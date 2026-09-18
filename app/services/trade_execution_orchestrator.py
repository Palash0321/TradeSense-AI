from app.services.trade_engine.risk_engine import RiskEngine
from app.services.trade_engine.position_sizing_engine import PositionSizingEngine
from app.services.trade_engine.order_intent_service import OrderIntentService


class TradeExecutionOrchestrator:
    """
    Coordinates the trade-preparation pipeline:

        Trade Candidate
              ↓
          RiskEngine
              ↓
      PositionSizingEngine
              ↓
       OrderIntentService

    This service does NOT:
    - generate trading signals
    - modify strategy logic
    - calculate entry
    - calculate stop loss
    - calculate targets
    - choose a risk percentage
    - place orders
    - communicate with a broker
    - execute paper trades
    - enforce portfolio exposure

    It only orchestrates the existing services and returns one
    canonical preparation result.
    """

    def __init__(
        self,
        trade_candidate,
        risk_budget,
        risk_engine=None,
        position_sizing_engine=None,
        order_intent_service=None,
    ):
        self.trade_candidate = trade_candidate or {}
        self.risk_budget = risk_budget

        self.risk_engine = risk_engine or RiskEngine(
            trade_candidate=self.trade_candidate
        )

        self.position_sizing_engine = position_sizing_engine

        self.order_intent_service = order_intent_service

    def prepare(self):
        """
        Run:

            Trade Candidate
                  ↓
              Risk Gate
                  ↓
            Position Sizing
                  ↓
             Order Intent

        The pipeline fails closed. A failed upstream stage prevents
        downstream stages from being executed.
        """

        # ============================================================
        # 1. Risk Gate
        # ============================================================

        risk_result = self.risk_engine.evaluate()

        if risk_result.get("risk_decision") != "PASS":
            return self._build_rejection(
                stage="RISK",
                risk_result=risk_result,
            )

        # ============================================================
        # 2. Position Sizing
        # ============================================================

        sizing_engine = self.position_sizing_engine or PositionSizingEngine(
            trade_candidate=self.trade_candidate,
            risk_result=risk_result,
            risk_budget=self.risk_budget,
        )

        sizing_result = sizing_engine.calculate()

        if sizing_result.get("sizing_decision") != "PASS":
            return self._build_rejection(
                stage="POSITION_SIZING",
                risk_result=risk_result,
                sizing_result=sizing_result,
            )

        # ============================================================
        # 3. Order Intent
        # ============================================================

        order_intent_service = (
            self.order_intent_service
            or OrderIntentService(
                trade_candidate=self.trade_candidate,
                risk_result=risk_result,
                sizing_result=sizing_result,
            )
        )

        order_intent = order_intent_service.build()

        if order_intent.get("intent_decision") != "PASS":
            return self._build_rejection(
                stage="ORDER_INTENT",
                risk_result=risk_result,
                sizing_result=sizing_result,
                order_intent=order_intent,
            )

        # ============================================================
        # 4. Fully Prepared
        # ============================================================

        return {
            "execution_decision": "PASS",
            "ready_for_execution": True,
            "failed_stage": None,
            "trade_candidate": self.trade_candidate,
            "risk": risk_result,
            "position_sizing": sizing_result,
            "order_intent": order_intent,
            "source": "TradeExecutionOrchestrator",
        }

    @staticmethod
    def _build_rejection(
        stage,
        risk_result=None,
        sizing_result=None,
        order_intent=None,
    ):
        """
        Build one canonical fail-closed result.

        Downstream stages that were not executed remain None.
        """

        return {
            "execution_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": stage,
            "trade_candidate": None,
            "risk": risk_result,
            "position_sizing": sizing_result,
            "order_intent": order_intent,
            "source": "TradeExecutionOrchestrator",
        }