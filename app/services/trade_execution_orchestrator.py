from app.services.trade_engine.risk_engine import RiskEngine
from app.services.trade_engine.position_sizing_engine import (
    PositionSizingEngine,
)
from app.services.trade_engine.order_intent_service import (
    OrderIntentService,
)
from app.services.risk_budget_service import RiskBudgetService


class TradeExecutionOrchestrator:
    """
    Coordinates the pre-execution risk pipeline.

    Pipeline:

        Trade Candidate
              ↓
          Risk Engine
              ↓
       Portfolio Risk State
              ↓
        Risk Budget Validation
              ↓
       Position Sizing Engine
              ↓
       Order Intent Service
              ↓
       Execution-ready result

    This orchestrator does NOT:
    - generate trading signals
    - modify strategy logic
    - choose a risk-per-trade percentage
    - invent a risk budget
    - place broker orders
    - execute trades
    """

    def __init__(
        self,
        trade_candidate,
        risk_budget=None,
        portfolio_state=None,
        risk_budget_result=None,
        risk_engine=None,
        position_sizing_engine=None,
        order_intent_service=None,
    ):
        self.trade_candidate = trade_candidate or {}

        self.risk_budget = risk_budget
        self.portfolio_state = portfolio_state
        self.risk_budget_result = risk_budget_result

        self.risk_engine = (
            risk_engine
            or RiskEngine(
                trade_candidate=self.trade_candidate
            )
        )

        self.position_sizing_engine = (
            position_sizing_engine
        )

        self.order_intent_service = (
            order_intent_service
        )

    def prepare(self):
        # ----------------------------------------------------------
        # Stage 1: Trade Risk Gate
        # ----------------------------------------------------------

        risk_result = self.risk_engine.evaluate()

        if risk_result.get("risk_decision") != "PASS":
            return self._reject(
                failed_stage="risk",
                risk=risk_result,
            )

        # ----------------------------------------------------------
        # Stage 2: Risk Budget Validation
        # ----------------------------------------------------------

        risk_budget_result = (
            self._resolve_risk_budget()
        )

        if risk_budget_result.get("status") != "PASS":
            return self._reject(
                failed_stage="risk_budget",
                risk=risk_result,
                risk_budget=risk_budget_result,
            )

        validated_risk_budget = (
            risk_budget_result.get("risk_budget")
        )

        # ----------------------------------------------------------
        # Stage 3: Position Sizing
        # ----------------------------------------------------------

        sizing_engine = (
            self.position_sizing_engine
            or PositionSizingEngine(
                trade_candidate=self.trade_candidate,
                risk_result=risk_result,
                risk_budget=validated_risk_budget,
            )
        )

        sizing_result = sizing_engine.calculate()

        if sizing_result.get("sizing_decision") != "PASS":
            return self._reject(
                failed_stage="position_sizing",
                risk=risk_result,
                risk_budget=risk_budget_result,
                position_sizing=sizing_result,
            )

        # ----------------------------------------------------------
        # Stage 4: Order Intent
        # ----------------------------------------------------------

        intent_service = (
            self.order_intent_service
            or OrderIntentService(
                trade_candidate=self.trade_candidate,
                risk_result=risk_result,
                sizing_result=sizing_result,
            )
        )

        order_intent = intent_service.build()

        if order_intent.get("intent_decision") != "PASS":
            return self._reject(
                failed_stage="order_intent",
                risk=risk_result,
                risk_budget=risk_budget_result,
                position_sizing=sizing_result,
                order_intent=order_intent,
            )

        # ----------------------------------------------------------
        # Final execution-ready result
        # ----------------------------------------------------------

        return {
            "execution_decision": "PASS",
            "ready_for_execution": True,
            "failed_stage": None,
            "trade_candidate": self.trade_candidate,
            "risk": risk_result,
            "risk_budget": risk_budget_result,
            "position_sizing": sizing_result,
            "order_intent": order_intent,
            "source": "TradeExecutionOrchestrator",
        }

    def _resolve_risk_budget(self):
        """
        Resolve and validate the monetary risk budget.

        Priority:
        1. Already validated RiskBudgetService result.
        2. Portfolio state + explicit risk budget.
        3. Legacy direct risk_budget path.

        The legacy path is retained so existing callers/tests
        continue to work. No percentage-based budget is created.
        """

        # ----------------------------------------------------------
        # Already validated result
        # ----------------------------------------------------------

        if self.risk_budget_result is not None:
            return self.risk_budget_result

        # ----------------------------------------------------------
        # New portfolio-state-aware path
        # ----------------------------------------------------------

        if self.portfolio_state is not None:
            return RiskBudgetService(
                portfolio_state=self.portfolio_state,
                risk_budget=self.risk_budget,
            ).validate()

        # ----------------------------------------------------------
        # Backward-compatible direct budget path
        # ----------------------------------------------------------

        if self.risk_budget is None:
            return {
                "status": "REJECT",
                "passed": False,
                "user_id": None,
                "equity": None,
                "cash_balance": None,
                "gross_exposure": None,
                "risk_budget": None,
                "risk_budget_policy_defined": False,
                "checks": {
                    "portfolio_state_valid": False,
                    "risk_budget_present": False,
                    "risk_budget_numeric": False,
                    "risk_budget_positive": False,
                },
                "reasons": [
                    "Risk budget was not supplied."
                ],
                "source": "RiskBudgetService",
            }

        return {
            "status": "PASS",
            "passed": True,
            "user_id": None,
            "equity": None,
            "cash_balance": None,
            "gross_exposure": None,
            "risk_budget": float(
                self.risk_budget
            ),
            "risk_budget_policy_defined": False,
            "checks": {
                "portfolio_state_valid": False,
                "risk_budget_present": True,
                "risk_budget_numeric": True,
                "risk_budget_positive": (
                    float(self.risk_budget) > 0
                ),
            },
            "reasons": [],
            "source": "TradeExecutionOrchestrator",
        }

    @staticmethod
    def _reject(
        failed_stage,
        risk=None,
        risk_budget=None,
        position_sizing=None,
        order_intent=None,
    ):
        return {
            "execution_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": failed_stage,
            "trade_candidate": None,
            "risk": risk,
            "risk_budget": risk_budget,
            "position_sizing": position_sizing,
            "order_intent": order_intent,
            "source": "TradeExecutionOrchestrator",
        }