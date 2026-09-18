from app.services.portfolio_risk_state_service import PortfolioRiskStateService
from app.services.trade_execution_orchestrator import TradeExecutionOrchestrator


class TradeExecutionRuntimeService:
    """
    Composes real portfolio state with the existing execution-preparation
    pipeline.

    Runtime flow:

        Portfolio Risk State
                ↓
        Trade Execution Orchestrator
                ↓
        Risk Gate
                ↓
        Risk Budget Validation
                ↓
        Position Sizing
                ↓
        Order Intent

    This service does NOT:
    - generate trading signals
    - modify strategy logic
    - choose a risk-per-trade percentage
    - invent a risk budget
    - place broker orders
    - execute paper trades
    - modify portfolio holdings

    It is only responsible for runtime composition.
    """

    def __init__(
        self,
        db,
        user_id,
        trade_candidate,
        risk_budget=None,
        portfolio_risk_state_service=None,
        trade_execution_orchestrator=None,
    ):
        self.db = db
        self.user_id = user_id
        self.trade_candidate = trade_candidate or {}
        self.risk_budget = risk_budget

        self.portfolio_risk_state_service = (
            portfolio_risk_state_service
            or PortfolioRiskStateService(
                db=db,
                user_id=user_id,
            )
        )

        self.trade_execution_orchestrator = trade_execution_orchestrator

    def prepare(self):
        portfolio_state = self.portfolio_risk_state_service.get_state()

        if portfolio_state.get("status") != "PASS":
            return self._reject(
                failed_stage="portfolio_risk_state",
                portfolio_state=portfolio_state,
            )

        orchestrator = self.trade_execution_orchestrator

        if orchestrator is None:
            orchestrator = TradeExecutionOrchestrator(
                trade_candidate=self.trade_candidate,
                risk_budget=self.risk_budget,
                portfolio_state=portfolio_state,
            )

        execution_result = orchestrator.prepare()

        return {
            "runtime_decision": execution_result.get(
                "execution_decision",
                "REJECT",
            ),
            "ready_for_execution": execution_result.get(
                "ready_for_execution",
                False,
            ),
            "failed_stage": execution_result.get("failed_stage"),
            "portfolio_risk_state": portfolio_state,
            "execution": execution_result,
            "source": "TradeExecutionRuntimeService",
        }

    @staticmethod
    def _reject(
        failed_stage,
        portfolio_state,
    ):
        return {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": failed_stage,
            "portfolio_risk_state": portfolio_state,
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }