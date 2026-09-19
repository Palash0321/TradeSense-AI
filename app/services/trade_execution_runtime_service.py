from app.services.portfolio_risk_state_service import (
    PortfolioRiskStateService,
)
from app.services.trade_execution_orchestrator import (
    TradeExecutionOrchestrator,
)
from app.services.execution_ledger_service import (
    ExecutionLedgerService,
)


class TradeExecutionRuntimeService:
    """
    Composes real portfolio state with the existing execution-preparation
    pipeline and optionally persists an execution ledger record.

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
                ↓
        Execution Ledger

    This service does NOT:
    - generate trading signals
    - modify strategy logic
    - choose a risk-per-trade percentage
    - invent a risk budget
    - place broker orders
    - execute paper trades
    - modify portfolio holdings
    """

    def __init__(
        self,
        db,
        user_id,
        trade_candidate,
        risk_budget=None,
        portfolio_risk_state_service=None,
        trade_execution_orchestrator=None,
        execution_ledger_service=None,
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

        self.trade_execution_orchestrator = (
            trade_execution_orchestrator
        )

        self.execution_ledger_service = (
            execution_ledger_service
        )

    def prepare(self):
        portfolio_state = (
            self.portfolio_risk_state_service.get_state()
        )

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

        runtime_result = {
            "runtime_decision": execution_result.get(
                "execution_decision",
                "REJECT",
            ),
            "ready_for_execution": execution_result.get(
                "ready_for_execution",
                False,
            ),
            "failed_stage": execution_result.get(
                "failed_stage"
            ),
            "portfolio_risk_state": portfolio_state,
            "execution": execution_result,
            "ledger": None,
            "source": "TradeExecutionRuntimeService",
        }

        if not runtime_result["ready_for_execution"]:
            return runtime_result

        order_intent = execution_result.get(
            "order_intent"
        )

        if not isinstance(order_intent, dict):
            runtime_result["ledger"] = {
                "status": "REJECT",
                "reason": (
                    "Execution pipeline did not produce "
                    "an order intent."
                ),
                "source": "TradeExecutionRuntimeService",
            }
            return runtime_result

        if self.execution_ledger_service is None:
            if self.db is None:
                return runtime_result

            ledger_service = self._build_ledger_service(
                order_intent
            )
        else:
            ledger_service = (
                self.execution_ledger_service
            )

        ledger_result = self._create_ledger(
            ledger_service
        )

        runtime_result["ledger"] = ledger_result

        return runtime_result

    def _build_ledger_service(self, order_intent):
        execution_id = order_intent.get("execution_id")

        if not execution_id:
            execution_id = self._generate_execution_id()

        return ExecutionLedgerService(
            db=self.db,
            execution_id=execution_id,
            user_id=self.user_id,
            symbol=order_intent.get(
                "symbol",
                self.trade_candidate.get("symbol"),
            ),
            direction=order_intent.get(
                "direction",
                self.trade_candidate.get("direction"),
            ),
            order_type=order_intent.get(
                "order_type"
            ),
            quantity=order_intent.get(
                "quantity"
            ),
            entry_price=order_intent.get(
                "entry"
            ),
        )

    @staticmethod
    def _create_ledger(ledger_service):
        return ledger_service.create()

    @staticmethod
    def _generate_execution_id():
        import uuid

        return str(uuid.uuid4())

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
            "ledger": None,
            "source": "TradeExecutionRuntimeService",
        }