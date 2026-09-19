from app.services.portfolio_risk_state_service import (
    PortfolioRiskStateService,
)
from app.services.trade_execution_orchestrator import (
    TradeExecutionOrchestrator,
)
from app.services.execution_ledger_service import (
    ExecutionLedgerService,
)
from app.services.broker_execution_service import (
    BrokerExecutionService,
)


class TradeExecutionRuntimeService:
    """
    Composes portfolio state, execution preparation, persistent execution
    lifecycle tracking, and the broker execution boundary.

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
        Execution Ledger CREATED
                ↓
        Execution Ledger VALIDATED
                ↓
        Broker Execution Boundary
                ↓
        Explicit Broker Result
                ↓
        Persistent Broker State

    This service does NOT:
    - generate trading signals
    - modify strategy logic
    - choose a risk-per-trade percentage
    - invent a risk budget
    - calculate position size
    - infer FILLED/COMPLETED state
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
        broker_execution_service=None,
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

        self.broker_execution_service = (
            broker_execution_service
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
            "broker": None,
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
            runtime_result["runtime_decision"] = "REJECT"
            runtime_result["ready_for_execution"] = False
            runtime_result["failed_stage"] = "order_intent"
            return runtime_result

        ledger_service = self._get_ledger_service(
            order_intent
        )

        if ledger_service is None:
            runtime_result["ledger"] = {
                "status": "REJECT",
                "reason": (
                    "Execution ledger could not be initialized."
                ),
                "source": "TradeExecutionRuntimeService",
            }
            runtime_result["runtime_decision"] = "REJECT"
            runtime_result["ready_for_execution"] = False
            runtime_result["failed_stage"] = "execution_ledger"
            return runtime_result

        ledger_result = self._create_ledger(
            ledger_service
        )

        runtime_result["ledger"] = ledger_result

        if ledger_result.get("status") != "PASS":
            runtime_result["runtime_decision"] = "REJECT"
            runtime_result["ready_for_execution"] = False
            runtime_result["failed_stage"] = "execution_ledger"
            return runtime_result

        validation_result = ledger_service.transition(
            new_state="VALIDATED",
            reason=(
                "Execution preparation passed and the "
                "order intent was validated for broker submission."
            ),
            metadata={
                "source": "TradeExecutionRuntimeService",
            },
        )

        runtime_result["ledger_validation"] = (
            validation_result
        )

        if validation_result.get("status") != "PASS":
            runtime_result["runtime_decision"] = "REJECT"
            runtime_result["ready_for_execution"] = False
            runtime_result["failed_stage"] = "execution_ledger_validation"
            return runtime_result

        broker_service = self._get_broker_service()

        if broker_service is None:
            runtime_result["broker"] = {
                "status": "REJECT",
                "executed": False,
                "broker_status": "NOT_CONFIGURED",
                "reason": (
                    "Broker execution service is not configured."
                ),
                "source": "TradeExecutionRuntimeService",
            }

            ledger_service.update_broker_state(
                broker_status="NOT_CONFIGURED",
                failure_reason=(
                    "Broker execution service is not configured."
                ),
            )

            return self._broker_rejection(
                runtime_result
            )

        broker_result = broker_service.execute(
            order_intent
        )

        runtime_result["broker"] = broker_result

        self._persist_broker_result(
            ledger_service=ledger_service,
            broker_result=broker_result,
        )

        if broker_result.get("status") != "PASS":
            return self._broker_rejection(
                runtime_result
            )

        return runtime_result

    def _get_ledger_service(self, order_intent):
        if self.execution_ledger_service is not None:
            return self.execution_ledger_service

        if self.db is None:
            return None

        return self._build_ledger_service(
            order_intent
        )

    def _get_broker_service(self):
        return (
            self.broker_execution_service
            or BrokerExecutionService()
        )

    def _build_ledger_service(self, order_intent):
        execution_id = order_intent.get(
            "execution_id"
        )

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
    def _persist_broker_result(
        ledger_service,
        broker_result,
    ):
        broker_status = broker_result.get(
            "broker_status"
        )

        broker_result_payload = broker_result.get(
            "broker_result"
        )

        broker_order_id = None

        if isinstance(
            broker_result_payload,
            dict,
        ):
            broker_order_id = (
                broker_result_payload.get(
                    "broker_order_id"
                )
            )

        failure_reason = broker_result.get(
            "reason"
        )

        if broker_result.get("status") == "PASS":
            failure_reason = None

        ledger_service.update_broker_state(
            broker_status=broker_status,
            broker_order_id=broker_order_id,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _broker_rejection(runtime_result):
        runtime_result["runtime_decision"] = "REJECT"
        runtime_result["ready_for_execution"] = False
        runtime_result["failed_stage"] = (
            "broker_execution"
        )
        return runtime_result

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
            "broker": None,
            "source": "TradeExecutionRuntimeService",
        }