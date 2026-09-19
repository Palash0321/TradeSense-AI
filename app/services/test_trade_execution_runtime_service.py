import uuid

from app.core.database import SessionLocal
from app.models.execution_record import ExecutionRecord
from app.services.trade_execution_runtime_service import (
    TradeExecutionRuntimeService,
)


class FakePortfolioRiskStateService:
    def __init__(self, state):
        self.state = state
        self.calls = 0

    def get_state(self):
        self.calls += 1
        return self.state


class FakeTradeExecutionOrchestrator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def prepare(self):
        self.calls += 1
        return self.result


class FakeExecutionLedgerService:
    def __init__(
        self,
        create_result,
        transition_result=None,
        broker_state_result=None,
    ):
        self.create_result = create_result
        self.transition_result = (
            transition_result
            or {
                "status": "PASS",
                "execution_id": "execution-123",
                "state": "VALIDATED",
            }
        )
        self.broker_state_result = (
            broker_state_result
            or {
                "status": "PASS",
                "execution_id": "execution-123",
                "state": "VALIDATED",
            }
        )

        self.create_calls = 0
        self.transition_calls = []
        self.broker_state_calls = []

    def create(self):
        self.create_calls += 1
        return self.create_result

    def transition(
        self,
        new_state,
        reason=None,
        metadata=None,
    ):
        self.transition_calls.append(
            {
                "new_state": new_state,
                "reason": reason,
                "metadata": metadata,
            }
        )
        return self.transition_result

    def update_broker_state(
        self,
        broker_status=None,
        broker_order_id=None,
        failure_reason=None,
    ):
        self.broker_state_calls.append(
            {
                "broker_status": broker_status,
                "broker_order_id": broker_order_id,
                "failure_reason": failure_reason,
            }
        )
        return self.broker_state_result

class FakeBrokerExecutionService:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.received_order_intent = None

    def execute(self, order_intent):
        self.calls += 1
        self.received_order_intent = order_intent
        return self.result


def test_runtime_composes_portfolio_state_into_execution_pipeline():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 90000.0,
        "gross_exposure": 10000.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "PASS",
        "ready_for_execution": True,
        "failed_stage": None,
        "trade_candidate": {
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        "risk": {
            "risk_decision": "PASS",
        },
        "risk_budget": {
            "status": "PASS",
            "risk_budget": 1000.0,
        },
        "position_sizing": {
            "sizing_decision": "PASS",
            "quantity": 10,
        },
        "order_intent": {
            "status": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
            "entry": 2500.0,
        },
    }

    ledger_result = {
        "status": "PASS",
        "execution_id": "execution-123",
        "record_id": 1,
        "state": "CREATED",
    }

    validation_result = {
        "status": "PASS",
        "execution_id": "execution-123",
        "state": "VALIDATED",
    }

    broker_result = {
        "status": "PASS",
        "executed": True,
        "broker_status": "EXECUTED",
        "broker_result": {
            "broker_order_id": "TEST-ORDER-001",
        },
    }

    portfolio_service = FakePortfolioRiskStateService(
        portfolio_state
    )

    orchestrator = FakeTradeExecutionOrchestrator(
        execution_result
    )

    ledger_service = FakeExecutionLedgerService(
        create_result=ledger_result,
        transition_result=validation_result,
    )

    broker_service = FakeBrokerExecutionService(
        broker_result
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=1000.0,
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
        execution_ledger_service=ledger_service,
        broker_execution_service=broker_service,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "PASS"
    assert result["ready_for_execution"] is True
    assert result["failed_stage"] is None

    assert result["portfolio_risk_state"] == portfolio_state
    assert result["execution"] == execution_result

    assert result["ledger"] == ledger_result
    assert result["ledger_validation"] == validation_result
    assert result["broker"] == broker_result

    assert portfolio_service.calls == 1
    assert orchestrator.calls == 1

    assert ledger_service.create_calls == 1

    assert len(ledger_service.transition_calls) == 1
    assert (
        ledger_service.transition_calls[0]["new_state"]
        == "VALIDATED"
    )

    assert ledger_service.broker_state_calls == [
        {
            "broker_status": "EXECUTED",
            "broker_order_id": "TEST-ORDER-001",
            "failure_reason": None,
        }
    ]

    assert broker_service.calls == 1
    assert (
        broker_service.received_order_intent
        == execution_result["order_intent"]
    )


def test_runtime_rejects_when_portfolio_state_is_invalid():
    portfolio_state = {
        "status": "REJECT",
        "passed": False,
        "user_id": 1,
        "equity": None,
        "cash_balance": None,
        "gross_exposure": None,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
        "reasons": [
            "Paper account not found."
        ],
    }

    portfolio_service = FakePortfolioRiskStateService(
        portfolio_state
    )

    orchestrator = FakeTradeExecutionOrchestrator(
        {
            "execution_decision": "PASS",
            "ready_for_execution": True,
        }
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=1000.0,
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "portfolio_risk_state"

    assert result["portfolio_risk_state"] == portfolio_state
    assert result["execution"] is None
    assert result["ledger"] is None

    assert portfolio_service.calls == 1
    assert orchestrator.calls == 0


def test_runtime_propagates_execution_rejection():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 100000.0,
        "gross_exposure": 0.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "REJECT",
        "ready_for_execution": False,
        "failed_stage": "risk_budget",
        "trade_candidate": {
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        "risk": {
            "risk_decision": "PASS",
        },
        "risk_budget": {
            "status": "REJECT",
            "risk_budget": None,
        },
        "position_sizing": None,
        "order_intent": None,
    }

    portfolio_service = FakePortfolioRiskStateService(
        portfolio_state
    )

    orchestrator = FakeTradeExecutionOrchestrator(
        execution_result
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=None,
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "risk_budget"

    assert result["execution"] == execution_result
    assert result["ledger"] is None


def test_runtime_uses_injected_ledger_when_execution_is_ready():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 100000.0,
        "gross_exposure": 0.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "PASS",
        "ready_for_execution": True,
        "failed_stage": None,
        "order_intent": {
            "status": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
            "entry": 2500.0,
        },
    }

    ledger_result = {
        "status": "PASS",
        "execution_id": "execution-123",
        "record_id": 1,
        "state": "CREATED",
    }

    validation_result = {
        "status": "PASS",
        "execution_id": "execution-123",
        "state": "VALIDATED",
    }

    broker_result = {
        "status": "PASS",
        "executed": True,
        "broker_status": "EXECUTED",
        "broker_result": {
            "broker_order_id": "TEST-ORDER-001",
        },
    }

    portfolio_service = FakePortfolioRiskStateService(
        portfolio_state
    )

    orchestrator = FakeTradeExecutionOrchestrator(
        execution_result
    )

    ledger_service = FakeExecutionLedgerService(
        create_result=ledger_result,
        transition_result=validation_result,
    )

    broker_service = FakeBrokerExecutionService(
    {
        "status": "PASS",
        "executed": True,
        "broker_status": "EXECUTED",
        "broker_result": {
            "broker_order_id": "TEST-ORDER-001",
        },
    }
)

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=1000.0,
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
        execution_ledger_service=ledger_service,
        broker_execution_service=broker_service,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "PASS"
    assert result["ready_for_execution"] is True

    assert result["ledger"] == ledger_result
    assert result["ledger_validation"] == (
        validation_result
    )

    assert result["broker"] == broker_result

    assert ledger_service.create_calls == 1

    assert len(
        ledger_service.transition_calls
    ) == 1

    assert (
        ledger_service.transition_calls[0][
            "new_state"
        ]
        == "VALIDATED"
    )

    assert broker_service.calls == 1

    assert (
        broker_service.received_order_intent
        == execution_result["order_intent"]
    )

    assert len(
        ledger_service.broker_state_calls
    ) == 1

    assert (
        ledger_service.broker_state_calls[0][
            "broker_status"
        ]
        == "EXECUTED"
    )

    assert (
        ledger_service.broker_state_calls[0][
            "broker_order_id"
        ]
        == "TEST-ORDER-001"
    )


def test_runtime_rejects_when_order_intent_is_missing():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 100000.0,
        "gross_exposure": 0.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "PASS",
        "ready_for_execution": True,
        "failed_stage": None,
        "order_intent": None,
    }

    portfolio_service = FakePortfolioRiskStateService(
        portfolio_state
    )

    orchestrator = FakeTradeExecutionOrchestrator(
        execution_result
    )

    ledger_service = FakeExecutionLedgerService(
        create_result={
            "status": "PASS",
            "execution_id": "should-not-be-created",
        }
    )

    broker_service = FakeBrokerExecutionService(
        {
            "status": "PASS",
            "executed": True,
        }
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
        execution_ledger_service=ledger_service,
        broker_execution_service=broker_service,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "order_intent"

    assert result["ledger"]["status"] == "REJECT"

    assert ledger_service.create_calls == 0
    assert broker_service.calls == 0


def test_runtime_persists_real_execution_record():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        portfolio_state = {
            "status": "PASS",
            "passed": True,
            "user_id": 1,
            "equity": 100000.0,
            "cash_balance": 100000.0,
            "gross_exposure": 0.0,
            "open_risk_known": False,
            "open_risk": None,
            "risk_budget_policy_defined": False,
            "risk_budget": None,
        }

        execution_result = {
            "execution_decision": "PASS",
            "ready_for_execution": True,
            "failed_stage": None,
            "order_intent": {
                "status": "PASS",
                "ready_for_execution": True,
                "symbol": "RELIANCE.NS",
                "direction": "LONG",
                "order_type": "MARKET",
                "quantity": 10,
                "entry": 2500.0,
            },
        }

        portfolio_service = FakePortfolioRiskStateService(
            portfolio_state
        )

        orchestrator = FakeTradeExecutionOrchestrator(
            execution_result
        )

        broker_service = FakeBrokerExecutionService(
            {
                "status": "PASS",
                "executed": True,
                "broker_status": "EXECUTED",
                "broker_result": {
                    "broker_order_id": "TEST-ORDER-001",
                },
            }
        )

        runtime = TradeExecutionRuntimeService(
            db=db,
            user_id=1,
            trade_candidate={
                "symbol": "RELIANCE.NS",
                "direction": "LONG",
            },
            risk_budget=1000.0,
            portfolio_risk_state_service=portfolio_service,
            trade_execution_orchestrator=orchestrator,
            broker_execution_service=broker_service,
        )

        runtime._generate_execution_id = (
            lambda: execution_id
        )

        result = runtime.prepare()

        assert result["runtime_decision"] == "PASS"
        assert result["ready_for_execution"] is True
        assert result["ledger"]["status"] == "PASS"
        assert result["ledger"]["execution_id"] == execution_id
        assert result["ledger"]["state"] == "CREATED"
        assert result["ledger_validation"]["state"] == "VALIDATED"
        assert result["broker"]["broker_status"] == "EXECUTED"

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.user_id == 1
        assert record.symbol == "RELIANCE.NS"
        assert record.direction == "LONG"
        assert record.order_type == "MARKET"
        assert record.quantity == 10
        assert record.entry_price == 2500.0
        assert record.current_state == "VALIDATED"
        assert record.broker_status == "EXECUTED"
        assert record.broker_order_id == "TEST-ORDER-001"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()

def test_runtime_rejects_when_broker_rejects_order():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 100000.0,
        "gross_exposure": 0.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "PASS",
        "ready_for_execution": True,
        "failed_stage": None,
        "order_intent": {
            "status": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
            "entry": 2500.0,
        },
    }

    broker_result = {
        "status": "REJECT",
        "executed": False,
        "broker_status": "ADAPTER_ERROR",
        "reason": "Broker adapter failed.",
    }

    ledger_service = FakeExecutionLedgerService(
        create_result={
            "status": "PASS",
            "execution_id": "execution-123",
            "record_id": 1,
            "state": "CREATED",
        },
        transition_result={
            "status": "PASS",
            "execution_id": "execution-123",
            "state": "VALIDATED",
        },
    )

    broker_service = FakeBrokerExecutionService(
        broker_result
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=1000.0,
        portfolio_risk_state_service=(
            FakePortfolioRiskStateService(
                portfolio_state
            )
        ),
        trade_execution_orchestrator=(
            FakeTradeExecutionOrchestrator(
                execution_result
            )
        ),
        execution_ledger_service=ledger_service,
        broker_execution_service=broker_service,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "broker_execution"

    assert result["broker"] == broker_result

    assert (
        ledger_service.broker_state_calls[0][
            "broker_status"
        ]
        == "ADAPTER_ERROR"
    )

    assert (
        ledger_service.broker_state_calls[0][
            "failure_reason"
        ]
        == "Broker adapter failed."
    )

def test_runtime_does_not_infer_filled_or_completed_from_broker_execution():
    portfolio_state = {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "equity": 100000.0,
        "cash_balance": 100000.0,
        "gross_exposure": 0.0,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
    }

    execution_result = {
        "execution_decision": "PASS",
        "ready_for_execution": True,
        "failed_stage": None,
        "order_intent": {
            "status": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
            "entry": 2500.0,
        },
    }

    ledger_service = FakeExecutionLedgerService(
        create_result={
            "status": "PASS",
            "execution_id": "execution-123",
            "record_id": 1,
            "state": "CREATED",
        },
        transition_result={
            "status": "PASS",
            "execution_id": "execution-123",
            "state": "VALIDATED",
        },
    )

    broker_service = FakeBrokerExecutionService(
        {
            "status": "PASS",
            "executed": True,
            "broker_status": "EXECUTED",
            "broker_result": {
                "broker_order_id": "TEST-ORDER-001",
            },
        }
    )

    runtime = TradeExecutionRuntimeService(
        db=None,
        user_id=1,
        trade_candidate={
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
        },
        risk_budget=1000.0,
        portfolio_risk_state_service=(
            FakePortfolioRiskStateService(
                portfolio_state
            )
        ),
        trade_execution_orchestrator=(
            FakeTradeExecutionOrchestrator(
                execution_result
            )
        ),
        execution_ledger_service=ledger_service,
        broker_execution_service=broker_service,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "PASS"
    assert result["ready_for_execution"] is True

    assert (
        result["ledger_validation"]["state"]
        == "VALIDATED"
    )

    assert result["broker"]["executed"] is True

    transition_states = [
        call["new_state"]
        for call in ledger_service.transition_calls
    ]

    assert transition_states == ["VALIDATED"]
    assert "FILLED" not in transition_states
    assert "COMPLETED" not in transition_states