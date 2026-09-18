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
            "intent_decision": "PASS",
            "ready": True,
        },
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
        risk_budget=1000.0,
        portfolio_risk_state_service=portfolio_service,
        trade_execution_orchestrator=orchestrator,
    )

    result = runtime.prepare()

    assert result["runtime_decision"] == "PASS"
    assert result["ready_for_execution"] is True
    assert result["failed_stage"] is None

    assert result["portfolio_risk_state"] == portfolio_state
    assert result["execution"] == execution_result

    assert portfolio_service.calls == 1
    assert orchestrator.calls == 1


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