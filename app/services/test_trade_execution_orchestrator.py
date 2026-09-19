from app.services.trade_execution_orchestrator import (
    TradeExecutionOrchestrator,
)


VALID_CANDIDATE = {
    "symbol": "TEST",
    "direction": "LONG",
    "decision": "BUY",
    "actionable": True,
    "setup_type": "BREAKOUT",
    "preferred_setup": "BREAKOUT",
    "validation_status": "PASS",
    "entry": 100.0,
    "entry_low": None,
    "entry_high": None,
    "stop_loss": 98.0,
    "target1": 103.0,
    "target2": 105.0,
    "target3": 107.0,
    "risk_reward": {
        "target1": 1.5,
        "target2": 2.5,
        "target3": 3.5,
    },
    "ai_confidence": 80.0,
    "breakout_trigger": True,
    "breakout_level": 100.0,
}


VALID_PORTFOLIO_STATE = {
    "status": "PASS",
    "passed": True,
    "user_id": 1,
    "cash_balance": 900000.0,
    "holdings_market_value": 100000.0,
    "gross_exposure": 100000.0,
    "equity": 1000000.0,
    "holdings_count": 2,
    "open_risk_known": False,
    "open_risk": None,
    "risk_budget_policy_defined": False,
    "risk_budget": None,
    "checks": {
        "account_exists": True,
        "cash_balance_valid": True,
        "equity_valid": True,
        "portfolio_exposure_valid": True,
    },
    "reasons": [],
    "source": "PortfolioRiskStateService",
}


class FakeRiskEngine:
    def __init__(self, result):
        self.result = result
        self.called = False

    def evaluate(self):
        self.called = True
        return self.result


class FakeSizingEngine:
    def __init__(self, result):
        self.result = result
        self.called = False

    def calculate(self):
        self.called = True
        return self.result


class FakeOrderIntentService:
    def __init__(self, result):
        self.result = result
        self.called = False

    def build(self):
        self.called = True
        return self.result


def passed_risk():
    return {
        "risk_decision": "PASS",
        "checks": {},
        "reasons": [],
        "risk": 2.0,
        "source": "RiskEngine",
    }


def rejected_risk():
    return {
        "risk_decision": "REJECT",
        "checks": {},
        "reasons": ["Risk rejected."],
        "risk": None,
        "source": "RiskEngine",
    }


def passed_sizing():
    return {
        "sizing_decision": "PASS",
        "quantity": 500,
        "position_value": 50000.0,
        "actual_risk": 1000.0,
        "risk_budget": 1000.0,
        "source": "PositionSizingEngine",
    }


def rejected_sizing():
    return {
        "sizing_decision": "REJECT",
        "quantity": None,
        "position_value": None,
        "actual_risk": None,
        "risk_budget": 0.5,
        "reasons": ["Insufficient risk budget."],
        "source": "PositionSizingEngine",
    }


def passed_intent():
    return {
        "intent_decision": "PASS",
        "ready_for_execution": True,
        "symbol": "TEST",
        "direction": "LONG",
        "order_type": "STOP",
        "quantity": 500,
        "entry": 100.0,
        "stop_loss": 98.0,
        "source": "OrderIntentService",
    }


def rejected_intent():
    return {
        "intent_decision": "REJECT",
        "ready_for_execution": False,
        "symbol": "TEST",
        "direction": "LONG",
        "order_type": None,
        "quantity": None,
        "reasons": ["Order intent rejected."],
        "source": "OrderIntentService",
    }


def test_full_pipeline_passes_with_legacy_direct_budget():
    risk_engine = FakeRiskEngine(passed_risk())
    sizing_engine = FakeSizingEngine(passed_sizing())
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        risk_budget=1000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "PASS"
    assert result["ready_for_execution"] is True
    assert result["failed_stage"] is None

    assert result["risk"]["risk_decision"] == "PASS"
    assert result["risk_budget"]["status"] == "PASS"
    assert result["risk_budget"]["risk_budget"] == 1000.0

    assert (
        result["position_sizing"]["sizing_decision"]
        == "PASS"
    )

    assert (
        result["order_intent"]["intent_decision"]
        == "PASS"
    )

    assert risk_engine.called is True
    assert sizing_engine.called is True
    assert intent_service.called is True


def test_pipeline_uses_risk_budget_service_with_portfolio_state():
    risk_engine = FakeRiskEngine(passed_risk())
    sizing_engine = FakeSizingEngine(passed_sizing())
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        portfolio_state=VALID_PORTFOLIO_STATE,
        risk_budget=5000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "PASS"
    assert result["ready_for_execution"] is True

    assert result["risk_budget"]["status"] == "PASS"
    assert result["risk_budget"]["risk_budget"] == 5000.0
    assert result["risk_budget"]["user_id"] == 1
    assert result["risk_budget"]["equity"] == 1000000.0


def test_invalid_portfolio_state_fails_closed_before_sizing():
    risk_engine = FakeRiskEngine(passed_risk())
    sizing_engine = FakeSizingEngine(passed_sizing())
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    invalid_state = dict(
        VALID_PORTFOLIO_STATE
    )
    invalid_state["status"] = "REJECT"
    invalid_state["passed"] = False

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        portfolio_state=invalid_state,
        risk_budget=5000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "risk_budget"

    assert (
        result["risk_budget"]["status"]
        == "REJECT"
    )

    assert sizing_engine.called is False
    assert intent_service.called is False


def test_invalid_risk_budget_fails_closed_before_sizing():
    risk_engine = FakeRiskEngine(passed_risk())
    sizing_engine = FakeSizingEngine(passed_sizing())
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        portfolio_state=VALID_PORTFOLIO_STATE,
        risk_budget=0,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "risk_budget"

    assert (
        result["risk_budget"]["status"]
        == "REJECT"
    )

    assert sizing_engine.called is False
    assert intent_service.called is False


def test_risk_rejection_stops_pipeline():
    risk_engine = FakeRiskEngine(
        rejected_risk()
    )
    sizing_engine = FakeSizingEngine(
        passed_sizing()
    )
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        risk_budget=1000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "risk"

    assert sizing_engine.called is False
    assert intent_service.called is False


def test_sizing_rejection_stops_order_intent():
    risk_engine = FakeRiskEngine(
        passed_risk()
    )
    sizing_engine = FakeSizingEngine(
        rejected_sizing()
    )
    intent_service = FakeOrderIntentService(
        passed_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        risk_budget=1000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "position_sizing"

    assert sizing_engine.called is True
    assert intent_service.called is False


def test_order_intent_rejection_is_terminal():
    risk_engine = FakeRiskEngine(
        passed_risk()
    )
    sizing_engine = FakeSizingEngine(
        passed_sizing()
    )
    intent_service = FakeOrderIntentService(
        rejected_intent()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        risk_budget=1000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "order_intent"

    assert intent_service.called is True

def test_order_intent_pass_but_not_ready_fails_closed():
    risk_engine = FakeRiskEngine(
        passed_risk()
    )
    sizing_engine = FakeSizingEngine(
        passed_sizing()
    )

    not_ready_intent = passed_intent()
    not_ready_intent["ready_for_execution"] = False

    intent_service = FakeOrderIntentService(
        not_ready_intent
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=VALID_CANDIDATE,
        risk_budget=1000,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "order_intent"

    assert intent_service.called is True