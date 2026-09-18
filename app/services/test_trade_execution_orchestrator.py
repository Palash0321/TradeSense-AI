from app.services.trade_execution_orchestrator import (
    TradeExecutionOrchestrator,
)


class FakeRiskEngine:
    def __init__(self, result):
        self.result = result
        self.called = False

    def evaluate(self):
        self.called = True
        return self.result


class FakePositionSizingEngine:
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


def make_candidate():
    return {
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "decision": "BUY",
        "actionable": True,
        "setup_type": "BREAKOUT",
        "entry": 100.0,
        "stop_loss": 98.0,
        "target1": 103.0,
        "target2": 105.0,
        "target3": 107.0,
        "risk_reward": 1.5,
    }


def make_risk_pass():
    return {
        "risk_decision": "PASS",
        "passed": True,
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "entry": 100.0,
        "stop_loss": 98.0,
        "risk": 2.0,
        "target1": 103.0,
        "target2": 105.0,
        "target3": 107.0,
        "reasons": [],
        "source": "RiskEngine",
    }


def make_sizing_pass():
    return {
        "sizing_decision": "PASS",
        "passed": True,
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "entry": 100.0,
        "stop_loss": 98.0,
        "risk_per_unit": 2.0,
        "risk_budget": 1000.0,
        "quantity": 500,
        "position_value": 50000.0,
        "actual_risk": 1000.0,
        "reasons": [],
        "source": "PositionSizingEngine",
    }


def make_order_intent_pass():
    return {
        "intent_decision": "PASS",
        "ready_for_execution": True,
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "order_type": "MARKET",
        "quantity": 500,
        "entry": 100.0,
        "stop_loss": 98.0,
        "target1": 103.0,
        "target2": 105.0,
        "target3": 107.0,
        "risk_budget": 1000.0,
        "risk_per_unit": 2.0,
        "actual_risk": 1000.0,
        "position_value": 50000.0,
        "reasons": [],
        "source": "OrderIntentService",
    }


def test_full_pipeline_passes():
    candidate = make_candidate()

    risk_engine = FakeRiskEngine(make_risk_pass())
    sizing_engine = FakePositionSizingEngine(make_sizing_pass())
    intent_service = FakeOrderIntentService(make_order_intent_pass())

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=candidate,
        risk_budget=1000.0,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "PASS"
    assert result["ready_for_execution"] is True
    assert result["failed_stage"] is None

    assert result["trade_candidate"] == candidate
    assert result["risk"]["risk_decision"] == "PASS"
    assert result["position_sizing"]["sizing_decision"] == "PASS"
    assert result["order_intent"]["intent_decision"] == "PASS"

    assert risk_engine.called is True
    assert sizing_engine.called is True
    assert intent_service.called is True


def test_risk_rejection_stops_pipeline():
    candidate = make_candidate()

    risk_engine = FakeRiskEngine(
        {
            "risk_decision": "REJECT",
            "passed": False,
            "reasons": ["Invalid risk geometry."],
            "source": "RiskEngine",
        }
    )

    sizing_engine = FakePositionSizingEngine(
        make_sizing_pass()
    )

    intent_service = FakeOrderIntentService(
        make_order_intent_pass()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=candidate,
        risk_budget=1000.0,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "RISK"

    assert result["risk"]["risk_decision"] == "REJECT"
    assert result["position_sizing"] is None
    assert result["order_intent"] is None

    assert risk_engine.called is True
    assert sizing_engine.called is False
    assert intent_service.called is False


def test_position_sizing_rejection_stops_order_intent():
    candidate = make_candidate()

    risk_engine = FakeRiskEngine(make_risk_pass())

    sizing_engine = FakePositionSizingEngine(
        {
            "sizing_decision": "REJECT",
            "passed": False,
            "quantity": 0,
            "reasons": [
                "Risk budget is insufficient to purchase at least one unit."
            ],
            "source": "PositionSizingEngine",
        }
    )

    intent_service = FakeOrderIntentService(
        make_order_intent_pass()
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=candidate,
        risk_budget=0.5,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "POSITION_SIZING"

    assert result["risk"]["risk_decision"] == "PASS"
    assert result["position_sizing"]["sizing_decision"] == "REJECT"
    assert result["order_intent"] is None

    assert risk_engine.called is True
    assert sizing_engine.called is True
    assert intent_service.called is False


def test_order_intent_rejection_is_terminal():
    candidate = make_candidate()

    risk_engine = FakeRiskEngine(make_risk_pass())
    sizing_engine = FakePositionSizingEngine(make_sizing_pass())

    intent_service = FakeOrderIntentService(
        {
            "intent_decision": "REJECT",
            "ready_for_execution": False,
            "reasons": ["Invalid order contract."],
            "source": "OrderIntentService",
        }
    )

    orchestrator = TradeExecutionOrchestrator(
        trade_candidate=candidate,
        risk_budget=1000.0,
        risk_engine=risk_engine,
        position_sizing_engine=sizing_engine,
        order_intent_service=intent_service,
    )

    result = orchestrator.prepare()

    assert result["execution_decision"] == "REJECT"
    assert result["ready_for_execution"] is False
    assert result["failed_stage"] == "ORDER_INTENT"

    assert result["risk"]["risk_decision"] == "PASS"
    assert result["position_sizing"]["sizing_decision"] == "PASS"
    assert result["order_intent"]["intent_decision"] == "REJECT"

    assert risk_engine.called is True
    assert sizing_engine.called is True
    assert intent_service.called is True