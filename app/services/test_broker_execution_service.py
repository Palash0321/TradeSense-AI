from app.services.broker_execution_service import BrokerExecutionService


def valid_order_intent():
    return {
        "status": "PASS",
        "ready_for_execution": True,
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "order_type": "MARKET",
        "quantity": 100,
        "entry": 2500.0,
    }


def test_rejects_missing_order_intent():
    service = BrokerExecutionService()

    result = service.execute(None)

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "NOT_EXECUTED"


def test_rejects_order_intent_not_ready():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["ready_for_execution"] = False

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "NOT_EXECUTED"


def test_rejects_failed_order_intent():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["status"] = "REJECT"

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "NOT_EXECUTED"


def test_rejects_invalid_direction():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["direction"] = "INVALID"

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False


def test_rejects_invalid_order_type():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["order_type"] = "TRAILING_STOP"

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False


def test_rejects_invalid_quantity():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["quantity"] = 0

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False


def test_rejects_non_integer_quantity():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["quantity"] = 10.5

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False


def test_rejects_non_market_order_without_entry():
    service = BrokerExecutionService()

    order_intent = valid_order_intent()
    order_intent["order_type"] = "LIMIT"
    order_intent["entry"] = None

    result = service.execute(order_intent)

    assert result["status"] == "REJECT"
    assert result["executed"] is False


def test_valid_order_is_blocked_when_no_broker_adapter_exists():
    service = BrokerExecutionService()

    result = service.execute(valid_order_intent())

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "NOT_CONFIGURED"


class FakeBrokerAdapter:
    def __init__(self):
        self.received_order = None

    def execute(self, order_intent):
        self.received_order = order_intent

        return {
            "status": "PASS",
            "executed": True,
            "broker_status": "EXECUTED",
            "broker_order_id": "TEST-ORDER-001",
        }


def test_broker_adapter_receives_valid_order():
    adapter = FakeBrokerAdapter()
    service = BrokerExecutionService(
        broker_adapter=adapter
    )

    order_intent = valid_order_intent()

    result = service.execute(order_intent)

    assert result["status"] == "PASS"
    assert result["executed"] is True
    assert result["broker_status"] == "EXECUTED"
    assert result["broker_result"]["broker_order_id"] == (
        "TEST-ORDER-001"
    )
    assert adapter.received_order == order_intent


class FailingBrokerAdapter:
    def execute(self, order_intent):
        raise RuntimeError("Simulated broker failure")


def test_broker_adapter_failure_is_fail_closed():
    service = BrokerExecutionService(
        broker_adapter=FailingBrokerAdapter()
    )

    result = service.execute(valid_order_intent())

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "ADAPTER_ERROR"
    assert result["error_type"] == "RuntimeError"


class InvalidResponseBrokerAdapter:
    def execute(self, order_intent):
        return "INVALID RESPONSE"


def test_invalid_broker_adapter_response_is_rejected():
    service = BrokerExecutionService(
        broker_adapter=InvalidResponseBrokerAdapter()
    )

    result = service.execute(valid_order_intent())

    assert result["status"] == "REJECT"
    assert result["executed"] is False
    assert result["broker_status"] == "INVALID_ADAPTER_RESPONSE"