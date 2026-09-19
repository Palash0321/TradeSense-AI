from typing import Any


class BrokerExecutionService:
    """
    Broker execution boundary for TradeSense-AI.

    This service defines the boundary between the internal execution
    pipeline and an external broker adapter.

    Current implementation is intentionally fail-closed.

    It does NOT:
    - generate signals
    - modify strategy logic
    - calculate entry/stop/targets
    - calculate risk
    - calculate position size
    - communicate with a broker
    - place live orders
    - store broker credentials
    - make network calls

    A future broker adapter can be introduced behind this boundary
    without changing the upstream trading decision pipeline.
    """

    SUPPORTED_ORDER_TYPES = {"MARKET", "LIMIT", "STOP"}

    def __init__(self, broker_adapter=None):
        self.broker_adapter = broker_adapter

    def execute(self, order_intent: dict[str, Any] | None):
        """
        Attempt to execute a validated order intent through the broker
        boundary.

        The current implementation is deliberately fail-closed:
        without an explicitly supplied broker adapter, no order is sent.
        """

        validation = self._validate_order_intent(order_intent)

        if validation["status"] != "PASS":
            return {
                "status": "REJECT",
                "executed": False,
                "broker_status": "NOT_EXECUTED",
                "reason": validation["reason"],
                "checks": validation["checks"],
                "source": "BrokerExecutionService",
            }

        if self.broker_adapter is None:
            return {
                "status": "REJECT",
                "executed": False,
                "broker_status": "NOT_CONFIGURED",
                "reason": (
                    "No broker adapter is configured. "
                    "Live broker execution is disabled."
                ),
                "checks": validation["checks"],
                "order_intent": order_intent,
                "source": "BrokerExecutionService",
            }

        try:
            broker_result = self.broker_adapter.execute(order_intent)
        except Exception as exc:
            return {
                "status": "REJECT",
                "executed": False,
                "broker_status": "ADAPTER_ERROR",
                "reason": "Broker adapter execution failed.",
                "error_type": type(exc).__name__,
                "checks": validation["checks"],
                "source": "BrokerExecutionService",
            }

        if not isinstance(broker_result, dict):
            return {
                "status": "REJECT",
                "executed": False,
                "broker_status": "INVALID_ADAPTER_RESPONSE",
                "reason": (
                    "Broker adapter returned an invalid response. "
                    "Expected a dictionary."
                ),
                "checks": validation["checks"],
                "source": "BrokerExecutionService",
            }

        return {
            "status": broker_result.get("status", "REJECT"),
            "executed": bool(broker_result.get("executed", False)),
            "broker_status": broker_result.get(
                "broker_status",
                "UNKNOWN",
            ),
            "broker_result": broker_result,
            "checks": validation["checks"],
            "source": "BrokerExecutionService",
        }

    def _validate_order_intent(self, order_intent):
        checks = []

        if not isinstance(order_intent, dict):
            return {
                "status": "REJECT",
                "reason": "Order intent must be a dictionary.",
                "checks": checks,
            }

        checks.append(
            {
                "check": "order_intent_type",
                "status": "PASS",
            }
        )

        if order_intent.get("status") != "PASS":
            return {
                "status": "REJECT",
                "reason": "Order intent status is not PASS.",
                "checks": checks,
            }

        checks.append(
            {
                "check": "order_intent_status",
                "status": "PASS",
            }
        )

        if order_intent.get("ready_for_execution") is not True:
            return {
                "status": "REJECT",
                "reason": (
                    "Order intent is not marked ready_for_execution."
                ),
                "checks": checks,
            }

        checks.append(
            {
                "check": "execution_readiness",
                "status": "PASS",
            }
        )

        symbol = order_intent.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            return {
                "status": "REJECT",
                "reason": "Order intent requires a valid symbol.",
                "checks": checks,
            }

        checks.append(
            {
                "check": "symbol",
                "status": "PASS",
            }
        )

        direction = order_intent.get("direction")
        if direction not in {"LONG", "SHORT"}:
            return {
                "status": "REJECT",
                "reason": "Order intent requires LONG or SHORT direction.",
                "checks": checks,
            }

        checks.append(
            {
                "check": "direction",
                "status": "PASS",
            }
        )

        order_type = order_intent.get("order_type")
        if order_type not in self.SUPPORTED_ORDER_TYPES:
            return {
                "status": "REJECT",
                "reason": (
                    "Order intent contains an unsupported order type."
                ),
                "checks": checks,
            }

        checks.append(
            {
                "check": "order_type",
                "status": "PASS",
            }
        )

        quantity = order_intent.get("quantity")
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            return {
                "status": "REJECT",
                "reason": (
                    "Order intent requires a positive integer quantity."
                ),
                "checks": checks,
            }

        checks.append(
            {
                "check": "quantity",
                "status": "PASS",
            }
        )

        entry = order_intent.get("entry")

        if order_type == "MARKET":
            if entry is not None:
                if not self._is_positive_number(entry):
                    return {
                        "status": "REJECT",
                        "reason": (
                            "MARKET order entry must be positive when supplied."
                        ),
                        "checks": checks,
                    }
        else:
            if not self._is_positive_number(entry):
                return {
                    "status": "REJECT",
                    "reason": (
                        "Non-MARKET orders require a positive entry price."
                    ),
                    "checks": checks,
                }

        checks.append(
            {
                "check": "entry",
                "status": "PASS",
            }
        )

        return {
            "status": "PASS",
            "reason": None,
            "checks": checks,
        }

    @staticmethod
    def _is_positive_number(value):
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        return value > 0