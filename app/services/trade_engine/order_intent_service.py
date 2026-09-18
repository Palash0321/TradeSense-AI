class OrderIntentService:
    """
    Builds a canonical Order Intent from an approved Trade Candidate,
    RiskEngine result, and PositionSizingEngine result.

    This service does NOT:
    - place orders
    - communicate with a broker
    - execute trades
    - modify strategy logic
    - modify entry or stop loss
    - calculate position size
    - enforce portfolio exposure

    It only creates the execution-ready instruction contract.
    """

    def __init__(
        self,
        trade_candidate,
        risk_result,
        sizing_result,
    ):
        self.trade_candidate = trade_candidate or {}
        self.risk_result = risk_result or {}
        self.sizing_result = sizing_result or {}

    def build(self):
        candidate = self.trade_candidate
        risk_result = self.risk_result
        sizing_result = self.sizing_result

        reasons = []
        checks = {}

        # ============================================================
        # 1. Risk Gate
        # ============================================================

        risk_passed = (
            risk_result.get("risk_decision") == "PASS"
        )

        checks["risk_gate_passed"] = risk_passed

        if not risk_passed:
            reasons.append(
                "Order intent requires a PASS decision from RiskEngine."
            )

            return {
                "intent_decision": "REJECT",
                "ready_for_execution": False,
                "symbol": candidate.get("symbol"),
                "direction": candidate.get("direction"),
                "order_type": None,
                "quantity": None,
                "entry": candidate.get("entry"),
                "stop_loss": candidate.get("stop_loss"),
                "target1": candidate.get("target1"),
                "target2": candidate.get("target2"),
                "target3": candidate.get("target3"),
                "risk_budget": sizing_result.get("risk_budget"),
                "risk_per_unit": sizing_result.get("risk_per_unit"),
                "actual_risk": sizing_result.get("actual_risk"),
                "position_value": sizing_result.get("position_value"),
                "checks": checks,
                "reasons": reasons,
                "source": "OrderIntentService",
            }

        # ============================================================
        # 2. Position Sizing Gate
        # ============================================================

        sizing_passed = (
            sizing_result.get("sizing_decision") == "PASS"
        )

        checks["sizing_gate_passed"] = sizing_passed

        if not sizing_passed:
            reasons.append(
                "Order intent requires a PASS decision from PositionSizingEngine."
            )

            return {
                "intent_decision": "REJECT",
                "ready_for_execution": False,
                "symbol": candidate.get("symbol"),
                "direction": candidate.get("direction"),
                "order_type": None,
                "quantity": sizing_result.get("quantity"),
                "entry": candidate.get("entry"),
                "stop_loss": candidate.get("stop_loss"),
                "target1": candidate.get("target1"),
                "target2": candidate.get("target2"),
                "target3": candidate.get("target3"),
                "risk_budget": sizing_result.get("risk_budget"),
                "risk_per_unit": sizing_result.get("risk_per_unit"),
                "actual_risk": sizing_result.get("actual_risk"),
                "position_value": sizing_result.get("position_value"),
                "checks": checks,
                "reasons": reasons,
                "source": "OrderIntentService",
            }

        # ============================================================
        # 3. Symbol
        # ============================================================

        symbol = candidate.get("symbol")

        valid_symbol = (
            isinstance(symbol, str)
            and bool(symbol.strip())
        )

        checks["symbol_valid"] = valid_symbol

        if not valid_symbol:
            reasons.append(
                "Symbol must be a non-empty string."
            )

        # ============================================================
        # 4. Direction
        # ============================================================

        direction = candidate.get("direction")

        valid_direction = direction in ["LONG", "SHORT"]

        checks["direction_valid"] = valid_direction

        if not valid_direction:
            reasons.append(
                "Direction must be LONG or SHORT."
            )

        # ============================================================
        # 5. Quantity
        # ============================================================

        quantity = sizing_result.get("quantity")

        valid_quantity = (
            isinstance(quantity, int)
            and not isinstance(quantity, bool)
            and quantity > 0
        )

        checks["quantity_valid"] = valid_quantity

        if not valid_quantity:
            reasons.append(
                "Order quantity must be a positive whole number."
            )

        # ============================================================
        # 6. Entry
        # ============================================================

        entry = candidate.get("entry")

        valid_entry = self._is_positive_number(entry)

        checks["entry_valid"] = valid_entry

        if not valid_entry:
            reasons.append(
                "Entry must be a positive numeric value."
            )

        # ============================================================
        # 7. Stop Loss
        # ============================================================

        stop_loss = candidate.get("stop_loss")

        valid_stop_loss = self._is_positive_number(stop_loss)

        checks["stop_loss_valid"] = valid_stop_loss

        if not valid_stop_loss:
            reasons.append(
                "Stop loss must be a positive numeric value."
            )

        # ============================================================
        # 8. Order Type
        # ============================================================

        order_type = self._determine_order_type(candidate)

        valid_order_type = order_type in [
            "MARKET",
            "LIMIT",
            "STOP",
        ]

        checks["order_type_valid"] = valid_order_type

        if not valid_order_type:
            reasons.append(
                "Order type must be MARKET, LIMIT, or STOP."
            )

        # ============================================================
        # 9. Local Contract Validation
        # ============================================================

        local_validation_passed = (
            valid_symbol
            and valid_direction
            and valid_quantity
            and valid_entry
            and valid_stop_loss
            and valid_order_type
        )

        checks["local_validation_passed"] = local_validation_passed

        if not local_validation_passed:
            return {
                "intent_decision": "REJECT",
                "ready_for_execution": False,
                "symbol": symbol,
                "direction": direction,
                "order_type": order_type,
                "quantity": quantity,
                "entry": entry,
                "stop_loss": stop_loss,
                "target1": candidate.get("target1"),
                "target2": candidate.get("target2"),
                "target3": candidate.get("target3"),
                "risk_budget": sizing_result.get("risk_budget"),
                "risk_per_unit": sizing_result.get("risk_per_unit"),
                "actual_risk": sizing_result.get("actual_risk"),
                "position_value": sizing_result.get("position_value"),
                "checks": checks,
                "reasons": reasons,
                "source": "OrderIntentService",
            }

        # ============================================================
        # 10. Final Intent
        # ============================================================

        return {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": symbol,
            "direction": direction,
            "order_type": order_type,
            "quantity": quantity,
            "entry": entry,
            "stop_loss": stop_loss,
            "target1": candidate.get("target1"),
            "target2": candidate.get("target2"),
            "target3": candidate.get("target3"),
            "risk_budget": sizing_result.get("risk_budget"),
            "risk_per_unit": sizing_result.get("risk_per_unit"),
            "actual_risk": sizing_result.get("actual_risk"),
            "position_value": sizing_result.get("position_value"),
            "checks": checks,
            "reasons": reasons,
            "source": "OrderIntentService",
        }

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _determine_order_type(candidate):
        """
        Determine the execution instruction from the existing
        candidate structure without changing strategy logic.

        Current default:
            MARKET

        Existing breakout candidates:
            STOP

        Existing pullback entry ranges:
            LIMIT
        """

        if candidate.get("breakout_trigger") is True:
            return "STOP"

        if candidate.get("entry_low") is not None:
            return "LIMIT"

        return "MARKET"

    @staticmethod
    def _is_positive_number(value):
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        return value > 0