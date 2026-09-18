import math


class PositionSizingEngine:
    """
    Calculates position quantity from an approved trade candidate
    and an explicit monetary risk budget.

    This engine does NOT:
    - choose the risk-per-trade percentage
    - modify strategy signals
    - calculate or modify stop loss
    - approve/reject strategy logic
    - place orders
    - execute trades
    - enforce portfolio-level exposure limits

    It only converts:
        risk budget + entry/stop geometry
    into:
        quantity + position value + actual risk
    """

    def __init__(self, trade_candidate, risk_result, risk_budget):
        self.trade_candidate = trade_candidate or {}
        self.risk_result = risk_result or {}
        self.risk_budget = risk_budget

    def calculate(self):
        candidate = self.trade_candidate
        risk_result = self.risk_result

        reasons = []
        checks = {}

        # ============================================================
        # 1. Risk Gate
        # ============================================================

        risk_passed = risk_result.get("risk_decision") == "PASS"
        checks["risk_gate_passed"] = risk_passed

        if not risk_passed:
            reasons.append(
                "Position sizing requires a PASS decision from RiskEngine."
            )

        # ============================================================
        # 2. Risk Budget
        # ============================================================

        valid_risk_budget = self._is_positive_number(
            self.risk_budget
        )

        checks["risk_budget_valid"] = valid_risk_budget

        if not valid_risk_budget:
            reasons.append(
                "Risk budget must be a positive numeric value."
            )

        # ============================================================
        # 3. Direction
        # ============================================================

        direction = candidate.get("direction")

        valid_direction = direction in ["LONG", "SHORT"]
        checks["direction_valid"] = valid_direction

        if not valid_direction:
            reasons.append(
                "Direction must be LONG or SHORT."
            )

        # ============================================================
        # 4. Entry
        # ============================================================

        entry = candidate.get("entry")

        valid_entry = self._is_positive_number(entry)
        checks["entry_valid"] = valid_entry

        if not valid_entry:
            reasons.append(
                "Entry must be a positive numeric value."
            )

        # ============================================================
        # 5. Stop Loss
        # ============================================================

        stop_loss = candidate.get("stop_loss")

        valid_stop_loss = self._is_positive_number(stop_loss)
        checks["stop_loss_valid"] = valid_stop_loss

        if not valid_stop_loss:
            reasons.append(
                "Stop loss must be a positive numeric value."
            )

        # ============================================================
        # 6. Risk Per Unit
        # ============================================================

        risk_per_unit = None
        positive_risk_per_unit = False

        if valid_entry and valid_stop_loss and valid_direction:
            if direction == "LONG":
                risk_per_unit = entry - stop_loss
            else:
                risk_per_unit = stop_loss - entry

            positive_risk_per_unit = risk_per_unit > 0

        checks["positive_risk_per_unit"] = positive_risk_per_unit

        if not positive_risk_per_unit:
            reasons.append(
                "Entry and stop-loss geometry does not produce positive risk per unit."
            )

        # ============================================================
        # Stop here if prerequisite validation failed
        # ============================================================

        prerequisites_valid = (
            risk_passed
            and valid_risk_budget
            and valid_direction
            and valid_entry
            and valid_stop_loss
            and positive_risk_per_unit
        )

        if not prerequisites_valid:
            return {
                "sizing_decision": "REJECT",
                "passed": False,
                "symbol": candidate.get("symbol"),
                "direction": direction,
                "entry": entry,
                "stop_loss": stop_loss,
                "risk_per_unit": risk_per_unit,
                "risk_budget": self.risk_budget,
                "quantity": None,
                "position_value": None,
                "actual_risk": None,
                "checks": checks,
                "reasons": reasons,
                "source": "PositionSizingEngine",
            }

        # ============================================================
        # 7. Quantity Calculation
        # ============================================================

        quantity = math.floor(
            self.risk_budget / risk_per_unit
        )

        valid_quantity = (
            isinstance(quantity, int)
            and quantity > 0
        )

        checks["quantity_valid"] = valid_quantity

        if not valid_quantity:
            reasons.append(
                "Risk budget is insufficient to purchase at least one unit."
            )

            return {
                "sizing_decision": "REJECT",
                "passed": False,
                "symbol": candidate.get("symbol"),
                "direction": direction,
                "entry": entry,
                "stop_loss": stop_loss,
                "risk_per_unit": risk_per_unit,
                "risk_budget": self.risk_budget,
                "quantity": quantity,
                "position_value": None,
                "actual_risk": None,
                "checks": checks,
                "reasons": reasons,
                "source": "PositionSizingEngine",
            }

        # ============================================================
        # 8. Position Value
        # ============================================================

        position_value = quantity * entry

        valid_position_value = position_value > 0
        checks["position_value_valid"] = valid_position_value

        if not valid_position_value:
            reasons.append(
                "Position value could not be calculated correctly."
            )

        # ============================================================
        # 9. Actual Risk
        # ============================================================

        actual_risk = quantity * risk_per_unit

        valid_actual_risk = actual_risk > 0
        checks["actual_risk_valid"] = valid_actual_risk

        if not valid_actual_risk:
            reasons.append(
                "Actual position risk could not be calculated correctly."
            )

        # ============================================================
        # 10. Risk Budget Constraint
        # ============================================================

        risk_within_budget = actual_risk <= self.risk_budget
        checks["risk_within_budget"] = risk_within_budget

        if not risk_within_budget:
            reasons.append(
                "Calculated actual risk exceeds the supplied risk budget."
            )

        # ============================================================
        # Final Decision
        # ============================================================

        passed = len(reasons) == 0

        return {
            "sizing_decision": "PASS" if passed else "REJECT",
            "passed": passed,
            "symbol": candidate.get("symbol"),
            "direction": direction,
            "entry": entry,
            "stop_loss": stop_loss,
            "risk_per_unit": risk_per_unit,
            "risk_budget": self.risk_budget,
            "quantity": quantity,
            "position_value": position_value,
            "actual_risk": actual_risk,
            "checks": checks,
            "reasons": reasons,
            "source": "PositionSizingEngine",
        }

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _is_positive_number(value):
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        return value > 0