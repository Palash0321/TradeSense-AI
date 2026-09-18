class RiskEngine:
    """
    Structural risk gate for a canonical Trade Candidate.

    This engine does NOT:
    - calculate risk-per-trade %
    - calculate position size
    - modify strategy signals
    - apply the historical Early-Adverse filter
    - place orders
    - execute trades

    It only validates whether a trade candidate is structurally
    suitable to proceed to the next stage.
    """

    def __init__(self, trade_candidate):
        self.trade_candidate = trade_candidate or {}

    def evaluate(self):
        candidate = self.trade_candidate

        reasons = []
        checks = {}

        # ============================================================
        # 1. Actionability
        # ============================================================

        actionable = candidate.get("actionable") is True
        checks["actionable"] = actionable

        if not actionable:
            reasons.append("Trade candidate is not actionable.")

        # ============================================================
        # 2. Direction
        # ============================================================

        direction = candidate.get("direction")
        decision = candidate.get("decision")

        valid_direction = direction in ["LONG", "SHORT"]
        checks["direction_valid"] = valid_direction

        if not valid_direction:
            reasons.append(
                "Direction must be LONG or SHORT."
            )

        # ============================================================
        # 3. Decision / Direction consistency
        # ============================================================

        direction_consistent = True

        if decision == "BUY" and direction != "LONG":
            direction_consistent = False

        if decision == "SELL" and direction != "SHORT":
            direction_consistent = False

        checks["direction_consistent"] = direction_consistent

        if not direction_consistent:
            reasons.append(
                "Decision and direction are inconsistent."
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
        # 6. Risk Geometry
        # ============================================================

        risk = None
        positive_risk = False

        if valid_entry and valid_stop_loss:
            if direction == "LONG":
                risk = entry - stop_loss
            elif direction == "SHORT":
                risk = stop_loss - entry

            if risk is not None:
                positive_risk = risk > 0

        checks["positive_risk"] = positive_risk

        if not positive_risk:
            reasons.append(
                "Stop-loss geometry does not produce positive risk."
            )

        # ============================================================
        # 7. Target Validation
        # ============================================================

        targets = [
            candidate.get("target1"),
            candidate.get("target2"),
            candidate.get("target3"),
        ]

        valid_targets = True

        for target in targets:
            if target is not None and not self._is_positive_number(target):
                valid_targets = False

        checks["targets_numeric"] = valid_targets

        if not valid_targets:
            reasons.append(
                "Targets must be positive numeric values when provided."
            )

        # ============================================================
        # 8. Target Ordering
        # ============================================================

        target_order_valid = self._validate_target_order(
            direction=direction,
            targets=targets
        )

        checks["target_order_valid"] = target_order_valid

        if not target_order_valid:
            reasons.append(
                "Target ordering is inconsistent with trade direction."
            )

        # ============================================================
        # 9. Target Direction Relative To Entry
        # ============================================================

        target_direction_valid = self._validate_target_direction(
            direction=direction,
            entry=entry,
            targets=targets
        )

        checks["target_direction_valid"] = target_direction_valid

        if not target_direction_valid:
            reasons.append(
                "Targets are not positioned favorably relative to entry."
            )

        # ============================================================
        # 10. R:R Structure
        # ============================================================

        risk_reward = candidate.get("risk_reward")

        rr_valid, rr_reasons = self._validate_risk_reward(
            risk_reward=risk_reward,
            targets=targets
        )

        checks["risk_reward_valid"] = rr_valid

        if not rr_valid:
            reasons.extend(rr_reasons)

        # ============================================================
        # Final Decision
        # ============================================================

        passed = len(reasons) == 0

        return {
            "risk_decision": "PASS" if passed else "REJECT",
            "passed": passed,
            "symbol": candidate.get("symbol"),
            "direction": direction,
            "decision": decision,
            "entry": entry,
            "stop_loss": stop_loss,
            "risk": risk,
            "target1": candidate.get("target1"),
            "target2": candidate.get("target2"),
            "target3": candidate.get("target3"),
            "risk_reward": risk_reward,
            "checks": checks,
            "reasons": reasons,
            "source": "RiskEngine",
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

    @staticmethod
    def _validate_target_order(direction, targets):
        supplied_targets = [
            target for target in targets
            if target is not None
        ]

        if len(supplied_targets) <= 1:
            return True

        if direction == "LONG":
            return all(
                supplied_targets[i] < supplied_targets[i + 1]
                for i in range(len(supplied_targets) - 1)
            )

        if direction == "SHORT":
            return all(
                supplied_targets[i] > supplied_targets[i + 1]
                for i in range(len(supplied_targets) - 1)
            )

        return False

    @staticmethod
    def _validate_target_direction(direction, entry, targets):
        if entry is None:
            return False

        supplied_targets = [
            target for target in targets
            if target is not None
        ]

        if not supplied_targets:
            return True

        if direction == "LONG":
            return all(target > entry for target in supplied_targets)

        if direction == "SHORT":
            return all(target < entry for target in supplied_targets)

        return False

    @staticmethod
    def _validate_risk_reward(risk_reward, targets):
        reasons = []

        if risk_reward is None:
            return True, reasons

        if not isinstance(risk_reward, dict):
            return False, [
                "Risk/reward structure must be a dictionary."
            ]

        supplied_targets = {
            "target1": targets[0],
            "target2": targets[1],
            "target3": targets[2],
        }

        for target_name, target_value in supplied_targets.items():
            rr_value = risk_reward.get(target_name)

            if target_value is None:
                if rr_value is not None:
                    reasons.append(
                        f"{target_name} has R:R data but no target value."
                    )
                continue

            if rr_value is None:
                reasons.append(
                    f"{target_name} is missing its R:R value."
                )
                continue

            if isinstance(rr_value, bool):
                reasons.append(
                    f"{target_name} R:R must be numeric."
                )
                continue

            if not isinstance(rr_value, (int, float)):
                reasons.append(
                    f"{target_name} R:R must be numeric."
                )
                continue

            if rr_value <= 0:
                reasons.append(
                    f"{target_name} R:R must be greater than zero."
                )

        return len(reasons) == 0, reasons