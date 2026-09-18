class TradeCandidateService:
    """
    Converts the existing FinalDecisionService output into one canonical
    trade-candidate contract.

    This service does NOT:
    - calculate entries
    - calculate stop losses
    - calculate targets
    - change strategy logic
    - approve/reject risk
    - calculate position size
    - place orders

    It only normalizes the already-existing decision and setup information.
    """

    def __init__(
        self,
        symbol,
        final_decision,
        setup=None,
        ai_confidence=None,
    ):
        self.symbol = symbol
        self.final_decision = final_decision or {}
        self.setup = setup or {}
        self.ai_confidence = ai_confidence

    def build(self):
        decision = self.final_decision.get(
            "decision",
            "WAIT"
        )

        preferred_setup = self.final_decision.get(
            "preferred_setup",
            "NO_SETUP"
        )

        validation_status = self.final_decision.get(
            "validation_status",
            "WAIT"
        )

        setup_details = self.final_decision.get(
            "setup_details"
        ) or {}

        setup_type = setup_details.get(
            "type",
            "NONE"
        )

        direction = self.setup.get(
            "direction"
        )

        if direction not in ["LONG", "SHORT"]:
            if decision == "BUY":
                direction = "LONG"
            elif decision == "SELL":
                direction = "SHORT"

        entry = setup_details.get("entry")

        entry_low = setup_details.get(
            "entry_low"
        )

        entry_high = setup_details.get(
            "entry_high"
        )

        stop_loss = setup_details.get(
            "stop_loss"
        )

        target1 = setup_details.get(
            "target1"
        )

        target2 = setup_details.get(
            "target2"
        )

        target3 = setup_details.get(
            "target3"
        )

        risk_reward = setup_details.get(
            "risk_reward"
        )

        breakout_trigger = self.final_decision.get(
            "breakout_trigger"
        )

        breakout_level = self.final_decision.get(
            "breakout_level"
        )

        confidence = self.final_decision.get(
            "ai_confidence",
            self.ai_confidence
        )

        if confidence is not None:
            confidence = round(
                float(confidence),
                2
            )

        actionable = decision in [
            "BUY",
            "SELL"
        ]

        return {
            "symbol": self.symbol,
            "direction": direction,
            "decision": decision,
            "actionable": actionable,

            "setup_type": setup_type,
            "preferred_setup": preferred_setup,

            "validation_status": validation_status,

            "entry": entry,
            "entry_low": entry_low,
            "entry_high": entry_high,

            "stop_loss": stop_loss,

            "target1": target1,
            "target2": target2,
            "target3": target3,

            "risk_reward": risk_reward,

            "ai_confidence": confidence,

            "breakout_trigger": breakout_trigger,
            "breakout_level": breakout_level,

            "source": "FinalDecisionService",
        }