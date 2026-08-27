class FinalDecisionService:

    def __init__(
        self,
        opportunity,
        trade_validation,
        ai_confidence
    ):

        self.opportunity = opportunity
        self.trade_validation = trade_validation
        self.ai_confidence = float(ai_confidence)

    def decide(self):

        action = self.opportunity.get(
            "action",
            "WAIT"
        )

        validation_status = self.trade_validation.get(
            "status",
            "WAIT"
        )

        preferred_setup = self.opportunity.get(
            "preferred_setup",
            "WAIT"
        )

        # ----------------------------------
        # Validated BUY
        # ----------------------------------

        if (
            action == "BUY"
            and
            validation_status == "VALID"
        ):

            decision = "BUY"

            message = (
                "Trade setup is validated and "
                "currently actionable."
            )

        # ----------------------------------
        # Strong setup but not confirmed
        # ----------------------------------

        elif preferred_setup == "WAIT_FOR_BREAKOUT":

            decision = "WAIT"

            message = (
                "Bullish setup detected. "
                "Wait for breakout confirmation."
            )

        # ----------------------------------
        # Pullback opportunity
        # ----------------------------------

        elif preferred_setup == "PULLBACK":

            decision = "WAIT"

            message = (
                "Wait for the preferred pullback "
                "entry near support."
            )

        # ----------------------------------
        # Everything else
        # ----------------------------------

        else:

            decision = "WAIT"

            message = (
                "No high-quality trade setup "
                "is currently confirmed."
            )

        return {
    "decision": decision,
    "message": message,
    "preferred_setup": preferred_setup,
    "validation_status": validation_status,
    "ai_confidence": round(
        self.ai_confidence,
        2
    ),

    "breakout_trigger": self.opportunity.get(
        "breakout_trigger"
    ),

    "breakout_level": self.opportunity.get(
        "breakout_level"
    )
}
