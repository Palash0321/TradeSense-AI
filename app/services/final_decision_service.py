class FinalDecisionService:

    def __init__(
        self,
        opportunity,
        trade_validation,
        ai_confidence,
        entry_engine,
        setup_details=None
    ):

        self.opportunity = opportunity
        self.trade_validation = trade_validation
        self.ai_confidence = float(ai_confidence)
        self.entry_engine = entry_engine
        self.setup_details = setup_details

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

        setup_details = None

        current_price = float(
            self.opportunity.get(
                "current_price",
                0
            ) or 0
        )

        breakout_level = float(
            self.opportunity.get(
                "breakout_level",
                0
            ) or 0
        )

        breakout_confirmed = (
            breakout_level > 0
            and current_price >= breakout_level
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

            if (
                breakout_confirmed
                and
                validation_status == "VALID"
            ):

                decision = "BUY"

                message = (
                    "Breakout confirmed and "
                    "trade setup is validated."
                )

            elif breakout_confirmed:

                decision = "WAIT"

                message = (
                    "Breakout level crossed, but "
                    "trade validation is still pending."
                )

            else:

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

            setup_details = None

        if preferred_setup in [
    "BREAKOUT",
    "WAIT_FOR_BREAKOUT"
]:

            setup = self.entry_engine.get(
                "breakout",
                {}
            )

            setup_details = {
                "type": "BREAKOUT",
                "entry": setup.get("entry"),
                "stop_loss": setup.get("stop_loss"),
                "target1": setup.get("target1"),
                "target2": setup.get("target2"),
                "target3": setup.get("target3"),
                "risk_reward": setup.get(
                    "risk_reward",
                    {}
                )
            }

        elif preferred_setup == "PULLBACK":

            setup = self.entry_engine.get(
                "pullback",
                {}
            )

            setup_details = {
                "type": "PULLBACK",
                "entry_low": setup.get("entry_low"),
                "entry_high": setup.get("entry_high"),
                "stop_loss": setup.get("stop_loss"),
                "target1": setup.get("target1"),
                "target2": setup.get("target2"),
                "target3": setup.get("target3"),
                "risk_reward": setup.get(
                    "risk_reward",
                    {}
                )
            }

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
    ),

    "setup_details": setup_details
}
