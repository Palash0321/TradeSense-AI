class TradeValidationService:

    def __init__(
        self,
        opportunity,
        entry_engine,
        ai_confidence,
        setup_risk_reward=None
    ):

        self.opportunity = opportunity
        self.entry_engine = entry_engine
        self.ai_confidence = float(ai_confidence)
        self.setup_risk_reward = setup_risk_reward

    def validate(self):

        action = self.opportunity.get(
            "action",
            "WAIT"
        )

        preferred = self.opportunity.get(
            "preferred_setup",
            "WAIT"
        )

        if preferred in ["NO_SETUP", "WAIT"]:

            return {
                "status": "NOT_APPLICABLE",
                "message": "No trade setup is currently available for validation.",
                "passed_checks": 0,
                "total_checks": 0,
                "checks": [],
                "preferred_setup": preferred,
                "risk_reward": 0.0
            }

        if self.setup_risk_reward is None:

            rr = 0.0

        else:

            rr = float(
                self.setup_risk_reward
            )

        breakout_score = float(
            self.opportunity.get(
                "breakout_confirmation_score",
                0
            ) or 0
        )

        mtf = float(
            self.opportunity.get(
                "mtf_probability",
                0
            ) or 0
        )

        checks = []

        # ----------------------------------
        # AI Confidence
        # ----------------------------------

        confidence_ok = self.ai_confidence >= 70

        checks.append({
            "name": "AI Confidence",
            "passed": confidence_ok
        })

        # ----------------------------------
        # Risk / Reward
        # ----------------------------------

        rr_ok = rr >= 1.5

        checks.append({
            "name": "Risk / Reward",
            "passed": rr_ok
        })

        # ----------------------------------
        # Multi Timeframe
        # ----------------------------------

        mtf_ok = mtf >= 60

        checks.append({
            "name": "Multi-Timeframe",
            "passed": mtf_ok
        })

        # ----------------------------------
        # Breakout Confirmation
        # ----------------------------------

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

        breakout_state = self.opportunity.get(
            "breakout_state",
            "BELOW_RESISTANCE"
        )

        breakout_evidence_ok = (
            breakout_score >= 70
        )

        breakout_confirmed = (
            breakout_state == "BREAKOUT_CONFIRMED"
            and
            breakout_level > 0
            and
            current_price >= breakout_level
        )

        checks.append({
            "name": "Breakout Evidence",
            "passed": breakout_evidence_ok
        })

        checks.append({
            "name": "Breakout Confirmed",
            "passed": breakout_confirmed
        })

        passed = sum(
            1
            for check in checks
            if check["passed"]
        )

        total = len(checks)

        # ----------------------------------
        # Final validation
        # ----------------------------------

        if (
            action == "BUY"
            and rr_ok
            and mtf_ok
            and confidence_ok
            and breakout_evidence_ok
            and breakout_confirmed
        ):

            status = "VALID"

            message = (
                "Trade setup passed the "
                "required validation checks."
            )

        else:

            status = "WAIT"

            failed = [
                check["name"]
                for check in checks
                if not check["passed"]
            ]

            if (
                preferred == "WAIT_FOR_BREAKOUT"
                and
                breakout_evidence_ok
                and
                not breakout_confirmed
            ):

                message = (
                    "Setup is validated, but the breakout "
                    "has not yet been confirmed."
                )

            elif failed:

                message = (
                    "Trade validation pending: "
                    + ", ".join(failed)
                )

            else:

                message = (
                    "Trade validation is pending."
                )

        return {

            "status": status,

            "message": message,

            "passed_checks": passed,

            "total_checks": total,

            "checks": checks,

            "preferred_setup": preferred,

            "risk_reward": rr

        }