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
        self.setup_risk_reward = (
            setup_risk_reward
            if setup_risk_reward is not None
            else float(
                self.opportunity.get(
                    "risk_reward",
                    0
                ) or 0
            )
        )

    def validate(self):

        action = self.opportunity.get(
            "action",
            "WAIT"
        )

        preferred = self.opportunity.get(
            "preferred_setup",
            "WAIT"
        )

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

        breakout_ok = (
            breakout_score >= 70
        )

        checks.append({
            "name": "Breakout Confirmation",
            "passed": breakout_ok
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
    and
    rr_ok
    and
    mtf_ok
    and
    confidence_ok
    and
    breakout_ok
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

            message = (
                "Trade validation failed: "
                + ", ".join(failed)
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