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

        setup_direction = self.opportunity.get(
            "setup_direction"
        )

        setup_type = self.opportunity.get(
            "setup_type",
            preferred
        )

        setup_confidence = float(
            self.opportunity.get(
                "setup_confidence",
                0
            ) or 0
        )

        is_long_setup = setup_type in [
            "LONG_CONTINUATION",
            "LONG_REVERSAL"
        ]

        is_short_setup = setup_type in [
            "SHORT_CONTINUATION",
            "SHORT_REVERSAL"
        ]

        is_continuation = setup_type in [
            "LONG_CONTINUATION",
            "SHORT_CONTINUATION"
        ]

        is_reversal = setup_type in [
            "LONG_REVERSAL",
            "SHORT_REVERSAL"
        ]

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
        # Directional Confirmation
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

        # ----------------------------------
        # Setup-specific confirmation
        # ----------------------------------

        if is_reversal:

            directional_confirmation_ok = (
                setup_direction in [
                    "LONG",
                    "SHORT"
                ]
                and
                setup_confidence >= 60
            )

        elif setup_direction == "SHORT":

            directional_confirmation_ok = (
                breakout_level > 0
                and
                current_price <= breakout_level
            )

        else:

            directional_confirmation_ok = (
                breakout_state == "BREAKOUT_CONFIRMED"
                and
                breakout_level > 0
                and
                current_price >= breakout_level
            )

        # ----------------------------------
        # Setup-specific validation checks
        # ----------------------------------

        if is_reversal:

            checks.append({
                "name": "Directional Confirmation",
                "passed": directional_confirmation_ok
            })

        else:

            checks.append({
                "name": "Breakout Evidence",
                "passed": breakout_evidence_ok
            })

            checks.append({
                "name": "Directional Confirmation",
                "passed": directional_confirmation_ok
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

        expected_action = (
            "SELL"
            if setup_direction == "SHORT"
            else "BUY"
        )

        confirmation_ok = (
            directional_confirmation_ok
            if is_reversal
            else (
                breakout_evidence_ok
                and
                directional_confirmation_ok
            )
        )

        if (
            action == expected_action
            and
            rr_ok
            and
            mtf_ok
            and
            confidence_ok
            and
            confirmation_ok
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
                not directional_confirmation_ok
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

            "setup_type": setup_type,

            "setup_direction": setup_direction,

            "setup_confidence": setup_confidence,

            "confirmation_type": (
                "REVERSAL"
                if is_reversal
                else "BREAKOUT"
            ),

            "risk_reward": rr

        }
