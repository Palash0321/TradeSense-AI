class OpportunityService:

    def __init__(
        self,
        analysis,
        support,
        resistance,
        current_price,
        risk_reward=None,
        volume=None,
        patterns=None,
        multi_timeframe=None
    ):

        self.analysis = analysis
        self.support = float(support)
        self.resistance = float(resistance)
        self.price = float(current_price)
        self.risk_reward = risk_reward or {}

        self.volume = volume or {}
        self.patterns = patterns or []
        self.multi_timeframe = multi_timeframe or {}

    def analyze(self):

        recommendation = self.analysis.get(
            "recommendation",
            "HOLD"
        )

        if self.price <= 0:

            return {
                "status": "WAIT",
                "action": "WAIT",
                "score": 0,
                "message": "Invalid current price.",
                "reason": "Current price is unavailable.",
                "pullback_trigger": None,
                "breakout_trigger": None
            }

        support_distance = (
            (self.price - self.support)
            / self.price
        ) * 100

        resistance_distance = (
            (self.resistance - self.price)
            / self.price
        ) * 100

        rr_ratio = float(
            self.risk_reward.get("ratio", 0)
            or 0
        )

        breakout_setup_rr = 1.5

        pullback_trigger = round(
            self.support,
            2
        )

        breakout_trigger = round(
            self.resistance,
            2
        )

        # ----------------------------------
        # Non-bullish setup
        # ----------------------------------

        if recommendation not in [
            "BUY",
            "STRONG BUY",
            "ACCUMULATE"
        ]:

            return {
                "status": "WAIT",
                "action": "WAIT",
                "score": 30,
                "message": "No immediate bullish setup.",
                "reason": (
                    "Wait for a stronger bullish setup "
                    "near support or a confirmed breakout."
                ),

                "preferred_setup": "NO_SETUP",
                
                "trigger_price": pullback_trigger,
                "pullback_trigger": pullback_trigger,
                "breakout_trigger": breakout_trigger,
                "support_distance": round(
                    support_distance,
                    2
                ),
                "resistance_distance": round(
                    resistance_distance,
                    2
                ),
                "risk_reward": rr_ratio
            }

        # ----------------------------------
        # Opportunity scoring
        # ----------------------------------

        score = 100
        reasons = []

        # ----------------------------------
        # Breakout confirmation evidence
        # ----------------------------------

        volume_ratio = float(
            self.volume.get("ratio", 0) or 0
        )

        bullish_patterns = [
            "Hammer",
            "Bullish Engulfing"
        ]

        bearish_patterns = [
            "Shooting Star",
            "Bearish Engulfing"
        ]

        bullish_candle = any(
            pattern in bullish_patterns
            for pattern in self.patterns
        )

        bearish_candle = any(
            pattern in bearish_patterns
            for pattern in self.patterns
        )

        breakout_confirmation_score = 0

        # ----------------------------------
        # Multi-Timeframe confirmation
        # ----------------------------------

        mtf_probability = float(
            self.multi_timeframe.get(
                "overall_probability",
                0
            ) or 0
        )

        if mtf_probability >= 80:

            breakout_confirmation_score += 30

            reasons.append(
                "Strong multi-timeframe bullish alignment."
            )

        elif mtf_probability >= 60:

            breakout_confirmation_score += 20

            reasons.append(
                "Most major timeframes support the bullish setup."
            )

        elif mtf_probability <= 40:

            breakout_confirmation_score -= 20

            reasons.append(
                "Multi-timeframe alignment is weak."
            )

        if volume_ratio >= 2.0:

            breakout_confirmation_score += 40

            reasons.append(
                "Very high volume supports a potential breakout."
            )

        elif volume_ratio >= 1.5:

            breakout_confirmation_score += 30

            reasons.append(
                "High volume supports a potential breakout."
            )

        elif volume_ratio >= 1.0:

            breakout_confirmation_score += 15

            reasons.append(
                "Normal volume provides limited breakout confirmation."
            )

        else:

            reasons.append(
                "Low volume does not confirm a breakout."
            )

        if bullish_candle:

            breakout_confirmation_score += 30

            reasons.append(
                "Bullish candlestick confirmation detected."
            )

        if bearish_candle:

            breakout_confirmation_score -= 30

            reasons.append(
                "Bearish candlestick pattern weakens breakout confirmation."
            )

        breakout_confirmation_score = max(
            0,
            min(
                100,
                breakout_confirmation_score
            )
        )

        if rr_ratio < 1.0:

            score -= 35

            reasons.append(
                "Risk/reward is below 1.0."
            )

        elif rr_ratio < 1.5:

            score -= 20

            reasons.append(
                "Risk/reward is below the preferred 1.5 level."
            )

        elif rr_ratio < 2.0:

            score -= 10

            reasons.append(
                "Risk/reward is acceptable but below 2.0."
            )

        else:

            reasons.append(
                "Risk/reward is favorable."
            )

        # ----------------------------------
        # Resistance proximity
        # ----------------------------------

        if resistance_distance <= 2:

            score -= 30

            reasons.append(
                "Price is very close to resistance."
            )

        elif resistance_distance <= 5:

            score -= 15

            reasons.append(
                "Price is approaching resistance."
            )

        else:

            reasons.append(
                "Sufficient room remains before resistance."
            )

        # ----------------------------------
        # Support proximity
        # ----------------------------------

        if support_distance <= 5:

            score += 5

            reasons.append(
                "Price is relatively close to support."
            )


                # ----------------------------------
        # Breakout State
        # ----------------------------------

        breakout_buffer = self.resistance * 0.0025

        breakout_level = (
            self.resistance
            + breakout_buffer
        )

        if self.price < self.resistance:

            breakout_state = "BELOW_RESISTANCE"

        elif self.price < breakout_level:

            breakout_state = "BREAKOUT_ATTEMPT"

        else:

            # Price has moved above resistance.
            # Confirmation still requires sufficient
            # risk/reward and confirmation evidence.

            if (
                breakout_confirmation_score >= 70
                and breakout_setup_rr >= 1.5
            ):

                breakout_state = "BREAKOUT_CONFIRMED"

            else:

                breakout_state = "BREAKOUT_ATTEMPT"

        # ----------------------------------
        # Final score
        # ----------------------------------

        score = max(
            0,
            min(100, score)
        )

        # ----------------------------------
        # Actionability
        # ----------------------------------

        if (
            score >= 70
            and
            rr_ratio >= 1.5
            and
            resistance_distance > 2
        ):

            status = "READY"
            action = "BUY"

            message = (
                "Trade setup is currently actionable."
            )

            trigger_price = round(
                self.price,
                2
            )

        else:

            status = "WAIT"
            action = "WAIT"

            if resistance_distance <= 2:

                message = (
                    "Bullish setup exists, but price is "
                    "too close to resistance."
                )

                trigger_price = breakout_trigger

            elif rr_ratio < 1.5:

                message = (
                    "Bullish setup exists, but the "
                    "current risk/reward is not attractive."
                )

                trigger_price = pullback_trigger

            else:

                message = (
                    "Bullish setup exists, but the "
                    "entry is not currently optimal."
                )

                trigger_price = pullback_trigger

        # ----------------------------------
        # Preferred Setup
        # ----------------------------------

        if (
            action == "BUY"
            and
            breakout_confirmation_score >= 70
            and
            breakout_state == "BREAKOUT_CONFIRMED"
        ):

            preferred_setup = "BREAKOUT"

            setup_message = (
                "Breakout confirmed. Trade can be considered."
            )

        elif (
            support_distance <= 5
            and
            rr_ratio >= 1.5
        ):

            preferred_setup = "PULLBACK"

            setup_message = (
                "Pullback setup is attractive near support."
            )

        elif (
            resistance_distance <= 5
            and
            breakout_confirmation_score <= 70
        ):

            preferred_setup = "WAIT_FOR_BREAKOUT"

            setup_message = (
                "Wait for a confirmed breakout above resistance."
            )

        else:

            preferred_setup = "WAIT"

            setup_message = (
                "No high-quality entry is currently confirmed."
            )

        return {
            "status": status,
            "action": action,
            "score": score,
            "message": message,
            "reason": " ".join(reasons),

            "trigger_price": trigger_price,

            "pullback_trigger": pullback_trigger,

            "breakout_trigger": breakout_trigger,

            "support_distance": round(
                support_distance,
                2
            ),

            "resistance_distance": round(
                resistance_distance,
                2
            ),

            "risk_reward": rr_ratio,

            "mtf_probability": round(
                mtf_probability,
                1
            ),

            "breakout_confirmation_score": breakout_confirmation_score,

            "breakout_state": breakout_state,

            "breakout_level": round(
                breakout_level,
                2
            ),

            "current_price": round(
                self.price,
                2
            ),

            "volume_ratio": round(
                volume_ratio,
                2
            ),

            "bullish_candle_confirmation": bullish_candle,

            "bearish_candle_confirmation": bearish_candle,

            "preferred_setup": preferred_setup,
            "setup_message": setup_message,
        }
