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
        multi_timeframe=None,
        setup_risk_reward=None,
        market_structure=None,
        liquidity=None,
        setup=None
    ):

        self.analysis = analysis
        self.support = float(support)
        self.resistance = float(resistance)
        self.price = float(current_price)
        self.risk_reward = risk_reward or {}

        self.volume = volume or {}
        self.patterns = patterns or []
        self.multi_timeframe = multi_timeframe or {}
        self.setup_risk_reward = setup_risk_reward or {}
        self.market_structure = market_structure or {}
        self.liquidity = liquidity or {}
        self.setup = setup or {}

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

        breakout_setup_rr = float(
            self.setup_risk_reward.get(
                "breakout",
                0
            ) or 0
        )

        pullback_setup_rr = float(
            self.setup_risk_reward.get(
                "pullback",
                0
            ) or 0
        )

        pullback_trigger = round(
            self.support,
            2
        )

        breakout_trigger = round(
            self.resistance,
            2
        )

        # ----------------------------------
        # Structural Setup
        # ----------------------------------

        setup_type = self.setup.get(
            "setup",
            "WAIT"
        )

        setup_direction = self.setup.get(
            "direction"
        )

        setup_confidence = float(
            self.setup.get(
                "confidence",
                0
            ) or 0
        )

        # ----------------------------------
        # No directional setup
        # ----------------------------------

        if setup_direction not in [
            "LONG",
            "SHORT"
        ]:

            structure = self.market_structure.get(
                "structure",
                "NEUTRAL"
            )

            break_direction = self.market_structure.get(
                "break_direction"
            )

            break_confirmed = self.market_structure.get(
                "break_confirmed",
                False
            )

            # ----------------------------------
            # Base market structure
            # ----------------------------------

            if structure == "HH_HL":

                structure_score = 15

            elif structure == "PARTIAL_BULLISH":

                structure_score = 8

            elif structure == "LH_LL":

                structure_score = -15

            elif structure == "PARTIAL_BEARISH":

                structure_score = -8

            else:

                structure_score = 0

            # ----------------------------------
            # Confirmed structural event
            # ----------------------------------

            if break_confirmed:

                if break_direction == "BULLISH":

                    structure_score += 20

                elif break_direction == "BEARISH":

                    structure_score -= 20

            # ----------------------------------
            # Liquidity Confirmation
            # ----------------------------------

            liquidity_score = 0

            nearest_liquidity = self.liquidity.get(
                "nearest_liquidity",
                {}
            )

            sweep = self.liquidity.get(
                "sweep",
                {}
            )

            nearest_above = nearest_liquidity.get(
                "above"
            )

            nearest_below = nearest_liquidity.get(
                "below"
            )

            # ----------------------------------
            # Liquidity location
            # ----------------------------------

            structure_bias = self.market_structure.get(
                "bias",
                "NEUTRAL"
            )

            if structure_bias == "BULLISH":

                if nearest_below is not None:

                    liquidity_score += 5

            elif structure_bias == "BEARISH":

                if nearest_above is not None:

                    liquidity_score += 5

            # ----------------------------------
            # Liquidity sweep
            # ----------------------------------

            if sweep.get("detected"):

                sweep_direction = sweep.get(
                    "direction"
                )

                if (
                    structure_bias == "BULLISH"
                    and
                    sweep_direction == "SELL_SIDE"
                ):

                    liquidity_score += 10

                elif (
                    structure_bias == "BEARISH"
                    and
                    sweep_direction == "BUY_SIDE"
                ):

                    liquidity_score += 10

                elif (
                    structure_bias == "BULLISH"
                    and
                    sweep_direction == "BUY_SIDE"
                ):

                    liquidity_score -= 10

                elif (
                    structure_bias == "BEARISH"
                    and
                    sweep_direction == "SELL_SIDE"
                ):

                    liquidity_score -= 10

            liquidity_score = max(
                -15,
                min(
                    15,
                    liquidity_score
                )
            )

            structure_adjusted_score = max(
                0,
                min(
                    100,
                    30
                    + structure_score
                    + liquidity_score
                )
            )

            return {
                "status": "WAIT",

                "action": "WAIT",

                "score": structure_adjusted_score,

                "message": (
                    "Bearish setup detected, but no immediate short entry is confirmed."
                    if setup_direction == "SHORT"
                    else
                    "No immediate bullish setup."
                ),

                "reason": (
    (
                    "Wait for stronger bearish confirmation "
                    "before considering a short entry."
                )
                if setup_direction == "SHORT"
                else
                (
                    "Wait for a stronger bullish setup "
                    "near support or a confirmed breakout."
                )
            ),

                "market_structure": self.market_structure,

                "liquidity": self.liquidity,

                "structure_score": structure_score,

                "liquidity_score": liquidity_score,

                "structure_bias": self.market_structure.get(
                    "bias",
                    "NEUTRAL"
                ),

                "structure": self.market_structure.get(
                    "structure",
                    "NEUTRAL"
                ),

                "structure_strength": self.market_structure.get(
                    "strength",
                    0
                ),

                "structure_break_direction": break_direction,

                "structure_break_confirmed": break_confirmed,

                "preferred_setup": setup_type,

                "setup": self.setup,

                "setup_type": setup_type,

                "setup_direction": setup_direction,

                "setup_confidence": setup_confidence,

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
                "risk_reward": rr_ratio,

                "setup_risk_reward": {
                    "breakout": breakout_setup_rr,
                    "pullback": pullback_setup_rr
                },

                "mtf_probability": round(
                    float(
                        self.multi_timeframe.get(
                            "overall_probability",
                            0
                        ) or 0
                    ),
                    1
                ),

                "breakout_confirmation_score": 0,

                "breakout_state": "NO_BREAKOUT",

                "breakout_level": round(
                    breakout_trigger,
                    2
                ),

                "current_price": round(
                    self.price,
                    2
                ),

                "volume_ratio": round(
                    float(
                        self.volume.get(
                            "ratio",
                            0
                        ) or 0
                    ),
                    2
                )
            }

        # ----------------------------------
        # Opportunity scoring
        # ----------------------------------

        score = 100
        reasons = []

        # ----------------------------------
        # Market Structure Confirmation
        # ----------------------------------

        structure = self.market_structure.get(
            "structure",
            "NEUTRAL"
        )

        structure_bias = self.market_structure.get(
            "bias",
            "NEUTRAL"
        )

        break_direction = self.market_structure.get(
            "break_direction"
        )

        break_confirmed = self.market_structure.get(
            "break_confirmed",
            False
        )

        structure_score = 0

        # ----------------------------------
        # Base market structure
        # ----------------------------------

        if structure == "HH_HL":

            structure_score = 15

            reasons.append(
                "Bullish market structure confirmed by HH/HL."
            )

        elif structure == "PARTIAL_BULLISH":

            structure_score = 8

            reasons.append(
                "Market structure has a partial bullish bias."
            )

        elif structure == "LH_LL":

            structure_score = -15

            reasons.append(
                "Bearish market structure confirmed by LH/LL."
            )

        elif structure == "PARTIAL_BEARISH":

            structure_score = -8

            reasons.append(
                "Market structure has a partial bearish bias."
            )

        # ----------------------------------
        # Structural event
        #
        # Confirmed structural events have
        # priority over background structure.
        # ----------------------------------

        if break_confirmed:

            if break_direction == "BULLISH":

                structure_score += 20

                reasons.append(
                    "Confirmed bullish structural break detected."
                )

            elif break_direction == "BEARISH":

                structure_score -= 20

                reasons.append(
                    "Confirmed bearish structural break detected."
                )

        score += structure_score


        # ----------------------------------
        # Liquidity Confirmation
        # ----------------------------------

        liquidity_score = 0

        nearest_liquidity = self.liquidity.get(
            "nearest_liquidity",
            {}
        )

        sweep = self.liquidity.get(
            "sweep",
            {}
        )

        nearest_above = nearest_liquidity.get(
            "above"
        )

        nearest_below = nearest_liquidity.get(
            "below"
        )

        # ----------------------------------
        # Liquidity location
        # ----------------------------------

        if structure_bias == "BULLISH":

            if nearest_below is not None:

                liquidity_score += 5

                reasons.append(
                    "Sell-side liquidity is available below price."
                )

        elif structure_bias == "BEARISH":

            if nearest_above is not None:

                liquidity_score += 5

                reasons.append(
                    "Buy-side liquidity is available above price."
                )

        # ----------------------------------
        # Liquidity sweep
        # ----------------------------------

        if sweep.get("detected"):

            sweep_direction = sweep.get(
                "direction"
            )

            if (
                structure_bias == "BULLISH"
                and
                sweep_direction == "SELL_SIDE"
            ):

                liquidity_score += 10

                reasons.append(
                    "Sell-side liquidity sweep supports the bullish structure."
                )

            elif (
                structure_bias == "BEARISH"
                and
                sweep_direction == "BUY_SIDE"
            ):

                liquidity_score += 10

                reasons.append(
                    "Buy-side liquidity sweep supports the bearish structure."
                )

            elif (
                structure_bias == "BULLISH"
                and
                sweep_direction == "BUY_SIDE"
            ):

                liquidity_score -= 10

                reasons.append(
                    "Buy-side liquidity sweep conflicts with the bullish structure."
                )

            elif (
                structure_bias == "BEARISH"
                and
                sweep_direction == "SELL_SIDE"
            ):

                liquidity_score -= 10

                reasons.append(
                    "Sell-side liquidity sweep conflicts with the bearish structure."
                )

        liquidity_score = max(
            -15,
            min(
                15,
                liquidity_score
            )
        )

        score += liquidity_score


        # ----------------------------------
        # Directional confirmation evidence
        # ----------------------------------

        volume_ratio = float(
            self.volume.get(
                "ratio",
                0
            ) or 0
        )

        mtf_probability = float(
            self.multi_timeframe.get(
                "overall_probability",
                0
            ) or 0
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

        # ----------------------------------
        # Multi-Timeframe confirmation
        # ----------------------------------

        frame_signals = self.multi_timeframe.get(
            "frames",
            {}
        )

        bullish_frames = [
            name
            for name, data in frame_signals.items()
            if data.get("signal") == "BUY"
        ]

        bearish_frames = [
            name
            for name, data in frame_signals.items()
            if data.get("signal") == "SELL"
        ]

        neutral_frames = [
            name
            for name, data in frame_signals.items()
            if data.get("signal") == "HOLD"
        ]

        bullish_count = len(
            bullish_frames
        )

        bearish_count = len(
            bearish_frames
        )

        total_directional_frames = (
            bullish_count
            +
            bearish_count
        )

        # ----------------------------------
        # Directional confirmation score
        # ----------------------------------

        breakout_confirmation_score = 0

        if setup_direction == "SHORT":

            if bearish_count >= 4:

                breakout_confirmation_score += 30

                reasons.append(
                    "Strong bearish multi-timeframe alignment."
                )

            elif bearish_count >= 3:

                breakout_confirmation_score += 25

                reasons.append(
                    "Most major timeframes support the bearish setup."
                )

            elif bearish_count >= 2:

                breakout_confirmation_score += 15

                reasons.append(
                    "Some timeframes support the bearish setup."
                )

            elif bullish_count >= 3:

                breakout_confirmation_score -= 20

                reasons.append(
                    "Multi-timeframe alignment conflicts with the bearish setup."
                )

            else:

                reasons.append(
                    "Multi-timeframe bearish confirmation is limited."
                )

        else:

            if bullish_count >= 4:

                breakout_confirmation_score += 30

                reasons.append(
                    "Strong bullish multi-timeframe alignment."
                )

            elif bullish_count >= 3:

                breakout_confirmation_score += 25

                reasons.append(
                    "Most major timeframes support the bullish setup."
                )

            elif bullish_count >= 2:

                breakout_confirmation_score += 15

                reasons.append(
                    "Some timeframes support the bullish setup."
                )

            elif bearish_count >= 3:

                breakout_confirmation_score -= 20

                reasons.append(
                    "Multi-timeframe alignment conflicts with the bullish setup."
                )

            else:

                reasons.append(
                    "Multi-timeframe bullish confirmation is limited."
                )

        # ----------------------------------
        # Volume confirmation
        # ----------------------------------

        if volume_ratio >= 2.0:

            breakout_confirmation_score += 30

            reasons.append(
                "Very high volume supports directional confirmation."
            )

        elif volume_ratio >= 1.5:

            breakout_confirmation_score += 20

            reasons.append(
                "High volume supports directional confirmation."
            )

        elif volume_ratio >= 1.0:

            breakout_confirmation_score += 10

            reasons.append(
                "Normal volume provides limited directional confirmation."
            )

        else:

            reasons.append(
                "Low volume does not strongly confirm the setup."
            )

        # ----------------------------------
        # Candlestick confirmation
        # ----------------------------------

        if setup_direction == "SHORT":

            if bearish_candle:

                breakout_confirmation_score += 30

                reasons.append(
                    "Bearish candlestick confirmation detected."
                )

            elif bullish_candle:

                breakout_confirmation_score -= 20

                reasons.append(
                    "Bullish candlestick pattern conflicts with the bearish setup."
                )

        else:

            if bullish_candle:

                breakout_confirmation_score += 30

                reasons.append(
                    "Bullish candlestick confirmation detected."
                )

            elif bearish_candle:

                breakout_confirmation_score -= 20

                reasons.append(
                    "Bearish candlestick pattern conflicts with the bullish setup."
                )

        breakout_confirmation_score = max(
            0,
            min(
                100,
                breakout_confirmation_score
            )
        )

        # ----------------------------------
        # Directional Breakout State
        # ----------------------------------

        breakout_buffer = self.resistance * 0.0025

        breakdown_buffer = self.support * 0.0025

        if setup_direction == "SHORT":

            breakdown_level = (
                self.support
                - breakdown_buffer
            )

            breakout_level = breakdown_level

            if self.price > self.support:

                breakout_state = "ABOVE_SUPPORT"

            elif self.price > breakdown_level:

                breakout_state = "BREAKDOWN_ATTEMPT"

            else:

                if (
                    breakout_confirmation_score >= 70
                    and
                    pullback_setup_rr >= 1.5
                ):

                    breakout_state = "BREAKDOWN_CONFIRMED"

                else:

                    breakout_state = "BREAKDOWN_ATTEMPT"

        else:

            breakout_level = (
                self.resistance
                + breakout_buffer
            )

            if self.price < self.resistance:

                breakout_state = "BELOW_RESISTANCE"

            elif self.price < breakout_level:

                breakout_state = "BREAKOUT_ATTEMPT"

            else:

                if (
                    breakout_confirmation_score >= 70
                    and
                    breakout_setup_rr >= 1.5
                ):

                    breakout_state = "BREAKOUT_CONFIRMED"

                else:

                    breakout_state = "BREAKOUT_ATTEMPT"
        


        # ----------------------------------
        # Risk / Reward
        # ----------------------------------

        if setup_direction == "SHORT":

            effective_rr = max(
                rr_ratio,
                pullback_setup_rr,
                breakout_setup_rr
            )

        else:

            effective_rr = max(
                rr_ratio,
                pullback_setup_rr,
                breakout_setup_rr
            )

        if effective_rr >= 2.0:

            reasons.append(
                "Risk/reward is favorable."
            )

        elif effective_rr >= 1.5:

            score -= 10

            reasons.append(
                "Risk/reward is acceptable but below 2.0."
            )

        elif effective_rr >= 1.0:

            score -= 25

            reasons.append(
                "Risk/reward is weak and below the preferred 1.5 level."
            )

        else:

            score -= 40

            reasons.append(
                "Risk/reward is below 1.0."
            )

        # ----------------------------------
        # Resistance proximity
        # ----------------------------------

        if breakout_state == "BREAKOUT_CONFIRMED":

            reasons.append(
                "Price has confirmed the breakout above resistance."
            )

        elif resistance_distance <= 2:

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
        # Final score
        # ----------------------------------

        score = max(
            0,
            min(100, score)
        )

        breakout_confirmed = (
    breakout_state == "BREAKOUT_CONFIRMED"
)

        # ----------------------------------
        # Actionability
        # ----------------------------------

        is_short_setup = (
            setup_direction == "SHORT"
        )

        is_long_setup = (
            setup_direction == "LONG"
        )

        if is_short_setup:

            short_rr_ok = (
                pullback_setup_rr >= 1.5
                or
                breakout_setup_rr >= 1.5
                or
                rr_ratio >= 1.5
            )

            short_confirmation_ok = (

                (
                    break_direction == "BEARISH"
                    and
                    break_confirmed
                )

                or

                breakout_state == "BREAKDOWN_CONFIRMED"

                or

                breakout_confirmation_score >= 70

            )

            if (
                score >= 70
                and
                short_rr_ok
                and
                short_confirmation_ok
            ):

                status = "READY"

                action = "SELL"

                message = (
                    "Bearish trade setup is currently actionable."
                )

                trigger_price = round(
                    self.price,
                    2
                )

            else:

                status = "WAIT"

                action = "WAIT"

                if setup_type in [
                    "SHORT_CONTINUATION",
                    "SHORT_REVERSAL"
                ]:

                    message = (
                        "Bearish setup detected, but "
                        "short-entry conditions are not "
                        "fully confirmed."
                    )

                    trigger_price = pullback_trigger

                else:

                    message = (
                        "No immediate bearish setup."
                    )

                    trigger_price = pullback_trigger

        elif is_long_setup:

            long_rr_ok = (
                breakout_setup_rr >= 1.5
                or
                pullback_setup_rr >= 1.5
            )

            long_confirmation_ok = (
                breakout_confirmed
                or
                setup_type in [
                    "LONG_CONTINUATION",
                    "LONG_REVERSAL"
                ]
            )

            if (
                score >= 70
                and
                long_rr_ok
                and
                long_confirmation_ok
            ):

                status = "READY"

                action = "BUY"

                message = (
                    "Bullish trade setup is currently actionable."
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

        else:

            status = "WAIT"

            action = "WAIT"

            message = (
                "No directional trade setup is currently confirmed."
            )

            trigger_price = pullback_trigger

        # ----------------------------------
        # Preferred Setup
        # ----------------------------------

        if setup_type in [
            "LONG_CONTINUATION",
            "LONG_REVERSAL",
            "SHORT_CONTINUATION",
            "SHORT_REVERSAL"
        ]:

            preferred_setup = setup_type

            setup_message = (
                self.setup.get(
                    "reason",
                    "Structural setup detected."
                )
            )

        elif (
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
            pullback_setup_rr >= 1.5
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

            "market_structure": self.market_structure,

            "liquidity": self.liquidity,

            "structure_score": structure_score,

            "liquidity_score": liquidity_score,

            "setup": self.setup,

            "setup_type": setup_type,

            "setup_direction": setup_direction,

            "setup_confidence": setup_confidence,

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

            "setup_risk_reward": {
            "breakout": breakout_setup_rr,
            "pullback": pullback_setup_rr
        },

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
