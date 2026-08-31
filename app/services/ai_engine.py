from app.services.trade_quality_service import TradeQualityService
from app.services.trade_planner_service import TradePlannerService
from app.services.opportunity_service import OpportunityService
from app.services.probability_service import ProbabilityService
from app.services.multi_timeframe_service import MultiTimeframeService
from app.services.candlestick_service import CandlestickService
from app.services.ai_confidence_service import AIConfidenceService
from app.services.volume_service import VolumeService
from app.services.rule_engine import RuleEngine
from app.services.entry_engine_service import EntryEngineService
from app.services.trade_validation_service import TradeValidationService
from app.services.option_chain_service import OptionChainService
from app.services.final_decision_service import FinalDecisionService
from app.core.levels.market_structure import calculate_market_structure
from app.core.liquidity.liquidity import calculate_liquidity
from app.core.setup.setup_engine import calculate_setup

class AIEngine:

    def __init__(
        self,
        symbol,
        history,
        latest,
        levels,
        risk_reward,
        setup_direction=None,
        trend_strength=None,
        macd_status=None,
        rsi_status=None,
        option_context=None
    ):

        self.symbol = symbol
        self.history = history
        self.latest = latest
        self.levels = levels
        self.risk_reward = risk_reward
        self.setup_direction = setup_direction
        self.trend_strength = trend_strength
        self.macd_status = macd_status
        self.rsi_status = rsi_status
        self.option_context = option_context or {}

    def analyze(self):

        # =====================================
        # Market Structure
        # =====================================

        market_structure = calculate_market_structure(
            self.history
        )

        # =====================================
        # Liquidity
        # =====================================

        liquidity = calculate_liquidity(
            self.history,
            market_structure
        )

        # =====================================
        # Setup Engine
        # =====================================

        setup = calculate_setup(
            market_structure=market_structure,
            liquidity=liquidity,
            current_price=float(
                self.latest["Close"]
            )
        )

        # =====================================
        # Trade Quality
        # =====================================

        trade_quality = TradeQualityService(
            self.history,
            setup_direction=setup.get(
                "direction"
            )
        )

        analysis = trade_quality.analyze()

        # =====================================
        # Rule Engine
        # =====================================

        rules = RuleEngine()

        # =====================================
        # Option Chain Analysis
        # =====================================

        option_chain = OptionChainService(
            self.option_context
        ).analyze()

        analysis["atr"] = round(
            float(self.latest["ATR"]),
            2
        )

        # -------------------------
        # Trend
        # -------------------------

        if analysis["trend"] >= 20:

            if setup.get("direction") == "SHORT":

                trend_direction = "bearish"
                trend_reason = "Strong bearish trend"

            else:

                trend_direction = "bullish"
                trend_reason = "Strong bullish trend"

            rules.add(

                module="Trend",

                weight=20,

                direction=trend_direction,

                reason=trend_reason,

                confidence=90

            )

        else:

            rules.add(

                module="Trend",

                weight=20,

                direction="neutral",

                reason="Trend confirmation is weak",

                confidence=90

            )


        # -------------------------
        # Momentum
        # -------------------------

        if analysis["momentum"] >= 20:

            if setup.get("direction") == "SHORT":

                momentum_direction = "bearish"
                momentum_reason = "Strong bearish momentum"

            else:

                momentum_direction = "bullish"
                momentum_reason = "Strong bullish momentum"

            rules.add(

                module="Momentum",

                weight=15,

                direction=momentum_direction,

                reason=momentum_reason,

                confidence=85

            )

        else:

            rules.add(

                module="Momentum",

                weight=15,

                direction="neutral",

                reason="Momentum confirmation is weak",

                confidence=85

            )

        # =====================================
        # Candlestick Analysis
        # =====================================

        patterns = CandlestickService(
            self.history
        ).detect()

        bullish = [

            "Hammer",

            "Bullish Engulfing"

        ]

        bearish = [

            "Shooting Star",

            "Bearish Engulfing"

        ]

        for pattern in patterns:

            if pattern in bullish:

                rules.add(

                    module="Candlestick",

                    weight=10,

                    direction="bullish",

                    reason=pattern,

                    confidence=75

                )

            elif pattern in bearish:

                rules.add(

                    module="Candlestick",

                    weight=10,

                    direction="bearish",

                    reason=pattern,

                    confidence=75

                )

        analysis["candlestick_patterns"] = patterns

        # -------------------------
        # Volume Analysis
        # -------------------------

        volume = VolumeService(
            self.history
        ).analyze()

        if volume["status"] in [

            "High",

            "Very High"

        ]:

            volume_direction = (
                "bearish"
                if setup.get("direction") == "SHORT"
                else "bullish"
            )

            volume_reason = (
                "High trading volume supports the bearish setup."
                if setup.get("direction") == "SHORT"
                else "High trading volume supports the bullish setup."
            )

            rules.add(

                module="Volume",

                weight=10,

                direction=volume_direction,

                reason=volume_reason,

                confidence=80

            )

        else:

            rules.add(

                module="Volume",

                weight=0,

                direction="neutral",

                reason="Low trading volume does not strongly confirm the setup.",

                confidence=80

            )

        analysis["volume_analysis"] = volume

        # =====================================
        # Option Chain Rule
        # =====================================

        if option_chain["available"]:

            if option_chain["bias"] in [
                "BULLISH",
                "MILD_BULLISH"
            ]:

                rules.add(

                    module="Option Chain",

                    weight=10,

                    direction="bullish",

                    reason=option_chain["reason"],

                    confidence=option_chain["confidence"]

                )

            elif option_chain["bias"] in [
                "BEARISH",
                "MILD_BEARISH"
            ]:

                rules.add(

                    module="Option Chain",

                    weight=10,

                    direction="bearish",

                    reason=option_chain["reason"],

                    confidence=option_chain["confidence"]

                )

            else:

                rules.add(

                    module="Option Chain",

                    weight=0,

                    direction="neutral",

                    reason=option_chain["reason"],

                    confidence=option_chain["confidence"]

                )

        
        # =====================================
        # Trade Planner
        # =====================================

        trade_plan = TradePlannerService(

            analysis,

            float(self.latest["Close"]),

            self.levels["support"],

            self.levels["resistance"],

            self.risk_reward,

            setup_direction=setup.get(
                "direction"
            )

        ).generate()

        # =====================================
        # Multi Timeframe
        # =====================================

        multi_timeframe = MultiTimeframeService(
            self.symbol,
            setup_direction=setup.get(
                "direction"
            )
        ).analyze()

        # =====================================
        # Opportunity
        # =====================================

        opportunity = OpportunityService(

            analysis,

            self.levels["support"],

            self.levels["resistance"],

            float(self.latest["Close"]),

            self.risk_reward,

            volume,

            patterns,

            multi_timeframe,

            market_structure=market_structure,
            liquidity=liquidity,

            setup=setup

        ).analyze()

        # =====================================
        # Entry Engine
        # =====================================

        breakout_level = round(
    float(self.levels["resistance"]) * 1.0025,
    2
)

        entry_engine = EntryEngineService(

            current_price=float(
                self.latest["Close"]
            ),

            support=self.levels["support"],

            resistance=self.levels["resistance"],

            atr=float(
                self.latest["ATR"]
            ),

            risk_reward=self.risk_reward,

            breakout_level=breakout_level,

            setup=setup

        ).generate()

        # =====================================
        # Refine Opportunity With Setup R/R
        # =====================================

        candidate_setup = opportunity.get(
            "preferred_setup",
            "NO_SETUP"
        )

        setup_risk_reward = {
            "pullback": entry_engine[
                "pullback"
            ]["risk_reward"]["target1"],

            "breakout": entry_engine[
                "breakout"
            ]["risk_reward"]["target1"]
        }

        opportunity = OpportunityService(

            analysis,

            self.levels["support"],

            self.levels["resistance"],

            float(self.latest["Close"]),

            self.risk_reward,

            volume,

            patterns,

            multi_timeframe,

            setup_risk_reward,

            market_structure=market_structure,
            liquidity=liquidity,
            setup=setup

        ).analyze()

        # =====================================
        # Probability
        # =====================================

        probability = ProbabilityService(

            analysis,

            self.risk_reward,

            opportunity

        ).calculate()
        # =====================================
        # Multi-Timeframe Rule
        # =====================================

        mtf_probability = multi_timeframe["overall_probability"]

        # Build the explanation from the actual timeframe signals.
        # The probability is already direction-aware because
        # MultiTimeframeService uses setup_direction.

        frame_signals = multi_timeframe.get(
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

        unknown_frames = [
            name
            for name, data in frame_signals.items()
            if data.get("signal") == "UNKNOWN"
        ]


        def format_frames(frames):

            return ", ".join(frames)


        # -------------------------------------
        # Direction-aware MTF interpretation
        # -------------------------------------

        if setup.get("direction") == "SHORT":

            if mtf_probability >= 80:

                reason_parts = []

                if bearish_frames:
                    reason_parts.append(
                        f"{format_frames(bearish_frames)} bearish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                reason = (
                    "Strong bearish multi-timeframe alignment — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=20,

                    direction="bearish",

                    reason=reason,

                    confidence=95

                )

            elif mtf_probability >= 60:

                reason_parts = []

                if bearish_frames:
                    reason_parts.append(
                        f"{format_frames(bearish_frames)} bearish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                if bullish_frames:
                    reason_parts.append(
                        f"{format_frames(bullish_frames)} bullish"
                    )

                reason = (
                    "Bearish multi-timeframe bias — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=10,

                    direction="bearish",

                    reason=reason,

                    confidence=85

                )

            elif mtf_probability <= 40:

                reason_parts = []

                if bullish_frames:
                    reason_parts.append(
                        f"{format_frames(bullish_frames)} bullish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                if bearish_frames:
                    reason_parts.append(
                        f"{format_frames(bearish_frames)} bearish"
                    )

                reason = (
                    "Weak bearish multi-timeframe alignment — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=10,

                    direction="bullish",

                    reason=reason,

                    confidence=80

                )

            else:

                rules.add(

                    module="Multi Timeframe",

                    weight=0,

                    direction="neutral",

                    reason="Mixed multi-timeframe signals.",

                    confidence=80

                )


        else:

            if mtf_probability >= 80:

                reason_parts = []

                if bullish_frames:
                    reason_parts.append(
                        f"{format_frames(bullish_frames)} bullish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                reason = (
                    "Strong bullish multi-timeframe alignment — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=20,

                    direction="bullish",

                    reason=reason,

                    confidence=95

                )

            elif mtf_probability >= 60:

                reason_parts = []

                if bullish_frames:
                    reason_parts.append(
                        f"{format_frames(bullish_frames)} bullish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                if bearish_frames:
                    reason_parts.append(
                        f"{format_frames(bearish_frames)} bearish"
                    )

                reason = (
                    "Bullish multi-timeframe bias — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=10,

                    direction="bullish",

                    reason=reason,

                    confidence=85

                )

            elif mtf_probability <= 40:

                reason_parts = []

                if bearish_frames:
                    reason_parts.append(
                        f"{format_frames(bearish_frames)} bearish"
                    )

                if neutral_frames:
                    reason_parts.append(
                        f"{format_frames(neutral_frames)} neutral"
                    )

                reason = (
                    "Weak bullish multi-timeframe alignment — "
                    + "; ".join(reason_parts)
                )

                rules.add(

                    module="Multi Timeframe",

                    weight=10,

                    direction="bearish",

                    reason=reason,

                    confidence=80

                )

            else:

                rules.add(

                    module="Multi Timeframe",

                    weight=0,

                    direction="neutral",

                    reason="Mixed multi-timeframe signals.",

                    confidence=80

                )

        analysis["option_chain"] = option_chain

        result = rules.final_score()
        
        analysis["score"] = result["score"]
        
        analysis["reasons"] = result["reasons"]
        
        analysis["bullish_signals"] = result["bullish_signals"]
        
        analysis["bearish_signals"] = result["bearish_signals"]
        
        analysis["neutral_signals"] = result["neutral_signals"]
        
        analysis["total_signals"] = result["total_signals"]

        # =====================================
        # AI Confidence
        # =====================================
        
        ai_confidence = AIConfidenceService(
        
                trade_quality=analysis,
        
                probability=probability,
        
                multi_timeframe=multi_timeframe,
        
                opportunity=opportunity,
        
                trade_plan=trade_plan,
        
                patterns=patterns
        
            ).calculate()

        # =====================================
        # Confidence Adjustment
        # =====================================

        confidence_bonus = 0

        if self.trend_strength == "Strong Bullish":

            confidence_bonus += 5

        if self.macd_status == "Bullish":

            confidence_bonus += 3

        if self.rsi_status == "Neutral":

            confidence_bonus += 2

        ai_confidence = min(
            ai_confidence + confidence_bonus,
            100
        )

        

        # =====================================
        # Setup Risk / Reward
        # =====================================

        preferred_setup = opportunity.get(
            "preferred_setup",
            "NO_SETUP"
        )

        # =====================================
        # Setup Details
        # =====================================

        if preferred_setup in [
            "BREAKOUT",
            "WAIT_FOR_BREAKOUT"
        ]:

            setup_details = {
                "type": "BREAKOUT",
                **entry_engine["breakout"]
            }

        elif preferred_setup in [
            "PULLBACK",
            "LONG_CONTINUATION",
            "LONG_REVERSAL",
            "SHORT_CONTINUATION",
            "SHORT_REVERSAL"
        ]:

            setup_details = {
                "type": "PULLBACK",
                **entry_engine["pullback"]
            }

        else:

            setup_details = {
                "type": "NONE",

                "pullback": entry_engine["pullback"],

                "breakout": entry_engine["breakout"]
            }

        # =====================================
        # Setup Risk / Reward
        # =====================================

        if preferred_setup in [
            "BREAKOUT",
            "WAIT_FOR_BREAKOUT",
            "LONG_CONTINUATION",
            "LONG_REVERSAL",
            "SHORT_CONTINUATION",
            "SHORT_REVERSAL"
        ]:

            if preferred_setup in [
                "SHORT_CONTINUATION",
                "SHORT_REVERSAL"
            ]:

                setup_risk_reward = entry_engine[
                    "pullback"
                ]["risk_reward"]["target1"]

            else:

                setup_risk_reward = entry_engine[
                    "breakout"
                ]["risk_reward"]["target1"]

        elif preferred_setup == "PULLBACK":

            setup_risk_reward = entry_engine[
                "pullback"
            ]["risk_reward"]["target1"]

        else:

            setup_risk_reward = None

        # =====================================
        # Trade Validation
        # =====================================

        trade_validation = TradeValidationService(

            opportunity=opportunity,

            entry_engine=entry_engine,

            ai_confidence=ai_confidence,

            setup_risk_reward=setup_risk_reward

        ).validate()

        # =====================================
        # Final Decision
        # =====================================

        final_decision = FinalDecisionService(

            opportunity=opportunity,

            trade_validation=trade_validation,

            ai_confidence=ai_confidence,

            entry_engine=entry_engine,

            setup_details=setup_details


        ).decide()

        return {

            "trade_quality": analysis,

            "trade_plan": trade_plan,

            "market_structure": market_structure,

            "liquidity": liquidity,

            "setup": setup,

            "opportunity": opportunity,

            "probability": probability,

            "ai_confidence": ai_confidence,

            "multi_timeframe": multi_timeframe,

            "entry_engine": entry_engine,

            "trade_validation": trade_validation,

            "final_decision": final_decision,

            "candlestick_patterns": patterns,

            "volume_analysis": volume

        }
