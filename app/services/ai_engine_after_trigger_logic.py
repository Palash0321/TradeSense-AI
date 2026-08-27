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
from app.services.final_decision_service import FinalDecisionService

class AIEngine:

    def __init__(
        self,
        symbol,
        history,
        latest,
        levels,
        risk_reward
    ):

        self.symbol = symbol
        self.history = history
        self.latest = latest
        self.levels = levels
        self.risk_reward = risk_reward

    def analyze(self):

        # =====================================
        # Trade Quality
        # =====================================

        trade_quality = TradeQualityService(
            self.history
        )

        analysis = trade_quality.analyze()

        # =====================================
        # Rule Engine
        # =====================================

        rules = RuleEngine()

        analysis["atr"] = round(
            float(self.latest["ATR"]),
            2
        )

        # -------------------------
        # Trend
        # -------------------------

        if analysis["trend"] >= 20:

            rules.add(

                module="Trend",

                weight=20,

                direction="bullish",

                reason="Strong bullish trend",

                confidence=90

            )

        else:

            rules.add(

                module="Trend",

                weight=20,

                direction="bearish",

                reason="Weak trend",

                confidence=90

            )

        # -------------------------
        # Momentum
        # -------------------------

        if analysis["momentum"] >= 20:

            rules.add(

                module="Momentum",

                weight=15,

                direction="bullish",

                reason="Momentum is strong",

                confidence=85

            )

        else:

            rules.add(

                module="Momentum",

                weight=15,

                direction="bearish",

                reason="Momentum is weak",

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

            rules.add(

                module="Volume",

                weight=10,

                direction="bullish",

                reason="High trading volume",

                confidence=80

            )

        else:

            rules.add(

                module="Volume",

                weight=10,

                direction="bearish",

                reason="Low trading volume",

                confidence=80

            )

        analysis["volume_analysis"] = volume

        
        # =====================================
        # Trade Planner
        # =====================================

        trade_plan = TradePlannerService(

            analysis,

            float(self.latest["Close"]),

            self.levels["support"],

            self.levels["resistance"],

            self.risk_reward

        ).generate()

        # =====================================
        # Multi Timeframe
        # =====================================

        multi_timeframe = MultiTimeframeService(
            self.symbol
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

            multi_timeframe

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

            breakout_level=breakout_level

        ).generate()    

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
        # This prevents the score from making claims that the individual
        # timeframe results do not support.

        frame_signals = multi_timeframe.get("frames", {})

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

        def format_frames(frames):
            return ", ".join(frames)

        if mtf_probability >= 80:

            if bullish_frames and neutral_frames:

                reason = (
                    "Strong bullish multi-timeframe alignment — "
                    f"{format_frames(bullish_frames)} bullish; "
                    f"{format_frames(neutral_frames)} neutral"
                )

            elif bullish_frames:

                reason = (
                    "Strong bullish multi-timeframe alignment — "
                    f"{format_frames(bullish_frames)} bullish"
                )

            else:

                reason = (
                    "Strong bullish multi-timeframe score"
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

        elif mtf_probability <= 20:

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
                "Bearish multi-timeframe bias — "
                + "; ".join(reason_parts)
            )

            rules.add(

                module="Multi Timeframe",

                weight=10,

                direction="bearish",

                reason=reason,

                confidence=80

            )

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
        # Setup Risk / Reward
        # =====================================

        preferred_setup = opportunity.get(
            "preferred_setup",
            "WAIT"
        )

        if preferred_setup in [
    "BREAKOUT",
    "WAIT_FOR_BREAKOUT"
]:

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

            ai_confidence=ai_confidence

        ).decide()

        return {

            "trade_quality": analysis,

            "trade_plan": trade_plan,

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
