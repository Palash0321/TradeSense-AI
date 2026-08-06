from app.services.trade_quality_service import TradeQualityService
from app.services.trade_planner_service import TradePlannerService
from app.services.opportunity_service import OpportunityService
from app.services.probability_service import ProbabilityService
from app.services.multi_timeframe_service import MultiTimeframeService
from app.services.candlestick_service import CandlestickService
from app.services.ai_confidence_service import AIConfidenceService
from app.services.volume_service import VolumeService
from app.services.rule_engine import RuleEngine

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
        # Limit Score
        # =====================================

        
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
        # Opportunity
        # =====================================

        opportunity = OpportunityService(

            analysis,

            self.levels["support"],

            self.levels["resistance"],

            float(self.latest["Close"])

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
        # Multi Timeframe
        # =====================================

        multi_timeframe = MultiTimeframeService(
            self.symbol
        ).analyze()

        # =====================================
        # Multi-Timeframe Rule
        # =====================================

        mtf_probability = multi_timeframe["overall_probability"]

        if mtf_probability >= 80:

            rules.add(

                module="Multi Timeframe",

                weight=20,

                direction="bullish",

                reason="All major timeframes are bullish",

                confidence=95

            )

        elif mtf_probability >= 60:

            rules.add(

                module="Multi Timeframe",

                weight=10,

                direction="bullish",

                reason="Most timeframes are bullish",

                confidence=85

            )

        elif mtf_probability <= 20:

            rules.add(

                module="Multi Timeframe",

                weight=20,

                direction="bearish",

                reason="Most timeframes are bearish",

                confidence=95

            )

        elif mtf_probability <= 40:

            rules.add(

                module="Multi Timeframe",

                weight=10,

                direction="bearish",

                reason="Higher timeframes are weak",

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

        return {

            "trade_quality": analysis,

            "trade_plan": trade_plan,

            "opportunity": opportunity,

            "probability": probability,

            "ai_confidence": ai_confidence,

            "multi_timeframe": multi_timeframe,

            "candlestick_patterns": patterns,

            "volume_analysis": volume

        }