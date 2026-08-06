class AIConfidenceService:

    def __init__(

        self,

        trade_quality,

        probability,

        multi_timeframe,

        opportunity,

        trade_plan,

        patterns

    ):

        self.trade_quality = trade_quality
        self.probability = probability
        self.multi_timeframe = multi_timeframe
        self.opportunity = opportunity
        self.trade_plan = trade_plan
        self.patterns = patterns

    def calculate(self):

        score = 0

        # --------------------------
        # Trade Quality
        # --------------------------

        score += self.trade_quality["score"] * 0.30

        # --------------------------
        # Probability
        # --------------------------

        score += self.probability * 0.25

        # --------------------------
        # Multi Timeframe
        # --------------------------

        score += (
            self.multi_timeframe["overall_probability"]
            * 0.20
        )

        # --------------------------
        # Opportunity
        # --------------------------

        if self.opportunity["status"] == "BUY":

            score += 10

        elif self.opportunity["status"] == "WAIT":

            score += 5

        # --------------------------
        # Trade Plan
        # --------------------------

        if self.trade_plan["recommendation"] == "BUY":

            score += 10

        # --------------------------
        # Candlestick
        # --------------------------

        bullish = [

            "Hammer",

            "Bullish Engulfing"

        ]

        bearish = [

            "Shooting Star",

            "Bearish Engulfing"

        ]

        for pattern in self.patterns:

            if pattern in bullish:

                score += 5

            elif pattern in bearish:

                score -= 5

        return round(

            max(

                0,

                min(100, score)

            ),

            2

        )