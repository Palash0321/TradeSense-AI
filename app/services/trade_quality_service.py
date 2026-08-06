import pandas as pd


class TradeQualityService:

    def __init__(self, dataframe):

        self.df = dataframe

    # ==========================
    # Trend Score (0-25)
    # ==========================

    def trend_score(self):

            latest = self.df.iloc[-1]

            score = 0

            if latest["MA20"] > latest["MA50"]:
                score += 15

            if latest["Close"] > latest["MA20"]:
                score += 10

            return score

    # ==========================
    # Momentum Score (0-20)
    # ==========================

    def momentum_score(self):

        latest = self.df.iloc[-1]

        score = 0

        if 50 <= latest["RSI"] <= 70:
            score += 10

        if latest["MACD"] > latest["Signal"]:
            score += 10

        return score

    # ==========================
    # Volume Score (0-15)
    # ==========================

    def volume_score(self):

        latest = self.df.iloc[-1]

        avg_volume = (

            self.df["Volume"]

            .tail(20)

            .mean()

        )

        if latest["Volume"] > avg_volume:

            return 15

        return 5

        # ==========================
    # Overall Trade Score (0-100)
    # ==========================

    def overall_score(self):

        trend = self.trend_score()

        momentum = self.momentum_score()

        volume = self.volume_score()

        total = (

            trend

            +

            momentum

            +

            volume

        )

        # Remaining 40 points
        # will be added later

        return {

            "trend": trend,

            "momentum": momentum,

            "volume": volume,

            "total_score": total

        }

        # ==========================
    # Trade Grade
    # ==========================

    def trade_grade(self):

        score = (

            self.overall_score()

            ["total_score"]

        )

        if score >= 90:

            return "★★★★★ Excellent"

        elif score >= 80:

            return "★★★★ Very Good"

        elif score >= 70:

            return "★★★ Good"

        elif score >= 60:

            return "★★ Average"

        else:

            return "★ Avoid"

            # ==========================
    # AI Confidence
    # ==========================

    def confidence(self):

        score = (

            self.overall_score()

            ["total_score"]

        )

        confidence = (

            score

            / 100

        ) * 100

        return round(

            confidence,

            1

        )

        # ==========================
    # Recommendation
    # ==========================

    def recommendation(self):

        score = self.overall_score()["total_score"]

        if score >= 90:
            return "STRONG BUY"

        elif score >= 75:
            return "BUY"

        elif score >= 60:
            return "ACCUMULATE"

        elif score >= 40:
            return "HOLD"

        else:
            return "AVOID"


        # ==========================
    # Risk Rating
    # ==========================

    def risk_rating(self):

        score = self.overall_score()["total_score"]

        if score >= 85:
            return "Very Low"

        elif score >= 70:
            return "Low"

        elif score >= 55:
            return "Medium"

        elif score >= 40:
            return "High"

        else:
            return "Very High"


        # ==========================
    # Complete Analysis
    # ==========================

    def analyze(self):

        quality = self.overall_score()

        return {

            "score": quality["total_score"],

            "trend": quality["trend"],

            "momentum": quality["momentum"],

            "volume": quality["volume"],

            "grade": self.trade_grade(),

            "confidence": self.confidence(),

            "recommendation": self.recommendation(),

            "risk": self.risk_rating()

        }