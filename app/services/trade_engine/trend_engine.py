from dataclasses import dataclass


@dataclass
class TrendResult:

    score: int

    trend: str

    strength: str

    confidence: int

    reasons: list[str]


class TrendEngine:

    def analyze(self, indicators: dict) -> TrendResult:

        score = 0

        reasons = []

        # -------------------------
        # EMA20 vs EMA50
        # -------------------------

        if indicators["ema20"] > indicators["ema50"]:

            score += 25

            reasons.append(
                "EMA20 above EMA50"
            )

        else:

            score -= 25

            reasons.append(
                "EMA20 below EMA50"
            )

        # -------------------------
        # Price vs EMA20
        # -------------------------

        if indicators["price"] > indicators["ema20"]:

            score += 15

            reasons.append(
                "Price above EMA20"
            )

        else:

            score -= 15

            reasons.append(
                "Price below EMA20"
            )

        # -------------------------
        # Price vs EMA50
        # -------------------------

        if indicators["price"] > indicators["ema50"]:

            score += 15

            reasons.append(
                "Price above EMA50"
            )

        else:

            score -= 15

            reasons.append(
                "Price below EMA50"
            )

        # -------------------------
        # ADX
        # -------------------------

        adx = indicators.get("adx", 20)

        if adx >= 30:

            score += 20

            reasons.append(
                "Strong trend (ADX)"
            )

        elif adx >= 20:

            score += 10

            reasons.append(
                "Moderate trend (ADX)"
            )

        else:

            reasons.append(
                "Weak trend"
            )

        # -------------------------
        # Higher Highs
        # -------------------------

        if indicators.get("higher_highs", False):

            score += 15

            reasons.append(
                "Higher highs confirmed"
            )

        # -------------------------

        confidence = min(abs(score), 100)

        if score >= 60:

            trend = "Bullish"

            strength = "Strong"

        elif score >= 25:

            trend = "Bullish"

            strength = "Moderate"

        elif score <= -60:

            trend = "Bearish"

            strength = "Strong"

        elif score <= -25:

            trend = "Bearish"

            strength = "Moderate"

        else:

            trend = "Sideways"

            strength = "Weak"

        return TrendResult(

            score=score,

            trend=trend,

            strength=strength,

            confidence=confidence,

            reasons=reasons

        )