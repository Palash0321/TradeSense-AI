from dataclasses import dataclass


@dataclass
class MomentumResult:

    score: int

    momentum: str

    confidence: int

    reasons: list[str]


class MomentumEngine:

    def analyze(self, indicators: dict) -> MomentumResult:

        score = 0

        reasons = []

        # -----------------------------
        # RSI
        # -----------------------------

        rsi = indicators["rsi"]

        if 55 <= rsi <= 70:

            score += 20

            reasons.append(
                "Healthy bullish RSI"
            )

        elif 45 <= rsi < 55:

            score += 10

            reasons.append(
                "Neutral RSI"
            )

        elif rsi > 70:

            score -= 10

            reasons.append(
                "Overbought"
            )

        elif rsi < 30:

            score += 15

            reasons.append(
                "Oversold Bounce"
            )

        else:

            score -= 15

            reasons.append(
                "Weak RSI"
            )

        # -----------------------------
        # MACD
        # -----------------------------

        if indicators["macd"] > indicators["signal"]:

            score += 20

            reasons.append(
                "MACD Bullish Crossover"
            )

        else:

            score -= 20

            reasons.append(
                "MACD Bearish Crossover"
            )

        # -----------------------------
        # MACD Histogram
        # -----------------------------

        histogram = indicators.get("histogram", 0)

        if histogram > 0:

            score += 10

            reasons.append(
                "Positive Histogram"
            )

        else:

            score -= 10

            reasons.append(
                "Negative Histogram"
            )

        # -----------------------------
        # ROC
        # -----------------------------

        roc = indicators.get("roc", 0)

        if roc > 0:

            score += 10

            reasons.append(
                "Positive Rate of Change"
            )

        else:

            score -= 10

            reasons.append(
                "Negative Rate of Change"
            )

        # -----------------------------
        # Stochastic RSI
        # -----------------------------

        stoch = indicators.get("stoch_rsi", 50)

        if stoch > 80:

            score -= 5

            reasons.append(
                "Stochastic Overbought"
            )

        elif stoch < 20:

            score += 10

            reasons.append(
                "Stochastic Oversold"
            )

        confidence = min(abs(score), 100)

        if score >= 45:

            momentum = "Strong Bullish"

        elif score >= 20:

            momentum = "Bullish"

        elif score <= -45:

            momentum = "Strong Bearish"

        elif score <= -20:

            momentum = "Bearish"

        else:

            momentum = "Neutral"

        return MomentumResult(

            score=score,

            momentum=momentum,

            confidence=confidence,

            reasons=reasons

        )