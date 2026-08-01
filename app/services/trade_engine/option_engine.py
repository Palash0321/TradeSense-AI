from dataclasses import dataclass


@dataclass
class OptionResult:

    score: int

    sentiment: str

    confidence: int

    support: int

    resistance: int

    max_pain: int

    strategy: str

    reasons: list[str]


class OptionEngine:

    def analyze(self, option_chain: dict) -> OptionResult:

        score = 0

        reasons = []

        pcr = option_chain["pcr"]

        bias = option_chain["bias"]

        support = option_chain["support"]

        resistance = option_chain["resistance"]

        max_pain = option_chain["max_pain"]

        # ---------------------------------
        # PCR
        # ---------------------------------

        if pcr >= 1.20:

            score += 25

            reasons.append(
                "Bullish PCR"
            )

        elif pcr >= 1:

            score += 15

            reasons.append(
                "Positive PCR"
            )

        elif pcr <= 0.80:

            score -= 25

            reasons.append(
                "Bearish PCR"
            )

        else:

            score -= 10

            reasons.append(
                "Weak PCR"
            )

        # ---------------------------------
        # Option Bias
        # ---------------------------------

        if bias == "Bullish":

            score += 30

            reasons.append(
                "Bullish OI Structure"
            )

        elif bias == "Bearish":

            score -= 30

            reasons.append(
                "Bearish OI Structure"
            )

        # ---------------------------------
        # Support / Resistance
        # ---------------------------------

        if support > 0:

            score += 10

            reasons.append(
                "Strong Support Identified"
            )

        if resistance > 0:

            score += 10

            reasons.append(
                "Resistance Identified"
            )

        # ---------------------------------
        # Strategy
        # ---------------------------------

        if score >= 50:

            strategy = "Bullish Option Buying"

            sentiment = "Bullish"

        elif score >= 20:

            strategy = "Bull Put Spread"

            sentiment = "Bullish"

        elif score <= -50:

            strategy = "Bearish Option Buying"

            sentiment = "Bearish"

        elif score <= -20:

            strategy = "Bear Call Spread"

            sentiment = "Bearish"

        else:

            strategy = "Iron Condor"

            sentiment = "Neutral"

        confidence = min(abs(score), 100)

        return OptionResult(

            score=score,

            sentiment=sentiment,

            confidence=confidence,

            support=support,

            resistance=resistance,

            max_pain=max_pain,

            strategy=strategy,

            reasons=reasons

        )