from dataclasses import dataclass


@dataclass
class TradeResult:

    score: int

    signal: str

    confidence: int

    reasons: list

    entry: float | None

    stop_loss: float | None

    target1: float | None

    target2: float | None


class TradeEngine:

    def analyze(

        self,

        indicators: dict,

        option_chain: dict

    ) -> TradeResult:

        score = 0

        reasons = []

        # ---------------------------
        # EMA
        # ---------------------------

        if indicators["ema20"] > indicators["ema50"]:

            score += 20

            reasons.append(
                "EMA20 above EMA50"
            )

        else:

            score -= 20

            reasons.append(
                "EMA20 below EMA50"
            )

        # ---------------------------
        # RSI
        # ---------------------------

        rsi = indicators["rsi"]

        if 45 <= rsi <= 65:

            score += 15

            reasons.append(
                "Healthy RSI"
            )

        elif rsi > 70:

            score -= 10

            reasons.append(
                "Overbought"
            )

        elif rsi < 30:

            score += 10

            reasons.append(
                "Oversold Bounce"
            )

        # ---------------------------
        # MACD
        # ---------------------------

        if indicators["macd"] > indicators["signal"]:

            score += 15

            reasons.append(
                "MACD Bullish"
            )

        else:

            score -= 15

            reasons.append(
                "MACD Bearish"
            )

        # ---------------------------
        # PCR
        # ---------------------------

        pcr = option_chain["pcr"]

        if pcr > 1:

            score += 15

            reasons.append(
                "Bullish PCR"
            )

        else:

            score -= 15

            reasons.append(
                "Bearish PCR"
            )

        # ---------------------------
        # Option Bias
        # ---------------------------

        if option_chain["bias"] == "Bullish":

            score += 20

            reasons.append(
                "Bullish Option Chain"
            )

        elif option_chain["bias"] == "Bearish":

            score -= 20

            reasons.append(
                "Bearish Option Chain"
            )

        # ---------------------------

        confidence = min(
            abs(score),
            100
        )

        if score >= 70:

            signal = "STRONG BUY"

        elif score >= 40:

            signal = "BUY"

        elif score >= 0:

            signal = "HOLD"

        elif score >= -40:

            signal = "SELL"

        else:

            signal = "STRONG SELL"

        entry = indicators["price"]

        stop = entry * 0.995

        target1 = entry * 1.01

        target2 = entry * 1.02

        return TradeResult(

            score=score,

            signal=signal,

            confidence=confidence,

            reasons=reasons,

            entry=round(entry, 2),

            stop_loss=round(stop, 2),

            target1=round(target1, 2),

            target2=round(target2, 2),

        )