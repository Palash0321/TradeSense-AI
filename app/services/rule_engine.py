class RuleEngine:

    def __init__(self):
        self.signals = []

    # ----------------------------------
    # Add a signal
    # ----------------------------------

    def add(
        self,
        module: str,
        weight: float,
        direction: str,
        reason: str,
        confidence: float = 100
    ):

        direction = direction.lower()

        if direction not in ("bullish", "bearish", "neutral"):
            raise ValueError(
                f"Invalid direction: {direction}"
            )

        self.signals.append({

            "module": module,

            "weight": float(weight),

            "direction": direction,

            "reason": reason,

            "confidence": float(confidence)

        })

    # ----------------------------------
    # Calculate final score
    # ----------------------------------

    def final_score(self):

        score = 60.0

        reasons = []

        bullish = 0

        bearish = 0

        neutral = 0

        for signal in self.signals:

            weight = signal["weight"]

            confidence = signal["confidence"] / 100

            value = weight * confidence

            reasons.append(signal["reason"])

            if signal["direction"] == "bullish":

                bullish += 1

                score += value

            elif signal["direction"] == "bearish":

                bearish += 1

                score -= value

            else:

                neutral += 1

        score = max(
            10,
            min(100, round(score, 2))
        )

        return {

            "score": score,

            "reasons": reasons,

            "bullish_signals": bullish,

            "bearish_signals": bearish,

            "neutral_signals": neutral,

            "total_signals": len(self.signals)

        }