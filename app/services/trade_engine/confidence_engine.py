from dataclasses import dataclass

from .trend_engine import TrendResult
from .momentum_engine import MomentumResult
from .option_engine import OptionResult


@dataclass
class ConfidenceResult:

    score: int

    confidence: int

    signal: str

    reasons: list[str]


class ConfidenceEngine:

    def analyze(

        self,

        trend: TrendResult,

        momentum: MomentumResult,

        option: OptionResult

    ) -> ConfidenceResult:

        score = (

            trend.score +

            momentum.score +

            option.score

        )

        score = max(-100, min(score, 100))

        confidence = abs(score)

        reasons = []

        reasons.extend(trend.reasons)

        reasons.extend(momentum.reasons)

        reasons.extend(option.reasons)

        if score >= 80:

            signal = "STRONG BUY"

        elif score >= 55:

            signal = "BUY"

        elif score >= 20:

            signal = "WEAK BUY"

        elif score > -20:

            signal = "WAIT"

        elif score > -55:

            signal = "SELL"

        else:

            signal = "STRONG SELL"

        return ConfidenceResult(

            score=score,

            confidence=confidence,

            signal=signal,

            reasons=reasons

        )