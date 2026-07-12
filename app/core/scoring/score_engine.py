from app.core.scoring.trend_score import trend_score
from app.core.scoring.rsi_score import rsi_score
from app.core.scoring.macd_score import macd_score
from app.core.scoring.price_score import price_score


def calculate_score(latest):

    score = 0

    reasons = []

    modules = [

        trend_score,

        rsi_score,

        macd_score,

        price_score

    ]

    for module in modules:

        points, reason = module(latest)

        score += points

        reasons.append(reason)

    return score, reasons