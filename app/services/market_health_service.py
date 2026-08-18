import time

from app.services.market_service import get_market_breadth


_CACHE_DURATION = 300  # 5 minutes

_cached_health = None
_cached_at = 0

def calculate_market_health():

    global _cached_health
    global _cached_at

    now = time.time()

    if (
        _cached_health is not None
        and now - _cached_at < _CACHE_DURATION
    ):

        return _cached_health

    breadth = get_market_breadth()

    analyzed = breadth["analyzed_stocks"]

    if analyzed == 0:
        return {
            "score": 50,
            "status": "⚪ Neutral",
            "breadth_ratio": 0,
            "analyzed_stocks": 0,
        }

    advancing = breadth["advancing"]
    declining = breadth["declining"]

    # -----------------------------------------
    # Market Breadth Score
    # -----------------------------------------

    breadth_ratio = (
        advancing / declining
        if declining > 0
        else float(advancing)
    )

    if breadth_ratio >= 2:
        breadth_score = 90

    elif breadth_ratio >= 1.5:
        breadth_score = 80

    elif breadth_ratio >= 1:
        breadth_score = 65

    elif breadth_ratio >= 0.75:
        breadth_score = 50

    elif breadth_ratio >= 0.5:
        breadth_score = 35

    else:
        breadth_score = 20

    # -----------------------------------------
    # Final Market Health
    # -----------------------------------------

    market_health = round(breadth_score)

    if market_health >= 75:
        status = "🟢 Strong"

    elif market_health >= 60:
        status = "🟡 Healthy"

    elif market_health >= 40:
        status = "🟠 Neutral"

    else:
        status = "🔴 Weak"

    result = {

    "score": market_health,

    "status": status,

    "breadth_ratio": round(
        breadth_ratio,
        2
    ),

    "advancing": advancing,

    "declining": declining,

    "unchanged": breadth["unchanged"],

    "analyzed_stocks": analyzed,

}

    _cached_health = result
    _cached_at = now

    return result