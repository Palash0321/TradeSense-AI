from app.services.cache_service import get, set

from app.config import MARKET_CACHE


async def get_fear_greed():

    cached = get("fear")

    if cached:

        return cached

    score = 68

    if score <= 25:

        label = "Extreme Fear"

    elif score <= 45:

        label = "Fear"

    elif score <= 55:

        label = "Neutral"

    elif score <= 75:

        label = "Greed"

    else:

        label = "Extreme Greed"

    result = {

        "score": score,

        "label": label

    }

    set("fear", result, MARKET_CACHE)

    return result