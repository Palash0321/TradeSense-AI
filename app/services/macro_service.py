from app.services.cache_service import get, set

from app.config import MARKET_CACHE


async def get_macro_data():

    cached = get("macro")

    if cached:

        return cached

    data = [

        {
            "title": "Gold",
            "value": "$3,350",
            "desc": "Safe Haven"
        },

        {
            "title": "Crude Oil",
            "value": "$71.2",
            "desc": "Energy"
        },

        {
            "title": "USD / INR",
            "value": "86.21",
            "desc": "Currency"
        },

        {
            "title": "Bitcoin",
            "value": "$118K",
            "desc": "Crypto"
        },

        {
            "title": "India VIX",
            "value": "13.4",
            "desc": "Volatility"
        }

    ]

    set("macro", data, MARKET_CACHE)

    return data