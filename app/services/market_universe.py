# =========================================================
# TRADE SENSE AI - MARKET UNIVERSE
# =========================================================
#
# Shared stock universe used by:
# - Market Breadth
# - Sector Performance
#
# This is intentionally a curated liquid-stock universe
# instead of querying the entire NSE universe.
# =========================================================


INDIA_MARKET_UNIVERSE = [

    # Energy
    "RELIANCE.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",

    # Technology
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",

    # Financial Services
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",

    # Consumer
    "ITC.NS",
    "HINDUNILVR.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",

    # Automobile
    "MARUTI.NS",
    "M&M.NS",
    "HEROMOTOCO.NS",

    # Industrials
    "LT.NS",
    "BHARTIARTL.NS",
    "ADANIPORTS.NS",

    # Pharmaceuticals
    "SUNPHARMA.NS",

    # Metals
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "HINDALCO.NS",

    # Diversified / Infrastructure
    "ADANIENT.NS",
]


def get_india_market_universe():

    return INDIA_MARKET_UNIVERSE.copy()