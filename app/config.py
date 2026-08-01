import os

# -----------------------------
# Application
# -----------------------------
APP_NAME = "StockPilot"

DEFAULT_MARKET = "NSE"

DEFAULT_PERIOD = "1y"

CACHE_TIMEOUT = 300


# -----------------------------
# API Keys
# -----------------------------

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")


# -----------------------------
# Cache
# -----------------------------

LIVE_PRICE_CACHE = 60

NEWS_CACHE = 300

MARKET_CACHE = 300