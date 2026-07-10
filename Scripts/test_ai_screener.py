import sys
import os

sys.path.append(os.getcwd())

from app.services.ai_screener_service import get_ai_picks

symbols = [

    "RELIANCE.NS",

    "TCS.NS",

    "INFY.NS",

    "HDFCBANK.NS",

    "SBIN.NS",

    "AAPL",

    "MSFT",

    "NVDA",

    "TSLA",

    "META"

]

results = get_ai_picks(symbols)

for stock in results:

    print(stock)