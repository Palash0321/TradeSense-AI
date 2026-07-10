import yfinance as yf


def lookup_stock(symbol):

    try:

        india = yf.Ticker(symbol + ".NS")

        info = india.info

        if info.get("longName"):

            return {

                "symbol": symbol,

                "name": info.get("longName"),

                "market": "india",

                "exchange": "NSE"

            }

    except:

        pass

    try:

        us = yf.Ticker(symbol)

        info = us.info

        if info.get("longName"):

            return {

                "symbol": symbol,

                "name": info.get("longName"),

                "market": "us",

                "exchange": "NASDAQ"

            }

    except:

        pass

    return None