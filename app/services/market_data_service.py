import yfinance as yf


def get_stock(symbol):
    return yf.Ticker(symbol)


def get_stock_info(symbol):

    stock = get_stock(symbol)

    try:
        return stock.fast_info
    except Exception:
        return {}


def get_stock_history(symbol, period="6mo"):

    stock = get_stock(symbol)

    return stock.history(period=period)


def get_stock_data(symbol, period="6mo"):

    stock = get_stock(symbol)

    history = stock.history(period=period)

    try:
        info = stock.info
    except Exception:

        info = {}

        try:

            fast = stock.fast_info

            info = {

                "currentPrice": fast.get("lastPrice"),

                "previousClose": fast.get("previousClose"),

                "open": fast.get("open"),

                "dayHigh": fast.get("dayHigh"),

                "dayLow": fast.get("dayLow"),

                "volume": fast.get("lastVolume"),

                "marketCap": fast.get("marketCap"),

                "trailingPE": None,

                "longName": symbol,

                "sector": "N/A",

                "industry": "N/A"

            }

        except Exception:
            pass

    return {

        "info": info,

        "history": history

    }