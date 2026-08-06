import yfinance as yf


def get_live_price(symbol: str):

    try:

        yahoo_symbol = symbol.upper()

        if not yahoo_symbol.endswith(".NS"):

            yahoo_symbol += ".NS"

        ticker = yf.Ticker(yahoo_symbol)

        history = ticker.history(period="1d")

        if history.empty:

            return 0.0

        return float(history["Close"].iloc[-1])

    except Exception:

        return 0.0