import yfinance as yf


def get_live_price(symbol: str):

    try:

        yahoo_symbol = symbol.upper()

        # Support Indian NSE symbols
        if not yahoo_symbol.endswith(".NS"):
            yahoo_symbol += ".NS"

        ticker = yf.Ticker(yahoo_symbol)

        history = ticker.history(period="5d")

        if history.empty:
            return {
                "price": 0.0,
                "change": 0.0,
                "change_percent": 0.0
            }

        closes = history["Close"].dropna()

        if closes.empty:
            return {
                "price": 0.0,
                "change": 0.0,
                "change_percent": 0.0
            }

        latest = float(closes.iloc[-1])

        if len(closes) >= 2:
            previous = float(closes.iloc[-2])
        else:
            previous = latest

        change = latest - previous

        change_percent = (
            (change / previous) * 100
            if previous != 0
            else 0.0
        )

        return {
            "price": round(latest, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2)
        }

    except Exception as error:

        print(
            f"Live price error for {symbol}:",
            error
        )

        return {
            "price": 0.0,
            "change": 0.0,
            "change_percent": 0.0
        }
