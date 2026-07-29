import yfinance as yf


INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI"
}


def get_market_indices():

    data = []

    for name, symbol in INDICES.items():

        try:

            ticker = yf.Ticker(symbol)

            history = ticker.history(period="5d")

            if history.empty:
                continue

            closes = history["Close"].dropna()

            current = float(closes.iloc[-1])
            previous = float(closes.iloc[-2])

            change = current - previous
            change_percent = (change / previous) * 100

            data.append({
                "name": name,
                "price": round(current, 2),
                "change": round(change, 2),
                "change_percent": round(change_percent, 2),
                "positive": change >= 0
            })

        except Exception as e:
            print(f"{name}: {e}")

    return data