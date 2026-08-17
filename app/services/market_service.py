import json

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

def get_market_breadth():

    with open(
        "data/market_breadth_stocks.json",
        "r"
    ) as file:

        symbols = json.load(file)

    advancing = 0
    declining = 0
    unchanged = 0

    for symbol in symbols:

        try:

            ticker = yf.Ticker(symbol)

            history = ticker.history(
                period="5d"
            )

            if history.empty:
                continue

            closes = (
                history["Close"]
                .dropna()
            )

            if len(closes) < 2:
                continue

            current = float(
                closes.iloc[-1]
            )

            previous = float(
                closes.iloc[-2]
            )

            if current > previous:

                advancing += 1

            elif current < previous:

                declining += 1

            else:

                unchanged += 1

        except Exception as e:

            print(
                f"{symbol}: {e}"
            )

    if declining == 0:

        ratio = (
            float(advancing)
            if advancing > 0
            else 0
        )

    else:

        ratio = round(
            advancing / declining,
            2
        )

    if ratio > 1.5:

        health = "🟢 Strong"

    elif ratio > 1:

        health = "🟡 Healthy"

    else:

        health = "🔴 Weak"

    return {

        "advancing": advancing,

        "declining": declining,

        "unchanged": unchanged,

        "ratio": ratio,

        "health": health,

        "tracked_stocks": len(symbols)

    }