import yfinance as yf


# Controlled universe for live sector analysis.
# We intentionally start small to avoid excessive Yahoo Finance requests.
SECTOR_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BHARTIARTL.NS",
]


def get_sector_performance():

    sectors = {}

    for symbol in SECTOR_STOCKS:

        try:

            ticker = yf.Ticker(symbol)

            info = ticker.info

            sector = info.get("sector")

            if not sector:
                continue

            history = ticker.history(period="5d")

            if history.empty or len(history) < 2:
                continue

            closes = history["Close"].dropna()

            if len(closes) < 2:
                continue

            current = float(closes.iloc[-1])

            previous = float(closes.iloc[-2])

            if previous == 0:
                continue

            change_percent = (
                (current - previous)
                / previous
            ) * 100

            if sector not in sectors:

                sectors[sector] = []

            sectors[sector].append(
                change_percent
            )

        except Exception as e:

            print(
                f"Sector data error for {symbol}: {e}"
            )


    result = []

    for sector, changes in sectors.items():

        if not changes:
            continue

        average_change = (
            sum(changes)
            / len(changes)
        )

        result.append({

            "name": sector,

            "change": round(
                average_change,
                2
            ),

            "stocks": len(changes)

        })


    result.sort(
        key=lambda x: x["change"],
        reverse=True
    )

    return result