import yfinance as yf

def get_stock_price(symbol: str):
    stock = yf.Ticker(symbol)
    info = stock.info

    return {
        "symbol": symbol.upper(),
        "company": info.get("longName"),
        "price": info.get("currentPrice"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap")
    }
def get_stock_history(symbol: str, period: str = "1mo"):
    stock = yf.Ticker(symbol)

    history = stock.history(period=period)

    data = []

    for date, row in history.iterrows():
        data.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            }
        )

    return data