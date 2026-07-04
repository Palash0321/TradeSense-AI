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