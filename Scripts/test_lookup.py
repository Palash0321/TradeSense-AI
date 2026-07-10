import yfinance as yf

for symbol in [
    "LTIM.NS",
    "LTIM",
    "LTIMINDTREE.NS",
    "WAAREE.NS",
    "WAAREEENER.NS",
]:
    print(f"\nTesting: {symbol}")

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        print(info.get("longName"))
    except Exception as e:
        print(e)