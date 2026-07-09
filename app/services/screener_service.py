from app.services.signal_service import generate_signal


def get_top_stocks(market: str):

    if market == "india":

        stocks = [

        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS"

    ]

    else:

        stocks = [

        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "META",
        "GOOGL",
        "TSLA",
        "NFLX"

    ]

    results = []

    for symbol in stocks:

        try:

            signal = generate_signal(symbol)

            results.append(signal)

        except:

            pass

    return results