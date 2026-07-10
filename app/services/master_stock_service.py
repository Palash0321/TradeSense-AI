from yahooquery import Screener


def get_indian_stocks():

    screener = Screener()

    data = screener.get_screeners("day_gainers")

    print(data)