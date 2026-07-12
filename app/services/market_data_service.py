import yfinance as yf


def get_stock(symbol):

    return yf.Ticker(symbol)


def get_stock_info(symbol):

    stock = get_stock(symbol)

    return stock.info


def get_stock_history(symbol, period="6mo"):

    stock = get_stock(symbol)

    return stock.history(period=period)


def get_stock_data(symbol, period="6mo"):

    stock = get_stock(symbol)

    return {

        "info": stock.info,

        "history": stock.history(period=period)

    }