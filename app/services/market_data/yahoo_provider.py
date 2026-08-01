import yfinance as yf

from .base import MarketDataProvider


class YahooProvider(MarketDataProvider):

    def get_quote(self, symbol):

        ticker = yf.Ticker(symbol)

        info = ticker.fast_info

        return {
            "symbol": symbol,
            "price": info.get("lastPrice")
        }

    def get_history(self, symbol,
                    period="6mo",
                    interval="1d"):

        ticker = yf.Ticker(symbol)

        return ticker.history(
            period=period,
            interval=interval
        )

    def get_option_chain(self, symbol):

        ticker = yf.Ticker(symbol)

        expiries = ticker.options

        if not expiries:
            return None

        chain = ticker.option_chain(expiries[0])

        return chain