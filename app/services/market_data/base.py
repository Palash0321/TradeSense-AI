from abc import ABC, abstractmethod


class MarketDataProvider(ABC):

    @abstractmethod
    def get_quote(self, symbol):
        pass

    @abstractmethod
    def get_history(self, symbol, period="6mo", interval="1d"):
        pass

    @abstractmethod
    def get_option_chain(self, symbol):
        pass