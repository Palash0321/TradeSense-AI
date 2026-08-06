import pandas as pd


class CandlestickService:

    def __init__(self, history):
        self.history = history

    def detect(self):

        df = self.history.copy()

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        open_price = float(latest["Open"])
        close_price = float(latest["Close"])
        high = float(latest["High"])
        low = float(latest["Low"])

        body = abs(close_price - open_price)
        candle_range = high - low

        upper_shadow = high - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low

        patterns = []

        # --------------------------
        # Doji
        # --------------------------

        if candle_range > 0:

            if body / candle_range < 0.1:
                patterns.append("Doji")

        # --------------------------
        # Hammer
        # --------------------------

        if lower_shadow > body * 2 and upper_shadow < body:
            patterns.append("Hammer")

        # --------------------------
        # Shooting Star
        # --------------------------

        if upper_shadow > body * 2 and lower_shadow < body:
            patterns.append("Shooting Star")

        # --------------------------
        # Bullish Engulfing
        # --------------------------

        if (

            previous["Close"] < previous["Open"]

            and

            close_price > open_price

            and

            close_price > previous["Open"]

            and

            open_price < previous["Close"]

        ):

            patterns.append("Bullish Engulfing")

        # --------------------------
        # Bearish Engulfing
        # --------------------------

        if (

            previous["Close"] > previous["Open"]

            and

            close_price < open_price

            and

            open_price > previous["Close"]

            and

            close_price < previous["Open"]

        ):

            patterns.append("Bearish Engulfing")

        return patterns