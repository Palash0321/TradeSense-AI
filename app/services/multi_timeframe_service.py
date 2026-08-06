import yfinance as yf
import pandas as pd


class MultiTimeframeService:

    def __init__(self, symbol):

        self.symbol = symbol

    def analyze_timeframe(self, interval, period):

        df = yf.download(

            self.symbol,

            interval=interval,

            period=period,

            progress=False

        )

        if df.empty:

            return {

                "signal": "UNKNOWN",

                "score": 0

            }

        # Flatten MultiIndex (new yfinance)

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        df["MA20"] = (

            df["Close"]

            .rolling(20)

            .mean()

        )

        df["MA50"] = (

            df["Close"]

            .rolling(50)

            .mean()

        )

        latest = df.iloc[-1]

        score = 0

        if latest["MA20"] > latest["MA50"]:

            score += 1

        if latest["Close"] > latest["MA20"]:

            score += 1

        if score == 2:

            signal = "BUY"

        elif score == 1:

            signal = "HOLD"

        else:

            signal = "SELL"

        return {

            "signal": signal,

            "score": score

        }

    def analyze(self):

        frames = {

            "15m": ("15m", "5d"),

            "1h": ("1h", "1mo"),

            "4h": ("1h", "3mo"),

            "1d": ("1d", "1y"),

            "1w": ("1wk", "5y")

        }

        results = {}

        total = 0

        for name, values in frames.items():

            result = self.analyze_timeframe(

                values[0],

                values[1]

            )

            results[name] = result

            total += result["score"]

        probability = round(

            (total / 10) * 100,

            1

        )

        return {

            "frames": results,

            "overall_probability": probability

        }