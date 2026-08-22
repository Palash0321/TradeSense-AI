import yfinance as yf
import pandas as pd


class MultiTimeframeService:

    def __init__(self, symbol):

        self.symbol = symbol

    def _download(self, interval, period):

        df = yf.download(
            self.symbol,
            interval=interval,
            period=period,
            progress=False
        )

        if df.empty:
            return df

        # Flatten MultiIndex returned by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        return df

    def _calculate_signal(self, df):

        if df.empty:
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

        # Make sure enough candles exist for MA50
        if len(df) < 50:
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

        df = df.copy()

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

        if (
            pd.isna(latest["MA20"])
            or
            pd.isna(latest["MA50"])
        ):
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

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

    def analyze_timeframe(self, interval, period):

        df = self._download(
            interval,
            period
        )

        return self._calculate_signal(df)

    def analyze_4h(self):

        # Yahoo provides reliable 1h intraday data for this use case.
        # We construct genuine 4h candles instead of treating 1h
        # candles as 4h candles.

        df = self._download(
            "1h",
            "60d"
        )

        if df.empty:
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

        # Convert to a clean timezone-aware index.
        if df.index.tz is None:

            df.index = df.index.tz_localize(
                "UTC"
            )

        df.index = df.index.tz_convert(
            "Asia/Kolkata"
        )

        # Keep normal NSE cash-market session.
        df = df.between_time(
            "09:15",
            "15:30"
        )

        if df.empty:
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

        # Build 4-hour candles from 1-hour candles.
        #
        # The origin is aligned with the NSE session rather than
        # midnight so the first candle starts around 09:15.

        df = df.copy()

        df["session_date"] = df.index.date

        grouped = []

        for _, day in df.groupby("session_date"):

            day = day.drop(
                columns=["session_date"]
            )

            if day.empty:
                continue

            candle = day.resample(
                "4h",
                origin="start_day",
                offset="9h15min",
                label="right",
                closed="right"
            ).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            })

            candle = candle.dropna(
                subset=["Open", "High", "Low", "Close"]
            )

            if not candle.empty:
                grouped.append(candle)

        if not grouped:
            return {
                "signal": "UNKNOWN",
                "score": 0
            }

        four_hour = pd.concat(
            grouped
        ).sort_index()

        return self._calculate_signal(
            four_hour
        )

    def analyze(self):

        frames = {

            "15m": ("15m", "5d"),

            "1h": ("1h", "1mo"),

            "1d": ("1d", "1y"),

            "1w": ("1wk", "5y")

        }

        results = {}

        total = 0

        # Normal timeframes
        for name, values in frames.items():

            result = self.analyze_timeframe(
                values[0],
                values[1]
            )

            results[name] = result

            total += result["score"]

        # Genuine 4-hour timeframe
        four_hour_result = self.analyze_4h()

        results["4h"] = four_hour_result

        total += four_hour_result["score"]

        probability = round(
            (total / 10) * 100,
            1
        )

        return {

            "frames": results,

            "overall_probability": probability

        }
