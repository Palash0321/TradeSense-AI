import pandas as pd


def calculate_atr(history, period=14):

    high_low = history["High"] - history["Low"]

    high_close = (
        history["High"] -
        history["Close"].shift()
    ).abs()

    low_close = (
        history["Low"] -
        history["Close"].shift()
    ).abs()

    tr = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    history["ATR"] = (

        tr

        .rolling(period)

        .mean()

    )

    return history