def calculate_macd(history):

    history["EMA12"] = history["Close"].ewm(span=12, adjust=False).mean()

    history["EMA26"] = history["Close"].ewm(span=26, adjust=False).mean()

    history["MACD"] = history["EMA12"] - history["EMA26"]

    history["Signal"] = history["MACD"].ewm(span=9, adjust=False).mean()

    return history