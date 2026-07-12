def calculate_moving_averages(history):

    history["MA20"] = history["Close"].rolling(window=20).mean()

    history["MA50"] = history["Close"].rolling(window=50).mean()

    return history