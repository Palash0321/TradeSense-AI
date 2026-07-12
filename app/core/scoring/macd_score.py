def macd_score(latest):

    diff = latest["MACD"] - latest["Signal"]

    if diff > 2:
        return 20, "Strong Bullish MACD"

    elif diff > 0:
        return 12, "Bullish MACD"

    elif diff > -2:
        return -10, "Weak Bearish MACD"

    return -20, "Strong Bearish MACD"