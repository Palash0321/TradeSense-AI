def rsi_score(latest):

    rsi = latest["RSI"]

    if rsi < 30:
        return 15, "RSI Oversold"

    elif rsi < 45:
        return 8, "RSI Recovering"

    elif rsi <= 65:
        return 12, "Healthy RSI"

    elif rsi <= 75:
        return -5, "RSI Slightly Overbought"

    return -15, "RSI Extremely Overbought"