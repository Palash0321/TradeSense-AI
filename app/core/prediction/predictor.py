def generate_prediction(latest, score):

    close = float(latest["Close"])

    rsi = float(latest["RSI"])

    macd = float(latest["MACD"])

    signal = float(latest["Signal"])

    ma20 = float(latest["MA20"])

    prediction = {}

    # Tomorrow

    if score >= 50:

        tomorrow = "Bullish"

        expected_move = 2.5

    elif score <= -50:

        tomorrow = "Bearish"

        expected_move = -2.5

    else:

        tomorrow = "Sideways"

        expected_move = 0.5

    # Weekly Outlook

    if ma20 > latest["MA50"] and macd > signal:

        weekly = "Strong Bullish"

    elif ma20 < latest["MA50"] and macd < signal:

        weekly = "Strong Bearish"

    else:

        weekly = "Neutral"

    target = close * (1 + expected_move / 100)

    stoploss = close * (1 - abs(expected_move) / 100)

    prediction["tomorrow"] = tomorrow

    prediction["weekly"] = weekly

    prediction["expected_move"] = round(expected_move, 2)

    prediction["target"] = round(target, 2)

    prediction["stoploss"] = round(stoploss, 2)

    return prediction