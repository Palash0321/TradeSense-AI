def explain(result):

    explanation = []

    score = result["score"]

    rsi = result["RSI"]

    macd = result["MACD"]

    signal = result["Signal"]

    # ------------------------
    # Trend
    # ------------------------

    if score > 40:

        explanation.append(

            "Strong bullish momentum detected."

        )

    elif score < -40:

        explanation.append(

            "Strong bearish momentum detected."

        )

    else:

        explanation.append(

            "Market is moving sideways."

        )

    # ------------------------
    # RSI
    # ------------------------

    if rsi < 30:

        explanation.append(

            "RSI indicates the stock is oversold."

        )

    elif rsi > 70:

        explanation.append(

            "RSI indicates the stock is overbought."

        )

    else:

        explanation.append(

            "RSI is in a healthy range."

        )

    # ------------------------
    # MACD
    # ------------------------

    if macd > signal:

        explanation.append(

            "MACD confirms bullish momentum."

        )

    else:

        explanation.append(

            "MACD confirms bearish momentum."

        )

    return explanation