def get_recommendation(score):

    if score >= 50:
        return "BUY"

    elif score <= -50:
        return "SELL"

    return "HOLD"