def get_recommendation(score):

    if score >= 45:
        return "STRONG BUY"

    elif score >= 20:
        return "BUY"

    elif score > -20:
        return "HOLD"

    elif score > -45:
        return "SELL"

    else:
        return "STRONG SELL"