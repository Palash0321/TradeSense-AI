def price_score(latest):

    distance = (
        latest["Close"] - latest["MA20"]
    ) / latest["MA20"] * 100

    if distance > 5:
        return 15, "Price strongly above MA20"

    elif distance > 0:
        return 10, "Price above MA20"

    elif distance > -5:
        return -5, "Price slightly below MA20"

    return -15, "Price well below MA20"