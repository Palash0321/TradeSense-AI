def trend_score(latest):

    if latest["MA20"] > latest["MA50"]:
        return 20, "Strong Uptrend (MA20 > MA50)"

    return -20, "Downtrend (MA20 < MA50)"