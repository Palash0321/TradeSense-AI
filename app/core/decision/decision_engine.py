def generate_decision(result):

    score = result["score"]
    signal = result["signal"]
    prediction = result["prediction"]
    reasons = result["reasons"]

    current_price = float(
        str(result["price"]).replace(",", "")
    )

    # -----------------------------
    # Probability
    # -----------------------------

    probability = min(abs(score) + 10, 95)

    # -----------------------------
    # Entry Range
    # -----------------------------

    buffer = current_price * 0.005

    entry_low = round(current_price - buffer, 2)
    entry_high = round(current_price + buffer, 2)

    entry = f"₹ {entry_low} - ₹ {entry_high}"

    # -----------------------------
    # Action
    # -----------------------------

    if signal == "BUY":

        action = "BUY"

        target = prediction["target"]

        stoploss = prediction["stoploss"]

    elif signal == "SELL":

        action = "SELL"

        target = prediction["stoploss"]

        stoploss = prediction["target"]

    else:

        action = "WAIT"

        target = prediction["target"]

        stoploss = prediction["stoploss"]

    return {

        "action": action,

        "probability": probability,

        "entry": entry,

        "target": target,

        "stoploss": stoploss,

        "reasons": reasons[:4]

    }