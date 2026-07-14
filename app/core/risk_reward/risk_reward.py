def calculate_risk_reward(
    current_price,
    target,
    stoploss
):

    reward = abs(target - current_price)

    risk = abs(current_price - stoploss)

    if risk == 0:

        ratio = 0

    else:

        ratio = round(reward / risk, 2)

    if ratio >= 2:

        verdict = "Excellent"

    elif ratio >= 1.5:

        verdict = "Good"

    elif ratio >= 1:

        verdict = "Average"

    else:

        verdict = "Poor"

    return {

        "risk": round(risk,2),

        "reward": round(reward,2),

        "ratio": ratio,

        "verdict": verdict

    }