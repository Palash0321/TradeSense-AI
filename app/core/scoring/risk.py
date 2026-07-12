def calculate_risk(score):

    score = abs(score)

    if score >= 85:
        return "Low"

    elif score >= 70:
        return "Medium"

    elif score >= 50:
        return "High"

    return "Very High"