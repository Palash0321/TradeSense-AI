def calculate_confidence(score):

    confidence = max(0, min(abs(score), 100))

    return f"{confidence}%"