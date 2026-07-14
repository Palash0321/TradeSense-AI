def calculate_support_resistance(history):

    support = round(
        history["Low"].tail(20).min(),
        2
    )

    resistance = round(
        history["High"].tail(20).max(),
        2
    )

    return {

        "support": support,

        "resistance": resistance

    }