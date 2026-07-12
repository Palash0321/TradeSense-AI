from datetime import datetime


def get_market_status():

    now = datetime.now()

    weekday = now.weekday()

    hour = now.hour

    minute = now.minute

    if weekday >= 5:
        return "CLOSED"

    if (
        (hour == 9 and minute >= 15)
        or (9 < hour < 15)
        or (hour == 15 and minute <= 30)
    ):
        return "OPEN"

    return "CLOSED"