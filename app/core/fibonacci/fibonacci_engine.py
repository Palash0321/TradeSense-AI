import pandas as pd


def calculate_fibonacci_context(
    history: pd.DataFrame,
    market_structure: dict
):
    """
    Fibonacci diagnostic engine.

    Purpose:
    - Identify the most recent structural impulse.
    - Calculate Fibonacci retracement levels.
    - Measure current price location within the impulse.
    - Identify premium / discount.
    - Measure distance from key Fibonacci levels.
    - Detect extension / overextension.

    IMPORTANT:
    This module is diagnostic only.

    It does NOT:
    - generate BUY signals
    - generate SELL signals
    - modify BOS / CHOCH
    - modify setup confidence
    - filter trades
    """

    empty_result = {
        "fib_valid": False,
        "fib_direction": None,

        "swing_low": None,
        "swing_high": None,
        "swing_range": None,

        "fib_382": None,
        "fib_500": None,
        "fib_618": None,

        "current_price": None,

        # Normalized structural price position
        # 0.00 = swing low
        # 0.50 = midpoint
        # 1.00 = swing high
        "price_position": None,

        "retracement_percent": None,
        "premium_discount": None,

        "distance_to_382_percent": None,
        "distance_to_500_percent": None,
        "distance_to_618_percent": None,

        "fib_zone": None,
        "extension_percent": None,
        "extension_state": None,
    }

    if history is None or history.empty:
        return empty_result

    if not isinstance(market_structure, dict):
        return empty_result

    required_columns = {
        "High",
        "Low",
        "Close"
    }

    if not required_columns.issubset(history.columns):
        return empty_result

    swing_highs = market_structure.get(
        "swing_highs",
        []
    )

    swing_lows = market_structure.get(
        "swing_lows",
        []
    )

    break_direction = market_structure.get(
        "break_direction"
    )

    if not swing_highs or not swing_lows:
        return empty_result

    df = history.copy()

    df = df.dropna(
        subset=[
            "High",
            "Low",
            "Close"
        ]
    )

    if df.empty:
        return empty_result

    try:
        current_price = float(
            df["Close"].iloc[-1]
        )
    except (TypeError, ValueError):
        return empty_result

    # ==========================================================
    # Determine structural impulse
    # ==========================================================

    latest_high = swing_highs[-1]
    latest_low = swing_lows[-1]

    try:
        high_index = df.index.get_loc(
            latest_high["index"]
        )

        low_index = df.index.get_loc(
            latest_low["index"]
        )
    except (KeyError, TypeError, ValueError):
        return empty_result

    # ----------------------------------------------------------
    # Bullish impulse:
    #
    # swing low occurred before swing high
    #
    # Example:
    #
    # Low ---------> High
    #
    # Fibonacci retracement is measured downward
    # from the high.
    # ----------------------------------------------------------

    if low_index < high_index:

        swing_low = float(
            latest_low["price"]
        )

        swing_high = float(
            latest_high["price"]
        )

        fib_direction = "BULLISH"

    # ----------------------------------------------------------
    # Bearish impulse:
    #
    # swing high occurred before swing low
    #
    # Example:
    #
    # High ---------> Low
    #
    # Fibonacci retracement is measured upward
    # from the low.
    # ----------------------------------------------------------

    elif high_index < low_index:

        swing_high = float(
            latest_high["price"]
        )

        swing_low = float(
            latest_low["price"]
        )

        fib_direction = "BEARISH"

    else:
        return empty_result

    swing_range = swing_high - swing_low

    if swing_range <= 0:
        return empty_result

    # ==========================================================
    # Normalized structural price position
    #
    # 0.00 = swing low
    # 0.50 = midpoint
    # 1.00 = swing high
    #
    # This is an entry-time diagnostic feature only.
    # ==========================================================

    price_position = (
        (current_price - swing_low)
        / swing_range
    )

    # ==========================================================
    # Fibonacci levels
    # ==========================================================

    if fib_direction == "BULLISH":

        fib_382 = (
            swing_high
            - swing_range * 0.382
        )

        fib_500 = (
            swing_high
            - swing_range * 0.500
        )

        fib_618 = (
            swing_high
            - swing_range * 0.618
        )

        # Percentage of impulse retraced downward
        retracement_percent = (
            (swing_high - current_price)
            / swing_range
        ) * 100

    else:

        fib_382 = (
            swing_low
            + swing_range * 0.382
        )

        fib_500 = (
            swing_low
            + swing_range * 0.500
        )

        fib_618 = (
            swing_low
            + swing_range * 0.618
        )

        # Percentage of impulse retraced upward
        retracement_percent = (
            (current_price - swing_low)
            / swing_range
        ) * 100

    # ==========================================================
    # Premium / Discount
    # ==========================================================

    equilibrium = (
        swing_low
        + swing_range * 0.500
    )

    if current_price > equilibrium:

        premium_discount = "PREMIUM"

    elif current_price < equilibrium:

        premium_discount = "DISCOUNT"

    else:

        premium_discount = "EQUILIBRIUM"

    # ==========================================================
    # Distance from Fibonacci levels
    #
    # Percentage is measured relative to current price.
    # ==========================================================

    distance_to_382_percent = (
        (current_price - fib_382)
        / current_price
    ) * 100

    distance_to_500_percent = (
        (current_price - fib_500)
        / current_price
    ) * 100

    distance_to_618_percent = (
        (current_price - fib_618)
        / current_price
    ) * 100

    # ==========================================================
    # Fibonacci zone
    # ==========================================================

    if fib_direction == "BULLISH":

        if current_price >= fib_382:
            fib_zone = "ABOVE_382"

        elif current_price >= fib_500:
            fib_zone = "382_500"

        elif current_price >= fib_618:
            fib_zone = "500_618"

        else:
            fib_zone = "BELOW_618"

    else:

        if current_price <= fib_382:
            fib_zone = "BELOW_382"

        elif current_price <= fib_500:
            fib_zone = "382_500"

        elif current_price <= fib_618:
            fib_zone = "500_618"

        else:
            fib_zone = "ABOVE_618"

    # ==========================================================
    # Extension
    #
    # Negative retracement means price has moved beyond
    # the impulse extreme.
    # ==========================================================

    extension_percent = None
    extension_state = "NORMAL"

    if fib_direction == "BULLISH":

        if current_price > swing_high:

            extension_percent = (
                (current_price - swing_high)
                / swing_range
            ) * 100

            extension_state = "BULLISH_EXTENSION"

        elif current_price < swing_low:

            extension_percent = (
                (swing_low - current_price)
                / swing_range
            ) * 100

            extension_state = "BEARISH_EXTENSION"

    else:

        if current_price < swing_low:

            extension_percent = (
                (swing_low - current_price)
                / swing_range
            ) * 100

            extension_state = "BEARISH_EXTENSION"

        elif current_price > swing_high:

            extension_percent = (
                (current_price - swing_high)
                / swing_range
            ) * 100

            extension_state = "BULLISH_EXTENSION"

    # ==========================================================
    # Structural direction override
    #
    # If a confirmed structural break exists, preserve it
    # as diagnostic metadata.
    # ==========================================================

    if break_direction == "BULLISH":
        diagnostic_direction = "BULLISH"

    elif break_direction == "BEARISH":
        diagnostic_direction = "BEARISH"

    else:
        diagnostic_direction = fib_direction

    # ==========================================================
    # Return
    # ==========================================================

    return {

        "fib_valid": True,

        "fib_direction": diagnostic_direction,

        "swing_low": round(
            swing_low,
            2
        ),

        "swing_high": round(
            swing_high,
            2
        ),

        "swing_range": round(
            swing_range,
            2
        ),

        "fib_382": round(
            fib_382,
            2
        ),

        "fib_500": round(
            fib_500,
            2
        ),

        "fib_618": round(
            fib_618,
            2
        ),

        "current_price": round(
            current_price,
            2
        ),

        "price_position": round(
            price_position,
            4
        ),

        "retracement_percent": round(
            retracement_percent,
            2
        ),

        "premium_discount": premium_discount,

        "distance_to_382_percent": round(
            distance_to_382_percent,
            3
        ),

        "distance_to_500_percent": round(
            distance_to_500_percent,
            3
        ),

        "distance_to_618_percent": round(
            distance_to_618_percent,
            3
        ),

        "fib_zone": fib_zone,

        "extension_percent": (
            round(extension_percent, 2)
            if extension_percent is not None
            else None
        ),

        "extension_state": extension_state,
    }