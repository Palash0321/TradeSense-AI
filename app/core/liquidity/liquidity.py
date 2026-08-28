from __future__ import annotations



def _find_equal_levels(
    prices,
    tolerance_percent=0.3
):
    """
    Identify approximately equal price levels.

    Two levels are considered equal when their difference
    is within tolerance_percent of the reference level.
    """

    levels = []

    for price in prices:

        price = float(price)

        matched = False

        for level in levels:

            reference = level["price"]

            difference_percent = (
                abs(price - reference)
                / reference
            ) * 100

            if difference_percent <= tolerance_percent:

                level["touches"] += 1

                level["prices"].append(
                    round(price, 2)
                )

                level["price"] = round(
                    sum(level["prices"])
                    / len(level["prices"]),
                    2
                )

                matched = True
                break

        if not matched:

            levels.append(
                {
                    "price": round(price, 2),
                    "touches": 1,
                    "prices": [
                        round(price, 2)
                    ]
                }
            )

    return [
        level
        for level in levels
        if level["touches"] >= 2
    ]


def calculate_liquidity(
    history,
    market_structure=None,
    tolerance_percent=0.3
):
    """
    Analyze liquidity around recent swing structure.

    This module is intentionally analysis-only.
    It does not generate BUY/SELL decisions.
    """

    if (
        history is None
        or history.empty
    ):

        return {
            "equal_highs": [],
            "equal_lows": [],
            "buy_side_liquidity": [],
            "sell_side_liquidity": [],
            "sweep": {
                "detected": False,
                "direction": None,
                "level": None,
                "date": None
            },
            "nearest_liquidity": {
                "above": None,
                "below": None
            },
            "strength": 0
        }

    required_columns = {
        "High",
        "Low",
        "Close"
    }

    if not required_columns.issubset(
        history.columns
    ):

        return {
            "equal_highs": [],
            "equal_lows": [],
            "buy_side_liquidity": [],
            "sell_side_liquidity": [],
            "sweep": {
                "detected": False,
                "direction": None,
                "level": None,
                "date": None
            },
            "nearest_liquidity": {
                "above": None,
                "below": None
            },
            "strength": 0
        }

    structure = (
        market_structure
        if market_structure
        else {}
    )

    swing_highs = structure.get(
        "swing_highs",
        []
    )

    swing_lows = structure.get(
        "swing_lows",
        []
    )

    # ------------------------------------------
    # Extract swing prices
    # ------------------------------------------

    high_prices = [
        float(item["price"])
        for item in swing_highs
        if "price" in item
    ]

    low_prices = [
        float(item["price"])
        for item in swing_lows
        if "price" in item
    ]

    # ------------------------------------------
    # Equal highs / equal lows
    # ------------------------------------------

    equal_highs = _find_equal_levels(
        high_prices,
        tolerance_percent
    )

    equal_lows = _find_equal_levels(
        low_prices,
        tolerance_percent
    )

    # ------------------------------------------
    # Liquidity pools
    # ------------------------------------------

    buy_side_liquidity = [
        {
            "price": level["price"],
            "touches": level["touches"],
            "type": "EQUAL_HIGH"
        }
        for level in equal_highs
    ]

    sell_side_liquidity = [
        {
            "price": level["price"],
            "touches": level["touches"],
            "type": "EQUAL_LOW"
        }
        for level in equal_lows
    ]

    # ------------------------------------------
    # Current price
    # ------------------------------------------

    current_price = float(
        history["Close"].iloc[-1]
    )

    # ------------------------------------------
    # Directional nearest liquidity
    # ------------------------------------------
    #
    # Buy-side liquidity is used as upside
    # liquidity and therefore searched above price.
    #
    # Sell-side liquidity is used as downside
    # liquidity and therefore searched below price.
    #

    above = [
        item
        for item in buy_side_liquidity
        if item["price"] > current_price
    ]

    below = [
        item
        for item in sell_side_liquidity
        if item["price"] < current_price
    ]

    nearest_above = None

    if above:

        nearest_above = min(
            above,
            key=lambda item: item["price"]
        )

    nearest_below = None

    if below:

        nearest_below = max(
            below,
            key=lambda item: item["price"]
        )

    # ------------------------------------------
    # Liquidity sweep detection
    # ------------------------------------------

    sweep = {
        "detected": False,
        "direction": None,
        "level": None,
        "date": None
    }

    if len(history) >= 2:

        previous = history.iloc[-2]
        current = history.iloc[-1]

        current_high = float(
            current["High"]
        )

        current_low = float(
            current["Low"]
        )

        current_close = float(
            current["Close"]
        )

        previous_close = float(
            previous["Close"]
        )

        # Buy-side sweep:
        # Price trades above a liquidity pool
        # but closes back below it.

        for level in buy_side_liquidity:

            price = level["price"]

            if (
                current_high > price
                and
                current_close < price
                and
                previous_close <= price
            ):

                sweep = {
                    "detected": True,
                    "direction": "BUY_SIDE",
                    "level": price,
                    "date": str(
                        history.index[-1].date()
                    )
                }

                break

        # Sell-side sweep:
        # Price trades below a liquidity pool
        # but closes back above it.

        if not sweep["detected"]:

            for level in sell_side_liquidity:

                price = level["price"]

                if (
                    current_low < price
                    and
                    current_close > price
                    and
                    previous_close >= price
                ):

                    sweep = {
                        "detected": True,
                        "direction": "SELL_SIDE",
                        "level": price,
                        "date": str(
                            history.index[-1].date()
                        )
                    }

                    break

    # ------------------------------------------
    # Liquidity strength
    # ------------------------------------------

    strength = 0

    if equal_highs:

        strength += 25

    if equal_lows:

        strength += 25

    if nearest_above is not None:

        strength += 15

    if nearest_below is not None:

        strength += 15

    if sweep["detected"]:

        strength += 20

    strength = max(
        0,
        min(100, strength)
    )

    return {
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "buy_side_liquidity": buy_side_liquidity,
        "sell_side_liquidity": sell_side_liquidity,
        "sweep": sweep,
        "nearest_liquidity": {
            "above": nearest_above,
            "below": nearest_below
        },
        "strength": strength
    }
