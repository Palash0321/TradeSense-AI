import pandas as pd


def calculate_market_structure(
    history: pd.DataFrame,
    swing_window: int = 3
):
    """
    Market structure engine.

    Detects:
    - Swing highs / lows
    - HH / HL / LH / LL
    - Structural bias
    - Confirmed BOS
    - CHOCH
    - Broken structure level
    - Break date
    - Structure strength

    BOS is treated as a structural EVENT rather than
    simply a permanent state.
    """

    empty_result = {
        "bias": "NEUTRAL",
        "structure": "UNKNOWN",
        "swing_highs": [],
        "swing_lows": [],
        "last_swing_high": None,
        "last_swing_low": None,
        "bos": None,
        "choch": None,
        "break_direction": None,
        "break_level": None,
        "break_date": None,
        "break_confirmed": False,
        "strength": 0
    }

    if history is None or history.empty:
        return empty_result

    required_columns = {
        "High",
        "Low",
        "Close"
    }

    if not required_columns.issubset(
        history.columns
    ):
        return empty_result

    df = history.copy()

    df = df.dropna(
        subset=[
            "High",
            "Low",
            "Close"
        ]
    )

    if len(df) < (
        swing_window * 2 + 1
    ):
        result = empty_result.copy()
        result["structure"] = "INSUFFICIENT_DATA"
        return result

    # ==========================================
    # Swing detection
    # ==========================================

    swing_highs = []
    swing_lows = []

    for i in range(
        swing_window,
        len(df) - swing_window
    ):

        high = float(
            df["High"].iloc[i]
        )

        low = float(
            df["Low"].iloc[i]
        )

        left_highs = df[
            "High"
        ].iloc[
            i - swing_window:i
        ]

        right_highs = df[
            "High"
        ].iloc[
            i + 1:i + swing_window + 1
        ]

        left_lows = df[
            "Low"
        ].iloc[
            i - swing_window:i
        ]

        right_lows = df[
            "Low"
        ].iloc[
            i + 1:i + swing_window + 1
        ]

        if (
            high > left_highs.max()
            and
            high > right_highs.max()
        ):
            swing_highs.append({
                "index": df.index[i],
                "price": round(high, 2)
            })

        if (
            low < left_lows.min()
            and
            low < right_lows.min()
        ):
            swing_lows.append({
                "index": df.index[i],
                "price": round(low, 2)
            })

    last_swing_high = (
        swing_highs[-1]["price"]
        if swing_highs
        else None
    )

    last_swing_low = (
        swing_lows[-1]["price"]
        if swing_lows
        else None
    )

    # ==========================================
    # HH / HL / LH / LL
    # ==========================================

    higher_high = False
    higher_low = False
    lower_high = False
    lower_low = False

    if len(swing_highs) >= 2:

        previous_high = swing_highs[-2]["price"]
        current_high = swing_highs[-1]["price"]

        higher_high = (
            current_high > previous_high
        )

        lower_high = (
            current_high < previous_high
        )

    if len(swing_lows) >= 2:

        previous_low = swing_lows[-2]["price"]
        current_low = swing_lows[-1]["price"]

        higher_low = (
            current_low > previous_low
        )

        lower_low = (
            current_low < previous_low
        )

    # ==========================================
    # Structure
    # ==========================================

    if higher_high and higher_low:

        structure = "HH_HL"
        bias = "BULLISH"

    elif lower_high and lower_low:

        structure = "LH_LL"
        bias = "BEARISH"

    elif higher_high or higher_low:

        structure = "PARTIAL_BULLISH"
        bias = "BULLISH"

    elif lower_high or lower_low:

        structure = "PARTIAL_BEARISH"
        bias = "BEARISH"

    else:

        structure = "NEUTRAL"
        bias = "NEUTRAL"

    # ==========================================
    # Confirmed structural break
    # ==========================================

        # ==========================================
    # Structural break confirmation buffer
    # ==========================================

    if "ATR" in df.columns:

        atr_series = pd.to_numeric(
            df["ATR"],
            errors="coerce"
        )

        current_atr = float(
            atr_series.iloc[-1]
        ) if pd.notna(
            atr_series.iloc[-1]
        ) else 0.0

    else:

        true_range = pd.concat(
            [
                df["High"] - df["Low"],
                (
                    df["High"]
                    - df["Close"].shift(1)
                ).abs(),
                (
                    df["Low"]
                    - df["Close"].shift(1)
                ).abs()
            ],
            axis=1
        ).max(axis=1)

        current_atr = float(
            true_range.rolling(14).mean().iloc[-1]
        )

    confirmation_buffer = (
        current_atr * 0.20
    )

    bos = None
    choch = None

    break_direction = None
    break_level = None
    break_date = None
    break_confirmed = False

    # ------------------------------------------
    # Find the most recent relevant swing
    # ------------------------------------------

    candidate_high = (
        swing_highs[-1]
        if swing_highs
        else None
    )

    candidate_low = (
        swing_lows[-1]
        if swing_lows
        else None
    )

    # ------------------------------------------
    # Search chronologically for a CLOSE beyond
    # the most recent structural swing.
    #
    # This prevents every subsequent candle from
    # being treated as a new BOS.
    # ------------------------------------------

    break_event = None

    # ------------------------------------------
    # Find the FIRST confirmed structural break.
    #
    # A close beyond a structural swing level
    # creates the event. Later candles continuing
    # beyond that level do not create another BOS.
    # ------------------------------------------

    break_candidates = []

    if candidate_high is not None:

        high_index = df.index.get_loc(
            candidate_high["index"]
        )

        for i in range(
            high_index + 1,
            len(df)
        ):

            close = float(
                df["Close"].iloc[i]
            )

            if close > (
                candidate_high["price"]
                + confirmation_buffer
            ):

                break_candidates.append({
                    "direction": "BULLISH",
                    "level": candidate_high["price"],
                    "index": df.index[i]
                })

                break

    if candidate_low is not None:

        low_index = df.index.get_loc(
            candidate_low["index"]
        )

        for i in range(
            low_index + 1,
            len(df)
        ):

            close = float(
                df["Close"].iloc[i]
            )

            if close < (
                candidate_low["price"]
                - confirmation_buffer
            ):

                break_candidates.append({
                    "direction": "BEARISH",
                    "level": candidate_low["price"],
                    "index": df.index[i]
                })

                break

    # ------------------------------------------
    # Select the earliest structural event.
    # ------------------------------------------

    if break_candidates:

        break_event = min(
            break_candidates,
            key=lambda event: event["index"]
        )

    # ==========================================
    # Confirm latest break
    # ==========================================

    if break_event is not None:

        break_direction = (
            break_event["direction"]
        )

        break_level = round(
            break_event["level"],
            2
        )

        break_date = (
            break_event["index"]
            .strftime("%Y-%m-%d")
        )

        break_confirmed = True

        # --------------------------------------
        # BOS vs CHOCH
        # --------------------------------------

        if (
            break_direction == "BULLISH"
            and
            bias == "BULLISH"
        ):

            bos = "BULLISH"

        elif (
            break_direction == "BEARISH"
            and
            bias == "BEARISH"
        ):

            bos = "BEARISH"

        elif (
            break_direction == "BULLISH"
            and
            bias == "BEARISH"
        ):

            choch = "BULLISH"

        elif (
            break_direction == "BEARISH"
            and
            bias == "BULLISH"
        ):

            choch = "BEARISH"

    # ==========================================
    # Structure strength
    # ==========================================

    strength = 0

    if higher_high:
        strength += 25

    if higher_low:
        strength += 25

    if lower_high:
        strength += 25

    if lower_low:
        strength += 25

    if break_confirmed:
        strength += 20

    strength = min(
        strength,
        100
    )

    return {

        "bias": bias,

        "structure": structure,

        "swing_highs": swing_highs,

        "swing_lows": swing_lows,

        "last_swing_high": last_swing_high,

        "last_swing_low": last_swing_low,

        "bos": bos,

        "choch": choch,

        "break_direction": break_direction,

        "break_level": break_level,

        "break_date": break_date,

        "break_confirmed": break_confirmed,

        "strength": strength
    }