from __future__ import annotations

import pandas as pd


def calculate_trend_context(
    df=None,
    fast_period=21,
    slow_period=50
):
    """
    Calculate historical trend context.

    This module is analysis-only.

    It uses only candles supplied in df.
    No future candles are used.

    Returns:

        trend:
            BULLISH
            BEARISH
            NEUTRAL

        trend_strength:
            0 - 100

        fast_ema
        slow_ema
        fast_slope
        price_position
    """

    result = {
        "trend": "NEUTRAL",
        "trend_strength": 0,
        "fast_ema": None,
        "slow_ema": None,
        "fast_slope": 0,
        "price_position": "NEUTRAL"
    }

    if df is None or df.empty:

        return result

    if "Close" not in df.columns:

        return result

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    ).dropna()

    minimum_history = max(
        fast_period,
        slow_period
    )

    if len(close) < minimum_history:

        return result

    fast_ema = (
        close
        .ewm(
            span=fast_period,
            adjust=False
        )
        .mean()
    )

    slow_ema = (
        close
        .ewm(
            span=slow_period,
            adjust=False
        )
        .mean()
    )

    current_close = float(
        close.iloc[-1]
    )

    current_fast = float(
        fast_ema.iloc[-1]
    )

    current_slow = float(
        slow_ema.iloc[-1]
    )

    previous_fast = float(
        fast_ema.iloc[-2]
    )

    fast_slope = (
        current_fast
        - previous_fast
    )

    # ---------------------------------
    # Price position
    # ---------------------------------

    if current_close > current_fast:

        price_position = "ABOVE_FAST"

    elif current_close < current_fast:

        price_position = "BELOW_FAST"

    else:

        price_position = "AT_FAST"

    # ---------------------------------
    # Trend classification
    # ---------------------------------

    if (
        current_fast > current_slow
        and
        fast_slope > 0
        and
        current_close > current_fast
    ):

        trend = "BULLISH"

    elif (
        current_fast < current_slow
        and
        fast_slope < 0
        and
        current_close < current_fast
    ):

        trend = "BEARISH"

    else:

        trend = "NEUTRAL"

    # ---------------------------------
    # Trend strength
    # ---------------------------------

    ema_distance = abs(
        current_fast
        - current_slow
    )

    if current_slow != 0:

        ema_gap_pct = (
            ema_distance
            / abs(current_slow)
        ) * 100

    else:

        ema_gap_pct = 0

    slope_pct = 0

    if current_fast != 0:

        slope_pct = (
            abs(fast_slope)
            / abs(current_fast)
        ) * 100

    # Conservative bounded strength.
    gap_score = min(
        70,
        ema_gap_pct * 20
    )

    slope_score = min(
        30,
        slope_pct * 100
    )

    trend_strength = (
        gap_score
        + slope_score
    )

    if trend == "NEUTRAL":

        trend_strength *= 0.5

    trend_strength = max(
        0,
        min(
            100,
            trend_strength
        )
    )

    return {

        "trend":
            trend,

        "trend_strength":
            round(
                float(trend_strength),
                2
            ),

        "fast_ema":
            round(
                current_fast,
                4
            ),

        "slow_ema":
            round(
                current_slow,
                4
            ),

        "fast_slope":
            round(
                fast_slope,
                4
            ),

        "price_position":
            price_position

    }