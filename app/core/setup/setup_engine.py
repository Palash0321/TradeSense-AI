from __future__ import annotations


def calculate_setup(
    market_structure=None,
    liquidity=None,
    current_price=None,
):
    """
    Determine the structural trading setup.

    This module is analysis-only.
    It does not generate BUY/SELL decisions.

    Possible setups:

    LONG_CONTINUATION
    LONG_REVERSAL
    SHORT_CONTINUATION
    SHORT_REVERSAL
    WAIT
    """

    market_structure = (
        market_structure
        if market_structure
        else {}
    )

    liquidity = (
        liquidity
        if liquidity
        else {}
    )

    if current_price is None:

        return {
            "setup": "WAIT",
            "direction": None,
            "confidence": 0,
            "reason": "Current price is unavailable."
        }

    current_price = float(current_price)

    bias = market_structure.get(
        "bias",
        "NEUTRAL"
    )

    structure = market_structure.get(
        "structure",
        "UNKNOWN"
    )

    bos = market_structure.get(
        "bos"
    )

    choch = market_structure.get(
        "choch"
    )

    break_confirmed = market_structure.get(
        "break_confirmed",
        False
    )

    sweep = liquidity.get(
        "sweep",
        {}
    )

    sweep_detected = sweep.get(
        "detected",
        False
    )

    sweep_direction = sweep.get(
        "direction"
    )

    # ==========================================
    # Default
    # ==========================================

    setup = "WAIT"
    direction = None
    confidence = 0

    reasons = []

    # ==========================================
    # Bullish continuation
    # ==========================================

    if (
        bias == "BULLISH"
        and
        bos == "BULLISH"
        and
        break_confirmed
    ):

        setup = "LONG_CONTINUATION"
        direction = "LONG"

        confidence = 70

        reasons.append(
            "Bullish structure confirmed by bullish BOS."
        )

        if sweep_detected:

            if sweep_direction == "SELL_SIDE":

                confidence += 15

                reasons.append(
                    "Sell-side liquidity sweep supports "
                    "bullish continuation."
                )

            elif sweep_direction == "BUY_SIDE":

                confidence -= 15

                reasons.append(
                    "Buy-side liquidity sweep weakens "
                    "bullish continuation."
                )

    # ==========================================
    # Bearish reversal
    # ==========================================

    elif (
        bias == "BULLISH"
        and
        choch == "BEARISH"
        and
        break_confirmed
    ):

        setup = "SHORT_REVERSAL"
        direction = "SHORT"

        confidence = 65

        reasons.append(
            "Bearish CHOCH against bullish structure "
            "indicates a potential bearish reversal."
        )

    # ==========================================
    # Bearish continuation
    # ==========================================

    elif (
        bias == "BEARISH"
        and
        bos == "BEARISH"
        and
        break_confirmed
    ):

        setup = "SHORT_CONTINUATION"
        direction = "SHORT"

        confidence = 70

        reasons.append(
            "Bearish structure confirmed by bearish BOS."
        )

        if sweep_detected:

            if sweep_direction == "BUY_SIDE":

                confidence += 15

                reasons.append(
                    "Buy-side liquidity sweep supports "
                    "bearish continuation."
                )

            elif sweep_direction == "SELL_SIDE":

                confidence -= 15

                reasons.append(
                    "Sell-side liquidity sweep weakens "
                    "bearish continuation."
                )

    # ==========================================
    # Bullish reversal
    # ==========================================

    elif (
        bias == "BEARISH"
        and
        choch == "BULLISH"
        and
        break_confirmed
    ):

        setup = "LONG_REVERSAL"
        direction = "LONG"

        confidence = 65

        reasons.append(
            "Bullish CHOCH against bearish structure "
            "indicates a potential bullish reversal."
        )

    # ==========================================
    # Liquidity-only confirmation
    # ==========================================

    elif sweep_detected:

        if (
            bias == "BULLISH"
            and
            sweep_direction == "SELL_SIDE"
        ):

            setup = "LONG_REVERSAL"
            direction = "LONG"

            confidence = 55

            reasons.append(
                "Sell-side liquidity sweep occurred "
                "within a bullish structural environment."
            )

        elif (
            bias == "BEARISH"
            and
            sweep_direction == "BUY_SIDE"
        ):

            setup = "SHORT_REVERSAL"
            direction = "SHORT"

            confidence = 55

            reasons.append(
                "Buy-side liquidity sweep occurred "
                "within a bearish structural environment."
            )

    # ==========================================
    # Safety adjustment
    # ==========================================

    confidence = max(
        0,
        min(
            100,
            confidence
        )
    )

    if not reasons:

        reasons.append(
            "No confirmed structural setup is currently present."
        )

    return {
        "setup": setup,
        "direction": direction,
        "confidence": confidence,
        "reason": " ".join(reasons),
        "structure": structure,
        "bias": bias,
        "bos": bos,
        "choch": choch,
        "break_confirmed": break_confirmed,
        "liquidity_sweep": sweep
    }