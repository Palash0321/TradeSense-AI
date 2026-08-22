import math


class TradeHorizonService:

    def __init__(
        self,
        preferred_setup,
        current_price,
        atr,
        entry_engine
    ):

        self.preferred_setup = preferred_setup
        self.current_price = float(
            current_price or 0
        )
        self.atr = float(
            atr or 0
        )
        self.entry_engine = entry_engine or {}

    # -----------------------------------------
    # Estimate trading days using daily ATR
    # -----------------------------------------

    def _estimate_days(
        self,
        start_price,
        target_price
    ):

        if (
            start_price <= 0
            or target_price is None
            or self.atr <= 0
        ):

            return None

        distance = abs(
            float(target_price)
            - float(start_price)
        )

        days = math.ceil(
            distance / self.atr
        )

        return max(
            1,
            days
        )

    # -----------------------------------------
    # Format trading days
    # -----------------------------------------

    def _format_days(
        self,
        days
    ):

        if days is None:
            return None

        if days == 1:
            return "≈ 1 trading day"

        return (
            f"≈ {days} trading days"
        )

    # -----------------------------------------
    # Main calculation
    # -----------------------------------------

    def calculate(self):

        if self.preferred_setup in [
            "BREAKOUT",
            "WAIT_FOR_BREAKOUT"
        ]:

            setup = self.entry_engine.get(
                "breakout",
                {}
            )

            entry = setup.get(
                "entry"
            )

            setup_type = "BREAKOUT"

        elif self.preferred_setup == "PULLBACK":

            setup = self.entry_engine.get(
                "pullback",
                {}
            )

            entry_low = setup.get(
                "entry_low"
            )

            entry_high = setup.get(
                "entry_high"
            )

            if (
                entry_low is None
                or entry_high is None
            ):

                return None

            entry = (
                float(entry_low)
                + float(entry_high)
            ) / 2

            setup_type = "PULLBACK"

        else:

            return None

        if entry is None:

            return None

        target1 = setup.get(
            "target1"
        )

        target2 = setup.get(
            "target2"
        )

        target3 = setup.get(
            "target3"
        )

        if target1 is None:

            return None

        # -----------------------------------------
        # Time to reach confirmed entry
        # -----------------------------------------

        entry_days = self._estimate_days(
            self.current_price,
            entry
        )

        # -----------------------------------------
        # Time to reach each target
        # -----------------------------------------

        target1_days = self._estimate_days(
            entry,
            target1
        )

        target2_days = self._estimate_days(
            entry,
            target2
        )

        target3_days = self._estimate_days(
            entry,
            target3
        )

        valid_target_days = [
            days
            for days in [
                target1_days,
                target2_days,
                target3_days
            ]
            if days is not None
        ]

        if not valid_target_days:

            return None

        first_target_days = min(
            valid_target_days
        )

        last_target_days = max(
            valid_target_days
        )

        if first_target_days == last_target_days:

            overall = (
                f"≈ {first_target_days} "
                "trading day"
            )

        else:

            overall = (
                f"≈ {first_target_days}–"
                f"{last_target_days} "
                "trading days after entry"
            )

        # -----------------------------------------
        # Trading horizon classification
        # -----------------------------------------

        if last_target_days <= 5:

            horizon = "Short-term"

        elif last_target_days <= 15:

            horizon = "Swing"

        elif last_target_days <= 30:

            horizon = "Medium-term"

        else:

            horizon = "Long-term"

        return {

            "setup_type": setup_type,

            "horizon": horizon,

            "overall": overall,

            "entry_activation": (
                self._format_days(
                    entry_days
                )
            ),

            "target1": (
                self._format_days(
                    target1_days
                )
            ),

            "target2": (
                self._format_days(
                    target2_days
                )
            ),

            "target3": (
                self._format_days(
                    target3_days
                )
            ),

            "atr": round(
                self.atr,
                2
            ),

            "basis": (
                "Daily ATR volatility estimate"
            ),

            "note": (
                "Estimated timeframe only. "
                "Actual price movement may take "
                "longer or shorter."
            )
        }