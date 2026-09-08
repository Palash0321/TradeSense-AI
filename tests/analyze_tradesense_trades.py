import csv
import random
import os
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


TRADES_FILE = "tests/output/tradesense_trades.csv"


def load_trades():

    with open(
        TRADES_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def to_float(value, default=0.0):

    if value in (
        None,
        "",
        "None",
        "nan",
    ):
        return default

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

def to_bool(value, default=False):
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        value = value.strip().lower()

        if value in (
            "true",
            "1",
            "yes",
        ):
            return True

        if value in (
            "false",
            "0",
            "no",
            "",
            "none",
            "nan",
        ):
            return False

    return default


def performance_label(trades):

    if not trades:
        return (
            "Trades=0 "
            "Win=0.00% "
            "PF=0.00 "
            "P&L=0.00 "
            "R=0.00 "
            "AvgR=0.000"
        )

    r_values = [
        to_float(
            trade.get("r_multiple")
        )
        for trade in trades
    ]

    profit_values = [
        to_float(
            trade.get("profit")
        )
        for trade in trades
    ]

    wins = [
        r
        for r in r_values
        if r > 0
    ]

    losses = [
        r
        for r in r_values
        if r <= 0
    ]

    gross_profit = sum(wins)
    gross_loss = abs(
        sum(losses)
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit
            / gross_loss
        )
    else:
        profit_factor = 0.0

    win_rate = (
        len(wins)
        / len(r_values)
        * 100
    )

    total_r = sum(r_values)

    avg_r = (
        total_r
        / len(r_values)
    )

    total_profit = sum(
        profit_values
    )

    return (
        f"Trades={len(r_values):>3} "
        f"Win={win_rate:6.2f}% "
        f"PF={profit_factor:5.2f} "
        f"P&L={total_profit:9.2f} "
        f"R={total_r:7.2f} "
        f"AvgR={avg_r:6.3f}"
    )


def print_comparison(
    title,
    baseline,
    filtered,
):

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    print()
    print("BASELINE")
    print(
        performance_label(
            baseline
        )
    )

    print()
    print("FILTERED")
    print(
        performance_label(
            filtered
        )
    )

    print()
    print(
        f"Trades removed: "
        f"{len(baseline) - len(filtered)}"
    )


def setup_exclusion_test(
    trades,
    excluded_setup,
):

    filtered = [
        trade
        for trade in trades
        if trade.get(
            "Diagnostic_Setup",
            trade.get("setup", "")
        ) != excluded_setup
    ]

    print_comparison(
        f"EXCLUDE {excluded_setup}",
        trades,
        filtered,
    )

def setup_trend_exclusion_matrix(trades):

    print()
    print("=" * 110)
    print("SETUP × TREND EXCLUSION MATRIX")
    print("=" * 110)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(trade)

    baseline_r = sum(
        to_float(
            trade.get(
                "r_multiple"
            )
        ) or 0.0
        for trade in trades
    )

    print()
    print(
        f"BASELINE "
        f"Trades={len(trades):>3} "
        f"R={baseline_r:>7.2f}"
    )

    for (
        setup,
        trend
    ) in sorted(groups.keys()):

        excluded = groups[
            (
                setup,
                trend
            )
        ]

        filtered = [
            trade
            for trade in trades
            if trade not in excluded
        ]

        if not filtered:
            continue

        winners = [
            trade
            for trade in filtered
            if (
                to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) or 0.0
            ) > 0
        ]

        gross_profit = sum(
            max(
                (
                    to_float(
                        trade.get(
                            "profit"
                        )
                    ) or 0.0
                ),
                0.0
            )
            for trade in filtered
        )

        gross_loss = sum(
            abs(
                min(
                    (
                        to_float(
                            trade.get(
                                "profit"
                            )
                        ) or 0.0
                    ),
                    0.0
                )
            )
            for trade in filtered
        )

        total_r = sum(
            (
                to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) or 0.0
            )
            for trade in filtered
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        win_rate = (
            len(winners)
            / len(filtered)
            * 100
        )

        avg_r = (
            total_r
            / len(filtered)
        )

        delta_r = (
            total_r
            - baseline_r
        )

        print()
        print(
            f"EXCLUDE {setup} × {trend}"
        )

        print(
            f"  Removed={len(excluded):>3} "
            f"Remaining={len(filtered):>3} "
            f"Win={win_rate:>6.2f}% "
            f"PF={profit_factor:>6.2f} "
            f"R={total_r:>7.2f} "
            f"AvgR={avg_r:>7.3f} "
            f"DeltaR={delta_r:>7.2f}"
        )

def setup_trend_combination_counterfactual(trades):

    print()
    print("=" * 120)
    print("SETUP × TREND COMBINATION COUNTERFACTUAL")
    print("=" * 120)

    def group_key(trade):

        setup = str(
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        )

        trend = str(
            trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        )

        return (
            setup,
            trend
        )

    scenarios = [
        (
            "NO FILTER",
            set()
        ),
        (
            "EXCLUDE LC × BULLISH",
            {
                (
                    "LONG_CONTINUATION",
                    "BULLISH"
                )
            }
        ),
        (
            "EXCLUDE SR × BEARISH",
            {
                (
                    "SHORT_REVERSAL",
                    "BEARISH"
                )
            }
        ),
        (
            "EXCLUDE SR × NEUTRAL",
            {
                (
                    "SHORT_REVERSAL",
                    "NEUTRAL"
                )
            }
        ),
        (
            "EXCLUDE ALL SHORT_REVERSAL",
            {
                (
                    "SHORT_REVERSAL",
                    "BEARISH"
                ),
                (
                    "SHORT_REVERSAL",
                    "NEUTRAL"
                ),
                (
                    "SHORT_REVERSAL",
                    "BULLISH"
                )
            }
        ),
        (
            "EXCLUDE LC BULLISH + ALL SR",
            {
                (
                    "LONG_CONTINUATION",
                    "BULLISH"
                ),
                (
                    "SHORT_REVERSAL",
                    "BEARISH"
                ),
                (
                    "SHORT_REVERSAL",
                    "NEUTRAL"
                ),
                (
                    "SHORT_REVERSAL",
                    "BULLISH"
                )
            }
        ),
        (
            "KEEP LC NEUTRAL + LR BULLISH + SC BEARISH",
            None
        ),
    ]

    baseline_r = sum(
        to_float(
            trade.get(
                "r_multiple"
            )
        )
        for trade in trades
    )

    for name, exclusions in scenarios:

        if exclusions is None:

            keep_groups = {
                (
                    "LONG_CONTINUATION",
                    "NEUTRAL"
                ),
                (
                    "LONG_REVERSAL",
                    "BULLISH"
                ),
                (
                    "SHORT_CONTINUATION",
                    "BEARISH"
                )
            }

            filtered = [
                trade
                for trade in trades
                if group_key(trade)
                in keep_groups
            ]

        else:

            filtered = [
                trade
                for trade in trades
                if group_key(trade)
                not in exclusions
            ]

        if not filtered:
            continue

        wins = 0
        gross_profit = 0.0
        gross_loss = 0.0
        total_r = 0.0

        for trade in filtered:

            profit = to_float(
                trade.get(
                    "profit"
                )
            )

            r_multiple = to_float(
                trade.get(
                    "r_multiple"
                )
            )

            total_r += r_multiple

            if r_multiple > 0:
                wins += 1

            if profit > 0:
                gross_profit += profit
            elif profit < 0:
                gross_loss += abs(
                    profit
                )

        win_rate = (
            wins
            / len(filtered)
            * 100
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else 0.0
        )

        avg_r = (
            total_r
            / len(filtered)
        )

        delta_r = (
            total_r
            - baseline_r
        )

        print()
        print(
            name
        )

        print(
            f"Trades={len(filtered):>3} "
            f"Win={win_rate:>6.2f}% "
            f"PF={profit_factor:>6.2f} "
            f"R={total_r:>7.2f} "
            f"AvgR={avg_r:>7.3f} "
            f"DeltaR={delta_r:>7.2f}"
        )

def entry_quality_early_path_diagnostic(trades):
    """
    Research-only diagnostic.

    Research question:
        Can ENTRY-TIME information identify trades that are likely
        to experience an early failure / slow path toward +0.5R?

    IMPORTANT:
        - Uses only information available at entry.
        - Does NOT use early_adverse_r_bar_* as a feature.
        - Does NOT use bars_to_0_5r as a feature.
        - Does NOT modify production engine logic.
        - Does NOT create a trading filter.
    """

    print()
    print("=" * 150)
    print("ENTRY-QUALITY → EARLY-PATH DIAGNOSTIC")
    print("=" * 150)

    print()
    print(
        "Goal: identify entry-time characteristics associated with "
        "early failure / slow movement toward +0.5R."
    )

    # --------------------------------------------------------------
    # TARGET DEFINITION
    # --------------------------------------------------------------

    def to_number(value):
        try:
            if value is None:
                return None

            if isinstance(value, str):
                value = value.strip()

                if value == "":
                    return None

                if value.lower() in {
                    "none",
                    "nan",
                    "null",
                }:
                    return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    def normalize_bool(value):
        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    def get_r(trade):
        value = to_number(
            trade.get("r_multiple")
        )

        return value if value is not None else 0.0

    def get_target_label(trade):
        """
        EARLY_FAILURE:
            Either the engine already marked immediate_failure,
            OR the trade failed to reach +0.5R within 5 bars.

        This is an OUTCOME label only.
        It is never used as an entry-time feature.
        """

        immediate_failure = normalize_bool(
            trade.get(
                "immediate_failure",
                False
            )
        )

        bars_to_05 = to_number(
            trade.get(
                "bars_to_0_5r"
            )
        )

        if immediate_failure:
            return "EARLY_FAILURE"

        if bars_to_05 is None:
            return "NO_0_5R"

        if bars_to_05 <= 5:
            return "FAST_0_5R"

        return "SLOW_0_5R"

    # --------------------------------------------------------------
    # ENTRY-TIME FEATURE HELPERS
    # --------------------------------------------------------------

    def get_setup(trade):
        return str(
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        )

    def get_trend(trade):
        return str(
            trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        )

    def get_strength_bucket(trade):
        value = to_number(
            trade.get(
                "entry_trend_strength"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < 20:
            return "<20"

        if value < 40:
            return "20-40"

        if value < 60:
            return "40-60"

        return "60+"

    def get_slope_bucket(trade):
        value = to_number(
            trade.get(
                "entry_trend_slope"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < -0.50:
            return "< -0.50"

        if value < 0:
            return "-0.50 to 0"

        if value < 0.50:
            return "0 to 0.50"

        return ">= 0.50"

    def get_price_position_bucket(trade):
        value = to_number(
            trade.get(
                "entry_price_position_value"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < 0.25:
            return "<25%"

        if value < 0.50:
            return "25-50%"

        if value < 0.75:
            return "50-75%"

        return "75%+"

    def get_break_age_bucket(trade):
        value = to_number(
            trade.get(
                "break_age",
                trade.get(
                    "Break_Age"
                )
            )
        )

        if value is None:
            return "UNKNOWN"

        if value <= 1:
            return "0-1"

        if value <= 3:
            return "2-3"

        if value <= 5:
            return "4-5"

        return "6+"

    def get_liquidity_bucket(trade):
        value = trade.get(
            "liquidity_sweep",
            trade.get(
                "Liquidity_Sweep"
            )
        )

        return (
            "SWEEP"
            if normalize_bool(value)
            else "NO_SWEEP"
        )

    def get_feature_value(trade, feature_name):

        if feature_name == "SETUP":
            return get_setup(trade)

        if feature_name == "ENTRY_TREND":
            return get_trend(trade)

        if feature_name == "TREND_STRENGTH":
            return get_strength_bucket(trade)

        if feature_name == "TREND_SLOPE":
            return get_slope_bucket(trade)

        if feature_name == "PRICE_POSITION":
            return get_price_position_bucket(trade)

        if feature_name == "STRUCTURE_BIAS":
            return str(
                trade.get(
                    "structure_bias",
                    trade.get(
                        "Structure_Bias",
                        "UNKNOWN"
                    )
                )
            )

        if feature_name == "LIQUIDITY_SWEEP":
            return get_liquidity_bucket(trade)

        if feature_name == "BREAK_AGE":
            return get_break_age_bucket(trade)

        if feature_name == "FIB_ZONE":
            return str(
                trade.get(
                    "Fib_Zone",
                    "UNKNOWN"
                )
            )

        if feature_name == "FIB_PREMIUM_DISCOUNT":
            return str(
                trade.get(
                    "Fib_Premium_Discount",
                    "UNKNOWN"
                )
            )

        if feature_name == "FIB_EXTENSION_STATE":
            return str(
                trade.get(
                    "Fib_Extension_State",
                    "UNKNOWN"
                )
            )

        if feature_name == "CONFIDENCE_BUCKET":
            return str(
                trade.get(
                    "Diagnostic_Confidence_Bucket",
                    "UNKNOWN"
                )
            )

        return "UNKNOWN"

    # --------------------------------------------------------------
    # FEATURES TO TEST
    # --------------------------------------------------------------

    features = [
        "SETUP",
        "ENTRY_TREND",
        "TREND_STRENGTH",
        "TREND_SLOPE",
        "PRICE_POSITION",
        "STRUCTURE_BIAS",
        "LIQUIDITY_SWEEP",
        "BREAK_AGE",
        "FIB_ZONE",
        "FIB_PREMIUM_DISCOUNT",
        "FIB_EXTENSION_STATE",
        "CONFIDENCE_BUCKET",
    ]

    # --------------------------------------------------------------
    # BUILD OUTCOME LABELS
    # --------------------------------------------------------------

    labelled_trades = []

    for trade in trades:

        if not trade.get("entry_date"):
            continue

        label = get_target_label(
            trade
        )

        labelled_trades.append(
            (
                trade,
                label
            )
        )

    if not labelled_trades:
        print()
        print("No valid labelled trades found.")
        return

    # --------------------------------------------------------------
    # OVERALL TARGET DISTRIBUTION
    # --------------------------------------------------------------

    print()
    print("-" * 150)
    print("TARGET DISTRIBUTION")
    print("-" * 150)

    target_groups = defaultdict(list)

    for trade, label in labelled_trades:
        target_groups[label].append(
            trade
        )

    total = len(labelled_trades)

    for label in [
        "EARLY_FAILURE",
        "NO_0_5R",
        "FAST_0_5R",
        "SLOW_0_5R",
    ]:

        sample = target_groups.get(
            label,
            []
        )

        if not sample:
            continue

        total_r = sum(
            get_r(trade)
            for trade in sample
        )

        avg_r = (
            total_r / len(sample)
            if sample
            else 0.0
        )

        print(
            f"{label:<18}"
            f"Trades={len(sample):>4} "
            f"Share={len(sample) / total * 100:>6.2f}% "
            f"R={total_r:>8.2f} "
            f"AvgR={avg_r:>7.3f}"
        )

    # --------------------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------------------

    for feature_name in features:

        print()
        print("=" * 150)
        print(
            f"ENTRY FEATURE: {feature_name}"
        )
        print("=" * 150)

        groups = defaultdict(list)

        for trade, label in labelled_trades:

            value = get_feature_value(
                trade,
                feature_name
            )

            groups[value].append(
                (
                    trade,
                    label
                )
            )

        print()
        print(
            f"{'BUCKET':<28}"
            f"{'TRADES':>8}"
            f"{'EARLY%':>10}"
            f"{'FAST%':>10}"
            f"{'SLOW%':>10}"
            f"{'R':>10}"
            f"{'AVG R':>10}"
        )

        print("-" * 90)

        for bucket in sorted(
            groups.keys(),
            key=lambda x: str(x)
        ):

            rows = groups[bucket]

            if len(rows) < 3:
                continue

            early_count = sum(
                1
                for (
                    trade,
                    label
                ) in rows
                if label == "EARLY_FAILURE"
            )

            fast_count = sum(
                1
                for (
                    trade,
                    label
                ) in rows
                if label == "FAST_0_5R"
            )

            slow_count = sum(
                1
                for (
                    trade,
                    label
                ) in rows
                if label == "SLOW_0_5R"
            )

            total_r = sum(
                get_r(trade)
                for (
                    trade,
                    label
                ) in rows
            )

            avg_r = (
                total_r / len(rows)
            )

            early_pct = (
                early_count
                / len(rows)
                * 100
            )

            fast_pct = (
                fast_count
                / len(rows)
                * 100
            )

            slow_pct = (
                slow_count
                / len(rows)
                * 100
            )

            print(
                f"{str(bucket):<28}"
                f"{len(rows):>8}"
                f"{early_pct:>9.1f}%"
                f"{fast_pct:>9.1f}%"
                f"{slow_pct:>9.1f}%"
                f"{total_r:>10.2f}"
                f"{avg_r:>10.3f}"
            )

    # --------------------------------------------------------------
    # SETUP × ENTRY TREND
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print("SETUP × ENTRY TREND → EARLY PATH")
    print("=" * 150)

    groups = defaultdict(list)

    for trade, label in labelled_trades:

        key = (
            get_setup(trade),
            get_trend(trade)
        )

        groups[key].append(
            (
                trade,
                label
            )
        )

    print()
    print(
        f"{'SETUP × TREND':<42}"
        f"{'TRADES':>8}"
        f"{'EARLY%':>10}"
        f"{'FAST%':>10}"
        f"{'SLOW%':>10}"
        f"{'R':>10}"
        f"{'AVG R':>10}"
    )

    print("-" * 100)

    for key in sorted(
        groups.keys(),
        key=lambda x: (
            str(x[0]),
            str(x[1])
        )
    ):

        rows = groups[key]

        if len(rows) < 3:
            continue

        early_count = sum(
            1
            for trade, label in rows
            if label == "EARLY_FAILURE"
        )

        fast_count = sum(
            1
            for trade, label in rows
            if label == "FAST_0_5R"
        )

        slow_count = sum(
            1
            for trade, label in rows
            if label == "SLOW_0_5R"
        )

        total_r = sum(
            get_r(trade)
            for trade, label in rows
        )

        avg_r = (
            total_r / len(rows)
        )

        print(
            f"{(str(key[0]) + ' × ' + str(key[1])):<42}"
            f"{len(rows):>8}"
            f"{early_count / len(rows) * 100:>9.1f}%"
            f"{fast_count / len(rows) * 100:>9.1f}%"
            f"{slow_count / len(rows) * 100:>9.1f}%"
            f"{total_r:>10.2f}"
            f"{avg_r:>10.3f}"
        )

    # --------------------------------------------------------------
    # RESEARCH INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print("ENTRY-QUALITY DIAGNOSTIC STATUS")
    print("=" * 150)

    print()
    print(
        "This output is hypothesis-generation only."
    )

    print(
        "Do NOT use any bucket as a live filter yet."
    )

    print(
        "A candidate feature must subsequently survive "
        "year-based walk-forward validation."
    )

    print(
        "The target variables EARLY_FAILURE / FAST_0_5R / "
        "SLOW_0_5R are outcomes and must never be supplied "
        "to the model at entry."
    )

    print()
    print(
        "NEXT STEP: identify only the strongest entry-time "
        "features/buckets, then test them using walk-forward "
        "validation on unseen years."
    )

def management_05r_counterfactual_holdout(trades):
    """
    RESEARCH-ONLY executable management counterfactual.

    Compare:
        1. BASELINE
        2. BE_0.5R       -> move stop to entry after +0.5R
        3. LOCK_0.25R    -> move stop to +0.25R after +0.5R

    The management rule activates only AFTER the bar that first
    reaches +0.5R, avoiding intrabar look-ahead.

    Target remains fixed at +2.0R.
    Initial stop remains the original initial stop until +0.5R
    has been reached.

    No production engine logic is modified.
    """

    print()
    print("=" * 120)
    print("EXECUTABLE +0.5R MANAGEMENT COUNTERFACTUAL")
    print("=" * 120)

    print()
    print("Hypotheses:")
    print("  BASELINE     = original realized trade outcome")
    print("  BE_0.5R      = after +0.5R, move stop to 0R")
    print("  LOCK_0.25R   = after +0.5R, move stop to +0.25R")
    print()
    print("Execution assumptions:")
    print("  Initial stop = original initial stop")
    print("  Target       = +2.0R")
    print("  Slippage     = 0.10%")
    print("  Brokerage    = Rs 20")
    print("  Management activates on the NEXT bar after +0.5R")
    print()

    split_date = datetime(
        2025,
        1,
        1
    )

    TARGET_R = 2.0
    TRIGGER_R = 0.5
    LOCK_R = 0.25

    SLIPPAGE_RATE = 0.001
    EXIT_BROKERAGE = 20.0

    cache = {}

    def get_history(symbol, entry_date):

        key = (
            str(symbol),
            str(entry_date),
        )

        if key in cache:
            return cache[key]

        try:

            entry_dt = pd.to_datetime(
                entry_date
            )

            start_date = (
                entry_dt
                - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            end_date = (
                entry_dt
                + timedelta(days=120)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                cache[key] = None
                return None

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                cache[key] = None
                return None

            data = data[
                required_columns
            ].copy()

            data.index = pd.to_datetime(
                data.index
            )

            data = data.sort_index()

            cache[key] = data

            return data

        except Exception:

            cache[key] = None
            return None

    def simulate_management(
        trade,
        management_mode
    ):

        try:

            symbol = trade.get(
                "_symbol"
            )

            entry_date = trade.get(
                "entry_date"
            )

            direction = str(
                trade.get(
                    "direction",
                    ""
                )
            ).upper()

            entry_price = to_float(
                trade.get(
                    "entry_price"
                ),
                None,
            )

            initial_risk = to_float(
                trade.get(
                    "initial_risk"
                ),
                None,
            )

            shares = to_float(
                trade.get(
                    "shares"
                ),
                None,
            )

            if (
                not symbol
                or not entry_date
                or direction not in (
                    "LONG",
                    "SHORT",
                )
                or entry_price is None
                or initial_risk is None
                or initial_risk <= 0
                or shares is None
                or shares <= 0
            ):
                return None

            data = get_history(
                symbol,
                entry_date
            )

            if data is None:
                return None

            entry_dt = pd.to_datetime(
                entry_date
            ).normalize()

            matching = data[
                data.index.normalize()
                == entry_dt
            ]

            if matching.empty:
                return None

            entry_position = data.index.get_loc(
                matching.index[0]
            )

            risk_per_share = (
                initial_risk
                / shares
            )

            if risk_per_share <= 0:
                return None

            trigger_move = (
                risk_per_share
                * TRIGGER_R
            )

            target_move = (
                risk_per_share
                * TARGET_R
            )

            lock_move = (
                risk_per_share
                * LOCK_R
            )

            trigger_price = (
                entry_price
                + trigger_move
                if direction == "LONG"
                else
                entry_price
                - trigger_move
            )

            target_price = (
                entry_price
                + target_move
                if direction == "LONG"
                else
                entry_price
                - target_move
            )

            breakeven_price = entry_price

            lock_price = (
                entry_price
                + lock_move
                if direction == "LONG"
                else
                entry_price
                - lock_move
            )

            initial_stop = to_float(
                trade.get(
                    "initial_stop"
                ),
                None,
            )

            if initial_stop is None:
                return None

            management_active = False

            for position in range(
                entry_position,
                len(data)
            ):

                bar = data.iloc[
                    position
                ]

                high = to_float(
                    bar.get("High"),
                    None,
                )

                low = to_float(
                    bar.get("Low"),
                    None,
                )

                if (
                    high is None
                    or low is None
                ):
                    continue

                # --------------------------------------------------
                # BEFORE +0.5R:
                # original stop remains active.
                # --------------------------------------------------

                if not management_active:

                    if direction == "LONG":

                        if low <= initial_stop:
                            exit_price = (
                                initial_stop
                                * (
                                    1.0
                                    - SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    exit_price
                                    - entry_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "INITIAL_STOP",
                                "triggered":
                                    False,
                            }

                        if high >= target_price:

                            exit_price = (
                                target_price
                                * (
                                    1.0
                                    - SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    exit_price
                                    - entry_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "TARGET",
                                "triggered":
                                    False,
                            }

                        if high >= trigger_price:

                            management_active = True

                            # Management begins on the NEXT bar.
                            continue

                    else:

                        if high >= initial_stop:
                            exit_price = (
                                initial_stop
                                * (
                                    1.0
                                    + SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    entry_price
                                    - exit_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "INITIAL_STOP",
                                "triggered":
                                    False,
                            }

                        if low <= target_price:

                            exit_price = (
                                target_price
                                * (
                                    1.0
                                    + SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    entry_price
                                    - exit_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "TARGET",
                                "triggered":
                                    False,
                            }

                        if low <= trigger_price:

                            management_active = True

                            # Management begins on the NEXT bar.
                            continue

                # --------------------------------------------------
                # AFTER +0.5R:
                # apply selected management stop.
                # --------------------------------------------------

                if management_active:

                    if management_mode == "BE_0.5R":
                        active_stop = (
                            breakeven_price
                        )

                    elif management_mode == "LOCK_0.25R":
                        active_stop = (
                            lock_price
                        )

                    else:
                        return None

                    if direction == "LONG":

                        if low <= active_stop:

                            exit_price = (
                                active_stop
                                * (
                                    1.0
                                    - SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    exit_price
                                    - entry_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    management_mode,
                                "triggered":
                                    True,
                            }

                        if high >= target_price:

                            exit_price = (
                                target_price
                                * (
                                    1.0
                                    - SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    exit_price
                                    - entry_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "TARGET",
                                "triggered":
                                    True,
                            }

                    else:

                        if high >= active_stop:

                            exit_price = (
                                active_stop
                                * (
                                    1.0
                                    + SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    entry_price
                                    - exit_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    management_mode,
                                "triggered":
                                    True,
                            }

                        if low <= target_price:

                            exit_price = (
                                target_price
                                * (
                                    1.0
                                    + SLIPPAGE_RATE
                                )
                            )

                            profit = (
                                (
                                    entry_price
                                    - exit_price
                                )
                                * shares
                                - EXIT_BROKERAGE
                            )

                            return {
                                "valid": True,
                                "r": (
                                    profit
                                    / initial_risk
                                ),
                                "exit_reason":
                                    "TARGET",
                                "triggered":
                                    True,
                            }

            return None

        except Exception:
            return None

    def evaluate(
        trade_list,
        management_mode
    ):

        results = []

        for trade in trade_list:

            result = simulate_management(
                trade,
                management_mode
            )

            if result is not None:
                results.append(
                    result
                )

        if not results:
            return {
                "evaluated": 0,
                "r": 0.0,
                "wins": 0,
                "losses": 0,
                "triggered": 0,
            }

        total_r = sum(
            result["r"]
            for result in results
        )

        wins = sum(
            1
            for result in results
            if result["r"] > 0
        )

        losses = sum(
            1
            for result in results
            if result["r"] <= 0
        )

        triggered = sum(
            1
            for result in results
            if result["triggered"]
        )

        return {
            "evaluated": len(results),
            "r": total_r,
            "wins": wins,
            "losses": losses,
            "triggered": triggered,
        }

    def baseline_evaluate(
        trade_list
    ):

        valid = []

        for trade in trade_list:

            r_value = to_float(
                trade.get(
                    "r_multiple"
                ),
                None,
            )

            if r_value is not None:
                valid.append(
                    r_value
                )

        if not valid:
            return {
                "evaluated": 0,
                "r": 0.0,
                "wins": 0,
                "losses": 0,
            }

        return {
            "evaluated": len(valid),
            "r": sum(valid),
            "wins": sum(
                1
                for value in valid
                if value > 0
            ),
            "losses": sum(
                1
                for value in valid
                if value <= 0
            ),
        }

    training = [
        trade
        for trade in trades
        if pd.to_datetime(
            trade.get(
                "entry_date"
            )
        ) < split_date
    ]

    holdout = [
        trade
        for trade in trades
        if pd.to_datetime(
            trade.get(
                "entry_date"
            )
        ) >= split_date
    ]

    print()
    print("=" * 120)
    print("TRAINING PERIOD: BEFORE 2025-01-01")
    print("=" * 120)

    baseline_train = baseline_evaluate(
        training
    )

    be_train = evaluate(
        training,
        "BE_0.5R"
    )

    lock_train = evaluate(
        training,
        "LOCK_0.25R"
    )

    print(
        f"Baseline      Trades={baseline_train['evaluated']:3d} "
        f"R={baseline_train['r']:8.2f} "
        f"W={baseline_train['wins']:3d} "
        f"L={baseline_train['losses']:3d}"
    )

    print(
        f"BE_0.5R       Trades={be_train['evaluated']:3d} "
        f"R={be_train['r']:8.2f} "
        f"W={be_train['wins']:3d} "
        f"L={be_train['losses']:3d} "
        f"Triggered={be_train['triggered']:3d}"
    )

    print(
        f"LOCK_0.25R    Trades={lock_train['evaluated']:3d} "
        f"R={lock_train['r']:8.2f} "
        f"W={lock_train['wins']:3d} "
        f"L={lock_train['losses']:3d} "
        f"Triggered={lock_train['triggered']:3d}"
    )

    print()
    print("=" * 120)
    print("HOLDOUT PERIOD: 2025-01-01 ONWARD")
    print("=" * 120)

    baseline_holdout = baseline_evaluate(
        holdout
    )

    be_holdout = evaluate(
        holdout,
        "BE_0.5R"
    )

    lock_holdout = evaluate(
        holdout,
        "LOCK_0.25R"
    )

    print(
        f"Baseline      Trades={baseline_holdout['evaluated']:3d} "
        f"R={baseline_holdout['r']:8.2f} "
        f"W={baseline_holdout['wins']:3d} "
        f"L={baseline_holdout['losses']:3d}"
    )

    print(
        f"BE_0.5R       Trades={be_holdout['evaluated']:3d} "
        f"R={be_holdout['r']:8.2f} "
        f"W={be_holdout['wins']:3d} "
        f"L={be_holdout['losses']:3d} "
        f"Triggered={be_holdout['triggered']:3d}"
    )

    print(
        f"LOCK_0.25R    Trades={lock_holdout['evaluated']:3d} "
        f"R={lock_holdout['r']:8.2f} "
        f"W={lock_holdout['wins']:3d} "
        f"L={lock_holdout['losses']:3d} "
        f"Triggered={lock_holdout['triggered']:3d}"
    )

    be_delta = (
        be_holdout["r"]
        - baseline_holdout["r"]
    )

    lock_delta = (
        lock_holdout["r"]
        - baseline_holdout["r"]
    )

    print()
    print("=" * 120)
    print("HOLDOUT MANAGEMENT VERDICT")
    print("=" * 120)

    print(
        f"BE_0.5R DeltaR:       "
        f"{be_delta:8.2f}R"
    )

    print(
        f"LOCK_0.25R DeltaR:    "
        f"{lock_delta:8.2f}R"
    )

    if (
        be_delta > 0
        and lock_delta > 0
    ):

        print(
            "RESEARCH RESULT: BOTH MANAGEMENT RULES "
            "IMPROVED HOLDOUT R."
        )

    elif be_delta > 0:

        print(
            "RESEARCH RESULT: BE_0.5R IMPROVED HOLDOUT R; "
            "LOCK_0.25R DID NOT."
        )

    elif lock_delta > 0:

        print(
            "RESEARCH RESULT: LOCK_0.25R IMPROVED HOLDOUT R; "
            "BE_0.5R DID NOT."
        )

    else:

        print(
            "RESEARCH RESULT: NO MANAGEMENT RULE "
            "IMPROVED HOLDOUT R."
        )

    print()
    print(
        "STATUS: RESEARCH ONLY — DO NOT MODIFY ENGINE LOGIC."
    )

def frozen_timeout_5_holdout_test(trades):

    print()
    print("=" * 120)
    print("FROZEN EXECUTABLE TIMEOUT_5 HOLDOUT TEST")
    print("=" * 120)

    print()
    print("Hypothesis:")
    print("  Freeze executable TIMEOUT_5")
    print("  If +0.5R is not reached within first 5 bars,")
    print("  exit at the 5th bar close using realistic slippage")
    print("  and brokerage.")
    print()

    print("IMPORTANT: TIMEOUT_5 is frozen from prior walk-forward research.")
    print("No parameter selection is performed here.")

    split_date = datetime(
        2025,
        1,
        1
    )

    TIMEOUT_BARS = 5
    SLIPPAGE_RATE = 0.001
    EXIT_BROKERAGE = 20.0

    cache = {}

    def get_history(symbol, entry_date):

        key = (
            str(symbol),
            str(entry_date),
        )

        if key in cache:
            return cache[key]

        try:

            entry_dt = pd.to_datetime(
                entry_date
            )

            start_date = (
                entry_dt
                - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            end_date = (
                entry_dt
                + timedelta(days=120)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                cache[key] = None
                return None

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                cache[key] = None
                return None

            data = data[
                required_columns
            ].copy()

            data.index = pd.to_datetime(
                data.index
            )

            data = data.sort_index()

            cache[key] = data

            return data

        except Exception:

            cache[key] = None
            return None

    def simulate_trade(trade):

        try:

            symbol = trade.get("_symbol")

            entry_date = trade.get("entry_date")

            direction = str(
                trade.get(
                    "direction",
                    ""
                )
            ).upper()

            entry_price = to_float(
                trade.get("entry_price"),
                None,
            )

            initial_risk = to_float(
                trade.get("initial_risk"),
                None,
            )

            shares = to_float(
                trade.get("shares"),
                None,
            )

            baseline_r = to_float(
                trade.get("r_multiple"),
                None,
            )

            baseline_profit = to_float(
                trade.get(
                    "profit",
                    trade.get("pnl")
                ),
                None,
            )

            if (
                not symbol
                or not entry_date
                or direction not in (
                    "LONG",
                    "SHORT",
                )
                or entry_price is None
                or initial_risk is None
                or initial_risk <= 0
                or shares is None
                or shares <= 0
                or baseline_r is None
                or baseline_profit is None
            ):
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            data = get_history(
                symbol,
                entry_date,
            )

            if data is None or len(data) == 0:
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            entry_dt = pd.to_datetime(
                entry_date
            )

            matching = data[
                data.index.normalize()
                == entry_dt.normalize()
            ]

            if matching.empty:
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            entry_position = data.index.get_loc(
                matching.index[0]
            )

            timeout_position = (
                entry_position
                + TIMEOUT_BARS
                - 1
            )

            if timeout_position >= len(data):
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            entry_bar = data.iloc[
                entry_position
            ]

            entry_open = to_float(
                entry_bar.get("Open"),
                None,
            )

            if entry_open is None:
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            expected_entry = (
                entry_open
                * (
                    1.0 + SLIPPAGE_RATE
                )
                if direction == "LONG"
                else
                entry_open
                * (
                    1.0 - SLIPPAGE_RATE
                )
            )

            tolerance = max(
                0.05,
                abs(entry_price) * 0.0005,
            )

            if abs(
                entry_price - expected_entry
            ) > tolerance:
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            # Convert total trade risk into risk per share.
            risk_per_share = (
                initial_risk
                / shares
            )

            target_move = (
                risk_per_share
                * 0.5
            )

            target_price = (
                entry_price
                + target_move
                if direction == "LONG"
                else
                entry_price
                - target_move
            )

            for position in range(
                entry_position,
                timeout_position + 1
            ):

                row = data.iloc[position]

                high = to_float(
                    row.get("High"),
                    None,
                )

                low = to_float(
                    row.get("Low"),
                    None,
                )

                if (
                    high is None
                    or low is None
                ):
                    continue

                if direction == "LONG":

                    if high >= target_price:

                        return {
                            "valid": True,
                            "r": baseline_r,
                            "profit": baseline_profit,
                            "timed_out": False,
                        }

                else:

                    if low <= target_price:

                        return {
                            "valid": True,
                            "r": baseline_r,
                            "profit": baseline_profit,
                            "timed_out": False,
                        }

            timeout_close = to_float(
                data.iloc[
                    timeout_position
                ].get("Close"),
                None,
            )

            if timeout_close is None:
                return {
                    "valid": False,
                    "r": 0.0,
                    "profit": 0.0,
                    "timed_out": False,
                }

            if direction == "LONG":

                timeout_exit_price = (
                    timeout_close
                    * (
                        1.0
                        - SLIPPAGE_RATE
                    )
                )

                timeout_profit = (
                    (
                        timeout_exit_price
                        - entry_price
                    )
                    * shares
                    - EXIT_BROKERAGE
                )

            else:

                timeout_exit_price = (
                    timeout_close
                    * (
                        1.0
                        + SLIPPAGE_RATE
                    )
                )

                timeout_profit = (
                    (
                        entry_price
                        - timeout_exit_price
                    )
                    * shares
                    - EXIT_BROKERAGE
                )

            timeout_r = (
                timeout_profit
                / initial_risk
            )

            return {
                "valid": True,
                "r": timeout_r,
                "profit": timeout_profit,
                "timed_out": True,
            }

        except Exception:

            return {
                "valid": False,
                "r": 0.0,
                "profit": 0.0,
                "timed_out": False,
            }

    def evaluate(sample):

        baseline_r = sum(
            (
                to_float(
                    trade.get(
                        "r_multiple"
                    ),
                    0.0,
                )
                or 0.0
            )
            for trade in sample
        )

        baseline_profit = sum(
            (
                to_float(
                    trade.get(
                        "profit",
                        trade.get("pnl")
                    ),
                    0.0,
                )
                or 0.0
            )
            for trade in sample
        )

        timeout_results = []

        for trade in sample:

            result = simulate_trade(
                trade
            )

            if result["valid"]:

                timeout_results.append(
                    result
                )

        timeout_r = sum(
            result["r"]
            for result in timeout_results
        )

        timeout_profit = sum(
            result["profit"]
            for result in timeout_results
        )

        timed_out = sum(
            1
            for result in timeout_results
            if result["timed_out"]
        )

        return {
            "baseline_r": baseline_r,
            "baseline_profit": baseline_profit,
            "timeout_r": timeout_r,
            "timeout_profit": timeout_profit,
            "timed_out": timed_out,
            "valid": len(timeout_results),
        }

    dated_trades = [
        trade
        for trade in trades
        if trade.get("entry_date") is not None
    ]

    training = [
        trade
        for trade in dated_trades
        if pd.to_datetime(
            trade.get("entry_date")
        ) < split_date
    ]

    holdout = [
        trade
        for trade in dated_trades
        if pd.to_datetime(
            trade.get("entry_date")
        ) >= split_date
    ]

    training_result = evaluate(
        training
    )

    holdout_result = evaluate(
        holdout
    )

    print()
    print("=" * 120)
    print("TRAINING PERIOD: BEFORE 2025-01-01")
    print("=" * 120)

    print(
        f"Baseline Trades={len(training):>3} "
        f"R={training_result['baseline_r']:>8.2f} "
        f"P&L={training_result['baseline_profit']:>10.2f}"
    )

    print(
        f"TIMEOUT_5 Executable Trades={training_result['valid']:>3} "
        f"R={training_result['timeout_r']:>8.2f} "
        f"P&L={training_result['timeout_profit']:>10.2f} "
        f"TimedOut={training_result['timed_out']:>3} "
        f"DeltaR={training_result['timeout_r'] - training_result['baseline_r']:>8.2f}"
    )

    print()
    print("=" * 120)
    print("HOLDOUT PERIOD: 2025-01-01 ONWARD")
    print("=" * 120)

    print(
        f"Baseline Trades={len(holdout):>3} "
        f"R={holdout_result['baseline_r']:>8.2f} "
        f"P&L={holdout_result['baseline_profit']:>10.2f}"
    )

    print(
        f"TIMEOUT_5 Executable Trades={holdout_result['valid']:>3} "
        f"R={holdout_result['timeout_r']:>8.2f} "
        f"P&L={holdout_result['timeout_profit']:>10.2f} "
        f"TimedOut={holdout_result['timed_out']:>3} "
        f"DeltaR={holdout_result['timeout_r'] - holdout_result['baseline_r']:>8.2f}"
    )

    print()
    print("=" * 120)
    print("FROZEN TIMEOUT_5 HOLDOUT VERDICT")
    print("=" * 120)

    if (
        holdout_result["timeout_r"]
        > holdout_result["baseline_r"]
    ):

        print(
            "PROMISING — frozen executable TIMEOUT_5 improved holdout R."
        )

        print(
            "DO NOT MODIFY ENGINE LOGIC YET; continue validation."
        )

    else:

        print(
            "FAILED — frozen executable TIMEOUT_5 did not improve holdout R."
        )

        print(
            "DO NOT MODIFY ENGINE LOGIC."
        )

def frozen_hypothesis_holdout_test(trades):

    print()
    print("=" * 120)
    print("FROZEN HYPOTHESIS HOLDOUT TEST")
    print("=" * 120)

    print()
    print(
        "Hypothesis:"
    )
    print(
        "  Exclude LONG_CONTINUATION × BULLISH"
    )
    print(
        "  Exclude SHORT_REVERSAL × BEARISH"
    )
    print(
        "  Exclude SHORT_REVERSAL × NEUTRAL"
    )

    print()
    print(
        "IMPORTANT: This is a frozen-hypothesis holdout sanity check."
    )
    print(
        "The hypothesis was discovered using the full historical dataset."
    )

    split_date = datetime(
        2025,
        1,
        1
    )

    def trade_date(trade):

        value = trade.get(
            "entry_date"
        )

        if value is None:
            return None

        try:
            return datetime.fromisoformat(
                str(value)
            )
        except ValueError:
            return None

    def is_excluded(trade):

        setup = str(
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        )

        trend = str(
            trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        )

        return (
            (
                setup == "LONG_CONTINUATION"
                and trend == "BULLISH"
            )
            or
            (
                setup == "SHORT_REVERSAL"
                and trend in {
                    "BEARISH",
                    "NEUTRAL"
                }
            )
        )

    def print_result(
        label,
        sample
    ):

        if not sample:

            print(
                f"{label:<25} "
                "Trades=0"
            )

            return

        total_r = sum(
            to_float(
                trade.get(
                    "r_multiple"
                )
            ) or 0.0
            for trade in sample
        )

        wins = sum(
            1
            for trade in sample
            if (
                (
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    )
                    or 0.0
                )
                > 0
            )
        )

        gross_profit = sum(
            max(
                (
                    to_float(
                        trade.get(
                            "profit"
                        )
                    )
                    or 0.0
                ),
                0.0
            )
            for trade in sample
        )

        gross_loss = sum(
            abs(
                min(
                    (
                        to_float(
                            trade.get(
                                "profit"
                            )
                        )
                        or 0.0
                    ),
                    0.0
                )
            )
            for trade in sample
        )

        win_rate = (
            wins
            / len(sample)
            * 100
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else 0.0
        )

        avg_r = (
            total_r
            / len(sample)
        )

        print(
            f"{label:<25} "
            f"Trades={len(sample):>3} "
            f"Win={win_rate:>6.2f}% "
            f"PF={profit_factor:>6.2f} "
            f"R={total_r:>7.2f} "
            f"AvgR={avg_r:>7.3f}"
        )

    dated_trades = [
        trade
        for trade in trades
        if trade_date(trade) is not None
    ]

    training = [
        trade
        for trade in dated_trades
        if trade_date(trade) < split_date
    ]

    holdout = [
        trade
        for trade in dated_trades
        if trade_date(trade) >= split_date
    ]

    training_filtered = [
        trade
        for trade in training
        if not is_excluded(trade)
    ]

    holdout_filtered = [
        trade
        for trade in holdout
        if not is_excluded(trade)
    ]

    print()
    print("=" * 120)
    print("TRAINING PERIOD: BEFORE 2025-01-01")
    print("=" * 120)

    print_result(
        "BASELINE",
        training
    )

    print_result(
        "FROZEN HYPOTHESIS",
        training_filtered
    )

    print()
    print("=" * 120)
    print("HOLDOUT PERIOD: 2025-01-01 ONWARD")
    print("=" * 120)

    print_result(
        "BASELINE",
        holdout
    )

    print_result(
        "FROZEN HYPOTHESIS",
        holdout_filtered
    )

    print()
    print("=" * 120)
    print("HOLDOUT BY STOCK")
    print("=" * 120)

    symbols = sorted(
        {
            str(
                trade.get(
                    "_symbol",
                    "UNKNOWN"
                )
            )
            for trade in holdout
        }
    )

    for symbol in symbols:

        stock_baseline = [
            trade
            for trade in holdout
            if str(
                trade.get(
                    "_symbol",
                    "UNKNOWN"
                )
            ) == symbol
        ]

        stock_filtered = [
            trade
            for trade in stock_baseline
            if not is_excluded(trade)
        ]

        print()
        print(
            symbol
        )

        print_result(
            "  BASELINE",
            stock_baseline
        )

        print_result(
            "  FROZEN HYPOTHESIS",
            stock_filtered
        )

def setup_trend_mfe_diagnostic(trades):

    print()
    print("=" * 110)
    print("SETUP × TREND × MFE OUTCOME")
    print("=" * 110)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        key = (
            setup,
            trend
        )

        groups[key].append(
            trade
        )

    for (
        setup,
        trend
    ), group_trades in sorted(
        groups.items()
    ):

        losers = [
            trade
            for trade in group_trades
            if to_float(
                trade.get("r_multiple")
            ) <= 0
        ]

        if not losers:
            continue

        losers_05r = [
            trade
            for trade in losers
            if to_float(
                trade.get("max_favorable_r")
            ) >= 0.5
        ]

        losers_1r = [
            trade
            for trade in losers
            if to_float(
                trade.get("max_favorable_r")
            ) >= 1.0
        ]

        losers_15r = [
            trade
            for trade in losers
            if to_float(
                trade.get("max_favorable_r")
            ) >= 1.5
        ]

        def avg(values):

            if not values:
                return 0.0

            return (
                sum(values)
                / len(values)
            )

        mfe_05r = [
            to_float(
                trade.get("max_favorable_r")
            )
            for trade in losers_05r
        ]

        realized_05r = [
            to_float(
                trade.get("r_multiple")
            )
            for trade in losers_05r
        ]

        giveback_05r = [
            to_float(
                trade.get("max_favorable_r")
            )
            - to_float(
                trade.get("r_multiple")
            )
            for trade in losers_05r
        ]

        bars_05r = [
            to_float(
                trade.get("bars_to_0_5r")
            )
            for trade in losers_05r
            if trade.get("bars_to_0_5r")
            not in (
                None,
                "",
            )
        ]

        bars_1r = [
            to_float(
                trade.get("bars_to_1r")
            )
            for trade in losers_1r
            if trade.get("bars_to_1r")
            not in (
                None,
                "",
            )
        ]

        print(
            f"{setup:<25}"
            f"{trend:<12}"
            f"Trades={len(group_trades):>3} "
            f"Losers={len(losers):>3} "
            f">=0.5R={len(losers_05r):>3} "
            f">=1R={len(losers_1r):>3} "
            f">=1.5R={len(losers_15r):>3} "
            f"AvgMFE05={avg(mfe_05r):5.2f}R "
            f"AvgR05={avg(realized_05r):6.2f}R "
            f"Giveback05={avg(giveback_05r):5.2f}R "
            f"Bars05={avg(bars_05r):5.1f} "
            f"Bars1={avg(bars_1r):5.1f}"
        )

def setup_mfe_bucket_diagnostic(trades):

    print()
    print("=" * 110)
    print("SETUP × MFE BUCKET OUTCOME")
    print("=" * 110)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        mfe = to_float(
            trade.get("max_favorable_r")
        )

        if mfe < 0.5:
            mfe_bucket = "<0.5R"

        elif mfe < 1.0:
            mfe_bucket = "0.5R-1R"

        elif mfe < 1.5:
            mfe_bucket = "1R-1.5R"

        else:
            mfe_bucket = ">=1.5R"

        groups[
            (
                setup,
                mfe_bucket
            )
        ].append(
            trade
        )

    bucket_order = {
        "<0.5R": 0,
        "0.5R-1R": 1,
        "1R-1.5R": 2,
        ">=1.5R": 3,
    }

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            bucket_order.get(
                item[0][1],
                99
            )
        )
    )

    for (
        setup,
        mfe_bucket
    ), group_trades in sorted_groups:

        r_values = [
            to_float(
                trade.get("r_multiple")
            )
            for trade in group_trades
        ]

        mfe_values = [
            to_float(
                trade.get("max_favorable_r")
            )
            for trade in group_trades
        ]

        giveback_values = [
            to_float(
                trade.get("max_favorable_r")
            )
            - to_float(
                trade.get("r_multiple")
            )
            for trade in group_trades
        ]

        wins = [
            r
            for r in r_values
            if r > 0
        ]

        losses = [
            r
            for r in r_values
            if r <= 0
        ]

        gross_profit = sum(wins)
        gross_loss = abs(
            sum(losses)
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = 0.0

        win_rate = (
            len(wins)
            / len(r_values)
            * 100
        )

        total_r = sum(r_values)

        avg_r = (
            total_r
            / len(r_values)
        )

        avg_mfe = (
            sum(mfe_values)
            / len(mfe_values)
        )

        avg_giveback = (
            sum(giveback_values)
            / len(giveback_values)
        )

        print(
            f"{setup:<25}"
            f"{mfe_bucket:<12}"
            f"Trades={len(r_values):>3} "
            f"Win={win_rate:6.2f}% "
            f"PF={profit_factor:5.2f} "
            f"R={total_r:7.2f} "
            f"AvgR={avg_r:6.3f} "
            f"AvgMFE={avg_mfe:5.2f}R "
            f"Giveback={avg_giveback:5.2f}R"
        )

def setup_trend_mfe_bucket_diagnostic(trades):

    print()
    print("=" * 110)
    print("SETUP × TREND × MFE BUCKET")
    print("=" * 110)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        mfe = to_float(
            trade.get("max_favorable_r")
        )

        if mfe < 0.5:
            mfe_bucket = "<0.5R"

        elif mfe < 1.0:
            mfe_bucket = "0.5R-1R"

        elif mfe < 1.5:
            mfe_bucket = "1R-1.5R"

        else:
            mfe_bucket = ">=1.5R"

        groups[
            (
                setup,
                trend,
                mfe_bucket
            )
        ].append(
            trade
        )

    bucket_order = {
        "<0.5R": 0,
        "0.5R-1R": 1,
        "1R-1.5R": 2,
        ">=1.5R": 3,
    }

    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            bucket_order.get(
                item[0][2],
                99
            )
        )
    )

    for (
        setup,
        trend,
        mfe_bucket
    ), group_trades in sorted_groups:

        print(
            f"{setup:<25}"
            f"{trend:<12}"
            f"{mfe_bucket:<12}"
            f"{performance_label(group_trades)}"
        )

def threshold_outcome_diagnostic(trades):

    print()
    print("=" * 110)
    print("MFE THRESHOLD OUTCOME ANALYSIS")
    print("=" * 110)

    thresholds = [
        ("0.5R", "reached_0_5r"),
        ("1.0R", "reached_1r"),
        ("1.5R", "reached_1_5r"),
    ]

    setups = sorted(
        {
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
            for trade in trades
        }
    )

    for setup in setups:

        setup_trades = [
            trade
            for trade in trades
            if trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            ) == setup
        ]

        print()
        print(
            f"SETUP: {setup}"
        )

        print("-" * 110)

        for threshold_name, field in thresholds:

            reached = [
                trade
                for trade in setup_trades
                if str(
                    trade.get(field, "")
                ).strip().lower()
                == "true"
            ]

            if not reached:
                print(
                    f"{threshold_name:<6}"
                    f"Trades=  0"
                )
                continue

            winners = [
                trade
                for trade in reached
                if to_float(
                    trade.get("r_multiple")
                ) > 0
            ]

            losers = [
                trade
                for trade in reached
                if to_float(
                    trade.get("r_multiple")
                ) <= 0
            ]

            target_exits = [
                trade
                for trade in reached
                if trade.get(
                    "exit_reason",
                    ""
                ) == "TARGET"
            ]

            stop_exits = [
                trade
                for trade in reached
                if trade.get(
                    "exit_reason",
                    ""
                ) == "STOP_LOSS"
            ]

            liquidation_exits = [
                trade
                for trade in reached
                if trade.get(
                    "exit_reason",
                    ""
                ) == "FINAL_LIQUIDATION"
            ]

            r_values = [
                to_float(
                    trade.get("r_multiple")
                )
                for trade in reached
            ]

            avg_r = (
                sum(r_values)
                / len(r_values)
            )

            win_rate = (
                len(winners)
                / len(reached)
                * 100
            )

            print(
                f"{threshold_name:<6}"
                f"Trades={len(reached):>3} "
                f"Win={win_rate:6.2f}% "
                f"AvgR={avg_r:6.3f} "
                f"Winners={len(winners):>3} "
                f"Losers={len(losers):>3} "
                f"Target={len(target_exits):>3} "
                f"Stop={len(stop_exits):>3} "
                f"Liquidation={len(liquidation_exits):>3}"
            )

def setup_context_diagnostic(trades):

    print()
    print("=" * 120)
    print("SETUP × CONFIDENCE × LIQUIDITY × BREAK AGE")
    print("=" * 120)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            "UNKNOWN"
        )

        confidence = trade.get(
            "Diagnostic_Confidence_Bucket",
            "UNKNOWN"
        )

        liquidity = trade.get(
            "Diagnostic_Liquidity",
            "UNKNOWN"
        )

        break_age = trade.get(
            "Diagnostic_Break_Age",
            "UNKNOWN"
        )

        groups[
            (
                setup,
                confidence,
                liquidity,
                break_age
            )
        ].append(
            trade
        )

    print()
    print(
        f"{'SETUP':<25}"
        f"{'CONF':<10}"
        f"{'LIQ':<10}"
        f"{'AGE':<10}"
        f"{'PERFORMANCE'}"
    )

    print("-" * 120)

    for (
        setup,
        confidence,
        liquidity,
        break_age
    ), group_trades in sorted(
        groups.items()
    ):

        print(
            f"{str(setup):<25}"
            f"{str(confidence):<10}"
            f"{str(liquidity):<10}"
            f"{str(break_age):<10}"
            f"{performance_label(group_trades)}"
        )

def setup_trend_liquidity_age_diagnostic(trades):

    print()
    print("=" * 120)
    print("SETUP × TREND × LIQUIDITY × BREAK AGE")
    print("=" * 120)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        liquidity = trade.get(
            "Diagnostic_Liquidity",
            "UNKNOWN"
        )

        break_age = trade.get(
            "Diagnostic_Break_Age",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend),
                str(liquidity),
                str(break_age)
            )
        ].append(
            trade
        )

    print()

    for (
        setup,
        trend,
        liquidity,
        break_age
    ), group_trades in sorted(
        groups.items()
    ):

        print(
            f"{setup:<25}"
            f"{trend:<12}"
            f"{liquidity:<12}"
            f"{break_age:<10}"
            f"{performance_label(group_trades)}"
        )

def entry_context_mfe_diagnostic(trades):

    print()
    print("=" * 110)
    print("ENTRY CONTEXT × MFE THRESHOLD")
    print("=" * 110)

    thresholds = [
        ("0.25R", 0.25),
        ("0.5R", 0.5),
        ("0.75R", 0.75),
        ("1.0R", 1.0),
        ("1.25R", 1.25),
        ("1.5R", 1.5),
    ]

    context_fields = [
        ("SETUP", "setup"),
        ("TREND", "entry_trend"),
        ("PRICE POSITION", "entry_price_position"),
    ]

    for context_label, context_field in context_fields:

        print()
        print("-" * 110)
        print(
            f"{context_label} × MFE THRESHOLD"
        )
        print("-" * 110)

        groups = defaultdict(list)

        for trade in trades:

            value = trade.get(
                "Diagnostic_Setup"
                if context_field == "setup"
                else context_field,
                "UNKNOWN"
            )

            groups[
                str(value)
            ].append(
                trade
            )

        for context_value in sorted(
            groups.keys()
        ):

            group_trades = groups[
                context_value
            ]

            print()
            print(
                f"{context_value}"
            )

            for threshold_label, threshold in thresholds:

                reached = [
                    trade
                    for trade in group_trades
                    if to_float(
                        trade.get(
                            "max_favorable_r"
                        )
                    ) >= threshold
                ]

                reached_count = len(
                    reached
                )

                total_count = len(
                    group_trades
                )

                reach_rate = (
                    reached_count
                    / total_count
                    * 100
                    if total_count > 0
                    else 0.0
                )

                avg_r = (
                    sum(
                        to_float(
                            trade.get(
                                "r_multiple"
                            )
                        )
                        for trade in reached
                    )
                    / reached_count
                    if reached_count > 0
                    else 0.0
                )

                print(
                    f"  {threshold_label:<6}"
                    f" Trades={total_count:>3} "
                    f"Reached={reached_count:>3} "
                    f"Rate={reach_rate:>6.2f}% "
                    f"AvgR={avg_r:>7.3f}"
                )


    print()
    print("-" * 110)
    print("SETUP × TREND × MFE THRESHOLD")
    print("-" * 110)

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(
            trade
        )

    for (
        setup,
        trend
    ) in sorted(
        groups.keys()
    ):

        group_trades = groups[
            (
                setup,
                trend
            )
        ]

        print()
        print(
            f"{setup} × {trend}"
        )

        for threshold_label, threshold in thresholds:

            reached = [
                trade
                for trade in group_trades
                if to_float(
                    trade.get(
                        "max_favorable_r"
                    )
                ) >= threshold
            ]

            total_count = len(
                group_trades
            )

            reached_count = len(
                reached
            )

            reach_rate = (
                reached_count
                / total_count
                * 100
                if total_count > 0
                else 0.0
            )

            avg_r = (
                sum(
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    )
                    for trade in reached
                )
                / reached_count
                if reached_count > 0
                else 0.0
            )

            print(
                f"  {threshold_label:<6}"
                f" Trades={total_count:>3} "
                f"Reached={reached_count:>3} "
                f"Rate={reach_rate:>6.2f}% "
                f"AvgR={avg_r:>7.3f}"
            )

def entry_context_failure_diagnostic(trades):

    print()
    print("=" * 120)
    print("INDIVIDUAL ENTRY VARIABLE × BAD PATH")
    print("=" * 120)

    print()
    print(
        "BAD PATH = MFE < 0.5R AND MAE >= 1R"
    )

    variables = {
        "TREND STRENGTH": "strength",
        "TREND SLOPE": "slope",
        "PRICE POSITION": "price_position",
        "ENTRY TREND": "trend",
        "SETUP": "setup",
    }

    def get_bucket(
        trade,
        variable
    ):

        if variable == "strength":

            strength = to_float(
                trade.get(
                    "entry_trend_strength"
                )
            )

            if strength < 20:
                return "STRENGTH_<20"

            elif strength < 40:
                return "STRENGTH_20-40"

            elif strength < 60:
                return "STRENGTH_40-60"

            else:
                return "STRENGTH_60+"

        if variable == "slope":

            slope = to_float(
                trade.get(
                    "entry_trend_slope"
                )
            )

            if slope < 0:
                return "SLOPE_NEG"

            elif slope == 0:
                return "SLOPE_ZERO"

            else:
                return "SLOPE_POS"

        if variable == "price_position":

            return str(
                trade.get(
                    "entry_price_position",
                    "UNKNOWN"
                )
            )

        if variable == "trend":

            return str(
                trade.get(
                    "entry_trend",
                    "UNKNOWN"
                )
            )

        if variable == "setup":

            return str(
                trade.get(
                    "Diagnostic_Setup",
                    trade.get(
                        "setup",
                        "UNKNOWN"
                    )
                )
            )

        return "UNKNOWN"

    for variable_name, variable_key in variables.items():

        print()
        print("-" * 120)
        print(variable_name)
        print("-" * 120)

        groups = defaultdict(list)

        for trade in trades:

            mfe = to_float(
                trade.get(
                    "max_favorable_r"
                )
            )

            mae = to_float(
                trade.get(
                    "max_adverse_r"
                )
            )

            bucket = get_bucket(
                trade,
                variable_key
            )

            groups[bucket].append(
                (
                    trade,
                    mfe,
                    mae
                )
            )

        for bucket in sorted(
            groups.keys()
        ):

            group = groups[
                bucket
            ]

            if not group:
                continue

            group_trades = [
                item[0]
                for item in group
            ]

            bad_path_trades = [
                item
                for item in group
                if (
                    item[1] < 0.5
                    and item[2] >= 1.0
                )
            ]

            bad_path_count = len(
                bad_path_trades
            )

            bad_path_rate = (
                bad_path_count
                / len(group)
                * 100
            )

            r_values = [
                to_float(
                    trade.get(
                        "r_multiple"
                    )
                )
                for trade in group_trades
            ]

            mfe_values = [
                item[1]
                for item in group
            ]

            mae_values = [
                item[2]
                for item in group
            ]

            gross_profit = sum(
                max(
                    r,
                    0.0
                )
                for r in r_values
            )

            gross_loss = sum(
                abs(
                    min(
                        r,
                        0.0
                    )
                )
                for r in r_values
            )

            profit_factor = (
                gross_profit
                / gross_loss
                if gross_loss > 0
                else 0.0
            )

            total_r = sum(
                r_values
            )

            avg_r = (
                total_r
                / len(r_values)
            )

            avg_mfe = (
                sum(mfe_values)
                / len(mfe_values)
            )

            avg_mae = (
                sum(mae_values)
                / len(mae_values)
            )

            print(
                f"{bucket:<20}"
                f"Trades={len(group):>3} "
                f"BadPath={bad_path_count:>3} "
                f"BadRate={bad_path_rate:>6.2f}% "
                f"AvgR={avg_r:>7.3f} "
                f"PF={profit_factor:>5.2f} "
                f"AvgMFE={avg_mfe:>6.2f}R "
                f"AvgMAE={avg_mae:>6.2f}R"
            )

def entry_variable_stability_diagnostic(trades):

    print()
    print("=" * 120)
    print("ENTRY VARIABLE STABILITY ACROSS TIME PERIODS")
    print("=" * 120)

    print()
    print("BAD PATH = MFE < 0.5R AND MAE >= 1R")
    print("MINIMUM SAMPLE SIZE = 5 TRADES")
    print()

    period_order = [
        "2021-2023",
        "2024",
        "2025",
        "2026",
    ]

    variable_definitions = [
        (
            "TREND STRENGTH",
            "entry_trend_strength",
            [
                ("STRENGTH_<20", lambda value: value < 20),
                ("STRENGTH_20-40", lambda value: 20 <= value < 40),
                ("STRENGTH_40-60", lambda value: 40 <= value < 60),
                ("STRENGTH_60+", lambda value: value >= 60),
            ],
        ),
        (
            "TREND SLOPE",
            "entry_trend_slope",
            [
                ("SLOPE_NEG", lambda value: value < 0),
                ("SLOPE_ZERO", lambda value: value == 0),
                ("SLOPE_POS", lambda value: value > 0),
            ],
        ),
        (
            "PRICE POSITION",
            "entry_price_position",
            [
                ("ABOVE_FAST", lambda value: value > 0),
                ("BELOW_FAST", lambda value: value < 0),
                ("NEUTRAL", lambda value: value == 0),
            ],
        ),
        (
            "ENTRY TREND",
            "entry_trend",
            [
                ("BULLISH", lambda value: str(value).upper() == "BULLISH"),
                ("BEARISH", lambda value: str(value).upper() == "BEARISH"),
                ("NEUTRAL", lambda value: str(value).upper() == "NEUTRAL"),
            ],
        ),
        (
            "SETUP",
            "Diagnostic_Setup",
            [
                (
                    "LONG_CONTINUATION",
                    lambda value: str(value).upper() == "LONG_CONTINUATION",
                ),
                (
                    "LONG_REVERSAL",
                    lambda value: str(value).upper() == "LONG_REVERSAL",
                ),
                (
                    "SHORT_CONTINUATION",
                    lambda value: str(value).upper() == "SHORT_CONTINUATION",
                ),
                (
                    "SHORT_REVERSAL",
                    lambda value: str(value).upper() == "SHORT_REVERSAL",
                ),
            ],
        ),
    ]

    period_groups = defaultdict(list)

    for trade in trades:

        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        ).strip()

        if len(entry_date) < 4:
            continue

        try:
            year = int(
                entry_date[:4]
            )
        except (TypeError, ValueError):
            continue

        if 2021 <= year <= 2023:
            period = "2021-2023"

        elif year == 2024:
            period = "2024"

        elif year == 2025:
            period = "2025"

        elif year == 2026:
            period = "2026"

        else:
            continue

        period_groups[period].append(
            trade
        )

    for period in period_order:

        period_trades = period_groups.get(
            period,
            []
        )

        print()
        print("-" * 120)
        print(
            f"PERIOD: {period} "
            f"(Trades={len(period_trades)})"
        )
        print("-" * 120)

        if not period_trades:
            print("NO TRADES")
            continue

        for (
            variable_name,
            field_name,
            buckets,
        ) in variable_definitions:

            print()
            print(
                f"{variable_name}"
            )
            print("-" * 120)

            bucket_groups = defaultdict(list)

            for trade in period_trades:

                raw_value = trade.get(
                    field_name
                )

                if raw_value in (
                    None,
                    "",
                    "None",
                ):
                    continue

                if field_name in (
                    "entry_trend_strength",
                    "entry_trend_slope",
                    "entry_price_position",
                ):

                    numeric_value = to_float(
                        raw_value
                    )

                    if numeric_value is None:
                        continue

                    for (
                        bucket_name,
                        condition,
                    ) in buckets:

                        if condition(
                            numeric_value
                        ):
                            bucket_groups[
                                bucket_name
                            ].append(
                                trade
                            )
                            break

                else:

                    for (
                        bucket_name,
                        condition,
                    ) in buckets:

                        if condition(
                            raw_value
                        ):
                            bucket_groups[
                                bucket_name
                            ].append(
                                trade
                            )
                            break

            for (
                bucket_name,
                _condition,
            ) in buckets:

                group_trades = bucket_groups.get(
                    bucket_name,
                    []
                )

                trade_count = len(
                    group_trades
                )

                if trade_count < 5:

                    print(
                        f"{bucket_name:<22} "
                        f"Trades={trade_count:>3} "
                        f"INSUFFICIENT_SAMPLE"
                    )

                    continue

                bad_path_count = 0

                total_r = 0.0
                total_mfe = 0.0
                total_mae = 0.0

                gross_profit = 0.0
                gross_loss = 0.0

                for trade in group_trades:

                    r_multiple = (
                        to_float(
                            trade.get(
                                "r_multiple"
                            )
                        )
                        or 0.0
                    )

                    mfe = (
                        to_float(
                            trade.get(
                                "max_favorable_r"
                            )
                        )
                        or 0.0
                    )

                    mae = (
                        to_float(
                            trade.get(
                                "max_adverse_r"
                            )
                        )
                        or 0.0
                    )

                    profit = (
                        to_float(
                            trade.get(
                                "profit"
                            )
                        )
                        or 0.0
                    )

                    total_r += r_multiple
                    total_mfe += mfe
                    total_mae += mae

                    if (
                        mfe < 0.5
                        and mae >= 1.0
                    ):
                        bad_path_count += 1

                    if profit > 0:
                        gross_profit += profit

                    elif profit < 0:
                        gross_loss += abs(
                            profit
                        )

                bad_path_rate = (
                    bad_path_count
                    / trade_count
                    * 100
                )

                avg_r = (
                    total_r
                    / trade_count
                )

                avg_mfe = (
                    total_mfe
                    / trade_count
                )

                avg_mae = (
                    total_mae
                    / trade_count
                )

                if gross_loss > 0:

                    profit_factor = (
                        gross_profit
                        / gross_loss
                    )

                    pf_text = (
                        f"{profit_factor:>5.2f}"
                    )

                else:

                    pf_text = "  INF"

                print(
                    f"{bucket_name:<22} "
                    f"Trades={trade_count:>3} "
                    f"BadPath={bad_path_count:>3} "
                    f"BadRate={bad_path_rate:>6.2f}% "
                    f"AvgR={avg_r:>7.3f} "
                    f"PF={pf_text} "
                    f"AvgMFE={avg_mfe:>5.2f}R "
                    f"AvgMAE={avg_mae:>5.2f}R"
                )

def setup_trend_mfe_timing_diagnostic(trades):

    print()
    print("=" * 120)
    print("SETUP × TREND × MFE THRESHOLD × TIMING / OUTCOME")
    print("=" * 120)

    thresholds = [
        ("0.5R", "bars_to_0_5r"),
        ("1.0R", "bars_to_1r"),
        ("1.5R", "bars_to_1_5r"),
    ]

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(
            trade
        )

    for (
        setup,
        trend
    ) in sorted(
        groups.keys()
    ):

        group_trades = groups[
            (
                setup,
                trend
            )
        ]

        print()
        print(
            f"{setup} × {trend}"
        )

        print("-" * 120)

        for (
            threshold_label,
            timing_field
        ) in thresholds:

            reached = [
                trade
                for trade in group_trades
                if trade.get(
                    timing_field
                ) not in (
                    None,
                    "",
                )
            ]

            if not reached:

                print(
                    f"{threshold_label:<6}"
                    f" Reached=  0"
                )

                continue

            winners = [
                trade
                for trade in reached
                if to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) > 0
            ]

            losers = [
                trade
                for trade in reached
                if to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) <= 0
            ]

            timing_values = [
                to_float(
                    trade.get(
                        timing_field
                    )
                )
                for trade in reached
            ]

            winner_timing = [
                to_float(
                    trade.get(
                        timing_field
                    )
                )
                for trade in winners
            ]

            loser_timing = [
                to_float(
                    trade.get(
                        timing_field
                    )
                )
                for trade in losers
            ]

            avg_bars = (
                sum(timing_values)
                / len(timing_values)
            )

            avg_winner_bars = (
                sum(winner_timing)
                / len(winner_timing)
                if winner_timing
                else 0.0
            )

            avg_loser_bars = (
                sum(loser_timing)
                / len(loser_timing)
                if loser_timing
                else 0.0
            )

            avg_winner_r = (
                sum(
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    )
                    for trade in winners
                )
                / len(winners)
                if winners
                else 0.0
            )

            avg_loser_r = (
                sum(
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    )
                    for trade in losers
                )
                / len(losers)
                if losers
                else 0.0
            )

            print(
                f"{threshold_label:<6}"
                f" Reached={len(reached):>3} "
                f"Winners={len(winners):>3} "
                f"Losers={len(losers):>3} "
                f"AvgBars={avg_bars:>6.1f} "
                f"WinBars={avg_winner_bars:>6.1f} "
                f"LoseBars={avg_loser_bars:>6.1f} "
                f"WinAvgR={avg_winner_r:>7.3f} "
                f"LoseAvgR={avg_loser_r:>7.3f}"
            )

def time_to_05r_timeout_counterfactual_walk_forward(trades):

    print()
    print("=" * 120)
    print("TIME-TO-0.5R EXECUTABLE TIMEOUT WALK-FORWARD")
    print("=" * 120)

    timeout_bars = [
        ("NO_FILTER", None),
        ("TIMEOUT_5", 5),
        ("TIMEOUT_10", 10),
        ("TIMEOUT_20", 20),
    ]

    MIN_TRAIN_TRADES = 10
    MIN_SELECTED_TRAIN_TRADES = 10

    SLIPPAGE_RATE = 0.001
    EXIT_BROKERAGE = 20.0

    cache = {}

    def get_history(symbol, entry_date):

        key = (
            str(symbol),
            str(entry_date),
        )

        if key in cache:
            return cache[key]

        try:
            entry_dt = pd.to_datetime(
                entry_date
            )

            start_date = (
                entry_dt
                - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            end_date = (
                entry_dt
                + timedelta(days=120)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                cache[key] = None
                return None

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                cache[key] = None
                return None

            data = data[
                required_columns
            ].copy()

            data.index = pd.to_datetime(
                data.index
            )

            data = data.sort_index()

            cache[key] = data

            return data

        except Exception:
            cache[key] = None
            return None

    def simulate_trade(
        trade,
        timeout,
    ):

        if timeout is None:
            return {
                "valid": True,
                "r_multiple": to_float(
                    trade.get("r_multiple")
                ),
                "profit": to_float(
                    trade.get("profit")
                ),
            }

        symbol = trade.get(
            "_symbol"
        )

        entry_date = trade.get(
            "entry_date"
        )

        direction = str(
            trade.get(
                "direction",
                ""
            )
        ).upper()

        entry_price = to_float(
            trade.get("entry_price"),
            None,
        )

        initial_risk = to_float(
            trade.get("initial_risk"),
            None,
        )

        shares = to_float(
            trade.get("shares"),
            None,
        )

        if (
            not symbol
            or not entry_date
            or direction
            not in (
                "LONG",
                "SHORT",
            )
            or entry_price is None
            or initial_risk is None
            or initial_risk <= 0
            or shares is None
            or shares <= 0
        ):
            return {
                "valid": False,
            }

        data = get_history(
            symbol,
            entry_date,
        )

        if data is None:
            return {
                "valid": False,
            }

        entry_dt = pd.to_datetime(
            entry_date
        ).normalize()

        matching = data[
            data.index.normalize()
            == entry_dt
        ]

        if matching.empty:
            return {
                "valid": False,
            }

        entry_position = data.index.get_loc(
            matching.index[0]
        )

        timeout_position = (
            entry_position
            + timeout
            - 1
        )

        if (
            timeout_position
            < entry_position
            or timeout_position
            >= len(data)
        ):
            return {
                "valid": False,
            }

        entry_bar = data.iloc[
            entry_position
        ]

        entry_open = float(
            entry_bar["Open"]
        )

        if direction == "LONG":

            expected_entry = (
                entry_open
                * (1 + SLIPPAGE_RATE)
            )

        else:

            expected_entry = (
                entry_open
                * (1 - SLIPPAGE_RATE)
            )

        tolerance = max(
            0.05,
            abs(entry_price)
            * 0.0005,
        )

        if abs(
            expected_entry
            - entry_price
        ) > tolerance:

            return {
                "valid": False,
            }

        risk_per_share = (
            initial_risk
            / shares
        )

        target_move = (
            risk_per_share
            * 0.5
        )

        target_price = (
            entry_price
            + target_move
            if direction == "LONG"
            else entry_price
            - target_move
        )

        for position in range(
            entry_position,
            timeout_position + 1,
        ):

            bar = data.iloc[
                position
            ]

            high = float(
                bar["High"]
            )

            low = float(
                bar["Low"]
            )

            if direction == "LONG":

                if high >= target_price:

                    return {
                        "valid": True,
                        "r_multiple": to_float(
                            trade.get(
                                "r_multiple"
                            )
                        ),
                        "profit": to_float(
                            trade.get(
                                "profit"
                            )
                        ),
                    }

            else:

                if low <= target_price:

                    return {
                        "valid": True,
                        "r_multiple": to_float(
                            trade.get(
                                "r_multiple"
                            )
                        ),
                        "profit": to_float(
                            trade.get(
                                "profit"
                            )
                        ),
                    }

        timeout_bar = data.iloc[
            timeout_position
        ]

        timeout_close = float(
            timeout_bar["Close"]
        )

        if direction == "LONG":

            timeout_exit_price = (
                timeout_close
                * (1 - SLIPPAGE_RATE)
            )

            timeout_profit = (
                (
                    timeout_exit_price
                    - entry_price
                )
                * shares
                - EXIT_BROKERAGE
            )

        else:

            timeout_exit_price = (
                timeout_close
                * (1 + SLIPPAGE_RATE)
            )

            timeout_profit = (
                (
                    entry_price
                    - timeout_exit_price
                )
                * shares
                - EXIT_BROKERAGE
            )

        timeout_r = (
            timeout_profit
            / initial_risk
        )

        return {
            "valid": True,
            "r_multiple": timeout_r,
            "profit": timeout_profit,
        }

    def evaluate(
        trade_list,
        timeout,
    ):

        results = []

        for trade in trade_list:

            result = simulate_trade(
                trade,
                timeout,
            )

            if result.get(
                "valid",
                False,
            ):
                results.append(
                    result
                )

        if not results:
            return {
                "trades": 0,
                "r": 0.0,
            }

        total_r = sum(
            result["r_multiple"]
            for result in results
        )

        return {
            "trades": len(results),
            "r": total_r,
        }

    years = sorted(
        {
            pd.to_datetime(
                trade.get(
                    "entry_date"
                )
            ).year
            for trade in trades
            if trade.get(
                "entry_date"
            )
        }
    )

    all_validation_results = []

    for year in years:

        train_trades = [
            trade
            for trade in trades
            if (
                trade.get(
                    "entry_date"
                )
                and pd.to_datetime(
                    trade.get(
                        "entry_date"
                    )
                ).year
                < year
            )
        ]

        validation_trades = [
            trade
            for trade in trades
            if (
                trade.get(
                    "entry_date"
                )
                and pd.to_datetime(
                    trade.get(
                        "entry_date"
                    )
                ).year
                == year
            )
        ]

        if (
            len(train_trades)
            < MIN_TRAIN_TRADES
            or not validation_trades
        ):
            continue

        print()
        print(
            f"YEAR {year}"
        )
        print("-" * 120)

        candidate_results = []

        for (
            label,
            timeout,
        ) in timeout_bars:

            train_result = evaluate(
                train_trades,
                timeout,
            )

            if (
                train_result["trades"]
                >= MIN_SELECTED_TRAIN_TRADES
            ):

                candidate_results.append(
                    (
                        train_result["r"],
                        label,
                        timeout,
                        train_result["trades"],
                    )
                )

        if not candidate_results:
            continue

        candidate_results.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        (
            train_r,
            selected_label,
            selected_timeout,
            selected_train_trades,
        ) = candidate_results[0]

        baseline = evaluate(
            validation_trades,
            None,
        )

        filtered = evaluate(
            validation_trades,
            selected_timeout,
        )

        delta_r = (
            filtered["r"]
            - baseline["r"]
        )

        print(
            f"Selected={selected_label:<12} "
            f"TrainTrades={selected_train_trades:>3} "
            f"TrainR={train_r:>8.2f} "
            f"ValidationTrades={baseline['trades']:>3} "
            f"BaselineR={baseline['r']:>8.2f} "
            f"FilteredR={filtered['r']:>8.2f} "
            f"DeltaR={delta_r:>8.2f}"
        )

        all_validation_results.append(
            {
                "year": year,
                "selected": selected_label,
                "train_r": train_r,
                "validation_trades": baseline[
                    "trades"
                ],
                "baseline_r": baseline[
                    "r"
                ],
                "filtered_r": filtered[
                    "r"
                ],
                "delta_r": delta_r,
            }
        )

    print()
    print("=" * 120)
    print("WALK-FORWARD SUMMARY")
    print("=" * 120)

    if not all_validation_results:

        print(
            "No eligible year-years."
        )
        return

    baseline_total = sum(
        result["baseline_r"]
        for result in all_validation_results
    )

    filtered_total = sum(
        result["filtered_r"]
        for result in all_validation_results
    )

    delta_total = (
        filtered_total
        - baseline_total
    )

    improved_years = sum(
        1
        for result
        in all_validation_results
        if result["delta_r"] > 0
    )

    print(
        f"Eligible years: "
        f"{len(all_validation_results)}"
    )

    print(
        f"Baseline validation total: "
        f"{baseline_total:.2f}R"
    )

    print(
        f"Filtered validation total: "
        f"{filtered_total:.2f}R"
    )

    print(
        f"Total DeltaR: "
        f"{delta_total:.2f}R"
    )

    print(
        f"Improved years: "
        f"{improved_years}/"
        f"{len(all_validation_results)}"
    )

    if (
        improved_years
        >= 2
        and delta_total > 0
    ):

        print(
            "RESEARCH RULE: "
            "PROMISING — executable "
            "time-to-0.5R timeout "
            "improved at least 2 "
            "validation years. "
            "DO NOT DEPLOY YET; "
            "continue research."
        )

    else:

        print(
            "RESEARCH RULE: "
            "NOT ROBUST ENOUGH — "
            "executable time-to-0.5R "
            "timeout does not yet "
            "provide sufficient "
            "out-of-sample evidence. "
            "DO NOT MODIFY ENGINE LOGIC."
        )

def setup_trend_timeout_5_walk_forward(trades):
    print()
    print("=" * 120)
    print("SETUP × TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD")
    print("=" * 120)

    TIMEOUT_BARS = 5
    MIN_TRAIN_TRADES = 5
    MIN_SELECTED_TRAIN_TRADES = 5

    SLIPPAGE_RATE = 0.001
    EXIT_BROKERAGE = 20.0

    cache = {}

    def get_history(symbol, entry_date):
        key = (
            str(symbol),
            str(entry_date),
        )

        if key in cache:
            return cache[key]

        try:
            entry_dt = pd.to_datetime(
                entry_date
            )

            start_date = (
                entry_dt
                - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            end_date = (
                entry_dt
                + timedelta(days=120)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                cache[key] = None
                return None

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                cache[key] = None
                return None

            data = data[
                required_columns
            ].copy()

            data.index = pd.to_datetime(
                data.index
            )

            data = data.sort_index()

            cache[key] = data

            return data

        except Exception:
            cache[key] = None
            return None

    def simulate_timeout_5(trade):
        try:
            symbol = trade.get(
                "_symbol"
            )

            entry_date = trade.get(
                "entry_date"
            )

            direction = str(
                trade.get(
                    "direction",
                    ""
                )
            ).upper()

            entry_price = to_float(
                trade.get(
                    "entry_price"
                ),
                None,
            )

            initial_risk = to_float(
                trade.get(
                    "initial_risk"
                ),
                None,
            )

            shares = to_float(
                trade.get(
                    "shares"
                ),
                None,
            )

            baseline_r = to_float(
                trade.get(
                    "r_multiple"
                ),
                None,
            )

            baseline_profit = to_float(
                trade.get(
                    "profit",
                    trade.get(
                        "pnl"
                    )
                ),
                None,
            )

            if (
                not symbol
                or not entry_date
                or direction
                not in (
                    "LONG",
                    "SHORT",
                )
                or entry_price is None
                or initial_risk is None
                or initial_risk <= 0
                or shares is None
                or shares <= 0
            ):
                return None

            data = get_history(
                symbol,
                entry_date,
            )

            if data is None:
                return None

            entry_dt = pd.to_datetime(
                entry_date
            ).normalize()

            matching = data[
                data.index.normalize()
                == entry_dt
            ]

            if matching.empty:
                return None

            entry_position = data.index.get_loc(
                matching.index[0]
            )

            timeout_position = (
                entry_position
                + TIMEOUT_BARS
                - 1
            )

            if (
                timeout_position
                < entry_position
                or timeout_position
                >= len(data)
            ):
                return None

            entry_bar = data.iloc[
                entry_position
            ]

            entry_open = float(
                entry_bar["Open"]
            )

            if direction == "LONG":
                expected_entry = (
                    entry_open
                    * (
                        1
                        + SLIPPAGE_RATE
                    )
                )
            else:
                expected_entry = (
                    entry_open
                    * (
                        1
                        - SLIPPAGE_RATE
                    )
                )

            tolerance = max(
                0.05,
                abs(entry_price)
                * 0.0005,
            )

            if abs(
                expected_entry
                - entry_price
            ) > tolerance:
                return None

            risk_per_share = (
                initial_risk
                / shares
            )

            if risk_per_share <= 0:
                return None

            target_move = (
                risk_per_share
                * 0.5
            )

            target_price = (
                entry_price
                + target_move
                if direction == "LONG"
                else entry_price
                - target_move
            )

            for position in range(
                entry_position,
                timeout_position + 1,
            ):
                bar = data.iloc[
                    position
                ]

                high = float(
                    bar["High"]
                )

                low = float(
                    bar["Low"]
                )

                if direction == "LONG":
                    if high >= target_price:
                        return {
                            "r": baseline_r,
                            "profit": baseline_profit,
                            "timed_out": False,
                        }

                else:
                    if low <= target_price:
                        return {
                            "r": baseline_r,
                            "profit": baseline_profit,
                            "timed_out": False,
                        }

            timeout_bar = data.iloc[
                timeout_position
            ]

            timeout_close = float(
                timeout_bar["Close"]
            )

            if direction == "LONG":
                timeout_exit_price = (
                    timeout_close
                    * (
                        1
                        - SLIPPAGE_RATE
                    )
                )

                timeout_profit = (
                    (
                        timeout_exit_price
                        - entry_price
                    )
                    * shares
                    - EXIT_BROKERAGE
                )

            else:
                timeout_exit_price = (
                    timeout_close
                    * (
                        1
                        + SLIPPAGE_RATE
                    )
                )

                timeout_profit = (
                    (
                        entry_price
                        - timeout_exit_price
                    )
                    * shares
                    - EXIT_BROKERAGE
                )

            timeout_r = (
                timeout_profit
                / initial_risk
            )

            return {
                "r": timeout_r,
                "profit": timeout_profit,
                "timed_out": True,
            }

        except Exception:
            return None

    def evaluate(
        trade_list
    ):
        total_r = 0.0
        total_profit = 0.0
        evaluated = 0

        for trade in trade_list:
            result = simulate_timeout_5(
                trade
            )

            if result is None:
                continue

            total_r += result["r"]
            total_profit += result["profit"]
            evaluated += 1

        return {
            "r": total_r,
            "profit": total_profit,
            "evaluated": evaluated,
        }

    def baseline_evaluate(
        trade_list
    ):
        total_r = 0.0
        total_profit = 0.0
        evaluated = 0

        for trade in trade_list:
            try:
                r_value = to_float(
                    trade.get(
                        "r_multiple"
                    )
                )

                profit_value = to_float(
                    trade.get(
                        "profit",
                        trade.get(
                            "pnl"
                        )
                    )
                )

                if (
                    r_value is None
                    or profit_value is None
                ):
                    continue

                total_r += r_value
                total_profit += profit_value
                evaluated += 1

            except Exception:
                continue

        return {
            "r": total_r,
            "profit": total_profit,
            "evaluated": evaluated,
        }

    grouped = defaultdict(list)

    for trade in trades:
        try:
            entry_date = pd.to_datetime(
                trade.get(
                    "entry_date"
                )
            )

            year = entry_date.year

            setup = trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )

            trend = trade.get(
                "entry_trend",
                "UNKNOWN"
            )

            setup = str(
                setup
            ).strip()

            trend = str(
                trend
            ).strip()

            if (
                not setup
                or setup == "UNKNOWN"
                or not trend
                or trend == "UNKNOWN"
            ):
                continue

            grouped[
                (
                    year,
                    setup,
                    trend,
                )
            ].append(
                trade
            )

        except Exception:
            continue

    years = sorted(
        {
            year
            for year, _, _
            in grouped.keys()
        }
    )

    total_baseline_r = 0.0
    total_timeout_r = 0.0
    total_delta_r = 0.0

    improved_groups = 0
    eligible_groups = 0

    for year in years:
        print()
        print(
            f"YEAR {year}"
        )
        print(
            "-" * 120
        )

        group_keys = sorted(
            [
                key
                for key in grouped.keys()
                if key[0] == year
            ],
            key=lambda x: (
                x[1],
                x[2],
            ),
        )

        for (
            _,
            setup,
            trend,
        ) in group_keys:

            validation_trades = grouped[
                (
                    year,
                    setup,
                    trend,
                )
            ]

            train_trades = [
                trade
                for (
                    trade_year,
                    trade_setup,
                    trade_trend,
                ), group
                in grouped.items()
                if (
                    trade_year < year
                    and trade_setup == setup
                    and trade_trend == trend
                )
                for trade in group
            ]

            if (
                len(train_trades)
                < MIN_TRAIN_TRADES
            ):
                continue

            train_result = evaluate(
                train_trades
            )

            if (
                train_result["evaluated"]
                < MIN_SELECTED_TRAIN_TRADES
            ):
                continue

            baseline = baseline_evaluate(
                validation_trades
            )

            timeout_result = evaluate(
                validation_trades
            )

            if (
                baseline["evaluated"] == 0
                or timeout_result["evaluated"] == 0
            ):
                continue

            delta_r = (
                timeout_result["r"]
                - baseline["r"]
            )

            eligible_groups += 1

            total_baseline_r += (
                baseline["r"]
            )

            total_timeout_r += (
                timeout_result["r"]
            )

            total_delta_r += delta_r

            if delta_r > 0:
                improved_groups += 1

            print(
                f"{setup} × "
                f"{trend:<10} "
                f"TrainTrades="
                f"{train_result['evaluated']:3d} "
                f"TrainR="
                f"{train_result['r']:8.2f} "
                f"ValTrades="
                f"{baseline['evaluated']:3d} "
                f"BaselineR="
                f"{baseline['r']:8.2f} "
                f"Timeout5R="
                f"{timeout_result['r']:8.2f} "
                f"DeltaR="
                f"{delta_r:8.2f}"
            )

    print()
    print(
        "=" * 120
    )
    print(
        "SETUP × TREND × EXECUTABLE "
        "TIMEOUT_5 WALK-FORWARD SUMMARY"
    )
    print(
        "=" * 120
    )

    print(
        f"Eligible groups: "
        f"{eligible_groups}"
    )

    print(
        f"Baseline validation total: "
        f"{total_baseline_r:.2f}R"
    )

    print(
        f"Timeout_5 validation total: "
        f"{total_timeout_r:.2f}R"
    )

    print(
        f"Total DeltaR: "
        f"{total_delta_r:.2f}R"
    )

    print(
        f"Improved groups: "
        f"{improved_groups}/"
        f"{eligible_groups}"
    )

    if (
        eligible_groups > 0
        and improved_groups
        >= max(
            2,
            eligible_groups // 2 + 1,
        )
        and total_delta_r > 0
    ):
        print(
            "RESEARCH RULE: "
            "PROMISING — executable "
            "TIMEOUT_5 improved a "
            "majority of eligible "
            "setup × trend validation "
            "groups. DO NOT DEPLOY YET; "
            "continue research."
        )
    else:
        print(
            "RESEARCH RULE: "
            "NOT ROBUST ENOUGH — executable "
            "TIMEOUT_5 does not provide "
            "sufficient setup × trend "
            "out-of-sample evidence. "
            "DO NOT MODIFY ENGINE LOGIC."
        )

def entry_trend_timeout_5_walk_forward(trades):

    print()
    print("=" * 120)
    print("ENTRY TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD")
    print("=" * 120)

    TIMEOUT_BARS = 5

    MIN_TRAIN_TRADES = 10
    MIN_VALIDATION_TRADES = 5

    SLIPPAGE_RATE = 0.001
    EXIT_BROKERAGE = 20.0

    cache = {}

    def get_history(symbol, entry_date):

        key = (
            str(symbol),
            str(entry_date),
        )

        if key in cache:
            return cache[key]

        try:

            entry_dt = pd.to_datetime(entry_date)

            start_date = (
                entry_dt
                - timedelta(days=10)
            ).strftime("%Y-%m-%d")

            end_date = (
                entry_dt
                + timedelta(days=120)
            ).strftime("%Y-%m-%d")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=15,
            )

            if data is None or data.empty:
                cache[key] = None
                return None

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):
                data.columns = [
                    column[0]
                    for column in data.columns
                ]

            required_columns = [
                "Open",
                "High",
                "Low",
                "Close",
            ]

            if not all(
                column in data.columns
                for column in required_columns
            ):
                cache[key] = None
                return None

            data = data[
                required_columns
            ].copy()

            data.index = pd.to_datetime(
                data.index
            )

            data = data.sort_index()

            cache[key] = data

            return data

        except Exception:

            cache[key] = None

            return None

    def simulate_timeout_5(trade):

        symbol = trade.get("_symbol")

        entry_date = trade.get(
            "entry_date"
        )

        direction = str(
            trade.get(
                "direction",
                ""
            )
        ).upper()

        entry_price = to_float(
            trade.get("entry_price"),
            None,
        )

        initial_risk = to_float(
            trade.get("initial_risk"),
            None,
        )

        shares = to_float(
            trade.get("shares"),
            None,
        )

        if (
            not symbol
            or not entry_date
            or direction not in (
                "LONG",
                "SHORT",
            )
            or entry_price is None
            or initial_risk is None
            or shares is None
            or initial_risk <= 0
            or shares <= 0
        ):
            return {
                "valid": False,
            }

        data = get_history(
            symbol,
            entry_date,
        )

        if data is None or data.empty:
            return {
                "valid": False,
            }

        entry_dt = pd.to_datetime(
            entry_date
        )

        matching = data[
            data.index >= entry_dt
        ]

        if matching.empty:
            return {
                "valid": False,
            }

        entry_index = data.index.get_loc(
            matching.index[0]
        )

        if entry_index >= len(data):
            return {
                "valid": False,
            }

        tolerance = max(
            0.05,
            abs(entry_price) * 0.0005,
        )

        selected_entry_index = None

        for idx in range(
            entry_index,
            min(
                entry_index + 3,
                len(data),
            ),
        ):

            open_price = float(
                data.iloc[idx]["Open"]
            )

            if abs(
                open_price - entry_price
            ) <= tolerance:

                selected_entry_index = idx

                break

        if selected_entry_index is None:

            selected_entry_index = entry_index

        start_idx = (
            selected_entry_index
            + 1
        )

        if start_idx >= len(data):

            return {
                "valid": False,
            }

        target_risk = initial_risk * 0.5

        if direction == "LONG":

            target_price = (
                entry_price
                + target_risk
            )

        else:

            target_price = (
                entry_price
                - target_risk
            )

        max_bars = min(
            TIMEOUT_BARS,
            len(data) - start_idx,
        )

        if max_bars <= 0:

            return {
                "valid": False,
            }

        for offset in range(
            max_bars
        ):

            bar = data.iloc[
                start_idx + offset
            ]

            high_price = float(
                bar["High"]
            )

            low_price = float(
                bar["Low"]
            )

            if direction == "LONG":

                if high_price >= target_price:

                    return {
                        "valid": True,
                        "r_multiple": to_float(
                            trade.get(
                                "r_multiple"
                            )
                        ),
                        "profit": to_float(
                            trade.get(
                                "profit"
                            )
                        ),
                    }

            else:

                if low_price <= target_price:

                    return {
                        "valid": True,
                        "r_multiple": to_float(
                            trade.get(
                                "r_multiple"
                            )
                        ),
                        "profit": to_float(
                            trade.get(
                                "profit"
                            )
                        ),
                    }

        timeout_bar = data.iloc[
            start_idx
            + max_bars
            - 1
        ]

        timeout_close = float(
            timeout_bar["Close"]
        )

        if direction == "LONG":

            exit_price = (
                timeout_close
                * (
                    1
                    - SLIPPAGE_RATE
                )
            )

            entry_value = (
                entry_price
                * shares
            )

            exit_value = (
                exit_price
                * shares
            )

            profit = (
                exit_value
                - EXIT_BROKERAGE
                - entry_value
                - EXIT_BROKERAGE
            )

        else:

            exit_price = (
                timeout_close
                * (
                    1
                    + SLIPPAGE_RATE
                )
            )

            entry_value = (
                entry_price
                * shares
            )

            exit_value = (
                exit_price
                * shares
            )

            profit = (
                entry_value
                - exit_value
                - EXIT_BROKERAGE
                - EXIT_BROKERAGE
            )

        r_multiple = (
            profit
            / (
                initial_risk
                * shares
            )
        )

        return {
            "valid": True,
            "r_multiple": r_multiple,
            "profit": profit,
        }

    grouped = defaultdict(
        lambda: defaultdict(list)
    )

    for trade in trades:

        entry_trend = str(
            trade.get(
                "entry_trend",
                ""
            )
        ).strip().upper()

        if entry_trend not in (
            "BULLISH",
            "BEARISH",
            "NEUTRAL",
        ):
            continue

        entry_date = trade.get(
            "entry_date"
        )

        if not entry_date:
            continue

        try:

            year = pd.to_datetime(
                entry_date
            ).year

        except Exception:

            continue

        grouped[
            entry_trend
        ][year].append(
            trade
        )

    eligible_groups = []

    total_baseline_r = 0.0
    total_timeout_r = 0.0
    total_delta_r = 0.0

    improved_groups = 0

    for entry_trend in (
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    ):

        years = sorted(
            grouped[
                entry_trend
            ].keys()
        )

        print()
        print(
            entry_trend
        )
        print("-" * 120)

        for validation_year in years:

            training_trades = []

            for year in years:

                if year >= validation_year:
                    continue

                training_trades.extend(
                    grouped[
                        entry_trend
                    ][year]
                )

            validation_trades = grouped[
                entry_trend
            ][validation_year]

            if len(training_trades) < MIN_TRAIN_TRADES:
                continue

            if len(validation_trades) < MIN_VALIDATION_TRADES:
                continue

            train_baseline_r = sum(
                to_float(
                    trade.get(
                        "r_multiple"
                    ),
                    0.0,
                )
                for trade in training_trades
            )

            train_filtered = []

            for trade in training_trades:

                result = simulate_timeout_5(
                    trade
                )

                if not result.get(
                    "valid",
                    False,
                ):
                    continue

                train_filtered.append(
                    result
                )

            if len(train_filtered) < MIN_TRAIN_TRADES:
                continue

            train_timeout_r = sum(
                result.get(
                    "r_multiple",
                    0.0,
                )
                for result in train_filtered
            )

            if train_timeout_r > train_baseline_r:

                selected_filter = "TIMEOUT_5"

            else:

                selected_filter = "NO_FILTER"

            baseline_validation_r = sum(
                to_float(
                    trade.get(
                        "r_multiple"
                    ),
                    0.0,
                )
                for trade in validation_trades
            )

            if selected_filter == "TIMEOUT_5":

                validation_results = []

                for trade in validation_trades:

                    result = simulate_timeout_5(
                        trade
                    )

                    if result.get(
                        "valid",
                        False,
                    ):
                        validation_results.append(
                            result
                        )

                if len(validation_results) < MIN_VALIDATION_TRADES:

                    continue

                timeout_validation_r = sum(
                    result.get(
                        "r_multiple",
                        0.0,
                    )
                    for result in validation_results
                )

            else:

                timeout_validation_r = (
                    baseline_validation_r
                )

            delta_r = (
                timeout_validation_r
                - baseline_validation_r
            )

            print(
                f"{validation_year} "
                f"TrainTrades={len(training_trades):3d} "
                f"TrainR={train_baseline_r:8.2f} "
                f"Selected={selected_filter:10s} "
                f"ValTrades={len(validation_trades):3d} "
                f"BaselineR={baseline_validation_r:8.2f} "
                f"Timeout5R={timeout_validation_r:8.2f} "
                f"DeltaR={delta_r:8.2f}"
            )

            eligible_groups.append(
                (
                    entry_trend,
                    validation_year,
                )
            )

            total_baseline_r += (
                baseline_validation_r
            )

            total_timeout_r += (
                timeout_validation_r
            )

            total_delta_r += delta_r

            if delta_r > 0:
                improved_groups += 1

    print()
    print("=" * 120)
    print(
        "ENTRY TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD SUMMARY"
    )
    print("=" * 120)

    print(
        f"Eligible groups: {len(eligible_groups)}"
    )

    print(
        f"Baseline validation total: "
        f"{total_baseline_r:.2f}R"
    )

    print(
        f"Timeout_5 validation total: "
        f"{total_timeout_r:.2f}R"
    )

    print(
        f"Total DeltaR: "
        f"{total_delta_r:.2f}R"
    )

    print(
        f"Improved groups: "
        f"{improved_groups}/{len(eligible_groups)}"
    )

    if (
        len(eligible_groups) >= 6
        and improved_groups
        >= len(eligible_groups) * 0.60
        and total_delta_r > 0
    ):

        print(
            "RESEARCH RULE: "
            "PROMISING — entry-trend-conditioned executable "
            "TIMEOUT_5 shows positive out-of-sample evidence. "
            "DO NOT DEPLOY YET; continue validation."
        )

    else:

        print(
            "RESEARCH RULE: "
            "NOT ROBUST ENOUGH — entry-trend-conditioned "
            "TIMEOUT_5 does not provide sufficient out-of-sample "
            "evidence. DO NOT MODIFY ENGINE LOGIC."
        )

def stock_timeout_5_walk_forward(trades):
    """
    Stock-conditioned executable TIMEOUT_5 research diagnostic.

    Reuses the already validated executable TIMEOUT_5
    walk-forward engine. Does NOT modify production logic.
    """

    print()
    print("=" * 120)
    print("STOCK × EXECUTABLE TIMEOUT_5 WALK-FORWARD")
    print("=" * 120)

    grouped = defaultdict(list)

    for trade in trades:
        symbol = str(
            trade.get(
                "_symbol",
                ""
            )
        ).strip()

        if not symbol:
            continue

        grouped[symbol].append(
            trade
        )

    eligible_stocks = 0

    for symbol in sorted(
        grouped.keys()
    ):
        stock_trades = grouped[
            symbol
        ]

        if len(stock_trades) < 10:
            continue

        print()
        print("#" * 120)
        print(
            f"STOCK: {symbol} | "
            f"TRADES: {len(stock_trades)}"
        )
        print("#" * 120)

        eligible_stocks += 1

        time_to_05r_timeout_counterfactual_walk_forward(
            stock_trades
        )

    print()
    print("=" * 120)
    print(
        "STOCK × EXECUTABLE TIMEOUT_5 "
        "WALK-FORWARD COMPLETE"
    )
    print("=" * 120)

    print(
        f"Eligible stocks: "
        f"{eligible_stocks}"
    )

def stock_setup_trend_timeout_5_walk_forward(trades):
    """
    Stock × Setup × Trend conditioned executable TIMEOUT_5
    walk-forward research diagnostic.

    Reuses the existing executable TIMEOUT_5
    setup × trend walk-forward engine.

    Research-only.
    Does NOT modify production engine logic.
    """

    print()
    print("=" * 120)
    print(
        "STOCK × SETUP × TREND × EXECUTABLE "
        "TIMEOUT_5 WALK-FORWARD"
    )
    print("=" * 120)

    stock_groups = defaultdict(list)

    for trade in trades:
        symbol = str(
            trade.get(
                "_symbol",
                ""
            )
        ).strip()

        if not symbol:
            continue

        stock_groups[symbol].append(
            trade
        )

    eligible_groups = 0

    for symbol in sorted(
        stock_groups.keys()
    ):
        stock_trades = stock_groups[
            symbol
        ]

        setup_trend_groups = defaultdict(list)

        for trade in stock_trades:
            setup = str(
                trade.get(
                    "setup",
                    ""
                )
            ).strip()

            trend = str(
                trade.get(
                    "entry_trend",
                    ""
                )
            ).strip()

            if not setup or not trend:
                continue

            setup_trend_groups[
                (
                    setup,
                    trend
                )
            ].append(
                trade
            )

        stock_has_eligible_group = False

        for (
            setup,
            trend
        ) in sorted(
            setup_trend_groups.keys()
        ):
            group_trades = setup_trend_groups[
                (
                    setup,
                    trend
                )
            ]

            if len(group_trades) < 10:
                continue

            stock_has_eligible_group = True
            eligible_groups += 1

            print()
            print("#" * 120)
            print(
                f"STOCK: {symbol} | "
                f"SETUP: {setup} | "
                f"TREND: {trend} | "
                f"TRADES: {len(group_trades)}"
            )
            print("#" * 120)

            setup_trend_timeout_5_walk_forward(
                group_trades
            )

        if not stock_has_eligible_group:
            continue

    print()
    print("=" * 120)
    print(
        "STOCK × SETUP × TREND × EXECUTABLE "
        "TIMEOUT_5 WALK-FORWARD COMPLETE"
    )
    print("=" * 120)

    print(
        f"Eligible stock × setup × trend groups: "
        f"{eligible_groups}"
    )

def stock_setup_timeout_5_walk_forward(trades):
    """
    Stock × Setup conditioned executable TIMEOUT_5
    walk-forward research diagnostic.

    Research-only.
    Does NOT modify production engine logic.
    """

    print()
    print("=" * 120)
    print(
        "STOCK × SETUP × EXECUTABLE "
        "TIMEOUT_5 WALK-FORWARD"
    )
    print("=" * 120)

    stock_groups = defaultdict(list)

    for trade in trades:
        symbol = str(
            trade.get(
                "_symbol",
                ""
            )
        ).strip()

        if not symbol:
            continue

        stock_groups[symbol].append(
            trade
        )

    eligible_groups = 0

    for symbol in sorted(
        stock_groups.keys()
    ):
        stock_trades = stock_groups[
            symbol
        ]

        setup_groups = defaultdict(list)

        for trade in stock_trades:
            setup = str(
                trade.get(
                    "setup",
                    ""
                )
            ).strip()

            if not setup:
                continue

            setup_groups[
                setup
            ].append(
                trade
            )

        for setup in sorted(
            setup_groups.keys()
        ):
            group_trades = setup_groups[
                setup
            ]

            if len(group_trades) < 10:
                continue

            eligible_groups += 1

            print()
            print("#" * 120)
            print(
                f"STOCK: {symbol} | "
                f"SETUP: {setup} | "
                f"TRADES: {len(group_trades)}"
            )
            print("#" * 120)

            time_to_05r_timeout_counterfactual_walk_forward(
                group_trades
            )

    print()
    print("=" * 120)
    print(
        "STOCK × SETUP × EXECUTABLE "
        "TIMEOUT_5 WALK-FORWARD COMPLETE"
    )
    print("=" * 120)

    print(
        f"Eligible stock × setup groups: "
        f"{eligible_groups}"
    )

def trend_strength_timeout_5_walk_forward(trades):
    """
    Trend-strength-conditioned executable TIMEOUT_5 walk-forward validation.

    Research-only diagnostic.
    Does NOT modify production engine logic.
    """

    print()
    print("=" * 120)
    print("TREND STRENGTH × EXECUTABLE TIMEOUT_5 WALK-FORWARD")
    print("=" * 120)

TIMEOUT_BARS = 5
MIN_TRAIN_TRADES = 10
MIN_VALIDATION_TRADES = 5

SLIPPAGE_RATE = 0.001
EXIT_BROKERAGE = 20.0

cache = {}

def get_history(symbol, entry_date):
    key = (
        str(symbol),
        str(entry_date),
    )

    if key in cache:
        return cache[key]

    try:
        entry_dt = pd.to_datetime(
            entry_date
        )

        start_date = (
            entry_dt
            - timedelta(days=10)
        ).strftime("%Y-%m-%d")

        end_date = (
            entry_dt
            + timedelta(days=120)
        ).strftime("%Y-%m-%d")

        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=15,
        )

        if data is None or data.empty:
            cache[key] = None
            return None

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):
            data.columns = [
                column[0]
                for column in data.columns
            ]

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
        ]

        if not all(
            column in data.columns
            for column in required_columns
        ):
            cache[key] = None
            return None

        data = data[
            required_columns
        ].copy()

        data.index = pd.to_datetime(
            data.index
        )

        data = data.sort_index()

        cache[key] = data

        return data

    except Exception:
        cache[key] = None
        return None

    strength_buckets = [
        ("STRENGTH_<20", lambda x: x < 20),
        ("STRENGTH_20-40", lambda x: 20 <= x < 40),
        ("STRENGTH_40-60", lambda x: 40 <= x < 60),
        ("STRENGTH_60+", lambda x: x >= 60),
    ]

    def get_strength_bucket(value):
        value = to_float(value)

        if value is None:
            return None

        for bucket_name, condition in strength_buckets:
            if condition(value):
                return bucket_name

        return None

    # --------------------------------------------------------------
    # Build groups by YEAR × TREND STRENGTH
    # --------------------------------------------------------------

    groups = defaultdict(list)

    for trade in trades:
        strength = to_float(
            trade.get("entry_trend_strength")
        )

        if trade.get("entry_trend_strength") in (
            None,
            "",
            "None",
            "nan",
        ):
            continue

        bucket = get_strength_bucket(strength)

        if bucket is None:
            continue

        entry_date = trade.get("entry_date")

        if not entry_date:
            continue

        try:
            year = int(str(entry_date)[:4])
        except Exception:
            continue

        groups[(bucket, year)].append(trade)

    # --------------------------------------------------------------
    # Print raw group counts first
    # --------------------------------------------------------------

    for bucket_name, _ in strength_buckets:
        print()
        print(bucket_name)
        print("-" * 120)

        for year in sorted(
            year
            for bucket, year in groups.keys()
            if bucket == bucket_name
        ):
            print(
                f"{year}: Trades={len(groups[(bucket_name, year)]):3d}"
            )

    # --------------------------------------------------------------
    # Reuse executable reconstruction mechanics
    # --------------------------------------------------------------

    def get_trade_year(trade):
        entry_date = trade.get("entry_date")

        if not entry_date:
            return None

        try:
            return int(str(entry_date)[:4])
        except Exception:
            return None

    def get_entry_price(trade):
        return to_float(
            trade.get("entry_price")
        )

    def get_initial_atr(trade):
        atr = to_float(
            trade.get("entry_atr")
        )

        if atr is None:
            atr = to_float(
                trade.get("ATR")
            )

        return atr

    def get_direction(trade):
        direction = trade.get("direction")

        if direction:
            return str(direction).upper()

        setup = str(
            trade.get(
                "Diagnostic_Setup",
                trade.get("setup", "")
            )
        ).upper()

        if setup.startswith("LONG"):
            return "LONG"

        if setup.startswith("SHORT"):
            return "SHORT"

        return None

    def simulate_timeout_5(trade):
        """
        Reconstruct the executable TIMEOUT_5 counterfactual
        using the same mechanics as the proven overall
        executable timeout diagnostic.
        """

        entry_price = get_entry_price(trade)
        entry_atr = get_initial_atr(trade)
        direction = get_direction(trade)

        if (
            entry_price is None
            or entry_atr is None
            or entry_atr <= 0
            or direction not in ("LONG", "SHORT")
        ):
            return {
                "valid": False,
                "r": 0.0,
            }

        symbol = (
            trade.get("symbol")
            or trade.get("Symbol")
            or trade.get("ticker")
            or trade.get("Ticker")
        )

        entry_date = trade.get("entry_date")

        if not symbol or not entry_date:
            return {
                "valid": False,
                "r": 0.0,
            }

        data = get_history(
            symbol,
            entry_date,
        )

        if data is None or len(data) == 0:
            return {
                "valid": False,
                "r": 0.0,
            }

        try:
            matching = data[
                data.index >= pd.Timestamp(entry_date)
            ]

            if matching.empty:
                return {
                    "valid": False,
                    "r": 0.0,
                }

            start_idx = data.index.get_loc(
                matching.index[0]
            )

        except Exception:
            return {
                "valid": False,
                "r": 0.0,
            }

        initial_risk_per_share = 2.0 * entry_atr

        if initial_risk_per_share <= 0:
            return {
                "valid": False,
                "r": 0.0,
            }

        target_price = (
            entry_price
            + 0.5 * initial_risk_per_share
            if direction == "LONG"
            else
            entry_price
            - 0.5 * initial_risk_per_share
        )

        # ----------------------------------------------------------
        # Find whether +0.5R is reached within first 5 bars
        # ----------------------------------------------------------

        reached = False
        timeout_close = None

        for offset in range(TIMEOUT_BARS):
            idx = start_idx + offset

            if idx >= len(data):
                break

            row = data.iloc[idx]

            high = to_float(row.get("High"))
            low = to_float(row.get("Low"))
            close = to_float(row.get("Close"))

            if (
                high is None
                or low is None
                or close is None
            ):
                continue

            if direction == "LONG":
                if high >= target_price:
                    reached = True
                    break
            else:
                if low <= target_price:
                    reached = True
                    break

            timeout_close = close

        # ----------------------------------------------------------
        # If +0.5R was reached, preserve original trade outcome
        # ----------------------------------------------------------

        if reached:
            original_r = to_float(
                trade.get("r_multiple")
            )

            if original_r is None:
                original_r = 0.0

            return {
                "valid": True,
                "r": original_r,
            }

        # ----------------------------------------------------------
        # Timeout at close of fifth bar
        # ----------------------------------------------------------

        if timeout_close is None:
            return {
                "valid": False,
                "r": 0.0,
            }

        slippage_rate = 0.001
        brokerage = 20.0

        risk_budget = to_float(
            trade.get("risk_budget")
        )

        if risk_budget is None:
            risk_budget = to_float(
                trade.get("initial_cash")
            )

            if risk_budget is not None:
                risk_budget *= 0.01

        if risk_budget is None:
            return {
                "valid": False,
                "r": 0.0,
            }

        quantity = (
            risk_budget / initial_risk_per_share
        )

        if quantity <= 0:
            return {
                "valid": False,
                "r": 0.0,
            }

        entry_cost = (
            entry_price * quantity
            + brokerage
        )

        if direction == "LONG":
            exit_price = (
                timeout_close
                * (1 - slippage_rate)
            )

            exit_value = exit_price * quantity

            pnl = (
                exit_value
                - brokerage
                - entry_cost
            )

        else:
            exit_price = (
                timeout_close
                * (1 + slippage_rate)
            )

            exit_value = exit_price * quantity

            pnl = (
                entry_cost
                - exit_value
                - brokerage
            )

        r_multiple = (
            pnl / risk_budget
        )

        return {
            "valid": True,
            "r": r_multiple,
        }

    # --------------------------------------------------------------
    # Walk-forward evaluation
    # --------------------------------------------------------------

    total_baseline_r = 0.0
    total_timeout_r = 0.0
    total_delta_r = 0.0
    eligible_groups = 0
    improved_groups = 0

    all_years = sorted(
        set(
            year
            for _, year in groups.keys()
        )
    )

    for bucket_name, _ in strength_buckets:

        print()
        print(bucket_name)
        print("-" * 120)

        for year in all_years:

            validation_trades = groups.get(
                (bucket_name, year),
                []
            )

            if len(validation_trades) < MIN_VALIDATION_TRADES:
                continue

            train_trades = []

            for prior_year in all_years:
                if prior_year >= year:
                    continue

                train_trades.extend(
                    groups.get(
                        (bucket_name, prior_year),
                        []
                    )
                )

            if len(train_trades) < MIN_TRAIN_TRADES:
                continue

            # ------------------------------------------------------
            # Evaluate training baseline and TIMEOUT_5
            # ------------------------------------------------------

            train_baseline_r = 0.0
            train_timeout_r = 0.0
            train_timeout_valid = 0

            for trade in train_trades:

                original_r = to_float(
                    trade.get("r_multiple")
                )

                if original_r is not None:
                    train_baseline_r += original_r

                result = simulate_timeout_5(trade)

                if result["valid"]:
                    train_timeout_r += result["r"]
                    train_timeout_valid += 1

            # Need enough executable training observations.
            if train_timeout_valid < MIN_TRAIN_TRADES:
                continue

            if train_timeout_r > train_baseline_r:
                selected = "TIMEOUT_5"
            else:
                selected = "NO_FILTER"

            # ------------------------------------------------------
            # Validation baseline
            # ------------------------------------------------------

            baseline_r = 0.0

            for trade in validation_trades:
                original_r = to_float(
                    trade.get("r_multiple")
                )

                if original_r is not None:
                    baseline_r += original_r

            # ------------------------------------------------------
            # Validation executable result
            # ------------------------------------------------------

            timeout_r = 0.0
            valid_timeout_trades = 0

            for trade in validation_trades:

                result = simulate_timeout_5(trade)

                if not result["valid"]:
                    continue

                timeout_r += result["r"]
                valid_timeout_trades += 1

            if valid_timeout_trades < MIN_VALIDATION_TRADES:
                continue

            if selected == "TIMEOUT_5":
                filtered_r = timeout_r
            else:
                filtered_r = baseline_r

            delta_r = (
                filtered_r
                - baseline_r
            )

            eligible_groups += 1
            total_baseline_r += baseline_r
            total_timeout_r += filtered_r
            total_delta_r += delta_r

            if delta_r > 0:
                improved_groups += 1

            print(
                f"{year} "
                f"TrainTrades={len(train_trades):3d} "
                f"TrainR={train_baseline_r:8.2f} "
                f"Selected={selected:<9} "
                f"ValTrades={len(validation_trades):3d} "
                f"BaselineR={baseline_r:8.2f} "
                f"Timeout5R={filtered_r:8.2f} "
                f"DeltaR={delta_r:8.2f}"
            )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "TREND STRENGTH × EXECUTABLE TIMEOUT_5 WALK-FORWARD SUMMARY"
    )
    print("=" * 120)

    print(
        f"Eligible groups: {eligible_groups}"
    )

    print(
        f"Baseline validation total: "
        f"{total_baseline_r:.2f}R"
    )

    print(
        f"Timeout_5 validation total: "
        f"{total_timeout_r:.2f}R"
    )

    print(
        f"Total DeltaR: "
        f"{total_delta_r:.2f}R"
    )

    print(
        f"Improved groups: "
        f"{improved_groups}/{eligible_groups}"
    )

    if (
        eligible_groups > 0
        and improved_groups >= max(
            2,
            eligible_groups // 2 + 1
        )
        and total_delta_r > 0
    ):
        print(
            "RESEARCH RULE: PROMISING — trend-strength-conditioned "
            "TIMEOUT_5 improved a majority of eligible validation "
            "groups. DO NOT DEPLOY YET; continue research."
        )
    else:
        print(
            "RESEARCH RULE: NOT ROBUST ENOUGH — trend-strength-conditioned "
            "TIMEOUT_5 does not provide sufficient out-of-sample evidence. "
            "DO NOT MODIFY ENGINE LOGIC."
        )



def setup_trend_time_to_05r_diagnostic(trades):

    print()
    print("=" * 120)
    print("SETUP × TREND × TIME-TO-0.5R BUCKET")
    print("=" * 120)

    buckets = [
        ("0-5", 0, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-40", 21, 40),
        ("40+", 41, float("inf")),
    ]

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get("setup", "UNKNOWN")
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        bars = to_float(
            trade.get("bars_to_0_5r")
        )

        if bars is None:
            continue

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(trade)

    for (
        setup,
        trend
    ) in sorted(groups.keys()):

        group_trades = groups[
            (
                setup,
                trend
            )
        ]

        print()
        print(
            f"{setup} × {trend}"
        )

        print("-" * 120)

        for (
            bucket_name,
            lower,
            upper
        ) in buckets:

            bucket_trades = []

            for trade in group_trades:

                bars = to_float(
                    trade.get(
                        "bars_to_0_5r"
                    )
                )

                if bars is None:
                    continue

                if (
                    bars >= lower
                    and bars <= upper
                ):
                    bucket_trades.append(
                        trade
                    )

            if not bucket_trades:

                continue

            wins = [
                trade
                for trade in bucket_trades
                if to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) > 0
            ]

            losses = [
                trade
                for trade in bucket_trades
                if to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) <= 0
            ]

            gross_profit = sum(
                max(
                    to_float(
                        trade.get(
                            "profit"
                        )
                    ) or 0.0,
                    0.0
                )
                for trade in bucket_trades
            )

            gross_loss = sum(
                abs(
                    min(
                        to_float(
                            trade.get(
                                "profit"
                            )
                        ) or 0.0,
                        0.0
                    )
                )
                for trade in bucket_trades
            )

            profit_factor = (
                gross_profit / gross_loss
                if gross_loss > 0
                else 0.0
            )

            total_r = sum(
                to_float(
                    trade.get(
                        "r_multiple"
                    )
                ) or 0.0
                for trade in bucket_trades
            )

            avg_r = (
                total_r / len(bucket_trades)
                if bucket_trades
                else 0.0
            )

            win_rate = (
                len(wins)
                / len(bucket_trades)
                * 100
            )

            print(
                f"{bucket_name:<6}"
                f" Trades={len(bucket_trades):>3} "
                f"Wins={len(wins):>3} "
                f"Losses={len(losses):>3} "
                f"Win={win_rate:>6.2f}% "
                f"PF={profit_factor:>6.2f} "
                f"R={total_r:>7.2f} "
                f"AvgR={avg_r:>7.3f}"
            )

def setup_trend_time_filter_counterfactual(trades):

    print()
    print("=" * 120)
    print("SETUP × TREND × TIME-TO-0.5R COUNTERFACTUAL FILTER")
    print("=" * 120)

    cutoffs = [
        ("NO FILTER", None),
        ("EXCLUDE <=5", 5),
        ("EXCLUDE <=10", 10),
        ("EXCLUDE <=20", 20),
    ]

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(trade)

    for (
        setup,
        trend
    ) in sorted(groups.keys()):

        group_trades = groups[
            (
                setup,
                trend
            )
        ]

        print()
        print(
            f"{setup} × {trend}"
        )

        print("-" * 120)

        for (
            label,
            cutoff
        ) in cutoffs:

            filtered = []

            for trade in group_trades:

                bars = to_float(
                    trade.get(
                        "bars_to_0_5r"
                    )
                )

                if cutoff is None:

                    filtered.append(
                        trade
                    )

                elif (
                    bars is not None
                    and bars > cutoff
                ):

                    filtered.append(
                        trade
                    )

                elif bars is None:

                    filtered.append(
                        trade
                    )

            if not filtered:
                continue

            wins = [
                trade
                for trade in filtered
                if (
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    ) or 0.0
                ) > 0
            ]

            gross_profit = sum(
                max(
                    (
                        to_float(
                            trade.get(
                                "profit"
                            )
                        ) or 0.0
                    ),
                    0.0
                )
                for trade in filtered
            )

            gross_loss = sum(
                abs(
                    min(
                        (
                            to_float(
                                trade.get(
                                    "profit"
                                )
                            ) or 0.0
                        ),
                        0.0
                    )
                )
                for trade in filtered
            )

            total_r = sum(
                (
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    ) or 0.0
                )
                for trade in filtered
            )

            profit_factor = (
                gross_profit / gross_loss
                if gross_loss > 0
                else 0.0
            )

            win_rate = (
                len(wins)
                / len(filtered)
                * 100
            )

            avg_r = (
                total_r
                / len(filtered)
            )

            print(
                f"{label:<16}"
                f" Trades={len(filtered):>3} "
                f"Wins={len(wins):>3} "
                f"Win={win_rate:>6.2f}% "
                f"PF={profit_factor:>6.2f} "
                f"R={total_r:>7.2f} "
                f"AvgR={avg_r:>7.3f}"
            )

def stock_setup_trend_time_diagnostic(trades):

    print()
    print("=" * 120)
    print("STOCK × SETUP × TREND × TIME-TO-0.5R")
    print("=" * 120)

    buckets = [
        ("0-5", 0, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-40", 21, 40),
        ("40+", 41, float("inf")),
    ]

    groups = defaultdict(list)

    for trade in trades:

        symbol = trade.get(
            "_symbol",
            trade.get(
                "symbol",
                "UNKNOWN"
            )
        )

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        bars = to_float(
            trade.get(
                "bars_to_0_5r"
            )
        )

        if bars is None:
            continue

        groups[
            (
                str(symbol),
                str(setup),
                str(trend)
            )
        ].append(trade)

    for (
        symbol,
        setup,
        trend
    ) in sorted(groups.keys()):

        group_trades = groups[
            (
                symbol,
                setup,
                trend
            )
        ]

        print()
        print(
            f"{symbol} × {setup} × {trend}"
        )

        print("-" * 120)

        for (
            bucket_name,
            lower,
            upper
        ) in buckets:

            bucket_trades = []

            for trade in group_trades:

                bars = to_float(
                    trade.get(
                        "bars_to_0_5r"
                    )
                )

                if (
                    bars >= lower
                    and bars <= upper
                ):
                    bucket_trades.append(
                        trade
                    )

            if not bucket_trades:
                continue

            wins = [
                trade
                for trade in bucket_trades
                if (
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    ) or 0.0
                ) > 0
            ]

            gross_profit = sum(
                max(
                    (
                        to_float(
                            trade.get(
                                "profit"
                            )
                        ) or 0.0
                    ),
                    0.0
                )
                for trade in bucket_trades
            )

            gross_loss = sum(
                abs(
                    min(
                        (
                            to_float(
                                trade.get(
                                    "profit"
                                )
                            ) or 0.0
                        ),
                        0.0
                    )
                )
                for trade in bucket_trades
            )

            total_r = sum(
                (
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    ) or 0.0
                )
                for trade in bucket_trades
            )

            profit_factor = (
                gross_profit / gross_loss
                if gross_loss > 0
                else 0.0
            )

            win_rate = (
                len(wins)
                / len(bucket_trades)
                * 100
            )

            avg_r = (
                total_r
                / len(bucket_trades)
            )

            print(
                f"{bucket_name:<6}"
                f" Trades={len(bucket_trades):>3} "
                f"Wins={len(wins):>3} "
                f"Win={win_rate:>6.2f}% "
                f"PF={profit_factor:>6.2f} "
                f"R={total_r:>7.2f} "
                f"AvgR={avg_r:>7.3f}"
            )

def immediate_failure_diagnostic(trades):

    print()
    print("=" * 120)
    print("EARLY POST-ENTRY FAILURE DIAGNOSTIC")
    print("=" * 120)

    print()
    print(
        "IMMEDIATE_FAILURE = adverse movement >= 0.5R "
        "within first 2 bars after entry"
    )

    def normalize_bool(value):

        if isinstance(value, bool):
            return value

        if value is None:
            return False

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes"
        }

    def print_group(label, group_trades):

        if not group_trades:
            print(
                f"{label:<30}"
                f"Trades=  0"
            )
            return

        wins = [
            trade
            for trade in group_trades
            if to_float(
                trade.get("r_multiple")
            ) > 0
        ]

        losses = [
            trade
            for trade in group_trades
            if to_float(
                trade.get("r_multiple")
            ) <= 0
        ]

        total_r = sum(
            to_float(
                trade.get("r_multiple")
            )
            for trade in group_trades
        )

        gross_profit = sum(
            to_float(
                trade.get("r_multiple")
            )
            for trade in wins
        )

        gross_loss = abs(
            sum(
                to_float(
                    trade.get("r_multiple")
                )
                for trade in losses
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        win_rate = (
            len(wins)
            / len(group_trades)
            * 100
        )

        avg_r = (
            total_r
            / len(group_trades)
        )

        print(
            f"{label:<30}"
            f"Trades={len(group_trades):>3} "
            f"Win={win_rate:>6.2f}% "
            f"PF={profit_factor:>6.2f} "
            f"R={total_r:>7.2f} "
            f"AvgR={avg_r:>7.3f}"
        )

    # --------------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------------

    immediate_failure_trades = [
        trade
        for trade in trades
        if normalize_bool(
            trade.get(
                "immediate_failure",
                False
            )
        )
    ]

    no_immediate_failure_trades = [
        trade
        for trade in trades
        if not normalize_bool(
            trade.get(
                "immediate_failure",
                False
            )
        )
    ]

    print()
    print("-" * 120)
    print("OVERALL")
    print("-" * 120)

    print_group(
        "IMMEDIATE_FAILURE",
        immediate_failure_trades
    )

    print_group(
        "NO_IMMEDIATE_FAILURE",
        no_immediate_failure_trades
    )

    # --------------------------------------------------------------
    # SETUP
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("SETUP × EARLY FAILURE")
    print("-" * 120)

    setup_groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        failure_label = (
            "IMMEDIATE_FAILURE"
            if normalize_bool(
                trade.get(
                    "immediate_failure",
                    False
                )
            )
            else "NO_IMMEDIATE_FAILURE"
        )

        setup_groups[
            (
                str(setup),
                failure_label
            )
        ].append(
            trade
        )

    for (
        setup,
        failure_label
    ), group_trades in sorted(
        setup_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print_group(
            f"{setup:<25} {failure_label}",
            group_trades
        )

    # --------------------------------------------------------------
    # ENTRY TREND
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("ENTRY TREND × EARLY FAILURE")
    print("-" * 120)

    trend_groups = defaultdict(list)

    for trade in trades:

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        failure_label = (
            "IMMEDIATE_FAILURE"
            if normalize_bool(
                trade.get(
                    "immediate_failure",
                    False
                )
            )
            else "NO_IMMEDIATE_FAILURE"
        )

        trend_groups[
            (
                str(trend),
                failure_label
            )
        ].append(
            trade
        )

    for (
        trend,
        failure_label
    ), group_trades in sorted(
        trend_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print_group(
            f"{trend:<25} {failure_label}",
            group_trades
        )

    # --------------------------------------------------------------
    # YEAR STABILITY
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("YEAR × EARLY FAILURE")
    print("-" * 120)

    year_groups = defaultdict(list)

    for trade in trades:

        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        )

        year = (
            entry_date[:4]
            if len(entry_date) >= 4
            else "UNKNOWN"
        )

        failure_label = (
            "IMMEDIATE_FAILURE"
            if normalize_bool(
                trade.get(
                    "immediate_failure",
                    False
                )
            )
            else "NO_IMMEDIATE_FAILURE"
        )

        year_groups[
            (
                year,
                failure_label
            )
        ].append(
            trade
        )

    for (
        year,
        failure_label
    ), group_trades in sorted(
        year_groups.items()
    ):

        print_group(
            f"{year:<10} {failure_label}",
            group_trades
        )

    # --------------------------------------------------------------
    # TRAINING VS HOLDOUT
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("TRAINING VS HOLDOUT")
    print("-" * 120)

    training_trades = []
    holdout_trades = []

    for trade in trades:

        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        )

        if entry_date < "2025-01-01":
            training_trades.append(
                trade
            )
        else:
            holdout_trades.append(
                trade
            )

    for period_name, period_trades in [
        (
            "TRAINING < 2025",
            training_trades
        ),
        (
            "HOLDOUT >= 2025",
            holdout_trades
        )
    ]:

        print()
        print(
            f"{period_name}"
        )

        period_failure = [
            trade
            for trade in period_trades
            if normalize_bool(
                trade.get(
                    "immediate_failure",
                    False
                )
            )
        ]

        period_no_failure = [
            trade
            for trade in period_trades
            if not normalize_bool(
                trade.get(
                    "immediate_failure",
                    False
                )
            )
        ]

        print_group(
            "IMMEDIATE_FAILURE",
            period_failure
        )

        print_group(
            "NO_IMMEDIATE_FAILURE",
            period_no_failure
        )

    # --------------------------------------------------------------
    # COUNTERFACTUAL REMOVAL TEST
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("EARLY FAILURE REMOVAL — RESEARCH ONLY")
    print("-" * 120)

    print()
    print("BASELINE")

    print_group(
        "ALL TRADES",
        trades
    )

    print()
    print("REMOVE IMMEDIATE_FAILURE")

    print_group(
        "REMAINING TRADES",
        no_immediate_failure_trades
    )

    print()
    print(
        f"Trades removed: "
        f"{len(immediate_failure_trades)}"
    )

def setup_trend_early_adverse_diagnostic(trades):
    """
    Research-only diagnostic.

    Tests whether early adverse movement remains informative
    after controlling for SETUP + TREND and TIME PERIOD.

    This does NOT modify the engine or historical execution.
    """

    windows = [
        2,
        3,
        5,
    ]

    thresholds = [
        0.25,
        0.35,
        0.50,
    ]

    # --------------------------------------------------------------
    # BUILD SETUP × TREND GROUPS
    # --------------------------------------------------------------

    groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        groups[
            (
                str(setup),
                str(trend)
            )
        ].append(
            trade
        )

    # --------------------------------------------------------------
    # OVERALL SETUP × TREND × EARLY ADVERSE-R
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print("SETUP × TREND × EARLY ADVERSE-R")
    print("=" * 150)

    print()
    print(
        "Research question: Does early adverse movement remain "
        "informative after controlling for SETUP + TREND?"
    )

    print()
    print(
        "Flagged = trade whose cumulative maximum adverse movement "
        "reached the threshold within the specified number of bars."
    )

    for (
        setup,
        trend
    ) in sorted(groups.keys()):

        group_trades = groups[
            (
                setup,
                trend
            )
        ]

        print()
        print("=" * 150)
        print(
            f"{setup} × {trend}"
        )
        print("=" * 150)

        print(
            f"{'WINDOW':<9}"
            f"{'THRESH':<9}"
            f"{'FLAGGED':<9}"
            f"{'FLAG WIN':<10}"
            f"{'FLAG LOSS':<11}"
            f"{'REMAIN':<9}"
            f"{'REM WIN':<9}"
            f"{'WIN%':<9}"
            f"{'PF':<9}"
            f"{'TOTAL R':<11}"
            f"{'AVG R':<9}"
        )

        print("-" * 150)

        for window in windows:

            column = (
                f"early_adverse_r_bar_{window}"
            )

            for threshold in thresholds:

                flagged = []
                remaining = []

                for trade in group_trades:

                    value = to_float(
                        trade.get(column)
                    )

                    if (
                        value is not None
                        and value >= threshold
                    ):
                        flagged.append(
                            trade
                        )
                    else:
                        remaining.append(
                            trade
                        )

                if not flagged:
                    continue

                flagged_wins = [
                    trade
                    for trade in flagged
                    if (
                        to_float(
                            trade.get("r_multiple")
                        ) or 0.0
                    ) > 0
                ]

                flagged_losses = [
                    trade
                    for trade in flagged
                    if (
                        to_float(
                            trade.get("r_multiple")
                        ) or 0.0
                    ) <= 0
                ]

                remaining_r = [
                    to_float(
                        trade.get("r_multiple")
                    ) or 0.0
                    for trade in remaining
                ]

                remaining_wins = [
                    r
                    for r in remaining_r
                    if r > 0
                ]

                remaining_losses = [
                    r
                    for r in remaining_r
                    if r < 0
                ]

                gross_profit = sum(
                    remaining_wins
                )

                gross_loss = abs(
                    sum(
                        remaining_losses
                    )
                )

                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else 0.0
                )

                total_r = sum(
                    remaining_r
                )

                avg_r = (
                    total_r / len(remaining)
                    if remaining
                    else 0.0
                )

                win_rate = (
                    len(remaining_wins)
                    / len(remaining)
                    * 100
                    if remaining
                    else 0.0
                )

                print(
                    f"{window:<9}"
                    f"{threshold:<9.2f}"
                    f"{len(flagged):<9}"
                    f"{len(flagged_wins):<10}"
                    f"{len(flagged_losses):<11}"
                    f"{len(remaining):<9}"
                    f"{len(remaining_wins):<9}"
                    f"{win_rate:<9.2f}"
                    f"{profit_factor:<9.2f}"
                    f"{total_r:<11.2f}"
                    f"{avg_r:<9.3f}"
                )

    # --------------------------------------------------------------
    # TIME VALIDATION
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print("SETUP × TREND × EARLY ADVERSE-R × TIME VALIDATION")
    print("=" * 150)

    print()
    print(
        "Training = before 2025"
    )

    print(
        "Holdout = 2025 onward"
    )

    print(
        "2026 = calendar year 2026"
    )

    periods = [
        (
            "TRAINING < 2025",
            lambda year: year < 2025
        ),
        (
            "HOLDOUT >= 2025",
            lambda year: year >= 2025
        ),
        (
            "2026",
            lambda year: year == 2026
        ),
    ]

    for (
        period_name,
        period_filter
    ) in periods:

        period_trades = []

        for trade in trades:

            entry_date = str(
                trade.get(
                    "entry_date",
                    ""
                )
            ).strip()

            if len(entry_date) < 4:
                continue

            try:
                year = int(
                    entry_date[:4]
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            if period_filter(year):
                period_trades.append(
                    trade
                )

        print()
        print("-" * 150)
        print(
            period_name
        )
        print("-" * 150)

        period_groups = defaultdict(list)

        for trade in period_trades:

            setup = trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )

            trend = trade.get(
                "entry_trend",
                "UNKNOWN"
            )

            period_groups[
                (
                    str(setup),
                    str(trend)
                )
            ].append(
                trade
            )

        for (
            setup,
            trend
        ) in sorted(period_groups.keys()):

            group_trades = period_groups[
                (
                    setup,
                    trend
                )
            ]

            print()
            print(
                f"{setup} × {trend}"
            )

            print(
                f"{'WINDOW':<9}"
                f"{'THRESH':<9}"
                f"{'FLAGGED':<9}"
                f"{'REMAIN':<9}"
                f"{'WIN%':<9}"
                f"{'PF':<9}"
                f"{'TOTAL R':<11}"
                f"{'AVG R':<9}"
            )

            print("-" * 100)

            for window in windows:

                column = (
                    f"early_adverse_r_bar_{window}"
                )

                for threshold in thresholds:

                    remaining = []
                    flagged_count = 0

                    for trade in group_trades:

                        value = to_float(
                            trade.get(column)
                        )

                        if (
                            value is not None
                            and value >= threshold
                        ):
                            flagged_count += 1
                        else:
                            remaining.append(
                                trade
                            )

                    if (
                        not remaining
                        and flagged_count == 0
                    ):
                        continue

                    r_values = [
                        to_float(
                            trade.get(
                                "r_multiple"
                            )
                        ) or 0.0
                        for trade in remaining
                    ]

                    wins = [
                        r
                        for r in r_values
                        if r > 0
                    ]

                    losses = [
                        r
                        for r in r_values
                        if r < 0
                    ]

                    gross_profit = sum(
                        wins
                    )

                    gross_loss = abs(
                        sum(losses)
                    )

                    profit_factor = (
                        gross_profit / gross_loss
                        if gross_loss > 0
                        else 0.0
                    )

                    total_r = sum(
                        r_values
                    )

                    avg_r = (
                        total_r / len(r_values)
                        if r_values
                        else 0.0
                    )

                    win_rate = (
                        len(wins)
                        / len(r_values)
                        * 100
                        if r_values
                        else 0.0
                    )

                    print(
                        f"{window:<9}"
                        f"{threshold:<9.2f}"
                        f"{flagged_count:<9}"
                        f"{len(remaining):<9}"
                        f"{win_rate:<9.2f}"
                        f"{profit_factor:<9.2f}"
                        f"{total_r:<11.2f}"
                        f"{avg_r:<9.3f}"
                    )

def early_adverse_r_walk_forward_diagnostic(trades):
    """
    Research-only walk-forward validation.

    For each validation year:
        1. Use only prior years as training data.
        2. Select the best early-adverse filter from training data.
        3. Apply that selected filter to the validation year.
        4. Report validation performance.

    This does NOT modify the engine or historical execution.
    """

    candidates = [
        ("NO_FILTER", None, None),
        ("1B_0.25R", 1, 0.25),
        ("2B_0.25R", 2, 0.25),
        ("3B_0.25R", 3, 0.25),
        ("4B_0.25R", 4, 0.25),
        ("5B_0.25R", 5, 0.25),
        ("5B_0.35R", 5, 0.35),
        ("5B_0.50R", 5, 0.50),
    ]

    validation_years = [2024, 2025, 2026]

    minimum_training_trades = 10

    def trade_year(trade):
        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        ).strip()

        if len(entry_date) < 4:
            return None

        try:
            return int(
                entry_date[:4]
            )
        except (TypeError, ValueError):
            return None

    def filter_trade(
        trade,
        window,
        threshold
    ):
        if window is None:
            return True

        field = (
            f"early_adverse_r_bar_{window}"
        )

        value = to_float(
            trade.get(field),
            default=None
        )

        if value is None:
            return True

        return value < threshold

    def evaluate(
        period_trades,
        window,
        threshold
    ):
        remaining = [
            trade
            for trade in period_trades
            if filter_trade(
                trade,
                window,
                threshold
            )
        ]

        total_r = sum(
            to_float(
                trade.get(
                    "r_multiple"
                )
            )
            for trade in remaining
        )

        wins = [
            trade
            for trade in remaining
            if to_float(
                trade.get(
                    "r_multiple"
                )
            ) > 0
        ]

        gross_profit = sum(
            max(
                to_float(
                    trade.get(
                        "r_multiple"
                    )
                ),
                0.0
            )
            for trade in remaining
        )

        gross_loss = sum(
            abs(
                min(
                    to_float(
                        trade.get(
                            "r_multiple"
                        )
                    ),
                    0.0
                )
            )
            for trade in remaining
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        win_rate = (
            len(wins)
            / len(remaining)
            * 100
            if remaining
            else 0.0
        )

        return {
            "trades": len(remaining),
            "wins": len(wins),
            "win_rate": win_rate,
            "pf": profit_factor,
            "total_r": total_r,
        }

    print()
    print("=" * 150)
    print(
        "EARLY ADVERSE-R WALK-FORWARD VALIDATION"
    )
    print("=" * 150)

    print()
    print(
        "IMPORTANT: each validation year is evaluated "
        "using parameters selected ONLY from prior years."
    )

    walk_forward_results = []

    for validation_year in validation_years:

        training_trades = [
            trade
            for trade in trades
            if (
                trade_year(trade) is not None
                and trade_year(trade) < validation_year
            )
        ]

        validation_trades = [
            trade
            for trade in trades
            if (
                trade_year(trade) == validation_year
            )
        ]

        print()
        print("-" * 150)
        print(
            f"TRAINING: < {validation_year} "
            f"→ VALIDATION: {validation_year}"
        )
        print("-" * 150)

        print(
            f"Training trades   : "
            f"{len(training_trades)}"
        )

        print(
            f"Validation trades : "
            f"{len(validation_trades)}"
        )

        if (
            len(training_trades)
            < minimum_training_trades
        ):
            print(
                "SKIPPED: insufficient "
                "training sample."
            )
            continue

        training_results = []

        for (
            name,
            window,
            threshold
        ) in candidates:

            result = evaluate(
                training_trades,
                window,
                threshold
            )

            training_results.append(
                (
                    name,
                    window,
                    threshold,
                    result
                )
            )

        eligible_results = [
            item
            for item in training_results
            if (
                item[3]["trades"]
                >= minimum_training_trades
            )
        ]

        if not eligible_results:
            print(
                "SKIPPED: no candidate has "
                "enough remaining training trades."
            )
            continue

        best = max(
            eligible_results,
            key=lambda item: (
                item[3]["total_r"],
                item[3]["pf"],
                item[3]["trades"]
            )
        )

        (
            best_name,
            best_window,
            best_threshold,
            best_training_result
        ) = best

        baseline_training = evaluate(
            training_trades,
            None,
            None
        )

        validation_result = evaluate(
            validation_trades,
            best_window,
            best_threshold
        )

        validation_baseline = evaluate(
            validation_trades,
            None,
            None
        )

        print()
        print("TRAINING SELECTION")
        print(
            f"Selected filter   : "
            f"{best_name}"
        )

        print(
            f"Training result   : "
            f"Trades={best_training_result['trades']} "
            f"Win={best_training_result['win_rate']:.2f}% "
            f"PF={best_training_result['pf']:.2f} "
            f"R={best_training_result['total_r']:.2f}"
        )

        print(
            f"Training baseline : "
            f"Trades={baseline_training['trades']} "
            f"Win={baseline_training['win_rate']:.2f}% "
            f"PF={baseline_training['pf']:.2f} "
            f"R={baseline_training['total_r']:.2f}"
        )

        print()
        print("UNSEEN VALIDATION")

        print(
            f"Baseline          : "
            f"Trades={validation_baseline['trades']} "
            f"Win={validation_baseline['win_rate']:.2f}% "
            f"PF={validation_baseline['pf']:.2f} "
            f"R={validation_baseline['total_r']:.2f}"
        )

        print(
            f"Selected filter   : "
            f"Trades={validation_result['trades']} "
            f"Win={validation_result['win_rate']:.2f}% "
            f"PF={validation_result['pf']:.2f} "
            f"R={validation_result['total_r']:.2f}"
        )

        validation_delta = (
            validation_result["total_r"]
            - validation_baseline["total_r"]
        )

        print(
            f"Validation ΔR     : "
            f"{validation_delta:.2f}"
        )

        if validation_delta > 0:
            print(
                "Validation verdict: IMPROVED"
            )
        elif validation_delta < 0:
            print(
                "Validation verdict: WORSE"
            )
        else:
            print(
                "Validation verdict: UNCHANGED"
            )

        walk_forward_results.append(
            {
                "validation_year": validation_year,
                "selected_filter": best_name,
                "training_trades": len(
                    training_trades
                ),
                "validation_trades": len(
                    validation_trades
                ),
                "validation_baseline_r": (
                    validation_baseline["total_r"]
                ),
                "validation_filtered_r": (
                    validation_result["total_r"]
                ),
                "validation_delta_r": (
                    validation_delta
                ),
            }
        )

    print()
    print("=" * 150)
    print(
        "WALK-FORWARD SUMMARY"
    )
    print("=" * 150)

    if not walk_forward_results:
        print(
            "NO VALID WALK-FORWARD RESULTS"
        )
        return

    total_baseline_r = sum(
        item["validation_baseline_r"]
        for item in walk_forward_results
    )

    total_filtered_r = sum(
        item["validation_filtered_r"]
        for item in walk_forward_results
    )

    improved_years = sum(
        1
        for item in walk_forward_results
        if item["validation_delta_r"] > 0
    )

    print()

    for item in walk_forward_results:
        print(
            f"{item['validation_year']} | "
            f"Selected={item['selected_filter']:<12} "
            f"BaselineR={item['validation_baseline_r']:>7.2f} "
            f"FilteredR={item['validation_filtered_r']:>7.2f} "
            f"ΔR={item['validation_delta_r']:>7.2f}"
        )

    print()
    print(
        f"Validation baseline total R : "
        f"{total_baseline_r:.2f}"
    )

    print(
        f"Validation filtered total R : "
        f"{total_filtered_r:.2f}"
    )

    print(
        f"Total validation ΔR         : "
        f"{total_filtered_r - total_baseline_r:.2f}"
    )

    print(
        f"Years improved              : "
        f"{improved_years}/"
        f"{len(walk_forward_results)}"
    )

    print()
    print(
        "RESEARCH RULE:"
    )

    if (
        total_filtered_r > total_baseline_r
        and improved_years
        >= 2
    ):
        print(
            "PROMISING — filter improved "
            "at least 2 validation years."
        )
        print(
            "Do NOT deploy yet; continue "
            "with setup-level walk-forward validation."
        )
    else:
        print(
            "NOT ROBUST — early-adverse filter "
            "did not consistently improve "
            "unseen validation years."
        )
        print(
            "Do NOT modify engine logic."
        )

def setup_early_adverse_walk_forward_diagnostic(trades):
    """
    Walk-forward validation of Early Adverse-R filters at the
    Setup level.

    For each validation year:
      1. Use only prior-year data from the same Setup.
      2. Select the best Early Adverse-R filter on training data.
      3. Apply that filter to the unseen validation year.
      4. Compare filtered validation R against validation baseline.

    Research focus:
      - Does the Early Adverse-R effect survive out-of-sample
        at the Setup level?
      - Which setups consistently benefit?
    """

    print("\n" + "=" * 90)
    print("SETUP EARLY ADVERSE-R WALK-FORWARD VALIDATION")
    print("=" * 90)

    candidates = [
        ("NO_FILTER", None, None),
        ("1B_0.25R", 1, 0.25),
        ("2B_0.25R", 2, 0.25),
        ("3B_0.25R", 3, 0.25),
        ("4B_0.25R", 4, 0.25),
        ("5B_0.25R", 5, 0.25),
        ("5B_0.35R", 5, 0.35),
        ("5B_0.50R", 5, 0.50),
    ]

    MIN_TRAIN_TRADES = 10
    MIN_SELECTED_TRAIN_TRADES = 10

    def get_setup(trade):
        return (
            trade.get("Diagnostic_Setup")
            or trade.get("setup")
            or "UNKNOWN"
        )

    def get_year(trade):
        entry_date = str(trade.get("entry_date", ""))

        try:
            return int(entry_date[:4])
        except (ValueError, TypeError):
            return None

    def get_r(trade):
        value = trade.get("R_multiple")

        if value is None:
            value = trade.get("r_multiple")

        if value is None:
            value = trade.get("R")

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def get_early_adverse_r(trade):
        value = trade.get("early_adverse_r")

        if value is None:
            value = trade.get("Early_Adverse_R")

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def passes_filter(trade, bars, threshold):
        if bars is None:
            return True

        adverse_r = get_early_adverse_r(trade)

        # Missing early adverse-R information is retained,
        # matching the existing Early Adverse-R diagnostic logic.
        if adverse_r is None:
            return True

        return adverse_r < threshold

    def evaluate(trade_list, bars, threshold):
        selected = [
            trade
            for trade in trade_list
            if passes_filter(trade, bars, threshold)
        ]

        total_r = sum(
            get_r(trade)
            for trade in selected
        )

        wins = sum(
            1
            for trade in selected
            if get_r(trade) > 0
        )

        losses = [
            get_r(trade)
            for trade in selected
            if get_r(trade) < 0
        ]

        winners = [
            get_r(trade)
            for trade in selected
            if get_r(trade) > 0
        ]

        gross_profit = sum(winners)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0:
            pf = float("inf")
        else:
            pf = 0.0

        win_rate = (
            wins / len(selected) * 100
            if selected
            else 0.0
        )

        return {
            "trades": len(selected),
            "wins": wins,
            "win_rate": win_rate,
            "pf": pf,
            "total_r": total_r,
        }

    def fmt_pf(value):
        if value == float("inf"):
            return "inf"

        return f"{value:.2f}"

    valid_trades = [
        trade
        for trade in trades
        if get_year(trade) is not None
    ]

    years = sorted(
        set(
            get_year(trade)
            for trade in valid_trades
        )
    )

    validation_years = [
        year
        for year in years
        if year >= 2024
    ]

    total_baseline_r = 0.0
    total_filtered_r = 0.0
    total_delta_r = 0.0

    improved_cases = 0
    eligible_cases = 0

    setup_results = {}

    for validation_year in validation_years:

        print(f"\n{'-' * 90}")
        print(f"VALIDATION YEAR: {validation_year}")
        print(f"{'-' * 90}")

        validation_trades = [
            trade
            for trade in valid_trades
            if get_year(trade) == validation_year
        ]

        groups = sorted(
            set(
                get_setup(trade)
                for trade in validation_trades
            )
        )

        year_baseline_r = 0.0
        year_filtered_r = 0.0

        for setup in groups:

            training_trades = [
                trade
                for trade in valid_trades
                if (
                    get_year(trade) < validation_year
                    and get_setup(trade) == setup
                )
            ]

            validation_setup_trades = [
                trade
                for trade in validation_trades
                if get_setup(trade) == setup
            ]

            if len(training_trades) < MIN_TRAIN_TRADES:
                continue

            if len(validation_setup_trades) == 0:
                continue

            best_name = None
            best_bars = None
            best_threshold = None
            best_training_metrics = None

            for name, bars, threshold in candidates:

                training_metrics = evaluate(
                    training_trades,
                    bars,
                    threshold
                )

                if (
                    training_metrics["trades"]
                    < MIN_SELECTED_TRAIN_TRADES
                ):
                    continue

                if best_training_metrics is None:
                    best_name = name
                    best_bars = bars
                    best_threshold = threshold
                    best_training_metrics = training_metrics
                    continue

                current_key = (
                    training_metrics["total_r"],
                    training_metrics["pf"],
                    training_metrics["trades"],
                )

                best_key = (
                    best_training_metrics["total_r"],
                    best_training_metrics["pf"],
                    best_training_metrics["trades"],
                )

                if current_key > best_key:
                    best_name = name
                    best_bars = bars
                    best_threshold = threshold
                    best_training_metrics = training_metrics

            if best_training_metrics is None:
                continue

            baseline_validation = evaluate(
                validation_setup_trades,
                None,
                None
            )

            filtered_validation = evaluate(
                validation_setup_trades,
                best_bars,
                best_threshold
            )

            delta_r = (
                filtered_validation["total_r"]
                - baseline_validation["total_r"]
            )

            eligible_cases += 1

            year_baseline_r += (
                baseline_validation["total_r"]
            )

            year_filtered_r += (
                filtered_validation["total_r"]
            )

            total_baseline_r += (
                baseline_validation["total_r"]
            )

            total_filtered_r += (
                filtered_validation["total_r"]
            )

            total_delta_r += delta_r

            if delta_r > 0:
                improved_cases += 1

            setup_results.setdefault(
                setup,
                []
            ).append(
                {
                    "year": validation_year,
                    "training_trades": len(training_trades),
                    "validation_trades": len(
                        validation_setup_trades
                    ),
                    "filter": best_name,
                    "baseline_r": (
                        baseline_validation["total_r"]
                    ),
                    "filtered_r": (
                        filtered_validation["total_r"]
                    ),
                    "delta_r": delta_r,
                }
            )

            print(
                f"\n{setup}"
            )

            print(
                f"  Training: "
                f"{len(training_trades)} trades"
            )

            print(
                f"  Validation: "
                f"{len(validation_setup_trades)} trades"
            )

            print(
                f"  Selected filter: "
                f"{best_name}"
            )

            print(
                f"  Training selected: "
                f"{best_training_metrics['trades']} trades | "
                f"Win {best_training_metrics['win_rate']:.2f}% | "
                f"PF {fmt_pf(best_training_metrics['pf'])} | "
                f"R {best_training_metrics['total_r']:+.2f}"
            )

            print(
                f"  Validation baseline: "
                f"{baseline_validation['trades']} trades | "
                f"Win {baseline_validation['win_rate']:.2f}% | "
                f"PF {fmt_pf(baseline_validation['pf'])} | "
                f"R {baseline_validation['total_r']:+.2f}"
            )

            print(
                f"  Validation filtered: "
                f"{filtered_validation['trades']} trades | "
                f"Win {filtered_validation['win_rate']:.2f}% | "
                f"PF {fmt_pf(filtered_validation['pf'])} | "
                f"R {filtered_validation['total_r']:+.2f}"
            )

            print(
                f"  ΔR: {delta_r:+.2f} | Verdict: "
                f"{'IMPROVED' if delta_r > 0 else 'NOT IMPROVED'}"
            )

        print(
            f"\nYEAR {validation_year} SUMMARY"
        )

        print(
            f"  Baseline R: "
            f"{year_baseline_r:+.2f}"
        )

        print(
            f"  Filtered R: "
            f"{year_filtered_r:+.2f}"
        )

        print(
            f"  ΔR: "
            f"{year_filtered_r - year_baseline_r:+.2f}"
        )

    print("\n" + "=" * 90)
    print("SETUP WALK-FORWARD SUMMARY")
    print("=" * 90)

    print(
        f"Eligible setup-years: "
        f"{eligible_cases}"
    )

    print(
        f"Validation baseline total R : "
        f"{total_baseline_r:.2f}"
    )

    print(
        f"Validation filtered total R : "
        f"{total_filtered_r:.2f}"
    )

    print(
        f"Total validation ΔR         : "
        f"{total_delta_r:.2f}"
    )

    print(
        f"Improved setup-years        : "
        f"{improved_cases}/"
        f"{eligible_cases}"
    )

    print()

    print(
        "SETUP-LEVEL RESULTS"
    )

    for setup in sorted(setup_results):

        results = setup_results[setup]

        setup_baseline_r = sum(
            item["baseline_r"]
            for item in results
        )

        setup_filtered_r = sum(
            item["filtered_r"]
            for item in results
        )

        setup_delta_r = (
            setup_filtered_r
            - setup_baseline_r
        )

        improved_years = sum(
            1
            for item in results
            if item["delta_r"] > 0
        )

        print(
            f"{setup:<22} "
            f"Years={len(results):<2} "
            f"BaselineR={setup_baseline_r:>7.2f} "
            f"FilteredR={setup_filtered_r:>7.2f} "
            f"ΔR={setup_delta_r:>7.2f} "
            f"Improved={improved_years}/{len(results)}"
        )

    print()
    print(
        "RESEARCH RULE:"
    )

    if (
        total_filtered_r > total_baseline_r
        and improved_cases >= 2
    ):
        print(
            "PROMISING — setup-level walk-forward "
            "shows improvement across at least 2 "
            "validation cases."
        )
        print(
            "Do NOT deploy yet; continue robustness "
            "testing."
        )
    else:
        print(
            "NOT ROBUST ENOUGH — setup-level "
            "walk-forward does not yet provide "
            "sufficient out-of-sample evidence."
        )
        print(
            "Do NOT modify engine logic."
        )

def setup_trend_early_adverse_walk_forward_diagnostic(trades):
    """
    Walk-forward validation of Early Adverse-R filters at the
    Setup x Entry-Trend level.

    For each validation year:
      1. Use only prior-year data from the same Setup x Trend group.
      2. Select the best Early Adverse-R filter on training data.
      3. Apply that filter to the unseen validation year.
      4. Compare filtered validation R against validation baseline.

    Research focus:
      - Does the Early Adverse-R effect survive out-of-sample
        at the Setup x Trend level?
      - Especially LONG_CONTINUATION x BULLISH.
    """

    print("\n" + "=" * 90)
    print("SETUP × TREND EARLY ADVERSE-R WALK-FORWARD VALIDATION")
    print("=" * 90)

    candidates = [
        ("NO_FILTER", None, None),
        ("1B_0.25R", 1, 0.25),
        ("2B_0.25R", 2, 0.25),
        ("3B_0.25R", 3, 0.25),
        ("4B_0.25R", 4, 0.25),
        ("5B_0.25R", 5, 0.25),
        ("5B_0.35R", 5, 0.35),
        ("5B_0.50R", 5, 0.50),
    ]

    MIN_TRAIN_TRADES = 10
    MIN_SELECTED_TRAIN_TRADES = 10

    def get_setup(trade):
        return (
            trade.get("Diagnostic_Setup")
            or trade.get("setup")
            or "UNKNOWN"
        )

    def get_trend(trade):
        return trade.get("entry_trend") or "UNKNOWN"

    def get_year(trade):
        entry_date = str(trade.get("entry_date", ""))
        try:
            return int(entry_date[:4])
        except (ValueError, TypeError):
            return None

    def get_r(trade):
        value = trade.get("R_multiple")

        if value is None:
            value = trade.get("r_multiple")

        if value is None:
            value = trade.get("R")

        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def get_early_adverse_r(trade):
        value = trade.get("early_adverse_r")

        if value is None:
            value = trade.get("Early_Adverse_R")

        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def passes_filter(trade, bars, threshold):
        if bars is None:
            return True

        adverse_r = get_early_adverse_r(trade)

        # Missing early adverse-R information is retained,
        # matching the existing Early Adverse-R diagnostic logic.
        if adverse_r is None:
            return True

        return adverse_r < threshold

    def evaluate(trade_list, bars, threshold):
        selected = [
            trade
            for trade in trade_list
            if passes_filter(trade, bars, threshold)
        ]

        total_r = sum(get_r(trade) for trade in selected)

        wins = sum(1 for trade in selected if get_r(trade) > 0)

        losses = [
            get_r(trade)
            for trade in selected
            if get_r(trade) < 0
        ]

        winners = [
            get_r(trade)
            for trade in selected
            if get_r(trade) > 0
        ]

        gross_profit = sum(winners)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:
            pf = gross_profit / gross_loss
        elif gross_profit > 0:
            pf = float("inf")
        else:
            pf = 0.0

        win_rate = (
            wins / len(selected) * 100
            if selected
            else 0.0
        )

        return {
            "trades": len(selected),
            "wins": wins,
            "win_rate": win_rate,
            "pf": pf,
            "total_r": total_r,
        }

    def fmt_pf(value):
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"

    valid_trades = [
        trade
        for trade in trades
        if get_year(trade) is not None
    ]

    years = sorted(
        set(get_year(trade) for trade in valid_trades)
    )

    validation_years = [
        year
        for year in years
        if year >= 2024
    ]

    total_baseline_r = 0.0
    total_filtered_r = 0.0
    total_delta_r = 0.0
    improved_cases = 0
    eligible_cases = 0

    lc_bullish_results = []

    for validation_year in validation_years:

        print(f"\n{'-' * 90}")
        print(f"VALIDATION YEAR: {validation_year}")
        print(f"{'-' * 90}")

        validation_trades = [
            trade
            for trade in valid_trades
            if get_year(trade) == validation_year
        ]

        groups = sorted(
            set(
                (
                    get_setup(trade),
                    get_trend(trade)
                )
                for trade in validation_trades
            )
        )

        year_baseline_r = 0.0
        year_filtered_r = 0.0

        for setup, trend in groups:

            training_trades = [
                trade
                for trade in valid_trades
                if (
                    get_year(trade) < validation_year
                    and get_setup(trade) == setup
                    and get_trend(trade) == trend
                )
            ]

            validation_group_trades = [
                trade
                for trade in validation_trades
                if (
                    get_setup(trade) == setup
                    and get_trend(trade) == trend
                )
            ]

            if len(training_trades) < MIN_TRAIN_TRADES:
                continue

            if len(validation_group_trades) == 0:
                continue

            best_name = None
            best_bars = None
            best_threshold = None
            best_training_metrics = None

            for name, bars, threshold in candidates:

                training_metrics = evaluate(
                    training_trades,
                    bars,
                    threshold
                )

                if (
                    training_metrics["trades"]
                    < MIN_SELECTED_TRAIN_TRADES
                ):
                    continue

                if best_training_metrics is None:
                    best_name = name
                    best_bars = bars
                    best_threshold = threshold
                    best_training_metrics = training_metrics
                    continue

                current_key = (
                    training_metrics["total_r"],
                    training_metrics["pf"],
                    training_metrics["trades"],
                )

                best_key = (
                    best_training_metrics["total_r"],
                    best_training_metrics["pf"],
                    best_training_metrics["trades"],
                )

                if current_key > best_key:
                    best_name = name
                    best_bars = bars
                    best_threshold = threshold
                    best_training_metrics = training_metrics

            if best_training_metrics is None:
                continue

            baseline_validation = evaluate(
                validation_group_trades,
                None,
                None
            )

            filtered_validation = evaluate(
                validation_group_trades,
                best_bars,
                best_threshold
            )

            delta_r = (
                filtered_validation["total_r"]
                - baseline_validation["total_r"]
            )

            eligible_cases += 1

            year_baseline_r += baseline_validation["total_r"]
            year_filtered_r += filtered_validation["total_r"]

            total_baseline_r += baseline_validation["total_r"]
            total_filtered_r += filtered_validation["total_r"]
            total_delta_r += delta_r

            if delta_r > 0:
                improved_cases += 1

            print(
                f"\n{setup} × {trend}"
            )

            print(
                f"  Training: {len(training_trades)} trades"
            )

            print(
                f"  Validation: {len(validation_group_trades)} trades"
            )

            print(
                f"  Selected filter: {best_name}"
            )

            print(
                f"  Training selected: "
                f"{best_training_metrics['trades']} trades | "
                f"Win {best_training_metrics['win_rate']:.2f}% | "
                f"PF {fmt_pf(best_training_metrics['pf'])} | "
                f"R {best_training_metrics['total_r']:+.2f}"
            )

            print(
                f"  Validation baseline: "
                f"{baseline_validation['trades']} trades | "
                f"Win {baseline_validation['win_rate']:.2f}% | "
                f"PF {fmt_pf(baseline_validation['pf'])} | "
                f"R {baseline_validation['total_r']:+.2f}"
            )

            print(
                f"  Validation filtered: "
                f"{filtered_validation['trades']} trades | "
                f"Win {filtered_validation['win_rate']:.2f}% | "
                f"PF {fmt_pf(filtered_validation['pf'])} | "
                f"R {filtered_validation['total_r']:+.2f}"
            )

            print(
                f"  ΔR: {delta_r:+.2f} "
                f"| Verdict: "
                f"{'IMPROVED' if delta_r > 0 else 'NOT IMPROVED'}"
            )

            if (
                setup == "LONG_CONTINUATION"
                and trend == "BULLISH"
            ):
                lc_bullish_results.append(
                    {
                        "year": validation_year,
                        "training_trades": len(training_trades),
                        "validation_trades": len(validation_group_trades),
                        "filter": best_name,
                        "baseline_r": baseline_validation["total_r"],
                        "filtered_r": filtered_validation["total_r"],
                        "delta_r": delta_r,
                    }
                )

        print(
            f"\nYEAR {validation_year} SUMMARY"
        )

        print(
            f"  Baseline R: {year_baseline_r:+.2f}"
        )

        print(
            f"  Filtered R: {year_filtered_r:+.2f}"
        )

        print(
            f"  ΔR: "
            f"{year_filtered_r - year_baseline_r:+.2f}"
        )

    print("\n" + "=" * 90)
    print("SETUP × TREND WALK-FORWARD SUMMARY")
    print("=" * 90)

    print(
        f"Eligible group-years: {eligible_cases}"
    )

    print(
        f"Baseline validation total R: "
        f"{total_baseline_r:+.2f}"
    )

    print(
        f"Filtered validation total R: "
        f"{total_filtered_r:+.2f}"
    )

    print(
        f"Total ΔR: "
        f"{total_delta_r:+.2f}"
    )

    print(
        f"Improved group-years: "
        f"{improved_cases}/{eligible_cases}"
    )

    print("\nLONG_CONTINUATION × BULLISH")
    print("-" * 90)

    if lc_bullish_results:
        lc_baseline_r = sum(
            result["baseline_r"]
            for result in lc_bullish_results
        )

        lc_filtered_r = sum(
            result["filtered_r"]
            for result in lc_bullish_results
        )

        lc_delta_r = lc_filtered_r - lc_baseline_r

        print(
            f"Validation years tested: "
            f"{len(lc_bullish_results)}"
        )

        for result in lc_bullish_results:
            print(
                f"  {result['year']}: "
                f"train {result['training_trades']} | "
                f"validation {result['validation_trades']} | "
                f"{result['filter']} | "
                f"baseline {result['baseline_r']:+.2f}R | "
                f"filtered {result['filtered_r']:+.2f}R | "
                f"Δ {result['delta_r']:+.2f}R"
            )

        print(
            f"\n  LC × Bullish baseline total: "
            f"{lc_baseline_r:+.2f}R"
        )

        print(
            f"  LC × Bullish filtered total: "
            f"{lc_filtered_r:+.2f}R"
        )

        print(
            f"  LC × Bullish total ΔR: "
            f"{lc_delta_r:+.2f}R"
        )

    else:
        print(
            "No LONG_CONTINUATION × BULLISH "
            "group-year passed the minimum training requirement."
        )

    print("\nRESEARCH RULE")

    if (
        eligible_cases > 0
        and total_delta_r > 0
        and improved_cases >= max(2, eligible_cases // 2)
    ):
        print(
            "PROMISING — setup × trend walk-forward "
            "shows positive out-of-sample improvement. "
            "Do NOT deploy yet; continue robustness testing."
        )
    else:
        print(
            "NOT ROBUST ENOUGH — setup × trend walk-forward "
            "does not yet provide sufficient out-of-sample evidence. "
            "Do NOT deploy; continue research."
        )

def full_population_regime_comparison_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Compare the COMPLETE trade population in 2022-2025
        against 2026 to determine whether 2026 represents a
        materially different trade environment.

    IMPORTANT:
        - Uses ALL trades.
        - Does NOT optimize any rule.
        - Does NOT create a trading filter.
        - Does NOT modify production engine logic.
        - Uses descriptive population comparison only.
    """

    print()
    print("=" * 120)
    print("FULL TRADE-POPULATION REGIME COMPARISON — 2022-2025 VS 2026")
    print("=" * 120)

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_number(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is None:
                continue

            try:
                return float(value)
            except Exception:
                continue

        return None

    def get_text(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is not None and str(value).strip() != "":
                return str(value)

        return "UNKNOWN"

    def get_bool(trade, *names):
        value = get_text(
            trade,
            *names
        )

        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y"
        }

    def get_setup(trade):
        return get_text(
            trade,
            "Diagnostic_Setup",
            "setup",
            "Setup"
        )

    def get_entry_trend(trade):
        return get_text(
            trade,
            "entry_trend",
            "Entry_Trend"
        )

    def get_structure_bias(trade):
        return get_text(
            trade,
            "structure_bias",
            "Structure_Bias"
        )

    def get_strength_bucket(trade):
        value = get_number(
            trade,
            "entry_trend_strength",
            "Entry_Trend_Strength"
        )

        if value is None:
            return "UNKNOWN"

        if value < 20:
            return "<20"

        if value < 40:
            return "20-40"

        if value < 60:
            return "40-60"

        return "60+"

    def get_slope_bucket(trade):
        value = get_number(
            trade,
            "entry_trend_slope",
            "Entry_Trend_Slope"
        )

        if value is None:
            return "UNKNOWN"

        if value < -0.50:
            return "< -0.50"

        if value < 0:
            return "-0.50 to 0"

        if value < 0.50:
            return "0 to 0.50"

        return ">= 0.50"

    def get_fibonacci_bucket(trade):
        value = get_number(
            trade,
            "entry_price_position_value",
            "Entry_Price_Position_Value",
            "Price_Position_Value"
        )

        if value is None:
            return "UNKNOWN"

        if value < 0:
            return "<0"

        if value < 0.25:
            return "0.00-0.25"

        if value < 0.50:
            return "0.25-0.50"

        if value < 0.75:
            return "0.50-0.75"

        if value <= 1.00:
            return "0.75-1.00"

        return ">1.00"

    def get_mfe_bucket(trade):
        value = get_number(
            trade,
            "max_favorable_r",
            "Max_Favorable_R"
        )

        if value is None:
            return "UNKNOWN"

        if value < 0:
            return "<0R"

        if value < 0.25:
            return "0-0.25R"

        if value < 0.50:
            return "0.25-0.50R"

        if value < 1.00:
            return "0.50-1.00R"

        return ">=1R"

    def get_mae_bucket(trade):
        value = get_number(
            trade,
            "max_adverse_r",
            "Max_Adverse_R"
        )

        if value is None:
            return "UNKNOWN"

        if value < 0.25:
            return "0-0.25R"

        if value < 0.50:
            return "0.25-0.50R"

        if value < 1.00:
            return "0.50-1.00R"

        return ">=1R"

    def get_early_adverse_trigger(trade):
        value = get_number(
            trade,
            "early_adverse_r_bar_3",
            "Early_Adverse_R_Bar_3"
        )

        if value is None:
            return "UNKNOWN"

        return (
            "TRIGGERED"
            if value >= 0.25
            else "NOT_TRIGGERED"
        )

    def performance_stats(group):
        if not group:
            return {
                "n": 0,
                "share": 0.0,
                "win_rate": 0.0,
                "pf": 0.0,
                "r": 0.0,
                "avg_r": 0.0
            }

        r_values = [
            get_number(
                trade,
                "r_multiple",
                "R_Multiple",
                "R"
            )
            for trade in group
        ]

        r_values = [
            value
            for value in r_values
            if value is not None
        ]

        if not r_values:
            return {
                "n": len(group),
                "share": 0.0,
                "win_rate": 0.0,
                "pf": 0.0,
                "r": 0.0,
                "avg_r": 0.0
            }

        wins = [
            value
            for value in r_values
            if value > 0
        ]

        losses = [
            value
            for value in r_values
            if value <= 0
        ]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        total_r = sum(r_values)

        return {
            "n": len(r_values),
            "share": 0.0,
            "win_rate": (
                len(wins)
                / len(r_values)
                * 100
            ),
            "pf": profit_factor,
            "r": total_r,
            "avg_r": (
                total_r
                / len(r_values)
            )
        }

    # --------------------------------------------------------------
    # PERIOD SPLIT
    # --------------------------------------------------------------

    earlier = [
        trade
        for trade in trades
        if get_year(trade) in {
            2022,
            2023,
            2024,
            2025
        }
    ]

    current = [
        trade
        for trade in trades
        if get_year(trade) == 2026
    ]

    print()
    print("-" * 120)
    print("PERIOD SUMMARY")
    print("-" * 120)

    for label, group in [
        ("2022-2025", earlier),
        ("2026", current)
    ]:

        stats = performance_stats(group)

        print()
        print(label)
        print(
            f"Trades={stats['n']:>4} "
            f"Win={stats['win_rate']:>6.2f}% "
            f"PF={stats['pf']:>5.2f} "
            f"R={stats['r']:>8.2f} "
            f"AvgR={stats['avg_r']:>7.3f}"
        )

    # --------------------------------------------------------------
    # POPULATION DIMENSION COMPARISON
    # --------------------------------------------------------------

    dimensions = [
        (
            "SETUP",
            get_setup
        ),
        (
            "ENTRY TREND",
            get_entry_trend
        ),
        (
            "TREND STRENGTH",
            get_strength_bucket
        ),
        (
            "TREND SLOPE",
            get_slope_bucket
        ),
        (
            "STRUCTURE BIAS",
            get_structure_bias
        ),
        (
            "LIQUIDITY SWEEP",
            lambda trade: (
                "SWEEP"
                if get_bool(
                    trade,
                    "liquidity_sweep",
                    "Liquidity_Sweep"
                )
                else "NO_SWEEP"
            )
        ),
        (
            "FIBONACCI ENTRY POSITION",
            get_fibonacci_bucket
        ),
        (
            "MFE",
            get_mfe_bucket
        ),
        (
            "MAE",
            get_mae_bucket
        ),
        (
            "3B_0.25R EARLY-ADVERSE",
            get_early_adverse_trigger
        )
    ]

    for title, getter in dimensions:

        print()
        print("=" * 120)
        print(title)
        print("=" * 120)

        earlier_groups = defaultdict(list)
        current_groups = defaultdict(list)

        for trade in earlier:
            earlier_groups[
                getter(trade)
            ].append(trade)

        for trade in current:
            current_groups[
                getter(trade)
            ].append(trade)

        all_groups = sorted(
            set(earlier_groups.keys())
            | set(current_groups.keys())
        )

        print()
        print(
            f"{'GROUP':28}"
            f"{'2022-25 N':>12}"
            f"{'2022-25 %':>12}"
            f"{'2026 N':>10}"
            f"{'2026 %':>10}"
            f"{'SHARE Δ':>12}"
            f"{'2022-25 R':>13}"
            f"{'2026 R':>11}"
        )

        print("-" * 110)

        for group_name in all_groups:

            earlier_group = earlier_groups.get(
                group_name,
                []
            )

            current_group = current_groups.get(
                group_name,
                []
            )

            earlier_stats = performance_stats(
                earlier_group
            )

            current_stats = performance_stats(
                current_group
            )

            earlier_share = (
                len(earlier_group)
                / len(earlier)
                * 100
                if earlier
                else 0.0
            )

            current_share = (
                len(current_group)
                / len(current)
                * 100
                if current
                else 0.0
            )

            share_delta = (
                current_share
                - earlier_share
            )

            print(
                f"{str(group_name)[:28]:28}"
                f"{len(earlier_group):12d}"
                f"{earlier_share:12.2f}"
                f"{len(current_group):10d}"
                f"{current_share:10.2f}"
                f"{share_delta:12.2f}"
                f"{earlier_stats['r']:13.2f}"
                f"{current_stats['r']:11.2f}"
            )

    # --------------------------------------------------------------
    # AVERAGE MFE / MAE / EARLY ADVERSE
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("AVERAGE TRADE PATH COMPARISON")
    print("=" * 120)

    def average_metric(group, names):
        values = [
            get_number(
                trade,
                *names
            )
            for trade in group
        ]

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return None

        return sum(values) / len(values)

    metrics = [
        (
            "MFE",
            (
                "max_favorable_r",
                "Max_Favorable_R"
            )
        ),
        (
            "MAE",
            (
                "max_adverse_r",
                "Max_Adverse_R"
            )
        ),
        (
            "Early Adverse Bar 1",
            (
                "early_adverse_r_bar_1",
                "Early_Adverse_R_Bar_1"
            )
        ),
        (
            "Early Adverse Bar 3",
            (
                "early_adverse_r_bar_3",
                "Early_Adverse_R_Bar_3"
            )
        ),
        (
            "Early Adverse Bar 5",
            (
                "early_adverse_r_bar_5",
                "Early_Adverse_R_Bar_5"
            )
        )
    ]

    print()
    print(
        f"{'METRIC':30}"
        f"{'2022-25':>15}"
        f"{'2026':>15}"
        f"{'CHANGE':>15}"
    )

    print("-" * 80)

    for label, names in metrics:

        earlier_value = average_metric(
            earlier,
            names
        )

        current_value = average_metric(
            current,
            names
        )

        if (
            earlier_value is None
            or current_value is None
        ):
            print(
                f"{label:<30}"
                f"{'N/A':>15}"
                f"{'N/A':>15}"
                f"{'N/A':>15}"
            )
            continue

        change = (
            current_value
            - earlier_value
        )

        print(
            f"{label:<30}"
            f"{earlier_value:>15.3f}"
            f"{current_value:>15.3f}"
            f"{change:>15.3f}"
        )

    # --------------------------------------------------------------
    # SYMBOL DISTRIBUTION
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("SYMBOL DISTRIBUTION")
    print("=" * 120)

    symbol_groups_earlier = defaultdict(list)
    symbol_groups_current = defaultdict(list)

    for trade in earlier:

        symbol = get_text(
            trade,
            "_symbol",
            "symbol",
            "Symbol",
            "ticker",
            "Ticker"
        )

        symbol_groups_earlier[
            symbol
        ].append(trade)

    for trade in current:

        symbol = get_text(
            trade,
            "_symbol",
            "symbol",
            "Symbol",
            "ticker",
            "Ticker"
        )

        symbol_groups_current[
            symbol
        ].append(trade)

    symbols = sorted(
        set(symbol_groups_earlier.keys())
        | set(symbol_groups_current.keys())
    )

    print()
    print(
        f"{'SYMBOL':25}"
        f"{'2022-25 N':>12}"
        f"{'2022-25 %':>12}"
        f"{'2026 N':>10}"
        f"{'2026 %':>10}"
    )

    print("-" * 75)

    for symbol in symbols:

        earlier_count = len(
            symbol_groups_earlier.get(
                symbol,
                []
            )
        )

        current_count = len(
            symbol_groups_current.get(
                symbol,
                []
            )
        )

        earlier_share = (
            earlier_count
            / len(earlier)
            * 100
            if earlier
            else 0.0
        )

        current_share = (
            current_count
            / len(current)
            * 100
            if current
            else 0.0
        )

        print(
            f"{str(symbol)[:25]:25}"
            f"{earlier_count:12d}"
            f"{earlier_share:12.2f}"
            f"{current_count:10d}"
            f"{current_share:10.2f}"
        )

    # --------------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("REGIME COMPARISON INTERPRETATION")
    print("=" * 120)

    print()
    print(
        "This diagnostic compares the complete trade population "
        "rather than optimizing or filtering individual subgroups."
    )

    print(
        "The key question is whether the 2026 trade population "
        "has materially different composition or trade-path behavior "
        "from 2022-2025."
    )

    print()
    print(
        "Large SHARE Δ values indicate population-composition changes."
    )

    print(
        "Changes in MFE, MAE, and early-adverse metrics indicate "
        "changes in trade-path behavior."
    )

    print()
    print(
        "IMPORTANT: 2026 currently contains a small sample, so "
        "population differences must be treated as preliminary evidence."
    )

    print(
        "Do NOT create subgroup-specific production rules from this "
        "diagnostic."
    )

    print()
    print("RESEARCH STATUS: DIAGNOSTIC ONLY")
    print("NO PRODUCTION ENGINE CHANGES")
    print("=" * 120)

def stock_setup_regime_comparison_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Compare stock × setup performance between the
        earlier period (2022-2025) and 2026.

    IMPORTANT:
        - Uses descriptive population comparison only.
        - Does NOT optimize any rule.
        - Does NOT create a trading filter.
        - Does NOT modify production engine logic.
        - 2026 subgroup samples may be very small.
    """

    from collections import defaultdict

    print()
    print("=" * 140)
    print("STOCK × SETUP × REGIME COMPARISON — 2022-2025 VS 2026")
    print("=" * 140)

    print()
    print(
        "Goal: determine whether the 2026 regime improvement is "
        "explained by stock/setup composition or by improvement "
        "within the same stock/setup combinations."
    )

    print()
    print(
        "IMPORTANT: 2026 subgroup samples are small. "
        "This is descriptive research evidence only."
    )

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_number(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is None:
                continue

            try:
                return float(value)
            except Exception:
                continue

        return None

    def get_text(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is not None and str(value).strip() != "":
                return str(value)

        return "UNKNOWN"

    def get_symbol(trade):
        return get_text(
            trade,
            "_symbol",
            "symbol",
            "Symbol",
            "ticker",
            "Ticker"
        )

    def get_setup(trade):
        return get_text(
            trade,
            "Diagnostic_Setup",
            "setup",
            "Setup"
        )

    def get_r(trade):
        value = get_number(
            trade,
            "r_multiple",
            "R_Multiple"
        )

        return value if value is not None else 0.0

    def calculate_metrics(group):
        if not group:
            return {
                "n": 0,
                "wins": 0,
                "win_pct": None,
                "pf": None,
                "total_r": 0.0,
                "avg_r": None,
            }

        r_values = [
            get_r(trade)
            for trade in group
        ]

        wins = [
            r
            for r in r_values
            if r > 0
        ]

        losses = [
            r
            for r in r_values
            if r < 0
        ]

        total_r = sum(r_values)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:
            pf = gross_profit / gross_loss
        else:
            pf = None

        return {
            "n": len(group),
            "wins": len(wins),
            "win_pct": (
                len(wins) / len(group) * 100
            ),
            "pf": pf,
            "total_r": total_r,
            "avg_r": total_r / len(group),
        }

    earlier = []
    current = []

    for trade in trades:
        year = get_year(trade)

        if year is None:
            continue

        if 2022 <= year <= 2025:
            earlier.append(trade)

        elif year == 2026:
            current.append(trade)

    earlier_groups = defaultdict(list)
    current_groups = defaultdict(list)

    for trade in earlier:
        key = (
            get_symbol(trade),
            get_setup(trade)
        )

        earlier_groups[key].append(trade)

    for trade in current:
        key = (
            get_symbol(trade),
            get_setup(trade)
        )

        current_groups[key].append(trade)

    keys = sorted(
        set(earlier_groups.keys())
        | set(current_groups.keys())
    )

    print()
    print("=" * 140)
    print("STOCK × SETUP PERFORMANCE")
    print("=" * 140)

    print()
    print(
        f"{'SYMBOL':18}"
        f"{'SETUP':22}"
        f"{'22-25 N':>9}"
        f"{'22-25 %':>10}"
        f"{'22-25 Win%':>12}"
        f"{'22-25 PF':>11}"
        f"{'22-25 R':>11}"
        f"{'26 N':>8}"
        f"{'26 %':>9}"
        f"{'26 Win%':>11}"
        f"{'26 PF':>10}"
        f"{'26 R':>11}"
        f"{'ΔR':>11}"
    )

    print("-" * 140)

    for symbol, setup in keys:

        earlier_group = earlier_groups.get(
            (symbol, setup),
            []
        )

        current_group = current_groups.get(
            (symbol, setup),
            []
        )

        earlier_metrics = calculate_metrics(
            earlier_group
        )

        current_metrics = calculate_metrics(
            current_group
        )

        earlier_share = (
            earlier_metrics["n"]
            / len(earlier)
            * 100
            if earlier
            else 0.0
        )

        current_share = (
            current_metrics["n"]
            / len(current)
            * 100
            if current
            else 0.0
        )

        delta_r = (
            current_metrics["total_r"]
            - earlier_metrics["total_r"]
        )

        earlier_pf = (
            f"{earlier_metrics['pf']:.2f}"
            if earlier_metrics["pf"] is not None
            else "N/A"
        )

        current_pf = (
            f"{current_metrics['pf']:.2f}"
            if current_metrics["pf"] is not None
            else "N/A"
        )

        earlier_win = (
            f"{earlier_metrics['win_pct']:.2f}"
            if earlier_metrics["win_pct"] is not None
            else "N/A"
        )

        current_win = (
            f"{current_metrics['win_pct']:.2f}"
            if current_metrics["win_pct"] is not None
            else "N/A"
        )

        print(
            f"{symbol[:18]:18}"
            f"{setup[:22]:22}"
            f"{earlier_metrics['n']:9d}"
            f"{earlier_share:10.2f}"
            f"{earlier_win:>12}"
            f"{earlier_pf:>11}"
            f"{earlier_metrics['total_r']:11.2f}"
            f"{current_metrics['n']:8d}"
            f"{current_share:9.2f}"
            f"{current_win:>11}"
            f"{current_pf:>10}"
            f"{current_metrics['total_r']:11.2f}"
            f"{delta_r:11.2f}"
        )

    print()
    print("=" * 140)
    print("2026 ACTIVE STOCK × SETUP GROUPS")
    print("=" * 140)

    active_groups = [
        key
        for key in keys
        if len(current_groups.get(key, [])) > 0
    ]

    if active_groups:

        print()

        for symbol, setup in active_groups:

            group = current_groups[
                (symbol, setup)
            ]

            metrics = calculate_metrics(group)

            print(
                f"{symbol} × {setup}: "
                f"N={metrics['n']} | "
                f"Win={metrics['win_pct']:.2f}% | "
                f"PF="
                f"{metrics['pf']:.2f}"
                if metrics["pf"] is not None
                else
                f"{symbol} × {setup}: "
                f"N={metrics['n']} | "
                f"Win={metrics['win_pct']:.2f}% | "
                f"PF=N/A",
                end=""
            )

            print(
                f" | R={metrics['total_r']:+.2f} | "
                f"AvgR={metrics['avg_r']:+.3f}"
            )

    else:
        print("No stock × setup groups contain 2026 trades.")

    print()
    print("=" * 140)
    print("STOCK × SETUP × REGIME INTERPRETATION")
    print("=" * 140)

    positive_delta_groups = 0
    negative_delta_groups = 0
    active_2026_groups = 0

    for symbol, setup in keys:

        earlier_group = earlier_groups.get(
            (symbol, setup),
            []
        )

        current_group = current_groups.get(
            (symbol, setup),
            []
        )

        if not current_group:
            continue

        active_2026_groups += 1

        earlier_r = sum(
            get_r(trade)
            for trade in earlier_group
        )

        current_r = sum(
            get_r(trade)
            for trade in current_group
        )

        delta_r = current_r - earlier_r

        if delta_r > 0:
            positive_delta_groups += 1

        elif delta_r < 0:
            negative_delta_groups += 1

    print()
    print(
        f"Active stock × setup groups in 2026: "
        f"{active_2026_groups}"
    )

    print(
        f"Groups with positive 2026-vs-earlier ΔR: "
        f"{positive_delta_groups}"
    )

    print(
        f"Groups with negative 2026-vs-earlier ΔR: "
        f"{negative_delta_groups}"
    )

    print()
    print(
        "INTERPRETATION:"
    )

    print(
        "Use this diagnostic to determine whether the "
        "2026 improvement is concentrated in particular "
        "stock/setup combinations or is broadly distributed."
    )

    print()
    print(
        "A positive ΔR does NOT automatically mean the "
        "stock/setup is profitable in 2026."
    )

    print(
        "It means its 2026 total R is better than its "
        "2022-2025 total-R baseline."
    )

    print()
    print(
        "IMPORTANT: groups with very small 2026 N should "
        "not be treated as reliable evidence."
    )

    print(
        "NO PRODUCTION ENGINE CHANGES."
    )

    print("=" * 140)

def stock_setup_early_adverse_regime_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Determine whether the frozen 3B_0.25R Early-Adverse-R
        behavior differs across STOCK × SETUP combinations
        between 2022-2025 and 2026.

    IMPORTANT:
        - Uses the already-frozen 3B_0.25R rule.
        - No subgroup-specific threshold optimization.
        - No production engine changes.
        - Descriptive research only.
        - 2026 subgroup samples are expected to be small.
    """

    print()
    print("=" * 140)
    print("STOCK × SETUP × EARLY-ADVERSE-R REGIME COMPARISON — 2022-2025 VS 2026")
    print("=" * 140)

    WINDOW = 3
    THRESHOLD = 0.25

    print()
    print(
        "Goal: determine whether early-adverse behavior differs within "
        "the same stock/setup combinations across regimes."
    )
    print(
        "Frozen rule: 3 bars / 0.25R"
    )
    print(
        "IMPORTANT: no subgroup-specific optimization is performed."
    )

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_symbol(trade):
        value = trade.get("_symbol")

        if value is None:
            value = trade.get("symbol")

        if value is None:
            value = trade.get("Symbol")

        if value is None:
            value = trade.get("ticker")

        if value is None:
            value = trade.get("Ticker")

        if value is None or str(value).strip() == "":
            return "UNKNOWN"

        return str(value).strip()

    def get_setup(trade):
        value = trade.get("Diagnostic_Setup")

        if value is None:
            value = trade.get("setup")

        if value is None:
            value = trade.get("Setup")

        if value is None or str(value).strip() == "":
            return "UNKNOWN"

        return str(value).strip()

    def get_r(trade):
        value = trade.get("r_multiple")

        if value is None:
            value = trade.get("R_Multiple")

        if value is None:
            value = trade.get("R")

        try:
            return float(value)
        except Exception:
            return 0.0

    def get_early_adverse(trade):
        field = f"early_adverse_r_bar_{WINDOW}"

        value = trade.get(field)

        if value is None:
            value = trade.get(
                field.replace(
                    "early_",
                    "Early_"
                ).replace(
                    "_r_",
                    "_R_"
                )
            )

        try:
            return float(value)
        except Exception:
            return None

    def triggered(trade):
        value = get_early_adverse(trade)

        return (
            value is not None
            and value >= THRESHOLD
        )

    def calculate_stats(group):
        total_r = sum(
            get_r(trade)
            for trade in group
        )

        triggered_count = sum(
            1
            for trade in group
            if triggered(trade)
        )

        trigger_rate = (
            triggered_count
            / len(group)
            * 100
            if group
            else 0.0
        )

        triggered_r = sum(
            get_r(trade)
            for trade in group
            if triggered(trade)
        )

        non_triggered_r = sum(
            get_r(trade)
            for trade in group
            if not triggered(trade)
        )

        avg_r = (
            total_r / len(group)
            if group
            else 0.0
        )

        return {
            "n": len(group),
            "total_r": total_r,
            "avg_r": avg_r,
            "triggered": triggered_count,
            "trigger_rate": trigger_rate,
            "triggered_r": triggered_r,
            "non_triggered_r": non_triggered_r,
        }

    # --------------------------------------------------------------
    # PERIODS
    # --------------------------------------------------------------

    earlier = []
    current = []

    for trade in trades:
        year = get_year(trade)

        if year is None:
            continue

        if 2022 <= year <= 2025:
            earlier.append(trade)

        elif year == 2026:
            current.append(trade)

    # --------------------------------------------------------------
    # STOCK × SETUP GROUPS
    # --------------------------------------------------------------

    groups = {}

    for trade in earlier:
        key = (
            get_symbol(trade),
            get_setup(trade)
        )

        groups.setdefault(key, {
            "earlier": [],
            "current": []
        })

        groups[key]["earlier"].append(trade)

    for trade in current:
        key = (
            get_symbol(trade),
            get_setup(trade)
        )

        groups.setdefault(key, {
            "earlier": [],
            "current": []
        })

        groups[key]["current"].append(trade)

    # --------------------------------------------------------------
    # MAIN TABLE
    # --------------------------------------------------------------

    print()
    print("-" * 160)
    print("STOCK × SETUP EARLY-ADVERSE-R BEHAVIOR")
    print("-" * 160)

    print(
        f"{'SYMBOL':18}"
        f"{'SETUP':22}"
        f"{'22-25 N':>9}"
        f"{'22-25 TRIG':>12}"
        f"{'22-25 RATE':>12}"
        f"{'22-25 R':>12}"
        f"{'26 N':>8}"
        f"{'26 TRIG':>10}"
        f"{'26 RATE':>11}"
        f"{'26 R':>12}"
        f"{'Δ RATE':>11}"
        f"{'Δ R':>11}"
    )

    print("-" * 160)

    for (symbol, setup), data in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1]
        )
    ):

        earlier_stats = calculate_stats(
            data["earlier"]
        )

        current_stats = calculate_stats(
            data["current"]
        )

        delta_rate = (
            current_stats["trigger_rate"]
            - earlier_stats["trigger_rate"]
            if current_stats["n"] > 0
            and earlier_stats["n"] > 0
            else None
        )

        delta_r = (
            current_stats["total_r"]
            - (
                earlier_stats["total_r"]
                / earlier_stats["n"]
                * current_stats["n"]
            )
            if current_stats["n"] > 0
            and earlier_stats["n"] > 0
            else None
        )

        earlier_rate = (
            f"{earlier_stats['trigger_rate']:>11.2f}"
            if earlier_stats["n"] > 0
            else f"{'N/A':>11}"
        )

        current_rate = (
            f"{current_stats['trigger_rate']:>10.2f}"
            if current_stats["n"] > 0
            else f"{'N/A':>10}"
        )

        delta_rate_text = (
            f"{delta_rate:>10.2f}"
            if delta_rate is not None
            else f"{'N/A':>10}"
        )

        delta_r_text = (
            f"{delta_r:>10.2f}"
            if delta_r is not None
            else f"{'N/A':>10}"
        )

        print(
            f"{symbol[:18]:18}"
            f"{setup[:22]:22}"
            f"{earlier_stats['n']:9d}"
            f"{earlier_stats['triggered']:12d}"
            f"{earlier_rate}"
            f"{earlier_stats['total_r']:12.2f}"
            f"{current_stats['n']:8d}"
            f"{current_stats['triggered']:10d}"
            f"{current_rate}"
            f"{current_stats['total_r']:12.2f}"
            f"{delta_rate_text:>11}"
            f"{delta_r_text:>11}"
        )

    # --------------------------------------------------------------
    # 2026 ACTIVE GROUPS
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("2026 ACTIVE STOCK × SETUP GROUPS")
    print("-" * 120)

    active_groups = []

    for (symbol, setup), data in groups.items():

        if not data["current"]:
            continue

        stats = calculate_stats(
            data["current"]
        )

        active_groups.append(
            (
                symbol,
                setup,
                stats
            )
        )

    for symbol, setup, stats in sorted(
        active_groups,
        key=lambda item: (
            item[0],
            item[1]
        )
    ):

        print(
            f"{symbol:18} × "
            f"{setup:22} | "
            f"N={stats['n']} | "
            f"Triggered={stats['triggered']} | "
            f"TriggerRate={stats['trigger_rate']:.2f}% | "
            f"R={stats['total_r']:+.2f} | "
            f"AvgR={stats['avg_r']:+.3f}"
        )

    # --------------------------------------------------------------
    # 2026 TRIGGER DISTRIBUTION
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("2026 EARLY-ADVERSE-R TRIGGER DISTRIBUTION")
    print("-" * 120)

    total_2026 = len(current)

    triggered_2026 = sum(
        1
        for trade in current
        if triggered(trade)
    )

    print(
        f"2026 trades:        {total_2026}"
    )

    print(
        f"2026 triggered:     {triggered_2026}"
    )

    print(
        f"2026 trigger rate:  "
        f"{(
            triggered_2026 / total_2026 * 100
            if total_2026
            else 0.0
        ):.2f}%"
    )

    # --------------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("RESEARCH INTERPRETATION")
    print("=" * 120)

    print()

    if triggered_2026 == 0:
        print(
            "OBSERVATION: 2026 contains no trades triggering "
            "the frozen 3B_0.25R condition."
        )

        print(
            "Therefore, the frozen early-adverse rule has very "
            "little opportunity to explain the 2026 improvement."
        )

    else:
        print(
            "OBSERVATION: 2026 contains trades triggering the "
            "frozen 3B_0.25R condition."
        )

        print(
            "The stock × setup breakdown should be used to determine "
            "whether those triggers are concentrated in specific groups."
        )

    print()

    print(
        "IMPORTANT: A positive or negative ΔR here is descriptive."
    )

    print(
        "It does NOT justify creating stock-specific or setup-specific "
        "early-adverse thresholds."
    )

    print()

    print(
        "RESEARCH STATUS: DIAGNOSTIC ONLY"
    )

    print(
        "NO PRODUCTION ENGINE CHANGES"
    )

    print("=" * 120)

def stock_setup_early_adverse_mfe_final_r_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Determine whether Early-Adverse-R behavior is associated
        with subsequent MFE and final R within the same
        STOCK × SETUP combinations.

    Frozen rule:
        3 bars / 0.25R

    IMPORTANT:
        - No subgroup-specific optimization.
        - No production engine changes.
        - Descriptive research only.
        - 2026 subgroup samples may be very small.
    """

    print()
    print("=" * 140)
    print(
        "STOCK × SETUP × EARLY-ADVERSE-R × MFE × FINAL-R DIAGNOSTIC"
    )
    print("=" * 140)

    WINDOW = 3
    THRESHOLD = 0.25

    print()
    print(
        "Goal: compare triggered vs non-triggered Early-Adverse-R "
        "trades within the same stock/setup combination."
    )

    print(
        "Frozen rule: 3 bars / 0.25R"
    )

    print(
        "Metrics: N, Avg Early Adverse-R, Avg MFE, Win%, Avg Final R, Total R"
    )

    print(
        "NO PRODUCTION ENGINE CHANGES."
    )

    print("=" * 140)

    groups = {}

    for trade in trades:

        entry_date = trade.get(
            "entry_date",
            trade.get("Entry_Date")
        )

        if not entry_date:
            continue

        try:
            year = int(str(entry_date)[:4])
        except (TypeError, ValueError):
            continue

        if year < 2022:
            continue

        if year >= 2026:
            regime = "2026"
        else:
            regime = "2022-2025"

        symbol = trade.get(
            "_symbol",
            trade.get(
                "symbol",
                trade.get(
                    "Symbol",
                    trade.get(
                        "ticker",
                        trade.get(
                            "Ticker",
                            "UNKNOWN"
                        )
                    )
                )
            )
        )

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                trade.get(
                    "Setup",
                    "UNKNOWN"
                )
            )
        )

        if symbol is None or str(symbol).strip() == "":
            symbol = "UNKNOWN"

        if setup is None or str(setup).strip() == "":
            setup = "UNKNOWN"

        key = (
            regime,
            str(symbol),
            str(setup)
        )

        groups.setdefault(
            key,
            []
        ).append(trade)

    print()

    for regime in (
        "2022-2025",
        "2026"
    ):

        print()
        print(
            f"REGIME: {regime}"
        )

        print("-" * 140)

        regime_groups = {
            key: value
            for key, value in groups.items()
            if key[0] == regime
        }

        for (
            _,
            symbol,
            setup
        ), group_trades in sorted(
            regime_groups.items()
        ):

            triggered = []
            non_triggered = []

            for trade in group_trades:

                early_adverse = trade.get(
                    "early_adverse_r_bar_3"
                )

                try:
                    early_adverse = float(
                        early_adverse
                    )
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if early_adverse >= THRESHOLD:
                    triggered.append(
                        trade
                    )
                else:
                    non_triggered.append(
                        trade
                    )

            print()
            print(
                f"{symbol} × {setup}"
            )

            print("-" * 140)

            for label, bucket in (
                (
                    "TRIGGERED",
                    triggered
                ),
                (
                    "NON-TRIGGERED",
                    non_triggered
                )
            ):

                if not bucket:
                    print(
                        f"{label:<14} "
                        f"N=0"
                    )
                    continue

                early_adverse_values = []
                mfe_values = []
                final_r_values = []

                wins = 0

                for trade in bucket:

                    early_adverse = trade.get(
                        "early_adverse_r_bar_3"
                    )

                    try:
                        early_adverse = float(
                            early_adverse
                        )
                    except (
                        TypeError,
                        ValueError
                    ):
                        continue

                    mfe = trade.get(
                        "max_favorable_r"
                    )

                    try:
                        mfe = float(
                            mfe
                        )
                    except (
                        TypeError,
                        ValueError
                    ):
                        mfe = None

                    final_r = trade.get(
                        "r_multiple"
                    )

                    try:
                        final_r = float(
                            final_r
                        )
                    except (
                        TypeError,
                        ValueError
                    ):
                        final_r = None

                    early_adverse_values.append(
                        early_adverse
                    )

                    if mfe is not None:
                        mfe_values.append(
                            mfe
                        )

                    if final_r is not None:
                        final_r_values.append(
                            final_r
                        )

                        if final_r > 0:
                            wins += 1

                if not final_r_values:
                    print(
                        f"{label:<14} "
                        f"N={len(bucket):>3} "
                        f"AvgEarly=N/A "
                        f"AvgMFE=N/A "
                        f"Win%=N/A "
                        f"AvgFinalR=N/A "
                        f"TotalR=N/A"
                    )
                    continue

                avg_early_adverse = (
                    sum(early_adverse_values)
                    / len(early_adverse_values)
                    if early_adverse_values
                    else 0.0
                )

                avg_mfe = (
                    sum(mfe_values)
                    / len(mfe_values)
                    if mfe_values
                    else 0.0
                )

                win_rate = (
                    wins
                    / len(final_r_values)
                ) * 100

                avg_final_r = (
                    sum(final_r_values)
                    / len(final_r_values)
                )

                total_r = sum(
                    final_r_values
                )

                print(
                    f"{label:<14} "
                    f"N={len(bucket):>3} "
                    f"AvgEarly={avg_early_adverse:>7.3f}R "
                    f"AvgMFE={avg_mfe:>7.3f}R "
                    f"Win%={win_rate:>6.2f}% "
                    f"AvgFinalR={avg_final_r:>8.3f}R "
                    f"TotalR={total_r:>8.3f}R"
                )

    print()
    print("=" * 140)
    print(
        "INTERPRETATION:"
    )
    print(
        "This diagnostic is descriptive only."
    )
    print(
        "It is intended to show whether trades experiencing the "
        "frozen Early-Adverse-R condition subsequently recover "
        "through MFE or continue toward poor final R."
    )
    print(
        "Small subgroup samples, especially in 2026, should not "
        "be treated as reliable evidence."
    )
    print(
        "NO PRODUCTION ENGINE CHANGES."
    )
    print("=" * 140)

def regime_break_early_adverse_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Identify where the frozen 3B_0.25R Early-Adverse-R rule
        behaves differently in 2026 versus the earlier validation
        period.

    IMPORTANT:
        - Does not optimize any rule.
        - Does not modify trades.
        - Does not modify the production engine.
        - Uses the already-frozen 3B_0.25R rule only.
    """

    print()
    print("=" * 100)
    print("REGIME-BREAK EARLY-ADVERSE-R DIAGNOSTIC")
    print("=" * 100)

    WINDOW = 3
    THRESHOLD = 0.25

    print()
    print("FROZEN RULE: 3B_0.25R")
    print("Window:", WINDOW)
    print("Threshold:", THRESHOLD)
    print("No subgroup-specific optimization is performed.")

    def get_year(trade):
        date_value = trade.get("entry_date")

        if date_value is None:
            date_value = trade.get("Entry_Date")

        if date_value is None:
            return None

        try:
            return int(str(date_value)[:4])
        except Exception:
            return None

    def get_r_multiple(trade):
        value = trade.get("r_multiple")

        if value is None:
            value = trade.get("R_Multiple")

        if value is None:
            value = trade.get("R")

        try:
            return float(value)
        except Exception:
            return 0.0

    def get_early_adverse(trade):
        field = f"early_adverse_r_bar_{WINDOW}"

        value = trade.get(field)

        if value is None:
            value = trade.get(
                field.replace("early_", "Early_").replace("_r_", "_R_")
            )

        try:
            return float(value)
        except Exception:
            return None

    def counterfactual_r(trade):
        baseline_r = get_r_multiple(trade)
        early_adverse = get_early_adverse(trade)

        if early_adverse is not None and early_adverse >= THRESHOLD:
            return -THRESHOLD

        return baseline_r

    def get_field(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is not None and str(value).strip() != "":
                return value

        return "UNKNOWN"

    def calculate_group(group):
        baseline = sum(get_r_multiple(t) for t in group)
        filtered = sum(counterfactual_r(t) for t in group)
        delta = filtered - baseline

        exits = sum(
            1
            for t in group
            if (
                get_early_adverse(t) is not None
                and get_early_adverse(t) >= THRESHOLD
            )
        )

        return baseline, filtered, delta, exits

    def print_dimension(title, field_names):
        print()
        print("-" * 100)
        print(title)
        print("-" * 100)

        groups = {}

        for trade in trades:
            value = get_field(trade, *field_names)
            value = str(value)

            groups.setdefault(value, []).append(trade)

        header = (
            f"{'GROUP':30}"
            f"{'N':>6}"
            f"{'BASE R':>12}"
            f"{'FILTERED R':>14}"
            f"{'DELTA R':>12}"
            f"{'EXITS':>10}"
        )

        print(header)
        print("-" * 100)

        for group_name, group_trades in sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0])
        ):
            baseline, filtered, delta, exits = calculate_group(group_trades)

            print(
                f"{group_name[:30]:30}"
                f"{len(group_trades):6d}"
                f"{baseline:12.2f}"
                f"{filtered:14.2f}"
                f"{delta:12.2f}"
                f"{exits:10d}"
            )

    # ------------------------------------------------------------------
    # YEAR COMPARISON
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("YEAR COMPARISON")
    print("-" * 100)

    years = {}

    for trade in trades:
        year = get_year(trade)

        if year is not None:
            years.setdefault(year, []).append(trade)

    print(
        f"{'YEAR':10}"
        f"{'N':>8}"
        f"{'BASE R':>12}"
        f"{'FILTERED R':>14}"
        f"{'DELTA R':>12}"
        f"{'EXITS':>10}"
    )

    print("-" * 100)

    for year in sorted(years):
        group = years[year]

        baseline, filtered, delta, exits = calculate_group(group)

        print(
            f"{year:<10}"
            f"{len(group):8d}"
            f"{baseline:12.2f}"
            f"{filtered:14.2f}"
            f"{delta:12.2f}"
            f"{exits:10d}"
        )

    # ------------------------------------------------------------------
    # EARLIER PERIOD VS 2026
    # ------------------------------------------------------------------

    earlier_trades = [
        t
        for t in trades
        if get_year(t) is not None and get_year(t) < 2026
    ]

    trades_2026 = [
        t
        for t in trades
        if get_year(t) == 2026
    ]

    print()
    print("-" * 100)
    print("EARLIER PERIOD VS 2026")
    print("-" * 100)

    for label, group in [
        ("2022-2025", earlier_trades),
        ("2026", trades_2026),
    ]:
        baseline, filtered, delta, exits = calculate_group(group)

        print()
        print(label)
        print("Trades:", len(group))
        print(f"Baseline R: {baseline:+.2f}")
        print(f"Filtered R: {filtered:+.2f}")
        print(f"Delta R: {delta:+.2f}")
        print("Threshold exits:", exits)

    # ------------------------------------------------------------------
    # CONTEXT DIMENSIONS
    # ------------------------------------------------------------------

    dimensions = [
        (
            "SETUP",
            (
                "Diagnostic_Setup",
                "setup",
                "Setup",
            ),
        ),
        (
            "ENTRY TREND",
            (
                "entry_trend",
                "Entry_Trend",
            ),
        ),
        (
            "TREND STRENGTH",
            (
                "entry_trend_strength",
                "Entry_Trend_Strength",
            ),
        ),
        (
            "TREND SLOPE",
            (
                "entry_trend_slope",
                "Entry_Trend_Slope",
            ),
        ),
        (
            "STRUCTURE BIAS",
            (
                "structure_bias",
                "Structure_Bias",
            ),
        ),
        (
            "LIQUIDITY SWEEP",
            (
                "liquidity_sweep",
                "Liquidity_Sweep",
            ),
        ),
    ]

    # ------------------------------------------------------------------
    # EACH DIMENSION — ALL YEARS
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CONTEXT BREAKDOWN — ALL YEARS")
    print("=" * 100)

    for title, field_names in dimensions:
        print_dimension(title, field_names)

    # ------------------------------------------------------------------
    # EACH DIMENSION — 2026 ONLY
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("2026 CONTEXT BREAKDOWN")
    print("=" * 100)

    original_trades = trades

    for title, field_names in dimensions:
        print()
        print("-" * 100)
        print(title)
        print("-" * 100)

        groups = {}

        for trade in trades_2026:
            value = get_field(trade, *field_names)
            value = str(value)

            groups.setdefault(value, []).append(trade)

        print(
            f"{'GROUP':30}"
            f"{'N':>6}"
            f"{'BASE R':>12}"
            f"{'FILTERED R':>14}"
            f"{'DELTA R':>12}"
            f"{'EXITS':>10}"
        )

        print("-" * 100)

        for group_name, group_trades in sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0])
        ):
            baseline, filtered, delta, exits = calculate_group(group_trades)

            print(
                f"{group_name[:30]:30}"
                f"{len(group_trades):6d}"
                f"{baseline:12.2f}"
                f"{filtered:14.2f}"
                f"{delta:12.2f}"
                f"{exits:10d}"
            )

    # ------------------------------------------------------------------
    # FIBONACCI ENTRY POSITION
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("FIBONACCI ENTRY POSITION — 2026")
    print("-" * 100)

    position_groups = {}

    for trade in trades_2026:
        value = get_field(
            trade,
            "entry_price_position_value",
            "Entry_Price_Position_Value",
            "Price_Position_Value",
        )

        try:
            numeric_value = float(value)

            if numeric_value < 0:
                bucket = "<0"
            elif numeric_value < 0.25:
                bucket = "0.00-0.25"
            elif numeric_value < 0.50:
                bucket = "0.25-0.50"
            elif numeric_value < 0.75:
                bucket = "0.50-0.75"
            elif numeric_value <= 1.00:
                bucket = "0.75-1.00"
            else:
                bucket = ">1.00"

        except Exception:
            bucket = "UNKNOWN"

        position_groups.setdefault(bucket, []).append(trade)

    print(
        f"{'POSITION':30}"
        f"{'N':>6}"
        f"{'BASE R':>12}"
        f"{'FILTERED R':>14}"
        f"{'DELTA R':>12}"
        f"{'EXITS':>10}"
    )

    print("-" * 100)

    for group_name, group_trades in sorted(
        position_groups.items(),
        key=lambda item: (-len(item[1]), item[0])
    ):
        baseline, filtered, delta, exits = calculate_group(group_trades)

        print(
            f"{group_name[:30]:30}"
            f"{len(group_trades):6d}"
            f"{baseline:12.2f}"
            f"{filtered:14.2f}"
            f"{delta:12.2f}"
            f"{exits:10d}"
        )

    # ------------------------------------------------------------------
    # FINAL DIAGNOSTIC
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 100)

    baseline_2026, filtered_2026, delta_2026, exits_2026 = calculate_group(
        trades_2026
    )

    print()
    print(f"2026 baseline: {baseline_2026:+.2f}R")
    print(f"2026 filtered: {filtered_2026:+.2f}R")
    print(f"2026 delta:    {delta_2026:+.2f}R")
    print(f"2026 exits:    {exits_2026}")

    if delta_2026 < 0:
        print()
        print(
            "RESULT: The frozen 3B_0.25R rule fails in 2026."
        )
        print(
            "The context breakdown above is diagnostic only."
        )
        print(
            "No subgroup-specific rule should be created from this output yet."
        )
    else:
        print()
        print(
            "RESULT: The frozen 3B_0.25R rule remains positive in 2026."
        )

    print()
    print("RESEARCH STATUS: DIAGNOSTIC ONLY")
    print("NO PRODUCTION ENGINE CHANGES")
    print("=" * 100)

def early_adverse_trade_profile_regime_comparison(trades):
    """
    Research-only diagnostic.

    Purpose:
        Compare the actual trades that trigger the frozen
        3B_0.25R Early-Adverse-R condition in 2022-2025
        versus 2026.

    This diagnostic is descriptive only.

    IMPORTANT:
        - No optimization.
        - No subgroup-specific rule creation.
        - No production engine changes.
        - No trade modification.
        - Uses only the frozen 3B_0.25R condition.
    """

    print()
    print("=" * 120)
    print("EARLY-ADVERSE TRADE PROFILE — 2022-2025 VS 2026")
    print("=" * 120)

    WINDOW = 3
    THRESHOLD = 0.25

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_early_adverse(trade):
        value = trade.get(
            f"early_adverse_r_bar_{WINDOW}"
        )

        if value is None:
            value = trade.get(
                f"Early_Adverse_R_Bar_{WINDOW}"
            )

        return to_float(
            value,
            default=None
        )

    def get_r(trade):
        return to_float(
            trade.get(
                "r_multiple"
            ),
            default=0.0
        )

    def get_mfe(trade):
        return to_float(
            trade.get(
                "max_favorable_r"
            ),
            default=None
        )

    def get_mae(trade):
        return to_float(
            trade.get(
                "max_adverse_r"
            ),
            default=None
        )

    def is_triggered(trade):
        early_adverse = get_early_adverse(
            trade
        )

        return (
            early_adverse is not None
            and early_adverse >= THRESHOLD
        )

    def period_trades(label):
        if label == "2022-2025":
            return [
                trade
                for trade in trades
                if get_year(trade) in {
                    2022,
                    2023,
                    2024,
                    2025,
                }
            ]

        if label == "2026":
            return [
                trade
                for trade in trades
                if get_year(trade) == 2026
            ]

        return []

    def summarize(label, group):
        triggered = [
            trade
            for trade in group
            if is_triggered(trade)
        ]

        non_triggered = [
            trade
            for trade in group
            if not is_triggered(trade)
        ]

        early_values = [
            get_early_adverse(trade)
            for trade in triggered
            if get_early_adverse(trade) is not None
        ]

        mfe_values = [
            get_mfe(trade)
            for trade in triggered
            if get_mfe(trade) is not None
        ]

        mae_values = [
            get_mae(trade)
            for trade in triggered
            if get_mae(trade) is not None
        ]

        triggered_r = [
            get_r(trade)
            for trade in triggered
        ]

        non_triggered_r = [
            get_r(trade)
            for trade in non_triggered
        ]

        def average(values):
            if not values:
                return None

            return sum(values) / len(values)

        triggered_winners = sum(
            1
            for value in triggered_r
            if value > 0
        )

        triggered_losers = sum(
            1
            for value in triggered_r
            if value < 0
        )

        non_triggered_winners = sum(
            1
            for value in non_triggered_r
            if value > 0
        )

        non_triggered_losers = sum(
            1
            for value in non_triggered_r
            if value < 0
        )

        print()
        print("-" * 120)
        print(label)
        print("-" * 120)

        print(
            f"Total trades:              {len(group)}"
        )

        print(
            f"Triggered 3B_0.25R:        {len(triggered)}"
        )

        if group:
            print(
                f"Trigger rate:              "
                f"{len(triggered) / len(group) * 100:.2f}%"
            )
        else:
            print(
                "Trigger rate:              N/A"
            )

        print()

        print("TRIGGERED TRADES")
        print(
            f"Average early adverse:     "
            f"{average(early_values):+.3f}R"
            if early_values
            else
            "Average early adverse:     N/A"
        )

        print(
            f"Average MFE:               "
            f"{average(mfe_values):+.3f}R"
            if mfe_values
            else
            "Average MFE:               N/A"
        )

        print(
            f"Average MAE:               "
            f"{average(mae_values):+.3f}R"
            if mae_values
            else
            "Average MAE:               N/A"
        )

        print(
            f"Final R:                   "
            f"{sum(triggered_r):+.2f}R"
        )

        print(
            f"Average final R:           "
            f"{average(triggered_r):+.3f}R"
            if triggered_r
            else
            "Average final R:           N/A"
        )

        print(
            f"Winners / Losers:          "
            f"{triggered_winners} / {triggered_losers}"
        )

        print()

        print("NON-TRIGGERED TRADES")

        print(
            f"Final R:                   "
            f"{sum(non_triggered_r):+.2f}R"
        )

        print(
            f"Average final R:           "
            f"{average(non_triggered_r):+.3f}R"
            if non_triggered_r
            else
            "Average final R:           N/A"
        )

        print(
            f"Winners / Losers:          "
            f"{non_triggered_winners} / "
            f"{non_triggered_losers}"
        )

        return triggered

    periods = {
        "2022-2025": period_trades(
            "2022-2025"
        ),
        "2026": period_trades(
            "2026"
        ),
    }

    triggered_by_period = {}

    for label, group in periods.items():
        triggered_by_period[label] = summarize(
            label,
            group
        )

    # ------------------------------------------------------------------
    # CONTEXT PROFILE OF TRIGGERED TRADES
    # ------------------------------------------------------------------

    context_dimensions = [
        (
            "SETUP",
            (
                "Diagnostic_Setup",
                "setup",
                "Setup",
            ),
        ),
        (
            "ENTRY_TREND",
            (
                "entry_trend",
                "Entry_Trend",
            ),
        ),
        (
            "STRUCTURE_BIAS",
            (
                "structure_bias",
                "Structure_Bias",
            ),
        ),
        (
            "LIQUIDITY_SWEEP",
            (
                "liquidity_sweep",
                "Liquidity_Sweep",
            ),
        ),
    ]

    print()
    print("=" * 120)
    print("TRIGGERED TRADE CONTEXT PROFILE")
    print("=" * 120)

    for dimension_name, field_names in context_dimensions:

        print()
        print("-" * 120)
        print(dimension_name)
        print("-" * 120)

        groups = {}

        for label, triggered in triggered_by_period.items():

            period_groups = {}

            for trade in triggered:

                value = None

                for field_name in field_names:
                    value = trade.get(
                        field_name
                    )

                    if (
                        value is not None
                        and str(value).strip() != ""
                    ):
                        break

                if value is None:
                    value = "UNKNOWN"

                period_groups.setdefault(
                    str(value),
                    []
                ).append(
                    trade
                )

            groups[label] = period_groups

        all_group_names = sorted(
            set(
                list(
                    groups["2022-2025"].keys()
                )
                +
                list(
                    groups["2026"].keys()
                )
            )
        )

        print(
            f"{'GROUP':30}"
            f"{'22-25 N':>10}"
            f"{'22-25 R':>14}"
            f"{'2026 N':>10}"
            f"{'2026 R':>14}"
        )

        print("-" * 120)

        for group_name in all_group_names:

            earlier_group = groups[
                "2022-2025"
            ].get(
                group_name,
                []
            )

            current_group = groups[
                "2026"
            ].get(
                group_name,
                []
            )

            earlier_r = sum(
                get_r(trade)
                for trade in earlier_group
            )

            current_r = sum(
                get_r(trade)
                for trade in current_group
            )

            print(
                f"{group_name[:30]:30}"
                f"{len(earlier_group):10d}"
                f"{earlier_r:14.2f}"
                f"{len(current_group):10d}"
                f"{current_r:14.2f}"
            )

    # ------------------------------------------------------------------
    # TREND STRENGTH / SLOPE BUCKETS
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("TRIGGERED TRADE TREND PROFILE")
    print("=" * 120)

    def strength_bucket(trade):

        value = to_float(
            trade.get(
                "entry_trend_strength"
            ),
            default=None
        )

        if value is None:
            return "UNKNOWN"

        if value < 20:
            return "<20"

        if value < 40:
            return "20-40"

        if value < 60:
            return "40-60"

        return "60+"

    def slope_bucket(trade):

        value = to_float(
            trade.get(
                "entry_trend_slope"
            ),
            default=None
        )

        if value is None:
            return "UNKNOWN"

        if value < -0.50:
            return "<-0.50"

        if value >= 0.50:
            return ">=0.50"

        return "MID"

    for title, bucket_function in [
        (
            "TREND STRENGTH",
            strength_bucket,
        ),
        (
            "TREND SLOPE",
            slope_bucket,
        ),
    ]:

        print()
        print("-" * 120)
        print(title)
        print("-" * 120)

        for label, triggered in triggered_by_period.items():

            groups = {}

            for trade in triggered:

                bucket = bucket_function(
                    trade
                )

                groups.setdefault(
                    bucket,
                    []
                ).append(
                    trade
                )

            print()
            print(label)

            for bucket, group in sorted(
                groups.items(),
                key=lambda item: (
                    -len(item[1]),
                    item[0]
                )
            ):

                total_r = sum(
                    get_r(trade)
                    for trade in group
                )

                print(
                    f"{bucket:<15} "
                    f"N={len(group):>3} "
                    f"FinalR={total_r:+.2f}R"
                )

    # ------------------------------------------------------------------
    # FIBONACCI ENTRY POSITION
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("TRIGGERED TRADE FIBONACCI ENTRY POSITION")
    print("=" * 120)

    def fibonacci_bucket(trade):

        value = trade.get(
            "entry_price_position_value"
        )

        if value is None:
            value = trade.get(
                "Entry_Price_Position_Value"
            )

        if value is None:
            value = trade.get(
                "Price_Position_Value"
            )

        value = to_float(
            value,
            default=None
        )

        if value is None:
            return "UNKNOWN"

        if value < 0:
            return "<0"

        if value < 0.25:
            return "0.00-0.25"

        if value < 0.50:
            return "0.25-0.50"

        if value < 0.75:
            return "0.50-0.75"

        if value <= 1.00:
            return "0.75-1.00"

        return ">1.00"

    for label, triggered in triggered_by_period.items():

        groups = {}

        for trade in triggered:

            bucket = fibonacci_bucket(
                trade
            )

            groups.setdefault(
                bucket,
                []
            ).append(
                trade
            )

        print()
        print(label)

        print(
            f"{'POSITION':20}"
            f"{'N':>8}"
            f"{'FINAL R':>14}"
        )

        print("-" * 60)

        for bucket, group in sorted(
            groups.items(),
            key=lambda item: (
                -len(item[1]),
                item[0]
            )
        ):

            total_r = sum(
                get_r(trade)
                for trade in group
            )

            print(
                f"{bucket:<20}"
                f"{len(group):8d}"
                f"{total_r:14.2f}"
            )

    # ------------------------------------------------------------------
    # FINAL INTERPRETATION
    # ------------------------------------------------------------------

    earlier_triggered = triggered_by_period[
        "2022-2025"
    ]

    current_triggered = triggered_by_period[
        "2026"
    ]

    print()
    print("=" * 120)
    print("RESEARCH INTERPRETATION")
    print("=" * 120)

    print()
    print(
        "This diagnostic compares the actual trades that "
        "triggered the frozen 3B_0.25R condition."
    )

    print(
        "It does not create, optimize, or recommend "
        "a new subgroup rule."
    )

    print()
    print(
        f"2022-2025 triggered trades: "
        f"{len(earlier_triggered)}"
    )

    print(
        f"2026 triggered trades:      "
        f"{len(current_triggered)}"
    )

    if len(current_triggered) == 0:
        print()
        print(
            "2026 contains no 3B_0.25R-triggering trades."
        )
        print(
            "Therefore the frozen early-adverse rule has "
            "almost no opportunity to affect 2026."
        )

    print()
    print(
        "RESEARCH STATUS: DIAGNOSTIC ONLY"
    )

    print(
        "NO PRODUCTION ENGINE CHANGES"
    )

    print("=" * 120)

def stock_regime_comparison_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Compare stock-level performance between 2022-2025
        and 2026.

    IMPORTANT:
        - Uses the complete trade population.
        - Does NOT optimize any rule.
        - Does NOT create a trading filter.
        - Does NOT modify production engine logic.
        - 2026 stock-level samples are small.
        - This is descriptive regime analysis only.
    """

    print()
    print("=" * 120)
    print("STOCK × REGIME COMPARISON — 2022-2025 VS 2026")
    print("=" * 120)

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_number(trade, *names):
        for name in names:
            value = trade.get(name)

            if value is None:
                continue

            try:
                return float(value)
            except Exception:
                continue

        return None

    def get_symbol(trade):
        value = trade.get("_symbol")

        if value is None:
            value = trade.get("symbol")

        if value is None:
            value = trade.get("Symbol")

        if value is None:
            value = trade.get("ticker")

        if value is None:
            value = trade.get("Ticker")

        if value is None:
            return "UNKNOWN"

        value = str(value).strip()

        return value if value else "UNKNOWN"

    def get_r(trade):
        value = get_number(
            trade,
            "r_multiple",
            "R_Multiple"
        )

        return value if value is not None else 0.0

    def calculate_metrics(group):
        if not group:
            return {
                "trades": 0,
                "wins": 0,
                "win_rate": None,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": None,
                "total_r": 0.0,
                "avg_r": None,
            }

        r_values = [
            get_r(trade)
            for trade in group
        ]

        wins = [
            value
            for value in r_values
            if value > 0
        ]

        losses = [
            value
            for value in r_values
            if value < 0
        ]

        gross_profit = sum(wins)

        gross_loss = abs(
            sum(losses)
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else None
        )

        return {
            "trades": len(group),
            "wins": len(wins),
            "win_rate": (
                len(wins) / len(group) * 100
            ),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "total_r": sum(r_values),
            "avg_r": (
                sum(r_values) / len(r_values)
            ),
        }

    # --------------------------------------------------------------
    # SPLIT INTO REGIMES
    # --------------------------------------------------------------

    earlier = []
    current = []

    for trade in trades:

        year = get_year(trade)

        if year is None:
            continue

        if 2022 <= year <= 2025:
            earlier.append(trade)

        elif year == 2026:
            current.append(trade)

    # --------------------------------------------------------------
    # GROUP BY STOCK
    # --------------------------------------------------------------

    stock_groups_earlier = defaultdict(list)
    stock_groups_current = defaultdict(list)

    for trade in earlier:

        symbol = get_symbol(trade)

        stock_groups_earlier[
            symbol
        ].append(trade)

    for trade in current:

        symbol = get_symbol(trade)

        stock_groups_current[
            symbol
        ].append(trade)

    symbols = sorted(
        set(stock_groups_earlier.keys())
        | set(stock_groups_current.keys())
    )

    # --------------------------------------------------------------
    # HEADER
    # --------------------------------------------------------------

    print()
    print(
        "Goal: determine whether the 2026 regime improvement "
        "is broadly distributed across stocks."
    )

    print()
    print(
        "IMPORTANT: 2026 stock-level samples are small, so "
        "these results are descriptive and NOT statistically "
        "conclusive."
    )

    # --------------------------------------------------------------
    # PERFORMANCE TABLE
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("STOCK-LEVEL PERFORMANCE")
    print("=" * 120)

    print()
    print(
        f"{'SYMBOL':20}"
        f"{'22-25 N':>10}"
        f"{'22-25 Win%':>13}"
        f"{'22-25 PF':>11}"
        f"{'22-25 R':>12}"
        f"{'26 N':>8}"
        f"{'26 Win%':>11}"
        f"{'26 PF':>10}"
        f"{'26 R':>11}"
        f"{'ΔR':>11}"
    )

    print("-" * 120)

    stock_results = []

    for symbol in symbols:

        earlier_group = (
            stock_groups_earlier.get(
                symbol,
                []
            )
        )

        current_group = (
            stock_groups_current.get(
                symbol,
                []
            )
        )

        earlier_metrics = calculate_metrics(
            earlier_group
        )

        current_metrics = calculate_metrics(
            current_group
        )

        delta_r = (
            current_metrics["total_r"]
            - earlier_metrics["total_r"]
        )

        stock_results.append(
            {
                "symbol": symbol,
                "earlier": earlier_metrics,
                "current": current_metrics,
                "delta_r": delta_r,
            }
        )

        earlier_pf = (
            f"{earlier_metrics['profit_factor']:.2f}"
            if earlier_metrics["profit_factor"] is not None
            else "N/A"
        )

        current_pf = (
            f"{current_metrics['profit_factor']:.2f}"
            if current_metrics["profit_factor"] is not None
            else "N/A"
        )

        earlier_win = (
            f"{earlier_metrics['win_rate']:.2f}"
            if earlier_metrics["win_rate"] is not None
            else "N/A"
        )

        current_win = (
            f"{current_metrics['win_rate']:.2f}"
            if current_metrics["win_rate"] is not None
            else "N/A"
        )

        print(
            f"{str(symbol)[:20]:20}"
            f"{earlier_metrics['trades']:10d}"
            f"{earlier_win:>13}"
            f"{earlier_pf:>11}"
            f"{earlier_metrics['total_r']:>12.2f}"
            f"{current_metrics['trades']:>8d}"
            f"{current_win:>11}"
            f"{current_pf:>10}"
            f"{current_metrics['total_r']:>11.2f}"
            f"{delta_r:>11.2f}"
        )

    # --------------------------------------------------------------
    # 2026 CONTRIBUTION
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("2026 STOCK CONTRIBUTION")
    print("=" * 120)

    total_current_trades = len(current)

    total_current_r = sum(
        get_r(trade)
        for trade in current
    )

    print()
    print(
        f"{'SYMBOL':20}"
        f"{'2026 N':>10}"
        f"{'TRADE SHARE':>15}"
        f"{'TOTAL R':>15}"
        f"{'R SHARE':>15}"
    )

    print("-" * 80)

    for result in stock_results:

        symbol = result["symbol"]
        current_metrics = result["current"]

        trade_count = current_metrics["trades"]
        total_r = current_metrics["total_r"]

        trade_share = (
            trade_count
            / total_current_trades
            * 100
            if total_current_trades
            else 0.0
        )

        r_share = (
            total_r
            / total_current_r
            * 100
            if total_current_r != 0
            else 0.0
        )

        print(
            f"{str(symbol)[:20]:20}"
            f"{trade_count:10d}"
            f"{trade_share:15.2f}"
            f"{total_r:15.2f}"
            f"{r_share:15.2f}"
        )

    # --------------------------------------------------------------
    # EARLY-ADVERSE RATE BY STOCK
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("2026 EARLY-ADVERSE-R TRIGGER RATE BY STOCK")
    print("=" * 120)

    print()
    print(
        f"{'SYMBOL':20}"
        f"{'2026 N':>10}"
        f"{'TRIGGERED':>15}"
        f"{'TRIGGER RATE':>15}"
    )

    print("-" * 65)

    for symbol in symbols:

        current_group = (
            stock_groups_current.get(
                symbol,
                []
            )
        )

        triggered = 0
        known = 0

        for trade in current_group:

            value = get_number(
                trade,
                "early_adverse_r_bar_3",
                "Early_Adverse_R_Bar_3"
            )

            if value is None:
                continue

            known += 1

            if value >= 0.25:
                triggered += 1

        trigger_rate = (
            triggered / known * 100
            if known
            else None
        )

        trigger_text = (
            f"{trigger_rate:.2f}"
            if trigger_rate is not None
            else "N/A"
        )

        print(
            f"{str(symbol)[:20]:20}"
            f"{len(current_group):10d}"
            f"{triggered:15d}"
            f"{trigger_text:15}"
        )

    # --------------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("STOCK × REGIME INTERPRETATION")
    print("=" * 120)

    print()

    positive_stocks = 0
    negative_stocks = 0
    active_stocks = 0

    for result in stock_results:

        current_n = result["current"]["trades"]
        delta_r = result["delta_r"]

        if current_n == 0:
            continue

        active_stocks += 1

        if delta_r > 0:
            positive_stocks += 1
        elif delta_r < 0:
            negative_stocks += 1

    print(
        f"Stocks with 2026 trades: "
        f"{active_stocks}"
    )

    print(
        f"Stocks with positive 2026-vs-earlier ΔR: "
        f"{positive_stocks}"
    )

    print(
        f"Stocks with negative 2026-vs-earlier ΔR: "
        f"{negative_stocks}"
    )

    print()

    if positive_stocks > negative_stocks:

        print(
            "OBSERVATION: 2026 improvement is positive "
            "across more stocks than negative stocks."
        )

    elif positive_stocks < negative_stocks:

        print(
            "OBSERVATION: 2026 improvement is concentrated "
            "in a minority of stocks."
        )

    else:

        print(
            "OBSERVATION: stock-level direction is mixed."
        )

    print()
    print(
        "This diagnostic does NOT establish that the regime "
        "shift is stock-independent."
    )

    print(
        "Small 2026 samples per stock require additional "
        "validation before any production decision."
    )

    print()
    print(
        "RESEARCH RULE: use this output to identify whether "
        "stock concentration deserves deeper investigation."
    )

    print(
        "NO PRODUCTION ENGINE CHANGES."
    )

def early_adverse_mfe_recovery_diagnostic(trades):
    """
    Research diagnostic:
    Among trades that trigger the frozen early-adverse rule
    (within first 3 bars and >= 0.25R adverse), measure how
    much favorable movement occurs AFTER the trigger.

    This is telemetry only.
    It does NOT modify trade outcomes.
    """

    triggered = [
        trade
        for trade in trades 
        if to_bool(
            trade.get(
                "early_adverse_triggered"
            )
        )
    ]

    if not triggered:
        print(
            "\n================================="
        )
        print(
            "Early-Adverse Post-Trigger Recovery Diagnostic"
        )
        print(
            "================================="
        )
        print(
            "No trades triggered the frozen "
            "early-adverse rule."
        )
        return

    recovered_entry = [
        trade
        for trade in triggered
        if to_bool(
            trade.get(
                "post_trigger_recovered_to_entry"
            )
        )
    ]

    recovered_025 = [
        trade
        for trade in triggered
        if to_bool(
            trade.get(
                "post_trigger_recovered_0_25r"
            )
        )
    ]

    recovered_050 = [
        trade
        for trade in triggered
        if to_bool(
            trade.get(
                "post_trigger_recovered_0_5r"
            )
        )
    ]

    recovered_1r = [
        trade
        for trade in triggered
        if to_bool(
            trade.get(
                "post_trigger_recovered_1r"
            )
        )
    ]

    post_trigger_mfe = [
        to_float(
            trade.get(
                "post_trigger_max_favorable_r"
            )
        )
        for trade in triggered
        if to_float(
            trade.get(
                "post_trigger_max_favorable_r"
            )
        ) is not None
    ]

    post_trigger_mae = [
        to_float(
            trade.get(
                "post_trigger_max_adverse_r"
            )
        )
        for trade in triggered
        if to_float(
            trade.get(
                "post_trigger_max_adverse_r"
            )
        ) is not None
    ]

    def average(values):
        if not values:
            return None

        return sum(values) / len(values)

    print(
        "\n================================="
    )
    print(
        "Early-Adverse Post-Trigger Recovery Diagnostic"
    )
    print(
        "================================="
    )

    print(
        f"Triggered Trades: {len(triggered)}"
    )

    print(
        f"Recovered to Entry: "
        f"{len(recovered_entry)} "
        f"({len(recovered_entry) / len(triggered) * 100:.2f}%)"
    )

    print(
        f"Recovered to +0.25R: "
        f"{len(recovered_025)} "
        f"({len(recovered_025) / len(triggered) * 100:.2f}%)"
    )

    print(
        f"Recovered to +0.50R: "
        f"{len(recovered_050)} "
        f"({len(recovered_050) / len(triggered) * 100:.2f}%)"
    )

    print(
        f"Recovered to +1.00R: "
        f"{len(recovered_1r)} "
        f"({len(recovered_1r) / len(triggered) * 100:.2f}%)"
    )

    avg_mfe = average(post_trigger_mfe)
    avg_mae = average(post_trigger_mae)

    print(
        f"Avg Post-Trigger MFE: "
        f"{avg_mfe:.2f}R"
        if avg_mfe is not None
        else "Avg Post-Trigger MFE: N/A"
    )

    print(
        f"Avg Post-Trigger MAE: "
        f"{avg_mae:.2f}R"
        if avg_mae is not None
        else "Avg Post-Trigger MAE: N/A"
    )

    recovery_times = [
        to_float(
            trade.get(
                "post_trigger_bars_to_entry"
            )
        )
        for trade in triggered
        if to_float(
            trade.get(
                "post_trigger_bars_to_entry"
            )
        ) is not None
    ]

    if recovery_times:
        print(
            f"Avg Bars to Entry Recovery: "
            f"{average(recovery_times):.2f}"
        )
    else:
        print(
            "Avg Bars to Entry Recovery: N/A"
        )

    print(
        "\nResearch interpretation:"
    )

    if len(recovered_entry) / len(triggered) >= 0.50:
        print(
            "A majority of early-adverse trades "
            "recover to entry after the trigger."
        )
    else:
        print(
            "Less than half of early-adverse trades "
            "recover to entry after the trigger."
        )

    if avg_mfe is not None and avg_mfe >= 0.50:
        print(
            "Post-trigger favorable movement remains "
            "meaningful and should be investigated."
        )
    else:
        print(
            "Post-trigger favorable movement is limited."
        )

    print(
        "\nResearch only — no execution rule changed."
    )
def early_adverse_trigger_discrepancy_audit(trades):
    """
    Research-only audit.

    Purpose:
        Explain why different Early-Adverse-R diagnostics may report
        different triggered-trade counts.

    Frozen condition:
        early_adverse_r_bar_3 >= 0.25R

    This diagnostic does NOT change any trading rule.
    """

    print()
    print("=" * 120)
    print("EARLY-ADVERSE TRIGGER DISCREPANCY AUDIT")
    print("=" * 120)

    def get_value(trade, key):
        return to_float(
            trade.get(key),
            default=None
        )

    frozen_triggered = []
    post_trigger_triggered = []

    for trade in trades:

        bar_3 = get_value(
            trade,
            "early_adverse_r_bar_3"
        )

        if (
            bar_3 is not None
            and bar_3 >= 0.25
        ):
            frozen_triggered.append(trade)

        if to_bool(
            trade.get(
                "early_adverse_triggered"
            )
        ):
            post_trigger_triggered.append(trade)

    frozen_keys = {
        (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
            str(trade.get("direction", "")),
            str(trade.get("entry_price", "")),
        )
        for trade in frozen_triggered
    }

    post_trigger_keys = {
        (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
            str(trade.get("direction", "")),
            str(trade.get("entry_price", "")),
        )
        for trade in post_trigger_triggered
    }

    only_frozen = [
        trade
        for trade in frozen_triggered
        if (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
            str(trade.get("direction", "")),
            str(trade.get("entry_price", "")),
        )
        not in post_trigger_keys
    ]

    only_post_trigger = [
        trade
        for trade in post_trigger_triggered
        if (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
            str(trade.get("direction", "")),
            str(trade.get("entry_price", "")),
        )
        not in frozen_keys
    ]

    overlap = [
        trade
        for trade in frozen_triggered
        if (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
            str(trade.get("direction", "")),
            str(trade.get("entry_price", "")),
        )
        in post_trigger_keys
    ]

    print()
    print("-" * 120)
    print("TRIGGER COUNTS")
    print("-" * 120)

    print(
        f"Frozen 3B_0.25R condition: "
        f"{len(frozen_triggered)}"
    )

    print(
        f"Post-trigger telemetry flag: "
        f"{len(post_trigger_triggered)}"
    )

    print(
        f"Overlap: "
        f"{len(overlap)}"
    )

    print(
        f"Only frozen condition: "
        f"{len(only_frozen)}"
    )

    print(
        f"Only post-trigger flag: "
        f"{len(only_post_trigger)}"
    )

    print()
    print("-" * 120)
    print("ONLY POST-TRIGGER FLAG")
    print("-" * 120)

    if not only_post_trigger:
        print("None")

    for trade in only_post_trigger:

        print(
            f"Date={trade.get('entry_date')} | "
            f"Symbol={trade.get('symbol')} | "
            f"Direction={trade.get('direction')} | "
            f"Bar1={trade.get('early_adverse_r_bar_1')} | "
            f"Bar2={trade.get('early_adverse_r_bar_2')} | "
            f"Bar3={trade.get('early_adverse_r_bar_3')} | "
            f"TriggerBar={trade.get('early_adverse_trigger_bar')} | "
            f"FinalR={trade.get('r_multiple')}"
        )

    print()
    print("-" * 120)
    print("ONLY FROZEN 3B_0.25R CONDITION")
    print("-" * 120)

    if not only_frozen:
        print("None")

    for trade in only_frozen:

        print(
            f"Date={trade.get('entry_date')} | "
            f"Symbol={trade.get('symbol')} | "
            f"Direction={trade.get('direction')} | "
            f"Bar1={trade.get('early_adverse_r_bar_1')} | "
            f"Bar2={trade.get('early_adverse_r_bar_2')} | "
            f"Bar3={trade.get('early_adverse_r_bar_3')} | "
            f"TriggerBar={trade.get('early_adverse_trigger_bar')} | "
            f"FinalR={trade.get('r_multiple')}"
        )

    print()
    print("=" * 120)
    print("RESEARCH STATUS: DIAGNOSTIC ONLY")
    print("NO PRODUCTION ENGINE CHANGES")
    print("=" * 120)

def fixed_early_adverse_stock_year_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Validate the fixed 5B_0.25R Early Adverse-R rule
        across STOCK × YEAR combinations.

    IMPORTANT:
        - The rule is completely frozen.
        - No stock-specific optimization.
        - No year-specific optimization.
        - No threshold selection.
        - No engine changes.
        - No execution changes.
        - This is descriptive robustness analysis only.

    Fixed rule:
        Reject a trade only when early_adverse_r_bar_5 >= 0.25R.

    Missing early-adverse values are retained.
    """

    print()
    print("=" * 120)
    print("FIXED EARLY-ADVERSE-R STOCK × YEAR VALIDATION")
    print("=" * 120)

    print()
    print(
        "Purpose: test the fixed 5B_0.25R rule across stock × year combinations."
    )
    print(
        "IMPORTANT: no stock-specific or year-specific optimization is performed."
    )

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25
    MIN_SAMPLE_SIZE = 5

    def get_symbol(trade):
        value = trade.get("_symbol")

        if value is None:
            value = trade.get("symbol")

        if value is None:
            value = trade.get("Symbol")

        if value is None:
            value = trade.get("ticker")

        if value is None:
            value = trade.get("Ticker")

        if value is None:
            return "UNKNOWN"

        value = str(value).strip()

        return value if value else "UNKNOWN"

    def get_year(trade):
        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        )

        if len(entry_date) >= 4:
            try:
                return int(
                    entry_date[:4]
                )
            except Exception:
                return None

        return None

    def rule_passes(trade):
        value = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if value is None:
            return True

        return value < RULE_THRESHOLD

    def trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=0.0
        )

        return value if value is not None else 0.0

    def evaluate(group_trades):
        baseline_r = sum(
            trade_r(trade)
            for trade in group_trades
        )

        filtered_trades = [
            trade
            for trade in group_trades
            if rule_passes(trade)
        ]

        filtered_r = sum(
            trade_r(trade)
            for trade in filtered_trades
        )

        removed_trades = [
            trade
            for trade in group_trades
            if not rule_passes(trade)
        ]

        return {
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "delta_r": filtered_r - baseline_r,
            "trades": len(group_trades),
            "retained": len(filtered_trades),
            "removed": len(removed_trades),
        }

    # ------------------------------------------------------------------
    # BUILD STOCK × YEAR GROUPS
    # ------------------------------------------------------------------

    groups = {}

    for trade in trades:

        symbol = get_symbol(
            trade
        )

        year = get_year(
            trade
        )

        if year is None:
            continue

        key = (
            symbol,
            year
        )

        groups.setdefault(
            key,
            []
        ).append(trade)

    if not groups:
        print()
        print(
            "No valid stock × year groups were found."
        )
        return

    # ------------------------------------------------------------------
    # STOCK × YEAR RESULTS
    # ------------------------------------------------------------------

    print()
    print("-" * 120)
    print("STOCK × YEAR RESULTS")
    print("-" * 120)

    print(
        f"Fixed Rule: "
        f"{RULE_WINDOW} bars / "
        f"{RULE_THRESHOLD:.2f}R"
    )

    print(
        f"Minimum sample size for interpretation: "
        f"{MIN_SAMPLE_SIZE} trades"
    )

    print()

    print(
        f"{'STOCK':<18} "
        f"{'YEAR':<6} "
        f"{'TRADES':>6} "
        f"{'RETAINED':>9} "
        f"{'REMOVED':>8} "
        f"{'BASELINE R':>11} "
        f"{'FILTERED R':>11} "
        f"{'DELTA R':>9} "
        f"{'STATUS'}"
    )

    print(
        "-" * 120
    )

    positive_cells = []
    negative_cells = []
    neutral_cells = []
    interpretable_cells = []

    stock_results = {}

    for (
        symbol,
        year
    ) in sorted(
        groups.keys(),
        key=lambda item: (
            item[0],
            item[1]
        )
    ):

        group_trades = groups[
            (
                symbol,
                year
            )
        ]

        result = evaluate(
            group_trades
        )

        if result["trades"] < MIN_SAMPLE_SIZE:
            status = "INSUFFICIENT_SAMPLE"

        elif result["delta_r"] > 0:
            status = "IMPROVED"

            positive_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

            interpretable_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

        elif result["delta_r"] < 0:
            status = "DETERIORATED"

            negative_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

            interpretable_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

        else:
            status = "NO_CHANGE"

            neutral_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

            interpretable_cells.append(
                (
                    symbol,
                    year,
                    result
                )
            )

        stock_results.setdefault(
            symbol,
            []
        ).append(
            result
        )

        print(
            f"{symbol:<18} "
            f"{year:<6} "
            f"{result['trades']:>6} "
            f"{result['retained']:>9} "
            f"{result['removed']:>8} "
            f"{result['baseline_r']:>11.2f} "
            f"{result['filtered_r']:>11.2f} "
            f"{result['delta_r']:>9.2f} "
            f"{status}"
        )

    # ------------------------------------------------------------------
    # STOCK-LEVEL AGGREGATION
    # ------------------------------------------------------------------

    print()
    print("-" * 120)
    print("STOCK-LEVEL AGGREGATION")
    print("-" * 120)

    print(
        "This aggregation uses the same frozen rule for every stock."
    )

    print()

    stock_groups = {}

    for trade in trades:

        symbol = get_symbol(
            trade
        )

        stock_groups.setdefault(
            symbol,
            []
        ).append(
            trade
        )

    print(
        f"{'STOCK':<18} "
        f"{'TRADES':>7} "
        f"{'REMOVED':>9} "
        f"{'BASELINE R':>12} "
        f"{'FILTERED R':>12} "
        f"{'DELTA R':>10}"
    )

    print(
        "-" * 90
    )

    stock_positive = 0
    stock_negative = 0
    stock_neutral = 0

    for symbol in sorted(
        stock_groups.keys()
    ):

        result = evaluate(
            stock_groups[symbol]
        )

        if result["delta_r"] > 0:
            stock_positive += 1

        elif result["delta_r"] < 0:
            stock_negative += 1

        else:
            stock_neutral += 1

        print(
            f"{symbol:<18} "
            f"{result['trades']:>7} "
            f"{result['removed']:>9} "
            f"{result['baseline_r']:>12.2f} "
            f"{result['filtered_r']:>12.2f} "
            f"{result['delta_r']:>10.2f}"
        )

    # ------------------------------------------------------------------
    # STOCK × YEAR ROBUSTNESS SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("STOCK × YEAR ROBUSTNESS SUMMARY")
    print("=" * 120)

    total_cells = len(
        groups
    )

    interpretable_count = len(
        interpretable_cells
    )

    positive_count = len(
        positive_cells
    )

    negative_count = len(
        negative_cells
    )

    neutral_count = len(
        neutral_cells
    )

    insufficient_count = (
        total_cells
        - interpretable_count
    )

    print()
    print(
        f"Total stock × year cells: "
        f"{total_cells}"
    )

    print(
        f"Interpretable cells: "
        f"{interpretable_count}"
    )

    print(
        f"Insufficient-sample cells: "
        f"{insufficient_count}"
    )

    print(
        f"Improved cells: "
        f"{positive_count}"
    )

    print(
        f"Deteriorated cells: "
        f"{negative_count}"
    )

    print(
        f"No-change cells: "
        f"{neutral_count}"
    )

    if interpretable_count > 0:

        print()
        print(
            f"Positive-cell rate: "
            f"{positive_count / interpretable_count * 100:.2f}%"
        )

    # ------------------------------------------------------------------
    # STOCK × YEAR DELTA CONTRIBUTION
    # ------------------------------------------------------------------

    if positive_cells:

        print()
        print("-" * 120)
        print("POSITIVE STOCK × YEAR CELLS")
        print("-" * 120)

        for (
            symbol,
            year,
            result
        ) in sorted(
            positive_cells,
            key=lambda item: item[2]["delta_r"],
            reverse=True
        ):

            print(
                f"{symbol:<18} "
                f"{year:<6} "
                f"Trades={result['trades']:>3} "
                f"BaselineR={result['baseline_r']:>8.2f} "
                f"FilteredR={result['filtered_r']:>8.2f} "
                f"DeltaR={result['delta_r']:>8.2f}"
            )

    if negative_cells:

        print()
        print("-" * 120)
        print("NEGATIVE STOCK × YEAR CELLS")
        print("-" * 120)

        for (
            symbol,
            year,
            result
        ) in sorted(
            negative_cells,
            key=lambda item: item[2]["delta_r"]
        ):

            print(
                f"{symbol:<18} "
                f"{year:<6} "
                f"Trades={result['trades']:>3} "
                f"BaselineR={result['baseline_r']:>8.2f} "
                f"FilteredR={result['filtered_r']:>8.2f} "
                f"DeltaR={result['delta_r']:>8.2f}"
            )

    # ------------------------------------------------------------------
    # FINAL RESEARCH INTERPRETATION
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("STOCK × YEAR VALIDATION INTERPRETATION")
    print("=" * 120)

    if (
        interpretable_count > 0
        and positive_count == interpretable_count
    ):

        print(
            "STRONG ROBUSTNESS SIGNAL:"
        )

        print(
            "The fixed 5B_0.25R rule improves every "
            "interpretable stock × year cell."
        )

        print(
            "This reduces concern that the global improvement "
            "is driven by a small number of stock-year combinations."
        )

    elif (
        interpretable_count > 0
        and positive_count > negative_count
    ):

        print(
            "MIXED BUT POSITIVE ROBUSTNESS SIGNAL:"
        )

        print(
            "More interpretable stock × year cells improve "
            "than deteriorate."
        )

        print(
            "The fixed rule remains promising, but the effect "
            "is not uniformly stable across stock × year cells."
        )

    elif (
        interpretable_count > 0
        and positive_count == negative_count
    ):

        print(
            "MIXED ROBUSTNESS SIGNAL:"
        )

        print(
            "Improving and deteriorating stock × year cells "
            "are balanced."
        )

        print(
            "The global improvement should not yet be treated "
            "as robust across stock × year dimensions."
        )

    else:

        print(
            "WEAK ROBUSTNESS SIGNAL:"
        )

        print(
            "The fixed rule does not show consistent improvement "
            "across interpretable stock × year cells."
        )

    print()
    print(
        f"Stock-level result: "
        f"{stock_positive} positive, "
        f"{stock_negative} negative, "
        f"{stock_neutral} unchanged."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

    print(
        "This diagnostic validates robustness only."
    )

def fixed_early_adverse_entry_quality_diagnostic(trades):
    """
    Research-only diagnostic.

    Purpose:
        Test the single frozen 5B_0.25R Early Adverse-R rule
        across entry-time contexts.

    IMPORTANT:
        - No subgroup-specific optimization.
        - No rule selection.
        - No engine changes.
        - No execution changes.
        - This is descriptive robustness analysis only.

    Fixed rule:
        Reject a trade only when early_adverse_r_bar_5 >= 0.25R.

    Missing early-adverse values are retained.
    """

    print()
    print("=" * 120)
    print("FIXED EARLY-ADVERSE-R + ENTRY QUALITY VALIDATION")
    print("=" * 120)

    print()
    print(
        "Purpose: test the fixed 5B_0.25R rule across entry-time contexts."
    )
    print(
        "IMPORTANT: no subgroup-specific optimization is performed."
    )

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    def rule_passes(trade):
        value = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if value is None:
            return True

        return value < RULE_THRESHOLD

    def trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=0.0
        )

        return value if value is not None else 0.0

    def evaluate(group_trades):
        baseline_r = sum(
            trade_r(trade)
            for trade in group_trades
        )

        filtered_trades = [
            trade
            for trade in group_trades
            if rule_passes(trade)
        ]

        filtered_r = sum(
            trade_r(trade)
            for trade in filtered_trades
        )

        removed_trades = [
            trade
            for trade in group_trades
            if not rule_passes(trade)
        ]

        return {
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "delta_r": filtered_r - baseline_r,
            "trades": len(group_trades),
            "retained": len(filtered_trades),
            "removed": len(removed_trades),
        }

    # ------------------------------------------------------------------
    # GLOBAL
    # ------------------------------------------------------------------

    global_result = evaluate(trades)

    print()
    print("-" * 120)
    print("GLOBAL FIXED RULE")
    print("-" * 120)

    print(
        f"Rule: {RULE_WINDOW} bars / "
        f"{RULE_THRESHOLD:.2f}R"
    )

    print(
        f"Baseline R={global_result['baseline_r']:>8.2f} | "
        f"Filtered R={global_result['filtered_r']:>8.2f} | "
        f"DeltaR={global_result['delta_r']:>8.2f} | "
        f"Trades={global_result['trades']:>3} | "
        f"Retained={global_result['retained']:>3} | "
        f"Removed={global_result['removed']:>3}"
    )

    # ------------------------------------------------------------------
    # YEAR-BY-YEAR
    # ------------------------------------------------------------------

    print()
    print("-" * 120)
    print("YEAR-BY-YEAR FIXED RULE")
    print("-" * 120)

    years = [2024, 2025, 2026]

    for year in years:

        year_trades = [
            trade
            for trade in trades
            if str(
                trade.get(
                    "entry_date",
                    ""
                )
            ).startswith(str(year))
        ]

        if not year_trades:
            continue

        result = evaluate(year_trades)

        print(
            f"YEAR {year} | "
            f"BaselineR={result['baseline_r']:>7.2f} | "
            f"FilteredR={result['filtered_r']:>7.2f} | "
            f"DeltaR={result['delta_r']:>7.2f} | "
            f"Trades={result['trades']:>3} | "
            f"Removed={result['removed']:>3}"
        )

    # ------------------------------------------------------------------
    # ENTRY-TIME CONTEXTS
    # ------------------------------------------------------------------

    contexts = [
        (
            "SETUP",
            lambda trade: trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        ),
        (
            "ENTRY_TREND",
            lambda trade: trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        ),
        (
            "TREND_STRENGTH",
            lambda trade: (
                "60+"
                if (
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                        default=None
                    ) is not None
                    and
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                        default=None
                    ) >= 60
                )
                else
                "40-60"
                if (
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                        default=None
                    ) is not None
                    and
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                            default=None
                    ) >= 40
                )
                else
                "20-40"
                if (
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                        default=None
                    ) is not None
                    and
                    to_float(
                        trade.get(
                            "entry_trend_strength"
                        ),
                            default=None
                    ) >= 20
                )
                else
                "<20"
                if to_float(
                    trade.get(
                        "entry_trend_strength"
                    ),
                    default=None
                ) is not None
                else
                "UNKNOWN"
            )
        ),
        (
            "TREND_SLOPE",
            lambda trade: (
                ">=0.50"
                if (
                    to_float(
                        trade.get(
                            "entry_trend_slope"
                        ),
                        default=None
                    ) is not None
                    and
                    to_float(
                        trade.get(
                            "entry_trend_slope"
                        ),
                        default=None
                    ) >= 0.50
                )
                else
                "<-0.50"
                if (
                    to_float(
                        trade.get(
                            "entry_trend_slope"
                        ),
                        default=None
                    ) is not None
                    and
                    to_float(
                        trade.get(
                            "entry_trend_slope"
                        ),
                            default=None
                    ) < -0.50
                )
                else
                "MID"
                if to_float(
                    trade.get(
                        "entry_trend_slope"
                    ),
                    default=None
                ) is not None
                else
                "UNKNOWN"
            )
        ),
        (
            "STRUCTURE_BIAS",
            lambda trade: trade.get(
                "Diagnostic_Structure_Bias",
                trade.get(
                    "Structure_Bias",
                    "UNKNOWN"
                )
            )
        ),
        (
            "LIQUIDITY_SWEEP",
            lambda trade: (
                "SWEEP"
                if str(
                    trade.get(
                        "Liquidity_Sweep",
                        ""
                    )
                ).strip().lower()
                in {
                    "true",
                    "1",
                    "yes"
                }
                else
                "NO_SWEEP"
                if str(
                    trade.get(
                        "Liquidity_Sweep",
                        ""
                    )
                ).strip().lower()
                in {
                    "false",
                    "0",
                    "no"
                }
                else
                "UNKNOWN"
            )
        ),
        (
            "BREAK_AGE",
            lambda trade: (
                str(
                    trade.get(
                        "Diagnostic_Break_Age",
                        trade.get(
                            "Break_Age",
                            "UNKNOWN"
                        )
                    )
                )
            )
        ),
    ]

    # ------------------------------------------------------------------
    # CONTEXT STABILITY
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("ENTRY-TIME CONTEXT STABILITY")
    print("=" * 120)

    for context_name, context_getter in contexts:

        groups = {}

        for trade in trades:
            key = str(
                context_getter(trade)
            )

            groups.setdefault(
                key,
                []
            ).append(trade)

        print()
        print(f"[{context_name}]")
        print("-" * 120)

        for group_name in sorted(groups.keys()):

            group_trades = groups[group_name]

            # Avoid interpreting tiny samples.
            if len(group_trades) < 5:
                continue

            result = evaluate(
                group_trades
            )

            print(
                f"{group_name:<25} "
                f"Trades={result['trades']:>3} "
                f"Removed={result['removed']:>3} "
                f"BaselineR={result['baseline_r']:>8.2f} "
                f"FilteredR={result['filtered_r']:>8.2f} "
                f"DeltaR={result['delta_r']:>8.2f}"
            )

    # ------------------------------------------------------------------
    # LEAVE-ONE-SETUP-OUT
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("LEAVE-ONE-SETUP-OUT STRESS TEST")
    print("=" * 120)

    setups = sorted(
        {
            str(
                trade.get(
                    "Diagnostic_Setup",
                    trade.get(
                        "setup",
                        "UNKNOWN"
                    )
                )
            )
            for trade in trades
        }
    )

    for excluded_setup in setups:

        remaining = [
            trade
            for trade in trades
            if str(
                trade.get(
                    "Diagnostic_Setup",
                    trade.get(
                        "setup",
                        "UNKNOWN"
                    )
                )
            ) != excluded_setup
        ]

        if len(remaining) < 10:
            continue

        result = evaluate(
            remaining
        )

        print(
            f"EXCLUDE {excluded_setup:<25} "
            f"Trades={result['trades']:>3} "
            f"BaselineR={result['baseline_r']:>8.2f} "
            f"FilteredR={result['filtered_r']:>8.2f} "
            f"DeltaR={result['delta_r']:>8.2f}"
        )

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("FIXED ENTRY-QUALITY VALIDATION VERDICT")
    print("=" * 120)

    year_deltas = []

    for year in years:

        year_trades = [
            trade
            for trade in trades
            if str(
                trade.get(
                    "entry_date",
                    ""
                )
            ).startswith(str(year))
        ]

        if not year_trades:
            continue

        result = evaluate(
            year_trades
        )

        year_deltas.append(
            result["delta_r"]
        )

    positive_years = sum(
        1
        for delta in year_deltas
        if delta > 0
    )

    if (
        global_result["delta_r"] > 0
        and positive_years == len(year_deltas)
    ):
        print(
            "PROMISING: fixed 5B_0.25R improvement "
            "remains positive across all tested years."
        )
    elif global_result["delta_r"] > 0:
        print(
            "MIXED: fixed 5B_0.25R improves globally "
            "but is not positive in every tested year."
        )
    else:
        print(
            "WEAK: fixed 5B_0.25R does not improve globally."
        )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )
    print(
        "This diagnostic only tests robustness of the observed "
        "counterfactual relationship."
    )

def fixed_early_adverse_integrity_audit(trades):
    """
    Research-only integrity audit for the frozen 5B_0.25R
    Early Adverse-R rule.

    Frozen rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    Missing early_adverse_r_bar_5 values are retained.

    This audit does NOT:
        - change trade outcomes
        - change execution
        - optimize thresholds
        - select a strategy
        - modify production engine logic

    Purpose:
        Prove that the observed filtered result is internally
        consistent with the frozen rule and that no final-outcome
        field is being used to decide selection.
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25
    EPSILON = 1e-9

    def get_early_adverse(trade):
        return to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

    def get_trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=None
        )

        return value

    def trade_date(trade):
        return str(
            trade.get(
                "entry_date",
                ""
            )
        )

    def trade_symbol(trade):
        return str(
            trade.get(
                "Symbol",
                trade.get(
                    "symbol",
                    "UNKNOWN"
                )
            )
        )

    def trade_direction(trade):
        return str(
            trade.get(
                "Direction",
                trade.get(
                    "direction",
                    "UNKNOWN"
                )
            )
        )

    def trade_key(trade):
        return (
            trade_date(trade),
            trade_symbol(trade),
            trade_direction(trade),
        )

    # --------------------------------------------------------------
    # BASIC COUNTS
    # --------------------------------------------------------------

    total_trades = len(trades)

    missing_bar5 = []
    invalid_r = []

    for trade in trades:

        bar5 = get_early_adverse(trade)
        final_r = get_trade_r(trade)

        if bar5 is None:
            missing_bar5.append(trade)

        if final_r is None:
            invalid_r.append(trade)

    # --------------------------------------------------------------
    # FROZEN RULE CLASSIFICATION
    # --------------------------------------------------------------

    retained = []
    removed = []

    for trade in trades:

        bar5 = get_early_adverse(trade)

        if bar5 is None:
            retained.append(trade)

        elif bar5 < RULE_THRESHOLD:
            retained.append(trade)

        else:
            removed.append(trade)

    # --------------------------------------------------------------
    # RULE MEMBERSHIP CHECK
    # --------------------------------------------------------------

    rule_membership_failures = []

    for trade in removed:

        bar5 = get_early_adverse(trade)

        if (
            bar5 is None
            or bar5 < RULE_THRESHOLD
        ):
            rule_membership_failures.append(
                trade
            )

    for trade in retained:

        bar5 = get_early_adverse(trade)

        if (
            bar5 is not None
            and bar5 >= RULE_THRESHOLD
        ):
            rule_membership_failures.append(
                trade
            )

    # --------------------------------------------------------------
    # BOUNDARY CASES
    # --------------------------------------------------------------

    exact_boundary = []
    just_below_boundary = []
    just_above_boundary = []

    for trade in trades:

        bar5 = get_early_adverse(trade)

        if bar5 is None:
            continue

        if abs(bar5 - RULE_THRESHOLD) <= EPSILON:
            exact_boundary.append(trade)

        elif (
            RULE_THRESHOLD - 0.01
            <= bar5
            < RULE_THRESHOLD
        ):
            just_below_boundary.append(trade)

        elif (
            RULE_THRESHOLD
            < bar5
            <= RULE_THRESHOLD + 0.01
        ):
            just_above_boundary.append(trade)

    # --------------------------------------------------------------
    # WINNERS REMOVED
    # --------------------------------------------------------------

    removed_winners = [
        trade
        for trade in removed
        if (
            get_trade_r(trade) is not None
            and get_trade_r(trade) > 0
        )
    ]

    # --------------------------------------------------------------
    # LOSERS RETAINED
    # --------------------------------------------------------------

    retained_losers = [
        trade
        for trade in retained
        if (
            get_trade_r(trade) is not None
            and get_trade_r(trade) <= 0
        )
    ]

    # --------------------------------------------------------------
    # R ACCOUNTING
    # --------------------------------------------------------------

    baseline_r = sum(
        get_trade_r(trade)
        for trade in trades
        if get_trade_r(trade) is not None
    )

    retained_r = sum(
        get_trade_r(trade)
        for trade in retained
        if get_trade_r(trade) is not None
    )

    removed_r = sum(
        get_trade_r(trade)
        for trade in removed
        if get_trade_r(trade) is not None
    )

    reconstruction_r = (
        retained_r
        + removed_r
    )

    accounting_error = (
        baseline_r
        - reconstruction_r
    )

    delta_from_removed = (
        baseline_r
        - retained_r
    )

    delta_reconciliation_error = (
        delta_from_removed
        - removed_r
    )

    # --------------------------------------------------------------
    # TRADE COUNT ACCOUNTING
    # --------------------------------------------------------------

    count_reconstruction_ok = (
        len(retained)
        + len(removed)
        == total_trades
    )

    # --------------------------------------------------------------
    # DUPLICATE IDENTITY CHECK
    # --------------------------------------------------------------

    keys = [
        trade_key(trade)
        for trade in trades
    ]

    duplicate_keys = []

    seen = set()

    for key in keys:

        if key in seen:
            duplicate_keys.append(
                key
            )

        else:
            seen.add(key)

    # --------------------------------------------------------------
    # TEMPORAL FIELD AUDIT
    # --------------------------------------------------------------

    selection_fields = {
        "early_adverse_r_bar_5"
    }

    outcome_fields = {
        "r_multiple",
        "max_favorable_r",
        "Max_Favorable_R",
        "max_adverse_r",
        "Max_Adverse_R",
        "post_trigger_max_favorable_r",
        "post_trigger_max_adverse_r",
        "post_trigger_recovered_to_entry",
        "post_trigger_recovered_0_25r",
        "post_trigger_recovered_0_5r",
        "post_trigger_recovered_1r",
    }

    # The classification above explicitly uses only bar 5.
    # This is an implementation-level audit, not proof of how the
    # CSV field itself was originally generated.
    selection_uses_outcome_fields = False

    # --------------------------------------------------------------
    # PASS / FAIL
    # --------------------------------------------------------------

    checks = {
        "FIELD_COMPLETENESS_R":
            len(invalid_r) == 0,

        "RULE_MEMBERSHIP":
            len(rule_membership_failures) == 0,

        "TRADE_COUNT_RECONCILIATION":
            count_reconstruction_ok,

        "R_RECONCILIATION":
            abs(accounting_error) <= EPSILON,

        "DELTA_RECONCILIATION":
            abs(delta_reconciliation_error) <= EPSILON,

        "SELECTION_OUTCOME_LEAK":
            selection_uses_outcome_fields is False,
    }

    overall_pass = all(
        checks.values()
    )

    # --------------------------------------------------------------
    # REPORT
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "FIXED 5B_0.25R COUNTERFACTUAL INTEGRITY AUDIT"
    )
    print("=" * 120)

    print()
    print(
        "Frozen rule:"
    )

    print(
        "Reject when "
        f"early_adverse_r_bar_5 >= "
        f"{RULE_THRESHOLD:.2f}R"
    )

    print(
        "Missing Bar-5 values are retained."
    )

    print()
    print("-" * 120)
    print("DATA COMPLETENESS")
    print("-" * 120)

    print(
        f"Total trades: {total_trades}"
    )

    print(
        f"Missing Bar-5: "
        f"{len(missing_bar5)}"
    )

    print(
        f"Missing/invalid Final-R: "
        f"{len(invalid_r)}"
    )

    print()
    print("-" * 120)
    print("RULE CLASSIFICATION")
    print("-" * 120)

    print(
        f"Retained: {len(retained)}"
    )

    print(
        f"Removed: {len(removed)}"
    )

    print(
        f"Retained + Removed: "
        f"{len(retained) + len(removed)}"
    )

    print(
        f"Count reconciliation: "
        f"{'PASS' if count_reconstruction_ok else 'FAIL'}"
    )

    print()
    print("-" * 120)
    print("BOUNDARY AUDIT")
    print("-" * 120)

    print(
        f"Exactly 0.25R: "
        f"{len(exact_boundary)}"
    )

    print(
        f"0.24R <= Bar5 < 0.25R: "
        f"{len(just_below_boundary)}"
    )

    print(
        f"0.25R < Bar5 <= 0.26R: "
        f"{len(just_above_boundary)}"
    )

    if exact_boundary:

        print()
        print(
            "Exact-boundary trades:"
        )

        for trade in exact_boundary:

            print(
                f"Date={trade_date(trade)} | "
                f"Symbol={trade_symbol(trade)} | "
                f"Direction={trade_direction(trade)} | "
                f"Bar5={get_early_adverse(trade)} | "
                f"FinalR={get_trade_r(trade)}"
            )

    print()
    print("-" * 120)
    print("REMOVED WINNERS / RETAINED LOSERS")
    print("-" * 120)

    print(
        f"Removed winners: "
        f"{len(removed_winners)}"
    )

    print(
        f"Retained losers: "
        f"{len(retained_losers)}"
    )

    print()
    print("-" * 120)
    print("R ACCOUNTING")
    print("-" * 120)

    print(
        f"Baseline R: "
        f"{baseline_r:.10f}"
    )

    print(
        f"Retained R: "
        f"{retained_r:.10f}"
    )

    print(
        f"Removed R: "
        f"{removed_r:.10f}"
    )

    print(
        f"Retained + Removed R: "
        f"{reconstruction_r:.10f}"
    )

    print(
        f"Accounting error: "
        f"{accounting_error:.10f}"
    )

    print(
        f"Baseline - Retained: "
        f"{delta_from_removed:.10f}"
    )

    print(
        f"Delta reconciliation error: "
        f"{delta_reconciliation_error:.10f}"
    )

    print(
        f"R reconciliation: "
        f"{'PASS' if abs(accounting_error) <= EPSILON else 'FAIL'}"
    )

    print(
        f"Delta reconciliation: "
        f"{'PASS' if abs(delta_reconciliation_error) <= EPSILON else 'FAIL'}"
    )

    print()
    print("-" * 120)
    print("RULE MEMBERSHIP")
    print("-" * 120)

    print(
        f"Membership failures: "
        f"{len(rule_membership_failures)}"
    )

    print(
        f"Rule membership: "
        f"{'PASS' if len(rule_membership_failures) == 0 else 'FAIL'}"
    )

    print()
    print("-" * 120)
    print("DUPLICATE TRADE IDENTITY CHECK")
    print("-" * 120)

    print(
        f"Duplicate Date/Symbol/Direction keys: "
        f"{len(duplicate_keys)}"
    )

    if duplicate_keys:

        for key in duplicate_keys[:20]:

            print(
                f"Duplicate: "
                f"Date={key[0]} | "
                f"Symbol={key[1]} | "
                f"Direction={key[2]}"
            )

    print()
    print("-" * 120)
    print("TEMPORAL / OUTCOME-LEAK CHECK")
    print("-" * 120)

    print(
        "Selection decision uses: "
        "early_adverse_r_bar_5 only"
    )

    print(
        "Final-R/MFE/MAE/recovery fields used "
        "for selection: NO"
    )

    print(
        "Implementation-level outcome leak: "
        f"{'PASS' if selection_uses_outcome_fields is False else 'FAIL'}"
    )

    print()
    print("-" * 120)
    print("FINAL INTEGRITY VERDICT")
    print("-" * 120)

    for name, passed in checks.items():

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if overall_pass:

        print(
            "OVERALL: PASS"
        )

        print(
            "The frozen 5B_0.25R counterfactual "
            "classification is internally consistent."
        )

    else:

        print(
            "OVERALL: FAIL"
        )

        print(
            "Do NOT proceed to holdout validation "
            "until the failed integrity checks are resolved."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This audit verifies the analyzer's "
        "classification and accounting."
    )

    print(
        "It does NOT by itself prove that "
        "early_adverse_r_bar_5 was generated "
        "without future-bar leakage upstream."
    )

    print(
        "Upstream generation logic must be inspected "
        "separately for that final temporal-integrity claim."
    )

    print("=" * 120)

def fixed_early_adverse_trade_level_audit(trades):
    """
    Research-only trade-level audit.

    Purpose:
        Compare trades removed by the fixed 5B_0.25R rule
        against trades retained by the rule.

    Fixed rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    This diagnostic:
        - does not modify trade outcomes
        - does not modify execution
        - does not select a rule
        - does not optimize subgroups
        - does not change the trading engine
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    def rule_passes(trade):
        value = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if value is None:
            return True

        return value < RULE_THRESHOLD

    def trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=0.0
        )

        return value if value is not None else 0.0

    def trade_date(trade):
        return str(
            trade.get(
                "entry_date",
                ""
            )
        )

    def trade_symbol(trade):
        return str(
            trade.get(
                "Symbol",
                trade.get(
                    "symbol",
                    "UNKNOWN"
                )
            )
        )

    def trade_direction(trade):
        return str(
            trade.get(
                "Direction",
                trade.get(
                    "direction",
                    "UNKNOWN"
                )
            )
        )

    def average(values):
        if not values:
            return None

        return sum(values) / len(values)

    retained = [
        trade
        for trade in trades
        if rule_passes(trade)
    ]

    removed = [
        trade
        for trade in trades
        if not rule_passes(trade)
    ]

    print()
    print("=" * 120)
    print(
        "FIXED EARLY-ADVERSE-R TRADE-LEVEL AUDIT"
    )
    print("=" * 120)

    print()
    print(
        "Fixed rule: "
        f"{RULE_WINDOW} bars / "
        f"{RULE_THRESHOLD:.2f}R"
    )

    print(
        "Purpose: compare removed trades against retained trades."
    )

    def print_group(name, group):
        total_r = sum(
            trade_r(trade)
            for trade in group
        )

        winners = [
            trade
            for trade in group
            if trade_r(trade) > 0
        ]

        losers = [
            trade
            for trade in group
            if trade_r(trade) <= 0
        ]

        win_rate = (
            len(winners) / len(group) * 100
            if group
            else 0.0
        )

        early_triggered = [
            trade
            for trade in group
            if to_bool(
                trade.get(
                    "early_adverse_triggered"
                )
            )
        ]

        recovered_entry = [
            trade
            for trade in group
            if to_bool(
                trade.get(
                    "post_trigger_recovered_to_entry"
                )
            )
        ]

        recovered_025 = [
            trade
            for trade in group
            if to_bool(
                trade.get(
                    "post_trigger_recovered_0_25r"
                )
            )
        ]

        recovered_050 = [
            trade
            for trade in group
            if to_bool(
                trade.get(
                    "post_trigger_recovered_0_5r"
                )
            )
        ]

        recovered_1r = [
            trade
            for trade in group
            if to_bool(
                trade.get(
                    "post_trigger_recovered_1r"
                )
            )
        ]

        mfe_values = [
            to_float(
                trade.get(
                    "post_trigger_max_favorable_r"
                )
            )
            for trade in group
            if to_float(
                trade.get(
                    "post_trigger_max_favorable_r"
                )
            ) is not None
        ]

        mae_values = [
            to_float(
                trade.get(
                    "post_trigger_max_adverse_r"
                )
            )
            for trade in group
            if to_float(
                trade.get(
                    "post_trigger_max_adverse_r"
                )
            ) is not None
        ]

        print()
        print(
            "-" * 120
        )
        print(
            f"{name}"
        )
        print(
            "-" * 120
        )

        print(
            f"Trades: {len(group)}"
        )

        print(
            f"Total R: {total_r:.2f}"
        )

        print(
            f"Avg R: "
            f"{average([trade_r(trade) for trade in group]):.2f}"
            if group
            else "Avg R: N/A"
        )

        print(
            f"Winners: {len(winners)}"
        )

        print(
            f"Losers: {len(losers)}"
        )

        print(
            f"Win Rate: {win_rate:.2f}%"
        )

        print(
            f"Early-Adverse Triggered: "
            f"{len(early_triggered)} "
            f"({len(early_triggered) / len(group) * 100:.2f}%)"
            if group
            else "Early-Adverse Triggered: 0"
        )

        print(
            f"Recovered to Entry: "
            f"{len(recovered_entry)} "
            f"({len(recovered_entry) / len(group) * 100:.2f}%)"
            if group
            else "Recovered to Entry: 0"
        )

        print(
            f"Recovered to +0.25R: "
            f"{len(recovered_025)} "
            f"({len(recovered_025) / len(group) * 100:.2f}%)"
            if group
            else "Recovered to +0.25R: 0"
        )

        print(
            f"Recovered to +0.50R: "
            f"{len(recovered_050)} "
            f"({len(recovered_050) / len(group) * 100:.2f}%)"
            if group
            else "Recovered to +0.50R: 0"
        )

        print(
            f"Recovered to +1.00R: "
            f"{len(recovered_1r)} "
            f"({len(recovered_1r) / len(group) * 100:.2f}%)"
            if group
            else "Recovered to +1.00R: 0"
        )

        print(
            f"Avg Post-Trigger MFE: "
            f"{average(mfe_values):.2f}R"
            if mfe_values
            else "Avg Post-Trigger MFE: N/A"
        )

        print(
            f"Avg Post-Trigger MAE: "
            f"{average(mae_values):.2f}R"
            if mae_values
            else "Avg Post-Trigger MAE: N/A"
        )

    print_group(
        "RETAINED TRADES",
        retained
    )

    print_group(
        "REMOVED TRADES",
        removed
    )

    # --------------------------------------------------------------
    # REMOVED TRADE DETAIL
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print(
        "REMOVED TRADE DETAIL"
    )
    print("=" * 120)

    print(
        "Date | Symbol | Direction | "
        "Bar1 | Bar2 | Bar3 | Bar4 | Bar5 | "
        "FinalR | MFE | MAE | RecoveredEntry"
    )

    print("-" * 120)

    for trade in sorted(
        removed,
        key=trade_date
    ):

        bar_values = []

        for bar_number in range(1, 6):
            value = to_float(
                trade.get(
                    f"early_adverse_r_bar_{bar_number}"
                ),
                default=None
            )

            if value is None:
                bar_values.append(
                    "-"
                )
            else:
                bar_values.append(
                    f"{value:.2f}"
                )

        final_r = trade_r(
            trade
        )

        mfe = to_float(
            trade.get(
                "post_trigger_max_favorable_r"
            ),
            default=None
        )

        mae = to_float(
            trade.get(
                "post_trigger_max_adverse_r"
            ),
            default=None
        )

        recovered = to_bool(
            trade.get(
                "post_trigger_recovered_to_entry"
            )
        )

        print(
            f"{trade_date(trade)} | "
            f"{trade_symbol(trade)} | "
            f"{trade_direction(trade)} | "
            f"{bar_values[0]} | "
            f"{bar_values[1]} | "
            f"{bar_values[2]} | "
            f"{bar_values[3]} | "
            f"{bar_values[4]} | "
            f"{final_r:.2f} | "
            f"{mfe:.2f}" if mfe is not None
            else
            f"{trade_date(trade)} | "
            f"{trade_symbol(trade)} | "
            f"{trade_direction(trade)} | "
            f"{bar_values[0]} | "
            f"{bar_values[1]} | "
            f"{bar_values[2]} | "
            f"{bar_values[3]} | "
            f"{bar_values[4]} | "
            f"{final_r:.2f} | -",
            end=""
        )

        print(
            f" | "
            f"{mae:.2f}" if mae is not None
            else
            " | -",
            end=""
        )

        print(
            f" | "
            f"{recovered}"
        )

    print()
    print(
        "Research only — no execution rule changed."
    )

def fixed_early_adverse_stock_level_diagnostic(trades):
    """
    Research-only stock-level diagnostic.

    Purpose:
        Evaluate the same frozen 5B_0.25R Early Adverse-R rule
        independently across each stock.

    Fixed rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    IMPORTANT:
        - No stock-specific thresholds.
        - No stock-specific optimization.
        - No production engine changes.
        - No execution changes.
        - Descriptive validation only.
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    def trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=0.0
        )

        return value if value is not None else 0.0

    def trade_symbol(trade):
        value = trade.get("_symbol")

        if value is None:
            value = trade.get("symbol")

        if value is None:
            value = trade.get("Symbol")

        if value is None:
            value = trade.get("ticker")

        if value is None:
            value = trade.get("Ticker")

        if value is None:
            value = "UNKNOWN"

        return str(value)

    def rule_passes(trade):
        value = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if value is None:
            return True

        return value < RULE_THRESHOLD

    def evaluate(group_trades):
        baseline_r = sum(
            trade_r(trade)
            for trade in group_trades
        )

        retained = [
            trade
            for trade in group_trades
            if rule_passes(trade)
        ]

        removed = [
            trade
            for trade in group_trades
            if not rule_passes(trade)
        ]

        filtered_r = sum(
            trade_r(trade)
            for trade in retained
        )

        removed_r = sum(
            trade_r(trade)
            for trade in removed
        )

        winners_retained = sum(
            1
            for trade in retained
            if trade_r(trade) > 0
        )

        winners_removed = sum(
            1
            for trade in removed
            if trade_r(trade) > 0
        )

        return {
            "trades": len(group_trades),
            "retained": len(retained),
            "removed": len(removed),
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "removed_r": removed_r,
            "delta_r": filtered_r - baseline_r,
            "retained_win_rate": (
                winners_retained
                / len(retained)
                * 100
                if retained
                else 0.0
            ),
            "removed_win_rate": (
                winners_removed
                / len(removed)
                * 100
                if removed
                else 0.0
            ),
        }

    groups = {}

    for trade in trades:
        symbol = trade_symbol(trade)

        groups.setdefault(
            symbol,
            []
        ).append(trade)

    print()
    print("=" * 150)
    print(
        "FIXED EARLY-ADVERSE-R × STOCK VALIDATION"
    )
    print("=" * 150)

    print()
    print(
        "Fixed rule: "
        f"{RULE_WINDOW} bars / "
        f"{RULE_THRESHOLD:.2f}R"
    )

    print(
        "Purpose: determine whether the fixed rule's "
        "performance improvement is distributed across stocks."
    )

    print(
        "IMPORTANT: no stock-specific rule or threshold is being selected."
    )

    print()
    print("-" * 150)

    print(
        f"{'STOCK':<18}"
        f"{'TRADES':>8}"
        f"{'RETAINED':>10}"
        f"{'REMOVED':>9}"
        f"{'BASELINE R':>13}"
        f"{'FILTERED R':>13}"
        f"{'REMOVED R':>12}"
        f"{'DELTA R':>11}"
        f"{'RETAIN WIN%':>13}"
        f"{'REMOVED WIN%':>14}"
    )

    print("-" * 150)

    stock_results = []

    for symbol in sorted(
        groups.keys()
    ):
        group_trades = groups[symbol]

        result = evaluate(
            group_trades
        )

        stock_results.append(
            (
                symbol,
                result
            )
        )

        print(
            f"{symbol:<18}"
            f"{result['trades']:>8}"
            f"{result['retained']:>10}"
            f"{result['removed']:>9}"
            f"{result['baseline_r']:>13.2f}"
            f"{result['filtered_r']:>13.2f}"
            f"{result['removed_r']:>12.2f}"
            f"{result['delta_r']:>11.2f}"
            f"{result['retained_win_rate']:>13.2f}"
            f"{result['removed_win_rate']:>14.2f}"
        )

    print("-" * 150)

    # --------------------------------------------------------------
    # STOCK CONTRIBUTION TO GLOBAL DELTA
    # --------------------------------------------------------------

    global_delta = sum(
        result["delta_r"]
        for _, result in stock_results
    )

    print()
    print("-" * 150)
    print(
        "STOCK CONTRIBUTION TO GLOBAL DELTA"
    )
    print("-" * 150)

    print(
        f"{'STOCK':<18}"
        f"{'DELTA R':>12}"
        f"{'SHARE OF TOTAL DELTA':>24}"
    )

    print("-" * 150)

    for symbol, result in sorted(
        stock_results,
        key=lambda item: item[1]["delta_r"],
        reverse=True
    ):
        contribution = (
            result["delta_r"]
            / global_delta
            * 100
            if global_delta != 0
            else 0.0
        )

        print(
            f"{symbol:<18}"
            f"{result['delta_r']:>12.2f}"
            f"{contribution:>23.2f}%"
        )

    # --------------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "RESEARCH INTERPRETATION"
    )
    print("=" * 150)

    positive_stocks = [
        symbol
        for symbol, result in stock_results
        if result["delta_r"] > 0
    ]

    negative_stocks = [
        symbol
        for symbol, result in stock_results
        if result["delta_r"] < 0
    ]

    print()

    print(
        f"Stocks with positive DeltaR: "
        f"{len(positive_stocks)}"
    )

    print(
        f"Stocks with negative DeltaR: "
        f"{len(negative_stocks)}"
    )

    if positive_stocks:
        print(
            "Positive DeltaR stocks: "
            + ", ".join(
                positive_stocks
            )
        )

    if negative_stocks:
        print(
            "Negative DeltaR stocks: "
            + ", ".join(
                negative_stocks
            )
        )

    print()

    print(
        "This diagnostic is descriptive only."
    )

    print(
        "A strong result in one stock does NOT justify "
        "creating a stock-specific Early-Adverse-R rule."
    )

    print(
        "The next validation step, if warranted, is temporal "
        "holdout testing rather than stock-specific optimization."
    )

    print()

    print(
        "RESEARCH STATUS: DIAGNOSTIC ONLY"
    )

    print(
        "NO PRODUCTION ENGINE CHANGES"
    )

    print("=" * 150)

def early_adverse_r_sensitivity_diagnostic(trades):
    """
    Research-only counterfactual analysis.

    Tests whether excluding trades that experience a specified
    amount of adverse movement within the first N bars improves
    historical performance.

    IMPORTANT:
    This does NOT modify the engine or actual trade execution.
    It only evaluates hypothetical exclusions on completed trades.
    """

    thresholds = [
        0.25,
        0.35,
        0.50,
        0.65,
        0.75,
    ]

    windows = [
        1,
        2,
        3,
        4,
        5,
    ]

    print()
    print("=" * 150)
    print("EARLY ADVERSE-R SENSITIVITY / COUNTERFACTUAL FILTER")
    print("=" * 150)

    print()
    print(
        "Interpretation: exclude a trade if cumulative maximum "
        "adverse movement reaches the threshold within the first N bars."
    )

    print()
    print(
        "This is research-only. No engine logic or historical trade "
        "execution has been changed."
    )

    print()
    print("-" * 150)
    print("OVERALL SENSITIVITY GRID")
    print("-" * 150)

    print(
        f"{'WINDOW':<8}"
        f"{'THRESH':<9}"
        f"{'FLAGGED':<9}"
        f"{'FLAGGED WIN':<13}"
        f"{'FLAGGED LOSS':<14}"
        f"{'WIN REMOVED':<13}"
        f"{'LOSS REMOVED':<14}"
        f"{'REMAIN':<9}"
        f"{'WIN%':<8}"
        f"{'PF':<8}"
        f"{'TOTAL R':<10}"
        f"{'AVG R':<9}"
    )

    print("-" * 150)

    results = []

    for window in windows:

        column = (
            f"early_adverse_r_bar_{window}"
        )

        for threshold in thresholds:

            flagged = []
            remaining = []

            for trade in trades:

                value = to_float(
                    trade.get(column)
                )

                if (
                    value is not None
                    and value >= threshold
                ):
                    flagged.append(trade)
                else:
                    remaining.append(trade)

            flagged_wins = [
                trade
                for trade in flagged
                if (to_float(trade.get("r_multiple")) or 0.0) > 0
            ]

            flagged_losses = [
                trade
                for trade in flagged
                if (to_float(trade.get("r_multiple")) or 0.0) <= 0
            ]

            remaining_r = [
                to_float(trade.get("r_multiple")) or 0.0
                for trade in remaining
            ]

            remaining_wins = [
                r
                for r in remaining_r
                if r > 0
            ]

            remaining_losses = [
                r
                for r in remaining_r
                if r < 0
            ]

            gross_profit = sum(
                remaining_wins
            )

            gross_loss = abs(
                sum(remaining_losses)
            )

            profit_factor = (
                gross_profit / gross_loss
                if gross_loss > 0
                else 0.0
            )

            total_r = sum(
                remaining_r
            )

            avg_r = (
                total_r / len(remaining)
                if remaining
                else 0.0
            )

            win_rate = (
                len(remaining_wins)
                / len(remaining)
                * 100
                if remaining
                else 0.0
            )

            result = {
                "window": window,
                "threshold": threshold,
                "flagged": len(flagged),
                "flagged_wins": len(flagged_wins),
                "flagged_losses": len(flagged_losses),
                "remaining": len(remaining),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_r": total_r,
                "avg_r": avg_r,
            }

            results.append(result)

            print(
                f"{window:<8}"
                f"{threshold:<9.2f}"
                f"{len(flagged):<9}"
                f"{len(flagged_wins):<13}"
                f"{len(flagged_losses):<14}"
                f"{len(flagged_wins):<13}"
                f"{len(flagged_losses):<14}"
                f"{len(remaining):<9}"
                f"{win_rate:<8.2f}"
                f"{profit_factor:<8.2f}"
                f"{total_r:<10.2f}"
                f"{avg_r:<9.3f}"
            )

    # ------------------------------------------------------------------
    # BEST COUNTERFACTUAL RESULTS
    # ------------------------------------------------------------------

    print()
    print("-" * 150)
    print("BEST COUNTERFACTUAL RESULTS BY TOTAL R")
    print("-" * 150)

    for result in sorted(
        results,
        key=lambda item: (
            -item["total_r"],
            -item["profit_factor"],
        )
    )[:10]:

        print(
            f"Window={result['window']} "
            f"Threshold={result['threshold']:.2f}R "
            f"Flagged={result['flagged']} "
            f"Remaining={result['remaining']} "
            f"Win={result['win_rate']:.2f}% "
            f"PF={result['profit_factor']:.2f} "
            f"R={result['total_r']:.2f} "
            f"AvgR={result['avg_r']:.3f}"
        )

    # ------------------------------------------------------------------
    # TRAINING / HOLDOUT / 2026 VALIDATION
    # ------------------------------------------------------------------

    print()
    print("-" * 150)
    print("TRAINING / HOLDOUT / 2026 VALIDATION")
    print("-" * 150)

    periods = [
        (
            "TRAINING",
            lambda year: year < 2025,
        ),
        (
            "HOLDOUT",
            lambda year: year >= 2025,
        ),
        (
            "2026",
            lambda year: year == 2026,
        ),
    ]

    for period_name, period_filter in periods:

        period_trades = []

        for trade in trades:

            entry_date = trade.get(
                "entry_date",
                ""
            )

            try:
                year = int(
                    str(entry_date)[:4]
                )
            except (
                ValueError,
                TypeError
            ):
                continue

            if period_filter(year):
                period_trades.append(trade)

        print()
        print(
            f"{period_name}"
        )

        print(
            f"{'WINDOW':<8}"
            f"{'THRESH':<9}"
            f"{'FLAGGED':<9}"
            f"{'REMAIN':<9}"
            f"{'WIN%':<8}"
            f"{'PF':<8}"
            f"{'TOTAL R':<10}"
            f"{'AVG R':<9}"
        )

        print("-" * 90)

        for window in windows:

            column = (
                f"early_adverse_r_bar_{window}"
            )

            for threshold in thresholds:

                remaining = []

                flagged_count = 0

                for trade in period_trades:

                    value = to_float(
                        trade.get(column)
                    )

                    if (
                        value is not None
                        and value >= threshold
                    ):
                        flagged_count += 1
                    else:
                        remaining.append(
                            trade
                        )

                r_values = [
                    to_float(
                        trade.get("r_multiple")
                    ) or 0.0
                    for trade in remaining
                ]

                wins = [
                    r
                    for r in r_values
                    if r > 0
                ]

                losses = [
                    r
                    for r in r_values
                    if r < 0
                ]

                gross_profit = sum(
                    wins
                )

                gross_loss = abs(
                    sum(losses)
                )

                profit_factor = (
                    gross_profit / gross_loss
                    if gross_loss > 0
                    else 0.0
                )

                total_r = sum(
                    r_values
                )

                avg_r = (
                    total_r / len(r_values)
                    if r_values
                    else 0.0
                )

                win_rate = (
                    len(wins)
                    / len(r_values)
                    * 100
                    if r_values
                    else 0.0
                )

                print(
                    f"{window:<8}"
                    f"{threshold:<9.2f}"
                    f"{flagged_count:<9}"
                    f"{len(remaining):<9}"
                    f"{win_rate:<8.2f}"
                    f"{profit_factor:<8.2f}"
                    f"{total_r:<10.2f}"
                    f"{avg_r:<9.3f}"
                )

def entry_quality_early_adverse_walk_forward_diagnostic(trades):
    """
    Research-only combined walk-forward validation.

    Combines:
        1. ENTRY-TIME quality filter
        2. EARLY ADVERSE-R delayed path filter

    IMPORTANT:
        - Entry-quality variables are available at entry.
        - Early Adverse-R is treated as a delayed executable condition.
        - Parameters are selected ONLY from prior years.
        - No production engine logic is modified.
    """

    print()
    print("=" * 150)
    print("ENTRY QUALITY + EARLY ADVERSE-R WALK-FORWARD VALIDATION")
    print("=" * 150)

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------

    def to_number(value):
        try:
            if value is None:
                return None

            if isinstance(value, str):
                value = value.strip()

                if value == "":
                    return None

                if value.lower() in {
                    "none",
                    "nan",
                    "null",
                }:
                    return None

            return float(value)

        except (TypeError, ValueError):
            return None

    def get_year(trade):
        try:
            return int(
                str(
                    trade.get(
                        "entry_date",
                        ""
                    )
                )[:4]
            )
        except (TypeError, ValueError):
            return None

    def get_setup(trade):
        return str(
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        )

    def get_trend(trade):
        return str(
            trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        )

    def get_strength(trade):
        return to_number(
            trade.get(
                "entry_trend_strength"
            )
        )

    def get_slope(trade):
        return to_number(
            trade.get(
                "entry_trend_slope"
            )
        )

    def get_price_position(trade):
        value = trade.get(
            "entry_price_position_value"
        )

        if value is None:
            value = trade.get(
                "Price_Position_Value"
            )

        return to_number(value)

    def get_structure_bias(trade):
        return str(
            trade.get(
                "structure_bias",
                trade.get(
                    "Structure_Bias",
                    "UNKNOWN"
                )
            )
        )

    def get_liquidity(trade):
        value = trade.get(
            "liquidity_sweep",
            trade.get(
                "Liquidity_Sweep"
            )
        )

        if isinstance(value, bool):
            return value

        return str(value).strip().lower() in {
            "true",
            "1",
            "yes",
            "y"
        }

    def get_break_age(trade):
        return to_number(
            trade.get(
                "break_age",
                trade.get(
                    "Break_Age"
                )
            )
        )

    def get_r(trade):
        value = to_number(
            trade.get(
                "r_multiple"
            )
        )

        return value if value is not None else 0.0

    # --------------------------------------------------------------
    # ENTRY-QUALITY CANDIDATES
    #
    # Keep this deliberately small.
    # We are testing robustness, not searching hundreds of rules.
    # --------------------------------------------------------------

    entry_filters = [
        (
            "NO_ENTRY_FILTER",
            lambda trade: True
        ),

        (
            "EXCLUDE_SHORT_REVERSAL",
            lambda trade:
                get_setup(trade) != "SHORT_REVERSAL"
        ),

        (
            "EXCLUDE_LONG_CONTINUATION",
            lambda trade:
                get_setup(trade) != "LONG_CONTINUATION"
        ),

        (
            "EXCLUDE_BOTH_WEAK_SETUPS",
            lambda trade:
                get_setup(trade) not in {
                    "SHORT_REVERSAL",
                    "LONG_CONTINUATION"
                }
        ),

        (
            "TREND_NOT_NEUTRAL",
            lambda trade:
                get_trend(trade) != "NEUTRAL"
        ),

        (
            "STRENGTH_GE_40",
            lambda trade:
                (
                    get_strength(trade) is None
                    or get_strength(trade) >= 40
                )
        ),

        (
            "BREAK_AGE_LE_3",
            lambda trade:
                (
                    get_break_age(trade) is None
                    or get_break_age(trade) <= 3
                )
        ),

        (
            "STRUCTURE_ALIGNED",
            lambda trade:
                (
                    get_structure_bias(trade) == "UNKNOWN"
                    or (
                        get_structure_bias(trade) == "BULLISH"
                        and get_trend(trade) == "BULLISH"
                    )
                    or (
                        get_structure_bias(trade) == "BEARISH"
                        and get_trend(trade) == "BEARISH"
                    )
                )
        ),
    ]

    # --------------------------------------------------------------
    # EARLY ADVERSE-R CANDIDATES
    # --------------------------------------------------------------

    adverse_filters = [
        ("NO_ADVERSE_FILTER", None, None),
        ("1B_0.25R", 1, 0.25),
        ("2B_0.25R", 2, 0.25),
        ("3B_0.25R", 3, 0.25),
        ("4B_0.25R", 4, 0.25),
        ("5B_0.25R", 5, 0.25),
        ("5B_0.35R", 5, 0.35),
        ("5B_0.50R", 5, 0.50),
    ]

    MIN_TRAINING_TRADES = 15
    MIN_SELECTED_TRADES = 10

    def passes_adverse_filter(
        trade,
        bars,
        threshold
    ):
        if bars is None:
            return True

        value = to_number(
            trade.get(
                f"early_adverse_r_bar_{bars}"
            )
        )

        # Preserve existing behavior:
        # missing adverse-path information is retained.
        if value is None:
            return True

        return value < threshold

    def evaluate(
        period_trades,
        entry_filter,
        adverse_bars,
        adverse_threshold
    ):
        selected = []

        for trade in period_trades:

            if not entry_filter(trade):
                continue

            if not passes_adverse_filter(
                trade,
                adverse_bars,
                adverse_threshold
            ):
                continue

            selected.append(trade)

        r_values = [
            get_r(trade)
            for trade in selected
        ]

        wins = [
            r
            for r in r_values
            if r > 0
        ]

        losses = [
            r
            for r in r_values
            if r < 0
        ]

        gross_profit = sum(wins)
        gross_loss = abs(
            sum(losses)
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        total_r = sum(r_values)

        win_rate = (
            len(wins)
            / len(r_values)
            * 100
            if r_values
            else 0.0
        )

        return {
            "trades": len(selected),
            "wins": len(wins),
            "win_rate": win_rate,
            "pf": profit_factor,
            "total_r": total_r,
        }

    valid_trades = [
        trade
        for trade in trades
        if get_year(trade) is not None
    ]

    validation_years = sorted(
        set(
            get_year(trade)
            for trade in valid_trades
            if get_year(trade) >= 2024
        )
    )

    total_baseline_r = 0.0
    total_filtered_r = 0.0
    total_delta_r = 0.0
    improved_years = 0
    tested_years = 0

    # --------------------------------------------------------------
    # WALK FORWARD
    # --------------------------------------------------------------

    for validation_year in validation_years:

        training_trades = [
            trade
            for trade in valid_trades
            if get_year(trade) < validation_year
        ]

        validation_trades = [
            trade
            for trade in valid_trades
            if get_year(trade) == validation_year
        ]

        print()
        print("-" * 150)
        print(
            f"TRAINING: < {validation_year} "
            f"→ VALIDATION: {validation_year}"
        )
        print("-" * 150)

        print(
            f"Training trades   : "
            f"{len(training_trades)}"
        )

        print(
            f"Validation trades : "
            f"{len(validation_trades)}"
        )

        if len(training_trades) < MIN_TRAINING_TRADES:
            print(
                "SKIPPED: insufficient training sample."
            )
            continue

        # ----------------------------------------------------------
        # TRAINING GRID
        # ----------------------------------------------------------

        candidates = []

        for (
            entry_name,
            entry_filter
        ) in entry_filters:

            for (
                adverse_name,
                adverse_bars,
                adverse_threshold
            ) in adverse_filters:

                metrics = evaluate(
                    training_trades,
                    entry_filter,
                    adverse_bars,
                    adverse_threshold
                )

                if (
                    metrics["trades"]
                    < MIN_SELECTED_TRADES
                ):
                    continue

                candidates.append(
                    (
                        entry_name,
                        entry_filter,
                        adverse_name,
                        adverse_bars,
                        adverse_threshold,
                        metrics
                    )
                )

        if not candidates:
            print(
                "SKIPPED: no combined candidate "
                "has enough training trades."
            )
            continue

        # ----------------------------------------------------------
        # SELECT BEST TRAINING COMBINATION
        # ----------------------------------------------------------

        best = max(
            candidates,
            key=lambda item: (
                item[5]["total_r"],
                item[5]["pf"],
                item[5]["trades"]
            )
        )

        (
            best_entry_name,
            best_entry_filter,
            best_adverse_name,
            best_adverse_bars,
            best_adverse_threshold,
            best_training_metrics
        ) = best

        # ----------------------------------------------------------
        # BASELINE
        # ----------------------------------------------------------

        baseline_validation = evaluate(
            validation_trades,
            lambda trade: True,
            None,
            None
        )

        # ----------------------------------------------------------
        # SELECTED COMBINATION ON UNSEEN DATA
        # ----------------------------------------------------------

        filtered_validation = evaluate(
            validation_trades,
            best_entry_filter,
            best_adverse_bars,
            best_adverse_threshold
        )

        delta_r = (
            filtered_validation["total_r"]
            - baseline_validation["total_r"]
        )

        tested_years += 1

        total_baseline_r += (
            baseline_validation["total_r"]
        )

        total_filtered_r += (
            filtered_validation["total_r"]
        )

        total_delta_r += delta_r

        if delta_r > 0:
            improved_years += 1

        print()
        print("TRAINING SELECTION")

        print(
            f"Entry filter      : "
            f"{best_entry_name}"
        )

        print(
            f"Early Adverse-R   : "
            f"{best_adverse_name}"
        )

        print(
            f"Training result   : "
            f"Trades={best_training_metrics['trades']} "
            f"Win={best_training_metrics['win_rate']:.2f}% "
            f"PF={best_training_metrics['pf']:.2f} "
            f"R={best_training_metrics['total_r']:+.2f}"
        )

        print()
        print("UNSEEN VALIDATION")

        print(
            f"Baseline          : "
            f"Trades={baseline_validation['trades']} "
            f"Win={baseline_validation['win_rate']:.2f}% "
            f"PF={baseline_validation['pf']:.2f} "
            f"R={baseline_validation['total_r']:+.2f}"
        )

        print(
            f"Combined filter   : "
            f"Trades={filtered_validation['trades']} "
            f"Win={filtered_validation['win_rate']:.2f}% "
            f"PF={filtered_validation['pf']:.2f} "
            f"R={filtered_validation['total_r']:+.2f}"
        )

        print(
            f"Validation ΔR     : "
            f"{delta_r:+.2f}"
        )

        print(
            "Validation verdict: "
            f"{'IMPROVED' if delta_r > 0 else 'WORSE' if delta_r < 0 else 'UNCHANGED'}"
        )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "ENTRY QUALITY + EARLY ADVERSE-R "
        "WALK-FORWARD SUMMARY"
    )
    print("=" * 150)

    print(
        f"Validation years tested : "
        f"{tested_years}"
    )

    print(
        f"Baseline validation R   : "
        f"{total_baseline_r:+.2f}"
    )

    print(
        f"Combined validation R   : "
        f"{total_filtered_r:+.2f}"
    )

    print(
        f"Total validation ΔR     : "
        f"{total_delta_r:+.2f}"
    )

    print(
        f"Years improved          : "
        f"{improved_years}/{tested_years}"
    )

    print()
    print("RESEARCH RULE")

    if (
        tested_years > 0
        and total_delta_r > 0
        and improved_years >= 2
    ):
        print(
            "PROMISING — combined entry-quality + "
            "Early Adverse-R filter improved at least "
            "2 unseen validation years."
        )
        print(
            "DO NOT DEPLOY YET — continue robustness testing."
        )
    else:
        print(
            "NOT ROBUST ENOUGH — combined filter does not "
            "yet provide sufficient out-of-sample evidence."
        )
        print(
            "DO NOT DEPLOY — continue research."
        )

def frozen_global_early_adverse_robustness_diagnostic(trades):
    """
    Research-only robustness test.

    Purpose:
        Test whether a SINGLE frozen Early Adverse-R rule remains
        useful across years and entry-time contexts.

    This diagnostic does NOT:
        - optimize a separate rule for each subgroup
        - modify production engine logic
        - modify historical execution
        - use future information as an entry-time feature

    The Early Adverse-R variable is treated as a post-entry
    diagnostic / hypothetical exclusion rule.

    Candidate rules:
        NO_FILTER
        1B_0.25R
        2B_0.25R
        3B_0.25R
        4B_0.25R
        5B_0.25R
        5B_0.35R
        5B_0.50R
    """

    candidates = [
        ("NO_FILTER", None, None),
        ("1B_0.25R", 1, 0.25),
        ("2B_0.25R", 2, 0.25),
        ("3B_0.25R", 3, 0.25),
        ("4B_0.25R", 4, 0.25),
        ("5B_0.25R", 5, 0.25),
        ("5B_0.35R", 5, 0.35),
        ("5B_0.50R", 5, 0.50),
    ]

    validation_years = [
        2024,
        2025,
        2026,
    ]

    context_features = [
        "SETUP",
        "ENTRY_TREND",
        "TREND_STRENGTH",
        "TREND_SLOPE",
        "PRICE_POSITION",
        "STRUCTURE_BIAS",
        "LIQUIDITY_SWEEP",
        "BREAK_AGE",
    ]

    MIN_GROUP_TRADES = 8

    def to_number(value):
        try:
            if value is None:
                return None

            if isinstance(value, str):
                value = value.strip()

                if value == "":
                    return None

                if value.lower() in {
                    "none",
                    "nan",
                    "null",
                }:
                    return None

            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    def trade_year(trade):
        entry_date = str(
            trade.get(
                "entry_date",
                ""
            )
        ).strip()

        if len(entry_date) < 4:
            return None

        try:
            return int(
                entry_date[:4]
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    def get_r(trade):
        value = to_number(
            trade.get(
                "r_multiple"
            )
        )

        return (
            value
            if value is not None
            else 0.0
        )

    def get_setup(trade):
        return str(
            trade.get(
                "Diagnostic_Setup",
                trade.get(
                    "setup",
                    "UNKNOWN"
                )
            )
        )

    def get_trend(trade):
        return str(
            trade.get(
                "entry_trend",
                "UNKNOWN"
            )
        )

    def get_strength_bucket(trade):
        value = to_number(
            trade.get(
                "entry_trend_strength"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < 20:
            return "<20"

        if value < 40:
            return "20-40"

        if value < 60:
            return "40-60"

        return "60+"

    def get_slope_bucket(trade):
        value = to_number(
            trade.get(
                "entry_trend_slope"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < -0.50:
            return "< -0.50"

        if value < 0:
            return "-0.50 to 0"

        if value < 0.50:
            return "0 to 0.50"

        return ">= 0.50"

    def get_price_position_bucket(trade):
        value = to_number(
            trade.get(
                "entry_price_position_value"
            )
        )

        if value is None:
            return "UNKNOWN"

        if value < 0.25:
            return "<25%"

        if value < 0.50:
            return "25-50%"

        if value < 0.75:
            return "50-75%"

        return "75%+"

    def get_structure_bias(trade):
        return str(
            trade.get(
                "structure_bias",
                trade.get(
                    "Structure_Bias",
                    "UNKNOWN"
                )
            )
        )

    def get_liquidity(trade):
        value = trade.get(
            "liquidity_sweep",
            trade.get(
                "Liquidity_Sweep"
            )
        )

        if isinstance(value, bool):
            return (
                "SWEEP"
                if value
                else "NO_SWEEP"
            )

        return (
            "SWEEP"
            if str(value).strip().lower()
            in {
                "true",
                "1",
                "yes",
                "y",
            }
            else "NO_SWEEP"
        )

    def get_break_age_bucket(trade):
        value = to_number(
            trade.get(
                "break_age",
                trade.get(
                    "Break_Age"
                )
            )
        )

        if value is None:
            return "UNKNOWN"

        if value <= 1:
            return "0-1"

        if value <= 3:
            return "2-3"

        if value <= 5:
            return "4-5"

        return "6+"

    def get_context_value(
        trade,
        feature
    ):
        if feature == "SETUP":
            return get_setup(trade)

        if feature == "ENTRY_TREND":
            return get_trend(trade)

        if feature == "TREND_STRENGTH":
            return get_strength_bucket(trade)

        if feature == "TREND_SLOPE":
            return get_slope_bucket(trade)

        if feature == "PRICE_POSITION":
            return get_price_position_bucket(trade)

        if feature == "STRUCTURE_BIAS":
            return get_structure_bias(trade)

        if feature == "LIQUIDITY_SWEEP":
            return get_liquidity(trade)

        if feature == "BREAK_AGE":
            return get_break_age_bucket(trade)

        return "UNKNOWN"

    def passes_filter(
        trade,
        window,
        threshold
    ):
        if window is None:
            return True

        field = (
            f"early_adverse_r_bar_{window}"
        )

        value = to_number(
            trade.get(field)
        )

        # Preserve trades where the
        # diagnostic field is unavailable.
        if value is None:
            return True

        return value < threshold

    def evaluate(
        period_trades,
        window,
        threshold
    ):
        remaining = [
            trade
            for trade in period_trades
            if passes_filter(
                trade,
                window,
                threshold
            )
        ]

        r_values = [
            get_r(trade)
            for trade in remaining
        ]

        wins = [
            r
            for r in r_values
            if r > 0
        ]

        losses = [
            r
            for r in r_values
            if r < 0
        ]

        gross_profit = sum(wins)
        gross_loss = abs(
            sum(losses)
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        )

        total_r = sum(r_values)

        win_rate = (
            len(wins)
            / len(r_values)
            * 100
            if r_values
            else 0.0
        )

        return {
            "trades": len(remaining),
            "win_rate": win_rate,
            "pf": profit_factor,
            "total_r": total_r,
        }

    print()
    print("=" * 150)
    print(
        "FROZEN GLOBAL EARLY-ADVERSE-R ROBUSTNESS"
    )
    print("=" * 150)

    print()
    print(
        "Purpose: test one fixed Early Adverse-R rule "
        "across years and entry-time contexts."
    )

    print()
    print(
        "IMPORTANT: no subgroup-specific optimization "
        "is performed in this diagnostic."
    )

    # --------------------------------------------------------------
    # GLOBAL FIXED-RULE MATRIX
    # --------------------------------------------------------------

    print()
    print("-" * 150)
    print(
        "GLOBAL FIXED-RULE PERFORMANCE"
    )
    print("-" * 150)

    baseline = evaluate(
        trades,
        None,
        None
    )

    global_results = []

    for (
        name,
        window,
        threshold
    ) in candidates:

        result = evaluate(
            trades,
            window,
            threshold
        )

        delta_r = (
            result["total_r"]
            - baseline["total_r"]
        )

        global_results.append(
            {
                "name": name,
                "window": window,
                "threshold": threshold,
                "trades": result["trades"],
                "win_rate": result["win_rate"],
                "pf": result["pf"],
                "total_r": result["total_r"],
                "delta_r": delta_r,
            }
        )

        print(
            f"{name:<15}"
            f"Trades={result['trades']:<5} "
            f"Win={result['win_rate']:>6.2f}% "
            f"PF={result['pf']:>6.2f} "
            f"R={result['total_r']:>8.2f} "
            f"DeltaR={delta_r:>8.2f}"
        )

    # --------------------------------------------------------------
    # YEAR-BY-YEAR FIXED-RULE TEST
    # --------------------------------------------------------------

    print()
    print("-" * 150)
    print(
        "YEAR-BY-YEAR FIXED-RULE TEST"
    )
    print("-" * 150)

    year_results = defaultdict(list)

    for year in validation_years:

        year_trades = [
            trade
            for trade in trades
            if trade_year(trade) == year
        ]

        baseline_year = evaluate(
            year_trades,
            None,
            None
        )

        print()
        print(
            f"YEAR {year} | "
            f"Baseline R={baseline_year['total_r']:.2f}"
        )

        for (
            name,
            window,
            threshold
        ) in candidates:

            result = evaluate(
                year_trades,
                window,
                threshold
            )

            delta_r = (
                result["total_r"]
                - baseline_year["total_r"]
            )

            year_results[name].append(
                {
                    "year": year,
                    "delta_r": delta_r,
                    "filtered_r": result["total_r"],
                    "trades": result["trades"],
                }
            )

            print(
                f"  {name:<15}"
                f"R={result['total_r']:>7.2f} "
                f"DeltaR={delta_r:>7.2f}"
            )

    # --------------------------------------------------------------
    # STABILITY SUMMARY
    # --------------------------------------------------------------

    print()
    print("-" * 150)
    print(
        "FIXED-RULE STABILITY SUMMARY"
    )
    print("-" * 150)

    for (
        name,
        window,
        threshold
    ) in candidates:

        results = year_results[name]

        improved = sum(
            1
            for item in results
            if item["delta_r"] > 0
        )

        total_delta = sum(
            item["delta_r"]
            for item in results
        )

        print(
            f"{name:<15}"
            f"Years improved={improved}/"
            f"{len(results)} "
            f"Total DeltaR={total_delta:>8.2f}"
        )

    # --------------------------------------------------------------
    # CONTEXT STABILITY
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONTEXT STABILITY OF FIXED RULES"
    )
    print("=" * 150)

    for feature in context_features:

        print()
        print("-" * 150)
        print(
            f"CONTEXT: {feature}"
        )
        print("-" * 150)

        groups = defaultdict(list)

        for trade in trades:

            key = get_context_value(
                trade,
                feature
            )

            groups[key].append(
                trade
            )

        for (
            context_value,
            group_trades
        ) in sorted(
            groups.items(),
            key=lambda item: str(item[0])
        ):

            if len(group_trades) < MIN_GROUP_TRADES:
                continue

            group_baseline = evaluate(
                group_trades,
                None,
                None
            )

            print()
            print(
                f"{str(context_value):<25}"
                f"Trades={len(group_trades):<5} "
                f"BaselineR="
                f"{group_baseline['total_r']:>7.2f}"
            )

            for (
                name,
                window,
                threshold
            ) in candidates:

                if name == "NO_FILTER":
                    continue

                result = evaluate(
                    group_trades,
                    window,
                    threshold
                )

                delta_r = (
                    result["total_r"]
                    - group_baseline["total_r"]
                )

                print(
                    f"  {name:<15}"
                    f"R={result['total_r']:>7.2f} "
                    f"DeltaR={delta_r:>7.2f} "
                    f"Remain={result['trades']}"
                )

    # --------------------------------------------------------------
    # CONCENTRATION CHECK
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "CONCENTRATION CHECK — SETUP"
    )
    print("=" * 150)

    setup_groups = defaultdict(list)

    for trade in trades:
        setup_groups[
            get_setup(trade)
        ].append(
            trade
        )

    for (
        name,
        window,
        threshold
    ) in candidates:

        if name == "NO_FILTER":
            continue

        total_delta = (
            global_results[
                next(
                    i
                    for i, item
                    in enumerate(global_results)
                    if item["name"] == name
                )
            ]["delta_r"]
        )

        if total_delta == 0:
            print(
                f"{name:<15}"
                "Total DeltaR=0.00 "
                "Concentration=N/A"
            )
            continue

        contributions = []

        for (
            setup,
            setup_trades
        ) in setup_groups.items():

            if len(setup_trades) < MIN_GROUP_TRADES:
                continue

            baseline_setup = evaluate(
                setup_trades,
                None,
                None
            )

            filtered_setup = evaluate(
                setup_trades,
                window,
                threshold
            )

            delta = (
                filtered_setup["total_r"]
                - baseline_setup["total_r"]
            )

            contributions.append(
                (
                    setup,
                    delta
                )
            )

        contributions.sort(
            key=lambda item: abs(item[1]),
            reverse=True
        )

        if contributions:

            largest_setup = contributions[0]

            concentration = (
                abs(largest_setup[1])
                / abs(total_delta)
                * 100
            )

            print(
                f"{name:<15}"
                f"TotalDeltaR={total_delta:>8.2f} "
                f"LargestSetup="
                f"{largest_setup[0]:<25} "
                f"Contribution="
                f"{largest_setup[1]:>7.2f} "
                f"Concentration="
                f"{concentration:>6.1f}%"
            )

    # --------------------------------------------------------------
    # FINAL RESEARCH VERDICT
    # --------------------------------------------------------------

    print()
    print("=" * 150)
    print(
        "FROZEN EARLY-ADVERSE-R RESEARCH VERDICT"
    )
    print("=" * 150)

    ranked = sorted(
        [
            item
            for item in global_results
            if item["name"] != "NO_FILTER"
        ],
        key=lambda item: (
            -item["delta_r"],
            -item["pf"],
        )
    )

    if not ranked:
        print(
            "No candidate rules available."
        )
        return

    best = ranked[0]

    best_year_results = year_results[
        best["name"]
    ]

    best_improved_years = sum(
        1
        for item in best_year_results
        if item["delta_r"] > 0
    )

    print()
    print(
        f"Best fixed rule by total DeltaR: "
        f"{best['name']}"
    )

    print(
        f"Global DeltaR: "
        f"{best['delta_r']:.2f}"
    )

    print(
        f"Years improved: "
        f"{best_improved_years}/"
        f"{len(best_year_results)}"
    )

    print()

    if (
        best["delta_r"] > 0
        and best_improved_years >= 2
    ):
        print(
            "PROMISING ROBUSTNESS SIGNAL."
        )
        print(
            "A fixed Early-Adverse-R rule "
            "shows positive aggregate improvement "
            "and improves at least 2 validation years."
        )
        print(
            "Do NOT deploy yet."
        )
    else:
        print(
            "NOT ROBUST ENOUGH."
        )
        print(
            "No single frozen Early-Adverse-R rule "
            "shows sufficiently consistent improvement."
        )
        print(
            "Do NOT modify production engine logic."
        )

def frozen_early_adverse_chronological_stability_test(trades):
    """
    Research-only chronological stability test.

    Frozen rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    Missing Bar-5 values are retained.

    IMPORTANT:
        This is a retrospective stability test.
        It is NOT an untouched holdout test.
        The rule is fixed in advance and is never optimized here.

    This diagnostic:
        - does not select a threshold
        - does not select a window
        - does not modify execution
        - does not modify trade outcomes
        - does not optimize stocks
        - does not optimize setups
        - does not optimize trends
        - does not modify production engine logic
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    print()
    print("=" * 140)
    print("FROZEN 5B_0.25R CHRONOLOGICAL STABILITY TEST")
    print("=" * 140)

    print()
    print("Frozen rule:")
    print(
        "  Reject when "
        "early_adverse_r_bar_5 >= 0.25R"
    )
    print(
        "  Missing Bar-5 values are retained."
    )
    print(
        "  No threshold/window/subgroup optimization is performed."
    )
    print()
    print(
        "IMPORTANT: This is retrospective chronological stability, "
        "NOT an untouched future holdout."
    )

    def trade_year(trade):
        value = str(
            trade.get(
                "entry_date",
                ""
            )
        ).strip()

        if len(value) < 4:
            return None

        try:
            return int(value[:4])
        except (TypeError, ValueError):
            return None

    def trade_r(trade):
        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=None
        )

        return value

    def passes_rule(trade):
        value = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if value is None:
            return True

        return value < RULE_THRESHOLD

    def evaluate(sample):
        baseline_r = 0.0
        filtered_r = 0.0

        baseline_wins = 0
        filtered_wins = 0

        removed_count = 0
        removed_winners = 0

        retained = []
        removed = []

        for trade in sample:
            r_value = trade_r(trade)

            if r_value is None:
                continue

            baseline_r += r_value

            if r_value > 0:
                baseline_wins += 1

            if passes_rule(trade):
                retained.append(trade)
                filtered_r += r_value

                if r_value > 0:
                    filtered_wins += 1
            else:
                removed.append(trade)
                removed_count += 1

                if r_value > 0:
                    removed_winners += 1

        baseline_trades = len(
            [
                trade
                for trade in sample
                if trade_r(trade) is not None
            ]
        )

        filtered_trades = len(retained)

        baseline_losses = (
            baseline_trades - baseline_wins
        )

        filtered_losses = (
            filtered_trades - filtered_wins
        )

        baseline_win_rate = (
            baseline_wins
            / baseline_trades
            * 100
            if baseline_trades > 0
            else 0.0
        )

        filtered_win_rate = (
            filtered_wins
            / filtered_trades
            * 100
            if filtered_trades > 0
            else 0.0
        )

        baseline_gross_profit = sum(
            max(
                trade_r(trade) or 0.0,
                0.0
            )
            for trade in sample
            if trade_r(trade) is not None
        )

        baseline_gross_loss = sum(
            abs(
                min(
                    trade_r(trade) or 0.0,
                    0.0
                )
            )
            for trade in sample
            if trade_r(trade) is not None
        )

        filtered_gross_profit = sum(
            max(
                trade_r(trade) or 0.0,
                0.0
            )
            for trade in retained
        )

        filtered_gross_loss = sum(
            abs(
                min(
                    trade_r(trade) or 0.0,
                    0.0
                )
            )
            for trade in retained
        )

        baseline_pf = (
            baseline_gross_profit
            / baseline_gross_loss
            if baseline_gross_loss > 0
            else 0.0
        )

        filtered_pf = (
            filtered_gross_profit
            / filtered_gross_loss
            if filtered_gross_loss > 0
            else 0.0
        )

        return {
            "baseline_trades": baseline_trades,
            "filtered_trades": filtered_trades,
            "baseline_wins": baseline_wins,
            "filtered_wins": filtered_wins,
            "baseline_losses": baseline_losses,
            "filtered_losses": filtered_losses,
            "baseline_win_rate": baseline_win_rate,
            "filtered_win_rate": filtered_win_rate,
            "baseline_pf": baseline_pf,
            "filtered_pf": filtered_pf,
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "delta_r": filtered_r - baseline_r,
            "removed_count": removed_count,
            "removed_winners": removed_winners,
        }

    years = sorted(
        {
            trade_year(trade)
            for trade in trades
            if trade_year(trade) is not None
        }
    )

    results = []

    print()
    print(
        f"{'Year':<8}"
        f"{'Base N':<9}"
        f"{'Filt N':<9}"
        f"{'Base WR':<11}"
        f"{'Filt WR':<11}"
        f"{'Base PF':<10}"
        f"{'Filt PF':<10}"
        f"{'Base R':<11}"
        f"{'Filt R':<11}"
        f"{'Delta R':<11}"
        f"{'Removed':<10}"
        f"{'Rem Winners':<13}"
    )

    print("-" * 140)

    for year in years:
        year_trades = [
            trade
            for trade in trades
            if trade_year(trade) == year
        ]

        result = evaluate(
            year_trades
        )

        results.append(
            (
                year,
                result
            )
        )

        print(
            f"{year:<8}"
            f"{result['baseline_trades']:<9}"
            f"{result['filtered_trades']:<9}"
            f"{result['baseline_win_rate']:<11.2f}"
            f"{result['filtered_win_rate']:<11.2f}"
            f"{result['baseline_pf']:<10.2f}"
            f"{result['filtered_pf']:<10.2f}"
            f"{result['baseline_r']:<11.2f}"
            f"{result['filtered_r']:<11.2f}"
            f"{result['delta_r']:<11.2f}"
            f"{result['removed_count']:<10}"
            f"{result['removed_winners']:<13}"
        )

    total_baseline_r = sum(
        result["baseline_r"]
        for _, result in results
    )

    total_filtered_r = sum(
        result["filtered_r"]
        for _, result in results
    )

    positive_years = sum(
        1
        for _, result in results
        if result["delta_r"] > 0
    )

    negative_years = sum(
        1
        for _, result in results
        if result["delta_r"] < 0
    )

    flat_years = sum(
        1
        for _, result in results
        if result["delta_r"] == 0
    )

    total_removed = sum(
        result["removed_count"]
        for _, result in results
    )

    total_removed_winners = sum(
        result["removed_winners"]
        for _, result in results
    )

    print()
    print("-" * 140)
    print("CHRONOLOGICAL STABILITY SUMMARY")
    print("-" * 140)

    print(
        f"Baseline total R        : "
        f"{total_baseline_r:.2f}"
    )

    print(
        f"Frozen-filter total R   : "
        f"{total_filtered_r:.2f}"
    )

    print(
        f"Total Delta R           : "
        f"{total_filtered_r - total_baseline_r:.2f}"
    )

    print(
        f"Positive years          : "
        f"{positive_years}"
    )

    print(
        f"Negative years          : "
        f"{negative_years}"
    )

    print(
        f"Flat years              : "
        f"{flat_years}"
    )

    print(
        f"Total removed trades    : "
        f"{total_removed}"
    )

    print(
        f"Removed winners         : "
        f"{total_removed_winners}"
    )

    print()
    print("-" * 140)
    print("RESEARCH INTERPRETATION")
    print("-" * 140)

    if (
        total_filtered_r > total_baseline_r
        and positive_years == len(results)
    ):
        print(
            "STRONG RETROSPECTIVE STABILITY: "
            "the frozen 5B_0.25R rule improved every tested year."
        )
    elif (
        total_filtered_r > total_baseline_r
        and positive_years >= 2
    ):
        print(
            "RETROSPECTIVELY PROMISING: "
            "the frozen 5B_0.25R rule improved most tested years."
        )
    elif total_filtered_r > total_baseline_r:
        print(
            "MIXED RETROSPECTIVE STABILITY: "
            "the frozen 5B_0.25R rule improved overall "
            "but was inconsistent across years."
        )
    else:
        print(
            "WEAK RETROSPECTIVE STABILITY: "
            "the frozen 5B_0.25R rule did not improve overall."
        )

    print()
    print(
        "HOLDOUT STATUS: NOT AN UNTOUCHED HOLDOUT."
    )

    print(
        "The genuine future holdout must consist of trades "
        "generated after the rule was frozen."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

def frozen_early_adverse_holdout_ledger_integrity_audit(trades):

    print()
    print("=" * 120)
    print("FROZEN 5B_0.25R HOLDOUT LEDGER INTEGRITY AUDIT")
    print("=" * 120)

    HOLDOUT_START_DATE = datetime(
        2026,
        9,
        7
    )

    RULE_THRESHOLD = 0.25

    LEDGER_FILE = (
        "tests/output/frozen_5b_025_holdout_ledger.csv"
    )

    print()
    print("AUDIT SCOPE:")
    print(
        "  Persistent holdout ledger only."
    )
    print(
        "  No strategy optimization."
    )
    print(
        "  No threshold optimization."
    )
    print(
        "  No window optimization."
    )
    print(
        "  No subgroup optimization."
    )
    print(
        "  No changes to trading logic."
    )

    print()
    print(
        f"Holdout start date: "
        f"{HOLDOUT_START_DATE.date()}"
    )

    print(
        f"Ledger file: "
        f"{LEDGER_FILE}"
    )

    def trade_date(trade):

        value = trade.get(
            "entry_date"
        )

        if value is None:
            return None

        try:
            return datetime.fromisoformat(
                str(value)
            )

        except ValueError:
            return None

    def ledger_trade_key(trade):

        return "|".join(
            [
                str(
                    trade.get(
                        "entry_date",
                        ""
                    )
                ),
                str(
                    trade.get(
                        "symbol",
                        trade.get(
                            "ticker",
                            ""
                        )
                    )
                ),
                str(
                    trade.get(
                        "direction",
                        ""
                    )
                ).upper(),
                str(
                    trade.get(
                        "entry_price",
                        ""
                    )
                ),
                str(
                    trade.get(
                        "initial_risk",
                        ""
                    )
                ),
            ]
        )

    def expected_decision(
        early_adverse
    ):

        if early_adverse is None:
            return "RETAIN"

        return (
            "RETAIN"
            if early_adverse < RULE_THRESHOLD
            else "REJECT"
        )

    def load_ledger():

        if not os.path.exists(
            LEDGER_FILE
        ):
            return []

        with open(
            LEDGER_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            return list(
                csv.DictReader(
                    file
                )
            )

    ledger_rows = load_ledger()

    print()
    print(
        f"Ledger rows loaded: "
        f"{len(ledger_rows)}"
    )

    # --------------------------------------------------------------
    # Build current genuine post-freeze trade lookup
    # --------------------------------------------------------------

    current_holdout = []

    for trade in trades:

        entry_date = trade_date(
            trade
        )

        if (
            entry_date is not None
            and entry_date >= HOLDOUT_START_DATE
        ):

            current_holdout.append(
                trade
            )

    current_lookup = {}

    for trade in current_holdout:

        key = ledger_trade_key(
            trade
        )

        if key not in current_lookup:

            current_lookup[key] = trade

    # --------------------------------------------------------------
    # AUDIT 1 — duplicate ledger keys
    # --------------------------------------------------------------

    ledger_keys = []

    for row in ledger_rows:

        ledger_keys.append(
            row.get(
                "trade_key",
                ""
            )
        )

    duplicate_keys = {
        key
        for key in ledger_keys
        if key
        and ledger_keys.count(key) > 1
    }

    print()
    print(
        "AUDIT 1 — DUPLICATE LEDGER KEYS"
    )

    print(
        f"  Duplicate keys: "
        f"{len(duplicate_keys)}"
    )

    duplicate_key_pass = (
        len(duplicate_keys) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if duplicate_key_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 2 — all ledger rows are post-freeze
    # --------------------------------------------------------------

    pre_freeze_rows = []

    invalid_date_rows = []

    for row in ledger_rows:

        value = row.get(
            "entry_date",
            ""
        )

        try:

            entry_date = datetime.fromisoformat(
                str(value)
            )

        except ValueError:

            invalid_date_rows.append(
                row
            )

            continue

        if entry_date < HOLDOUT_START_DATE:

            pre_freeze_rows.append(
                row
            )

    print()
    print(
        "AUDIT 2 — POST-FREEZE MEMBERSHIP"
    )

    print(
        f"  Pre-freeze ledger rows: "
        f"{len(pre_freeze_rows)}"
    )

    print(
        f"  Invalid-date ledger rows: "
        f"{len(invalid_date_rows)}"
    )

    post_freeze_pass = (
        len(pre_freeze_rows) == 0
        and len(invalid_date_rows) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if post_freeze_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 3 — ledger rows match current trade dataset
    # --------------------------------------------------------------

    missing_current_trades = []

    for row in ledger_rows:

        key = row.get(
            "trade_key",
            ""
        )

        if key not in current_lookup:

            missing_current_trades.append(
                row
            )

    print()
    print(
        "AUDIT 3 — CURRENT DATASET RECONCILIATION"
    )

    print(
        f"  Ledger rows not found in "
        f"current post-freeze dataset: "
        f"{len(missing_current_trades)}"
    )

    current_dataset_pass = (
        len(missing_current_trades) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if current_dataset_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 4 — immutable entry-level fields
    # --------------------------------------------------------------

    entry_field_mismatches = []

    for row in ledger_rows:

        key = row.get(
            "trade_key",
            ""
        )

        current_trade = current_lookup.get(
            key
        )

        if current_trade is None:
            continue

        checks = [
            (
                "entry_date",
                str(
                    row.get(
                        "entry_date",
                        ""
                    )
                ),
                str(
                    current_trade.get(
                        "entry_date",
                        ""
                    )
                )
            ),
            (
                "symbol",
                str(
                    row.get(
                        "symbol",
                        ""
                    )
                ),
                str(
                    current_trade.get(
                        "symbol",
                        current_trade.get(
                            "ticker",
                            ""
                        )
                    )
                )
            ),
            (
                "direction",
                str(
                    row.get(
                        "direction",
                        ""
                    )
                ).upper(),
                str(
                    current_trade.get(
                        "direction",
                        ""
                    )
                ).upper()
            ),
            (
                "entry_price",
                str(
                    row.get(
                        "entry_price",
                        ""
                    )
                ),
                str(
                    current_trade.get(
                        "entry_price",
                        ""
                    )
                )
            ),
            (
                "initial_risk",
                str(
                    row.get(
                        "initial_risk",
                        ""
                    )
                ),
                str(
                    current_trade.get(
                        "initial_risk",
                        ""
                    )
                )
            ),
        ]

        for (
            field,
            ledger_value,
            current_value
        ) in checks:

            if ledger_value != current_value:

                entry_field_mismatches.append(
                    (
                        key,
                        field,
                        ledger_value,
                        current_value
                    )
                )

    print()
    print(
        "AUDIT 4 — ENTRY-LEVEL FIELD IMMUTABILITY"
    )

    print(
        f"  Field mismatches: "
        f"{len(entry_field_mismatches)}"
    )

    entry_field_pass = (
        len(entry_field_mismatches) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if entry_field_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 5 — Bar-5 value integrity
    # --------------------------------------------------------------

    bar5_mismatches = []

    for row in ledger_rows:

        key = row.get(
            "trade_key",
            ""
        )

        current_trade = current_lookup.get(
            key
        )

        if current_trade is None:
            continue

        ledger_bar5 = row.get(
            "early_adverse_r_bar_5",
            ""
        )

        current_bar5 = current_trade.get(
            "early_adverse_r_bar_5",
            ""
        )

        ledger_bar5_value = to_float(
            ledger_bar5,
            default=None
        )

        current_bar5_value = to_float(
            current_bar5,
            default=None
        )

        if (
            ledger_bar5_value is None
            and current_bar5_value is None
        ):

            continue

        if (
            ledger_bar5_value is None
            or current_bar5_value is None
            or abs(
                ledger_bar5_value
                - current_bar5_value
            ) > 1e-12
        ):

            bar5_mismatches.append(
                (
                    key,
                    ledger_bar5,
                    current_bar5
                )
            )

    print()
    print(
        "AUDIT 5 — BAR-5 VALUE INTEGRITY"
    )

    print(
        f"  Bar-5 mismatches: "
        f"{len(bar5_mismatches)}"
    )

    bar5_pass = (
        len(bar5_mismatches) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if bar5_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 6 — frozen decision integrity
    # --------------------------------------------------------------

    decision_mismatches = []

    for row in ledger_rows:

        early_adverse = to_float(
            row.get(
                "early_adverse_r_bar_5"
            ),
            default=None
        )

        expected = expected_decision(
            early_adverse
        )

        actual = str(
            row.get(
                "decision",
                ""
            )
        ).upper()

        if actual != expected:

            decision_mismatches.append(
                (
                    row.get(
                        "trade_key",
                        ""
                    ),
                    actual,
                    expected,
                    early_adverse
                )
            )

    print()
    print(
        "AUDIT 6 — FROZEN DECISION INTEGRITY"
    )

    print(
        f"  Decision mismatches: "
        f"{len(decision_mismatches)}"
    )

    decision_pass = (
        len(decision_mismatches) == 0
    )

    print(
        "  Frozen rule: "
        "Bar-5 >= 0.25R -> REJECT"
    )

    print(
        "  Status: "
        f"{'PASS' if decision_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 7 — Final R integrity
    # --------------------------------------------------------------

    invalid_r_rows = []

    for row in ledger_rows:

        value = to_float(
            row.get(
                "r_multiple"
            ),
            default=None
        )

        if value is None:

            invalid_r_rows.append(
                row
            )

    print()
    print(
        "AUDIT 7 — FINAL-R INTEGRITY"
    )

    print(
        f"  Missing/invalid Final-R rows: "
        f"{len(invalid_r_rows)}"
    )

    final_r_pass = (
        len(invalid_r_rows) == 0
    )

    print(
        "  Status: "
        f"{'PASS' if final_r_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # AUDIT 8 — ledger count / unique-key reconciliation
    # --------------------------------------------------------------

    unique_ledger_keys = {
        key
        for key in ledger_keys
        if key
    }

    ledger_count_pass = (
        len(ledger_rows)
        == len(unique_ledger_keys)
    )

    print()
    print(
        "AUDIT 8 — LEDGER COUNT RECONCILIATION"
    )

    print(
        f"  Ledger rows: "
        f"{len(ledger_rows)}"
    )

    print(
        f"  Unique non-empty keys: "
        f"{len(unique_ledger_keys)}"
    )

    print(
        "  Status: "
        f"{'PASS' if ledger_count_pass else 'FAIL'}"
    )

    # --------------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------------

    overall_pass = all(
        [
            duplicate_key_pass,
            post_freeze_pass,
            current_dataset_pass,
            entry_field_pass,
            bar5_pass,
            decision_pass,
            final_r_pass,
            ledger_count_pass,
        ]
    )

    print()
    print("=" * 120)
    print(
        "HOLDOUT LEDGER INTEGRITY RESULT"
    )
    print("=" * 120)

    print(
        f"  Overall status: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    if overall_pass:

        print()
        print(
            "  The persistent holdout ledger passes "
            "all integrity checks."
        )

        print(
            "  No strategy or holdout-rule changes "
            "were made by this audit."
        )

    else:

        print()
        print(
            "  IMPORTANT: Ledger integrity failure "
            "detected."
        )

        print(
            "  Do NOT use the affected holdout "
            "as validation evidence until resolved."
        )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

def frozen_early_adverse_profit_concentration_audit(trades):
    """
    Research-only profit concentration / fragility audit.

    Frozen rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    Purpose:
        Determine whether the historical improvement produced by the
        frozen rule depends excessively on a small number of trades.

    This audit:
        - does not optimize the threshold
        - does not optimize the window
        - does not optimize stocks
        - does not optimize setups
        - does not optimize trends
        - does not modify trade outcomes
        - does not modify execution
        - does not modify production engine logic

    Stress tests are predefined:
        Remove best 1 retained trade
        Remove best 3 retained trades
        Remove best 5 retained trades
        Remove best 10 retained trades

    IMPORTANT:
        This is retrospective robustness analysis.
        It is NOT an untouched future holdout.
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    print()
    print("=" * 150)
    print(
        "FROZEN 5B_0.25R PROFIT CONCENTRATION / FRAGILITY AUDIT"
    )
    print("=" * 150)

    print()
    print("Frozen rule:")
    print(
        "  Reject when "
        "early_adverse_r_bar_5 >= 0.25R"
    )
    print(
        "  Missing Bar-5 values are retained."
    )

    print()
    print(
        "Purpose:"
    )
    print(
        "  Test whether the historical improvement depends "
        "excessively on a small number of trades."
    )

    print()
    print(
        "Predefined stress tests:"
    )
    print(
        "  Remove best 1 retained trade"
    )
    print(
        "  Remove best 3 retained trades"
    )
    print(
        "  Remove best 5 retained trades"
    )
    print(
        "  Remove best 10 retained trades"
    )

    print()
    print(
        "IMPORTANT: Retrospective robustness test only."
    )
    print(
        "This is NOT an untouched future holdout."
    )

    def get_r(trade):

        return to_float(
            trade.get(
                "r_multiple"
            ),
            default=None
        )

    def passes_rule(trade):

        early_adverse = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if early_adverse is None:

            return True

        return (
            early_adverse
            < RULE_THRESHOLD
        )

    valid_trades = [
        trade
        for trade in trades
        if get_r(trade) is not None
    ]

    retained = [
        trade
        for trade in valid_trades
        if passes_rule(trade)
    ]

    removed = [
        trade
        for trade in valid_trades
        if not passes_rule(trade)
    ]

    baseline_r = sum(
        get_r(trade)
        for trade in valid_trades
    )

    filtered_r = sum(
        get_r(trade)
        for trade in retained
    )

    removed_r = sum(
        get_r(trade)
        for trade in removed
    )

    total_delta_r = (
        filtered_r
        - baseline_r
    )

    print()
    print("=" * 150)
    print(
        "BASELINE CONCENTRATION"
    )
    print("=" * 150)

    print(
        f"Valid trades       : "
        f"{len(valid_trades)}"
    )

    print(
        f"Retained trades    : "
        f"{len(retained)}"
    )

    print(
        f"Removed trades     : "
        f"{len(removed)}"
    )

    print(
        f"Baseline R         : "
        f"{baseline_r:.2f}"
    )

    print(
        f"Filtered R         : "
        f"{filtered_r:.2f}"
    )

    print(
        f"Removed R          : "
        f"{removed_r:.2f}"
    )

    print(
        f"Total Delta R      : "
        f"{total_delta_r:.2f}"
    )

    if retained:

        ranked_retained = sorted(
            retained,
            key=lambda trade: get_r(trade),
            reverse=True
        )

    else:

        ranked_retained = []

    def stressed_filtered_r(remove_count):

        if not ranked_retained:

            return 0.0

        remaining = ranked_retained[
            remove_count:
        ]

        return sum(
            get_r(trade)
            for trade in remaining
        )

    print()
    print("=" * 150)
    print(
        "TOP-WINNER REMOVAL STRESS TEST"
    )
    print("=" * 150)

    print()
    print(
        f"{'Stress Test':<28}"
        f"{'Removed':<10}"
        f"{'Filtered R':<14}"
        f"{'Delta R':<14}"
        f"{'Remaining N':<14}"
        f"{'Status':<20}"
    )

    print(
        "-" * 150
    )

    stress_results = []

    for remove_count in [
        0,
        1,
        3,
        5,
        10
    ]:

        stressed_r = (
            stressed_filtered_r(
                remove_count
            )
        )

        stressed_delta = (
            stressed_r
            - baseline_r
        )

        remaining_n = max(
            len(ranked_retained)
            - remove_count,
            0
        )

        if (
            remove_count == 0
            and stressed_delta > 0
        ):

            status = "REFERENCE"

        elif stressed_delta > 0:

            status = "POSITIVE"

        elif stressed_delta == 0:

            status = "FLAT"

        else:

            status = "NEGATIVE"

        stress_results.append(
            (
                remove_count,
                stressed_r,
                stressed_delta,
                remaining_n,
                status
            )
        )

        if remove_count == 0:

            label = "FULL FILTER"

        else:

            label = (
                f"REMOVE TOP {remove_count}"
            )

        print(
            f"{label:<28}"
            f"{remove_count:<10}"
            f"{stressed_r:<14.2f}"
            f"{stressed_delta:<14.2f}"
            f"{remaining_n:<14}"
            f"{status:<20}"
        )

    print()
    print("=" * 150)
    print(
        "TOP RETAINED TRADE CONTRIBUTIONS"
    )
    print("=" * 150)

    if ranked_retained:

        top_trade_r = get_r(
            ranked_retained[0]
        )

        top_3_r = sum(
            get_r(trade)
            for trade in ranked_retained[:3]
        )

        top_5_r = sum(
            get_r(trade)
            for trade in ranked_retained[:5]
        )

        top_10_r = sum(
            get_r(trade)
            for trade in ranked_retained[:10]
        )

        print(
            f"Top 1 retained R     : "
            f"{top_trade_r:.2f}"
        )

        print(
            f"Top 3 retained R     : "
            f"{top_3_r:.2f}"
        )

        print(
            f"Top 5 retained R     : "
            f"{top_5_r:.2f}"
        )

        print(
            f"Top 10 retained R    : "
            f"{top_10_r:.2f}"
        )

        print()

        if total_delta_r != 0:

            top_1_concentration = (
                abs(top_trade_r)
                / abs(total_delta_r)
                * 100
            )

            top_3_concentration = (
                abs(top_3_r)
                / abs(total_delta_r)
                * 100
            )

            top_5_concentration = (
                abs(top_5_r)
                / abs(total_delta_r)
                * 100
            )

            top_10_concentration = (
                abs(top_10_r)
                / abs(total_delta_r)
                * 100
            )

            print(
                f"Top 1 / total Delta R : "
                f"{top_1_concentration:.1f}%"
            )

            print(
                f"Top 3 / total Delta R : "
                f"{top_3_concentration:.1f}%"
            )

            print(
                f"Top 5 / total Delta R : "
                f"{top_5_concentration:.1f}%"
            )

            print(
                f"Top 10 / total Delta R: "
                f"{top_10_concentration:.1f}%"
            )

    else:

        print(
            "No retained trades available."
        )

    print()
    print("=" * 150)
    print(
        "REMOVED-TRADE CONTRIBUTION"
    )
    print("=" * 150)

    print(
        f"Removed trades       : "
        f"{len(removed)}"
    )

    print(
        f"Removed-trade R      : "
        f"{removed_r:.2f}"
    )

    removed_winners = sum(
        1
        for trade in removed
        if get_r(trade) > 0
    )

    removed_losers = sum(
        1
        for trade in removed
        if get_r(trade) <= 0
    )

    print(
        f"Removed winners      : "
        f"{removed_winners}"
    )

    print(
        f"Removed non-winners  : "
        f"{removed_losers}"
    )

    print()
    print("=" * 150)
    print(
        "FRAGILITY DECISION"
    )
    print("=" * 150)

    result_map = {
        remove_count: stressed_delta
        for (
            remove_count,
            _,
            stressed_delta,
            _,
            _
        ) in stress_results
    }

    delta_after_1 = result_map.get(
        1,
        0.0
    )

    delta_after_3 = result_map.get(
        3,
        0.0
    )

    delta_after_5 = result_map.get(
        5,
        0.0
    )

    delta_after_10 = result_map.get(
        10,
        0.0
    )

    if (
        total_delta_r > 0
        and delta_after_10 > 0
    ):

        verdict = (
            "ROBUST"
        )

        explanation = (
            "The historical improvement remains "
            "positive even after removing the "
            "10 largest retained winners."
        )

    elif (
        total_delta_r > 0
        and delta_after_5 > 0
    ):

        verdict = (
            "MODERATELY ROBUST"
        )

        explanation = (
            "The historical improvement survives "
            "removal of the 5 largest retained "
            "winners but not the larger stress test."
        )

    elif (
        total_delta_r > 0
        and delta_after_3 > 0
    ):

        verdict = (
            "MODERATELY FRAGILE"
        )

        explanation = (
            "The historical improvement survives "
            "removal of the 3 largest retained "
            "winners but becomes vulnerable "
            "under larger concentration stress."
        )

    elif (
        total_delta_r > 0
        and delta_after_1 > 0
    ):

        verdict = (
            "FRAGILE"
        )

        explanation = (
            "The historical improvement survives "
            "removal of only the single largest "
            "retained winner."
        )

    else:

        verdict = (
            "HIGHLY FRAGILE"
        )

        explanation = (
            "The historical improvement disappears "
            "when only a small number of the largest "
            "retained winners are removed."
        )

    print(
        f"Concentration verdict: "
        f"{verdict}"
    )

    print(
        f"Interpretation: "
        f"{explanation}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  This verdict does NOT approve deployment."
    )

    print(
        "  This analysis is retrospective."
    )

    print(
        "  The genuine post-freeze holdout remains "
        "the decisive future validation."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

def frozen_early_adverse_leave_one_out_audit(trades):
    """
    Research-only leave-one-out robustness audit.

    Purpose:
        Remove each retained trade individually and determine
        whether the frozen 5B_0.25R edge remains positive.

    Frozen rule:
        Reject only when early_adverse_r_bar_5 >= 0.25R.
        Missing Bar-5 values are retained.

    IMPORTANT:
        - Does not modify the frozen rule.
        - Does not optimize anything.
        - Does not alter trade outcomes.
        - Does not alter execution logic.
        - Does not use future holdout data.
    """

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    def get_r(trade):
        value = to_float(
            trade.get("r_multiple"),
            default=None
        )

        return value if value is not None else 0.0

    def is_retained(trade):
        early_adverse = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if early_adverse is None:
            return True

        return early_adverse < RULE_THRESHOLD

    def trade_date(trade):
        return str(
            trade.get(
                "entry_date",
                ""
            )
        )

    def trade_symbol(trade):
        return str(
            trade.get(
                "Symbol",
                trade.get(
                    "symbol",
                    "UNKNOWN"
                )
            )
        )

    def trade_direction(trade):
        return str(
            trade.get(
                "Direction",
                trade.get(
                    "direction",
                    "UNKNOWN"
                )
            )
        )

    def trade_label(trade):
        return (
            f"{trade_date(trade)} | "
            f"{trade_symbol(trade)} | "
            f"{trade_direction(trade)} | "
            f"R={get_r(trade):.2f}"
        )

    valid_trades = [
        trade
        for trade in trades
        if to_float(
            trade.get("r_multiple"),
            default=None
        ) is not None
    ]

    retained = [
        trade
        for trade in valid_trades
        if is_retained(trade)
    ]

    baseline_r = sum(
        get_r(trade)
        for trade in valid_trades
    )

    filtered_r = sum(
        get_r(trade)
        for trade in retained
    )

    full_delta = (
        filtered_r - baseline_r
    )

    results = []

    for index, removed_trade in enumerate(retained):

        remaining = [
            trade
            for position, trade in enumerate(retained)
            if position != index
        ]

        leave_one_out_r = sum(
            get_r(trade)
            for trade in remaining
        )

        leave_one_out_delta = (
            leave_one_out_r - baseline_r
        )

        winners = [
            trade
            for trade in remaining
            if get_r(trade) > 0
        ]

        losers = [
            trade
            for trade in remaining
            if get_r(trade) <= 0
        ]

        gross_profit = sum(
            get_r(trade)
            for trade in winners
        )

        gross_loss = abs(
            sum(
                get_r(trade)
                for trade in losers
            )
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf")
        )

        win_rate = (
            len(winners) / len(remaining) * 100
            if remaining
            else 0.0
        )

        results.append(
            {
                "trade": removed_trade,
                "r": get_r(removed_trade),
                "filtered_r": leave_one_out_r,
                "delta_r": leave_one_out_delta,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
            }
        )

    if not results:
        print()
        print("=" * 120)
        print(
            "FROZEN 5B_0.25R LEAVE-ONE-OUT AUDIT"
        )
        print("=" * 120)
        print("No retained trades available.")
        return

    min_result = min(
        results,
        key=lambda x: x["delta_r"]
    )

    max_result = max(
        results,
        key=lambda x: x["delta_r"]
    )

    positive_filtered_count = sum(
        1
        for result in results
        if result["filtered_r"] > 0
    )

    positive_delta_count = sum(
        1
        for result in results
        if result["delta_r"] > 0
    )

    negative_filtered_count = sum(
        1
        for result in results
        if result["filtered_r"] <= 0
    )

    print()
    print("=" * 120)
    print(
        "FROZEN 5B_0.25R LEAVE-ONE-OUT ROBUSTNESS AUDIT"
    )
    print("=" * 120)

    print()
    print(
        "Frozen rule: "
        "Reject when early_adverse_r_bar_5 >= 0.25R"
    )

    print(
        "Purpose: "
        "remove each retained trade individually and test "
        "whether the historical edge survives."
    )

    print()
    print("-" * 120)
    print("REFERENCE")
    print("-" * 120)

    print(
        f"Valid trades       : {len(valid_trades)}"
    )

    print(
        f"Retained trades    : {len(retained)}"
    )

    print(
        f"Baseline R         : {baseline_r:.2f}"
    )

    print(
        f"Filtered R         : {filtered_r:.2f}"
    )

    print(
        f"Full Delta R       : {full_delta:.2f}"
    )

    print()
    print("-" * 120)
    print("LEAVE-ONE-OUT SUMMARY")
    print("-" * 120)

    print(
        f"Cases tested                       : "
        f"{len(results)}"
    )

    print(
        f"Cases with Filtered R > 0         : "
        f"{positive_filtered_count}/{len(results)}"
    )

    print(
        f"Cases with Delta R > 0             : "
        f"{positive_delta_count}/{len(results)}"
    )

    print(
        f"Cases with Filtered R <= 0        : "
        f"{negative_filtered_count}/{len(results)}"
    )

    print()
    print(
        "Worst leave-one-out case:"
    )

    print(
        f"  Removed trade    : "
        f"{trade_label(min_result['trade'])}"
    )

    print(
        f"  Remaining R      : "
        f"{min_result['filtered_r']:.2f}"
    )

    print(
        f"  Remaining DeltaR : "
        f"{min_result['delta_r']:.2f}"
    )

    print(
        f"  Win Rate         : "
        f"{min_result['win_rate']:.2f}%"
    )

    print(
        f"  Profit Factor    : "
        f"{min_result['profit_factor']:.2f}"
    )

    print()
    print(
        "Most favorable leave-one-out case:"
    )

    print(
        f"  Removed trade    : "
        f"{trade_label(max_result['trade'])}"
    )

    print(
        f"  Remaining R      : "
        f"{max_result['filtered_r']:.2f}"
    )

    print(
        f"  Remaining DeltaR : "
        f"{max_result['delta_r']:.2f}"
    )

    print()
    print("-" * 120)
    print("ALL LEAVE-ONE-OUT RESULTS")
    print("-" * 120)

    print(
        "Removed Trade | Removed R | Remaining R | "
        "Delta R | Win Rate | PF"
    )

    print("-" * 120)

    sorted_results = sorted(
        results,
        key=lambda x: x["delta_r"]
    )

    for result in sorted_results:

        print(
            f"{trade_label(result['trade']):<48} "
            f"{result['r']:>8.2f} "
            f"{result['filtered_r']:>13.2f} "
            f"{result['delta_r']:>9.2f} "
            f"{result['win_rate']:>8.2f}% "
            f"{result['profit_factor']:>6.2f}"
        )

    print()
    print("=" * 120)
    print("LEAVE-ONE-OUT DECISION")
    print("=" * 120)

    if (
        positive_filtered_count == len(results)
        and positive_delta_count == len(results)
    ):
        verdict = "STRONG ROBUSTNESS"

        interpretation = (
            "The frozen historical edge remains positive "
            "after removal of every individual retained trade."
        )

    elif positive_delta_count == len(results):
        verdict = "ROBUST WITH CAUTION"

        interpretation = (
            "The filter retains positive improvement after "
            "removing every individual retained trade, although "
            "the filtered strategy may become negative in some cases."
        )

    else:
        verdict = "TRADE-DEPENDENT"

        interpretation = (
            "At least one individual retained trade is necessary "
            "for the historical Delta R to remain positive."
        )

    print()
    print(
        f"Leave-one-out verdict: {verdict}"
    )

    print(
        f"Interpretation: {interpretation}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a retrospective robustness test."
    )

    print(
        "It does NOT prove future profitability."
    )

    print(
        "It does NOT replace the genuine post-freeze holdout."
    )

    print(
        "It does NOT approve production deployment."
    )

    print(
        "The frozen 5B_0.25R rule remains unchanged."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY"
    )

    print("=" * 120)

def frozen_early_adverse_dimension_robustness_audit(trades):

    print()
    print("=" * 120)
    print("FROZEN 5B_0.25R LEAVE-ONE-DIMENSION-OUT ROBUSTNESS AUDIT")
    print("=" * 120)

    RULE_THRESHOLD = 0.25

    print()
    print("Frozen rule:")
    print(
        "  Reject when early_adverse_r_bar_5 >= "
        f"{RULE_THRESHOLD:.2f}R"
    )
    print("  Missing Bar-5 values are retained.")

    print()
    print("Purpose:")
    print(
        "  Test whether the historical frozen-rule edge depends "
        "disproportionately on any single year, stock, or direction."
    )

    print()
    print("IMPORTANT:")
    print("  No threshold optimization.")
    print("  No window optimization.")
    print("  No subgroup-specific rule creation.")
    print("  No strategy modification.")
    print("  Research-only retrospective robustness test.")

    def get_r(trade):
        return to_float(
            trade.get("r_multiple"),
            default=None
        )

    def get_early_adverse(trade):
        return to_float(
            trade.get("early_adverse_r_bar_5"),
            default=None
        )

    def get_year(trade):
        value = trade.get("entry_date")

        if value is None:
            value = trade.get("Entry_Date")

        if value is None:
            return None

        try:
            return int(str(value)[:4])
        except Exception:
            return None

    def get_symbol(trade):
        value = trade.get("_symbol")

        if value is None:
            value = trade.get("symbol")

        if value is None:
            value = trade.get("Symbol")

        if value is None:
            value = trade.get("ticker")

        if value is None:
            value = trade.get("Ticker")

        if value is None:
            return "UNKNOWN"

        value = str(value).strip()

        if not value:
            return "UNKNOWN"

        return value

    def get_direction(trade):
        value = trade.get("direction")

        if value is None:
            value = trade.get("Direction")

        if value is None:
            return "UNKNOWN"

        value = str(value).strip().upper()

        if not value:
            return "UNKNOWN"

        return value

    def valid_trades(sample):
        result = []

        for trade in sample:
            r = get_r(trade)

            if r is not None:
                result.append(trade)

        return result

    def filter_trade(trade):
        early_adverse = get_early_adverse(trade)

        if early_adverse is None:
            return True

        return early_adverse < RULE_THRESHOLD

    def total_r(sample):
        return sum(
            get_r(trade)
            for trade in sample
            if get_r(trade) is not None
        )

    def filtered_sample(sample):
        return [
            trade
            for trade in sample
            if filter_trade(trade)
        ]

    def performance(sample):
        sample = valid_trades(sample)

        if not sample:
            return {
                "n": 0,
                "wins": 0,
                "win_rate": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "pf": 0.0,
                "r": 0.0,
            }

        values = [
            get_r(trade)
            for trade in sample
            if get_r(trade) is not None
        ]

        wins = sum(
            1
            for value in values
            if value > 0
        )

        gross_profit = sum(
            max(value, 0.0)
            for value in values
        )

        gross_loss = sum(
            abs(min(value, 0.0))
            for value in values
        )

        if gross_loss > 0:
            pf = gross_profit / gross_loss
        else:
            pf = float("inf") if gross_profit > 0 else 0.0

        return {
            "n": len(values),
            "wins": wins,
            "win_rate": (
                wins / len(values) * 100
                if values
                else 0.0
            ),
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "pf": pf,
            "r": sum(values),
        }

    all_valid = valid_trades(trades)

    baseline = performance(all_valid)

    actual_filtered = filtered_sample(all_valid)

    frozen = performance(actual_filtered)

    full_delta = frozen["r"] - baseline["r"]

    print()
    print("=" * 120)
    print("FULL FROZEN-RULE REFERENCE")
    print("=" * 120)

    print(
        f"Valid trades       : {baseline['n']}"
    )
    print(
        f"Retained trades    : {frozen['n']}"
    )
    print(
        f"Rejected trades    : "
        f"{baseline['n'] - frozen['n']}"
    )
    print(
        f"Baseline R         : {baseline['r']:.2f}"
    )
    print(
        f"Filtered R         : {frozen['r']:.2f}"
    )
    print(
        f"Full Delta R       : {full_delta:.2f}"
    )

    dimension_results = []

    def test_dimension(
        dimension_name,
        values,
        value_getter
    ):

        print()
        print("=" * 120)
        print(
            f"LEAVE-ONE-{dimension_name.upper()}-OUT"
        )
        print("=" * 120)

        print()
        print(
            f"{'Removed Group':<25}"
            f"{'N':>6}"
            f"{'Remain':>8}"
            f"{'BaselineR':>12}"
            f"{'FilteredR':>12}"
            f"{'DeltaR':>12}"
            f"{'Win%':>9}"
            f"{'PF':>9}"
        )
        print("-" * 120)

        for value in values:

            remaining = [
                trade
                for trade in all_valid
                if value_getter(trade) != value
            ]

            remaining_filtered = filtered_sample(
                remaining
            )

            remaining_baseline = performance(
                remaining
            )

            remaining_frozen = performance(
                remaining_filtered
            )

            delta_r = (
                remaining_frozen["r"]
                - remaining_baseline["r"]
            )

            pf_text = (
                f"{remaining_frozen['pf']:.2f}"
                if remaining_frozen["pf"] != float("inf")
                else "INF"
            )

            print(
                f"{str(value):<25}"
                f"{remaining_baseline['n']:>6}"
                f"{remaining_frozen['n']:>8}"
                f"{remaining_baseline['r']:>12.2f}"
                f"{remaining_frozen['r']:>12.2f}"
                f"{delta_r:>12.2f}"
                f"{remaining_frozen['win_rate']:>8.2f}%"
                f"{pf_text:>9}"
            )

            dimension_results.append({
                "dimension": dimension_name,
                "value": value,
                "remaining_n": remaining_baseline["n"],
                "remaining_filtered_n": remaining_frozen["n"],
                "baseline_r": remaining_baseline["r"],
                "filtered_r": remaining_frozen["r"],
                "delta_r": delta_r,
            })

    years = sorted(
        {
            get_year(trade)
            for trade in all_valid
            if get_year(trade) is not None
        }
    )

    symbols = sorted(
        {
            get_symbol(trade)
            for trade in all_valid
        }
    )

    directions = sorted(
        {
            get_direction(trade)
            for trade in all_valid
        }
    )

    test_dimension(
        "YEAR",
        years,
        get_year
    )

    test_dimension(
        "STOCK",
        symbols,
        get_symbol
    )

    test_dimension(
        "DIRECTION",
        directions,
        get_direction
    )

    print()
    print("=" * 120)
    print("DIMENSION ROBUSTNESS SUMMARY")
    print("=" * 120)

    for dimension in [
        "YEAR",
        "STOCK",
        "DIRECTION"
    ]:

        rows = [
            row
            for row in dimension_results
            if row["dimension"] == dimension
        ]

        positive_filtered = sum(
            1
            for row in rows
            if row["filtered_r"] > 0
        )

        positive_delta = sum(
            1
            for row in rows
            if row["delta_r"] > 0
        )

        print()
        print(
            f"{dimension}:"
        )

        print(
            f"  Groups tested          : {len(rows)}"
        )

        print(
            f"  Filtered R > 0         : "
            f"{positive_filtered}/{len(rows)}"
        )

        print(
            f"  Delta R > 0            : "
            f"{positive_delta}/{len(rows)}"
        )

    all_filtered_positive = all(
        row["filtered_r"] > 0
        for row in dimension_results
    )

    all_delta_positive = all(
        row["delta_r"] > 0
        for row in dimension_results
    )

    if (
        all_filtered_positive
        and all_delta_positive
    ):
        verdict = "STRONG DIMENSIONAL ROBUSTNESS"
    elif (
        all_delta_positive
    ):
        verdict = "POSITIVE EDGE — SOME SUBGROUP DEPENDENCE"
    else:
        verdict = "DIMENSIONAL FRAGILITY DETECTED"

    print()
    print("=" * 120)
    print("LEAVE-ONE-DIMENSION-OUT DECISION")
    print("=" * 120)

    print(
        f"Dimension robustness verdict: {verdict}"
    )

    if verdict == "STRONG DIMENSIONAL ROBUSTNESS":
        print(
            "Interpretation: the frozen historical edge remains "
            "positive after removing every individual year, "
            "stock, and direction."
        )
    elif verdict == "POSITIVE EDGE — SOME SUBGROUP DEPENDENCE":
        print(
            "Interpretation: the frozen rule retains positive "
            "incremental edge across the tested dimensions, "
            "but at least one exclusion causes filtered R to "
            "become non-positive."
        )
    else:
        print(
            "Interpretation: at least one dimension exclusion "
            "eliminates the positive Delta R. The historical "
            "edge may depend materially on that subgroup."
        )

    print()
    print("IMPORTANT:")
    print(
        "  This is a retrospective robustness test."
    )
    print(
        "  It does NOT prove future profitability."
    )
    print(
        "  It does NOT replace the genuine post-freeze holdout."
    )
    print(
        "  It does NOT approve production deployment."
    )
    print(
        "  The frozen 5B_0.25R rule remains unchanged."
    )

    print()
    print("RESEARCH STATUS: DO NOT DEPLOY")
    print("=" * 120)

def frozen_early_adverse_randomization_audit(trades):
    """
    Research-only statistical randomization audit.

    Frozen rule:
        Reject a trade only when
        early_adverse_r_bar_5 >= 0.25R.

    Purpose:
        Test whether the observed +DeltaR from the frozen rule
        is materially stronger than what would be expected from
        randomly selecting the same number of trades.

    IMPORTANT:
        This does NOT optimize:
            - threshold
            - window
            - stocks
            - setups
            - trends
            - years
            - trade count

        The actual frozen rule determines the observed retained
        trade count.

    The randomization test preserves:
        - total number of valid trades
        - total number of retained trades

    It randomly assigns the same number of trades to a hypothetical
    retained group and calculates the resulting DeltaR.

    This is retrospective statistical evidence only.
    It is NOT a future holdout and does NOT approve deployment.
    """

    import random

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    RANDOM_SEED = 20260908
    N_PERMUTATIONS = 10000

    print()
    print("=" * 150)
    print(
        "FROZEN 5B_0.25R RANDOMIZATION / STATISTICAL SIGNIFICANCE AUDIT"
    )
    print("=" * 150)

    print()
    print("Frozen rule:")
    print(
        "  Reject when "
        "early_adverse_r_bar_5 >= 0.25R"
    )

    print()
    print(
        "Purpose:"
    )
    print(
        "  Test whether the observed filter improvement is "
        "stronger than random trade selection."
    )

    print()
    print(
        f"Permutations       : {N_PERMUTATIONS}"
    )

    print(
        f"Random seed        : {RANDOM_SEED}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  The actual frozen rule is NOT changed."
    )

    print(
        "  This is a statistical stress test only."
    )

    def get_r(trade):

        return to_float(
            trade.get(
                "r_multiple"
            ),
            default=None
        )

    def passes_rule(trade):

        early_adverse = to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

        if early_adverse is None:

            return True

        return (
            early_adverse
            < RULE_THRESHOLD
        )

    valid_trades = [
        trade
        for trade in trades
        if get_r(trade) is not None
    ]

    if not valid_trades:

        print()
        print(
            "No valid trades available."
        )

        print(
            "RANDOMIZATION AUDIT: NOT READY"
        )

        return

    actual_retained = [
        trade
        for trade in valid_trades
        if passes_rule(trade)
    ]

    baseline_r = sum(
        get_r(trade)
        for trade in valid_trades
    )

    actual_filtered_r = sum(
        get_r(trade)
        for trade in actual_retained
    )

    actual_delta_r = (
        actual_filtered_r
        - baseline_r
    )

    total_n = len(
        valid_trades
    )

    retained_n = len(
        actual_retained
    )

    print()
    print("=" * 150)
    print(
        "OBSERVED FROZEN-RULE RESULT"
    )
    print("=" * 150)

    print(
        f"Valid trades       : {total_n}"
    )

    print(
        f"Actual retained    : {retained_n}"
    )

    print(
        f"Actual rejected    : "
        f"{total_n - retained_n}"
    )

    print(
        f"Baseline R         : "
        f"{baseline_r:.2f}"
    )

    print(
        f"Filtered R         : "
        f"{actual_filtered_r:.2f}"
    )

    print(
        f"Observed Delta R   : "
        f"{actual_delta_r:.2f}"
    )

    if retained_n <= 0 or retained_n >= total_n:

        print()
        print(
            "Randomization test cannot be performed because "
            "the frozen rule does not create two groups."
        )

        print(
            "RANDOMIZATION AUDIT: NOT READY"
        )

        return

    rng = random.Random(
        RANDOM_SEED
    )

    all_r = [
        get_r(trade)
        for trade in valid_trades
    ]

    permutation_deltas = []

    trade_indices = list(
        range(total_n)
    )

    for _ in range(
        N_PERMUTATIONS
    ):

        selected_indices = rng.sample(
            trade_indices,
            retained_n
        )

        random_filtered_r = sum(
            all_r[index]
            for index in selected_indices
        )

        random_delta_r = (
            random_filtered_r
            - baseline_r
        )

        permutation_deltas.append(
            random_delta_r
        )

    permutation_deltas.sort()

    mean_random_delta = (
        sum(permutation_deltas)
        / len(permutation_deltas)
    )

    median_random_delta = (
        permutation_deltas[
            len(permutation_deltas) // 2
        ]
    )

    p_count = sum(
        1
        for delta in permutation_deltas
        if delta >= actual_delta_r
    )

    empirical_p_value = (
        (p_count + 1)
        / (N_PERMUTATIONS + 1)
    )

    percentile_count = sum(
        1
        for delta in permutation_deltas
        if delta <= actual_delta_r
    )

    empirical_percentile = (
        percentile_count
        / N_PERMUTATIONS
        * 100
    )

    lower_index = int(
        0.025
        * N_PERMUTATIONS
    )

    upper_index = int(
        0.975
        * N_PERMUTATIONS
    ) - 1

    lower_index = max(
        0,
        min(
            lower_index,
            N_PERMUTATIONS - 1
        )
    )

    upper_index = max(
        0,
        min(
            upper_index,
            N_PERMUTATIONS - 1
        )
    )

    lower_95 = permutation_deltas[
        lower_index
    ]

    upper_95 = permutation_deltas[
        upper_index
    ]

    print()
    print("=" * 150)
    print(
        "RANDOMIZATION DISTRIBUTION"
    )
    print("=" * 150)

    print(
        f"Mean random Delta R       : "
        f"{mean_random_delta:.2f}"
    )

    print(
        f"Median random Delta R     : "
        f"{median_random_delta:.2f}"
    )

    print(
        f"Random 2.5th percentile   : "
        f"{lower_95:.2f}"
    )

    print(
        f"Random 97.5th percentile  : "
        f"{upper_95:.2f}"
    )

    print()
    print(
        f"Observed Delta R          : "
        f"{actual_delta_r:.2f}"
    )

    print(
        f"Observed percentile       : "
        f"{empirical_percentile:.2f}%"
    )

    print(
        f"Empirical one-sided p     : "
        f"{empirical_p_value:.4f}"
    )

    print(
        f"Permutations >= observed  : "
        f"{p_count}"
    )

    print()
    print("=" * 150)
    print(
        "RANDOMIZATION INTERPRETATION"
    )
    print("=" * 150)

    if (
        actual_delta_r > 0
        and empirical_p_value < 0.01
    ):

        verdict = (
            "STRONG STATISTICAL SIGNAL"
        )

        explanation = (
            "The observed frozen-rule improvement is "
            "rare under random selection of the same "
            "number of trades."
        )

    elif (
        actual_delta_r > 0
        and empirical_p_value < 0.05
    ):

        verdict = (
            "STATISTICALLY PROMISING"
        )

        explanation = (
            "The observed improvement is stronger than "
            "most random selections of the same size."
        )

    elif (
        actual_delta_r > 0
        and empirical_p_value < 0.10
    ):

        verdict = (
            "WEAK STATISTICAL SIGNAL"
        )

        explanation = (
            "The observed improvement is positive but "
            "not especially rare under random selection."
        )

    else:

        verdict = (
            "NO CLEAR STATISTICAL EDGE"
        )

        explanation = (
            "The observed improvement is not sufficiently "
            "rare under the predefined randomization test."
        )

    print(
        f"Randomization verdict : "
        f"{verdict}"
    )

    print(
        f"Interpretation         : "
        f"{explanation}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  This does NOT prove causality."
    )

    print(
        "  This does NOT replace chronological validation."
    )

    print(
        "  This does NOT replace the genuine future holdout."
    )

    print(
        "  This does NOT approve production deployment."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

def frozen_early_adverse_drawdown_risk_audit(trades):
    """
    Research-only drawdown / risk-of-ruin / equity-curve stability audit
    for the already-frozen 5B_0.25R rule.

    Frozen rule:
        Reject only when early_adverse_r_bar_5 >= 0.25R.

    IMPORTANT:
        - Does NOT optimize the threshold.
        - Does NOT modify the strategy.
        - Does NOT use future information to make the filter decision.
        - Uses the frozen counterfactual classification already established.
        - Compares chronological equity-curve behavior of baseline vs filtered trades.
        - This is NOT a future holdout.
    """

    print()
    print("=" * 120)
    print("FROZEN 5B_0.25R DRAWDOWN / RISK / EQUITY-CURVE STABILITY AUDIT")
    print("=" * 120)

    print()
    print("Frozen rule: reject only when Bar-5 early adverse R >= 0.25R")
    print("Purpose: evaluate drawdown, losing streaks, recovery, and equity-curve stability.")
    print("Research-only: no threshold optimization and no strategy modification.")

    # --------------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------------

    def get_r(trade):
        value = to_float(
            trade.get("r_multiple"),
            default=None
        )

        if value is None:
            return None

        return value

    def get_bar5(trade):
        return to_float(
            trade.get("early_adverse_r_bar_5"),
            default=None
        )

    def get_entry_date(trade):
        value = str(
            trade.get(
                "entry_date",
                ""
            )
        ).strip()

        if not value:
            return None

        try:
            return datetime.strptime(
                value[:10],
                "%Y-%m-%d"
            )
        except ValueError:
            return None

    def is_rejected(trade):
        bar5 = get_bar5(trade)

        return (
            bar5 is not None
            and bar5 >= 0.25
        )

    # --------------------------------------------------------------
    # PREPARE VALID TRADES
    # --------------------------------------------------------------

    valid_trades = []

    for trade in trades:

        entry_date = get_entry_date(
            trade
        )

        r_multiple = get_r(
            trade
        )

        if entry_date is None:
            continue

        if r_multiple is None:
            continue

        valid_trades.append(
            (
                entry_date,
                trade
            )
        )

    valid_trades.sort(
        key=lambda item: (
            item[0],
            str(
                item[1].get(
                    "symbol",
                    ""
                )
            ),
            str(
                item[1].get(
                    "direction",
                    ""
                )
            ),
            str(
                item[1].get(
                    "entry_price",
                    ""
                )
            )
        )
    )

    if not valid_trades:
        print()
        print("No valid trades available for drawdown audit.")
        return

    baseline_trades = [
        trade
        for _, trade in valid_trades
    ]

    filtered_trades = [
        trade
        for _, trade in valid_trades
        if not is_rejected(trade)
    ]

    # --------------------------------------------------------------
    # EQUITY CURVE METRICS
    # --------------------------------------------------------------

    def equity_curve_metrics(
        sample_trades
    ):
        if not sample_trades:
            return {
                "trades": 0,
                "total_r": 0.0,
                "max_drawdown_r": 0.0,
                "max_drawdown_pct_of_peak_r": 0.0,
                "longest_losing_streak": 0,
                "worst_single_trade": 0.0,
                "worst_5_trade_sum": 0.0,
                "worst_10_trade_sum": 0.0,
                "recovery_trades": None,
                "recovery_days": None,
                "peak_r": 0.0,
                "ending_r": 0.0,
            }

        running_r = 0.0
        peak_r = 0.0
        max_drawdown_r = 0.0

        longest_losing_streak = 0
        current_losing_streak = 0

        worst_single_trade = 0.0

        cumulative_values = []

        for trade in sample_trades:

            r_multiple = get_r(
                trade
            )

            if r_multiple is None:
                continue

            running_r += r_multiple

            cumulative_values.append(
                running_r
            )

            if running_r > peak_r:
                peak_r = running_r

            drawdown = (
                peak_r
                - running_r
            )

            if drawdown > max_drawdown_r:
                max_drawdown_r = drawdown

            if r_multiple <= 0:
                current_losing_streak += 1

                if (
                    current_losing_streak
                    > longest_losing_streak
                ):
                    longest_losing_streak = (
                        current_losing_streak
                    )
            else:
                current_losing_streak = 0

            if r_multiple < worst_single_trade:
                worst_single_trade = r_multiple

        # ----------------------------------------------------------
        # WORST CONSECUTIVE TRADE WINDOWS
        # ----------------------------------------------------------

        def worst_window(
            values,
            window_size
        ):
            if len(values) < window_size:
                return None

            worst = None

            for index in range(
                0,
                len(values) - window_size + 1
            ):
                window_sum = sum(
                    values[
                        index:
                        index + window_size
                    ]
                )

                if (
                    worst is None
                    or window_sum < worst
                ):
                    worst = window_sum

            return worst

        worst_5_trade_sum = (
            worst_window(
                [
                    get_r(trade)
                    for trade in sample_trades
                    if get_r(trade) is not None
                ],
                5
            )
        )

        worst_10_trade_sum = (
            worst_window(
                [
                    get_r(trade)
                    for trade in sample_trades
                    if get_r(trade) is not None
                ],
                10
            )
        )

        if worst_5_trade_sum is None:
            worst_5_trade_sum = 0.0

        if worst_10_trade_sum is None:
            worst_10_trade_sum = 0.0

        # ----------------------------------------------------------
        # RECOVERY ANALYSIS
        # ----------------------------------------------------------

        running_r = 0.0
        peak_index = 0
        peak_value = 0.0

        recovery_trades = None
        recovery_days = None

        trade_dates = [
            get_entry_date(trade)
            for trade in sample_trades
        ]

        for index, trade in enumerate(
            sample_trades
        ):

            r_multiple = get_r(
                trade
            )

            if r_multiple is None:
                continue

            running_r += r_multiple

            if running_r >= peak_value:
                peak_value = running_r
                peak_index = index
                continue

            if (
                recovery_trades is None
                and index > peak_index
                and running_r >= peak_value
            ):
                recovery_trades = (
                    index - peak_index
                )

                if (
                    peak_index
                    < len(trade_dates)
                    and index
                    < len(trade_dates)
                ):
                    start_date = trade_dates[
                        peak_index
                    ]

                    end_date = trade_dates[
                        index
                    ]

                    if (
                        start_date is not None
                        and end_date is not None
                    ):
                        recovery_days = (
                            end_date
                            - start_date
                        ).days
                break

        # ----------------------------------------------------------
        # MORE RELIABLE RECOVERY MEASUREMENT
        # ----------------------------------------------------------

        running_r = 0.0
        peak_value = 0.0

        max_drawdown_recovery_trades = 0
        max_drawdown_recovery_days = 0

        drawdown_start_index = None

        for index, trade in enumerate(
            sample_trades
        ):

            r_multiple = get_r(
                trade
            )

            if r_multiple is None:
                continue

            running_r += r_multiple

            if running_r >= peak_value:

                if (
                    drawdown_start_index
                    is not None
                ):

                    recovery_trade_count = (
                        index
                        - drawdown_start_index
                    )

                    if (
                        recovery_trade_count
                        > max_drawdown_recovery_trades
                    ):
                        max_drawdown_recovery_trades = (
                            recovery_trade_count
                        )

                    start_date = get_entry_date(
                        sample_trades[
                            drawdown_start_index
                        ]
                    )

                    end_date = get_entry_date(
                        trade
                    )

                    if (
                        start_date is not None
                        and end_date is not None
                    ):
                        recovery_day_count = (
                            end_date
                            - start_date
                        ).days

                        if (
                            recovery_day_count
                            > max_drawdown_recovery_days
                        ):
                            max_drawdown_recovery_days = (
                                recovery_day_count
                            )

                peak_value = running_r
                drawdown_start_index = None

            else:

                if drawdown_start_index is None:
                    drawdown_start_index = index

        # ----------------------------------------------------------
        # PEAK-NORMALIZED DRAW DOWN
        # ----------------------------------------------------------

        if peak_r > 0:
            drawdown_pct_of_peak_r = (
                max_drawdown_r
                / peak_r
                * 100.0
            )
        else:
            drawdown_pct_of_peak_r = 0.0

        return {
            "trades": len(sample_trades),
            "total_r": sum(
                get_r(trade)
                for trade in sample_trades
                if get_r(trade) is not None
            ),
            "max_drawdown_r": max_drawdown_r,
            "max_drawdown_pct_of_peak_r": (
                drawdown_pct_of_peak_r
            ),
            "longest_losing_streak": (
                longest_losing_streak
            ),
            "worst_single_trade": (
                worst_single_trade
            ),
            "worst_5_trade_sum": (
                worst_5_trade_sum
            ),
            "worst_10_trade_sum": (
                worst_10_trade_sum
            ),
            "recovery_trades": (
                max_drawdown_recovery_trades
            ),
            "recovery_days": (
                max_drawdown_recovery_days
            ),
            "peak_r": peak_r,
            "ending_r": running_r,
        }

    # --------------------------------------------------------------
    # CALCULATE BASELINE / FILTERED METRICS
    # --------------------------------------------------------------

    baseline_metrics = equity_curve_metrics(
        baseline_trades
    )

    filtered_metrics = equity_curve_metrics(
        filtered_trades
    )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("EQUITY-CURVE SUMMARY")
    print("-" * 120)

    print()
    print(
        f"BASELINE  "
        f"Trades={baseline_metrics['trades']:>3} "
        f"R={baseline_metrics['total_r']:>8.2f} "
        f"MaxDD={baseline_metrics['max_drawdown_r']:>7.2f}R "
        f"LosingStreak={baseline_metrics['longest_losing_streak']:>3} "
        f"Worst1={baseline_metrics['worst_single_trade']:>7.2f}R "
        f"Worst5={baseline_metrics['worst_5_trade_sum']:>7.2f}R "
        f"Worst10={baseline_metrics['worst_10_trade_sum']:>7.2f}R"
    )

    print()

    print(
        f"FILTERED  "
        f"Trades={filtered_metrics['trades']:>3} "
        f"R={filtered_metrics['total_r']:>8.2f} "
        f"MaxDD={filtered_metrics['max_drawdown_r']:>7.2f}R "
        f"LosingStreak={filtered_metrics['longest_losing_streak']:>3} "
        f"Worst1={filtered_metrics['worst_single_trade']:>7.2f}R "
        f"Worst5={filtered_metrics['worst_5_trade_sum']:>7.2f}R "
        f"Worst10={filtered_metrics['worst_10_trade_sum']:>7.2f}R"
    )

    # --------------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("DRAWDOWN RECOVERY")
    print("-" * 120)

    print()
    print(
        "BASELINE:"
    )

    print(
        f"  Max drawdown: "
        f"{baseline_metrics['max_drawdown_r']:.2f}R"
    )

    print(
        f"  Longest recovery: "
        f"{baseline_metrics['recovery_trades']} trades"
    )

    print(
        f"  Longest recovery: "
        f"{baseline_metrics['recovery_days']} calendar days"
    )

    print()

    print(
        "FILTERED:"
    )

    print(
        f"  Max drawdown: "
        f"{filtered_metrics['max_drawdown_r']:.2f}R"
    )

    print(
        f"  Longest recovery: "
        f"{filtered_metrics['recovery_trades']} trades"
    )

    print(
        f"  Longest recovery: "
        f"{filtered_metrics['recovery_days']} calendar days"
    )

    # --------------------------------------------------------------
    # DRAWDOWN IMPROVEMENT
    # --------------------------------------------------------------

    baseline_dd = (
        baseline_metrics[
            "max_drawdown_r"
        ]
    )

    filtered_dd = (
        filtered_metrics[
            "max_drawdown_r"
        ]
    )

    dd_improvement = (
        baseline_dd
        - filtered_dd
    )

    print()
    print("-" * 120)
    print("DRAWDOWN IMPACT")
    print("-" * 120)

    print()
    print(
        f"Baseline MaxDD: "
        f"{baseline_dd:.2f}R"
    )

    print(
        f"Filtered MaxDD: "
        f"{filtered_dd:.2f}R"
    )

    print(
        f"MaxDD improvement: "
        f"{dd_improvement:+.2f}R"
    )

    if baseline_dd > 0:
        dd_improvement_pct = (
            dd_improvement
            / baseline_dd
            * 100.0
        )
    else:
        dd_improvement_pct = 0.0

    print(
        f"MaxDD improvement %: "
        f"{dd_improvement_pct:+.2f}%"
    )

    # --------------------------------------------------------------
    # CAPITAL REQUIREMENT
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("R-BASED CAPITAL / RISK REQUIREMENT")
    print("-" * 120)

    print()
    print(
        "Because the audit is expressed in R, "
        "capital requirement is shown as a multiple of 1R."
    )

    print()

    print(
        f"Baseline worst drawdown requirement: "
        f"{baseline_dd:.2f}R"
    )

    print(
        f"Filtered worst drawdown requirement: "
        f"{filtered_dd:.2f}R"
    )

    print()

    print(
        "Example:"
    )

    print(
        "If 1R = ₹10,000 risk per trade, "
        "multiply the R drawdown by ₹10,000 "
        "to estimate the historical peak-to-trough capital impact."
    )

    # --------------------------------------------------------------
    # YEAR-BY-YEAR DRAWDOWN
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("YEAR-BY-YEAR DRAWDOWN")
    print("=" * 120)

    years = sorted(
        {
            entry_date.year
            for entry_date, _ in valid_trades
        }
    )

    for year in years:

        year_baseline = [
            trade
            for entry_date, trade in valid_trades
            if entry_date.year == year
        ]

        year_filtered = [
            trade
            for trade in year_baseline
            if not is_rejected(trade)
        ]

        baseline_year_metrics = (
            equity_curve_metrics(
                year_baseline
            )
        )

        filtered_year_metrics = (
            equity_curve_metrics(
                year_filtered
            )
        )

        print()
        print(
            f"{year}: "
            f"BaselineR={baseline_year_metrics['total_r']:>7.2f} "
            f"BaselineDD={baseline_year_metrics['max_drawdown_r']:>6.2f}R "
            f"FilteredR={filtered_year_metrics['total_r']:>7.2f} "
            f"FilteredDD={filtered_year_metrics['max_drawdown_r']:>6.2f}R "
            f"FilteredStreak={filtered_year_metrics['longest_losing_streak']:>2}"
        )

    # --------------------------------------------------------------
    # RISK VERDICT
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("RISK / EQUITY-CURVE VERDICT")
    print("=" * 120)

    dd_better = (
        filtered_dd < baseline_dd
    )

    streak_better = (
        filtered_metrics[
            "longest_losing_streak"
        ]
        <=
        baseline_metrics[
            "longest_losing_streak"
        ]
    )

    worst_window_better = (
        filtered_metrics[
            "worst_10_trade_sum"
        ]
        >=
        baseline_metrics[
            "worst_10_trade_sum"
        ]
    )

    positive_total = (
        filtered_metrics[
            "total_r"
        ] > 0
    )

    checks_passed = sum(
        [
            dd_better,
            streak_better,
            worst_window_better,
            positive_total,
        ]
    )

    print()
    print(
        f"MaxDD improved: "
        f"{'PASS' if dd_better else 'FAIL'}"
    )

    print(
        f"Losing-streak stability: "
        f"{'PASS' if streak_better else 'FAIL'}"
    )

    print(
        f"Worst-10-trade stability: "
        f"{'PASS' if worst_window_better else 'FAIL'}"
    )

    print(
        f"Filtered total R positive: "
        f"{'PASS' if positive_total else 'FAIL'}"
    )

    print()

    if checks_passed == 4:

        print(
            "VERDICT: STRONG EQUITY-CURVE / RISK SIGNAL"
        )

    elif checks_passed >= 3:

        print(
            "VERDICT: PROMISING EQUITY-CURVE / RISK SIGNAL"
        )

    elif checks_passed >= 2:

        print(
            "VERDICT: MIXED EQUITY-CURVE / RISK SIGNAL"
        )

    else:

        print(
            "VERDICT: WEAK EQUITY-CURVE / RISK SIGNAL"
        )

    print()
    print(
        "IMPORTANT: This audit does not establish production readiness."
    )

    print(
        "It is retrospective and must be followed by genuine future holdout "
        "and paper/shadow validation."
    )

def frozen_early_adverse_monte_carlo_audit(trades):
    """
    Research-only Monte Carlo trade-sequence stress test.

    Purpose:
        Evaluate how the frozen 5B_0.25R retained trade set behaves
        under randomized trade ordering.

    Frozen rule:
        Reject when early_adverse_r_bar_5 >= 0.25R

    IMPORTANT:
        - Does not change the frozen strategy.
        - Does not optimize any threshold.
        - Does not add/remove trades.
        - Uses exactly the historically retained trades.
        - Randomizes only trade sequence/order.
        - This is a risk-path stress test, not future validation.
    """

    MONTE_CARLO_RUNS = 20000
    RANDOM_SEED = 20260908

    print()
    print("=" * 140)
    print(
        "FROZEN 5B_0.25R MONTE CARLO / TRADE-SEQUENCE STRESS TEST"
    )
    print("=" * 140)

    print()
    print(
        "Frozen rule: reject only when Bar-5 early adverse R >= 0.25R"
    )
    print(
        "Purpose: evaluate drawdown, losing streak, recovery, and "
        "capital-risk behavior under randomized trade ordering."
    )
    print(
        "Research-only: no threshold optimization and no strategy modification."
    )

    # --------------------------------------------------------------
    # Build the exact historically retained trade set.
    # --------------------------------------------------------------

    valid_trades = []

    for trade in trades:

        final_r = to_float(
            trade.get(
                "r_multiple"
            )
        )

        if final_r is None:
            continue

        early_adverse = to_float(
            trade.get(
                "early_adverse_r_bar_5"
            )
        )

        # Missing Bar-5 values are retained under the frozen rule.
        if (
            early_adverse is None
            or early_adverse < 0.25
        ):
            valid_trades.append(
                trade
            )

    if not valid_trades:

        print()
        print(
            "Monte Carlo status: NOT RUN"
        )
        print(
            "Reason: no valid retained trades."
        )
        print("=" * 140)

        return

    observed_r_values = [
        to_float(
            trade.get(
                "r_multiple"
            )
        )
        for trade in valid_trades
    ]

    observed_r_values = [
        value
        for value in observed_r_values
        if value is not None
    ]

    if not observed_r_values:

        print()
        print(
            "Monte Carlo status: NOT RUN"
        )
        print(
            "Reason: retained trades contain no valid Final-R values."
        )
        print("=" * 140)

        return

    baseline_r_values = []

    for trade in trades:

        final_r = to_float(
            trade.get(
                "r_multiple"
            )
        )

        if final_r is not None:
            baseline_r_values.append(
                final_r
            )

    baseline_r = sum(
        baseline_r_values
    )

    observed_filtered_r = sum(
        observed_r_values
    )

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    def path_metrics(values):

        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        current_losing_streak = 0
        longest_losing_streak = 0

        worst_trade = 0.0

        worst_5 = 0.0
        worst_10 = 0.0

        peak_trade_index = 0
        recovery_lengths = []
        unrecovered_from_peak = False

        for index, value in enumerate(values):

            equity += value

            if value < worst_trade:
                worst_trade = value

            if value < 0:
                current_losing_streak += 1

                longest_losing_streak = max(
                    longest_losing_streak,
                    current_losing_streak
                )
            else:
                current_losing_streak = 0

            if equity > peak:

                if (
                    index > peak_trade_index
                    and peak > 0
                ):

                    recovery_length = (
                        index
                        - peak_trade_index
                    )

                    recovery_lengths.append(
                        recovery_length
                    )

                peak = equity
                peak_trade_index = index

            drawdown = (
                peak
                - equity
            )

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

        if equity < peak:
            unrecovered_from_peak = True

        longest_recovery = (
            max(
                recovery_lengths
            )
            if recovery_lengths
            else 0
        )

        return {
            "ending_r": equity,
            "max_drawdown": max_drawdown,
            "losing_streak": longest_losing_streak,
            "worst_trade": worst_trade,
            "longest_recovery": longest_recovery,
            "unrecovered": unrecovered_from_peak,
        }

    def percentile(values, probability):

        if not values:
            return 0.0

        ordered = sorted(
            values
        )

        position = (
            probability
            * (len(ordered) - 1)
        )

        lower = int(
            position
        )

        upper = min(
            lower + 1,
            len(ordered) - 1
        )

        fraction = (
            position
            - lower
        )

        return (
            ordered[lower]
            + (
                ordered[upper]
                - ordered[lower]
            )
            * fraction
        )

    # --------------------------------------------------------------
    # Observed chronological path
    # --------------------------------------------------------------

    observed_metrics = path_metrics(
        observed_r_values
    )

    # --------------------------------------------------------------
    # Monte Carlo
    # --------------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    max_drawdowns = []
    losing_streaks = []
    ending_r_values = []
    recovery_lengths = []

    unrecovered_count = 0

    dd_5_count = 0
    dd_10_count = 0
    dd_15_count = 0
    dd_20_count = 0

    for _ in range(
        MONTE_CARLO_RUNS
    ):

        randomized_values = list(
            observed_r_values
        )

        rng.shuffle(
            randomized_values
        )

        metrics = path_metrics(
            randomized_values
        )

        max_drawdown = metrics[
            "max_drawdown"
        ]

        max_drawdowns.append(
            max_drawdown
        )

        losing_streaks.append(
            metrics[
                "losing_streak"
            ]
        )

        ending_r_values.append(
            metrics[
                "ending_r"
            ]
        )

        if metrics[
            "unrecovered"
        ]:

            unrecovered_count += 1

        else:

            recovery_lengths.append(
                metrics[
                    "longest_recovery"
                ]
            )

        if max_drawdown >= 5.0:
            dd_5_count += 1

        if max_drawdown >= 10.0:
            dd_10_count += 1

        if max_drawdown >= 15.0:
            dd_15_count += 1

        if max_drawdown >= 20.0:
            dd_20_count += 1

    # --------------------------------------------------------------
    # Monte Carlo distributions
    # --------------------------------------------------------------

    mean_max_dd = (
        sum(max_drawdowns)
        / len(max_drawdowns)
    )

    median_max_dd = percentile(
        max_drawdowns,
        0.50
    )

    p95_max_dd = percentile(
        max_drawdowns,
        0.95
    )

    p99_max_dd = percentile(
        max_drawdowns,
        0.99
    )

    mean_losing_streak = (
        sum(losing_streaks)
        / len(losing_streaks)
    )

    median_losing_streak = percentile(
        losing_streaks,
        0.50
    )

    p95_losing_streak = percentile(
        losing_streaks,
        0.95
    )

    p99_losing_streak = percentile(
        losing_streaks,
        0.99
    )

    unrecovered_probability = (
        unrecovered_count
        / MONTE_CARLO_RUNS
        * 100
    )

    # --------------------------------------------------------------
    # Capital-risk stress test
    #
    # Fixed risk-per-trade is expressed as a percentage of starting
    # capital. This deliberately uses a simple fixed-risk model.
    # --------------------------------------------------------------

    risk_levels = [
        0.25,
        0.50,
        1.00,
        2.00,
        3.00,
        5.00,
    ]

    risk_results = []

    for risk_percent in risk_levels:

        loss_10 = 0
        loss_20 = 0
        loss_30 = 0
        loss_40 = 0
        loss_50 = 0
        ruin = 0

        for max_dd in max_drawdowns:

            capital_drawdown_percent = (
                max_dd
                * risk_percent
            )

            if capital_drawdown_percent >= 10:
                loss_10 += 1

            if capital_drawdown_percent >= 20:
                loss_20 += 1

            if capital_drawdown_percent >= 30:
                loss_30 += 1

            if capital_drawdown_percent >= 40:
                loss_40 += 1

            if capital_drawdown_percent >= 50:
                loss_50 += 1

            if capital_drawdown_percent >= 100:
                ruin += 1

        risk_results.append(
            {
                "risk": risk_percent,
                "loss_10": (
                    loss_10
                    / MONTE_CARLO_RUNS
                    * 100
                ),
                "loss_20": (
                    loss_20
                    / MONTE_CARLO_RUNS
                    * 100
                ),
                "loss_30": (
                    loss_30
                    / MONTE_CARLO_RUNS
                    * 100
                ),
                "loss_40": (
                    loss_40
                    / MONTE_CARLO_RUNS
                    * 100
                ),
                "loss_50": (
                    loss_50
                    / MONTE_CARLO_RUNS
                    * 100
                ),
                "ruin": (
                    ruin
                    / MONTE_CARLO_RUNS
                    * 100
                ),
            }
        )

    # --------------------------------------------------------------
    # Recovery statistics
    # --------------------------------------------------------------

    if recovery_lengths:

        mean_recovery = (
            sum(recovery_lengths)
            / len(recovery_lengths)
        )

        median_recovery = percentile(
            recovery_lengths,
            0.50
        )

        p95_recovery = percentile(
            recovery_lengths,
            0.95
        )

    else:

        mean_recovery = 0.0
        median_recovery = 0.0
        p95_recovery = 0.0

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    print()
    print("-" * 140)
    print(
        "MONTE CARLO CONFIGURATION"
    )
    print("-" * 140)

    print(
        f"Retained trades       : {len(observed_r_values)}"
    )

    print(
        f"Monte Carlo runs      : {MONTE_CARLO_RUNS}"
    )

    print(
        f"Random seed           : {RANDOM_SEED}"
    )

    print(
        f"Baseline R            : {baseline_r:.2f}"
    )

    print(
        f"Observed filtered R   : {observed_filtered_r:.2f}"
    )

    print(
        "Ending R after shuffle: "
        f"{sum(ending_r_values) / len(ending_r_values):.2f}"
    )

    print()
    print("-" * 140)
    print(
        "OBSERVED CHRONOLOGICAL PATH"
    )
    print("-" * 140)

    print(
        f"Observed MaxDD        : "
        f"{observed_metrics['max_drawdown']:.2f}R"
    )

    print(
        f"Observed losing streak: "
        f"{observed_metrics['losing_streak']}"
    )

    print(
        f"Observed longest recovery: "
        f"{observed_metrics['longest_recovery']} trades"
    )

    print()
    print("-" * 140)
    print(
        "MONTE CARLO MAXIMUM DRAWDOWN DISTRIBUTION"
    )
    print("-" * 140)

    print(
        f"Mean MaxDD            : "
        f"{mean_max_dd:.2f}R"
    )

    print(
        f"Median MaxDD          : "
        f"{median_max_dd:.2f}R"
    )

    print(
        f"95th percentile MaxDD : "
        f"{p95_max_dd:.2f}R"
    )

    print(
        f"99th percentile MaxDD : "
        f"{p99_max_dd:.2f}R"
    )

    print()
    print(
        f"P(MaxDD >= 5R)       : "
        f"{dd_5_count / MONTE_CARLO_RUNS * 100:.2f}%"
    )

    print(
        f"P(MaxDD >= 10R)      : "
        f"{dd_10_count / MONTE_CARLO_RUNS * 100:.2f}%"
    )

    print(
        f"P(MaxDD >= 15R)      : "
        f"{dd_15_count / MONTE_CARLO_RUNS * 100:.2f}%"
    )

    print(
        f"P(MaxDD >= 20R)      : "
        f"{dd_20_count / MONTE_CARLO_RUNS * 100:.2f}%"
    )

    print()
    print("-" * 140)
    print(
        "LOSING-STREAK DISTRIBUTION"
    )
    print("-" * 140)

    print(
        f"Mean losing streak    : "
        f"{mean_losing_streak:.2f}"
    )

    print(
        f"Median losing streak  : "
        f"{median_losing_streak:.0f}"
    )

    print(
        f"95th percentile       : "
        f"{p95_losing_streak:.0f}"
    )

    print(
        f"99th percentile       : "
        f"{p99_losing_streak:.0f}"
    )

    print()
    print("-" * 140)
    print(
        "RECOVERY STRESS"
    )
    print("-" * 140)

    print(
        f"Paths not recovered to final peak: "
        f"{unrecovered_probability:.2f}%"
    )

    print(
        f"Mean longest recovery "
        f"(recovered paths): "
        f"{mean_recovery:.2f} trades"
    )

    print(
        f"Median longest recovery "
        f"(recovered paths): "
        f"{median_recovery:.0f} trades"
    )

    print(
        f"95th percentile recovery "
        f"(recovered paths): "
        f"{p95_recovery:.0f} trades"
    )

    print()
    print("-" * 140)
    print(
        "CAPITAL RISK STRESS TEST"
    )
    print("-" * 140)

    print(
        "Risk/Trade | P(>=10% DD) | P(>=20% DD) | "
        "P(>=30% DD) | P(>=40% DD) | P(>=50% DD) | P(Ruin)"
    )

    print("-" * 140)

    for result in risk_results:

        print(
            f"{result['risk']:>8.2f}% | "
            f"{result['loss_10']:>11.2f}% | "
            f"{result['loss_20']:>11.2f}% | "
            f"{result['loss_30']:>11.2f}% | "
            f"{result['loss_40']:>11.2f}% | "
            f"{result['loss_50']:>11.2f}% | "
            f"{result['ruin']:>7.2f}%"
        )

    # --------------------------------------------------------------
    # Practical risk interpretation
    # --------------------------------------------------------------

    print()
    print("-" * 140)
    print(
        "PRACTICAL RISK INTERPRETATION"
    )
    print("-" * 140)

    if p99_max_dd <= 10:

        risk_verdict = (
            "LOWER HISTORICAL SEQUENCE RISK"
        )

    elif p99_max_dd <= 15:

        risk_verdict = (
            "MODERATE HISTORICAL SEQUENCE RISK"
        )

    elif p99_max_dd <= 20:

        risk_verdict = (
            "ELEVATED HISTORICAL SEQUENCE RISK"
        )

    else:

        risk_verdict = (
            "HIGH HISTORICAL SEQUENCE RISK"
        )

    print(
        f"Monte Carlo risk verdict: {risk_verdict}"
    )

    print(
        "The 95th/99th percentile MaxDD should be treated "
        "as a stress envelope, not a guaranteed future limit."
    )

    print(
        "Risk-per-trade should be chosen so that a severe "
        "Monte Carlo drawdown remains financially tolerable."
    )

    # --------------------------------------------------------------
    # Final research verdict
    # --------------------------------------------------------------

    if (
        observed_metrics["max_drawdown"]
        <= p95_max_dd
        and p99_max_dd <= 20
        and p95_losing_streak <= 10
        and observed_filtered_r > 0
    ):

        verdict = (
            "PASS — HISTORICAL PATH IS WITHIN A REASONABLE "
            "MONTE CARLO STRESS ENVELOPE"
        )

    elif (
        observed_filtered_r > 0
    ):

        verdict = (
            "CAUTION — EDGE IS POSITIVE BUT MONTE CARLO "
            "SEQUENCE RISK IS ELEVATED"
        )

    else:

        verdict = (
            "FAIL — RETAINED TRADE SET DOES NOT PROVIDE "
            "A POSITIVE HISTORICAL EDGE"
        )

    print()
    print("=" * 140)
    print(
        "MONTE CARLO DECISION"
    )
    print("=" * 140)

    print(
        f"Monte Carlo verdict: {verdict}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  This is a retrospective sequence stress test."
    )

    print(
        "  It does NOT establish future profitability."
    )

    print(
        "  It does NOT replace the genuine post-freeze holdout."
    )

    print(
        "  It does NOT prove causality."
    )

    print(
        "  It does NOT approve production deployment."
    )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY."
    )

    print("=" * 140)

def frozen_early_adverse_future_holdout_test(trades):

    print()
    print("=" * 120)
    print("FROZEN 5B_0.25R GENUINE FUTURE HOLDOUT")
    print("=" * 120)

    HOLDOUT_START_DATE = datetime(
        2026,
        9,
        7
    )

    MIN_HOLDOUT_TRADES = 30
    MIN_HOLDOUT_CALENDAR_DAYS = 90

    RULE_WINDOW = 5
    RULE_THRESHOLD = 0.25

    LEDGER_FILE = (
        "tests/output/"
        "frozen_5b_025_holdout_ledger.csv"
    )

    print()
    print("FROZEN RULE:")
    print(
        f"  Reject when early_adverse_r_bar_{RULE_WINDOW} "
        f">= {RULE_THRESHOLD:.2f}R"
    )
    print("  Missing Bar-5 values are retained.")
    print("  No threshold optimization.")
    print("  No window optimization.")
    print("  No subgroup optimization.")

    print()
    print(
        f"HOLDOUT START: "
        f"{HOLDOUT_START_DATE.date()}"
    )

    print()
    print(
        "IMPORTANT: This is a genuine future holdout only for "
        "trades generated on or after the freeze date."
    )
    print(
        "Historical trades before the freeze date are NOT counted "
        "as holdout evidence."
    )

    def trade_date(trade):

        value = trade.get(
            "entry_date"
        )

        if value is None:
            return None

        try:
            return datetime.fromisoformat(
                str(value)
            )

        except ValueError:
            return None

    def get_trade_r(trade):

        value = to_float(
            trade.get(
                "r_multiple"
            ),
            default=None
        )

        return value

    def get_early_adverse(trade):

        return to_float(
            trade.get(
                f"early_adverse_r_bar_{RULE_WINDOW}"
            ),
            default=None
        )

    def filter_trade(trade):

        early_adverse = get_early_adverse(
            trade
        )

        if early_adverse is None:
            return True

        return (
            early_adverse
            < RULE_THRESHOLD
        )

    def trade_key(trade):

        return (
            str(
                trade.get(
                    "entry_date",
                    ""
                )
            ),
            str(
                trade.get(
                    "symbol",
                    trade.get(
                        "Symbol",
                        ""
                    )
                )
            ),
            str(
                trade.get(
                    "direction",
                    trade.get(
                        "Direction",
                        ""
                    )
                )
            ).upper(),
        )

    def performance(sample):

        values = [
            get_trade_r(trade)
            for trade in sample
        ]

        values = [
            value
            for value in values
            if value is not None
        ]

        if not values:
            return {
                "n": 0,
                "wins": 0,
                "win_rate": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "pf": 0.0,
                "r": 0.0,
            }

        wins = sum(
            1
            for value in values
            if value > 0
        )

        gross_profit = sum(
            max(
                value,
                0.0
            )
            for value in values
        )

        gross_loss = sum(
            abs(
                min(
                    value,
                    0.0
                )
            )
            for value in values
        )

        total_r = sum(
            values
        )

        win_rate = (
            wins
            / len(values)
            * 100
        )

        profit_factor = (
            gross_profit
            / gross_loss
            if gross_loss > 0
            else 0.0
        )

        return {
            "n": len(values),
            "wins": wins,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "pf": profit_factor,
            "r": total_r,
        }

    # --------------------------------------------------------------
    # GENUINE POST-FREEZE TRADES
    # --------------------------------------------------------------

    holdout = [
        trade
        for trade in trades
        if (
            trade_date(trade) is not None
            and trade_date(trade)
            >= HOLDOUT_START_DATE
        )
    ]

    # --------------------------------------------------------------
    # PERSISTENT HOLDOUT LEDGER
    #
    # The ledger preserves the first observed post-freeze trade
    # record so the future holdout cannot silently change because
    # the upstream CSV is regenerated.
    # --------------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            LEDGER_FILE
        ),
        exist_ok=True
    )

    ledger = {}

    if os.path.exists(
        LEDGER_FILE
    ):

        try:

            with open(
                LEDGER_FILE,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:

                reader = csv.DictReader(
                    file
                )

                for row in reader:

                    key = (
                        row.get(
                            "entry_date",
                            ""
                        ),
                        row.get(
                            "symbol",
                            ""
                        ),
                        row.get(
                            "direction",
                            ""
                        ).upper(),
                    )

                    ledger[key] = row

        except (
            OSError,
            csv.Error,
        ):

            ledger = {}

    new_records = 0

    for trade in holdout:

        key = trade_key(
            trade
        )

        if key in ledger:
            continue

        early_adverse = get_early_adverse(
            trade
        )

        final_r = get_trade_r(
            trade
        )

        decision = (
            "RETAIN"
            if early_adverse is None
            or early_adverse < RULE_THRESHOLD
            else "REJECT"
        )

        ledger[key] = {
            "entry_date": key[0],
            "symbol": key[1],
            "direction": key[2],
            "early_adverse_r_bar_5": (
                ""
                if early_adverse is None
                else f"{early_adverse:.6f}"
            ),
            "r_multiple": (
                ""
                if final_r is None
                else f"{final_r:.6f}"
            ),
            "decision": decision,
        }

        new_records += 1

    # --------------------------------------------------------------
    # WRITE LEDGER
    # --------------------------------------------------------------

    ledger_rows = list(
        ledger.values()
    )

    ledger_rows.sort(
        key=lambda row: (
            row.get(
                "entry_date",
                ""
            ),
            row.get(
                "symbol",
                ""
            ),
            row.get(
                "direction",
                ""
            ),
        )
    )

    if ledger_rows:

        with open(
            LEDGER_FILE,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            fieldnames = [
                "entry_date",
                "symbol",
                "direction",
                "early_adverse_r_bar_5",
                "r_multiple",
                "decision",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                ledger_rows
            )

    # --------------------------------------------------------------
    # RECONSTRUCT HOLDOUT FROM PERSISTENT LEDGER
    # --------------------------------------------------------------

    holdout_records = []

    for row in ledger_rows:

        entry_date = row.get(
            "entry_date",
            ""
        )

        try:

            parsed_date = datetime.fromisoformat(
                entry_date
            )

        except ValueError:

            continue

        if parsed_date < HOLDOUT_START_DATE:
            continue

        early_adverse_value = (
            to_float(
                row.get(
                    "early_adverse_r_bar_5"
                ),
                default=None
            )
        )

        final_r_value = (
            to_float(
                row.get(
                    "r_multiple"
                ),
                default=None
            )
        )

        holdout_records.append(
            {
                "entry_date": entry_date,
                "symbol": row.get(
                    "symbol",
                    ""
                ),
                "direction": row.get(
                    "direction",
                    ""
                ),
                "early_adverse_r_bar_5": (
                    early_adverse_value
                ),
                "r_multiple": final_r_value,
            }
        )

    holdout = holdout_records

    holdout_filtered = [
        trade
        for trade in holdout
        if filter_trade(
            trade
        )
    ]

    baseline = performance(
        holdout
    )

    filtered = performance(
        holdout_filtered
    )

    delta_r = (
        filtered["r"]
        - baseline["r"]
    )

    print()
    print("=" * 120)
    print("PERSISTENT HOLDOUT LEDGER")
    print("=" * 120)

    print(
        f"Ledger file: "
        f"{LEDGER_FILE}"
    )

    print(
        f"New post-freeze trades recorded: "
        f"{new_records}"
    )

    print(
        f"Persistent holdout trades: "
        f"{baseline['n']}"
    )

    # --------------------------------------------------------------
    # HOLDOUT SAMPLE
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("HOLDOUT SAMPLE")
    print("=" * 120)

    print(
        f"Holdout trades: "
        f"{baseline['n']}"
    )

    if holdout:

        dates = []

        for trade in holdout:

            parsed = trade_date(
                trade
            )

            if parsed is not None:
                dates.append(
                    parsed
                )

        first_date = min(
            dates
        )

        last_date = max(
            dates
        )

        calendar_days = (
            last_date.date()
            - first_date.date()
        ).days + 1

        print(
            f"First holdout entry: "
            f"{first_date}"
        )

        print(
            f"Last holdout entry:  "
            f"{last_date}"
        )

        print(
            f"Calendar span: "
            f"{calendar_days} days"
        )

    else:

        calendar_days = 0

        print(
            "No genuine post-freeze trades "
            "are currently available."
        )

    # --------------------------------------------------------------
    # BASELINE VS FROZEN RULE
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("BASELINE VS FROZEN 5B_0.25R")
    print("=" * 120)

    print(
        f"{'METRIC':<25}"
        f"{'BASELINE':>15}"
        f"{'FILTERED':>15}"
    )

    print("-" * 60)

    print(
        f"{'Trades':<25}"
        f"{baseline['n']:>15}"
        f"{filtered['n']:>15}"
    )

    print(
        f"{'Win Rate':<25}"
        f"{baseline['win_rate']:>14.2f}%"
        f"{filtered['win_rate']:>14.2f}%"
    )

    print(
        f"{'Profit Factor':<25}"
        f"{baseline['pf']:>15.2f}"
        f"{filtered['pf']:>15.2f}"
    )

    print(
        f"{'Total R':<25}"
        f"{baseline['r']:>15.2f}"
        f"{filtered['r']:>15.2f}"
    )

    print(
        f"{'Delta R':<25}"
        f"{'N/A':>15}"
        f"{delta_r:>15.2f}"
    )

    # --------------------------------------------------------------
    # HOLDOUT READINESS
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("HOLDOUT READINESS")
    print("=" * 120)

    trade_count_ready = (
        baseline["n"]
        >= MIN_HOLDOUT_TRADES
    )

    calendar_ready = (
        calendar_days
        >= MIN_HOLDOUT_CALENDAR_DAYS
    )

    sufficient_trades = (
        trade_count_ready
        and calendar_ready
    )

    print(
        f"Minimum required trades: "
        f"{MIN_HOLDOUT_TRADES}"
    )

    print(
        f"Current holdout trades: "
        f"{baseline['n']}"
    )

    print(
        f"Minimum required calendar span: "
        f"{MIN_HOLDOUT_CALENDAR_DAYS} days"
    )

    print(
        f"Current holdout calendar span: "
        f"{calendar_days} days"
    )

    print(
        "Trade-count requirement: "
        f"{'PASS' if trade_count_ready else 'NOT READY'}"
    )

    print(
        "Calendar-span requirement: "
        f"{'PASS' if calendar_ready else 'NOT READY'}"
    )

    print(
        "Overall holdout readiness: "
        f"{'READY' if sufficient_trades else 'NOT READY'}"
    )

    # --------------------------------------------------------------
    # EVALUATION ONLY AFTER BOTH GATES PASS
    # --------------------------------------------------------------

    if sufficient_trades:

        print()
        print(
            "PRELIMINARY HOLDOUT EVALUATION"
        )

        positive_delta = (
            delta_r > 0
        )

        filtered_positive = (
            filtered["r"] > 0
        )

        filtered_pf_valid = (
            filtered["pf"] >= 1.0
        )

        print(
            f"Positive Delta R: "
            f"{'PASS' if positive_delta else 'FAIL'}"
        )

        print(
            f"Filtered Total R > 0: "
            f"{'PASS' if filtered_positive else 'FAIL'}"
        )

        print(
            f"Filtered PF >= 1.00: "
            f"{'PASS' if filtered_pf_valid else 'FAIL'}"
        )

        holdout_pass = (
            positive_delta
            and filtered_positive
            and filtered_pf_valid
        )

        print()

        if holdout_pass:

            print(
                "HOLDOUT RESULT: "
                "PRELIMINARY PASS"
            )

            print(
                "This does NOT authorize production deployment."
            )

        else:

            print(
                "HOLDOUT RESULT: "
                "FAIL / NOT ROBUST"
            )

            print(
                "Do NOT modify the rule to rescue the result."
            )

    else:

        print()
        print(
            "HOLDOUT RESULT: NOT READY"
        )

        print(
            "Do NOT evaluate the strategy using the "
            "current holdout sample."
        )

        print(
            "Continue generating genuinely unseen trades "
            "under the frozen 5B_0.25R rule."
        )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY"
    )

def main():

    trades = load_trades()

    print("=" * 110)
    print("FAST TRADESENSE RESEARCH")
    print("=" * 110)

    print()
    print(
        f"Loaded trades: {len(trades)}"
    )

    print()
    print("DATASET RANGE")

    dates = [
        trade.get("entry_date", "")
        for trade in trades
        if trade.get("entry_date")
    ]

    if dates:

        print(
            f"First entry: {min(dates)}"
        )

        print(
            f"Last entry:  {max(dates)}"
        )

    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------

    print()
    print("=" * 110)
    print("BASELINE PERFORMANCE")
    print("=" * 110)

    print(
        performance_label(
            trades
        )
    )

    # ------------------------------------------------------------------
    # SHORT REVERSAL EXCLUSION
    # ------------------------------------------------------------------

    setup_exclusion_test(
        trades,
        "SHORT_REVERSAL",
    )

    # ------------------------------------------------------------------
    # LONG CONTINUATION EXCLUSION
    # ------------------------------------------------------------------

    setup_exclusion_test(
        trades,
        "LONG_CONTINUATION",
    )

    # ------------------------------------------------------------------
    # SHORT CONTINUATION EXCLUSION
    # ------------------------------------------------------------------

    setup_exclusion_test(
        trades,
        "SHORT_CONTINUATION",
    )

    # ------------------------------------------------------------------
    # LONG REVERSAL EXCLUSION
    # ------------------------------------------------------------------

    setup_exclusion_test(
        trades,
        "LONG_REVERSAL",
    )

    # ------------------------------------------------------------------
    # SETUP × TREND EXCLUSION MATRIX
    # ------------------------------------------------------------------

    setup_trend_exclusion_matrix(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND COMBINATION COUNTERFACTUAL
    # ------------------------------------------------------------------

    setup_trend_combination_counterfactual(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN HYPOTHESIS HOLDOUT TEST
    # ------------------------------------------------------------------

    frozen_hypothesis_holdout_test(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × PRICE POSITION
    # ------------------------------------------------------------------

    print()
    print("=" * 110)
    print("SETUP × TREND × PRICE POSITION")
    print("=" * 110)

    three_way_groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        price_position = trade.get(
            "entry_price_position",
            "UNKNOWN"
        )

        key = (
            setup,
            trend,
            price_position
        )

        three_way_groups[key].append(
            trade
        )

    for (
        setup,
        trend,
        price_position
    ), group_trades in sorted(
        three_way_groups.items()
    ):

        print(
            f"{setup:<25}"
            f"{trend:<12}"
            f"{price_position:<15}"
            f"{performance_label(group_trades)}"
        )

    # ------------------------------------------------------------------
    # SETUP × TREND × LIQUIDITY × PRICE POSITION
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("SETUP × TREND × LIQUIDITY × PRICE POSITION")
    print("=" * 120)

    four_way_groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        liquidity = trade.get(
            "Diagnostic_Liquidity",
            "UNKNOWN"
        )

        price_position = trade.get(
            "entry_price_position",
            "UNKNOWN"
        )

        key = (
            setup,
            trend,
            liquidity,
            price_position
        )

        four_way_groups[key].append(
            trade
        )

    for (
        setup,
        trend,
        liquidity,
        price_position
    ), group_trades in sorted(
        four_way_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print(
            f"{setup:<25}"
            f"{trend:<12}"
            f"{liquidity:<12}"
            f"{price_position:<15}"
            f"{performance_label(group_trades)}"
        )

    # ------------------------------------------------------------------
    # STOCK × SETUP × TREND × PRICE POSITION
    # ------------------------------------------------------------------

    print()
    print("=" * 130)
    print("STOCK × SETUP × TREND × PRICE POSITION")
    print("=" * 130)

    stock_four_way_groups = defaultdict(list)

    for trade in trades:

        stock = trade.get(
            "_symbol",
            "UNKNOWN"
        )

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        price_position = trade.get(
            "entry_price_position",
            "UNKNOWN"
        )

        key = (
            stock,
            setup,
            trend,
            price_position
        )

        stock_four_way_groups[key].append(
            trade
        )

    print()
    print(
        f"{'STOCK':<16}"
        f"{'SETUP':<25}"
        f"{'TREND':<12}"
        f"{'PRICE POSITION':<15}"
        f"PERFORMANCE"
    )

    print("-" * 130)

    for (
        stock,
        setup,
        trend,
        price_position
    ), group_trades in sorted(
        stock_four_way_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print(
            f"{str(stock):<16}"
            f"{str(setup):<25}"
            f"{str(trend):<12}"
            f"{str(price_position):<15}"
            f"{performance_label(group_trades)}"
        )

    # ------------------------------------------------------------------
    # SETUP × TREND × MFE × MAE
    # ------------------------------------------------------------------

    print()
    print("=" * 130)
    print("SETUP × TREND × MFE × MAE")
    print("=" * 130)

    mfe_mae_groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        max_favorable_r = float(
            trade.get(
                "max_favorable_r",
                0
            )
        )

        max_adverse_r = float(
            trade.get(
                "max_adverse_r",
                0
            )
        )

        if max_favorable_r < 0.5:
            mfe_bucket = "<0.5R"
        elif max_favorable_r < 1.0:
            mfe_bucket = "0.5R-1R"
        elif max_favorable_r < 1.5:
            mfe_bucket = "1R-1.5R"
        else:
            mfe_bucket = ">=1.5R"

        if max_adverse_r < 0.5:
            mae_bucket = "<0.5R"
        elif max_adverse_r < 1.0:
            mae_bucket = "0.5R-1R"
        else:
            mae_bucket = ">=1R"

        key = (
            setup,
            trend,
            mfe_bucket,
            mae_bucket
        )

        mfe_mae_groups[key].append(
            trade
        )

    print()
    print(
        f"{'SETUP':<25}"
        f"{'TREND':<12}"
        f"{'MFE':<12}"
        f"{'MAE':<12}"
        f"PERFORMANCE"
    )

    print("-" * 130)

    for (
        setup,
        trend,
        mfe_bucket,
        mae_bucket
    ), group_trades in sorted(
        mfe_mae_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print(
            f"{str(setup):<25}"
            f"{str(trend):<12}"
            f"{str(mfe_bucket):<12}"
            f"{str(mae_bucket):<12}"
            f"{performance_label(group_trades)}"
        )

    # ------------------------------------------------------------------
    # SETUP × TREND × FAVORABLE FIRST × IMMEDIATE FAILURE
    # ------------------------------------------------------------------

    print()
    print("=" * 130)
    print("SETUP × TREND × FAVORABLE FIRST × IMMEDIATE FAILURE")
    print("=" * 130)

    path_groups = defaultdict(list)

    for trade in trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get(
                "setup",
                "UNKNOWN"
            )
        )

        trend = trade.get(
            "entry_trend",
            "UNKNOWN"
        )

        favorable_first = trade.get(
            "favorable_first"
        )

        immediate_failure = trade.get(
            "immediate_failure",
            False
        )

        favorable_label = (
            str(favorable_first)
            if favorable_first is not None
            else "UNKNOWN"
        )

        immediate_label = (
            "IMMEDIATE_FAILURE"
            if immediate_failure
            else "NO_IMMEDIATE_FAILURE"
        )

        key = (
            setup,
            trend,
            favorable_label,
            immediate_label
        )

        path_groups[key].append(
            trade
        )

    print()
    print(
        f"{'SETUP':<25}"
        f"{'TREND':<12}"
        f"{'FIRST MOVE':<18}"
        f"{'FAILURE':<22}"
        f"PERFORMANCE"
    )

    print("-" * 130)

    for (
        setup,
        trend,
        favorable_label,
        immediate_label
    ), group_trades in sorted(
        path_groups.items(),
        key=lambda item: (
            -len(item[1]),
            item[0]
        )
    ):

        print(
            f"{str(setup):<25}"
            f"{str(trend):<12}"
            f"{str(favorable_label):<18}"
            f"{str(immediate_label):<22}"
            f"{performance_label(group_trades)}"
        )

    # ------------------------------------------------------------------
    # SETUP × TREND × MFE OUTCOME
    # ------------------------------------------------------------------

    setup_trend_mfe_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × MFE BUCKET OUTCOME
    # ------------------------------------------------------------------

    setup_mfe_bucket_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × MFE BUCKET
    # ------------------------------------------------------------------

    setup_trend_mfe_bucket_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # MFE THRESHOLD OUTCOME ANALYSIS
    # ------------------------------------------------------------------

    threshold_outcome_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × CONFIDENCE × LIQUIDITY × BREAK AGE
    # ------------------------------------------------------------------

    setup_context_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × LIQUIDITY × BREAK AGE
    # ------------------------------------------------------------------

    setup_trend_liquidity_age_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # ENTRY CONTEXT × MFE THRESHOLD
    # ------------------------------------------------------------------

    entry_context_mfe_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # ENTRY CONTEXT × FAILURE OUTCOME
    # ------------------------------------------------------------------

    entry_context_failure_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # ENTRY VARIABLE STABILITY ACROSS TIME
    # ------------------------------------------------------------------

    entry_variable_stability_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × MFE THRESHOLD × TIMING / OUTCOME
    # ------------------------------------------------------------------

    setup_trend_mfe_timing_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # TIME-TO-0.5R EXECUTABLE TIMEOUT WALK-FORWARD
    # ------------------------------------------------------------------

    time_to_05r_timeout_counterfactual_walk_forward(
        trades
    )

    # SETUP × TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    setup_trend_timeout_5_walk_forward(trades)

    # ENTRY TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    entry_trend_timeout_5_walk_forward(trades)

    # TREND STRENGTH × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    trend_strength_timeout_5_walk_forward(trades)

    # STOCK × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    stock_timeout_5_walk_forward(trades)

    # STOCK × SETUP × TREND × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    stock_setup_trend_timeout_5_walk_forward(trades)

    # STOCK × SETUP × EXECUTABLE TIMEOUT_5 WALK-FORWARD
    stock_setup_timeout_5_walk_forward(trades)

    # ------------------------------------------------------------------
    # EXECUTABLE +0.5R MANAGEMENT COUNTERFACTUAL
    # ------------------------------------------------------------------

    management_05r_counterfactual_holdout(
        trades
    )

    # FROZEN EXECUTABLE TIMEOUT_5 HOLDOUT
    frozen_timeout_5_holdout_test(trades)

    entry_quality_early_path_diagnostic(
    trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × TIME-TO-0.5R BUCKET
    # ------------------------------------------------------------------

    setup_trend_time_to_05r_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × TIME-TO-0.5R COUNTERFACTUAL FILTER
    # ------------------------------------------------------------------

    setup_trend_time_filter_counterfactual(
        trades
    )

    # ------------------------------------------------------------------ 
    # STOCK × SETUP × TREND × TIME-TO-0.5R 
    # ------------------------------------------------------------------ 
 
    stock_setup_trend_time_diagnostic( 
        trades 
    ) 
 
    # ------------------------------------------------------------------ 
    # EARLY POST-ENTRY FAILURE DIAGNOSTIC 
    # ------------------------------------------------------------------ 
 
    immediate_failure_diagnostic( 
        trades 
    )

    # ------------------------------------------------------------------
    # SETUP × TREND × EARLY ADVERSE-R
    # ------------------------------------------------------------------

    setup_trend_early_adverse_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # EARLY ADVERSE-R SENSITIVITY / COUNTERFACTUAL FILTER
    # ------------------------------------------------------------------

    early_adverse_r_walk_forward_diagnostic(
    trades
    )

    setup_trend_early_adverse_walk_forward_diagnostic(
    trades
    )

    setup_early_adverse_walk_forward_diagnostic(
    trades
    )

    early_adverse_r_sensitivity_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # REGIME-BREAK EARLY-ADVERSE-R DIAGNOSTIC
    # ------------------------------------------------------------------

    regime_break_early_adverse_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # EARLY-ADVERSE TRADE PROFILE — 2022-2025 VS 2026
    # ------------------------------------------------------------------

    early_adverse_trade_profile_regime_comparison(
        trades
    )

    # ------------------------------------------------------------------
    # FULL TRADE-POPULATION REGIME COMPARISON
    # ------------------------------------------------------------------

    full_population_regime_comparison_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # STOCK × REGIME COMPARISON
    # ------------------------------------------------------------------

    stock_regime_comparison_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # STOCK × SETUP × REGIME COMPARISON
    # ------------------------------------------------------------------

    stock_setup_regime_comparison_diagnostic(
        trades
    )

    # STOCK × SETUP × EARLY-ADVERSE-R REGIME COMPARISON

    stock_setup_early_adverse_regime_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # STOCK × SETUP × EARLY-ADVERSE-R × MFE × FINAL-R
    # ------------------------------------------------------------------

    stock_setup_early_adverse_mfe_final_r_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # EARLY-ADVERSE-R × MFE RECOVERY GAP × FINAL-R
    # ------------------------------------------------------------------

    early_adverse_mfe_recovery_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # EARLY-ADVERSE TRIGGER DISCREPANCY AUDIT
    # ------------------------------------------------------------------

    early_adverse_trigger_discrepancy_audit(
        trades
    )

    # ------------------------------------------------------------------
    # FIXED EARLY-ADVERSE-R + ENTRY QUALITY VALIDATION
    # ------------------------------------------------------------------

    fixed_early_adverse_entry_quality_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # FIXED EARLY-ADVERSE-R STOCK × YEAR VALIDATION
    # ------------------------------------------------------------------

    fixed_early_adverse_stock_year_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # FIXED 5B_0.25R COUNTERFACTUAL INTEGRITY AUDIT
    # ------------------------------------------------------------------

    fixed_early_adverse_integrity_audit(
        trades
    )

    # ------------------------------------------------------------------
    # FIXED EARLY-ADVERSE-R TRADE-LEVEL AUDIT
    # ------------------------------------------------------------------

    fixed_early_adverse_trade_level_audit(
        trades
    )

    # ------------------------------------------------------------------
    # FIXED EARLY-ADVERSE-R × STOCK VALIDATION
    # ------------------------------------------------------------------

    fixed_early_adverse_stock_level_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN GLOBAL EARLY-ADVERSE-R ROBUSTNESS
    # ------------------------------------------------------------------

    frozen_global_early_adverse_robustness_diagnostic(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R HOLDOUT LEDGER INTEGRITY AUDIT
    # ------------------------------------------------------------------

    frozen_early_adverse_holdout_ledger_integrity_audit(
        trades
    )

    frozen_early_adverse_monte_carlo_audit(
    trades
    )

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R GENUINE FUTURE HOLDOUT
    # ------------------------------------------------------------------

    frozen_early_adverse_future_holdout_test(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R CHRONOLOGICAL STABILITY
    # ------------------------------------------------------------------

    frozen_early_adverse_chronological_stability_test(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R PROFIT CONCENTRATION / FRAGILITY
    # ------------------------------------------------------------------

    frozen_early_adverse_profit_concentration_audit(
        trades
    )

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R LEAVE-ONE-OUT ROBUSTNESS AUDIT
    # ------------------------------------------------------------------

    frozen_early_adverse_leave_one_out_audit(
        trades
    )

    frozen_early_adverse_dimension_robustness_audit(trades)

    # ------------------------------------------------------------------
    # FROZEN 5B_0.25R RANDOMIZATION / STATISTICAL SIGNIFICANCE
    # ------------------------------------------------------------------

    frozen_early_adverse_randomization_audit(
        trades
    )

    frozen_early_adverse_drawdown_risk_audit(trades)



    # ------------------------------------------------------------------
    # ENTRY QUALITY + EARLY ADVERSE-R COMBINED WALK-FORWARD
    # ------------------------------------------------------------------

    entry_quality_early_adverse_walk_forward_diagnostic(
        trades
    )
 
if __name__ == "__main__": 
    main()