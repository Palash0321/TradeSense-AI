import csv
import math
import os

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure


# ======================================================================
# CONFIGURATION
# ======================================================================

INITIAL_CAPITAL = 100000

STABLE_START_DATE = "2021-01-01"
ROLLING_START_DATE = "2021-09-13"

OUTPUT_FILE = (
    "tests/output/reconciliation/"
    "signal_root_cause_forensic.csv"
)

# These are the ONLY four cases where the previous production-field
# forensic found an actual Signal difference while both contexts had
# a usable signal row.
TARGET_CASES = [
    {
        "symbol": "RELIANCE.NS",
        "entry_date": "2021-09-24",
        "direction": "LONG",
    },
    {
        "symbol": "SBIN.NS",
        "entry_date": "2021-09-16",
        "direction": "LONG",
    },
    {
        "symbol": "TCS.NS",
        "entry_date": "2021-09-16",
        "direction": "LONG",
    },
    {
        "symbol": "INFY.NS",
        "entry_date": "2021-10-04",
        "direction": "SHORT",
    },
]


# ======================================================================
# FIELDS TO TRACE
# ======================================================================

PRODUCTION_FIELDS = [
    "Signal",
    "Setup",
    "Setup_Direction",
    "Setup_Confidence",
    "Setup_Reason",

    "Structure_Bias",
    "Structure",
    "BOS",
    "CHOCH",
    "Break_Direction",
    "Break_Level",
    "Break_Date",
    "Break_Confirmed",
    "Break_Age",

    "Liquidity_Sweep",

    "Trend",
    "Trend_Strength",
    "Trend_Slope",

    "Fast_EMA",
    "Slow_EMA",

    "Price_Position",
    "Price_Position_Value",

    "ATR",

    "Fib_Retracement",
    "Fib_Zone",
    "Fib_Extension_Percent",
    "Fib_Extension_State",
    "Fib_Premium_Discount",
]


# ======================================================================
# GENERIC HELPERS
# ======================================================================

def normalize_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def values_equal(left, right):
    left = normalize_value(left)
    right = normalize_value(right)

    if left is None and right is None:
        return True

    if left is None or right is None:
        return False

    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        try:
            return math.isclose(
                float(left),
                float(right),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        except Exception:
            pass

    return str(left) == str(right)


def printable(value):
    value = normalize_value(value)

    if value is None:
        return "None"

    return str(value)


def normalize_frame(frame):
    """
    Normalize an OHLCV DataFrame without changing its values.
    """

    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):

        result.columns = [
            "_".join(
                str(part)
                for part in column
                if str(part) != "nan"
            )
            for column in result.columns
        ]

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[
        ~result.index.isna()
    ].copy()

    result.index = result.index.normalize()

    result = result[
        ~result.index.duplicated(
            keep="last"
        )
    ].sort_index()

    return result


def get_row_field(row, field):
    if row is None:
        return None

    if field in row.index:
        return row[field]

    wanted = field.lower()

    for column in row.index:

        if str(column).lower() == wanted:
            return row[column]

    return None


# ======================================================================
# PRODUCTION SIGNAL NORMALIZATION
# ======================================================================

def prepare_signal_frame(signals):
    """
    Convert the actual generate_tradesense_signals() result into a
    date-indexed DataFrame.

    IMPORTANT:
    No signal-generation logic is recreated here.
    """

    if isinstance(signals, pd.DataFrame):

        frame = signals.copy()

    elif isinstance(signals, dict):

        if isinstance(
            signals.get("signals"),
            pd.DataFrame,
        ):
            frame = signals["signals"].copy()

        elif isinstance(
            signals.get("data"),
            pd.DataFrame,
        ):
            frame = signals["data"].copy()

        else:
            frame = pd.DataFrame(signals)

    elif isinstance(signals, list):

        frame = pd.DataFrame(signals)

    else:

        raise TypeError(
            "Unexpected production signal return type: "
            f"{type(signals)}"
        )

    if frame.empty:
        return frame

    date_column = None

    for candidate in [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Signal_Date",
    ]:

        if candidate in frame.columns:
            date_column = candidate
            break

    if date_column is not None:

        frame.index = pd.to_datetime(
            frame[date_column],
            errors="coerce",
        )

    else:

        frame.index = pd.to_datetime(
            frame.index,
            errors="coerce",
        )

    frame = frame[
        ~frame.index.isna()
    ].copy()

    frame.index = frame.index.normalize()

    frame = frame[
        ~frame.index.duplicated(
            keep="last"
        )
    ].sort_index()

    return frame


def find_signal_row(
    signal_frame,
    signal_date,
):
    signal_date = pd.Timestamp(
        signal_date
    ).normalize()

    if signal_date not in signal_frame.index:
        return None

    rows = signal_frame.loc[
        signal_frame.index == signal_date
    ]

    if rows.empty:
        return None

    return rows.iloc[-1]


# ======================================================================
# INTERNAL MARKET-STRUCTURE SNAPSHOT
# ======================================================================

def structure_snapshot(history):
    """
    Call the ACTUAL production market-structure function.

    Production calculate_market_structure() returns a dictionary.
    Swing highs/lows are dictionaries containing:
        {"index": Timestamp(...), "price": float}

    This function converts those swing objects into comparable
    date lists while preserving the production structure output.

    This function does NOT modify production code.
    """

    history = normalize_frame(history)

    try:

        result = calculate_market_structure(
            history
        )

    except Exception as exc:

        return {
            "error": (
                f"{type(exc).__name__}: {exc}"
            )
        }

    snapshot = {}

    for key in [
        "bias",
        "structure",
        "last_swing_high",
        "last_swing_low",
        "bos",
        "choch",
        "break_direction",
        "break_level",
        "break_date",
        "break_confirmed",
        "break_age",
        "strength",
    ]:

        value = result.get(key)

        if isinstance(value, pd.Timestamp):

            value = value.date().isoformat()

        snapshot[key] = normalize_value(
            value
        )

    swing_highs = result.get(
        "swing_highs",
        [],
    )

    swing_lows = result.get(
        "swing_lows",
        [],
    )

    snapshot["swing_high_dates"] = [
        pd.Timestamp(
            item["index"]
        ).normalize()
        if isinstance(item, dict)
        else (
            pd.Timestamp(item[0]).normalize()
            if isinstance(item, (tuple, list))
            else pd.Timestamp(item).normalize()
        )
        for item in swing_highs
    ]

    snapshot["swing_low_dates"] = [
        pd.Timestamp(
            item["index"]
        ).normalize()
        if isinstance(item, dict)
        else (
            pd.Timestamp(item[0]).normalize()
            if isinstance(item, (tuple, list))
            else pd.Timestamp(item).normalize()
        )
        for item in swing_lows
    ]

    return snapshot


def comparable_snapshot_fields():
    return [
        "bias",
        "structure",
        "last_swing_high",
        "last_swing_low",
        "bos",
        "choch",
        "break_direction",
        "break_level",
        "break_date",
        "break_confirmed",
        "break_age",
        "strength",
    ]


# ======================================================================
# FIND FIRST SWING DIFFERENCE
# ======================================================================

def normalize_swing_list(swings):
    """
    Production calculate_market_structure() returns swings as:

        {
            "index": Timestamp(...),
            "price": float
        }

    Normalize them into:

        (Timestamp, price)

    so stable and rolling production swing sequences
    can be compared safely.
    """

    normalized = []

    for swing in swings or []:

        if isinstance(swing, dict):

            swing_date = pd.Timestamp(
                swing["index"]
            ).normalize()

            swing_price = float(
                swing["price"]
            )

            normalized.append(
                (
                    swing_date,
                    swing_price
                )
            )

        elif isinstance(swing, (tuple, list)):

            if len(swing) >= 2:

                normalized.append(
                    (
                        pd.Timestamp(
                            swing[0]
                        ).normalize(),
                        float(swing[1])
                    )
                )

            elif len(swing) == 1:

                normalized.append(
                    (
                        pd.Timestamp(
                            swing[0]
                        ).normalize(),
                        None
                    )
                )

        else:

            normalized.append(
                (
                    pd.Timestamp(
                        swing
                    ).normalize(),
                    None
                )
            )

    return normalized


def first_list_difference(
    stable_list,
    rolling_list
):
    """
    Compare swing dates and prices after the rolling-context
    boundary.

    IMPORTANT:
    The stable context has additional historical swings before
    2021-09-13. Those swings are intentionally ignored here.

    We only want to know whether the two production contexts
    disagree about swings inside the overlapping period.
    """

    stable_by_date = {
        date: price
        for date, price in stable_list
    }

    rolling_by_date = {
        date: price
        for date, price in rolling_list
    }

    all_dates = sorted(
        set(stable_by_date.keys())
        | set(rolling_by_date.keys())
    )

    for date in all_dates:

        stable_exists = (
            date in stable_by_date
        )

        rolling_exists = (
            date in rolling_by_date
        )

        if stable_exists != rolling_exists:

            return {
                "type": "SWING_DATE_DIFFERENCE",
                "date": date,
                "stable": (
                    stable_by_date.get(date)
                    if stable_exists
                    else None
                ),
                "rolling": (
                    rolling_by_date.get(date)
                    if rolling_exists
                    else None
                ),
            }

        stable_price = stable_by_date.get(
            date
        )

        rolling_price = rolling_by_date.get(
            date
        )

        if (
            stable_price is not None
            and rolling_price is not None
            and not math.isclose(
                float(stable_price),
                float(rolling_price),
                rel_tol=1e-9,
                abs_tol=1e-6,
            )
        ):

            return {
                "type": "SWING_PRICE_DIFFERENCE",
                "date": date,
                "stable": stable_price,
                "rolling": rolling_price,
            }

    return None

def filter_swings_to_window(
    swings,
    start_date,
    end_date
):
    """
    Keep only production swings whose dates fall inside
    the requested overlapping comparison window.
    """

    start_date = pd.Timestamp(
        start_date
    ).normalize()

    end_date = pd.Timestamp(
        end_date
    ).normalize()

    return [
        swing
        for swing in swings
        if (
            start_date
            <= swing[0]
            <= end_date
        )
    ]


def find_first_swing_difference(
    stable_df,
    rolling_df,
    signal_date,
):
    """
    Find the first GENUINE production swing divergence
    inside the overlapping stable-vs-rolling window.

    Stable context:
        2021-01-01 onward

    Rolling context:
        2021-09-13 onward

    IMPORTANT:

    We deliberately ignore stable swings before
    ROLLING_START_DATE.

    Otherwise the forensic would always report the
    rolling-context boundary itself as the first difference.

    The function walks forward one trading bar at a time
    and compares only swings whose dates are inside the
    common window.
    """

    signal_date = pd.Timestamp(
        signal_date
    ).normalize()

    rolling_start_date = pd.Timestamp(
        ROLLING_START_DATE
    ).normalize()

    common_dates = sorted(
        set(stable_df.index)
        & set(rolling_df.index)
    )

    common_dates = [
        date
        for date in common_dates
        if (
            rolling_start_date
            <= date
            <= signal_date
        )
    ]

    for date in common_dates:

        stable_history = stable_df.loc[
            :date
        ]

        rolling_history = rolling_df.loc[
            :date
        ]

        stable_result = (
            calculate_market_structure(
                stable_history
            )
        )

        rolling_result = (
            calculate_market_structure(
                rolling_history
            )
        )

        stable_highs = normalize_swing_list(
            stable_result.get(
                "swing_highs",
                []
            )
        )

        rolling_highs = normalize_swing_list(
            rolling_result.get(
                "swing_highs",
                []
            )
        )

        stable_lows = normalize_swing_list(
            stable_result.get(
                "swing_lows",
                []
            )
        )

        rolling_lows = normalize_swing_list(
            rolling_result.get(
                "swing_lows",
                []
            )
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        # Ignore stable swings that existed before the rolling
        # context began.
        # ----------------------------------------------------------

        stable_highs = filter_swings_to_window(
            stable_highs,
            rolling_start_date,
            date
        )

        rolling_highs = filter_swings_to_window(
            rolling_highs,
            rolling_start_date,
            date
        )

        stable_lows = filter_swings_to_window(
            stable_lows,
            rolling_start_date,
            date
        )

        rolling_lows = filter_swings_to_window(
            rolling_lows,
            rolling_start_date,
            date
        )

        # ----------------------------------------------------------
        # Compare HIGH swings.
        # ----------------------------------------------------------

        high_difference = (
            first_list_difference(
                stable_highs,
                rolling_highs
            )
        )

        if high_difference is not None:

            return {
                "date": pd.Timestamp(
                    date
                ),

                "swing_type": "HIGH",

                "difference": (
                    high_difference
                ),

                "stable_swing_high_count": (
                    len(stable_highs)
                ),

                "rolling_swing_high_count": (
                    len(rolling_highs)
                ),

                "stable_swing_low_count": (
                    len(stable_lows)
                ),

                "rolling_swing_low_count": (
                    len(rolling_lows)
                ),
            }

        # ----------------------------------------------------------
        # Compare LOW swings.
        # ----------------------------------------------------------

        low_difference = (
            first_list_difference(
                stable_lows,
                rolling_lows
            )
        )

        if low_difference is not None:

            return {
                "date": pd.Timestamp(
                    date
                ),

                "swing_type": "LOW",

                "difference": (
                    low_difference
                ),

                "stable_swing_high_count": (
                    len(stable_highs)
                ),

                "rolling_swing_high_count": (
                    len(rolling_highs)
                ),

                "stable_swing_low_count": (
                    len(stable_lows)
                ),

                "rolling_swing_low_count": (
                    len(rolling_lows)
                ),
            }

    return None


# ======================================================================
# FIND FIRST STRUCTURE SUMMARY DIFFERENCE
# ======================================================================

def find_first_structure_difference(
    stable_history,
    rolling_history,
):
    """
    Compare actual calculate_market_structure() summary state at every
    overlapping bar.

    This identifies the first date where the structural output itself
    differs, even if swing lists happen to be identical.
    """

    common_dates = sorted(
        set(stable_history.index)
        & set(rolling_history.index)
    )

    fields = comparable_snapshot_fields()

    previous_difference = False

    for date in common_dates:

        stable_snapshot = structure_snapshot(
            stable_history.loc[:date]
        )

        rolling_snapshot = structure_snapshot(
            rolling_history.loc[:date]
        )

        if (
            "error" in stable_snapshot
            or "error" in rolling_snapshot
        ):
            continue

        differences = []

        for field in fields:

            stable_value = stable_snapshot.get(
                field
            )

            rolling_value = rolling_snapshot.get(
                field
            )

            if not values_equal(
                stable_value,
                rolling_value,
            ):

                differences.append(
                    field
                )

        current_difference = bool(
            differences
        )

        if (
            current_difference
            and not previous_difference
        ):

            return {
                "date": date,
                "fields": differences,
                "stable_snapshot": stable_snapshot,
                "rolling_snapshot": rolling_snapshot,
            }

        previous_difference = (
            current_difference
        )

    return None


# ======================================================================
# PRODUCTION SIGNAL COMPARISON
# ======================================================================

def compare_production_signal(
    stable_row,
    rolling_row,
):
    differences = []

    for field in PRODUCTION_FIELDS:

        stable_value = get_row_field(
            stable_row,
            field,
        )

        rolling_value = get_row_field(
            rolling_row,
            field,
        )

        if not values_equal(
            stable_value,
            rolling_value,
        ):

            differences.append(
                {
                    "field": field,
                    "stable": stable_value,
                    "rolling": rolling_value,
                }
            )

    return differences


# ======================================================================
# CASE ANALYSIS
# ======================================================================

def analyse_case(case):

    symbol = case["symbol"]

    entry_date = pd.Timestamp(
        case["entry_date"]
    ).normalize()

    direction = case["direction"]

    print()
    print("=" * 120)
    print(
        f"ROOT-CAUSE CASE | "
        f"{symbol} | "
        f"{entry_date.date()} | "
        f"{direction}"
    )
    print("=" * 120)

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=INITIAL_CAPITAL,
    )

    # --------------------------------------------------------------
    # Load both contexts.
    # --------------------------------------------------------------

    stable_data = normalize_frame(
        service.load_data(
            start_date=STABLE_START_DATE
        )
    )

    rolling_data = normalize_frame(
        service.load_data(
            start_date=ROLLING_START_DATE
        )
    )

    signal_date = entry_date - pd.Timedelta(
        days=1
    )

    # The actual signal date must be the preceding trading bar.
    stable_signal_dates = stable_data.index[
        stable_data.index < entry_date
    ]

    rolling_signal_dates = rolling_data.index[
        rolling_data.index < entry_date
    ]

    if len(stable_signal_dates) == 0:
        raise RuntimeError(
            f"No stable signal bar before {entry_date.date()}"
        )

    if len(rolling_signal_dates) == 0:
        raise RuntimeError(
            f"No rolling signal bar before {entry_date.date()}"
        )

    stable_signal_date = stable_signal_dates[-1]
    rolling_signal_date = rolling_signal_dates[-1]

    # --------------------------------------------------------------
    # Generate ACTUAL production signals.
    # --------------------------------------------------------------

    stable_signals = (
        service.generate_tradesense_signals(
            df=stable_data
        )
    )

    rolling_signals = (
        service.generate_tradesense_signals(
            df=rolling_data
        )
    )

    stable_frame = prepare_signal_frame(
        stable_signals
    )

    rolling_frame = prepare_signal_frame(
        rolling_signals
    )

    stable_row = find_signal_row(
        stable_frame,
        stable_signal_date,
    )

    rolling_row = find_signal_row(
        rolling_frame,
        rolling_signal_date,
    )

    if stable_row is None:
        raise RuntimeError(
            f"Stable production signal missing "
            f"for {stable_signal_date.date()}"
        )

    if rolling_row is None:
        raise RuntimeError(
            f"Rolling production signal missing "
            f"for {rolling_signal_date.date()}"
        )

    # --------------------------------------------------------------
    # Actual production field differences.
    # --------------------------------------------------------------

    production_differences = (
        compare_production_signal(
            stable_row,
            rolling_row,
        )
    )

    stable_signal = get_row_field(
        stable_row,
        "Signal",
    )

    rolling_signal = get_row_field(
        rolling_row,
        "Signal",
    )

    print()
    print(
        f"Signal date: "
        f"stable={stable_signal_date.date()} "
        f"rolling={rolling_signal_date.date()}"
    )

    print(
        f"Final Signal: "
        f"stable={printable(stable_signal)} "
        f"rolling={printable(rolling_signal)}"
    )

    # --------------------------------------------------------------
    # FIRST SWING DIFFERENCE.
    # --------------------------------------------------------------

    print()
    print(
        "SEARCHING FOR FIRST PRODUCTION SWING DIFFERENCE..."
    )

    first_swing_difference = (
        find_first_swing_difference(
            stable_data,
            rolling_data,
            stable_signal_date,
        )
    )

    if first_swing_difference is None:

        print(
            "No swing-list divergence found."
        )

    else:

        date = first_swing_difference[
            "date"
        ]

        print(
            f"FIRST SWING DIFFERENCE: "
            f"{date.date()}"
        )

    # --------------------------------------------------------------
    # FIRST STRUCTURE SUMMARY DIFFERENCE.
    # --------------------------------------------------------------

    print()
    print(
        "SEARCHING FOR FIRST MARKET-STRUCTURE OUTPUT DIFFERENCE..."
    )

    first_structure_difference = (
        find_first_structure_difference(
            stable_data,
            rolling_data,
        )
    )

    if first_structure_difference is None:

        print(
            "No structure-summary divergence found."
        )

    else:

        date = first_structure_difference[
            "date"
        ]

        fields = first_structure_difference[
            "fields"
        ]

        print(
            f"FIRST STRUCTURE OUTPUT DIFFERENCE: "
            f"{date.date()}"
        )

        print(
            f"Fields: {', '.join(fields)}"
        )

    # --------------------------------------------------------------
    # Determine which difference occurs first.
    # --------------------------------------------------------------

    root_cause_stage = (
        "UNKNOWN"
    )

    root_cause_date = None

    if first_swing_difference is not None:

        root_cause_stage = "SWING_STATE"

        root_cause_date = (
            first_swing_difference[
                "date"
            ]
        )

    if first_structure_difference is not None:

        structure_date = (
            first_structure_difference[
                "date"
            ]
        )

        if (
            root_cause_date is None
            or structure_date < root_cause_date
        ):

            root_cause_stage = (
                "STRUCTURE_OUTPUT"
            )

            root_cause_date = (
                structure_date
            )

    # --------------------------------------------------------------
    # Print production signal differences.
    # --------------------------------------------------------------

    print()
    print(
        "PRODUCTION SIGNAL DIFFERENCES AT TARGET BAR"
    )
    print("-" * 120)

    if production_differences:

        for difference in production_differences:

            print(
                f"{difference['field']:<30} "
                f"STABLE={printable(difference['stable']):<35} "
                f"ROLLING={printable(difference['rolling'])}"
            )

    else:

        print(
            "NONE"
        )

    # --------------------------------------------------------------
    # Print first structure snapshots.
    # --------------------------------------------------------------

    if first_structure_difference is not None:

        print()
        print(
            "FIRST STRUCTURE-DIVERGENCE SNAPSHOT"
        )
        print("-" * 120)

        stable_snapshot = (
            first_structure_difference[
                "stable_snapshot"
            ]
        )

        rolling_snapshot = (
            first_structure_difference[
                "rolling_snapshot"
            ]
        )

        for field in comparable_snapshot_fields():

            stable_value = stable_snapshot.get(
                field
            )

            rolling_value = rolling_snapshot.get(
                field
            )

            if not values_equal(
                stable_value,
                rolling_value,
            ):

                print(
                    f"{field:<30} "
                    f"STABLE={printable(stable_value):<35} "
                    f"ROLLING={printable(rolling_value)}"
                )

    # --------------------------------------------------------------
    # Return compact result.
    # --------------------------------------------------------------

    return {
        "symbol": symbol,
        "entry_date": entry_date.date().isoformat(),
        "direction": direction,

        "stable_signal_date":
            stable_signal_date.date().isoformat(),

        "rolling_signal_date":
            rolling_signal_date.date().isoformat(),

        "stable_signal":
            printable(stable_signal),

        "rolling_signal":
            printable(rolling_signal),

        "signal_differs":
            not values_equal(
                stable_signal,
                rolling_signal,
            ),

        "first_swing_difference_date":
            (
                first_swing_difference["date"]
                .date()
                .isoformat()
                if first_swing_difference
                is not None
                else ""
            ),

        "first_structure_difference_date":
            (
                first_structure_difference["date"]
                .date()
                .isoformat()
                if first_structure_difference
                is not None
                else ""
            ),

        "first_structure_difference_fields":
            (
                "|".join(
                    first_structure_difference[
                        "fields"
                    ]
                )
                if first_structure_difference
                is not None
                else ""
            ),

        "root_cause_stage":
            root_cause_stage,

        "root_cause_date":
            (
                root_cause_date
                .date()
                .isoformat()
                if root_cause_date is not None
                else ""
            ),

        "production_difference_count":
            len(production_differences),

        "production_difference_fields":
            "|".join(
                item["field"]
                for item in production_differences
            ),
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print()
    print("=" * 120)
    print("FOUR-CASE SIGNAL ROOT-CAUSE FORENSIC")
    print("=" * 120)

    print()
    print(
        f"Stable context : {STABLE_START_DATE}"
    )

    print(
        f"Rolling context: {ROLLING_START_DATE}"
    )

    print()
    print(
        "This diagnostic investigates ONLY the four "
        "comparable cases where the production Signal changed."
    )

    print()
    print(
        "Production code: UNCHANGED"
    )

    print(
        "Frozen dataset : UNCHANGED"
    )

    print(
        "Holdout        : UNCHANGED"
    )

    results = []

    for case in TARGET_CASES:

        try:

            result = analyse_case(
                case
            )

            results.append(
                result
            )

        except Exception as exc:

            print()
            print(
                "CASE FAILED"
            )

            print(
                f"{case['symbol']} "
                f"{case['entry_date']}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "symbol": case["symbol"],
                    "entry_date": case["entry_date"],
                    "direction": case["direction"],
                    "error":
                        f"{type(exc).__name__}: {exc}",
                }
            )

    # ==================================================================
    # SAVE RESULTS
    # ==================================================================

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True,
    )

    fieldnames = []

    for row in results:

        for key in row.keys():

            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:

            writer.writerow(row)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print()
    print("=" * 120)
    print("ROOT-CAUSE SUMMARY")
    print("=" * 120)

    print()

    for result in results:

        print(
            f"{result.get('symbol', ''):<15} "
            f"{result.get('entry_date', ''):<12} "
            f"root={result.get('root_cause_stage', ''):<20} "
            f"date={result.get('root_cause_date', ''):<12} "
            f"signal="
            f"{result.get('stable_signal', '')}"
            f"→"
            f"{result.get('rolling_signal', '')}"
        )

    print()
    print(
        "OUTPUT FILE"
    )
    print("-" * 120)

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 120)
    print("ROOT-CAUSE FORENSIC COMPLETE")
    print("=" * 120)

    print()
    print(
        "IMPORTANT: No production code, frozen historical "
        "data, candidate data, or holdout data was modified."
    )


if __name__ == "__main__":
    main()