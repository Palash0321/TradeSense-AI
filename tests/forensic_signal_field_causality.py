import csv
import math
import os

import pandas as pd

from app.services.backtest_service import BacktestService


# ======================================================================
# CONFIGURATION
# ======================================================================

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

# Stable research context.
STABLE_START_DATE = "2021-01-01"

# Current rolling production-style context.
ROLLING_START_DATE = "2021-09-13"

FRESH_ONLY_FILE = (
    "tests/output/reconciliation/fresh_only_trades.csv"
)

HISTORICAL_ONLY_FILE = (
    "tests/output/reconciliation/historical_only_trades.csv"
)

OUTPUT_FILE = (
    "tests/output/reconciliation/"
    "signal_field_causality_forensic.csv"
)


# ======================================================================
# PRODUCTION SIGNAL FIELDS
# ======================================================================
#
# Ordered approximately along the production decision chain.
#
# The important field is FIRST_DIFFERING_FIELD.
#
# We deliberately do NOT sort these alphabetically because we want to
# identify the earliest meaningful production-output divergence.
# ======================================================================

SIGNAL_FIELDS = [
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
    "Fib_Distance_382",
    "Fib_Distance_500",
    "Fib_Distance_618",
    "Fib_Extension_Percent",
    "Fib_Extension_State",
    "Fib_Premium_Discount",
]


# ======================================================================
# HELPERS
# ======================================================================

def normalize_symbol(value):
    if value is None:
        return None

    value = str(value).strip()

    if value.endswith(".NS"):
        return value

    return f"{value}.NS"


def normalize_date(value):
    if value is None:
        return None

    try:
        timestamp = pd.Timestamp(value)

        if pd.isna(timestamp):
            return None

        return timestamp.normalize()
    except Exception:
        return None


def normalize_value(value):
    """
    Convert pandas/numpy values into stable Python values for comparison.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    return value


def values_equal(left, right):
    """
    Comparison that handles:
    - None / NaN
    - numeric floating-point differences
    - timestamps
    - strings
    """

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


def display_value(value):
    value = normalize_value(value)

    if value is None:
        return ""

    return str(value)


def load_trade_identities():
    """
    Load the exact divergent identities produced by reconciliation.

    Identity:
        _symbol + entry_date + direction

    We intentionally use BOTH fresh-only and historical-only trades.
    """

    cases = []

    for file_path, category in [
        (FRESH_ONLY_FILE, "FRESH_ONLY"),
        (HISTORICAL_ONLY_FILE, "HISTORICAL_ONLY"),
    ]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required reconciliation file not found: {file_path}"
            )

        with open(
            file_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                symbol = (
                    row.get("_symbol")
                    or row.get("symbol")
                    or row.get("Symbol")
                )

                entry_date = row.get("entry_date")

                direction = (
                    row.get("direction")
                    or row.get("Direction")
                )

                symbol = normalize_symbol(symbol)
                entry_date = normalize_date(entry_date)

                if (
                    symbol is None
                    or entry_date is None
                    or direction is None
                ):
                    continue

                cases.append(
                    {
                        "category": category,
                        "symbol": symbol,
                        "entry_date": entry_date,
                        "direction": str(direction).upper(),
                    }
                )

    # Deduplicate defensively.
    unique = {}

    for case in cases:

        key = (
            case["symbol"],
            case["entry_date"],
            case["direction"],
        )

        unique[key] = case

    cases = list(unique.values())

    cases.sort(
        key=lambda item: (
            item["symbol"],
            item["entry_date"],
            item["direction"],
        )
    )

    return cases


def prepare_signal_frame(signals):
    """
    Normalize the actual production output returned by
    generate_tradesense_signals() into a DataFrame indexed by date.

    No signal logic is recreated here.
    """

    if isinstance(signals, pd.DataFrame):

        frame = signals.copy()

    elif isinstance(signals, dict):

        # Common possible production shapes.
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
            "Unexpected generate_tradesense_signals() "
            f"return type: {type(signals)}"
        )

    if frame.empty:
        return frame

    # --------------------------------------------------------------
    # Normalize date index.
    # --------------------------------------------------------------

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

        dates = pd.to_datetime(
            frame[date_column],
            errors="coerce",
        )

        frame = frame.copy()
        frame.index = dates

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

    # --------------------------------------------------------------
    # Flatten MultiIndex columns if production output ever contains
    # them.
    # --------------------------------------------------------------

    flattened_columns = []

    for column in frame.columns:

        if isinstance(column, tuple):

            parts = [
                str(part)
                for part in column
                if str(part) != "nan"
            ]

            flattened_columns.append(
                "_".join(parts)
            )

        else:

            flattened_columns.append(
                str(column)
            )

    frame.columns = flattened_columns

    return frame


def find_signal_before_entry(
    signal_frame,
    entry_date,
):
    """
    Production enters on the bar after the signal bar.

    Therefore find the latest actual production signal row whose
    date is strictly BEFORE the entry date.
    """

    if signal_frame is None or signal_frame.empty:
        return None, None

    entry_date = normalize_date(entry_date)

    eligible = signal_frame[
        signal_frame.index < entry_date
    ]

    if eligible.empty:
        return None, None

    signal_date = eligible.index[-1]

    return signal_date, eligible.iloc[-1]


def get_field_value(row, field):
    if row is None:
        return None

    if field in row.index:
        return row[field]

    # Case-insensitive fallback.
    field_lower = field.lower()

    for column in row.index:

        if str(column).lower() == field_lower:
            return row[column]

    return None


def classify_first_difference(
    first_difference,
):
    if first_difference is None:
        return "NO_PRODUCTION_FIELD_DIFFERENCE"

    if first_difference in {
        "Structure_Bias",
        "Structure",
        "BOS",
        "CHOCH",
        "Break_Direction",
        "Break_Level",
        "Break_Date",
        "Break_Confirmed",
        "Break_Age",
    }:
        return "STRUCTURE_BREAK"

    if first_difference == "Liquidity_Sweep":
        return "LIQUIDITY"

    if first_difference in {
        "Trend",
        "Trend_Strength",
        "Trend_Slope",
    }:
        return "TREND"

    if first_difference in {
        "Fast_EMA",
        "Slow_EMA",
        "Price_Position",
        "Price_Position_Value",
        "ATR",
    }:
        return "PRICE_INDICATOR"

    if first_difference in {
        "Fib_Retracement",
        "Fib_Zone",
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
        "Fib_Extension_Percent",
        "Fib_Extension_State",
        "Fib_Premium_Discount",
    }:
        return "FIBONACCI"

    if first_difference in {
        "Setup",
        "Setup_Direction",
        "Setup_Confidence",
        "Setup_Reason",
    }:
        return "SETUP"

    if first_difference == "Signal":
        return "SIGNAL"

    return "OTHER"


# ======================================================================
# MAIN FORENSIC
# ======================================================================

def main():

    print()
    print("=" * 120)
    print("PRODUCTION SIGNAL FIELD CAUSALITY FORENSIC")
    print("=" * 120)

    print()
    print(
        "Stable context : "
        f"{STABLE_START_DATE}"
    )

    print(
        "Rolling context: "
        f"{ROLLING_START_DATE}"
    )

    print(
        "Purpose        : Compare ACTUAL production "
        "generate_tradesense_signals() outputs."
    )

    print(
        "Production code: UNCHANGED"
    )

    print(
        "Frozen dataset : UNCHANGED"
    )

    print(
        "Holdout        : UNCHANGED"
    )

    # --------------------------------------------------------------
    # Load divergent identities.
    # --------------------------------------------------------------

    cases = load_trade_identities()

    print()
    print(
        f"Divergent identities loaded: {len(cases)}"
    )

    if not cases:
        print()
        print("No divergent identities found.")
        return

    # --------------------------------------------------------------
    # Cache production signal frames by symbol/context.
    # --------------------------------------------------------------

    signal_cache = {}

    for symbol in SYMBOLS:

        print()
        print("-" * 120)
        print(f"LOADING PRODUCTION SIGNALS: {symbol}")
        print("-" * 120)

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        # ----------------------------------------------------------
        # Stable context.
        # ----------------------------------------------------------

        print(
            f"[{symbol}] Stable data "
            f"from {STABLE_START_DATE}"
        )

        stable_data = service.load_data(
            start_date=STABLE_START_DATE
        )

        stable_signals = (
            service.generate_tradesense_signals(
                df=stable_data
            )
        )

        stable_frame = prepare_signal_frame(
            stable_signals
        )

        print(
            f"[{symbol}] Stable signal rows: "
            f"{len(stable_frame)}"
        )

        # ----------------------------------------------------------
        # Rolling context.
        # ----------------------------------------------------------

        print(
            f"[{symbol}] Rolling data "
            f"from {ROLLING_START_DATE}"
        )

        rolling_data = service.load_data(
            start_date=ROLLING_START_DATE
        )

        rolling_signals = (
            service.generate_tradesense_signals(
                df=rolling_data
            )
        )

        rolling_frame = prepare_signal_frame(
            rolling_signals
        )

        print(
            f"[{symbol}] Rolling signal rows: "
            f"{len(rolling_frame)}"
        )

        signal_cache[symbol] = {
            "stable": stable_frame,
            "rolling": rolling_frame,
        }

    # --------------------------------------------------------------
    # Analyse each divergent identity.
    # --------------------------------------------------------------

    report_rows = []

    aggregate_first_field = {}

    status_counts = {}

    for index, case in enumerate(
        cases,
        start=1,
    ):

        symbol = case["symbol"]
        entry_date = case["entry_date"]
        direction = case["direction"]
        category = case["category"]

        print()
        print("=" * 120)
        print(
            f"CASE {index}/{len(cases)} | "
            f"{category} | "
            f"{symbol} | "
            f"{entry_date.date()} | "
            f"{direction}"
        )
        print("=" * 120)

        if symbol not in signal_cache:

            print(
                f"WARNING: {symbol} is not in configured symbols."
            )

            report_rows.append(
                {
                    "category": category,
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat(),
                    "direction": direction,
                    "stable_signal_date": "",
                    "rolling_signal_date": "",
                    "stable_signal_available": False,
                    "rolling_signal_available": False,
                    "first_differing_field": "SYMBOL_NOT_CONFIGURED",
                    "first_difference_layer": "OTHER",
                    "signal_differs": False,
                    "setup_differs": False,
                    "structure_differs": False,
                    "liquidity_differs": False,
                    "trend_differs": False,
                    "status": "SYMBOL_NOT_CONFIGURED",
                }
            )

            continue

        stable_frame = signal_cache[
            symbol
        ]["stable"]

        rolling_frame = signal_cache[
            symbol
        ]["rolling"]

        # ----------------------------------------------------------
        # Locate the exact signal bar preceding the trade entry.
        # ----------------------------------------------------------

        stable_signal_date, stable_row = (
            find_signal_before_entry(
                stable_frame,
                entry_date,
            )
        )

        rolling_signal_date, rolling_row = (
            find_signal_before_entry(
                rolling_frame,
                entry_date,
            )
        )

        stable_available = (
            stable_row is not None
        )

        rolling_available = (
            rolling_row is not None
        )

        print(
            f"Stable signal date : "
            f"{stable_signal_date.date() 
            if stable_signal_date is not None
            else 'UNAVAILABLE'}"
        )

        print(
            f"Rolling signal date: "
            f"{rolling_signal_date.date() 
            if rolling_signal_date is not None
            else 'UNAVAILABLE'}"
        )

        # ----------------------------------------------------------
        # Rolling context unavailable.
        #
        # These are expected for early trades before 2021-09-13.
        # ----------------------------------------------------------

        if not rolling_available:

            print()
            print(
                "STATUS: ROLLING_CONTEXT_UNAVAILABLE"
            )

            status = (
                "ROLLING_CONTEXT_UNAVAILABLE"
            )

            status_counts[status] = (
                status_counts.get(status, 0) + 1
            )

            report_rows.append(
                {
                    "category": category,
                    "symbol": symbol,
                    "entry_date": entry_date.date().isoformat(),
                    "direction": direction,

                    "stable_signal_date": (
                        stable_signal_date.date().isoformat()
                        if stable_signal_date is not None
                        else ""
                    ),

                    "rolling_signal_date": "",

                    "stable_signal_available": stable_available,
                    "rolling_signal_available": False,

                    "first_differing_field":
                        "ROLLING_CONTEXT_UNAVAILABLE",

                    "first_difference_layer":
                        "CONTEXT",

                    "signal_differs": "",
                    "setup_differs": "",
                    "structure_differs": "",
                    "liquidity_differs": "",
                    "trend_differs": "",

                    "status": status,
                }
            )

            continue

        # ----------------------------------------------------------
        # Compare actual production fields.
        # ----------------------------------------------------------

        differing_fields = []

        for field in SIGNAL_FIELDS:

            stable_value = get_field_value(
                stable_row,
                field,
            )

            rolling_value = get_field_value(
                rolling_row,
                field,
            )

            if not values_equal(
                stable_value,
                rolling_value,
            ):

                differing_fields.append(
                    (
                        field,
                        stable_value,
                        rolling_value,
                    )
                )

        if differing_fields:

            first_field = (
                differing_fields[0][0]
            )

            first_layer = classify_first_difference(
                first_field
            )

        else:

            first_field = None
            first_layer = (
                "NO_PRODUCTION_FIELD_DIFFERENCE"
            )

        # ----------------------------------------------------------
        # Specific causality flags.
        # ----------------------------------------------------------

        signal_differs = not values_equal(
            get_field_value(
                stable_row,
                "Signal",
            ),
            get_field_value(
                rolling_row,
                "Signal",
            ),
        )

        setup_differs = not values_equal(
            get_field_value(
                stable_row,
                "Setup",
            ),
            get_field_value(
                rolling_row,
                "Setup",
            ),
        )

        structure_fields = {
            "Structure_Bias",
            "Structure",
            "BOS",
            "CHOCH",
            "Break_Direction",
            "Break_Level",
            "Break_Date",
            "Break_Confirmed",
            "Break_Age",
        }

        structure_differs = any(
            field in structure_fields
            for field, _, _ in differing_fields
        )

        liquidity_differs = any(
            field == "Liquidity_Sweep"
            for field, _, _ in differing_fields
        )

        trend_fields = {
            "Trend",
            "Trend_Strength",
            "Trend_Slope",
        }

        trend_differs = any(
            field in trend_fields
            for field, _, _ in differing_fields
        )

        if not differing_fields:

            status = (
                "PRODUCTION_SIGNAL_FIELDS_MATCH"
            )

        elif signal_differs:

            status = (
                "PRODUCTION_SIGNAL_DIFFERS"
            )

        else:

            status = (
                "METADATA_DIFFERS_SIGNAL_SAME"
            )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        if first_field is not None:

            aggregate_first_field[first_field] = (
                aggregate_first_field.get(
                    first_field,
                    0,
                )
                + 1
            )

        # ----------------------------------------------------------
        # Print exact differences.
        # ----------------------------------------------------------

        print()
        print(
            f"First differing field: "
            f"{first_field or 'NONE'}"
        )

        print(
            f"First difference layer: "
            f"{first_layer}"
        )

        print(
            f"Signal differs        : "
            f"{signal_differs}"
        )

        print(
            f"Setup differs         : "
            f"{setup_differs}"
        )

        print(
            f"Structure differs     : "
            f"{structure_differs}"
        )

        print(
            f"Liquidity differs     : "
            f"{liquidity_differs}"
        )

        print(
            f"Trend differs         : "
            f"{trend_differs}"
        )

        if differing_fields:

            print()
            print(
                "ACTUAL PRODUCTION FIELD DIFFERENCES"
            )
            print("-" * 120)

            for (
                field,
                stable_value,
                rolling_value,
            ) in differing_fields:

                print(
                    f"{field:<30} "
                    f"STABLE={display_value(stable_value):<30} "
                    f"ROLLING={display_value(rolling_value)}"
                )

        else:

            print()
            print(
                "NO PRODUCTION FIELD DIFFERENCES"
            )

        # ----------------------------------------------------------
        # Build complete report row.
        # ----------------------------------------------------------

        report_row = {
            "category": category,
            "symbol": symbol,
            "entry_date": entry_date.date().isoformat(),
            "direction": direction,

            "stable_signal_date": (
                stable_signal_date.date().isoformat()
            ),

            "rolling_signal_date": (
                rolling_signal_date.date().isoformat()
            ),

            "stable_signal_available": True,
            "rolling_signal_available": True,

            "first_differing_field": (
                first_field
                if first_field is not None
                else ""
            ),

            "first_difference_layer": first_layer,

            "signal_differs": signal_differs,
            "setup_differs": setup_differs,
            "structure_differs": structure_differs,
            "liquidity_differs": liquidity_differs,
            "trend_differs": trend_differs,

            "status": status,
        }

        # ----------------------------------------------------------
        # Store every requested production field.
        # ----------------------------------------------------------

        for field in SIGNAL_FIELDS:

            stable_value = get_field_value(
                stable_row,
                field,
            )

            rolling_value = get_field_value(
                rolling_row,
                field,
            )

            report_row[
                f"stable_{field}"
            ] = display_value(
                stable_value
            )

            report_row[
                f"rolling_{field}"
            ] = display_value(
                rolling_value
            )

        report_rows.append(
            report_row
        )

    # ==================================================================
    # SAVE REPORT
    # ==================================================================

    output_dir = os.path.dirname(
        OUTPUT_FILE
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # Ensure all rows have identical columns.
    fieldnames = []

    for row in report_rows:

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

        for row in report_rows:

            writer.writerow(row)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print()
    print("=" * 120)
    print("FORENSIC SUMMARY")
    print("=" * 120)

    print()
    print(
        f"Cases analysed          : "
        f"{len(cases)}"
    )

    print()
    print(
        "STATUS COUNTS"
    )
    print("-" * 120)

    for status in sorted(
        status_counts
    ):

        print(
            f"{status:<45} "
            f"{status_counts[status]}"
        )

    print()
    print(
        "FIRST DIFFERING PRODUCTION FIELD"
    )
    print("-" * 120)

    if aggregate_first_field:

        ordered = sorted(
            aggregate_first_field.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        for field, count in ordered:

            print(
                f"{field:<35} "
                f"{count}"
            )

    else:

        print(
            "No comparable production "
            "field differences."
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
    print("FORENSIC COMPLETE")
    print("=" * 120)

    print()
    print(
        "IMPORTANT: This diagnostic did not modify "
        "production code, frozen historical trades, "
        "candidate trades, or holdout trades."
    )


if __name__ == "__main__":
    main()