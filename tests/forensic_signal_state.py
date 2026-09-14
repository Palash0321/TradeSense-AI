"""
TradeSense-AI
Signal-State Forensic Analysis

Purpose:
    Inspect the actual TradeSense signal / market-structure state
    around the trade divergences between:

        1. Frozen historical dataset
        2. Fresh research candidate dataset

This is DIAGNOSTIC ONLY.

It does NOT:
    - modify the frozen historical dataset
    - modify the research candidate
    - modify strategy logic
    - modify market structure logic
    - modify backtest logic

Research question:
    Are the fresh-only trades caused by legitimate signal states
    available after using a longer historical warm-up period?

    Or do the signal / structure states themselves differ in a
    suspicious way?

Important:
    Signals are generated using a stable historical start date
    of 2021-01-01, matching the warm-up correction already tested.
"""

from pathlib import Path
import csv
import pandas as pd

from app.services.backtest_service import BacktestService


# ==============================================================
# CONFIGURATION
# ==============================================================

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

SIGNAL_START_DATE = "2021-01-01"

HISTORICAL_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

CANDIDATE_FILE = Path(
    "tests/output/tradesense_trades_research_candidate.csv"
)

OUTPUT_DIR = Path(
    "tests/output/reconciliation"
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "signal_state_forensic.csv"
)


# ==============================================================
# HELPERS
# ==============================================================

def load_trade_file(file_path):
    """
    Load trade identities from CSV.

    Identity:
        _symbol + entry_date + direction
    """

    trades = []

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if not row:
                continue

            if (
                row.get("exit_reason", "").strip()
                == "FINAL_LIQUIDATION"
            ):
                continue

            symbol = row.get("_symbol", "").strip()
            entry_date = row.get("entry_date", "").strip()
            direction = row.get("direction", "").strip()

            if not symbol or not entry_date or not direction:
                continue

            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date[:10],
                    "direction": direction,
                    "setup": row.get("setup", "").strip(),
                    "r_multiple": row.get(
                        "r_multiple",
                        ""
                    ),
                }
            )

    return trades


def build_identity_set(trades):
    """Build trade identity set."""

    return {
        (
            trade["symbol"],
            trade["entry_date"],
            trade["direction"],
        )
        for trade in trades
    }


def get_trade_rows_for_symbol(
    trades,
    symbol
):
    """Return trades for one symbol."""

    return [
        trade
        for trade in trades
        if trade["symbol"] == symbol
    ]


def normalize_signal_dataframe(signals):
    """
    Normalize signal dataframe/index into a clean date index.
    """

    df = signals.copy()

    if not isinstance(df.index, pd.DatetimeIndex):

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.set_index("Date")

        elif "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

            df = df.set_index("date")

        else:
            df.index = pd.to_datetime(
                df.index,
                errors="coerce"
            )

    else:
        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

    df = df[~df.index.isna()].copy()

    return df.sort_index()


def get_signal_row(
    signals,
    date
):
    """
    Retrieve the exact signal row for a date.

    If the requested date does not exist, return None.
    """

    target_date = pd.Timestamp(date).normalize()

    matches = signals[
        signals.index.normalize()
        == target_date
    ]

    if matches.empty:
        return None

    return matches.iloc[-1]


def value_from_row(
    row,
    possible_columns
):
    """
    Return the first available column value.
    """

    if row is None:
        return ""

    for column in possible_columns:

        if column in row.index:

            value = row[column]

            if pd.isna(value):
                return ""

            return value

    return ""


def format_value(value):
    """Format values for terminal/report output."""

    if value is None:
        return ""

    if isinstance(value, float):

        if pd.isna(value):
            return ""

        return f"{value:.6f}"

    return str(value)


# ==============================================================
# LOAD TRADE DATA
# ==============================================================

print()
print("=" * 110)
print("TRADESENSE-AI — SIGNAL STATE FORENSIC ANALYSIS")
print("=" * 110)

print()
print(f"Historical file : {HISTORICAL_FILE}")
print(f"Candidate file  : {CANDIDATE_FILE}")
print(
    f"Signal start    : {SIGNAL_START_DATE}"
)

if not HISTORICAL_FILE.exists():
    raise FileNotFoundError(
        f"Historical dataset not found: {HISTORICAL_FILE}"
    )

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Research candidate not found: {CANDIDATE_FILE}"
    )


historical_trades = load_trade_file(
    HISTORICAL_FILE
)

candidate_trades = load_trade_file(
    CANDIDATE_FILE
)

historical_ids = build_identity_set(
    historical_trades
)

candidate_ids = build_identity_set(
    candidate_trades
)

fresh_only_ids = (
    candidate_ids
    - historical_ids
)

historical_only_ids = (
    historical_ids
    - candidate_ids
)


fresh_only_trades = [
    trade
    for trade in candidate_trades
    if (
        trade["symbol"],
        trade["entry_date"],
        trade["direction"],
    )
    in fresh_only_ids
]

historical_only_trades = [
    trade
    for trade in historical_trades
    if (
        trade["symbol"],
        trade["entry_date"],
        trade["direction"],
    )
    in historical_only_ids
]


# ==============================================================
# CREATE TARGET DATE SET
# ==============================================================

# We inspect:
#
#   1. fresh-only trade dates
#   2. historical-only trade dates
#
# The purpose is to see the actual signal state on those dates.

targets_by_symbol = {}

for symbol in SYMBOLS:

    fresh_dates = sorted(
        {
            trade["entry_date"]
            for trade in fresh_only_trades
            if trade["symbol"] == symbol
        }
    )

    historical_dates = sorted(
        {
            trade["entry_date"]
            for trade in historical_only_trades
            if trade["symbol"] == symbol
        }
    )

    targets_by_symbol[symbol] = {
        "fresh_only": fresh_dates,
        "historical_only": historical_dates,
    }


# ==============================================================
# SIGNAL COLUMNS
# ==============================================================

SIGNAL_COLUMNS = [
    "Structure_Bias",
    "Structure",
    "BOS",
    "CHOCH",
    "Break_Direction",
    "Break_Confirmed",
    "Break_Age",
    "Setup",
    "Setup_Direction",
    "Setup_Confidence",
    "Setup_Reason",
    "Trend",
    "Trend_Strength",
    "Fast_EMA",
    "Slow_EMA",
    "Trend_Slope",
    "Fibonacci",
    "Liquidity_Sweep",
    "Open",
    "High",
    "Low",
    "Close",
    "ATR",
]


# ==============================================================
# RUN SYMBOL-BY-SYMBOL SIGNAL ANALYSIS
# ==============================================================

results = []


for symbol in SYMBOLS:

    print()
    print("=" * 110)
    print(f"SYMBOL: {symbol}")
    print("=" * 110)

    fresh_dates = targets_by_symbol[symbol][
        "fresh_only"
    ]

    historical_dates = targets_by_symbol[symbol][
        "historical_only"
    ]

    target_dates = sorted(
        set(fresh_dates)
        | set(historical_dates)
    )

    print()
    print(
        f"Fresh-only target dates      : "
        f"{len(fresh_dates)}"
    )

    print(
        f"Historical-only target dates : "
        f"{len(historical_dates)}"
    )

    if not target_dates:

        print("No divergence dates.")

        continue

    # ----------------------------------------------------------
    # Load stable historical data
    # ----------------------------------------------------------

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=INITIAL_CAPITAL,
    )

    historical_data = service.load_data(
        start_date=SIGNAL_START_DATE
    )

    signals = service.generate_tradesense_signals(
        df=historical_data
    )

    signals = normalize_signal_dataframe(
        signals
    )

    print()
    print(
        f"Signal rows generated: {len(signals)}"
    )

    # ----------------------------------------------------------
    # Inspect every divergence date
    # ----------------------------------------------------------

    for target_date in target_dates:

        row = get_signal_row(
            signals,
            target_date
        )

        fresh_trade = next(
            (
                trade
                for trade in fresh_only_trades
                if (
                    trade["symbol"] == symbol
                    and trade["entry_date"]
                    == target_date
                )
            ),
            None
        )

        historical_trade = next(
            (
                trade
                for trade in historical_only_trades
                if (
                    trade["symbol"] == symbol
                    and trade["entry_date"]
                    == target_date
                )
            ),
            None
        )

        if fresh_trade:

            classification = "FRESH_ONLY"

            trade_direction = (
                fresh_trade["direction"]
            )

            trade_setup = (
                fresh_trade["setup"]
            )

        else:

            classification = "HISTORICAL_ONLY"

            trade_direction = (
                historical_trade["direction"]
            )

            trade_setup = (
                historical_trade["setup"]
            )

        print()
        print("-" * 110)
        print(
            f"{classification} | {symbol} | "
            f"{target_date}"
        )
        print("-" * 110)

        print(
            f"Trade direction : {trade_direction}"
        )

        print(
            f"Trade setup     : {trade_setup}"
        )

        if row is None:

            print(
                "WARNING: No signal row found for "
                f"{target_date}"
            )

            results.append(
                {
                    "symbol": symbol,
                    "date": target_date,
                    "classification": classification,
                    "trade_direction": trade_direction,
                    "trade_setup": trade_setup,
                    "signal_row_found": False,
                }
            )

            continue

        print()
        print("SIGNAL / STRUCTURE STATE")
        print("-" * 110)

        row_data = {}

        for column in SIGNAL_COLUMNS:

            value = value_from_row(
                row,
                [column]
            )

            row_data[column] = value

            print(
                f"{column:22s}: "
                f"{format_value(value)}"
            )

        # ------------------------------------------------------
        # Direction compatibility diagnostic
        # ------------------------------------------------------

        setup_direction = value_from_row(
            row,
            ["Setup_Direction"]
        )

        structure_bias = value_from_row(
            row,
            ["Structure_Bias"]
        )

        setup_value = value_from_row(
            row,
            ["Setup"]
        )

        trend_value = value_from_row(
            row,
            ["Trend"]
        )

        direction_match = ""

        if trade_direction == "LONG":

            direction_match = (
                setup_direction == "LONG"
                or structure_bias in (
                    "BULLISH",
                    "PARTIAL_BULLISH",
                )
            )

        elif trade_direction == "SHORT":

            direction_match = (
                setup_direction == "SHORT"
                or structure_bias in (
                    "BEARISH",
                    "PARTIAL_BEARISH",
                )
            )

        print()
        print("COMPATIBILITY CHECK")
        print("-" * 110)

        print(
            f"Trade direction : {trade_direction}"
        )

        print(
            f"Setup direction  : "
            f"{format_value(setup_direction)}"
        )

        print(
            f"Structure bias   : "
            f"{format_value(structure_bias)}"
        )

        print(
            f"Trend            : "
            f"{format_value(trend_value)}"
        )

        print(
            f"Direction broadly compatible: "
            f"{direction_match}"
        )

        # ------------------------------------------------------
        # Save forensic row
        # ------------------------------------------------------

        result = {
            "symbol": symbol,
            "date": target_date,
            "classification": classification,
            "trade_direction": trade_direction,
            "trade_setup": trade_setup,
            "signal_row_found": True,
        }

        for column in SIGNAL_COLUMNS:

            result[column] = format_value(
                row_data[column]
            )

        result["direction_broadly_compatible"] = (
            direction_match
        )

        results.append(result)


# ==============================================================
# SAVE REPORT
# ==============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


output_fields = [
    "symbol",
    "date",
    "classification",
    "trade_direction",
    "trade_setup",
    "signal_row_found",
    *SIGNAL_COLUMNS,
    "direction_broadly_compatible",
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=output_fields,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(results)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()
print("=" * 110)
print("SIGNAL FORENSIC SUMMARY")
print("=" * 110)

print()

for symbol in SYMBOLS:

    symbol_results = [
        result
        for result in results
        if result["symbol"] == symbol
    ]

    fresh_count = sum(
        result["classification"]
        == "FRESH_ONLY"
        for result in symbol_results
    )

    historical_count = sum(
        result["classification"]
        == "HISTORICAL_ONLY"
        for result in symbol_results
    )

    missing_count = sum(
        not result["signal_row_found"]
        for result in symbol_results
    )

    print(
        f"{symbol:18s} | "
        f"fresh-only={fresh_count:2d} | "
        f"historical-only={historical_count:2d} | "
        f"missing-signal-row={missing_count:2d}"
    )


print()
print("=" * 110)
print("REPORT SAVED")
print("=" * 110)

print(
    f"File: {OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT: This diagnostic does not modify "
    "any dataset or strategy code."
)

print("=" * 110)