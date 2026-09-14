"""
TradeSense-AI
Signal-Bar Forensic Diagnostic

Purpose
-------
Investigate fresh-only vs historical-only trades using the ACTUAL
signal bar that generated the trade.

Engine timing:
    signal generated on bar i
    entry executed at bar i+1 OPEN

Therefore:
    entry_date  = execution bar
    signal_date = immediately preceding trading bar

This script is DIAGNOSTIC ONLY.

It does NOT:
- modify strategy logic
- modify market structure
- modify setup logic
- modify backtest engine
- modify the frozen historical dataset

Outputs:
    tests/output/reconciliation/signal_bar_forensic.csv
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

HISTORICAL_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

CANDIDATE_FILE = Path(
    "tests/output/tradesense_trades_research_candidate.csv"
)

OUTPUT_DIR = Path(
    "tests/output/reconciliation"
)

OUTPUT_FILE = OUTPUT_DIR / "signal_bar_forensic.csv"


# ==============================================================
# HELPERS
# ==============================================================

def normalize_date(value):
    """
    Convert a date-like value into YYYY-MM-DD.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value[:10]


def normalize_symbol(value):
    """
    Normalize symbol representation.
    """
    if value is None:
        return ""

    return str(value).strip().upper()


def trade_identity(trade):
    """
    Identity used by reconciliation.
    """
    return (
        normalize_symbol(trade.get("_symbol")),
        normalize_date(trade.get("entry_date")),
        str(trade.get("direction", "")).strip().upper(),
    )


def safe_value(row, column):
    """
    Safely extract a dataframe value.
    """
    if row is None:
        return None

    if column not in row.index:
        return None

    value = row[column]

    if pd.isna(value):
        return None

    return value


def format_value(value):
    """
    Keep diagnostic CSV readable.
    """
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


# ==============================================================
# LOAD TRADE DATASETS
# ==============================================================

print()
print("=" * 100)
print("SIGNAL-BAR FORENSIC DIAGNOSTIC")
print("=" * 100)

print()
print("Frozen historical dataset:")
print(f"  {HISTORICAL_FILE}")

print()
print("Fresh research candidate:")
print(f"  {CANDIDATE_FILE}")


historical_df = pd.read_csv(
    HISTORICAL_FILE
)

candidate_df = pd.read_csv(
    CANDIDATE_FILE
)


# ==============================================================
# NORMALIZE TRADE IDENTITIES
# ==============================================================

historical_trades = {}

for _, row in historical_df.iterrows():

    trade = row.to_dict()

    key = trade_identity(trade)

    historical_trades[key] = trade


candidate_trades = {}

for _, row in candidate_df.iterrows():

    trade = row.to_dict()

    key = trade_identity(trade)

    candidate_trades[key] = trade


historical_keys = set(historical_trades)
candidate_keys = set(candidate_trades)


fresh_only_keys = sorted(
    candidate_keys - historical_keys,
    key=lambda x: (
        x[0],
        x[1] or "",
        x[2],
    )
)

historical_only_keys = sorted(
    historical_keys - candidate_keys,
    key=lambda x: (
        x[0],
        x[1] or "",
        x[2],
    )
)


# ==============================================================
# DISPLAY RECONCILIATION COUNTS
# ==============================================================

print()
print("=" * 100)
print("TRADE IDENTITY RECONCILIATION")
print("=" * 100)

print(
    f"Historical trades : {len(historical_trades)}"
)

print(
    f"Candidate trades  : {len(candidate_trades)}"
)

print(
    f"Fresh-only trades : {len(fresh_only_keys)}"
)

print(
    f"Historical-only   : {len(historical_only_keys)}"
)

print(
    f"Common trades     : "
    f"{len(historical_keys & candidate_keys)}"
)


# ==============================================================
# IMPORTANT:
# ANALYZE BOTH SIDES
# ==============================================================

cases = []

for key in fresh_only_keys:

    trade = candidate_trades[key]

    cases.append(
        {
            "case_type": "FRESH_ONLY",
            "trade": trade,
        }
    )


for key in historical_only_keys:

    trade = historical_trades[key]

    cases.append(
        {
            "case_type": "HISTORICAL_ONLY",
            "trade": trade,
        }
    )


# ==============================================================
# LOAD STABLE RESEARCH DATA
# ==============================================================

print()
print("=" * 100)
print("LOADING STABLE RESEARCH DATA")
print("=" * 100)

print(
    "Using start_date=2021-01-01 for all symbols."
)

print(
    "This provides a stable warm-up boundary."
)


signal_cache = {}

for symbol in SYMBOLS:

    print()
    print(f"Loading {symbol}...")

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    historical_data = service.load_data(
        start_date="2021-01-01"
    )

    signals = service.generate_tradesense_signals(
        df=historical_data
    )

    if signals is None or signals.empty:

        print(
            f"  WARNING: No signals returned for {symbol}"
        )

        signal_cache[symbol] = None
        continue

    signals = signals.copy()

    signals.index = pd.to_datetime(
        signals.index
    )

    signals = signals.sort_index()

    signal_cache[symbol] = signals

    print(
        f"  Signal rows: {len(signals)}"
    )

    print(
        f"  First date : {signals.index.min().date()}"
    )

    print(
        f"  Last date  : {signals.index.max().date()}"
    )


# ==============================================================
# FIELDS TO INSPECT
# ==============================================================

INSPECT_FIELDS = [
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
    "Trend_Slope",
    "Liquidity_Sweep",
    "Open",
    "High",
    "Low",
    "Close",
    "ATR",
]


# ==============================================================
# FORENSIC ANALYSIS
# ==============================================================

results = []


for case in cases:

    case_type = case["case_type"]
    trade = case["trade"]

    symbol = normalize_symbol(
        trade.get("_symbol")
    )

    entry_date = normalize_date(
        trade.get("entry_date")
    )

    direction = str(
        trade.get("direction", "")
    ).strip().upper()

    setup_recorded = str(
        trade.get("setup", "")
    ).strip()

    setup_direction_recorded = str(
        trade.get("setup_direction", "")
    ).strip()

    signals = signal_cache.get(symbol)

    print()
    print("-" * 100)
    print(
        f"{case_type}: "
        f"{symbol} | "
        f"{entry_date} | "
        f"{direction}"
    )
    print("-" * 100)

    # ----------------------------------------------------------
    # Missing symbol
    # ----------------------------------------------------------

    if signals is None:

        print(
            "  SIGNAL DATA: MISSING"
        )

        results.append(
            {
                "case_type": case_type,
                "symbol": symbol,
                "entry_date": entry_date,
                "signal_date": "",
                "direction": direction,
                "recorded_setup": setup_recorded,
                "recorded_setup_direction":
                    setup_direction_recorded,
                "signal_row_found": False,
                "signal_position": "",
                "entry_row_found": False,
                "notes": "No signal dataframe",
            }
        )

        continue

    # ----------------------------------------------------------
    # Find execution / entry date
    # ----------------------------------------------------------

    entry_timestamp = pd.Timestamp(
        entry_date
    )

    matching_positions = [
        i
        for i, timestamp in enumerate(signals.index)
        if timestamp.normalize() == entry_timestamp.normalize()
    ]

    if not matching_positions:

        print(
            f"  ENTRY DATE {entry_date} NOT FOUND"
        )

        results.append(
            {
                "case_type": case_type,
                "symbol": symbol,
                "entry_date": entry_date,
                "signal_date": "",
                "direction": direction,
                "recorded_setup": setup_recorded,
                "recorded_setup_direction":
                    setup_direction_recorded,
                "signal_row_found": False,
                "signal_position": "",
                "entry_row_found": False,
                "notes": "Entry date not found in signal dataframe",
            }
        )

        continue

    entry_position = matching_positions[0]

    # ----------------------------------------------------------
    # ACTUAL ENGINE TIMING:
    #
    # signal = previous row
    # entry  = current row
    # ----------------------------------------------------------

    if entry_position <= 0:

        print(
            "  SIGNAL BAR: unavailable "
            "(entry is first dataframe row)"
        )

        results.append(
            {
                "case_type": case_type,
                "symbol": symbol,
                "entry_date": entry_date,
                "signal_date": "",
                "direction": direction,
                "recorded_setup": setup_recorded,
                "recorded_setup_direction":
                    setup_direction_recorded,
                "signal_row_found": False,
                "signal_position": entry_position,
                "entry_row_found": True,
                "notes": "No preceding signal bar",
            }
        )

        continue

    signal_position = entry_position - 1

    signal_timestamp = signals.index[
        signal_position
    ]

    entry_timestamp_actual = signals.index[
        entry_position
    ]

    signal_row = signals.iloc[
        signal_position
    ]

    entry_row = signals.iloc[
        entry_position
    ]

    signal_date = (
        signal_timestamp.strftime("%Y-%m-%d")
    )

    actual_entry_date = (
        entry_timestamp_actual.strftime("%Y-%m-%d")
    )

    print()
    print(
        f"  SIGNAL DATE : {signal_date}"
    )

    print(
        f"  ENTRY DATE  : {actual_entry_date}"
    )

    print(
        f"  TRADE DIR   : {direction}"
    )

    print(
        f"  RECORDED SETUP: {setup_recorded}"
    )

    print(
        f"  RECORDED SETUP DIRECTION: "
        f"{setup_direction_recorded}"
    )

    print()
    print("  SIGNAL-BAR STATE:")
    print()

    result = {
        "case_type": case_type,
        "symbol": symbol,
        "entry_date": entry_date,
        "signal_date": signal_date,
        "direction": direction,
        "recorded_setup": setup_recorded,
        "recorded_setup_direction":
            setup_direction_recorded,
        "signal_row_found": True,
        "signal_position": signal_position,
        "entry_row_found": True,
        "notes": "",
    }

    # ----------------------------------------------------------
    # Extract signal-bar fields
    # ----------------------------------------------------------

    for field in INSPECT_FIELDS:

        signal_value = safe_value(
            signal_row,
            field
        )

        entry_value = safe_value(
            entry_row,
            field
        )

        result[
            f"signal_{field}"
        ] = format_value(signal_value)

        result[
            f"entry_{field}"
        ] = format_value(entry_value)

        print(
            f"    {field:<22} "
            f"signal={format_value(signal_value):<25} "
            f"entry={format_value(entry_value)}"
        )

    # ----------------------------------------------------------
    # Timing sanity checks
    # ----------------------------------------------------------

    timing_ok = (
        actual_entry_date != signal_date
        and signal_position == entry_position - 1
    )

    result[
        "timing_alignment"
    ] = (
        "PASS"
        if timing_ok
        else "FAIL"
    )

    # ----------------------------------------------------------
    # Setup consistency
    #
    # Trade record can use different capitalization/string
    # representation, so normalize before comparison.
    # ----------------------------------------------------------

    signal_setup = str(
        safe_value(
            signal_row,
            "Setup"
        ) or ""
    ).strip().upper()

    recorded_setup_normalized = (
        setup_recorded.strip().upper()
    )

    setup_match = (
        signal_setup == recorded_setup_normalized
    )

    result[
        "signal_setup_matches_trade"
    ] = (
        "YES"
        if setup_match
        else "NO"
    )

    # ----------------------------------------------------------
    # Setup direction consistency
    # ----------------------------------------------------------

    signal_setup_direction = str(
        safe_value(
            signal_row,
            "Setup_Direction"
        ) or ""
    ).strip().upper()

    recorded_setup_direction_normalized = (
        setup_direction_recorded.strip().upper()
    )

    setup_direction_match = (
        signal_setup_direction
        == recorded_setup_direction_normalized
    )

    result[
        "signal_setup_direction_matches_trade"
    ] = (
        "YES"
        if setup_direction_match
        else "NO"
    )

    # ----------------------------------------------------------
    # Direction compatibility
    # ----------------------------------------------------------

    direction_compatible = None

    if signal_setup_direction:

        if direction == "LONG":

            direction_compatible = (
                signal_setup_direction == "LONG"
            )

        elif direction == "SHORT":

            direction_compatible = (
                signal_setup_direction == "SHORT"
            )

    result[
        "direction_compatible_with_signal_setup"
    ] = (
        ""
        if direction_compatible is None
        else (
            "YES"
            if direction_compatible
            else "NO"
        )
    )

    # ----------------------------------------------------------
    # Structure / break consistency
    # ----------------------------------------------------------

    signal_break_direction = str(
        safe_value(
            signal_row,
            "Break_Direction"
        ) or ""
    ).strip().upper()

    signal_break_confirmed = str(
        safe_value(
            signal_row,
            "Break_Confirmed"
        ) or ""
    ).strip().upper()

    result[
        "signal_break_direction"
    ] = signal_break_direction

    result[
        "signal_break_confirmed"
    ] = signal_break_confirmed

    # ----------------------------------------------------------
    # Final diagnostic interpretation
    #
    # IMPORTANT:
    # This is descriptive only.
    # ----------------------------------------------------------

    notes = []

    if not timing_ok:
        notes.append(
            "TIMING_ALIGNMENT_PROBLEM"
        )

    if (
        signal_setup
        and recorded_setup_normalized
        and not setup_match
    ):
        notes.append(
            "SETUP_FIELD_DIFFERS"
        )

    if (
        signal_setup_direction
        and recorded_setup_direction_normalized
        and not setup_direction_match
    ):
        notes.append(
            "SETUP_DIRECTION_FIELD_DIFFERS"
        )

    if direction_compatible is False:
        notes.append(
            "TRADE_DIRECTION_NOT_COMPATIBLE_WITH_SIGNAL_SETUP"
        )

    if (
        signal_break_confirmed
        not in {
            "",
            "TRUE",
            "1",
            "1.0",
        }
    ):
        notes.append(
            "BREAK_NOT_CONFIRMED_ON_SIGNAL_BAR"
        )

    result["notes"] = (
        "; ".join(notes)
        if notes
        else "NO_IMMEDIATE_FORENSIC_ANOMALY"
    )

    results.append(result)


# ==============================================================
# SAVE REPORT
# ==============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if results:

    fieldnames = sorted(
        {
            key
            for result in results
            for key in result.keys()
        }
    )

    # Keep the most important columns first.

    priority_columns = [
        "case_type",
        "symbol",
        "entry_date",
        "signal_date",
        "direction",
        "recorded_setup",
        "recorded_setup_direction",
        "signal_row_found",
        "entry_row_found",
        "signal_position",
        "timing_alignment",
        "signal_setup_matches_trade",
        "signal_setup_direction_matches_trade",
        "direction_compatible_with_signal_setup",
        "signal_break_direction",
        "signal_break_confirmed",
        "notes",
    ]

    ordered_fields = [
        field
        for field in priority_columns
        if field in fieldnames
    ]

    ordered_fields += [
        field
        for field in fieldnames
        if field not in ordered_fields
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=ordered_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)


# ==============================================================
# SUMMARY
# ==============================================================

print()
print("=" * 100)
print("SIGNAL-BAR FORENSIC SUMMARY")
print("=" * 100)

print(
    f"Cases analysed: {len(results)}"
)

timing_pass = sum(
    1
    for result in results
    if result.get("timing_alignment") == "PASS"
)

print(
    f"Timing alignment PASS: "
    f"{timing_pass}/{len(results)}"
)

setup_mismatches = sum(
    1
    for result in results
    if result.get(
        "signal_setup_matches_trade"
    ) == "NO"
)

print(
    f"Signal Setup vs trade Setup mismatches: "
    f"{setup_mismatches}"
)

direction_mismatches = sum(
    1
    for result in results
    if result.get(
        "direction_compatible_with_signal_setup"
    ) == "NO"
)

print(
    f"Direction incompatibilities: "
    f"{direction_mismatches}"
)

print()
print(
    f"Report saved to:"
)

print(
    f"  {OUTPUT_FILE}"
)

print()
print("=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)