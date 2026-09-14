"""
TradeSense-AI
Forensic Structure Context Comparison

Purpose
-------
Compare TradeSense signal/market-structure states generated from:

1. Stable research warm-up:
       start_date="2021-01-01"

2. Rolling production-style window:
       load_data()  -> current 5-year window

The purpose is NOT to change production logic.

It is purely diagnostic and investigates whether historical
trade identity differences are caused by different historical
warm-up boundaries.

IMPORTANT
---------
This script does not modify:
- production code
- frozen historical trades
- holdout trades

Output:
    tests/output/reconciliation/structure_context_forensic.csv
"""

import csv
import os

import pandas as pd

from app.services.backtest_service import BacktestService


# --------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

HISTORICAL_FILE = "tests/output/tradesense_trades.csv"
CANDIDATE_FILE = "tests/output/tradesense_trades_research_candidate.csv"

OUTPUT_DIR = "tests/output/reconciliation"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "structure_context_forensic.csv",
)

STABLE_START_DATE = "2021-01-01"


# --------------------------------------------------------------
# SIGNAL FIELDS TO COMPARE
# --------------------------------------------------------------

COMPARE_FIELDS = [
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
]


# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------

def clean_value(value):
    """
    Normalize values so comparison is stable.
    """
    if pd.isna(value):
        return ""

    return str(value)


def load_trade_identities():
    """
    Load all identity differences between frozen historical
    and fresh research candidate datasets.
    """

    historical = pd.read_csv(HISTORICAL_FILE)
    candidate = pd.read_csv(CANDIDATE_FILE)

    historical["_symbol"] = historical["_symbol"].astype(str)
    candidate["_symbol"] = candidate["_symbol"].astype(str)

    historical["entry_date"] = (
        historical["entry_date"]
        .astype(str)
        .str[:10]
    )

    candidate["entry_date"] = (
        candidate["entry_date"]
        .astype(str)
        .str[:10]
    )

    historical["direction"] = (
        historical["direction"]
        .astype(str)
    )

    candidate["direction"] = (
        candidate["direction"]
        .astype(str)
    )

    historical_ids = set(
        zip(
            historical["_symbol"],
            historical["entry_date"],
            historical["direction"],
        )
    )

    candidate_ids = set(
        zip(
            candidate["_symbol"],
            candidate["entry_date"],
            candidate["direction"],
        )
    )

    fresh_only = sorted(
        candidate_ids - historical_ids
    )

    historical_only = sorted(
        historical_ids - candidate_ids
    )

    return fresh_only, historical_only


def create_signal_map(signals):
    """
    Create date -> signal row mapping.

    A signal is generated on the signal bar and the trade enters
    on the next bar.

    Therefore the trade entry date maps to the immediately
    preceding signal bar.
    """

    signal_map = {}

    if signals is None or len(signals) == 0:
        return signal_map

    working = signals.copy()

    if not isinstance(working.index, pd.DatetimeIndex):
        working.index = pd.to_datetime(
            working.index,
            errors="coerce",
        )

    for index, row in working.iterrows():

        if pd.isna(index):
            continue

        signal_date = index.strftime("%Y-%m-%d")

        signal_map[signal_date] = row

    return signal_map


def get_signal_for_entry(signal_map, entry_date):
    """
    Entry date -> previous signal bar.

    Since execution occurs at the next bar open,
    signal date is the immediately preceding trading bar.
    """

    try:
        entry_timestamp = pd.Timestamp(entry_date)
    except Exception:
        return None, None

    signal_date = (
        entry_timestamp - pd.tseries.offsets.BDay(1)
    ).strftime("%Y-%m-%d")

    # First try exact previous business day.
    if signal_date in signal_map:
        return signal_date, signal_map[signal_date]

    # Fallback: find the latest available signal date
    # strictly before entry date.
    available_dates = []

    for date_string in signal_map:
        try:
            timestamp = pd.Timestamp(date_string)

            if timestamp < entry_timestamp:
                available_dates.append(
                    (timestamp, date_string)
                )

        except Exception:
            continue

    if not available_dates:
        return None, None

    available_dates.sort()

    _, best_date = available_dates[-1]

    return best_date, signal_map[best_date]


# --------------------------------------------------------------
# LOAD DATA / GENERATE SIGNALS
# --------------------------------------------------------------

def generate_stable_signals(symbol):
    """
    Stable research context.

    Long historical warm-up beginning 2021-01-01.
    """

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    data = service.load_data(
        start_date=STABLE_START_DATE
    )

    signals = service.generate_tradesense_signals(
        df=data
    )

    return signals


def generate_rolling_signals(symbol):
    """
    Rolling production-style context.

    Uses BacktestService.load_data() with no explicit start date.
    Current implementation uses the rolling 5-year Yahoo Finance
    window.
    """

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    data = service.load_data()

    signals = service.generate_tradesense_signals(
        df=data
    )

    return signals


# --------------------------------------------------------------
# MAIN FORENSIC ANALYSIS
# --------------------------------------------------------------

def main():

    print()
    print("=" * 110)
    print("TRADESENSE-AI STRUCTURE CONTEXT FORENSIC")
    print("=" * 110)

    print()
    print(
        "Comparing stable 2021-01-01 warm-up "
        "against rolling 5-year warm-up."
    )

    print()
    print(
        "Frozen dataset:",
        HISTORICAL_FILE
    )

    print(
        "Candidate dataset:",
        CANDIDATE_FILE
    )

    print(
        "Output:",
        OUTPUT_FILE
    )

    print()

    # ----------------------------------------------------------
    # LOAD DIVERGENT TRADE IDENTITIES
    # ----------------------------------------------------------

    fresh_only, historical_only = load_trade_identities()

    divergent_ids = sorted(
        set(fresh_only) | set(historical_only)
    )

    print("=" * 110)
    print("DIVERGENT TRADE IDENTITIES")
    print("=" * 110)

    print(
        f"Fresh-only trades:       {len(fresh_only)}"
    )

    print(
        f"Historical-only trades:  {len(historical_only)}"
    )

    print(
        f"Total divergent cases:   {len(divergent_ids)}"
    )

    print()

    # ----------------------------------------------------------
    # PREPARE OUTPUT
    # ----------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    rows = []

    total_cases = 0
    stable_signal_found = 0
    rolling_signal_found = 0
    state_differences = 0

    # ----------------------------------------------------------
    # PROCESS SYMBOL BY SYMBOL
    # ----------------------------------------------------------

    for symbol in SYMBOLS:

        symbol_cases = [
            item
            for item in divergent_ids
            if item[0] == symbol
        ]

        if not symbol_cases:
            continue

        print()
        print("=" * 110)
        print(f"SYMBOL: {symbol}")
        print("=" * 110)

        print(
            f"Divergent cases for {symbol}: "
            f"{len(symbol_cases)}"
        )

        print()
        print("Generating stable signals...")

        stable_signals = generate_stable_signals(
            symbol
        )

        print(
            f"Stable signal rows: "
            f"{len(stable_signals)}"
        )

        print()
        print("Generating rolling signals...")

        rolling_signals = generate_rolling_signals(
            symbol
        )

        print(
            f"Rolling signal rows: "
            f"{len(rolling_signals)}"
        )

        stable_map = create_signal_map(
            stable_signals
        )

        rolling_map = create_signal_map(
            rolling_signals
        )

        # ------------------------------------------------------
        # PROCESS EACH DIVERGENT TRADE
        # ------------------------------------------------------

        for (
            case_symbol,
            entry_date,
            direction,
        ) in symbol_cases:

            total_cases += 1

            stable_signal_date, stable_row = (
                get_signal_for_entry(
                    stable_map,
                    entry_date,
                )
            )

            rolling_signal_date, rolling_row = (
                get_signal_for_entry(
                    rolling_map,
                    entry_date,
                )
            )

            stable_found = stable_row is not None
            rolling_found = rolling_row is not None

            if stable_found:
                stable_signal_found += 1

            if rolling_found:
                rolling_signal_found += 1

            differences = []

            for field in COMPARE_FIELDS:

                stable_value = ""

                rolling_value = ""

                if stable_found and field in stable_row.index:
                    stable_value = clean_value(
                        stable_row[field]
                    )

                if rolling_found and field in rolling_row.index:
                    rolling_value = clean_value(
                        rolling_row[field]
                    )

                if stable_value != rolling_value:
                    differences.append(field)

            context_status = (
                "STATE_DIFFERENCE"
                if differences
                else "STATE_MATCH"
            )

            if differences:
                state_differences += 1

            # --------------------------------------------------
            # PRINT CASE
            # --------------------------------------------------

            print()
            print("-" * 110)

            print(
                f"Trade: {case_symbol} | "
                f"{entry_date} | "
                f"{direction}"
            )

            print(
                f"Stable signal bar:  "
                f"{stable_signal_date or 'NOT FOUND'}"
            )

            print(
                f"Rolling signal bar: "
                f"{rolling_signal_date or 'NOT FOUND'}"
            )

            print(
                f"Context result:      "
                f"{context_status}"
            )

            if differences:
                print(
                    "Different fields:"
                )

                print(
                    "  " + ", ".join(differences)
                )

            else:
                print(
                    "Different fields: NONE"
                )

            # --------------------------------------------------
            # SAVE FIELD-LEVEL RECORD
            # --------------------------------------------------

            output_row = {
                "symbol": case_symbol,
                "entry_date": entry_date,
                "direction": direction,
                "stable_signal_date": (
                    stable_signal_date or ""
                ),
                "rolling_signal_date": (
                    rolling_signal_date or ""
                ),
                "stable_signal_found": stable_found,
                "rolling_signal_found": rolling_found,
                "context_status": context_status,
                "different_fields": "|".join(
                    differences
                ),
            }

            for field in COMPARE_FIELDS:

                stable_value = ""

                rolling_value = ""

                if stable_found and field in stable_row.index:
                    stable_value = clean_value(
                        stable_row[field]
                    )

                if rolling_found and field in rolling_row.index:
                    rolling_value = clean_value(
                        rolling_row[field]
                    )

                output_row[
                    f"stable_{field}"
                ] = stable_value

                output_row[
                    f"rolling_{field}"
                ] = rolling_value

            rows.append(output_row)

    # ----------------------------------------------------------
    # WRITE REPORT
    # ----------------------------------------------------------

    fieldnames = [
        "symbol",
        "entry_date",
        "direction",
        "stable_signal_date",
        "rolling_signal_date",
        "stable_signal_found",
        "rolling_signal_found",
        "context_status",
        "different_fields",
    ]

    for field in COMPARE_FIELDS:
        fieldnames.append(
            f"stable_{field}"
        )
        fieldnames.append(
            f"rolling_{field}"
        )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    print()
    print()
    print("=" * 110)
    print("FORENSIC SUMMARY")
    print("=" * 110)

    print(
        f"Total divergent cases:       {total_cases}"
    )

    print(
        f"Stable signal found:         "
        f"{stable_signal_found}/{total_cases}"
    )

    print(
        f"Rolling signal found:        "
        f"{rolling_signal_found}/{total_cases}"
    )

    print(
        f"Cases with state difference: "
        f"{state_differences}/{total_cases}"
    )

    state_matches = (
        total_cases - state_differences
    )

    print(
        f"Cases with state match:      "
        f"{state_matches}/{total_cases}"
    )

    print()
    print(
        f"Report saved to: {OUTPUT_FILE}"
    )

    print()

    # ----------------------------------------------------------
    # INTERPRETATION
    # ----------------------------------------------------------

    print("=" * 110)
    print("INTERPRETATION")
    print("=" * 110)

    if total_cases == 0:

        print(
            "No divergent trade identities were found."
        )

    elif (
        stable_signal_found == total_cases
        and rolling_signal_found == total_cases
        and state_differences == total_cases
    ):

        print(
            "RESULT: STRONG WARM-UP STATE DIFFERENCE SIGNAL"
        )

        print(
            "Every divergent case has both signal contexts, "
            "and every case has at least one market-structure/"
            "signal-state difference."
        )

        print(
            "This strongly supports different historical "
            "warm-up boundaries as the cause of the trade "
            "identity differences."
        )

        print(
            "DO NOT MODIFY STRATEGY OR FROZEN DATA YET."
        )

    elif (
        state_differences > 0
        and state_differences < total_cases
    ):

        print(
            "RESULT: PARTIAL WARM-UP STATE DIFFERENCE"
        )

        print(
            "Some divergent cases have different states, "
            "while others do not."
        )

        print(
            "Further investigation is required before "
            "concluding that warm-up alone explains the "
            "identity differences."
        )

    elif state_differences == 0:

        print(
            "RESULT: NO STRUCTURE-STATE DIFFERENCE"
        )

        print(
            "The two warm-up contexts produce matching "
            "states for the divergent trade dates."
        )

        print(
            "Therefore the identity divergence requires "
            "another explanation."
        )

    else:

        print(
            "RESULT: INCOMPLETE CONTEXT ALIGNMENT"
        )

        print(
            "One or both signal contexts could not be "
            "reconstructed for some cases."
        )

        print(
            "Do not draw conclusions from this run."
        )

    print()

    print("=" * 110)
    print("END OF FORENSIC ANALYSIS")
    print("=" * 110)


if __name__ == "__main__":
    main()