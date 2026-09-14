"""
TradeSense-AI
Forensic Structure Convergence Audit

Purpose
-------
Compare market-structure / signal states generated using:

1. Stable research context:
       start_date="2021-01-01"

2. Rolling 5-year context:
       load_data()

The audit examines the ENTIRE OVERLAPPING DATE RANGE,
not only divergent trades.

It identifies:
- first overlapping date
- first state divergence
- total divergent dates
- longest consecutive divergence period
- last divergent date
- final convergence
- divergence by field
- divergence by symbol

IMPORTANT
---------
Diagnostic only.

This script does NOT modify:
- production code
- frozen historical trades
- research candidate trades
- holdout trades

Output:
    tests/output/reconciliation/
        structure_convergence_forensic.csv
        structure_convergence_summary.csv
"""

import csv
import os

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

STABLE_START_DATE = "2021-01-01"

OUTPUT_DIR = "tests/output/reconciliation"

DETAIL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "structure_convergence_forensic.csv",
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "structure_convergence_summary.csv",
)


# ==============================================================
# FIELDS
# ==============================================================

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


# ==============================================================
# HELPERS
# ==============================================================

def normalize_value(value):
    """
    Convert values into stable comparable strings.
    """

    if pd.isna(value):
        return ""

    return str(value)


def normalize_index(df):
    """
    Ensure the dataframe has a normalized DatetimeIndex.
    """

    result = df.copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[
        ~result.index.isna()
    ]

    result = result.sort_index()

    return result


def generate_stable_signals(symbol):
    """
    Generate signals with the stable research warm-up.
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

    return normalize_index(signals)


def generate_rolling_signals(symbol):
    """
    Generate signals using the current rolling 5-year
    load_data() behaviour.
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

    return normalize_index(signals)


def compare_rows(stable_row, rolling_row):
    """
    Compare the requested signal/structure fields.

    Returns:
        differences: list[str]
        stable_values: dict
        rolling_values: dict
    """

    differences = []

    stable_values = {}
    rolling_values = {}

    for field in COMPARE_FIELDS:

        if field in stable_row.index:
            stable_value = normalize_value(
                stable_row[field]
            )
        else:
            stable_value = ""

        if field in rolling_row.index:
            rolling_value = normalize_value(
                rolling_row[field]
            )
        else:
            rolling_value = ""

        stable_values[field] = stable_value
        rolling_values[field] = rolling_value

        if stable_value != rolling_value:
            differences.append(field)

    return (
        differences,
        stable_values,
        rolling_values,
    )


def calculate_divergence_periods(detail_rows):
    """
    Calculate consecutive divergence periods.

    detail_rows must be sorted by date.

    Returns:
        list of dictionaries
    """

    periods = []

    current_start = None
    current_end = None
    current_count = 0

    previous_date = None

    for row in detail_rows:

        date = pd.Timestamp(
            row["date"]
        )

        is_different = (
            row["state_status"]
            == "STATE_DIFFERENCE"
        )

        if is_different:

            if current_start is None:

                current_start = date
                current_end = date
                current_count = 1

            else:

                # We consider the sequence continuous
                # when the next available comparison row
                # is the next signal date in the dataset.
                current_end = date
                current_count += 1

        else:

            if current_start is not None:

                periods.append(
                    {
                        "start": current_start,
                        "end": current_end,
                        "comparison_rows": current_count,
                        "calendar_days": (
                            current_end - current_start
                        ).days + 1,
                    }
                )

                current_start = None
                current_end = None
                current_count = 0

        previous_date = date

    if current_start is not None:

        periods.append(
            {
                "start": current_start,
                "end": current_end,
                "comparison_rows": current_count,
                "calendar_days": (
                    current_end - current_start
                ).days + 1,
            }
        )

    return periods


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 115)
    print("TRADESENSE-AI FULL STRUCTURE CONVERGENCE FORENSIC")
    print("=" * 115)

    print()
    print(
        "Stable context :",
        f"{STABLE_START_DATE} warm-up"
    )

    print(
        "Rolling context:",
        "current rolling 5-year window"
    )

    print()
    print(
        "This audit compares the entire overlapping "
        "signal history."
    )

    print()
    print(
        "NO PRODUCTION DATA WILL BE MODIFIED."
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    all_detail_rows = []
    summary_rows = []

    # ==========================================================
    # SYMBOL LOOP
    # ==========================================================

    for symbol in SYMBOLS:

        print()
        print("=" * 115)
        print(f"SYMBOL: {symbol}")
        print("=" * 115)

        # ------------------------------------------------------
        # GENERATE BOTH CONTEXTS
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # OVERLAPPING DATES
        # ------------------------------------------------------

        stable_dates = set(
            stable_signals.index
        )

        rolling_dates = set(
            rolling_signals.index
        )

        overlap_dates = sorted(
            stable_dates & rolling_dates
        )

        if not overlap_dates:

            print()
            print(
                "No overlapping signal dates."
            )

            continue

        print()
        print(
            f"Overlapping dates: "
            f"{len(overlap_dates)}"
        )

        first_overlap = overlap_dates[0]
        last_overlap = overlap_dates[-1]

        print(
            "First overlap:",
            first_overlap.strftime("%Y-%m-%d")
        )

        print(
            "Last overlap :",
            last_overlap.strftime("%Y-%m-%d")
        )

        # ------------------------------------------------------
        # COMPARE ALL OVERLAPPING DATES
        # ------------------------------------------------------

        symbol_detail_rows = []

        field_difference_counts = {
            field: 0
            for field in COMPARE_FIELDS
        }

        total_different = 0
        total_matching = 0

        for date in overlap_dates:

            stable_row = stable_signals.loc[
                date
            ]

            rolling_row = rolling_signals.loc[
                date
            ]

            (
                differences,
                stable_values,
                rolling_values,
            ) = compare_rows(
                stable_row,
                rolling_row,
            )

            if differences:

                state_status = (
                    "STATE_DIFFERENCE"
                )

                total_different += 1

                for field in differences:
                    field_difference_counts[
                        field
                    ] += 1

            else:

                state_status = (
                    "STATE_MATCH"
                )

                total_matching += 1

            row = {
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "state_status": state_status,
                "different_fields": "|".join(
                    differences
                ),
            }

            for field in COMPARE_FIELDS:

                row[
                    f"stable_{field}"
                ] = stable_values[field]

                row[
                    f"rolling_{field}"
                ] = rolling_values[field]

            symbol_detail_rows.append(row)
            all_detail_rows.append(row)

        # ------------------------------------------------------
        # DIVERGENCE PERIODS
        # ------------------------------------------------------

        periods = calculate_divergence_periods(
            symbol_detail_rows
        )

        if periods:

            longest_period = max(
                periods,
                key=lambda x: x[
                    "comparison_rows"
                ],
            )

            first_divergence = periods[0][
                "start"
            ]

            last_divergence = periods[-1][
                "end"
            ]

        else:

            longest_period = None
            first_divergence = None
            last_divergence = None

        # ------------------------------------------------------
        # PRINT SYMBOL SUMMARY
        # ------------------------------------------------------

        divergence_pct = (
            (
                total_different
                / len(overlap_dates)
            )
            * 100
            if overlap_dates
            else 0
        )

        print()
        print("-" * 115)
        print("SYMBOL CONVERGENCE SUMMARY")
        print("-" * 115)

        print(
            f"Overlap dates:          {len(overlap_dates)}"
        )

        print(
            f"State differences:      {total_different}"
        )

        print(
            f"State matches:          {total_matching}"
        )

        print(
            f"Divergence percentage:  "
            f"{divergence_pct:.2f}%"
        )

        print(
            "First divergence:       "
            + (
                first_divergence.strftime(
                    "%Y-%m-%d"
                )
                if first_divergence is not None
                else "NONE"
            )
        )

        print(
            "Last divergence:        "
            + (
                last_divergence.strftime(
                    "%Y-%m-%d"
                )
                if last_divergence is not None
                else "NONE"
            )
        )

        if longest_period:

            print(
                "Longest divergence:     "
                f"{longest_period['start'].strftime('%Y-%m-%d')}"
                " -> "
                f"{longest_period['end'].strftime('%Y-%m-%d')}"
            )

            print(
                "Longest comparison rows:"
                f" {longest_period['comparison_rows']}"
            )

            print(
                "Longest calendar span:  "
                f"{longest_period['calendar_days']} days"
            )

        else:

            print(
                "Longest divergence:     NONE"
            )

        print()
        print("FIELD-LEVEL DIFFERENCES")

        for field in COMPARE_FIELDS:

            count = field_difference_counts[
                field
            ]

            if count:

                print(
                    f"  {field:<22} {count}"
                )

        # ------------------------------------------------------
        # SUMMARY ROW
        # ------------------------------------------------------

        summary_row = {
            "symbol": symbol,
            "stable_start_date": STABLE_START_DATE,
            "overlap_start": (
                first_overlap.strftime(
                    "%Y-%m-%d"
                )
            ),
            "overlap_end": (
                last_overlap.strftime(
                    "%Y-%m-%d"
                )
            ),
            "overlap_rows": len(
                overlap_dates
            ),
            "state_difference_rows": (
                total_different
            ),
            "state_match_rows": (
                total_matching
            ),
            "divergence_percentage": (
                round(
                    divergence_pct,
                    4,
                )
            ),
            "first_divergence": (
                first_divergence.strftime(
                    "%Y-%m-%d"
                )
                if first_divergence is not None
                else ""
            ),
            "last_divergence": (
                last_divergence.strftime(
                    "%Y-%m-%d"
                )
                if last_divergence is not None
                else ""
            ),
            "divergence_period_count": len(
                periods
            ),
            "longest_divergence_start": (
                longest_period["start"].strftime(
                    "%Y-%m-%d"
                )
                if longest_period
                else ""
            ),
            "longest_divergence_end": (
                longest_period["end"].strftime(
                    "%Y-%m-%d"
                )
                if longest_period
                else ""
            ),
            "longest_divergence_rows": (
                longest_period[
                    "comparison_rows"
                ]
                if longest_period
                else 0
            ),
            "longest_divergence_calendar_days": (
                longest_period[
                    "calendar_days"
                ]
                if longest_period
                else 0
            ),
        }

        for field in COMPARE_FIELDS:

            summary_row[
                f"difference_count_{field}"
            ] = field_difference_counts[
                field
            ]

        summary_rows.append(
            summary_row
        )

    # ==========================================================
    # WRITE DETAIL CSV
    # ==========================================================

    detail_fields = [
        "symbol",
        "date",
        "state_status",
        "different_fields",
    ]

    for field in COMPARE_FIELDS:

        detail_fields.append(
            f"stable_{field}"
        )

        detail_fields.append(
            f"rolling_{field}"
        )

    with open(
        DETAIL_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=detail_fields,
        )

        writer.writeheader()
        writer.writerows(
            all_detail_rows
        )

    # ==========================================================
    # WRITE SUMMARY CSV
    # ==========================================================

    summary_fields = [
        "symbol",
        "stable_start_date",
        "overlap_start",
        "overlap_end",
        "overlap_rows",
        "state_difference_rows",
        "state_match_rows",
        "divergence_percentage",
        "first_divergence",
        "last_divergence",
        "divergence_period_count",
        "longest_divergence_start",
        "longest_divergence_end",
        "longest_divergence_rows",
        "longest_divergence_calendar_days",
    ]

    for field in COMPARE_FIELDS:

        summary_fields.append(
            f"difference_count_{field}"
        )

    with open(
        SUMMARY_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # ==========================================================
    # GLOBAL SUMMARY
    # ==========================================================

    total_overlap = sum(
        int(row["overlap_rows"])
        for row in summary_rows
    )

    total_difference = sum(
        int(row["state_difference_rows"])
        for row in summary_rows
    )

    total_match = sum(
        int(row["state_match_rows"])
        for row in summary_rows
    )

    print()
    print()
    print("=" * 115)
    print("GLOBAL CONVERGENCE SUMMARY")
    print("=" * 115)

    print(
        f"Symbols analysed:        "
        f"{len(summary_rows)}"
    )

    print(
        f"Total overlapping rows:  "
        f"{total_overlap}"
    )

    print(
        f"State differences:       "
        f"{total_difference}"
    )

    print(
        f"State matches:           "
        f"{total_match}"
    )

    if total_overlap:

        print(
            f"Overall divergence:      "
            f"{(total_difference / total_overlap) * 100:.2f}%"
        )

    print()
    print(
        "Detail report:"
    )

    print(
        DETAIL_OUTPUT
    )

    print()
    print(
        "Summary report:"
    )

    print(
        SUMMARY_OUTPUT
    )

    print()
    print("=" * 115)
    print("FORENSIC CONCLUSION")
    print("=" * 115)

    if total_difference == 0:

        print(
            "RESULT: COMPLETE STATE CONVERGENCE"
        )

        print(
            "The stable and rolling contexts produce "
            "identical states across their entire overlap."
        )

        print(
            "Warm-up differences do NOT explain the "
            "trade identity divergence."
        )

    elif total_match == 0:

        print(
            "RESULT: PERSISTENT STATE DIVERGENCE"
        )

        print(
            "The stable and rolling contexts differ "
            "throughout the entire overlapping period."
        )

        print(
            "Warm-up has a persistent effect on the "
            "signal-state calculation."
        )

    else:

        print(
            "RESULT: PARTIAL / EVENTUAL CONVERGENCE"
        )

        print(
            "The stable and rolling contexts differ "
            "during part of the overlap and match during "
            "another part."
        )

        print(
            "This is consistent with a warm-up-dependent "
            "state that eventually converges."
        )

        print(
            "The convergence timeline must be examined "
            "before changing the research dataset."
        )

    print()
    print(
        "NO PRODUCTION OR FROZEN DATA WAS MODIFIED."
    )

    print()
    print("=" * 115)
    print("END OF FORENSIC ANALYSIS")
    print("=" * 115)


if __name__ == "__main__":
    main()