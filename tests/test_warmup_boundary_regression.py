"""
TradeSense-AI
Warm-Up Boundary Determinism Test

Purpose:
    Separate three questions:

    1. Does changing the starting boundary change the signals?
    2. Is signal generation deterministic when the exact same
       dataframe is supplied twice?
    3. Does filtering a stable full-history result reproduce itself?

Diagnostic only.
No production data is modified.
"""

import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "TCS.NS"

REFERENCE_START = "2021-01-01"
ROLLING_START = "2021-09-13"
ANALYSIS_START = "2021-09-13"


COMPARE_COLUMNS = [
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
    "Break_Confirmed",
    "Break_Age",
    "Liquidity_Sweep",
    "Trend",
    "Trend_Strength",
    "Trend_Slope",
]


def prepare(df):
    if df is None or df.empty:
        return pd.DataFrame()

    columns = [
        column
        for column in COMPARE_COLUMNS
        if column in df.columns
    ]

    result = df[columns].copy()

    result.index = pd.to_datetime(result.index)

    return result


def compare(reference, candidate, label):
    common_index = reference.index.intersection(
        candidate.index
    )

    reference = reference.loc[common_index].copy()
    candidate = candidate.loc[common_index].copy()

    common_columns = [
        column
        for column in reference.columns
        if column in candidate.columns
    ]

    differences = (
        reference[common_columns].astype(str)
        != candidate[common_columns].astype(str)
    )

    different_rows = differences.any(axis=1)

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    print(
        f"Common rows: {len(common_index)}"
    )

    print(
        f"Different rows: {int(different_rows.sum())}"
    )

    if len(common_index):
        match_rate = (
            (len(common_index) - int(different_rows.sum()))
            / len(common_index)
        ) * 100
    else:
        match_rate = 0

    print(
        f"Match rate: {match_rate:.2f}%"
    )

    if different_rows.any():

        print()
        print("FIELD DIFFERENCE COUNTS:")

        for column in common_columns:

            count = int(
                differences[column].sum()
            )

            if count > 0:

                print(
                    f"  {column}: {count}"
                )

        print()
        print("FIRST DIFFERENT ROW:")

        first_index = differences.any(axis=1).idxmax()

        print(
            f"Date: {first_index}"
        )

        for column in common_columns:

            left = reference.loc[
                first_index,
                column
            ]

            right = candidate.loc[
                first_index,
                column
            ]

            if str(left) != str(right):

                print(
                    f"  {column}: "
                    f"{left!r} -> {right!r}"
                )

    return {
        "rows": len(common_index),
        "different": int(different_rows.sum()),
    }


def main():

    print("=" * 100)
    print("TRADESENSE WARM-UP / DETERMINISM FORENSIC")
    print("=" * 100)

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    # ------------------------------------------------------------------
    # LOAD STABLE DATA ONCE
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("LOADING STABLE DATA")
    print("-" * 100)

    stable_data = service.load_data(
        start_date=REFERENCE_START
    )

    if stable_data is None or stable_data.empty:
        raise RuntimeError(
            "Stable data could not be loaded."
        )

    print(
        f"Stable data rows: {len(stable_data)}"
    )

    print(
        f"Stable first date: {stable_data.index.min()}"
    )

    print(
        f"Stable last date: {stable_data.index.max()}"
    )

    # ------------------------------------------------------------------
    # GENERATE STABLE SIGNALS — RUN 1
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("STABLE SIGNAL RUN 1")
    print("-" * 100)

    stable_run_1 = service.generate_tradesense_signals(
        df=stable_data.copy()
    )

    stable_run_1 = prepare(stable_run_1)

    # ------------------------------------------------------------------
    # GENERATE STABLE SIGNALS — RUN 2
    # EXACT SAME DATAFRAME
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("STABLE SIGNAL RUN 2")
    print("-" * 100)

    stable_run_2 = service.generate_tradesense_signals(
        df=stable_data.copy()
    )

    stable_run_2 = prepare(stable_run_2)

    # ------------------------------------------------------------------
    # DETERMINISM TEST
    # ------------------------------------------------------------------

    deterministic = compare(
        stable_run_1[
            stable_run_1.index >= pd.Timestamp(
                ANALYSIS_START
            )
        ],
        stable_run_2[
            stable_run_2.index >= pd.Timestamp(
                ANALYSIS_START
            )
        ],
        "TEST 1: SAME DATAFRAME RUN TWICE"
    )

    # ------------------------------------------------------------------
    # ROLLING DATA
    # ------------------------------------------------------------------

    print()
    print("-" * 100)
    print("LOADING ROLLING DATA")
    print("-" * 100)

    rolling_data = service.load_data(
        start_date=ROLLING_START
    )

    if rolling_data is None or rolling_data.empty:
        raise RuntimeError(
            "Rolling data could not be loaded."
        )

    print(
        f"Rolling data rows: {len(rolling_data)}"
    )

    print(
        f"Rolling first date: {rolling_data.index.min()}"
    )

    print(
        f"Rolling last date: {rolling_data.index.max()}"
    )

    rolling_signals = service.generate_tradesense_signals(
        df=rolling_data.copy()
    )

    rolling_signals = prepare(
        rolling_signals
    )

    rolling_signals = rolling_signals[
        rolling_signals.index >= pd.Timestamp(
            ANALYSIS_START
        )
    ]

    stable_signals = stable_run_1[
        stable_run_1.index >= pd.Timestamp(
            ANALYSIS_START
        )
    ]

    # ------------------------------------------------------------------
    # BOUNDARY TEST
    # ------------------------------------------------------------------

    boundary = compare(
        stable_signals,
        rolling_signals,
        "TEST 2: STABLE HISTORY VS ROLLING HISTORY"
    )

    # ------------------------------------------------------------------
    # FINAL VERDICT
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)

    deterministic_pass = (
        deterministic["different"] == 0
    )

    boundary_detected = (
        boundary["different"] > 0
    )

    print(
        f"Same-data determinism: "
        f"{'PASS' if deterministic_pass else 'FAIL'}"
    )

    print(
        f"Boundary sensitivity detected: "
        f"{'YES' if boundary_detected else 'NO'}"
    )

    print()

    if deterministic_pass and boundary_detected:

        print(
            "RESULT: PASS"
        )

        print(
            "The signal engine is deterministic for identical "
            "input history."
        )

        print(
            "Changing the historical boundary changes the "
            "resulting state."
        )

        print()
        print(
            "NEXT: implement explicit warm-up architecture."
        )

    elif not deterministic_pass:

        print(
            "RESULT: INVESTIGATE DETERMINISM"
        )

        print(
            "The exact same input dataframe produced "
            "different signals."
        )

        print(
            "DO NOT MODIFY PRODUCTION CODE YET."
        )

    else:

        print(
            "RESULT: NO BOUNDARY EFFECT DETECTED"
        )

        print(
            "DO NOT MODIFY PRODUCTION CODE YET."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()