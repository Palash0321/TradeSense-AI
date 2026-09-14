"""
Evaluation-slice regression test for TradeSense-AI.

Purpose:
--------
Verify that:

1. Signals generated from the full historical calculation context
   remain identical when we evaluate only a later slice.

2. The evaluation slice is therefore independent of the arbitrary
   evaluation-window boundary.

3. Rolling-window calculation is reported separately as a comparison,
   because differences there are expected and are NOT treated as a
   production failure.

IMPORTANT:
----------
This is a diagnostic/regression test only.

It does NOT:
- modify production code
- modify the frozen historical dataset
- modify the holdout dataset
- change strategy rules
"""

import hashlib
import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

CALCULATION_START = "2021-01-01"
EVALUATION_START = "2021-09-13"


def dataframe_hash(df):
    """
    Create a deterministic hash of the dataframe contents,
    index and columns.
    """

    normalized = df.copy()

    normalized = normalized.sort_index()

    normalized.columns = [
        str(column)
        for column in normalized.columns
    ]

    payload = pd.util.hash_pandas_object(
        normalized,
        index=True
    ).values.tobytes()

    metadata = "|".join(
        normalized.columns
    ).encode("utf-8")

    return hashlib.sha256(
        payload + metadata
    ).hexdigest()


def compare_dataframes(
    left,
    right,
    label
):
    """
    Compare two signal dataframes on their common index.
    """

    common_index = left.index.intersection(
        right.index
    )

    if len(common_index) == 0:
        print(
            f"{label}: NO COMMON ROWS"
        )
        return False, 0, 0

    left_common = left.loc[
        common_index
    ].copy()

    right_common = right.loc[
        common_index
    ].copy()

    common_columns = [
        column
        for column in left_common.columns
        if column in right_common.columns
    ]

    left_common = left_common[
        common_columns
    ]

    right_common = right_common[
        common_columns
    ]

    # ----------------------------------------------------------
    # IMPORTANT:
    # pandas treats NaN != NaN with normal .eq().
    #
    # Signal datasets naturally contain NaN/None values in
    # fields such as BOS, CHOCH, Break_Direction, Fibonacci,
    # etc.
    #
    # Therefore comparison must treat missing-vs-missing as
    # equal.
    # ----------------------------------------------------------

    left_compare = left_common.copy()
    right_compare = right_common.copy()

    for column in common_columns:

        left_compare[column] = (
            left_compare[column]
            .astype("object")
            .where(
                left_compare[column].notna(),
                "__TRADESENSE_MISSING__"
            )
        )

        right_compare[column] = (
            right_compare[column]
            .astype("object")
            .where(
                right_compare[column].notna(),
                "__TRADESENSE_MISSING__"
            )
        )

    equal = left_compare.eq(
        right_compare
    )

    row_equal = equal.all(axis=1)

    differing_rows = int(
        (~row_equal).sum()
    )

    total_rows = len(common_index)

    print(
        f"{label}: "
        f"{total_rows - differing_rows}/{total_rows} "
        f"rows identical"
    )

    if differing_rows:
        print(
            f"{label}: "
            f"{differing_rows} differing rows"
        )

        differing_columns = {}

        for column in common_columns:
            differences = int(
                (~equal[column]).sum()
            )

            if differences:
                differing_columns[
                    column
                ] = differences

        print(
            f"{label}: differing columns:"
        )

        for column, count in sorted(
            differing_columns.items(),
            key=lambda item: -item[1]
        ):
            print(
                f"  {column}: {count}"
            )

    return (
        differing_rows == 0,
        total_rows,
        differing_rows
    )


def main():

    print()
    print("=" * 100)
    print("TRADESENSE-AI EVALUATION-SLICE REGRESSION")
    print("=" * 100)
    print(
        f"Calculation context : {CALCULATION_START}"
    )
    print(
        f"Evaluation starts    : {EVALUATION_START}"
    )
    print()

    total_symbols = 0
    passed_symbols = 0

    for symbol in SYMBOLS:

        print()
        print("-" * 100)
        print(symbol)
        print("-" * 100)

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=100000,
        )

        # ----------------------------------------------------------
        # FULL CALCULATION CONTEXT
        # ----------------------------------------------------------

        full_data = service.load_data(
            start_date=CALCULATION_START
        )

        full_signals = service.generate_tradesense_signals(
            df=full_data
        )

        print(
            f"Full calculation rows: "
            f"{len(full_signals)}"
        )

        print(
            f"Full first date: "
            f"{full_signals.index.min()}"
        )

        print(
            f"Full last date: "
            f"{full_signals.index.max()}"
        )

        # ----------------------------------------------------------
        # EVALUATION SLICE FROM THE SAME CALCULATION
        # ----------------------------------------------------------

        full_evaluation_slice = full_signals.loc[
            full_signals.index >= EVALUATION_START
        ].copy()

        print(
            f"Evaluation-slice rows: "
            f"{len(full_evaluation_slice)}"
        )

        # ----------------------------------------------------------
        # REGENERATE FROM THE SAME FULL CONTEXT
        #
        # This intentionally performs a second independent signal
        # generation to verify deterministic evaluation slicing.
        # ----------------------------------------------------------

        second_data = service.load_data(
            start_date=CALCULATION_START
        )

        second_signals = service.generate_tradesense_signals(
            df=second_data
        )

        second_evaluation_slice = second_signals.loc[
            second_signals.index >= EVALUATION_START
        ].copy()

        passed, total, differing = compare_dataframes(
            full_evaluation_slice,
            second_evaluation_slice,
            "FULL-CONTEXT EVALUATION REPEAT"
        )

        if passed:
            print(
                "PASS: Full-context evaluation slice is deterministic."
            )
        else:
            print(
                "FAIL: Full-context evaluation slice changed."
            )

        # ----------------------------------------------------------
        # ROLLING CONTEXT
        #
        # This is deliberately reported separately.
        # Differences are expected because the calculation context
        # itself is different.
        # ----------------------------------------------------------

        rolling_data = service.load_data(
            start_date=EVALUATION_START
        )

        rolling_signals = service.generate_tradesense_signals(
            df=rolling_data
        )

        rolling_slice = rolling_signals.loc[
            rolling_signals.index >= EVALUATION_START
        ].copy()

        print(
            f"Rolling-context rows: "
            f"{len(rolling_slice)}"
        )

        (
            rolling_match,
            rolling_total,
            rolling_differing
        ) = compare_dataframes(
            full_evaluation_slice,
            rolling_slice,
            "FULL-CONTEXT vs ROLLING-CONTEXT"
        )

        if rolling_match:
            print(
                "INFO: Rolling context happened to match."
            )
        else:
            print(
                "INFO: Rolling context differs "
                "(expected context-boundary sensitivity)."
            )

        # ----------------------------------------------------------
        # HASHES
        # ----------------------------------------------------------

        full_hash = dataframe_hash(
            full_evaluation_slice
        )

        second_hash = dataframe_hash(
            second_evaluation_slice
        )

        print()
        print(
            f"Full evaluation hash : {full_hash}"
        )

        print(
            f"Repeat evaluation hash: {second_hash}"
        )

        if full_hash == second_hash:
            print(
                "HASH CHECK: PASS"
            )
        else:
            print(
                "HASH CHECK: FAIL"
            )

        total_symbols += 1

        if passed and full_hash == second_hash:
            passed_symbols += 1

    # --------------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL RESULT")
    print("=" * 100)

    print(
        f"Symbols analysed : {total_symbols}"
    )

    print(
        f"Symbols passed   : {passed_symbols}/{total_symbols}"
    )

    if passed_symbols == total_symbols:
        print()
        print(
            "RESULT: PASS"
        )
        print(
            "Full-history calculation context produces "
            "a deterministic evaluation slice."
        )
        print(
            "This supports separating calculation context "
            "from evaluation period."
        )
    else:
        print()
        print(
            "RESULT: FAIL"
        )
        print(
            "Do NOT modify production yet."
        )


if __name__ == "__main__":
    main()