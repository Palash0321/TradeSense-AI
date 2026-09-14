from __future__ import annotations

import hashlib
import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure


SYMBOL = "TCS.NS"
START_DATE = "2021-01-01"
CHECK_DATE = "2021-09-13"


def dataframe_hash(df: pd.DataFrame) -> str:
    """
    Stable hash of the complete dataframe values,
    index, columns and dtypes.
    """

    payload = pd.util.hash_pandas_object(
        df,
        index=True
    ).values.tobytes()

    metadata = (
        str(list(df.columns))
        + str(df.dtypes.astype(str).tolist())
        + str(df.index.tolist())
    ).encode("utf-8")

    return hashlib.sha256(
        payload + metadata
    ).hexdigest()


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def compare_values(label, first, second):

    if first == second:
        print(f"{label}: SAME")
        return True

    print(f"{label}: DIFFERENT")
    print(f"  RUN 1: {first}")
    print(f"  RUN 2: {second}")

    return False


def main():

    print_section(
        "TRADESENSE SIGNAL DETERMINISM — LAYER ISOLATION FORENSIC"
    )

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    # ==========================================================
    # LOAD DATA ONCE
    # ==========================================================

    print_section("1. LOAD STABLE DATA")

    stable_data = service.load_data(
        start_date=START_DATE
    )

    print(
        f"Rows: {len(stable_data)}"
    )

    print(
        f"First date: {stable_data.index.min()}"
    )

    print(
        f"Last date: {stable_data.index.max()}"
    )

    original_hash = dataframe_hash(
        stable_data
    )

    print(
        f"Original input hash: {original_hash}"
    )

    original_columns = list(
        stable_data.columns
    )

    original_dtypes = (
        stable_data.dtypes.astype(str).to_dict()
    )

    # ==========================================================
    # TEST A
    # SAME DATAFRAME CONTENT BEFORE / AFTER RUN
    # ==========================================================

    print_section(
        "2. INPUT MUTATION TEST"
    )

    input_before = stable_data.copy(
        deep=True
    )

    input_before_hash = dataframe_hash(
        input_before
    )

    signals_run_1 = (
        service.generate_tradesense_signals(
            df=stable_data
        )
    )

    input_after_hash = dataframe_hash(
        stable_data
    )

    print(
        f"Input hash BEFORE run: {input_before_hash}"
    )

    print(
        f"Input hash AFTER run : {input_after_hash}"
    )

    if input_before_hash == input_after_hash:

        print(
            "INPUT MUTATION: PASS"
        )

    else:

        print(
            "INPUT MUTATION: FAIL"
        )

        print(
            "The production signal generator modified the supplied dataframe."
        )

    print(
        f"Columns before: {original_columns}"
    )

    print(
        f"Columns after : {list(stable_data.columns)}"
    )

    if (
        original_columns
        == list(stable_data.columns)
    ):

        print(
            "COLUMN MUTATION: PASS"
        )

    else:

        print(
            "COLUMN MUTATION: FAIL"
        )

    after_dtypes = (
        stable_data.dtypes.astype(str).to_dict()
    )

    if original_dtypes == after_dtypes:

        print(
            "DTYPE MUTATION: PASS"
        )

    else:

        print(
            "DTYPE MUTATION: FAIL"
        )

        print(
            "Before:",
            original_dtypes
        )

        print(
            "After:",
            after_dtypes
        )

    # ==========================================================
    # TEST B
    # MARKET STRUCTURE DIRECT DETERMINISM
    # ==========================================================

    print_section(
        "3. DIRECT MARKET STRUCTURE DETERMINISM"
    )

    target_position = (
        stable_data.index.get_loc(
            pd.Timestamp(CHECK_DATE)
        )
    )

    structure_history = stable_data.iloc[
        :target_position + 1
    ].copy(
        deep=True
    )

    print(
        f"Structure history rows: {len(structure_history)}"
    )

    structure_run_1 = calculate_market_structure(
        structure_history.copy(deep=True),
        swing_window=3
    )

    structure_run_2 = calculate_market_structure(
        structure_history.copy(deep=True),
        swing_window=3
    )

    structure_fields = [
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

    structure_deterministic = True

    for field in structure_fields:

        first = structure_run_1.get(field)
        second = structure_run_2.get(field)

        if first != second:

            structure_deterministic = False

        compare_values(
            f"Market structure field: {field}",
            first,
            second
        )

    # ==========================================================
    # TEST C
    # COMPLETE SIGNAL GENERATION DETERMINISM
    # ==========================================================

    print_section(
        "4. COMPLETE SIGNAL GENERATION DETERMINISM"
    )

    # IMPORTANT:
    # Use two completely independent deep copies.
    # This eliminates shared dataframe state as a possibility.

    input_run_1 = input_before.copy(
        deep=True
    )

    input_run_2 = input_before.copy(
        deep=True
    )

    signals_1 = (
        service.generate_tradesense_signals(
            df=input_run_1
        )
    )

    signals_2 = (
        service.generate_tradesense_signals(
            df=input_run_2
        )
    )

    comparison_dates = (
        signals_1.index.intersection(
            signals_2.index
        )
    )

    print(
        f"Common rows: {len(comparison_dates)}"
    )

    comparison_fields = [
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

    differing_rows = 0

    field_difference_counts = {
        field: 0
        for field in comparison_fields
    }

    first_difference = None

    for date in comparison_dates:

        row_1 = signals_1.loc[date]
        row_2 = signals_2.loc[date]

        row_different = False

        for field in comparison_fields:

            value_1 = row_1[field]
            value_2 = row_2[field]

            if pd.isna(value_1) and pd.isna(value_2):

                continue

            if value_1 != value_2:

                field_difference_counts[
                    field
                ] += 1

                row_different = True

                if first_difference is None:

                    first_difference = {
                        "date": date,
                        "field": field,
                        "run_1": value_1,
                        "run_2": value_2,
                    }

        if row_different:

            differing_rows += 1

    match_rate = (
        (
            len(comparison_dates)
            - differing_rows
        )
        / len(comparison_dates)
        * 100
        if len(comparison_dates) > 0
        else 0
    )

    print()
    print(
        f"Differing rows: {differing_rows}"
    )

    print(
        f"Match rate: {match_rate:.2f}%"
    )

    print()
    print(
        "FIELD DIFFERENCE COUNTS:"
    )

    for field, count in (
        field_difference_counts.items()
    ):

        if count > 0:

            print(
                f"  {field}: {count}"
            )

    if first_difference:

        print()
        print(
            "FIRST DIFFERENCE:"
        )

        print(
            f"  Date   : {first_difference['date']}"
        )

        print(
            f"  Field  : {first_difference['field']}"
        )

        print(
            f"  Run 1  : {first_difference['run_1']}"
        )

        print(
            f"  Run 2  : {first_difference['run_2']}"
        )

    # ==========================================================
    # FINAL VERDICT
    # ==========================================================

    print_section(
        "5. FINAL VERDICT"
    )

    print(
        f"Direct market_structure deterministic: "
        f"{'PASS' if structure_deterministic else 'FAIL'}"
    )

    print(
        f"Complete signal generation deterministic: "
        f"{'PASS' if differing_rows == 0 else 'FAIL'}"
    )

    print(
        f"Input dataframe mutation: "
        f"{'PASS' if input_before_hash == input_after_hash else 'FAIL'}"
    )

    print()

    if (
        structure_deterministic
        and differing_rows > 0
        and input_before_hash == input_after_hash
    ):

        print(
            "RESULT: NON-DETERMINISM IS INSIDE THE SIGNAL-GENERATION PIPELINE."
        )

        print(
            "Market structure itself is deterministic."
        )

        print(
            "The next investigation must isolate trend/liquidity/setup/event state."
        )

    elif not structure_deterministic:

        print(
            "RESULT: MARKET STRUCTURE ENGINE IS NON-DETERMINISTIC."
        )

        print(
            "DO NOT MODIFY PRODUCTION YET."
        )

    elif input_before_hash != input_after_hash:

        print(
            "RESULT: INPUT DATAFRAME IS BEING MUTATED."
        )

        print(
            "DO NOT MODIFY PRODUCTION YET."
        )

    else:

        print(
            "RESULT: SIGNAL GENERATION IS DETERMINISTIC."
        )

        print(
            "Proceed to warm-up boundary analysis."
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()