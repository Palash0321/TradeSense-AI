"""
TradeSense-AI
TCS Structural Event State Forensic

Purpose
-------
Compare the stateful structural-event logic used by
BacktestService.generate_tradesense_signals() under:

1. Stable research context:
   2021-01-01 -> 2026-09-11

2. Rolling 5-year context:
   2021-09-13 -> 2026-09-11

This diagnostic specifically traces:

    structural_event
    last_structural_event
    new_structural_event
    setup
    setup_direction
    signal

at selected TCS dates.

IMPORTANT
---------
Diagnostic only.

Do NOT modify:
- production code
- frozen historical dataset
- research candidate dataset
"""

import os
import csv
import pandas as pd

from app.services.backtest_service import BacktestService

from app.core.levels.market_structure import (
    calculate_market_structure
)

from app.core.liquidity.liquidity import (
    calculate_liquidity
)

from app.core.setup.setup_engine import (
    calculate_setup
)


SYMBOL = "TCS.NS"

STABLE_START = "2021-01-01"
ROLLING_START = "2021-09-13"

OUTPUT_FILE = (
    "tests/output/reconciliation/"
    "tcs_structural_event_state_forensic.csv"
)

SELECTED_DATES = [
    "2021-10-11",
    "2021-10-12",
    "2021-10-20",

    "2022-02-22",
    "2022-02-23",

    "2022-07-14",
    "2022-07-15",

    "2022-08-11",
    "2022-08-12",

    "2022-09-16",
    "2022-09-19",

    "2022-11-11",
    "2022-11-14",

    "2023-03-31",
    "2023-04-03",

    "2023-10-03",
    "2023-10-04",

    "2023-10-19",
    "2023-10-20",

    "2024-01-04",
]


def load_context(start_date):
    """
    Load TCS data using the same BacktestService data loader
    used by the research pipeline.
    """

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    df = service.load_data(
        start_date=start_date
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {SYMBOL} "
            f"starting {start_date}"
        )

    return df


def trace_structural_events(df):
    """
    Reproduce the stateful structural-event section of
    generate_tradesense_signals() for diagnostic purposes.

    We intentionally trace only the relevant state.

    The production function:
        - walks forward candle by candle
        - supplies history only through current candle
        - calculates market structure
        - creates structural_event
        - compares it with last_structural_event
        - calculates setup
        - generates signal
        - updates last_structural_event
    """

    result = df.copy()

    minimum_history = (
        3 * 2 + 1
    )

    last_structural_event = None

    rows = []

    for i in range(
        minimum_history - 1,
        len(result)
    ):

        current_date = result.index[i]

        history = result.iloc[
            :i + 1
        ]

        current_price = float(
            history["Close"].iloc[-1]
        )

        # --------------------------------------------------
        # Market Structure
        # --------------------------------------------------

        market_structure = (
            calculate_market_structure(
                history,
                swing_window=3
            )
        )

        # --------------------------------------------------
        # Liquidity
        # --------------------------------------------------

        liquidity = calculate_liquidity(
            history,
            market_structure
        )

        # --------------------------------------------------
        # Extract structural state
        # --------------------------------------------------

        break_confirmed = bool(
            market_structure.get(
                "break_confirmed",
                False
            )
        )

        break_direction = (
            market_structure.get(
                "break_direction"
            )
        )

        break_level = (
            market_structure.get(
                "break_level"
            )
        )

        break_date = (
            market_structure.get(
                "break_date"
            )
        )

        # --------------------------------------------------
        # Exact production structural-event construction
        # --------------------------------------------------

        structural_event = None

        if (
            break_confirmed
            and
            break_direction is not None
            and
            break_level is not None
            and
            break_date is not None
        ):

            structural_event = (
                break_direction,
                round(
                    float(break_level),
                    2
                ),
                str(break_date)
            )

        # --------------------------------------------------
        # Exact production new-event test
        # --------------------------------------------------

        previous_last_event = (
            last_structural_event
        )

        new_structural_event = (
            structural_event is not None
            and
            structural_event
            != last_structural_event
        )

        # --------------------------------------------------
        # Setup
        # --------------------------------------------------

        setup = calculate_setup(
            market_structure=market_structure,
            liquidity=liquidity,
            current_price=current_price
        )

        setup_type = setup.get(
            "setup",
            "WAIT"
        )

        setup_direction = setup.get(
            "direction"
        )

        setup_confidence = float(
            setup.get(
                "confidence",
                0
            )
            or 0
        )

        # --------------------------------------------------
        # Signal logic
        # --------------------------------------------------

        signal = 0

        structural_setup = (
            setup_type in [
                "LONG_CONTINUATION",
                "SHORT_CONTINUATION",
                "LONG_REVERSAL",
                "SHORT_REVERSAL"
            ]
        )

        if (
            structural_setup
            and
            new_structural_event
        ):

            if setup_direction == "LONG":
                signal = 1

            elif setup_direction == "SHORT":
                signal = -1

        elif (
            not structural_setup
            and
            setup_type in [
                "LONG_REVERSAL",
                "SHORT_REVERSAL"
            ]
        ):

            if setup_direction == "LONG":
                signal = 1

            elif setup_direction == "SHORT":
                signal = -1

        # --------------------------------------------------
        # Exact production event-memory update
        # --------------------------------------------------

        if new_structural_event:

            last_structural_event = (
                structural_event
            )

        # --------------------------------------------------
        # Record every selected date only
        # --------------------------------------------------

        date_string = (
            pd.Timestamp(current_date)
            .strftime("%Y-%m-%d")
        )

        if date_string in SELECTED_DATES:

            rows.append(
                {
                    "date": date_string,

                    "previous_last_structural_event":
                        repr(previous_last_event),

                    "structural_event":
                        repr(structural_event),

                    "new_structural_event":
                        bool(new_structural_event),

                    "last_structural_event_after":
                        repr(last_structural_event),

                    "break_direction":
                        break_direction,

                    "break_level":
                        break_level,

                    "break_date":
                        str(break_date)
                        if break_date is not None
                        else None,

                    "break_confirmed":
                        break_confirmed,

                    "BOS":
                        market_structure.get(
                            "bos"
                        ),

                    "CHOCH":
                        market_structure.get(
                            "choch"
                        ),

                    "Structure_Bias":
                        market_structure.get(
                            "bias"
                        ),

                    "Structure":
                        market_structure.get(
                            "structure"
                        ),

                    "Break_Age":
                        market_structure.get(
                            "break_age"
                        ),

                    "Setup":
                        setup_type,

                    "Setup_Direction":
                        setup_direction,

                    "Setup_Confidence":
                        setup_confidence,

                    "Signal":
                        signal,
                }
            )

    return rows


def main():

    print()
    print("=" * 110)
    print("TCS STRUCTURAL EVENT STATE FORENSIC")
    print("=" * 110)

    print()
    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Stable context: {STABLE_START}"
    )

    print(
        f"Rolling context: {ROLLING_START}"
    )

    print()
    print(
        "Loading stable context..."
    )

    stable_df = load_context(
        STABLE_START
    )

    print(
        f"Stable rows: {len(stable_df)}"
    )

    print(
        f"Stable range: "
        f"{stable_df.index.min().date()} -> "
        f"{stable_df.index.max().date()}"
    )

    print()
    print(
        "Loading rolling context..."
    )

    rolling_df = load_context(
        ROLLING_START
    )

    print(
        f"Rolling rows: {len(rolling_df)}"
    )

    print(
        f"Rolling range: "
        f"{rolling_df.index.min().date()} -> "
        f"{rolling_df.index.max().date()}"
    )

    print()
    print(
        "Tracing stable structural-event state..."
    )

    stable_rows = trace_structural_events(
        stable_df
    )

    print(
        f"Stable selected rows: "
        f"{len(stable_rows)}"
    )

    print()
    print(
        "Tracing rolling structural-event state..."
    )

    rolling_rows = trace_structural_events(
        rolling_df
    )

    print(
        f"Rolling selected rows: "
        f"{len(rolling_rows)}"
    )

    stable_map = {
        row["date"]: row
        for row in stable_rows
    }

    rolling_map = {
        row["date"]: row
        for row in rolling_rows
    }

    comparison_rows = []

    state_difference_count = 0
    event_difference_count = 0
    signal_difference_count = 0

    print()
    print("=" * 110)
    print("CHRONOLOGICAL COMPARISON")
    print("=" * 110)

    for date in SELECTED_DATES:

        stable = stable_map.get(date)
        rolling = rolling_map.get(date)

        if stable is None or rolling is None:
            print()
            print(
                f"{date}: "
                f"MISSING CONTEXT ROW"
            )
            continue

        state_fields = [
            "previous_last_structural_event",
            "structural_event",
            "new_structural_event",
            "last_structural_event_after",
            "break_direction",
            "break_level",
            "break_date",
            "break_confirmed",
            "BOS",
            "CHOCH",
            "Structure_Bias",
            "Structure",
            "Break_Age",
            "Setup",
            "Setup_Direction",
            "Setup_Confidence",
            "Signal",
        ]

        differences = []

        for field in state_fields:

            if stable[field] != rolling[field]:

                differences.append(
                    field
                )

        if differences:
            state_difference_count += 1

        if (
            stable["structural_event"]
            !=
            rolling["structural_event"]
        ):
            event_difference_count += 1

        if (
            stable["Signal"]
            !=
            rolling["Signal"]
        ):
            signal_difference_count += 1

        print()
        print("-" * 110)

        print(
            f"DATE: {date}"
        )

        print()

        print(
            f"Stable previous_last_event : "
            f"{stable['previous_last_structural_event']}"
        )

        print(
            f"Rolling previous_last_event: "
            f"{rolling['previous_last_structural_event']}"
        )

        print()

        print(
            f"Stable structural_event : "
            f"{stable['structural_event']}"
        )

        print(
            f"Rolling structural_event: "
            f"{rolling['structural_event']}"
        )

        print()

        print(
            f"Stable new_event : "
            f"{stable['new_structural_event']}"
        )

        print(
            f"Rolling new_event: "
            f"{rolling['new_structural_event']}"
        )

        print()

        print(
            f"Stable Setup : "
            f"{stable['Setup']}"
        )

        print(
            f"Rolling Setup: "
            f"{rolling['Setup']}"
        )

        print()

        print(
            f"Stable Signal : "
            f"{stable['Signal']}"
        )

        print(
            f"Rolling Signal: "
            f"{rolling['Signal']}"
        )

        print()

        if differences:

            print(
                "DIFFERENCES:"
            )

            for field in differences:

                print(
                    f"  {field}: "
                    f"{stable[field]!r} "
                    f"-> "
                    f"{rolling[field]!r}"
                )

        else:

            print(
                "STATE DIFFERENCES: NONE"
            )

        comparison_rows.append(
            {
                "date": date,

                "stable_previous_last_event":
                    stable["previous_last_structural_event"],

                "rolling_previous_last_event":
                    rolling["previous_last_structural_event"],

                "stable_structural_event":
                    stable["structural_event"],

                "rolling_structural_event":
                    rolling["structural_event"],

                "stable_new_event":
                    stable["new_structural_event"],

                "rolling_new_event":
                    rolling["new_structural_event"],

                "stable_setup":
                    stable["Setup"],

                "rolling_setup":
                    rolling["Setup"],

                "stable_direction":
                    stable["Setup_Direction"],

                "rolling_direction":
                    rolling["Setup_Direction"],

                "stable_signal":
                    stable["Signal"],

                "rolling_signal":
                    rolling["Signal"],

                "state_differences":
                    " | ".join(differences),
            }
        )

    # ------------------------------------------------------
    # Save forensic output
    # ------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
        exist_ok=True
    )

    fieldnames = [
        "date",

        "stable_previous_last_event",
        "rolling_previous_last_event",

        "stable_structural_event",
        "rolling_structural_event",

        "stable_new_event",
        "rolling_new_event",

        "stable_setup",
        "rolling_setup",

        "stable_direction",
        "rolling_direction",

        "stable_signal",
        "rolling_signal",

        "state_differences",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    print()
    print("=" * 110)
    print("FORENSIC SUMMARY")
    print("=" * 110)

    print(
        f"Dates analysed: "
        f"{len(comparison_rows)}"
    )

    print(
        f"Dates with state differences: "
        f"{state_difference_count}"
    )

    print(
        f"Dates with structural-event differences: "
        f"{event_difference_count}"
    )

    print(
        f"Dates with signal differences: "
        f"{signal_difference_count}"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("=" * 110)

    if event_difference_count > 0:

        print(
            "RESULT: STRUCTURAL EVENT STATE DIFFERS"
        )

    elif signal_difference_count > 0:

        print(
            "RESULT: SIGNAL DIFFERS WITHOUT EVENT DIFF"
        )

    else:

        print(
            "RESULT: NO STRUCTURAL EVENT DIFFERENCE "
            "AT SELECTED DATES"
        )

    print("=" * 110)


if __name__ == "__main__":
    main()