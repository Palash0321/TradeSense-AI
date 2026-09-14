"""
TCS chronological market-structure event forensic.

Purpose:
    Compare the production market-structure engine when it receives:

        A) Stable history starting 2021-01-01
        B) Rolling history starting 2021-09-13

    at exactly the same evaluation dates.

    The goal is to identify exactly which internal market-structure
    components differ and why.

IMPORTANT:
    Diagnostic only.
    Does NOT modify production code.
    Does NOT modify the frozen historical trade dataset.
"""

from pathlib import Path

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure


SYMBOL = "TCS.NS"

STABLE_START = "2021-01-01"
ROLLING_START = "2021-09-13"

DATES_TO_INSPECT = [
    "2021-09-13",
    "2021-09-16",
    "2021-09-30",
    "2021-10-11",
    "2021-10-12",
    "2021-10-20",
    "2022-02-22",
    "2022-02-23",
    "2022-07-14",
    "2022-07-15",
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

OUTPUT_FILE = Path(
    "tests/output/reconciliation/tcs_structure_events_forensic.csv"
)


def load_data(start_date):
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
            f"No data returned from {start_date}"
        )

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def format_date(value):
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def normalize_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value


def extract_swings(result, key):
    swings = result.get(key, [])

    output = []

    for swing in swings:
        if isinstance(swing, dict):
            output.append(
                {
                    "date": format_date(
                        swing.get("index")
                    ),
                    "price": swing.get("price"),
                }
            )
        else:
            output.append(
                {
                    "date": format_date(swing),
                    "price": None,
                }
            )

    return output


def print_swings(label, swings):
    print()
    print(label)

    if not swings:
        print("  NONE")
        return

    # Show the most recent 5 swings because these are the
    # ones directly involved in the production calculations.
    for swing in swings[-5:]:
        print(
            f"  {swing['date']} "
            f"@ {swing['price']}"
        )


def get_state(result):
    return {
        "bias": normalize_value(
            result.get("bias")
        ),
        "structure": normalize_value(
            result.get("structure")
        ),
        "bos": normalize_value(
            result.get("bos")
        ),
        "choch": normalize_value(
            result.get("choch")
        ),
        "break_direction": normalize_value(
            result.get("break_direction")
        ),
        "break_level": normalize_value(
            result.get("break_level")
        ),
        "break_date": normalize_value(
            result.get("break_date")
        ),
        "break_confirmed": normalize_value(
            result.get("break_confirmed")
        ),
        "break_age": normalize_value(
            result.get("break_age")
        ),
        "strength": normalize_value(
            result.get("strength")
        ),
    }


def compare_state(stable_state, rolling_state):
    differences = {}

    for key in stable_state:
        if (
            stable_state[key]
            != rolling_state[key]
        ):
            differences[key] = (
                stable_state[key],
                rolling_state[key],
            )

    return differences


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 110)
    print("TCS CHRONOLOGICAL MARKET-STRUCTURE EVENT FORENSIC")
    print("=" * 110)
    print()

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Stable context: {STABLE_START} -> present"
    )

    print(
        f"Rolling context: {ROLLING_START} -> present"
    )

    print()

    stable_df = load_data(
        STABLE_START
    )

    rolling_df = load_data(
        ROLLING_START
    )

    rows = []

    for date_string in DATES_TO_INSPECT:

        current_date = pd.Timestamp(
            date_string
        )

        if current_date not in stable_df.index:
            print(
                f"SKIP {date_string}: "
                "not available in stable data"
            )
            continue

        if current_date not in rolling_df.index:
            print(
                f"SKIP {date_string}: "
                "not available in rolling data"
            )
            continue

        stable_prefix = stable_df.loc[
            :current_date
        ]

        rolling_prefix = rolling_df.loc[
            :current_date
        ]

        stable_result = calculate_market_structure(
            stable_prefix
        )

        rolling_result = calculate_market_structure(
            rolling_prefix
        )

        stable_state = get_state(
            stable_result
        )

        rolling_state = get_state(
            rolling_result
        )

        differences = compare_state(
            stable_state,
            rolling_state
        )

        stable_highs = extract_swings(
            stable_result,
            "swing_highs"
        )

        rolling_highs = extract_swings(
            rolling_result,
            "swing_highs"
        )

        stable_lows = extract_swings(
            stable_result,
            "swing_lows"
        )

        rolling_lows = extract_swings(
            rolling_result,
            "swing_lows"
        )

        # Compare overlapping swing identities.
        stable_high_dates = [
            item["date"]
            for item in stable_highs
        ]

        rolling_high_dates = [
            item["date"]
            for item in rolling_highs
        ]

        stable_low_dates = [
            item["date"]
            for item in stable_lows
        ]

        rolling_low_dates = [
            item["date"]
            for item in rolling_lows
        ]

        high_dates_match = (
            stable_high_dates
            == rolling_high_dates
        )

        low_dates_match = (
            stable_low_dates
            == rolling_low_dates
        )

        print()
        print("=" * 110)
        print(
            f"DATE: {date_string}"
        )
        print("=" * 110)

        print()
        print("STABLE")

        for key, value in stable_state.items():
            print(
                f"  {key}: {value}"
            )

        print_swings(
            "  Recent swing highs:",
            stable_highs
        )

        print_swings(
            "  Recent swing lows:",
            stable_lows
        )

        print()
        print("ROLLING")

        for key, value in rolling_state.items():
            print(
                f"  {key}: {value}"
            )

        print_swings(
            "  Recent swing highs:",
            rolling_highs
        )

        print_swings(
            "  Recent swing lows:",
            rolling_lows
        )

        print()
        print("COMPARISON")

        print(
            "  Swing-high date sequence identical:",
            high_dates_match
        )

        print(
            "  Swing-low date sequence identical:",
            low_dates_match
        )

        if differences:

            print(
                "  STATE DIFFERENCES:"
            )

            for key, values in differences.items():
                print(
                    f"    {key}: "
                    f"{values[0]} -> {values[1]}"
                )

        else:

            print(
                "  STATE DIFFERENCES: NONE"
            )

        rows.append(
            {
                "symbol": SYMBOL,
                "date": date_string,

                "stable_swing_high_count": len(
                    stable_highs
                ),

                "rolling_swing_high_count": len(
                    rolling_highs
                ),

                "stable_swing_low_count": len(
                    stable_lows
                ),

                "rolling_swing_low_count": len(
                    rolling_lows
                ),

                "swing_high_dates_match": (
                    high_dates_match
                ),

                "swing_low_dates_match": (
                    low_dates_match
                ),

                "state_different": bool(
                    differences
                ),

                "different_fields": "|".join(
                    differences.keys()
                ),

                "stable_bias": stable_state["bias"],
                "rolling_bias": rolling_state["bias"],

                "stable_structure": stable_state[
                    "structure"
                ],
                "rolling_structure": rolling_state[
                    "structure"
                ],

                "stable_bos": stable_state["bos"],
                "rolling_bos": rolling_state["bos"],

                "stable_choch": stable_state["choch"],
                "rolling_choch": rolling_state["choch"],

                "stable_break_direction": stable_state[
                    "break_direction"
                ],
                "rolling_break_direction": rolling_state[
                    "break_direction"
                ],

                "stable_break_level": stable_state[
                    "break_level"
                ],
                "rolling_break_level": rolling_state[
                    "break_level"
                ],

                "stable_break_date": stable_state[
                    "break_date"
                ],
                "rolling_break_date": rolling_state[
                    "break_date"
                ],

                "stable_break_confirmed": stable_state[
                    "break_confirmed"
                ],
                "rolling_break_confirmed": rolling_state[
                    "break_confirmed"
                ],

                "stable_break_age": stable_state[
                    "break_age"
                ],
                "rolling_break_age": rolling_state[
                    "break_age"
                ],

                "stable_strength": stable_state[
                    "strength"
                ],
                "rolling_strength": rolling_state[
                    "strength"
                ],
            }
        )

    output_df = pd.DataFrame(
        rows
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 110)
    print("FORENSIC COMPLETE")
    print("=" * 110)

    print(
        f"Dates analysed: {len(output_df)}"
    )

    print(
        f"Dates with state differences: "
        f"{output_df['state_different'].sum()}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print(
        "NO PRODUCTION CODE WAS MODIFIED."
    )

    print(
        "NO FROZEN DATASET WAS MODIFIED."
    )


if __name__ == "__main__":
    main()