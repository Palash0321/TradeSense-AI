"""
Chronological market-structure prefix forensic.

Purpose:
    Determine exactly where stable-history and rolling-5Y market-structure
    states diverge when the overlapping OHLCV data is identical.

IMPORTANT:
    Diagnostic only.
    Does NOT modify production code.
    Does NOT modify the frozen historical trade dataset.
"""

from pathlib import Path

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure


SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

STABLE_START = "2021-01-01"
ROLLING_START = "2021-09-13"

OUTPUT_FILE = Path(
    "tests/output/reconciliation/market_structure_prefix_forensic.csv"
)


STATE_FIELDS = [
    "bias",
    "structure",
    "bos",
    "choch",
    "break_direction",
    "break_level",
    "break_date",
    "break_confirmed",
    "break_age",
    "strength",
]


def load_context(symbol, start_date):
    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    df = service.load_data(start_date=start_date)

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {symbol} from {start_date}"
        )

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def normalize_value(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value


def state_snapshot(result):
    snapshot = {}

    for field in STATE_FIELDS:
        value = result.get(field)

        if isinstance(value, pd.Timestamp):
            value = value.strftime("%Y-%m-%d")

        elif pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
            value = None

        snapshot[field] = normalize_value(value)

    return snapshot


def compare_states(stable_state, rolling_state):
    differences = {}

    for field in STATE_FIELDS:
        stable_value = stable_state.get(field)
        rolling_value = rolling_state.get(field)

        if stable_value != rolling_value:
            differences[field] = (
                stable_value,
                rolling_value,
            )

    return differences


def swing_dates(result, key):
    values = result.get(key, [])

    dates = []

    for value in values:

        if isinstance(value, tuple):
            value = value[0]

        if isinstance(value, pd.Timestamp):
            dates.append(value.strftime("%Y-%m-%d"))

        else:
            try:
                dates.append(
                    pd.Timestamp(value).strftime("%Y-%m-%d")
                )
            except Exception:
                dates.append(str(value))

    return dates


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    print()
    print("=" * 110)
    print("CHRONOLOGICAL MARKET-STRUCTURE PREFIX FORENSIC")
    print("=" * 110)
    print()
    print(
        "Stable context:",
        STABLE_START,
        "→ present"
    )
    print(
        "Rolling context:",
        ROLLING_START,
        "→ present"
    )
    print()

    for symbol in SYMBOLS:

        print()
        print("-" * 110)
        print(symbol)
        print("-" * 110)

        stable_df = load_context(
            symbol,
            STABLE_START
        )

        rolling_df = load_context(
            symbol,
            ROLLING_START
        )

        common_dates = sorted(
            set(stable_df.index)
            & set(rolling_df.index)
        )

        print(
            f"Common dates: {len(common_dates)}"
        )

        first_divergence = None
        first_swing_divergence = None

        state_difference_count = 0
        swing_difference_count = 0

        selected_dates = []

        # Always inspect the beginning of the overlap.
        selected_dates.extend(
            common_dates[:20]
        )

        # Inspect every known divergence-sensitive area
        # plus the first ~120 common dates.
        selected_dates.extend(
            common_dates[20:120]
        )

        # Inspect the entire overlap after that at a reduced cadence.
        selected_dates.extend(
            common_dates[120::20]
        )

        selected_dates = sorted(
            set(selected_dates)
        )

        for current_date in selected_dates:

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

            stable_state = state_snapshot(
                stable_result
            )

            rolling_state = state_snapshot(
                rolling_result
            )

            state_differences = compare_states(
                stable_state,
                rolling_state
            )

            stable_highs = swing_dates(
                stable_result,
                "swing_highs"
            )

            rolling_highs = swing_dates(
                rolling_result,
                "swing_highs"
            )

            stable_lows = swing_dates(
                stable_result,
                "swing_lows"
            )

            rolling_lows = swing_dates(
                rolling_result,
                "swing_lows"
            )

            high_difference = (
                stable_highs != rolling_highs
            )

            low_difference = (
                stable_lows != rolling_lows
            )

            swing_difference = (
                high_difference
                or low_difference
            )

            if state_differences:
                state_difference_count += 1

                if first_divergence is None:
                    first_divergence = current_date

                    print()
                    print(
                        "FIRST STATE DIVERGENCE:"
                    )
                    print(
                        "Date:",
                        current_date.strftime("%Y-%m-%d")
                    )

                    for field, values in state_differences.items():
                        print(
                            f"  {field}: "
                            f"{values[0]} -> {values[1]}"
                        )

            if swing_difference:
                swing_difference_count += 1

                if first_swing_divergence is None:
                    first_swing_divergence = current_date

                    print()
                    print(
                        "FIRST SWING DIVERGENCE:"
                    )
                    print(
                        "Date:",
                        current_date.strftime("%Y-%m-%d")
                    )

                    print(
                        "Stable swing highs:",
                        len(stable_highs)
                    )

                    print(
                        "Rolling swing highs:",
                        len(rolling_highs)
                    )

                    print(
                        "Stable swing lows:",
                        len(stable_lows)
                    )

                    print(
                        "Rolling swing lows:",
                        len(rolling_lows)
                    )

            all_rows.append(
                {
                    "_symbol": symbol,
                    "date": current_date.strftime(
                        "%Y-%m-%d"
                    ),
                    "state_different": bool(
                        state_differences
                    ),
                    "swing_different": bool(
                        swing_difference
                    ),
                    "different_fields": "|".join(
                        state_differences.keys()
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
                }
            )

        print()
        print(
            "First state divergence:",
            (
                first_divergence.strftime("%Y-%m-%d")
                if first_divergence is not None
                else "NONE"
            )
        )

        print(
            "First swing divergence:",
            (
                first_swing_divergence.strftime("%Y-%m-%d")
                if first_swing_divergence is not None
                else "NONE"
            )
        )

        print(
            "Checked prefix dates:",
            len(selected_dates)
        )

        print(
            "State-different prefixes:",
            state_difference_count
        )

        print(
            "Swing-different prefixes:",
            swing_difference_count
        )

    result_df = pd.DataFrame(all_rows)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 110)
    print("FORENSIC COMPLETE")
    print("=" * 110)
    print(
        f"Rows saved: {len(result_df)}"
    )
    print(
        f"File: {OUTPUT_FILE}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This is diagnostic only."
    )
    print(
        "No production strategy code was changed."
    )
    print(
        "The frozen historical dataset was not modified."
    )


if __name__ == "__main__":
    main()