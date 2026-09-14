"""
TradeSense-AI
Internal Market Structure Root-Cause Forensic

Purpose
-------
Directly compare the ACTUAL output of:

app.core.levels.market_structure.calculate_market_structure()

between:

1. Stable research context
   2021-01-01 -> present

2. Rolling production context
   BacktestService.load_data() default

Focus:
    raw OHLCV
        ↓
    swing_highs / swing_lows
        ↓
    structure
        ↓
    BOS / CHOCH / break
        ↓
    bias / strength

Diagnostic only.
DO NOT modify production code or frozen datasets.
"""

from pathlib import Path

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import (
    calculate_market_structure,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SYMBOL = "TCS.NS"

STABLE_START_DATE = "2021-01-01"

OUTPUT_DIR = Path(
    "tests/output/reconciliation"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "internal_market_structure_forensic.csv"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_timestamp(value):
    """Return a timezone-naive pandas Timestamp."""

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp


def normalize_swing_list(swing_list):
    """
    Convert production swing dictionaries into a clean DataFrame.
    """

    if not swing_list:
        return pd.DataFrame(
            columns=[
                "date",
                "price",
            ]
        )

    rows = []

    for swing in swing_list:

        rows.append(
            {
                "date": normalize_timestamp(
                    swing["index"]
                ),
                "price": float(
                    swing["price"]
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = (
        result
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return result


def compare_swing_lists(
    stable_list,
    rolling_list,
    swing_type,
):
    """
    Compare stable and rolling swing points by date.

    Returns:
        list of forensic records
    """

    stable_df = normalize_swing_list(
        stable_list
    )

    rolling_df = normalize_swing_list(
        rolling_list
    )

    stable_map = dict(
        zip(
            stable_df["date"],
            stable_df["price"],
        )
    )

    rolling_map = dict(
        zip(
            rolling_df["date"],
            rolling_df["price"],
        )
    )

    all_dates = sorted(
        set(stable_map)
        | set(rolling_map)
    )

    records = []

    for date in all_dates:

        stable_price = stable_map.get(
            date
        )

        rolling_price = rolling_map.get(
            date
        )

        if (
            stable_price is not None
            and rolling_price is not None
        ):

            if abs(
                stable_price
                - rolling_price
            ) < 1e-9:

                status = "MATCH"

            else:

                status = "PRICE_MISMATCH"

        elif stable_price is not None:

            status = "STABLE_ONLY"

        else:

            status = "ROLLING_ONLY"

        records.append(
            {
                "symbol": SYMBOL,
                "swing_type": swing_type,
                "date": date.strftime(
                    "%Y-%m-%d"
                ),
                "stable_price": stable_price,
                "rolling_price": rolling_price,
                "status": status,
            }
        )

    return records


def get_state_dict(
    structure_result,
):
    """
    Extract non-swing market-structure state.
    """

    return {
        key: structure_result.get(key)
        for key in [
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
    }


def compare_state(
    stable_result,
    rolling_result,
):
    """
    Compare final state returned by market structure.
    """

    stable_state = get_state_dict(
        stable_result
    )

    rolling_state = get_state_dict(
        rolling_result
    )

    differences = {}

    for key in stable_state:

        stable_value = stable_state[key]
        rolling_value = rolling_state[key]

        if pd.isna(stable_value):
            stable_value = None

        if pd.isna(rolling_value):
            rolling_value = None

        if stable_value != rolling_value:

            differences[key] = (
                stable_value,
                rolling_value,
            )

    return differences


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI INTERNAL MARKET STRUCTURE FORENSIC"
    )
    print("=" * 100)

    print()
    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Stable start: {STABLE_START_DATE}"
    )

    print(
        "Rolling context: "
        "BacktestService.load_data() default"
    )

    print()
    print(
        "Production function:"
    )

    print(
        "app.core.levels.market_structure."
        "calculate_market_structure"
    )

    # -----------------------------------------------------------------
    # LOAD STABLE DATA
    # -----------------------------------------------------------------

    stable_service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    stable_raw = stable_service.load_data(
        start_date=STABLE_START_DATE
    )

    # -----------------------------------------------------------------
    # LOAD ROLLING DATA
    # -----------------------------------------------------------------

    rolling_service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    rolling_raw = rolling_service.load_data()

    print()
    print("=" * 100)
    print("RAW DATA COVERAGE")
    print("=" * 100)

    print(
        "Stable:",
        stable_raw.index.min(),
        "→",
        stable_raw.index.max(),
        "rows:",
        len(stable_raw),
    )

    print(
        "Rolling:",
        rolling_raw.index.min(),
        "→",
        rolling_raw.index.max(),
        "rows:",
        len(rolling_raw),
    )

    # -----------------------------------------------------------------
    # RUN ACTUAL PRODUCTION MARKET STRUCTURE
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "RUNNING ACTUAL MARKET STRUCTURE"
    )
    print("=" * 100)

    stable_result = calculate_market_structure(
        stable_raw.copy()
    )

    rolling_result = calculate_market_structure(
        rolling_raw.copy()
    )

    # -----------------------------------------------------------------
    # BASIC RESULT INFORMATION
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("STABLE MARKET STRUCTURE RESULT")
    print("=" * 100)

    print(
        "Bias:",
        stable_result.get("bias")
    )

    print(
        "Structure:",
        stable_result.get("structure")
    )

    print(
        "Swing highs:",
        len(
            stable_result.get(
                "swing_highs",
                [],
            )
        ),
    )

    print(
        "Swing lows:",
        len(
            stable_result.get(
                "swing_lows",
                [],
            )
        ),
    )

    print(
        "BOS:",
        stable_result.get("bos")
    )

    print(
        "CHOCH:",
        stable_result.get("choch")
    )

    print(
        "Break direction:",
        stable_result.get(
            "break_direction"
        ),
    )

    print(
        "Break confirmed:",
        stable_result.get(
            "break_confirmed"
        ),
    )

    print(
        "Break age:",
        stable_result.get(
            "break_age"
        ),
    )

    print(
        "Strength:",
        stable_result.get(
            "strength"
        ),
    )

    print()
    print("=" * 100)
    print("ROLLING MARKET STRUCTURE RESULT")
    print("=" * 100)

    print(
        "Bias:",
        rolling_result.get("bias")
    )

    print(
        "Structure:",
        rolling_result.get("structure")
    )

    print(
        "Swing highs:",
        len(
            rolling_result.get(
                "swing_highs",
                [],
            )
        ),
    )

    print(
        "Swing lows:",
        len(
            rolling_result.get(
                "swing_lows",
                [],
            )
        ),
    )

    print(
        "BOS:",
        rolling_result.get("bos")
    )

    print(
        "CHOCH:",
        rolling_result.get("choch")
    )

    print(
        "Break direction:",
        rolling_result.get(
            "break_direction"
        ),
    )

    print(
        "Break confirmed:",
        rolling_result.get(
            "break_confirmed"
        ),
    )

    print(
        "Break age:",
        rolling_result.get(
            "break_age"
        ),
    )

    print(
        "Strength:",
        rolling_result.get(
            "strength"
        ),
    )

    # -----------------------------------------------------------------
    # SWING FORENSICS
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "SWING DETECTION FORENSIC"
    )
    print("=" * 100)

    stable_highs = normalize_swing_list(
        stable_result.get(
            "swing_highs",
            [],
        )
    )

    rolling_highs = normalize_swing_list(
        rolling_result.get(
            "swing_highs",
            [],
        )
    )

    stable_lows = normalize_swing_list(
        stable_result.get(
            "swing_lows",
            [],
        )
    )

    rolling_lows = normalize_swing_list(
        rolling_result.get(
            "swing_lows",
            [],
        )
    )

    # -----------------------------------------------------------------
    # OVERLAPPING PERIOD
    # -----------------------------------------------------------------

    overlap_start = max(
        stable_raw.index.min(),
        rolling_raw.index.min(),
    )

    overlap_end = min(
        stable_raw.index.max(),
        rolling_raw.index.max(),
    )

    overlap_start = normalize_timestamp(
        overlap_start
    )

    overlap_end = normalize_timestamp(
        overlap_end
    )

    print()
    print(
        "Overlap:",
        overlap_start.strftime(
            "%Y-%m-%d"
        ),
        "→",
        overlap_end.strftime(
            "%Y-%m-%d"
        ),
    )

    stable_highs_overlap = stable_highs[
        (
            stable_highs["date"]
            >= overlap_start
        )
        & (
            stable_highs["date"]
            <= overlap_end
        )
    ]

    rolling_highs_overlap = rolling_highs[
        (
            rolling_highs["date"]
            >= overlap_start
        )
        & (
            rolling_highs["date"]
            <= overlap_end
        )
    ]

    stable_lows_overlap = stable_lows[
        (
            stable_lows["date"]
            >= overlap_start
        )
        & (
            stable_lows["date"]
            <= overlap_end
        )
    ]

    rolling_lows_overlap = rolling_lows[
        (
            rolling_lows["date"]
            >= overlap_start
        )
        & (
            rolling_lows["date"]
            <= overlap_end
        )
    ]

    # -----------------------------------------------------------------
    # SET COMPARISON
    # -----------------------------------------------------------------

    def swing_map(df):

        return {
            row["date"]: row["price"]
            for _, row in df.iterrows()
        }

    stable_high_map = swing_map(
        stable_highs_overlap
    )

    rolling_high_map = swing_map(
        rolling_highs_overlap
    )

    stable_low_map = swing_map(
        stable_lows_overlap
    )

    rolling_low_map = swing_map(
        rolling_lows_overlap
    )

    high_dates = sorted(
        set(stable_high_map)
        | set(rolling_high_map)
    )

    low_dates = sorted(
        set(stable_low_map)
        | set(rolling_low_map)
    )

    high_differences = []

    for date in high_dates:

        stable_price = stable_high_map.get(
            date
        )

        rolling_price = rolling_high_map.get(
            date
        )

        if (
            stable_price is None
            or rolling_price is None
            or abs(
                stable_price
                - rolling_price
            ) > 1e-9
        ):

            high_differences.append(
                (
                    date,
                    stable_price,
                    rolling_price,
                )
            )

    low_differences = []

    for date in low_dates:

        stable_price = stable_low_map.get(
            date
        )

        rolling_price = rolling_low_map.get(
            date
        )

        if (
            stable_price is None
            or rolling_price is None
            or abs(
                stable_price
                - rolling_price
            ) > 1e-9
        ):

            low_differences.append(
                (
                    date,
                    stable_price,
                    rolling_price,
                )
            )

    # -----------------------------------------------------------------
    # PRINT SWING RESULTS
    # -----------------------------------------------------------------

    print()
    print(
        "Stable overlapping swing highs:",
        len(stable_high_map),
    )

    print(
        "Rolling overlapping swing highs:",
        len(rolling_high_map),
    )

    print(
        "Different swing-high dates:",
        len(high_differences),
    )

    print()
    print(
        "Stable overlapping swing lows:",
        len(stable_low_map),
    )

    print(
        "Rolling overlapping swing lows:",
        len(rolling_low_map),
    )

    print(
        "Different swing-low dates:",
        len(low_differences),
    )

    # -----------------------------------------------------------------
    # FIRST SWING DIVERGENCE
    # -----------------------------------------------------------------

    all_swing_differences = []

    for date, stable_price, rolling_price in (
        high_differences
    ):

        all_swing_differences.append(
            (
                date,
                "HIGH",
                stable_price,
                rolling_price,
            )
        )

    for date, stable_price, rolling_price in (
        low_differences
    ):

        all_swing_differences.append(
            (
                date,
                "LOW",
                stable_price,
                rolling_price,
            )
        )

    all_swing_differences.sort(
        key=lambda item: item[0]
    )

    first_swing_divergence = (
        all_swing_differences[0]
        if all_swing_differences
        else None
    )

    print()
    print("=" * 100)
    print(
        "FIRST SWING DIVERGENCE"
    )
    print("=" * 100)

    if first_swing_divergence:

        date, swing_type, stable_price, rolling_price = (
            first_swing_divergence
        )

        print(
            "Date:",
            date.strftime(
                "%Y-%m-%d"
            ),
        )

        print(
            "Type:",
            swing_type,
        )

        print(
            "Stable price:",
            stable_price,
        )

        print(
            "Rolling price:",
            rolling_price,
        )

    else:

        print(
            "NO SWING DIVERGENCE FOUND "
            "IN OVERLAPPING PERIOD."
        )

    # -----------------------------------------------------------------
    # PRINT FIRST 20 SWING DIFFERENCES
    # -----------------------------------------------------------------

    if all_swing_differences:

        print()
        print("=" * 100)
        print(
            "FIRST 20 SWING DIFFERENCES"
        )
        print("=" * 100)

        for (
            date,
            swing_type,
            stable_price,
            rolling_price,
        ) in all_swing_differences[:20]:

            print(
                date.strftime(
                    "%Y-%m-%d"
                ),
                "|",
                swing_type,
                "| stable =",
                stable_price,
                "| rolling =",
                rolling_price,
            )

    # -----------------------------------------------------------------
    # FINAL STATE COMPARISON
    # -----------------------------------------------------------------

    state_differences = compare_state(
        stable_result,
        rolling_result,
    )

    print()
    print("=" * 100)
    print(
        "FINAL MARKET STRUCTURE STATE COMPARISON"
    )
    print("=" * 100)

    if state_differences:

        for key, values in (
            state_differences.items()
        ):

            print()
            print(
                key,
                ":",
            )

            print(
                "  Stable :",
                values[0],
            )

            print(
                "  Rolling:",
                values[1],
            )

    else:

        print(
            "FINAL STATE MATCHES."
        )

    # -----------------------------------------------------------------
    # SAVE DETAILED SWING FORENSICS
    # -----------------------------------------------------------------

    records = []

    records.extend(
        compare_swing_lists(
            stable_result.get(
                "swing_highs",
                [],
            ),
            rolling_result.get(
                "swing_highs",
                [],
            ),
            "HIGH",
        )
    )

    records.extend(
        compare_swing_lists(
            stable_result.get(
                "swing_lows",
                [],
            ),
            rolling_result.get(
                "swing_lows",
                [],
            ),
            "LOW",
        )
    )

    forensic_df = pd.DataFrame(
        records
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forensic_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------------------
    # FINAL CONCLUSION
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "FORENSIC CONCLUSION"
    )
    print("=" * 100)

    if all_swing_differences:

        print(
            "RESULT: INTERNAL SWING DETECTION "
            "DIFFERS BETWEEN CONTEXTS."
        )

        print()
        print(
            "This means the divergence exists "
            "at or before swing detection."
        )

        print()
        print(
            "Next investigation:"
        )

        print(
            "Trace the swing algorithm and determine "
            "why identical overlapping OHLC produces "
            "different swing history."
        )

    else:

        print(
            "RESULT: SWING POINTS MATCH "
            "ACROSS THE OVERLAPPING PERIOD."
        )

        print()
        print(
            "The divergence is therefore downstream "
            "of swing detection."
        )

    print()
    print(
        f"Detailed output saved to: {OUTPUT_FILE}"
    )

    print()
    print(
        "Diagnostic complete."
    )


if __name__ == "__main__":
    main()