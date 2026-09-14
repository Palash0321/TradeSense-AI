"""
TradeSense-AI
Setup Input Forensic

Purpose
-------
Compare the exact inputs and outputs of calculate_setup()
between:

    Stable context:
        2021-01-01 -> 2026-09-11

    Rolling context:
        2021-09-13 -> 2026-09-11

Focus:
    Divergent TCS trade identities.

This is DIAGNOSTIC ONLY.

It must NOT:
- modify production code
- modify frozen historical trades
- modify the research candidate
- modify strategy logic
"""

from pathlib import Path
import math

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure
from app.core.liquidity.liquidity import calculate_liquidity
from app.core.setup.setup_engine import calculate_setup


SYMBOL = "TCS.NS"

STABLE_START = "2021-01-01"
ROLLING_START = "2021-09-13"

HISTORICAL_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

CANDIDATE_FILE = Path(
    "tests/output/tradesense_trades_research_candidate.csv"
)

OUTPUT_FILE = Path(
    "tests/output/reconciliation/setup_inputs_forensic.csv"
)


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def normalize_value(value):
    """
    Make nested diagnostic values CSV-safe and comparable.
    """

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value):
            return None

        return round(value, 8)

    if isinstance(value, dict):

        return {
            str(k): normalize_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            normalize_value(v)
            for v in value
        ]

    return value


def values_equal(a, b):
    """
    Robust equality for nested dictionaries/lists/scalars.
    """

    a = normalize_value(a)
    b = normalize_value(b)

    return a == b


def load_price_history(service, start_date):
    """
    Load the same historical data used by the research
    diagnostics.
    """

    return service.load_data(
        start_date=start_date
    )


def get_row_by_date(df, date_string):
    """
    Return the exact row for a calendar date.
    """

    target = pd.Timestamp(date_string)

    index = pd.to_datetime(df.index)

    matches = df.loc[index == target]

    if matches.empty:
        raise ValueError(
            f"{date_string} not found in dataframe."
        )

    return matches.iloc[-1]


# ------------------------------------------------------------------
# Divergent TCS dates
# ------------------------------------------------------------------

def load_divergent_dates():
    """
    Build the TCS divergent trade-date set from the frozen
    historical dataset and fresh research candidate.

    Identity:
        _symbol + entry_date + direction
    """

    historical = pd.read_csv(
        HISTORICAL_FILE
    )

    candidate = pd.read_csv(
        CANDIDATE_FILE
    )

    historical = historical[
        historical["_symbol"].astype(str).str.upper()
        == SYMBOL
    ].copy()

    candidate = candidate[
        candidate["_symbol"].astype(str).str.upper()
        == SYMBOL
    ].copy()

    def make_key(df):

        return (
            df["_symbol"].astype(str)
            + "|"
            + pd.to_datetime(
                df["entry_date"]
            ).dt.strftime("%Y-%m-%d")
            + "|"
            + df["direction"].astype(str)
        )

    historical["_key"] = make_key(
        historical
    )

    candidate["_key"] = make_key(
        candidate
    )

    historical_keys = set(
        historical["_key"]
    )

    candidate_keys = set(
        candidate["_key"]
    )

    divergent_keys = (
        historical_keys
        ^ candidate_keys
    )

    dates = set()

    for key in divergent_keys:

        parts = key.split("|")

        if len(parts) != 3:
            continue

        dates.add(parts[1])

    return sorted(dates)


# ------------------------------------------------------------------
# Context calculation
# ------------------------------------------------------------------

def calculate_context(
    full_df,
    date_string
):
    """
    Calculate production context exactly as of date_string.

    IMPORTANT:
    Only candles through date_string are supplied.
    """

    target = pd.Timestamp(
        date_string
    )

    index = pd.to_datetime(
        full_df.index
    )

    historical = full_df.loc[
        index <= target
    ].copy()

    if historical.empty:
        raise ValueError(
            f"No history available through {date_string}"
        )

    current_price = float(
        historical["Close"].iloc[-1]
    )

    market_structure = calculate_market_structure(
        historical,
        swing_window=3
    )

    liquidity = calculate_liquidity(
        historical,
        market_structure
    )

    setup = calculate_setup(
        market_structure=market_structure,
        liquidity=liquidity,
        current_price=current_price
    )

    return {
        "history_rows": len(historical),
        "current_price": current_price,
        "market_structure": market_structure,
        "liquidity": liquidity,
        "setup": setup,
    }


# ------------------------------------------------------------------
# Extract useful setup fields
# ------------------------------------------------------------------

def extract_setup_fields(setup):

    return {
        "setup": setup.get(
            "setup"
        ),
        "direction": setup.get(
            "direction"
        ),
        "confidence": setup.get(
            "confidence"
        ),
        "reason": setup.get(
            "reason"
        ),
    }


def extract_structure_fields(structure):

    fields = [
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

    return {
        field: structure.get(
            field
        )
        for field in fields
    }


def extract_liquidity_fields(liquidity):

    if not isinstance(
        liquidity,
        dict
    ):
        return {
            "raw": liquidity
        }

    return {
        key: normalize_value(value)
        for key, value in liquidity.items()
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    print()
    print("=" * 110)
    print("TCS SETUP INPUT FORENSIC")
    print("=" * 110)

    print(
        f"Stable context : {STABLE_START}"
    )

    print(
        f"Rolling context: {ROLLING_START}"
    )

    print()

    # --------------------------------------------------------------
    # Load service
    # --------------------------------------------------------------

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    # --------------------------------------------------------------
    # Load both histories
    # --------------------------------------------------------------

    print(
        "Loading stable history..."
    )

    stable_df = load_price_history(
        service,
        STABLE_START
    )

    print(
        "Loading rolling history..."
    )

    rolling_df = load_price_history(
        service,
        ROLLING_START
    )

    print()

    print(
        f"Stable rows : {len(stable_df)}"
    )

    print(
        f"Rolling rows: {len(rolling_df)}"
    )

    print()

    # --------------------------------------------------------------
    # Divergent dates
    # --------------------------------------------------------------

    dates = load_divergent_dates()

    print(
        f"Divergent TCS dates: {len(dates)}"
    )

    print()

    if not dates:

        print(
            "No divergent TCS dates found."
        )

        return

    # --------------------------------------------------------------
    # Output preparation
    # --------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    comparable_dates = 0

    setup_output_differences = 0
    liquidity_differences = 0
    structure_differences = 0
    price_differences = 0

    # --------------------------------------------------------------
    # Analyze every divergent date
    # --------------------------------------------------------------

    for date_string in dates:

        print("-" * 110)
        print(
            f"DATE: {date_string}"
        )
        print("-" * 110)

        stable_context = calculate_context(
            stable_df,
            date_string
        )

        # ----------------------------------------------------------
        # The rolling context starts on 2021-09-13.
        #
        # Therefore dates before 2021-09-13 cannot be compared
        # against the rolling context.
        #
        # They are naturally "fresh-only" candidates caused by
        # the additional historical coverage.
        # ----------------------------------------------------------

        target_date = pd.Timestamp(
            date_string
        )

        rolling_start = pd.Timestamp(
            ROLLING_START
        )

        if target_date < rolling_start:

            print(
                "ROLLING CONTEXT: NOT AVAILABLE"
            )

            print(
                "Reason: this date is before "
                f"the rolling start date {ROLLING_START}."
            )

            print(
                "Classification: PRE-ROLLING "
                "FRESH-ONLY TRADE"
            )

            print()

            stable_price = (
                stable_context[
                    "current_price"
                ]
            )

            stable_structure = (
                stable_context[
                    "market_structure"
                ]
            )

            stable_liquidity = (
                stable_context[
                    "liquidity"
                ]
            )

            stable_setup = (
                stable_context[
                    "setup"
                ]
            )

            rows.append(
                {
                    "symbol": SYMBOL,
                    "date": date_string,

                    "stable_history_rows":
                        stable_context[
                            "history_rows"
                        ],

                    "rolling_history_rows":
                        0,

                    "stable_price":
                        stable_price,

                    "rolling_price":
                        None,

                    "price_same":
                        None,

                    "structure_same":
                        None,

                    "liquidity_same":
                        None,

                    "setup_same":
                        None,

                    "setup_diff_fields":
                        "",

                    "liquidity_diff_fields":
                        "",

                    "structure_diff_fields":
                        "",

                    "stable_setup":
                        str(
                            normalize_value(
                                stable_setup
                            )
                        ),

                    "rolling_setup":
                        "",

                    "stable_liquidity":
                        str(
                            normalize_value(
                                stable_liquidity
                            )
                        ),

                    "rolling_liquidity":
                        "",

                    "stable_structure":
                        str(
                            normalize_value(
                                stable_structure
                            )
                        ),

                    "rolling_structure":
                        "",
                }
            )

            continue

        # ----------------------------------------------------------
        # Genuine overlapping-context comparison
        # ----------------------------------------------------------

        rolling_context = calculate_context(
            rolling_df,
            date_string
        )

        stable_price = (
            stable_context[
                "current_price"
            ]
        )

        rolling_price = (
            rolling_context[
                "current_price"
            ]
        )

        stable_structure = (
            stable_context[
                "market_structure"
            ]
        )

        rolling_structure = (
            rolling_context[
                "market_structure"
            ]
        )

        stable_liquidity = (
            stable_context[
                "liquidity"
            ]
        )

        rolling_liquidity = (
            rolling_context[
                "liquidity"
            ]
        )

        stable_setup = (
            stable_context[
                "setup"
            ]
        )

        rolling_setup = (
            rolling_context[
                "setup"
            ]
        )

        # ----------------------------------------------------------
        # Equality checks
        # ----------------------------------------------------------

        price_same = values_equal(
            stable_price,
            rolling_price
        )

        structure_same = values_equal(
            extract_structure_fields(
                stable_structure
            ),
            extract_structure_fields(
                rolling_structure
            )
        )

        liquidity_same = values_equal(
            extract_liquidity_fields(
                stable_liquidity
            ),
            extract_liquidity_fields(
                rolling_liquidity
            )
        )

        setup_same = values_equal(
            extract_setup_fields(
                stable_setup
            ),
            extract_setup_fields(
                rolling_setup
            )
        )

        comparable_dates += 1

        if not price_same:
            price_differences += 1

        if not structure_same:
            structure_differences += 1

        if not liquidity_same:
            liquidity_differences += 1

        if not setup_same:
            setup_output_differences += 1

        # ----------------------------------------------------------
        # Print concise result
        # ----------------------------------------------------------

        print(
            f"History rows : "
            f"{stable_context['history_rows']} "
            f"vs "
            f"{rolling_context['history_rows']}"
        )

        print(
            f"Current price: "
            f"{stable_price} "
            f"vs "
            f"{rolling_price} "
            f"| SAME={price_same}"
        )

        print()

        print(
            "STRUCTURE:"
        )

        print(
            "  Stable : "
            f"{extract_structure_fields(stable_structure)}"
        )

        print(
            "  Rolling: "
            f"{extract_structure_fields(rolling_structure)}"
        )

        print(
            f"  SAME={structure_same}"
        )

        print()

        print(
            "LIQUIDITY:"
        )

        print(
            "  Stable : "
            f"{extract_liquidity_fields(stable_liquidity)}"
        )

        print(
            "  Rolling: "
            f"{extract_liquidity_fields(rolling_liquidity)}"
        )

        print(
            f"  SAME={liquidity_same}"
        )

        print()

        print(
            "SETUP:"
        )

        print(
            "  Stable : "
            f"{extract_setup_fields(stable_setup)}"
        )

        print(
            "  Rolling: "
            f"{extract_setup_fields(rolling_setup)}"
        )

        print(
            f"  SAME={setup_same}"
        )

        # ----------------------------------------------------------
        # Setup field-level differences
        # ----------------------------------------------------------

        stable_setup_fields = (
            extract_setup_fields(
                stable_setup
            )
        )

        rolling_setup_fields = (
            extract_setup_fields(
                rolling_setup
            )
        )

        setup_diff_fields = []

        for field in stable_setup_fields:

            if not values_equal(
                stable_setup_fields[field],
                rolling_setup_fields[field]
            ):
                setup_diff_fields.append(
                    field
                )

        # ----------------------------------------------------------
        # Liquidity field-level differences
        # ----------------------------------------------------------

        stable_liquidity_fields = (
            extract_liquidity_fields(
                stable_liquidity
            )
        )

        rolling_liquidity_fields = (
            extract_liquidity_fields(
                rolling_liquidity
            )
        )

        liquidity_diff_fields = sorted(
            set(
                stable_liquidity_fields
            )
            |
            set(
                rolling_liquidity_fields
            )
        )

        liquidity_diff_fields = [
            field
            for field in liquidity_diff_fields
            if not values_equal(
                stable_liquidity_fields.get(
                    field
                ),
                rolling_liquidity_fields.get(
                    field
                )
            )
        ]

        # ----------------------------------------------------------
        # Structure field-level differences
        # ----------------------------------------------------------

        stable_structure_fields = (
            extract_structure_fields(
                stable_structure
            )
        )

        rolling_structure_fields = (
            extract_structure_fields(
                rolling_structure
            )
        )

        structure_diff_fields = [
            field
            for field in stable_structure_fields
            if not values_equal(
                stable_structure_fields[field],
                rolling_structure_fields[field]
            )
        ]

        print()

        print(
            f"Setup differing fields: "
            f"{setup_diff_fields}"
        )

        print(
            f"Liquidity differing fields: "
            f"{liquidity_diff_fields}"
        )

        print(
            f"Structure differing fields: "
            f"{structure_diff_fields}"
        )

        # ----------------------------------------------------------
        # CSV row
        # ----------------------------------------------------------

        rows.append(
            {
                "symbol": SYMBOL,
                "date": date_string,

                "stable_history_rows":
                    stable_context[
                        "history_rows"
                    ],

                "rolling_history_rows":
                    rolling_context[
                        "history_rows"
                    ],

                "stable_price":
                    stable_price,

                "rolling_price":
                    rolling_price,

                "price_same":
                    price_same,

                "structure_same":
                    structure_same,

                "liquidity_same":
                    liquidity_same,

                "setup_same":
                    setup_same,

                "setup_diff_fields":
                    "|".join(
                        setup_diff_fields
                    ),

                "liquidity_diff_fields":
                    "|".join(
                        liquidity_diff_fields
                    ),

                "structure_diff_fields":
                    "|".join(
                        structure_diff_fields
                    ),

                "stable_setup":
                    str(
                        normalize_value(
                            stable_setup
                        )
                    ),

                "rolling_setup":
                    str(
                        normalize_value(
                            rolling_setup
                        )
                    ),

                "stable_liquidity":
                    str(
                        normalize_value(
                            stable_liquidity
                        )
                    ),

                "rolling_liquidity":
                    str(
                        normalize_value(
                            rolling_liquidity
                        )
                    ),

                "stable_structure":
                    str(
                        normalize_value(
                            stable_structure
                        )
                    ),

                "rolling_structure":
                    str(
                        normalize_value(
                            rolling_structure
                        )
                    ),
            }
        )

    # --------------------------------------------------------------
    # Save forensic output
    # --------------------------------------------------------------

    output_df = pd.DataFrame(
        rows
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------------
    # Final summary
    # --------------------------------------------------------------

    print()
    print("=" * 110)
    print("FORENSIC SUMMARY")
    print("=" * 110)

    print(
        f"Dates analysed              : {len(dates)}"
    )

    print(
        f"Comparable dates             : {comparable_dates}"
    )

    print(
        f"Price differences           : "
        f"{price_differences}"
    )

    print(
        f"Structure differences       : "
        f"{structure_differences}"
    )

    print(
        f"Liquidity differences       : "
        f"{liquidity_differences}"
    )

    print(
        f"Setup-output differences    : "
        f"{setup_output_differences}"
    )

    print()

    if setup_output_differences == 0:

        print(
            "RESULT: calculate_setup() output "
            "is identical across all divergent TCS dates."
        )

        print(
            "NEXT LAYER: investigate trade-generation/"
            "backtest execution logic."
        )

    else:

        print(
            "RESULT: calculate_setup() differs on "
            "at least one divergent TCS date."
        )

        print(
            "NEXT LAYER: isolate the exact setup input "
            "or setup rule causing the divergence."
        )

    print()

    print(
        f"Output saved to: {OUTPUT_FILE}"
    )

    print(
        "=" * 110
    )


if __name__ == "__main__":
    main()