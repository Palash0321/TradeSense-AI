"""
TradeSense-AI
Structural Event Causality Forensic

Purpose
-------
Determine whether differences in:

    last_structural_event
    structural_event
    new_structural_event

are responsible for the 41 historical-vs-candidate trade identity
differences.

IMPORTANT
---------
Diagnostic only.

This script MUST NOT:
- modify production code
- modify the frozen historical dataset
- modify the research candidate dataset
- modify the holdout dataset

Contexts compared:
    STABLE  = 2021-01-01
    ROLLING = 2021-09-13

The structural-event state machine mirrors the production
generate_tradesense_signals() logic.
"""

import csv
import os

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
ROLLING_START_DATE = "2021-09-13"

FROZEN_HISTORICAL_FILE = (
    "tests/output/tradesense_trades.csv"
)

FRESH_ONLY_FILE = (
    "tests/output/reconciliation/fresh_only_trades.csv"
)

HISTORICAL_ONLY_FILE = (
    "tests/output/reconciliation/historical_only_trades.csv"
)

OUTPUT_FILE = (
    "tests/output/reconciliation/"
    "structural_event_causality_forensic.csv"
)

SWING_WINDOW = 3


# ==============================================================
# HELPERS
# ==============================================================

def normalize_date(value):
    """
    Convert any date-like value to YYYY-MM-DD.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except Exception:
        return text[:10]


def normalize_symbol(value):
    """
    Normalize symbol representation.
    """
    if value is None:
        return ""

    return str(value).strip().upper()


def normalize_direction(value):
    """
    Normalize LONG / SHORT.
    """
    if value is None:
        return ""

    return str(value).strip().upper()


def make_trade_key(symbol, entry_date, direction):
    """
    Identity used by the reconciliation workflow.
    """
    return (
        normalize_symbol(symbol),
        normalize_date(entry_date),
        normalize_direction(direction),
    )


def read_trade_file(path):
    """
    Read a trade CSV into a list of dictionaries.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(file)
        )


def get_trade_identity(row):
    """
    Extract trade identity from a row.
    """
    symbol = (
        row.get("_symbol")
        or row.get("symbol")
        or row.get("Symbol")
    )

    entry_date = (
        row.get("entry_date")
        or row.get("Entry_Date")
        or row.get("ENTRY_DATE")
    )

    direction = (
        row.get("direction")
        or row.get("Direction")
        or row.get("DIR")
    )

    return make_trade_key(
        symbol,
        entry_date,
        direction
    )


def load_divergent_cases():
    """
    Load fresh-only and historical-only identities.

    The union represents every identity that differs between
    frozen historical and fresh candidate datasets.
    """

    fresh_only = read_trade_file(
        FRESH_ONLY_FILE
    )

    historical_only = read_trade_file(
        HISTORICAL_ONLY_FILE
    )

    cases = {}

    for row in fresh_only:

        key = get_trade_identity(row)

        cases[key] = {
            "identity": key,
            "fresh_only": True,
            "historical_only": False,
        }

    for row in historical_only:

        key = get_trade_identity(row)

        if key in cases:
            cases[key]["historical_only"] = True
        else:
            cases[key] = {
                "identity": key,
                "fresh_only": False,
                "historical_only": True,
            }

    return sorted(
        cases.values(),
        key=lambda item: (
            item["identity"][0],
            item["identity"][1],
            item["identity"][2],
        )
    )


def build_date_index(df):
    """
    Build normalized YYYY-MM-DD -> positional index map.
    """
    result = {}

    for position, value in enumerate(df.index):

        date_key = normalize_date(value)

        if date_key is not None:
            result[date_key] = position

    return result


# ==============================================================
# PRODUCTION-EQUIVALENT STRUCTURAL EVENT TRACE
# ==============================================================

def trace_context(
    service,
    df,
    target_signal_dates
):
    """
    Reconstruct the production structural-event state machine
    over the complete supplied historical context.

    Returns a dictionary keyed by signal date.

    This is intentionally walk-forward:
        history = result.iloc[:i + 1]

    so no future candles are supplied to any calculation.
    """

    result = df.copy()

    date_index = build_date_index(result)

    traces = {}

    minimum_history = (
        SWING_WINDOW * 2 + 1
    )

    last_structural_event = None

    for i in range(
        minimum_history - 1,
        len(result)
    ):

        history = result.iloc[
            :i + 1
        ]

        current_date = normalize_date(
            result.index[i]
        )

        current_price = float(
            history["Close"].iloc[-1]
        )

        # ------------------------------------------------------
        # MARKET STRUCTURE
        # ------------------------------------------------------

        market_structure = (
            calculate_market_structure(
                history,
                swing_window=SWING_WINDOW
            )
        )

        # ------------------------------------------------------
        # LIQUIDITY
        # ------------------------------------------------------

        liquidity = calculate_liquidity(
            history,
            market_structure
        )

        # ------------------------------------------------------
        # STRUCTURAL EVENT
        # ------------------------------------------------------

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

        structural_event = None

        if (
            break_confirmed
            and break_direction is not None
            and break_level is not None
            and break_date is not None
        ):

            structural_event = (
                break_direction,
                round(
                    float(break_level),
                    2
                ),
                str(break_date)
            )

        previous_last_event = (
            last_structural_event
        )

        new_structural_event = (
            structural_event is not None
            and
            structural_event
            != last_structural_event
        )

        # ------------------------------------------------------
        # SETUP
        # ------------------------------------------------------

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

        # ------------------------------------------------------
        # PRODUCTION-EQUIVALENT SIGNAL LOGIC
        # ------------------------------------------------------

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
            and new_structural_event
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

        # ------------------------------------------------------
        # PRODUCTION STATE UPDATE
        # ------------------------------------------------------

        if new_structural_event:

            last_structural_event = (
                structural_event
            )

        # ------------------------------------------------------
        # STORE ONLY TARGET DATES
        # ------------------------------------------------------

        if current_date in target_signal_dates:

            traces[current_date] = {
                "signal_date": current_date,

                "previous_last_event": (
                    previous_last_event
                ),

                "structural_event": (
                    structural_event
                ),

                "new_structural_event": (
                    bool(new_structural_event)
                ),

                "raw_setup": (
                    setup_type
                ),

                "raw_setup_direction": (
                    setup_direction
                ),

                "raw_setup_confidence": (
                    setup_confidence
                ),

                "signal": (
                    signal
                ),

                "structure_bias": (
                    market_structure.get(
                        "bias"
                    )
                ),

                "structure": (
                    market_structure.get(
                        "structure"
                    )
                ),

                "break_direction": (
                    break_direction
                ),

                "break_confirmed": (
                    break_confirmed
                ),

                "break_level": (
                    break_level
                ),

                "break_date": (
                    break_date
                ),

                "break_age": (
                    market_structure.get(
                        "break_age"
                    )
                ),
            }

    return traces


# ==============================================================
# SIGNAL-DATE RESOLUTION
# ==============================================================

def resolve_signal_date(
    df,
    entry_date
):
    """
    Entry happens on the bar immediately following the signal bar.

    Therefore:
        signal_date = bar immediately preceding entry_date.
    """

    normalized_dates = [
        normalize_date(value)
        for value in df.index
    ]

    entry_key = normalize_date(
        entry_date
    )

    try:
        entry_position = (
            normalized_dates.index(
                entry_key
            )
        )
    except ValueError:
        return None

    if entry_position <= 0:
        return None

    return normalized_dates[
        entry_position - 1
    ]


# ==============================================================
# CASE COMPARISON
# ==============================================================

def classify_case(
    stable,
    rolling
):
    """
    Classify the causal relationship.

    Priority:
        SETUP_AND_EVENT
        EVENT_CAUSES_SIGNAL_DIVERGENCE
        SETUP_CAUSES_SIGNAL_DIVERGENCE
        EVENT_STATE_DIFFERENCE_NO_SIGNAL_EFFECT
        NO_SIGNAL_DIVERGENCE
    """

    if stable is None or rolling is None:
        return (
            "ROLLING_CONTEXT_UNAVAILABLE"
        )

    setup_diff = (
        stable["raw_setup"]
        != rolling["raw_setup"]
        or
        stable["raw_setup_direction"]
        != rolling["raw_setup_direction"]
    )

    event_diff = (
        stable["previous_last_event"]
        != rolling["previous_last_event"]
        or
        stable["structural_event"]
        != rolling["structural_event"]
        or
        stable["new_structural_event"]
        != rolling["new_structural_event"]
    )

    signal_diff = (
        stable["signal"]
        != rolling["signal"]
    )

    if signal_diff and setup_diff and event_diff:
        return (
            "SETUP_AND_EVENT_CAUSE_SIGNAL_DIVERGENCE"
        )

    if signal_diff and event_diff:
        return (
            "EVENT_CAUSES_SIGNAL_DIVERGENCE"
        )

    if signal_diff and setup_diff:
        return (
            "SETUP_CAUSES_SIGNAL_DIVERGENCE"
        )

    if event_diff and not signal_diff:
        return (
            "EVENT_STATE_DIFFERENCE_NO_SIGNAL_EFFECT"
        )

    return (
        "NO_SIGNAL_DIVERGENCE"
    )


# ==============================================================
# MAIN
# ==============================================================

def main():

    print()
    print("=" * 120)
    print("STRUCTURAL EVENT CAUSALITY FORENSIC")
    print("=" * 120)

    print()
    print(
        f"Stable context : {STABLE_START_DATE}"
    )

    print(
        f"Rolling context: {ROLLING_START_DATE}"
    )

    print()

    # ----------------------------------------------------------
    # LOAD DIVERGENT IDENTITIES
    # ----------------------------------------------------------

    cases = load_divergent_cases()

    print(
        f"Divergent trade identities: "
        f"{len(cases)}"
    )

    if len(cases) != 41:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected 41 divergent identities, "
            f"found {len(cases)}."
        )

        print(
            "Continue only if reconciliation files "
            "are intentionally different."
        )

    # ----------------------------------------------------------
    # TARGET SIGNAL DATES BY SYMBOL
    # ----------------------------------------------------------

    symbol_targets = {
        symbol: set()
        for symbol in SYMBOLS
    }

    case_metadata = []

    # We resolve signal dates separately under each context.
    # First collect entry dates.

    for case in cases:

        symbol, entry_date, direction = (
            case["identity"]
        )

        if symbol not in symbol_targets:
            continue

        case_metadata.append(
            {
                "symbol": symbol,
                "entry_date": entry_date,
                "direction": direction,
            }
        )

    # ----------------------------------------------------------
    # PROCESS SYMBOL BY SYMBOL
    # ----------------------------------------------------------

    output_rows = []

    for symbol in SYMBOLS:

        print()
        print("=" * 120)
        print(
            f"SYMBOL: {symbol}"
        )
        print("=" * 120)

        symbol_cases = [
            case
            for case in case_metadata
            if case["symbol"] == symbol
        ]

        if not symbol_cases:
            print(
                "No divergent cases."
            )
            continue

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=100000
        )

        # ------------------------------------------------------
        # STABLE DATA
        # ------------------------------------------------------

        print(
            "Loading stable context..."
        )

        stable_df = service.load_data(
            start_date=STABLE_START_DATE
        )

        if stable_df.empty:

            print(
                "ERROR: Stable dataset is empty."
            )

            continue

        # ------------------------------------------------------
        # ROLLING DATA
        # ------------------------------------------------------

        print(
            "Loading rolling context..."
        )

        rolling_df = service.load_data(
            start_date=ROLLING_START_DATE
        )

        if rolling_df.empty:

            print(
                "ERROR: Rolling dataset is empty."
            )

            continue

        # ------------------------------------------------------
        # RESOLVE SIGNAL DATES
        # ------------------------------------------------------

        stable_targets = set()
        rolling_targets = set()

        per_case_dates = {}

        for case in symbol_cases:

            entry_date = case["entry_date"]

            stable_signal_date = (
                resolve_signal_date(
                    stable_df,
                    entry_date
                )
            )

            rolling_signal_date = (
                resolve_signal_date(
                    rolling_df,
                    entry_date
                )
            )

            per_case_dates[
                (
                    entry_date,
                    case["direction"]
                )
            ] = {
                "stable": stable_signal_date,
                "rolling": rolling_signal_date,
            }

            if stable_signal_date:
                stable_targets.add(
                    stable_signal_date
                )

            if rolling_signal_date:
                rolling_targets.add(
                    rolling_signal_date
                )

        # ------------------------------------------------------
        # TRACE COMPLETE HISTORICAL STATE MACHINES
        # ------------------------------------------------------

        print(
            "Tracing stable structural-event state..."
        )

        stable_traces = trace_context(
            service=service,
            df=stable_df,
            target_signal_dates=stable_targets
        )

        print(
            "Tracing rolling structural-event state..."
        )

        rolling_traces = trace_context(
            service=service,
            df=rolling_df,
            target_signal_dates=rolling_targets
        )

        # ------------------------------------------------------
        # COMPARE CASES
        # ------------------------------------------------------

        for case in symbol_cases:

            entry_date = case["entry_date"]
            direction = case["direction"]

            dates = per_case_dates[
                (
                    entry_date,
                    direction
                )
            ]

            stable_signal_date = (
                dates["stable"]
            )

            rolling_signal_date = (
                dates["rolling"]
            )

            stable = (
                stable_traces.get(
                    stable_signal_date
                )
                if stable_signal_date
                else None
            )

            rolling = (
                rolling_traces.get(
                    rolling_signal_date
                )
                if rolling_signal_date
                else None
            )

            classification = classify_case(
                stable,
                rolling
            )

            row = {
                "symbol": symbol,
                "entry_date": entry_date,
                "direction": direction,

                "stable_signal_date": (
                    stable_signal_date
                ),

                "rolling_signal_date": (
                    rolling_signal_date
                ),

                "stable_previous_last_event": (
                    repr(
                        stable[
                            "previous_last_event"
                        ]
                    )
                    if stable
                    else None
                ),

                "rolling_previous_last_event": (
                    repr(
                        rolling[
                            "previous_last_event"
                        ]
                    )
                    if rolling
                    else None
                ),

                "stable_structural_event": (
                    repr(
                        stable[
                            "structural_event"
                        ]
                    )
                    if stable
                    else None
                ),

                "rolling_structural_event": (
                    repr(
                        rolling[
                            "structural_event"
                        ]
                    )
                    if rolling
                    else None
                ),

                "stable_new_structural_event": (
                    stable[
                        "new_structural_event"
                    ]
                    if stable
                    else None
                ),

                "rolling_new_structural_event": (
                    rolling[
                        "new_structural_event"
                    ]
                    if rolling
                    else None
                ),

                "stable_raw_setup": (
                    stable["raw_setup"]
                    if stable
                    else None
                ),

                "rolling_raw_setup": (
                    rolling["raw_setup"]
                    if rolling
                    else None
                ),

                "stable_raw_setup_direction": (
                    stable[
                        "raw_setup_direction"
                    ]
                    if stable
                    else None
                ),

                "rolling_raw_setup_direction": (
                    rolling[
                        "raw_setup_direction"
                    ]
                    if rolling
                    else None
                ),

                "stable_raw_setup_confidence": (
                    stable[
                        "raw_setup_confidence"
                    ]
                    if stable
                    else None
                ),

                "rolling_raw_setup_confidence": (
                    rolling[
                        "raw_setup_confidence"
                    ]
                    if rolling
                    else None
                ),

                "stable_signal": (
                    stable["signal"]
                    if stable
                    else None
                ),

                "rolling_signal": (
                    rolling["signal"]
                    if rolling
                    else None
                ),

                "stable_structure_bias": (
                    stable["structure_bias"]
                    if stable
                    else None
                ),

                "rolling_structure_bias": (
                    rolling["structure_bias"]
                    if rolling
                    else None
                ),

                "stable_structure": (
                    stable["structure"]
                    if stable
                    else None
                ),

                "rolling_structure": (
                    rolling["structure"]
                    if rolling
                    else None
                ),

                "stable_break_direction": (
                    stable["break_direction"]
                    if stable
                    else None
                ),

                "rolling_break_direction": (
                    rolling["break_direction"]
                    if rolling
                    else None
                ),

                "stable_break_confirmed": (
                    stable["break_confirmed"]
                    if stable
                    else None
                ),

                "rolling_break_confirmed": (
                    rolling["break_confirmed"]
                    if rolling
                    else None
                ),

                "stable_break_level": (
                    stable["break_level"]
                    if stable
                    else None
                ),

                "rolling_break_level": (
                    rolling["break_level"]
                    if rolling
                    else None
                ),

                "stable_break_date": (
                    stable["break_date"]
                    if stable
                    else None
                ),

                "rolling_break_date": (
                    rolling["break_date"]
                    if rolling
                    else None
                ),

                "stable_break_age": (
                    stable["break_age"]
                    if stable
                    else None
                ),

                "rolling_break_age": (
                    rolling["break_age"]
                    if rolling
                    else None
                ),

                "classification": classification,
            }

            output_rows.append(row)

            # --------------------------------------------------
            # CONSOLE OUTPUT
            # --------------------------------------------------

            print()
            print("-" * 120)
            print(
                f"ENTRY: {entry_date} | "
                f"{direction}"
            )
            print("-" * 120)

            print(
                f"Stable signal date : "
                f"{stable_signal_date}"
            )

            print(
                f"Rolling signal date: "
                f"{rolling_signal_date}"
            )

            if stable:

                print()
                print(
                    "STABLE"
                )

                print(
                    "  Previous event : "
                    f"{stable['previous_last_event']}"
                )

                print(
                    "  Current event  : "
                    f"{stable['structural_event']}"
                )

                print(
                    "  New event      : "
                    f"{stable['new_structural_event']}"
                )

                print(
                    "  Raw setup      : "
                    f"{stable['raw_setup']}"
                )

                print(
                    "  Setup direction: "
                    f"{stable['raw_setup_direction']}"
                )

                print(
                    "  Signal         : "
                    f"{stable['signal']}"
                )

            else:

                print()
                print(
                    "STABLE: unavailable"
                )

            if rolling:

                print()
                print(
                    "ROLLING"
                )

                print(
                    "  Previous event : "
                    f"{rolling['previous_last_event']}"
                )

                print(
                    "  Current event  : "
                    f"{rolling['structural_event']}"
                )

                print(
                    "  New event      : "
                    f"{rolling['new_structural_event']}"
                )

                print(
                    "  Raw setup      : "
                    f"{rolling['raw_setup']}"
                )

                print(
                    "  Setup direction: "
                    f"{rolling['raw_setup_direction']}"
                )

                print(
                    "  Signal         : "
                    f"{rolling['signal']}"
                )

            else:

                print()
                print(
                    "ROLLING: unavailable"
                )

            print()
            print(
                "CLASSIFICATION: "
                f"{classification}"
            )

    # ----------------------------------------------------------
    # SAVE REPORT
    # ----------------------------------------------------------

    output_dir = os.path.dirname(
        OUTPUT_FILE
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    if output_rows:

        fieldnames = list(
            output_rows[0].keys()
        )

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
                output_rows
            )

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    counts = {}

    for row in output_rows:

        classification = row[
            "classification"
        ]

        counts[classification] = (
            counts.get(
                classification,
                0
            )
            + 1
        )

    print()
    print("=" * 120)
    print("FINAL STRUCTURAL EVENT CAUSALITY SUMMARY")
    print("=" * 120)

    print()
    print(
        f"Cases analysed: "
        f"{len(output_rows)}"
    )

    print()

    for classification in sorted(
        counts
    ):

        print(
            f"{classification:<50} "
            f"{counts[classification]}"
        )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print("=" * 120)

    # ----------------------------------------------------------
    # INTERPRETATION
    # ----------------------------------------------------------

    event_caused = (
        counts.get(
            "EVENT_CAUSES_SIGNAL_DIVERGENCE",
            0
        )
        +
        counts.get(
            "SETUP_AND_EVENT_CAUSE_SIGNAL_DIVERGENCE",
            0
        )
    )

    setup_caused = (
        counts.get(
            "SETUP_CAUSES_SIGNAL_DIVERGENCE",
            0
        )
        +
        counts.get(
            "SETUP_AND_EVENT_CAUSE_SIGNAL_DIVERGENCE",
            0
        )
    )

    print()

    print(
        f"Signal divergences with event involvement: "
        f"{event_caused}"
    )

    print(
        f"Signal divergences with setup involvement: "
        f"{setup_caused}"
    )

    print()

    if event_caused == 0:

        print(
            "RESULT: STRUCTURAL EVENT STATE MACHINE "
            "DOES NOT EXPLAIN THE DIVERGENT SIGNALS "
            "IN THIS AUDIT."
        )

    else:

        print(
            "RESULT: STRUCTURAL EVENT STATE MACHINE "
            "IS INVOLVED IN AT LEAST SOME DIVERGENT SIGNALS."
        )

    print()

    print(
        "IMPORTANT: This is a forensic diagnostic only."
    )

    print(
        "No production code or frozen dataset was modified."
    )


if __name__ == "__main__":
    main()