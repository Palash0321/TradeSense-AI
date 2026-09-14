import csv
import os

import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure
from app.core.liquidity.liquidity import calculate_liquidity
from app.core.setup.setup_engine import calculate_setup


SYMBOL = "TCS.NS"

STABLE_START = "2021-01-01"
ROLLING_START = "2021-09-13"

FRESH_ONLY_FILE = (
    "tests/output/reconciliation/fresh_only_trades.csv"
)

HISTORICAL_ONLY_FILE = (
    "tests/output/reconciliation/historical_only_trades.csv"
)

OUTPUT_FILE = (
    "tests/output/reconciliation/"
    "setup_causality_forensic.csv"
)


def normalize_date(value):
    return pd.Timestamp(value).normalize()


def load_divergent_entry_dates():
    dates = set()

    for filename in [
        FRESH_ONLY_FILE,
        HISTORICAL_ONLY_FILE,
    ]:

        if not os.path.exists(filename):
            raise FileNotFoundError(
                f"Required reconciliation file not found: {filename}"
            )

        with open(
            filename,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row.get("_symbol") != SYMBOL:
                    continue

                entry_date = row.get("entry_date")

                if not entry_date:
                    continue

                dates.add(
                    normalize_date(
                        entry_date[:10]
                    )
                )

    return sorted(dates)


def load_contexts():
    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    stable_data = service.load_data(
        start_date=STABLE_START
    )

    rolling_data = service.load_data(
        start_date=ROLLING_START
    )

    stable_data.index = pd.to_datetime(
        stable_data.index
    ).normalize()

    rolling_data.index = pd.to_datetime(
        rolling_data.index
    ).normalize()

    stable_data = stable_data.sort_index()
    rolling_data = rolling_data.sort_index()

    return stable_data, rolling_data


def get_signal_date(data, entry_date):
    """
    Production execution enters on the next candle open.
    Therefore the signal candle is the immediately
    preceding available trading row.
    """

    eligible = data.index[
        data.index < entry_date
    ]

    if len(eligible) == 0:
        return None

    return eligible[-1]


def calculate_inputs(data, signal_date):
    history = data.loc[
        data.index <= signal_date
    ].copy()

    if history.empty:
        return None

    current_price = float(
        history.iloc[-1]["Close"]
    )

    market_structure = calculate_market_structure(
        history
    )

    liquidity = calculate_liquidity(
        history,
        market_structure
    )

    setup = calculate_setup(
        market_structure=market_structure,
        liquidity=liquidity,
        current_price=current_price
    )

    return {
        "history": history,
        "price": current_price,
        "structure": market_structure,
        "liquidity": liquidity,
        "setup": setup,
    }


def setup_signature(setup):
    return (
        setup.get("setup", "WAIT"),
        setup.get("direction"),
        float(
            setup.get("confidence", 0) or 0
        ),
        setup.get("reason", ""),
    )


def structure_signature(structure):
    return {
        "bias": structure.get("bias"),
        "structure": structure.get("structure"),
        "bos": structure.get("bos"),
        "choch": structure.get("choch"),
        "break_direction": structure.get(
            "break_direction"
        ),
        "break_level": structure.get(
            "break_level"
        ),
        "break_date": structure.get(
            "break_date"
        ),
        "break_confirmed": structure.get(
            "break_confirmed"
        ),
        "break_age": structure.get(
            "break_age"
        ),
        "strength": structure.get(
            "strength"
        ),
    }


def liquidity_signature(liquidity):
    return {
        "equal_highs": liquidity.get(
            "equal_highs",
            []
        ),
        "equal_lows": liquidity.get(
            "equal_lows",
            []
        ),
        "buy_side_liquidity": liquidity.get(
            "buy_side_liquidity",
            []
        ),
        "sell_side_liquidity": liquidity.get(
            "sell_side_liquidity",
            []
        ),
        "sweep": liquidity.get(
            "sweep",
            {}
        ),
        "nearest_liquidity": liquidity.get(
            "nearest_liquidity",
            {}
        ),
        "strength": liquidity.get(
            "strength",
            0
        ),
    }


def run_mixed_setup(
    stable_inputs,
    rolling_inputs,
    structure_source,
    liquidity_source
):

    if structure_source == "STABLE":
        structure = stable_inputs["structure"]
    else:
        structure = rolling_inputs["structure"]

    if liquidity_source == "STABLE":
        liquidity = stable_inputs["liquidity"]
    else:
        liquidity = rolling_inputs["liquidity"]

    return calculate_setup(
        market_structure=structure,
        liquidity=liquidity,
        current_price=stable_inputs["price"]
    )


def classify_cause(
    actual_stable,
    actual_rolling,
    stable_structure_rolling_liquidity,
    rolling_structure_stable_liquidity
):

    stable_sig = setup_signature(
        actual_stable
    )

    rolling_sig = setup_signature(
        actual_rolling
    )

    mixed_sr_sig = setup_signature(
        stable_structure_rolling_liquidity
    )

    mixed_rs_sig = setup_signature(
        rolling_structure_stable_liquidity
    )

    if stable_sig == rolling_sig:
        return "NO_SETUP_DIVERGENCE"

    # Stable structure + rolling liquidity reproduces
    # the rolling result.
    if mixed_sr_sig == rolling_sig:

        # And rolling structure + stable liquidity
        # reproduces stable.
        if mixed_rs_sig == stable_sig:
            return "BOTH_INPUTS_INTERACT"

        return "LIQUIDITY_CAUSES_DIVERGENCE"

    # Rolling structure + stable liquidity reproduces
    # the rolling result.
    if mixed_rs_sig == rolling_sig:
        return "STRUCTURE_CAUSES_DIVERGENCE"

    # Neither isolated replacement reproduces the
    # other side.
    return "NON_SEPARABLE_OR_RULE_INTERACTION"


def main():

    print()
    print("=" * 110)
    print("TRADESENSE SETUP CAUSALITY FORENSIC")
    print("=" * 110)

    divergent_entry_dates = (
        load_divergent_entry_dates()
    )

    print()
    print(
        f"Symbol: {SYMBOL}"
    )
    print(
        f"Divergent entry dates: "
        f"{len(divergent_entry_dates)}"
    )

    stable_data, rolling_data = load_contexts()

    rows = []

    for entry_date in divergent_entry_dates:

        stable_signal_date = get_signal_date(
            stable_data,
            entry_date
        )

        rolling_signal_date = get_signal_date(
            rolling_data,
            entry_date
        )

        if (
            stable_signal_date is None
            or rolling_signal_date is None
        ):
            print()
            print(
                f"{entry_date.date()} "
                f"SKIPPED — no signal date"
            )
            continue

        # The signal-date alignment itself must be
        # checked before causal attribution.
        if stable_signal_date != rolling_signal_date:

            print()
            print("=" * 110)
            print(
                f"ENTRY DATE: {entry_date.date()}"
            )
            print(
                "SIGNAL DATE ALIGNMENT DIFFERENCE"
            )
            print(
                f"Stable : "
                f"{stable_signal_date.date()}"
            )
            print(
                f"Rolling: "
                f"{rolling_signal_date.date()}"
            )
            print("=" * 110)

            rows.append({
                "entry_date": str(
                    entry_date.date()
                ),
                "signal_date": None,
                "status": "SIGNAL_DATE_MISMATCH",
            })

            continue

        signal_date = stable_signal_date

        stable_inputs = calculate_inputs(
            stable_data,
            signal_date
        )

        rolling_inputs = calculate_inputs(
            rolling_data,
            signal_date
        )

        if (
            stable_inputs is None
            or rolling_inputs is None
        ):
            continue

        actual_stable = stable_inputs["setup"]
        actual_rolling = rolling_inputs["setup"]

        mixed_sr = run_mixed_setup(
            stable_inputs,
            rolling_inputs,
            structure_source="STABLE",
            liquidity_source="ROLLING"
        )

        mixed_rs = run_mixed_setup(
            stable_inputs,
            rolling_inputs,
            structure_source="ROLLING",
            liquidity_source="STABLE"
        )

        stable_sig = setup_signature(
            actual_stable
        )

        rolling_sig = setup_signature(
            actual_rolling
        )

        mixed_sr_sig = setup_signature(
            mixed_sr
        )

        mixed_rs_sig = setup_signature(
            mixed_rs
        )

        cause = classify_cause(
            actual_stable,
            actual_rolling,
            mixed_sr,
            mixed_rs
        )

        structure_same = (
            structure_signature(
                stable_inputs["structure"]
            )
            ==
            structure_signature(
                rolling_inputs["structure"]
            )
        )

        liquidity_same = (
            liquidity_signature(
                stable_inputs["liquidity"]
            )
            ==
            liquidity_signature(
                rolling_inputs["liquidity"]
            )
        )

        print()
        print("=" * 110)
        print(
            f"ENTRY DATE: {entry_date.date()}"
        )
        print(
            f"SIGNAL DATE: {signal_date.date()}"
        )
        print("=" * 110)

        print()
        print("ACTUAL SETUPS")
        print(
            f"Stable : {stable_sig}"
        )
        print(
            f"Rolling: {rolling_sig}"
        )

        print()
        print("MIXED INPUT TESTS")
        print(
            "Stable Structure + Rolling Liquidity:"
        )
        print(
            f"    {mixed_sr_sig}"
        )

        print(
            "Rolling Structure + Stable Liquidity:"
        )
        print(
            f"    {mixed_rs_sig}"
        )

        print()
        print(
            f"Structure identical: {structure_same}"
        )
        print(
            f"Liquidity identical: {liquidity_same}"
        )

        print()
        print(
            f"CAUSAL CLASSIFICATION: {cause}"
        )

        rows.append({
            "entry_date": str(
                entry_date.date()
            ),
            "signal_date": str(
                signal_date.date()
            ),
            "stable_setup": stable_sig[0],
            "stable_direction": stable_sig[1],
            "stable_confidence": stable_sig[2],
            "rolling_setup": rolling_sig[0],
            "rolling_direction": rolling_sig[1],
            "rolling_confidence": rolling_sig[2],
            "mixed_stable_structure_setup": (
                mixed_sr_sig[0]
            ),
            "mixed_stable_structure_direction": (
                mixed_sr_sig[1]
            ),
            "mixed_stable_structure_confidence": (
                mixed_sr_sig[2]
            ),
            "mixed_rolling_structure_setup": (
                mixed_rs_sig[0]
            ),
            "mixed_rolling_structure_direction": (
                mixed_rs_sig[1]
            ),
            "mixed_rolling_structure_confidence": (
                mixed_rs_sig[2]
            ),
            "structure_same": structure_same,
            "liquidity_same": liquidity_same,
            "cause": cause,
            "status": "ANALYSED",
        })

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
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
        writer.writerows(rows)

    print()
    print("=" * 110)
    print("CAUSAL FORENSIC SUMMARY")
    print("=" * 110)

    analysed = [
        row
        for row in rows
        if row.get("status") == "ANALYSED"
    ]

    counts = {}

    for row in analysed:

        cause = row.get(
            "cause",
            "UNKNOWN"
        )

        counts[cause] = (
            counts.get(cause, 0) + 1
        )

    print(
        f"Analysed: {len(analysed)}"
    )

    for cause, count in sorted(
        counts.items()
    ):
        print(
            f"{cause:<40} {count}"
        )

    print()
    print(
        f"Output saved to: {OUTPUT_FILE}"
    )

    print()
    print("=" * 110)


if __name__ == "__main__":
    main()