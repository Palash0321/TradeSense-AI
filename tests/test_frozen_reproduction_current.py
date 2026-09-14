import csv
import math
import os

import pandas as pd

from app.services.backtest_service import BacktestService


# ======================================================================
# CONFIGURATION
# ======================================================================

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

# Exact boundary previously proven to reproduce the frozen dataset.
START_DATE = "2021-09-07"
END_DATE = "2026-09-08"

# Frozen historical dataset.
HISTORICAL_FILE = (
    "tests/output/tradesense_trades.csv"
)

# Temporary diagnostic output.
REPRODUCED_FILE = (
    "tests/output/tradesense_trades_current_reproduction.csv"
)


# ======================================================================
# HELPERS
# ======================================================================

def normalize_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return str(value)

    return value


def values_equal(left, right):

    left = normalize_value(left)
    right = normalize_value(right)

    if left is None and right is None:
        return True

    if left is None or right is None:
        return False

    try:
        left_float = float(left)
        right_float = float(right)

        if math.isfinite(left_float) and math.isfinite(right_float):
            return math.isclose(
                left_float,
                right_float,
                rel_tol=1e-9,
                abs_tol=1e-6,
            )

    except (TypeError, ValueError):
        pass

    return str(left) == str(right)


def normalize_trade(trade):

    normalized = {}

    for key, value in trade.items():

        value = normalize_value(value)

        if isinstance(value, pd.Timestamp):
            value = str(value)

        normalized[key] = value

    return normalized


def trade_identity(trade):

    return (
        str(trade.get("_symbol", "")),
        str(trade.get("entry_date", ""))[:10],
        str(trade.get("direction", "")),
    )


# ======================================================================
# LOAD FROZEN DATASET
# ======================================================================

print()
print("=" * 100)
print("TRADESENSE-AI — CURRENT FROZEN DATASET REPRODUCTION")
print("=" * 100)

if not os.path.exists(HISTORICAL_FILE):

    raise FileNotFoundError(
        f"Frozen dataset not found: {HISTORICAL_FILE}"
    )


with open(
    HISTORICAL_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    historical_trades = list(reader)

historical_trades = [
    normalize_trade(trade)
    for trade in historical_trades
]


print()
print("FROZEN DATASET")
print("-" * 100)
print(f"File       : {HISTORICAL_FILE}")
print(f"Trades     : {len(historical_trades)}")


historical_columns = list(
    historical_trades[0].keys()
) if historical_trades else []


print(
    f"Columns    : {len(historical_columns)}"
)


# ======================================================================
# GENERATE USING EXACT HISTORICAL BOUNDARY
# ======================================================================

all_trades = []


for symbol in SYMBOLS:

    print()
    print("-" * 100)
    print(f"PROCESSING: {symbol}")
    print("-" * 100)

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=INITIAL_CAPITAL,
    )

    # IMPORTANT:
    # Explicit historical boundary.
    #
    # We are intentionally NOT using:
    #
    #     service.generate_tradesense_signals()
    #
    # because that uses the default rolling data boundary.
    #
    # This test is specifically checking whether the historical
    # frozen dataset is reproducible under its original boundary.
    # ------------------------------------------------------------------

    historical_data = service.load_data(
        start_date=START_DATE,
        end_date=END_DATE
    )

    print(
        f"Data first : {historical_data.index.min()}"
    )

    print(
        f"Data last  : {historical_data.index.max()}"
    )

    print(
        f"Data rows  : {len(historical_data)}"
    )

    signals = service.generate_tradesense_signals(
        df=historical_data
    )

    print(
        f"Signal rows: {len(signals)}"
    )

    backtest_result = service.run_directional_backtest(
        signals,
        signal_column="Signal",
        initial_stop_atr=3.5,
        target_r=2.0,
        risk_per_trade=0.01,
    )

    trades = backtest_result.get(
        "trades",
        []
    )

    print(
        f"Trades     : {len(trades)}"
    )

    for trade in trades:

        trade["_symbol"] = symbol

    all_trades.extend(trades)


# ======================================================================
# SAVE TEMPORARY REPRODUCTION
# ======================================================================

print()
print("=" * 100)
print("CURRENT REPRODUCTION RESULT")
print("=" * 100)

print(
    f"Start date          : {START_DATE}"
)

print(
    f"Historical trades   : {len(historical_trades)}"
)

print(
    f"Current reproduction: {len(all_trades)}"
)


if all_trades:

    reproduced_columns = sorted(
        {
            key
            for trade in all_trades
            for key in trade.keys()
        }
    )

    with open(
        REPRODUCED_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=reproduced_columns,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(all_trades)


# ======================================================================
# NORMALIZE CURRENT TRADES
# ======================================================================

reproduced_trades = [
    normalize_trade(trade)
    for trade in all_trades
]


# ======================================================================
# IDENTITY COMPARISON
# ======================================================================

historical_map = {
    trade_identity(trade): trade
    for trade in historical_trades
}

reproduced_map = {
    trade_identity(trade): trade
    for trade in reproduced_trades
}


historical_keys = set(
    historical_map.keys()
)

reproduced_keys = set(
    reproduced_map.keys()
)


common_keys = (
    historical_keys
    & reproduced_keys
)

historical_only = (
    historical_keys
    - reproduced_keys
)

reproduced_only = (
    reproduced_keys
    - historical_keys
)


# ======================================================================
# FIELD COMPARISON
# ======================================================================

field_changes = []


for key in sorted(common_keys):

    old_trade = historical_map[key]
    new_trade = reproduced_map[key]

    all_fields = (
        set(old_trade.keys())
        | set(new_trade.keys())
    )

    for field in sorted(all_fields):

        old_value = old_trade.get(field)
        new_value = new_trade.get(field)

        if not values_equal(
            old_value,
            new_value
        ):

            field_changes.append(
                (
                    key,
                    field,
                    old_value,
                    new_value,
                )
            )


# ======================================================================
# PRINT IDENTITY DIFFERENCES
# ======================================================================

print()
print("=" * 100)
print("TRADE IDENTITY COMPARISON")
print("=" * 100)

print(
    f"Historical identities : {len(historical_keys)}"
)

print(
    f"Reproduced identities : {len(reproduced_keys)}"
)

print(
    f"Common identities     : {len(common_keys)}"
)

print(
    f"Historical-only       : {len(historical_only)}"
)

print(
    f"Reproduced-only       : {len(reproduced_only)}"
)


if historical_only:

    print()
    print("HISTORICAL-ONLY TRADES")
    print("-" * 100)

    for key in sorted(historical_only):

        print(
            f"{key[0]:<18} "
            f"{key[1]} "
            f"{key[2]}"
        )


if reproduced_only:

    print()
    print("REPRODUCED-ONLY TRADES")
    print("-" * 100)

    for key in sorted(reproduced_only):

        print(
            f"{key[0]:<18} "
            f"{key[1]} "
            f"{key[2]}"
        )


# ======================================================================
# FIELD DIFFERENCES
# ======================================================================

print()
print("=" * 100)
print("COMMON TRADE FIELD COMPARISON")
print("=" * 100)

print(
    f"Common trades         : {len(common_keys)}"
)

print(
    f"Field changes         : {len(field_changes)}"
)


if field_changes:

    print()
    print("FIRST 30 FIELD CHANGES")
    print("-" * 100)

    for (
        key,
        field,
        old_value,
        new_value,
    ) in field_changes[:30]:

        print(
            f"{key[0]:<18} "
            f"{key[1]} "
            f"{key[2]:<6} "
            f"{field:<30} "
            f"OLD={old_value!r} "
            f"NEW={new_value!r}"
        )


# ======================================================================
# R-MULTIPLE COMPARISON
# ======================================================================

historical_total_r = sum(
    float(
        trade.get("r_multiple", 0)
        or 0
    )
    for trade in historical_trades
)

reproduced_total_r = sum(
    float(
        trade.get("r_multiple", 0)
        or 0
    )
    for trade in reproduced_trades
)


print()
print("=" * 100)
print("R-MULTIPLE COMPARISON")
print("=" * 100)

print(
    f"Historical total R : {historical_total_r:.6f}"
)

print(
    f"Reproduced total R : {reproduced_total_r:.6f}"
)

print(
    f"R difference        : "
    f"{reproduced_total_r - historical_total_r:.6f}"
)


# ======================================================================
# FINAL VERDICT
# ======================================================================

exact_identity_match = (
    historical_keys == reproduced_keys
)

exact_field_match = (
    len(field_changes) == 0
)

exact_trade_count = (
    len(historical_trades)
    == len(reproduced_trades)
)


print()
print("=" * 100)
print("FINAL VERDICT")
print("=" * 100)

print(
    f"Trade count match     : "
    f"{'PASS' if exact_trade_count else 'FAIL'}"
)

print(
    f"Trade identity match  : "
    f"{'PASS' if exact_identity_match else 'FAIL'}"
)

print(
    f"Common field match     : "
    f"{'PASS' if exact_field_match else 'FAIL'}"
)


if (
    exact_trade_count
    and exact_identity_match
    and exact_field_match
):

    print()
    print(
        "RESULT: FROZEN 106-TRADE DATASET "
        "IS CURRENTLY REPRODUCIBLE."
    )

else:

    print()
    print(
        "RESULT: FROZEN DATASET DOES NOT "
        "CURRENTLY REPRODUCE EXACTLY."
    )

print()
print(
    f"Temporary reproduction: "
    f"{REPRODUCED_FILE}"
)

print()
print("=" * 100)
print("IMPORTANT: NO PRODUCTION OR FROZEN DATA WAS MODIFIED")
print("=" * 100)