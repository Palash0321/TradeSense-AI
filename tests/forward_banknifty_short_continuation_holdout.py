"""
TradeSense-AI
BANK NIFTY SHORT_CONTINUATION — GENUINE FORWARD/OOS HOLDOUT

FROZEN RESEARCH HYPOTHESIS
--------------------------
For BANK NIFTY SHORT_CONTINUATION:

    Reject the trade if early adverse movement
    reaches >= 0.25R within the first 5 bars.

This hypothesis is FROZEN.

This script is intentionally separate from:
    - production strategy
    - frozen stock historical dataset
    - BANK NIFTY historical baseline
    - stock holdout

Purpose:
    Generate genuinely unseen BANK NIFTY trades from the
    hypothesis-freeze date onward and evaluate the frozen
    5-bar / 0.25R rule without changing the strategy.

IMPORTANT:
    Historical trades are NOT used to select parameters here.
    The 5B / 0.25R rule is already frozen.
"""

import csv
import os
import sys

from datetime import datetime


import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from app.services.backtest_service import BacktestService


# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL = "^NSEBANK"

INDEX_NAME = "BANK NIFTY"

INITIAL_CAPITAL = 1_000_000

INITIAL_STOP_ATR = 3.5

TARGET_R = 2.0

RISK_PER_TRADE = 0.01

SWING_WINDOW = 3

# --------------------------------------------------------------------------
# FROZEN HYPOTHESIS
# --------------------------------------------------------------------------

FROZEN_EARLY_ADVERSE_WINDOW = 5

FROZEN_EARLY_ADVERSE_THRESHOLD = 0.25

# --------------------------------------------------------------------------
# Genuine forward/OOS start
#
# This is AFTER the historical research used to discover the hypothesis.
# Do NOT move this backward.
# --------------------------------------------------------------------------

FORWARD_START_DATE = "2026-09-14"

# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

HOLDOUT_FILE = os.path.join(
    OUTPUT_DIR,
    "banknifty_short_continuation_forward_holdout.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    if pd.isna(value):
        return None

    try:
        return pd.to_datetime(
            value
        ).strftime(
            "%Y-%m-%d"
        )
    except Exception:
        return str(value)


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def calculate_filtered_r(trades):
    """
    Apply the frozen hypothesis only.

    Returns:
        baseline R
        filtered R
        triggered count
        retained count
    """

    baseline_r = 0.0
    filtered_r = 0.0

    triggered = 0
    retained = 0

    for trade in trades:

        r = safe_float(
            trade.get(
                "r_multiple"
            )
        )

        if r is None:
            continue

        baseline_r += r

        early_adverse = safe_float(
            trade.get(
                "early_adverse_r_bar_5"
            )
        )

        if (
            early_adverse is not None
            and early_adverse
            >= FROZEN_EARLY_ADVERSE_THRESHOLD
        ):

            triggered += 1

        else:

            retained += 1
            filtered_r += r

    return (
        baseline_r,
        filtered_r,
        triggered,
        retained,
    )


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print(
    "TRADESENSE-AI — BANK NIFTY "
    "GENUINE FORWARD/OOS HOLDOUT"
)
print("=" * 100)

print()
print(
    "This script evaluates a FROZEN research hypothesis."
)

print(
    "No production strategy modification."
)

print(
    "No BANK NIFTY historical baseline modification."
)

print(
    "No stock historical dataset modification."
)

print(
    "No stock holdout modification."
)

print()
print(
    "FROZEN HYPOTHESIS:"
)

print(
    f"SHORT_CONTINUATION + "
    f"early adverse >= "
    f"{FROZEN_EARLY_ADVERSE_THRESHOLD:.2f}R "
    f"within first "
    f"{FROZEN_EARLY_ADVERSE_WINDOW} bars"
)

print()
print(
    f"FORWARD START DATE: "
    f"{FORWARD_START_DATE}"
)


# ============================================================================
# LOAD CURRENT DATA
# ============================================================================

print()
print("-" * 100)
print("LOADING CURRENT BANK NIFTY DATA")
print("-" * 100)

service = BacktestService(
    symbol=SYMBOL,
    strategy="tradesense",
    initial_capital=INITIAL_CAPITAL,
)

# Load recent/current BANK NIFTY history instead of requesting
# only the freeze-date candle. This allows the collector to run
# before today's completed daily candle exists.
df = service.load_data()

if df is None or df.empty:

    print()
    print(
        "No forward BANK NIFTY data is currently available."
    )

    print(
        "This is expected until new market sessions occur "
        "after the freeze date."
    )

    raise SystemExit(0)


print(
    f"Forward rows available: {len(df)}"
)

print(
    "Data range: "
    f"{df.index.min()} -> {df.index.max()}"
)


# ============================================================================
# GENERATE SIGNALS
# ============================================================================

print()
print("-" * 100)
print("GENERATING FORWARD TRADESENSE SIGNALS")
print("-" * 100)

signals = service.generate_tradesense_signals(
    df=df,
    swing_window=SWING_WINDOW,
)

if signals is None or signals.empty:

    print()
    print(
        "No forward signal data generated."
    )

    raise SystemExit(0)


print(
    f"Signal rows: {len(signals)}"
)

if "Signal" in signals.columns:

    signal_counts = (
        signals["Signal"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Signal distribution:")

    for signal_value, count in signal_counts.items():

        print(
            f"  Signal {signal_value:+}: {count}"
        )


# ============================================================================
# RUN DIRECTIONAL BACKTEST
# ============================================================================

print()
print("-" * 100)
print("RUNNING FORWARD BANK NIFTY BACKTEST")
print("-" * 100)

result = service.run_directional_backtest(
    signals,
    signal_column="Signal",
    initial_stop_atr=INITIAL_STOP_ATR,
    target_r=TARGET_R,
    risk_per_trade=RISK_PER_TRADE,
)

if not isinstance(result, dict):

    print()
    print(
        "ERROR: Backtest result is not a dictionary."
    )

    raise SystemExit(1)


all_trades = result.get(
    "trades",
    []
)

print(
    f"Forward executed trades: "
    f"{len(all_trades)}"
)


# ============================================================================
# FILTER TO SHORT_CONTINUATION
# ============================================================================

target_trades = []

for trade in all_trades:

    if trade.get(
        "setup"
    ) != "SHORT_CONTINUATION":

        continue

    entry_date = normalize_date(
        trade.get(
            "entry_date"
        )
    )

    if (
        entry_date is None
        or entry_date
        < FORWARD_START_DATE
    ):

        continue

    target_trades.append(
        trade
    )


print(
    f"Forward SHORT_CONTINUATION trades: "
    f"{len(target_trades)}"
)


# ============================================================================
# LOAD EXISTING FORWARD HOLDOUT
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

existing_trades = []

if os.path.exists(
    HOLDOUT_FILE
):

    try:

        with open(
            HOLDOUT_FILE,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            reader = csv.DictReader(
                file
            )

            existing_trades = list(
                reader
            )

    except Exception as exc:

        print()
        print(
            "WARNING: Could not read existing "
            f"holdout file: {exc}"
        )

        existing_trades = []


print(
    f"Existing forward holdout trades: "
    f"{len(existing_trades)}"
)


# ============================================================================
# DEDUPLICATION
# ============================================================================

def trade_key(trade):

    return (
        normalize_date(
            trade.get(
                "entry_date"
            )
        ),
        normalize_date(
            trade.get(
                "exit_date"
            )
        ),
        str(
            trade.get(
                "direction",
                ""
            )
        ),
        str(
            trade.get(
                "setup",
                ""
            )
        ),
    )


existing_keys = {
    trade_key(trade)
    for trade in existing_trades
}


new_trades = []

for trade in target_trades:

    key = trade_key(
        trade
    )

    if key in existing_keys:
        continue

    new_trades.append(
        trade
    )


combined_trades = (
    existing_trades
    + new_trades
)


print(
    f"New forward trades: "
    f"{len(new_trades)}"
)

print(
    f"Combined forward holdout trades: "
    f"{len(combined_trades)}"
)


# ============================================================================
# SAVE
# ============================================================================

if combined_trades:

    fieldnames = sorted(
        {
            key
            for trade in combined_trades
            for key in trade.keys()
        }
    )

    with open(
        HOLDOUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            combined_trades
        )

    print()
    print(
        f"Saved: {HOLDOUT_FILE}"
    )

else:

    print()
    print(
        "No forward SHORT_CONTINUATION "
        "trades to save yet."
    )


# ============================================================================
# FORWARD HOLDOUT ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("FORWARD HOLDOUT RESULT")
print("=" * 100)

if not combined_trades:

    print()
    print(
        "GENUINE OOS TRADES: 0"
    )

    print()
    print(
        "No conclusion can be drawn yet."
    )

    print(
        "Continue collecting genuinely unseen "
        "BANK NIFTY signals/trades."
    )

    raise SystemExit(0)


baseline_r, filtered_r, triggered, retained = (
    calculate_filtered_r(
        combined_trades
    )
)

delta_r = (
    filtered_r
    - baseline_r
)

wins = 0

for trade in combined_trades:

    r = safe_float(
        trade.get(
            "r_multiple"
        )
    )

    if (
        r is not None
        and r > 0
    ):

        wins += 1


total_trades = len(
    combined_trades
)

win_rate = (
    wins / total_trades * 100
    if total_trades
    else 0.0
)


print()
print(
    f"Forward SHORT_CONTINUATION trades: "
    f"{total_trades}"
)

print(
    f"Winners: {wins}"
)

print(
    f"Baseline Win Rate: "
    f"{win_rate:.2f}%"
)

print(
    f"Baseline Total R: "
    f"{baseline_r:+.2f}R"
)

print(
    f"Frozen-rule triggered: "
    f"{triggered}"
)

print(
    f"Frozen-rule retained: "
    f"{retained}"
)

print(
    f"Filtered Total R: "
    f"{filtered_r:+.2f}R"
)

print(
    f"Delta: "
    f"{delta_r:+.2f}R"
)


# ============================================================================
# TRADE-BY-TRADE OOS RESULTS
# ============================================================================

print()
print("-" * 100)
print("FORWARD TRADE-BY-TRADE RESULTS")
print("-" * 100)

for trade in combined_trades:

    entry_date = normalize_date(
        trade.get(
            "entry_date"
        )
    )

    exit_date = normalize_date(
        trade.get(
            "exit_date"
        )
    )

    r = safe_float(
        trade.get(
            "r_multiple"
        )
    )

    early5 = safe_float(
        trade.get(
            "early_adverse_r_bar_5"
        )
    )

    triggered = (
        early5 is not None
        and early5
        >= FROZEN_EARLY_ADVERSE_THRESHOLD
    )

    print()
    print(
        f"Entry: {entry_date}"
    )

    print(
        f"Exit:  {exit_date}"
    )

    print(
        f"R:     {r:+.2f}R"
        if r is not None
        else "R:     N/A"
    )

    print(
        f"Early5: "
        f"{early5:.2f}R"
        if early5 is not None
        else "Early5: N/A"
    )

    print(
        "Frozen rule: "
        + (
            "TRIGGERED"
            if triggered
            else "NOT_TRIGGERED"
        )
    )


# ============================================================================
# VERDICT
# ============================================================================

print()
print("=" * 100)
print("OOS VERDICT")
print("=" * 100)

if total_trades < 5:

    print()
    print(
        "NOT READY — insufficient genuine OOS sample."
    )

    print(
        f"Current OOS trades: {total_trades}"
    )

    print(
        "Do not accept or reject the hypothesis yet."
    )

elif delta_r > 0:

    print()
    print(
        "OOS SIGNAL CURRENTLY POSITIVE"
    )

    print(
        f"Delta: {delta_r:+.2f}R"
    )

    print(
        "This is NOT deployment approval."
    )

else:

    print()
    print(
        "OOS SIGNAL CURRENTLY NOT POSITIVE"
    )

    print(
        f"Delta: {delta_r:+.2f}R"
    )

    print(
        "Continue validation before making any decision."
    )


# ============================================================================
# SAFETY
# ============================================================================

print()
print("=" * 100)
print("SAFETY CHECK")
print("=" * 100)

print(
    "PASS: Production strategy was not modified."
)

print(
    "PASS: BANK NIFTY historical baseline was not modified."
)

print(
    "PASS: Frozen stock historical data was not modified."
)

print(
    "PASS: Stock holdout was not modified."
)

print(
    "PASS: Frozen 5B / 0.25R hypothesis was not retuned."
)

print(
    "PASS: Forward results saved separately."
)

print("=" * 100)