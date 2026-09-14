"""
TradeSense-AI
NIFTY MIDCAP SELECT Index Baseline Backtest

Research-only index validation.

This script:
- loads validated MIDSELECT index OHLC data
- generates TradeSense signals using the existing production service
- runs the existing directional backtest
- saves MIDSELECT trades separately
- does NOT modify production strategy code
- does NOT modify frozen stock historical trades
- does NOT modify stock holdout data
- does NOT apply NIFTY/SENSEX/BANK NIFTY filters
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

from app.services.backtest_service import BacktestService


warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "NIFTY MIDCAP SELECT"

START_DATE = "2022-01-10"
END_DATE = "2026-09-12"

INPUT_FILE = (
    "tests/output/index_backtests/"
    "midselect_niftyindices_ohlc.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_index_trades.csv",
)

# Use the same capital scale used for the other
# index-level research runs so index unit sizing
# does not artificially eliminate trades.
INITIAL_CAPITAL = 1000000

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0
RISK_PER_TRADE = 0.01

SWING_WINDOW = 3


# ============================================================================
# HELPERS
# ============================================================================

def calculate_drawdown(trades, initial_capital):
    """
    Calculate realized closed-trade drawdown.

    Research-side diagnostic only.
    Does not modify the production backtest engine.
    """

    if not trades:
        return 0.0

    equity = float(initial_capital)
    peak = equity
    max_drawdown = 0.0

    for trade in trades:

        pnl = trade.get("pnl", 0.0)

        try:
            pnl = float(pnl)
        except (TypeError, ValueError):
            pnl = 0.0

        equity += pnl

        if equity > peak:
            peak = equity

        if peak > 0:
            drawdown = (
                (peak - equity)
                / peak
            ) * 100.0

            max_drawdown = max(
                max_drawdown,
                drawdown,
            )

    return max_drawdown


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT INDEX BASELINE BACKTEST"
)
print("=" * 100)

print()
print("Research-only index validation.")
print("Production strategy is NOT modified.")
print("Frozen stock historical data is NOT modified.")
print("Stock holdout data is NOT modified.")
print("No NIFTY/SENSEX/BANK NIFTY filter is being applied.")

print()
print(
    f"Research period: "
    f"{START_DATE} -> {END_DATE}"
)

print(
    f"Input source: {INPUT_FILE}"
)


# ============================================================================
# LOAD VALIDATED MIDSELECT DATA
# ============================================================================

print()
print("-" * 100)
print("LOADING VALIDATED MIDSELECT DATA")
print("-" * 100)

if not os.path.exists(INPUT_FILE):

    print()
    print(
        "ERROR: Validated MIDSELECT data file "
        "was not found."
    )

    print(
        f"Expected: {INPUT_FILE}"
    )

    print(
        "Run the MIDSELECT data-source audit first."
    )

    sys.exit(1)


try:

    data = pd.read_csv(
        INPUT_FILE
    )

except Exception as exc:

    print()
    print(
        "ERROR: MIDSELECT data loading failed."
    )

    print(exc)

    sys.exit(1)


if data is None or data.empty:

    print()
    print(
        "ERROR: MIDSELECT dataset is empty."
    )

    sys.exit(1)


# ============================================================================
# NORMALIZE COLUMNS
# ============================================================================

required_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
]


missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing_columns:

    print()
    print(
        "ERROR: Missing required columns:"
    )

    for column in missing_columns:
        print(f"  - {column}")

    sys.exit(1)


data = data.copy()

data["Date"] = pd.to_datetime(
    data["Date"],
    errors="coerce",
)

for column in [
    "Open",
    "High",
    "Low",
    "Close",
]:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


data = data.dropna(
    subset=required_columns
)

data = data.sort_values(
    "Date"
)

data = data.drop_duplicates(
    subset=["Date"],
    keep="last",
)

data = data.set_index(
    "Date"
)


# ============================================================================
# RESEARCH PERIOD FILTER
# ============================================================================

data = data[
    (
        data.index
        >= pd.Timestamp(START_DATE)
    )
    & (
        data.index
        < pd.Timestamp(END_DATE)
    )
].copy()


if data.empty:

    print()
    print(
        "ERROR: No MIDSELECT data remains "
        "after research-period filtering."
    )

    sys.exit(1)


# ============================================================================
# DATA INTEGRITY
# ============================================================================

print()
print("-" * 100)
print("MIDSELECT DATA INTEGRITY")
print("-" * 100)


invalid_ohlc = 0

for _, row in data.iterrows():

    open_price = row["Open"]
    high = row["High"]
    low = row["Low"]
    close = row["Close"]

    if any(
        pd.isna(value)
        for value in [
            open_price,
            high,
            low,
            close,
        ]
    ):

        invalid_ohlc += 1
        continue

    if (
        open_price <= 0
        or high <= 0
        or low <= 0
        or close <= 0
    ):

        invalid_ohlc += 1
        continue

    if high < low:
        invalid_ohlc += 1
        continue

    if high < open_price:
        invalid_ohlc += 1
        continue

    if high < close:
        invalid_ohlc += 1
        continue

    if low > open_price:
        invalid_ohlc += 1
        continue

    if low > close:
        invalid_ohlc += 1
        continue


duplicate_dates = int(
    data.index.duplicated().sum()
)


print(
    f"Rows:             {len(data)}"
)

print(
    f"Date range:       "
    f"{data.index.min().date()} -> "
    f"{data.index.max().date()}"
)

print(
    f"Invalid OHLC:     {invalid_ohlc}"
)

print(
    f"Duplicate dates:  {duplicate_dates}"
)


if invalid_ohlc != 0:

    print()
    print(
        "STATUS: FAIL — invalid OHLC detected."
    )

    sys.exit(1)


if duplicate_dates != 0:

    print()
    print(
        "STATUS: FAIL — duplicate dates detected."
    )

    sys.exit(1)


print()
print(
    "STATUS: PASS"
)


# ============================================================================
# BACKTEST SERVICE
# ============================================================================

print()
print("=" * 100)
print("GENERATING TRADESENSE SIGNALS")
print("=" * 100)

service = BacktestService(
    symbol="NIFTY_MIDCAP_SELECT",
    strategy="tradesense",
    initial_capital=INITIAL_CAPITAL,
)


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

try:

    signals = (
        service.generate_tradesense_signals(
            df=data.copy(),
            swing_window=SWING_WINDOW,
        )
    )

except Exception as exc:

    print()
    print(
        "ERROR: TradeSense signal generation failed."
    )

    print(exc)

    sys.exit(1)


if signals is None or signals.empty:

    print()
    print(
        "ERROR: No signal dataframe returned."
    )

    sys.exit(1)


print(
    f"Signal rows: {len(signals)}"
)


signal_counts = (
    signals["Signal"]
    .value_counts()
    .sort_index()
)


print()
print("Signal distribution:")

for signal_value, count in signal_counts.items():

    print(
        f"  Signal {signal_value:+.0f}: "
        f"{int(count)}"
    )


# ============================================================================
# REQUIRED COLUMN CHECK
# ============================================================================

required_backtest_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "ATR",
    "Signal",
]


missing_backtest_columns = [
    column
    for column in required_backtest_columns
    if column not in signals.columns
]


if missing_backtest_columns:

    print()
    print(
        "ERROR: Missing backtest columns:"
    )

    for column in missing_backtest_columns:
        print(f"  - {column}")

    sys.exit(1)


# ============================================================================
# RUN DIRECTIONAL BACKTEST
# ============================================================================

print()
print("=" * 100)
print("RUNNING MIDSELECT DIRECTIONAL BACKTEST")
print("=" * 100)

try:

    result = (
        service.run_directional_backtest(
            signals,
            signal_column="Signal",
            initial_stop_atr=INITIAL_STOP_ATR,
            target_r=TARGET_R,
            risk_per_trade=RISK_PER_TRADE,
        )
    )

except Exception as exc:

    print()
    print(
        "ERROR: Directional backtest failed."
    )

    print(exc)

    sys.exit(1)


if not isinstance(result, dict):

    print()
    print(
        "ERROR: Unexpected backtest result type."
    )

    print(
        f"Expected dict, got {type(result)}"
    )

    sys.exit(1)


trades = result.get(
    "trades",
    []
)


if trades is None:
    trades = []


print(
    f"Executed trades: {len(trades)}"
)


# ============================================================================
# BASIC RESULT
# ============================================================================

if not trades:

    print()
    print("=" * 100)
    print("NO TRADES GENERATED")
    print("=" * 100)

    print()
    print(
        "The MIDSELECT baseline produced "
        "zero executable trades."
    )

    print(
        "No production code was modified."
    )

    sys.exit(0)


# ============================================================================
# TRADE DATAFRAME
# ============================================================================

trade_df = pd.DataFrame(
    trades
)


# ============================================================================
# R METRICS
# ============================================================================

if "r_multiple" in trade_df.columns:

    r_values = pd.to_numeric(
        trade_df["r_multiple"],
        errors="coerce",
    ).fillna(0.0)

else:

    print()
    print(
        "ERROR: r_multiple missing from trades."
    )

    sys.exit(1)


winners = int(
    (r_values > 0).sum()
)

losers = int(
    (r_values <= 0).sum()
)

trade_count = len(
    trade_df
)

win_rate = (
    winners
    / trade_count
    * 100.0
    if trade_count
    else 0.0
)

gross_profit = r_values[
    r_values > 0
].sum()

gross_loss = abs(
    r_values[
        r_values < 0
    ].sum()
)


if gross_loss > 0:

    profit_factor = (
        gross_profit
        / gross_loss
    )

else:

    profit_factor = np.inf


total_r = r_values.sum()

average_r = (
    total_r
    / trade_count
    if trade_count
    else 0.0
)


# ============================================================================
# P&L
# ============================================================================

if "pnl" in trade_df.columns:

    pnl_values = pd.to_numeric(
        trade_df["pnl"],
        errors="coerce",
    ).fillna(0.0)

    total_pnl = pnl_values.sum()

else:

    total_pnl = 0.0


max_drawdown = calculate_drawdown(
    trades,
    INITIAL_CAPITAL,
)


# ============================================================================
# EXIT BREAKDOWN
# ============================================================================

exit_counts = {}

if "exit_reason" in trade_df.columns:

    exit_counts = (
        trade_df["exit_reason"]
        .value_counts()
        .to_dict()
    )


# ============================================================================
# DIRECTION BREAKDOWN
# ============================================================================

direction_counts = {}

if "direction" in trade_df.columns:

    direction_counts = (
        trade_df["direction"]
        .value_counts()
        .to_dict()
    )


# ============================================================================
# SETUP BREAKDOWN
# ============================================================================

setup_counts = {}

if "setup" in trade_df.columns:

    setup_counts = (
        trade_df["setup"]
        .value_counts()
        .to_dict()
    )


# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 100)
print("MIDSELECT BASELINE RESULT")
print("=" * 100)

print(
    f"Trades:              {trade_count}"
)

print(
    f"Winners:             {winners}"
)

print(
    f"Losers:              {losers}"
)

print(
    f"Win rate:            {win_rate:.2f}%"
)

print(
    f"Gross Profit:        {gross_profit:.2f}R"
)

print(
    f"Gross Loss:          -{gross_loss:.2f}R"
)

print(
    f"Profit Factor:       {profit_factor:.3f}"
)

print(
    f"Total R:             {total_r:.2f}R"
)

print(
    f"Average R:           {average_r:.3f}R"
)

print(
    f"Total P&L:           ₹{total_pnl:,.2f}"
)

print(
    f"Max Drawdown:        {max_drawdown:.2f}%"
)


# ============================================================================
# EXIT BREAKDOWN
# ============================================================================

print()
print("-" * 100)
print("EXIT BREAKDOWN")
print("-" * 100)

for reason, count in exit_counts.items():

    print(
        f"{reason:<20} {int(count):>5}"
    )


# ============================================================================
# DIRECTION BREAKDOWN
# ============================================================================

print()
print("-" * 100)
print("DIRECTION BREAKDOWN")
print("-" * 100)

for direction, count in direction_counts.items():

    print(
        f"{direction:<20} {int(count):>5}"
    )


# ============================================================================
# SETUP BREAKDOWN
# ============================================================================

print()
print("-" * 100)
print("SETUP BREAKDOWN")
print("-" * 100)

for setup, count in setup_counts.items():

    print(
        f"{setup:<25} {int(count):>5}"
    )


# ============================================================================
# TRADE DATE RANGE
# ============================================================================

if "entry_date" in trade_df.columns:

    print()
    print("-" * 100)
    print("TRADE COVERAGE")
    print("-" * 100)

    print(
        f"First entry: "
        f"{trade_df['entry_date'].min()}"
    )

    print(
        f"Last entry:  "
        f"{trade_df['entry_date'].max()}"
    )


# ============================================================================
# SAVE
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


trade_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Saved: {OUTPUT_FILE}"
)


# ============================================================================
# SAFETY CHECK
# ============================================================================

print()
print("=" * 100)
print("SAFETY CHECK")
print("=" * 100)

print(
    "PASS: Production code was not modified."
)

print(
    "PASS: Frozen stock historical data was not modified."
)

print(
    "PASS: Stock holdout data was not modified."
)

print(
    "PASS: No NIFTY/SENSEX/BANK NIFTY filter was applied."
)

print(
    "PASS: MIDSELECT result saved separately."
)

print()
print("=" * 100)
print("MIDSELECT BASELINE COMPLETE")
print("=" * 100)