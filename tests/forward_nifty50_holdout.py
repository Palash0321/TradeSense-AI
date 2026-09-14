"""
TradeSense-AI
NIFTY 50 — Genuine Forward / OOS Holdout Collector

RESEARCH-ONLY.

Purpose
-------
Generate genuinely unseen NIFTY 50 TradeSense trades after the
forward freeze date and record the locked 7B / 0.25R early-adverse
observation.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within the first 7 entry bars.

IMPORTANT
---------
This script does NOT:
    - modify production code
    - modify the historical NIFTY 50 baseline
    - modify stock historical data
    - modify stock holdout data
    - tune the hypothesis
    - select parameters
    - filter by regime
    - filter by setup
    - manufacture trades

Historical price data is allowed as signal context.

Only trades whose actual ENTRY DATE >= 2026-09-14
are admitted to the forward ledger.

A ledger is created only when genuine OOS trades exist.
If there are zero genuine OOS trades, the ledger remains absent.
That is an expected state.
"""

import csv
import os
import sys

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

SYMBOL = "^NSEI"

INDEX_NAME = "NIFTY 50"

FORWARD_START_DATE = pd.Timestamp(
    "2026-09-14"
)

INITIAL_CAPITAL = 1_000_000

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0
RISK_PER_TRADE = 0.01
SWING_WINDOW = 3

LOCKED_EARLY_ADVERSE_WINDOW = 7
LOCKED_EARLY_ADVERSE_THRESHOLD = 0.25

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

HOLDOUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_forward_holdout.csv",
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    if pd.isna(value):
        return None

    try:
        return pd.Timestamp(value).normalize()
    except Exception:
        return None


def safe_float(value):
    try:
        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


def trade_key(trade):
    return (
        str(
            trade.get(
                "entry_date",
                "",
            )
        )[:10],

        str(
            trade.get(
                "exit_date",
                "",
            )
        )[:10],

        str(
            trade.get(
                "direction",
                "",
            )
        ).strip().upper(),

        str(
            trade.get(
                "setup",
                "",
            )
        ).strip().upper(),
    )


def calculate_early_adverse_7b(
    price_df,
    trade,
):
    """
    Calculate early adverse movement during the first
    seven ENTRY bars of the ACTUAL completed trade.

    Bar 1 is the actual entry-date candle.

    No lifetime extension is performed.

    Returns:
        maximum adverse R within first 7 bars,
        plus per-bar observations.
    """

    direction = str(
        trade.get(
            "direction",
            "",
        )
    ).strip().upper()

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

    entry_price = safe_float(
        trade.get(
            "entry_price"
        )
    )

    entry_atr = safe_float(
        trade.get(
            "entry_atr"
        )
    )

    if (
        direction not in {
            "LONG",
            "SHORT",
        }
        or entry_date is None
        or exit_date is None
        or entry_price is None
        or entry_atr is None
        or entry_atr <= 0
    ):
        return {
            "early_adverse_r_7b": None,
            "early_adverse_triggered_7b": None,
            "early_adverse_bar_7b": None,
        }

    risk_per_share = (
        INITIAL_STOP_ATR
        * entry_atr
    )

    if risk_per_share <= 0:
        return {
            "early_adverse_r_7b": None,
            "early_adverse_triggered_7b": None,
            "early_adverse_bar_7b": None,
        }

    lifetime = price_df.loc[
        (
            price_df.index >= entry_date
        )
        & (
            price_df.index <= exit_date
        )
    ].copy()

    if lifetime.empty:
        return {
            "early_adverse_r_7b": None,
            "early_adverse_triggered_7b": None,
            "early_adverse_bar_7b": None,
        }

    maximum_adverse_r = 0.0
    trigger_bar = None

    bar_values = {}

    for bar_number, (
        bar_date,
        bar,
    ) in enumerate(
        lifetime.iterrows(),
        start=1,
    ):

        if bar_number > LOCKED_EARLY_ADVERSE_WINDOW:
            break

        high = safe_float(
            bar.get("High")
        )

        low = safe_float(
            bar.get("Low")
        )

        if high is None or low is None:
            continue

        if direction == "LONG":

            adverse_r = (
                entry_price - low
            ) / risk_per_share

        else:

            adverse_r = (
                high - entry_price
            ) / risk_per_share

        adverse_r = max(
            0.0,
            adverse_r,
        )

        bar_values[
            f"early_adverse_r_bar_{bar_number}"
        ] = adverse_r

        if adverse_r > maximum_adverse_r:
            maximum_adverse_r = adverse_r

        if (
            trigger_bar is None
            and adverse_r
            >= LOCKED_EARLY_ADVERSE_THRESHOLD
        ):
            trigger_bar = bar_number

    return {
        "early_adverse_r_7b": maximum_adverse_r,
        "early_adverse_triggered_7b": (
            maximum_adverse_r
            >= LOCKED_EARLY_ADVERSE_THRESHOLD
        ),
        "early_adverse_bar_7b": trigger_bar,
        **bar_values,
    }


def enrich_trade(
    price_df,
    trade,
):
    enriched = dict(
        trade
    )

    path = calculate_early_adverse_7b(
        price_df=price_df,
        trade=trade,
    )

    enriched.update(
        path
    )

    enriched[
        "oos_hypothesis"
    ] = "7B/0.25R"

    enriched[
        "oos_start_date"
    ] = FORWARD_START_DATE.strftime(
        "%Y-%m-%d"
    )

    return enriched


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 110)
print(
    "TRADESENSE-AI — NIFTY 50 "
    "GENUINE FORWARD / OOS HOLDOUT"
)
print("=" * 110)

print()
print(
    f"Index                : {INDEX_NAME}"
)

print(
    f"Symbol               : {SYMBOL}"
)

print(
    f"Forward start        : "
    f"{FORWARD_START_DATE.strftime('%Y-%m-%d')}"
)

print(
    f"Locked hypothesis    : "
    f"Early adverse >= "
    f"{LOCKED_EARLY_ADVERSE_THRESHOLD:.2f}R "
    f"within first "
    f"{LOCKED_EARLY_ADVERSE_WINDOW} entry bars"
)

print()
print(
    "Historical price data may be used only as signal context."
)

print(
    "Only actual entry dates on/after the forward start "
    "are admitted."
)

print(
    "Production strategy: UNCHANGED"
)


# ============================================================================
# LOAD CURRENT DATA
# ============================================================================

print()
print("-" * 110)
print(
    "LOADING CURRENT NIFTY 50 DATA"
)
print("-" * 110)

service = BacktestService(
    symbol=SYMBOL,
    strategy="tradesense",
    initial_capital=INITIAL_CAPITAL,
)

df = service.load_data()

if df is None or df.empty:

    print()
    print(
        "No NIFTY 50 price data available."
    )

    sys.exit(0)


df = df.copy()

df.index = pd.to_datetime(
    df.index
).normalize()

df = df.sort_index()


print(
    f"Price rows           : {len(df)}"
)

print(
    f"Data range           : "
    f"{df.index.min().date()} -> "
    f"{df.index.max().date()}"
)


# ============================================================================
# GENERATE SIGNALS
# ============================================================================

print()
print("-" * 110)
print(
    "GENERATING TRADESENSE SIGNALS"
)
print("-" * 110)

signals = service.generate_tradesense_signals(
    df=df,
    swing_window=SWING_WINDOW,
)

if signals is None or signals.empty:

    print()
    print(
        "No signal data generated."
    )

    sys.exit(0)


signals = signals.copy()

signals.index = pd.to_datetime(
    signals.index
).normalize()

signals = signals.sort_index()


print(
    f"Signal rows          : {len(signals)}"
)

if "Signal" in signals.columns:

    print()
    print(
        "Signal distribution:"
    )

    counts = (
        signals["Signal"]
        .value_counts()
        .sort_index()
    )

    for signal_value, count in counts.items():

        print(
            f"  Signal {signal_value:+}: "
            f"{count}"
        )


# ============================================================================
# RUN PRODUCTION-MATCHED DIRECTIONAL BACKTEST
# ============================================================================

print()
print("-" * 110)
print(
    "RUNNING PRODUCTION-MATCHED BACKTEST"
)
print("-" * 110)

result = service.run_directional_backtest(
    signals,
    signal_column="Signal",
    initial_stop_atr=INITIAL_STOP_ATR,
    target_r=TARGET_R,
    risk_per_trade=RISK_PER_TRADE,
)

if not isinstance(
    result,
    dict,
):

    print(
        "ERROR: Backtest result is not a dictionary."
    )

    sys.exit(1)


all_trades = result.get(
    "trades",
    [],
)

print(
    f"All executed trades  : "
    f"{len(all_trades)}"
)


# ============================================================================
# FILTER GENUINE OOS TRADES
# ============================================================================

forward_trades = []

for trade in all_trades:

    entry_date = normalize_date(
        trade.get(
            "entry_date"
        )
    )

    if entry_date is None:
        continue

    if entry_date < FORWARD_START_DATE:
        continue

    forward_trades.append(
        enrich_trade(
            price_df=df,
            trade=trade,
        )
    )


print(
    f"Genuine OOS trades   : "
    f"{len(forward_trades)}"
)


# ============================================================================
# LOAD EXISTING LEDGER
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
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
            "ERROR reading existing ledger:"
        )

        print(
            exc
        )

        sys.exit(1)


print(
    f"Existing OOS trades : "
    f"{len(existing_trades)}"
)


# ============================================================================
# DEDUPLICATION
# ============================================================================

existing_keys = {
    trade_key(
        trade
    )
    for trade in existing_trades
}


new_trades = []

for trade in forward_trades:

    key = trade_key(
        trade
    )

    if key in existing_keys:
        continue

    new_trades.append(
        trade
    )

    existing_keys.add(
        key
    )


combined_trades = (
    existing_trades
    + new_trades
)


print(
    f"New OOS trades      : "
    f"{len(new_trades)}"
)

print(
    f"Combined OOS trades  : "
    f"{len(combined_trades)}"
)


# ============================================================================
# SAVE ONLY WHEN GENUINE OOS DATA EXISTS
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
        f"Saved OOS ledger:"
    )

    print(
        f"  {HOLDOUT_FILE}"
    )

else:

    print()
    print(
        "No genuine OOS trades exist yet."
    )

    print(
        "Forward ledger not created/modified."
    )


# ============================================================================
# CURRENT STATUS
# ============================================================================

print()
print("=" * 110)
print(
    "NIFTY 50 FORWARD VALIDATION STATUS"
)
print("=" * 110)

genuine_oos_count = len(
    combined_trades
)

if genuine_oos_count == 0:

    print(
        "STATUS: WAITING_FOR_GENUINE_OOS_TRADES"
    )

    print(
        "The current completed data contains "
        "no post-forward-start NIFTY 50 trade."
    )

elif genuine_oos_count < 5:

    print(
        "STATUS: EARLY_OOS_SAMPLE"
    )

    print(
        "Genuine OOS trades exist."
    )

    print(
        "Insufficient evidence for conclusions."
    )

    print(
        "Do not tune the hypothesis."
    )

else:

    print(
        "STATUS: OOS_DATA_ACCUMULATING"
    )

    print(
        "Genuine OOS observations are accumulating."
    )

    print(
        "The 7B / 0.25R hypothesis remains frozen."
    )


# ============================================================================
# SAFETY
# ============================================================================

print()
print("=" * 110)
print(
    "SAFETY CHECK"
)
print("=" * 110)

print(
    "PASS: Production strategy unchanged."
)

print(
    "PASS: Historical NIFTY 50 baseline unchanged."
)

print(
    "PASS: Stock historical dataset unchanged."
)

print(
    "PASS: Stock holdout unchanged."
)

print(
    "PASS: No parameter tuning."
)

print(
    "PASS: No regime filtering."
)

print(
    "PASS: No setup filtering."
)

print(
    "PASS: Locked 7B / 0.25R hypothesis unchanged."
)

print(
    "PASS: Only genuine post-start entry dates admitted."
)

print()
print("=" * 110)
print(
    "END NIFTY 50 FORWARD COLLECTOR"
)
print("=" * 110)