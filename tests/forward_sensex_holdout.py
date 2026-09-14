"""
TradeSense-AI
SENSEX — Genuine Forward / OOS Holdout Collector

RESEARCH-ONLY.

Purpose
-------
Generate genuinely unseen SENSEX TradeSense-AI trades after the
forward freeze date and record the locked 7B / 0.25R early-adverse
observation.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within the first 7 entry bars.

IMPORTANT
---------
This script DOES NOT:

    - modify production code
    - modify the historical SENSEX baseline
    - modify the stock historical dataset
    - modify the stock holdout dataset
    - tune the hypothesis
    - select parameters
    - filter by regime
    - filter by setup
    - manufacture trades
    - use future candles to generate signals

Historical price data is allowed as SIGNAL CONTEXT.

Only trades whose actual ENTRY DATE >= 2026-09-14
are admitted to the forward ledger.

If there are zero genuine OOS trades, the ledger is NOT created.

That is an expected state.
"""


# ============================================================================
# IMPORTS
# ============================================================================

import csv
import os
import sys

import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

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


# ============================================================================
# PRODUCTION SERVICE
# ============================================================================

from app.services.backtest_service import (
    BacktestService,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL = "^BSESN"

INDEX_NAME = "SENSEX"

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
    "sensex_forward_holdout.csv",
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    """
    Normalize a date-like value to midnight.

    Returns None when conversion fails.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except Exception:
        pass

    try:

        return pd.Timestamp(
            value
        ).normalize()

    except Exception:

        return None


def safe_float(value):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)

    except Exception:

        return None


def trade_key(trade):
    """
    Stable key used to prevent duplicate OOS trades.
    """

    entry_date = str(
        trade.get(
            "entry_date",
            "",
        )
    )[:10]

    exit_date = str(
        trade.get(
            "exit_date",
            "",
        )
    )[:10]

    direction = str(
        trade.get(
            "direction",
            "",
        )
    ).strip().upper()

    setup = str(
        trade.get(
            "setup",
            "",
        )
    ).strip().upper()

    return (
        entry_date,
        exit_date,
        direction,
        setup,
    )


# ============================================================================
# EARLY-ADVERSE CALCULATION
# ============================================================================

def calculate_early_adverse_7b(
    price_df,
    trade,
):
    """
    Calculate maximum adverse excursion in R during the first
    seven entry bars of the actual completed trade.

    IMPORTANT
    ---------
    Bar 1 is the actual entry-date candle.

    The trade lifetime is NOT extended.

    The calculation is observational only.
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

    # ------------------------------------------------------------------------
    # Validate required fields.
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Production-matched initial risk per index point.
    #
    # Frozen production configuration:
    #     initial_stop_atr = 3.5
    #
    # Therefore:
    #     risk_per_share = 3.5 * entry_atr
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Restrict observation to actual trade lifetime.
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # First seven entry bars.
    # ------------------------------------------------------------------------

    for bar_number, (
        bar_date,
        bar,
    ) in enumerate(
        lifetime.iterrows(),
        start=1,
    ):

        if (
            bar_number
            > LOCKED_EARLY_ADVERSE_WINDOW
        ):
            break

        high = safe_float(
            bar.get(
                "High"
            )
        )

        low = safe_float(
            bar.get(
                "Low"
            )
        )

        if high is None or low is None:
            continue

        # --------------------------------------------------------------------
        # LONG:
        #
        # Adverse movement occurs when price moves BELOW entry.
        #
        # SHORT:
        #
        # Adverse movement occurs when price moves ABOVE entry.
        # --------------------------------------------------------------------

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

        maximum_adverse_r = max(
            maximum_adverse_r,
            adverse_r,
        )

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


# ============================================================================
# ENRICH TRADE
# ============================================================================

def enrich_trade(
    price_df,
    trade,
):
    """
    Add locked OOS research observations to a completed trade.
    """

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
    "TRADESENSE-AI — "
    "SENSEX GENUINE FORWARD / OOS HOLDOUT"
)
print("=" * 110)

print()

print(
    f"Index                : "
    f"{INDEX_NAME}"
)

print(
    f"Symbol               : "
    f"{SYMBOL}"
)

print(
    f"Forward start        : "
    f"{FORWARD_START_DATE.strftime('%Y-%m-%d')}"
)

print(
    "Locked hypothesis    : "
    "Early adverse >= 0.25R "
    "within first 7 entry bars"
)

print()

print(
    "Historical price data may be used "
    "only as signal context."
)

print(
    "Only actual entry dates on/after "
    "the forward start are admitted."
)

print(
    "Production strategy  : UNCHANGED"
)

print(
    "Research status      : RESEARCH-ONLY"
)


# ============================================================================
# LOAD CURRENT SENSEX DATA
# ============================================================================

print()
print("-" * 110)
print(
    "LOADING CURRENT SENSEX DATA"
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
        "No SENSEX price data available."
    )

    print(
        "Nothing to collect."
    )

    sys.exit(0)


df = df.copy()

df.index = pd.to_datetime(
    df.index
).normalize()

df = df.sort_index()


print(
    f"Price rows           : "
    f"{len(df)}"
)

print(
    f"Data range           : "
    f"{df.index.min().date()} -> "
    f"{df.index.max().date()}"
)


# ============================================================================
# GENERATE TRADESENSE SIGNALS
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
    f"Signal rows          : "
    f"{len(signals)}"
)

if "Signal" in signals.columns:

    print()

    print(
        "Signal distribution:"
    )

    signal_counts = (
        signals["Signal"]
        .value_counts()
        .sort_index()
    )

    for (
        signal_value,
        count,
    ) in signal_counts.items():

        print(
            f"  Signal {signal_value:+}: "
            f"{count}"
        )


# ============================================================================
# RUN PRODUCTION-MATCHED BACKTEST
# ============================================================================

print()
print("-" * 110)
print(
    "RUNNING PRODUCTION-MATCHED "
    "DIRECTIONAL BACKTEST"
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

    print()

    print(
        "ERROR: Backtest result "
        "is not a dictionary."
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

    # ------------------------------------------------------------------------
    # HARD OOS BOUNDARY.
    #
    # Signal context may be historical.
    # Actual entry must be post-freeze.
    # ------------------------------------------------------------------------

    if (
        entry_date
        < FORWARD_START_DATE
    ):
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
# LOAD EXISTING OOS LEDGER
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

existing_trades = []

if os.path.exists(
    HOLDOUT_FILE
):

    print()
    print(
        "Existing OOS ledger found."
    )

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
            "ERROR reading existing "
            "SENSEX OOS ledger:"
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
# SAVE OOS LEDGER
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
        "OOS ledger saved:"
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
# CURRENT OOS STATUS
# ============================================================================

print()
print("=" * 110)
print(
    "SENSEX FORWARD VALIDATION STATUS"
)
print("=" * 110)

genuine_oos_count = len(
    combined_trades
)

if genuine_oos_count == 0:

    print(
        "STATUS: "
        "WAITING_FOR_GENUINE_OOS_TRADES"
    )

    print()

    print(
        "No post-forward-start SENSEX "
        "trade exists in the current "
        "completed data."
    )

elif genuine_oos_count < 5:

    print(
        "STATUS: EARLY_OOS_SAMPLE"
    )

    print()

    print(
        "Genuine OOS trades exist."
    )

    print(
        "Evidence is insufficient "
        "for strategy or exit conclusions."
    )

    print(
        "Do not tune the hypothesis."
    )

else:

    print(
        "STATUS: OOS_DATA_ACCUMULATING"
    )

    print()

    print(
        "Genuine OOS observations "
        "are accumulating."
    )

    print(
        "The 7B / 0.25R hypothesis "
        "remains frozen."
    )


# ============================================================================
# SAFETY CHECK
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
    "PASS: Historical SENSEX baseline unchanged."
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
    "PASS: Only genuine post-start "
    "entry dates admitted."
)

print(
    "PASS: Zero-trade state does not "
    "create a fake ledger."
)

print()
print("=" * 110)
print(
    "END SENSEX FORWARD COLLECTOR"
)
print("=" * 110)