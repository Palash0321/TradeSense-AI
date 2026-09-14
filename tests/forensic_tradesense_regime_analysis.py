from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRADE_FILE = (
    BASE_DIR
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tests"
    / "output"
    / "reconciliation"
    / "tradesense_regime_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGIME_SUMMARY_FILE = (
    OUTPUT_DIR
    / "regime_summary.csv"
)

SETUP_REGIME_FILE = (
    OUTPUT_DIR
    / "setup_by_regime.csv"
)

DIRECTION_REGIME_FILE = (
    OUTPUT_DIR
    / "direction_by_regime.csv"
)

YEAR_REGIME_FILE = (
    OUTPUT_DIR
    / "year_by_regime.csv"
)

EARLY_ADVERSE_REGIME_FILE = (
    OUTPUT_DIR
    / "early_adverse_by_regime.csv"
)

TRADE_LEVEL_FILE = (
    OUTPUT_DIR
    / "trade_level_regime_classification.csv"
)


# ============================================================
# FIXED CONFIGURATION
# ============================================================

VOLATILITY_LOOKBACK = 252

EARLY_ADVERSE_WINDOW = 5
EARLY_ADVERSE_THRESHOLD = 0.25


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return np.nan


def calculate_stats(group):
    """
    Calculate basic trading statistics from frozen R-multiples.
    """

    if group.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": np.nan,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": np.nan,
            "total_r": 0.0,
            "avg_r": np.nan,
            "median_r": np.nan,
        }

    r = pd.to_numeric(
        group["r_multiple"],
        errors="coerce",
    ).dropna()

    if r.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": np.nan,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": np.nan,
            "total_r": 0.0,
            "avg_r": np.nan,
            "median_r": np.nan,
        }

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    return {
        "trades": int(len(r)),
        "wins": int(len(wins)),
        "losses": int(len(losses)),
        "win_rate_pct": (
            float(len(wins) / len(r) * 100.0)
        ),
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
        "profit_factor": profit_factor,
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
    }


def print_stats(label, stats):
    print(
        f"{label:<32}"
        f"Trades={stats['trades']:>3} "
        f"W={stats['wins']:>3} "
        f"L={stats['losses']:>3} "
        f"Win={stats['win_rate_pct']:>7.2f}% "
        f"PF={stats['profit_factor']:>7.3f} "
        f"R={stats['total_r']:>8.2f} "
        f"AvgR={stats['avg_r']:>7.3f}"
    )


# ============================================================
# LOAD FROZEN TRADES
# ============================================================

if not TRADE_FILE.exists():
    raise FileNotFoundError(
        f"Frozen trade file not found: {TRADE_FILE}"
    )

trades = pd.read_csv(
    TRADE_FILE,
    parse_dates=[
        "Signal_Date",
        "entry_date",
        "exit_date",
    ],
)

trades = (
    trades
    .sort_values(
        [
            "_symbol",
            "entry_date",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# FROZEN DATASET VALIDATION
# ============================================================

print("=" * 100)
print("TRADESENSE STRATEGY × MARKET REGIME FORENSIC")
print("=" * 100)

print()
print("INPUT VALIDATION")
print("-" * 100)

print(
    f"Frozen trade file : {TRADE_FILE}"
)

print(
    f"Feature file      : {FEATURE_FILE}"
)

print(
    f"Trade rows        : {len(trades)}"
)

if len(trades) != 106:
    raise ValueError(
        "Expected the frozen 106-trade dataset. "
        f"Found {len(trades)} rows."
    )

required_trade_columns = [
    "Signal_Date",
    "entry_date",
    "exit_date",
    "_symbol",
    "direction",
    "setup",
    "r_multiple",
    "mfe_percent",
    "mae_percent",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
]

missing_trade_columns = [
    column
    for column in required_trade_columns
    if column not in trades.columns
]

if missing_trade_columns:
    raise ValueError(
        "Missing required trade columns: "
        f"{missing_trade_columns}"
    )

print("Frozen trade count validation : PASS")


# ============================================================
# LOAD NIFTY FEATURES
# ============================================================

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"NIFTY feature file not found: {FEATURE_FILE}"
    )

features = pd.read_csv(
    FEATURE_FILE,
    parse_dates=["date"],
)

features = (
    features
    .sort_values("date")
    .drop_duplicates(
        subset=["date"],
        keep="last",
    )
    .reset_index(drop=True)
)

required_feature_columns = [
    "date",
    "nifty_close_ema50_ratio",
    "nifty_atr_pct",
    "nifty_return_1d",
]

missing_feature_columns = [
    column
    for column in required_feature_columns
    if column not in features.columns
]

if missing_feature_columns:
    raise ValueError(
        "Missing required feature columns: "
        f"{missing_feature_columns}"
    )


# ============================================================
# DEFINE FIXED REGIMES
#
# IMPORTANT:
# These are the SAME fixed definitions used by Model 9.
#
# Trend:
#   close / EMA50 >= 1.0 -> UPTREND
#   close / EMA50 <  1.0 -> DOWNTREND
#
# Volatility:
#   ATR% >= trailing 252-day median -> HIGH_VOL
#   ATR% <  trailing 252-day median -> LOW_VOL
#
# The volatility median uses information through the
# signal date only.
# ============================================================

features["volatility_median_252d"] = (
    features["nifty_atr_pct"]
    .rolling(
        window=VOLATILITY_LOOKBACK,
        min_periods=VOLATILITY_LOOKBACK,
    )
    .median()
)

features["trend_regime"] = np.where(
    features["nifty_close_ema50_ratio"] >= 1.0,
    "UPTREND",
    "DOWNTREND",
)

features["volatility_regime"] = np.where(
    features["nifty_atr_pct"]
    >= features["volatility_median_252d"],
    "HIGH_VOL",
    "LOW_VOL",
)

features["regime"] = (
    features["trend_regime"]
    + "_"
    + features["volatility_regime"]
)


# ============================================================
# PRIMARY CAUSAL MERGE
#
# We classify the trade using SIGNAL DATE.
#
# Why?
#
# The strategy generates the signal on Signal_Date and enters
# on the following trading bar.
#
# Therefore Signal_Date is the latest completed market state
# available when the trading decision is made.
#
# Using entry_date would risk using information from the
# execution day's close, which is not available at entry.
# ============================================================

regime_features = features[
    [
        "date",
        "nifty_close_ema50_ratio",
        "nifty_atr_pct",
        "volatility_median_252d",
        "trend_regime",
        "volatility_regime",
        "regime",
    ]
].copy()

regime_features = regime_features.rename(
    columns={
        "date": "Signal_Date",
        "nifty_close_ema50_ratio":
            "signal_close_ema50_ratio",
        "nifty_atr_pct":
            "signal_atr_pct",
        "volatility_median_252d":
            "signal_volatility_median_252d",
        "trend_regime":
            "signal_trend_regime",
        "volatility_regime":
            "signal_volatility_regime",
        "regime":
            "signal_regime",
    }
)

analysis = trades.merge(
    regime_features,
    on="Signal_Date",
    how="left",
    validate="many_to_one",
)


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

missing_regime = analysis[
    analysis["signal_regime"].isna()
]

if not missing_regime.empty:
    print()
    print(
        "MISSING REGIME CLASSIFICATION"
    )
    print(
        missing_regime[
            [
                "_symbol",
                "Signal_Date",
                "entry_date",
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "Some frozen trades could not be assigned "
        "a regime using Signal_Date."
    )

if (
    analysis["Signal_Date"]
    >= analysis["entry_date"]
).any():
    raise ValueError(
        "Temporal validation failed: "
        "Signal_Date must precede entry_date."
    )

print(
    "Signal_Date → entry_date temporal ordering : PASS"
)

print(
    "Regime classification coverage             : PASS"
)


# ============================================================
# EARLY-ADVERSE FLAG
# ============================================================

early_columns = [
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
]

early_values = (
    analysis[early_columns]
    .apply(pd.to_numeric, errors="coerce")
)

analysis["early_adverse_5b_025r"] = (
    early_values
    >= EARLY_ADVERSE_THRESHOLD
).any(axis=1)


# ============================================================
# REGIME ORDER
# ============================================================

regime_order = [
    "UPTREND_LOW_VOL",
    "UPTREND_HIGH_VOL",
    "DOWNTREND_LOW_VOL",
    "DOWNTREND_HIGH_VOL",
]


# ============================================================
# REGIME SUMMARY
# ============================================================

print()
print("=" * 100)
print("REGIME SUMMARY")
print("=" * 100)

regime_rows = []

for regime in regime_order:

    group = analysis[
        analysis["signal_regime"] == regime
    ].copy()

    stats = calculate_stats(group)

    mfe = pd.to_numeric(
        group["mfe_percent"],
        errors="coerce",
    ).dropna()

    mae = pd.to_numeric(
        group["mae_percent"],
        errors="coerce",
    ).dropna()

    triggered = group[
        group["early_adverse_5b_025r"]
    ]

    retained = group[
        ~group["early_adverse_5b_025r"]
    ]

    retained_stats = calculate_stats(
        retained
    )

    regime_rows.append(
        {
            "regime": regime,
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate_pct": stats["win_rate_pct"],
            "profit_factor": stats["profit_factor"],
            "total_r": stats["total_r"],
            "avg_r": stats["avg_r"],
            "median_r": stats["median_r"],
            "avg_mfe_percent": (
                float(mfe.mean())
                if not mfe.empty
                else np.nan
            ),
            "avg_mae_percent": (
                float(mae.mean())
                if not mae.empty
                else np.nan
            ),
            "early_adverse_triggered_trades": (
                len(triggered)
            ),
            "early_adverse_trigger_rate_pct": (
                len(triggered)
                / len(group)
                * 100.0
                if len(group) > 0
                else np.nan
            ),
            "retained_trades_5b_025r": (
                retained_stats["trades"]
            ),
            "retained_win_rate_pct_5b_025r": (
                retained_stats["win_rate_pct"]
            ),
            "retained_profit_factor_5b_025r": (
                retained_stats["profit_factor"]
            ),
            "retained_total_r_5b_025r": (
                retained_stats["total_r"]
            ),
            "delta_r_5b_025r": (
                retained_stats["total_r"]
                - stats["total_r"]
            ),
        }
    )

    print()
    print_stats(
        regime,
        stats,
    )

    print(
        f"{'  5B/0.25R retained':<32}"
        f"Trades={retained_stats['trades']:>3} "
        f"W={retained_stats['wins']:>3} "
        f"L={retained_stats['losses']:>3} "
        f"Win={retained_stats['win_rate_pct']:>7.2f}% "
        f"PF={retained_stats['profit_factor']:>7.3f} "
        f"R={retained_stats['total_r']:>8.2f} "
        f"Delta={retained_stats['total_r'] - stats['total_r']:>7.2f}"
    )


regime_summary = pd.DataFrame(
    regime_rows
)

regime_summary.to_csv(
    REGIME_SUMMARY_FILE,
    index=False,
)


# ============================================================
# SETUP × REGIME
# ============================================================

print()
print("=" * 100)
print("SETUP × REGIME")
print("=" * 100)

setup_rows = []

for (
    regime,
    setup,
), group in (
    analysis
    .groupby(
        [
            "signal_regime",
            "setup",
        ],
        dropna=False,
    )
):

    stats = calculate_stats(group)

    setup_rows.append(
        {
            "regime": regime,
            "setup": setup,
            **stats,
        }
    )

setup_regime = pd.DataFrame(
    setup_rows
)

setup_regime = (
    setup_regime
    .sort_values(
        [
            "regime",
            "total_r",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

print(
    setup_regime.to_string(
        index=False
    )
)

setup_regime.to_csv(
    SETUP_REGIME_FILE,
    index=False,
)


# ============================================================
# DIRECTION × REGIME
# ============================================================

print()
print("=" * 100)
print("DIRECTION × REGIME")
print("=" * 100)

direction_rows = []

for (
    regime,
    direction,
), group in (
    analysis
    .groupby(
        [
            "signal_regime",
            "direction",
        ],
        dropna=False,
    )
):

    stats = calculate_stats(group)

    direction_rows.append(
        {
            "regime": regime,
            "direction": direction,
            **stats,
        }
    )

direction_regime = pd.DataFrame(
    direction_rows
)

direction_regime = (
    direction_regime
    .sort_values(
        [
            "regime",
            "total_r",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

print(
    direction_regime.to_string(
        index=False
    )
)

direction_regime.to_csv(
    DIRECTION_REGIME_FILE,
    index=False,
)


# ============================================================
# YEAR × REGIME
# ============================================================

print()
print("=" * 100)
print("YEAR × REGIME")
print("=" * 100)

analysis["year"] = (
    analysis["entry_date"]
    .dt.year
)

year_rows = []

for (
    year,
    regime,
), group in (
    analysis
    .groupby(
        [
            "year",
            "signal_regime",
        ],
        dropna=False,
    )
):

    stats = calculate_stats(group)

    year_rows.append(
        {
            "year": int(year),
            "regime": regime,
            **stats,
        }
    )

year_regime = pd.DataFrame(
    year_rows
)

year_regime = (
    year_regime
    .sort_values(
        [
            "year",
            "regime",
        ]
    )
    .reset_index(drop=True)
)

print(
    year_regime.to_string(
        index=False
    )
)

year_regime.to_csv(
    YEAR_REGIME_FILE,
    index=False,
)


# ============================================================
# EARLY-ADVERSE × REGIME
# ============================================================

print()
print("=" * 100)
print("EARLY-ADVERSE 5B / 0.25R × REGIME")
print("=" * 100)

early_rows = []

for regime in regime_order:

    group = analysis[
        analysis["signal_regime"] == regime
    ].copy()

    for triggered, label in [
        (
            True,
            "TRIGGERED",
        ),
        (
            False,
            "NOT_TRIGGERED",
        ),
    ]:

        subgroup = group[
            group["early_adverse_5b_025r"]
            == triggered
        ]

        stats = calculate_stats(
            subgroup
        )

        early_rows.append(
            {
                "regime": regime,
                "early_adverse_state": label,
                **stats,
            }
        )

early_regime = pd.DataFrame(
    early_rows
)

print(
    early_regime.to_string(
        index=False
    )
)

early_regime.to_csv(
    EARLY_ADVERSE_REGIME_FILE,
    index=False,
)


# ============================================================
# TRADE-LEVEL CLASSIFICATION
# ============================================================

trade_level_columns = [
    "index",
    "_symbol",
    "Signal_Date",
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "signal_trend_regime",
    "signal_volatility_regime",
    "signal_regime",
    "r_multiple",
    "mfe_percent",
    "mae_percent",
    "early_adverse_5b_025r",
]

available_trade_level_columns = [
    column
    for column in trade_level_columns
    if column in analysis.columns
]

trade_level = analysis[
    available_trade_level_columns
].copy()

trade_level.to_csv(
    TRADE_LEVEL_FILE,
    index=False,
)


# ============================================================
# GLOBAL CROSS-CHECK
# ============================================================

print()
print("=" * 100)
print("GLOBAL CROSS-CHECK")
print("=" * 100)

overall_stats = calculate_stats(
    analysis
)

print_stats(
    "ALL FROZEN TRADES",
    overall_stats,
)

print()
print(
    f"Regime classified trades : "
    f"{len(analysis)}"
)

print(
    f"Unclassified trades      : "
    f"{analysis['signal_regime'].isna().sum()}"
)

print(
    f"Unique regimes           : "
    f"{analysis['signal_regime'].nunique()}"
)

print(
    f"Unique symbols           : "
    f"{analysis['_symbol'].nunique()}"
)

print(
    f"Date range               : "
    f"{analysis['entry_date'].min().date()} -> "
    f"{analysis['entry_date'].max().date()}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

regime_count = (
    regime_summary["trades"].sum()
)

if regime_count != len(analysis):
    raise ValueError(
        "Regime counts do not reconcile with "
        "the frozen trade count."
    )

if abs(
    regime_summary["total_r"].sum()
    - overall_stats["total_r"]
) > 1e-9:
    raise ValueError(
        "Regime R totals do not reconcile "
        "with overall frozen R."
    )

print()
print("=" * 100)
print("VALIDATION")
print("=" * 100)

print(
    "Frozen trade count reconciliation : PASS"
)

print(
    "Regime trade count reconciliation : PASS"
)

print(
    "Regime R reconciliation           : PASS"
)

print(
    "Temporal classification           : PASS"
)

print()
print("OUTPUT FILES")
print(
    f"Regime summary       : {REGIME_SUMMARY_FILE}"
)
print(
    f"Setup × regime       : {SETUP_REGIME_FILE}"
)
print(
    f"Direction × regime   : {DIRECTION_REGIME_FILE}"
)
print(
    f"Year × regime        : {YEAR_REGIME_FILE}"
)
print(
    f"Early adverse × reg : {EARLY_ADVERSE_REGIME_FILE}"
)
print(
    f"Trade classifications: {TRADE_LEVEL_FILE}"
)

print()
print("=" * 100)
print("TRADESENSE × REGIME FORENSIC COMPLETE")
print("=" * 100)