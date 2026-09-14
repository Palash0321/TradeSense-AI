from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRADES_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "tradesense_regime_analysis"
)

OUTPUT_INTERACTION = (
    OUTPUT_DIR
    / "early_adverse_regime_interaction.csv"
)

OUTPUT_REGIME_WITHIN_STATE = (
    OUTPUT_DIR
    / "regime_effect_within_early_adverse_state.csv"
)

OUTPUT_STATE_WITHIN_REGIME = (
    OUTPUT_DIR
    / "early_adverse_effect_within_regime.csv"
)

OUTPUT_TRADE_CLASSIFICATION = (
    OUTPUT_DIR
    / "early_adverse_regime_interaction_trade_level.csv"
)

EXPECTED_TRADES = 106
EARLY_ADVERSE_WINDOW = 5
EARLY_ADVERSE_THRESHOLD_R = 0.25
VOLATILITY_LOOKBACK = 252


# =============================================================================
# HELPERS
# =============================================================================

def calculate_stats(df: pd.DataFrame) -> dict:
    """
    Calculate outcome statistics in R.
    """

    trades = len(df)

    if trades == 0:
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

    r = pd.to_numeric(df["r_multiple"], errors="coerce").dropna()

    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())

    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    return {
        "trades": int(len(r)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": (
            wins / len(r) * 100
            if len(r) > 0
            else np.nan
        ),
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
        "profit_factor": profit_factor,
        "total_r": float(r.sum()),
        "avg_r": float(r.mean()),
        "median_r": float(r.median()),
    }


def format_pf(value):
    if pd.isna(value):
        return "NA"
    if np.isinf(value):
        return "inf"
    return f"{value:.3f}"


# =============================================================================
# LOAD FROZEN TRADES
# =============================================================================

print("=" * 100)
print("TRADESENSE EARLY-ADVERSE × MARKET-REGIME INTERACTION FORENSIC")
print("=" * 100)

print("\nINPUT VALIDATION")
print("-" * 100)

trades = pd.read_csv(TRADES_FILE)

print(f"Frozen trade file : {TRADES_FILE}")
print(f"Trade rows        : {len(trades)}")

if len(trades) != EXPECTED_TRADES:
    raise RuntimeError(
        f"Expected {EXPECTED_TRADES} frozen trades, found {len(trades)}"
    )

print("Frozen trade count validation : PASS")


# =============================================================================
# REQUIRED TRADE FIELDS
# =============================================================================

required_trade_columns = [
    "Signal_Date",
    "entry_date",
    "exit_date",
    "_symbol",
    "direction",
    "setup",
    "r_multiple",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
]

missing_trade_columns = [
    col
    for col in required_trade_columns
    if col not in trades.columns
]

if missing_trade_columns:
    raise RuntimeError(
        f"Missing required trade columns: {missing_trade_columns}"
    )


# =============================================================================
# DATE NORMALIZATION
# =============================================================================

trades["Signal_Date"] = pd.to_datetime(
    trades["Signal_Date"],
    errors="coerce",
)

trades["entry_date"] = pd.to_datetime(
    trades["entry_date"],
    errors="coerce",
)

trades["exit_date"] = pd.to_datetime(
    trades["exit_date"],
    errors="coerce",
)

if trades["Signal_Date"].isna().any():
    raise RuntimeError("Invalid Signal_Date values found.")

if trades["entry_date"].isna().any():
    raise RuntimeError("Invalid entry_date values found.")

if trades["exit_date"].isna().any():
    raise RuntimeError("Invalid exit_date values found.")


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================

if not (trades["Signal_Date"] < trades["entry_date"]).all():
    bad = trades.loc[
        ~(
            trades["Signal_Date"]
            < trades["entry_date"]
        )
    ]

    raise RuntimeError(
        "Signal_Date must be strictly before entry_date.\n"
        f"Bad rows:\n{bad}"
    )

print("Signal_Date → entry_date temporal ordering : PASS")


# =============================================================================
# LOAD NIFTY FEATURES
# =============================================================================

features = pd.read_csv(FEATURE_FILE)

required_feature_columns = [
    "date",
    "nifty_close_ema50_ratio",
    "nifty_atr_pct",
]

missing_feature_columns = [
    col
    for col in required_feature_columns
    if col not in features.columns
]

if missing_feature_columns:
    raise RuntimeError(
        f"Missing required feature columns: {missing_feature_columns}"
    )

features["date"] = pd.to_datetime(
    features["date"],
    errors="coerce",
)

features = features.sort_values("date").reset_index(drop=True)


# =============================================================================
# MARKET REGIME DEFINITION
# =============================================================================
#
# IMPORTANT:
# Regime is assigned using Signal_Date.
#
# The strategy generates the signal using information available at the
# signal-date close and enters on the next bar.
#
# Therefore entry_date is deliberately NOT used for causal regime assignment.
#


features["trend_regime"] = np.where(
    features["nifty_close_ema50_ratio"] >= 1.0,
    "UPTREND",
    "DOWNTREND",
)

rolling_median_atr = (
    features["nifty_atr_pct"]
    .rolling(
        VOLATILITY_LOOKBACK,
        min_periods=VOLATILITY_LOOKBACK,
    )
    .median()
)

features["volatility_regime"] = np.where(
    features["nifty_atr_pct"] >= rolling_median_atr,
    "HIGH_VOL",
    "LOW_VOL",
)

features["regime"] = (
    features["trend_regime"]
    + "_"
    + features["volatility_regime"]
)


regime_features = features[
    [
        "date",
        "regime",
    ]
].copy()

regime_features = regime_features.rename(
    columns={
        "date": "Signal_Date",
    }
)


# =============================================================================
# MERGE REGIME TO TRADES
# =============================================================================

analysis = trades.merge(
    regime_features,
    on="Signal_Date",
    how="left",
    validate="many_to_one",
)

if analysis["regime"].isna().any():
    missing = analysis.loc[
        analysis["regime"].isna(),
        [
            "_symbol",
            "Signal_Date",
            "entry_date",
        ],
    ]

    raise RuntimeError(
        "Some trades could not be assigned a market regime.\n"
        f"{missing}"
    )

print("Regime classification coverage : PASS")


# =============================================================================
# EARLY-ADVERSE CLASSIFICATION
# =============================================================================

early_adverse_columns = [
    f"early_adverse_r_bar_{i}"
    for i in range(1, EARLY_ADVERSE_WINDOW + 1)
]

for col in early_adverse_columns:
    analysis[col] = pd.to_numeric(
        analysis[col],
        errors="coerce",
    )

analysis["max_early_adverse_r_5b"] = (
    analysis[early_adverse_columns]
    .max(axis=1)
)

analysis["early_adverse_triggered"] = (
    analysis["max_early_adverse_r_5b"]
    >= EARLY_ADVERSE_THRESHOLD_R
)

analysis["early_adverse_state"] = np.where(
    analysis["early_adverse_triggered"],
    "TRIGGERED",
    "NOT_TRIGGERED",
)


# =============================================================================
# GLOBAL CROSS-CHECK
# =============================================================================

overall = calculate_stats(analysis)

if overall["trades"] != EXPECTED_TRADES:
    raise RuntimeError(
        "Global trade count changed after classification."
    )

print("\nGLOBAL BASELINE")
print("-" * 100)
print(
    f"Trades={overall['trades']:3d} "
    f"W={overall['wins']:3d} "
    f"L={overall['losses']:3d} "
    f"Win={overall['win_rate_pct']:6.2f}% "
    f"PF={format_pf(overall['profit_factor']):>6} "
    f"R={overall['total_r']:8.2f}"
)


# =============================================================================
# 2 × 4 INTERACTION MATRIX
# =============================================================================

interaction_rows = []

regime_order = [
    "UPTREND_LOW_VOL",
    "UPTREND_HIGH_VOL",
    "DOWNTREND_LOW_VOL",
    "DOWNTREND_HIGH_VOL",
]

state_order = [
    "TRIGGERED",
    "NOT_TRIGGERED",
]

for regime in regime_order:

    for state in state_order:

        subset = analysis[
            (analysis["regime"] == regime)
            &
            (analysis["early_adverse_state"] == state)
        ]

        stats = calculate_stats(subset)

        interaction_rows.append(
            {
                "regime": regime,
                "early_adverse_state": state,
                **stats,
                "avg_max_early_adverse_r_5b": (
                    float(
                        subset["max_early_adverse_r_5b"].mean()
                    )
                    if len(subset) > 0
                    else np.nan
                ),
            }
        )

interaction = pd.DataFrame(interaction_rows)


# =============================================================================
# REGIME EFFECT WITHIN EACH EARLY-ADVERSE STATE
# =============================================================================
#
# For each early-adverse state:
#
#   best regime R - worst regime R
#
# This tells us whether regime still creates meaningful separation AFTER
# conditioning on early-adverse behavior.
#

regime_effect_rows = []

for state in state_order:

    state_data = interaction[
        interaction["early_adverse_state"] == state
    ].copy()

    state_data = state_data[
        state_data["trades"] > 0
    ]

    if len(state_data) == 0:
        continue

    best_idx = state_data["total_r"].idxmax()
    worst_idx = state_data["total_r"].idxmin()

    best = state_data.loc[best_idx]
    worst = state_data.loc[worst_idx]

    regime_effect_rows.append(
        {
            "early_adverse_state": state,
            "best_regime": best["regime"],
            "best_regime_trades": int(best["trades"]),
            "best_regime_total_r": float(best["total_r"]),
            "worst_regime": worst["regime"],
            "worst_regime_trades": int(worst["trades"]),
            "worst_regime_total_r": float(worst["total_r"]),
            "regime_spread_r": (
                float(best["total_r"])
                - float(worst["total_r"])
            ),
        }
    )

regime_effect = pd.DataFrame(
    regime_effect_rows
)


# =============================================================================
# EARLY-ADVERSE EFFECT WITHIN EACH REGIME
# =============================================================================
#
# This is the reverse comparison:
#
#   NOT_TRIGGERED R - TRIGGERED R
#
# inside each regime.
#
# If this remains strongly positive across regimes, it supports early
# adverse as the stronger discriminator.
#

state_effect_rows = []

for regime in regime_order:

    regime_data = interaction[
        interaction["regime"] == regime
    ].copy()

    triggered = regime_data[
        regime_data["early_adverse_state"] == "TRIGGERED"
    ]

    not_triggered = regime_data[
        regime_data["early_adverse_state"] == "NOT_TRIGGERED"
    ]

    if len(triggered) != 1 or len(not_triggered) != 1:
        continue

    triggered_row = triggered.iloc[0]
    not_triggered_row = not_triggered.iloc[0]

    state_effect_rows.append(
        {
            "regime": regime,
            "triggered_trades": int(
                triggered_row["trades"]
            ),
            "triggered_total_r": float(
                triggered_row["total_r"]
            ),
            "not_triggered_trades": int(
                not_triggered_row["trades"]
            ),
            "not_triggered_total_r": float(
                not_triggered_row["total_r"]
            ),
            "early_adverse_state_spread_r": (
                float(not_triggered_row["total_r"])
                - float(triggered_row["total_r"])
            ),
        }
    )

state_effect = pd.DataFrame(
    state_effect_rows
)


# =============================================================================
# TRADE-LEVEL OUTPUT
# =============================================================================

trade_columns = [
    "_symbol",
    "Signal_Date",
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "regime",
    "early_adverse_state",
    "max_early_adverse_r_5b",
    "r_multiple",
]

trade_level = analysis[
    trade_columns
].copy()

trade_level = trade_level.sort_values(
    [
        "Signal_Date",
        "_symbol",
    ]
).reset_index(drop=True)


# =============================================================================
# RECONCILIATION
# =============================================================================

interaction_trade_count = int(
    interaction["trades"].sum()
)

interaction_total_r = float(
    interaction["total_r"].sum()
)

if interaction_trade_count != EXPECTED_TRADES:
    raise RuntimeError(
        "Interaction matrix trade count reconciliation FAILED."
    )

if not np.isclose(
    interaction_total_r,
    overall["total_r"],
    atol=1e-6,
):
    raise RuntimeError(
        "Interaction matrix R reconciliation FAILED.\n"
        f"Matrix R={interaction_total_r}\n"
        f"Overall R={overall['total_r']}"
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

interaction.to_csv(
    OUTPUT_INTERACTION,
    index=False,
)

regime_effect.to_csv(
    OUTPUT_REGIME_WITHIN_STATE,
    index=False,
)

state_effect.to_csv(
    OUTPUT_STATE_WITHIN_REGIME,
    index=False,
)

trade_level.to_csv(
    OUTPUT_TRADE_CLASSIFICATION,
    index=False,
)


# =============================================================================
# PRINT INTERACTION MATRIX
# =============================================================================

print("\n")
print("=" * 100)
print("2 × 4 EARLY-ADVERSE × REGIME INTERACTION")
print("=" * 100)

for regime in regime_order:

    print(f"\n{regime}")

    regime_rows = interaction[
        interaction["regime"] == regime
    ]

    for state in state_order:

        row = regime_rows[
            regime_rows["early_adverse_state"] == state
        ]

        if len(row) == 0:
            continue

        row = row.iloc[0]

        print(
            f"  {state:<14} "
            f"Trades={int(row['trades']):3d} "
            f"W={int(row['wins']):2d} "
            f"L={int(row['losses']):2d} "
            f"Win={row['win_rate_pct']:6.2f}% "
            f"PF={format_pf(row['profit_factor']):>6} "
            f"R={row['total_r']:8.2f}"
        )


# =============================================================================
# PRINT REGIME EFFECT
# =============================================================================

print("\n")
print("=" * 100)
print("REGIME EFFECT WITHIN EARLY-ADVERSE STATE")
print("=" * 100)

for _, row in regime_effect.iterrows():

    print(
        f"{row['early_adverse_state']:<14} "
        f"Best={row['best_regime']:<22} "
        f"{row['best_regime_total_r']:7.2f}R "
        f"vs "
        f"Worst={row['worst_regime']:<22} "
        f"{row['worst_regime_total_r']:7.2f}R "
        f"Spread={row['regime_spread_r']:7.2f}R"
    )


# =============================================================================
# PRINT EARLY-ADVERSE EFFECT
# =============================================================================

print("\n")
print("=" * 100)
print("EARLY-ADVERSE EFFECT WITHIN EACH REGIME")
print("=" * 100)

for _, row in state_effect.iterrows():

    print(
        f"{row['regime']:<22} "
        f"Triggered={int(row['triggered_trades']):2d} "
        f"{row['triggered_total_r']:7.2f}R "
        f"| "
        f"NotTriggered={int(row['not_triggered_trades']):2d} "
        f"{row['not_triggered_total_r']:7.2f}R "
        f"| "
        f"StateSpread={row['early_adverse_state_spread_r']:7.2f}R"
    )


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print("\n")
print("=" * 100)
print("VALIDATION")
print("=" * 100)

print("Frozen trade count reconciliation : PASS")
print("Interaction trade count          : PASS")
print("Interaction R reconciliation     : PASS")
print("Temporal classification           : PASS")
print("Regime classification             : PASS")
print("Early-adverse classification      : PASS")

print("\nOUTPUT FILES")
print("-" * 100)

print(
    f"Interaction matrix      : {OUTPUT_INTERACTION}"
)

print(
    f"Regime effect by state  : {OUTPUT_REGIME_WITHIN_STATE}"
)

print(
    f"State effect by regime  : {OUTPUT_STATE_WITHIN_REGIME}"
)

print(
    f"Trade classifications   : {OUTPUT_TRADE_CLASSIFICATION}"
)

print("\n")
print("=" * 100)
print("EARLY-ADVERSE × REGIME INTERACTION FORENSIC COMPLETE")
print("=" * 100)