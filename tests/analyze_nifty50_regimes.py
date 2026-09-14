from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

MODEL7_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model7"
    / "model7_predictions.csv"
)

MODEL8_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model8_return"
    / "model8_return_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model9_regimes"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REGIME_SUMMARY_FILE = (
    OUTPUT_DIR
    / "regime_summary.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR
    / "regime_yearly_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

VOLATILITY_LOOKBACK = 252


# ============================================================
# LOAD BASE FEATURES
# ============================================================

features = pd.read_csv(
    FEATURE_FILE,
    parse_dates=["date"],
)

features = (
    features
    .sort_values("date")
    .reset_index(drop=True)
)


required_feature_columns = [
    "date",
    "nifty_close_ema50_ratio",
    "nifty_atr_pct",
    "nifty_return_1d",
]

missing_features = [
    column
    for column in required_feature_columns
    if column not in features.columns
]

if missing_features:
    raise ValueError(
        "Missing required feature columns: "
        f"{missing_features}"
    )


# ============================================================
# DEFINE REGIMES
#
# IMPORTANT:
# These definitions are fixed BEFORE evaluating performance.
#
# TREND:
#   close / EMA50 >= 1.0 -> UPTREND
#   close / EMA50 <  1.0 -> DOWNTREND
#
# VOLATILITY:
#   ATR% >= trailing 252-day median -> HIGH_VOL
#   ATR% <  trailing 252-day median -> LOW_VOL
#
# The rolling median uses information available through T only.
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
# LOAD MODEL 7
# ============================================================

model7 = pd.read_csv(
    MODEL7_FILE,
    parse_dates=["date"],
)

model7 = (
    model7
    .sort_values("date")
    .reset_index(drop=True)
)


required_model7_columns = [
    "date",
    "actual",
    "probability_up",
    "prediction",
]

missing_model7 = [
    column
    for column in required_model7_columns
    if column not in model7.columns
]

if missing_model7:
    raise ValueError(
        "Missing required Model 7 columns: "
        f"{missing_model7}"
    )


# ============================================================
# LOAD MODEL 8
# ============================================================

model8 = pd.read_csv(
    MODEL8_FILE,
    parse_dates=["date"],
)

model8 = (
    model8
    .sort_values("date")
    .reset_index(drop=True)
)


required_model8_columns = [
    "date",
    "actual_return",
    "predicted_return",
]

missing_model8 = [
    column
    for column in required_model8_columns
    if column not in model8.columns
]

if missing_model8:
    raise ValueError(
        "Missing required Model 8 columns: "
        f"{missing_model8}"
    )


# ============================================================
# MERGE MODEL OUTPUTS WITH REGIMES
# ============================================================

analysis = (
    model7
    .merge(
        model8,
        on="date",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        features[
            [
                "date",
                "nifty_return_1d",
                "nifty_close_ema50_ratio",
                "nifty_atr_pct",
                "volatility_median_252d",
                "trend_regime",
                "volatility_regime",
                "regime",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# VALIDATION
# ============================================================

if analysis.empty:
    raise ValueError(
        "No overlapping dates between Model 7, "
        "Model 8, and feature data."
    )

if analysis["regime"].isna().any():
    missing_count = int(
        analysis["regime"].isna().sum()
    )

    raise ValueError(
        "Missing regime classification for "
        f"{missing_count} observations."
    )


print("=" * 80)
print("NIFTY 50 MODEL 9")
print("REGIME-CONDITIONED ANALYSIS")
print("=" * 80)

print(
    f"Feature coverage: "
    f"{features['date'].min().date()} -> "
    f"{features['date'].max().date()}"
)

print(
    f"Model 7 observations: "
    f"{len(model7)}"
)

print(
    f"Model 8 observations: "
    f"{len(model8)}"
)

print(
    f"Overlapping observations: "
    f"{len(analysis)}"
)

print()

print("REGIME DEFINITIONS")
print(
    "Trend: close / EMA50 >= 1.0 -> UPTREND"
)
print(
    "Trend: close / EMA50 <  1.0 -> DOWNTREND"
)
print(
    "Volatility: ATR% >= trailing 252D median -> HIGH_VOL"
)
print(
    "Volatility: ATR% <  trailing 252D median -> LOW_VOL"
)

print()


# ============================================================
# PREVIOUS-DAY DIRECTION BASELINE
# ============================================================

analysis["previous_day_prediction"] = (
    analysis["nifty_return_1d"] > 0
).astype(int)


# ============================================================
# ZERO-RETURN MAGNITUDE BASELINE
# ============================================================

analysis["zero_return_prediction"] = 0.0


# ============================================================
# REGIME ANALYSIS
# ============================================================

regime_results = []

regime_order = [
    "UPTREND_LOW_VOL",
    "UPTREND_HIGH_VOL",
    "DOWNTREND_LOW_VOL",
    "DOWNTREND_HIGH_VOL",
]

for regime in regime_order:

    group = analysis[
        analysis["regime"] == regime
    ].copy()

    if group.empty:
        continue

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction_actual = (
        group["actual"]
        .astype(int)
    )

    direction_model7 = (
        group["prediction"]
        .astype(int)
    )

    direction_baseline = (
        group["previous_day_prediction"]
        .astype(int)
    )

    model7_accuracy = accuracy_score(
        direction_actual,
        direction_model7,
    )

    model7_balanced_accuracy = (
        balanced_accuracy_score(
            direction_actual,
            direction_model7,
        )
    )

    model7_auc = np.nan

    if direction_actual.nunique() == 2:
        model7_auc = roc_auc_score(
            direction_actual,
            group["probability_up"],
        )

    baseline_accuracy = accuracy_score(
        direction_actual,
        direction_baseline,
    )

    baseline_balanced_accuracy = (
        balanced_accuracy_score(
            direction_actual,
            direction_baseline,
        )
    )

    # --------------------------------------------------------
    # Magnitude
    # --------------------------------------------------------

    return_actual = (
        group["actual_return"]
        .astype(float)
    )

    return_model8 = (
        group["predicted_return"]
        .astype(float)
    )

    return_zero = np.zeros(
        len(group)
    )

    model8_mae = mean_absolute_error(
        return_actual,
        return_model8,
    )

    model8_rmse = np.sqrt(
        mean_squared_error(
            return_actual,
            return_model8,
        )
    )

    model8_r2 = r2_score(
        return_actual,
        return_model8,
    )

    model8_correlation = (
        return_actual.corr(
            return_model8
        )
    )

    zero_mae = mean_absolute_error(
        return_actual,
        return_zero,
    )

    zero_rmse = np.sqrt(
        mean_squared_error(
            return_actual,
            return_zero,
        )
    )

    zero_r2 = r2_score(
        return_actual,
        return_zero,
    )

    regime_results.append(
        {
            "regime": regime,
            "observations": len(group),

            "direction_actual_positive_rate": (
                direction_actual.mean()
            ),

            "model7_accuracy": (
                model7_accuracy
            ),

            "model7_balanced_accuracy": (
                model7_balanced_accuracy
            ),

            "model7_roc_auc": (
                model7_auc
            ),

            "previous_day_accuracy": (
                baseline_accuracy
            ),

            "previous_day_balanced_accuracy": (
                baseline_balanced_accuracy
            ),

            "model7_accuracy_delta_vs_previous_day": (
                model7_accuracy
                - baseline_accuracy
            ),

            "model7_balanced_accuracy_delta_vs_previous_day": (
                model7_balanced_accuracy
                - baseline_balanced_accuracy
            ),

            "model8_mae": model8_mae,

            "model8_rmse": model8_rmse,

            "model8_r2": model8_r2,

            "model8_correlation": (
                model8_correlation
            ),

            "zero_return_mae": zero_mae,

            "zero_return_rmse": zero_rmse,

            "zero_return_r2": zero_r2,

            "model8_mae_delta_vs_zero": (
                model8_mae
                - zero_mae
            ),

            "model8_rmse_delta_vs_zero": (
                model8_rmse
                - zero_rmse
            ),
        }
    )


regime_df = pd.DataFrame(
    regime_results
)


# ============================================================
# YEARLY × REGIME RESULTS
# ============================================================

yearly_results = []

for (
    year,
    regime,
), group in (
    analysis
    .groupby(
        [
            analysis["date"].dt.year,
            "regime",
        ]
    )
):

    direction_actual = (
        group["actual"]
        .astype(int)
    )

    direction_model7 = (
        group["prediction"]
        .astype(int)
    )

    direction_baseline = (
        group["previous_day_prediction"]
        .astype(int)
    )

    return_actual = (
        group["actual_return"]
        .astype(float)
    )

    return_model8 = (
        group["predicted_return"]
        .astype(float)
    )

    zero_prediction = np.zeros(
        len(group)
    )

    yearly_results.append(
        {
            "year": int(year),
            "regime": regime,
            "observations": len(group),

            "model7_accuracy": (
                accuracy_score(
                    direction_actual,
                    direction_model7,
                )
            ),

            "model7_balanced_accuracy": (
                balanced_accuracy_score(
                    direction_actual,
                    direction_model7,
                )
            ),

            "previous_day_accuracy": (
                accuracy_score(
                    direction_actual,
                    direction_baseline,
                )
            ),

            "previous_day_balanced_accuracy": (
                balanced_accuracy_score(
                    direction_actual,
                    direction_baseline,
                )
            ),

            "model8_mae": (
                mean_absolute_error(
                    return_actual,
                    return_model8,
                )
            ),

            "zero_return_mae": (
                mean_absolute_error(
                    return_actual,
                    zero_prediction,
                )
            ),

            "model8_rmse": (
                np.sqrt(
                    mean_squared_error(
                        return_actual,
                        return_model8,
                    )
                )
            ),

            "zero_return_rmse": (
                np.sqrt(
                    mean_squared_error(
                        return_actual,
                        zero_prediction,
                    )
                )
            ),
        }
    )


yearly_df = pd.DataFrame(
    yearly_results
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

regime_df.to_csv(
    REGIME_SUMMARY_FILE,
    index=False,
)

yearly_df.to_csv(
    YEARLY_FILE,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 80)
print("REGIME SUMMARY")
print("=" * 80)

print(
    regime_df.to_string(
        index=False
    )
)

print()

print("=" * 80)
print("YEAR × REGIME RESULTS")
print("=" * 80)

print(
    yearly_df.to_string(
        index=False
    )
)

print()

print("=" * 80)
print("REGIME OBSERVATION COUNTS")
print("=" * 80)

print(
    analysis["regime"]
    .value_counts()
    .reindex(regime_order)
    .fillna(0)
    .astype(int)
    .to_string()
)

print()

print("OUTPUT FILES")

print(
    f"Regime summary : "
    f"{REGIME_SUMMARY_FILE}"
)

print(
    f"Yearly results : "
    f"{YEARLY_FILE}"
)

print()

print("=" * 80)
print("MODEL 9 COMPLETE")
print("=" * 80)