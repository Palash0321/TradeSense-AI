from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


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

OUTPUT_DIR = BASE_DIR / "data" / "nifty50_model8_return"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    OUTPUT_DIR / "model8_return_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "model8_return_summary.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR / "model8_return_yearly_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TRAIN_ROWS = 252
RIDGE_ALPHA = 1.0


# ============================================================
# MODEL 8 FEATURES
# SAME 30 FEATURES USED BY MODEL 7
# ============================================================

NIFTY_FEATURES = [
    "nifty_return_1d",
    "nifty_return_3d",
    "nifty_return_5d",
    "nifty_return_10d",
    "nifty_return_20d",
    "nifty_close_ema20_ratio",
    "nifty_close_ema50_ratio",
    "nifty_close_ema200_ratio",
    "nifty_rsi14",
    "nifty_distance_20d_high",
    "nifty_distance_20d_low",
    "nifty_gap_pct",
    "nifty_atr_pct",
    "nifty_rolling_volatility_20d",
    "nifty_volume_ratio_20d",
]

BREADTH_FEATURES = [
    "breadth_1d",
    "breadth_5d",
    "breadth_20d",
    "above_ema20_breadth",
    "above_ema50_breadth",
    "above_ema200_breadth",
]

MOMENTUM_FEATURES = [
    "median_return_1d",
    "median_return_5d",
    "median_return_20d",
    "median_rsi14",
]

RELATIVE_STRENGTH_FEATURES = [
    "relative_strength_breadth",
    "median_relative_strength_20d",
]

VOLUME_FEATURES = [
    "high_volume_up_breadth",
    "high_volume_down_breadth",
    "median_volume_ratio_20d",
]

FEATURES = (
    NIFTY_FEATURES
    + BREADTH_FEATURES
    + MOMENTUM_FEATURES
    + RELATIVE_STRENGTH_FEATURES
    + VOLUME_FEATURES
)

TARGET = "nifty_t1_return"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    FEATURE_FILE,
    parse_dates=["date"],
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


required_columns = (
    ["date", TARGET]
    + FEATURES
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


print("=" * 80)
print("NIFTY 50 MODEL 8")
print("T+1 RETURN MAGNITUDE REGRESSION")
print("=" * 80)

print(f"Feature file: {FEATURE_FILE}")

print(f"Rows: {len(df)}")

print(
    "Coverage: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print(f"Total features: {len(FEATURES)}")
print(f"Target: {TARGET}")
print()


# ============================================================
# USABLE DATA
# ============================================================

model_df = df[
    ["date", TARGET] + FEATURES
].copy()

model_df = model_df.dropna(
    subset=[TARGET] + FEATURES
).reset_index(drop=True)


print(
    f"Usable rows: {len(model_df)}"
)

print(
    "Target mean: "
    f"{model_df[TARGET].mean():.8f}"
)

print(
    "Target median: "
    f"{model_df[TARGET].median():.8f}"
)

print(
    "Target standard deviation: "
    f"{model_df[TARGET].std():.8f}"
)

print()


# ============================================================
# ARRAYS
# ============================================================

X_all = (
    model_df[FEATURES]
    .astype(float)
    .values
)

y_all = (
    model_df[TARGET]
    .astype(float)
    .values
)

dates = model_df["date"].values


# ============================================================
# EXPANDING CHRONOLOGICAL WALK-FORWARD
# ============================================================

predictions = []

for i in range(
    MIN_TRAIN_ROWS,
    len(model_df),
):

    X_train = X_all[:i]
    y_train = y_all[:i]

    X_test = X_all[i:i + 1]
    y_test = y_all[i:i + 1]

    test_date = pd.Timestamp(
        dates[i]
    )

    # --------------------------------------------------------
    # SCALE USING TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_test_scaled = (
        scaler.transform(X_test)
    )

    # --------------------------------------------------------
    # RIDGE REGRESSION
    # --------------------------------------------------------

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    predicted_return = model.predict(
        X_test_scaled
    )[0]

    actual_return = y_test[0]

    predictions.append(
        {
            "date": test_date,
            "actual_return": actual_return,
            "predicted_return": predicted_return,
        }
    )


predictions_df = pd.DataFrame(
    predictions
)


# ============================================================
# MODEL METRICS
# ============================================================

actual = (
    predictions_df["actual_return"]
    .astype(float)
)

predicted = (
    predictions_df["predicted_return"]
    .astype(float)
)


model_mae = mean_absolute_error(
    actual,
    predicted,
)

model_rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted,
    )
)

model_r2 = r2_score(
    actual,
    predicted,
)

model_correlation = (
    actual.corr(predicted)
)


# Directional accuracy

actual_direction = (
    actual > 0
).astype(int)

predicted_direction = (
    predicted > 0
).astype(int)

directional_accuracy = (
    actual_direction
    == predicted_direction
).mean()


# ============================================================
# BASELINE 1
# ZERO RETURN
# ============================================================

zero_prediction = np.zeros(
    len(actual)
)

zero_mae = mean_absolute_error(
    actual,
    zero_prediction,
)

zero_rmse = np.sqrt(
    mean_squared_error(
        actual,
        zero_prediction,
    )
)

zero_r2 = r2_score(
    actual,
    zero_prediction,
)


# ============================================================
# BASELINE 2
# PREVIOUS-DAY RETURN
# ============================================================

previous_day_prediction = (
    model_df.loc[
        MIN_TRAIN_ROWS:,
        "nifty_return_1d"
    ]
    .astype(float)
    .values
)

previous_day_mae = (
    mean_absolute_error(
        actual,
        previous_day_prediction,
    )
)

previous_day_rmse = np.sqrt(
    mean_squared_error(
        actual,
        previous_day_prediction,
    )
)

previous_day_r2 = r2_score(
    actual,
    previous_day_prediction,
)

previous_day_direction = (
    previous_day_prediction > 0
).astype(int)

previous_day_directional_accuracy = (
    previous_day_direction
    == actual_direction
).mean()


# ============================================================
# YEARLY RESULTS
# ============================================================

predictions_df["year"] = (
    predictions_df["date"].dt.year
)

yearly_results = []

for year, group in (
    predictions_df.groupby("year")
):

    year_actual = (
        group["actual_return"]
        .astype(float)
    )

    year_predicted = (
        group["predicted_return"]
        .astype(float)
    )

    year_zero = np.zeros(
        len(group)
    )

    year_previous_day = (
        model_df.loc[
            group.index,
            "nifty_return_1d"
        ]
        .astype(float)
        .values
    )

    year_actual_direction = (
        year_actual > 0
    ).astype(int)

    year_predicted_direction = (
        year_predicted > 0
    ).astype(int)

    yearly_results.append(
        {
            "year": year,
            "observations": len(group),

            "model_mae": mean_absolute_error(
                year_actual,
                year_predicted,
            ),

            "model_rmse": np.sqrt(
                mean_squared_error(
                    year_actual,
                    year_predicted,
                )
            ),

            "model_r2": r2_score(
                year_actual,
                year_predicted,
            ),

            "model_correlation": (
                year_actual.corr(
                    year_predicted
                )
            ),

            "model_directional_accuracy": (
                (
                    year_actual_direction
                    == year_predicted_direction
                ).mean()
            ),

            "zero_return_mae": (
                mean_absolute_error(
                    year_actual,
                    year_zero,
                )
            ),

            "zero_return_rmse": np.sqrt(
                mean_squared_error(
                    year_actual,
                    year_zero,
                )
            ),

            "previous_day_mae": (
                mean_absolute_error(
                    year_actual,
                    year_previous_day,
                )
            ),

            "previous_day_rmse": np.sqrt(
                mean_squared_error(
                    year_actual,
                    year_previous_day,
                )
            ),

            "previous_day_directional_accuracy": (
                (
                    (
                        year_previous_day > 0
                    ).astype(int)
                    == year_actual_direction
                ).mean()
            ),
        }
    )


yearly_df = pd.DataFrame(
    yearly_results
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "model": "Model 8",
            "description": (
                "T+1 return regression using "
                "the same 30 features as Model 7"
            ),

            "target": TARGET,

            "total_feature_rows": len(df),

            "usable_rows": len(model_df),

            "walk_forward_observations": (
                len(predictions_df)
            ),

            "coverage_start": (
                df["date"].min().date()
            ),

            "coverage_end": (
                df["date"].max().date()
            ),

            "feature_count": len(FEATURES),

            "min_train_rows": (
                MIN_TRAIN_ROWS
            ),

            "ridge_alpha": RIDGE_ALPHA,

            "model_mae": model_mae,

            "model_rmse": model_rmse,

            "model_r2": model_r2,

            "model_correlation": (
                model_correlation
            ),

            "model_directional_accuracy": (
                directional_accuracy
            ),

            "zero_return_mae": zero_mae,

            "zero_return_rmse": zero_rmse,

            "zero_return_r2": zero_r2,

            "previous_day_mae": (
                previous_day_mae
            ),

            "previous_day_rmse": (
                previous_day_rmse
            ),

            "previous_day_r2": (
                previous_day_r2
            ),

            "previous_day_directional_accuracy": (
                previous_day_directional_accuracy
            ),
        }
    ]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

predictions_to_save = (
    predictions_df
    .drop(columns=["year"])
)

predictions_to_save.to_csv(
    PREDICTIONS_FILE,
    index=False,
)

summary_df.to_csv(
    SUMMARY_FILE,
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
print("MODEL 8 RESULTS")
print("=" * 80)

print(
    "Walk-forward observations : "
    f"{len(predictions_df)}"
)

print()

print("MODEL 8 — RIDGE")

print(
    f"MAE                       : "
    f"{model_mae:.8f}"
)

print(
    f"RMSE                      : "
    f"{model_rmse:.8f}"
)

print(
    f"R²                        : "
    f"{model_r2:.8f}"
)

print(
    f"Correlation               : "
    f"{model_correlation:.8f}"
)

print(
    f"Directional accuracy      : "
    f"{directional_accuracy:.4f}"
)

print()

print("BASELINE — ZERO RETURN")

print(
    f"MAE                       : "
    f"{zero_mae:.8f}"
)

print(
    f"RMSE                      : "
    f"{zero_rmse:.8f}"
)

print(
    f"R²                        : "
    f"{zero_r2:.8f}"
)

print()

print("BASELINE — PREVIOUS-DAY RETURN")

print(
    f"MAE                       : "
    f"{previous_day_mae:.8f}"
)

print(
    f"RMSE                      : "
    f"{previous_day_rmse:.8f}"
)

print(
    f"R²                        : "
    f"{previous_day_r2:.8f}"
)

print(
    f"Directional accuracy      : "
    f"{previous_day_directional_accuracy:.4f}"
)

print()

print("YEARLY RESULTS")

print(
    yearly_df.to_string(
        index=False
    )
)

print()

print("OUTPUT FILES")

print(
    f"Predictions : "
    f"{PREDICTIONS_FILE}"
)

print(
    f"Summary     : "
    f"{SUMMARY_FILE}"
)

print(
    f"Yearly      : "
    f"{YEARLY_FILE}"
)

print()

print("=" * 80)
print("MODEL 8 COMPLETE")
print("=" * 80)