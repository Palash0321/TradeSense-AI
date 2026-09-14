from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
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

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "model10_large_move_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "model10_large_move_summary.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR
    / "model10_large_move_yearly_results.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "model10_large_move_confusion_matrix.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TRAIN_ROWS = 252
THRESHOLD = 0.50
LARGE_MOVE_THRESHOLD = 0.01
RANDOM_STATE = 42


# ============================================================
# FEATURES
# SAME 30 FEATURES AS MODEL 7
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
    ["date", "nifty_t1_return"]
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


# ============================================================
# CREATE LARGE-MOVE TARGET
#
# FIXED BEFORE TESTING:
#
# abs(T+1 return) >= 1%
# ============================================================

df["target_large_move"] = (
    df["nifty_t1_return"].abs()
    >= LARGE_MOVE_THRESHOLD
).astype(int)


# ============================================================
# USABLE DATA
# ============================================================

model_df = df[
    ["date", "nifty_t1_return", "target_large_move"]
    + FEATURES
].copy()

model_df = model_df.dropna(
    subset=[
        "nifty_t1_return",
        "target_large_move",
    ] + FEATURES
).reset_index(drop=True)


print("=" * 80)
print("NIFTY 50 MODEL 10")
print("T+1 LARGE-MOVE PREDICTION")
print("=" * 80)

print(
    f"Feature file: {FEATURE_FILE}"
)

print(
    f"Rows: {len(df)}"
)

print(
    "Coverage: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print(
    f"Total features: {len(FEATURES)}"
)

print(
    "Large-move threshold: "
    f"{LARGE_MOVE_THRESHOLD:.2%}"
)

print(
    f"Usable rows: {len(model_df)}"
)

print(
    "Large-move rate: "
    f"{model_df['target_large_move'].mean():.4f}"
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
    model_df["target_large_move"]
    .astype(int)
    .values
)

dates = model_df["date"].values


# ============================================================
# BASELINE
#
# Predict the majority class using the full usable dataset
# only as a descriptive reference.
# ============================================================

majority_class = int(
    model_df["target_large_move"].mean() >= 0.5
)

majority_prediction = np.full(
    len(model_df),
    majority_class,
    dtype=int,
)

majority_accuracy = accuracy_score(
    y_all,
    majority_prediction,
)

majority_balanced_accuracy = (
    balanced_accuracy_score(
        y_all,
        majority_prediction,
    )
)


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

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_test_scaled = (
        scaler.transform(X_test)
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probability_large_move = (
        model.predict_proba(
            X_test_scaled
        )[0, 1]
    )

    prediction = int(
        probability_large_move
        >= THRESHOLD
    )

    predictions.append(
        {
            "date": test_date,
            "actual_large_move": int(
                y_test[0]
            ),
            "probability_large_move": (
                probability_large_move
            ),
            "prediction": prediction,
        }
    )


predictions_df = pd.DataFrame(
    predictions
)


# ============================================================
# OVERALL METRICS
# ============================================================

y_true = (
    predictions_df["actual_large_move"]
    .astype(int)
)

y_pred = (
    predictions_df["prediction"]
    .astype(int)
)

y_prob = (
    predictions_df["probability_large_move"]
    .astype(float)
)


accuracy = accuracy_score(
    y_true,
    y_pred,
)

balanced_accuracy = (
    balanced_accuracy_score(
        y_true,
        y_pred,
    )
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

auc = roc_auc_score(
    y_true,
    y_prob,
)

brier = brier_score_loss(
    y_true,
    y_prob,
)

predicted_large_move_rate = (
    y_pred.mean()
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

confusion_df = pd.DataFrame(
    cm,
    index=[
        "actual_0",
        "actual_1",
    ],
    columns=[
        "predicted_0",
        "predicted_1",
    ],
)

confusion_df.to_csv(
    CONFUSION_FILE
)


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

    year_true = (
        group["actual_large_move"]
        .astype(int)
    )

    year_pred = (
        group["prediction"]
        .astype(int)
    )

    year_prob = (
        group["probability_large_move"]
        .astype(float)
    )

    yearly_results.append(
        {
            "year": year,
            "observations": len(group),

            "actual_large_move_rate": (
                year_true.mean()
            ),

            "predicted_large_move_rate": (
                year_pred.mean()
            ),

            "accuracy": (
                accuracy_score(
                    year_true,
                    year_pred,
                )
            ),

            "balanced_accuracy": (
                balanced_accuracy_score(
                    year_true,
                    year_pred,
                )
            ),

            "precision": (
                precision_score(
                    year_true,
                    year_pred,
                    zero_division=0,
                )
            ),

            "recall": (
                recall_score(
                    year_true,
                    year_pred,
                    zero_division=0,
                )
            ),

            "f1": (
                f1_score(
                    year_true,
                    year_pred,
                    zero_division=0,
                )
            ),

            "roc_auc": (
                roc_auc_score(
                    year_true,
                    year_prob,
                )
                if year_true.nunique() == 2
                else np.nan
            ),

            "brier": (
                brier_score_loss(
                    year_true,
                    year_prob,
                )
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
            "model": "Model 10",

            "description": (
                "Predict whether absolute NIFTY "
                "T+1 return is at least 1%"
            ),

            "target": "abs(nifty_t1_return) >= 1%",

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

            "large_move_threshold": (
                LARGE_MOVE_THRESHOLD
            ),

            "actual_large_move_rate": (
                y_true.mean()
            ),

            "predicted_large_move_rate": (
                predicted_large_move_rate
            ),

            "min_train_rows": (
                MIN_TRAIN_ROWS
            ),

            "threshold": THRESHOLD,

            "majority_accuracy": (
                majority_accuracy
            ),

            "majority_balanced_accuracy": (
                majority_balanced_accuracy
            ),

            "model_accuracy": accuracy,

            "model_balanced_accuracy": (
                balanced_accuracy
            ),

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "roc_auc": auc,

            "brier": brier,
        }
    ]
)


# ============================================================
# SAVE
# ============================================================

predictions_df.drop(
    columns=["year"]
).to_csv(
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
# PRINT
# ============================================================

print("=" * 80)
print("MODEL 10 RESULTS")
print("=" * 80)

print(
    f"Walk-forward observations : "
    f"{len(predictions_df)}"
)

print(
    f"Actual large-move rate    : "
    f"{y_true.mean():.4f}"
)

print(
    f"Predicted large-move rate : "
    f"{predicted_large_move_rate:.4f}"
)

print()

print("BASELINE — MAJORITY CLASS")

print(
    f"Accuracy                  : "
    f"{majority_accuracy:.4f}"
)

print(
    f"Balanced accuracy         : "
    f"{majority_balanced_accuracy:.4f}"
)

print()

print("MODEL 10")

print(
    f"Accuracy                  : "
    f"{accuracy:.4f}"
)

print(
    f"Balanced accuracy         : "
    f"{balanced_accuracy:.4f}"
)

print(
    f"Precision                 : "
    f"{precision:.4f}"
)

print(
    f"Recall                    : "
    f"{recall:.4f}"
)

print(
    f"F1                        : "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC                   : "
    f"{auc:.4f}"
)

print(
    f"Brier score               : "
    f"{brier:.4f}"
)

print()

print("CONFUSION MATRIX")

print(confusion_df)

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

print(
    f"Confusion   : "
    f"{CONFUSION_FILE}"
)

print()

print("=" * 80)
print("MODEL 10 COMPLETE")
print("=" * 80)