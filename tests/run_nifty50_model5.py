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

OUTPUT_DIR = BASE_DIR / "data" / "nifty50_model5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL 5 FEATURES
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

CONSTITUENT_VOLUME_FEATURES = [
    "high_volume_up_breadth",
    "high_volume_down_breadth",
    "median_volume_ratio_20d",
]

FEATURES = (
    NIFTY_FEATURES
    + CONSTITUENT_VOLUME_FEATURES
)

TARGET = "target_t1_direction"

DATE_COLUMN = "date"

MIN_TRAIN_ROWS = 252

RANDOM_STATE = 42


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "observations": len(y_true),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_prob,
        ),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NIFTY 50 MODEL 5")
print("NIFTY HISTORY + CONSTITUENT VOLUME PRESSURE")
print("=" * 70)

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found: {FEATURE_FILE}"
    )

df = pd.read_csv(
    FEATURE_FILE
)

df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN]
)

df = (
    df
    .sort_values(DATE_COLUMN)
    .reset_index(drop=True)
)

print(
    f"Feature rows: {len(df):,}"
)

print(
    f"Coverage: "
    f"{df[DATE_COLUMN].min().date()} -> "
    f"{df[DATE_COLUMN].max().date()}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES + [TARGET]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# MODEL DATA
# ============================================================

model_df = (
    df
    .dropna(
        subset=FEATURES + [TARGET]
    )
    .copy()
)

model_df = (
    model_df
    .sort_values(DATE_COLUMN)
    .reset_index(drop=True)
)

print(
    f"Usable Model 5 rows: "
    f"{len(model_df):,}"
)

print(
    f"Target positive rate: "
    f"{model_df[TARGET].mean():.4f}"
)


# ============================================================
# BASELINE 1
# UNCONDITIONAL MAJORITY
# ============================================================

baseline_df = (
    df
    .dropna(
        subset=[TARGET]
    )
    .copy()
)

y_baseline = (
    baseline_df[TARGET]
    .astype(int)
)

majority_class = int(
    y_baseline.mean() >= 0.5
)

majority_pred = np.full(
    len(y_baseline),
    majority_class,
)

majority_metrics = {
    "model": "UNCONDITIONAL_MAJORITY",
    "observations": len(y_baseline),
    "accuracy": accuracy_score(
        y_baseline,
        majority_pred,
    ),
    "balanced_accuracy": balanced_accuracy_score(
        y_baseline,
        majority_pred,
    ),
}


# ============================================================
# BASELINE 2
# PREVIOUS DAY DIRECTION
# ============================================================

previous_day_df = (
    df
    .dropna(
        subset=[
            "nifty_return_1d",
            TARGET,
        ]
    )
    .copy()
)

previous_day_df = previous_day_df[
    previous_day_df["nifty_return_1d"] != 0
].copy()

previous_day_pred = (
    previous_day_df["nifty_return_1d"] > 0
).astype(int)

previous_day_actual = (
    previous_day_df[TARGET]
    .astype(int)
)

previous_day_metrics = {
    "model": "PREVIOUS_DAY_DIRECTION",
    "observations": len(previous_day_actual),
    "accuracy": accuracy_score(
        previous_day_actual,
        previous_day_pred,
    ),
    "balanced_accuracy": balanced_accuracy_score(
        previous_day_actual,
        previous_day_pred,
    ),
}


# ============================================================
# PRINT BASELINES
# ============================================================

print()
print("=" * 70)
print("BASELINE COMPARISON")
print("=" * 70)

print(
    "UNCONDITIONAL_MAJORITY: "
    f"{majority_metrics['observations']} obs, "
    f"accuracy "
    f"{majority_metrics['accuracy']:.6f}, "
    f"balanced_accuracy "
    f"{majority_metrics['balanced_accuracy']:.6f}"
)

print(
    "PREVIOUS_DAY_DIRECTION: "
    f"{previous_day_metrics['observations']} obs, "
    f"accuracy "
    f"{previous_day_metrics['accuracy']:.6f}, "
    f"balanced_accuracy "
    f"{previous_day_metrics['balanced_accuracy']:.6f}"
)


# ============================================================
# EXPANDING WALK-FORWARD
# ============================================================

predictions = []

for i in range(
    MIN_TRAIN_ROWS,
    len(model_df),
):

    train_df = model_df.iloc[:i]

    test_df = model_df.iloc[
        i:i + 1
    ]

    X_train = train_df[FEATURES]

    y_train = (
        train_df[TARGET]
        .astype(int)
    )

    X_test = test_df[FEATURES]

    y_test = int(
        test_df[TARGET].iloc[0]
    )

    # --------------------------------------------------------
    # SCALE USING TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    probability_up = (
        model
        .predict_proba(
            X_test_scaled
        )[0, 1]
    )

    prediction = int(
        probability_up >= 0.50
    )

    predictions.append(
        {
            "date": test_df[
                DATE_COLUMN
            ].iloc[0],
            "actual": y_test,
            "predicted": prediction,
            "probability_up": probability_up,
        }
    )


# ============================================================
# PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    predictions
)

if predictions_df.empty:
    raise ValueError(
        "No walk-forward predictions were generated."
    )

y_true = (
    predictions_df["actual"]
    .astype(int)
)

y_pred = (
    predictions_df["predicted"]
    .astype(int)
)

y_prob = (
    predictions_df["probability_up"]
    .astype(float)
)


# ============================================================
# OVERALL RESULTS
# ============================================================

overall_metrics = calculate_metrics(
    y_true,
    y_pred,
    y_prob,
)

print()
print("=" * 70)
print("MODEL 5 RESULTS")
print("=" * 70)

print(
    f"Walk-forward observations: "
    f"{overall_metrics['observations']}"
)

print(
    f"Actual positive rate: "
    f"{y_true.mean():.4f}"
)

print(
    f"Predicted positive rate: "
    f"{y_pred.mean():.4f}"
)

print(
    f"Accuracy: "
    f"{overall_metrics['accuracy']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{overall_metrics['balanced_accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{overall_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{overall_metrics['recall']:.4f}"
)

print(
    f"F1: "
    f"{overall_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{overall_metrics['roc_auc']:.4f}"
)

print(
    f"Brier Score: "
    f"{overall_metrics['brier_score']:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = pd.crosstab(
    y_true,
    y_pred,
    rownames=["actual"],
    colnames=["predicted"],
    dropna=False,
)

confusion = confusion.reindex(
    index=[0, 1],
    columns=[0, 1],
    fill_value=0,
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    confusion.rename(
        columns={
            0: "predicted_0",
            1: "predicted_1",
        },
        index={
            0: "actual_0",
            1: "actual_1",
        },
    )
)


# ============================================================
# YEARLY RESULTS
# ============================================================

predictions_df["year"] = (
    predictions_df["date"]
    .dt.year
)

yearly_results = []

for year, group in predictions_df.groupby(
    "year"
):

    y_true_year = (
        group["actual"]
        .astype(int)
    )

    y_pred_year = (
        group["predicted"]
        .astype(int)
    )

    y_prob_year = (
        group["probability_up"]
        .astype(float)
    )

    metrics = calculate_metrics(
        y_true_year,
        y_pred_year,
        y_prob_year,
    )

    metrics["year"] = year

    yearly_results.append(
        metrics
    )


yearly_df = pd.DataFrame(
    yearly_results
)

yearly_df = (
    yearly_df[
        [
            "year",
            "observations",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "brier_score",
        ]
    ]
    .sort_values("year")
    .reset_index(drop=True)
)


print()
print("=" * 70)
print("YEARLY RESULTS")
print("=" * 70)

print(
    yearly_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    [
        {
            "model": "MODEL_5",
            "description": (
                "NIFTY history + "
                "constituent volume pressure"
            ),
            **overall_metrics,
        },
        majority_metrics,
        previous_day_metrics,
    ]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

predictions_file = (
    OUTPUT_DIR
    / "model5_predictions.csv"
)

summary_file = (
    OUTPUT_DIR
    / "model5_summary.csv"
)

yearly_file = (
    OUTPUT_DIR
    / "model5_yearly_results.csv"
)

confusion_file = (
    OUTPUT_DIR
    / "model5_confusion_matrix.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False,
)

summary_df.to_csv(
    summary_file,
    index=False,
)

yearly_df.to_csv(
    yearly_file,
    index=False,
)

confusion.rename(
    columns={
        0: "predicted_0",
        1: "predicted_1",
    },
    index={
        0: "actual_0",
        1: "actual_1",
    },
).to_csv(
    confusion_file
)


# ============================================================
# FILE OUTPUT
# ============================================================

print()
print("=" * 70)
print("FILES WRITTEN")
print("=" * 70)

print(predictions_file)
print(summary_file)
print(yearly_file)
print(confusion_file)

print()
print("=" * 70)
print("MODEL 5 COMPLETE")
print("=" * 70)