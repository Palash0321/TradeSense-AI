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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONFIGURATION
# =========================================================

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
    / "nifty50_model0"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "model0_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "model0_summary.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR
    / "model0_yearly_results.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "model0_confusion_matrix.csv"
)

MIN_TRAIN_ROWS = 252


# =========================================================
# MODEL 0 FEATURE SET
# =========================================================
#
# ONLY NIFTY'S OWN INFORMATION.
#
# NO constituent intelligence is allowed here.
#
# These are all features already present in the
# validated feature dataset.
# =========================================================

MODEL0_FEATURES = [
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


TARGET_COLUMN = "target_t1_direction"


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset():
    print()
    print("=" * 70)
    print("MODEL 0 — NIFTY-ONLY BASELINE")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    data = pd.read_csv(FEATURE_FILE)

    required_columns = set(
        ["date", TARGET_COLUMN]
        + MODEL0_FEATURES
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Feature dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if data["date"].duplicated().any():
        raise RuntimeError(
            "Duplicate dates found in feature dataset."
        )

    return data


# =========================================================
# PREPARE MODEL DATA
# =========================================================

def prepare_model_data(data):
    model_data = data[
        [
            "date",
            TARGET_COLUMN,
        ]
        + MODEL0_FEATURES
    ].copy()

    model_data = model_data.dropna(
        subset=[TARGET_COLUMN]
    ).reset_index(drop=True)

    model_data[TARGET_COLUMN] = (
        pd.to_numeric(
            model_data[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    invalid_targets = sorted(
        set(model_data[TARGET_COLUMN].unique())
        - {0, 1}
    )

    if invalid_targets:
        raise RuntimeError(
            "Invalid target values found: "
            f"{invalid_targets}"
        )

    return model_data


# =========================================================
# BASELINE 1
# UNCONDITIONAL MAJORITY-CLASS BASELINE
# =========================================================

def calculate_unconditional_baseline(model_data):
    y = model_data[TARGET_COLUMN]

    positive_rate = float(
        (y == 1).mean()
    )

    negative_rate = float(
        (y == 0).mean()
    )

    majority_class = int(
        positive_rate >= negative_rate
    )

    predictions = np.full(
        len(y),
        majority_class,
        dtype=int,
    )

    return {
        "baseline": "UNCONDITIONAL_MAJORITY",
        "observations": len(y),
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "prediction_class": majority_class,
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            predictions,
        ),
    }


# =========================================================
# BASELINE 2
# PREVIOUS-DAY DIRECTION
# =========================================================

def calculate_previous_day_baseline(model_data):
    data = model_data.copy()

    previous_direction = (
        data["nifty_return_1d"]
        > 0
    ).astype(int)

    valid_mask = (
        data["nifty_return_1d"].notna()
    )

    y_true = data.loc[
        valid_mask,
        TARGET_COLUMN,
    ]

    y_pred = previous_direction.loc[
        valid_mask
    ]

    return {
        "baseline": "PREVIOUS_DAY_DIRECTION",
        "observations": len(y_true),
        "positive_rate": float(
            y_true.mean()
        ),
        "negative_rate": float(
            1 - y_true.mean()
        ),
        "prediction_class": "DIRECTION",
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }


# =========================================================
# LOGISTIC REGRESSION MODEL
# =========================================================

def create_model():
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight=None,
                ),
            ),
        ]
    )


# =========================================================
# EXPANDING WALK-FORWARD VALIDATION
# =========================================================
#
# At date T:
#
#   train = all eligible observations before T
#   test  = observation T
#
# No future observation is used to train the model.
#
# This is deliberately chronological.
# =========================================================

def run_walk_forward(model_data):
    print()
    print("=" * 70)
    print("EXPANDING WALK-FORWARD VALIDATION")
    print("=" * 70)

    feature_data = model_data[
        MODEL0_FEATURES
    ].copy()

    target = model_data[
        TARGET_COLUMN
    ].copy()

    predictions = []

    for test_index in range(
        MIN_TRAIN_ROWS,
        len(model_data),
    ):
        train_indices = np.arange(
            0,
            test_index,
        )

        test_indices = np.array(
            [test_index]
        )

        X_train = feature_data.iloc[
            train_indices
        ]

        y_train = target.iloc[
            train_indices
        ]

        X_test = feature_data.iloc[
            test_indices
        ]

        y_test = target.iloc[
            test_indices
        ]

        # -------------------------------------------------
        # Training data must contain complete feature rows.
        # -------------------------------------------------

        train_valid = (
            X_train.notna().all(axis=1)
            & y_train.notna()
        )

        X_train = X_train.loc[
            train_valid
        ]

        y_train = y_train.loc[
            train_valid
        ]

        # -------------------------------------------------
        # Test observation must contain complete features.
        # -------------------------------------------------

        if X_test.isna().any(axis=None):
            continue

        if len(X_train) < MIN_TRAIN_ROWS:
            continue

        if y_train.nunique() < 2:
            continue

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        probability = float(
            model.predict_proba(
                X_test
            )[0, 1]
        )

        prediction = int(
            probability >= 0.50
        )

        predictions.append(
            {
                "date": model_data.iloc[
                    test_index
                ]["date"],
                "actual": int(
                    y_test.iloc[0]
                ),
                "predicted": prediction,
                "probability_up": probability,
                "train_rows": len(X_train),
            }
        )

        if len(predictions) % 250 == 0:
            print(
                f"Walk-forward predictions: "
                f"{len(predictions)}"
            )

    predictions_df = pd.DataFrame(
        predictions
    )

    if predictions_df.empty:
        raise RuntimeError(
            "No walk-forward predictions were generated."
        )

    return predictions_df


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "observations": len(y_true),
        "positive_actual_rate": float(
            np.mean(y_true)
        ),
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
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "predicted_positive_rate": float(
            np.mean(y_pred)
        ),
    }

    return metrics


# =========================================================
# YEARLY ANALYSIS
# =========================================================

def calculate_yearly_results(predictions):
    predictions = predictions.copy()

    predictions["year"] = (
        pd.to_datetime(
            predictions["date"]
        ).dt.year
    )

    rows = []

    for year, group in (
        predictions
        .groupby("year")
    ):
        y_true = group["actual"].astype(int)
        y_pred = group["predicted"].astype(int)
        probabilities = group[
            "probability_up"
        ].astype(float)

        metrics = calculate_metrics(
            y_true,
            y_pred,
            probabilities,
        )

        metrics["year"] = int(year)

        rows.append(metrics)

    yearly = pd.DataFrame(rows)

    if not yearly.empty:
        yearly = yearly[
            [
                "year",
                "observations",
                "positive_actual_rate",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "predicted_positive_rate",
            ]
        ]

    return yearly


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_dataset()

    print(
        f"Feature rows: {len(data)}"
    )

    print(
        f"Coverage: "
        f"{data['date'].min().date()} -> "
        f"{data['date'].max().date()}"
    )

    model_data = prepare_model_data(
        data
    )

    print(
        f"Target rows: {len(model_data)}"
    )

    print(
        f"Target positive rate: "
        f"{model_data[TARGET_COLUMN].mean():.4f}"
    )

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    unconditional = (
        calculate_unconditional_baseline(
            model_data
        )
    )

    previous_day = (
        calculate_previous_day_baseline(
            model_data
        )
    )

    baseline_df = pd.DataFrame(
        [
            unconditional,
            previous_day,
        ]
    )

    print(
        baseline_df[
            [
                "baseline",
                "observations",
                "accuracy",
                "balanced_accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    predictions = run_walk_forward(
        model_data
    )

    predictions["date"] = pd.to_datetime(
        predictions["date"]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    predictions.to_csv(
        PREDICTION_FILE,
        index=False,
    )

    y_true = predictions[
        "actual"
    ].astype(int)

    y_pred = predictions[
        "predicted"
    ].astype(int)

    probabilities = predictions[
        "probability_up"
    ].astype(float)

    metrics = calculate_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    yearly = calculate_yearly_results(
        predictions
    )

    yearly.to_csv(
        YEARLY_FILE,
        index=False,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    confusion_df = pd.DataFrame(
        confusion,
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

    summary_rows = []

    summary_rows.append(
        {
            "model": "UNCONDITIONAL_MAJORITY",
            **{
                key: unconditional[key]
                for key in [
                    "observations",
                    "accuracy",
                    "balanced_accuracy",
                ]
            },
        }
    )

    summary_rows.append(
        {
            "model": "PREVIOUS_DAY_DIRECTION",
            **{
                key: previous_day[key]
                for key in [
                    "observations",
                    "accuracy",
                    "balanced_accuracy",
                ]
            },
        }
    )

    summary_rows.append(
        {
            "model": "MODEL_0_NIFTY_ONLY_LOGISTIC",
            **{
                key: metrics[key]
                for key in [
                    "observations",
                    "positive_actual_rate",
                    "accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "brier_score",
                    "true_negative",
                    "false_positive",
                    "false_negative",
                    "true_positive",
                    "predicted_positive_rate",
                ]
            },
        }
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # =====================================================
    # REPORT
    # =====================================================

    print()
    print("=" * 70)
    print("MODEL 0 RESULTS")
    print("=" * 70)

    print(
        f"Walk-forward observations: "
        f"{metrics['observations']}"
    )

    print(
        f"Actual positive rate: "
        f"{metrics['positive_actual_rate']:.4f}"
    )

    print(
        f"Predicted positive rate: "
        f"{metrics['predicted_positive_rate']:.4f}"
    )

    print()
    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy:    "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision:            "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:              "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score:          "
        f"{metrics['brier_score']:.4f}"
    )

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        confusion_df.to_string()
    )

    print()
    print(
        "=" * 70
    )
    print(
        "YEAR-BY-YEAR RESULTS"
    )
    print(
        "=" * 70
    )

    if yearly.empty:
        print(
            "No yearly results generated."
        )
    else:
        print(
            yearly[
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
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 70
    )
    print(
        "OUTPUT FILES"
    )
    print(
        "=" * 70
    )

    print(
        f"Predictions: {PREDICTION_FILE}"
    )

    print(
        f"Summary:     {SUMMARY_FILE}"
    )

    print(
        f"Yearly:      {YEARLY_FILE}"
    )

    print(
        f"Confusion:   {CONFUSION_FILE}"
    )

    print()
    print(
        "STATUS: PASS"
    )


if __name__ == "__main__":
    main()