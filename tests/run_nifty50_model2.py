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

FEATURE_FILE = Path(
    "data/nifty50_model2/nifty50_model2_features.csv"
)

OUTPUT_DIR = Path(
    "data/nifty50_model2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL 2 FEATURES
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

WEIGHTED_FEATURES = [
    "weighted_return_1d",
    "weighted_return_20d",
    "weighted_positive_pressure_1d",
    "weighted_positive_pressure_20d",
    "weighted_negative_pressure_1d",
    "weighted_negative_pressure_20d",
    "weighted_constituent_count",
]

FEATURES = (
    NIFTY_FEATURES
    + WEIGHTED_FEATURES
)

TARGET = "target_t1_direction"

DATE_COLUMN = "date"


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

MIN_TRAIN_SIZE = 252

PROBABILITY_THRESHOLD = 0.50


# ============================================================
# METRIC HELPER
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    metrics = {
        "observations": len(y_true),

        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "roc_auc": np.nan,

        "brier_score":
            brier_score_loss(
                y_true,
                y_prob
            ),
    }

    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                y_prob
            )
        )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NIFTY 50 MODEL 2")
    print("NIFTY HISTORY + DYNAMIC WEIGHTED CONSTITUENT PRESSURE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load feature file
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found: "
            f"{FEATURE_FILE}"
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
        f"Feature rows: {len(df)}"
    )

    print(
        f"Coverage: "
        f"{df[DATE_COLUMN].min().date()} "
        f"-> "
        f"{df[DATE_COLUMN].max().date()}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            TARGET,
            "nifty_t1_return",
        ]
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

    # --------------------------------------------------------
    # Usable observations
    # --------------------------------------------------------

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
        f"Usable Model 2 rows: "
        f"{len(model_df)}"
    )

    print(
        f"Target positive rate: "
        f"{model_df[TARGET].mean():.4f}"
    )

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------

    X = model_df[
        FEATURES
    ].astype(float)

    y = model_df[
        TARGET
    ].astype(int)

    dates = model_df[
        DATE_COLUMN
    ]

    # --------------------------------------------------------
    # Expanding chronological walk-forward
    # --------------------------------------------------------

    predictions = []

    for test_index in range(
        MIN_TRAIN_SIZE,
        len(model_df)
    ):

        train_indices = np.arange(
            0,
            test_index
        )

        X_train = X.iloc[
            train_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        X_test = X.iloc[
            [test_index]
        ]

        y_test = y.iloc[
            test_index
        ]

        # ----------------------------------------------------
        # Scale using TRAINING DATA ONLY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------

        model = LogisticRegression(
            max_iter=2000,
            random_state=42
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        probability_up = (
            model
            .predict_proba(
                X_test_scaled
            )[0, 1]
        )

        prediction = int(
            probability_up
            >= PROBABILITY_THRESHOLD
        )

        predictions.append(
            {
                "date":
                    dates.iloc[
                        test_index
                    ],

                "actual":
                    int(y_test),

                "predicted":
                    prediction,

                "probability_up":
                    probability_up,
            }
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    prediction_file = (
        OUTPUT_DIR
        / "model2_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_file,
        index=False
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall_metrics = calculate_metrics(
        predictions_df["actual"],
        predictions_df["predicted"],
        predictions_df["probability_up"],
    )

    # --------------------------------------------------------
    # Majority baseline
    # --------------------------------------------------------

    full_target = (
        df[TARGET]
        .dropna()
        .astype(int)
    )

    majority_class = int(
        full_target.mean()
        >= 0.50
    )

    majority_pred = np.full(
        len(full_target),
        majority_class
    )

    majority_accuracy = (
        accuracy_score(
            full_target,
            majority_pred
        )
    )

    majority_balanced_accuracy = (
        balanced_accuracy_score(
            full_target,
            majority_pred
        )
    )

    # --------------------------------------------------------
    # Previous-day direction baseline
    # --------------------------------------------------------

    baseline_df = (
        df[
            [
                "date",
                "nifty_return_1d",
                TARGET,
            ]
        ]
        .dropna()
        .copy()
    )

    baseline_df[
        "previous_day_prediction"
    ] = (
        baseline_df[
            "nifty_return_1d"
        ] > 0
    ).astype(int)

    previous_day_accuracy = (
        accuracy_score(
            baseline_df[TARGET],
            baseline_df[
                "previous_day_prediction"
            ],
        )
    )

    previous_day_balanced_accuracy = (
        balanced_accuracy_score(
            baseline_df[TARGET],
            baseline_df[
                "previous_day_prediction"
            ],
        )
    )

    # --------------------------------------------------------
    # Print baseline comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"UNCONDITIONAL_MAJORITY: "
        f"{len(full_target)} obs, "
        f"accuracy "
        f"{majority_accuracy:.6f}, "
        f"balanced_accuracy "
        f"{majority_balanced_accuracy:.6f}"
    )

    print(
        f"PREVIOUS_DAY_DIRECTION: "
        f"{len(baseline_df)} obs, "
        f"accuracy "
        f"{previous_day_accuracy:.6f}, "
        f"balanced_accuracy "
        f"{previous_day_balanced_accuracy:.6f}"
    )

    # --------------------------------------------------------
    # Print Model 2 results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL 2 RESULTS")
    print("=" * 70)

    print(
        f"Walk-forward observations: "
        f"{overall_metrics['observations']}"
    )

    print(
        f"Actual positive rate: "
        f"{predictions_df['actual'].mean():.4f}"
    )

    print(
        f"Predicted positive rate: "
        f"{predictions_df['predicted'].mean():.4f}"
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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        predictions_df["actual"],
        predictions_df["predicted"]
    )

    confusion_df = pd.DataFrame(
        cm,
        index=[
            "actual_0",
            "actual_1"
        ],
        columns=[
            "predicted_0",
            "predicted_1"
        ],
    )

    confusion_file = (
        OUTPUT_DIR
        / "model2_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_file
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df
    )

    # --------------------------------------------------------
    # Yearly results
    # --------------------------------------------------------

    predictions_df["year"] = (
        predictions_df[
            "date"
        ].dt.year
    )

    yearly_results = []

    for year, group in (
        predictions_df.groupby(
            "year"
        )
    ):

        metrics = calculate_metrics(
            group["actual"],
            group["predicted"],
            group["probability_up"],
        )

        yearly_results.append(
            {
                "year": year,
                **metrics,
            }
        )

    yearly_df = pd.DataFrame(
        yearly_results
    )

    yearly_file = (
        OUTPUT_DIR
        / "model2_yearly_results.csv"
    )

    yearly_df.to_csv(
        yearly_file,
        index=False
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

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "model":
            "MODEL_2",

        "feature_group":
            "NIFTY_HISTORY_PLUS_DYNAMIC_WEIGHTED_PRESSURE",

        "feature_count":
            len(FEATURES),

        "nifty_feature_count":
            len(NIFTY_FEATURES),

        "weighted_feature_count":
            len(WEIGHTED_FEATURES),

        "walk_forward_observations":
            overall_metrics[
                "observations"
            ],

        "actual_positive_rate":
            predictions_df[
                "actual"
            ].mean(),

        "predicted_positive_rate":
            predictions_df[
                "predicted"
            ].mean(),

        "accuracy":
            overall_metrics[
                "accuracy"
            ],

        "balanced_accuracy":
            overall_metrics[
                "balanced_accuracy"
            ],

        "precision":
            overall_metrics[
                "precision"
            ],

        "recall":
            overall_metrics[
                "recall"
            ],

        "f1":
            overall_metrics[
                "f1"
            ],

        "roc_auc":
            overall_metrics[
                "roc_auc"
            ],

        "brier_score":
            overall_metrics[
                "brier_score"
            ],

        "previous_day_accuracy":
            previous_day_accuracy,

        "previous_day_balanced_accuracy":
            previous_day_balanced_accuracy,

        "majority_accuracy":
            majority_accuracy,

        "majority_balanced_accuracy":
            majority_balanced_accuracy,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_file = (
        OUTPUT_DIR
        / "model2_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Final files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(
        prediction_file
    )

    print(
        summary_file
    )

    print(
        yearly_file
    )

    print(
        confusion_file
    )

    print()
    print("=" * 70)
    print("MODEL 2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()