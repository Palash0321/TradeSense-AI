from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model6"
    / "nifty50_model6_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "model6_predictions.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "model6_summary.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR
    / "model6_yearly_results.csv"
)

CONFUSION_FILE = (
    OUTPUT_DIR
    / "model6_confusion_matrix.csv"
)


# ============================================================
# MODEL CONFIGURATION
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


SECTOR_FEATURES = [
    "sector_return_1d_mean",
    "sector_return_5d_mean",
    "sector_positive_breadth_mean",
    "sector_negative_breadth_mean",
    "sector_dispersion_mean",
    "sector_leadership",
    "sector_agreement",
]


FEATURES = (
    NIFTY_FEATURES
    + SECTOR_FEATURES
)

TARGET = "target_t1_direction"

MIN_TRAIN_ROWS = 252

PROBABILITY_THRESHOLD = 0.50


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NIFTY 50 MODEL 6")
    print("NIFTY HISTORY + SECTOR PRESSURE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            FEATURE_FILE
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Feature rows: {len(df):,}"
    )

    print(
        f"Coverage: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = (
        FEATURES
        + [
            TARGET,
            "nifty_t1_return",
            "nifty_t1_close",
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
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # MODEL DATA
    # --------------------------------------------------------

    model_df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    model_df[TARGET] = (
        model_df[TARGET]
        .astype(int)
    )

    print(
        f"Usable Model 6 rows: "
        f"{len(model_df):,}"
    )

    print(
        f"Target positive rate: "
        f"{model_df[TARGET].mean():.4f}"
    )

    # --------------------------------------------------------
    # BASELINES
    # --------------------------------------------------------

    baseline_df = df.dropna(
        subset=[
            TARGET
        ]
    ).copy()

    y_baseline = (
        baseline_df[TARGET]
        .astype(int)
        .values
    )

    majority_class = int(
        y_baseline.mean() >= 0.5
    )

    majority_pred = (
        pd.Series(
            majority_class,
            index=baseline_df.index,
        )
        .values
    )

    majority_metrics = {
        "baseline": "UNCONDITIONAL_MAJORITY",
        **calculate_metrics(
            y_baseline,
            majority_pred,
            majority_pred.astype(float),
        ),
    }

    # Previous-day direction baseline
    previous_day = df[
        [
            "date",
            "nifty_return_1d",
            TARGET,
        ]
    ].dropna(
        subset=[
            "nifty_return_1d",
            TARGET,
        ]
    ).copy()

    previous_day_pred = (
        previous_day[
            "nifty_return_1d"
        ]
        > 0
    ).astype(int).values

    previous_day_true = (
        previous_day[TARGET]
        .astype(int)
        .values
    )

    previous_day_prob = (
        previous_day[
            "nifty_return_1d"
        ]
        .rank(
            pct=True
        )
        .values
    )

    previous_day_metrics = {
        "baseline": "PREVIOUS_DAY_DIRECTION",
        **calculate_metrics(
            previous_day_true,
            previous_day_pred,
            previous_day_prob,
        ),
    }

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

    # --------------------------------------------------------
    # WALK-FORWARD
    # --------------------------------------------------------

    predictions = []

    for i in range(
        MIN_TRAIN_ROWS,
        len(model_df),
    ):

        train = model_df.iloc[
            :i
        ]

        test = model_df.iloc[
            i:i + 1
        ]

        X_train = train[
            FEATURES
        ]

        y_train = train[
            TARGET
        ]

        X_test = test[
            FEATURES
        ]

        y_test = int(
            test[TARGET].iloc[0]
        )

        pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probability = float(
            pipeline.predict_proba(
                X_test
            )[0, 1]
        )

        prediction = int(
            probability
            >= PROBABILITY_THRESHOLD
        )

        predictions.append(
            {
                "date": test[
                    "date"
                ].iloc[0],
                "actual": y_test,
                "predicted": prediction,
                "probability_up": probability,
                "nifty_t1_return": test[
                    "nifty_t1_return"
                ].iloc[0],
            }
        )

    predictions_df = pd.DataFrame(
        predictions
    )

    y_true = predictions_df[
        "actual"
    ].astype(int).values

    y_pred = predictions_df[
        "predicted"
    ].astype(int).values

    y_prob = predictions_df[
        "probability_up"
    ].astype(float).values

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    predicted_positive_rate = (
        y_pred.mean()
    )

    actual_positive_rate = (
        y_true.mean()
    )

    print()
    print("=" * 70)
    print("MODEL 6 RESULTS")
    print("=" * 70)

    print(
        f"Walk-forward observations: "
        f"{metrics['observations']}"
    )

    print(
        f"Actual positive rate: "
        f"{actual_positive_rate:.4f}"
    )

    print(
        f"Predicted positive rate: "
        f"{predicted_positive_rate:.4f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{metrics['brier_score']:.4f}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
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

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        confusion_df
    )

    # --------------------------------------------------------
    # YEARLY RESULTS
    # --------------------------------------------------------

    predictions_df["year"] = (
        predictions_df[
            "date"
        ].dt.year
    )

    yearly_rows = []

    for year, group in predictions_df.groupby(
        "year"
    ):

        group_true = group[
            "actual"
        ].astype(int).values

        group_pred = group[
            "predicted"
        ].astype(int).values

        group_prob = group[
            "probability_up"
        ].astype(float).values

        yearly_metrics = calculate_metrics(
            group_true,
            group_pred,
            group_prob,
        )

        yearly_rows.append(
            {
                "year": year,
                **yearly_metrics,
            }
        )

    yearly_df = pd.DataFrame(
        yearly_rows
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
    # SUMMARY
    # --------------------------------------------------------

    summary_rows = [
        {
            "model": "UNCONDITIONAL_MAJORITY",
            **{
                key: value
                for key, value
                in majority_metrics.items()
                if key != "baseline"
            },
        },
        {
            "model": "PREVIOUS_DAY_DIRECTION",
            **{
                key: value
                for key, value
                in previous_day_metrics.items()
                if key != "baseline"
            },
        },
        {
            "model": "MODEL_6",
            **metrics,
            "actual_positive_rate":
                actual_positive_rate,
            "predicted_positive_rate":
                predicted_positive_rate,
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # SAVE FILES
    # --------------------------------------------------------

    predictions_df.to_csv(
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

    confusion_df.to_csv(
        CONFUSION_FILE
    )

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(
        PREDICTIONS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        YEARLY_FILE
    )

    print(
        CONFUSION_FILE
    )

    print()
    print("=" * 70)
    print("MODEL 6 COMPLETE")
    print("=" * 70)

    print()
    print(
        "RESEARCH STATUS:"
    )

    print(
        "EXPLORATORY ONLY — sector classifications "
        "are current official classifications, "
        "not point-in-time historical classifications."
    )


if __name__ == "__main__":
    main()