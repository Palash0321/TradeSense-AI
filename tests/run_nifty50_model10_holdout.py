from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# NIFTY 50 MODEL 10 — PROSPECTIVE FUTURE HOLDOUT
# EXPANDING WALK-FORWARD VERSION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

HOLDOUT_LOCK_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
    / "model10_holdout_lock.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "model10_holdout_predictions.csv"
)


# ============================================================
# FROZEN HOLDOUT DEFINITION
# ============================================================

HOLDOUT_START_DATE = pd.Timestamp("2026-09-11")


# ============================================================
# FROZEN MODEL 10 FEATURE SET — 30 FEATURES
# ============================================================

FEATURES = [

    # --------------------------------------------------------
    # NIFTY HISTORY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BREADTH
    # --------------------------------------------------------

    "breadth_1d",
    "breadth_5d",
    "breadth_20d",
    "above_ema20_breadth",
    "above_ema50_breadth",
    "above_ema200_breadth",

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    "median_return_1d",
    "median_return_5d",
    "median_return_20d",
    "median_rsi14",

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    # --------------------------------------------------------

    "relative_strength_breadth",
    "median_relative_strength_20d",

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    "high_volume_up_breadth",
    "high_volume_down_breadth",
    "median_volume_ratio_20d",
]


# ============================================================
# FROZEN MODEL PARAMETERS
# ============================================================

MIN_TRAIN_ROWS = 252

RANDOM_STATE = 42

PREDICTION_THRESHOLD = 0.50


def train_and_predict(train_data, prediction_row):

    """
    Train Model 10 using ONLY observations strictly before
    the prediction date, then generate one probability.

    No future target or future feature is used.
    """

    train_data = train_data.copy()

    # --------------------------------------------------------
    # Training target
    # --------------------------------------------------------

    train_data["actual_large_move"] = (
        train_data["nifty_t1_return"].abs() >= 0.01
    ).astype(int)

    # --------------------------------------------------------
    # Remove incomplete training observations
    # --------------------------------------------------------

    train_data = train_data.dropna(
        subset=FEATURES + ["nifty_t1_return"]
    ).copy()

    train_data = (
        train_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(train_data) < MIN_TRAIN_ROWS:
        return None

    # --------------------------------------------------------
    # Safety:
    # training dates must be strictly before prediction date
    # --------------------------------------------------------

    prediction_date = prediction_row["date"]

    if (
        train_data["date"] >= prediction_date
    ).any():

        raise AssertionError(
            "Training data contains an observation on or after "
            "the prediction date."
        )

    X_train = train_data[FEATURES]

    y_train = train_data["actual_large_move"]

    X_prediction = prediction_row[FEATURES].to_frame().T

    # --------------------------------------------------------
    # Frozen Model 10 preprocessing
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_prediction_scaled = scaler.transform(
        X_prediction
    )

    # --------------------------------------------------------
    # Frozen Model 10 model
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    probability = model.predict_proba(
        X_prediction_scaled
    )[0, 1]

    prediction = int(
        probability >= PREDICTION_THRESHOLD
    )

    return {
        "probability_large_move": float(probability),
        "prediction": prediction,
        "training_end_date": train_data["date"].max(),
        "training_rows": len(train_data),
        "training_large_move_rate": float(y_train.mean()),
    }


def main():

    print("=" * 70)
    print(
        "NIFTY 50 MODEL 10 — PROSPECTIVE FUTURE HOLDOUT"
    )
    print(
        "EXPANDING WALK-FORWARD VERSION"
    )
    print("=" * 70)

    # ========================================================
    # 1. FILE CHECKS
    # ========================================================

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Missing feature file:\n{FEATURE_FILE}"
        )

    if not HOLDOUT_LOCK_FILE.exists():

        raise FileNotFoundError(
            f"Missing holdout lock file:\n"
            f"{HOLDOUT_LOCK_FILE}"
        )

    # ========================================================
    # 2. VERIFY HOLDOUT LOCK
    # ========================================================

    lock = pd.read_csv(
        HOLDOUT_LOCK_FILE
    )

    if len(lock) != 1:

        raise ValueError(
            "Holdout lock must contain exactly one row."
        )

    locked_start = pd.Timestamp(
        lock.loc[0, "holdout_start_date"]
    )

    if locked_start != HOLDOUT_START_DATE:

        raise ValueError(
            "Holdout start date does not match the "
            "frozen holdout definition."
        )

    if not bool(
        lock.loc[0, "model_frozen"]
    ):

        raise ValueError(
            "Holdout lock does not confirm frozen model."
        )

    if not bool(
        lock.loc[0, "features_frozen"]
    ):

        raise ValueError(
            "Holdout lock does not confirm frozen features."
        )

    # ========================================================
    # 3. LOAD FEATURES
    # ========================================================

    data = pd.read_csv(
        FEATURE_FILE
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = (
        data
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    required_columns = set(FEATURES)

    required_columns.add(
        "nifty_t1_return"
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # 4. SEPARATE HISTORICAL AND HOLDOUT DATES
    # ========================================================

    historical = data[
        data["date"] < HOLDOUT_START_DATE
    ].copy()

    holdout = data[
        data["date"] >= HOLDOUT_START_DATE
    ].copy()

    historical = (
        historical
        .sort_values("date")
        .reset_index(drop=True)
    )

    holdout = (
        holdout
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ========================================================
    # 5. IDENTIFY PREDICTABLE HOLDOUT ROWS
    # ========================================================
    #
    # IMPORTANT:
    #
    # We only require the 30 T-date features.
    #
    # We DO NOT require T+1 return.
    #
    # That future outcome is intentionally unknown.
    # ========================================================

    holdout_predictable = holdout[
        holdout[FEATURES].notna().all(axis=1)
    ].copy()

    # ========================================================
    # 6. GENERATE ONE PREDICTION PER HOLDOUT DATE
    # ========================================================

    new_predictions = []

    print()
    print("HOLDOUT PROCESSING")
    print("-" * 70)

    if len(holdout_predictable) == 0:

        print(
            "No complete holdout feature rows are currently "
            "available."
        )

    else:

        for _, prediction_row in (
            holdout_predictable.iterrows()
        ):

            prediction_date = prediction_row[
                "date"
            ]

            # ------------------------------------------------
            # EXPANDING TRAINING WINDOW
            #
            # ONLY dates strictly before T
            # ------------------------------------------------

            training_data = historical[
                historical["date"] < prediction_date
            ].copy()

            result = train_and_predict(
                training_data,
                prediction_row
            )

            if result is None:

                print(
                    f"{prediction_date.date()} | "
                    "SKIPPED | insufficient training data"
                )

                continue

            record = {
                "date": prediction_date,
                "probability_large_move": (
                    result[
                        "probability_large_move"
                    ]
                ),
                "prediction": (
                    result["prediction"]
                ),
                "training_end_date": (
                    result["training_end_date"]
                ),
                "training_rows": (
                    result["training_rows"]
                ),
                "training_large_move_rate": (
                    result[
                        "training_large_move_rate"
                    ]
                ),
                "holdout_start_date": (
                    HOLDOUT_START_DATE
                ),
            }

            new_predictions.append(
                record
            )

            print(
                f"{prediction_date.date()} | "
                f"probability="
                f"{result['probability_large_move']:.6f} | "
                f"prediction="
                f"{result['prediction']} | "
                f"training_end="
                f"{result['training_end_date'].date()} | "
                f"training_rows="
                f"{result['training_rows']}"
            )

    # ========================================================
    # 7. CREATE OUTPUT DATAFRAME
    # ========================================================

    output_columns = [
        "date",
        "probability_large_move",
        "prediction",
        "training_end_date",
        "training_rows",
        "training_large_move_rate",
        "holdout_start_date",
    ]

    new_output = pd.DataFrame(
        new_predictions,
        columns=output_columns
    )

    # ========================================================
    # 8. NEVER OVERWRITE EXISTING HOLDOUT PREDICTIONS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():

        existing = pd.read_csv(
            OUTPUT_FILE
        )

        if len(existing) > 0:

            existing["date"] = pd.to_datetime(
                existing["date"]
            )

            new_dates = set(
                new_output["date"]
            )

            existing_dates = set(
                existing["date"]
            )

            overlap = (
                new_dates
                .intersection(existing_dates)
            )

            if overlap:

                raise ValueError(
                    "Prediction already exists for "
                    f"holdout date(s): "
                    f"{sorted(overlap)}"
                )

            combined = pd.concat(
                [
                    existing,
                    new_output
                ],
                ignore_index=True
            )

        else:

            combined = new_output.copy()

    else:

        combined = new_output.copy()

    if len(combined) > 0:

        combined = (
            combined
            .sort_values("date")
            .drop_duplicates(
                "date",
                keep="first"
            )
            .reset_index(drop=True)
        )

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # 9. SAFETY AUDIT
    # ========================================================

    if len(combined) > 0:

        # No pre-holdout prediction
        if (
            combined["date"]
            < HOLDOUT_START_DATE
        ).any():

            raise AssertionError(
                "Pre-holdout prediction detected."
            )

        # No duplicate dates
        if combined["date"].duplicated().any():

            raise AssertionError(
                "Duplicate holdout dates detected."
            )

        # Probability bounds
        if (
            (
                combined[
                    "probability_large_move"
                ] < 0
            )
            |
            (
                combined[
                    "probability_large_move"
                ] > 1
            )
        ).any():

            raise AssertionError(
                "Invalid probability detected."
            )

        # Training cutoff must precede prediction date
        training_dates = pd.to_datetime(
            combined["training_end_date"]
        )

        if (
            training_dates
            >= combined["date"]
        ).any():

            raise AssertionError(
                "Training cutoff is not strictly before "
                "prediction date."
            )

    # ========================================================
    # 10. FINAL REPORT
    # ========================================================

    print()
    print("FROZEN MODEL")
    print("-" * 70)
    print("Target                  : |T+1 return| >= 1.0%")
    print("Features                : 30")
    print(
        "Training method         : Expanding walk-forward"
    )
    print(
        "Minimum training rows  : "
        f"{MIN_TRAIN_ROWS}"
    )
    print(
        "Model                   : Logistic Regression"
    )
    print(
        "Scaling                 : StandardScaler"
    )
    print(
        f"Random state            : {RANDOM_STATE}"
    )
    print(
        f"Classification threshold: "
        f"{PREDICTION_THRESHOLD}"
    )

    print()
    print("HOLDOUT")
    print("-" * 70)
    print(
        f"Holdout starts          : "
        f"{HOLDOUT_START_DATE.date()}"
    )

    print(
        f"Complete feature dates  : "
        f"{len(holdout_predictable)}"
    )

    print(
        f"New predictions created : "
        f"{len(new_output)}"
    )

    print(
        f"Stored holdout rows     : "
        f"{len(combined)}"
    )

    print()
    print("LEAKAGE CONTROLS")
    print("-" * 70)
    print(
        "Training date < prediction date : PASS"
    )
    print(
        "Future T+1 target used for X    : NO"
    )
    print(
        "Future-result tuning             : NO"
    )
    print(
        "Feature changes                  : NO"
    )
    print(
        "Model changes                    : NO"
    )
    print(
        "Threshold optimization           : NO"
    )
    print(
        "Historical Model 10 files        : UNTOUCHED"
    )

    print()
    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print()
    print("STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()