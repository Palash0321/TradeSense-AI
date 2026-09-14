from pathlib import Path
import pandas as pd


# ============================================================
# MODEL 10 — GENUINE FUTURE HOLDOUT LOCK
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
    / "model10_large_move_predictions.csv"
)

HOLDOUT_START_DATE = pd.Timestamp("2026-09-11")

OUTPUT_DIR = BASE_DIR / "data" / "nifty50_model10_large_move"

LOCK_FILE = OUTPUT_DIR / "model10_holdout_lock.csv"


def main():

    print("=" * 70)
    print("NIFTY 50 MODEL 10 — GENUINE FUTURE HOLDOUT LOCK")
    print("=" * 70)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Model 10 predictions file:\n{PREDICTIONS_FILE}"
        )

    data = pd.read_csv(PREDICTIONS_FILE)

    required_columns = {
        "date",
        "actual_large_move",
        "probability_large_move",
        "prediction",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values("date").reset_index(drop=True)

    historical_start = data["date"].min()
    historical_end = data["date"].max()

    historical_rows = len(data)

    # --------------------------------------------------------
    # Check that no existing Model 10 prediction belongs to
    # the future holdout period.
    # --------------------------------------------------------

    accidental_holdout_rows = data[
        data["date"] >= HOLDOUT_START_DATE
    ]

    accidental_count = len(accidental_holdout_rows)

    # --------------------------------------------------------
    # Fixed holdout specification
    # --------------------------------------------------------

    lock = pd.DataFrame(
        [
            {
                "model": "Model 10",
                "target": "ABS_T1_RETURN_GE_1_PERCENT",
                "holdout_start_date": HOLDOUT_START_DATE.date(),
                "historical_prediction_start": historical_start.date(),
                "historical_prediction_end": historical_end.date(),
                "historical_prediction_rows": historical_rows,
                "accidental_holdout_rows_present": accidental_count,
                "ranking_method": "GLOBAL_TOP_BOTTOM_20_PERCENT",
                "top_fraction": 0.20,
                "bottom_fraction": 0.20,
                "model_frozen": True,
                "features_frozen": True,
                "threshold_optimization_allowed": False,
                "future_retraining_allowed_before_holdout_evaluation": False,
                "status": (
                    "PASS"
                    if accidental_count == 0
                    else "FAIL"
                ),
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lock.to_csv(LOCK_FILE, index=False)

    print()
    print(f"Historical predictions : {historical_rows}")
    print(
        f"Historical coverage    : "
        f"{historical_start.date()} -> {historical_end.date()}"
    )
    print(f"Holdout starts         : {HOLDOUT_START_DATE.date()}")
    print(f"Existing holdout rows  : {accidental_count}")

    print()
    print("FIXED HOLDOUT RULES")
    print("-" * 70)
    print("Target                 : |T+1 return| >= 1.0%")
    print("Ranking                : Global probability ranking")
    print("Top group              : Top 20%")
    print("Bottom group           : Bottom 20%")
    print("Threshold optimization : NOT ALLOWED")
    print("Feature changes         : NOT ALLOWED")
    print("Model changes           : NOT ALLOWED")
    print("Future-result tuning    : NOT ALLOWED")

    print()
    print(f"Lock file written      : {LOCK_FILE}")

    print()
    if accidental_count == 0:
        print("STATUS: PASS")
        print()
        print(
            "The genuine future holdout is cleanly separated from "
            "all existing Model 10 predictions."
        )
    else:
        print("STATUS: FAIL")
        print(
            "Existing Model 10 predictions already contain observations "
            "inside the intended holdout period."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()