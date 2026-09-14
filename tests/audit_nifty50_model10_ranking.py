from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
    / "model10_large_move_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model10_large_move"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "model10_ranking_audit.csv"
)

YEARLY_FILE = (
    OUTPUT_DIR
    / "model10_ranking_yearly.csv"
)


# ============================================================
# FIXED RANKING GROUP
#
# Predefined:
#   TOP 20%    = highest predicted probabilities
#   BOTTOM 20% = lowest predicted probabilities
#
# No threshold optimization.
# ============================================================

QUANTILE = 0.20


# ============================================================
# LOAD EXISTING MODEL 10 PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTIONS_FILE,
    parse_dates=["date"],
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


required_columns = [
    "date",
    "actual_large_move",
    "probability_large_move",
]

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
# VALIDATE
# ============================================================

df = df.dropna(
    subset=[
        "actual_large_move",
        "probability_large_move",
    ]
).reset_index(drop=True)

df["actual_large_move"] = (
    df["actual_large_move"]
    .astype(int)
)

df["probability_large_move"] = (
    df["probability_large_move"]
    .astype(float)
)


# ============================================================
# FIXED TOP/BOTTOM 20% RANKING
# ============================================================

n = len(df)

group_size = int(
    np.floor(n * QUANTILE)
)

if group_size < 1:
    raise ValueError(
        "Not enough observations for ranking audit."
    )


ranked = df.sort_values(
    "probability_large_move",
    ascending=False,
).reset_index(drop=True)

ranked["ranking_group"] = "MIDDLE_60"

ranked.loc[
    :group_size - 1,
    "ranking_group",
] = "TOP_20"

ranked.loc[
    n - group_size:,
    "ranking_group",
] = "BOTTOM_20"


# ============================================================
# OVERALL SUMMARY
# ============================================================

overall_event_rate = (
    ranked["actual_large_move"]
    .mean()
)

summary_rows = []

for group_name in [
    "TOP_20",
    "MIDDLE_60",
    "BOTTOM_20",
]:

    group = ranked[
        ranked["ranking_group"]
        == group_name
    ]

    event_rate = (
        group["actual_large_move"]
        .mean()
    )

    average_probability = (
        group["probability_large_move"]
        .mean()
    )

    summary_rows.append(
        {
            "group": group_name,
            "observations": len(group),
            "actual_large_move_rate": (
                event_rate
            ),
            "average_predicted_probability": (
                average_probability
            ),
            "lift_vs_overall": (
                event_rate
                / overall_event_rate
                if overall_event_rate > 0
                else np.nan
            ),
            "delta_vs_overall": (
                event_rate
                - overall_event_rate
            ),
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# YEARLY RANKING AUDIT
# ============================================================

yearly_rows = []

ranked["year"] = (
    ranked["date"].dt.year
)

for year, year_group in (
    ranked.groupby("year")
):

    year_group = (
        year_group
        .sort_values(
            "probability_large_move",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    year_n = len(year_group)

    year_group_size = int(
        np.floor(
            year_n * QUANTILE
        )
    )

    if year_group_size < 1:
        continue

    top = year_group.iloc[
        :year_group_size
    ]

    bottom = year_group.iloc[
        -year_group_size:
    ]

    overall_rate = (
        year_group["actual_large_move"]
        .mean()
    )

    top_rate = (
        top["actual_large_move"]
        .mean()
    )

    bottom_rate = (
        bottom["actual_large_move"]
        .mean()
    )

    yearly_rows.append(
        {
            "year": year,
            "observations": year_n,

            "overall_large_move_rate": (
                overall_rate
            ),

            "top_20_observations": (
                len(top)
            ),

            "top_20_large_move_rate": (
                top_rate
            ),

            "top_20_lift": (
                top_rate / overall_rate
                if overall_rate > 0
                else np.nan
            ),

            "bottom_20_observations": (
                len(bottom)
            ),

            "bottom_20_large_move_rate": (
                bottom_rate
            ),

            "bottom_20_lift": (
                bottom_rate / overall_rate
                if overall_rate > 0
                else np.nan
            ),

            "top_minus_bottom": (
                top_rate
                - bottom_rate
            ),
        }
    )


yearly_df = pd.DataFrame(
    yearly_rows
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    OUTPUT_FILE,
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
print("MODEL 10A")
print("LARGE-MOVE RANKING AUDIT")
print("=" * 80)

print(
    f"Prediction file: "
    f"{PREDICTIONS_FILE}"
)

print(
    f"Observations: {len(ranked)}"
)

print(
    f"Overall large-move rate: "
    f"{overall_event_rate:.4f}"
)

print(
    f"Fixed ranking group: "
    f"TOP/BOTTOM {QUANTILE:.0%}"
)

print()

print("=" * 80)
print("OVERALL RANKING RESULTS")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print()

print("=" * 80)
print("YEARLY RANKING RESULTS")
print("=" * 80)

print(
    yearly_df.to_string(
        index=False
    )
)

print()

print("=" * 80)
print("MODEL 10A COMPLETE")
print("=" * 80)

print()

print(
    f"Output: {OUTPUT_FILE}"
)

print(
    f"Yearly: {YEARLY_FILE}"
)