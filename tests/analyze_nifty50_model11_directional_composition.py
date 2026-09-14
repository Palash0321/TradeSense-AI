from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL10_DIR = BASE_DIR / "data" / "nifty50_model10_large_move"
FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model11_directional_composition"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    MODEL10_DIR
    / "model10_large_move_predictions.csv"
)

OVERALL_OUTPUT = (
    OUTPUT_DIR
    / "model11_directional_composition.csv"
)

YEARLY_OUTPUT = (
    OUTPUT_DIR
    / "model11_directional_composition_yearly.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

pred = pd.read_csv(PREDICTIONS_FILE)
features = pd.read_csv(FEATURE_FILE)

pred["date"] = pd.to_datetime(pred["date"])
features["date"] = pd.to_datetime(features["date"])


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_pred_cols = [
    "date",
    "probability_large_move",
]

required_feature_cols = [
    "date",
    "nifty_t1_return",
    "target_t1_direction",
]


missing_pred = [
    c for c in required_pred_cols
    if c not in pred.columns
]

missing_features = [
    c for c in required_feature_cols
    if c not in features.columns
]


if missing_pred:
    raise ValueError(
        f"Missing Model 10 prediction columns: {missing_pred}"
    )

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )


# ============================================================
# MERGE MODEL 10 PREDICTIONS WITH ACTUAL T+1 OUTCOME
# ============================================================

df = pred[
    required_pred_cols
].merge(
    features[
        required_feature_cols
    ],
    on="date",
    how="inner",
)


# ============================================================
# REMOVE ROWS WITHOUT REQUIRED INFORMATION
# ============================================================

df = df.dropna(
    subset=[
        "probability_large_move",
        "nifty_t1_return",
        "target_t1_direction",
    ]
).copy()

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(df) < 10:
    raise ValueError(
        f"Too few observations for ranking analysis: {len(df)}"
    )


# ============================================================
# FIXED RANKING BUCKETS
#
# TOP 20%
# MIDDLE 60%
# BOTTOM 20%
#
# Same ranking concept as Model 10A.
# ============================================================

n = len(df)

df["rank"] = (
    df["probability_large_move"]
    .rank(
        method="first",
        ascending=False
    )
)

top_n = int(np.ceil(n * 0.20))
bottom_n = int(np.ceil(n * 0.20))

df["ranking_group"] = "MIDDLE_60"

df.loc[
    df["rank"] <= top_n,
    "ranking_group"
] = "TOP_20"

df.loc[
    df["rank"] > (n - bottom_n),
    "ranking_group"
] = "BOTTOM_20"


# ============================================================
# DIRECTIONAL METRICS
# ============================================================

def calculate_metrics(group):

    returns = group["nifty_t1_return"]

    positive = returns > 0
    negative = returns < 0
    zero = returns == 0

    positive_returns = returns[positive]
    negative_returns = returns[negative]

    return pd.Series({

        "observations":
            len(group),

        "avg_predicted_large_move_probability":
            group[
                "probability_large_move"
            ].mean(),

        "positive_count":
            positive.sum(),

        "negative_count":
            negative.sum(),

        "zero_return_count":
            zero.sum(),

        "positive_pct":
            positive.mean(),

        "negative_pct":
            negative.mean(),

        "zero_return_pct":
            zero.mean(),

        "avg_t1_return":
            returns.mean(),

        "median_t1_return":
            returns.median(),

        "avg_positive_return":
            (
                positive_returns.mean()
                if len(positive_returns) > 0
                else np.nan
            ),

        "avg_negative_return":
            (
                negative_returns.mean()
                if len(negative_returns) > 0
                else np.nan
            ),

        "avg_absolute_return":
            returns.abs().mean(),

        "median_absolute_return":
            returns.abs().median(),

        "large_move_rate":
            (
                returns.abs() >= 0.01
            ).mean(),
    })


# ============================================================
# OVERALL ANALYSIS
# ============================================================

overall = (
    df
    .groupby(
        "ranking_group",
        sort=False
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


group_order = [
    "TOP_20",
    "MIDDLE_60",
    "BOTTOM_20",
]

overall["ranking_group"] = pd.Categorical(
    overall["ranking_group"],
    categories=group_order,
    ordered=True,
)

overall = (
    overall
    .sort_values("ranking_group")
    .reset_index(drop=True)
)


# ============================================================
# TOP VS BOTTOM COMPARISON
# ============================================================

top = overall[
    overall["ranking_group"] == "TOP_20"
].iloc[0]

bottom = overall[
    overall["ranking_group"] == "BOTTOM_20"
].iloc[0]


comparison = pd.DataFrame([{

    "comparison":
        "TOP_20_MINUS_BOTTOM_20",

    "observations_top":
        top["observations"],

    "observations_bottom":
        bottom["observations"],

    "positive_pct_delta":
        (
            top["positive_pct"]
            - bottom["positive_pct"]
        ),

    "negative_pct_delta":
        (
            top["negative_pct"]
            - bottom["negative_pct"]
        ),

    "avg_t1_return_delta":
        (
            top["avg_t1_return"]
            - bottom["avg_t1_return"]
        ),

    "avg_absolute_return_delta":
        (
            top["avg_absolute_return"]
            - bottom["avg_absolute_return"]
        ),

    "large_move_rate_delta":
        (
            top["large_move_rate"]
            - bottom["large_move_rate"]
        ),

    "avg_probability_delta":
        (
            top[
                "avg_predicted_large_move_probability"
            ]
            -
            bottom[
                "avg_predicted_large_move_probability"
            ]
        ),
}])


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

overall.to_csv(
    OVERALL_OUTPUT,
    index=False
)

comparison.to_csv(
    OUTPUT_DIR
    / "model11_top_vs_bottom_comparison.csv",
    index=False
)


# ============================================================
# YEARLY ANALYSIS
# ============================================================

df["year"] = df["date"].dt.year

yearly = (
    df
    .groupby(
        ["year", "ranking_group"],
        sort=True
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)

yearly["ranking_group"] = pd.Categorical(
    yearly["ranking_group"],
    categories=group_order,
    ordered=True,
)

yearly = (
    yearly
    .sort_values(
        ["year", "ranking_group"]
    )
    .reset_index(drop=True)
)

yearly.to_csv(
    YEARLY_OUTPUT,
    index=False
)


# ============================================================
# YEARLY TOP VS BOTTOM
# ============================================================

yearly_comparisons = []

for year in sorted(df["year"].unique()):

    year_data = yearly[
        yearly["year"] == year
    ]

    top_rows = year_data[
        year_data["ranking_group"] == "TOP_20"
    ]

    bottom_rows = year_data[
        year_data["ranking_group"] == "BOTTOM_20"
    ]

    if (
        len(top_rows) != 1
        or len(bottom_rows) != 1
    ):
        continue

    top_y = top_rows.iloc[0]
    bottom_y = bottom_rows.iloc[0]

    yearly_comparisons.append({

        "year":
            year,

        "top_observations":
            top_y["observations"],

        "bottom_observations":
            bottom_y["observations"],

        "top_positive_pct":
            top_y["positive_pct"],

        "bottom_positive_pct":
            bottom_y["positive_pct"],

        "top_negative_pct":
            top_y["negative_pct"],

        "bottom_negative_pct":
            bottom_y["negative_pct"],

        "top_avg_t1_return":
            top_y["avg_t1_return"],

        "bottom_avg_t1_return":
            bottom_y["avg_t1_return"],

        "top_avg_absolute_return":
            top_y["avg_absolute_return"],

        "bottom_avg_absolute_return":
            bottom_y["avg_absolute_return"],

        "top_large_move_rate":
            top_y["large_move_rate"],

        "bottom_large_move_rate":
            bottom_y["large_move_rate"],

        "top_minus_bottom_positive_pct":
            (
                top_y["positive_pct"]
                -
                bottom_y["positive_pct"]
            ),

        "top_minus_bottom_avg_return":
            (
                top_y["avg_t1_return"]
                -
                bottom_y["avg_t1_return"]
            ),

        "top_minus_bottom_large_move_rate":
            (
                top_y["large_move_rate"]
                -
                bottom_y["large_move_rate"]
            ),
    })


yearly_comparison_df = pd.DataFrame(
    yearly_comparisons
)

yearly_comparison_df.to_csv(
    OUTPUT_DIR
    / "model11_top_vs_bottom_yearly.csv",
    index=False
)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("=" * 80)
print("MODEL 11 — DIRECTIONAL COMPOSITION DIAGNOSTIC")
print("=" * 80)

print(
    f"Observations: {len(df)}"
)

print(
    f"Coverage: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print()
print("OVERALL RANKING GROUPS")
print("-" * 80)

display_cols = [

    "ranking_group",

    "observations",

    "avg_predicted_large_move_probability",

    "positive_pct",

    "negative_pct",

    "avg_t1_return",

    "median_t1_return",

    "avg_positive_return",

    "avg_negative_return",

    "avg_absolute_return",

    "large_move_rate",
]

print(
    overall[
        display_cols
    ].to_string(index=False)
)

print()
print("TOP 20% VS BOTTOM 20%")
print("-" * 80)

print(
    comparison.to_string(index=False)
)

print()
print("YEARLY TOP VS BOTTOM")
print("-" * 80)

print(
    yearly_comparison_df
    .to_string(index=False)
)

print()
print("OUTPUT FILES")
print("-" * 80)

print(
    OVERALL_OUTPUT
)

print(
    OUTPUT_DIR
    / "model11_top_vs_bottom_comparison.csv"
)

print(
    YEARLY_OUTPUT
)

print(
    OUTPUT_DIR
    / "model11_top_vs_bottom_yearly.csv"
)

print()
print("=" * 80)
print(
    "MODEL 11 STATUS: "
    "DIAGNOSTIC ONLY — DO NOT DEPLOY"
)
print("=" * 80)