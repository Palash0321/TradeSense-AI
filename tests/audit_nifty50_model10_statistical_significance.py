from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Fixed in advance — same ranking definition as Model 10A
TOP_FRACTION = 0.20
BOTTOM_FRACTION = 0.20

# Statistical settings
BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 10000

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL10_FILE = (
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

BOOTSTRAP_OUTPUT = (
    OUTPUT_DIR
    / "model10_bootstrap_results.csv"
)

PERMUTATION_OUTPUT = (
    OUTPUT_DIR
    / "model10_permutation_results.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "model10_statistical_significance_summary.csv"
)


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# LOAD MODEL 10 PREDICTIONS
# ============================================================

df = pd.read_csv(MODEL10_FILE)

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
        f"Missing required Model 10 columns: "
        f"{missing_columns}"
    )


df["date"] = pd.to_datetime(df["date"])

df = (
    df[
        required_columns
    ]
    .dropna()
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if len(df) < 20:
    raise ValueError(
        f"Too few observations: {len(df)}"
    )


df["actual_large_move"] = (
    df["actual_large_move"]
    .astype(int)
)

df["probability_large_move"] = (
    df["probability_large_move"]
    .astype(float)
)


# ============================================================
# FIXED TOP / BOTTOM 20% RANKING
# ============================================================

n = len(df)

top_n = int(
    np.ceil(
        n * TOP_FRACTION
    )
)

bottom_n = int(
    np.ceil(
        n * BOTTOM_FRACTION
    )
)


def calculate_top_bottom_difference(
    probabilities,
    outcomes
):
    """
    Rank observations by probability and calculate:

    TOP 20% event rate
    minus
    BOTTOM 20% event rate
    """

    probabilities = np.asarray(
        probabilities
    )

    outcomes = np.asarray(
        outcomes
    )

    order = np.argsort(
        -probabilities,
        kind="mergesort"
    )

    top_indices = order[:top_n]

    bottom_indices = order[
        -bottom_n:
    ]

    top_rate = outcomes[
        top_indices
    ].mean()

    bottom_rate = outcomes[
        bottom_indices
    ].mean()

    difference = (
        top_rate
        - bottom_rate
    )

    return (
        top_rate,
        bottom_rate,
        difference
    )


# ============================================================
# OBSERVED STATISTIC
# ============================================================

observed_top_rate, observed_bottom_rate, observed_difference = (
    calculate_top_bottom_difference(
        df["probability_large_move"].to_numpy(),
        df["actual_large_move"].to_numpy()
    )
)


# ============================================================
# BOOTSTRAP
#
# Resample paired observations with replacement.
#
# This estimates uncertainty around the observed
# TOP-vs-BOTTOM ranking difference.
# ============================================================

bootstrap_differences = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=float
)

bootstrap_top_rates = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=float
)

bootstrap_bottom_rates = np.empty(
    BOOTSTRAP_ITERATIONS,
    dtype=float
)


probabilities = (
    df["probability_large_move"]
    .to_numpy()
)

outcomes = (
    df["actual_large_move"]
    .to_numpy()
)


for i in range(
    BOOTSTRAP_ITERATIONS
):

    sample_indices = rng.integers(
        0,
        n,
        size=n
    )

    sample_probabilities = (
        probabilities[
            sample_indices
        ]
    )

    sample_outcomes = (
        outcomes[
            sample_indices
        ]
    )

    (
        top_rate,
        bottom_rate,
        difference
    ) = calculate_top_bottom_difference(
        sample_probabilities,
        sample_outcomes
    )

    bootstrap_top_rates[i] = (
        top_rate
    )

    bootstrap_bottom_rates[i] = (
        bottom_rate
    )

    bootstrap_differences[i] = (
        difference
    )


# ============================================================
# BOOTSTRAP 95% CONFIDENCE INTERVAL
#
# Percentile method.
# ============================================================

bootstrap_ci_lower = np.percentile(
    bootstrap_differences,
    2.5
)

bootstrap_ci_upper = np.percentile(
    bootstrap_differences,
    97.5
)


bootstrap_probability_positive = (
    np.mean(
        bootstrap_differences > 0
    )
)


# ============================================================
# PERMUTATION TEST
#
# Keep the actual outcomes fixed.
#
# Randomly shuffle the predicted probabilities,
# destroying the relationship between predictions
# and outcomes.
#
# Recalculate TOP-vs-BOTTOM difference.
#
# This tests the null hypothesis that the ranking
# contains no information about large-move outcomes.
# ============================================================

permutation_differences = np.empty(
    PERMUTATION_ITERATIONS,
    dtype=float
)


for i in range(
    PERMUTATION_ITERATIONS
):

    shuffled_probabilities = rng.permutation(
        probabilities
    )

    (
        top_rate,
        bottom_rate,
        difference
    ) = calculate_top_bottom_difference(
        shuffled_probabilities,
        outcomes
    )

    permutation_differences[i] = (
        difference
    )


# ============================================================
# ONE-SIDED PERMUTATION P-VALUE
#
# We specifically test whether the observed
# TOP 20% rate is greater than the BOTTOM 20% rate.
#
# +1 correction prevents zero p-values.
# ============================================================

permutation_extreme_count = (
    np.sum(
        permutation_differences
        >= observed_difference
    )
)

permutation_p_value = (
    permutation_extreme_count + 1
) / (
    PERMUTATION_ITERATIONS + 1
)


# ============================================================
# TWO-SIDED PERMUTATION P-VALUE
#
# Useful as an additional robustness reference.
# ============================================================

permutation_two_sided_count = (
    np.sum(
        np.abs(
            permutation_differences
        )
        >= abs(
            observed_difference
        )
    )
)

permutation_two_sided_p_value = (
    permutation_two_sided_count + 1
) / (
    PERMUTATION_ITERATIONS + 1
)


# ============================================================
# BOOTSTRAP OUTPUT
# ============================================================

bootstrap_results = pd.DataFrame({
    "bootstrap_iteration":
        np.arange(
            1,
            BOOTSTRAP_ITERATIONS + 1
        ),

    "top_20_rate":
        bootstrap_top_rates,

    "bottom_20_rate":
        bootstrap_bottom_rates,

    "top_minus_bottom":
        bootstrap_differences,
})

bootstrap_results.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False
)


# ============================================================
# PERMUTATION OUTPUT
# ============================================================

permutation_results = pd.DataFrame({
    "permutation_iteration":
        np.arange(
            1,
            PERMUTATION_ITERATIONS + 1
        ),

    "randomized_top_minus_bottom":
        permutation_differences,
})

permutation_results.to_csv(
    PERMUTATION_OUTPUT,
    index=False
)


# ============================================================
# INTERPRETATION FLAGS
# ============================================================

bootstrap_ci_excludes_zero = (
    bootstrap_ci_lower > 0
    or bootstrap_ci_upper < 0
)

permutation_significant_05 = (
    permutation_p_value < 0.05
)

permutation_significant_01 = (
    permutation_p_value < 0.01
)


if (
    observed_difference > 0
    and bootstrap_ci_lower > 0
    and permutation_p_value < 0.05
):
    statistical_status = (
        "STATISTICALLY INTERESTING"
    )

elif (
    observed_difference > 0
    and (
        bootstrap_ci_lower > 0
        or permutation_p_value < 0.10
    )
):
    statistical_status = (
        "WEAK / BORDERLINE EVIDENCE"
    )

else:
    statistical_status = (
        "NO STRONG STATISTICAL EVIDENCE"
    )


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "observations":
        n,

    "coverage_start":
        df["date"].min().date(),

    "coverage_end":
        df["date"].max().date(),

    "top_20_observations":
        top_n,

    "bottom_20_observations":
        bottom_n,

    "overall_large_move_rate":
        outcomes.mean(),

    "observed_top_20_rate":
        observed_top_rate,

    "observed_bottom_20_rate":
        observed_bottom_rate,

    "observed_top_minus_bottom":
        observed_difference,

    "bootstrap_iterations":
        BOOTSTRAP_ITERATIONS,

    "bootstrap_ci_95_lower":
        bootstrap_ci_lower,

    "bootstrap_ci_95_upper":
        bootstrap_ci_upper,

    "bootstrap_probability_difference_positive":
        bootstrap_probability_positive,

    "permutation_iterations":
        PERMUTATION_ITERATIONS,

    "permutation_one_sided_p_value":
        permutation_p_value,

    "permutation_two_sided_p_value":
        permutation_two_sided_p_value,

    "bootstrap_ci_excludes_zero":
        bootstrap_ci_excludes_zero,

    "permutation_significant_at_5pct":
        permutation_significant_05,

    "permutation_significant_at_1pct":
        permutation_significant_01,

    "statistical_status":
        statistical_status,
}])

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("=" * 80)
print(
    "MODEL 10 — STATISTICAL SIGNIFICANCE AUDIT"
)
print("=" * 80)

print(
    f"Observations: {n}"
)

print(
    f"Coverage: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print()
print(
    "FIXED RANKING"
)
print("-" * 80)

print(
    f"TOP 20% observations: {top_n}"
)

print(
    f"BOTTOM 20% observations: {bottom_n}"
)

print(
    f"Overall large-move rate: "
    f"{outcomes.mean():.6f}"
)

print(
    f"Observed TOP 20% rate: "
    f"{observed_top_rate:.6f}"
)

print(
    f"Observed BOTTOM 20% rate: "
    f"{observed_bottom_rate:.6f}"
)

print(
    f"Observed TOP-BOTTOM difference: "
    f"{observed_difference:.6f}"
)

print()
print(
    "BOOTSTRAP"
)
print("-" * 80)

print(
    f"Iterations: "
    f"{BOOTSTRAP_ITERATIONS}"
)

print(
    f"95% CI lower: "
    f"{bootstrap_ci_lower:.6f}"
)

print(
    f"95% CI upper: "
    f"{bootstrap_ci_upper:.6f}"
)

print(
    f"Probability bootstrap difference > 0: "
    f"{bootstrap_probability_positive:.6f}"
)

print()
print(
    "PERMUTATION TEST"
)
print("-" * 80)

print(
    f"Iterations: "
    f"{PERMUTATION_ITERATIONS}"
)

print(
    f"One-sided p-value: "
    f"{permutation_p_value:.6f}"
)

print(
    f"Two-sided p-value: "
    f"{permutation_two_sided_p_value:.6f}"
)

print()
print(
    "STATISTICAL INTERPRETATION"
)
print("-" * 80)

print(
    f"Bootstrap CI excludes zero: "
    f"{bootstrap_ci_excludes_zero}"
)

print(
    f"Permutation significant at 5%: "
    f"{permutation_significant_05}"
)

print(
    f"Permutation significant at 1%: "
    f"{permutation_significant_01}"
)

print(
    f"STATUS: {statistical_status}"
)

print()
print(
    "OUTPUT FILES"
)
print("-" * 80)

print(
    BOOTSTRAP_OUTPUT
)

print(
    PERMUTATION_OUTPUT
)

print(
    SUMMARY_OUTPUT
)

print()
print("=" * 80)
print(
    "MODEL 10 STATUS: "
    "STATISTICAL AUDIT COMPLETE — "
    "NOT FOR PRODUCTION"
)
print("=" * 80)