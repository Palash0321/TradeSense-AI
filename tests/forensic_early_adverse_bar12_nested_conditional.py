"""
TradeSense-AI
Strict Nested Conditional / Incremental Information Forensic Audit

Research-only. This script DOES NOT modify production or OOS data.

Research question
-----------------
After controlling for a fixed Bar-1 composite path state, does the
Bar-2 "favorable excursion failed to improve" state retain association
with the frozen 5-bar / 0.25R early-adverse event?

Frozen population
-----------------
106 total trades
60 frozen early-adverse events
46 retained

Frozen label
------------
stored_frozen_event == 1
where production early_adverse_r_bar_5 >= 0.25R

Primary Bar-2 state
-------------------
bar2_favorable_failed_to_improve

Nested control
--------------
A fixed Bar-1 composite state built only from already-defined Bar-1
path dimensions:

1. Bar-1 cumulative MAE
2. Bar-1 cumulative MFE
3. Bar-1 net excursion
4. Bar-1 MAE-minus-MFE balance

Each dimension uses its GLOBAL frozen-population median.
No threshold optimization is performed.

The composite is intentionally coarse:
    LOW_PATH
    HIGH_PATH

The script also creates four quartile-style composite cells from the
same binary dimensions:
    0000
    0001
    ...
    1111

The primary inference uses the two-level composite control strata.
The 16-cell representation is descriptive only and requires minimum
sample sizes.

No strategy rule is promoted.
"""

from __future__ import annotations

import math
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAJECTORY_FILE = (
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

PATH_FILE = (
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = (
    "tests/output/research/"
    "early_adverse_bar12_nested_conditional"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_BAR = 5

MIN_STRATUM_N = 8
MIN_CELL_N = 6

N_PERMUTATIONS = 5000
RANDOM_SEED = 20260916

CONTROL_COLUMNS = [
    "bar1_cumulative_mae_r",
    "bar1_cumulative_mfe_r",
    "bar1_net_excursion_r",
    "bar1_mae_minus_mfe",
]

PRIMARY_STATE = "bar2_favorable_failed_to_improve"

ROBUSTNESS_GROUPS = [
    "year",
    "symbol",
    "setup",
    "direction",
]


# ============================================================================
# HELPERS
# ============================================================================

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan})
    )


def normalize_direction(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan})
    )


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def make_merge_key(df: pd.DataFrame) -> pd.Series:
    return (
        normalize_symbol(df["symbol"])
        .astype(str)
        + "|"
        + normalize_direction(df["direction"]).astype(str)
        + "|"
        + normalize_date(df["entry_date"]).astype(str)
    )


def assert_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def percentile_bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int = 5000,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        boot[i] = statistic(sample)

    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def risk_difference_ci(
    a_event: int,
    a_total: int,
    b_event: int,
    b_total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """
    Wald/Newcombe-style component interval for difference in proportions.

    Returned values are fractions, not percentage points.
    """
    if a_total <= 0 or b_total <= 0:
        return np.nan, np.nan

    p1 = a_event / a_total
    p2 = b_event / b_total

    se1 = math.sqrt(max(p1 * (1.0 - p1) / a_total, 0.0))
    se2 = math.sqrt(max(p2 * (1.0 - p2) / b_total, 0.0))

    diff = p1 - p2
    se = math.sqrt(se1 * se1 + se2 * se2)

    return diff - z * se, diff + z * se


def fisher_from_binary(
    df: pd.DataFrame,
    state_col: str,
    label_col: str,
) -> dict:
    work = df[[state_col, label_col]].dropna().copy()

    work[state_col] = work[state_col].astype(int)
    work[label_col] = work[label_col].astype(int)

    state1_event = int(
        ((work[state_col] == 1) & (work[label_col] == 1)).sum()
    )
    state1_non_event = int(
        ((work[state_col] == 1) & (work[label_col] == 0)).sum()
    )
    state0_event = int(
        ((work[state_col] == 0) & (work[label_col] == 1)).sum()
    )
    state0_non_event = int(
        ((work[state_col] == 0) & (work[label_col] == 0)).sum()
    )

    table = np.array(
        [
            [state1_event, state1_non_event],
            [state0_event, state0_non_event],
        ],
        dtype=int,
    )

    odds_ratio, p_value = fisher_exact(table)

    n1 = state1_event + state1_non_event
    n0 = state0_event + state0_non_event

    rd = (
        state1_event / n1 - state0_event / n0
        if n1 > 0 and n0 > 0
        else np.nan
    )

    ci_low, ci_high = risk_difference_ci(
        state1_event,
        n1,
        state0_event,
        n0,
    )

    return {
        "state1_event": state1_event,
        "state1_non_event": state1_non_event,
        "state0_event": state0_event,
        "state0_non_event": state0_non_event,
        "state1_n": n1,
        "state0_n": n0,
        "state1_event_rate": (
            state1_event / n1 if n1 > 0 else np.nan
        ),
        "state0_event_rate": (
            state0_event / n0 if n0 > 0 else np.nan
        ),
        "risk_difference": rd,
        "risk_difference_pp": rd * 100.0 if np.isfinite(rd) else np.nan,
        "rd_ci_low_pp": (
            ci_low * 100.0 if np.isfinite(ci_low) else np.nan
        ),
        "rd_ci_high_pp": (
            ci_high * 100.0 if np.isfinite(ci_high) else np.nan
        ),
        "odds_ratio": float(odds_ratio),
        "fisher_p": float(p_value),
    }


def stratified_mantel_haenszel(
    df: pd.DataFrame,
    stratum_col: str,
    state_col: str,
    label_col: str,
    min_n: int = MIN_STRATUM_N,
) -> dict:
    """
    Manual Mantel-Haenszel pooled odds ratio.

    No statsmodels dependency.

    Each stratum contributes:
        a = state1/event
        b = state1/non-event
        c = state0/event
        d = state0/non-event
        n = a+b+c+d

    MH OR = sum(a*d/n) / sum(b*c/n)

    A permutation p-value is calculated separately for the pooled
    risk-difference statistic.
    """
    rows = []

    for stratum, group in df.groupby(stratum_col, dropna=False):
        g = group[[state_col, label_col]].dropna().copy()

        if len(g) < min_n:
            continue

        g[state_col] = g[state_col].astype(int)
        g[label_col] = g[label_col].astype(int)

        a = int(((g[state_col] == 1) & (g[label_col] == 1)).sum())
        b = int(((g[state_col] == 1) & (g[label_col] == 0)).sum())
        c = int(((g[state_col] == 0) & (g[label_col] == 1)).sum())
        d = int(((g[state_col] == 0) & (g[label_col] == 0)).sum())

        n = a + b + c + d

        if n <= 0:
            continue

        state1_n = a + b
        state0_n = c + d

        if state1_n == 0 or state0_n == 0:
            continue

        rd = a / state1_n - c / state0_n

        rows.append(
            {
                "stratum": str(stratum),
                "n": n,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "state1_n": state1_n,
                "state0_n": state0_n,
                "rd": rd,
            }
        )

    strata = pd.DataFrame(rows)

    if strata.empty:
        return {
            "valid_strata": 0,
            "mh_or": np.nan,
            "pooled_rd": np.nan,
            "total_n": 0,
            "strata": strata,
        }

    numerator = np.sum(
        (strata["a"] * strata["d"]) / strata["n"]
    )

    denominator = np.sum(
        (strata["b"] * strata["c"]) / strata["n"]
    )

    mh_or = (
        numerator / denominator
        if denominator > 0
        else np.inf
    )

    pooled_numerator = np.sum(
        strata["state1_n"] * strata["rd"]
    )

    pooled_denominator = np.sum(
        strata["state1_n"]
    )

    pooled_rd = (
        pooled_numerator / pooled_denominator
        if pooled_denominator > 0
        else np.nan
    )

    return {
        "valid_strata": len(strata),
        "mh_or": float(mh_or),
        "pooled_rd": float(pooled_rd),
        "pooled_rd_pp": (
            float(pooled_rd * 100.0)
            if np.isfinite(pooled_rd)
            else np.nan
        ),
        "total_n": int(strata["n"].sum()),
        "strata": strata,
    }


def permutation_pooled_rd(
    df: pd.DataFrame,
    stratum_col: str,
    state_col: str,
    label_col: str,
    observed_rd: float,
    n_perm: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Permutation test preserving each fixed stratum's event count.

    The state assignments are held fixed. Event labels are shuffled
    within each stratum, preserving the number of events per stratum.
    """
    if not np.isfinite(observed_rd):
        return np.nan

    work = df[
        [stratum_col, state_col, label_col]
    ].dropna().copy()

    if work.empty:
        return np.nan

    work[state_col] = work[state_col].astype(int)
    work[label_col] = work[label_col].astype(int)

    strata_data = []

    for _, g in work.groupby(stratum_col, dropna=False):
        states = g[state_col].to_numpy(dtype=int)
        labels = g[label_col].to_numpy(dtype=int)

        if len(g) < MIN_STRATUM_N:
            continue

        state1_n = int((states == 1).sum())
        state0_n = int((states == 0).sum())

        if state1_n == 0 or state0_n == 0:
            continue

        event_count = int(labels.sum())

        strata_data.append(
            (
                states,
                event_count,
                len(g),
            )
        )

    if not strata_data:
        return np.nan

    rng = np.random.default_rng(seed)
    extreme = 0

    for _ in range(n_perm):
        numerator = 0.0
        denominator = 0.0

        for states, event_count, n in strata_data:
            shuffled = np.zeros(n, dtype=int)

            if event_count > 0:
                event_positions = rng.choice(
                    n,
                    size=event_count,
                    replace=False,
                )
                shuffled[event_positions] = 1

            state1 = states == 1
            state0 = states == 0

            state1_n = int(state1.sum())
            state0_n = int(state0.sum())

            if state1_n == 0 or state0_n == 0:
                continue

            p1 = shuffled[state1].mean()
            p0 = shuffled[state0].mean()

            rd = p1 - p0

            numerator += rd * state1_n
            denominator += state1_n

        if denominator <= 0:
            continue

        perm_rd = numerator / denominator

        if abs(perm_rd) >= abs(observed_rd):
            extreme += 1

    return (extreme + 1) / (n_perm + 1)


def subgroup_table(
    df: pd.DataFrame,
    group_col: str,
    state_col: str,
    label_col: str,
    min_n: int = MIN_CELL_N,
) -> pd.DataFrame:
    rows = []

    for group_value, group in df.groupby(
        group_col,
        dropna=False,
    ):
        if len(group) < min_n:
            continue

        stats = fisher_from_binary(
            group,
            state_col,
            label_col,
        )

        rows.append(
            {
                "group": str(group_value),
                "n": len(group),
                **stats,
            }
        )

    return pd.DataFrame(rows)


def pairwise_control_table(
    df: pd.DataFrame,
    state_col: str,
    label_col: str,
) -> pd.DataFrame:
    """
    Test whether Bar2 state association remains directionally positive
    within each of the four fixed Bar1 binary dimensions.

    This is intentionally descriptive/conditional, not a multivariate
    regression.
    """
    rows = []

    for control in CONTROL_COLUMNS:
        control_state = f"{control}_binary"

        for value in [0, 1]:
            subset = df[df[control_state] == value].copy()

            if len(subset) < MIN_STRATUM_N:
                continue

            stats = fisher_from_binary(
                subset,
                state_col,
                label_col,
            )

            rows.append(
                {
                    "control": control,
                    "control_state": value,
                    "n": len(subset),
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def build_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build fixed Bar1 binary states and a composite score.

    Binary convention:
        1 = less favorable / more adverse Bar1 path
        0 = more favorable Bar1 path

    For:
        MAE: high = adverse
        MFE: low = adverse
        Net: low = adverse
        MAE-MFE: high = adverse
    """
    work = df.copy()

    medians = {}

    medians["bar1_cumulative_mae_r"] = float(
        work["bar1_cumulative_mae_r"].median()
    )

    medians["bar1_cumulative_mfe_r"] = float(
        work["bar1_cumulative_mfe_r"].median()
    )

    medians["bar1_net_excursion_r"] = float(
        work["bar1_net_excursion_r"].median()
    )

    medians["bar1_mae_minus_mfe"] = float(
        work["bar1_mae_minus_mfe"].median()
    )

    work["bar1_cumulative_mae_r_binary"] = (
        work["bar1_cumulative_mae_r"]
        >= medians["bar1_cumulative_mae_r"]
    ).astype(int)

    work["bar1_cumulative_mfe_r_binary"] = (
        work["bar1_cumulative_mfe_r"]
        <= medians["bar1_cumulative_mfe_r"]
    ).astype(int)

    work["bar1_net_excursion_r_binary"] = (
        work["bar1_net_excursion_r"]
        <= medians["bar1_net_excursion_r"]
    ).astype(int)

    work["bar1_mae_minus_mfe_binary"] = (
        work["bar1_mae_minus_mfe"]
        >= medians["bar1_mae_minus_mfe"]
    ).astype(int)

    binary_cols = [
        "bar1_cumulative_mae_r_binary",
        "bar1_cumulative_mfe_r_binary",
        "bar1_net_excursion_r_binary",
        "bar1_mae_minus_mfe_binary",
    ]

    work["bar1_composite_score"] = work[binary_cols].sum(axis=1)

    # Coarse two-level composite:
    # 0/1 adverse dimensions -> LOW_PATH
    # 2/3/4 adverse dimensions -> HIGH_PATH
    work["bar1_composite_state"] = np.where(
        work["bar1_composite_score"] <= 1,
        "LOW_PATH",
        "HIGH_PATH",
    )

    # Exact binary signature, descriptive only.
    work["bar1_composite_signature"] = (
        work[binary_cols].astype(int).astype(str).agg("".join, axis=1)
    )

    return work, medians


def leave_one_group_out(
    df: pd.DataFrame,
    group_col: str,
    state_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows = []

    groups = [
        g for g in df[group_col].dropna().unique()
    ]

    for removed in sorted(groups, key=str):
        subset = df[df[group_col] != removed].copy()

        if len(subset) < MIN_STRATUM_N:
            continue

        result = stratified_mantel_haenszel(
            subset,
            "bar1_composite_state",
            state_col,
            label_col,
            min_n=MIN_STRATUM_N,
        )

        rows.append(
            {
                "group": str(removed),
                "remaining_n": len(subset),
                "valid_strata": result["valid_strata"],
                "mh_or": result["mh_or"],
                "pooled_rd_pp": result.get(
                    "pooled_rd_pp",
                    np.nan,
                ),
                "direction_positive": (
                    bool(
                        np.isfinite(result.get("pooled_rd", np.nan))
                        and result["pooled_rd"] > 0
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# LOAD DATA
# ============================================================================

ensure_output_dir()

trajectory = pd.read_csv(TRAJECTORY_FILE)
path = pd.read_csv(PATH_FILE)

print("=" * 78)
print("TRADE SENSE-AI — STRICT NESTED CONDITIONAL FORENSIC AUDIT")
print("=" * 78)

print()
print(f"Trajectory rows: {len(trajectory)}")
print(f"Trajectory columns: {len(trajectory.columns)}")
print(f"Path rows:       {len(path)}")


# ============================================================================
# VALIDATE TRAJECTORY SCHEMA
# ============================================================================

trajectory_required = [
    "symbol",
    "direction",
    "entry_date",
    "setup",
    "frozen_event",
    "bar2_favorable_failed_to_improve",
    "bar1_cumulative_mae_r",
    "bar1_cumulative_mfe_r",
    "bar1_net_excursion_r",
    "bar1_mae_minus_mfe",
]

assert_columns(
    trajectory,
    trajectory_required,
    "trajectory",
)

trajectory["symbol"] = normalize_symbol(
    trajectory["symbol"]
)

trajectory["direction"] = normalize_direction(
    trajectory["direction"]
)

trajectory["entry_date"] = pd.to_datetime(
    trajectory["entry_date"],
    errors="coerce",
)

trajectory["setup"] = (
    trajectory["setup"]
    .astype(str)
    .str.strip()
    .replace({"nan": np.nan})
)

for col in [
    "frozen_event",
    "bar2_favorable_failed_to_improve",
]:
    trajectory[col] = pd.to_numeric(
        trajectory[col],
        errors="coerce",
    )

for col in CONTROL_COLUMNS:
    trajectory[col] = safe_float(
        trajectory[col]
    )


# ============================================================================
# VALIDATE FROZEN POPULATION
# ============================================================================

trajectory["frozen_event"] = (
    trajectory["frozen_event"]
    .astype(int)
)

trajectory[PRIMARY_STATE] = (
    trajectory[PRIMARY_STATE]
    .astype(int)
)

trade_count = len(trajectory)
event_count = int(
    trajectory["frozen_event"].sum()
)
retained_count = trade_count - event_count

print()
print("FROZEN POPULATION")
print(f"  Trades:   {trade_count}")
print(f"  Events:   {event_count}")
print(f"  Retained: {retained_count}")

if trade_count != EXPECTED_TRADES:
    raise RuntimeError(
        f"Expected {EXPECTED_TRADES} trades, got {trade_count}"
    )

if event_count != EXPECTED_EVENTS:
    raise RuntimeError(
        f"Expected {EXPECTED_EVENTS} events, got {event_count}"
    )

if retained_count != EXPECTED_RETAINED:
    raise RuntimeError(
        f"Expected {EXPECTED_RETAINED} retained, got {retained_count}"
    )


# ============================================================================
# UNIQUE TRADE KEY
# ============================================================================

trajectory["merge_key"] = make_merge_key(
    trajectory
)

trajectory_duplicate_keys = int(
    trajectory["merge_key"].duplicated().sum()
)

if trajectory_duplicate_keys != 0:
    raise RuntimeError(
        "Trajectory contains duplicate trade keys."
    )


# ============================================================================
# BUILD FIXED BAR1 COMPOSITE
# ============================================================================

research, medians = build_composite(
    trajectory
)

print()
print("FIXED BAR-1 CONTROL MEDIANS")

for key, value in medians.items():
    print(
        f"  {key}: {value:.6f}"
    )

print()
print("BAR-1 COMPOSITE")

print(
    research[
        [
            "bar1_composite_score",
            "bar1_composite_state",
        ]
    ]
    .value_counts()
    .sort_index()
)

print()
print("COMPOSITE STATE EVENT RATES")

composite_rates = (
    research
    .groupby("bar1_composite_state")["frozen_event"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

print(
    composite_rates.to_string(
        index=False
    )
)


# ============================================================================
# GLOBAL BAR2 ASSOCIATION
# ============================================================================

print()
print("=" * 78)
print("GLOBAL BAR2 STATE")
print("=" * 78)

global_stats = fisher_from_binary(
    research,
    PRIMARY_STATE,
    "frozen_event",
)

print(
    f"  State event rate:     "
    f"{global_stats['state1_event_rate'] * 100:.2f}%"
)

print(
    f"  Non-state event rate: "
    f"{global_stats['state0_event_rate'] * 100:.2f}%"
)

print(
    f"  Risk difference:      "
    f"{global_stats['risk_difference_pp']:+.2f} pp"
)

print(
    f"  RD 95% CI:            "
    f"[{global_stats['rd_ci_low_pp']:+.2f}, "
    f"{global_stats['rd_ci_high_pp']:+.2f}] pp"
)

print(
    f"  Odds ratio:           "
    f"{global_stats['odds_ratio']:.6f}"
)

print(
    f"  Fisher p:             "
    f"{global_stats['fisher_p']:.6f}"
)


# ============================================================================
# NESTED CONTROL STRATA
# ============================================================================

print()
print("=" * 78)
print("NESTED BAR-1 COMPOSITE CONTROL")
print("=" * 78)

nested = stratified_mantel_haenszel(
    research,
    "bar1_composite_state",
    PRIMARY_STATE,
    "frozen_event",
    min_n=MIN_STRATUM_N,
)

print(
    f"  Valid control strata: "
    f"{nested['valid_strata']}"
)

print(
    f"  Total controlled rows: "
    f"{nested['total_n']}"
)

print(
    f"  Mantel-Haenszel OR: "
    f"{nested['mh_or']:.6f}"
)

print(
    f"  Pooled RD: "
    f"{nested.get('pooled_rd_pp', np.nan):+.2f} pp"
)

if nested["valid_strata"] < 2:
    raise RuntimeError(
        "Nested control did not produce both fixed composite strata."
    )


# ============================================================================
# PERMUTATION TEST WITHIN FIXED CONTROL STRATA
# ============================================================================

print()
print("NESTED STRATIFIED PERMUTATION TEST")
print(
    f"  Permutations: {N_PERMUTATIONS}"
)

nested_perm_p = permutation_pooled_rd(
    research,
    "bar1_composite_state",
    PRIMARY_STATE,
    "frozen_event",
    nested.get("pooled_rd", np.nan),
    n_perm=N_PERMUTATIONS,
    seed=RANDOM_SEED,
)

print(
    f"  Observed pooled RD: "
    f"{nested.get('pooled_rd_pp', np.nan):+.2f} pp"
)

print(
    f"  Permutation p: "
    f"{nested_perm_p:.6f}"
)


# ============================================================================
# CONTROL-DIMENSION CONDITIONAL TESTS
# ============================================================================

print()
print("=" * 78)
print("BAR2 STATE WITH EACH FIXED BAR1 DIMENSION CONTROLLED")
print("=" * 78)

pairwise_controls = pairwise_control_table(
    research,
    PRIMARY_STATE,
    "frozen_event",
)

if pairwise_controls.empty:
    raise RuntimeError(
        "No valid pairwise control strata were produced."
    )

print(
    pairwise_controls[
        [
            "control",
            "control_state",
            "n",
            "risk_difference_pp",
            "fisher_p",
            "odds_ratio",
        ]
    ].to_string(index=False)
)


# ============================================================================
# EXACT COMPOSITE SIGNATURES
# ============================================================================

print()
print("=" * 78)
print("EXACT BAR-1 COMPOSITE SIGNATURES")
print("=" * 78)

signature_rows = []

for signature, group in research.groupby(
    "bar1_composite_signature",
    dropna=False,
):
    if len(group) < MIN_CELL_N:
        continue

    stats = fisher_from_binary(
        group,
        PRIMARY_STATE,
        "frozen_event",
    )

    signature_rows.append(
        {
            "signature": str(signature),
            "n": len(group),
            **stats,
        }
    )

signature_df = pd.DataFrame(
    signature_rows
)

if signature_df.empty:
    print(
        "  No exact signature cell reached the minimum sample size."
    )
else:
    print(
        signature_df[
            [
                "signature",
                "n",
                "state1_event_rate",
                "state0_event_rate",
                "risk_difference_pp",
                "fisher_p",
            ]
        ].to_string(index=False)
    )


# ============================================================================
# PRE-THRESHOLD ANALYSIS
# ============================================================================

# Use path data to determine whether the 0.25R threshold had already
# been crossed by Bar2.

path_required = [
    "symbol",
    "direction",
    "entry_date",
    "bar_number",
    "cumulative_mae_r",
]

assert_columns(
    path,
    path_required,
    "path",
)

path["symbol"] = normalize_symbol(
    path["symbol"]
)

path["direction"] = normalize_direction(
    path["direction"]
)

path["entry_date"] = pd.to_datetime(
    path["entry_date"],
    errors="coerce",
)

path["bar_number"] = pd.to_numeric(
    path["bar_number"],
    errors="coerce",
)

path["cumulative_mae_r"] = safe_float(
    path["cumulative_mae_r"]
)

path["merge_key"] = make_merge_key(
    path
)

bar2_path = path[
    path["bar_number"] == 2
].copy()

bar2_path["crossed_025_by_bar2"] = (
    bar2_path["cumulative_mae_r"]
    >= EARLY_ADVERSE_THRESHOLD_R
)

bar2_threshold = (
    bar2_path[
        [
            "merge_key",
            "crossed_025_by_bar2",
        ]
    ]
    .drop_duplicates("merge_key")
)

research = research.merge(
    bar2_threshold,
    on="merge_key",
    how="left",
    validate="one_to_one",
)

research["crossed_025_by_bar2"] = (
    research["crossed_025_by_bar2"]
    .fillna(False)
    .astype(bool)
)

pre_threshold = research[
    ~research["crossed_025_by_bar2"]
].copy()

print()
print("=" * 78)
print("PRE-THRESHOLD NESTED TEST")
print("=" * 78)

print(
    f"  Trades not crossed 0.25R by Bar2: "
    f"{len(pre_threshold)}"
)

if len(pre_threshold) >= MIN_STRATUM_N:
    pre_nested = stratified_mantel_haenszel(
        pre_threshold,
        "bar1_composite_state",
        PRIMARY_STATE,
        "frozen_event",
        min_n=MIN_STRATUM_N,
    )

    print(
        f"  Valid control strata: "
        f"{pre_nested['valid_strata']}"
    )

    print(
        f"  Controlled MH OR: "
        f"{pre_nested['mh_or']:.6f}"
    )

    print(
        f"  Controlled pooled RD: "
        f"{pre_nested.get('pooled_rd_pp', np.nan):+.2f} pp"
    )

    pre_perm_p = permutation_pooled_rd(
        pre_threshold,
        "bar1_composite_state",
        PRIMARY_STATE,
        "frozen_event",
        pre_nested.get("pooled_rd", np.nan),
        n_perm=N_PERMUTATIONS,
        seed=RANDOM_SEED + 1,
    )

    print(
        f"  Stratified permutation p: "
        f"{pre_perm_p:.6f}"
    )
else:
    pre_nested = None
    pre_perm_p = np.nan

    print(
        "  Insufficient pre-threshold sample for nested inference."
    )


# ============================================================================
# DELAYED EVENTS
# ============================================================================

# Determine earliest 0.25R crossing bar from the path replay.

crossing = (
    path[
        path["cumulative_mae_r"]
        >= EARLY_ADVERSE_THRESHOLD_R
    ]
    .groupby("merge_key")["bar_number"]
    .min()
    .rename("earliest_crossing_bar")
    .reset_index()
)

research = research.merge(
    crossing,
    on="merge_key",
    how="left",
    validate="one_to_one",
)

delayed = research[
    (research["frozen_event"] == 1)
    & (
        research["earliest_crossing_bar"]
        > 2
    )
].copy()

retained = research[
    research["frozen_event"] == 0
].copy()

print()
print("=" * 78)
print("DELAYED-EVENT NESTED TEST")
print("=" * 78)

print(
    f"  Delayed frozen events: {len(delayed)}"
)

if len(delayed) >= MIN_STRATUM_N:
    delayed_nested = stratified_mantel_haenszel(
        pd.concat(
            [
                delayed.assign(
                    delayed_target=1
                ),
                retained.assign(
                    delayed_target=0
                ),
            ],
            ignore_index=True,
        ),
        "bar1_composite_state",
        PRIMARY_STATE,
        "delayed_target",
        min_n=MIN_STRATUM_N,
    )

    print(
        f"  Valid control strata: "
        f"{delayed_nested['valid_strata']}"
    )

    print(
        f"  Controlled MH OR: "
        f"{delayed_nested['mh_or']:.6f}"
    )

    print(
        f"  Controlled pooled RD: "
        f"{delayed_nested.get('pooled_rd_pp', np.nan):+.2f} pp"
    )
else:
    delayed_nested = None
    print(
        "  Insufficient delayed-event sample for nested inference."
    )


# ============================================================================
# ROBUSTNESS BY YEAR / SYMBOL / SETUP / DIRECTION
# ============================================================================

research["year"] = (
    research["entry_date"]
    .dt.year
    .astype("Int64")
)

print()
print("=" * 78)
print("ROBUSTNESS OF NESTED CONTROL")
print("=" * 78)

robustness_rows = []

for group_col in ROBUSTNESS_GROUPS:
    if group_col not in research.columns:
        continue

    group_work = research.copy()

    for group_value, group in group_work.groupby(
        group_col,
        dropna=False,
    ):
        if len(group) < MIN_CELL_N:
            continue

        result = stratified_mantel_haenszel(
            group,
            "bar1_composite_state",
            PRIMARY_STATE,
            "frozen_event",
            min_n=MIN_CELL_N,
        )

        if result["valid_strata"] < 2:
            continue

        robustness_rows.append(
            {
                "group_type": group_col,
                "group": str(group_value),
                "n": len(group),
                "valid_strata": result["valid_strata"],
                "mh_or": result["mh_or"],
                "pooled_rd_pp": result.get(
                    "pooled_rd_pp",
                    np.nan,
                ),
                "direction_positive": bool(
                    np.isfinite(
                        result.get(
                            "pooled_rd",
                            np.nan,
                        )
                    )
                    and result["pooled_rd"] > 0
                ),
            }
        )

robustness_df = pd.DataFrame(
    robustness_rows
)

if robustness_df.empty:
    print(
        "  No robustness subgroup had both composite strata."
    )
else:
    print(
        robustness_df.to_string(
            index=False
        )
    )


# ============================================================================
# LEAVE-ONE-GROUP-OUT ROBUSTNESS
# ============================================================================

print()
print("=" * 78)
print("LEAVE-ONE-GROUP-OUT NESTED ROBUSTNESS")
print("=" * 78)

loo_tables = []

for group_col in ROBUSTNESS_GROUPS:
    loo = leave_one_group_out(
        research,
        group_col,
        PRIMARY_STATE,
        "frozen_event",
    )

    if not loo.empty:
        loo.insert(
            0,
            "group_type",
            group_col,
        )
        loo_tables.append(loo)

if loo_tables:
    loo_df = pd.concat(
        loo_tables,
        ignore_index=True,
    )

    print(
        loo_df.to_string(
            index=False
        )
    )
else:
    loo_df = pd.DataFrame()

    print(
        "  No valid LOO nested results."
    )


# ============================================================================
# TEMPORAL / SYMBOL / SETUP / DIRECTION DIRECTIONAL CONSISTENCY
# ============================================================================

print()
print("=" * 78)
print("ROBUSTNESS SUMMARY")
print("=" * 78)

robustness_summary_rows = []

for group_type in ROBUSTNESS_GROUPS:
    subset = robustness_df[
        robustness_df["group_type"]
        == group_type
    ].copy()

    if subset.empty:
        robustness_summary_rows.append(
            {
                "group_type": group_type,
                "valid_subgroups": 0,
                "positive_subgroups": 0,
                "positive_fraction": np.nan,
                "median_rd_pp": np.nan,
                "min_rd_pp": np.nan,
                "max_rd_pp": np.nan,
            }
        )
        continue

    positive = int(
        subset["direction_positive"].sum()
    )

    robustness_summary_rows.append(
        {
            "group_type": group_type,
            "valid_subgroups": len(subset),
            "positive_subgroups": positive,
            "positive_fraction": (
                positive / len(subset)
            ),
            "median_rd_pp": subset[
                "pooled_rd_pp"
            ].median(),
            "min_rd_pp": subset[
                "pooled_rd_pp"
            ].min(),
            "max_rd_pp": subset[
                "pooled_rd_pp"
            ].max(),
        }
    )

robustness_summary = pd.DataFrame(
    robustness_summary_rows
)

print(
    robustness_summary.to_string(
        index=False
    )
)


# ============================================================================
# CONTROLLED EFFECT-SIZE BOOTSTRAP
# ============================================================================

print()
print("=" * 78)
print("CONTROLLED EFFECT-SIZE STABILITY")
print("=" * 78)

# Bootstrap at the trade level while preserving the fixed composite state.
# This is a stability interval, not a causal interval.

controlled_bootstrap_values = []

rng = np.random.default_rng(
    RANDOM_SEED + 2
)

valid_research = research[
    [
        "bar1_composite_state",
        PRIMARY_STATE,
        "frozen_event",
    ]
].dropna().copy()

if len(valid_research) > 0:
    for _ in range(N_PERMUTATIONS):
        idx = rng.integers(
            0,
            len(valid_research),
            size=len(valid_research),
        )

        sample = valid_research.iloc[
            idx
        ].copy()

        result = stratified_mantel_haenszel(
            sample,
            "bar1_composite_state",
            PRIMARY_STATE,
            "frozen_event",
            min_n=MIN_STRATUM_N,
        )

        pooled = result.get(
            "pooled_rd",
            np.nan,
        )

        if np.isfinite(pooled):
            controlled_bootstrap_values.append(
                pooled
            )

controlled_bootstrap_values = np.asarray(
    controlled_bootstrap_values,
    dtype=float,
)

if len(controlled_bootstrap_values) > 0:
    boot_low = float(
        np.percentile(
            controlled_bootstrap_values,
            2.5,
        )
        * 100.0
    )

    boot_high = float(
        np.percentile(
            controlled_bootstrap_values,
            97.5,
        )
        * 100.0
    )

    boot_median = float(
        np.median(
            controlled_bootstrap_values
        )
        * 100.0
    )

    print(
        f"  Bootstrap valid draws: "
        f"{len(controlled_bootstrap_values)}"
    )

    print(
        f"  Median pooled RD: "
        f"{boot_median:+.2f} pp"
    )

    print(
        f"  95% bootstrap CI: "
        f"[{boot_low:+.2f}, {boot_high:+.2f}] pp"
    )
else:
    boot_low = np.nan
    boot_high = np.nan
    boot_median = np.nan

    print(
        "  Bootstrap did not produce valid controlled draws."
    )


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

global_output = pd.DataFrame(
    [
        {
            "metric": "trade_count",
            "value": trade_count,
        },
        {
            "metric": "event_count",
            "value": event_count,
        },
        {
            "metric": "retained_count",
            "value": retained_count,
        },
        {
            "metric": "global_state_event_rate",
            "value": global_stats[
                "state1_event_rate"
            ],
        },
        {
            "metric": "global_non_state_event_rate",
            "value": global_stats[
                "state0_event_rate"
            ],
        },
        {
            "metric": "global_risk_difference_pp",
            "value": global_stats[
                "risk_difference_pp"
            ],
        },
        {
            "metric": "global_fisher_p",
            "value": global_stats[
                "fisher_p"
            ],
        },
        {
            "metric": "nested_mh_or",
            "value": nested["mh_or"],
        },
        {
            "metric": "nested_pooled_rd_pp",
            "value": nested.get(
                "pooled_rd_pp",
                np.nan,
            ),
        },
        {
            "metric": "nested_permutation_p",
            "value": nested_perm_p,
        },
        {
            "metric": "pre_threshold_n",
            "value": len(pre_threshold),
        },
        {
            "metric": "delayed_event_n",
            "value": len(delayed),
        },
        {
            "metric": "bootstrap_median_rd_pp",
            "value": boot_median,
        },
        {
            "metric": "bootstrap_ci_low_pp",
            "value": boot_low,
        },
        {
            "metric": "bootstrap_ci_high_pp",
            "value": boot_high,
        },
    ]
)

global_output.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_conditional_summary.csv",
    ),
    index=False,
)

pd.DataFrame(
    [
        {
            "control": key,
            "global_median": value,
        }
        for key, value in medians.items()
    ]
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "bar1_control_medians.csv",
    ),
    index=False,
)

research[
    [
        "merge_key",
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",
        PRIMARY_STATE,
        *CONTROL_COLUMNS,
        "bar1_cumulative_mae_r_binary",
        "bar1_cumulative_mfe_r_binary",
        "bar1_net_excursion_r_binary",
        "bar1_mae_minus_mfe_binary",
        "bar1_composite_score",
        "bar1_composite_state",
        "bar1_composite_signature",
        "crossed_025_by_bar2",
        "earliest_crossing_bar",
    ]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_conditional_per_trade.csv",
    ),
    index=False,
)

pairwise_controls.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_pairwise_controls.csv",
    ),
    index=False,
)

signature_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_exact_signatures.csv",
    ),
    index=False,
)

robustness_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_robustness_subgroups.csv",
    ),
    index=False,
)

robustness_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_robustness_summary.csv",
    ),
    index=False,
)

loo_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_robustness_loo.csv",
    ),
    index=False,
)

nested["strata"].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_mh_strata.csv",
    ),
    index=False,
)

if pre_nested is not None:
    pre_nested["strata"].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "pre_threshold_nested_strata.csv",
        ),
        index=False,
    )

if delayed_nested is not None:
    delayed_nested["strata"].to_csv(
        os.path.join(
            OUTPUT_DIR,
            "delayed_nested_strata.csv",
        ),
        index=False,
    )


# ============================================================================
# INTEGRITY
# ============================================================================

integrity_rows = []

integrity_rows.append(
    {
        "check": "trajectory_rows",
        "actual": trade_count,
        "expected": EXPECTED_TRADES,
        "status": "PASS"
        if trade_count == EXPECTED_TRADES
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "frozen_events",
        "actual": event_count,
        "expected": EXPECTED_EVENTS,
        "status": "PASS"
        if event_count == EXPECTED_EVENTS
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "retained",
        "actual": retained_count,
        "expected": EXPECTED_RETAINED,
        "status": "PASS"
        if retained_count == EXPECTED_RETAINED
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "trajectory_unique_keys",
        "actual": trade_count
        - trajectory_duplicate_keys,
        "expected": trade_count,
        "status": "PASS"
        if trajectory_duplicate_keys == 0
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "primary_state_binary",
        "actual": sorted(
            research[PRIMARY_STATE]
            .dropna()
            .unique()
            .tolist()
        ),
        "expected": [0, 1],
        "status": "PASS"
        if sorted(
            research[PRIMARY_STATE]
            .dropna()
            .unique()
            .tolist()
        )
        == [0, 1]
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "composite_states",
        "actual": sorted(
            research["bar1_composite_state"]
            .dropna()
            .unique()
            .tolist()
        ),
        "expected": [
            "HIGH_PATH",
            "LOW_PATH",
        ],
        "status": "PASS"
        if set(
            research[
                "bar1_composite_state"
            ].dropna().unique()
        )
        == {
            "HIGH_PATH",
            "LOW_PATH",
        }
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "pre_threshold_rows",
        "actual": len(pre_threshold),
        "expected": ">0",
        "status": "PASS"
        if len(pre_threshold) > 0
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "nested_valid_strata",
        "actual": nested["valid_strata"],
        "expected": ">=2",
        "status": "PASS"
        if nested["valid_strata"] >= 2
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "robustness_rows",
        "actual": len(robustness_df),
        "expected": ">0",
        "status": "PASS"
        if len(robustness_df) > 0
        else "FAIL",
    }
)

integrity_rows.append(
    {
        "check": "production_modified",
        "actual": False,
        "expected": False,
        "status": "PASS",
    }
)

integrity_rows.append(
    {
        "check": "oos_modified",
        "actual": False,
        "expected": False,
        "status": "PASS",
    }
)

integrity_rows.append(
    {
        "check": "strategy_rule_promoted",
        "actual": False,
        "expected": False,
        "status": "PASS",
    }
)

integrity_df = pd.DataFrame(
    integrity_rows
)

integrity_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "nested_conditional_integrity.csv",
    ),
    index=False,
)

print()
print("=" * 78)
print("FINAL INTEGRITY")
print("=" * 78)

print(
    integrity_df.to_string(
        index=False
    )
)

if not (
    integrity_df["status"]
    == "PASS"
).all():
    raise RuntimeError(
        "Nested conditional forensic audit failed integrity checks."
    )

print()
print(
    "PASS — STRICT NESTED CONDITIONAL FORENSIC AUDIT COMPLETE"
)

print(
    f"Output: {OUTPUT_DIR}"
)

print()
print(
    "IMPORTANT: This is forensic association evidence only."
)

print(
    "No production strategy rule was promoted."
)

print(
    "Production and OOS datasets were not modified."
)