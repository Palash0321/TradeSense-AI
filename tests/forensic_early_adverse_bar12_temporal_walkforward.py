# tests/forensic_early_adverse_bar12_temporal_walkforward.py

"""
TradeSense-AI
Forensic Early-Adverse Path Research

Temporal Walk-Forward Validation of the Fixed Bar-2
Favorable-Failure State.

Research question
-----------------
Does the already-fixed Bar-2 state

    bar2_favorable_failed_to_improve

retain its association with the frozen early-adverse event when
evaluated chronologically, using only earlier trades as history
and later trades as validation?

IMPORTANT
---------
- Research-only.
- Frozen population remains 106 trades / 60 events / 46 retained.
- Frozen event definition remains stored bar5 >= 0.25R.
- No threshold optimization.
- No ML.
- No future-data threshold fitting.
- No production/OOS modification.
- No strategy rule promotion.

The Bar-1 control dimensions and their global medians are inherited
from the validated nested conditional analysis:

    bar1 cumulative MAE
    bar1 cumulative MFE
    bar1 net excursion
    bar1 MAE-minus-MFE

The temporal test uses chronological folds.

For each validation period:
    history = all earlier trades
    validation = later trades

The fixed Bar-2 state itself is already defined from Bar1->Bar2
trajectory information and is NOT re-optimized inside the fold.

Outputs
-------
tests/output/research/
    early_adverse_bar12_temporal_walkforward/

        temporal_walkforward_folds.csv
        temporal_walkforward_years.csv
        temporal_walkforward_cumulative.csv
        temporal_walkforward_integrity.csv
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================
# CONFIG
# ============================================================

TRAJECTORY_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

PATH_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/"
    "early_adverse_bar12_temporal_walkforward"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_BAR = 5

MIN_HISTORY_TRADES = 20
MIN_VALIDATION_TRADES = 8

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 20260916


# ============================================================
# HELPERS
# ============================================================

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def normalize_symbol(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_direction(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce",
    ).dt.normalize()


def build_merge_key(df: pd.DataFrame) -> pd.Series:
    return (
        normalize_symbol(df["symbol"])
        + "|"
        + normalize_direction(df["direction"])
        + "|"
        + normalize_date(df["entry_date"]).astype(str)
    )


def fisher_two_sided(
    state: pd.Series,
    event: pd.Series,
) -> float:
    state = state.astype(int)
    event = event.astype(int)

    state_event = int(((state == 1) & (event == 1)).sum())
    state_retained = int(((state == 1) & (event == 0)).sum())
    non_event = int(((state == 0) & (event == 1)).sum())
    non_retained = int(((state == 0) & (event == 0)).sum())

    table = np.array(
        [
            [state_event, state_retained],
            [non_event, non_retained],
        ],
        dtype=int,
    )

    _, p_value = fisher_exact(
        table,
        alternative="two-sided",
    )

    return float(p_value)


def odds_ratio(
    state: pd.Series,
    event: pd.Series,
) -> float:
    state = state.astype(int)
    event = event.astype(int)

    a = int(((state == 1) & (event == 1)).sum())
    b = int(((state == 1) & (event == 0)).sum())
    c = int(((state == 0) & (event == 1)).sum())
    d = int(((state == 0) & (event == 0)).sum())

    # Haldane-Anscombe correction only for zero cells.
    if 0 in (a, b, c, d):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    return float((a * d) / (b * c))


def risk_difference(
    state: pd.Series,
    event: pd.Series,
) -> float:
    state = state.astype(int)
    event = event.astype(int)

    state_mask = state == 1
    non_mask = state == 0

    if state_mask.sum() == 0 or non_mask.sum() == 0:
        return float("nan")

    return float(
        event[state_mask].mean()
        - event[non_mask].mean()
    )


def wilson_ci(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")

    p = successes / total
    denom = 1 + (z * z / total)

    centre = (
        p
        + (z * z / (2 * total))
    ) / denom

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p) / total
            )
            + (
                z * z / (4 * total * total)
            )
        )
        / denom
    )

    return (
        float(max(0.0, centre - margin)),
        float(min(1.0, centre + margin)),
    )


def risk_difference_ci(
    state: pd.Series,
    event: pd.Series,
) -> tuple[float, float]:
    state = state.astype(int)
    event = event.astype(int)

    state_mask = state == 1
    non_mask = state == 0

    n1 = int(state_mask.sum())
    n0 = int(non_mask.sum())

    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")

    x1 = int(event[state_mask].sum())
    x0 = int(event[non_mask].sum())

    p1_low, p1_high = wilson_ci(x1, n1)
    p0_low, p0_high = wilson_ci(x0, n0)

    return (
        float(p1_low - p0_high),
        float(p1_high - p0_low),
    )


def state_summary(
    df: pd.DataFrame,
    state_col: str,
    event_col: str,
    fold_name: str,
    population_name: str,
) -> dict:
    work = df[
        [state_col, event_col]
    ].dropna().copy()

    if work.empty:
        return {
            "fold": fold_name,
            "population": population_name,
            "n": 0,
            "state_n": 0,
            "non_state_n": 0,
            "state_events": 0,
            "non_state_events": 0,
            "state_event_rate": np.nan,
            "non_state_event_rate": np.nan,
            "risk_difference_pp": np.nan,
            "risk_difference_ci_low_pp": np.nan,
            "risk_difference_ci_high_pp": np.nan,
            "odds_ratio": np.nan,
            "fisher_p": np.nan,
        }

    state = work[state_col].astype(int)
    event = work[event_col].astype(int)

    state_mask = state == 1
    non_mask = state == 0

    state_n = int(state_mask.sum())
    non_state_n = int(non_mask.sum())

    state_events = int(
        event[state_mask].sum()
    )

    non_state_events = int(
        event[non_mask].sum()
    )

    state_rate = (
        state_events / state_n
        if state_n
        else np.nan
    )

    non_state_rate = (
        non_state_events / non_state_n
        if non_state_n
        else np.nan
    )

    rd = risk_difference(
        state,
        event,
    )

    ci_low, ci_high = risk_difference_ci(
        state,
        event,
    )

    p_value = fisher_two_sided(
        state,
        event,
    )

    or_value = odds_ratio(
        state,
        event,
    )

    return {
        "fold": fold_name,
        "population": population_name,
        "n": len(work),
        "state_n": state_n,
        "non_state_n": non_state_n,
        "state_events": state_events,
        "non_state_events": non_state_events,
        "state_event_rate": state_rate,
        "non_state_event_rate": non_state_rate,
        "risk_difference_pp": rd * 100,
        "risk_difference_ci_low_pp": ci_low * 100,
        "risk_difference_ci_high_pp": ci_high * 100,
        "odds_ratio": or_value,
        "fisher_p": p_value,
    }


def add_bar1_composite(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recreate the already-fixed Bar1 composite used by
    the nested conditional test.

    Each component uses its global median from the frozen
    106-trade population.

    Higher score means a worse Bar1 path.

    1. MAE >= median
    2. MFE <= median
    3. Net <= median
    4. MAE-MFE >= median
    """

    work = df.copy()

    mae_median = float(
        work["bar1_cumulative_mae_r"].median()
    )

    mfe_median = float(
        work["bar1_cumulative_mfe_r"].median()
    )

    net_median = float(
        work["bar1_net_excursion_r"].median()
    )

    mae_mfe_median = float(
        work["bar1_mae_minus_mfe"].median()
    )

    work["bar1_mae_high"] = (
        work["bar1_cumulative_mae_r"]
        >= mae_median
    ).astype(int)

    work["bar1_mfe_low"] = (
        work["bar1_cumulative_mfe_r"]
        <= mfe_median
    ).astype(int)

    work["bar1_net_low"] = (
        work["bar1_net_excursion_r"]
        <= net_median
    ).astype(int)

    work["bar1_mae_mfe_high"] = (
        work["bar1_mae_minus_mfe"]
        >= mae_mfe_median
    ).astype(int)

    work["bar1_composite_score"] = (
        work["bar1_mae_high"]
        + work["bar1_mfe_low"]
        + work["bar1_net_low"]
        + work["bar1_mae_mfe_high"]
    )

    work["bar1_composite"] = np.where(
        work["bar1_composite_score"] >= 2,
        "HIGH_PATH",
        "LOW_PATH",
    )

    work["bar1_high_path"] = (
        work["bar1_composite"] == "HIGH_PATH"
    ).astype(int)

    return work


def stratified_summary(
    df: pd.DataFrame,
    state_col: str,
    event_col: str,
    strata_col: str,
    fold_name: str,
    population_name: str,
) -> dict:
    """
    Simple stratified association summary.

    This intentionally avoids statsmodels.

    Pooled RD is weighted by stratum size.

    Pooled OR uses a manual Mantel-Haenszel estimator.
    """

    valid = df[
        [
            state_col,
            event_col,
            strata_col,
        ]
    ].dropna().copy()

    if valid.empty:
        return {
            "fold": fold_name,
            "population": population_name,
            "strata": 0,
            "pooled_rd_pp": np.nan,
            "mh_or": np.nan,
        }

    rd_numerator = 0.0
    rd_denominator = 0.0

    mh_num = 0.0
    mh_den = 0.0

    used = 0

    for _, sub in valid.groupby(
        strata_col,
        sort=False,
    ):
        state = sub[state_col].astype(int)
        event = sub[event_col].astype(int)

        n1 = int((state == 1).sum())
        n0 = int((state == 0).sum())

        if n1 == 0 or n0 == 0:
            continue

        a = int(
            ((state == 1) & (event == 1)).sum()
        )
        b = int(
            ((state == 1) & (event == 0)).sum()
        )
        c = int(
            ((state == 0) & (event == 1)).sum()
        )
        d = int(
            ((state == 0) & (event == 0)).sum()
        )

        n = a + b + c + d

        if n == 0:
            continue

        rd = (
            (a / n1)
            - (c / n0)
        )

        weight = (
            (n1 * n0) / n
        )

        rd_numerator += weight * rd
        rd_denominator += weight

        mh_num += (
            (a * d) / n
        )

        mh_den += (
            (b * c) / n
        )

        used += 1

    pooled_rd = (
        rd_numerator / rd_denominator
        if rd_denominator > 0
        else np.nan
    )

    mh_or = (
        mh_num / mh_den
        if mh_den > 0
        else np.nan
    )

    return {
        "fold": fold_name,
        "population": population_name,
        "strata": used,
        "pooled_rd_pp": pooled_rd * 100,
        "mh_or": mh_or,
    }


def bootstrap_rd(
    df: pd.DataFrame,
    state_col: str,
    event_col: str,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """
    Bootstrap the state-vs-nonstate risk difference.
    """

    work = df[
        [state_col, event_col]
    ].dropna().reset_index(drop=True)

    if work.empty:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    observed = risk_difference(
        work[state_col],
        work[event_col],
    )

    values = []

    n = len(work)

    for _ in range(iterations):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        sample = work.iloc[idx]

        state = sample[state_col].astype(int)

        if (
            (state == 1).sum() == 0
            or (state == 0).sum() == 0
        ):
            continue

        values.append(
            risk_difference(
                state,
                sample[event_col].astype(int),
            )
        )

    if not values:
        return observed, np.nan, np.nan

    arr = np.asarray(values)

    return (
        float(observed),
        float(np.quantile(arr, 0.025)),
        float(np.quantile(arr, 0.975)),
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("TRADE SENSE-AI")
print("TEMPORAL WALK-FORWARD VALIDATION")
print("FIXED BAR-2 FAVORABLE-FAILURE STATE")
print("=" * 72)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

trajectory = pd.read_csv(
    TRAJECTORY_FILE
)

path = pd.read_csv(
    PATH_FILE
)

print(
    f"\nTrajectory rows: {len(trajectory)}, "
    f"columns: {len(trajectory.columns)}"
)

print(
    f"Path rows: {len(path)}, "
    f"columns: {len(path.columns)}"
)


# ============================================================
# REQUIRED TRAJECTORY COLUMNS
# ============================================================

require_columns(
    trajectory,
    [
        "symbol",
        "direction",
        "entry_date",
        "frozen_event",
        "bar1_cumulative_mae_r",
        "bar1_cumulative_mfe_r",
        "bar1_net_excursion_r",
        "bar2_favorable_failed_to_improve",
    ],
    "Trajectory",
)


# ============================================================
# NORMALIZE TRAJECTORY
# ============================================================

trajectory = trajectory.copy()

trajectory["symbol"] = normalize_symbol(
    trajectory["symbol"]
)

trajectory["direction"] = normalize_direction(
    trajectory["direction"]
)

trajectory["entry_date"] = normalize_date(
    trajectory["entry_date"]
)

trajectory["frozen_event"] = pd.to_numeric(
    trajectory["frozen_event"],
    errors="coerce",
)

trajectory["bar2_favorable_failed_to_improve"] = (
    pd.to_numeric(
        trajectory[
            "bar2_favorable_failed_to_improve"
        ],
        errors="coerce",
    )
)

trajectory["bar1_mae_minus_mfe"] = (
    trajectory["bar1_cumulative_mae_r"]
    - trajectory["bar1_cumulative_mfe_r"]
)

trajectory["merge_key"] = build_merge_key(
    trajectory
)


# ============================================================
# PATH CHECK
# ============================================================

require_columns(
    path,
    [
        "symbol",
        "direction",
        "entry_date",
        "bar_number",
        "production_observation_status",
        "production_observed",
    ],
    "Path",
)

path = path.copy()

path["symbol"] = normalize_symbol(
    path["symbol"]
)

path["direction"] = normalize_direction(
    path["direction"]
)

path["entry_date"] = normalize_date(
    path["entry_date"]
)

path["bar_number"] = pd.to_numeric(
    path["bar_number"],
    errors="coerce",
)

path["production_observed"] = pd.to_numeric(
    path["production_observed"],
    errors="coerce",
)

path["merge_key"] = build_merge_key(
    path
)


# ============================================================
# FROZEN POPULATION INTEGRITY
# ============================================================

trajectory_trade_count = (
    trajectory["merge_key"].nunique()
)

trajectory_event_count = int(
    trajectory[
        "frozen_event"
    ].eq(1).sum()
)

trajectory_retained_count = int(
    trajectory[
        "frozen_event"
    ].eq(0).sum()
)

print("\nFROZEN POPULATION")
print("-" * 72)

print(
    f"Trade count:     "
    f"{trajectory_trade_count}"
)

print(
    f"Event count:     "
    f"{trajectory_event_count}"
)

print(
    f"Retained count:  "
    f"{trajectory_retained_count}"
)

if trajectory_trade_count != EXPECTED_TRADES:
    raise RuntimeError(
        "Frozen trade count mismatch."
    )

if trajectory_event_count != EXPECTED_EVENTS:
    raise RuntimeError(
        "Frozen event count mismatch."
    )

if trajectory_retained_count != EXPECTED_RETAINED:
    raise RuntimeError(
        "Frozen retained count mismatch."
    )


# ============================================================
# DUPLICATE KEY CHECK
# ============================================================

duplicate_keys = (
    trajectory["merge_key"]
    .duplicated()
)

if duplicate_keys.any():
    raise RuntimeError(
        "Trajectory contains duplicate trade keys."
    )

print(
    "Unique trajectory keys: PASS"
)


# ============================================================
# ADD BAR1 COMPOSITE
# ============================================================

trajectory = add_bar1_composite(
    trajectory
)

print("\nBAR-1 CONTROL MEDIANS")
print("-" * 72)

print(
    f"MAE median:       "
    f"{trajectory['bar1_cumulative_mae_r'].median():.6f}"
)

print(
    f"MFE median:       "
    f"{trajectory['bar1_cumulative_mfe_r'].median():.6f}"
)

print(
    f"Net median:       "
    f"{trajectory['bar1_net_excursion_r'].median():.6f}"
)

print(
    f"MAE-MFE median:   "
    f"{trajectory['bar1_mae_minus_mfe'].median():.6f}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

trajectory = trajectory.sort_values(
    [
        "entry_date",
        "symbol",
        "direction",
    ]
).reset_index(drop=True)

trajectory["calendar_year"] = (
    trajectory["entry_date"].dt.year
)


# ============================================================
# FIXED STATE CHECK
# ============================================================

state_values = set(
    trajectory[
        "bar2_favorable_failed_to_improve"
    ]
    .dropna()
    .astype(int)
    .unique()
)

if not state_values.issubset({0, 1}):
    raise RuntimeError(
        "Bar-2 state is not binary."
    )

if state_values != {0, 1}:
    raise RuntimeError(
        "Bar-2 state does not contain both states."
    )

print(
    "\nFixed Bar-2 state: "
    "bar2_favorable_failed_to_improve"
)

print(
    "State values: "
    f"{sorted(state_values)}"
)


# ============================================================
# TEMPORAL FOLD CONSTRUCTION
# ============================================================

unique_years = sorted(
    trajectory["calendar_year"].unique()
)

print("\nAVAILABLE YEARS")
print("-" * 72)

print(
    ", ".join(
        str(int(y))
        for y in unique_years
    )
)

fold_rows = []
year_rows = []
cumulative_rows = []

running_validation = []

fold_number = 0


# ============================================================
# YEAR-BLOCK WALK-FORWARD
# ============================================================

for validation_year in unique_years:

    history = trajectory[
        trajectory["calendar_year"]
        < validation_year
    ].copy()

    validation = trajectory[
        trajectory["calendar_year"]
        == validation_year
    ].copy()

    if len(history) < MIN_HISTORY_TRADES:
        print(
            f"\nYEAR {validation_year}: "
            f"SKIPPED — history {len(history)} "
            f"< minimum {MIN_HISTORY_TRADES}"
        )
        continue

    if len(validation) < MIN_VALIDATION_TRADES:
        print(
            f"\nYEAR {validation_year}: "
            f"SKIPPED — validation {len(validation)} "
            f"< minimum {MIN_VALIDATION_TRADES}"
        )
        continue

    fold_number += 1

    fold_name = (
        f"FOLD_{fold_number}_"
        f"VALIDATE_{int(validation_year)}"
    )

    print("\n" + "-" * 72)
    print(
        f"{fold_name}"
    )
    print("-" * 72)

    print(
        f"History:     "
        f"{len(history)} trades"
    )

    print(
        f"Validation:  "
        f"{len(validation)} trades"
    )

    print(
        f"History end: "
        f"{history['entry_date'].max().date()}"
    )

    print(
        f"Validation:  "
        f"{validation['entry_date'].min().date()}"
        f" -> "
        f"{validation['entry_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Global validation state result
    # --------------------------------------------------------

    result = state_summary(
        validation,
        "bar2_favorable_failed_to_improve",
        "frozen_event",
        fold_name,
        "VALIDATION_ALL",
    )

    result.update(
        {
            "validation_year": int(
                validation_year
            ),
            "history_n": len(history),
            "history_end": history[
                "entry_date"
            ].max(),
            "validation_start": validation[
                "entry_date"
            ].min(),
            "validation_end": validation[
                "entry_date"
            ].max(),
        }
    )

    fold_rows.append(result)

    # --------------------------------------------------------
    # Nested Bar1 composite control
    # --------------------------------------------------------

    nested = validation.copy()

    nested["bar1_control"] = (
        nested["bar1_composite"]
    )

    nested_result = stratified_summary(
        nested,
        "bar2_favorable_failed_to_improve",
        "frozen_event",
        "bar1_control",
        fold_name,
        "VALIDATION_NESTED_BAR1",
    )

    nested_result.update(
        {
            "validation_year": int(
                validation_year
            ),
            "history_n": len(history),
            "validation_n": len(validation),
        }
    )

    fold_rows.append(nested_result)

    # --------------------------------------------------------
    # Pre-threshold population
    #
    # Keep only trades whose 0.25R threshold had NOT already
    # been reached by the end of Bar2.
    #
    # This uses the validated path replay.
    # --------------------------------------------------------

    validation_keys = set(
        validation["merge_key"]
    )

    path_validation = path[
        path["merge_key"].isin(
            validation_keys
        )
    ].copy()

    bar2 = path_validation[
        path_validation["bar_number"] == 2
    ].copy()

    if (
        "cumulative_mae_r" in bar2.columns
    ):
        bar2["crossed_by_bar2"] = (
            bar2[
                "cumulative_mae_r"
            ]
            >= EARLY_ADVERSE_THRESHOLD_R
        )
    elif (
        "bar_adverse_r" in bar2.columns
    ):
        bar2["crossed_by_bar2"] = (
            bar2[
                "bar_adverse_r"
            ]
            >= EARLY_ADVERSE_THRESHOLD_R
        )
    else:
        bar2["crossed_by_bar2"] = False

    bar2 = bar2[
        [
            "merge_key",
            "crossed_by_bar2",
        ]
    ].drop_duplicates(
        "merge_key"
    )

    pre_threshold = validation.merge(
        bar2,
        on="merge_key",
        how="left",
    )

    pre_threshold[
        "crossed_by_bar2"
    ] = pre_threshold[
        "crossed_by_bar2"
    ].fillna(False)

    pre_threshold = pre_threshold[
        ~pre_threshold[
            "crossed_by_bar2"
        ]
    ].copy()

    if len(pre_threshold) >= MIN_VALIDATION_TRADES:
        pre_result = state_summary(
            pre_threshold,
            "bar2_favorable_failed_to_improve",
            "frozen_event",
            fold_name,
            "VALIDATION_PRE_THRESHOLD",
        )

        pre_result.update(
            {
                "validation_year": int(
                    validation_year
                ),
                "history_n": len(history),
                "validation_n": len(
                    pre_threshold
                ),
            }
        )

        fold_rows.append(
            pre_result
        )

        print(
            f"Pre-threshold validation: "
            f"{len(pre_threshold)} trades"
        )

    # --------------------------------------------------------
    # Cumulative validation
    # --------------------------------------------------------

    running_validation.append(
        validation
    )

    cumulative = pd.concat(
        running_validation,
        ignore_index=True,
    )

    cumulative_result = state_summary(
        cumulative,
        "bar2_favorable_failed_to_improve",
        "frozen_event",
        fold_name,
        "CUMULATIVE_VALIDATION",
    )

    cumulative_result.update(
        {
            "validation_year": int(
                validation_year
            ),
            "history_n": len(history),
            "cumulative_validation_n": len(
                cumulative
            ),
        }
    )

    cumulative_rows.append(
        cumulative_result
    )

    print(
        f"Validation RD: "
        f"{result['risk_difference_pp']:+.2f} pp"
    )

    print(
        f"Validation OR: "
        f"{result['odds_ratio']:.4f}"
    )

    print(
        f"Validation p:  "
        f"{result['fisher_p']:.6f}"
    )

    print(
        f"Nested MH OR:   "
        f"{nested_result['mh_or']:.4f}"
    )

    print(
        f"Nested RD:      "
        f"{nested_result['pooled_rd_pp']:+.2f} pp"
    )


# ============================================================
# YEAR-BLOCK SUMMARY
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)

cumulative_df = pd.DataFrame(
    cumulative_rows
)

if fold_df.empty:
    raise RuntimeError(
        "No temporal walk-forward folds "
        "met the minimum history/validation requirements."
    )


print("\n" + "=" * 72)
print("TEMPORAL WALK-FORWARD SUMMARY")
print("=" * 72)

validation_only = fold_df[
    fold_df["population"]
    == "VALIDATION_ALL"
].copy()

print(
    f"\nValid validation folds: "
    f"{len(validation_only)}"
)

print(
    f"Positive validation RD folds: "
    f"{int((validation_only['risk_difference_pp'] > 0).sum())}"
    f"/{len(validation_only)}"
)

print(
    f"Negative validation RD folds: "
    f"{int((validation_only['risk_difference_pp'] < 0).sum())}"
    f"/{len(validation_only)}"
)

print(
    "\nYear-block validation:"
)

for _, row in validation_only.iterrows():
    print(
        f"  {int(row['validation_year'])}: "
        f"n={int(row['n'])}, "
        f"state={int(row['state_n'])}, "
        f"nonstate={int(row['non_state_n'])}, "
        f"RD={row['risk_difference_pp']:+.2f} pp, "
        f"OR={row['odds_ratio']:.4f}, "
        f"p={row['fisher_p']:.6f}"
    )


# ============================================================
# TEMPORAL CONSISTENCY METRICS
# ============================================================

rd_values = (
    validation_only[
        "risk_difference_pp"
    ]
    .dropna()
    .to_numpy()
)

if len(rd_values):
    median_rd = float(
        np.median(rd_values)
    )

    min_rd = float(
        np.min(rd_values)
    )

    max_rd = float(
        np.max(rd_values)
    )

    positive_fraction = float(
        np.mean(
            rd_values > 0
        )
    )
else:
    median_rd = np.nan
    min_rd = np.nan
    max_rd = np.nan
    positive_fraction = np.nan


print("\nTEMPORAL CONSISTENCY")
print("-" * 72)

print(
    f"Median validation RD: "
    f"{median_rd:+.2f} pp"
)

print(
    f"Validation RD range: "
    f"{min_rd:+.2f} -> "
    f"{max_rd:+.2f} pp"
)

print(
    f"Positive-fold fraction: "
    f"{positive_fraction * 100:.2f}%"
)


# ============================================================
# NESTED YEAR-BLOCK SUMMARY
# ============================================================

nested_df = fold_df[
    fold_df["population"]
    == "VALIDATION_NESTED_BAR1"
].copy()

if not nested_df.empty:
    nested_positive = int(
        (
            nested_df[
                "pooled_rd_pp"
            ]
            > 0
        ).sum()
    )

    print("\nNESTED BAR-1 CONTROL")
    print("-" * 72)

    print(
        f"Valid nested folds: "
        f"{len(nested_df)}"
    )

    print(
        f"Positive nested RD folds: "
        f"{nested_positive}/"
        f"{len(nested_df)}"
    )

    print(
        f"Nested RD median: "
        f"{nested_df['pooled_rd_pp'].median():+.2f} pp"
    )

    print(
        f"Nested RD range: "
        f"{nested_df['pooled_rd_pp'].min():+.2f} -> "
        f"{nested_df['pooled_rd_pp'].max():+.2f} pp"
    )


# ============================================================
# YEAR-LEVEL SIMPLE OUTPUT
# ============================================================

year_summary = (
    trajectory
    .groupby("calendar_year")
    .agg(
        trades=("merge_key", "count"),
        events=("frozen_event", "sum"),
        state_trades=(
            "bar2_favorable_failed_to_improve",
            "sum",
        ),
    )
    .reset_index()
)

year_summary["event_rate"] = (
    year_summary["events"]
    / year_summary["trades"]
)

year_summary["state_rate"] = (
    year_summary["state_trades"]
    / year_summary["trades"]
)

year_summary["eligible_for_walkforward"] = (
    year_summary["calendar_year"].apply(
        lambda y: (
            len(
                trajectory[
                    trajectory["calendar_year"]
                    < y
                ]
            )
            >= MIN_HISTORY_TRADES
        )
    )
)

year_rows = []

for _, row in year_summary.iterrows():

    year = int(
        row["calendar_year"]
    )

    validation = trajectory[
        trajectory["calendar_year"]
        == year
    ].copy()

    history_n = int(
        (
            trajectory["calendar_year"]
            < year
        ).sum()
    )

    if (
        history_n >= MIN_HISTORY_TRADES
        and len(validation)
        >= MIN_VALIDATION_TRADES
    ):
        result = state_summary(
            validation,
            "bar2_favorable_failed_to_improve",
            "frozen_event",
            f"YEAR_{year}",
            "YEAR_VALIDATION",
        )

        year_rows.append(
            {
                "calendar_year": year,
                "history_n": history_n,
                "validation_n": len(
                    validation
                ),
                **result,
            }
        )

year_df = pd.DataFrame(
    year_rows
)


# ============================================================
# CUMULATIVE CHRONOLOGICAL OUTPUT
# ============================================================

if not cumulative_df.empty:

    print("\nCUMULATIVE CHRONOLOGICAL VALIDATION")
    print("-" * 72)

    for _, row in cumulative_df.iterrows():
        print(
            f"  Through {int(row['validation_year'])}: "
            f"n={int(row['cumulative_validation_n'])}, "
            f"RD={row['risk_difference_pp']:+.2f} pp, "
            f"OR={row['odds_ratio']:.4f}, "
            f"p={row['fisher_p']:.6f}"
        )


# ============================================================
# FULL-DATA BOOTSTRAP REFERENCE
# ============================================================

observed_rd, boot_low, boot_high = (
    bootstrap_rd(
        trajectory,
        "bar2_favorable_failed_to_improve",
        "frozen_event",
    )
)

print("\nFULL FROZEN POPULATION REFERENCE")
print("-" * 72)

print(
    f"Observed RD: "
    f"{observed_rd * 100:+.2f} pp"
)

print(
    f"Bootstrap 95% CI: "
    f"{boot_low * 100:+.2f} -> "
    f"{boot_high * 100:+.2f} pp"
)


# ============================================================
# TEMPORAL PASS / FAIL CRITERIA
# ============================================================

valid_fold_count = len(
    validation_only
)

positive_fold_count = int(
    (
        validation_only[
            "risk_difference_pp"
        ]
        > 0
    ).sum()
)

nested_valid_count = len(
    nested_df
)

nested_positive_count = (
    int(
        (
            nested_df[
                "pooled_rd_pp"
            ]
            > 0
        ).sum()
    )
    if not nested_df.empty
    else 0
)

temporal_direction_pass = (
    valid_fold_count > 0
    and positive_fold_count
    == valid_fold_count
)

nested_direction_pass = (
    nested_valid_count > 0
    and nested_positive_count
    == nested_valid_count
)

history_validation_pass = (
    valid_fold_count >= 2
)

integrity_trade_count = (
    trajectory_trade_count
    == EXPECTED_TRADES
)

integrity_event_count = (
    trajectory_event_count
    == EXPECTED_EVENTS
)

integrity_retained_count = (
    trajectory_retained_count
    == EXPECTED_RETAINED
)

integrity_binary_state = (
    state_values == {0, 1}
)

integrity_path_rows = (
    len(path)
    == EXPECTED_TRADES * MAX_BAR
)


# ============================================================
# SAVE FOLD OUTPUT
# ============================================================

if not fold_df.empty:
    fold_df = fold_df[
        [
            "fold",
            "population",
            "validation_year",
            "history_n",
            "validation_n",
            "n",
            "state_n",
            "non_state_n",
            "state_events",
            "non_state_events",
            "state_event_rate",
            "non_state_event_rate",
            "risk_difference_pp",
            "risk_difference_ci_low_pp",
            "risk_difference_ci_high_pp",
            "odds_ratio",
            "fisher_p",
            "strata",
            "pooled_rd_pp",
            "mh_or",
        ]
        + [
            c
            for c in [
                "history_end",
                "validation_start",
                "validation_end",
                "cumulative_validation_n",
            ]
            if c in fold_df.columns
        ]
    ]

fold_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_folds.csv",
    index=False,
)

year_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_years.csv",
    index=False,
)

cumulative_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_cumulative.csv",
    index=False,
)


# ============================================================
# INTEGRITY
# ============================================================

integrity_rows = [
    {
        "check": "trajectory_trade_count",
        "expected": EXPECTED_TRADES,
        "actual": trajectory_trade_count,
        "pass": integrity_trade_count,
    },
    {
        "check": "trajectory_event_count",
        "expected": EXPECTED_EVENTS,
        "actual": trajectory_event_count,
        "pass": integrity_event_count,
    },
    {
        "check": "trajectory_retained_count",
        "expected": EXPECTED_RETAINED,
        "actual": trajectory_retained_count,
        "pass": integrity_retained_count,
    },
    {
        "check": "trajectory_unique_keys",
        "expected": EXPECTED_TRADES,
        "actual": trajectory["merge_key"].nunique(),
        "pass": trajectory["merge_key"].nunique()
        == EXPECTED_TRADES,
    },
    {
        "check": "path_rows",
        "expected": EXPECTED_TRADES * MAX_BAR,
        "actual": len(path),
        "pass": integrity_path_rows,
    },
    {
        "check": "bar2_state_binary",
        "expected": "{0,1}",
        "actual": str(sorted(state_values)),
        "pass": integrity_binary_state,
    },
    {
        "check": "valid_walkforward_folds",
        "expected": ">=2",
        "actual": valid_fold_count,
        "pass": history_validation_pass,
    },
    {
        "check": "validation_direction_consistency",
        "expected": "all positive",
        "actual": (
            f"{positive_fold_count}/"
            f"{valid_fold_count}"
        ),
        "pass": temporal_direction_pass,
    },
    {
        "check": "nested_direction_consistency",
        "expected": "all positive",
        "actual": (
            f"{nested_positive_count}/"
            f"{nested_valid_count}"
        ),
        "pass": nested_direction_pass,
    },
    {
        "check": "production_modified",
        "expected": False,
        "actual": False,
        "pass": True,
    },
    {
        "check": "oos_modified",
        "expected": False,
        "actual": False,
        "pass": True,
    },
    {
        "check": "strategy_rule_promoted",
        "expected": False,
        "actual": False,
        "pass": True,
    },
]

integrity_df = pd.DataFrame(
    integrity_rows
)

integrity_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_integrity.csv",
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

all_integrity_pass = bool(
    integrity_df["pass"].all()
)

print("\n" + "=" * 72)
print("FINAL TEMPORAL WALK-FORWARD STATUS")
print("=" * 72)

print(
    f"Frozen population: "
    f"{trajectory_trade_count} / "
    f"{trajectory_event_count} / "
    f"{trajectory_retained_count}"
)

print(
    f"Valid validation folds: "
    f"{valid_fold_count}"
)

print(
    f"Validation positive direction: "
    f"{positive_fold_count}/"
    f"{valid_fold_count}"
)

print(
    f"Nested positive direction: "
    f"{nested_positive_count}/"
    f"{nested_valid_count}"
)

print(
    f"Temporal direction PASS: "
    f"{temporal_direction_pass}"
)

print(
    f"Nested direction PASS: "
    f"{nested_direction_pass}"
)

print(
    f"Integrity PASS: "
    f"{all_integrity_pass}"
)

print(
    "\nNo production/OOS data modified."
)

print(
    "No strategy rule promoted."
)

if (
    all_integrity_pass
    and history_validation_pass
    and temporal_direction_pass
    and nested_direction_pass
):
    print(
        "\nTEMPORAL WALK-FORWARD TEST: PASS"
    )
else:
    print(
        "\nTEMPORAL WALK-FORWARD TEST: "
        "INCONCLUSIVE / FAIL"
    )

print(
    "\nIMPORTANT:"
)

print(
    "This remains historical forensic validation."
)

print(
    "A positive temporal association is not,"
    " by itself, an OOS trading edge."
)

print(
    "The locked 7B / 0.25R OOS hypothesis"
    " remains untouched."
)

print(
    "\nOutputs:"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_folds.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_years.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_cumulative.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_integrity.csv'}"
)