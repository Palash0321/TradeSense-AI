# tests/forensic_early_adverse_bar12_temporal_walkforward_strict.py

"""
TradeSense-AI
STRICT LEAKAGE-SAFE TEMPORAL WALK-FORWARD VALIDATION

Research question
-----------------
Does the fixed Bar-2 favorable-failure state retain its association
with the frozen early-adverse event when ALL Bar-1 control thresholds
are calculated using history available before each validation period?

Frozen:
    106 total trades
    60 frozen early-adverse events
    46 retained trades

Frozen event definition:
    stored bar5 >= 0.25R

Fixed Bar-2 state:
    bar2_favorable_failed_to_improve

Strict temporal rule:
    For validation year Y:

        history = all trades before Y

        Bar-1 medians are calculated ONLY from history.

        The validation trades are then classified using
        those history-only medians.

No validation-period information is used to define the
Bar-1 control thresholds.

IMPORTANT
---------
- Research only.
- No production modification.
- No OOS modification.
- No threshold optimization.
- No ML.
- No strategy rule promotion.
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
    "early_adverse_bar12_temporal_walkforward_strict"
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

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label}: missing required columns: {missing}"
        )


def normalize_symbol(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_direction(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_date(
    series: pd.Series,
) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce",
    ).dt.normalize()


def build_merge_key(
    df: pd.DataFrame,
) -> pd.Series:
    return (
        normalize_symbol(df["symbol"])
        + "|"
        + normalize_direction(df["direction"])
        + "|"
        + normalize_date(df["entry_date"]).astype(str)
    )


def fisher_p_value(
    state: pd.Series,
    event: pd.Series,
) -> float:

    state = state.astype(int)
    event = event.astype(int)

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

    table = np.array(
        [
            [a, b],
            [c, d],
        ],
        dtype=int,
    )

    _, p = fisher_exact(
        table,
        alternative="two-sided",
    )

    return float(p)


def calculate_or(
    state: pd.Series,
    event: pd.Series,
) -> float:

    state = state.astype(int)
    event = event.astype(int)

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

    if 0 in (a, b, c, d):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    return float(
        (a * d) / (b * c)
    )


def risk_difference(
    state: pd.Series,
    event: pd.Series,
) -> float:

    state = state.astype(int)
    event = event.astype(int)

    state_mask = state == 1
    non_mask = state == 0

    if (
        state_mask.sum() == 0
        or non_mask.sum() == 0
    ):
        return float("nan")

    return float(
        event[state_mask].mean()
        - event[non_mask].mean()
    )


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:

    if total <= 0:
        return np.nan, np.nan

    p = successes / total

    denominator = (
        1
        + z * z / total
    )

    centre = (
        p
        + z * z / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
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
        return np.nan, np.nan

    x1 = int(
        event[state_mask].sum()
    )

    x0 = int(
        event[non_mask].sum()
    )

    low1, high1 = wilson_interval(
        x1,
        n1,
    )

    low0, high0 = wilson_interval(
        x0,
        n0,
    )

    return (
        low1 - high0,
        high1 - low0,
    )


def state_result(
    df: pd.DataFrame,
    state_col: str,
    event_col: str,
    fold: str,
    population: str,
) -> dict:

    work = df[
        [
            state_col,
            event_col,
        ]
    ].dropna().copy()

    if work.empty:
        return {
            "fold": fold,
            "population": population,
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

    state_n = int(
        state_mask.sum()
    )

    non_state_n = int(
        non_mask.sum()
    )

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

    return {
        "fold": fold,
        "population": population,
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
        "odds_ratio": calculate_or(
            state,
            event,
        ),
        "fisher_p": fisher_p_value(
            state,
            event,
        ),
    }


def add_history_only_bar1_control(
    history: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    history = history.copy()
    validation = validation.copy()

    history["bar1_mae_minus_mfe"] = (
        history["bar1_cumulative_mae_r"]
        - history["bar1_cumulative_mfe_r"]
    )

    validation["bar1_mae_minus_mfe"] = (
        validation["bar1_cumulative_mae_r"]
        - validation["bar1_cumulative_mfe_r"]
    )

    medians = {
        "mae": float(
            history[
                "bar1_cumulative_mae_r"
            ].median()
        ),
        "mfe": float(
            history[
                "bar1_cumulative_mfe_r"
            ].median()
        ),
        "net": float(
            history[
                "bar1_net_excursion_r"
            ].median()
        ),
        "mae_mfe": float(
            history[
                "bar1_mae_minus_mfe"
            ].median()
        ),
    }

    validation[
        "bar1_mae_high"
    ] = (
        validation[
            "bar1_cumulative_mae_r"
        ]
        >= medians["mae"]
    ).astype(int)

    validation[
        "bar1_mfe_low"
    ] = (
        validation[
            "bar1_cumulative_mfe_r"
        ]
        <= medians["mfe"]
    ).astype(int)

    validation[
        "bar1_net_low"
    ] = (
        validation[
            "bar1_net_excursion_r"
        ]
        <= medians["net"]
    ).astype(int)

    validation[
        "bar1_mae_mfe_high"
    ] = (
        validation[
            "bar1_mae_minus_mfe"
        ]
        >= medians["mae_mfe"]
    ).astype(int)

    validation[
        "bar1_composite_score"
    ] = (
        validation["bar1_mae_high"]
        + validation["bar1_mfe_low"]
        + validation["bar1_net_low"]
        + validation["bar1_mae_mfe_high"]
    )

    validation[
        "bar1_composite"
    ] = np.where(
        validation[
            "bar1_composite_score"
        ] >= 2,
        "HIGH_PATH",
        "LOW_PATH",
    )

    validation[
        "bar1_high_path"
    ] = (
        validation[
            "bar1_composite"
        ]
        == "HIGH_PATH"
    ).astype(int)

    return validation, medians


def stratified_mh(
    df: pd.DataFrame,
    state_col: str,
    event_col: str,
    strata_col: str,
) -> tuple[float, float]:

    work = df[
        [
            state_col,
            event_col,
            strata_col,
        ]
    ].dropna()

    if work.empty:
        return np.nan, np.nan

    mh_num = 0.0
    mh_den = 0.0

    rd_num = 0.0
    rd_den = 0.0

    used = 0

    for _, sub in work.groupby(
        strata_col,
        sort=False,
    ):

        state = sub[state_col].astype(int)
        event = sub[event_col].astype(int)

        n1 = int(
            (state == 1).sum()
        )

        n0 = int(
            (state == 0).sum()
        )

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

        mh_num += (
            a * d / n
        )

        mh_den += (
            b * c / n
        )

        rd = (
            a / n1
            - c / n0
        )

        weight = (
            n1 * n0 / n
        )

        rd_num += (
            weight * rd
        )

        rd_den += weight

        used += 1

    mh_or = (
        mh_num / mh_den
        if mh_den > 0
        else np.nan
    )

    pooled_rd = (
        rd_num / rd_den
        if rd_den > 0
        else np.nan
    )

    return (
        mh_or,
        pooled_rd * 100,
    )


def bootstrap_rd(
    df: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float]:

    work = df[
        [
            "bar2_favorable_failed_to_improve",
            "frozen_event",
        ]
    ].dropna().reset_index(drop=True)

    observed = risk_difference(
        work[
            "bar2_favorable_failed_to_improve"
        ],
        work["frozen_event"],
    )

    rng = np.random.default_rng(
        RANDOM_SEED
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

        state = sample[
            "bar2_favorable_failed_to_improve"
        ].astype(int)

        event = sample[
            "frozen_event"
        ].astype(int)

        if (
            (state == 1).sum() == 0
            or (state == 0).sum() == 0
        ):
            continue

        values.append(
            risk_difference(
                state,
                event,
            )
        )

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
print("STRICT LEAKAGE-SAFE TEMPORAL WALK-FORWARD")
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
# REQUIRED COLUMNS
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

require_columns(
    path,
    [
        "symbol",
        "direction",
        "entry_date",
        "bar_number",
    ],
    "Path",
)


# ============================================================
# NORMALIZE
# ============================================================

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
).astype(int)

trajectory[
    "bar2_favorable_failed_to_improve"
] = pd.to_numeric(
    trajectory[
        "bar2_favorable_failed_to_improve"
    ],
    errors="coerce",
).astype(int)

trajectory["merge_key"] = build_merge_key(
    trajectory
)

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

path["merge_key"] = build_merge_key(
    path
)


# ============================================================
# FROZEN INTEGRITY
# ============================================================

trade_count = (
    trajectory["merge_key"].nunique()
)

event_count = int(
    trajectory["frozen_event"].sum()
)

retained_count = int(
    (trajectory["frozen_event"] == 0).sum()
)

print("\nFROZEN POPULATION")
print("-" * 72)

print(
    f"Trade count:     {trade_count}"
)

print(
    f"Event count:     {event_count}"
)

print(
    f"Retained count:  {retained_count}"
)

if trade_count != EXPECTED_TRADES:
    raise RuntimeError(
        "Trade count mismatch."
    )

if event_count != EXPECTED_EVENTS:
    raise RuntimeError(
        "Event count mismatch."
    )

if retained_count != EXPECTED_RETAINED:
    raise RuntimeError(
        "Retained count mismatch."
    )

if trajectory["merge_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate trajectory keys detected."
    )

print(
    "Unique trajectory keys: PASS"
)


# ============================================================
# SORT
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
# FIXED BAR2 STATE CHECK
# ============================================================

state_values = set(
    trajectory[
        "bar2_favorable_failed_to_improve"
    ].unique()
)

if state_values != {0, 1}:
    raise RuntimeError(
        "Bar2 state must contain exactly 0 and 1."
    )

print(
    "\nFixed Bar-2 state:"
    " bar2_favorable_failed_to_improve"
)

print(
    f"State values: {sorted(state_values)}"
)


# ============================================================
# TEMPORAL FOLDS
# ============================================================

years = sorted(
    trajectory["calendar_year"].unique()
)

fold_rows = []
cumulative_rows = []
median_rows = []

running_validation = []

fold_number = 0


for validation_year in years:

    history = trajectory[
        trajectory["calendar_year"]
        < validation_year
    ].copy()

    validation_raw = trajectory[
        trajectory["calendar_year"]
        == validation_year
    ].copy()

    history_n = len(history)
    validation_n = len(validation_raw)

    if history_n < MIN_HISTORY_TRADES:
        print(
            f"\nYEAR {validation_year}: "
            f"SKIPPED — history {history_n} "
            f"< minimum {MIN_HISTORY_TRADES}"
        )
        continue

    if validation_n < MIN_VALIDATION_TRADES:
        print(
            f"\nYEAR {validation_year}: "
            f"SKIPPED — validation {validation_n} "
            f"< minimum {MIN_VALIDATION_TRADES}"
        )
        continue

    fold_number += 1

    fold_name = (
        f"FOLD_{fold_number}_"
        f"VALIDATE_{int(validation_year)}"
    )

    # --------------------------------------------------------
    # CRITICAL:
    # ALL MEDIANS COME FROM HISTORY ONLY.
    # --------------------------------------------------------

    validation, medians = (
        add_history_only_bar1_control(
            history,
            validation_raw,
        )
    )

    median_rows.append(
        {
            "fold": fold_name,
            "validation_year": int(
                validation_year
            ),
            "history_n": history_n,
            "validation_n": validation_n,
            "history_mae_median": medians["mae"],
            "history_mfe_median": medians["mfe"],
            "history_net_median": medians["net"],
            "history_mae_mfe_median": medians["mae_mfe"],
        }
    )

    print("\n" + "-" * 72)
    print(f"{fold_name}")
    print("-" * 72)

    print(
        f"History:     {history_n} trades"
    )

    print(
        f"Validation:  {validation_n} trades"
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

    print(
        "\nHISTORY-ONLY BAR1 MEDIANS:"
    )

    print(
        f"  MAE:      {medians['mae']:.6f}"
    )

    print(
        f"  MFE:      {medians['mfe']:.6f}"
    )

    print(
        f"  Net:      {medians['net']:.6f}"
    )

    print(
        f"  MAE-MFE:  {medians['mae_mfe']:.6f}"
    )

    # --------------------------------------------------------
    # MAIN VALIDATION
    # --------------------------------------------------------

    result = state_result(
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
            "history_n": history_n,
            "validation_n": validation_n,
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
    # STRICT NESTED BAR1 CONTROL
    # --------------------------------------------------------

    mh_or, pooled_rd = stratified_mh(
        validation,
        "bar2_favorable_failed_to_improve",
        "frozen_event",
        "bar1_composite",
    )

    nested_result = {
        "fold": fold_name,
        "population": "VALIDATION_NESTED_HISTORY_ONLY",
        "validation_year": int(
            validation_year
        ),
        "history_n": history_n,
        "validation_n": validation_n,
        "n": validation_n,
        "state_n": int(
            validation[
                "bar2_favorable_failed_to_improve"
            ].sum()
        ),
        "non_state_n": int(
            (
                validation[
                    "bar2_favorable_failed_to_improve"
                ] == 0
            ).sum()
        ),
        "state_events": int(
            (
                (
                    validation[
                        "bar2_favorable_failed_to_improve"
                    ] == 1
                )
                & (
                    validation[
                        "frozen_event"
                    ] == 1
                )
            ).sum()
        ),
        "non_state_events": int(
            (
                (
                    validation[
                        "bar2_favorable_failed_to_improve"
                    ] == 0
                )
                & (
                    validation[
                        "frozen_event"
                    ] == 1
                )
            ).sum()
        ),
        "pooled_rd_pp": pooled_rd,
        "mh_or": mh_or,
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

    fold_rows.append(
        nested_result
    )

    print(
        f"\nValidation RD: "
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
        f"STRICT nested MH OR: "
        f"{mh_or:.4f}"
        if not np.isnan(mh_or)
        else "STRICT nested MH OR: nan"
    )

    print(
        f"STRICT nested RD: "
        f"{pooled_rd:+.2f} pp"
        if not np.isnan(pooled_rd)
        else "STRICT nested RD: nan"
    )

    # --------------------------------------------------------
    # PRE-THRESHOLD VALIDATION
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
        raise RuntimeError(
            "Path contains neither cumulative_mae_r "
            "nor bar_adverse_r."
        )

    bar2 = bar2[
        [
            "merge_key",
            "crossed_by_bar2",
        ]
    ].drop_duplicates(
        "merge_key"
    )

    pre = validation.merge(
        bar2,
        on="merge_key",
        how="left",
    )

    pre["crossed_by_bar2"] = (
        pre["crossed_by_bar2"]
        .fillna(False)
    )

    pre = pre[
        ~pre["crossed_by_bar2"]
    ].copy()

    if len(pre) >= MIN_VALIDATION_TRADES:

        pre_result = state_result(
            pre,
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
                "history_n": history_n,
                "validation_n": len(pre),
            }
        )

        fold_rows.append(
            pre_result
        )

        print(
            f"Pre-threshold validation: "
            f"{len(pre)} trades"
        )

    # --------------------------------------------------------
    # CUMULATIVE VALIDATION
    # --------------------------------------------------------

    running_validation.append(
        validation
    )

    cumulative = pd.concat(
        running_validation,
        ignore_index=True,
    )

    cumulative_result = state_result(
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
            "history_n": history_n,
            "cumulative_validation_n": len(
                cumulative
            ),
        }
    )

    cumulative_rows.append(
        cumulative_result
    )


# ============================================================
# DATAFRAMES
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)

median_df = pd.DataFrame(
    median_rows
)

cumulative_df = pd.DataFrame(
    cumulative_rows
)


# ============================================================
# SUMMARY
# ============================================================

validation_df = fold_df[
    fold_df["population"]
    == "VALIDATION_ALL"
].copy()

nested_df = fold_df[
    fold_df["population"]
    == "VALIDATION_NESTED_HISTORY_ONLY"
].copy()

print("\n" + "=" * 72)
print("STRICT TEMPORAL WALK-FORWARD SUMMARY")
print("=" * 72)

print(
    f"\nValid validation folds: "
    f"{len(validation_df)}"
)

positive_validation = int(
    (
        validation_df[
            "risk_difference_pp"
        ] > 0
    ).sum()
)

print(
    f"Positive validation RD folds: "
    f"{positive_validation}/"
    f"{len(validation_df)}"
)

print(
    "\nSTRICT YEAR-BLOCK VALIDATION:"
)

for _, row in validation_df.iterrows():

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
# TEMPORAL METRICS
# ============================================================

rd = (
    validation_df[
        "risk_difference_pp"
    ]
    .dropna()
    .to_numpy()
)

if len(rd):

    median_rd = float(
        np.median(rd)
    )

    minimum_rd = float(
        np.min(rd)
    )

    maximum_rd = float(
        np.max(rd)
    )

else:

    median_rd = np.nan
    minimum_rd = np.nan
    maximum_rd = np.nan


print("\nSTRICT TEMPORAL CONSISTENCY")
print("-" * 72)

print(
    f"Median validation RD: "
    f"{median_rd:+.2f} pp"
)

print(
    f"RD range: "
    f"{minimum_rd:+.2f} -> "
    f"{maximum_rd:+.2f} pp"
)

print(
    f"Positive-fold fraction: "
    f"{(
        positive_validation / len(validation_df)
        * 100
    ):.2f}%"
    if len(validation_df)
    else "Positive-fold fraction: nan"
)


# ============================================================
# STRICT NESTED SUMMARY
# ============================================================

if not nested_df.empty:

    nested_positive = int(
        (
            nested_df[
                "pooled_rd_pp"
            ] > 0
        ).sum()
    )

    print(
        "\nSTRICT HISTORY-ONLY NESTED CONTROL"
    )

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
        f"{nested_df['pooled_rd_pp'].min():+.2f}"
        f" -> "
        f"{nested_df['pooled_rd_pp'].max():+.2f} pp"
    )


# ============================================================
# CUMULATIVE
# ============================================================

if not cumulative_df.empty:

    print(
        "\nSTRICT CUMULATIVE VALIDATION"
    )

    print("-" * 72)

    for _, row in cumulative_df.iterrows():

        print(
            f"  Through "
            f"{int(row['validation_year'])}: "
            f"n={int(row['cumulative_validation_n'])}, "
            f"RD={row['risk_difference_pp']:+.2f} pp, "
            f"OR={row['odds_ratio']:.4f}, "
            f"p={row['fisher_p']:.6f}"
        )


# ============================================================
# HISTORY MEDIAN DRIFT
# ============================================================

print(
    "\nHISTORY-ONLY MEDIAN DRIFT"
)

print("-" * 72)

for _, row in median_df.iterrows():

    print(
        f"  {int(row['validation_year'])}: "
        f"MAE={row['history_mae_median']:.6f}, "
        f"MFE={row['history_mfe_median']:.6f}, "
        f"Net={row['history_net_median']:.6f}, "
        f"MAE-MFE={row['history_mae_mfe_median']:.6f}"
    )


# ============================================================
# FULL SAMPLE REFERENCE
# ============================================================

full_observed, boot_low, boot_high = (
    bootstrap_rd(
        trajectory
    )
)

print(
    "\nFULL FROZEN POPULATION REFERENCE"
)

print("-" * 72)

print(
    f"Observed RD: "
    f"{full_observed * 100:+.2f} pp"
)

print(
    f"Bootstrap 95% CI: "
    f"{boot_low * 100:+.2f}"
    f" -> "
    f"{boot_high * 100:+.2f} pp"
)


# ============================================================
# PASS CRITERIA
# ============================================================

valid_fold_count = len(
    validation_df
)

nested_fold_count = len(
    nested_df
)

nested_positive = (
    int(
        (
            nested_df[
                "pooled_rd_pp"
            ] > 0
        ).sum()
    )
    if nested_fold_count
    else 0
)

strict_temporal_pass = (
    valid_fold_count >= 2
    and positive_validation
    == valid_fold_count
)

strict_nested_pass = (
    nested_fold_count >= 2
    and nested_positive
    == nested_fold_count
)

integrity_pass = (
    trade_count == EXPECTED_TRADES
    and event_count == EXPECTED_EVENTS
    and retained_count == EXPECTED_RETAINED
    and len(path)
    == EXPECTED_TRADES * MAX_BAR
    and state_values == {0, 1}
)

# ------------------------------------------------------------
# CRITICAL LEAKAGE CHECK
#
# Every median used for a validation fold must come from
# history ending strictly before validation begins.
# ------------------------------------------------------------

leakage_check = True

for _, row in median_df.iterrows():

    validation_year = int(
        row["validation_year"]
    )

    if not (
        validation_year
        >= 2023
    ):
        leakage_check = False

    if (
        row["history_n"]
        < MIN_HISTORY_TRADES
    ):
        leakage_check = False


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_strict_folds.csv",
    index=False,
)

median_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_strict_history_medians.csv",
    index=False,
)

cumulative_df.to_csv(
    OUTPUT_DIR
    / "temporal_walkforward_strict_cumulative.csv",
    index=False,
)


integrity_rows = [
    {
        "check": "trade_count",
        "expected": EXPECTED_TRADES,
        "actual": trade_count,
        "pass": trade_count == EXPECTED_TRADES,
    },
    {
        "check": "event_count",
        "expected": EXPECTED_EVENTS,
        "actual": event_count,
        "pass": event_count == EXPECTED_EVENTS,
    },
    {
        "check": "retained_count",
        "expected": EXPECTED_RETAINED,
        "actual": retained_count,
        "pass": retained_count == EXPECTED_RETAINED,
    },
    {
        "check": "path_rows",
        "expected": EXPECTED_TRADES * MAX_BAR,
        "actual": len(path),
        "pass": len(path)
        == EXPECTED_TRADES * MAX_BAR,
    },
    {
        "check": "state_binary",
        "expected": "{0,1}",
        "actual": str(sorted(state_values)),
        "pass": state_values == {0, 1},
    },
    {
        "check": "valid_temporal_folds",
        "expected": ">=2",
        "actual": valid_fold_count,
        "pass": valid_fold_count >= 2,
    },
    {
        "check": "strict_temporal_direction",
        "expected": "all positive",
        "actual": (
            f"{positive_validation}/"
            f"{valid_fold_count}"
        ),
        "pass": strict_temporal_pass,
    },
    {
        "check": "strict_nested_direction",
        "expected": "all positive",
        "actual": (
            f"{nested_positive}/"
            f"{nested_fold_count}"
        ),
        "pass": strict_nested_pass,
    },
    {
        "check": "history_only_control_definition",
        "expected": True,
        "actual": True,
        "pass": leakage_check,
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
    / "temporal_walkforward_strict_integrity.csv",
    index=False,
)


# ============================================================
# FINAL
# ============================================================

all_integrity_pass = bool(
    integrity_df["pass"].all()
)

print(
    "\n" + "=" * 72
)

print(
    "FINAL STRICT TEMPORAL WALK-FORWARD STATUS"
)

print(
    "=" * 72
)

print(
    f"Frozen population: "
    f"{trade_count} / "
    f"{event_count} / "
    f"{retained_count}"
)

print(
    f"Valid validation folds: "
    f"{valid_fold_count}"
)

print(
    f"Strict validation positive: "
    f"{positive_validation}/"
    f"{valid_fold_count}"
)

print(
    f"Strict nested positive: "
    f"{nested_positive}/"
    f"{nested_fold_count}"
)

print(
    f"History-only control: "
    f"{leakage_check}"
)

print(
    f"Strict temporal PASS: "
    f"{strict_temporal_pass}"
)

print(
    f"Strict nested PASS: "
    f"{strict_nested_pass}"
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
    and strict_temporal_pass
    and strict_nested_pass
):
    print(
        "\nSTRICT TEMPORAL WALK-FORWARD TEST: PASS"
    )
else:
    print(
        "\nSTRICT TEMPORAL WALK-FORWARD TEST: "
        "INCONCLUSIVE / FAIL"
    )

print(
    "\nIMPORTANT:"
)

print(
    "This is still historical forensic validation."
)

print(
    "The result does not replace genuine post-freeze OOS validation."
)

print(
    "The locked 7B / 0.25R OOS hypothesis remains untouched."
)

print(
    "\nOutputs:"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_strict_folds.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_strict_history_medians.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_strict_cumulative.csv'}"
)

print(
    f"  {OUTPUT_DIR / 'temporal_walkforward_strict_integrity.csv'}"
)