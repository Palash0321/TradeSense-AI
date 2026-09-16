"""
TradeSense-AI
Research-only forensic audit:
Bar1 -> Bar2 state independence and label-proximity analysis.

Purpose
-------
Test whether the fixed Bar2 state
    bar2_favorable_failed_to_improve
contains information beyond Bar1 baseline path conditions, and whether
its apparent separation is merely a consequence of proximity to the
frozen 0.25R early-adverse label.

Frozen research population
--------------------------
106 trades
60 frozen early-adverse events
46 retained

Locked hypothesis used only for audit context:
    early adverse >= 0.25R within first 7 entry bars

Important
---------
- No production files are modified.
- No OOS files are modified.
- No threshold optimization.
- No ML/model fitting.
- No strategy rule is promoted.
"""

from __future__ import annotations

from pathlib import Path
import math

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
    "tests/output/research/early_adverse_bar12_independence"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

THRESHOLD_R = 0.25
MIN_SUBGROUP_N = 8

N_PERMUTATIONS = 10000
RANDOM_SEED = 20260916


# ============================================================
# HELPERS
# ============================================================

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def wilson_ci(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:

    if n <= 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1.0 + (z**2 / n)

    center = (
        p + (z**2 / (2.0 * n))
    ) / denominator

    half = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / n
            )
            + (
                z**2 / (4.0 * n**2)
            )
        )
        / denominator
    )

    return center - half, center + half


def rd_ci(
    state1_event: int,
    state1_n: int,
    state0_event: int,
    state0_n: int,
) -> tuple[float, float]:

    if state1_n <= 0 or state0_n <= 0:
        return np.nan, np.nan

    p1 = state1_event / state1_n
    p0 = state0_event / state0_n

    z = 1.959963984540054

    se = math.sqrt(
        (
            p1 * (1.0 - p1)
            / state1_n
        )
        +
        (
            p0 * (1.0 - p0)
            / state0_n
        )
    )

    rd = p1 - p0

    half = z * se

    return rd - half, rd + half


def state_event_statistics(
    state: pd.Series,
    event: pd.Series,
) -> dict:

    state = pd.to_numeric(
        state,
        errors="coerce",
    )

    event = pd.to_numeric(
        event,
        errors="coerce",
    )

    valid = (
        state.notna()
        & event.notna()
    )

    state = state.loc[valid].astype(int)
    event = event.loc[valid].astype(int)

    a = int(
        (
            (state == 1)
            & (event == 1)
        ).sum()
    )

    b = int(
        (
            (state == 1)
            & (event == 0)
        ).sum()
    )

    c = int(
        (
            (state == 0)
            & (event == 1)
        ).sum()
    )

    d = int(
        (
            (state == 0)
            & (event == 0)
        ).sum()
    )

    state1_n = a + b
    state0_n = c + d

    rate1 = (
        a / state1_n
        if state1_n
        else np.nan
    )

    rate0 = (
        c / state0_n
        if state0_n
        else np.nan
    )

    rd = (
        rate1 - rate0
        if state1_n and state0_n
        else np.nan
    )

    ci_low, ci_high = rd_ci(
        a,
        state1_n,
        c,
        state0_n,
    )

    odds_ratio, fisher_p = fisher_exact(
        [
            [a, b],
            [c, d],
        ]
    )

    return {
        "state1_event_n": a,
        "state1_non_event_n": b,
        "state0_event_n": c,
        "state0_non_event_n": d,
        "state1_n": state1_n,
        "state0_n": state0_n,
        "event_rate_state1": rate1,
        "event_rate_state0": rate0,
        "risk_difference_pp": rd * 100.0,
        "rd_ci_low_pp": ci_low * 100.0,
        "rd_ci_high_pp": ci_high * 100.0,
        "odds_ratio": odds_ratio,
        "fisher_p": fisher_p,
    }


def permutation_test(
    state: np.ndarray,
    event: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict:

    state = np.asarray(
        state,
        dtype=int,
    )

    event = np.asarray(
        event,
        dtype=int,
    )

    state1_mask = state == 1
    state0_mask = state == 0

    observed = (
        event[state1_mask].mean()
        -
        event[state0_mask].mean()
    )

    rng = np.random.default_rng(seed)

    null_distribution = np.empty(
        n_permutations,
        dtype=float,
    )

    n_state1 = int(
        state1_mask.sum()
    )

    for i in range(n_permutations):

        shuffled = rng.permutation(
            event
        )

        null_distribution[i] = (
            shuffled[:n_state1].mean()
            -
            shuffled[n_state1:].mean()
        )

    p_value = (
        np.sum(
            np.abs(null_distribution)
            >= abs(observed)
        )
        + 1
    ) / (
        n_permutations + 1
    )

    return {
        "observed_rd": observed,
        "permutation_p": p_value,
        "null_q025": np.quantile(
            null_distribution,
            0.025,
        ),
        "null_q975": np.quantile(
            null_distribution,
            0.975,
        ),
    }


def median_control_strata(
    df: pd.DataFrame,
    control_column: str,
    state_column: str,
    event_column: str,
) -> list[dict]:

    work = df[
        [
            control_column,
            state_column,
            event_column,
        ]
    ].copy()

    work[control_column] = pd.to_numeric(
        work[control_column],
        errors="coerce",
    )

    work = work.dropna()

    if len(work) < MIN_SUBGROUP_N:
        return []

    median = float(
        work[control_column].median()
    )

    work["control_high"] = (
        work[control_column] >= median
    ).astype(int)

    output = []

    for group_value, group_label in [
        (0, "LOW"),
        (1, "HIGH"),
    ]:

        subgroup = work[
            work["control_high"]
            == group_value
        ]

        if len(subgroup) < MIN_SUBGROUP_N:
            continue

        stats = state_event_statistics(
            subgroup[state_column],
            subgroup[event_column],
        )

        stats.update(
            {
                "control": control_column,
                "control_group": group_label,
                "control_median": median,
                "n": len(subgroup),
            }
        )

        output.append(stats)

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "TRADE SENSE-AI — BAR1/BAR2 INDEPENDENCE "
        "+ LABEL-PROXIMITY FORENSIC AUDIT"
    )
    print("=" * 78)
    print()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    trajectory = pd.read_csv(
        TRAJECTORY_FILE
    )

    path = pd.read_csv(
        PATH_FILE
    )

    print(
        f"Trajectory rows: {len(trajectory)}"
    )

    print(
        f"Trajectory columns: "
        f"{len(trajectory.columns)}"
    )

    print(
        f"Path rows:       {len(path)}"
    )

    print()

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    require_columns(
        trajectory,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",

            "bar1_cumulative_mae_r",
            "bar1_cumulative_mfe_r",
            "bar1_net_excursion_r",

            "bar2_cumulative_mae_r",
            "bar2_cumulative_mfe_r",
            "bar2_net_excursion_r",

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
            "cumulative_mae_r",
            "cumulative_mfe_r",
            "net_excursion_r",
            "crossed_0_25r_by_this_bar",
        ],
        "Path",
    )

    # --------------------------------------------------------
    # NORMALIZE KEYS
    # --------------------------------------------------------

    for df in [trajectory, path]:

        df["symbol"] = (
            df["symbol"]
            .astype(str)
            .str.upper()
        )

        df["direction"] = (
            df["direction"]
            .astype(str)
            .str.upper()
        )

        df["entry_date"] = (
            df["entry_date"]
            .astype(str)
        )

    trajectory["merge_key"] = (
        trajectory["symbol"]
        + "|"
        + trajectory["direction"]
        + "|"
        + trajectory["entry_date"]
    )

    path["merge_key"] = (
        path["symbol"]
        + "|"
        + path["direction"]
        + "|"
        + path["entry_date"]
    )

    duplicate_keys = int(
        trajectory["merge_key"]
        .duplicated()
        .sum()
    )

    if duplicate_keys:
        raise RuntimeError(
            f"Trajectory duplicate keys: "
            f"{duplicate_keys}"
        )

    # --------------------------------------------------------
    # FROZEN POPULATION
    # --------------------------------------------------------

    if len(trajectory) != EXPECTED_TRADES:

        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} trades, "
            f"got {len(trajectory)}"
        )

    trajectory["frozen_event"] = (
        pd.to_numeric(
            trajectory["frozen_event"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    event_count = int(
        trajectory["frozen_event"].sum()
    )

    retained_count = (
        len(trajectory)
        - event_count
    )

    if event_count != EXPECTED_EVENTS:

        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events, "
            f"got {event_count}"
        )

    if retained_count != EXPECTED_RETAINED:

        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED} retained, "
            f"got {retained_count}"
        )

    # --------------------------------------------------------
    # RECONSTRUCT EARLIEST 0.25R CROSSING
    # --------------------------------------------------------

    path["bar_number"] = pd.to_numeric(
        path["bar_number"],
        errors="coerce",
    )

    path[
        "crossed_0_25r_by_this_bar"
    ] = (
        pd.to_numeric(
            path[
                "crossed_0_25r_by_this_bar"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    crossed = path[
        path[
            "crossed_0_25r_by_this_bar"
        ] == 1
    ].copy()

    earliest = (
        crossed
        .groupby(
            "merge_key",
            as_index=False,
        )["bar_number"]
        .min()
        .rename(
            columns={
                "bar_number":
                    "earliest_crossing_bar"
            }
        )
    )

    trajectory = trajectory.merge(
        earliest,
        on="merge_key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # FIXED BAR2 STATE
    # --------------------------------------------------------

    trajectory["state"] = (
        pd.to_numeric(
            trajectory[
                "bar2_favorable_failed_to_improve"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # BAR1 DISTANCE TO FROZEN THRESHOLD
    # --------------------------------------------------------

    trajectory[
        "bar1_distance_to_025r"
    ] = (
        THRESHOLD_R
        -
        pd.to_numeric(
            trajectory[
                "bar1_cumulative_mae_r"
            ],
            errors="coerce",
        )
    )

    trajectory[
        "bar2_distance_to_025r"
    ] = (
        THRESHOLD_R
        -
        pd.to_numeric(
            trajectory[
                "bar2_cumulative_mae_r"
            ],
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # CROSSING TIMING
    # --------------------------------------------------------

    trajectory[
        "earliest_crossing_bar"
    ] = pd.to_numeric(
        trajectory[
            "earliest_crossing_bar"
        ],
        errors="coerce",
    )

    trajectory["crossed_by_bar2"] = (
        trajectory[
            "earliest_crossing_bar"
        ]
        <= 2
    ).fillna(False).astype(int)

    trajectory["crossed_after_bar2"] = (
        trajectory[
            "earliest_crossing_bar"
        ]
        > 2
    ).fillna(False).astype(int)

    # ========================================================
    # 1. GLOBAL STATE SEPARATION
    # ========================================================

    global_stats = state_event_statistics(
        trajectory["state"],
        trajectory["frozen_event"],
    )

    global_df = pd.DataFrame(
        [
            {
                "analysis":
                    "global_state_vs_frozen_label",
                **global_stats,
            }
        ]
    )

    # ========================================================
    # 2. BAR1 CONTROL STRATA
    # ========================================================

    controls = [
        "bar1_cumulative_mae_r",
        "bar1_cumulative_mfe_r",
        "bar1_net_excursion_r",
        "bar1_distance_to_025r",
    ]

    control_rows = []

    for control in controls:

        control_rows.extend(
            median_control_strata(
                trajectory,
                control,
                "state",
                "frozen_event",
            )
        )

    control_df = pd.DataFrame(
        control_rows
    )

    # ========================================================
    # 3. PRE-THRESHOLD POPULATION
    # ========================================================

    pre_threshold = trajectory[
        trajectory["crossed_by_bar2"] == 0
    ].copy()

    pre_threshold_rows = []

    if len(pre_threshold) > 0:

        stats = state_event_statistics(
            pre_threshold["state"],
            pre_threshold["frozen_event"],
        )

        pre_threshold_rows.append(
            {
                "population":
                    "not_crossed_025r_by_bar2",
                "n":
                    len(pre_threshold),
                **stats,
            }
        )

    pre_threshold_df = pd.DataFrame(
        pre_threshold_rows
    )

    # ========================================================
    # 4. DELAYED EVENTS
    # ========================================================

    delayed_events = trajectory[
        (
            trajectory["frozen_event"]
            == 1
        )
        &
        (
            trajectory[
                "crossed_after_bar2"
            ]
            == 1
        )
    ].copy()

    retained = trajectory[
        trajectory["frozen_event"] == 0
    ].copy()

    delayed_summary = pd.DataFrame(
        [
            {
                "population":
                    "delayed_events_after_bar2",
                "n":
                    len(delayed_events),
                "state_triggered":
                    int(
                        delayed_events[
                            "state"
                        ].sum()
                    ),
                "state_rate":
                    delayed_events[
                        "state"
                    ].mean()
                    if len(delayed_events)
                    else np.nan,
            },
            {
                "population":
                    "retained",
                "n":
                    len(retained),
                "state_triggered":
                    int(
                        retained[
                            "state"
                        ].sum()
                    ),
                "state_rate":
                    retained[
                        "state"
                    ].mean()
                    if len(retained)
                    else np.nan,
            },
        ]
    )

    # ========================================================
    # 5. FIXED DISTANCE BANDS
    # ========================================================

    trajectory[
        "bar1_distance_band"
    ] = pd.cut(
        trajectory[
            "bar1_distance_to_025r"
        ],
        bins=[
            -np.inf,
            0.05,
            0.10,
            0.15,
            0.20,
            THRESHOLD_R,
            np.inf,
        ],
        labels=[
            "<=0.05R",
            "0.05-0.10R",
            "0.10-0.15R",
            "0.15-0.20R",
            "0.20-0.25R",
            ">0.25R",
        ],
        include_lowest=True,
        right=True,
    )

    distance_rows = []

    for band in trajectory[
        "bar1_distance_band"
    ].cat.categories:

        sub = trajectory[
            trajectory[
                "bar1_distance_band"
            ]
            == band
        ]

        if sub.empty:
            continue

        event_sub = sub[
            sub["frozen_event"] == 1
        ]

        retained_sub = sub[
            sub["frozen_event"] == 0
        ]

        distance_rows.append(
            {
                "distance_band": str(band),
                "n": len(sub),
                "events": int(
                    sub[
                        "frozen_event"
                    ].sum()
                ),
                "state_triggered":
                    int(sub["state"].sum()),
                "state_rate_all":
                    sub["state"].mean(),
                "state_rate_event":
                    event_sub[
                        "state"
                    ].mean()
                    if len(event_sub)
                    else np.nan,
                "state_rate_retained":
                    retained_sub[
                        "state"
                    ].mean()
                    if len(retained_sub)
                    else np.nan,
            }
        )

    distance_df = pd.DataFrame(
        distance_rows
    )

    # ========================================================
    # 6. CROSSING TIMING
    # ========================================================

    timing_rows = []

    for bar in [1, 2, 3, 4, 5, 6, 7]:

        sub = trajectory[
            trajectory[
                "earliest_crossing_bar"
            ]
            == bar
        ]

        if sub.empty:
            continue

        event_sub = sub[
            sub["frozen_event"] == 1
        ]

        retained_sub = sub[
            sub["frozen_event"] == 0
        ]

        timing_rows.append(
            {
                "earliest_crossing_bar":
                    bar,
                "n":
                    len(sub),
                "events":
                    int(
                        sub[
                            "frozen_event"
                        ].sum()
                    ),
                "state_rate":
                    sub["state"].mean(),
                "state_rate_event":
                    event_sub[
                        "state"
                    ].mean()
                    if len(event_sub)
                    else np.nan,
                "state_rate_retained":
                    retained_sub[
                        "state"
                    ].mean()
                    if len(retained_sub)
                    else np.nan,
            }
        )

    timing_df = pd.DataFrame(
        timing_rows
    )

    # ========================================================
    # 7. EVENT TIMING ONLY
    # ========================================================

    event_timing = trajectory[
        trajectory["frozen_event"] == 1
    ].copy()

    event_timing_df = (
        event_timing
        .groupby(
            "earliest_crossing_bar",
            dropna=False,
        )
        .agg(
            n=("state", "size"),
            state_triggered=(
                "state",
                "sum",
            ),
            state_rate=(
                "state",
                "mean",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # 8. PERMUTATION TEST
    # ========================================================

    permutation = permutation_test(
        trajectory["state"].to_numpy(),
        trajectory["frozen_event"].to_numpy(),
        N_PERMUTATIONS,
        RANDOM_SEED,
    )

    permutation_df = pd.DataFrame(
        [permutation]
    )

    # ========================================================
    # 9. ROBUSTNESS
    # ========================================================

    trajectory["year"] = pd.to_datetime(
        trajectory["entry_date"],
        errors="coerce",
    ).dt.year

    subgroup_rows = []

    for dimension in [
        "year",
        "symbol",
        "setup",
        "direction",
    ]:

        for group, subgroup in trajectory.groupby(
            dimension,
            dropna=False,
        ):

            if len(subgroup) < MIN_SUBGROUP_N:
                continue

            stats = state_event_statistics(
                subgroup["state"],
                subgroup["frozen_event"],
            )

            subgroup_rows.append(
                {
                    "dimension":
                        dimension,
                    "group":
                        str(group),
                    "n":
                        len(subgroup),
                    **stats,
                }
            )

    subgroup_df = pd.DataFrame(
        subgroup_rows
    )

    # ========================================================
    # 10. INTERPRETATION FLAGS
    # ========================================================

    interpretation_df = pd.DataFrame(
        [
            {
                "check":
                    "global_association",
                "result":
                    (
                        "SIGNIFICANT"
                        if global_stats[
                            "fisher_p"
                        ] < 0.05
                        else
                        "NOT_SIGNIFICANT"
                    ),
                "value":
                    global_stats[
                        "fisher_p"
                    ],
            },
            {
                "check":
                    "pre_threshold_population_available",
                "result":
                    (
                        "YES"
                        if len(pre_threshold)
                        > 0
                        else
                        "NO"
                    ),
                "value":
                    len(pre_threshold),
            },
            {
                "check":
                    "delayed_event_population",
                "result":
                    (
                        "AVAILABLE"
                        if len(delayed_events)
                        > 0
                        else
                        "EMPTY"
                    ),
                "value":
                    len(delayed_events),
            },
            {
                "check":
                    "permutation_consistency",
                "result":
                    (
                        "SIGNIFICANT"
                        if permutation[
                            "permutation_p"
                        ] < 0.05
                        else
                        "NOT_SIGNIFICANT"
                    ),
                "value":
                    permutation[
                        "permutation_p"
                    ],
            },
            {
                "check":
                    "strategy_rule_promoted",
                "result":
                    "NO",
                "value":
                    np.nan,
            },
            {
                "check":
                    "production_modified",
                "result":
                    "NO",
                "value":
                    np.nan,
            },
            {
                "check":
                    "oos_modified",
                "result":
                    "NO",
                "value":
                    np.nan,
            },
        ]
    )

    # ========================================================
    # 11. INTEGRITY
    # ========================================================

    integrity_rows = [
        {
            "check":
                "trajectory_rows",
            "expected":
                EXPECTED_TRADES,
            "actual":
                len(trajectory),
            "pass":
                len(trajectory)
                == EXPECTED_TRADES,
        },
        {
            "check":
                "frozen_events",
            "expected":
                EXPECTED_EVENTS,
            "actual":
                event_count,
            "pass":
                event_count
                == EXPECTED_EVENTS,
        },
        {
            "check":
                "retained",
            "expected":
                EXPECTED_RETAINED,
            "actual":
                retained_count,
            "pass":
                retained_count
                == EXPECTED_RETAINED,
        },
        {
            "check":
                "trajectory_unique_keys",
            "expected":
                EXPECTED_TRADES,
            "actual":
                len(trajectory),
            "pass":
                duplicate_keys == 0,
        },
        {
            "check":
                "path_rows",
            "expected":
                530,
            "actual":
                len(path),
            "pass":
                len(path) == 530,
        },
        {
            "check":
                "state_binary",
            "expected":
                "0/1",
            "actual":
                sorted(
                    trajectory[
                        "state"
                    ].unique().tolist()
                ),
            "pass":
                set(
                    trajectory[
                        "state"
                    ].unique()
                ).issubset({0, 1}),
        },
        {
            "check":
                "pre_threshold_rows",
            "expected":
                ">0",
            "actual":
                len(pre_threshold),
            "pass":
                len(pre_threshold) > 0,
        },
        {
            "check":
                "subgroup_rows",
            "expected":
                ">0",
            "actual":
                len(subgroup_df),
            "pass":
                len(subgroup_df) > 0,
        },
        {
            "check":
                "production_modified",
            "expected":
                False,
            "actual":
                False,
            "pass":
                True,
        },
        {
            "check":
                "oos_modified",
            "expected":
                False,
            "actual":
                False,
            "pass":
                True,
        },
        {
            "check":
                "strategy_rule_promoted",
            "expected":
                False,
            "actual":
                False,
            "pass":
                True,
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    global_df.to_csv(
        OUTPUT_DIR
        / "global_state_separation.csv",
        index=False,
    )

    control_df.to_csv(
        OUTPUT_DIR
        / "bar1_control_strata.csv",
        index=False,
    )

    pre_threshold_df.to_csv(
        OUTPUT_DIR
        / "pre_threshold_separation.csv",
        index=False,
    )

    delayed_summary.to_csv(
        OUTPUT_DIR
        / "delayed_event_summary.csv",
        index=False,
    )

    distance_df.to_csv(
        OUTPUT_DIR
        / "distance_band_analysis.csv",
        index=False,
    )

    timing_df.to_csv(
        OUTPUT_DIR
        / "crossing_timing_analysis.csv",
        index=False,
    )

    event_timing_df.to_csv(
        OUTPUT_DIR
        / "event_timing_state_rate.csv",
        index=False,
    )

    permutation_df.to_csv(
        OUTPUT_DIR
        / "permutation_test.csv",
        index=False,
    )

    subgroup_df.to_csv(
        OUTPUT_DIR
        / "subgroup_robustness.csv",
        index=False,
    )

    interpretation_df.to_csv(
        OUTPUT_DIR
        / "interpretation_flags.csv",
        index=False,
    )

    integrity_df.to_csv(
        OUTPUT_DIR
        / "independence_integrity.csv",
        index=False,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("FROZEN POPULATION")
    print(
        f"  Trades:   {len(trajectory)}"
    )
    print(
        f"  Events:   {event_count}"
    )
    print(
        f"  Retained: {retained_count}"
    )
    print()

    print(
        "GLOBAL BAR2 STATE VS FROZEN LABEL"
    )

    print(
        f"  State event rate:     "
        f"{global_stats['event_rate_state1']:.4f}"
    )

    print(
        f"  Non-state event rate: "
        f"{global_stats['event_rate_state0']:.4f}"
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

    print()

    print("BAR1 CONTROL STRATA")

    if not control_df.empty:

        print(
            control_df[
                [
                    "control",
                    "control_group",
                    "n",
                    "risk_difference_pp",
                    "fisher_p",
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "  No valid control strata."
        )

    print()

    print(
        "PRE-THRESHOLD ANALYSIS"
    )

    print(
        "  Trades not crossed 0.25R "
        f"by Bar2: {len(pre_threshold)}"
    )

    if not pre_threshold_df.empty:

        row = pre_threshold_df.iloc[0]

        print(
            f"  State event rate: "
            f"{row['event_rate_state1']:.4f}"
        )

        print(
            f"  Non-state event rate: "
            f"{row['event_rate_state0']:.4f}"
        )

        print(
            f"  Difference: "
            f"{row['risk_difference_pp']:+.2f} pp"
        )

        print(
            f"  Fisher p: "
            f"{row['fisher_p']:.6f}"
        )

    print()

    print("DELAYED EVENTS")

    print(
        f"  Delayed frozen events: "
        f"{len(delayed_events)}"
    )

    if len(delayed_events):

        delayed_rate = (
            delayed_events["state"].mean()
        )

        retained_rate = (
            retained["state"].mean()
        )

        print(
            f"  Delayed-event state rate: "
            f"{delayed_rate:.4f}"
        )

        print(
            f"  Retained state rate:      "
            f"{retained_rate:.4f}"
        )

        print(
            f"  Difference: "
            f"{(delayed_rate - retained_rate) * 100:+.2f} pp"
        )

    print()

    print("PERMUTATION TEST")

    print(
        f"  Observed RD: "
        f"{permutation['observed_rd'] * 100:+.2f} pp"
    )

    print(
        f"  Permutation p: "
        f"{permutation['permutation_p']:.6f}"
    )

    print()

    print("DISTANCE-BAND ANALYSIS")

    if not distance_df.empty:

        print(
            distance_df.to_string(
                index=False
            )
        )

    print()

    print("CROSSING TIMING")

    if not timing_df.empty:

        print(
            timing_df.to_string(
                index=False
            )
        )

    print()

    print("FINAL INTEGRITY")

    all_pass = bool(
        integrity_df["pass"].all()
    )

    for _, row in integrity_df.iterrows():

        print(
            f"  {row['check']}: "
            f"{row['actual']} / "
            f"{row['expected']} "
            f"{'PASS' if row['pass'] else 'FAIL'}"
        )

    print()

    if all_pass:

        print(
            "PASS — BAR1/BAR2 INDEPENDENCE "
            "+ LABEL-PROXIMITY FORENSIC AUDIT COMPLETE"
        )

    else:

        print(
            "FAIL — INTEGRITY CHECK FAILED"
        )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())