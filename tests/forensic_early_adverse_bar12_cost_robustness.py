"""
TradeSense-AI — Bar 1→2 Cost-Adjusted Robustness Forensic Analysis

RESEARCH ONLY.

Purpose
-------
Stress-test the already established Bar-2 favorable-failure
counterfactual at fixed execution-cost assumptions.

Frozen population:
    106 total
    60 early-adverse events
    46 retained

Fixed state:
    bar2_favorable_failed_to_improve

Stress costs:
    0.10R
    0.20R

This script tests:
    1. Overall cost-adjusted improvement
    2. Event vs retained contribution
    3. Year robustness
    4. Symbol robustness
    5. Setup robustness
    6. Direction robustness
    7. Leave-one-year-out
    8. Leave-one-symbol-out
    9. Top-N contributor concentration
   10. Removal of largest positive contributors

No optimization is performed.
No production/OOS files are modified.
No strategy rule is promoted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

COUNTERFACTUAL_FILE = Path(
    "tests/output/research/early_adverse_bar12_counterfactual/"
    "bar12_counterfactual_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_cost_robustness"
)

COST_LEVELS_R = [0.10, 0.20]

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

MIN_SUBGROUP_TRADES = 3


# ============================================================
# HELPERS
# ============================================================

def first_existing(
    df: pd.DataFrame,
    candidates: List[str],
) -> str:

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not resolve any of:\n{candidates}\n\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def state_to_bool(series: pd.Series) -> pd.Series:

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        [
            "true",
            "1",
            "yes",
            "y",
            "t",
        ]
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int = 10000,
    seed: int = 42,
):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def summarize_values(
    values: pd.Series,
) -> Dict[str, float]:

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return {
            "n": 0,
            "total_r": np.nan,
            "mean_r": np.nan,
            "median_r": np.nan,
            "positive": 0,
            "negative": 0,
            "zero": 0,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    arr = values.to_numpy(dtype=float)

    ci_low, ci_high = bootstrap_mean_ci(arr)

    return {
        "n": int(len(arr)),
        "total_r": float(arr.sum()),
        "mean_r": float(arr.mean()),
        "median_r": float(np.median(arr)),
        "positive": int(np.sum(arr > 1e-12)),
        "negative": int(np.sum(arr < -1e-12)),
        "zero": int(np.sum(np.abs(arr) <= 1e-12)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================
# LOAD
# ============================================================

def load_data() -> pd.DataFrame:

    if not COUNTERFACTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing counterfactual file:\n"
            f"{COUNTERFACTUAL_FILE}"
        )

    df = pd.read_csv(
        COUNTERFACTUAL_FILE
    )

    print("=" * 72)
    print(
        "TRADE SENSE-AI — "
        "BAR 1→2 COST-ADJUSTED ROBUSTNESS"
    )
    print("=" * 72)
    print()
    print(
        f"Input file: {COUNTERFACTUAL_FILE}"
    )
    print(
        f"Input rows: {len(df)}"
    )
    print(
        f"Input columns: {len(df.columns)}"
    )
    print()

    return df


# ============================================================
# RESOLVE SCHEMA
# ============================================================

def resolve_columns(
    df: pd.DataFrame,
) -> Dict[str, str]:

    c = {}

    c["state"] = first_existing(
        df,
        [
            "bar2_favorable_failed_to_improve",
        ],
    )

    c["gross_delta"] = first_existing(
        df,
        [
            "bar2_favorable_failed_to_improve_counterfactual_delta_r",
        ],
    )

    c["event"] = first_existing(
        df,
        [
            "frozen_event",
            "event",
        ],
    )

    c["symbol"] = first_existing(
        df,
        [
            "symbol",
            "_symbol",
        ],
    )

    c["direction"] = first_existing(
        df,
        [
            "direction",
        ],
    )

    c["setup"] = first_existing(
        df,
        [
            "setup",
        ],
    )

    c["entry_date"] = first_existing(
        df,
        [
            "entry_date",
        ],
    )

    print("Resolved columns:")

    for key, value in c.items():
        print(
            f"  {key:15s}: {value}"
        )

    print()

    return c


# ============================================================
# PREPARE
# ============================================================

def prepare(
    df: pd.DataFrame,
    c: Dict[str, str],
) -> pd.DataFrame:

    work = df.copy()

    work["_state"] = state_to_bool(
        work[c["state"]]
    )

    work["_gross_delta_r"] = pd.to_numeric(
        work[c["gross_delta"]],
        errors="coerce",
    )

    work["_event"] = state_to_bool(
        work[c["event"]]
    )

    work["_symbol"] = (
        work[c["symbol"]]
        .astype(str)
        .str.strip()
    )

    work["_direction"] = (
        work[c["direction"]]
        .astype(str)
        .str.strip()
    )

    work["_setup"] = (
        work[c["setup"]]
        .astype(str)
        .str.strip()
    )

    work["_entry_date"] = pd.to_datetime(
        work[c["entry_date"]],
        errors="coerce",
    )

    work["_year"] = (
        work["_entry_date"]
        .dt.year
    )

    return work


# ============================================================
# INTEGRITY
# ============================================================

def integrity_check(
    df: pd.DataFrame,
) -> None:

    trade_count = len(df)

    event_count = int(
        df["_event"].sum()
    )

    retained_count = (
        trade_count - event_count
    )

    state_count = int(
        df["_state"].sum()
    )

    checks = {
        "trade_count": (
            trade_count == EXPECTED_TRADES
        ),
        "event_count": (
            event_count == EXPECTED_EVENTS
        ),
        "retained_count": (
            retained_count == EXPECTED_RETAINED
        ),
        "state_nonempty": (
            state_count > 0
        ),
        "gross_delta_complete": (
            df.loc[
                df["_state"],
                "_gross_delta_r",
            ].notna().all()
        ),
        "year_complete": (
            df["_year"].notna().all()
        ),
    }

    print("=" * 72)
    print("INTEGRITY CHECK")
    print("=" * 72)
    print()

    print(
        f"Trade count:       {trade_count}"
    )
    print(
        f"Frozen events:     {event_count}"
    )
    print(
        f"Frozen retained:   {retained_count}"
    )
    print(
        f"Triggered trades:  {state_count}"
    )
    print()

    for name, passed in checks.items():

        print(
            f"{name:30s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if not all(checks.values()):

        raise AssertionError(
            "Cost-adjusted robustness integrity failed."
        )


# ============================================================
# APPLY COST
# ============================================================

def apply_cost(
    df: pd.DataFrame,
    cost_r: float,
) -> pd.DataFrame:

    work = df.copy()

    # Only triggered trades incur the hypothetical
    # Bar-2 exit cost.
    work["cost_adjusted_delta_r"] = np.where(
        work["_state"],
        work["_gross_delta_r"] - cost_r,
        0.0,
    )

    return work


# ============================================================
# OVERALL SCENARIO
# ============================================================

def overall_scenario(
    df: pd.DataFrame,
    cost_r: float,
) -> Dict[str, float]:

    vals = df[
        "cost_adjusted_delta_r"
    ]

    summary = summarize_values(
        vals
    )

    summary["cost_r"] = cost_r

    return summary


# ============================================================
# EVENT / RETAINED
# ============================================================

def event_retained(
    df: pd.DataFrame,
    cost_r: float,
) -> pd.DataFrame:

    rows = []

    for label, mask in [
        (
            "event",
            df["_event"],
        ),
        (
            "retained",
            ~df["_event"],
        ),
    ]:

        summary = summarize_values(
            df.loc[
                mask,
                "cost_adjusted_delta_r",
            ]
        )

        rows.append(
            {
                "cost_r": cost_r,
                "population": label,
                **summary,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUBGROUP CONTRIBUTION
# ============================================================

def subgroup_contribution(
    df: pd.DataFrame,
    dimension: str,
    column: str,
    cost_r: float,
) -> pd.DataFrame:

    rows = []

    for group, sub in df.groupby(
        column,
        dropna=False,
    ):

        vals = pd.to_numeric(
            sub["cost_adjusted_delta_r"],
            errors="coerce",
        ).dropna()

        if len(vals) < MIN_SUBGROUP_TRADES:
            continue

        rows.append(
            {
                "cost_r": cost_r,
                "dimension": dimension,
                "group": str(group),
                "n": int(len(vals)),
                "total_delta_r": float(
                    vals.sum()
                ),
                "mean_delta_r": float(
                    vals.mean()
                ),
                "median_delta_r": float(
                    vals.median()
                ),
                "positive": int(
                    (vals > 1e-12).sum()
                ),
                "negative": int(
                    (vals < -1e-12).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LEAVE-ONE-OUT
# ============================================================

def leave_one_out(
    df: pd.DataFrame,
    dimension: str,
    column: str,
    cost_r: float,
) -> pd.DataFrame:

    rows = []

    full_total = float(
        df["cost_adjusted_delta_r"].sum()
    )

    groups = (
        df[column]
        .dropna()
        .unique()
    )

    for group in groups:

        mask = df[column] != group

        remaining = df.loc[
            mask,
            "cost_adjusted_delta_r",
        ].dropna()

        if len(remaining) < MIN_SUBGROUP_TRADES:
            continue

        total = float(
            remaining.sum()
        )

        rows.append(
            {
                "cost_r": cost_r,
                "dimension": dimension,
                "removed_group": str(group),
                "removed_delta_r": (
                    full_total - total
                ),
                "remaining_n": int(
                    len(remaining)
                ),
                "remaining_total_delta_r": total,
                "remaining_mean_delta_r": float(
                    remaining.mean()
                ),
                "remaining_median_delta_r": float(
                    remaining.median()
                ),
                "remaining_positive": int(
                    (remaining > 1e-12).sum()
                ),
                "remaining_negative": int(
                    (remaining < -1e-12).sum()
                ),
                "remaining_positive_total": (
                    total > 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# TOP-N CONCENTRATION
# ============================================================

def top_n_concentration(
    df: pd.DataFrame,
    cost_r: float,
) -> pd.DataFrame:

    vals = (
        df["cost_adjusted_delta_r"]
        .sort_values(
            ascending=False
        )
        .reset_index(drop=True)
    )

    total = float(
        vals.sum()
    )

    rows = []

    for n in [1, 3, 5, 10]:

        if len(vals) == 0:
            continue

        top_total = float(
            vals.head(
                min(n, len(vals))
            ).sum()
        )

        rows.append(
            {
                "cost_r": cost_r,
                "top_n": n,
                "top_n_delta_r": top_total,
                "total_delta_r": total,
                "contribution_pct": (
                    top_total / total * 100
                    if total != 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REMOVE TOP CONTRIBUTORS
# ============================================================

def top_removal(
    df: pd.DataFrame,
    cost_r: float,
) -> pd.DataFrame:

    ordered = (
        df[
            "cost_adjusted_delta_r"
        ]
        .sort_values(
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    for n in [1, 3, 5, 10]:

        if n > len(ordered):
            continue

        remaining = ordered.iloc[n:]

        rows.append(
            {
                "cost_r": cost_r,
                "removed_top_n": n,
                "removed_delta_r": float(
                    ordered.head(n).sum()
                ),
                "remaining_n": int(
                    len(remaining)
                ),
                "remaining_total_delta_r": float(
                    remaining.sum()
                ),
                "remaining_mean_delta_r": float(
                    remaining.mean()
                ),
                "remaining_positive": int(
                    (remaining > 1e-12).sum()
                ),
                "remaining_negative": int(
                    (remaining < -1e-12).sum()
                ),
                "remaining_still_positive": (
                    float(remaining.sum()) > 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_data()

    columns = resolve_columns(
        raw
    )

    df = prepare(
        raw,
        columns,
    )

    integrity_check(
        df
    )

    print("=" * 72)
    print("FIXED FORENSIC STATE")
    print("=" * 72)
    print()
    print(
        "bar2_favorable_failed_to_improve"
    )
    print(
        f"Triggered trades: "
        f"{int(df['_state'].sum())}"
    )
    print()

    # --------------------------------------------------------
    # COST SCENARIOS
    # --------------------------------------------------------

    scenario_rows = []
    event_rows = []

    subgroup_frames = []
    loo_frames = []
    concentration_frames = []
    removal_frames = []

    for cost_r in COST_LEVELS_R:

        scenario = apply_cost(
            df,
            cost_r,
        )

        # Overall
        scenario_rows.append(
            overall_scenario(
                scenario,
                cost_r,
            )
        )

        # Event / retained
        event_rows.append(
            event_retained(
                scenario,
                cost_r,
            )
        )

        # Subgroups
        subgroup_frames.extend(
            [
                subgroup_contribution(
                    scenario,
                    "year",
                    "_year",
                    cost_r,
                ),
                subgroup_contribution(
                    scenario,
                    "symbol",
                    "_symbol",
                    cost_r,
                ),
                subgroup_contribution(
                    scenario,
                    "setup",
                    "_setup",
                    cost_r,
                ),
                subgroup_contribution(
                    scenario,
                    "direction",
                    "_direction",
                    cost_r,
                ),
            ]
        )

        # LOO
        loo_frames.extend(
            [
                leave_one_out(
                    scenario,
                    "year",
                    "_year",
                    cost_r,
                ),
                leave_one_out(
                    scenario,
                    "symbol",
                    "_symbol",
                    cost_r,
                ),
            ]
        )

        # Concentration
        concentration_frames.append(
            top_n_concentration(
                scenario,
                cost_r,
            )
        )

        # Largest contributor removal
        removal_frames.append(
            top_removal(
                scenario,
                cost_r,
            )
        )

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    event_df = pd.concat(
        event_rows,
        ignore_index=True,
    )

    subgroup_df = pd.concat(
        subgroup_frames,
        ignore_index=True,
    )

    loo_df = pd.concat(
        loo_frames,
        ignore_index=True,
    )

    concentration_df = pd.concat(
        concentration_frames,
        ignore_index=True,
    )

    removal_df = pd.concat(
        removal_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # PRINT OVERALL
    # --------------------------------------------------------

    print("=" * 72)
    print("COST-ADJUSTED OVERALL RESULTS")
    print("=" * 72)
    print()

    print(
        scenario_df[
            [
                "cost_r",
                "n",
                "total_r",
                "mean_r",
                "median_r",
                "ci_low",
                "ci_high",
                "positive",
                "negative",
                "zero",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # EVENT / RETAINED
    # --------------------------------------------------------

    print("=" * 72)
    print("EVENT VS RETAINED")
    print("=" * 72)
    print()

    print(
        event_df[
            [
                "cost_r",
                "population",
                "n",
                "total_r",
                "mean_r",
                "median_r",
                "positive",
                "negative",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # SUBGROUPS
    # --------------------------------------------------------

    print("=" * 72)
    print("COST-ADJUSTED SUBGROUP CONTRIBUTION")
    print("=" * 72)
    print()

    print(
        subgroup_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # LOO
    # --------------------------------------------------------

    print("=" * 72)
    print("LEAVE-ONE-OUT ROBUSTNESS")
    print("=" * 72)
    print()

    for dimension in [
        "year",
        "symbol",
    ]:

        sub = loo_df[
            loo_df["dimension"]
            == dimension
        ]

        if sub.empty:
            continue

        print(
            f"{dimension.upper()}:"
        )

        for cost in COST_LEVELS_R:

            cost_sub = sub[
                sub["cost_r"] == cost
            ]

            if cost_sub.empty:
                continue

            positive_count = int(
                (
                    cost_sub[
                        "remaining_total_delta_r"
                    ] > 0
                ).sum()
            )

            total_count = len(
                cost_sub
            )

            min_total = float(
                cost_sub[
                    "remaining_total_delta_r"
                ].min()
            )

            max_total = float(
                cost_sub[
                    "remaining_total_delta_r"
                ].max()
            )

            print(
                f"  Cost {cost:.2f}R: "
                f"{positive_count}/{total_count} "
                f"positive | "
                f"range "
                f"{min_total:.4f}R → "
                f"{max_total:.4f}R"
            )

        print()

    # --------------------------------------------------------
    # CONCENTRATION
    # --------------------------------------------------------

    print("=" * 72)
    print("TOP-N CONCENTRATION")
    print("=" * 72)
    print()

    print(
        concentration_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # TOP REMOVAL
    # --------------------------------------------------------

    print("=" * 72)
    print("LARGEST POSITIVE CONTRIBUTOR REMOVAL")
    print("=" * 72)
    print()

    print(
        removal_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    scenario_path = (
        OUTPUT_DIR
        / "cost_robustness_scenarios.csv"
    )

    event_path = (
        OUTPUT_DIR
        / "cost_robustness_event_retained.csv"
    )

    subgroup_path = (
        OUTPUT_DIR
        / "cost_robustness_subgroups.csv"
    )

    loo_path = (
        OUTPUT_DIR
        / "cost_robustness_leave_one_out.csv"
    )

    concentration_path = (
        OUTPUT_DIR
        / "cost_robustness_concentration.csv"
    )

    removal_path = (
        OUTPUT_DIR
        / "cost_robustness_top_removal.csv"
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False,
    )

    loo_df.to_csv(
        loo_path,
        index=False,
    )

    concentration_df.to_csv(
        concentration_path,
        index=False,
    )

    removal_df.to_csv(
        removal_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL INTEGRITY
    # --------------------------------------------------------

    integrity = [
        {
            "check": "trade_count",
            "expected": EXPECTED_TRADES,
            "actual": len(df),
            "pass": (
                len(df)
                == EXPECTED_TRADES
            ),
        },
        {
            "check": "event_count",
            "expected": EXPECTED_EVENTS,
            "actual": int(
                df["_event"].sum()
            ),
            "pass": (
                int(df["_event"].sum())
                == EXPECTED_EVENTS
            ),
        },
        {
            "check": "retained_count",
            "expected": EXPECTED_RETAINED,
            "actual": int(
                (~df["_event"]).sum()
            ),
            "pass": (
                int((~df["_event"]).sum())
                == EXPECTED_RETAINED
            ),
        },
        {
            "check": "cost_scenario_count",
            "expected": len(
                COST_LEVELS_R
            ),
            "actual": len(
                scenario_df
            ),
            "pass": (
                len(scenario_df)
                == len(COST_LEVELS_R)
            ),
        },
        {
            "check": "event_retained_rows",
            "expected": len(
                COST_LEVELS_R
            ) * 2,
            "actual": len(
                event_df
            ),
            "pass": (
                len(event_df)
                == len(COST_LEVELS_R) * 2
            ),
        },
        {
            "check": "subgroup_output",
            "expected": True,
            "actual": not subgroup_df.empty,
            "pass": not subgroup_df.empty,
        },
        {
            "check": "loo_output",
            "expected": True,
            "actual": not loo_df.empty,
            "pass": not loo_df.empty,
        },
        {
            "check": "concentration_output",
            "expected": True,
            "actual": (
                not concentration_df.empty
            ),
            "pass": (
                not concentration_df.empty
            ),
        },
        {
            "check": "top_removal_output",
            "expected": True,
            "actual": not removal_df.empty,
            "pass": not removal_df.empty,
        },
    ]

    integrity_df = pd.DataFrame(
        integrity
    )

    integrity_path = (
        OUTPUT_DIR
        / "cost_robustness_integrity.csv"
    )

    integrity_df.to_csv(
        integrity_path,
        index=False,
    )

    print("=" * 72)
    print("FINAL INTEGRITY")
    print("=" * 72)
    print()

    print(
        integrity_df.to_string(
            index=False
        )
    )

    print()

    if not bool(
        integrity_df["pass"].all()
    ):
        raise AssertionError(
            "Final cost-adjusted robustness "
            "integrity failed."
        )

    print("=" * 72)
    print(
        "PASS — COST-ADJUSTED ROBUSTNESS COMPLETE"
    )
    print("=" * 72)
    print()
    print(
        "No production logic modified."
    )
    print(
        "No OOS data modified."
    )
    print(
        "No strategy rule promoted."
    )
    print()


if __name__ == "__main__":
    main()