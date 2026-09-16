"""
TradeSense-AI — Bar 1→2 Execution-Cost Sensitivity Forensic Analysis

RESEARCH ONLY.

Purpose
-------
Stress-test the previously defined Bar-2 favorable-failure counterfactual
under fixed hypothetical exit-cost assumptions expressed in R.

This script DOES NOT:
- modify production logic
- modify OOS data
- optimize a threshold
- search for a new state
- change the frozen 5-bar / 0.25R event definition
- claim that any tested cost is the actual market cost

Frozen population
-----------------
106 total trades
60 frozen early-adverse events
46 retained trades

Primary state
-------------
bar2_favorable_failed_to_improve

Counterfactual logic
--------------------
For trades where the fixed Bar-2 state is TRUE:

    adjusted_counterfactual_R = bar2_close_R - exit_cost_R

For trades where the state is FALSE:

    adjusted_counterfactual_R = original production R

Therefore:

    adjusted_delta_R
        = adjusted_counterfactual_R - original_production_R

Cost stress levels
------------------
0.00R
0.02R
0.05R
0.10R
0.15R
0.20R

The analysis reports:
- total improvement
- mean improvement
- median improvement
- bootstrap confidence intervals
- positive / negative / zero trade counts
- event vs retained contribution
- year / symbol / setup / direction contribution
- top-10 concentration
- break-even cost per triggered trade

No strategy rule is promoted by this script.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# ============================================================
# CONFIGURATION
# ============================================================

COUNTERFACTUAL_FILE = Path(
    "tests/output/research/early_adverse_bar12_counterfactual/"
    "bar12_counterfactual_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_execution_cost"
)

COST_LEVELS_R = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
]

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

PRIMARY_STATE = "bar2_favorable_failed_to_improve"

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42

MIN_SUBGROUP_TRADES = 3


# ============================================================
# HELPERS
# ============================================================

def first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not resolve any of these columns: {candidates}\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (float("nan"), float("nan"))

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


def bootstrap_total_ci(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    totals = samples.sum(axis=1)

    return (
        float(np.percentile(totals, 2.5)),
        float(np.percentile(totals, 97.5)),
    )


def fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


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


def contribution_table(
    df: pd.DataFrame,
    delta_col: str,
    group_col: str,
) -> pd.DataFrame:

    rows = []

    for group, sub in df.groupby(group_col, dropna=False):

        vals = pd.to_numeric(
            sub[delta_col],
            errors="coerce",
        ).dropna()

        if len(vals) < MIN_SUBGROUP_TRADES:
            continue

        total = float(vals.sum())

        rows.append(
            {
                "group": str(group),
                "n": int(len(vals)),
                "total_delta_r": total,
                "mean_delta_r": float(vals.mean()),
                "median_delta_r": float(vals.median()),
            }
        )

    out = pd.DataFrame(rows)

    if len(out):
        grand_total = float(df[delta_col].sum())

        if grand_total != 0:
            out["contribution_pct"] = (
                out["total_delta_r"] / grand_total * 100.0
            )
        else:
            out["contribution_pct"] = np.nan

        out = out.sort_values(
            "total_delta_r",
            ascending=False,
        ).reset_index(drop=True)

    return out


# ============================================================
# LOAD
# ============================================================

def load_data() -> pd.DataFrame:

    if not COUNTERFACTUAL_FILE.exists():
        raise FileNotFoundError(
            f"Counterfactual file not found:\n"
            f"{COUNTERFACTUAL_FILE}"
        )

    df = pd.read_csv(COUNTERFACTUAL_FILE)

    print("=" * 72)
    print("TRADE SENSE-AI — BAR 1→2 EXECUTION-COST SENSITIVITY")
    print("=" * 72)
    print()
    print(f"Input file: {COUNTERFACTUAL_FILE}")
    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")
    print()

    return df


# ============================================================
# COLUMN RESOLUTION
# ============================================================

def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:

    resolved = {}

    resolved["state"] = first_existing(
        df,
        [
            PRIMARY_STATE,
            "state",
            "triggered",
            "counterfactual_triggered",
        ],
    )

    resolved["original_r"] = first_existing(
        df,
        [
            "original_final_r",
            "original_r",
            "original_R",
            "production_r",
            "production_R",
            "r_multiple",
            "final_r",
        ],
    )

    resolved["bar2_close_r"] = first_existing(
        df,
        [
            "bar2_close_r",
            "bar2_close_R",
            "counterfactual_r",
            "bar2_counterfactual_r",
        ],
    )

    resolved["event"] = first_existing(
        df,
        [
            "frozen_event",
            "event",
            "early_adverse_event",
        ],
    )

    resolved["symbol"] = first_existing(
        df,
        [
            "symbol",
            "_symbol",
            "Symbol",
        ],
    )

    resolved["direction"] = first_existing(
        df,
        [
            "direction",
            "Direction",
        ],
    )

    resolved["setup"] = first_existing(
        df,
        [
            "setup",
            "Setup",
        ],
    )

    resolved["entry_date"] = first_existing(
        df,
        [
            "entry_date",
            "Entry Date",
        ],
    )

    print("Resolved columns:")
    for key, value in resolved.items():
        print(f"  {key:15s}: {value}")

    print()

    return resolved


# ============================================================
# PREPARE FROZEN DATA
# ============================================================

def prepare(df: pd.DataFrame, c: Dict[str, str]) -> pd.DataFrame:

    work = df.copy()

    work["_state"] = state_to_bool(work[c["state"]])

    work["_original_r"] = numeric_series(
        work,
        c["original_r"],
    )

    work["_bar2_close_r"] = numeric_series(
        work,
        c["bar2_close_r"],
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

    work["_year"] = work["_entry_date"].dt.year

    required = [
        "_state",
        "_original_r",
        "_bar2_close_r",
        "_event",
        "_symbol",
        "_direction",
        "_setup",
        "_entry_date",
    ]

    missing = [
        col
        for col in required
        if work[col].isna().any()
    ]

    if missing:
        print("WARNING: missing values detected in:")
        for col in missing:
            print(
                f"  {col}: "
                f"{int(work[col].isna().sum())}"
            )
        print()

    return work


# ============================================================
# INTEGRITY
# ============================================================

def integrity_check(df: pd.DataFrame) -> Dict[str, object]:

    results = {}

    trade_count = len(df)
    event_count = int(df["_event"].sum())
    retained_count = trade_count - event_count

    results["trade_count"] = (
        trade_count == EXPECTED_TRADES
    )

    results["event_count"] = (
        event_count == EXPECTED_EVENTS
    )

    results["retained_count"] = (
        retained_count == EXPECTED_RETAINED
    )

    results["state_nonempty"] = (
        int(df["_state"].sum()) > 0
    )

    results["original_nonempty"] = (
        df["_original_r"].notna().all()
    )

    results["bar2_nonempty_for_state"] = (
        df.loc[df["_state"], "_bar2_close_r"]
        .notna()
        .all()
    )

    print("=" * 72)
    print("INTEGRITY CHECK")
    print("=" * 72)
    print()
    print(f"Trade count:       {trade_count}")
    print(f"Frozen events:     {event_count}")
    print(f"Frozen retained:   {retained_count}")
    print(
        f"Triggered trades:  "
        f"{int(df['_state'].sum())}"
    )
    print()

    for key, passed in results.items():
        print(
            f"{key:30s}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if not all(results.values()):
        raise AssertionError(
            "Execution-cost integrity check failed."
        )

    return results


# ============================================================
# COST SCENARIO
# ============================================================

def calculate_cost_scenario(
    df: pd.DataFrame,
    cost_r: float,
) -> pd.DataFrame:

    work = df.copy()

    work["adjusted_counterfactual_r"] = np.where(
        work["_state"],
        work["_bar2_close_r"] - cost_r,
        work["_original_r"],
    )

    work["adjusted_delta_r"] = (
        work["adjusted_counterfactual_r"]
        - work["_original_r"]
    )

    return work


# ============================================================
# SCENARIO SUMMARY
# ============================================================

def scenario_summary(
    df: pd.DataFrame,
    cost_r: float,
) -> Dict[str, object]:

    vals = (
        pd.to_numeric(
            df["adjusted_delta_r"],
            errors="coerce",
        )
        .dropna()
        .to_numpy()
    )

    state_vals = (
        df.loc[df["_state"], "adjusted_delta_r"]
        .dropna()
        .to_numpy()
    )

    event_vals = (
        df.loc[df["_event"], "adjusted_delta_r"]
        .dropna()
        .to_numpy()
    )

    retained_vals = (
        df.loc[~df["_event"], "adjusted_delta_r"]
        .dropna()
        .to_numpy()
    )

    positive = int(np.sum(vals > 1e-12))
    negative = int(np.sum(vals < -1e-12))
    zero = int(np.sum(np.abs(vals) <= 1e-12))

    mean_ci = bootstrap_mean_ci(vals)
    total_ci = bootstrap_total_ci(vals)

    state_total = float(state_vals.sum())

    return {
        "cost_r": cost_r,
        "n": len(vals),
        "total_delta_r": float(vals.sum()),
        "mean_delta_r": float(vals.mean()),
        "median_delta_r": float(np.median(vals)),
        "bootstrap_mean_ci_low": mean_ci[0],
        "bootstrap_mean_ci_high": mean_ci[1],
        "bootstrap_total_ci_low": total_ci[0],
        "bootstrap_total_ci_high": total_ci[1],
        "positive_trades": positive,
        "negative_trades": negative,
        "zero_trades": zero,
        "triggered_n": len(state_vals),
        "triggered_total_delta_r": state_total,
        "triggered_mean_delta_r": (
            float(state_vals.mean())
            if len(state_vals)
            else np.nan
        ),
        "event_n": len(event_vals),
        "event_total_delta_r": float(event_vals.sum()),
        "event_mean_delta_r": (
            float(event_vals.mean())
            if len(event_vals)
            else np.nan
        ),
        "retained_n": len(retained_vals),
        "retained_total_delta_r": float(
            retained_vals.sum()
        ),
        "retained_mean_delta_r": (
            float(retained_vals.mean())
            if len(retained_vals)
            else np.nan
        ),
    }


# ============================================================
# TRIGGERED-TRADE COST ECONOMICS
# ============================================================

def triggered_cost_analysis(
    df: pd.DataFrame,
) -> Dict[str, float]:

    triggered = df.loc[df["_state"]].copy()

    gross = (
        triggered["_bar2_close_r"]
        - triggered["_original_r"]
    )

    gross = pd.to_numeric(
        gross,
        errors="coerce",
    ).dropna()

    n = len(gross)

    if n == 0:
        return {
            "triggered_n": 0,
            "gross_total": np.nan,
            "gross_mean": np.nan,
            "break_even_mean_cost_r": np.nan,
        }

    gross_total = float(gross.sum())
    gross_mean = float(gross.mean())

    return {
        "triggered_n": n,
        "gross_total": gross_total,
        "gross_mean": gross_mean,
        "break_even_mean_cost_r": gross_mean,
    }


# ============================================================
# TOP-N CONCENTRATION
# ============================================================

def concentration_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    vals = (
        df["adjusted_delta_r"]
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )

    total = float(vals.sum())

    rows = []

    for n in [1, 3, 5, 10]:

        if len(vals) == 0:
            continue

        top_n = float(
            vals.head(min(n, len(vals))).sum()
        )

        rows.append(
            {
                "top_n": n,
                "top_n_delta_r": top_n,
                "total_delta_r": total,
                "contribution_pct": (
                    top_n / total * 100.0
                    if total != 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# YEAR / SYMBOL / SETUP / DIRECTION
# ============================================================

def subgroup_outputs(
    df: pd.DataFrame,
    cost_r: float,
) -> Dict[str, pd.DataFrame]:

    outputs = {}

    outputs["year"] = contribution_table(
        df,
        "adjusted_delta_r",
        "_year",
    )

    outputs["symbol"] = contribution_table(
        df,
        "adjusted_delta_r",
        "_symbol",
    )

    outputs["setup"] = contribution_table(
        df,
        "adjusted_delta_r",
        "_setup",
    )

    outputs["direction"] = contribution_table(
        df,
        "adjusted_delta_r",
        "_direction",
    )

    return outputs


# ============================================================
# COST BREAKEVEN TABLE
# ============================================================

def build_break_even_table(
    gross_triggered_delta: float,
    triggered_n: int,
    costs: List[float],
) -> pd.DataFrame:

    rows = []

    for cost in costs:

        cost_drag = cost * triggered_n

        adjusted_total = (
            gross_triggered_delta
            - cost_drag
        )

        adjusted_mean = (
            adjusted_total / triggered_n
            if triggered_n
            else np.nan
        )

        rows.append(
            {
                "cost_r_per_trigger": cost,
                "triggered_n": triggered_n,
                "gross_triggered_delta_r": (
                    gross_triggered_delta
                ),
                "cost_drag_r": cost_drag,
                "adjusted_triggered_delta_r": (
                    adjusted_total
                ),
                "adjusted_triggered_mean_r": (
                    adjusted_mean
                ),
                "still_positive": (
                    adjusted_total > 0
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

    df_raw = load_data()

    columns = resolve_columns(df_raw)

    df = prepare(
        df_raw,
        columns,
    )

    integrity_check(df)

    print("=" * 72)
    print("PRIMARY STATE")
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

    scenario_data = {}

    for cost in COST_LEVELS_R:

        scenario = calculate_cost_scenario(
            df,
            cost,
        )

        scenario_data[cost] = scenario

        summary = scenario_summary(
            scenario,
            cost,
        )

        scenario_rows.append(summary)

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    print("=" * 72)
    print("EXECUTION-COST SENSITIVITY")
    print("=" * 72)
    print()

    display_cols = [
        "cost_r",
        "total_delta_r",
        "mean_delta_r",
        "median_delta_r",
        "bootstrap_mean_ci_low",
        "bootstrap_mean_ci_high",
        "positive_trades",
        "negative_trades",
        "event_total_delta_r",
        "retained_total_delta_r",
    ]

    print(
        scenario_df[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # BREAK-EVEN COST
    # --------------------------------------------------------

    gross_info = triggered_cost_analysis(df)

    break_even_df = build_break_even_table(
        gross_triggered_delta=gross_info[
            "gross_total"
        ],
        triggered_n=int(
            gross_info["triggered_n"]
        ),
        costs=COST_LEVELS_R,
    )

    print("=" * 72)
    print("TRIGGERED-TRADE COST BREAK-EVEN")
    print("=" * 72)
    print()
    print(
        f"Triggered trades: "
        f"{gross_info['triggered_n']}"
    )
    print(
        f"Gross triggered improvement: "
        f"{gross_info['gross_total']:.4f}R"
    )
    print(
        f"Gross improvement per triggered trade: "
        f"{gross_info['gross_mean']:.4f}R"
    )
    print(
        f"Break-even average cost per triggered trade: "
        f"{gross_info['break_even_mean_cost_r']:.4f}R"
    )
    print()

    print(
        break_even_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()

    # --------------------------------------------------------
    # SUBGROUPS
    # --------------------------------------------------------

    subgroup_records = []

    for cost in COST_LEVELS_R:

        scenario = scenario_data[cost]

        groups = subgroup_outputs(
            scenario,
            cost,
        )

        for dimension, table in groups.items():

            if table.empty:
                continue

            table = table.copy()

            table.insert(
                0,
                "cost_r",
                cost,
            )

            table.insert(
                1,
                "dimension",
                dimension,
            )

            subgroup_records.append(
                table
            )

    if subgroup_records:
        subgroup_df = pd.concat(
            subgroup_records,
            ignore_index=True,
        )
    else:
        subgroup_df = pd.DataFrame()

    # --------------------------------------------------------
    # TOP-N CONCENTRATION
    # --------------------------------------------------------

    concentration_records = []

    for cost in COST_LEVELS_R:

        scenario = scenario_data[cost]

        table = concentration_table(
            scenario
        )

        table.insert(
            0,
            "cost_r",
            cost,
        )

        concentration_records.append(
            table
        )

    concentration_df = pd.concat(
        concentration_records,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # EVENT / RETAINED
    # --------------------------------------------------------

    event_rows = []

    for cost in COST_LEVELS_R:

        scenario = scenario_data[cost]

        for label, mask in [
            ("event", scenario["_event"]),
            ("retained", ~scenario["_event"]),
        ]:

            vals = (
                scenario.loc[
                    mask,
                    "adjusted_delta_r",
                ]
                .dropna()
                .to_numpy()
            )

            event_rows.append(
                {
                    "cost_r": cost,
                    "population": label,
                    "n": len(vals),
                    "total_delta_r": (
                        float(vals.sum())
                        if len(vals)
                        else np.nan
                    ),
                    "mean_delta_r": (
                        float(vals.mean())
                        if len(vals)
                        else np.nan
                    ),
                    "median_delta_r": (
                        float(np.median(vals))
                        if len(vals)
                        else np.nan
                    ),
                }
            )

    event_df = pd.DataFrame(
        event_rows
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    scenario_path = (
        OUTPUT_DIR
        / "execution_cost_scenarios.csv"
    )

    break_even_path = (
        OUTPUT_DIR
        / "execution_cost_break_even.csv"
    )

    subgroup_path = (
        OUTPUT_DIR
        / "execution_cost_subgroups.csv"
    )

    concentration_path = (
        OUTPUT_DIR
        / "execution_cost_concentration.csv"
    )

    event_path = (
        OUTPUT_DIR
        / "execution_cost_event_retained.csv"
    )

    scenario_df.to_csv(
        scenario_path,
        index=False,
    )

    break_even_df.to_csv(
        break_even_path,
        index=False,
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False,
    )

    concentration_df.to_csv(
        concentration_path,
        index=False,
    )

    event_df.to_csv(
        event_path,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL INTEGRITY
    # --------------------------------------------------------

    integrity_rows = [
        {
            "check": "trade_count",
            "expected": EXPECTED_TRADES,
            "actual": len(df),
            "pass": len(df) == EXPECTED_TRADES,
        },
        {
            "check": "event_count",
            "expected": EXPECTED_EVENTS,
            "actual": int(df["_event"].sum()),
            "pass": int(df["_event"].sum())
            == EXPECTED_EVENTS,
        },
        {
            "check": "retained_count",
            "expected": EXPECTED_RETAINED,
            "actual": int((~df["_event"]).sum()),
            "pass": int((~df["_event"]).sum())
            == EXPECTED_RETAINED,
        },
        {
            "check": "cost_scenarios",
            "expected": len(COST_LEVELS_R),
            "actual": len(scenario_df),
            "pass": len(scenario_df)
            == len(COST_LEVELS_R),
        },
        {
            "check": "break_even_rows",
            "expected": len(COST_LEVELS_R),
            "actual": len(break_even_df),
            "pass": len(break_even_df)
            == len(COST_LEVELS_R),
        },
        {
            "check": "subgroup_output_nonempty",
            "expected": True,
            "actual": not subgroup_df.empty,
            "pass": not subgroup_df.empty,
        },
        {
            "check": "concentration_output_nonempty",
            "expected": True,
            "actual": not concentration_df.empty,
            "pass": not concentration_df.empty,
        },
        {
            "check": "event_retained_output_nonempty",
            "expected": True,
            "actual": not event_df.empty,
            "pass": not event_df.empty,
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    integrity_path = (
        OUTPUT_DIR
        / "execution_cost_integrity.csv"
    )

    integrity_df.to_csv(
        integrity_path,
        index=False,
    )

    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)
    print()
    print(scenario_path)
    print(break_even_path)
    print(subgroup_path)
    print(concentration_path)
    print(event_path)
    print(integrity_path)
    print()

    print("=" * 72)
    print("FINAL INTEGRITY")
    print("=" * 72)
    print()

    print(
        integrity_df[
            [
                "check",
                "expected",
                "actual",
                "pass",
            ]
        ].to_string(index=False)
    )

    print()

    if not bool(integrity_df["pass"].all()):
        raise AssertionError(
            "Final execution-cost integrity failed."
        )

    print("=" * 72)
    print("PASS — EXECUTION-COST SENSITIVITY COMPLETE")
    print("=" * 72)
    print()
    print(
        "No production logic modified."
    )
    print(
        "No OOS data modified."
    )
    print(
        "No deployment rule promoted."
    )
    print()


if __name__ == "__main__":
    main()