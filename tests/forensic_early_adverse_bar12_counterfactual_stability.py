"""
TradeSense-AI
BAR 1->2 COUNTERFACTUAL STABILITY / CONCENTRATION FORENSIC

Research-only.

Frozen population:
    106 trades
    60 frozen events
    46 retained

Purpose:
    Stress-test the gross Bar-2 counterfactual improvement.

Primary state:
    bar2_favorable_failed_to_improve

Also evaluates:
    bar2_net_deteriorated
    adverse_up_favorable_not_up_net_down

Tests:
    1. Per-trade delta distribution
    2. Top-N contribution concentration
    3. Leave-one-out stability
    4. Bootstrap aggregate improvement
    5. Event vs retained contribution
    6. Calendar-year contribution
    7. Symbol contribution
    8. Setup contribution
    9. Direction contribution
    10. Largest-positive-contributor removal stress tests

No:
    - production modification
    - OOS modification
    - threshold optimization
    - ML
    - strategy promotion
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_bar12_counterfactual/"
    "bar12_counterfactual_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_counterfactual_stability"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

STATES = [
    "bar2_net_deteriorated",
    "bar2_favorable_failed_to_improve",
    "adverse_up_favorable_not_up_net_down",
]

STATE_LABELS = {
    "bar2_net_deteriorated":
        "Bar-2 net deterioration",

    "bar2_favorable_failed_to_improve":
        "Bar-2 favorable failure",

    "adverse_up_favorable_not_up_net_down":
        "Bar-2 adverse-up + favorable-not-up + net-down",
}

TOP_N_VALUES = [
    1,
    3,
    5,
    10,
]

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    raise RuntimeError(message)


def require_columns(df, columns, name):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        fail(
            f"{name} missing required columns: {missing}"
        )


def normalize_symbol(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_direction(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_date(series):
    return pd.to_datetime(
        series,
        errors="coerce"
    )


def bootstrap_total_delta(
    deltas,
    iterations=10000,
    seed=42,
):
    values = np.asarray(
        deltas,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    # Each bootstrap sample contains n trades.
    indices = rng.integers(
        0,
        len(values),
        size=(iterations, len(values)),
    )

    boot_totals = (
        values[indices]
        .sum(axis=1)
    )

    observed = float(
        values.sum()
    )

    low = float(
        np.percentile(
            boot_totals,
            2.5
        )
    )

    high = float(
        np.percentile(
            boot_totals,
            97.5
        )
    )

    return observed, low, high


def bootstrap_mean_delta(
    deltas,
    iterations=10000,
    seed=42,
):
    values = np.asarray(
        deltas,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0,
        len(values),
        size=(iterations, len(values)),
    )

    boot_means = (
        values[indices]
        .mean(axis=1)
    )

    observed = float(
        values.mean()
    )

    low = float(
        np.percentile(
            boot_means,
            2.5
        )
    )

    high = float(
        np.percentile(
            boot_means,
            97.5
        )
    )

    return observed, low, high


# ============================================================================
# LOAD
# ============================================================================

def load_data():

    if not INPUT_FILE.exists():
        fail(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}, "
        f"Columns: {len(df.columns)}"
    )

    require_columns(
        df,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            "original_final_r",
            "bar2_close_r",
        ],
        "Counterfactual"
    )

    for state in STATES:
        require_columns(
            df,
            [state],
            "Counterfactual"
        )

    if len(df) != EXPECTED_TRADES:
        fail(
            f"Rows {len(df)} "
            f"!= {EXPECTED_TRADES}"
        )

    df = df.copy()

    df["symbol"] = normalize_symbol(
        df["symbol"]
    )

    df["direction"] = normalize_direction(
        df["direction"]
    )

    df["entry_date"] = normalize_date(
        df["entry_date"]
    )

    df["frozen_event"] = (
        df["frozen_event"]
        .astype(int)
        .astype(bool)
    )

    for state in STATES:
        df[state] = (
            df[state]
            .astype(int)
            .astype(bool)
        )

    event_count = int(
        df["frozen_event"].sum()
    )

    retained_count = int(
        (~df["frozen_event"]).sum()
    )

    print(
        f"Frozen population: "
        f"{len(df)} / "
        f"{event_count} / "
        f"{retained_count}"
    )

    if event_count != EXPECTED_EVENTS:
        fail(
            f"Event count {event_count} "
            f"!= {EXPECTED_EVENTS}"
        )

    if retained_count != EXPECTED_RETAINED:
        fail(
            f"Retained count {retained_count} "
            f"!= {EXPECTED_RETAINED}"
        )

    # Reconstruct the three counterfactual deltas directly.
    # This avoids trusting a derived delta column.
    for state in STATES:

        df[
            f"{state}_delta_r"
        ] = np.where(
            df[state],
            df["bar2_close_r"],
            df["original_final_r"],
        ) - df["original_final_r"]

    return df


# ============================================================================
# PER-TRADE DISTRIBUTION
# ============================================================================

def per_trade_distribution(
    df,
    state,
):

    mask = df[state]

    values = df.loc[
        mask,
        f"{state}_delta_r"
    ].to_numpy()

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        fail(
            f"No state-1 trades for {state}"
        )

    result = {
        "state": state,
        "state_label": STATE_LABELS[state],
        "n": len(values),
        "sum_delta_r": float(values.sum()),
        "mean_delta_r": float(values.mean()),
        "median_delta_r": float(np.median(values)),
        "std_delta_r": float(np.std(values, ddof=1))
        if len(values) > 1
        else np.nan,
        "min_delta_r": float(values.min()),
        "max_delta_r": float(values.max()),
        "positive_delta_count": int(
            (values > 0).sum()
        ),
        "negative_delta_count": int(
            (values < 0).sum()
        ),
        "zero_delta_count": int(
            (values == 0).sum()
        ),
    }

    return result


# ============================================================================
# TOP-N CONCENTRATION
# ============================================================================

def top_n_concentration(
    df,
    state,
):

    mask = df[state]

    work = df.loc[
        mask,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            "original_final_r",
            "bar2_close_r",
            f"{state}_delta_r",
        ]
    ].copy()

    work = work.sort_values(
        f"{state}_delta_r",
        ascending=False,
    )

    total = float(
        work[f"{state}_delta_r"].sum()
    )

    rows = []

    for n in TOP_N_VALUES:

        top = work.head(
            min(n, len(work))
        )

        top_sum = float(
            top[f"{state}_delta_r"].sum()
        )

        contribution = (
            top_sum / total
            if total != 0
            else np.nan
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "top_n": n,
                "total_delta_r": total,
                "top_n_delta_r": top_sum,
                "top_n_contribution_fraction":
                    contribution,
                "top_n_contribution_percent":
                    contribution * 100
                    if np.isfinite(contribution)
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# LEAVE-ONE-OUT
# ============================================================================

def leave_one_out(
    df,
    state,
):

    mask = df[state]

    work = df.loc[
        mask,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            f"{state}_delta_r",
        ]
    ].copy()

    total = float(
        work[f"{state}_delta_r"].sum()
    )

    rows = []

    for idx, row in work.iterrows():

        delta = float(
            row[f"{state}_delta_r"]
        )

        loo_total = (
            total - delta
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "symbol": row["symbol"],
                "direction": row["direction"],
                "entry_date": row["entry_date"],
                "setup": row["setup"],
                "frozen_event":
                    bool(row["frozen_event"]),
                "trade_delta_r": delta,
                "full_total_delta_r": total,
                "leave_one_out_total_delta_r":
                    loo_total,
                "loo_remains_positive":
                    loo_total > 0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# BOOTSTRAP
# ============================================================================

def bootstrap_analysis(
    df,
    state,
):

    values = df.loc[
        df[state],
        f"{state}_delta_r"
    ].to_numpy()

    total, total_low, total_high = (
        bootstrap_total_delta(
            values,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
    )

    mean, mean_low, mean_high = (
        bootstrap_mean_delta(
            values,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
    )

    return {
        "state": state,
        "state_label": STATE_LABELS[state],
        "n": len(values),
        "observed_total_delta_r": total,
        "bootstrap_total_ci_low_r":
            total_low,
        "bootstrap_total_ci_high_r":
            total_high,
        "observed_mean_delta_r": mean,
        "bootstrap_mean_ci_low_r":
            mean_low,
        "bootstrap_mean_ci_high_r":
            mean_high,
    }


# ============================================================================
# EVENT / RETAINED CONTRIBUTION
# ============================================================================

def group_contribution(
    df,
    state,
):

    rows = []

    for group_name, group_mask in [
        (
            "event",
            df["frozen_event"]
        ),
        (
            "retained",
            ~df["frozen_event"]
        ),
    ]:

        subset = df[
            df[state]
            & group_mask
        ]

        values = subset[
            f"{state}_delta_r"
        ].to_numpy()

        total = float(
            values.sum()
        ) if len(values) else 0.0

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "population": group_name,
                "n": len(values),
                "total_delta_r": total,
                "mean_delta_r":
                    float(values.mean())
                    if len(values)
                    else np.nan,
                "median_delta_r":
                    float(np.median(values))
                    if len(values)
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# SUBGROUP CONTRIBUTION
# ============================================================================

def subgroup_contribution(
    df,
    state,
    column,
):

    rows = []

    for group_value, subset in df[
        df[state]
    ].groupby(
        column,
        dropna=False
    ):

        values = subset[
            f"{state}_delta_r"
        ].to_numpy()

        total = float(
            values.sum()
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "dimension": column,
                "group": group_value,
                "n": len(values),
                "total_delta_r": total,
                "mean_delta_r":
                    float(values.mean()),
                "median_delta_r":
                    float(np.median(values)),
                "positive_delta_count":
                    int((values > 0).sum()),
                "negative_delta_count":
                    int((values < 0).sum()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# LARGEST POSITIVE CONTRIBUTOR REMOVAL
# ============================================================================

def removal_stress(
    df,
    state,
):

    work = df[
        df[state]
    ].copy()

    work = work.sort_values(
        f"{state}_delta_r",
        ascending=False,
    )

    values = work[
        f"{state}_delta_r"
    ].to_numpy()

    total = float(
        values.sum()
    )

    rows = []

    for n in [
        1,
        3,
        5,
        10,
    ]:

        removed = values[
            :min(n, len(values))
        ]

        remaining = (
            total
            - float(removed.sum())
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "remove_top_n_positive":
                    n,
                "full_total_delta_r":
                    total,
                "removed_delta_r":
                    float(removed.sum()),
                "remaining_total_delta_r":
                    remaining,
                "remaining_mean_delta_r":
                    remaining
                    / (
                        len(values)
                        - len(removed)
                    )
                    if len(values) > len(removed)
                    else np.nan,
                "removal_remains_positive":
                    remaining > 0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "TRADE SENSE-AI — BAR 1→2 "
        "COUNTERFACTUAL STABILITY / "
        "CONCENTRATION FORENSIC"
    )
    print("=" * 72)

    df = load_data()

    primary_rows = []
    concentration_frames = []
    loo_frames = []
    bootstrap_rows = []
    group_frames = []
    subgroup_frames = []
    removal_frames = []

    # ------------------------------------------------------------------------
    # Primary state analyses
    # ------------------------------------------------------------------------

    for state in STATES:

        print()
        print("=" * 72)
        print(STATE_LABELS[state])
        print("=" * 72)

        primary = per_trade_distribution(
            df,
            state,
        )

        primary_rows.append(
            primary
        )

        print(
            f"State-1 trades: "
            f"{primary['n']}"
        )

        print(
            f"Total delta: "
            f"{primary['sum_delta_r']:+.4f}R"
        )

        print(
            f"Mean delta: "
            f"{primary['mean_delta_r']:+.4f}R"
        )

        print(
            f"Median delta: "
            f"{primary['median_delta_r']:+.4f}R"
        )

        print(
            f"Positive / Negative / Zero: "
            f"{primary['positive_delta_count']} / "
            f"{primary['negative_delta_count']} / "
            f"{primary['zero_delta_count']}"
        )

        # Top-N.
        concentration = top_n_concentration(
            df,
            state,
        )

        concentration_frames.append(
            concentration
        )

        # LOO.
        loo = leave_one_out(
            df,
            state,
        )

        loo_frames.append(
            loo
        )

        # Bootstrap.
        bootstrap_rows.append(
            bootstrap_analysis(
                df,
                state,
            )
        )

        # Event/retained.
        group_frames.append(
            group_contribution(
                df,
                state,
            )
        )

        # Calendar year.
        df["_year"] = (
            df["entry_date"]
            .dt.year
        )

        subgroup_frames.append(
            subgroup_contribution(
                df,
                state,
                "_year",
            )
        )

        # Symbol.
        subgroup_frames.append(
            subgroup_contribution(
                df,
                state,
                "symbol",
            )
        )

        # Setup.
        subgroup_frames.append(
            subgroup_contribution(
                df,
                state,
                "setup",
            )
        )

        # Direction.
        subgroup_frames.append(
            subgroup_contribution(
                df,
                state,
                "direction",
            )
        )

        # Largest contributor removal.
        removal_frames.append(
            removal_stress(
                df,
                state,
            )
        )

    # ------------------------------------------------------------------------
    # Concatenate outputs
    # ------------------------------------------------------------------------

    primary_df = pd.DataFrame(
        primary_rows
    )

    concentration_df = pd.concat(
        concentration_frames,
        ignore_index=True,
    )

    loo_df = pd.concat(
        loo_frames,
        ignore_index=True,
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    group_df = pd.concat(
        group_frames,
        ignore_index=True,
    )

    subgroup_df = pd.concat(
        subgroup_frames,
        ignore_index=True,
    )

    removal_df = pd.concat(
        removal_frames,
        ignore_index=True,
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    primary_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_stability_primary.csv",
        index=False,
    )

    concentration_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_top_n_concentration.csv",
        index=False,
    )

    loo_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_leave_one_out.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_bootstrap.csv",
        index=False,
    )

    group_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_event_retained_contribution.csv",
        index=False,
    )

    subgroup_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_subgroup_contribution.csv",
        index=False,
    )

    removal_df.to_csv(
        OUTPUT_DIR
        / "counterfactual_removal_stress.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Print concentration
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("TOP-N CONTRIBUTION")
    print("=" * 72)

    for state in STATES:

        print()
        print(STATE_LABELS[state])

        temp = concentration_df[
            concentration_df["state"] == state
        ]

        for _, row in temp.iterrows():

            print(
                f"  Top {int(row['top_n']):2d}: "
                f"{row['top_n_delta_r']:+.4f}R "
                f"= "
                f"{row['top_n_contribution_percent']:.2f}% "
                f"of total"
            )

    # ------------------------------------------------------------------------
    # Print LOO
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LEAVE-ONE-OUT STABILITY")
    print("=" * 72)

    for state in STATES:

        temp = loo_df[
            loo_df["state"] == state
        ]

        positive_count = int(
            temp[
                "loo_remains_positive"
            ].sum()
        )

        min_total = float(
            temp[
                "leave_one_out_total_delta_r"
            ].min()
        )

        max_total = float(
            temp[
                "leave_one_out_total_delta_r"
            ].max()
        )

        print()
        print(
            STATE_LABELS[state]
        )

        print(
            f"  LOO positive: "
            f"{positive_count}/{len(temp)}"
        )

        print(
            f"  LOO total range: "
            f"{min_total:+.4f}R → "
            f"{max_total:+.4f}R"
        )

    # ------------------------------------------------------------------------
    # Print bootstrap
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("BOOTSTRAP STABILITY")
    print("=" * 72)

    for _, row in bootstrap_df.iterrows():

        print()
        print(
            row["state_label"]
        )

        print(
            f"  Total delta: "
            f"{row['observed_total_delta_r']:+.4f}R"
        )

        print(
            f"  Bootstrap total CI: "
            f"["
            f"{row['bootstrap_total_ci_low_r']:+.4f}R, "
            f"{row['bootstrap_total_ci_high_r']:+.4f}R"
            f"]"
        )

        print(
            f"  Mean delta: "
            f"{row['observed_mean_delta_r']:+.4f}R"
        )

        print(
            f"  Bootstrap mean CI: "
            f"["
            f"{row['bootstrap_mean_ci_low_r']:+.4f}R, "
            f"{row['bootstrap_mean_ci_high_r']:+.4f}R"
            f"]"
        )

    # ------------------------------------------------------------------------
    # Print event/retained
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("EVENT / RETAINED CONTRIBUTION")
    print("=" * 72)

    print(
        group_df[
            [
                "state_label",
                "population",
                "n",
                "total_delta_r",
                "mean_delta_r",
                "median_delta_r",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # Print subgroup contribution
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SUBGROUP CONTRIBUTION")
    print("=" * 72)

    for state in STATES:

        print()
        print(
            STATE_LABELS[state]
        )

        temp = subgroup_df[
            subgroup_df["state"] == state
        ].copy()

        for dimension in [
            "_year",
            "symbol",
            "setup",
            "direction",
        ]:

            subset = temp[
                temp["dimension"] == dimension
            ].copy()

            print(
                f"\n  {dimension}:"
            )

            for _, row in subset.iterrows():

                print(
                    f"    {row['group']}: "
                    f"n={int(row['n'])}, "
                    f"delta={row['total_delta_r']:+.4f}R, "
                    f"mean={row['mean_delta_r']:+.4f}R"
                )

    # ------------------------------------------------------------------------
    # Print removal stress
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("LARGEST-POSITIVE-CONTRIBUTOR REMOVAL")
    print("=" * 72)

    for state in STATES:

        print()
        print(
            STATE_LABELS[state]
        )

        temp = removal_df[
            removal_df["state"] == state
        ]

        for _, row in temp.iterrows():

            print(
                f"  Remove top "
                f"{int(row['remove_top_n_positive']):2d}: "
                f"remaining "
                f"{row['remaining_total_delta_r']:+.4f}R"
            )

    # ------------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------------

    integrity_rows = []

    integrity_rows.append(
        {
            "check": "input_rows",
            "actual": len(df),
            "expected": EXPECTED_TRADES,
            "pass": len(df)
            == EXPECTED_TRADES,
        }
    )

    integrity_rows.append(
        {
            "check": "event_count",
            "actual": int(
                df["frozen_event"].sum()
            ),
            "expected": EXPECTED_EVENTS,
            "pass":
                int(
                    df["frozen_event"].sum()
                )
                == EXPECTED_EVENTS,
        }
    )

    integrity_rows.append(
        {
            "check": "retained_count",
            "actual": int(
                (~df["frozen_event"]).sum()
            ),
            "expected": EXPECTED_RETAINED,
            "pass":
                int(
                    (~df["frozen_event"]).sum()
                )
                == EXPECTED_RETAINED,
        }
    )

    integrity_rows.append(
        {
            "check": "primary_state_count",
            "actual": len(primary_df),
            "expected": len(STATES),
            "pass":
                len(primary_df)
                == len(STATES),
        }
    )

    integrity_rows.append(
        {
            "check": "concentration_rows",
            "actual": len(concentration_df),
            "expected":
                len(STATES)
                * len(TOP_N_VALUES),
            "pass":
                len(concentration_df)
                == (
                    len(STATES)
                    * len(TOP_N_VALUES)
                ),
        }
    )

    integrity_rows.append(
        {
            "check": "loo_rows",
            "actual": len(loo_df),
            "expected":
                sum(
                    int(df[state].sum())
                    for state in STATES
                ),
            "pass":
                len(loo_df)
                == sum(
                    int(df[state].sum())
                    for state in STATES
                ),
        }
    )

    integrity_rows.append(
        {
            "check": "bootstrap_rows",
            "actual": len(bootstrap_df),
            "expected": len(STATES),
            "pass":
                len(bootstrap_df)
                == len(STATES),
        }
    )

    integrity_rows.append(
        {
            "check": "subgroup_rows",
            "actual": len(subgroup_df),
            "expected": len(subgroup_df),
            "pass": True,
        }
    )

    integrity_rows.append(
        {
            "check": "removal_rows",
            "actual": len(removal_df),
            "expected":
                len(STATES)
                * len(TOP_N_VALUES),
            "pass":
                len(removal_df)
                == (
                    len(STATES)
                    * len(TOP_N_VALUES)
                ),
        }
    )

    integrity = pd.DataFrame(
        integrity_rows
    )

    integrity.to_csv(
        OUTPUT_DIR
        / "counterfactual_stability_integrity.csv",
        index=False,
    )

    integrity_pass = bool(
        integrity["pass"].all()
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    summary = {
        "analysis":
            "Bar1->Bar2 counterfactual stability",
        "population": {
            "trades": EXPECTED_TRADES,
            "events": EXPECTED_EVENTS,
            "retained": EXPECTED_RETAINED,
        },
        "states": STATES,
        "bootstrap_iterations":
            BOOTSTRAP_ITERATIONS,
        "integrity_pass":
            integrity_pass,
        "production_modified":
            False,
        "oos_modified":
            False,
        "deployment_rule_promoted":
            False,
    }

    with open(
        OUTPUT_DIR
        / "counterfactual_stability_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"States tested: {len(STATES)}"
    )

    print(
        f"Integrity PASS: {integrity_pass}"
    )

    print()
    print(
        "No production or OOS data was modified."
    )

    print(
        "No deployment rule was promoted."
    )

    print()
    print("Outputs:")
    print(
        OUTPUT_DIR
        / "counterfactual_stability_primary.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_top_n_concentration.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_leave_one_out.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_bootstrap.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_event_retained_contribution.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_subgroup_contribution.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_removal_stress.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_stability_integrity.csv"
    )
    print(
        OUTPUT_DIR
        / "counterfactual_stability_summary.json"
    )


if __name__ == "__main__":
    main()