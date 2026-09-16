"""
TradeSense-AI
BAR 1->2 LEAD-TIME / PRE-THRESHOLD PREDICTIVITY FORENSIC ANALYSIS

Purpose
-------
Test whether the validated Bar-1 -> Bar-2 deterioration states appear
BEFORE the frozen 5-bar / 0.25R early-adverse threshold is reached.

Research-only.

Frozen population
-----------------
106 total trades
60 frozen early-adverse events
46 retained

Frozen event definition
-----------------------
stored_frozen_event == 1
i.e. production stored early_adverse_r_bar_5 >= 0.25R.

This script does NOT modify:
- production strategy logic
- frozen trade file
- OOS data

Questions
---------
1. Among eventual events whose 0.25R crossing occurs AFTER Bar 2,
   how often is the Bar-2 deterioration state already present?

2. How much lead time exists between the Bar-2 state and the eventual
   0.25R crossing?

3. How often does Bar-2 deterioration occur without a later 0.25R
   crossing in the extended 5-bar replay?

4. When Bar-2 deterioration occurs, does the path subsequently recover
   by Bar 5?

5. Are the timing/recovery patterns directionally stable across:
   - calendar year
   - symbol
   - setup
   - direction

Important
---------
This is descriptive/forensic evidence only.

It does NOT establish:
- causality
- independent OOS predictive power
- economic value
- transaction-cost-adjusted value
- a deployable trading rule
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAJECTORY_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

PATH_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_lead_time"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_BAR = 5

MIN_SUBGROUP_TRADES = 8

STATE_COLUMNS = [
    "bar2_net_deteriorated",
    "bar2_favorable_failed_to_improve",
    "adverse_up_favorable_not_up_net_down",
]

STATE_LABELS = {
    "bar2_net_deteriorated": "Bar-2 net deterioration",
    "bar2_favorable_failed_to_improve": "Bar-2 favorable failure",
    "adverse_up_favorable_not_up_net_down":
        "Bar-2 adverse-up + favorable-not-up + net-down",
}


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str):
    raise RuntimeError(message)


def require_columns(df: pd.DataFrame, columns, name: str):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(
            f"{name} is missing required columns: {missing}"
        )


def normalize_symbol(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def normalize_direction(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series,
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def make_key(df: pd.DataFrame) -> pd.Series:
    symbol = normalize_symbol(df["symbol"])
    direction = normalize_direction(df["direction"])
    entry_date = normalize_date(df["entry_date"])

    invalid = (
        symbol.isin(["", "NAN", "NONE"])
        | direction.isin(["", "NAN", "NONE"])
        | entry_date.isna()
    )

    if int(invalid.sum()) > 0:
        fail(
            f"Invalid identity rows detected: {int(invalid.sum())}"
        )

    return (
        symbol
        + "|"
        + direction
        + "|"
        + entry_date
    )


def bootstrap_median_difference(
    state1: pd.Series,
    state0: pd.Series,
    n_boot: int = 5000,
    seed: int = 42,
):
    """
    Bootstrap median(state1) - median(state0).
    """

    a = pd.to_numeric(state1, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(state0, errors="coerce").dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    observed = float(np.median(a) - np.median(b))

    boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)

        boot[i] = (
            np.median(sample_a)
            - np.median(sample_b)
        )

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))

    return observed, lo, hi


def rate_difference(
    state1: pd.Series,
    state0: pd.Series,
):
    """
    state1/state0 are boolean or 0/1 series.

    Returns:
        rate1, rate0, difference_pp
    """

    a = pd.Series(state1).dropna().astype(bool)
    b = pd.Series(state0).dropna().astype(bool)

    rate1 = float(a.mean()) if len(a) else np.nan
    rate0 = float(b.mean()) if len(b) else np.nan

    difference_pp = (
        (rate1 - rate0) * 100.0
        if np.isfinite(rate1) and np.isfinite(rate0)
        else np.nan
    )

    return rate1, rate0, difference_pp


def pct(value):
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.2f}%"


def pp(value):
    if pd.isna(value):
        return "NA"
    return f"{value:+.2f} pp"


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data():

    print(
        "TRADE SENSE-AI — BAR 1→2 LEAD-TIME / "
        "PRE-THRESHOLD PREDICTIVITY FORENSIC ANALYSIS"
    )
    print("=" * 72)

    if not TRAJECTORY_FILE.exists():
        fail(
            f"Trajectory file not found: {TRAJECTORY_FILE}"
        )

    if not PATH_FILE.exists():
        fail(
            f"Path file not found: {PATH_FILE}"
        )

    trajectory = pd.read_csv(TRAJECTORY_FILE)
    path = pd.read_csv(PATH_FILE)

    print(
        f"Trajectory input: {TRAJECTORY_FILE}"
    )
    print(
        f"Trajectory rows: {len(trajectory)}, "
        f"columns: {len(trajectory.columns)}"
    )

    print(
        f"Path input: {PATH_FILE}"
    )
    print(
        f"Path rows: {len(path)}, "
        f"columns: {len(path.columns)}"
    )

    if len(trajectory) != EXPECTED_TRADES:
        fail(
            f"Trajectory row count mismatch: "
            f"{len(trajectory)} != {EXPECTED_TRADES}"
        )

    if len(path) != EXPECTED_TRADES * MAX_BAR:
        fail(
            f"Path row count mismatch: "
            f"{len(path)} != {EXPECTED_TRADES * MAX_BAR}"
        )

    require_columns(
        trajectory,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            "bar2_net_deteriorated",
            "bar2_favorable_failed_to_improve",
            "adverse_up_favorable_not_up_net_down",
            "bar2_net_excursion_r",
            "bar1_net_excursion_r",
        ],
        "Trajectory input",
    )

    require_columns(
        path,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "bar_number",
            "stored_frozen_event",
            "bar_adverse_r",
            "cumulative_mae_r",
            "cumulative_mfe_r",
            "net_excursion_r",
        ],
        "Path input",
    )

    trajectory = trajectory.copy()
    path = path.copy()

    trajectory["_merge_key"] = make_key(trajectory)
    path["_merge_key"] = make_key(path)

    trajectory_duplicates = int(
        trajectory["_merge_key"].duplicated().sum()
    )

    if trajectory_duplicates:
        fail(
            f"Trajectory duplicate keys: "
            f"{trajectory_duplicates}"
        )

    path_duplicates = int(
        path.duplicated(
            subset=["_merge_key", "bar_number"]
        ).sum()
    )

    if path_duplicates:
        fail(
            f"Path duplicate trade/bar keys: "
            f"{path_duplicates}"
        )

    trajectory["frozen_event"] = (
        trajectory["frozen_event"]
        .astype(int)
        .astype(bool)
    )

    path["stored_frozen_event"] = (
        path["stored_frozen_event"]
        .astype(int)
        .astype(bool)
    )

    path["bar_number"] = pd.to_numeric(
        path["bar_number"],
        errors="coerce"
    )

    if path["bar_number"].isna().any():
        fail("Path contains invalid bar_number values.")

    path["bar_number"] = path["bar_number"].astype(int)

    if not set(path["bar_number"].unique()).issubset(
        set(range(1, MAX_BAR + 1))
    ):
        fail(
            "Path contains bar numbers outside 1..5."
        )

    event_count = int(trajectory["frozen_event"].sum())
    retained_count = int((~trajectory["frozen_event"]).sum())

    print(
        f"Frozen population: "
        f"{len(trajectory)} / {event_count} / {retained_count}"
    )

    if event_count != EXPECTED_EVENTS:
        fail(
            f"Frozen event count mismatch: "
            f"{event_count} != {EXPECTED_EVENTS}"
        )

    if retained_count != EXPECTED_RETAINED:
        fail(
            f"Frozen retained count mismatch: "
            f"{retained_count} != {EXPECTED_RETAINED}"
        )

    return trajectory, path


# ============================================================================
# BUILD CROSSING TIMING
# ============================================================================

def build_crossing_timing(path: pd.DataFrame):

    df = path.copy()

    df["threshold_reached_by_bar"] = (
        df["cumulative_mae_r"]
        >= EARLY_ADVERSE_THRESHOLD_R
    )

    crossing_rows = []

    for key, group in df.groupby("_merge_key", sort=False):

        group = group.sort_values("bar_number")

        crossing = group.loc[
            group["threshold_reached_by_bar"]
        ]

        if crossing.empty:
            first_crossing_bar = np.nan
        else:
            first_crossing_bar = int(
                crossing.iloc[0]["bar_number"]
            )

        frozen_event = bool(
            group["stored_frozen_event"].iloc[0]
        )

        symbol = group["symbol"].iloc[0]
        direction = group["direction"].iloc[0]
        entry_date = group["entry_date"].iloc[0]
        setup = group["setup"].iloc[0]

        crossing_rows.append(
            {
                "_merge_key": key,
                "symbol": symbol,
                "direction": direction,
                "entry_date": entry_date,
                "setup": setup,
                "frozen_event": frozen_event,
                "first_0_25r_crossing_bar": first_crossing_bar,
                "crossed_0_25r_by_bar2": (
                    pd.notna(first_crossing_bar)
                    and first_crossing_bar <= 2
                ),
                "crossed_0_25r_after_bar2": (
                    pd.notna(first_crossing_bar)
                    and first_crossing_bar > 2
                ),
                "never_crossed_0_25r_in_5bar_replay": (
                    pd.isna(first_crossing_bar)
                ),
            }
        )

    timing = pd.DataFrame(crossing_rows)

    if len(timing) != EXPECTED_TRADES:
        fail(
            f"Crossing timing rows: {len(timing)} "
            f"!= {EXPECTED_TRADES}"
        )

    return timing


# ============================================================================
# MERGE BAR2 STATES WITH TIMING
# ============================================================================

def build_analysis_frame(
    trajectory: pd.DataFrame,
    timing: pd.DataFrame,
):

    columns = [
        "_merge_key",
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",
        "bar1_net_excursion_r",
        "bar2_net_excursion_r",
        "bar2_net_deteriorated",
        "bar2_favorable_failed_to_improve",
        "adverse_up_favorable_not_up_net_down",
    ]

    base = trajectory[columns].copy()

    merged = base.merge(
        timing,
        on="_merge_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_timing"),
    )

    if len(merged) != EXPECTED_TRADES:
        fail(
            f"Analysis merge row count: {len(merged)} "
            f"!= {EXPECTED_TRADES}"
        )

    if merged["first_0_25r_crossing_bar"].isna().sum() == EXPECTED_TRADES:
        fail(
            "No threshold timing information was recovered."
        )

    return merged


# ============================================================================
# PRE-THRESHOLD LEAD-TIME ANALYSIS
# ============================================================================

def analyze_prethreshold(df: pd.DataFrame):

    rows = []

    print()
    print("=" * 72)
    print("PRE-THRESHOLD BAR-2 ANALYSIS")
    print("=" * 72)

    # Only eventual frozen events whose threshold crossing occurs
    # AFTER Bar 2.
    delayed_events = df[
        df["frozen_event"]
        & df["crossed_0_25r_after_bar2"]
    ].copy()

    print(
        f"Frozen events crossing 0.25R after Bar 2: "
        f"{len(delayed_events)}"
    )

    if len(delayed_events) == 0:
        fail(
            "No frozen events cross 0.25R after Bar 2."
        )

    retained = df[
        ~df["frozen_event"]
    ].copy()

    for state in STATE_COLUMNS:

        event_state = delayed_events[state].astype(bool)
        retained_state = retained[state].astype(bool)

        event_rate = float(event_state.mean())
        retained_rate = float(retained_state.mean())
        difference_pp = (
            (event_rate - retained_rate) * 100.0
        )

        rows.append(
            {
                "analysis": "prethreshold_bar2_state",
                "state": state,
                "state_label": STATE_LABELS[state],
                "event_n": len(delayed_events),
                "retained_n": len(retained),
                "event_state_n": int(event_state.sum()),
                "retained_state_n": int(retained_state.sum()),
                "event_state_rate": event_rate,
                "retained_state_rate": retained_rate,
                "rate_difference_pp": difference_pp,
            }
        )

        print()
        print(STATE_LABELS[state])
        print(
            f"  Delayed-event state rate: "
            f"{pct(event_rate)} "
            f"({int(event_state.sum())}/{len(event_state)})"
        )
        print(
            f"  Retained state rate:      "
            f"{pct(retained_rate)} "
            f"({int(retained_state.sum())}/{len(retained_state)})"
        )
        print(
            f"  Difference:               "
            f"{pp(difference_pp)}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# LEAD-TIME DISTRIBUTION
# ============================================================================

def analyze_lead_time(df: pd.DataFrame):

    events = df[
        df["frozen_event"]
        & df["crossed_0_25r_after_bar2"]
    ].copy()

    events["lead_time_bars"] = (
        events["first_0_25r_crossing_bar"]
        - 2
    )

    print()
    print("=" * 72)
    print("0.25R CROSSING LEAD-TIME")
    print("=" * 72)

    distribution = (
        events["lead_time_bars"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Delayed event count: {len(events)}"
    )

    for bar_count, count in distribution.items():
        print(
            f"  Lead time {int(bar_count)} bar(s): "
            f"{int(count)} trades "
            f"({count / len(events) * 100:.2f}%)"
        )

    median_lead = float(
        events["lead_time_bars"].median()
    )

    mean_lead = float(
        events["lead_time_bars"].mean()
    )

    print(
        f"  Median lead time: {median_lead:.2f} bars"
    )
    print(
        f"  Mean lead time:   {mean_lead:.2f} bars"
    )

    return events


# ============================================================================
# BAR2 STATE VS CROSSING TIMING
# ============================================================================

def analyze_state_timing(
    df: pd.DataFrame,
    delayed_events: pd.DataFrame,
):

    rows = []

    print()
    print("=" * 72)
    print("BAR-2 STATE VS EVENT CROSSING TIMING")
    print("=" * 72)

    for state in STATE_COLUMNS:

        state1 = delayed_events[
            delayed_events[state].astype(bool)
        ]

        state0 = delayed_events[
            ~delayed_events[state].astype(bool)
        ]

        lead1 = state1["lead_time_bars"]
        lead0 = state0["lead_time_bars"]

        median1 = (
            float(lead1.median())
            if len(lead1)
            else np.nan
        )

        median0 = (
            float(lead0.median())
            if len(lead0)
            else np.nan
        )

        difference = (
            median1 - median0
            if np.isfinite(median1)
            and np.isfinite(median0)
            else np.nan
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "state1_n": len(state1),
                "state0_n": len(state0),
                "state1_median_lead_bars": median1,
                "state0_median_lead_bars": median0,
                "median_lead_difference_bars": difference,
            }
        )

        print()
        print(STATE_LABELS[state])
        print(
            f"  State 1 n: {len(state1)}"
        )
        print(
            f"  State 0 n: {len(state0)}"
        )
        print(
            f"  State 1 median lead: "
            f"{median1 if np.isfinite(median1) else 'NA'}"
        )
        print(
            f"  State 0 median lead: "
            f"{median0 if np.isfinite(median0) else 'NA'}"
        )
        print(
            f"  Difference: "
            f"{difference:+.2f} bars"
            if np.isfinite(difference)
            else "  Difference: NA"
        )

    return pd.DataFrame(rows)


# ============================================================================
# RECOVERY ANALYSIS
# ============================================================================

def analyze_recovery(
    trajectory: pd.DataFrame,
    path: pd.DataFrame,
    df: pd.DataFrame,
):

    print()
    print("=" * 72)
    print("BAR-2 DETERIORATION RECOVERY ANALYSIS")
    print("=" * 72)

    bar5 = path[
        path["bar_number"] == MAX_BAR
    ][
        [
            "_merge_key",
            "cumulative_mae_r",
            "cumulative_mfe_r",
            "net_excursion_r",
        ]
    ].copy()

    bar5 = bar5.rename(
        columns={
            "cumulative_mae_r": "bar5_cumulative_mae_r",
            "cumulative_mfe_r": "bar5_cumulative_mfe_r",
            "net_excursion_r": "bar5_net_excursion_r",
        }
    )

    bar2 = path[
        path["bar_number"] == 2
    ][
        [
            "_merge_key",
            "cumulative_mae_r",
            "cumulative_mfe_r",
            "net_excursion_r",
        ]
    ].copy()

    bar2 = bar2.rename(
        columns={
            "cumulative_mae_r": "bar2_cumulative_mae_r_path",
            "cumulative_mfe_r": "bar2_cumulative_mfe_r_path",
            "net_excursion_r": "bar2_net_excursion_r_path",
        }
    )

    recovery = df.merge(
        bar2,
        on="_merge_key",
        how="left",
        validate="one_to_one",
    ).merge(
        bar5,
        on="_merge_key",
        how="left",
        validate="one_to_one",
    )

    recovery["net_recovered_by_bar5"] = (
        recovery["bar5_net_excursion_r"]
        > recovery["bar2_net_excursion_r_path"]
    )

    recovery["mfe_improved_after_bar2"] = (
        recovery["bar5_cumulative_mfe_r"]
        > recovery["bar2_cumulative_mfe_r_path"]
    )

    recovery["mae_worsened_after_bar2"] = (
        recovery["bar5_cumulative_mae_r"]
        > recovery["bar2_cumulative_mae_r_path"]
    )

    recovery["net_change_bar2_to_bar5"] = (
        recovery["bar5_net_excursion_r"]
        - recovery["bar2_net_excursion_r_path"]
    )

    recovery["mfe_change_bar2_to_bar5"] = (
        recovery["bar5_cumulative_mfe_r"]
        - recovery["bar2_cumulative_mfe_r_path"]
    )

    recovery["mae_change_bar2_to_bar5"] = (
        recovery["bar5_cumulative_mae_r"]
        - recovery["bar2_cumulative_mae_r_path"]
    )

    rows = []

    for state in STATE_COLUMNS:

        state1 = recovery[
            recovery[state].astype(bool)
        ].copy()

        state0 = recovery[
            ~recovery[state].astype(bool)
        ].copy()

        valid1 = state1[
            state1["bar5_net_excursion_r"].notna()
        ]

        valid0 = state0[
            state0["bar5_net_excursion_r"].notna()
        ]

        if len(valid1) == 0 or len(valid0) == 0:
            continue

        rate1 = float(
            valid1["net_recovered_by_bar5"].mean()
        )

        rate0 = float(
            valid0["net_recovered_by_bar5"].mean()
        )

        rd = (rate1 - rate0) * 100.0

        median_net_change1 = float(
            valid1["net_change_bar2_to_bar5"].median()
        )

        median_net_change0 = float(
            valid0["net_change_bar2_to_bar5"].median()
        )

        median_mfe_change1 = float(
            valid1["mfe_change_bar2_to_bar5"].median()
        )

        median_mfe_change0 = float(
            valid0["mfe_change_bar2_to_bar5"].median()
        )

        median_mae_change1 = float(
            valid1["mae_change_bar2_to_bar5"].median()
        )

        median_mae_change0 = float(
            valid0["mae_change_bar2_to_bar5"].median()
        )

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "state1_n": len(valid1),
                "state0_n": len(valid0),
                "state1_recovery_rate": rate1,
                "state0_recovery_rate": rate0,
                "recovery_rate_difference_pp": rd,
                "state1_median_net_change_bar2_to_bar5":
                    median_net_change1,
                "state0_median_net_change_bar2_to_bar5":
                    median_net_change0,
                "state1_median_mfe_change_bar2_to_bar5":
                    median_mfe_change1,
                "state0_median_mfe_change_bar2_to_bar5":
                    median_mfe_change0,
                "state1_median_mae_change_bar2_to_bar5":
                    median_mae_change1,
                "state0_median_mae_change_bar2_to_bar5":
                    median_mae_change0,
            }
        )

        print()
        print(STATE_LABELS[state])
        print(
            f"  State 1 n: {len(valid1)}"
        )
        print(
            f"  State 0 n: {len(valid0)}"
        )
        print(
            f"  Net recovery by Bar 5:"
            f" {pct(rate1)} vs {pct(rate0)}"
        )
        print(
            f"  Recovery-rate difference:"
            f" {pp(rd)}"
        )
        print(
            f"  Median net change Bar2→Bar5:"
            f" {median_net_change1:+.4f}R vs "
            f"{median_net_change0:+.4f}R"
        )
        print(
            f"  Median MFE change Bar2→Bar5:"
            f" {median_mfe_change1:+.4f}R vs "
            f"{median_mfe_change0:+.4f}R"
        )
        print(
            f"  Median MAE change Bar2→Bar5:"
            f" {median_mae_change1:+.4f}R vs "
            f"{median_mae_change0:+.4f}R"
        )

    return (
        recovery,
        pd.DataFrame(rows),
    )


# ============================================================================
# ROBUSTNESS
# ============================================================================

def robustness_analysis(
    delayed_events: pd.DataFrame,
):

    rows = []

    dimensions = {
        "year": (
            pd.to_datetime(
                delayed_events["entry_date"],
                errors="coerce"
            ).dt.year.astype(str)
        ),
        "symbol": delayed_events["symbol"].astype(str),
        "setup": delayed_events["setup"].astype(str),
        "direction": delayed_events["direction"].astype(str),
    }

    print()
    print("=" * 72)
    print("PRE-THRESHOLD STATE ROBUSTNESS")
    print("=" * 72)

    for dimension, groups in dimensions.items():

        working = delayed_events.copy()
        working["_group"] = groups.values

        for state in STATE_COLUMNS:

            subgroup_rows = []

            for group_name, subgroup in working.groupby(
                "_group",
                sort=True,
            ):

                if len(subgroup) < MIN_SUBGROUP_TRADES:
                    continue

                rate = float(
                    subgroup[state].astype(bool).mean()
                )

                subgroup_rows.append(
                    {
                        "dimension": dimension,
                        "state": state,
                        "group": group_name,
                        "n": len(subgroup),
                        "state_rate": rate,
                    }
                )

            if not subgroup_rows:
                continue

            sub = pd.DataFrame(subgroup_rows)

            # Here we are only measuring whether the state is present.
            # There is no subgroup-specific threshold optimization.
            median_rate = float(
                sub["state_rate"].median()
            )

            rows.extend(subgroup_rows)

            print()
            print(
                f"{STATE_LABELS[state]} | {dimension}"
            )
            print(
                f"  Valid subgroups: {len(sub)}"
            )
            print(
                f"  Median state rate: "
                f"{median_rate * 100:.2f}%"
            )

    return pd.DataFrame(rows)


# ============================================================================
# INTEGRITY
# ============================================================================

def integrity_checks(
    trajectory: pd.DataFrame,
    path: pd.DataFrame,
    timing: pd.DataFrame,
    analysis: pd.DataFrame,
    delayed_events: pd.DataFrame,
    recovery: pd.DataFrame,
):

    checks = []

    checks.append(
        {
            "check": "trajectory_trade_count",
            "actual": len(trajectory),
            "expected": EXPECTED_TRADES,
            "pass": len(trajectory) == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "frozen_event_count",
            "actual": int(trajectory["frozen_event"].sum()),
            "expected": EXPECTED_EVENTS,
            "pass":
                int(trajectory["frozen_event"].sum())
                == EXPECTED_EVENTS,
        }
    )

    checks.append(
        {
            "check": "frozen_retained_count",
            "actual": int((~trajectory["frozen_event"]).sum()),
            "expected": EXPECTED_RETAINED,
            "pass":
                int((~trajectory["frozen_event"]).sum())
                == EXPECTED_RETAINED,
        }
    )

    checks.append(
        {
            "check": "path_row_count",
            "actual": len(path),
            "expected": EXPECTED_TRADES * MAX_BAR,
            "pass":
                len(path)
                == EXPECTED_TRADES * MAX_BAR,
        }
    )

    checks.append(
        {
            "check": "timing_rows",
            "actual": len(timing),
            "expected": EXPECTED_TRADES,
            "pass": len(timing) == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "analysis_rows",
            "actual": len(analysis),
            "expected": EXPECTED_TRADES,
            "pass": len(analysis) == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "delayed_event_count_positive",
            "actual": len(delayed_events),
            "expected": "> 0",
            "pass": len(delayed_events) > 0,
        }
    )

    checks.append(
        {
            "check": "recovery_rows",
            "actual": len(recovery),
            "expected": EXPECTED_TRADES,
            "pass": len(recovery) == EXPECTED_TRADES,
        }
    )

    checks_df = pd.DataFrame(checks)

    return checks_df


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory, path = load_data()

    timing = build_crossing_timing(path)

    analysis = build_analysis_frame(
        trajectory,
        timing,
    )

    prethreshold = analyze_prethreshold(
        analysis
    )

    delayed_events = analyze_lead_time(
        analysis
    )

    timing_results = analyze_state_timing(
        analysis,
        delayed_events,
    )

    recovery, recovery_results = analyze_recovery(
        trajectory,
        path,
        analysis,
    )

    robustness = robustness_analysis(
        delayed_events
    )

    integrity = integrity_checks(
        trajectory,
        path,
        timing,
        analysis,
        delayed_events,
        recovery,
    )

    # ------------------------------------------------------------------------
    # Per-trade output
    # ------------------------------------------------------------------------

    per_trade = recovery.copy()

    per_trade.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_per_trade.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Timing output
    # ------------------------------------------------------------------------

    timing.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_crossing_timing.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Pre-threshold output
    # ------------------------------------------------------------------------

    prethreshold.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_prethreshold.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Timing-state output
    # ------------------------------------------------------------------------

    timing_results.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_state_timing.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Recovery output
    # ------------------------------------------------------------------------

    recovery_results.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_recovery.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Robustness output
    # ------------------------------------------------------------------------

    robustness.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_robustness.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Integrity output
    # ------------------------------------------------------------------------

    integrity.to_csv(
        OUTPUT_DIR
        / "bar12_lead_time_integrity.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------------

    summary = {
        "analysis":
            "Bar1->Bar2 lead-time / pre-threshold predictivity",
        "frozen_population": {
            "trades": EXPECTED_TRADES,
            "events": EXPECTED_EVENTS,
            "retained": EXPECTED_RETAINED,
        },
        "threshold_r":
            EARLY_ADVERSE_THRESHOLD_R,
        "max_bar":
            MAX_BAR,
        "delayed_event_count":
            int(len(delayed_events)),
        "integrity_pass":
            bool(integrity["pass"].all()),
        "production_modified":
            False,
        "oos_modified":
            False,
        "deployment_rule_promoted":
            False,
    }

    with open(
        OUTPUT_DIR / "bar12_lead_time_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final console result
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"Delayed frozen events "
        f"(0.25R crossing after Bar 2): "
        f"{len(delayed_events)}"
    )

    print(
        f"Integrity PASS: "
        f"{bool(integrity['pass'].all())}"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This remains a forensic lead-time/recovery analysis."
    )
    print(
        "It does NOT establish causality, independent OOS "
        "predictive power, economic value, or a deployable rule."
    )
    print(
        "Production strategy logic and OOS data remain untouched."
    )

    print()
    print("Outputs:")
    print(
        OUTPUT_DIR
        / "bar12_lead_time_per_trade.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_crossing_timing.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_prethreshold.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_state_timing.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_recovery.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_robustness.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_integrity.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_lead_time_summary.json"
    )


if __name__ == "__main__":
    main()