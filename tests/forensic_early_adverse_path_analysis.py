"""
TradeSense-AI — Early-Adverse Path Forensic Analysis

Research-only analysis of the validated First-5-Bar Path Foundation.

Frozen foundation:
    - 106 frozen trades
    - 60 frozen early-adverse events
    - 46 retained trades
    - 5 reconstructed post-entry bars per trade
    - frozen event label = stored production bar-5 >= 0.25R

IMPORTANT:
    - Does NOT modify production code.
    - Does NOT modify frozen trades.
    - Does NOT modify OOS / holdout data.
    - Does NOT promote any strategy rule.
    - Consumes only the validated First-5-Bar Path Foundation CSV.

Research objectives:
    1. Event vs retained trajectory across bars 1-5.
    2. Earliest warning-bar separation.
    3. Threshold progression.
    4. Candle / microstructure behavior.
    5. Direction robustness.
    6. Setup robustness.
    7. Symbol robustness.
    8. Sample-size and missingness integrity.
"""

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_path_analysis"
)

TRAJECTORY_FILE = (
    OUTPUT_DIR / "early_adverse_path_trajectory.csv"
)

EARLIEST_WARNING_FILE = (
    OUTPUT_DIR / "early_adverse_path_earliest_warning.csv"
)

THRESHOLD_FILE = (
    OUTPUT_DIR / "early_adverse_path_threshold_progression.csv"
)

MICROSTRUCTURE_FILE = (
    OUTPUT_DIR / "early_adverse_path_microstructure.csv"
)

DIRECTION_FILE = (
    OUTPUT_DIR / "early_adverse_path_direction.csv"
)

SETUP_FILE = (
    OUTPUT_DIR / "early_adverse_path_setup.csv"
)

SYMBOL_FILE = (
    OUTPUT_DIR / "early_adverse_path_symbol.csv"
)

INTEGRITY_FILE = (
    OUTPUT_DIR / "early_adverse_path_analysis_integrity.csv"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46
EXPECTED_BARS_PER_TRADE = 5
EXPECTED_PATH_ROWS = 530

FROZEN_THRESHOLD_R = 0.25

THRESHOLDS_R = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

EPSILON = 1e-12


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    print(f"[FAIL] {message}")
    sys.exit(1)


def median_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(s.median())


def mean_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(s.mean())


def std_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) <= 1:
        return np.nan

    return float(s.std(ddof=1))


def min_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(s.min())


def max_or_nan(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        return np.nan

    return float(s.max())


def difference(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan

    return float(a - b)


def require_columns(df, columns):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


# ============================================================================
# HEADER
# ============================================================================

print("=" * 78)
print("TRADE SENSE-AI — FIRST-5-BAR PATH FORENSIC ANALYSIS")
print("=" * 78)

print(f"[INFO] Input: {INPUT_FILE}")

if not INPUT_FILE.exists():
    fail(
        f"Input file does not exist: {INPUT_FILE}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(INPUT_FILE)

print(f"[INFO] Input rows: {len(df)}")
print(f"[INFO] Input columns: {len(df.columns)}")


# ============================================================================
# REQUIRED REAL SCHEMA
# ============================================================================

required_columns = [
    "symbol",
    "direction",
    "entry_date",
    "entry_timestamp",
    "signal_timestamp",
    "setup",
    "bar_number",
    "bar_timestamp",
    "entry_price_reconstructed",
    "entry_atr_exact",
    "risk_per_unit",
    "open",
    "high",
    "low",
    "close",
    "bar_adverse_r",
    "bar_favorable_r",
    "cumulative_mae_r",
    "cumulative_mfe_r",
    "incremental_mae_r",
    "incremental_mfe_r",
    "net_excursion_r",
    "mfe_to_mae_ratio",
    "open_displacement_r",
    "close_from_entry_r",
    "close_move_r",
    "close_direction",
    "range_price",
    "body_price",
    "upper_wick_price",
    "lower_wick_price",
    "adverse_wick_price",
    "favorable_wick_price",
    "body_pct_range",
    "upper_wick_pct_range",
    "lower_wick_pct_range",
    "close_location",
    "bar_range_atr",
    "body_atr",
    "adverse_wick_atr",
    "favorable_wick_atr",
    "bar_adverse_atr",
    "bar_favorable_atr",
    "crossed_0_25r_by_this_bar",
    "stored_frozen_event",
    "extended_5bar_event",
    "production_bars_in_trade",
    "bar5_reached_in_production",
    "production_observed",
    "production_observation_status",
]

require_columns(
    df,
    required_columns,
)


# ============================================================================
# NORMALIZATION
# ============================================================================

df["symbol"] = (
    df["symbol"]
    .astype(str)
    .str.strip()
)

df["direction"] = (
    df["direction"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["setup"] = (
    df["setup"]
    .astype(str)
    .str.strip()
)

df["bar_number"] = pd.to_numeric(
    df["bar_number"],
    errors="coerce",
)

numeric_columns = [
    "entry_price_reconstructed",
    "entry_atr_exact",
    "risk_per_unit",
    "open",
    "high",
    "low",
    "close",
    "bar_adverse_r",
    "bar_favorable_r",
    "cumulative_mae_r",
    "cumulative_mfe_r",
    "incremental_mae_r",
    "incremental_mfe_r",
    "net_excursion_r",
    "mfe_to_mae_ratio",
    "open_displacement_r",
    "close_from_entry_r",
    "close_move_r",
    "range_price",
    "body_price",
    "upper_wick_price",
    "lower_wick_price",
    "adverse_wick_price",
    "favorable_wick_price",
    "body_pct_range",
    "upper_wick_pct_range",
    "lower_wick_pct_range",
    "close_location",
    "bar_range_atr",
    "body_atr",
    "adverse_wick_atr",
    "favorable_wick_atr",
    "bar_adverse_atr",
    "bar_favorable_atr",
    "crossed_0_25r_by_this_bar",
    "stored_frozen_event",
    "extended_5bar_event",
    "production_bars_in_trade",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================================
# TRADE ID
# ============================================================================
#
# The validated CSV does not contain trade_index.
#
# Build a deterministic trade identity from the fields that identify
# one reconstructed trade. This is only an analysis key; it does not
# modify the source data.

trade_identity_columns = [
    "symbol",
    "direction",
    "entry_date",
    "entry_timestamp",
    "signal_timestamp",
    "setup",
]

for column in trade_identity_columns:
    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )

df["trade_id"] = (
    df[trade_identity_columns]
    .astype(str)
    .agg("|".join, axis=1)
)


# ============================================================================
# BASIC PATH INTEGRITY
# ============================================================================

unique_trade_ids = df["trade_id"].unique()

trade_count = len(unique_trade_ids)

if trade_count != EXPECTED_TRADES:
    fail(
        f"Expected {EXPECTED_TRADES} unique trades, "
        f"observed {trade_count}."
    )

if len(df) != EXPECTED_PATH_ROWS:
    fail(
        f"Expected {EXPECTED_PATH_ROWS} path rows, "
        f"observed {len(df)}."
    )

bars_per_trade = (
    df.groupby("trade_id")["bar_number"]
    .nunique()
)

bad_bar_counts = bars_per_trade[
    bars_per_trade != EXPECTED_BARS_PER_TRADE
]

if len(bad_bar_counts) > 0:
    fail(
        "One or more trades do not contain exactly "
        "five reconstructed path bars."
    )

expected_bar_set = {
    1,
    2,
    3,
    4,
    5,
}

bad_sequences = []

for trade_id, group in df.groupby("trade_id"):

    observed = set(
        int(value)
        for value in group["bar_number"]
        .dropna()
        .tolist()
    )

    if observed != expected_bar_set:
        bad_sequences.append(
            {
                "trade_id": trade_id,
                "observed_bars": sorted(
                    observed
                ),
            }
        )

if bad_sequences:
    fail(
        "Invalid bar sequence detected."
    )


# ============================================================================
# FROZEN EVENT LABEL
# ============================================================================

trade_event = (
    df.groupby("trade_id")["stored_frozen_event"]
    .first()
)

event_count = int(
    (trade_event == 1).sum()
)

retained_count = int(
    (trade_event == 0).sum()
)

if event_count != EXPECTED_EVENTS:
    fail(
        f"Expected {EXPECTED_EVENTS} frozen events, "
        f"observed {event_count}."
    )

if retained_count != EXPECTED_RETAINED:
    fail(
        f"Expected {EXPECTED_RETAINED} retained trades, "
        f"observed {retained_count}."
    )

# Verify event label is constant across all five bars.
event_consistency = (
    df.groupby("trade_id")["stored_frozen_event"]
    .nunique(dropna=True)
)

bad_event_consistency = event_consistency[
    event_consistency > 1
]

if len(bad_event_consistency) > 0:
    fail(
        "Frozen event label changes within one or more trades."
    )

# Use the frozen event label for all analysis.
df["frozen_event"] = (
    df["stored_frozen_event"]
    .astype(int)
)


print()
print("[PASS] Frozen population:")
print(f"       Trades   : {trade_count}")
print(f"       Events   : {event_count}")
print(f"       Retained : {retained_count}")
print(f"       Path rows: {len(df)}")


# ============================================================================
# 1. EVENT VS RETAINED TRAJECTORY
# ============================================================================

print()
print("-" * 78)
print("1. EVENT VS RETAINED TRAJECTORY")
print("-" * 78)

trajectory_metrics = [
    "bar_adverse_r",
    "cumulative_mae_r",
    "incremental_mae_r",
    "bar_favorable_r",
    "cumulative_mfe_r",
    "incremental_mfe_r",
    "net_excursion_r",
    "open_displacement_r",
    "close_from_entry_r",
    "close_move_r",
    "range_price",
    "body_price",
    "upper_wick_price",
    "lower_wick_price",
    "adverse_wick_price",
    "favorable_wick_price",
    "body_pct_range",
    "upper_wick_pct_range",
    "lower_wick_pct_range",
    "close_location",
    "bar_range_atr",
    "body_atr",
    "adverse_wick_atr",
    "favorable_wick_atr",
    "bar_adverse_atr",
    "bar_favorable_atr",
]

trajectory_rows = []

for bar_number in range(1, 6):

    bar_df = df[
        df["bar_number"] == bar_number
    ]

    for metric in trajectory_metrics:

        event_values = bar_df.loc[
            bar_df["frozen_event"] == 1,
            metric,
        ]

        retained_values = bar_df.loc[
            bar_df["frozen_event"] == 0,
            metric,
        ]

        event_median = median_or_nan(
            event_values
        )

        retained_median = median_or_nan(
            retained_values
        )

        event_mean = mean_or_nan(
            event_values
        )

        retained_mean = mean_or_nan(
            retained_values
        )

        trajectory_rows.append(
            {
                "bar_number": bar_number,
                "metric": metric,
                "event_n":
                    int(event_values.notna().sum()),
                "retained_n":
                    int(retained_values.notna().sum()),
                "event_median":
                    event_median,
                "retained_median":
                    retained_median,
                "median_difference_event_minus_retained":
                    difference(
                        event_median,
                        retained_median,
                    ),
                "event_mean":
                    event_mean,
                "retained_mean":
                    retained_mean,
                "mean_difference_event_minus_retained":
                    difference(
                        event_mean,
                        retained_mean,
                    ),
            }
        )

trajectory_output = pd.DataFrame(
    trajectory_rows
)

trajectory_output.to_csv(
    TRAJECTORY_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {TRAJECTORY_FILE}"
)


# ============================================================================
# 2. EARLIEST WARNING BAR
# ============================================================================

print()
print("-" * 78)
print("2. EARLIEST WARNING-BAR ANALYSIS")
print("-" * 78)

earliest_rows = []

for threshold in THRESHOLDS_R:

    for trade_id, group in (
        df.groupby("trade_id")
    ):

        group = group.sort_values(
            "bar_number"
        )

        reached = group[
            group["cumulative_mae_r"]
            >= threshold - EPSILON
        ]

        if len(reached) > 0:
            earliest_bar = int(
                reached.iloc[0]["bar_number"]
            )
        else:
            earliest_bar = np.nan

        frozen_event = int(
            trade_event.loc[trade_id]
        )

        earliest_rows.append(
            {
                "trade_id": trade_id,
                "frozen_event": frozen_event,
                "threshold_r": threshold,
                "earliest_reached_bar":
                    earliest_bar,
                "reached_within_5_bars":
                    int(
                        not pd.isna(
                            earliest_bar
                        )
                    ),
            }
        )

earliest_output = pd.DataFrame(
    earliest_rows
)


# Summary by threshold and event group.
warning_summary = (
    earliest_output
    .groupby(
        [
            "threshold_r",
            "frozen_event",
        ]
    )
    .agg(
        trades=(
            "trade_id",
            "nunique",
        ),
        reached_within_5_bars=(
            "reached_within_5_bars",
            "sum",
        ),
    )
    .reset_index()
)

warning_summary["reach_rate"] = (
    warning_summary["reached_within_5_bars"]
    / warning_summary["trades"]
)


# Distribution of the first bar at which threshold was reached.
warning_distribution = (
    earliest_output[
        earliest_output[
            "reached_within_5_bars"
        ] == 1
    ]
    .groupby(
        [
            "threshold_r",
            "frozen_event",
            "earliest_reached_bar",
        ]
    )
    .size()
    .reset_index(
        name="trades"
    )
)

warning_distribution[
    "group_total_reached"
] = (
    warning_distribution
    .groupby(
        [
            "threshold_r",
            "frozen_event",
        ]
    )["trades"]
    .transform("sum")
)

warning_distribution[
    "share_of_reached"
] = (
    warning_distribution["trades"]
    / warning_distribution[
        "group_total_reached"
    ]
)


warning_output = warning_summary.merge(
    warning_distribution,
    on=[
        "threshold_r",
        "frozen_event",
    ],
    how="left",
)

warning_output.to_csv(
    EARLIEST_WARNING_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {EARLIEST_WARNING_FILE}"
)


# ============================================================================
# 3. THRESHOLD PROGRESSION
# ============================================================================

print()
print("-" * 78)
print("3. THRESHOLD PROGRESSION")
print("-" * 78)

threshold_rows = []

for threshold in THRESHOLDS_R:

    for event_value, label in [
        (1, "EVENT"),
        (0, "RETAINED"),
    ]:

        subset = earliest_output[
            (
                earliest_output[
                    "threshold_r"
                ] == threshold
            )
            &
            (
                earliest_output[
                    "frozen_event"
                ] == event_value
            )
        ]

        n_trades = len(subset)

        reached_count = int(
            subset[
                "reached_within_5_bars"
            ].sum()
        )

        reached_rate = (
            reached_count / n_trades
            if n_trades > 0
            else np.nan
        )

        first_bars = subset.loc[
            subset[
                "reached_within_5_bars"
            ] == 1,
            "earliest_reached_bar",
        ]

        threshold_rows.append(
            {
                "threshold_r": threshold,
                "group": label,
                "n_trades": n_trades,
                "reached_count":
                    reached_count,
                "reached_rate":
                    reached_rate,
                "reached_rate_pct":
                    reached_rate * 100
                    if not pd.isna(
                        reached_rate
                    )
                    else np.nan,
                "median_first_reached_bar":
                    median_or_nan(
                        first_bars
                    ),
            }
        )

threshold_output = pd.DataFrame(
    threshold_rows
)

threshold_pivot = (
    threshold_output
    .pivot(
        index="threshold_r",
        columns="group",
        values="reached_rate_pct",
    )
    .reset_index()
)

if "EVENT" in threshold_pivot.columns:
    threshold_pivot[
        "EVENT_rate_pct"
    ] = threshold_pivot["EVENT"]

if "RETAINED" in threshold_pivot.columns:
    threshold_pivot[
        "RETAINED_rate_pct"
    ] = threshold_pivot["RETAINED"]

if (
    "EVENT_rate_pct" in threshold_pivot.columns
    and
    "RETAINED_rate_pct" in threshold_pivot.columns
):
    threshold_pivot[
        "event_minus_retained_pp"
    ] = (
        threshold_pivot[
            "EVENT_rate_pct"
        ]
        -
        threshold_pivot[
            "RETAINED_rate_pct"
        ]
    )

threshold_output.to_csv(
    THRESHOLD_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {THRESHOLD_FILE}"
)

print()
print(
    threshold_pivot.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================================
# 4. MICROSTRUCTURE ANALYSIS
# ============================================================================

print()
print("-" * 78)
print("4. MICROSTRUCTURE ANALYSIS")
print("-" * 78)

microstructure_metrics = [
    "bar_adverse_r",
    "incremental_mae_r",
    "cumulative_mae_r",
    "bar_favorable_r",
    "incremental_mfe_r",
    "cumulative_mfe_r",
    "net_excursion_r",
    "open_displacement_r",
    "close_from_entry_r",
    "close_move_r",
    "range_price",
    "body_price",
    "upper_wick_price",
    "lower_wick_price",
    "adverse_wick_price",
    "favorable_wick_price",
    "body_pct_range",
    "upper_wick_pct_range",
    "lower_wick_pct_range",
    "close_location",
    "bar_range_atr",
    "body_atr",
    "adverse_wick_atr",
    "favorable_wick_atr",
    "bar_adverse_atr",
    "bar_favorable_atr",
]

microstructure_rows = []

for bar_number in range(1, 6):

    bar_df = df[
        df["bar_number"] == bar_number
    ]

    for metric in microstructure_metrics:

        event_values = bar_df.loc[
            bar_df["frozen_event"] == 1,
            metric,
        ]

        retained_values = bar_df.loc[
            bar_df["frozen_event"] == 0,
            metric,
        ]

        event_median = median_or_nan(
            event_values
        )

        retained_median = median_or_nan(
            retained_values
        )

        event_mean = mean_or_nan(
            event_values
        )

        retained_mean = mean_or_nan(
            retained_values
        )

        microstructure_rows.append(
            {
                "bar_number": bar_number,
                "metric": metric,
                "event_n":
                    int(event_values.notna().sum()),
                "retained_n":
                    int(retained_values.notna().sum()),
                "event_median":
                    event_median,
                "retained_median":
                    retained_median,
                "event_minus_retained_median":
                    difference(
                        event_median,
                        retained_median,
                    ),
                "event_mean":
                    event_mean,
                "retained_mean":
                    retained_mean,
                "event_minus_retained_mean":
                    difference(
                        event_mean,
                        retained_mean,
                    ),
                "event_std":
                    std_or_nan(event_values),
                "retained_std":
                    std_or_nan(
                        retained_values
                    ),
            }
        )

microstructure_output = pd.DataFrame(
    microstructure_rows
)

microstructure_output.to_csv(
    MICROSTRUCTURE_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {MICROSTRUCTURE_FILE}"
)


# ============================================================================
# 5. DIRECTION ROBUSTNESS
# ============================================================================

print()
print("-" * 78)
print("5. DIRECTION ROBUSTNESS")
print("-" * 78)

direction_rows = []

for direction, direction_df in (
    df.groupby("direction")
):

    direction_trade_event = (
        direction_df
        .groupby("trade_id")[
            "frozen_event"
        ]
        .first()
    )

    direction_trade_count = len(
        direction_trade_event
    )

    direction_event_count = int(
        (
            direction_trade_event == 1
        ).sum()
    )

    direction_retained_count = int(
        (
            direction_trade_event == 0
        ).sum()
    )

    event_rate = (
        direction_event_count
        / direction_trade_count
        if direction_trade_count > 0
        else np.nan
    )

    for bar_number in range(1, 6):

        bar_df = direction_df[
            direction_df["bar_number"]
            == bar_number
        ]

        event_mae = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "cumulative_mae_r",
        ]

        retained_mae = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "cumulative_mae_r",
        ]

        event_close = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "close_from_entry_r",
        ]

        retained_close = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "close_from_entry_r",
        ]

        event_adverse = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "bar_adverse_r",
        ]

        retained_adverse = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "bar_adverse_r",
        ]

        direction_rows.append(
            {
                "direction": direction,
                "bar_number": bar_number,
                "trade_count":
                    direction_trade_count,
                "event_count":
                    direction_event_count,
                "retained_count":
                    direction_retained_count,
                "event_rate":
                    event_rate,
                "event_rate_pct":
                    event_rate * 100
                    if not pd.isna(event_rate)
                    else np.nan,
                "event_cumulative_mae_median":
                    median_or_nan(
                        event_mae
                    ),
                "retained_cumulative_mae_median":
                    median_or_nan(
                        retained_mae
                    ),
                "mae_median_difference":
                    difference(
                        median_or_nan(
                            event_mae
                        ),
                        median_or_nan(
                            retained_mae
                        ),
                    ),
                "event_close_from_entry_median":
                    median_or_nan(
                        event_close
                    ),
                "retained_close_from_entry_median":
                    median_or_nan(
                        retained_close
                    ),
                "close_difference":
                    difference(
                        median_or_nan(
                            event_close
                        ),
                        median_or_nan(
                            retained_close
                        ),
                    ),
                "event_bar_adverse_median":
                    median_or_nan(
                        event_adverse
                    ),
                "retained_bar_adverse_median":
                    median_or_nan(
                        retained_adverse
                    ),
            }
        )

direction_output = pd.DataFrame(
    direction_rows
)

direction_output.to_csv(
    DIRECTION_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {DIRECTION_FILE}"
)


# ============================================================================
# 6. SETUP ROBUSTNESS
# ============================================================================

print()
print("-" * 78)
print("6. SETUP ROBUSTNESS")
print("-" * 78)

setup_rows = []

for setup, setup_df in (
    df.groupby("setup")
):

    setup_trade_event = (
        setup_df
        .groupby("trade_id")[
            "frozen_event"
        ]
        .first()
    )

    setup_trade_count = len(
        setup_trade_event
    )

    setup_event_count = int(
        (
            setup_trade_event == 1
        ).sum()
    )

    setup_retained_count = int(
        (
            setup_trade_event == 0
        ).sum()
    )

    event_rate = (
        setup_event_count
        / setup_trade_count
        if setup_trade_count > 0
        else np.nan
    )

    for bar_number in range(1, 6):

        bar_df = setup_df[
            setup_df["bar_number"]
            == bar_number
        ]

        event_mae = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "cumulative_mae_r",
        ]

        retained_mae = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "cumulative_mae_r",
        ]

        event_adverse = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "bar_adverse_r",
        ]

        retained_adverse = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "bar_adverse_r",
        ]

        setup_rows.append(
            {
                "setup": setup,
                "bar_number": bar_number,
                "trade_count":
                    setup_trade_count,
                "event_count":
                    setup_event_count,
                "retained_count":
                    setup_retained_count,
                "event_rate":
                    event_rate,
                "event_rate_pct":
                    event_rate * 100
                    if not pd.isna(event_rate)
                    else np.nan,
                "event_mae_median":
                    median_or_nan(
                        event_mae
                    ),
                "retained_mae_median":
                    median_or_nan(
                        retained_mae
                    ),
                "mae_median_difference":
                    difference(
                        median_or_nan(
                            event_mae
                        ),
                        median_or_nan(
                            retained_mae
                        ),
                    ),
                "event_bar_adverse_median":
                    median_or_nan(
                        event_adverse
                    ),
                "retained_bar_adverse_median":
                    median_or_nan(
                        retained_adverse
                    ),
            }
        )

setup_output = pd.DataFrame(
    setup_rows
)

setup_output.to_csv(
    SETUP_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {SETUP_FILE}"
)


# ============================================================================
# 7. SYMBOL ROBUSTNESS
# ============================================================================

print()
print("-" * 78)
print("7. SYMBOL ROBUSTNESS")
print("-" * 78)

symbol_rows = []

for symbol, symbol_df in (
    df.groupby("symbol")
):

    symbol_trade_event = (
        symbol_df
        .groupby("trade_id")[
            "frozen_event"
        ]
        .first()
    )

    symbol_trade_count = len(
        symbol_trade_event
    )

    symbol_event_count = int(
        (
            symbol_trade_event == 1
        ).sum()
    )

    symbol_retained_count = int(
        (
            symbol_trade_event == 0
        ).sum()
    )

    event_rate = (
        symbol_event_count
        / symbol_trade_count
        if symbol_trade_count > 0
        else np.nan
    )

    for bar_number in range(1, 6):

        bar_df = symbol_df[
            symbol_df["bar_number"]
            == bar_number
        ]

        event_mae = bar_df.loc[
            bar_df["frozen_event"] == 1,
            "cumulative_mae_r",
        ]

        retained_mae = bar_df.loc[
            bar_df["frozen_event"] == 0,
            "cumulative_mae_r",
        ]

        symbol_rows.append(
            {
                "symbol": symbol,
                "bar_number": bar_number,
                "trade_count":
                    symbol_trade_count,
                "event_count":
                    symbol_event_count,
                "retained_count":
                    symbol_retained_count,
                "event_rate":
                    event_rate,
                "event_rate_pct":
                    event_rate * 100
                    if not pd.isna(event_rate)
                    else np.nan,
                "event_mae_median":
                    median_or_nan(
                        event_mae
                    ),
                "retained_mae_median":
                    median_or_nan(
                        retained_mae
                    ),
                "mae_median_difference":
                    difference(
                        median_or_nan(
                            event_mae
                        ),
                        median_or_nan(
                            retained_mae
                        ),
                    ),
            }
        )

symbol_output = pd.DataFrame(
    symbol_rows
)

symbol_output.to_csv(
    SYMBOL_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {SYMBOL_FILE}"
)


# ============================================================================
# 8. CROSS-THRESHOLD 0.25R STATE ANALYSIS
# ============================================================================

print()
print("-" * 78)
print("8. 0.25R CROSSING STATE")
print("-" * 78)

crossing_rows = []

for bar_number in range(1, 6):

    bar_df = df[
        df["bar_number"] == bar_number
    ]

    for event_value, label in [
        (1, "EVENT"),
        (0, "RETAINED"),
    ]:

        subset = bar_df[
            bar_df["frozen_event"]
            == event_value
        ]

        n = len(subset)

        crossed = int(
            (
                subset[
                    "crossed_0_25r_by_this_bar"
                ]
                == 1
            ).sum()
        )

        crossing_rows.append(
            {
                "bar_number": bar_number,
                "group": label,
                "n_trades": n,
                "crossed_0_25r_count":
                    crossed,
                "crossed_0_25r_rate_pct":
                    (
                        crossed / n * 100
                        if n > 0
                        else np.nan
                    ),
            }
        )

crossing_output = pd.DataFrame(
    crossing_rows
)

crossing_file = (
    OUTPUT_DIR
    / "early_adverse_path_025_crossing_state.csv"
)

crossing_output.to_csv(
    crossing_file,
    index=False,
)

print(
    f"[PASS] Saved: {crossing_file}"
)


# ============================================================================
# 9. INTEGRITY / MISSINGNESS
# ============================================================================

print()
print("-" * 78)
print("9. ANALYSIS INTEGRITY")
print("-" * 78)

integrity_rows = []


def add_integrity(
    check,
    expected,
    observed,
    status,
):
    integrity_rows.append(
        {
            "check": check,
            "expected": expected,
            "observed": observed,
            "status": status,
        }
    )


add_integrity(
    "input_file_exists",
    True,
    INPUT_FILE.exists(),
    "PASS"
    if INPUT_FILE.exists()
    else "FAIL",
)

add_integrity(
    "frozen_trade_count",
    EXPECTED_TRADES,
    trade_count,
    "PASS"
    if trade_count == EXPECTED_TRADES
    else "FAIL",
)

add_integrity(
    "path_row_count",
    EXPECTED_PATH_ROWS,
    len(df),
    "PASS"
    if len(df) == EXPECTED_PATH_ROWS
    else "FAIL",
)

add_integrity(
    "frozen_event_count",
    EXPECTED_EVENTS,
    event_count,
    "PASS"
    if event_count == EXPECTED_EVENTS
    else "FAIL",
)

add_integrity(
    "frozen_retained_count",
    EXPECTED_RETAINED,
    retained_count,
    "PASS"
    if retained_count == EXPECTED_RETAINED
    else "FAIL",
)

add_integrity(
    "bars_per_trade_min",
    EXPECTED_BARS_PER_TRADE,
    int(bars_per_trade.min()),
    "PASS"
    if bars_per_trade.min()
    == EXPECTED_BARS_PER_TRADE
    else "FAIL",
)

add_integrity(
    "bars_per_trade_max",
    EXPECTED_BARS_PER_TRADE,
    int(bars_per_trade.max()),
    "PASS"
    if bars_per_trade.max()
    == EXPECTED_BARS_PER_TRADE
    else "FAIL",
)

add_integrity(
    "bad_bar_sequences",
    0,
    len(bad_sequences),
    "PASS"
    if len(bad_sequences) == 0
    else "FAIL",
)

add_integrity(
    "event_label_inconsistency",
    0,
    len(bad_event_consistency),
    "PASS"
    if len(bad_event_consistency) == 0
    else "FAIL",
)


# Missingness for core analysis variables.
for column in trajectory_metrics:

    missing = int(
        df[column].isna().sum()
    )

    # Missing values are not automatically a failure.
    # We record them explicitly so that observed-bar limitations
    # remain visible.

    add_integrity(
        f"missing_{column}",
        0,
        missing,
        "PASS"
        if missing == 0
        else "INFO",
    )


integrity_output = pd.DataFrame(
    integrity_rows
)

integrity_output.to_csv(
    INTEGRITY_FILE,
    index=False,
)

print(
    f"[PASS] Saved: {INTEGRITY_FILE}"
)


# ============================================================================
# 10. CONSOLE SUMMARY
# ============================================================================

print()
print("=" * 78)
print("FORENSIC SUMMARY")
print("=" * 78)

print(
    f"Frozen trades       : {trade_count}"
)

print(
    f"Frozen events       : {event_count}"
)

print(
    f"Frozen retained     : {retained_count}"
)

print(
    f"Path rows           : {len(df)}"
)

print()
print(
    "Cumulative 0.25R reach rate by bar:"
)

for bar_number in range(1, 6):

    bar_df = df[
        df["bar_number"] == bar_number
    ]

    event_group = bar_df[
        bar_df["frozen_event"] == 1
    ]

    retained_group = bar_df[
        bar_df["frozen_event"] == 0
    ]

    event_rate = (
        event_group[
            "cumulative_mae_r"
        ]
        >= FROZEN_THRESHOLD_R
    ).mean()

    retained_rate = (
        retained_group[
            "cumulative_mae_r"
        ]
        >= FROZEN_THRESHOLD_R
    ).mean()

    difference_pp = (
        event_rate
        - retained_rate
    ) * 100.0

    print(
        f"  Bar {bar_number}: "
        f"event={event_rate * 100:.2f}% | "
        f"retained={retained_rate * 100:.2f}% | "
        f"Δ={difference_pp:+.2f} pp"
    )


print()
print(
    "Production observation coverage:"
)

production_observed = (
    df.groupby("trade_id")[
        "production_observed"
    ]
    .first()
)

production_bars = (
    df.groupby("trade_id")[
        "production_bars_in_trade"
    ]
    .first()
)

print(
    f"  Bar-5 reached in production: "
    f"{int((production_bars >= 5).sum())}"
)

print(
    f"  Bar-5 not reached in production: "
    f"{int((production_bars < 5).sum())}"
)


# ============================================================================
# GUARDRAILS
# ============================================================================

print()
print("=" * 78)
print("RESEARCH GUARDRAILS")
print("=" * 78)

print(
    "1. Frozen event label remains stored production "
    "bar-5 >= 0.25R."
)

print(
    "2. Production 3-bar trigger is not being redefined."
)

print(
    "3. Extended 5-bar event is kept separate from the "
    "frozen production label."
)

print(
    "4. This analysis is descriptive / forensic only."
)

print(
    "5. No threshold, bar, candle feature, setup, "
    "direction, or symbol filter is promoted."
)

print(
    "6. Small-sample subgroup differences are treated "
    "as investigation clues only."
)

print(
    "7. Production, frozen data, and OOS holdout remain untouched."
)


# ============================================================================
# FINAL
# ============================================================================

print()
print("=" * 78)
print("FIRST-5-BAR PATH FORENSIC ANALYSIS — PASS")
print("=" * 78)

print(
    f"Trajectory output:"
)
print(
    f"  {TRAJECTORY_FILE}"
)

print(
    f"Earliest-warning output:"
)
print(
    f"  {EARLIEST_WARNING_FILE}"
)

print(
    f"Threshold output:"
)
print(
    f"  {THRESHOLD_FILE}"
)

print(
    f"Microstructure output:"
)
print(
    f"  {MICROSTRUCTURE_FILE}"
)

print(
    f"Direction output:"
)
print(
    f"  {DIRECTION_FILE}"
)

print(
    f"Setup output:"
)
print(
    f"  {SETUP_FILE}"
)

print(
    f"Symbol output:"
)
print(
    f"  {SYMBOL_FILE}"
)

print(
    f"Integrity output:"
)
print(
    f"  {INTEGRITY_FILE}"
)

print()
print(
    "No strategy rule has been promoted."
)

print(
    "No production/OOS data has been modified."
)

print("=" * 78)