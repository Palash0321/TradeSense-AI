"""
TradeSense-AI — Forensic Early-Adverse Bar 1→2 Trajectory Analysis
===================================================================

Purpose
-------
Research-only analysis of whether information available by the end of
Bar 2 provides an earlier warning signature for the frozen early-adverse
event population.

Frozen research population
--------------------------
106 trades
60 frozen events
46 retained

Frozen event definition
-----------------------
stored_frozen_event == 1

The frozen event label originates from the validated first-5-bar path
foundation:

    stored production early_adverse_r_bar_5 >= 0.25R

This script DOES NOT redefine that label.

Research questions
------------------
1. How do Bar 1 and Bar 2 path states differ between events and retained?
2. How much does the path change from Bar 1 to Bar 2?
3. Does adverse excursion accelerate?
4. Does favorable excursion improve?
5. Does net excursion recover or deteriorate?
6. Does close displacement move in the intended direction?
7. Are these effects visible within LONG and SHORT separately?
8. Are they visible within setup categories?
9. What are the raw distributions before any threshold is considered?

Critical guardrails
-------------------
- Research only.
- No production strategy changes.
- No OOS modification.
- No frozen-data modification.
- No deployment threshold selection.
- No optimization.
- No rule promotion.
- Frozen event label remains stored_frozen_event.
- Bar 1 and Bar 2 are observed post-entry bars from the validated path
  foundation.
- This script studies trajectory information; it does not claim causality.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_trajectory"
)

OUTPUT_DISTRIBUTION = (
    OUTPUT_DIR / "early_adverse_bar12_raw_distributions.csv"
)

OUTPUT_TRAJECTORY = (
    OUTPUT_DIR / "early_adverse_bar12_trajectory_per_trade.csv"
)

OUTPUT_EFFECTS = (
    OUTPUT_DIR / "early_adverse_bar12_effect_sizes.csv"
)

OUTPUT_DIRECTION = (
    OUTPUT_DIR / "early_adverse_bar12_direction.csv"
)

OUTPUT_SETUP = (
    OUTPUT_DIR / "early_adverse_bar12_setup.csv"
)

OUTPUT_STATE_TRANSITIONS = (
    OUTPUT_DIR / "early_adverse_bar12_state_transitions.csv"
)

OUTPUT_INTEGRITY = (
    OUTPUT_DIR / "early_adverse_bar12_integrity.csv"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46
EXPECTED_ROWS = 530

BAR_1 = 1
BAR_2 = 2


# ============================================================================
# REQUIRED INPUT COLUMNS
# ============================================================================

REQUIRED_COLUMNS = [
    "symbol",
    "direction",
    "entry_date",
    "entry_timestamp",
    "signal_timestamp",
    "setup",
    "bar_number",
    "bar_timestamp",
    "entry_price_reconstructed",
    "risk_per_unit",
    "bar_adverse_r",
    "bar_favorable_r",
    "cumulative_mae_r",
    "cumulative_mfe_r",
    "net_excursion_r",
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
    "stored_frozen_event",
]


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def median(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.median())


def mean(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def std(series):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        return np.nan
    return float(series.std(ddof=1))


def quantile(series, q):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float(series.quantile(q))


def make_trade_id(df):
    return (
        df["symbol"].astype(str)
        + "|"
        + df["direction"].astype(str)
        + "|"
        + df["entry_date"].astype(str)
        + "|"
        + df["entry_timestamp"].astype(str)
        + "|"
        + df["signal_timestamp"].astype(str)
        + "|"
        + df["setup"].astype(str)
    )


def safe_difference(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def safe_ratio(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan

    if b == 0:
        return np.nan

    return float(a / b)


# ============================================================================
# LOAD + VALIDATE FOUNDATION DATA
# ============================================================================

def load_data():
    if not INPUT_FILE.exists():
        fail(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 78)
    print("TRADEsENSE-AI — BAR 1→2 TRAJECTORY ANALYSIS")
    print("=" * 78)

    print(f"[INFO] Input file : {INPUT_FILE}")
    print(f"[INFO] Input rows : {len(df)}")
    print(f"[INFO] Input cols : {len(df.columns)}")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        fail(
            "Missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    df["trade_id"] = make_trade_id(df)

    trade_counts = df.groupby("trade_id").size()

    if len(trade_counts) != EXPECTED_TRADES:
        fail(
            f"Expected {EXPECTED_TRADES} trades, "
            f"observed {len(trade_counts)}"
        )

    if len(df) != EXPECTED_ROWS:
        fail(
            f"Expected {EXPECTED_ROWS} path rows, "
            f"observed {len(df)}"
        )

    bad_trade_lengths = trade_counts[
        trade_counts != 5
    ]

    if not bad_trade_lengths.empty:
        fail(
            "Not every trade contains exactly five path rows."
        )

    event_values = (
        pd.to_numeric(
            df["stored_frozen_event"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["frozen_event"] = event_values

    event_by_trade = (
        df.groupby("trade_id")["frozen_event"]
        .first()
    )

    observed_events = int(event_by_trade.sum())
    observed_retained = int(
        (event_by_trade == 0).sum()
    )

    if observed_events != EXPECTED_EVENTS:
        fail(
            f"Expected {EXPECTED_EVENTS} frozen events, "
            f"observed {observed_events}"
        )

    if observed_retained != EXPECTED_RETAINED:
        fail(
            f"Expected {EXPECTED_RETAINED} retained trades, "
            f"observed {observed_retained}"
        )

    # Ensure the event label is constant within each trade.
    label_variation = (
        df.groupby("trade_id")["frozen_event"]
        .nunique()
    )

    if (label_variation > 1).any():
        fail(
            "Frozen event label changes within at least one trade."
        )

    print()
    print("[PASS] Frozen population:")
    print(f"       Trades   : {len(event_by_trade)}")
    print(f"       Events   : {observed_events}")
    print(f"       Retained : {observed_retained}")
    print(f"       Path rows: {len(df)}")

    return df


# ============================================================================
# BUILD ONE ROW PER TRADE WITH BAR 1 + BAR 2 STATES
# ============================================================================

PATH_VARIABLES = [
    "bar_adverse_r",
    "bar_favorable_r",
    "cumulative_mae_r",
    "cumulative_mfe_r",
    "net_excursion_r",
    "close_from_entry_r",
    "close_move_r",
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


def build_trade_level(df):
    bar1 = (
        df[df["bar_number"] == BAR_1]
        .copy()
        .set_index("trade_id")
    )

    bar2 = (
        df[df["bar_number"] == BAR_2]
        .copy()
        .set_index("trade_id")
    )

    if len(bar1) != EXPECTED_TRADES:
        fail(
            f"Expected {EXPECTED_TRADES} Bar-1 rows, "
            f"observed {len(bar1)}"
        )

    if len(bar2) != EXPECTED_TRADES:
        fail(
            f"Expected {EXPECTED_TRADES} Bar-2 rows, "
            f"observed {len(bar2)}"
        )

    trade = bar1[
        [
            "symbol",
            "direction",
            "entry_date",
            "entry_timestamp",
            "signal_timestamp",
            "setup",
            "frozen_event",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # BAR 1 / BAR 2 RAW STATES
    # ------------------------------------------------------------------

    for variable in PATH_VARIABLES:
        trade[f"bar1_{variable}"] = pd.to_numeric(
            bar1[variable],
            errors="coerce"
        )

        trade[f"bar2_{variable}"] = pd.to_numeric(
            bar2[variable],
            errors="coerce"
        )

    # ------------------------------------------------------------------
    # BAR 1 → BAR 2 CHANGES
    # ------------------------------------------------------------------

    for variable in PATH_VARIABLES:
        trade[f"delta_{variable}"] = (
            trade[f"bar2_{variable}"]
            - trade[f"bar1_{variable}"]
        )

    # ------------------------------------------------------------------
    # SPECIFIC TRAJECTORY FEATURES
    # ------------------------------------------------------------------

    # Cumulative adverse development.
    trade["delta_mae_r"] = (
        trade["bar2_cumulative_mae_r"]
        - trade["bar1_cumulative_mae_r"]
    )

    # Cumulative favorable development.
    trade["delta_mfe_r"] = (
        trade["bar2_cumulative_mfe_r"]
        - trade["bar1_cumulative_mfe_r"]
    )

    # Net path development.
    trade["delta_net_excursion_r"] = (
        trade["bar2_net_excursion_r"]
        - trade["bar1_net_excursion_r"]
    )

    # Close displacement development.
    trade["delta_close_from_entry_r"] = (
        trade["bar2_close_from_entry_r"]
        - trade["bar1_close_from_entry_r"]
    )

    # Single-bar adverse movement change.
    trade["delta_bar_adverse_r"] = (
        trade["bar2_bar_adverse_r"]
        - trade["bar1_bar_adverse_r"]
    )

    # Single-bar favorable movement change.
    trade["delta_bar_favorable_r"] = (
        trade["bar2_bar_favorable_r"]
        - trade["bar1_bar_favorable_r"]
    )

    # ------------------------------------------------------------------
    # BALANCE FEATURES
    # ------------------------------------------------------------------

    trade["bar1_mae_minus_mfe"] = (
        trade["bar1_cumulative_mae_r"]
        - trade["bar1_cumulative_mfe_r"]
    )

    trade["bar2_mae_minus_mfe"] = (
        trade["bar2_cumulative_mae_r"]
        - trade["bar2_cumulative_mfe_r"]
    )

    trade["delta_mae_minus_mfe"] = (
        trade["bar2_mae_minus_mfe"]
        - trade["bar1_mae_minus_mfe"]
    )

    trade["bar1_mae_to_mfe"] = trade.apply(
        lambda row: safe_ratio(
            row["bar1_cumulative_mae_r"],
            row["bar1_cumulative_mfe_r"],
        ),
        axis=1,
    )

    trade["bar2_mae_to_mfe"] = trade.apply(
        lambda row: safe_ratio(
            row["bar2_cumulative_mae_r"],
            row["bar2_cumulative_mfe_r"],
        ),
        axis=1,
    )

    # ------------------------------------------------------------------
    # EARLY-WARNING DESCRIPTIVE FLAGS
    # ------------------------------------------------------------------

    trade["bar2_mae_increased"] = (
        trade["delta_mae_r"] > 0
    ).astype(int)

    trade["bar2_mfe_increased"] = (
        trade["delta_mfe_r"] > 0
    ).astype(int)

    trade["bar2_net_deteriorated"] = (
        trade["delta_net_excursion_r"] < 0
    ).astype(int)

    trade["bar2_close_deteriorated"] = (
        trade["delta_close_from_entry_r"] < 0
    ).astype(int)

    trade["bar2_adverse_worsened"] = (
        trade["delta_bar_adverse_r"] > 0
    ).astype(int)

    trade["bar2_favorable_failed_to_improve"] = (
        trade["delta_bar_favorable_r"] <= 0
    ).astype(int)

    # Combination:
    # adverse path increases while favorable path does not improve.
    trade["adverse_up_favorable_not_up"] = (
        (trade["delta_mae_r"] > 0)
        & (trade["delta_mfe_r"] <= 0)
    ).astype(int)

    # Combination:
    # adverse path increases and net path deteriorates.
    trade["adverse_up_net_down"] = (
        (trade["delta_mae_r"] > 0)
        & (trade["delta_net_excursion_r"] < 0)
    ).astype(int)

    # Combination:
    # adverse single-bar excursion increases while close displacement
    # moves further against the intended direction.
    trade["adverse_up_close_down"] = (
        (trade["delta_bar_adverse_r"] > 0)
        & (trade["delta_close_from_entry_r"] < 0)
    ).astype(int)

    # Stronger descriptive combination.
    trade["adverse_up_favorable_not_up_net_down"] = (
        (trade["delta_mae_r"] > 0)
        & (trade["delta_mfe_r"] <= 0)
        & (trade["delta_net_excursion_r"] < 0)
    ).astype(int)

    trade = trade.reset_index()

    return trade


# ============================================================================
# RAW DISTRIBUTION ANALYSIS
# ============================================================================

DISTRIBUTION_VARIABLES = [
    # Bar 1 states
    "bar1_cumulative_mae_r",
    "bar1_cumulative_mfe_r",
    "bar1_net_excursion_r",
    "bar1_close_from_entry_r",
    "bar1_bar_adverse_r",
    "bar1_bar_favorable_r",

    # Bar 2 states
    "bar2_cumulative_mae_r",
    "bar2_cumulative_mfe_r",
    "bar2_net_excursion_r",
    "bar2_close_from_entry_r",
    "bar2_bar_adverse_r",
    "bar2_bar_favorable_r",

    # Trajectory changes
    "delta_mae_r",
    "delta_mfe_r",
    "delta_net_excursion_r",
    "delta_close_from_entry_r",
    "delta_bar_adverse_r",
    "delta_bar_favorable_r",
    "delta_mae_minus_mfe",
]


def build_distribution_table(trade):
    records = []

    for variable in DISTRIBUTION_VARIABLES:
        event_values = trade.loc[
            trade["frozen_event"] == 1,
            variable,
        ].dropna()

        retained_values = trade.loc[
            trade["frozen_event"] == 0,
            variable,
        ].dropna()

        records.append(
            {
                "variable": variable,
                "event_n": len(event_values),
                "retained_n": len(retained_values),

                "event_mean": mean(event_values),
                "retained_mean": mean(retained_values),
                "mean_difference": (
                    mean(event_values)
                    - mean(retained_values)
                ),

                "event_median": median(event_values),
                "retained_median": median(retained_values),
                "median_difference": (
                    median(event_values)
                    - median(retained_values)
                ),

                "event_std": std(event_values),
                "retained_std": std(retained_values),

                "event_p25": quantile(event_values, 0.25),
                "event_p75": quantile(event_values, 0.75),

                "retained_p25": quantile(
                    retained_values,
                    0.25,
                ),
                "retained_p75": quantile(
                    retained_values,
                    0.75,
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# EFFECT SIZE SUMMARY
# ============================================================================

def build_effect_table(trade):
    records = []

    effect_variables = [
        "delta_mae_r",
        "delta_mfe_r",
        "delta_net_excursion_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
        "delta_mae_minus_mfe",
    ]

    for variable in effect_variables:
        event = trade.loc[
            trade["frozen_event"] == 1,
            variable,
        ].dropna()

        retained = trade.loc[
            trade["frozen_event"] == 0,
            variable,
        ].dropna()

        event_median = median(event)
        retained_median = median(retained)

        event_mean = mean(event)
        retained_mean = mean(retained)

        records.append(
            {
                "feature": variable,
                "event_n": len(event),
                "retained_n": len(retained),

                "event_mean": event_mean,
                "retained_mean": retained_mean,
                "mean_difference": (
                    event_mean
                    - retained_mean
                ),

                "event_median": event_median,
                "retained_median": retained_median,
                "median_difference": (
                    event_median
                    - retained_median
                ),

                "event_std": std(event),
                "retained_std": std(retained),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# DIRECTION ANALYSIS
# ============================================================================

def build_direction_analysis(trade):
    records = []

    features = [
        "bar1_cumulative_mae_r",
        "bar2_cumulative_mae_r",
        "delta_mae_r",
        "bar1_cumulative_mfe_r",
        "bar2_cumulative_mfe_r",
        "delta_mfe_r",
        "bar1_net_excursion_r",
        "bar2_net_excursion_r",
        "delta_net_excursion_r",
        "bar1_close_from_entry_r",
        "bar2_close_from_entry_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]

    for direction in sorted(
        trade["direction"].dropna().unique()
    ):
        subset = trade[
            trade["direction"] == direction
        ]

        for feature in features:
            event = subset.loc[
                subset["frozen_event"] == 1,
                feature,
            ].dropna()

            retained = subset.loc[
                subset["frozen_event"] == 0,
                feature,
            ].dropna()

            records.append(
                {
                    "direction": direction,
                    "feature": feature,
                    "event_n": len(event),
                    "retained_n": len(retained),
                    "event_median": median(event),
                    "retained_median": median(retained),
                    "median_difference": (
                        median(event)
                        - median(retained)
                    ),
                    "event_mean": mean(event),
                    "retained_mean": mean(retained),
                    "mean_difference": (
                        mean(event)
                        - mean(retained)
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================================
# SETUP ANALYSIS
# ============================================================================

def build_setup_analysis(trade):
    records = []

    features = [
        "bar1_cumulative_mae_r",
        "bar2_cumulative_mae_r",
        "delta_mae_r",
        "bar1_cumulative_mfe_r",
        "bar2_cumulative_mfe_r",
        "delta_mfe_r",
        "bar1_net_excursion_r",
        "bar2_net_excursion_r",
        "delta_net_excursion_r",
        "bar1_close_from_entry_r",
        "bar2_close_from_entry_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]

    for setup in sorted(
        trade["setup"].dropna().unique()
    ):
        subset = trade[
            trade["setup"] == setup
        ]

        for feature in features:
            event = subset.loc[
                subset["frozen_event"] == 1,
                feature,
            ].dropna()

            retained = subset.loc[
                subset["frozen_event"] == 0,
                feature,
            ].dropna()

            records.append(
                {
                    "setup": setup,
                    "feature": feature,
                    "event_n": len(event),
                    "retained_n": len(retained),
                    "event_median": median(event),
                    "retained_median": median(retained),
                    "median_difference": (
                        median(event)
                        - median(retained)
                    ),
                    "event_mean": mean(event),
                    "retained_mean": mean(retained),
                    "mean_difference": (
                        mean(event)
                        - mean(retained)
                    ),
                }
            )

    return pd.DataFrame(records)


# ============================================================================
# STATE TRANSITION ANALYSIS
# ============================================================================

def build_state_transition_analysis(trade):
    records = []

    total_events = int(
        (trade["frozen_event"] == 1).sum()
    )

    total_retained = int(
        (trade["frozen_event"] == 0).sum()
    )

    flag_columns = [
        "bar2_mae_increased",
        "bar2_mfe_increased",
        "bar2_net_deteriorated",
        "bar2_close_deteriorated",
        "bar2_adverse_worsened",
        "bar2_favorable_failed_to_improve",
        "adverse_up_favorable_not_up",
        "adverse_up_net_down",
        "adverse_up_close_down",
        "adverse_up_favorable_not_up_net_down",
    ]

    for flag in flag_columns:
        event_subset = trade[
            trade["frozen_event"] == 1
        ][flag]

        retained_subset = trade[
            trade["frozen_event"] == 0
        ][flag]

        event_count = int(event_subset.sum())
        retained_count = int(retained_subset.sum())

        event_rate = (
            event_count / total_events
            if total_events
            else np.nan
        )

        retained_rate = (
            retained_count / total_retained
            if total_retained
            else np.nan
        )

        records.append(
            {
                "state_transition": flag,

                "event_count": event_count,
                "event_total": total_events,
                "event_rate": event_rate,
                "event_rate_pct": (
                    event_rate * 100
                    if not pd.isna(event_rate)
                    else np.nan
                ),

                "retained_count": retained_count,
                "retained_total": total_retained,
                "retained_rate": retained_rate,
                "retained_rate_pct": (
                    retained_rate * 100
                    if not pd.isna(retained_rate)
                    else np.nan
                ),

                "event_minus_retained_pp": (
                    (event_rate - retained_rate) * 100
                    if not (
                        pd.isna(event_rate)
                        or pd.isna(retained_rate)
                    )
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# INTEGRITY
# ============================================================================

def build_integrity(df, trade):
    records = []

    records.append(
        {
            "check": "input_row_count",
            "expected": EXPECTED_ROWS,
            "observed": len(df),
            "status": (
                "PASS"
                if len(df) == EXPECTED_ROWS
                else "FAIL"
            ),
        }
    )

    records.append(
        {
            "check": "trade_count",
            "expected": EXPECTED_TRADES,
            "observed": trade["trade_id"].nunique(),
            "status": (
                "PASS"
                if trade["trade_id"].nunique()
                == EXPECTED_TRADES
                else "FAIL"
            ),
        }
    )

    records.append(
        {
            "check": "event_count",
            "expected": EXPECTED_EVENTS,
            "observed": int(
                trade["frozen_event"].sum()
            ),
            "status": (
                "PASS"
                if int(trade["frozen_event"].sum())
                == EXPECTED_EVENTS
                else "FAIL"
            ),
        }
    )

    records.append(
        {
            "check": "retained_count",
            "expected": EXPECTED_RETAINED,
            "observed": int(
                (trade["frozen_event"] == 0).sum()
            ),
            "status": (
                "PASS"
                if int(
                    (trade["frozen_event"] == 0).sum()
                )
                == EXPECTED_RETAINED
                else "FAIL"
            ),
        }
    )

    records.append(
        {
            "check": "bar1_rows",
            "expected": EXPECTED_TRADES,
            "observed": int(
                (df["bar_number"] == BAR_1).sum()
            ),
            "status": (
                "PASS"
                if int(
                    (df["bar_number"] == BAR_1).sum()
                )
                == EXPECTED_TRADES
                else "FAIL"
            ),
        }
    )

    records.append(
        {
            "check": "bar2_rows",
            "expected": EXPECTED_TRADES,
            "observed": int(
                (df["bar_number"] == BAR_2).sum()
            ),
            "status": (
                "PASS"
                if int(
                    (df["bar_number"] == BAR_2).sum()
                )
                == EXPECTED_TRADES
                else "FAIL"
            ),
        }
    )

    # Every trajectory feature must have 106 trade-level rows.
    for feature in [
        "delta_mae_r",
        "delta_mfe_r",
        "delta_net_excursion_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]:
        records.append(
            {
                "check": f"{feature}_trade_rows",
                "expected": EXPECTED_TRADES,
                "observed": int(
                    trade[feature].notna().sum()
                ),
                "status": (
                    "PASS"
                    if int(
                        trade[feature].notna().sum()
                    )
                    == EXPECTED_TRADES
                    else "FAIL"
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    print()
    print("-" * 78)
    print("BUILDING BAR 1 → BAR 2 TRAJECTORY")
    print("-" * 78)

    trade = build_trade_level(df)

    print(
        f"[PASS] Trade-level trajectory rows: "
        f"{len(trade)}"
    )

    # ------------------------------------------------------------------
    # 1. RAW DISTRIBUTIONS
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("1. RAW DISTRIBUTION ANALYSIS")
    print("-" * 78)

    distribution = build_distribution_table(trade)
    distribution.to_csv(
        OUTPUT_DISTRIBUTION,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_DISTRIBUTION}"
    )

    # ------------------------------------------------------------------
    # 2. TRAJECTORY PER TRADE
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("2. PER-TRADE BAR 1 → BAR 2 TRAJECTORY")
    print("-" * 78)

    trade.to_csv(
        OUTPUT_TRAJECTORY,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_TRAJECTORY}"
    )

    # ------------------------------------------------------------------
    # 3. EFFECT SIZES
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("3. TRAJECTORY EFFECT SIZES")
    print("-" * 78)

    effects = build_effect_table(trade)

    effects.to_csv(
        OUTPUT_EFFECTS,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_EFFECTS}"
    )

    # ------------------------------------------------------------------
    # 4. DIRECTION
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("4. DIRECTION ROBUSTNESS")
    print("-" * 78)

    direction = build_direction_analysis(trade)

    direction.to_csv(
        OUTPUT_DIRECTION,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_DIRECTION}"
    )

    # ------------------------------------------------------------------
    # 5. SETUP
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("5. SETUP ROBUSTNESS")
    print("-" * 78)

    setup = build_setup_analysis(trade)

    setup.to_csv(
        OUTPUT_SETUP,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_SETUP}"
    )

    # ------------------------------------------------------------------
    # 6. STATE TRANSITIONS
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("6. BAR 1 → BAR 2 STATE TRANSITIONS")
    print("-" * 78)

    transitions = build_state_transition_analysis(
        trade
    )

    transitions.to_csv(
        OUTPUT_STATE_TRANSITIONS,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_STATE_TRANSITIONS}"
    )

    # ------------------------------------------------------------------
    # 7. INTEGRITY
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("7. ANALYSIS INTEGRITY")
    print("-" * 78)

    integrity = build_integrity(
        df,
        trade,
    )

    integrity.to_csv(
        OUTPUT_INTEGRITY,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_INTEGRITY}"
    )

    failed = integrity[
        integrity["status"] != "PASS"
    ]

    if not failed.empty:
        print()
        print("[FAIL] Integrity checks failed:")
        print(failed.to_string(index=False))
        sys.exit(1)

    # ------------------------------------------------------------------
    # 8. CONSOLE SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print("BAR 1 → BAR 2 TRAJECTORY FORENSIC SUMMARY")
    print("=" * 78)

    print(
        f"Frozen trades : {EXPECTED_TRADES}"
    )

    print(
        f"Frozen events : {EXPECTED_EVENTS}"
    )

    print(
        f"Retained      : {EXPECTED_RETAINED}"
    )

    print()

    summary_features = [
        "delta_mae_r",
        "delta_mfe_r",
        "delta_net_excursion_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]

    for feature in summary_features:
        row = effects[
            effects["feature"] == feature
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        print(
            f"{feature:<32} "
            f"event median={row['event_median']:+.4f}R  "
            f"retained median={row['retained_median']:+.4f}R  "
            f"Δ={row['median_difference']:+.4f}R"
        )

    print()
    print("State-transition rates:")
    print(
        transitions[
            [
                "state_transition",
                "event_rate_pct",
                "retained_rate_pct",
                "event_minus_retained_pp",
            ]
        ].to_string(index=False)
    )

    print()
    print("GUARDRAILS:")
    print("1. Frozen event label remains stored_frozen_event.")
    print("2. Bar 1 and Bar 2 are observed path states only.")
    print("3. No production/OOS data modified.")
    print("4. No threshold selected.")
    print("5. No deployment rule promoted.")
    print("6. Results are descriptive forensic evidence only.")

    print()
    print("BAR 1 → BAR 2 TRAJECTORY FOUNDATION — PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()