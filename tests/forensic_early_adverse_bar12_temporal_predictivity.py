"""
TRADE SENSE-AI
BAR 1 -> BAR 2 TEMPORAL PREDICTIVITY FORENSIC ANALYSIS

Purpose
-------
Research-only forensic analysis of whether information available by Bar 1
and Bar 2 has stable association with the frozen early-adverse event label.

Frozen research population
--------------------------
106 trades
60 eventual early-adverse events
46 retained trades

Frozen event definition
-----------------------
Stored production:
    early_adverse_r_bar_5 >= 0.25R

Important separation
--------------------
This script DOES NOT:
- modify production code
- modify frozen trade data
- modify OOS data
- optimize a deployment threshold
- create a trading rule
- use future bars beyond the information timestamp
- use final MAE/MFE/exit information as predictors

The purpose is to test whether the already identified Bar 1 -> Bar 2
trajectory information is temporally and cross-sectional robust.

Information sets
----------------
BAR1:
    Information available from Bar 1 only.

BAR12:
    Information available after Bar 2.

Primary state families
----------------------
1. bar2_mfe_increased
2. bar2_net_deteriorated
3. bar2_favorable_failed_to_improve
4. adverse_up_net_down

Robustness
----------
- calendar year
- symbol leave-one-out
- setup
- direction
- effect-size stability
- Bar1 vs Bar12 incremental comparison

No deployment decision is made by this script.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_temporal_predictivity"
)

OUTPUT_MAIN = OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity.csv"
OUTPUT_YEAR = OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_year.csv"
OUTPUT_SYMBOL_LOO = (
    OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_symbol_loo.csv"
)
OUTPUT_SETUP = OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_setup.csv"
OUTPUT_DIRECTION = (
    OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_direction.csv"
)
OUTPUT_EFFECTS = (
    OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_effects.csv"
)
OUTPUT_STABILITY = (
    OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_stability.csv"
)
OUTPUT_INTEGRITY = (
    OUTPUT_DIR / "early_adverse_bar12_temporal_predictivity_integrity.csv"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

ALPHA = 0.05


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denominator = 1.0 + (z ** 2) / total
    centre = (
        p + (z ** 2) / (2.0 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / total
                + (z ** 2) / (4.0 * total ** 2)
            )
        )
        / denominator
    )

    return centre - margin, centre + margin


def risk_difference_ci(
    event_success: int,
    event_total: int,
    retained_success: int,
    retained_total: int,
):
    """
    Conservative independent Wilson/Newcombe-style interval for
    difference in two proportions.
    """

    event_low, event_high = wilson_interval(
        event_success,
        event_total,
    )

    retained_low, retained_high = wilson_interval(
        retained_success,
        retained_total,
    )

    rd = (
        event_success / event_total
        - retained_success / retained_total
    )

    low = event_low - retained_high
    high = event_high - retained_low

    return rd, low, high


def fisher_test(
    event_success: int,
    event_total: int,
    retained_success: int,
    retained_total: int,
):
    table = np.array(
        [
            [
                event_success,
                event_total - event_success,
            ],
            [
                retained_success,
                retained_total - retained_success,
            ],
        ],
        dtype=int,
    )

    odds_ratio, p_value = fisher_exact(
        table,
        alternative="two-sided",
    )

    return odds_ratio, p_value


def build_binary_table(
    df: pd.DataFrame,
    state: str,
):
    event_df = df[df["frozen_event"] == 1]
    retained_df = df[df["frozen_event"] == 0]

    event_total = len(event_df)
    retained_total = len(retained_df)

    event_success = int(event_df[state].fillna(False).sum())
    retained_success = int(retained_df[state].fillna(False).sum())

    return (
        event_success,
        event_total,
        retained_success,
        retained_total,
    )


def state_statistics(
    df: pd.DataFrame,
    state: str,
):
    (
        event_success,
        event_total,
        retained_success,
        retained_total,
    ) = build_binary_table(df, state)

    if event_total == 0 or retained_total == 0:
        return {
            "event_success": event_success,
            "event_total": event_total,
            "event_rate_pct": np.nan,
            "retained_success": retained_success,
            "retained_total": retained_total,
            "retained_rate_pct": np.nan,
            "risk_difference_pp": np.nan,
            "risk_difference_ci_low_pp": np.nan,
            "risk_difference_ci_high_pp": np.nan,
            "odds_ratio": np.nan,
            "fisher_p": np.nan,
        }

    rd, ci_low, ci_high = risk_difference_ci(
        event_success,
        event_total,
        retained_success,
        retained_total,
    )

    odds_ratio, p_value = fisher_test(
        event_success,
        event_total,
        retained_success,
        retained_total,
    )

    return {
        "event_success": event_success,
        "event_total": event_total,
        "event_rate_pct": 100.0 * event_success / event_total,
        "retained_success": retained_success,
        "retained_total": retained_total,
        "retained_rate_pct": 100.0 * retained_success / retained_total,
        "risk_difference_pp": 100.0 * rd,
        "risk_difference_ci_low_pp": 100.0 * ci_low,
        "risk_difference_ci_high_pp": 100.0 * ci_high,
        "odds_ratio": odds_ratio,
        "fisher_p": p_value,
    }


def effect_size_stability(values: List[float]):
    clean = [
        float(x)
        for x in values
        if x is not None and np.isfinite(x)
    ]

    if not clean:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "min": np.nan,
            "max": np.nan,
            "positive_fraction": np.nan,
        }

    return {
        "n": len(clean),
        "median": float(np.median(clean)),
        "mean": float(np.mean(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "positive_fraction": float(
            np.mean(np.asarray(clean) > 0)
        ),
    }


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def load_input():
    print("=" * 78)
    print("TRADE SENSE-AI — BAR 1→2 TEMPORAL PREDICTIVITY FORENSIC ANALYSIS")
    print("=" * 78)

    print(f"[INFO] Input file: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required validated trajectory file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"[INFO] Input rows: {len(df)}")
    print(f"[INFO] Input columns: {len(df.columns)}")

    required = {
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",

        "bar1_cumulative_mae_r",
        "bar2_cumulative_mae_r",

        "bar1_cumulative_mfe_r",
        "bar2_cumulative_mfe_r",

        "bar1_net_excursion_r",
        "bar2_net_excursion_r",

        "bar1_close_from_entry_r",
        "bar2_close_from_entry_r",

        "bar1_bar_adverse_r",
        "bar2_bar_adverse_r",

        "bar1_bar_favorable_r",
        "bar2_bar_favorable_r",

        "delta_cumulative_mae_r",
        "delta_cumulative_mfe_r",
        "delta_net_excursion_r",
        "delta_close_from_entry_r",

        "delta_bar_adverse_r",
        "delta_bar_favorable_r",

        "bar2_mfe_increased",
        "bar2_net_deteriorated",
        "bar2_favorable_failed_to_improve",
        "adverse_up_net_down",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    trade_ids = (
        df[
            [
                "symbol",
                "direction",
                "entry_date",
            ]
        ]
        .astype(str)
        .drop_duplicates()
    )

    trade_count = len(trade_ids)
    event_count = int(
        df.groupby(
            [
                "symbol",
                "direction",
                "entry_date",
            ]
        )["frozen_event"]
        .first()
        .sum()
    )

    retained_count = trade_count - event_count

    if trade_count != EXPECTED_TRADES:
        raise ValueError(
            f"Frozen trade count mismatch: "
            f"expected {EXPECTED_TRADES}, observed {trade_count}"
        )

    if event_count != EXPECTED_EVENTS:
        raise ValueError(
            f"Frozen event count mismatch: "
            f"expected {EXPECTED_EVENTS}, observed {event_count}"
        )

    if retained_count != EXPECTED_RETAINED:
        raise ValueError(
            f"Frozen retained count mismatch: "
            f"expected {EXPECTED_RETAINED}, observed {retained_count}"
        )

    print("[PASS] Frozen population:")
    print(f"       Trades   : {trade_count}")
    print(f"       Events   : {event_count}")
    print(f"       Retained : {retained_count}")
    print(f"       Path rows: {len(df)}")

    return df


# ============================================================================
# PRIMARY STATES
# ============================================================================

PRIMARY_STATES = [
    "bar2_mfe_increased",
    "bar2_net_deteriorated",
    "bar2_favorable_failed_to_improve",
    "adverse_up_net_down",
]


# ============================================================================
# 1. MAIN TEMPORAL COMPARISON
# ============================================================================

def analyze_main(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("1. BAR 1 VS BAR 1→2 TEMPORAL INFORMATION")
    print("=" * 78)

    rows = []

    event_df = df[df["frozen_event"] == 1]
    retained_df = df[df["frozen_event"] == 0]

    features = [
        "bar1_cumulative_mae_r",
        "bar1_cumulative_mfe_r",
        "bar1_net_excursion_r",
        "bar1_close_from_entry_r",
        "bar1_bar_adverse_r",
        "bar1_bar_favorable_r",

        "bar2_cumulative_mae_r",
        "bar2_cumulative_mfe_r",
        "bar2_net_excursion_r",
        "bar2_close_from_entry_r",
        "bar2_bar_adverse_r",
        "bar2_bar_favorable_r",

        "delta_cumulative_mae_r",
        "delta_cumulative_mfe_r",
        "delta_net_excursion_r",
        "delta_close_from_entry_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]

    for feature in features:
        event_values = pd.to_numeric(
            event_df[feature],
            errors="coerce",
        ).dropna()

        retained_values = pd.to_numeric(
            retained_df[feature],
            errors="coerce",
        ).dropna()

        event_median = (
            float(event_values.median())
            if not event_values.empty
            else np.nan
        )

        retained_median = (
            float(retained_values.median())
            if not retained_values.empty
            else np.nan
        )

        event_mean = (
            float(event_values.mean())
            if not event_values.empty
            else np.nan
        )

        retained_mean = (
            float(retained_values.mean())
            if not retained_values.empty
            else np.nan
        )

        rows.append(
            {
                "information_stage": (
                    "BAR1"
                    if feature.startswith("bar1_")
                    else (
                        "BAR2_SNAPSHOT"
                        if feature.startswith("bar2_")
                        else "BAR2_INCREMENTAL"
                    )
                ),
                "feature": feature,
                "event_n": len(event_values),
                "retained_n": len(retained_values),
                "event_median": event_median,
                "retained_median": retained_median,
                "median_difference_event_minus_retained": (
                    event_median - retained_median
                    if np.isfinite(event_median)
                    and np.isfinite(retained_median)
                    else np.nan
                ),
                "event_mean": event_mean,
                "retained_mean": retained_mean,
                "mean_difference_event_minus_retained": (
                    event_mean - retained_mean
                    if np.isfinite(event_mean)
                    and np.isfinite(retained_mean)
                    else np.nan
                ),
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_MAIN, index=False)

    print(f"[PASS] Saved: {OUTPUT_MAIN}")

    return output


# ============================================================================
# 2. YEAR ROBUSTNESS
# ============================================================================

def analyze_year_robustness(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("2. CALENDAR-YEAR ROBUSTNESS")
    print("=" * 78)

    work = df.copy()

    work["entry_date_parsed"] = pd.to_datetime(
        work["entry_date"],
        errors="coerce",
    )

    if work["entry_date_parsed"].isna().any():
        raise ValueError(
            "Could not parse one or more entry dates."
        )

    work["year"] = work["entry_date_parsed"].dt.year

    rows = []

    for year in sorted(work["year"].unique()):
        subset = work[work["year"] == year]

        for state in PRIMARY_STATES:
            if state not in subset.columns:
                continue

            stats = state_statistics(
                subset,
                state,
            )

            rows.append(
                {
                    "year": int(year),
                    "state": state,
                    "trade_count": len(subset),
                    **stats,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_YEAR, index=False)

    print(f"[PASS] Saved: {OUTPUT_YEAR}")

    return output


# ============================================================================
# 3. SYMBOL LEAVE-ONE-OUT
# ============================================================================

def analyze_symbol_loo(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("3. SYMBOL LEAVE-ONE-OUT ROBUSTNESS")
    print("=" * 78)

    rows = []

    for symbol in sorted(
        df["symbol"].dropna().astype(str).unique()
    ):
        subset = df[
            df["symbol"].astype(str) != symbol
        ].copy()

        for state in PRIMARY_STATES:
            stats = state_statistics(
                subset,
                state,
            )

            rows.append(
                {
                    "excluded_symbol": symbol,
                    "remaining_trade_count": len(subset),
                    "state": state,
                    **stats,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_SYMBOL_LOO, index=False)

    print(f"[PASS] Saved: {OUTPUT_SYMBOL_LOO}")

    return output


# ============================================================================
# 4. SETUP ROBUSTNESS
# ============================================================================

def analyze_setup(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("4. SETUP ROBUSTNESS")
    print("=" * 78)

    rows = []

    for setup in sorted(
        df["setup"].dropna().astype(str).unique()
    ):
        subset = df[
            df["setup"].astype(str) == setup
        ].copy()

        for state in PRIMARY_STATES:
            stats = state_statistics(
                subset,
                state,
            )

            rows.append(
                {
                    "setup": setup,
                    "trade_count": len(subset),
                    "state": state,
                    **stats,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_SETUP, index=False)

    print(f"[PASS] Saved: {OUTPUT_SETUP}")

    return output


# ============================================================================
# 5. DIRECTION ROBUSTNESS
# ============================================================================

def analyze_direction(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("5. DIRECTION ROBUSTNESS")
    print("=" * 78)

    rows = []

    for direction in sorted(
        df["direction"].dropna().astype(str).unique()
    ):
        subset = df[
            df["direction"].astype(str) == direction
        ].copy()

        for state in PRIMARY_STATES:
            stats = state_statistics(
                subset,
                state,
            )

            rows.append(
                {
                    "direction": direction,
                    "trade_count": len(subset),
                    "state": state,
                    **stats,
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_DIRECTION, index=False)

    print(f"[PASS] Saved: {OUTPUT_DIRECTION}")

    return output


# ============================================================================
# 6. PRIMARY EFFECTS
# ============================================================================

def analyze_primary_effects(df: pd.DataFrame):
    print()
    print("=" * 78)
    print("6. PRIMARY STATE EFFECT SIZES")
    print("=" * 78)

    rows = []

    for state in PRIMARY_STATES:
        stats = state_statistics(
            df,
            state,
        )

        rows.append(
            {
                "state": state,
                **stats,
            }
        )

    output = pd.DataFrame(rows)

    # Explicitly mark redundancy.
    output["state_family"] = [
        "favorable_development",
        "net_path_deterioration",
        "favorable_development",
        "net_path_deterioration",
    ]

    output.to_csv(OUTPUT_EFFECTS, index=False)

    print(output.to_string(index=False))
    print(f"[PASS] Saved: {OUTPUT_EFFECTS}")

    return output


# ============================================================================
# 7. STABILITY SUMMARY
# ============================================================================

def analyze_stability(
    year_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    setup_df: pd.DataFrame,
    direction_df: pd.DataFrame,
):
    print()
    print("=" * 78)
    print("7. EFFECT-SIZE STABILITY")
    print("=" * 78)

    rows = []

    for state in PRIMARY_STATES:
        year_values = (
            year_df.loc[
                year_df["state"] == state,
                "risk_difference_pp",
            ]
            .dropna()
            .tolist()
        )

        loo_values = (
            loo_df.loc[
                loo_df["state"] == state,
                "risk_difference_pp",
            ]
            .dropna()
            .tolist()
        )

        setup_values = (
            setup_df.loc[
                setup_df["state"] == state,
                "risk_difference_pp",
            ]
            .dropna()
            .tolist()
        )

        direction_values = (
            direction_df.loc[
                direction_df["state"] == state,
                "risk_difference_pp",
            ]
            .dropna()
            .tolist()
        )

        year_stats = effect_size_stability(
            year_values
        )

        loo_stats = effect_size_stability(
            loo_values
        )

        setup_stats = effect_size_stability(
            setup_values
        )

        direction_stats = effect_size_stability(
            direction_values
        )

        rows.append(
            {
                "state": state,

                "year_n": year_stats["n"],
                "year_median_rd_pp": year_stats["median"],
                "year_min_rd_pp": year_stats["min"],
                "year_max_rd_pp": year_stats["max"],
                "year_positive_fraction": year_stats[
                    "positive_fraction"
                ],

                "symbol_loo_n": loo_stats["n"],
                "symbol_loo_median_rd_pp": loo_stats["median"],
                "symbol_loo_min_rd_pp": loo_stats["min"],
                "symbol_loo_max_rd_pp": loo_stats["max"],
                "symbol_loo_positive_fraction": loo_stats[
                    "positive_fraction"
                ],

                "setup_n": setup_stats["n"],
                "setup_median_rd_pp": setup_stats["median"],
                "setup_min_rd_pp": setup_stats["min"],
                "setup_max_rd_pp": setup_stats["max"],
                "setup_positive_fraction": setup_stats[
                    "positive_fraction"
                ],

                "direction_n": direction_stats["n"],
                "direction_median_rd_pp": direction_stats["median"],
                "direction_min_rd_pp": direction_stats["min"],
                "direction_max_rd_pp": direction_stats["max"],
                "direction_positive_fraction": direction_stats[
                    "positive_fraction"
                ],
            }
        )

    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_STABILITY, index=False)

    print(output.to_string(index=False))
    print(f"[PASS] Saved: {OUTPUT_STABILITY}")

    return output


# ============================================================================
# 8. INTEGRITY
# ============================================================================

def build_integrity(
    df: pd.DataFrame,
    main_df: pd.DataFrame,
    year_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    setup_df: pd.DataFrame,
    direction_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    stability_df: pd.DataFrame,
):
    trade_ids = (
        df[
            [
                "symbol",
                "direction",
                "entry_date",
            ]
        ]
        .astype(str)
        .drop_duplicates()
    )

    trade_count = len(trade_ids)

    event_count = int(
        df.groupby(
            [
                "symbol",
                "direction",
                "entry_date",
            ]
        )["frozen_event"]
        .first()
        .sum()
    )

    retained_count = trade_count - event_count

    checks = [
        (
            "trade_count",
            EXPECTED_TRADES,
            trade_count,
            trade_count == EXPECTED_TRADES,
        ),
        (
            "event_count",
            EXPECTED_EVENTS,
            event_count,
            event_count == EXPECTED_EVENTS,
        ),
        (
            "retained_count",
            EXPECTED_RETAINED,
            retained_count,
            retained_count == EXPECTED_RETAINED,
        ),
        (
            "trade_level_input_rows",
            EXPECTED_TRADES,
            len(df),
            len(df) == EXPECTED_TRADES,
        ),
        (
            "main_rows",
            18,
            len(main_df),
            len(main_df) == 18,
        ),
        (
            "primary_state_count",
            4,
            len(effects_df),
            len(effects_df) == 4,
        ),
        (
            "year_rows",
            ">0",
            len(year_df),
            len(year_df) > 0,
        ),
        (
            "symbol_loo_rows",
            ">0",
            len(loo_df),
            len(loo_df) > 0,
        ),
        (
            "setup_rows",
            ">0",
            len(setup_df),
            len(setup_df) > 0,
        ),
        (
            "direction_rows",
            ">0",
            len(direction_df),
            len(direction_df) > 0,
        ),
        (
            "stability_rows",
            4,
            len(stability_df),
            len(stability_df) == 4,
        ),
    ]

    output = pd.DataFrame(
        [
            {
                "check": name,
                "expected": expected,
                "observed": observed,
                "pass": passed,
            }
            for name, expected, observed, passed in checks
        ]
    )

    output.to_csv(OUTPUT_INTEGRITY, index=False)

    print()
    print("=" * 78)
    print("8. INTEGRITY")
    print("=" * 78)
    print(output.to_string(index=False))

    return output


# ============================================================================
# FINAL INTERPRETATION
# ============================================================================

def print_summary(
    effects_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    integrity_df: pd.DataFrame,
):
    print()
    print("=" * 78)
    print("FORENSIC TEMPORAL-PREDICTIVITY SUMMARY")
    print("=" * 78)

    print(f"Frozen trades : {EXPECTED_TRADES}")
    print(f"Frozen events : {EXPECTED_EVENTS}")
    print(f"Retained      : {EXPECTED_RETAINED}")

    print()
    print("Interpretation guardrails:")
    print("1. Bar1 is the earlier information set.")
    print("2. Bar1→Bar2 states use only information available by Bar2.")
    print("3. No threshold optimization is performed.")
    print("4. The frozen event label remains the stored production bar-5 label.")
    print("5. Year/symbol/setup/direction checks are robustness diagnostics.")
    print("6. Significant states are not treated as independent discoveries.")
    print("7. Positive robustness does not establish causality.")
    print("8. Positive robustness does not establish deployment performance.")
    print("9. No strategy rule has been promoted.")
    print("10. Production code remains untouched.")
    print("11. Frozen/OOS data remain untouched.")

    print()
    print("Primary state effect sizes:")

    display_cols = [
        "state",
        "event_rate_pct",
        "retained_rate_pct",
        "risk_difference_pp",
        "risk_difference_ci_low_pp",
        "risk_difference_ci_high_pp",
        "fisher_p",
    ]

    print(
        effects_df[
            display_cols
        ].to_string(index=False)
    )

    print()
    print("Stability summary:")

    stability_cols = [
        "state",
        "year_positive_fraction",
        "symbol_loo_positive_fraction",
        "setup_positive_fraction",
        "direction_positive_fraction",
    ]

    print(
        stability_df[
            stability_cols
        ].to_string(index=False)
    )

    all_pass = bool(
        integrity_df["pass"].all()
    )

    print()
    if all_pass:
        print("BAR 1→2 TEMPORAL PREDICTIVITY FORENSIC ANALYSIS — PASS")
        print("No strategy rule has been promoted.")
        print("No production/OOS data has been modified.")
        print("FINAL STATUS: PASS")
    else:
        print("BAR 1→2 TEMPORAL PREDICTIVITY FORENSIC ANALYSIS — FAIL")
        print("FINAL STATUS: FAIL")


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_input()

    main_df = analyze_main(df)

    year_df = analyze_year_robustness(df)

    loo_df = analyze_symbol_loo(df)

    setup_df = analyze_setup(df)

    direction_df = analyze_direction(df)

    effects_df = analyze_primary_effects(df)

    stability_df = analyze_stability(
        year_df,
        loo_df,
        setup_df,
        direction_df,
    )

    integrity_df = build_integrity(
        df,
        main_df,
        year_df,
        loo_df,
        setup_df,
        direction_df,
        effects_df,
        stability_df,
    )

    print_summary(
        effects_df,
        stability_df,
        integrity_df,
    )


if __name__ == "__main__":
    main()