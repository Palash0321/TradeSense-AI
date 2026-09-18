"""
TradeSense-AI
VOLUME × INCREMENTAL INFORMATION FORENSIC AUDIT

Purpose
-------
Test whether Volume adds incremental information about the frozen
Early-Adverse event beyond a small pre-specified entry-context baseline.

Frozen hypothesis
-----------------
Early adverse >= 0.25R within first 5 entry bars.

Frozen population
-----------------
106 trades
60 events
46 retained

Design principles
-----------------
- Research-only.
- Frozen 106-trade population.
- Volume exposure is FIXED at the global median.
- Continuous baseline variables use FIXED global median splits.
- No optimized thresholds.
- No feature selection based on outcome.
- No ML model.
- No production changes.
- No OOS changes.

Baseline context
----------------
1. Entry Trend Strength
2. Entry Trend Slope
3. Entry ATR
4. Setup Confidence

Models / information comparisons
--------------------------------
A. Volume alone
B. Baseline context only
C. Baseline context + Volume

The analysis focuses on whether adding Volume preserves a meaningful
directional association and improves simple information criteria.

This is association / sensitivity analysis only.
It does not establish causality or create a deployment rule.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

VOLUME_FILE = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_early_adverse"
    / "volume_early_adverse_trade_level.csv"
)

FROZEN_FILE = (
    ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_incremental_information"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_BARS = 5

BASELINE_CONTROLS = [
    "TREND_STRENGTH",
    "TREND_SLOPE",
    "ENTRY_ATR",
    "SETUP_CONFIDENCE",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_column_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def resolve_column(
    df: pd.DataFrame,
    aliases: list[str],
    required: bool = False,
) -> str | None:

    normalized = {}

    for col in df.columns:
        normalized.setdefault(
            normalize_column_name(col),
            col,
        )

    for alias in aliases:
        key = normalize_column_name(alias)

        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not resolve required column. "
            f"Tried aliases={aliases}. "
            f"Available columns={list(df.columns)}"
        )

    return None


def standardize_symbol(value) -> str:
    text = str(value).strip().upper()

    if text.endswith(".NS"):
        text = text[:-3]

    return text


def standardize_direction(value) -> str:
    text = str(value).strip().upper()

    if text in {"LONG", "BUY", "1", "+1"}:
        return "LONG"

    if text in {"SHORT", "SELL", "-1"}:
        return "SHORT"

    return text


def standardize_date(value) -> str:
    parsed = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def make_trade_key(
    symbol,
    direction,
    entry_date,
) -> str:

    return (
        f"{standardize_symbol(symbol)}|"
        f"{standardize_direction(direction)}|"
        f"{standardize_date(entry_date)}"
    )


def proportion(
    numerator,
    denominator,
) -> float:

    if denominator == 0:
        return 0.0

    return (
        float(numerator)
        / float(denominator)
        * 100.0
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_volume() -> pd.DataFrame:

    if not VOLUME_FILE.exists():
        raise FileNotFoundError(
            f"Volume file not found:\n{VOLUME_FILE}"
        )

    df = pd.read_csv(VOLUME_FILE)

    symbol_col = resolve_column(
        df,
        ["symbol", "_symbol", "Symbol"],
        required=True,
    )

    direction_col = resolve_column(
        df,
        ["direction", "Direction"],
        required=True,
    )

    date_col = resolve_column(
        df,
        ["entry_date", "Entry_Date", "Entry Date"],
        required=True,
    )

    volume_col = resolve_column(
        df,
        [
            "volume_ratio",
            "Volume_Ratio",
            "signal_volume_ratio",
        ],
        required=True,
    )

    result = pd.DataFrame()

    result["trade_key"] = [
        make_trade_key(
            s,
            d,
            dt,
        )
        for s, d, dt in zip(
            df[symbol_col],
            df[direction_col],
            df[date_col],
        )
    ]

    result["volume_ratio"] = pd.to_numeric(
        df[volume_col],
        errors="coerce",
    )

    return result


def load_frozen() -> pd.DataFrame:

    if not FROZEN_FILE.exists():
        raise FileNotFoundError(
            f"Frozen trade file not found:\n{FROZEN_FILE}"
        )

    df = pd.read_csv(FROZEN_FILE)

    symbol_col = resolve_column(
        df,
        ["_symbol", "symbol", "Symbol"],
        required=True,
    )

    direction_col = resolve_column(
        df,
        ["direction", "Direction"],
        required=True,
    )

    date_col = resolve_column(
        df,
        ["entry_date", "Entry_Date", "Entry Date"],
        required=True,
    )

    event_col = resolve_column(
        df,
        [
            "early_adverse_r_bar_5",
            "Early_Adverse_R_Bar_5",
        ],
        required=True,
    )

    result = df.copy()

    result["trade_key"] = [
        make_trade_key(
            s,
            d,
            dt,
        )
        for s, d, dt in zip(
            df[symbol_col],
            df[direction_col],
            df[date_col],
        )
    ]

    bar5 = pd.to_numeric(
        df[event_col],
        errors="coerce",
    )

    result["event"] = (
        bar5.notna()
        & (
            bar5
            >= EARLY_ADVERSE_THRESHOLD_R
        )
    ).astype(int)

    return result


# =============================================================================
# CONTEXT DEFINITIONS
# =============================================================================

CONTEXT_SPECS = {
    "TREND_STRENGTH": [
        "entry_trend_strength",
        "Entry_Trend_Strength",
        "trend_strength",
        "Trend_Strength",
    ],
    "TREND_SLOPE": [
        "entry_trend_slope",
        "Entry_Trend_Slope",
        "trend_slope",
        "Trend_Slope",
    ],
    "ENTRY_ATR": [
        "entry_atr",
        "Entry_ATR",
    ],
    "SETUP_CONFIDENCE": [
        "setup_confidence",
        "Setup_Confidence",
    ],
}


# =============================================================================
# BINARY TABLE / STATISTICS
# =============================================================================

def build_table(
    df: pd.DataFrame,
    exposure_col: str,
) -> np.ndarray:

    table = np.zeros(
        (2, 2),
        dtype=int,
    )

    for _, row in df.iterrows():

        exposure = int(
            row[exposure_col]
        )

        event = int(
            row["event"]
        )

        if exposure not in (0, 1):
            continue

        if event not in (0, 1):
            continue

        table[
            exposure,
            event,
        ] += 1

    return table


def fisher_stats(
    table: np.ndarray,
) -> tuple[float, float]:

    if table.sum() == 0:
        return np.nan, np.nan

    if (
        table.sum(axis=0)[0] == 0
        or table.sum(axis=0)[1] == 0
        or table.sum(axis=1)[0] == 0
        or table.sum(axis=1)[1] == 0
    ):
        return np.nan, np.nan

    odds_ratio, p_value = fisher_exact(
        table
    )

    return (
        float(odds_ratio),
        float(p_value),
    )


def risk_difference_pp(
    table: np.ndarray,
) -> float:

    low_n = int(
        table[0].sum()
    )

    high_n = int(
        table[1].sum()
    )

    if low_n == 0 or high_n == 0:
        return np.nan

    low_rate = (
        table[0, 1]
        / low_n
        * 100.0
    )

    high_rate = (
        table[1, 1]
        / high_n
        * 100.0
    )

    return (
        high_rate
        - low_rate
    )


# =============================================================================
# BASELINE COMPOSITE
# =============================================================================

def build_baseline(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    working = data.copy()

    medians = {}

    for control in BASELINE_CONTROLS:

        median = float(
            working[control].median()
        )

        medians[control] = median

        working[
            f"{control}_HIGH"
        ] = (
            working[control]
            > median
        ).astype(int)

    high_columns = [
        f"{control}_HIGH"
        for control in BASELINE_CONTROLS
    ]

    working["baseline_high_count"] = (
        working[high_columns]
        .sum(axis=1)
    )

    # Fixed, non-optimized definition:
    # baseline HIGH means at least half of the
    # four controls are above their global medians.
    working["baseline_high"] = (
        working["baseline_high_count"]
        >= 2
    ).astype(int)

    return working, medians


# =============================================================================
# MODEL-LIKE INFORMATION CRITERIA
# =============================================================================

def bernoulli_log_likelihood(
    observed: np.ndarray,
    probability: float,
) -> float:

    if probability <= 0:
        probability = 1e-12

    if probability >= 1:
        probability = 1.0 - 1e-12

    return float(
        np.sum(
            observed
            * np.log(probability)
            + (
                1 - observed
            )
            * np.log(1 - probability)
        )
    )


def null_log_likelihood(
    events: pd.Series,
) -> float:

    y = (
        events
        .astype(int)
        .to_numpy()
    )

    p = float(
        y.mean()
    )

    return bernoulli_log_likelihood(
        y,
        p,
    )


def group_probability_log_likelihood(
    df: pd.DataFrame,
    group_col: str,
) -> tuple[float, int]:

    total_ll = 0.0
    groups_used = 0

    for _, group in df.groupby(
        group_col,
        dropna=False,
    ):

        y = (
            group["event"]
            .astype(int)
            .to_numpy()
        )

        if len(y) == 0:
            continue

        p = float(
            y.mean()
        )

        total_ll += (
            bernoulli_log_likelihood(
                y,
                p,
            )
        )

        groups_used += 1

    return (
        float(total_ll),
        groups_used,
    )


def aic_from_ll(
    log_likelihood: float,
    parameters: int,
) -> float:

    return (
        2.0
        * parameters
        - 2.0
        * log_likelihood
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(
    data: pd.DataFrame,
    model_name: str,
    group_col: str,
    parameter_count: int,
) -> dict:

    ll, groups_used = (
        group_probability_log_likelihood(
            data,
            group_col,
        )
    )

    null_ll = null_log_likelihood(
        data["event"]
    )

    aic = aic_from_ll(
        ll,
        parameter_count,
    )

    null_aic = aic_from_ll(
        null_ll,
        1,
    )

    return {
        "model": model_name,
        "group_column": group_col,
        "n": len(data),
        "events": int(
            data["event"].sum()
        ),
        "retained": int(
            (data["event"] == 0).sum()
        ),
        "groups_used": groups_used,
        "log_likelihood": ll,
        "null_log_likelihood": null_ll,
        "aic": aic,
        "null_aic": null_aic,
        "delta_aic_vs_null":
            aic - null_aic,
    }


# =============================================================================
# STABILITY
# =============================================================================

def evaluate_subset(
    subset: pd.DataFrame,
    subset_name: str,
) -> dict:

    result = {
        "subset": subset_name,
        "n": len(subset),
        "events": int(
            subset["event"].sum()
        ),
    }

    for model_name, group_col in [
        (
            "VOLUME_ONLY",
            "volume_high",
        ),
        (
            "BASELINE_ONLY",
            "baseline_high",
        ),
        (
            "BASELINE_PLUS_VOLUME",
            "baseline_volume_group",
        ),
    ]:

        ll, groups_used = (
            group_probability_log_likelihood(
                subset,
                group_col,
            )
        )

        result[
            f"{model_name}_log_likelihood"
        ] = ll

        result[
            f"{model_name}_groups"
        ] = groups_used

    return result


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print("=" * 100)
    print(
        "TRADESENSE-AI — VOLUME × INCREMENTAL INFORMATION FORENSIC AUDIT"
    )
    print("=" * 100)

    print()
    print("FROZEN HYPOTHESIS:")
    print(
        "Early adverse >= 0.25R within first 5 entry bars"
    )

    print()
    print("FROZEN POPULATION:")
    print(
        f"{EXPECTED_TRADES} trades | "
        f"{EXPECTED_EVENTS} events | "
        f"{EXPECTED_RETAINED} retained"
    )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    volume = load_volume()
    frozen = load_frozen()

    volume_duplicate_keys = int(
        volume["trade_key"]
        .duplicated()
        .sum()
    )

    frozen_duplicate_keys = int(
        frozen["trade_key"]
        .duplicated()
        .sum()
    )

    data = frozen.merge(
        volume,
        on="trade_key",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # Context
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("BASELINE CONTEXT AVAILABILITY")
    print("=" * 100)

    resolved_contexts = {}

    for name, aliases in CONTEXT_SPECS.items():

        col = resolve_column(
            data,
            aliases,
            required=False,
        )

        if col is None:

            print(
                f"{name:<24} NOT AVAILABLE"
            )

            continue

        data[name] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

        missing = int(
            data[name]
            .isna()
            .sum()
        )

        unique = int(
            data[name]
            .nunique(
                dropna=True
            )
        )

        median = float(
            data[name]
            .median()
        )

        resolved_contexts[name] = name

        print(
            f"{name:<24} "
            f"source={col:<28} "
            f"unique={unique:<3} "
            f"missing={missing:<3} "
            f"median={median:.9f}"
        )

    missing_contexts = [
        control
        for control in BASELINE_CONTROLS
        if control
        not in resolved_contexts
    ]

    if missing_contexts:

        raise RuntimeError(
            "Required baseline context unavailable: "
            f"{missing_contexts}"
        )

    # -----------------------------------------------------------------
    # Integrity
    # -----------------------------------------------------------------

    actual_trades = len(data)

    actual_events = int(
        data["event"].sum()
    )

    actual_retained = int(
        (data["event"] == 0).sum()
    )

    missing_volume = int(
        data["volume_ratio"]
        .isna()
        .sum()
    )

    volume_median = float(
        data["volume_ratio"]
        .median()
    )

    data["volume_high"] = (
        data["volume_ratio"]
        > volume_median
    ).astype(int)

    # Fixed baseline.
    data, baseline_medians = (
        build_baseline(data)
    )

    # Four-state baseline × volume.
    data["baseline_volume_group"] = (
        data[
            [
                "baseline_high",
                "volume_high",
            ]
        ]
        .astype(str)
        .agg(
            "_".join,
            axis=1,
        )
    )

    print()
    print("=" * 100)
    print("MERGE / FROZEN INTEGRITY")
    print("=" * 100)

    print(
        f"Frozen trades          : "
        f"{actual_trades}/{EXPECTED_TRADES}"
    )

    print(
        f"Frozen events          : "
        f"{actual_events}/{EXPECTED_EVENTS}"
    )

    print(
        f"Frozen retained        : "
        f"{actual_retained}/{EXPECTED_RETAINED}"
    )

    print(
        f"Volume duplicate keys  : "
        f"{volume_duplicate_keys}"
    )

    print(
        f"Frozen duplicate keys  : "
        f"{frozen_duplicate_keys}"
    )

    print(
        f"Missing volume ratio   : "
        f"{missing_volume}"
    )

    print(
        f"Global volume median   : "
        f"{volume_median:.9f}"
    )

    # -----------------------------------------------------------------
    # Raw volume reference
    # -----------------------------------------------------------------

    volume_table = build_table(
        data,
        "volume_high",
    )

    volume_or, volume_p = (
        fisher_stats(volume_table)
    )

    volume_rd = risk_difference_pp(
        volume_table
    )

    # -----------------------------------------------------------------
    # Baseline model reference
    # -----------------------------------------------------------------

    baseline_table = build_table(
        data,
        "baseline_high",
    )

    baseline_or, baseline_p = (
        fisher_stats(baseline_table)
    )

    baseline_rd = risk_difference_pp(
        baseline_table
    )

    print()
    print("=" * 100)
    print("REFERENCE ASSOCIATIONS")
    print("=" * 100)

    print(
        f"Volume-low event rate    : "
        f"{proportion(volume_table[0, 1], volume_table[0].sum()):.6f}%"
    )

    print(
        f"Volume-high event rate   : "
        f"{proportion(volume_table[1, 1], volume_table[1].sum()):.6f}%"
    )

    print(
        f"Volume RD high-low       : "
        f"{volume_rd:.6f} pp"
    )

    print(
        f"Volume Fisher OR         : "
        f"{volume_or}"
    )

    print(
        f"Volume Fisher p          : "
        f"{volume_p}"
    )

    print()

    print(
        f"Baseline-low event rate  : "
        f"{proportion(baseline_table[0, 1], baseline_table[0].sum()):.6f}%"
    )

    print(
        f"Baseline-high event rate : "
        f"{proportion(baseline_table[1, 1], baseline_table[1].sum()):.6f}%"
    )

    print(
        f"Baseline RD high-low     : "
        f"{baseline_rd:.6f} pp"
    )

    print(
        f"Baseline Fisher OR       : "
        f"{baseline_or}"
    )

    print(
        f"Baseline Fisher p        : "
        f"{baseline_p}"
    )

    # -----------------------------------------------------------------
    # Model-like information comparison
    # -----------------------------------------------------------------

    models = []

    models.append(
        evaluate_model(
            data,
            "VOLUME_ONLY",
            "volume_high",
            parameter_count=2,
        )
    )

    models.append(
        evaluate_model(
            data,
            "BASELINE_ONLY",
            "baseline_high",
            parameter_count=2,
        )
    )

    models.append(
        evaluate_model(
            data,
            "BASELINE_PLUS_VOLUME",
            "baseline_volume_group",
            parameter_count=4,
        )
    )

    model_df = pd.DataFrame(
        models
    )

    print()
    print("=" * 100)
    print("INCREMENTAL INFORMATION COMPARISON")
    print("=" * 100)

    for row in models:

        print()
        print(
            row["model"]
        )

        print(
            f"  groups              : "
            f"{row['groups_used']}"
        )

        print(
            f"  log likelihood      : "
            f"{row['log_likelihood']:.6f}"
        )

        print(
            f"  AIC                 : "
            f"{row['aic']:.6f}"
        )

        print(
            f"  delta AIC vs null   : "
            f"{row['delta_aic_vs_null']:.6f}"
        )

    baseline_aic = float(
        model_df.loc[
            model_df["model"]
            == "BASELINE_ONLY",
            "aic",
        ].iloc[0]
    )

    combined_aic = float(
        model_df.loc[
            model_df["model"]
            == "BASELINE_PLUS_VOLUME",
            "aic",
        ].iloc[0]
    )

    volume_incremental_delta_aic = (
        combined_aic
        - baseline_aic
    )

    print()
    print(
        "Volume incremental ΔAIC "
        "(combined - baseline): "
        f"{volume_incremental_delta_aic:.6f}"
    )

    # -----------------------------------------------------------------
    # Stability subsets
    # -----------------------------------------------------------------

    stability_rows = []

    data["_year"] = pd.to_datetime(
        data["entry_date"],
        errors="coerce",
    ).dt.year

    data["_symbol"] = [
        standardize_symbol(x)
        for x in data[
            resolve_column(
                data,
                [
                    "_symbol",
                    "symbol",
                    "Symbol",
                ],
                required=True,
            )
        ]
    ]

    data["_direction"] = [
        standardize_direction(x)
        for x in data[
            resolve_column(
                data,
                [
                    "direction",
                    "Direction",
                ],
                required=True,
            )
        ]
    ]

    setup_col = resolve_column(
        data,
        [
            "Setup",
            "setup",
        ],
        required=False,
    )

    if setup_col is not None:
        data["_setup"] = (
            data[setup_col]
            .astype(str)
            .str.strip()
        )

    # Overall.
    stability_rows.append(
        evaluate_subset(
            data,
            "ALL",
        )
    )

    # Year.
    for value in sorted(
        data["_year"]
        .dropna()
        .unique()
    ):

        subset = data[
            data["_year"] == value
        ].copy()

        stability_rows.append(
            evaluate_subset(
                subset,
                f"YEAR_{int(value)}",
            )
        )

    # Symbol.
    for value in sorted(
        data["_symbol"]
        .dropna()
        .unique()
    ):

        subset = data[
            data["_symbol"] == value
        ].copy()

        stability_rows.append(
            evaluate_subset(
                subset,
                f"SYMBOL_{value}",
            )
        )

    # Direction.
    for value in sorted(
        data["_direction"]
        .dropna()
        .unique()
    ):

        subset = data[
            data["_direction"] == value
        ].copy()

        stability_rows.append(
            evaluate_subset(
                subset,
                f"DIRECTION_{value}",
            )
        )

    # Setup.
    if "_setup" in data.columns:

        for value in sorted(
            data["_setup"]
            .dropna()
            .unique()
        ):

            subset = data[
                data["_setup"] == value
            ].copy()

            stability_rows.append(
                evaluate_subset(
                    subset,
                    f"SETUP_{value}",
                )
            )

    stability_df = pd.DataFrame(
        stability_rows
    )

    # -----------------------------------------------------------------
    # Integrity dataframe
    # -----------------------------------------------------------------

    integrity_rows = [
        {
            "check":
                "FROZEN_TRADE_COUNT",
            "actual":
                actual_trades,
            "expected":
                EXPECTED_TRADES,
            "status":
                (
                    "PASS"
                    if actual_trades
                    == EXPECTED_TRADES
                    else "FAIL"
                ),
        },
        {
            "check":
                "FROZEN_EVENT_COUNT",
            "actual":
                actual_events,
            "expected":
                EXPECTED_EVENTS,
            "status":
                (
                    "PASS"
                    if actual_events
                    == EXPECTED_EVENTS
                    else "FAIL"
                ),
        },
        {
            "check":
                "FROZEN_RETAINED_COUNT",
            "actual":
                actual_retained,
            "expected":
                EXPECTED_RETAINED,
            "status":
                (
                    "PASS"
                    if actual_retained
                    == EXPECTED_RETAINED
                    else "FAIL"
                ),
        },
        {
            "check":
                "VOLUME_DUPLICATE_KEYS",
            "actual":
                volume_duplicate_keys,
            "expected":
                0,
            "status":
                (
                    "PASS"
                    if volume_duplicate_keys
                    == 0
                    else "FAIL"
                ),
        },
        {
            "check":
                "FROZEN_DUPLICATE_KEYS",
            "actual":
                frozen_duplicate_keys,
            "expected":
                0,
            "status":
                (
                    "PASS"
                    if frozen_duplicate_keys
                    == 0
                    else "FAIL"
                ),
        },
        {
            "check":
                "MISSING_VOLUME_RATIO",
            "actual":
                missing_volume,
            "expected":
                0,
            "status":
                (
                    "PASS"
                    if missing_volume
                    == 0
                    else "FAIL"
                ),
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    data.to_csv(
        OUTPUT_DIR
        / "volume_incremental_trade_level.csv",
        index=False,
    )

    model_df.to_csv(
        OUTPUT_DIR
        / "volume_incremental_model_comparison.csv",
        index=False,
    )

    stability_df.to_csv(
        OUTPUT_DIR
        / "volume_incremental_stability.csv",
        index=False,
    )

    integrity_df.to_csv(
        OUTPUT_DIR
        / "volume_incremental_integrity.csv",
        index=False,
    )

    summary_json = {
        "hypothesis": {
            "threshold_r":
                EARLY_ADVERSE_THRESHOLD_R,
            "max_bars":
                MAX_BARS,
            "definition":
                "5B_0.25R",
        },
        "population": {
            "trades":
                actual_trades,
            "events":
                actual_events,
            "retained":
                actual_retained,
        },
        "volume": {
            "global_median":
                volume_median,
            "raw_odds_ratio":
                volume_or,
            "raw_fisher_p":
                volume_p,
            "raw_risk_difference_pp":
                volume_rd,
        },
        "baseline": {
            "controls":
                BASELINE_CONTROLS,
            "medians":
                baseline_medians,
            "odds_ratio":
                baseline_or,
            "fisher_p":
                baseline_p,
            "risk_difference_pp":
                baseline_rd,
        },
        "models":
            models,
        "volume_incremental_delta_aic":
            volume_incremental_delta_aic,
        "safety": {
            "production_modified":
                False,
            "frozen_dataset_modified":
                False,
            "oos_modified":
                False,
            "threshold_optimized":
                False,
            "feature_selection":
                False,
            "ml_model":
                False,
            "deployment_rule_created":
                False,
        },
    }

    with open(
        OUTPUT_DIR
        / "volume_incremental_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary_json,
            f,
            indent=2,
            default=str,
        )

    # -----------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------

    integrity_pass = bool(
        (
            integrity_df["status"]
            == "PASS"
        ).all()
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        "Trade level       : "
        f"{OUTPUT_DIR / 'volume_incremental_trade_level.csv'}"
    )

    print(
        "Model comparison  : "
        f"{OUTPUT_DIR / 'volume_incremental_model_comparison.csv'}"
    )

    print(
        "Stability         : "
        f"{OUTPUT_DIR / 'volume_incremental_stability.csv'}"
    )

    print(
        "Integrity         : "
        f"{OUTPUT_DIR / 'volume_incremental_integrity.csv'}"
    )

    print(
        "Summary           : "
        f"{OUTPUT_DIR / 'volume_incremental_summary.json'}"
    )

    print()

    if integrity_pass:

        print(
            "STATUS: VOLUME INCREMENTAL INFORMATION AUDIT PASS"
        )

    else:

        print(
            "STATUS: VOLUME INCREMENTAL INFORMATION AUDIT FAIL"
        )

    print()
    print("IMPORTANT:")
    print(
        "This establishes incremental association evidence only."
    )
    print(
        "Baseline variables use fixed global median splits."
    )
    print(
        "Volume uses the fixed global median."
    )
    print(
        "No threshold optimization or feature selection was performed."
    )
    print(
        "No ML model was used."
    )
    print(
        "No Volume deployment rule was created."
    )
    print(
        "Production, frozen data, and OOS remain unchanged."
    )

    return (
        0
        if integrity_pass
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )