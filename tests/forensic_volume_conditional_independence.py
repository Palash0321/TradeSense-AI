"""
TradeSense-AI
VOLUME CONDITIONAL INDEPENDENCE FORENSIC AUDIT

Purpose
-------
Determine whether Volume contains information about the frozen
5-bar / 0.25R early-adverse event after conditioning on existing
trade context.

Frozen hypothesis
-----------------
Early adverse >= 0.25R within first 5 entry bars.

Frozen population
-----------------
106 trades
60 early-adverse events
46 retained

Primary question
----------------
Does a pre-specified Volume split remain associated with the
frozen early-adverse event after controlling for existing context?

Controls
--------
1. Setup
2. Direction
3. Market Structure
4. Liquidity
5. Trend context
6. Setup x Direction

Method
------
- Volume split is FIXED at the global median of Volume Ratio.
- No threshold optimization.
- No symbol selection.
- No year selection.
- No parameter tuning.
- Cochran-Mantel-Haenszel style stratified odds-ratio analysis.
- Small strata are retained but flagged.
- Strata with no 2x2 variation are not allowed to create a
  false association.
- Leave-one-stratum-out robustness is also evaluated.

This is a forensic association test only.
It does NOT create a deployment rule.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

VOLUME_TRADE_FILE = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_early_adverse"
    / "volume_early_adverse_trade_level.csv"
)

FROZEN_TRADE_FILE = (
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
    / "volume_conditional_independence"
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

MIN_STRATUM_SIZE = 4


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


def build_normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    mapping = {}

    for col in df.columns:
        mapping.setdefault(normalize_column_name(col), col)

    return mapping


def resolve_column(
    df: pd.DataFrame,
    aliases: list[str],
    required: bool = False,
) -> str | None:
    normalized = build_normalized_columns(df)

    for alias in aliases:
        key = normalize_column_name(alias)

        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not resolve required column. "
            f"Tried aliases: {aliases}. "
            f"Available columns: {list(df.columns)}"
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
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def trade_key(symbol, direction, entry_date) -> str:
    return (
        f"{standardize_symbol(symbol)}|"
        f"{standardize_direction(direction)}|"
        f"{standardize_date(entry_date)}"
    )


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def proportion(a: int, b: int) -> float:
    if b == 0:
        return np.nan

    return 100.0 * a / b


# =============================================================================
# DATA LOADING
# =============================================================================

def load_volume_data() -> pd.DataFrame:
    if not VOLUME_TRADE_FILE.exists():
        raise FileNotFoundError(
            f"Volume trade-level file not found:\n{VOLUME_TRADE_FILE}"
        )

    df = pd.read_csv(VOLUME_TRADE_FILE)

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

    volume_col = resolve_column(
        df,
        [
            "volume_ratio",
            "Volume_Ratio",
            "signal_volume_ratio",
            "volume",
        ],
        required=True,
    )

    df = df.copy()

    df["_symbol_key"] = df[symbol_col].map(standardize_symbol)
    df["_direction_key"] = df[direction_col].map(standardize_direction)
    df["_entry_date_key"] = df[date_col].map(standardize_date)

    df["trade_key"] = [
        trade_key(s, d, dt)
        for s, d, dt in zip(
            df[symbol_col],
            df[direction_col],
            df[date_col],
        )
    ]

    df["volume_ratio"] = safe_float_series(df[volume_col])

    return df


def load_frozen_data() -> pd.DataFrame:
    if not FROZEN_TRADE_FILE.exists():
        raise FileNotFoundError(
            f"Frozen trade file not found:\n{FROZEN_TRADE_FILE}"
        )

    df = pd.read_csv(FROZEN_TRADE_FILE)

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

    bar5_col = resolve_column(
        df,
        [
            "early_adverse_r_bar_5",
            "Early_Adverse_R_Bar_5",
            "early_adverse_bar_5",
        ],
        required=True,
    )

    df = df.copy()

    df["trade_key"] = [
        trade_key(s, d, dt)
        for s, d, dt in zip(
            df[symbol_col],
            df[direction_col],
            df[date_col],
        )
    ]

    df["_symbol_key"] = df[symbol_col].map(standardize_symbol)
    df["_direction_key"] = df[direction_col].map(standardize_direction)
    df["_entry_date_key"] = df[date_col].map(standardize_date)

    bar5 = safe_float_series(df[bar5_col])

    # IMPORTANT:
    # Frozen label is defined by stored production Bar5 >= 0.25R.
    df["event"] = (
        bar5.notna()
        & (bar5 >= EARLY_ADVERSE_THRESHOLD_R)
    ).astype(int)

    return df


# =============================================================================
# CONTEXT RESOLUTION
# =============================================================================

def find_context_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    return resolve_column(df, candidates, required=False)


def attach_context_columns(
    volume: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:

    merged = frozen.merge(
        volume[
            [
                "trade_key",
                "volume_ratio",
            ]
        ],
        on="trade_key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_volume"),
    )

    # ---------------------------------------------------------
    # Resolve context from frozen production dataset first.
    # ---------------------------------------------------------

    context_specs = {
        "setup": [
            "setup",
            "Setup",
            "setup_name",
            "Setup_Name",
        ],
        "structure": [
            "structure_bias",
            "Structure_Bias",
            "market_structure",
            "Market_Structure",
            "structure",
            "Structure",
        ],
        "liquidity": [
            "liquidity_sweep",
            "Liquidity_Sweep",
            "liquidity",
            "Liquidity",
        ],
        "trend_strength": [
            "trend_strength",
            "Trend_Strength",
            "entry_trend_strength",
            "Entry_Trend_Strength",
        ],
        "trend": [
            "trend",
            "Trend",
            "entry_trend",
            "Entry_Trend",
        ],
        "bos": [
            "bos",
            "BOS",
        ],
        "choch": [
            "choch",
            "CHOCH",
        ],
    }

    source = frozen

    for output_name, candidates in context_specs.items():

        col = find_context_column(
            source,
            candidates,
        )

        if col is not None:
            merged[output_name] = (
                source[col]
                .astype(str)
                .replace({"nan": "MISSING"})
                .fillna("MISSING")
            )

        else:
            merged[output_name] = "MISSING"

    # ---------------------------------------------------------
    # Fixed global median Volume split.
    # ---------------------------------------------------------

    median_volume = float(
        merged["volume_ratio"].median()
    )

    merged["volume_high"] = (
        merged["volume_ratio"] > median_volume
    ).astype(int)

    merged["volume_side"] = np.where(
        merged["volume_high"] == 1,
        "HIGHER_THAN_GLOBAL_MEDIAN",
        "AT_OR_BELOW_GLOBAL_MEDIAN",
    )

    return merged, median_volume


# =============================================================================
# 2x2 HELPERS
# =============================================================================

def make_table(
    df: pd.DataFrame,
    exposure_col: str = "volume_high",
) -> np.ndarray:

    table = np.zeros((2, 2), dtype=int)

    for _, row in df.iterrows():

        exposure = int(row[exposure_col])
        event = int(row["event"])

        if exposure not in (0, 1):
            continue

        if event not in (0, 1):
            continue

        table[exposure, event] += 1

    return table


def fisher_or_pvalue(table: np.ndarray) -> tuple[float, float]:

    if table.sum() == 0:
        return np.nan, np.nan

    odds_ratio, p_value = fisher_exact(table)

    return float(odds_ratio), float(p_value)


def risk_difference_pp(table: np.ndarray) -> float:

    low_n = int(table[0].sum())
    high_n = int(table[1].sum())

    low_events = int(table[0, 1])
    high_events = int(table[1, 1])

    if low_n == 0 or high_n == 0:
        return np.nan

    return (
        100.0 * high_events / high_n
        - 100.0 * low_events / low_n
    )


# =============================================================================
# STRATIFIED CMH
# =============================================================================

def cmh_or(tables: list[np.ndarray]) -> tuple[float, float, int]:
    """
    Mantel-Haenszel pooled odds ratio.

    Table layout:

        exposure 0 | exposure 1
    event 0
    event 1

    Internally represented as:

        [[non-event exposure0, event exposure0],
         [non-event exposure1, event exposure1]]
    """

    numerator = 0.0
    denominator = 0.0

    usable = 0

    for table in tables:

        if table.shape != (2, 2):
            continue

        a = float(table[0, 0])
        b = float(table[0, 1])
        c = float(table[1, 0])
        d = float(table[1, 1])

        n = a + b + c + d

        if n <= 0:
            continue

        if (a + b) == 0:
            continue

        if (c + d) == 0:
            continue

        if (a + c) == 0:
            continue

        if (b + d) == 0:
            continue

        numerator += (a * d) / n
        denominator += (b * c) / n

        usable += 1

    if denominator == 0:
        return np.nan, np.nan, usable

    pooled_or = numerator / denominator

    # Approximate large-sample variance of log MH OR.
    # With this small dataset, p-value is treated as a diagnostic,
    # not as definitive proof.

    variance_terms = []

    for table in tables:

        a = float(table[0, 0])
        b = float(table[0, 1])
        c = float(table[1, 0])
        d = float(table[1, 1])

        n = a + b + c + d

        if min(a, b, c, d) <= 0:
            continue

        variance_terms.append(
            1.0 / a
            + 1.0 / b
            + 1.0 / c
            + 1.0 / d
        )

    if not variance_terms:
        return float(pooled_or), np.nan, usable

    variance = float(np.mean(variance_terms))

    if variance <= 0:
        return float(pooled_or), np.nan, usable

    z = math.log(pooled_or) / math.sqrt(variance)

    # Two-sided normal approximation.
    p = math.erfc(abs(z) / math.sqrt(2.0))

    return float(pooled_or), float(p), usable


# =============================================================================
# CONDITIONAL ANALYSIS
# =============================================================================

def conditional_analysis(
    data: pd.DataFrame,
    context_name: str,
    context_col: str,
) -> dict:

    rows = []

    usable_tables = []

    total_strata = 0
    usable_strata = 0
    small_strata = 0
    zero_variation_strata = 0

    for stratum_value, group in data.groupby(
        context_col,
        dropna=False,
    ):

        total_strata += 1

        n = len(group)

        table = make_table(group)

        if n < MIN_STRATUM_SIZE:
            small_strata += 1

        # Need both volume sides and both event states
        # to estimate within-stratum association.
        row_totals = table.sum(axis=1)
        col_totals = table.sum(axis=0)

        has_two_volume_sides = bool(
            (row_totals > 0).sum() == 2
        )

        has_two_event_states = bool(
            (col_totals > 0).sum() == 2
        )

        if not (
            has_two_volume_sides
            and has_two_event_states
        ):
            zero_variation_strata += 1
            usable = False
        else:
            usable = True
            usable_strata += 1
            usable_tables.append(table)

        low_n = int(table[0].sum())
        high_n = int(table[1].sum())

        low_events = int(table[0, 1])
        high_events = int(table[1, 1])

        low_rate = (
            100.0 * low_events / low_n
            if low_n
            else np.nan
        )

        high_rate = (
            100.0 * high_events / high_n
            if high_n
            else np.nan
        )

        rd = (
            high_rate - low_rate
            if not pd.isna(low_rate)
            and not pd.isna(high_rate)
            else np.nan
        )

        or_value, p_value = fisher_or_pvalue(table)

        rows.append(
            {
                "context": context_name,
                "stratum": str(stratum_value),
                "n": n,
                "volume_low_n": low_n,
                "volume_high_n": high_n,
                "volume_low_events": low_events,
                "volume_high_events": high_events,
                "volume_low_event_rate_pct": low_rate,
                "volume_high_event_rate_pct": high_rate,
                "risk_difference_high_minus_low_pp": rd,
                "stratum_odds_ratio": or_value,
                "stratum_fisher_p": p_value,
                "usable_for_cmh": usable,
                "small_stratum": n < MIN_STRATUM_SIZE,
            }
        )

    pooled_or, pooled_p, cmh_usable = cmh_or(
        usable_tables
    )

    result = {
        "context": context_name,
        "context_column": context_col,
        "total_strata": total_strata,
        "usable_strata": usable_strata,
        "small_strata": small_strata,
        "zero_variation_strata": zero_variation_strata,
        "cmh_usable_strata": cmh_usable,
        "cmh_odds_ratio": pooled_or,
        "cmh_p_value_approx": pooled_p,
    }

    return result, rows


# =============================================================================
# LEAVE-ONE-STRATUM-OUT
# =============================================================================

def leave_one_stratum_out(
    data: pd.DataFrame,
    context_name: str,
    context_col: str,
) -> list[dict]:

    all_groups = list(
        data[context_col]
        .dropna()
        .astype(str)
        .unique()
    )

    results = []

    for removed in all_groups:

        remaining = data[
            data[context_col].astype(str) != removed
        ].copy()

        tables = []

        for _, group in remaining.groupby(
            context_col,
            dropna=False,
        ):

            table = make_table(group)

            row_totals = table.sum(axis=1)
            col_totals = table.sum(axis=0)

            if (
                (row_totals > 0).sum() == 2
                and (col_totals > 0).sum() == 2
            ):
                tables.append(table)

        pooled_or, pooled_p, usable = cmh_or(
            tables
        )

        results.append(
            {
                "context": context_name,
                "removed_stratum": str(removed),
                "remaining_strata": len(all_groups) - 1,
                "usable_strata": usable,
                "cmh_odds_ratio": pooled_or,
                "cmh_p_value_approx": pooled_p,
            }
        )

    return results


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print("=" * 100)
    print(
        "TRADESENSE-AI — VOLUME CONDITIONAL INDEPENDENCE FORENSIC AUDIT"
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

    volume = load_volume_data()
    frozen = load_frozen_data()

    duplicate_volume = int(
        volume["trade_key"].duplicated().sum()
    )

    duplicate_frozen = int(
        frozen["trade_key"].duplicated().sum()
    )

    merged, volume_median = attach_context_columns(
        volume,
        frozen,
    )

    # -----------------------------------------------------------------
    # Integrity
    # -----------------------------------------------------------------

    actual_trades = len(merged)
    actual_events = int(merged["event"].sum())
    actual_retained = int(
        (merged["event"] == 0).sum()
    )

    missing_volume = int(
        merged["volume_ratio"].isna().sum()
    )

    missing_keys = int(
        merged["volume_ratio"].isna().sum()
    )

    print()
    print("=" * 100)
    print("FROZEN / MERGE INTEGRITY")
    print("=" * 100)

    print(
        f"Frozen trades                 : "
        f"{actual_trades}/{EXPECTED_TRADES}"
    )

    print(
        f"Frozen events                 : "
        f"{actual_events}/{EXPECTED_EVENTS}"
    )

    print(
        f"Frozen retained               : "
        f"{actual_retained}/{EXPECTED_RETAINED}"
    )

    print(
        f"Volume duplicate keys        : "
        f"{duplicate_volume}"
    )

    print(
        f"Frozen duplicate keys        : "
        f"{duplicate_frozen}"
    )

    print(
        f"Missing volume ratio         : "
        f"{missing_volume}"
    )

    print()
    print(
        f"GLOBAL VOLUME MEDIAN SPLIT   : "
        f"{volume_median:.9f}"
    )

    # -----------------------------------------------------------------
    # Context availability
    # -----------------------------------------------------------------

    print()
    print("=" * 100)
    print("CONTEXT AVAILABILITY")
    print("=" * 100)

    context_columns = [
        ("SETUP", "setup"),
        ("DIRECTION", "direction"),
        ("STRUCTURE", "structure"),
        ("LIQUIDITY", "liquidity"),
        ("TREND", "trend"),
        ("TREND_STRENGTH", "trend_strength"),
        ("BOS", "bos"),
        ("CHOCH", "choch"),
        (
            "SETUP_X_DIRECTION",
            "setup_x_direction",
        ),
    ]

    merged["setup_x_direction"] = (
        merged["setup"].astype(str)
        + " | "
        + merged["direction"].astype(str)
    )

    available_contexts = []

    for display_name, column in context_columns:

        if column not in merged.columns:
            continue

        unique = (
            merged[column]
            .astype(str)
            .nunique()
        )

        missing_like = int(
            merged[column]
            .astype(str)
            .isin(
                [
                    "MISSING",
                    "nan",
                    "None",
                ]
            )
            .sum()
        )

        print(
            f"{display_name:<24} "
            f"unique={unique:<3} "
            f"missing={missing_like:<3}"
        )

        if (
            unique >= 2
            and missing_like < len(merged)
        ):
            available_contexts.append(
                (display_name, column)
            )

    # -----------------------------------------------------------------
    # Primary conditional tests
    # -----------------------------------------------------------------

    summary_rows = []
    stratum_rows = []
    loo_rows = []

    print()
    print("=" * 100)
    print("CONDITIONAL VOLUME TESTS")
    print("=" * 100)

    for display_name, column in available_contexts:

        result, strata = conditional_analysis(
            merged,
            display_name,
            column,
        )

        summary_rows.append(result)
        stratum_rows.extend(strata)

        loo_rows.extend(
            leave_one_stratum_out(
                merged,
                display_name,
                column,
            )
        )

        print()
        print(
            f"{display_name}"
        )

        print(
            f"  strata                 : "
            f"{result['total_strata']}"
        )

        print(
            f"  usable strata          : "
            f"{result['usable_strata']}"
        )

        print(
            f"  small strata           : "
            f"{result['small_strata']}"
        )

        print(
            f"  zero-variation strata  : "
            f"{result['zero_variation_strata']}"
        )

        print(
            f"  CMH pooled OR          : "
            f"{result['cmh_odds_ratio']}"
        )

        print(
            f"  CMH approx p           : "
            f"{result['cmh_p_value_approx']}"
        )

    # -----------------------------------------------------------------
    # Overall / raw association reference
    # -----------------------------------------------------------------

    global_table = make_table(merged)

    global_or, global_p = fisher_or_pvalue(
        global_table
    )

    global_rd = risk_difference_pp(
        global_table
    )

    # -----------------------------------------------------------------
    # Safety / integrity
    # -----------------------------------------------------------------

    integrity_rows = [
        {
            "check": "FROZEN_TRADE_COUNT",
            "actual": actual_trades,
            "expected": EXPECTED_TRADES,
            "status": (
                "PASS"
                if actual_trades == EXPECTED_TRADES
                else "FAIL"
            ),
        },
        {
            "check": "FROZEN_EVENT_COUNT",
            "actual": actual_events,
            "expected": EXPECTED_EVENTS,
            "status": (
                "PASS"
                if actual_events == EXPECTED_EVENTS
                else "FAIL"
            ),
        },
        {
            "check": "FROZEN_RETAINED_COUNT",
            "actual": actual_retained,
            "expected": EXPECTED_RETAINED,
            "status": (
                "PASS"
                if actual_retained == EXPECTED_RETAINED
                else "FAIL"
            ),
        },
        {
            "check": "VOLUME_DUPLICATE_KEYS",
            "actual": duplicate_volume,
            "expected": 0,
            "status": (
                "PASS"
                if duplicate_volume == 0
                else "FAIL"
            ),
        },
        {
            "check": "FROZEN_DUPLICATE_KEYS",
            "actual": duplicate_frozen,
            "expected": 0,
            "status": (
                "PASS"
                if duplicate_frozen == 0
                else "FAIL"
            ),
        },
        {
            "check": "MISSING_VOLUME_RATIO",
            "actual": missing_volume,
            "expected": 0,
            "status": (
                "PASS"
                if missing_volume == 0
                else "FAIL"
            ),
        },
        {
            "check": "MISSING_MERGE_VOLUME",
            "actual": missing_keys,
            "expected": 0,
            "status": (
                "PASS"
                if missing_keys == 0
                else "FAIL"
            ),
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    strata_df = pd.DataFrame(
        stratum_rows
    )

    loo_df = pd.DataFrame(
        loo_rows
    )

    reference_df = pd.DataFrame(
        [
            {
                "volume_global_median": volume_median,
                "global_fisher_odds_ratio": global_or,
                "global_fisher_p_value": global_p,
                "global_risk_difference_high_minus_low_pp": global_rd,
                "frozen_trade_count": actual_trades,
                "frozen_event_count": actual_events,
                "frozen_retained_count": actual_retained,
            }
        ]
    )

    merged_output = merged.copy()

    merged_output.to_csv(
        OUTPUT_DIR
        / "volume_conditional_trade_level.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "volume_conditional_summary.csv",
        index=False,
    )

    strata_df.to_csv(
        OUTPUT_DIR
        / "volume_conditional_strata.csv",
        index=False,
    )

    loo_df.to_csv(
        OUTPUT_DIR
        / "volume_conditional_loo.csv",
        index=False,
    )

    reference_df.to_csv(
        OUTPUT_DIR
        / "volume_conditional_reference.csv",
        index=False,
    )

    integrity_df.to_csv(
        OUTPUT_DIR
        / "volume_conditional_integrity.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # JSON summary
    # -----------------------------------------------------------------

    summary_json = {
        "hypothesis": {
            "threshold_r": EARLY_ADVERSE_THRESHOLD_R,
            "max_bars": MAX_BARS,
            "definition": "5B_0.25R",
        },
        "population": {
            "trades": actual_trades,
            "events": actual_events,
            "retained": actual_retained,
        },
        "volume": {
            "global_median": volume_median,
            "global_fisher_odds_ratio": global_or,
            "global_fisher_p_value": global_p,
            "global_risk_difference_high_minus_low_pp": global_rd,
        },
        "conditional_tests": summary_rows,
        "integrity": integrity_rows,
        "safety": {
            "production_modified": False,
            "frozen_dataset_modified": False,
            "oos_modified": False,
            "threshold_optimized": False,
            "volume_threshold_optimized": False,
            "deployment_rule_created": False,
        },
    }

    with open(
        OUTPUT_DIR
        / "volume_conditional_summary.json",
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
    # Final status
    # -----------------------------------------------------------------

    integrity_pass = bool(
        (integrity_df["status"] == "PASS").all()
    )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        "Trade level       : "
        f"{OUTPUT_DIR / 'volume_conditional_trade_level.csv'}"
    )

    print(
        "Conditional       : "
        f"{OUTPUT_DIR / 'volume_conditional_summary.csv'}"
    )

    print(
        "Strata            : "
        f"{OUTPUT_DIR / 'volume_conditional_strata.csv'}"
    )

    print(
        "Leave-one-out     : "
        f"{OUTPUT_DIR / 'volume_conditional_loo.csv'}"
    )

    print(
        "Reference         : "
        f"{OUTPUT_DIR / 'volume_conditional_reference.csv'}"
    )

    print(
        "Integrity         : "
        f"{OUTPUT_DIR / 'volume_conditional_integrity.csv'}"
    )

    print(
        "Summary           : "
        f"{OUTPUT_DIR / 'volume_conditional_summary.json'}"
    )

    print()

    if integrity_pass:
        print(
            "STATUS: VOLUME CONDITIONAL INDEPENDENCE AUDIT PASS"
        )
    else:
        print(
            "STATUS: VOLUME CONDITIONAL INDEPENDENCE AUDIT FAIL"
        )

    print()
    print("IMPORTANT:")
    print(
        "This establishes conditional association evidence only."
    )
    print(
        "No Volume deployment rule was created."
    )
    print(
        "Production, frozen data, and OOS remain unchanged."
    )

    return 0 if integrity_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())