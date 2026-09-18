"""
TradeSense-AI
VOLUME × CONTINUOUS-CONTEXT ADJUSTMENT FORENSIC AUDIT

Purpose
-------
Test whether the observed Volume -> Early-Adverse association remains
after accounting for continuous context variables.

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
- Continuous context variables use FIXED global median splits.
- No optimized thresholds.
- No feature selection based on outcome.
- No ML model.
- No production changes.
- No OOS changes.

Primary continuous controls
---------------------------
1. Entry Trend Strength
2. Entry Trend Slope
3. Entry ATR
4. Setup Confidence
5. Break Age

Secondary combined controls
---------------------------
- Trend Strength + Trend Slope
- Entry ATR + Setup Confidence
- Trend Strength + Entry ATR
- Setup Confidence + Break Age

Method
------
For each continuous control:
1. Split the control at its global median.
2. Within each control stratum, calculate Volume-high vs Volume-low
   event-rate difference.
3. Pool the strata with a Mantel-Haenszel-style odds ratio.
4. Perform leave-one-stratum-out stability.
5. Compare the controlled direction with the raw Volume association.

For combined controls:
- Use the Cartesian product of the fixed binary median splits.
- Tiny / zero-variation strata are retained in diagnostics but excluded
  from pooled association estimation.

This is association analysis only.
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
    / "volume_continuous_adjustment"
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

MIN_STRATUM_SIZE = 5


# =============================================================================
# COLUMN HELPERS
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

def proportion(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


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
        & (bar5 >= EARLY_ADVERSE_THRESHOLD_R)
    ).astype(int)

    return result


# =============================================================================
# CONTINUOUS CONTEXT DEFINITIONS
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
    "BREAK_AGE": [
        "break_age",
        "Break_Age",
    ],
}


# =============================================================================
# 2x2 STATISTICS
# =============================================================================

def build_2x2(
    df: pd.DataFrame,
) -> np.ndarray:

    # Rows = Volume Low / High
    # Cols = Event No / Event Yes

    table = np.zeros(
        (2, 2),
        dtype=int,
    )

    for _, row in df.iterrows():

        volume_side = int(
            row["volume_high"]
        )

        event = int(
            row["event"]
        )

        if volume_side not in (0, 1):
            continue

        if event not in (0, 1):
            continue

        table[
            volume_side,
            event,
        ] += 1

    return table


def risk_difference_pp(
    table: np.ndarray,
) -> float:

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


def fisher_statistics(
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


# =============================================================================
# MANTEL-HAENSZEL POOLED OR
# =============================================================================

def mh_pooled_or(
    tables: list[np.ndarray],
) -> tuple[float, float, int]:

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

        if (
            min(
                a + b,
                c + d,
                a + c,
                b + d,
            )
            <= 0
        ):
            continue

        numerator += (
            a * d / n
        )

        denominator += (
            b * c / n
        )

        usable += 1

    if denominator <= 0:
        return (
            np.nan,
            np.nan,
            usable,
        )

    pooled_or = (
        numerator
        / denominator
    )

    # Conservative diagnostic approximation.
    #
    # Because this is a small forensic sample, the p-value is NOT
    # treated as definitive evidence.

    log_terms = []

    for table in tables:

        a, b = float(table[0, 0]), float(table[0, 1])
        c, d = float(table[1, 0]), float(table[1, 1])

        if min(a, b, c, d) <= 0:
            continue

        log_terms.append(
            1.0 / a
            + 1.0 / b
            + 1.0 / c
            + 1.0 / d
        )

    if not log_terms:
        return (
            float(pooled_or),
            np.nan,
            usable,
        )

    variance = float(
        np.mean(log_terms)
    )

    if variance <= 0:
        return (
            float(pooled_or),
            np.nan,
            usable,
        )

    z = (
        math.log(pooled_or)
        / math.sqrt(variance)
    )

    p_value = math.erfc(
        abs(z)
        / math.sqrt(2.0)
    )

    return (
        float(pooled_or),
        float(p_value),
        usable,
    )


# =============================================================================
# SINGLE CONTINUOUS CONTROL
# =============================================================================

def analyze_single_control(
    data: pd.DataFrame,
    control_name: str,
    control_col: str,
) -> tuple[dict, list[dict], list[dict]]:

    working = data.copy()

    working[control_col] = pd.to_numeric(
        working[control_col],
        errors="coerce",
    )

    missing = int(
        working[control_col].isna().sum()
    )

    median = float(
        working[control_col].median()
    )

    working["control_high"] = (
        working[control_col]
        > median
    ).astype(int)

    working["control_stratum"] = np.where(
        working["control_high"] == 1,
        "HIGHER_THAN_MEDIAN",
        "AT_OR_BELOW_MEDIAN",
    )

    tables = []

    stratum_rows = []

    for stratum, group in working.groupby(
        "control_stratum",
        dropna=False,
    ):

        table = build_2x2(
            group
        )

        tables.append(table)

        or_value, p_value = (
            fisher_statistics(table)
        )

        stratum_rows.append(
            {
                "control": control_name,
                "control_median": median,
                "stratum": stratum,
                "n": len(group),
                "volume_low_n": int(
                    table[0].sum()
                ),
                "volume_high_n": int(
                    table[1].sum()
                ),
                "volume_low_events": int(
                    table[0, 1]
                ),
                "volume_high_events": int(
                    table[1, 1]
                ),
                "risk_difference_high_minus_low_pp":
                    risk_difference_pp(table),
                "stratum_odds_ratio":
                    or_value,
                "stratum_fisher_p":
                    p_value,
                "small_stratum":
                    len(group) < MIN_STRATUM_SIZE,
            }
        )

    pooled_or, pooled_p, usable = (
        mh_pooled_or(tables)
    )

    summary = {
        "control": control_name,
        "control_column": control_col,
        "control_median": median,
        "missing_values": missing,
        "n": len(working),
        "pooled_cmh_or": pooled_or,
        "pooled_cmh_p_approx": pooled_p,
        "usable_strata": usable,
    }

    # ---------------------------------------------------------
    # Leave-one-stratum-out
    # ---------------------------------------------------------

    loo_rows = []

    for removed_stratum in [
        "AT_OR_BELOW_MEDIAN",
        "HIGHER_THAN_MEDIAN",
    ]:

        remaining = working[
            working["control_stratum"]
            != removed_stratum
        ].copy()

        table = build_2x2(
            remaining
        )

        or_value, p_value = (
            fisher_statistics(table)
        )

        loo_rows.append(
            {
                "control": control_name,
                "removed_stratum":
                    removed_stratum,
                "remaining_n":
                    len(remaining),
                "volume_low_n":
                    int(table[0].sum()),
                "volume_high_n":
                    int(table[1].sum()),
                "risk_difference_high_minus_low_pp":
                    risk_difference_pp(table),
                "odds_ratio":
                    or_value,
                "fisher_p":
                    p_value,
            }
        )

    return (
        summary,
        stratum_rows,
        loo_rows,
    )


# =============================================================================
# COMBINED CONTROLS
# =============================================================================

COMBINED_CONTROLS = [
    (
        "TREND_STRENGTH_X_TREND_SLOPE",
        [
            "TREND_STRENGTH",
            "TREND_SLOPE",
        ],
    ),
    (
        "ENTRY_ATR_X_SETUP_CONFIDENCE",
        [
            "ENTRY_ATR",
            "SETUP_CONFIDENCE",
        ],
    ),
    (
        "TREND_STRENGTH_X_ENTRY_ATR",
        [
            "TREND_STRENGTH",
            "ENTRY_ATR",
        ],
    ),
    (
        "SETUP_CONFIDENCE_X_BREAK_AGE",
        [
            "SETUP_CONFIDENCE",
            "BREAK_AGE",
        ],
    ),
]


def analyze_combined_control(
    data: pd.DataFrame,
    name: str,
    controls: list[str],
) -> tuple[dict, list[dict]]:

    working = data.copy()

    for control in controls:

        median = float(
            working[control]
            .median()
        )

        working[
            f"{control}_HIGH"
        ] = (
            working[control]
            > median
        ).astype(int)

    stratum_columns = [
        f"{control}_HIGH"
        for control in controls
    ]

    working["combined_stratum"] = (
        working[stratum_columns]
        .astype(str)
        .agg(
            "_".join,
            axis=1,
        )
    )

    tables = []

    rows = []

    for stratum, group in working.groupby(
        "combined_stratum",
        dropna=False,
    ):

        table = build_2x2(
            group
        )

        row_totals = table.sum(
            axis=1
        )

        col_totals = table.sum(
            axis=0
        )

        usable = bool(
            (row_totals > 0).sum() == 2
            and
            (col_totals > 0).sum() == 2
        )

        if usable:
            tables.append(table)

        rows.append(
            {
                "combined_control": name,
                "stratum": stratum,
                "n": len(group),
                "usable_for_cmh": usable,
                "small_stratum":
                    len(group) < MIN_STRATUM_SIZE,
                "volume_low_n":
                    int(table[0].sum()),
                "volume_high_n":
                    int(table[1].sum()),
                "volume_low_events":
                    int(table[0, 1]),
                "volume_high_events":
                    int(table[1, 1]),
                "risk_difference_high_minus_low_pp":
                    risk_difference_pp(table),
            }
        )

    pooled_or, pooled_p, usable = (
        mh_pooled_or(tables)
    )

    summary = {
        "combined_control": name,
        "controls": controls,
        "strata": len(rows),
        "usable_strata": usable,
        "pooled_cmh_or": pooled_or,
        "pooled_cmh_p_approx": pooled_p,
    }

    return summary, rows


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print("=" * 100)
    print(
        "TRADESENSE-AI — VOLUME × CONTINUOUS-CONTEXT ADJUSTMENT FORENSIC AUDIT"
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
    # Resolve context fields
    # -----------------------------------------------------------------

    resolved_contexts = {}

    print()
    print("=" * 100)
    print("CONTINUOUS CONTEXT AVAILABILITY")
    print("=" * 100)

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
            data[name].isna().sum()
        )

        unique = int(
            data[name]
            .nunique(
                dropna=True
            )
        )

        median = float(
            data[name].median()
        )

        resolved_contexts[name] = name

        print(
            f"{name:<24} "
            f"source={col:<28} "
            f"unique={unique:<3} "
            f"missing={missing:<3} "
            f"median={median:.9f}"
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
        data["volume_ratio"].isna().sum()
    )

    # Fixed global volume median.
    volume_median = float(
        data["volume_ratio"].median()
    )

    data["volume_high"] = (
        data["volume_ratio"]
        > volume_median
    ).astype(int)

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
    # Raw reference
    # -----------------------------------------------------------------

    global_table = build_2x2(
        data
    )

    raw_or, raw_p = fisher_statistics(
        global_table
    )

    raw_rd = risk_difference_pp(
        global_table
    )

    print()
    print("=" * 100)
    print("RAW VOLUME REFERENCE")
    print("=" * 100)

    print(
        f"Volume-low event rate  : "
        f"{proportion(global_table[0, 1], global_table[0].sum()):.6f}%"
    )

    print(
        f"Volume-high event rate : "
        f"{proportion(global_table[1, 1], global_table[1].sum()):.6f}%"
    )

    print(
        f"Risk difference        : "
        f"{raw_rd:.6f} pp"
    )

    print(
        f"Fisher odds ratio      : "
        f"{raw_or}"
    )

    print(
        f"Fisher p-value         : "
        f"{raw_p}"
    )

    # -----------------------------------------------------------------
    # Single continuous controls
    # -----------------------------------------------------------------

    summary_rows = []
    stratum_rows = []
    loo_rows = []

    print()
    print("=" * 100)
    print("SINGLE CONTINUOUS CONTROLS")
    print("=" * 100)

    for name in resolved_contexts:

        summary, strata, loo = (
            analyze_single_control(
                data,
                name,
                name,
            )
        )

        summary_rows.append(
            summary
        )

        stratum_rows.extend(
            strata
        )

        loo_rows.extend(
            loo
        )

        print()
        print(name)

        print(
            f"  median              : "
            f"{summary['control_median']}"
        )

        print(
            f"  missing             : "
            f"{summary['missing_values']}"
        )

        print(
            f"  usable strata       : "
            f"{summary['usable_strata']}"
        )

        print(
            f"  pooled CMH OR       : "
            f"{summary['pooled_cmh_or']}"
        )

        print(
            f"  pooled CMH p        : "
            f"{summary['pooled_cmh_p_approx']}"
        )

    # -----------------------------------------------------------------
    # Combined continuous controls
    # -----------------------------------------------------------------

    combined_summary_rows = []
    combined_strata_rows = []

    print()
    print("=" * 100)
    print("COMBINED CONTINUOUS CONTROLS")
    print("=" * 100)

    for name, controls in COMBINED_CONTROLS:

        if not all(
            control in resolved_contexts
            for control in controls
        ):
            print()
            print(
                f"{name}: SKIPPED — required context unavailable"
            )
            continue

        summary, rows = (
            analyze_combined_control(
                data,
                name,
                controls,
            )
        )

        combined_summary_rows.append(
            summary
        )

        combined_strata_rows.extend(
            rows
        )

        print()
        print(name)

        print(
            f"  strata              : "
            f"{summary['strata']}"
        )

        print(
            f"  usable strata       : "
            f"{summary['usable_strata']}"
        )

        print(
            f"  pooled CMH OR       : "
            f"{summary['pooled_cmh_or']}"
        )

        print(
            f"  pooled CMH p        : "
            f"{summary['pooled_cmh_p_approx']}"
        )

    # -----------------------------------------------------------------
    # Integrity dataframe
    # -----------------------------------------------------------------

    integrity_rows = [
        {
            "check": "FROZEN_TRADE_COUNT",
            "actual": actual_trades,
            "expected": EXPECTED_TRADES,
            "status":
                "PASS"
                if actual_trades == EXPECTED_TRADES
                else "FAIL",
        },
        {
            "check": "FROZEN_EVENT_COUNT",
            "actual": actual_events,
            "expected": EXPECTED_EVENTS,
            "status":
                "PASS"
                if actual_events == EXPECTED_EVENTS
                else "FAIL",
        },
        {
            "check": "FROZEN_RETAINED_COUNT",
            "actual": actual_retained,
            "expected": EXPECTED_RETAINED,
            "status":
                "PASS"
                if actual_retained == EXPECTED_RETAINED
                else "FAIL",
        },
        {
            "check": "VOLUME_DUPLICATE_KEYS",
            "actual": volume_duplicate_keys,
            "expected": 0,
            "status":
                "PASS"
                if volume_duplicate_keys == 0
                else "FAIL",
        },
        {
            "check": "FROZEN_DUPLICATE_KEYS",
            "actual": frozen_duplicate_keys,
            "expected": 0,
            "status":
                "PASS"
                if frozen_duplicate_keys == 0
                else "FAIL",
        },
        {
            "check": "MISSING_VOLUME_RATIO",
            "actual": missing_volume,
            "expected": 0,
            "status":
                "PASS"
                if missing_volume == 0
                else "FAIL",
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    # -----------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------

    data.to_csv(
        OUTPUT_DIR
        / "volume_continuous_adjustment_trade_level.csv",
        index=False,
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        OUTPUT_DIR
        / "volume_continuous_adjustment_summary.csv",
        index=False,
    )

    pd.DataFrame(
        stratum_rows
    ).to_csv(
        OUTPUT_DIR
        / "volume_continuous_adjustment_strata.csv",
        index=False,
    )

    pd.DataFrame(
        loo_rows
    ).to_csv(
        OUTPUT_DIR
        / "volume_continuous_adjustment_loo.csv",
        index=False,
    )

    pd.DataFrame(
        combined_summary_rows
    ).to_csv(
        OUTPUT_DIR
        / "volume_continuous_combined_summary.csv",
        index=False,
    )

    pd.DataFrame(
        combined_strata_rows
    ).to_csv(
        OUTPUT_DIR
        / "volume_continuous_combined_strata.csv",
        index=False,
    )

    integrity_df.to_csv(
        OUTPUT_DIR
        / "volume_continuous_adjustment_integrity.csv",
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
                raw_or,
            "raw_fisher_p":
                raw_p,
            "raw_risk_difference_pp":
                raw_rd,
        },
        "continuous_controls":
            summary_rows,
        "combined_controls":
            combined_summary_rows,
        "integrity":
            integrity_rows,
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
        / "volume_continuous_adjustment_summary.json",
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
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_trade_level.csv'}"
    )

    print(
        "Single controls   : "
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_summary.csv'}"
    )

    print(
        "Strata            : "
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_strata.csv'}"
    )

    print(
        "Leave-one-out     : "
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_loo.csv'}"
    )

    print(
        "Combined summary  : "
        f"{OUTPUT_DIR / 'volume_continuous_combined_summary.csv'}"
    )

    print(
        "Combined strata   : "
        f"{OUTPUT_DIR / 'volume_continuous_combined_strata.csv'}"
    )

    print(
        "Integrity         : "
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_integrity.csv'}"
    )

    print(
        "Summary           : "
        f"{OUTPUT_DIR / 'volume_continuous_adjustment_summary.json'}"
    )

    print()

    if integrity_pass:
        print(
            "STATUS: VOLUME CONTINUOUS ADJUSTMENT AUDIT PASS"
        )
    else:
        print(
            "STATUS: VOLUME CONTINUOUS ADJUSTMENT AUDIT FAIL"
        )

    print()
    print("IMPORTANT:")
    print(
        "This establishes conditional association evidence only."
    )
    print(
        "Continuous controls use fixed global median splits."
    )
    print(
        "No threshold optimization or feature selection was performed."
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