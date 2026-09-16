"""
TradeSense-AI
Early-Adverse Entry Mechanics Forensic
======================================

RESEARCH ONLY.

Objective
---------
Investigate whether pre-entry execution / signal-quality characteristics
differentiate the frozen early-adverse population from retained trades.

FROZEN EARLY-ADVERSE RULE
--------------------------
Reject only when:

    early_adverse_r_bar_5 >= 0.25R

Missing Bar-5 remains retained.

This script MUST NOT:
    - modify production strategy
    - modify the frozen hypothesis
    - modify OOS logic
    - modify the frozen ledger
    - optimize thresholds for deployment
    - use post-entry information as a predictor

Research dimensions
-------------------
1. Entry ATR / volatility
2. Entry trend strength
3. Entry trend slope
4. Setup confidence
5. Break age / signal freshness
6. Break confirmation
7. BOS
8. CHOCH
9. Liquidity sweep
10. Direction
11. Setup
12. interactions between execution-quality dimensions
13. event-rate separation
14. setup-stratified permutation tests
15. bootstrap confidence intervals
16. symbol leave-one-out
17. year leave-one-out
18. correlation stability

The target is the frozen early-adverse event itself.
Final trade outcome is NOT used to define the predictor.

No production/OOS/frozen files are written.
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

RULE_WINDOW = 5
RULE_THRESHOLD = 0.25

N_PERMUTATIONS = 20000
N_BOOTSTRAPS = 10000

SEED = 20260915

ROOT = Path(__file__).resolve().parents[1]

TRADES_FILE = ROOT / "tests" / "output" / "tradesense_trades.csv"

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "early_adverse_entry_mechanics"
)

EVENT_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_event_level.csv"
CATEGORY_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_categories.csv"
NUMERIC_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_numeric.csv"
SETUP_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_setup.csv"
SYMBOL_LOO_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_symbol_loo.csv"
YEAR_LOO_OUTPUT = OUTPUT_DIR / "early_adverse_entry_mechanics_year_loo.csv"


# ============================================================================
# UTILITIES
# ============================================================================

def to_float(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        return None


def clean_text(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {"nan", "none", "null", "nat"}:
        return None

    return text


def normalize_symbol(value):
    value = clean_text(value)

    if value is None:
        return None

    return value.replace(".NS", "").upper()


def normalize_direction(value):
    value = clean_text(value)

    if value is None:
        return None

    value = value.upper()

    if value in {"LONG", "BUY"}:
        return "LONG"

    if value in {"SHORT", "SELL"}:
        return "SHORT"

    return value


def normalize_bool(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().upper()

    if text in {"TRUE", "1", "YES"}:
        return True

    if text in {"FALSE", "0", "NO"}:
        return False

    return None


def safe_mean(values):
    values = [x for x in values if x is not None and np.isfinite(x)]

    if not values:
        return np.nan

    return float(np.mean(values))


def safe_median(values):
    values = [x for x in values if x is not None and np.isfinite(x)]

    if not values:
        return np.nan

    return float(np.median(values))


def fmt(value, digits=4):
    if value is None:
        return "NA"

    try:
        if not np.isfinite(float(value)):
            return "NA"
    except Exception:
        return "NA"

    return f"{float(value):.{digits}f}"


def pct(value, digits=2):
    if value is None:
        return "NA"

    try:
        if not np.isfinite(float(value)):
            return "NA"
    except Exception:
        return "NA"

    return f"{float(value) * 100:.{digits}f}%"


# ============================================================================
# LOAD
# ============================================================================

def load_trades():
    if not TRADES_FILE.exists():
        raise FileNotFoundError(
            f"Frozen trade ledger not found:\n{TRADES_FILE}"
        )

    df = pd.read_csv(TRADES_FILE)

    if df.empty:
        raise ValueError("Frozen trade ledger is empty.")

    return df


# ============================================================================
# FROZEN EVENT RECONSTRUCTION
# ============================================================================

def reconstruct_frozen_event(row):
    """
    Authoritative frozen rule:

        early_adverse_r_bar_5 >= 0.25R
            => event / reject

        missing Bar-5
            => retain

        otherwise
            => retain
    """

    value = to_float(row.get("early_adverse_r_bar_5"))

    if value is None:
        return False

    return value >= RULE_THRESHOLD


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def add_event_target(df):
    df = df.copy()

    df["Frozen_Early_Adverse_Event"] = df.apply(
        reconstruct_frozen_event,
        axis=1,
    )

    df["Research_Symbol"] = df["_symbol"].apply(normalize_symbol)

    df["Direction_Normalized"] = df["direction"].apply(
        normalize_direction
    )

    df["Setup_Normalized"] = (
        df["Setup"]
        .fillna(df.get("setup", pd.Series(index=df.index)))
        .apply(clean_text)
    )

    df["Entry_Date_Normalized"] = pd.to_datetime(
        df["entry_date"],
        errors="coerce",
    )

    df["Research_Year"] = (
        df["Entry_Date_Normalized"]
        .dt.year
        .astype("Int64")
    )

    return df


def add_numeric_features(df):
    df = df.copy()

    numeric_columns = [
        "entry_atr",
        "initial_risk",
        "entry_trend_strength",
        "entry_trend_slope",
        "Setup_Confidence",
        "Break_Age",
        "bars_in_trade",
        "Fib_Retracement",
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ----------------------------------------------------------------------
    # Derived pre-entry features
    # ----------------------------------------------------------------------

    # Initial risk relative to ATR.
    #
    # This is descriptive only. We are NOT assuming a particular economic
    # interpretation until the data supports it.
    df["Risk_ATR_Ratio"] = np.where(
        df["entry_atr"].notna()
        & (df["entry_atr"] != 0)
        & df["initial_risk"].notna(),
        df["initial_risk"] / df["entry_atr"],
        np.nan,
    )

    # Absolute trend slope.
    df["Abs_Trend_Slope"] = df["entry_trend_slope"].abs()

    # Signed slope relative to direction.
    #
    # Positive means slope agrees with direction.
    df["Directional_Trend_Slope"] = np.where(
        df["Direction_Normalized"].eq("LONG"),
        df["entry_trend_slope"],
        np.where(
            df["Direction_Normalized"].eq("SHORT"),
            -df["entry_trend_slope"],
            np.nan,
        ),
    )

    # Entry trend strength already has a direction-aware meaning in the
    # frozen ledger, so preserve it without transformation.
    df["Trend_Strength"] = df["entry_trend_strength"]

    return df


# ============================================================================
# BASIC EVENT SUMMARY
# ============================================================================

def event_summary(df):
    events = df[df["Frozen_Early_Adverse_Event"]]
    retained = df[~df["Frozen_Early_Adverse_Event"]]

    total_r = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    ).sum()

    event_r = pd.to_numeric(
        events["r_multiple"],
        errors="coerce",
    ).sum()

    retained_r = pd.to_numeric(
        retained["r_multiple"],
        errors="coerce",
    ).sum()

    print()
    print("=" * 80)
    print("FROZEN EARLY-ADVERSE REFERENCE")
    print("=" * 80)

    print(f"Total trades        : {len(df)}")
    print(f"Early-adverse events: {len(events)}")
    print(f"Retained trades     : {len(retained)}")

    print(f"Baseline R          : {fmt(total_r, 2)}")
    print(f"Event R             : {fmt(event_r, 2)}")
    print(f"Retained R          : {fmt(retained_r, 2)}")

    print()
    print(
        f"Frozen rule: "
        f"early_adverse_r_bar_{RULE_WINDOW} >= {RULE_THRESHOLD:.2f}R"
    )

    print("=" * 80)


# ============================================================================
# CATEGORY AUDIT
# ============================================================================

def category_audit(df, column):
    rows = []

    work = df.copy()

    work["_category"] = work[column].apply(clean_text)

    for category, group in work.groupby(
        "_category",
        dropna=False,
    ):
        if category is None or (
            isinstance(category, float) and np.isnan(category)
        ):
            category = "UNKNOWN"

        events = group["Frozen_Early_Adverse_Event"]

        r = pd.to_numeric(
            group["r_multiple"],
            errors="coerce",
        )

        winners = (r > 0).sum()

        rows.append(
            {
                "Feature": column,
                "Category": category,
                "Trades": len(group),
                "Events": int(events.sum()),
                "Event_Rate": float(events.mean()),
                "Wins": int(winners),
                "Win_Rate": float(winners / len(group))
                if len(group)
                else np.nan,
                "Total_R": float(r.sum()),
                "Avg_R": float(r.mean()),
            }
        )

    return pd.DataFrame(rows)


def print_category_audit(df, column):
    result = category_audit(df, column)

    print()
    print("-" * 80)
    print(column)
    print("-" * 80)

    if result.empty:
        print("No usable observations.")
        return result

    for _, row in result.iterrows():
        print(
            f"{str(row['Category']):<24}"
            f" N={int(row['Trades']):>3}"
            f" Events={int(row['Events']):>3}"
            f" EventRate={pct(row['Event_Rate']):>8}"
            f" R={fmt(row['Total_R'], 2):>8}"
            f" AvgR={fmt(row['Avg_R'], 3):>8}"
        )

    return result


# ============================================================================
# NUMERIC CORRELATION
# ============================================================================

def numeric_event_correlations(df, numeric_features):
    rows = []

    target = df["Frozen_Early_Adverse_Event"].astype(float)

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid = x.notna() & target.notna()

        if valid.sum() < 5:
            rows.append(
                {
                    "Feature": feature,
                    "N": int(valid.sum()),
                    "Pearson": np.nan,
                    "Spearman": np.nan,
                }
            )
            continue

        xv = x[valid]
        yv = target[valid]

        pearson = xv.corr(yv, method="pearson")
        spearman = xv.corr(yv, method="spearman")

        rows.append(
            {
                "Feature": feature,
                "N": int(valid.sum()),
                "Pearson": float(pearson),
                "Spearman": float(spearman),
            }
        )

    return pd.DataFrame(rows)


def print_numeric_correlations(result):
    print()
    print("=" * 80)
    print("NUMERIC FEATURE → EARLY-ADVERSE EVENT CORRELATION")
    print("=" * 80)

    for _, row in result.iterrows():
        print(
            f"{row['Feature']:<28}"
            f"N={int(row['N']):>3}"
            f" Pearson={fmt(row['Pearson']):>9}"
            f" Spearman={fmt(row['Spearman']):>9}"
        )


# ============================================================================
# SETUP-STRATIFIED PERMUTATION
# ============================================================================

def setup_stratified_delta(
    df,
    feature,
):
    """
    Compare mean R between feature=1 and feature=0 while preserving
    Setup composition.

    This is deliberately outcome-based only for the robustness diagnostic.
    It is NOT a deployment optimization.

    The main target of this forensic remains the early-adverse event.
    """

    if feature not in df.columns:
        return np.nan, 0

    local = df.copy()

    values = pd.to_numeric(
        local[feature],
        errors="coerce",
    )

    valid = values.notna() & local["r_multiple"].notna()

    local = local.loc[valid].copy()

    if local.empty:
        return np.nan, 0

    local["_feature"] = values.loc[valid].astype(float)
    local["_R"] = pd.to_numeric(
        local["r_multiple"],
        errors="coerce",
    )

    deltas = []

    for _, group in local.groupby(
        "Setup_Normalized",
        dropna=False,
    ):
        positive = group[group["_feature"] > 0]
        negative = group[group["_feature"] <= 0]

        if len(positive) == 0 or len(negative) == 0:
            continue

        delta = (
            positive["_R"].mean()
            - negative["_R"].mean()
        )

        deltas.append(
            (
                delta,
                len(group),
            )
        )

    if not deltas:
        return np.nan, 0

    total_n = sum(n for _, n in deltas)

    weighted_delta = sum(
        delta * n
        for delta, n in deltas
    ) / total_n

    return float(weighted_delta), len(deltas)


def permutation_test_setup_stratified(
    df,
    feature,
    n_permutations=N_PERMUTATIONS,
    seed=SEED,
):
    """
    Setup-stratified permutation test.

    Labels are copied before shuffling to avoid numpy read-only-array
    failures from pandas-backed arrays.
    """

    if feature not in df.columns:
        return {
            "Observed_Delta_R": np.nan,
            "P_Value": np.nan,
            "Bootstrap_Low": np.nan,
            "Bootstrap_High": np.nan,
            "Usable_Setups": 0,
            "N": 0,
        }

    local = df.copy()

    feature_values = pd.to_numeric(
        local[feature],
        errors="coerce",
    )

    r_values = pd.to_numeric(
        local["r_multiple"],
        errors="coerce",
    )

    valid = (
        feature_values.notna()
        & r_values.notna()
        & local["Setup_Normalized"].notna()
    )

    local = local.loc[valid].copy()

    if local.empty:
        return {
            "Observed_Delta_R": np.nan,
            "P_Value": np.nan,
            "Bootstrap_Low": np.nan,
            "Bootstrap_High": np.nan,
            "Usable_Setups": 0,
            "N": 0,
        }

    local["_feature"] = feature_values.loc[valid].astype(float)
    local["_R"] = r_values.loc[valid].astype(float)

    setup_groups = []

    for setup, group in local.groupby(
        "Setup_Normalized",
        dropna=False,
    ):
        feature = group["_feature"].to_numpy(
            dtype=float,
            copy=True,
        )

        # Convert continuous/binary inputs to a strict binary partition.
        labels = feature > 0

        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        returns = group["_R"].to_numpy(
            dtype=float,
            copy=True,
        )

        setup_groups.append(
            {
                "setup": setup,
                "labels": labels,
                "returns": returns,
            }
        )

    if not setup_groups:
        return {
            "Observed_Delta_R": np.nan,
            "P_Value": np.nan,
            "Bootstrap_Low": np.nan,
            "Bootstrap_High": np.nan,
            "Usable_Setups": 0,
            "N": 0,
        }

    def calculate_delta(groups):
        numerator = 0.0
        denominator = 0

        for group in groups:
            labels = group["labels"]
            returns = group["returns"]

            a = returns[labels]
            b = returns[~labels]

            if len(a) == 0 or len(b) == 0:
                continue

            delta = a.mean() - b.mean()

            n = len(returns)

            numerator += delta * n
            denominator += n

        if denominator == 0:
            return np.nan

        return numerator / denominator

    observed = calculate_delta(setup_groups)

    rng = np.random.default_rng(seed)

    null_distribution = np.empty(
        n_permutations,
        dtype=float,
    )

    for i in range(n_permutations):
        permuted_groups = []

        for group in setup_groups:
            labels = group["labels"].copy()
            rng.shuffle(labels)

            permuted_groups.append(
                {
                    "labels": labels,
                    "returns": group["returns"],
                }
            )

        null_distribution[i] = calculate_delta(
            permuted_groups
        )

    null_distribution = null_distribution[
        np.isfinite(null_distribution)
    ]

    if len(null_distribution) == 0:
        p_value = np.nan
    else:
        p_value = (
            np.sum(
                np.abs(null_distribution)
                >= abs(observed)
            )
            + 1
        ) / (len(null_distribution) + 1)

    # ----------------------------------------------------------------------
    # Bootstrap
    # ----------------------------------------------------------------------

    bootstrap_rng = np.random.default_rng(seed + 1)

    bootstrap_values = np.empty(
        N_BOOTSTRAPS,
        dtype=float,
    )

    for i in range(N_BOOTSTRAPS):
        bootstrap_groups = []

        for group in setup_groups:
            labels = group["labels"]
            returns = group["returns"]

            indices = bootstrap_rng.integers(
                0,
                len(returns),
                size=len(returns),
            )

            bootstrap_groups.append(
                {
                    "labels": labels[indices],
                    "returns": returns[indices],
                }
            )

        bootstrap_values[i] = calculate_delta(
            bootstrap_groups
        )

    bootstrap_values = bootstrap_values[
        np.isfinite(bootstrap_values)
    ]

    if len(bootstrap_values):
        low = float(
            np.percentile(
                bootstrap_values,
                2.5,
            )
        )

        high = float(
            np.percentile(
                bootstrap_values,
                97.5,
            )
        )
    else:
        low = np.nan
        high = np.nan

    return {
        "Observed_Delta_R": float(observed),
        "P_Value": float(p_value),
        "Bootstrap_Low": low,
        "Bootstrap_High": high,
        "Usable_Setups": len(setup_groups),
        "N": len(local),
    }


# ============================================================================
# EVENT PREDICTION CATEGORY TEST
# ============================================================================

def event_rate_test(df, feature):
    """
    Pure event-rate comparison.

    For binary/category features:
        event rate when condition is present
        vs
        event rate when condition is absent.

    This is descriptive and is not a production filter.
    """

    if feature not in df.columns:
        return None

    local = df.copy()

    values = local[feature]

    if pd.api.types.is_numeric_dtype(values):
        values = pd.to_numeric(
            values,
            errors="coerce",
        )

        valid = values.notna()

        if valid.sum() == 0:
            return None

        median = values[valid].median()

        condition = values >= median
    else:
        values = values.apply(clean_text)

        valid = values.notna()

        if valid.sum() == 0:
            return None

        # For categorical variables, only use the most populated category.
        top_category = (
            values[valid]
            .value_counts()
            .index[0]
        )

        condition = values.eq(top_category)

    local = local.loc[valid].copy()
    condition = condition.loc[valid]

    events = local[
        "Frozen_Early_Adverse_Event"
    ].astype(int)

    positive = events[condition]
    negative = events[~condition]

    if len(positive) == 0 or len(negative) == 0:
        return None

    return {
        "Feature": feature,
        "Positive_N": len(positive),
        "Positive_Event_Rate": positive.mean(),
        "Negative_N": len(negative),
        "Negative_Event_Rate": negative.mean(),
        "Delta_Event_Rate": positive.mean() - negative.mean(),
    }


# ============================================================================
# SYMBOL LOO
# ============================================================================

def symbol_loo(df, numeric_features):
    rows = []

    symbols = [
        x
        for x in df["Research_Symbol"]
        .dropna()
        .unique()
    ]

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        for symbol in symbols:
            local = df[
                df["Research_Symbol"] != symbol
            ].copy()

            x = pd.to_numeric(
                local[feature],
                errors="coerce",
            )

            y = local[
                "Frozen_Early_Adverse_Event"
            ].astype(float)

            valid = x.notna()

            if valid.sum() < 5:
                corr = np.nan
            else:
                corr = x[valid].corr(
                    y[valid],
                    method="spearman",
                )

            rows.append(
                {
                    "Feature": feature,
                    "Excluded_Symbol": symbol,
                    "N": int(valid.sum()),
                    "Spearman": corr,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# YEAR LOO
# ============================================================================

def year_loo(df, numeric_features):
    rows = []

    years = [
        int(x)
        for x in df["Research_Year"]
        .dropna()
        .unique()
    ]

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        for year in years:
            local = df[
                df["Research_Year"] != year
            ].copy()

            x = pd.to_numeric(
                local[feature],
                errors="coerce",
            )

            y = local[
                "Frozen_Early_Adverse_Event"
            ].astype(float)

            valid = x.notna()

            if valid.sum() < 5:
                corr = np.nan
            else:
                corr = x[valid].corr(
                    y[valid],
                    method="spearman",
                )

            rows.append(
                {
                    "Feature": feature,
                    "Excluded_Year": year,
                    "N": int(valid.sum()),
                    "Spearman": corr,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# SETUP-CONDITIONAL EVENT ANALYSIS
# ============================================================================

def setup_event_analysis(df, features):
    rows = []

    for setup, group in df.groupby(
        "Setup_Normalized",
        dropna=False,
    ):
        setup_name = (
            setup
            if setup is not None
            else "UNKNOWN"
        )

        for feature in features:
            if feature not in group.columns:
                continue

            values = group[feature]

            if pd.api.types.is_numeric_dtype(values):
                values = pd.to_numeric(
                    values,
                    errors="coerce",
                )

                valid = values.notna()

                if valid.sum() < 4:
                    continue

                median = values[valid].median()

                condition = values >= median

                label = (
                    f">= median ({median:.4f})"
                )

            else:
                values = values.apply(clean_text)

                valid = values.notna()

                if valid.sum() < 4:
                    continue

                top = (
                    values[valid]
                    .value_counts()
                    .index[0]
                )

                condition = values.eq(top)

                label = str(top)

            events = group[
                "Frozen_Early_Adverse_Event"
            ]

            positive = events[
                valid & condition
            ]

            negative = events[
                valid & ~condition
            ]

            if len(positive) == 0 or len(negative) == 0:
                continue

            rows.append(
                {
                    "Setup": setup_name,
                    "Feature": feature,
                    "Condition": label,
                    "Positive_N": len(positive),
                    "Positive_Event_Rate": positive.mean(),
                    "Negative_N": len(negative),
                    "Negative_Event_Rate": negative.mean(),
                    "Delta_Event_Rate": (
                        positive.mean()
                        - negative.mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main():
    np.random.seed(SEED)
    random.seed(SEED)

    print()
    print("=" * 80)
    print("TRADESENSE-AI")
    print("EARLY-ADVERSE ENTRY MECHANICS FORENSIC")
    print("=" * 80)

    print()
    print(f"Ledger: {TRADES_FILE}")
    print(
        f"Frozen rule: "
        f"early_adverse_r_bar_{RULE_WINDOW} >= "
        f"{RULE_THRESHOLD:.2f}R"
    )

    df = load_trades()

    print()
    print(f"Loaded trades: {len(df)}")

    required = [
        "_symbol",
        "entry_date",
        "direction",
        "Setup",
        "r_multiple",
        "early_adverse_r_bar_5",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(missing)
        )

    df = add_event_target(df)
    df = add_numeric_features(df)

    # ----------------------------------------------------------------------
    # Frozen reference
    # ----------------------------------------------------------------------

    event_summary(df)

    # ----------------------------------------------------------------------
    # Event-level output
    # ----------------------------------------------------------------------

    event_columns = [
        "_symbol",
        "Research_Symbol",
        "entry_date",
        "Research_Year",
        "Direction_Normalized",
        "Setup_Normalized",
        "Frozen_Early_Adverse_Event",
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5",
        "entry_atr",
        "initial_risk",
        "Risk_ATR_Ratio",
        "entry_trend_strength",
        "entry_trend_slope",
        "Abs_Trend_Slope",
        "Directional_Trend_Slope",
        "Setup_Confidence",
        "Break_Age",
        "Break_Confirmed",
        "BOS",
        "CHOCH",
        "Liquidity_Sweep",
        "entry_price_position",
        "entry_trend",
        "r_multiple",
    ]

    event_columns = [
        c
        for c in event_columns
        if c in df.columns
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df[event_columns].to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Categorical dimensions
    # ----------------------------------------------------------------------

    categorical_features = [
        "Direction_Normalized",
        "Setup_Normalized",
        "entry_price_position",
        "entry_trend",
        "Break_Confirmed",
        "BOS",
        "CHOCH",
        "Liquidity_Sweep",
    ]

    all_category_results = []

    for feature in categorical_features:
        if feature not in df.columns:
            continue

        result = print_category_audit(
            df,
            feature,
        )

        all_category_results.append(result)

    if all_category_results:
        pd.concat(
            all_category_results,
            ignore_index=True,
        ).to_csv(
            CATEGORY_OUTPUT,
            index=False,
        )

    # ----------------------------------------------------------------------
    # Numeric dimensions
    # ----------------------------------------------------------------------

    numeric_features = [
        "entry_atr",
        "initial_risk",
        "Risk_ATR_Ratio",
        "entry_trend_strength",
        "entry_trend_slope",
        "Abs_Trend_Slope",
        "Directional_Trend_Slope",
        "Setup_Confidence",
        "Break_Age",
        "Fib_Retracement",
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
    ]

    numeric_result = numeric_event_correlations(
        df,
        numeric_features,
    )

    print_numeric_correlations(
        numeric_result
    )

    numeric_result.to_csv(
        NUMERIC_OUTPUT,
        index=False,
    )

    # ----------------------------------------------------------------------
    # Event-rate diagnostics
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PRE-ENTRY FEATURE → EVENT-RATE DIAGNOSTICS")
    print("=" * 80)

    event_rate_rows = []

    diagnostic_features = [
        "entry_atr",
        "Risk_ATR_Ratio",
        "entry_trend_strength",
        "entry_trend_slope",
        "Setup_Confidence",
        "Break_Age",
        "entry_price_position",
        "entry_trend",
        "Break_Confirmed",
        "BOS",
        "CHOCH",
        "Liquidity_Sweep",
    ]

    for feature in diagnostic_features:
        result = event_rate_test(
            df,
            feature,
        )

        if result is None:
            continue

        event_rate_rows.append(result)

        print(
            f"{feature:<28}"
            f" positive={int(result['Positive_N']):>3}"
            f" rate={pct(result['Positive_Event_Rate']):>8}"
            f" | negative={int(result['Negative_N']):>3}"
            f" rate={pct(result['Negative_Event_Rate']):>8}"
            f" | Δ={pct(result['Delta_Event_Rate']):>8}"
        )

    if event_rate_rows:
        pd.DataFrame(
            event_rate_rows
        ).to_csv(
            OUTPUT_DIR
            / "early_adverse_entry_mechanics_event_rate.csv",
            index=False,
        )

    # ----------------------------------------------------------------------
    # Setup-stratified formal tests
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SETUP-STRATIFIED FORMAL TESTS")
    print("=" * 80)

    formal_features = [
        "entry_price_position",
        "entry_trend",
        "Break_Confirmed",
        "BOS",
        "CHOCH",
        "Liquidity_Sweep",
    ]

    formal_rows = []

    for feature in formal_features:
        if feature not in df.columns:
            continue

        local = df.copy()

        # Convert categorical feature to binary using its most frequent
        # meaningful category. This is a diagnostic representation only.
        values = local[feature].apply(clean_text)

        valid = values.notna()

        if valid.sum() == 0:
            continue

        top_category = (
            values[valid]
            .value_counts()
            .index[0]
        )

        local["_binary_feature"] = (
            values.eq(top_category)
            .astype(float)
        )

        result = permutation_test_setup_stratified(
            local,
            "_binary_feature",
        )

        formal_rows.append(
            {
                "Feature": feature,
                "Positive_Category": top_category,
                **result,
            }
        )

        print(
            f"{feature:<28}"
            f" category={str(top_category):<18}"
            f" Δ={fmt(result['Observed_Delta_R'], 4):>9}R"
            f" p={fmt(result['P_Value'], 6):>9}"
            f" CI=["
            f"{fmt(result['Bootstrap_Low'], 4)}, "
            f"{fmt(result['Bootstrap_High'], 4)}]"
        )

    if formal_rows:
        pd.DataFrame(formal_rows).to_csv(
            SETUP_OUTPUT,
            index=False,
        )

    # ----------------------------------------------------------------------
    # Setup-specific event rates
    # ----------------------------------------------------------------------

    setup_features = [
        "entry_atr",
        "Risk_ATR_Ratio",
        "entry_trend_strength",
        "entry_trend_slope",
        "Setup_Confidence",
        "Break_Age",
        "entry_price_position",
        "entry_trend",
        "Break_Confirmed",
        "BOS",
        "CHOCH",
        "Liquidity_Sweep",
    ]

    setup_result = setup_event_analysis(
        df,
        setup_features,
    )

    if not setup_result.empty:
        print()
        print("=" * 80)
        print("SETUP × PRE-ENTRY FEATURE EVENT RATES")
        print("=" * 80)

        for _, row in setup_result.iterrows():
            print(
                f"{str(row['Setup']):<24}"
                f"{row['Feature']:<28}"
                f"{str(row['Condition']):<22}"
                f"N+={int(row['Positive_N']):>3}"
                f" rate+={pct(row['Positive_Event_Rate']):>8}"
                f"N-={int(row['Negative_N']):>3}"
                f" rate-={pct(row['Negative_Event_Rate']):>8}"
                f" Δ={pct(row['Delta_Event_Rate']):>8}"
            )

    # ----------------------------------------------------------------------
    # Symbol LOO
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SYMBOL LEAVE-ONE-OUT")
    print("=" * 80)

    symbol_result = symbol_loo(
        df,
        numeric_features,
    )

    if not symbol_result.empty:
        symbol_result.to_csv(
            SYMBOL_LOO_OUTPUT,
            index=False,
        )

        for feature in numeric_features:
            local = symbol_result[
                symbol_result["Feature"] == feature
            ]

            values = pd.to_numeric(
                local["Spearman"],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            print(
                f"{feature:<28}"
                f" median={fmt(values.median()):>9}"
                f" min={fmt(values.min()):>9}"
                f" max={fmt(values.max()):>9}"
            )

    # ----------------------------------------------------------------------
    # Year LOO
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("YEAR LEAVE-ONE-OUT")
    print("=" * 80)

    year_result = year_loo(
        df,
        numeric_features,
    )

    if not year_result.empty:
        year_result.to_csv(
            YEAR_LOO_OUTPUT,
            index=False,
        )

        for feature in numeric_features:
            local = year_result[
                year_result["Feature"] == feature
            ]

            values = pd.to_numeric(
                local["Spearman"],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            print(
                f"{feature:<28}"
                f" median={fmt(values.median()):>9}"
                f" min={fmt(values.min()):>9}"
                f" max={fmt(values.max()):>9}"
            )

    # ----------------------------------------------------------------------
    # Integrity checks
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("INTEGRITY")
    print("=" * 80)

    event_count = int(
        df["Frozen_Early_Adverse_Event"].sum()
    )

    retained_count = int(
        (~df["Frozen_Early_Adverse_Event"]).sum()
    )

    print(
        f"Event count    : {event_count}"
    )

    print(
        f"Retained count : {retained_count}"
    )

    print(
        f"Total count    : "
        f"{event_count + retained_count}"
    )

    if event_count == 60 and retained_count == 46:
        print(
            "Frozen population: PASS "
            "(60 events / 46 retained)"
        )
    else:
        print(
            "Frozen population: WARNING"
        )

    # ----------------------------------------------------------------------
    # Safety declaration
    # ----------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SAFETY / SCOPE")
    print("=" * 80)

    print("No production strategy changes were made.")
    print("No frozen hypothesis changes were made.")
    print("No OOS changes were made.")
    print("No trading parameters were modified.")
    print("No future information was used as a predictor.")

    print()
    print("Outputs:")
    print(f"Event   : {EVENT_OUTPUT}")
    print(f"Category: {CATEGORY_OUTPUT}")
    print(f"Numeric : {NUMERIC_OUTPUT}")
    print(f"Setup   : {SETUP_OUTPUT}")
    print(f"Symbol LOO: {SYMBOL_LOO_OUTPUT}")
    print(f"Year LOO  : {YEAR_LOO_OUTPUT}")

    print()
    print("=" * 80)
    print("EARLY_ADVERSE_ENTRY_MECHANICS_COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()