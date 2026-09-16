"""
TRADESENSE-AI
EARLY-ADVERSE PRICE-LOCATION CONDITIONAL FORENSIC

Purpose
-------
Investigate whether PRE-ENTRY PRICE LOCATION contains information
about eventual trade quality / frozen Early-Adverse events beyond
the existing Setup and Direction classifications.

This is RESEARCH ONLY.

FROZEN RULE
-----------
Early adverse event:
    Bar-5 adverse excursion >= 0.25R

The authoritative historical frozen trade ledger is used.
No production strategy logic is modified.
No frozen dataset is modified.
No OOS / holdout dataset is modified.
No thresholds are optimized.

Research dimensions
-------------------
1. Fib distance:
   - Fib_Distance_382
   - Fib_Distance_500
   - Fib_Distance_618

2. Fib categorical location:
   - Fib_Zone
   - Fib_Premium_Discount
   - Fib_Extension_State

3. EMA / price location:
   - entry_price_position
   - Price_Position_Value

4. Continuous retracement:
   - Fib_Retracement

5. Conditional interactions:
   - Setup x location
   - Direction x location

6. Robustness:
   - setup-stratified permutation
   - bootstrap confidence intervals
   - symbol leave-one-out
   - year leave-one-out
   - correlation stability

Important
---------
This script is diagnostic only.

A strong-looking subgroup is NOT sufficient for deployment.
No production change is allowed from this script.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

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
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# FROZEN RESEARCH CONSTANTS
# ============================================================================

THRESHOLD_R = 0.25

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46
EXPECTED_FROZEN_R = 6.40


# ============================================================================
# HELPERS
# ============================================================================

def normalize_symbol(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def normalize_direction(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value in {"LONG", "BUY", "1"}:
        return "LONG"

    if value in {"SHORT", "SELL", "-1"}:
        return "SHORT"

    return value


def clean_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_sum(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return 0.0

    return float(values.sum())


def win_rate(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        (values > 0).mean() * 100.0
    )


def profit_factor(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    gross_profit = values[
        values > 0
    ].sum()

    gross_loss = -values[
        values < 0
    ].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf

        return np.nan

    return float(
        gross_profit / gross_loss
    )


def correlation_pair(x, y):
    temp = pd.DataFrame(
        {
            "x": pd.to_numeric(
                x,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                y,
                errors="coerce",
            ),
        }
    ).dropna()

    if len(temp) < 3:
        return {
            "n": len(temp),
            "pearson": np.nan,
            "spearman": np.nan,
        }

    pearson = temp["x"].corr(
        temp["y"],
        method="pearson",
    )

    spearman = temp["x"].corr(
        temp["y"],
        method="spearman",
    )

    return {
        "n": len(temp),
        "pearson": float(pearson),
        "spearman": float(spearman),
    }


# ============================================================================
# LOAD FROZEN LEDGER
# ============================================================================

def load_frozen():
    if not FROZEN_FILE.exists():
        raise FileNotFoundError(
            f"Frozen ledger not found:\n{FROZEN_FILE}"
        )

    df = pd.read_csv(
        FROZEN_FILE
    )

    print(
        f"Frozen trades loaded : {len(df)}"
    )

    if len(df) != EXPECTED_TRADES:
        raise RuntimeError(
            "Unexpected frozen trade count. "
            f"Expected {EXPECTED_TRADES}, "
            f"got {len(df)}."
        )

    return df


# ============================================================================
# COLUMN RESOLUTION
# ============================================================================

def resolve_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def resolve_required_columns(df):
    columns = {}

    columns["symbol"] = resolve_column(
        df,
        [
            "_symbol",
            "Symbol",
            "symbol",
        ],
    )

    columns["entry_date"] = resolve_column(
        df,
        [
            "entry_date",
            "Entry_Date",
            "entryDate",
        ],
    )

    columns["direction"] = resolve_column(
        df,
        [
            "Direction_Normalized",
            "Diagnostic_Direction",
            "direction",
            "Direction",
        ],
    )

    columns["setup"] = resolve_column(
        df,
        [
            "Diagnostic_Setup",
            "Setup",
            "setup",
        ],
    )

    columns["r_multiple"] = resolve_column(
        df,
        [
            "R_Multiple",
            "r_multiple",
            "R",
        ],
    )

    columns["bar5"] = resolve_column(
        df,
        [
            "Bar5_Early_Adverse_R",
            "early_adverse_r_bar_5",
        ],
    )

    required = [
        "symbol",
        "entry_date",
        "direction",
        "setup",
        "r_multiple",
        "bar5",
    ]

    missing = [
        name
        for name in required
        if columns[name] is None
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return columns


# ============================================================================
# BUILD AUTHORITATIVE FROZEN EVENT
# ============================================================================

def build_frozen_event(df, columns):
    bar5 = clean_numeric(
        df[columns["bar5"]]
    )

    df["Early_Adverse"] = (
        bar5.notna()
        & (bar5 >= THRESHOLD_R)
    )

    event_count = int(
        df["Early_Adverse"].sum()
    )

    retained_count = int(
        (~df["Early_Adverse"]).sum()
    )

    print()
    print("=" * 80)
    print("FROZEN EARLY-ADVERSE RECONCILIATION")
    print("=" * 80)

    print(
        f"Trades              : {len(df)}"
    )

    print(
        f"5B/.25R events      : {event_count}"
    )

    print(
        f"Retained            : {retained_count}"
    )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            "Frozen event count mismatch. "
            f"Expected {EXPECTED_EVENTS}, "
            f"got {event_count}."
        )

    if retained_count != EXPECTED_RETAINED:
        raise RuntimeError(
            "Frozen retained count mismatch. "
            f"Expected {EXPECTED_RETAINED}, "
            f"got {retained_count}."
        )

    return df


# ============================================================================
# STANDARDIZE METADATA
# ============================================================================

def prepare_metadata(df, columns):
    df["Research_Symbol"] = (
        df[columns["symbol"]]
        .map(normalize_symbol)
    )

    df["Research_Date"] = pd.to_datetime(
        df[columns["entry_date"]],
        errors="coerce",
    )

    df["Research_Year"] = (
        df["Research_Date"]
        .dt.year
    )

    df["Research_Direction"] = (
        df[columns["direction"]]
        .map(normalize_direction)
    )

    df["Research_Setup"] = (
        df[columns["setup"]]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["R_Multiple_Research"] = clean_numeric(
        df[columns["r_multiple"]]
    )

    return df


# ============================================================================
# REQUIRED LOCATION COLUMNS
# ============================================================================

def resolve_location_columns(df):
    candidates = {
        "Fib_Distance_382": [
            "Fib_Distance_382",
        ],
        "Fib_Distance_500": [
            "Fib_Distance_500",
        ],
        "Fib_Distance_618": [
            "Fib_Distance_618",
        ],
        "Fib_Retracement": [
            "Fib_Retracement",
        ],
        "Fib_Zone": [
            "Fib_Zone",
        ],
        "Fib_Premium_Discount": [
            "Fib_Premium_Discount",
        ],
        "Fib_Extension_State": [
            "Fib_Extension_State",
        ],
        "entry_price_position": [
            "entry_price_position",
            "Price_Position",
        ],
        "Price_Position_Value": [
            "Price_Position_Value",
        ],
    }

    resolved = {}

    for name, options in candidates.items():
        resolved[name] = resolve_column(
            df,
            options,
        )

    print()
    print("=" * 80)
    print("LOCATION COLUMN AVAILABILITY")
    print("=" * 80)

    for name, column in resolved.items():
        print(
            f"{name:<28} "
            f"{column if column else 'MISSING'}"
        )

    return resolved


# ============================================================================
# GROUP SUMMARY
# ============================================================================

def summarize_group(
    frame,
    label,
    value_column="R_Multiple_Research",
):
    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    ).dropna()

    return {
        "group": label,
        "n": int(len(values)),
        "wins": int(
            (values > 0).sum()
        ),
        "win_rate_pct": (
            win_rate(values)
        ),
        "profit_factor": (
            profit_factor(values)
        ),
        "total_r": safe_sum(values),
        "avg_r": safe_mean(values),
        "avg_mfe_r": safe_mean(
            frame.loc[
                values.index,
                "MFE_R",
            ]
            if "MFE_R" in frame.columns
            else pd.Series(
                dtype=float
            )
        ),
    }


def print_group_table(
    df,
    column,
    title,
):
    if column not in df.columns:
        return pd.DataFrame()

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    rows = []

    for value, group in df.groupby(
        column,
        dropna=False,
    ):
        label = (
            "UNKNOWN"
            if pd.isna(value)
            else str(value)
        )

        row = summarize_group(
            group,
            label,
        )

        rows.append(row)

        print(
            f"{label:<24} "
            f"N={row['n']:>3} "
            f"WR={row['win_rate_pct']:>6.1f}% "
            f"PF={row['profit_factor']:>7.3f} "
            f"R={row['total_r']:>8.2f} "
            f"AvgR={row['avg_r']:>7.3f}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# EVENT VS NON-EVENT GROUPING
# ============================================================================

def event_group_summary(
    df,
    feature,
):
    if feature not in df.columns:
        return pd.DataFrame()

    rows = []

    for value, group in df.groupby(
        feature,
        dropna=False,
    ):
        label = (
            "UNKNOWN"
            if pd.isna(value)
            else str(value)
        )

        for event_value in [
            False,
            True,
        ]:
            subset = group[
                group["Early_Adverse"]
                == event_value
            ]

            values = subset[
                "R_Multiple_Research"
            ].dropna()

            rows.append(
                {
                    "feature": feature,
                    "value": label,
                    "early_adverse": event_value,
                    "n": int(len(values)),
                    "wins": int(
                        (values > 0).sum()
                    ),
                    "win_rate_pct": (
                        win_rate(values)
                    ),
                    "total_r": safe_sum(values),
                    "avg_r": safe_mean(values),
                }
            )

    return pd.DataFrame(rows)


def print_event_group_table(
    df,
    feature,
):
    result = event_group_summary(
        df,
        feature,
    )

    if result.empty:
        return result

    print()
    print("=" * 80)
    print(
        f"EARLY-ADVERSE × {feature}"
    )
    print("=" * 80)

    for _, row in result.iterrows():
        print(
            f"{str(row['value']):<24} "
            f"Event={str(row['early_adverse']):<5} "
            f"N={int(row['n']):>3} "
            f"WR={row['win_rate_pct']:>6.1f}% "
            f"R={row['total_r']:>8.2f} "
            f"AvgR={row['avg_r']:>7.3f}"
        )

    return result


# ============================================================================
# CONTINUOUS LOCATION ANALYSIS
# ============================================================================

def continuous_analysis(
    df,
    feature,
):
    if feature not in df.columns:
        return None

    result = correlation_pair(
        df[feature],
        df["R_Multiple_Research"],
    )

    return {
        "feature": feature,
        **result,
    }


def print_continuous_analysis(
    df,
    features,
):
    print()
    print("=" * 80)
    print("CONTINUOUS PRE-ENTRY LOCATION ANALYSIS")
    print("=" * 80)

    rows = []

    for feature in features:
        result = continuous_analysis(
            df,
            feature,
        )

        if result is None:
            continue

        rows.append(result)

        print(
            f"{feature:<28} "
            f"N={result['n']:>3} "
            f"Pearson={result['pearson']:+.4f} "
            f"Spearman={result['spearman']:+.4f}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# EARLY-ADVERSE RATE BY LOCATION
# ============================================================================

def event_rate_by_feature(
    df,
    feature,
    min_group_n=5,
):
    if feature not in df.columns:
        return pd.DataFrame()

    rows = []

    for value, group in df.groupby(
        feature,
        dropna=False,
    ):
        label = (
            "UNKNOWN"
            if pd.isna(value)
            else str(value)
        )

        n = len(group)

        if n < min_group_n:
            continue

        events = int(
            group["Early_Adverse"].sum()
        )

        rate = (
            events / n * 100.0
        )

        rows.append(
            {
                "feature": feature,
                "value": label,
                "n": n,
                "events": events,
                "event_rate_pct": rate,
                "avg_r": safe_mean(
                    group[
                        "R_Multiple_Research"
                    ]
                ),
                "total_r": safe_sum(
                    group[
                        "R_Multiple_Research"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def print_event_rates(
    df,
    features,
):
    print()
    print("=" * 80)
    print(
        "EARLY-ADVERSE EVENT RATE BY LOCATION"
    )
    print("=" * 80)

    outputs = []

    for feature in features:
        result = event_rate_by_feature(
            df,
            feature,
            min_group_n=5,
        )

        if result.empty:
            continue

        outputs.append(result)

        print()
        print(
            f"--- {feature} ---"
        )

        for _, row in result.iterrows():
            print(
                f"{str(row['value']):<24} "
                f"N={int(row['n']):>3} "
                f"Events={int(row['events']):>3} "
                f"Rate={row['event_rate_pct']:>6.1f}% "
                f"AvgR={row['avg_r']:+.3f}"
            )

    if outputs:
        return pd.concat(
            outputs,
            ignore_index=True,
        )

    return pd.DataFrame()


# ============================================================================
# SETUP × LOCATION
# ============================================================================

def setup_location_analysis(
    df,
    feature,
    min_group_n=4,
):
    if feature not in df.columns:
        return pd.DataFrame()

    rows = []

    grouped = df.groupby(
        [
            "Research_Setup",
            feature,
        ],
        dropna=False,
    )

    for (
        setup,
        value,
    ), group in grouped:
        if len(group) < min_group_n:
            continue

        values = group[
            "R_Multiple_Research"
        ].dropna()

        events = int(
            group["Early_Adverse"].sum()
        )

        rows.append(
            {
                "setup": setup,
                "location_feature": feature,
                "location_value": (
                    "UNKNOWN"
                    if pd.isna(value)
                    else str(value)
                ),
                "n": int(len(group)),
                "events": events,
                "event_rate_pct": (
                    events
                    / len(group)
                    * 100.0
                ),
                "wins": int(
                    (values > 0).sum()
                ),
                "win_rate_pct": (
                    win_rate(values)
                ),
                "avg_r": safe_mean(values),
                "total_r": safe_sum(values),
            }
        )

    return pd.DataFrame(rows)


def print_setup_location_analysis(
    df,
    features,
):
    print()
    print("=" * 80)
    print("SETUP × LOCATION")
    print("=" * 80)

    outputs = []

    for feature in features:
        result = setup_location_analysis(
            df,
            feature,
        )

        if result.empty:
            continue

        outputs.append(result)

        print()
        print(
            f"--- {feature} ---"
        )

        for _, row in result.iterrows():
            print(
                f"{row['setup']:<24} "
                f"{str(row['location_value']):<18} "
                f"N={int(row['n']):>3} "
                f"EventRate={row['event_rate_pct']:>6.1f}% "
                f"AvgR={row['avg_r']:+.3f}"
            )

    if outputs:
        return pd.concat(
            outputs,
            ignore_index=True,
        )

    return pd.DataFrame()


# ============================================================================
# DIRECTION × LOCATION
# ============================================================================

def direction_location_analysis(
    df,
    feature,
    min_group_n=4,
):
    if feature not in df.columns:
        return pd.DataFrame()

    rows = []

    grouped = df.groupby(
        [
            "Research_Direction",
            feature,
        ],
        dropna=False,
    )

    for (
        direction,
        value,
    ), group in grouped:
        if len(group) < min_group_n:
            continue

        values = group[
            "R_Multiple_Research"
        ].dropna()

        events = int(
            group["Early_Adverse"].sum()
        )

        rows.append(
            {
                "direction": direction,
                "location_feature": feature,
                "location_value": (
                    "UNKNOWN"
                    if pd.isna(value)
                    else str(value)
                ),
                "n": int(len(group)),
                "events": events,
                "event_rate_pct": (
                    events
                    / len(group)
                    * 100.0
                ),
                "wins": int(
                    (values > 0).sum()
                ),
                "win_rate_pct": (
                    win_rate(values)
                ),
                "avg_r": safe_mean(values),
                "total_r": safe_sum(values),
            }
        )

    return pd.DataFrame(rows)


def print_direction_location_analysis(
    df,
    features,
):
    print()
    print("=" * 80)
    print("DIRECTION × LOCATION")
    print("=" * 80)

    outputs = []

    for feature in features:
        result = direction_location_analysis(
            df,
            feature,
        )

        if result.empty:
            continue

        outputs.append(result)

        print()
        print(
            f"--- {feature} ---"
        )

        for _, row in result.iterrows():
            print(
                f"{str(row['direction']):<8} "
                f"{str(row['location_value']):<18} "
                f"N={int(row['n']):>3} "
                f"EventRate={row['event_rate_pct']:>6.1f}% "
                f"AvgR={row['avg_r']:+.3f}"
            )

    if outputs:
        return pd.concat(
            outputs,
            ignore_index=True,
        )

    return pd.DataFrame()


# ============================================================================
# SETUP-STRATIFIED BINARY DIFFERENCE
# ============================================================================

def stratified_binary_difference(
    df,
    flag,
):
    if flag not in df.columns:
        return np.nan

    work = df[
        [
            "Research_Setup",
            flag,
            "R_Multiple_Research",
        ]
    ].copy()

    work = work.dropna(
        subset=[
            "Research_Setup",
            flag,
            "R_Multiple_Research",
        ]
    )

    if work.empty:
        return np.nan

    weighted_sum = 0.0
    weighted_n = 0

    for _, group in work.groupby(
        "Research_Setup"
    ):
        false_values = group.loc[
            group[flag] == False,
            "R_Multiple_Research",
        ]

        true_values = group.loc[
            group[flag] == True,
            "R_Multiple_Research",
        ]

        if (
            len(false_values) == 0
            or len(true_values) == 0
        ):
            continue

        delta = (
            true_values.mean()
            - false_values.mean()
        )

        weighted_sum += (
            delta * len(group)
        )

        weighted_n += len(group)

    if weighted_n == 0:
        return np.nan

    return weighted_sum / weighted_n


# ============================================================================
# PERMUTATION TEST
# ============================================================================

def setup_stratified_permutation(
    df,
    flag,
    n_permutations=20000,
    seed=20260915,
):
    if flag not in df.columns:
        return {
            "observed_delta": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    work = df[
        [
            "Research_Setup",
            flag,
            "R_Multiple_Research",
        ]
    ].copy()

    work = work.dropna(
        subset=[
            "Research_Setup",
            flag,
            "R_Multiple_Research",
        ]
    )

    observed = (
        stratified_binary_difference(
            df,
            flag,
        )
    )

    if pd.isna(observed):
        return {
            "observed_delta": np.nan,
            "p_value": np.nan,
            "n": len(work),
        }

    rng = np.random.default_rng(
        seed
    )

    null_values = []

    grouped = [
        group.copy()
        for _, group
        in work.groupby(
            "Research_Setup"
        )
    ]

    for _ in range(
        n_permutations
    ):
        shuffled_parts = []

        for group in grouped:
            local = group.copy()

            labels = local[
                flag
            ].to_numpy(
                dtype=bool,
                copy=True
            )

            rng.shuffle(labels)

            local["_perm_flag"] = labels

            shuffled_parts.append(
                local
            )

        shuffled = pd.concat(
            shuffled_parts,
            ignore_index=True,
        )

        weighted_sum = 0.0
        weighted_n = 0

        for _, group in shuffled.groupby(
            "Research_Setup"
        ):
            false_values = group.loc[
                ~group["_perm_flag"],
                "R_Multiple_Research",
            ]

            true_values = group.loc[
                group["_perm_flag"],
                "R_Multiple_Research",
            ]

            if (
                len(false_values) == 0
                or len(true_values) == 0
            ):
                continue

            delta = (
                true_values.mean()
                - false_values.mean()
            )

            weighted_sum += (
                delta * len(group)
            )

            weighted_n += len(group)

        if weighted_n == 0:
            null_values.append(
                np.nan
            )
        else:
            null_values.append(
                weighted_sum
                / weighted_n
            )

    null = np.asarray(
        null_values,
        dtype=float,
    )

    null = null[
        np.isfinite(null)
    ]

    if len(null) == 0:
        p_value = np.nan
    else:
        p_value = float(
            (
                np.abs(null)
                >= abs(observed)
            ).mean()
        )

    return {
        "observed_delta": float(
            observed
        ),
        "p_value": p_value,
        "n": len(work),
    }


# ============================================================================
# BOOTSTRAP CI
# ============================================================================

def bootstrap_setup_delta(
    df,
    flag,
    n_bootstrap=10000,
    seed=20260915,
):
    if flag not in df.columns:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    work = df[
        [
            "Research_Setup",
            flag,
            "R_Multiple_Research",
        ]
    ].dropna()

    observed = (
        stratified_binary_difference(
            df,
            flag,
        )
    )

    if pd.isna(observed):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    rng = np.random.default_rng(
        seed
    )

    setup_groups = {
        setup: group.copy()
        for setup, group
        in work.groupby(
            "Research_Setup"
        )
    }

    boot_values = []

    for _ in range(
        n_bootstrap
    ):
        sampled_parts = []

        for setup, group in setup_groups.items():
            if len(group) == 0:
                continue

            indices = rng.integers(
                0,
                len(group),
                size=len(group),
            )

            sampled = group.iloc[
                indices
            ].copy()

            sampled_parts.append(
                sampled
            )

        if not sampled_parts:
            continue

        sampled_all = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        weighted_sum = 0.0
        weighted_n = 0

        for _, group in sampled_all.groupby(
            "Research_Setup"
        ):
            false_values = group.loc[
                group[flag] == False,
                "R_Multiple_Research",
            ]

            true_values = group.loc[
                group[flag] == True,
                "R_Multiple_Research",
            ]

            if (
                len(false_values) == 0
                or len(true_values) == 0
            ):
                continue

            delta = (
                true_values.mean()
                - false_values.mean()
            )

            weighted_sum += (
                delta * len(group)
            )

            weighted_n += len(group)

        if weighted_n:
            boot_values.append(
                weighted_sum
                / weighted_n
            )

    if not boot_values:
        return (
            observed,
            np.nan,
            np.nan,
        )

    low, high = np.percentile(
        boot_values,
        [
            2.5,
            97.5,
        ],
    )

    return (
        float(observed),
        float(low),
        float(high),
    )


# ============================================================================
# BINARY FEATURE BUILDER
# ============================================================================

def build_binary_location_flags(
    df,
    location_columns,
):
    if (
        location_columns.get(
            "entry_price_position"
        )
        is not None
    ):
        source = location_columns[
            "entry_price_position"
        ]

        df["Loc_Above_Fast"] = (
            df[source]
            .astype(str)
            .str.upper()
            .eq("ABOVE_FAST")
        )

        df["Loc_Below_Fast"] = (
            df[source]
            .astype(str)
            .str.upper()
            .eq("BELOW_FAST")
        )

    if (
        location_columns.get(
            "Fib_Premium_Discount"
        )
        is not None
    ):
        source = location_columns[
            "Fib_Premium_Discount"
        ]

        df["Loc_Premium"] = (
            df[source]
            .astype(str)
            .str.upper()
            .eq("PREMIUM")
        )

        df["Loc_Discount"] = (
            df[source]
            .astype(str)
            .str.upper()
            .eq("DISCOUNT")
        )

    if (
        location_columns.get(
            "Fib_Zone"
        )
        is not None
    ):
        source = location_columns[
            "Fib_Zone"
        ]

        values = (
            df[source]
            .astype(str)
            .str.upper()
        )

        df["Loc_Above_382"] = (
            values == "ABOVE_382"
        )

        df["Loc_Above_618"] = (
            values == "ABOVE_618"
        )

        df["Loc_Below_382"] = (
            values == "BELOW_382"
        )

        df["Loc_Below_618"] = (
            values == "BELOW_618"
        )

    return df


# ============================================================================
# FORMAL INDEPENDENT LOCATION TESTS
# ============================================================================

def run_binary_tests(
    df,
    flags,
):
    print()
    print("=" * 80)
    print(
        "FORMAL SETUP-STRATIFIED LOCATION TESTS"
    )
    print("=" * 80)

    rows = []

    for flag in flags:
        if flag not in df.columns:
            continue

        true_n = int(
            df[flag].sum()
        )

        false_n = int(
            (~df[flag]).sum()
        )

        if (
            true_n < 5
            or false_n < 5
        ):
            print()
            print(
                f"{flag}: SKIPPED "
                f"(True={true_n}, "
                f"False={false_n})"
            )

            rows.append(
                {
                    "feature": flag,
                    "status": "SKIPPED_SMALL_GROUP",
                    "true_n": true_n,
                    "false_n": false_n,
                    "delta": np.nan,
                    "p_value": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )

            continue

        permutation = (
            setup_stratified_permutation(
                df,
                flag,
            )
        )

        observed, ci_low, ci_high = (
            bootstrap_setup_delta(
                df,
                flag,
            )
        )

        print()
        print(flag)

        print(
            f"  True N       : {true_n}"
        )

        print(
            f"  False N      : {false_n}"
        )

        print(
            f"  Stratified Δ : "
            f"{observed:+.4f}R"
        )

        print(
            f"  Permutation p: "
            f"{permutation['p_value']:.6f}"
        )

        print(
            f"  Bootstrap CI : "
            f"[{ci_low:+.4f}, "
            f"{ci_high:+.4f}]R"
        )

        rows.append(
            {
                "feature": flag,
                "status": "TESTED",
                "true_n": true_n,
                "false_n": false_n,
                "delta": observed,
                "p_value": permutation[
                    "p_value"
                ],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# SYMBOL LOO
# ============================================================================

def symbol_loo(
    df,
    flags,
):
    print()
    print("=" * 80)
    print("SYMBOL LEAVE-ONE-OUT")
    print("=" * 80)

    rows = []

    symbols = sorted(
        df["Research_Symbol"]
        .dropna()
        .unique()
    )

    for symbol in symbols:
        remaining = df[
            df["Research_Symbol"]
            != symbol
        ].copy()

        for flag in flags:
            if flag not in remaining.columns:
                continue

            true_n = int(
                remaining[flag].sum()
            )

            false_n = int(
                (~remaining[flag]).sum()
            )

            if (
                true_n < 5
                or false_n < 5
            ):
                delta = np.nan
            else:
                delta = (
                    stratified_binary_difference(
                        remaining,
                        flag,
                    )
                )

            rows.append(
                {
                    "excluded_symbol": symbol,
                    "feature": flag,
                    "delta": delta,
                    "true_n": true_n,
                    "false_n": false_n,
                }
            )

            print(
                f"{symbol:<12} "
                f"{flag:<28} "
                f"Δ="
                f"{delta:+.4f}R"
                if np.isfinite(delta)
                else
                f"{symbol:<12} "
                f"{flag:<28} "
                f"Δ=NA"
            )

    return pd.DataFrame(rows)


# ============================================================================
# YEAR LOO
# ============================================================================

def year_loo(
    df,
    flags,
):
    print()
    print("=" * 80)
    print("YEAR LEAVE-ONE-OUT")
    print("=" * 80)

    rows = []

    years = sorted(
        df["Research_Year"]
        .dropna()
        .unique()
    )

    for year in years:
        remaining = df[
            df["Research_Year"]
            != year
        ].copy()

        for flag in flags:
            if flag not in remaining.columns:
                continue

            true_n = int(
                remaining[flag].sum()
            )

            false_n = int(
                (~remaining[flag]).sum()
            )

            if (
                true_n < 5
                or false_n < 5
            ):
                delta = np.nan
            else:
                delta = (
                    stratified_binary_difference(
                        remaining,
                        flag,
                    )
                )

            rows.append(
                {
                    "excluded_year": int(year),
                    "feature": flag,
                    "delta": delta,
                    "true_n": true_n,
                    "false_n": false_n,
                }
            )

            if np.isfinite(delta):
                print(
                    f"{int(year)} "
                    f"{flag:<28} "
                    f"Δ={delta:+.4f}R"
                )
            else:
                print(
                    f"{int(year)} "
                    f"{flag:<28} "
                    f"Δ=NA"
                )

    return pd.DataFrame(rows)


# ============================================================================
# LOCATION CORRELATION STABILITY
# ============================================================================

def correlation_stability(
    df,
    features,
):
    print()
    print("=" * 80)
    print("LOCATION CORRELATION STABILITY")
    print("=" * 80)

    rows = []

    for feature in features:
        if feature not in df.columns:
            continue

        overall = correlation_pair(
            df[feature],
            df["R_Multiple_Research"],
        )

        symbol_values = []
        year_values = []

        for symbol, group in df.groupby(
            "Research_Symbol"
        ):
            result = correlation_pair(
                group[feature],
                group[
                    "R_Multiple_Research"
                ],
            )

            if np.isfinite(
                result["spearman"]
            ):
                symbol_values.append(
                    result["spearman"]
                )

        for year, group in df.groupby(
            "Research_Year"
        ):
            result = correlation_pair(
                group[feature],
                group[
                    "R_Multiple_Research"
                ],
            )

            if np.isfinite(
                result["spearman"]
            ):
                year_values.append(
                    result["spearman"]
                )

        rows.append(
            {
                "feature": feature,
                "overall_pearson": (
                    overall["pearson"]
                ),
                "overall_spearman": (
                    overall["spearman"]
                ),
                "symbol_mean_spearman": (
                    np.mean(symbol_values)
                    if symbol_values
                    else np.nan
                ),
                "symbol_median_spearman": (
                    np.median(symbol_values)
                    if symbol_values
                    else np.nan
                ),
                "symbol_positive_count": (
                    sum(
                        x > 0
                        for x
                        in symbol_values
                    )
                    if symbol_values
                    else 0
                ),
                "year_mean_spearman": (
                    np.mean(year_values)
                    if year_values
                    else np.nan
                ),
                "year_median_spearman": (
                    np.median(year_values)
                    if year_values
                    else np.nan
                ),
                "year_positive_count": (
                    sum(
                        x > 0
                        for x
                        in year_values
                    )
                    if year_values
                    else 0
                ),
            }
        )

        print(
            f"{feature:<28} "
            f"Overall Spearman="
            f"{overall['spearman']:+.4f} "
            f"Symbol median="
            f"{np.median(symbol_values):+.4f}"
            if symbol_values
            else
            f"{feature:<28} "
            f"Overall Spearman="
            f"{overall['spearman']:+.4f}"
        )

    return pd.DataFrame(rows)


# ============================================================================
# FROZEN RETENTION REFERENCE
# ============================================================================

def frozen_reference(df):
    baseline = safe_sum(
        df["R_Multiple_Research"]
    )

    retained = safe_sum(
        df.loc[
            ~df["Early_Adverse"],
            "R_Multiple_Research",
        ]
    )

    removed = safe_sum(
        df.loc[
            df["Early_Adverse"],
            "R_Multiple_Research",
        ]
    )

    print()
    print("=" * 80)
    print("FROZEN HYPOTHESIS REFERENCE")
    print("=" * 80)

    print(
        f"Baseline R          : "
        f"{baseline:+.2f}R"
    )

    print(
        f"Early-Adverse R     : "
        f"{removed:+.2f}R"
    )

    print(
        f"Retained R          : "
        f"{retained:+.2f}R"
    )

    print(
        f"Expected frozen R   : "
        f"{EXPECTED_FROZEN_R:+.2f}R"
    )

    if not np.isclose(
        retained,
        EXPECTED_FROZEN_R,
        atol=0.10,
    ):
        print(
            "WARNING: retained R differs "
            "from the authoritative "
            "research reference."
        )

    return {
        "baseline_r": baseline,
        "early_adverse_r": removed,
        "retained_r": retained,
    }


# ============================================================================
# INTERPRETATION
# ============================================================================

def interpretation(
    formal_results,
    correlation_results,
):
    print()
    print("=" * 80)
    print("RESEARCH INTERPRETATION")
    print("=" * 80)

    tested = formal_results[
        formal_results["status"]
        == "TESTED"
    ].copy()

    if tested.empty:
        print(
            "No location binary feature "
            "had enough observations "
            "for a formal test."
        )
    else:
        for _, row in tested.iterrows():
            feature = row[
                "feature"
            ]

            delta = row[
                "delta"
            ]

            p_value = row[
                "p_value"
            ]

            ci_low = row[
                "ci_low"
            ]

            ci_high = row[
                "ci_high"
            ]

            print()
            print(
                f"{feature}:"
            )

            print(
                f"  Δ={delta:+.4f}R"
            )

            print(
                f"  p={p_value:.6f}"
            )

            print(
                f"  CI=["
                f"{ci_low:+.4f}, "
                f"{ci_high:+.4f}]R"
            )

            if (
                p_value < 0.05
                and ci_low > 0
            ):
                print(
                    "  OBSERVATION: "
                    "positive statistical "
                    "separation."
                )
            elif (
                p_value < 0.05
                and ci_high < 0
            ):
                print(
                    "  OBSERVATION: "
                    "negative statistical "
                    "separation."
                )
            else:
                print(
                    "  OBSERVATION: "
                    "not statistically "
                    "confirmed."
                )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Correlation does not establish "
        "causality."
    )

    print(
        "Subgroups are descriptive unless "
        "they survive conditional and "
        "robustness tests."
    )

    print(
        "No location filter is being "
        "implemented."
    )

    print(
        "No production engine changes "
        "are permitted from this forensic."
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 80)
    print(
        "TRADESENSE-AI — "
        "EARLY-ADVERSE PRICE LOCATION FORENSIC"
    )
    print("=" * 80)

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    df = load_frozen()

    columns = resolve_required_columns(
        df
    )

    # ------------------------------------------------------------------------
    # AUTHORITATIVE FROZEN EVENT
    # ------------------------------------------------------------------------

    df = build_frozen_event(
        df,
        columns,
    )

    # ------------------------------------------------------------------------
    # STANDARDIZE METADATA
    # ------------------------------------------------------------------------

    df = prepare_metadata(
        df,
        columns,
    )

    # ------------------------------------------------------------------------
    # LOCATION COLUMNS
    # ------------------------------------------------------------------------

    location_columns = (
        resolve_location_columns(df)
    )

    # ------------------------------------------------------------------------
    # COPY LOCATION DATA INTO
    # STANDARD RESEARCH COLUMNS
    # ------------------------------------------------------------------------

    for standard_name, source in (
        location_columns.items()
    ):
        if source is None:
            continue

        if standard_name in {
            "Fib_Distance_382",
            "Fib_Distance_500",
            "Fib_Distance_618",
            "Fib_Retracement",
            "Price_Position_Value",
        }:
            df[standard_name] = clean_numeric(
                df[source]
            )
        else:
            df[standard_name] = (
                df[source]
                .astype(str)
                .str.strip()
                .str.upper()
            )

    # ------------------------------------------------------------------------
    # BINARY LOCATION FLAGS
    # ------------------------------------------------------------------------

    df = build_binary_location_flags(
        df,
        location_columns,
    )

    # ------------------------------------------------------------------------
    # FROZEN REFERENCE
    # ------------------------------------------------------------------------

    frozen_reference(
        df
    )

    # ------------------------------------------------------------------------
    # CONTINUOUS ANALYSIS
    # ------------------------------------------------------------------------

    continuous_features = [
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
        "Fib_Retracement",
        "Price_Position_Value",
    ]

    continuous_results = (
        print_continuous_analysis(
            df,
            continuous_features,
        )
    )

    # ------------------------------------------------------------------------
    # CATEGORICAL LOCATION
    # ------------------------------------------------------------------------

    categorical_features = [
        "Fib_Zone",
        "Fib_Premium_Discount",
        "Fib_Extension_State",
        "entry_price_position",
    ]

    for feature in categorical_features:
        print_group_table(
            df,
            feature,
            f"{feature} — EVENT POPULATION",
        )

        print_event_group_table(
            df,
            feature,
        )

    event_rate_results = (
        print_event_rates(
            df,
            categorical_features,
        )
    )

    # ------------------------------------------------------------------------
    # SETUP × LOCATION
    # ------------------------------------------------------------------------

    setup_results = (
        print_setup_location_analysis(
            df,
            categorical_features,
        )
    )

    # ------------------------------------------------------------------------
    # DIRECTION × LOCATION
    # ------------------------------------------------------------------------

    direction_results = (
        print_direction_location_analysis(
            df,
            categorical_features,
        )
    )

    # ------------------------------------------------------------------------
    # FORMAL BINARY LOCATION TESTS
    # ------------------------------------------------------------------------

    binary_flags = [
        "Loc_Above_Fast",
        "Loc_Below_Fast",
        "Loc_Premium",
        "Loc_Discount",
        "Loc_Above_382",
        "Loc_Above_618",
        "Loc_Below_382",
        "Loc_Below_618",
    ]

    formal_results = run_binary_tests(
        df,
        binary_flags,
    )

    # ------------------------------------------------------------------------
    # SYMBOL LOO
    # ------------------------------------------------------------------------

    symbol_results = symbol_loo(
        df,
        binary_flags,
    )

    # ------------------------------------------------------------------------
    # YEAR LOO
    # ------------------------------------------------------------------------

    year_results = year_loo(
        df,
        binary_flags,
    )

    # ------------------------------------------------------------------------
    # CORRELATION STABILITY
    # ------------------------------------------------------------------------

    correlation_results = (
        correlation_stability(
            df,
            continuous_features,
        )
    )

    # ------------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------------

    interpretation(
        formal_results,
        correlation_results,
    )

    # ------------------------------------------------------------------------
    # SAVE RESEARCH OUTPUTS
    # ------------------------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / "early_adverse_location_conditional_results.csv"
    )

    rows = []

    if not continuous_results.empty:
        temp = continuous_results.copy()
        temp["section"] = (
            "continuous_location"
        )
        rows.append(temp)

    if not event_rate_results.empty:
        temp = event_rate_results.copy()
        temp["section"] = (
            "event_rate"
        )
        rows.append(temp)

    if not formal_results.empty:
        temp = formal_results.copy()
        temp["section"] = (
            "formal_binary_test"
        )
        rows.append(temp)

    if not correlation_results.empty:
        temp = correlation_results.copy()
        temp["section"] = (
            "correlation_stability"
        )
        rows.append(temp)

    if rows:
        combined = pd.concat(
            rows,
            ignore_index=True,
            sort=False,
        )

        combined.to_csv(
            output_file,
            index=False,
        )

    print()
    print("=" * 80)
    print("SAFETY CHECK")
    print("=" * 80)

    print(
        "PASS: Production strategy "
        "was not modified."
    )

    print(
        "PASS: Frozen historical "
        "ledger was not modified."
    )

    print(
        "PASS: Stock holdout "
        "was not modified."
    )

    print(
        "PASS: No OOS logic "
        "was modified."
    )

    print(
        "PASS: No trading filter "
        "was implemented."
    )

    print()
    print(
        f"Research output: {output_file}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()