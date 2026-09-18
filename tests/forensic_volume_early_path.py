"""
TradeSense-AI — Volume × Early-Path Forensic Audit

Research-only forensic analysis.

Purpose
-------
Determine whether entry-time Volume contains information about the
post-entry early path of trades, especially the Bar1 -> Bar2 path
associated with the frozen early-adverse hypothesis.

Frozen hypothesis
-----------------
Early adverse >= 0.25R within first 5 entry bars.

Frozen population
-----------------
106 trades
60 frozen early-adverse events
46 retained

Research guardrails
-------------------
- Read-only against production/frozen/OOS data.
- No production strategy modification.
- No frozen dataset modification.
- No OOS modification.
- No threshold optimization.
- Volume uses the already validated global median.
- No ML.
- No feature selection.
- Descriptive / association evidence only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    fisher_exact,
    mannwhitneyu,
    spearmanr,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

FROZEN_TRADES_FILE = (
    ROOT / "tests" / "output" / "tradesense_trades.csv"
)

VOLUME_FILE = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_lineage"
    / "volume_lineage_per_trade.csv"
)

EARLY_PATH_FILE = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "early_adverse_path"
    / "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_early_path"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FROZEN CONFIGURATION
# =============================================================================

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_BARS = 5

VOLUME_MEDIAN = 1.192481227

VOLUME_CATEGORY_ORDER = [
    "LOW",
    "NORMAL",
    "HIGH",
    "VERY_HIGH",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_symbol(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    aliases = {
        "HDFC BANK": "HDFCBANK",
        "HDFCBANK.NS": "HDFCBANK",
        "ICICI BANK": "ICICIBANK",
        "ICICIBANK.NS": "ICICIBANK",
        "INFOSYS": "INFY",
        "INFY.NS": "INFY",
        "RELIANCE.NS": "RELIANCE",
        "SBIN.NS": "SBIN",
        "TCS.NS": "TCS",
    }

    return aliases.get(value, value)


def normalize_direction(value) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip().upper()

    if value in {"LONG", "BUY", "1", "+1"}:
        return "LONG"

    if value in {"SHORT", "SELL", "-1"}:
        return "SHORT"

    return value


def normalize_date(value):
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return pd.NaT

    return parsed.normalize()


def make_trade_key(
    symbol,
    direction,
    entry_date,
) -> str:
    symbol = normalize_symbol(symbol)
    direction = normalize_direction(direction)
    entry_date = normalize_date(entry_date)

    if pd.isna(entry_date):
        date_text = ""
    else:
        date_text = entry_date.strftime("%Y-%m-%d")

    return f"{symbol}|{direction}|{date_text}"


def proportion(numerator, denominator) -> float:
    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100.0


def safe_float(value):
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def median_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def event_rate(series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean() * 100.0)


def risk_difference(
    event_rate_group_1: float,
    event_rate_group_0: float,
) -> float:
    if pd.isna(event_rate_group_1) or pd.isna(event_rate_group_0):
        return np.nan

    return float(event_rate_group_1 - event_rate_group_0)


def fisher_2x2(
    group,
    event,
):
    temp = pd.DataFrame(
        {
            "group": pd.to_numeric(group, errors="coerce"),
            "event": pd.to_numeric(event, errors="coerce"),
        }
    ).dropna()

    if temp.empty:
        return np.nan, np.nan

    table = pd.crosstab(
        temp["group"].astype(int),
        temp["event"].astype(int),
    )

    table = table.reindex(
        index=[0, 1],
        columns=[0, 1],
        fill_value=0,
    )

    try:
        odds_ratio, p_value = fisher_exact(table.values)

        return float(odds_ratio), float(p_value)

    except Exception:
        return np.nan, np.nan


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
):
    if total <= 0:
        return np.nan, np.nan

    p = successes / total

    denominator = 1.0 + (z * z / total)

    center = (
        p
        + (z * z / (2.0 * total))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / total
            )
            + (
                z * z / (4.0 * total * total)
            )
        )
        / denominator
    )

    return (
        float(center - margin),
        float(center + margin),
    )


def difference_ci(
    group_1,
    group_0,
):
    """
    Approximate Newcombe-style CI for difference of two proportions
    using Wilson intervals.

    Returns percentage-point bounds.
    """

    g1 = pd.to_numeric(group_1, errors="coerce").dropna().astype(int)
    g0 = pd.to_numeric(group_0, errors="coerce").dropna().astype(int)

    n1 = len(g1)
    n0 = len(g0)

    if n1 == 0 or n0 == 0:
        return np.nan, np.nan

    x1 = int(g1.sum())
    x0 = int(g0.sum())

    lo1, hi1 = wilson_interval(x1, n1)
    lo0, hi0 = wilson_interval(x0, n0)

    return (
        float((lo1 - hi0) * 100.0),
        float((hi1 - lo0) * 100.0),
    )


def mann_whitney_summary(
    event_values,
    retained_values,
):
    event_values = (
        pd.to_numeric(event_values, errors="coerce")
        .dropna()
        .to_numpy()
    )

    retained_values = (
        pd.to_numeric(retained_values, errors="coerce")
        .dropna()
        .to_numpy()
    )

    if len(event_values) == 0 or len(retained_values) == 0:
        return {
            "u": np.nan,
            "p": np.nan,
            "event_median": np.nan,
            "retained_median": np.nan,
            "median_difference": np.nan,
        }

    try:
        u_stat, p_value = mannwhitneyu(
            event_values,
            retained_values,
            alternative="two-sided",
        )
    except Exception:
        u_stat = np.nan
        p_value = np.nan

    return {
        "u": float(u_stat),
        "p": float(p_value),
        "event_median": float(np.median(event_values)),
        "retained_median": float(np.median(retained_values)),
        "median_difference": float(
            np.median(event_values)
            - np.median(retained_values)
        ),
    }


def correlation_summary(
    x,
    y,
):
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    if len(frame) < 3:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
        }

    pearson = frame["x"].corr(frame["y"], method="pearson")

    try:
        spearman, _ = spearmanr(
            frame["x"],
            frame["y"],
        )
    except Exception:
        spearman = np.nan

    return {
        "pearson": float(pearson)
        if pd.notna(pearson)
        else np.nan,
        "spearman": float(spearman)
        if pd.notna(spearman)
        else np.nan,
    }


def classify_volume(value):
    value = safe_float(value)

    if pd.isna(value):
        return np.nan

    if value < 1.0:
        return "LOW"

    if value < 1.5:
        return "NORMAL"

    if value < 2.0:
        return "HIGH"

    return "VERY_HIGH"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_frozen_trades():
    if not FROZEN_TRADES_FILE.exists():
        raise FileNotFoundError(
            f"Frozen trades file not found: {FROZEN_TRADES_FILE}"
        )

    trades = pd.read_csv(FROZEN_TRADES_FILE)

    symbol_col = None

    for candidate in [
        "_symbol",
        "symbol",
        "Symbol",
    ]:
        if candidate in trades.columns:
            symbol_col = candidate
            break

    if symbol_col is None:
        raise ValueError(
            "Could not locate frozen symbol column."
        )

    direction_col = None

    for candidate in [
        "direction",
        "Direction",
    ]:
        if candidate in trades.columns:
            direction_col = candidate
            break

    if direction_col is None:
        raise ValueError(
            "Could not locate frozen direction column."
        )

    date_col = None

    for candidate in [
        "entry_date",
        "Entry_Date",
        "Entry Date",
    ]:
        if candidate in trades.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError(
            "Could not locate frozen entry date column."
        )

    trades["_symbol"] = trades[symbol_col].map(
        normalize_symbol
    )

    trades["_direction"] = trades[direction_col].map(
        normalize_direction
    )

    trades["_entry_date"] = trades[date_col].map(
        normalize_date
    )

    trades["_trade_key"] = [
        make_trade_key(
            symbol,
            direction,
            entry_date,
        )
        for symbol, direction, entry_date
        in zip(
            trades["_symbol"],
            trades["_direction"],
            trades["_entry_date"],
        )
    ]

    if len(trades) != EXPECTED_TRADES:
        raise RuntimeError(
            "FROZEN DATASET INTEGRITY FAILURE: "
            f"expected {EXPECTED_TRADES}, "
            f"found {len(trades)}"
        )

    duplicate_count = int(
        trades["_trade_key"].duplicated().sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "FROZEN DATASET DUPLICATE KEY FAILURE: "
            f"{duplicate_count}"
        )

    event_col = None

    for candidate in [
        "early_adverse_r_bar_5",
        "early_adverse_r_5",
    ]:
        if candidate in trades.columns:
            event_col = candidate
            break

    if event_col is None:
        raise ValueError(
            "Could not locate frozen Bar5 early-adverse field."
        )

    frozen_bar5 = pd.to_numeric(
        trades[event_col],
        errors="coerce",
    )

    trades["frozen_event"] = (
        frozen_bar5 >= EARLY_ADVERSE_THRESHOLD_R
    ).astype(int)

    trades["frozen_event"] = trades["frozen_event"].where(
        frozen_bar5.notna(),
        0,
    )

    event_count = int(
        trades["frozen_event"].sum()
    )

    retained_count = (
        len(trades)
        - event_count
    )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            "FROZEN EVENT COUNT FAILURE: "
            f"expected {EXPECTED_EVENTS}, "
            f"found {event_count}"
        )

    if retained_count != EXPECTED_RETAINED:
        raise RuntimeError(
            "FROZEN RETAINED COUNT FAILURE: "
            f"expected {EXPECTED_RETAINED}, "
            f"found {retained_count}"
        )

    return trades


def load_volume():
    if not VOLUME_FILE.exists():
        raise FileNotFoundError(
            f"Volume lineage file not found: {VOLUME_FILE}"
        )

    volume = pd.read_csv(VOLUME_FILE)

    required_columns = [
        "trade_key",
        "symbol",
        "direction",
        "signal_date",
        "entry_date",
        "volume_ratio",
        "volume_category",
        "lineage_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in volume.columns
    ]

    if missing:
        raise ValueError(
            "Volume lineage file missing required columns: "
            + ", ".join(missing)
        )

    volume["_symbol"] = volume["symbol"].map(
        normalize_symbol
    )

    volume["_direction"] = volume["direction"].map(
        normalize_direction
    )

    volume["_entry_date"] = volume["entry_date"].map(
        normalize_date
    )

    volume["_trade_key"] = [
        make_trade_key(
            symbol,
            direction,
            entry_date,
        )
        for symbol, direction, entry_date
        in zip(
            volume["_symbol"],
            volume["_direction"],
            volume["_entry_date"],
        )
    ]

    duplicate_count = int(
        volume["_trade_key"].duplicated().sum()
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "VOLUME DUPLICATE KEY FAILURE: "
            f"{duplicate_count}"
        )

    volume["_volume_ratio"] = pd.to_numeric(
        volume["volume_ratio"],
        errors="coerce",
    )

    if volume["_volume_ratio"].isna().any():
        missing_ratio = int(
            volume["_volume_ratio"].isna().sum()
        )

        raise RuntimeError(
            "VOLUME RATIO AVAILABILITY FAILURE: "
            f"{missing_ratio} rows have missing volume_ratio"
        )

    volume["_volume_category"] = (
        volume["volume_category"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_categories = int(
        (
            ~volume["_volume_category"]
            .isin(VOLUME_CATEGORY_ORDER)
        ).sum()
    )

    if invalid_categories != 0:
        raise RuntimeError(
            "VOLUME CATEGORY FAILURE: "
            f"{invalid_categories} invalid categories"
        )

    invalid_lineage = int(
        (
            volume["lineage_status"]
            .astype(str)
            .str.upper()
            != "PASS"
        ).sum()
    )

    if invalid_lineage != 0:
        raise RuntimeError(
            "VOLUME LINEAGE STATUS FAILURE: "
            f"{invalid_lineage} rows are not PASS"
        )

    return volume


def load_early_path():
    if not EARLY_PATH_FILE.exists():
        raise FileNotFoundError(
            f"Early path file not found: {EARLY_PATH_FILE}"
        )

    path = pd.read_csv(EARLY_PATH_FILE)

    required = [
        "symbol",
        "direction",
        "entry_date",
        "bar_number",
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "bar_adverse_atr",
        "bar_favorable_atr",
        "close_from_entry_r",
        "event_label",
        "stored_frozen_event",
    ]

    missing = [
        column
        for column in required
        if column not in path.columns
    ]

    if missing:
        raise ValueError(
            "Early path file missing columns: "
            + ", ".join(missing)
        )

    path["_symbol"] = path["symbol"].map(
        normalize_symbol
    )

    path["_direction"] = path["direction"].map(
        normalize_direction
    )

    path["_entry_date"] = path["entry_date"].map(
        normalize_date
    )

    path["_trade_key"] = [
        make_trade_key(
            symbol,
            direction,
            entry_date,
        )
        for symbol, direction, entry_date
        in zip(
            path["_symbol"],
            path["_direction"],
            path["_entry_date"],
        )
    ]

    path["bar_number"] = pd.to_numeric(
        path["bar_number"],
        errors="coerce",
    )

    return path


# =============================================================================
# EARLY-PATH FEATURE EXTRACTION
# =============================================================================

def extract_path_features(path):
    """
    Convert the Bar1/Bar2 rows into one trade-level dataframe.
    """

    selected = path[
        path["bar_number"].isin([1, 2])
    ].copy()

    metric_columns = [
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "bar_adverse_atr",
        "bar_favorable_atr",
        "close_from_entry_r",
    ]

    records = []

    for trade_key, group in selected.groupby(
        "_trade_key",
        sort=False,
    ):
        group = group.sort_values("bar_number")

        record = {
            "_trade_key": trade_key,
        }

        for bar in [1, 2]:
            row = group[
                group["bar_number"] == bar
            ]

            if row.empty:
                for metric in metric_columns:
                    record[f"bar{bar}_{metric}"] = np.nan
            else:
                row = row.iloc[0]

                for metric in metric_columns:
                    record[f"bar{bar}_{metric}"] = (
                        safe_float(row[metric])
                    )

        # ---------------------------------------------------------------------
        # Bar1 -> Bar2 changes
        # ---------------------------------------------------------------------

        record["delta_mae_r"] = (
            record["bar2_cumulative_mae_r"]
            - record["bar1_cumulative_mae_r"]
        )

        record["delta_mfe_r"] = (
            record["bar2_cumulative_mfe_r"]
            - record["bar1_cumulative_mfe_r"]
        )

        record["delta_net_excursion_r"] = (
            record["bar2_net_excursion_r"]
            - record["bar1_net_excursion_r"]
        )

        record["delta_bar_adverse_r"] = (
            record["bar2_bar_adverse_r"]
            - record["bar1_bar_adverse_r"]
        )

        record["delta_bar_favorable_r"] = (
            record["bar2_bar_favorable_r"]
            - record["bar1_bar_favorable_r"]
        )

        record["delta_close_from_entry_r"] = (
            record["bar2_close_from_entry_r"]
            - record["bar1_close_from_entry_r"]
        )

        # ---------------------------------------------------------------------
        # Pre-specified Bar2 state flags
        # ---------------------------------------------------------------------

        record["bar2_mae_increased"] = int(
            pd.notna(record["delta_mae_r"])
            and record["delta_mae_r"] > 0
        )

        record["bar2_mfe_increased"] = int(
            pd.notna(record["delta_mfe_r"])
            and record["delta_mfe_r"] > 0
        )

        record["bar2_net_deteriorated"] = int(
            pd.notna(record["delta_net_excursion_r"])
            and record["delta_net_excursion_r"] < 0
        )

        record["bar2_close_deteriorated"] = int(
            pd.notna(record["delta_close_from_entry_r"])
            and record["delta_close_from_entry_r"] < 0
        )

        record["bar2_adverse_worsened"] = int(
            pd.notna(record["delta_bar_adverse_r"])
            and record["delta_bar_adverse_r"] > 0
        )

        record["bar2_favorable_failed_to_improve"] = int(
            pd.notna(record["delta_bar_favorable_r"])
            and record["delta_bar_favorable_r"] <= 0
        )

        record["adverse_up_favorable_not_up"] = int(
            record["bar2_adverse_worsened"] == 1
            and record["bar2_favorable_failed_to_improve"] == 1
        )

        record["adverse_up_net_down"] = int(
            record["bar2_adverse_worsened"] == 1
            and record["bar2_net_deteriorated"] == 1
        )

        record["adverse_up_favorable_not_up_net_down"] = int(
            record["bar2_adverse_worsened"] == 1
            and record["bar2_favorable_failed_to_improve"] == 1
            and record["bar2_net_deteriorated"] == 1
        )

        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# MERGE
# =============================================================================

def build_trade_level_dataset():
    frozen = load_frozen_trades()
    volume = load_volume()
    path = load_early_path()

    path_features = extract_path_features(path)

    merged = (
        frozen[
            [
                "_trade_key",
                "_symbol",
                "_direction",
                "_entry_date",
                "frozen_event",
            ]
        ]
        .merge(
            volume[
                [
                    "_trade_key",
                    "_volume_ratio",
                    "_volume_category",
                ]
            ],
            on="_trade_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            path_features,
            on="_trade_key",
            how="left",
            validate="one_to_one",
        )
    )

    merged["volume_high"] = (
        merged["_volume_ratio"]
        >= VOLUME_MEDIAN
    ).astype(int)

    merged["volume_low"] = (
        merged["volume_high"] == 0
    ).astype(int)

    merged["population"] = np.where(
        merged["frozen_event"] == 1,
        "EVENT",
        "RETAINED",
    )

    return frozen, volume, path, merged


# =============================================================================
# VOLUME × PATH CONTINUOUS ANALYSIS
# =============================================================================

PATH_METRICS = [
    ("bar1_bar_adverse_r", "Bar1 adverse excursion"),
    ("bar1_bar_favorable_r", "Bar1 favorable excursion"),
    ("bar1_cumulative_mae_r", "Bar1 cumulative MAE"),
    ("bar1_cumulative_mfe_r", "Bar1 cumulative MFE"),
    ("bar1_net_excursion_r", "Bar1 net excursion"),
    ("bar2_bar_adverse_r", "Bar2 adverse excursion"),
    ("bar2_bar_favorable_r", "Bar2 favorable excursion"),
    ("bar2_cumulative_mae_r", "Bar2 cumulative MAE"),
    ("bar2_cumulative_mfe_r", "Bar2 cumulative MFE"),
    ("bar2_net_excursion_r", "Bar2 net excursion"),
    ("delta_mae_r", "Bar1→Bar2 MAE change"),
    ("delta_mfe_r", "Bar1→Bar2 MFE change"),
    ("delta_net_excursion_r", "Bar1→Bar2 net change"),
    ("delta_bar_adverse_r", "Bar1→Bar2 adverse change"),
    ("delta_bar_favorable_r", "Bar1→Bar2 favorable change"),
    ("delta_close_from_entry_r", "Bar1→Bar2 close change"),
    (
        "bar1_bar_adverse_atr",
        "Bar1 adverse ATR",
    ),
    (
        "bar1_bar_favorable_atr",
        "Bar1 favorable ATR",
    ),
    (
        "bar2_bar_adverse_atr",
        "Bar2 adverse ATR",
    ),
    (
        "bar2_bar_favorable_atr",
        "Bar2 favorable ATR",
    ),
]


def analyze_continuous_volume_path(merged):
    rows = []

    for column, label in PATH_METRICS:

        if column not in merged.columns:
            continue

        high = merged[
            merged["volume_high"] == 1
        ][column]

        low = merged[
            merged["volume_high"] == 0
        ][column]

        high_values = pd.to_numeric(
            high,
            errors="coerce",
        ).dropna()

        low_values = pd.to_numeric(
            low,
            errors="coerce",
        ).dropna()

        if len(high_values) == 0 or len(low_values) == 0:
            continue

        try:
            u_stat, p_value = mannwhitneyu(
                high_values,
                low_values,
                alternative="two-sided",
            )
        except Exception:
            u_stat = np.nan
            p_value = np.nan

        rows.append(
            {
                "metric": column,
                "description": label,
                "high_volume_n": int(len(high_values)),
                "low_volume_n": int(len(low_values)),
                "high_volume_mean": float(
                    high_values.mean()
                ),
                "low_volume_mean": float(
                    low_values.mean()
                ),
                "high_volume_median": float(
                    high_values.median()
                ),
                "low_volume_median": float(
                    low_values.median()
                ),
                "median_difference_high_minus_low": float(
                    high_values.median()
                    - low_values.median()
                ),
                "mean_difference_high_minus_low": float(
                    high_values.mean()
                    - low_values.mean()
                ),
                "mann_whitney_u": float(u_stat)
                if pd.notna(u_stat)
                else np.nan,
                "p_value": float(p_value)
                if pd.notna(p_value)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# VOLUME × PATH EVENT-STATE ANALYSIS
# =============================================================================

PATH_STATES = [
    (
        "bar2_mae_increased",
        "Bar2 MAE increased",
    ),
    (
        "bar2_mfe_increased",
        "Bar2 MFE increased",
    ),
    (
        "bar2_net_deteriorated",
        "Bar2 net deteriorated",
    ),
    (
        "bar2_close_deteriorated",
        "Bar2 close deteriorated",
    ),
    (
        "bar2_adverse_worsened",
        "Bar2 adverse excursion worsened",
    ),
    (
        "bar2_favorable_failed_to_improve",
        "Bar2 favorable excursion failed to improve",
    ),
    (
        "adverse_up_favorable_not_up",
        "Adverse up + favorable not up",
    ),
    (
        "adverse_up_net_down",
        "Adverse up + net down",
    ),
    (
        "adverse_up_favorable_not_up_net_down",
        "Adverse up + favorable not up + net down",
    ),
]


def analyze_volume_path_states(merged):
    rows = []

    for column, label in PATH_STATES:

        if column not in merged.columns:
            continue

        temp = merged[
            [
                "volume_high",
                "frozen_event",
                column,
            ]
        ].dropna()

        if temp.empty:
            continue

        temp["volume_high"] = (
            temp["volume_high"].astype(int)
        )

        temp[column] = (
            temp[column].astype(int)
        )

        high = temp[
            temp["volume_high"] == 1
        ][column]

        low = temp[
            temp["volume_high"] == 0
        ][column]

        high_n = len(high)
        low_n = len(low)

        high_rate = (
            high.mean() * 100.0
            if high_n
            else np.nan
        )

        low_rate = (
            low.mean() * 100.0
            if low_n
            else np.nan
        )

        rd = (
            high_rate - low_rate
            if pd.notna(high_rate)
            and pd.notna(low_rate)
            else np.nan
        )

        ci_low, ci_high = difference_ci(
            high,
            low,
        )

        odds_ratio, p_value = fisher_2x2(
            temp["volume_high"],
            temp[column],
        )

        rows.append(
            {
                "state": column,
                "description": label,
                "high_volume_n": int(high_n),
                "high_volume_rate_pct": high_rate,
                "low_volume_n": int(low_n),
                "low_volume_rate_pct": low_rate,
                "risk_difference_high_minus_low_pp": rd,
                "rd_ci_low_pp": ci_low,
                "rd_ci_high_pp": ci_high,
                "fisher_or": odds_ratio,
                "fisher_p": p_value,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# EVENT-STRATIFIED PATH ANALYSIS
# =============================================================================

def analyze_event_stratified_path(merged):
    """
    Within frozen EVENT vs RETAINED populations, compare the early path
    between high and low Volume.

    This asks whether Volume is associated with the magnitude/direction
    of the early path even after the frozen event label is fixed.
    """

    rows = []

    for population_name, population_value in [
        ("EVENT", 1),
        ("RETAINED", 0),
    ]:

        subset = merged[
            merged["frozen_event"] == population_value
        ].copy()

        for column, label in PATH_METRICS:

            if column not in subset.columns:
                continue

            high = pd.to_numeric(
                subset.loc[
                    subset["volume_high"] == 1,
                    column,
                ],
                errors="coerce",
            ).dropna()

            low = pd.to_numeric(
                subset.loc[
                    subset["volume_high"] == 0,
                    column,
                ],
                errors="coerce",
            ).dropna()

            if len(high) == 0 or len(low) == 0:
                continue

            try:
                u_stat, p_value = mannwhitneyu(
                    high,
                    low,
                    alternative="two-sided",
                )
            except Exception:
                u_stat = np.nan
                p_value = np.nan

            rows.append(
                {
                    "population": population_name,
                    "metric": column,
                    "description": label,
                    "high_volume_n": int(len(high)),
                    "low_volume_n": int(len(low)),
                    "high_volume_median": float(
                        high.median()
                    ),
                    "low_volume_median": float(
                        low.median()
                    ),
                    "median_difference_high_minus_low": float(
                        high.median()
                        - low.median()
                    ),
                    "high_volume_mean": float(
                        high.mean()
                    ),
                    "low_volume_mean": float(
                        low.mean()
                    ),
                    "mean_difference_high_minus_low": float(
                        high.mean()
                        - low.mean()
                    ),
                    "mann_whitney_u": float(u_stat)
                    if pd.notna(u_stat)
                    else np.nan,
                    "p_value": float(p_value)
                    if pd.notna(p_value)
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# VOLUME CATEGORY ANALYSIS
# =============================================================================

def analyze_volume_categories(merged):
    rows = []

    for category in VOLUME_CATEGORY_ORDER:

        subset = merged[
            merged["_volume_category"] == category
        ]

        n = len(subset)

        if n == 0:
            continue

        events = int(
            subset["frozen_event"].sum()
        )

        rows.append(
            {
                "volume_category": category,
                "n": int(n),
                "events": events,
                "retained": int(n - events),
                "event_rate_pct": proportion(
                    events,
                    n,
                ),
                "bar2_net_deteriorated_rate_pct": (
                    subset[
                        "bar2_net_deteriorated"
                    ].mean()
                    * 100.0
                ),
                "bar2_favorable_failed_rate_pct": (
                    subset[
                        "bar2_favorable_failed_to_improve"
                    ].mean()
                    * 100.0
                ),
                "bar2_mae_increased_rate_pct": (
                    subset[
                        "bar2_mae_increased"
                    ].mean()
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# ROBUSTNESS
# =============================================================================

def analyze_subgroup(
    merged,
    subgroup_column,
):
    rows = []

    if subgroup_column not in merged.columns:
        return pd.DataFrame()

    values = (
        merged[subgroup_column]
        .dropna()
        .astype(str)
        .unique()
    )

    for value in sorted(values):

        subset = merged[
            merged[subgroup_column].astype(str)
            == value
        ]

        high = subset[
            subset["volume_high"] == 1
        ]

        low = subset[
            subset["volume_high"] == 0
        ]

        if len(high) == 0 or len(low) == 0:
            continue

        high_events = int(
            high["frozen_event"].sum()
        )

        low_events = int(
            low["frozen_event"].sum()
        )

        high_rate = (
            high["frozen_event"].mean()
            * 100.0
        )

        low_rate = (
            low["frozen_event"].mean()
            * 100.0
        )

        rd = (
            high_rate
            - low_rate
        )

        ci_low, ci_high = difference_ci(
            high["frozen_event"],
            low["frozen_event"],
        )

        odds_ratio, p_value = fisher_2x2(
            pd.concat(
                [
                    pd.Series(
                        [1] * len(high)
                    ),
                    pd.Series(
                        [0] * len(low)
                    ),
                ],
                ignore_index=True,
            ),
            pd.concat(
                [
                    high["frozen_event"],
                    low["frozen_event"],
                ],
                ignore_index=True,
            ),
        )

        rows.append(
            {
                "subgroup_dimension": subgroup_column,
                "subgroup": value,
                "n": int(len(subset)),
                "high_volume_n": int(len(high)),
                "low_volume_n": int(len(low)),
                "high_volume_event_rate_pct": high_rate,
                "low_volume_event_rate_pct": low_rate,
                "risk_difference_high_minus_low_pp": rd,
                "rd_ci_low_pp": ci_low,
                "rd_ci_high_pp": ci_high,
                "fisher_or": odds_ratio,
                "fisher_p": p_value,
                "high_volume_bar2_net_deteriorated_pct": (
                    high[
                        "bar2_net_deteriorated"
                    ].mean()
                    * 100.0
                ),
                "low_volume_bar2_net_deteriorated_pct": (
                    low[
                        "bar2_net_deteriorated"
                    ].mean()
                    * 100.0
                ),
                "high_volume_bar2_favorable_failed_pct": (
                    high[
                        "bar2_favorable_failed_to_improve"
                    ].mean()
                    * 100.0
                ),
                "low_volume_bar2_favorable_failed_pct": (
                    low[
                        "bar2_favorable_failed_to_improve"
                    ].mean()
                    * 100.0
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# LEAD / PATH TIMING
# =============================================================================

def analyze_volume_path_timing(
    merged,
    path,
):
    """
    Determine whether high vs low Volume differs in how early the
    frozen 0.25R threshold is reached.

    Timing is reconstructed directly from the validated five-bar
    early-path dataframe so Bars 1-5 are genuinely available.

    This is descriptive and uses the already frozen label/path.
    """

    rows = []

    path_timing = path[
        [
            "_trade_key",
            "bar_number",
            "cumulative_mae_r",
        ]
    ].copy()

    path_timing["bar_number"] = pd.to_numeric(
        path_timing["bar_number"],
        errors="coerce",
    )

    path_timing["cumulative_mae_r"] = pd.to_numeric(
        path_timing["cumulative_mae_r"],
        errors="coerce",
    )

    path_timing = path_timing[
        path_timing["bar_number"].isin([1, 2, 3, 4, 5])
    ].copy()

    path_timing = path_timing.merge(
        merged[
            [
                "_trade_key",
                "volume_high",
            ]
        ],
        on="_trade_key",
        how="inner",
        validate="many_to_one",
    )

    for volume_group, group_value in [
        ("LOW_VOLUME", 0),
        ("HIGH_VOLUME", 1),
    ]:

        subset = path_timing[
            path_timing["volume_high"] == group_value
        ].copy()

        if subset.empty:
            continue

        timing_values = []

        for trade_key, trade_path in subset.groupby(
            "_trade_key",
            sort=False,
        ):

            trade_path = trade_path.sort_values(
                "bar_number"
            )

            found_bar = np.nan

            for _, row in trade_path.iterrows():

                bar = safe_float(
                    row["bar_number"]
                )

                value = safe_float(
                    row["cumulative_mae_r"]
                )

                if (
                    pd.notna(bar)
                    and pd.notna(value)
                    and value >= EARLY_ADVERSE_THRESHOLD_R
                ):
                    found_bar = int(bar)
                    break

            if pd.notna(found_bar):
                timing_values.append(found_bar)

        if timing_values:
            rows.append(
                {
                    "volume_group": volume_group,
                    "n_reaching_0_25r": int(
                        len(timing_values)
                    ),
                    "median_first_crossing_bar": float(
                        np.median(timing_values)
                    ),
                    "mean_first_crossing_bar": float(
                        np.mean(timing_values)
                    ),
                    "bar1_cross_pct": proportion(
                        sum(
                            x == 1
                            for x in timing_values
                        ),
                        len(timing_values),
                    ),
                    "bar2_cross_pct": proportion(
                        sum(
                            x == 2
                            for x in timing_values
                        ),
                        len(timing_values),
                    ),
                    "bar3_cross_pct": proportion(
                        sum(
                            x == 3
                            for x in timing_values
                        ),
                        len(timing_values),
                    ),
                    "bar4_cross_pct": proportion(
                        sum(
                            x == 4
                            for x in timing_values
                        ),
                        len(timing_values),
                    ),
                    "bar5_cross_pct": proportion(
                        sum(
                            x == 5
                            for x in timing_values
                        ),
                        len(timing_values),
                    ),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# INTEGRITY
# =============================================================================

def build_integrity(
    frozen,
    volume,
    path,
    merged,
):
    rows = []

    frozen_duplicate_keys = int(
        frozen["_trade_key"].duplicated().sum()
    )

    volume_duplicate_keys = int(
        volume["_trade_key"].duplicated().sum()
    )

    path_duplicate_rows = int(
        path[
            [
                "_trade_key",
                "bar_number",
            ]
        ]
        .duplicated()
        .sum()
    )

    path_duplicate_trade_keys = int(
        path[
            "_trade_key"
        ]
        .duplicated()
        .sum()
    )

    path_missing_bar_number = int(
        path["bar_number"].isna().sum()
    )

    merged_missing_volume = int(
        merged["_volume_ratio"].isna().sum()
    )

    merged_missing_path = int(
        merged[
            "bar1_cumulative_mae_r"
        ].isna()
        .sum()
    )

    invalid_categories = int(
        (
            ~merged["_volume_category"]
            .isin(VOLUME_CATEGORY_ORDER)
        ).sum()
    )

    invalid_event_values = int(
        (
            ~merged["frozen_event"]
            .isin([0, 1])
        ).sum()
    )

    rows.extend(
        [
            {
                "check": "frozen_trade_count",
                "expected": EXPECTED_TRADES,
                "actual": len(frozen),
                "status": (
                    "PASS"
                    if len(frozen)
                    == EXPECTED_TRADES
                    else "FAIL"
                ),
            },
            {
                "check": "frozen_event_count",
                "expected": EXPECTED_EVENTS,
                "actual": int(
                    frozen["frozen_event"].sum()
                ),
                "status": (
                    "PASS"
                    if int(
                        frozen["frozen_event"].sum()
                    )
                    == EXPECTED_EVENTS
                    else "FAIL"
                ),
            },
            {
                "check": "frozen_retained_count",
                "expected": EXPECTED_RETAINED,
                "actual": int(
                    (
                        frozen["frozen_event"]
                        == 0
                    ).sum()
                ),
                "status": (
                    "PASS"
                    if int(
                        (
                            frozen["frozen_event"]
                            == 0
                        ).sum()
                    )
                    == EXPECTED_RETAINED
                    else "FAIL"
                ),
            },
            {
                "check": "frozen_duplicate_keys",
                "expected": 0,
                "actual": frozen_duplicate_keys,
                "status": (
                    "PASS"
                    if frozen_duplicate_keys == 0
                    else "FAIL"
                ),
            },
            {
                "check": "volume_duplicate_keys",
                "expected": 0,
                "actual": volume_duplicate_keys,
                "status": (
                    "PASS"
                    if volume_duplicate_keys == 0
                    else "FAIL"
                ),
            },
            
{
    "check": "path_duplicate_rows",
    "expected": 0,
    "actual": path_duplicate_rows,
    "status": (
        "PASS"
        if path_duplicate_rows == 0
        else "FAIL"
    ),
},
    {
        "check": "path_duplicate_trade_keys_allowed",
        "expected": "multiple bars per trade",
        "actual": path_duplicate_trade_keys,
        "status": (
            "PASS"
            if path_duplicate_trade_keys > 0
            else "FAIL"
        ),
    },
    {
        "check": "path_missing_bar_number",
        "expected": 0,
        "actual": path_missing_bar_number,
        "status": (
            "PASS"
            if path_missing_bar_number == 0
            else "FAIL"
        ),
    },
            {
                "check": "missing_volume_ratio",
                "expected": 0,
                "actual": merged_missing_volume,
                "status": (
                    "PASS"
                    if merged_missing_volume == 0
                    else "FAIL"
                ),
            },
            {
                "check": "missing_bar1_path",
                "expected": 0,
                "actual": merged_missing_path,
                "status": (
                    "PASS"
                    if merged_missing_path == 0
                    else "FAIL"
                ),
            },
            {
                "check": "invalid_volume_categories",
                "expected": 0,
                "actual": invalid_categories,
                "status": (
                    "PASS"
                    if invalid_categories == 0
                    else "FAIL"
                ),
            },
            {
                "check": "invalid_frozen_event_values",
                "expected": 0,
                "actual": invalid_event_values,
                "status": (
                    "PASS"
                    if invalid_event_values == 0
                    else "FAIL"
                ),
            },
            {
                "check": "merged_trade_count",
                "expected": EXPECTED_TRADES,
                "actual": len(merged),
                "status": (
                    "PASS"
                    if len(merged)
                    == EXPECTED_TRADES
                    else "FAIL"
                ),
            },
        ]
    )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "TRADESENSE-AI — VOLUME × EARLY-PATH FORENSIC AUDIT"
    )
    print("=" * 100)

    print()
    print("FROZEN HYPOTHESIS:")
    print(
        "Early adverse >= "
        f"{EARLY_ADVERSE_THRESHOLD_R:.2f}R "
        f"within first {MAX_BARS} entry bars"
    )

    print()
    print("FROZEN POPULATION:")
    print(
        f"{EXPECTED_TRADES} trades | "
        f"{EXPECTED_EVENTS} events | "
        f"{EXPECTED_RETAINED} retained"
    )

    print()
    print("=" * 100)
    print("LOADING VALIDATED RESEARCH INPUTS")
    print("=" * 100)

    (
        frozen,
        volume,
        path,
        merged,
    ) = build_trade_level_dataset()

    print(
        f"Frozen trades loaded       : {len(frozen)}"
    )

    print(
        f"Volume lineage rows        : {len(volume)}"
    )

    print(
        f"Early-path rows            : {len(path)}"
    )

    print(
        f"Trade-level merged rows    : {len(merged)}"
    )

    print()
    print("=" * 100)
    print("VOLUME REFERENCE")
    print("=" * 100)

    low = merged[
        merged["volume_high"] == 0
    ]

    high = merged[
        merged["volume_high"] == 1
    ]

    low_event_rate = (
        low["frozen_event"].mean()
        * 100.0
    )

    high_event_rate = (
        high["frozen_event"].mean()
        * 100.0
    )

    rd = (
        high_event_rate
        - low_event_rate
    )

    odds_ratio, p_value = fisher_2x2(
        merged["volume_high"],
        merged["frozen_event"],
    )

    print(
        f"Global Volume median       : "
        f"{VOLUME_MEDIAN:.9f}"
    )

    print(
        f"Low Volume n               : "
        f"{len(low)}"
    )

    print(
        f"High Volume n              : "
        f"{len(high)}"
    )

    print(
        f"Low Volume event rate      : "
        f"{low_event_rate:.6f}%"
    )

    print(
        f"High Volume event rate     : "
        f"{high_event_rate:.6f}%"
    )

    print(
        f"Volume RD high-low         : "
        f"{rd:.6f} pp"
    )

    print(
        f"Volume Fisher OR           : "
        f"{odds_ratio}"
    )

    print(
        f"Volume Fisher p            : "
        f"{p_value}"
    )

    # -------------------------------------------------------------------------
    # CONTINUOUS PATH
    # -------------------------------------------------------------------------

    continuous = analyze_continuous_volume_path(
        merged
    )

    print()
    print("=" * 100)
    print(
        "VOLUME × CONTINUOUS EARLY-PATH DIFFERENCES"
    )
    print("=" * 100)

    for _, row in continuous.iterrows():

        print()
        print(
            f"{row['metric']}"
        )

        print(
            f"  High Volume median       : "
            f"{row['high_volume_median']:.6f}"
        )

        print(
            f"  Low Volume median        : "
            f"{row['low_volume_median']:.6f}"
        )

        print(
            f"  Median difference        : "
            f"{row['median_difference_high_minus_low']:.6f}"
        )

        print(
            f"  Mean difference          : "
            f"{row['mean_difference_high_minus_low']:.6f}"
        )

        print(
            f"  Mann-Whitney p           : "
            f"{row['p_value']}"
        )

    # -------------------------------------------------------------------------
    # STATE ANALYSIS
    # -------------------------------------------------------------------------

    states = analyze_volume_path_states(
        merged
    )

    print()
    print("=" * 100)
    print(
        "VOLUME × BAR1→BAR2 PATH STATES"
    )
    print("=" * 100)

    for _, row in states.iterrows():

        print()
        print(
            f"{row['state']}"
        )

        print(
            f"  High Volume rate         : "
            f"{row['high_volume_rate_pct']:.6f}%"
        )

        print(
            f"  Low Volume rate          : "
            f"{row['low_volume_rate_pct']:.6f}%"
        )

        print(
            f"  RD high-low              : "
            f"{row['risk_difference_high_minus_low_pp']:.6f} pp"
        )

        print(
            f"  95% CI                   : "
            f"[{row['rd_ci_low_pp']:.6f}, "
            f"{row['rd_ci_high_pp']:.6f}]"
        )

        print(
            f"  Fisher OR                : "
            f"{row['fisher_or']}"
        )

        print(
            f"  Fisher p                 : "
            f"{row['fisher_p']}"
        )

    # -------------------------------------------------------------------------
    # EVENT-STRATIFIED ANALYSIS
    # -------------------------------------------------------------------------

    event_stratified = (
        analyze_event_stratified_path(
            merged
        )
    )

    print()
    print("=" * 100)
    print(
        "WITHIN-POPULATION VOLUME × EARLY-PATH ANALYSIS"
    )
    print("=" * 100)

    important_metrics = [
        "bar1_cumulative_mae_r",
        "bar1_cumulative_mfe_r",
        "bar1_net_excursion_r",
        "bar2_cumulative_mae_r",
        "bar2_cumulative_mfe_r",
        "bar2_net_excursion_r",
        "delta_mae_r",
        "delta_mfe_r",
        "delta_net_excursion_r",
        "delta_bar_adverse_r",
        "delta_bar_favorable_r",
    ]

    if event_stratified.empty:
        display_rows = pd.DataFrame()
    else:
        display_rows = event_stratified[
            event_stratified["metric"].isin(
                important_metrics
            )
        ]

    for _, row in display_rows.iterrows():

        print()
        print(
            f"{row['population']} — "
            f"{row['metric']}"
        )

        print(
            f"  High Volume median       : "
            f"{row['high_volume_median']:.6f}"
        )

        print(
            f"  Low Volume median        : "
            f"{row['low_volume_median']:.6f}"
        )

        print(
            f"  Difference               : "
            f"{row['median_difference_high_minus_low']:.6f}"
        )

        print(
            f"  Mann-Whitney p           : "
            f"{row['p_value']}"
        )

    # -------------------------------------------------------------------------
    # CATEGORY ANALYSIS
    # -------------------------------------------------------------------------

    categories = analyze_volume_categories(
        merged
    )

    print()
    print("=" * 100)
    print(
        "VOLUME CATEGORY × EARLY PATH"
    )
    print("=" * 100)

    for _, row in categories.iterrows():

        print(
            f"{row['volume_category']:12s} "
            f"n={int(row['n']):3d} "
            f"events={int(row['events']):3d} "
            f"event_rate={row['event_rate_pct']:.2f}% "
            f"| net_down={row['bar2_net_deteriorated_rate_pct']:.2f}% "
            f"| favorable_failed={row['bar2_favorable_failed_rate_pct']:.2f}%"
        )

    # -------------------------------------------------------------------------
    # ROBUSTNESS
    # -------------------------------------------------------------------------

    robustness_frames = []

    for dimension in [
        "_symbol",
        "_direction",
    ]:

        frame = analyze_subgroup(
            merged,
            dimension,
        )

        if not frame.empty:
            robustness_frames.append(
                frame
            )

    merged["_year"] = (
        merged["_entry_date"]
        .dt.year
    )

    frame = analyze_subgroup(
        merged,
        "_year",
    )

    if not frame.empty:
        robustness_frames.append(
            frame
        )

    # Setup column if available in frozen data.
    if "Setup" in frozen.columns:
        setup_map = frozen[
            [
                "_trade_key",
                "Setup",
            ]
        ].copy()

        merged = merged.merge(
            setup_map,
            on="_trade_key",
            how="left",
            validate="one_to_one",
        )

        frame = analyze_subgroup(
            merged,
            "Setup",
        )

        if not frame.empty:
            robustness_frames.append(
                frame
            )

    if robustness_frames:
        robustness = pd.concat(
            robustness_frames,
            ignore_index=True,
        )
    else:
        robustness = pd.DataFrame()

    print()
    print("=" * 100)
    print(
        "VOLUME × EARLY-PATH ROBUSTNESS"
    )
    print("=" * 100)

    if robustness.empty:
        print(
            "No subgroup robustness rows available."
        )
    else:
        for dimension in [
            "_year",
            "_symbol",
            "_direction",
            "Setup",
        ]:

            subset = robustness[
                robustness[
                    "subgroup_dimension"
                ]
                == dimension
            ]

            if subset.empty:
                continue

            print()
            print(
                f"[{dimension}]"
            )

            for _, row in subset.iterrows():

                print(
                    f"  {row['subgroup']}: "
                    f"n={int(row['n'])} "
                    f"RD={row['risk_difference_high_minus_low_pp']:.2f}pp "
                    f"p={row['fisher_p']}"
                )

    # -------------------------------------------------------------------------
    # TIMING
    # -------------------------------------------------------------------------

    timing = analyze_volume_path_timing(
        merged,
        path,
    )

    print()
    print("=" * 100)
    print(
        "VOLUME × 0.25R PATH TIMING"
    )
    print("=" * 100)

    if timing.empty:
        print(
            "No 0.25R crossing timing rows available."
        )
    else:
        for _, row in timing.iterrows():

            print(
                f"{row['volume_group']}: "
                f"n={int(row['n_reaching_0_25r'])}, "
                f"median first crossing="
                f"{row['median_first_crossing_bar']:.2f} bars, "
                f"mean="
                f"{row['mean_first_crossing_bar']:.2f}"
            )

    # -------------------------------------------------------------------------
    # INTEGRITY
    # -------------------------------------------------------------------------

    integrity = build_integrity(
        frozen,
        volume,
        path,
        merged,
    )

    integrity_pass = bool(
        (integrity["status"] == "PASS").all()
    )

    # -------------------------------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------------------------------

    trade_level_output = (
        OUTPUT_DIR
        / "volume_early_path_trade_level.csv"
    )

    continuous_output = (
        OUTPUT_DIR
        / "volume_early_path_continuous.csv"
    )

    states_output = (
        OUTPUT_DIR
        / "volume_early_path_states.csv"
    )

    event_stratified_output = (
        OUTPUT_DIR
        / "volume_early_path_event_stratified.csv"
    )

    categories_output = (
        OUTPUT_DIR
        / "volume_early_path_categories.csv"
    )

    robustness_output = (
        OUTPUT_DIR
        / "volume_early_path_robustness.csv"
    )

    timing_output = (
        OUTPUT_DIR
        / "volume_early_path_timing.csv"
    )

    integrity_output = (
        OUTPUT_DIR
        / "volume_early_path_integrity.csv"
    )

    summary_output = (
        OUTPUT_DIR
        / "volume_early_path_summary.json"
    )

    merged.to_csv(
        trade_level_output,
        index=False,
    )

    continuous.to_csv(
        continuous_output,
        index=False,
    )

    states.to_csv(
        states_output,
        index=False,
    )

    event_stratified.to_csv(
        event_stratified_output,
        index=False,
    )

    categories.to_csv(
        categories_output,
        index=False,
    )

    robustness.to_csv(
        robustness_output,
        index=False,
    )

    timing.to_csv(
        timing_output,
        index=False,
    )

    integrity.to_csv(
        integrity_output,
        index=False,
    )

    summary = {
        "frozen_hypothesis": (
            "Early adverse >= 0.25R within first "
            "5 entry bars"
        ),
        "frozen_trades": EXPECTED_TRADES,
        "frozen_events": EXPECTED_EVENTS,
        "frozen_retained": EXPECTED_RETAINED,
        "volume_global_median": VOLUME_MEDIAN,
        "volume_low_event_rate_pct": low_event_rate,
        "volume_high_event_rate_pct": high_event_rate,
        "volume_risk_difference_high_minus_low_pp": rd,
        "volume_fisher_or": odds_ratio,
        "volume_fisher_p": p_value,
        "continuous_path_rows": len(continuous),
        "path_state_rows": len(states),
        "event_stratified_rows": len(
            event_stratified
        ),
        "category_rows": len(categories),
        "robustness_rows": len(robustness),
        "timing_rows": len(timing),
        "integrity_pass": integrity_pass,
        "threshold_optimization": False,
        "feature_selection": False,
        "ml_used": False,
        "production_modified": False,
        "frozen_data_modified": False,
        "oos_modified": False,
        "deployment_rule_created": False,
        "interpretation": (
            "This audit establishes only descriptive and "
            "conditional association evidence between entry-time "
            "Volume and early post-entry path behavior. "
            "It does not establish causality, independent "
            "predictive performance, or a deployment rule."
        ),
    }

    with open(
        summary_output,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        f"Trade level       : {trade_level_output}"
    )

    print(
        f"Continuous path   : {continuous_output}"
    )

    print(
        f"Path states       : {states_output}"
    )

    print(
        f"Event stratified  : {event_stratified_output}"
    )

    print(
        f"Categories        : {categories_output}"
    )

    print(
        f"Robustness        : {robustness_output}"
    )

    print(
        f"Timing            : {timing_output}"
    )

    print(
        f"Integrity         : {integrity_output}"
    )

    print(
        f"Summary           : {summary_output}"
    )

    print()

    if integrity_pass:

        print(
            "STATUS: VOLUME × EARLY-PATH FORENSIC AUDIT PASS"
        )

        print()
        print("IMPORTANT:")
        print(
            "This establishes early-path association evidence only."
        )
        print(
            "Volume uses the fixed global median."
        )
        print(
            "No threshold optimization or feature selection "
            "was performed."
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

    else:

        print(
            "STATUS: VOLUME × EARLY-PATH FORENSIC AUDIT FAIL"
        )

        print()
        print(
            "Review volume_early_path_integrity.csv "
            "before interpreting any research output."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()