"""
TradeSense-AI — Bar 1→2 Outcome Separation Forensic Analysis

PURPOSE
-------
Test whether the already-identified Bar-1 → Bar-2 conditions separate
eventual trade outcomes, rather than merely reproducing the frozen
5-bar / 0.25R event label.

FROZEN POPULATION
-----------------
106 total trades
60 frozen early-adverse events
46 retained

PRIMARY EARLY CONDITIONS
------------------------
A. Bar-1 net path state
   -> Bar-2 net deterioration

B. Bar-1 favorable excursion state
   -> Bar-2 favorable failure

C. Bar-1 adverse excursion state
   -> Bar-2 net deterioration

OUTCOME VARIABLES
-----------------
The analysis attempts to identify the existing production outcome
columns from tests/output/tradesense_trades.csv.

Expected/common candidates include:
- final R / trade R
- MFE R
- MAE R

The script does NOT reconstruct or modify production outcomes.

METHODOLOGICAL RULES
--------------------
- Bar-1 states use the already-established GLOBAL medians.
- No threshold optimization.
- No machine learning.
- No new strategy rule.
- No OOS modification.
- Frozen event label is NOT used as the only outcome.
- Outcome metrics are evaluated independently of the frozen event label.
- Primary comparison is outcome distribution between Bar-2 state=1
  and Bar-2 state=0.
- Results are descriptive and inferential, not deployment evidence.

IMPORTANT
---------
The Bar-2 state itself is derived from the first two post-entry bars.
Therefore, outcome separation can establish association with later
trade outcomes, but does not by itself establish causality.

OUTPUTS
-------
tests/output/research/early_adverse_bar12_outcome_separation/
    bar12_outcome_separation_per_trade.csv
    bar12_outcome_separation_primary.csv
    bar12_outcome_separation_family_summary.csv
    bar12_outcome_separation_integrity.csv
    bar12_outcome_separation_summary.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    fisher_exact,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAJECTORY_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_path"
)

TRADE_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

TRAJECTORY_CANDIDATES = [
    Path(
        "tests/output/research/early_adverse_bar12_trajectory/"
        "early_adverse_bar12_trajectory_per_trade.csv"
    ),
]

OUTPUT_DIR = Path(
    "tests/output/research/"
    "early_adverse_bar12_outcome_separation"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

ALPHA = 0.05

FAMILIES = {
    "A_net_path": {
        "bar1_feature": "bar1_net_excursion_r",
        "bar2_state": "bar2_net_deteriorated",
        "state_name": "Bar-2 net deterioration",
    },
    "B_favorable": {
        "bar1_feature": "bar1_cumulative_mfe_r",
        "bar2_state": "bar2_favorable_failed_to_improve",
        "state_name": "Bar-2 favorable failure",
    },
    "C_adverse": {
        "bar1_feature": "bar1_cumulative_mae_r",
        "bar2_state": "bar2_net_deteriorated",
        "state_name": "Bar-2 net deterioration",
    },
}


# ============================================================
# COLUMN ALIASES
# ============================================================

FINAL_R_ALIASES = [
    "final_r",
    "trade_r",
    "net_r",
    "realized_r",
    "realized_trade_r",
    "pnl_r",
    "profit_r",
    "return_r",
    "r_multiple",
    "final_R",
    "trade_R",
]

MFE_R_ALIASES = [
    "mfe_r",
    "final_mfe_r",
    "max_favorable_excursion_r",
    "maximum_favorable_excursion_r",
    "max_favorable_r",
    "mfe_R",
]

MAE_R_ALIASES = [
    "mae_r",
    "final_mae_r",
    "max_adverse_excursion_r",
    "maximum_adverse_excursion_r",
    "max_adverse_r",
    "mae_R",
]

TRADE_ID_ALIASES = [
    "trade_id",
    "Trade_ID",
    "id",
]

SYMBOL_ALIASES = [
    "symbol",
    "Symbol",
    "_symbol",
]

ENTRY_DATE_ALIASES = [
    "entry_date",
    "Entry_Date",
    "date",
]

DIRECTION_ALIASES = [
    "direction",
    "Direction",
]


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def find_existing_file(
    candidates: List[Path],
) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the validated Bar-1→2 trajectory file.\n"
        "Checked:\n"
        + "\n".join(
            f"  - {path}"
            for path in candidates
        )
    )


def find_column(
    df: pd.DataFrame,
    aliases: List[str],
) -> Optional[str]:
    """
    Resolve a column using exact names first and then
    case-insensitive matching.
    """

    for alias in aliases:
        if alias in df.columns:
            return alias

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for alias in aliases:
        key = alias.strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


def normalize_key(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def median_split(
    series: pd.Series,
):
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    median = float(
        numeric.dropna().median()
    )

    state = (
        numeric >= median
    ).astype(int)

    return median, state


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def rank_biserial_from_u(
    u_stat: float,
    n1: int,
    n0: int,
):
    """
    Rank-biserial correlation:

        r_rb = 2U/(n1*n0) - 1

    Here U corresponds to the first group.
    """
    if n1 <= 0 or n0 <= 0:
        return np.nan

    return safe_float(
        (2.0 * u_stat / (n1 * n0)) - 1.0
    )


def bootstrap_median_difference(
    group1: np.ndarray,
    group0: np.ndarray,
    seed: int = 20260916,
    iterations: int = 5000,
):
    """
    Bootstrap 95% CI for:

        median(group1) - median(group0)

    Used only as an uncertainty estimate.
    """
    group1 = np.asarray(group1, dtype=float)
    group0 = np.asarray(group0, dtype=float)

    group1 = group1[np.isfinite(group1)]
    group0 = group0[np.isfinite(group0)]

    if len(group1) < 2 or len(group0) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples1 = rng.choice(
        group1,
        size=(iterations, len(group1)),
        replace=True,
    )

    samples0 = rng.choice(
        group0,
        size=(iterations, len(group0)),
        replace=True,
    )

    differences = (
        np.median(samples1, axis=1)
        - np.median(samples0, axis=1)
    )

    low, high = np.percentile(
        differences,
        [2.5, 97.5],
    )

    return (
        safe_float(low),
        safe_float(high),
    )


def outcome_statistics(
    frame: pd.DataFrame,
    state_column: str,
    outcome_column: str,
):
    """
    Compare outcome distribution between:

        state = 1
        state = 0

    Returns:
        sample sizes
        means
        medians
        median difference
        Mann-Whitney U
        p-value
        rank-biserial effect
        bootstrap median CI
    """

    data = frame[
        [state_column, outcome_column]
    ].copy()

    data[outcome_column] = pd.to_numeric(
        data[outcome_column],
        errors="coerce",
    )

    data = data.dropna(
        subset=[outcome_column]
    )

    group1 = data.loc[
        data[state_column] == 1,
        outcome_column,
    ].to_numpy(dtype=float)

    group0 = data.loc[
        data[state_column] == 0,
        outcome_column,
    ].to_numpy(dtype=float)

    n1 = len(group1)
    n0 = len(group0)

    if n1 == 0 or n0 == 0:
        return {
            "n_state1": n1,
            "n_state0": n0,
            "mean_state1": np.nan,
            "mean_state0": np.nan,
            "median_state1": np.nan,
            "median_state0": np.nan,
            "median_difference_state1_minus_state0": np.nan,
            "mw_u": np.nan,
            "mw_p": np.nan,
            "rank_biserial": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }

    mean1 = float(np.mean(group1))
    mean0 = float(np.mean(group0))

    median1 = float(np.median(group1))
    median0 = float(np.median(group0))

    median_difference = (
        median1 - median0
    )

    try:
        u_stat, p_value = mannwhitneyu(
            group1,
            group0,
            alternative="two-sided",
        )
    except Exception:
        u_stat = np.nan
        p_value = np.nan

    rank_biserial = (
        rank_biserial_from_u(
            u_stat,
            n1,
            n0,
        )
        if np.isfinite(u_stat)
        else np.nan
    )

    ci_low, ci_high = (
        bootstrap_median_difference(
            group1,
            group0,
        )
    )

    return {
        "n_state1": int(n1),
        "n_state0": int(n0),
        "mean_state1": safe_float(mean1),
        "mean_state0": safe_float(mean0),
        "median_state1": safe_float(median1),
        "median_state0": safe_float(median0),
        "median_difference_state1_minus_state0": safe_float(
            median_difference
        ),
        "mw_u": safe_float(u_stat),
        "mw_p": safe_float(p_value),
        "rank_biserial": safe_float(
            rank_biserial
        ),
        "bootstrap_ci_low": safe_float(
            ci_low
        ),
        "bootstrap_ci_high": safe_float(
            ci_high
        ),
    }


def event_rate_test(
    frame: pd.DataFrame,
    state_column: str,
):
    """
    Secondary check against the frozen event label.

    This is deliberately not the primary outcome test.
    """

    state = frame[state_column].astype(int)
    event = frame["frozen_event"].astype(int)

    a = int(
        ((state == 1) & (event == 1)).sum()
    )
    b = int(
        ((state == 0) & (event == 1)).sum()
    )
    c = int(
        ((state == 1) & (event == 0)).sum()
    )
    d = int(
        ((state == 0) & (event == 0)).sum()
    )

    table = np.array(
        [
            [a, b],
            [c, d],
        ],
        dtype=int,
    )

    try:
        odds_ratio, p_value = fisher_exact(
            table,
            alternative="two-sided",
        )
    except Exception:
        odds_ratio = np.nan
        p_value = np.nan

    n1 = a + c
    n0 = b + d

    rate1 = (
        a / n1
        if n1 > 0
        else np.nan
    )

    rate0 = (
        b / n0
        if n0 > 0
        else np.nan
    )

    return {
        "event_rate_state1_pct": safe_float(
            rate1 * 100.0
        ),
        "event_rate_state0_pct": safe_float(
            rate0 * 100.0
        ),
        "event_rate_difference_pp": safe_float(
            (rate1 - rate0) * 100.0
            if np.isfinite(rate1)
            and np.isfinite(rate0)
            else np.nan
        ),
        "event_fisher_or": safe_float(
            odds_ratio
        ),
        "event_fisher_p": safe_float(
            p_value
        ),
    }


# ============================================================
# LOAD TRAJECTORY
# ============================================================

def load_trajectory():
    path = find_existing_file(
        TRAJECTORY_CANDIDATES
    )

    df = pd.read_csv(path)

    print(
        f"Trajectory input: {path}"
    )

    print(
        f"Trajectory rows: "
        f"{len(df)}, columns: {len(df.columns)}"
    )

    required = {
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",
        "bar1_net_excursion_r",
        "bar1_cumulative_mfe_r",
        "bar1_cumulative_mae_r",
        "bar2_net_deteriorated",
        "bar2_favorable_failed_to_improve",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing trajectory columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if len(df) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} trajectory rows, "
            f"found {len(df)}."
        )

    df = df.copy()

    df["frozen_event"] = bool_series(
        df["frozen_event"]
    ).astype(int)

    for column in [
        "bar1_net_excursion_r",
        "bar1_cumulative_mfe_r",
        "bar1_cumulative_mae_r",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    for column in [
        "bar2_net_deteriorated",
        "bar2_favorable_failed_to_improve",
    ]:
        df[column] = bool_series(
            df[column]
        ).astype(int)

    df["entry_date"] = pd.to_datetime(
        df["entry_date"],
        errors="coerce",
    )

    event_count = int(
        df["frozen_event"].sum()
    )

    retained_count = int(
        (df["frozen_event"] == 0).sum()
    )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events, "
            f"found {event_count}."
        )

    if retained_count != EXPECTED_RETAINED:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED} retained, "
            f"found {retained_count}."
        )

    return df, path


# ============================================================
# LOAD PRODUCTION TRADE OUTCOMES
# ============================================================

def load_trade_outcomes():
    if not TRADE_FILE.exists():
        raise FileNotFoundError(
            f"Production trade file not found: "
            f"{TRADE_FILE}"
        )

    trades = pd.read_csv(
        TRADE_FILE
    )

    print(
        f"Production trade rows: "
        f"{len(trades)}, columns: {len(trades.columns)}"
    )

    if len(trades) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} production trades, "
            f"found {len(trades)}."
        )

    final_r_column = find_column(
        trades,
        FINAL_R_ALIASES,
    )

    mfe_column = find_column(
        trades,
        MFE_R_ALIASES,
    )

    mae_column = find_column(
        trades,
        MAE_R_ALIASES,
    )


    symbol_column = find_column(
        trades,
        SYMBOL_ALIASES,
    )

    entry_date_column = find_column(
        trades,
        ENTRY_DATE_ALIASES,
    )

    direction_column = find_column(
        trades,
        DIRECTION_ALIASES,
    )

    print("\nResolved production outcome columns:")

    print(
        f"  Final R : "
        f"{final_r_column}"
    )

    print(
        f"  MFE R   : "
        f"{mfe_column}"
    )

    print(
        f"  MAE R   : "
        f"{mae_column}"
    )

    print(
        f"  Symbol  : "
        f"{symbol_column}"
    )

    print(
        f"  Entry Date : "
        f"{entry_date_column}"
    )

    print(
        f"  Direction : "
        f"{direction_column}"
    )

    if final_r_column is None:
        raise RuntimeError(
            "\nCould not identify the production FINAL R column.\n"
            "Available columns are:\n"
            + "\n".join(
                f"  - {column}"
                for column in trades.columns
            )
            + "\n\n"
            "The script intentionally stops instead of guessing."
        )

    if (
        symbol_column is None
        or entry_date_column is None
        or direction_column is None
    ):
        raise RuntimeError(
            "\nCould not identify all required production "
            "identity columns.\n"
            "Required: symbol + direction + entry_date.\n"
            "The script intentionally stops instead of guessing."
        )

    # --------------------------------------------------------
    # Build validated composite merge key
    # --------------------------------------------------------
    #
    # Production trades.csv has no trade_id.
    # Therefore we use the strongest common identifying
    # fields available in BOTH datasets:
    #
    #     symbol + direction + entry_date
    #
    # Direction is included deliberately so that the key is
    # not based only on symbol/date.
    # --------------------------------------------------------

    normalized_symbol = (
        trades[symbol_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    normalized_direction = (
        trades[direction_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    normalized_entry_date = (
        pd.to_datetime(
            trades[entry_date_column],
            errors="coerce",
        )
        .dt.strftime("%Y-%m-%d")
    )

    invalid_identity = (
        normalized_symbol.isin(["", "NAN", "NONE"])
        | normalized_direction.isin(["", "NAN", "NONE"])
        | normalized_entry_date.isna()
    )

    if int(invalid_identity.sum()) > 0:
        raise RuntimeError(
            "Production outcome identity contains "
            f"{int(invalid_identity.sum())} invalid rows. "
            "Refusing ambiguous merge."
        )

    trades["_merge_key"] = (
        normalized_symbol
        + "|"
        + normalized_direction
        + "|"
        + normalized_entry_date
    )

    merge_type = "symbol|direction|entry_date"

    duplicate_keys = int(
        trades["_merge_key"].duplicated().sum()
    )

    if duplicate_keys > 0:
        duplicate_values = (
            trades.loc[
                trades["_merge_key"].duplicated(
                    keep=False
                ),
                "_merge_key",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Production outcome merge key has "
            f"{duplicate_keys} duplicate rows. "
            "Refusing ambiguous merge.\n"
            "Duplicate keys:\n"
            + "\n".join(
                f"  - {value}"
                for value in duplicate_values
            )
        )

    print(
        f"  Merge key : {merge_type}"
    )

    print(
        f"  Unique production keys : "
        f"{trades['_merge_key'].nunique()}"
    )
    outcome_columns = [
        "_merge_key",
        final_r_column,
    ]

    if mfe_column is not None:
        outcome_columns.append(
            mfe_column
        )

    if mae_column is not None:
        outcome_columns.append(
            mae_column
        )

    outcome = trades[
        outcome_columns
    ].copy()

    rename_map = {
        final_r_column: "outcome_final_r",
    }

    if mfe_column is not None:
        rename_map[mfe_column] = (
            "outcome_mfe_r"
        )

    if mae_column is not None:
        rename_map[mae_column] = (
            "outcome_mae_r"
        )

    outcome = outcome.rename(
        columns=rename_map
    )

    outcome["outcome_final_r"] = pd.to_numeric(
        outcome["outcome_final_r"],
        errors="coerce",
    )

    if "outcome_mfe_r" in outcome.columns:
        outcome["outcome_mfe_r"] = pd.to_numeric(
            outcome["outcome_mfe_r"],
            errors="coerce",
        )

    if "outcome_mae_r" in outcome.columns:
        outcome["outcome_mae_r"] = pd.to_numeric(
            outcome["outcome_mae_r"],
            errors="coerce",
        )

    duplicate_keys = int(
        outcome["_merge_key"].duplicated().sum()
    )

    if duplicate_keys > 0:
        raise RuntimeError(
            f"Production outcome merge key has "
            f"{duplicate_keys} duplicate rows. "
            f"Refusing ambiguous merge."
        )

    return (
        outcome,
        merge_type,
        final_r_column,
        mfe_column,
        mae_column,
    )


# ============================================================
# MERGE TRAJECTORY + OUTCOMES
# ============================================================

def merge_data(
    trajectory: pd.DataFrame,
    outcomes: pd.DataFrame,
    trade_merge_type: str,
):
    df = trajectory.copy()

    # --------------------------------------------------------
    # Construct the same merge key used for production data.
    # --------------------------------------------------------

    if trade_merge_type == "trade_id":

        trajectory_trade_id = find_column(
            df,
            TRADE_ID_ALIASES,
        )

        if trajectory_trade_id is None:
            raise RuntimeError(
                "Production file uses trade_id for merging, "
                "but trajectory file has no trade_id."
            )

        df["_merge_key"] = (
            df[trajectory_trade_id]
            .astype(str)
            .str.strip()
        )

    else:

        normalized_symbol = (
            df["symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        normalized_direction = (
            df["direction"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        normalized_entry_date = (
            pd.to_datetime(
                df["entry_date"],
                errors="coerce",
            )
            .dt.strftime("%Y-%m-%d")
        )

        invalid_identity = (
            normalized_symbol.isin(
                ["", "NAN", "NONE"]
            )
            | normalized_direction.isin(
                ["", "NAN", "NONE"]
            )
            | normalized_entry_date.isna()
        )

        if int(invalid_identity.sum()) > 0:
            raise RuntimeError(
                "Trajectory outcome identity contains "
                f"{int(invalid_identity.sum())} invalid rows. "
                "Refusing ambiguous merge."
            )

        df["_merge_key"] = (
            normalized_symbol
            + "|"
            + normalized_direction
            + "|"
            + normalized_entry_date
        )

        print(
            "  Trajectory merge key: "
            "symbol|direction|entry_date"
        )

    duplicate_keys = int(
        df["_merge_key"].duplicated().sum()
    )

    if duplicate_keys > 0:
        raise RuntimeError(
            f"Trajectory merge key has "
            f"{duplicate_keys} duplicates."
        )

    outcome_duplicate_keys = int(
        outcomes["_merge_key"].duplicated().sum()
    )

    if outcome_duplicate_keys > 0:
        raise RuntimeError(
            "Production outcome merge key has "
            f"{outcome_duplicate_keys} duplicates."
        )

    merged = df.merge(
        outcomes,
        on="_merge_key",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != EXPECTED_TRADES:
        raise RuntimeError(
            "Merge changed the expected trade count."
        )

    final_missing = int(
        merged["outcome_final_r"]
        .isna()
        .sum()
    )

    if final_missing > 0:
        raise RuntimeError(
            f"{final_missing} trades have no matched "
            f"final-R outcome."
        )

    return merged


# ============================================================
# BUILD FIXED BAR-1 STATES
# ============================================================

def add_global_states(
    df: pd.DataFrame,
):
    df = df.copy()

    medians = {}

    for family_name, config in FAMILIES.items():

        feature = config[
            "bar1_feature"
        ]

        median, state = median_split(
            df[feature]
        )

        state_column = (
            f"{family_name}_bar1_state"
        )

        df[state_column] = state

        medians[feature] = median

    print("\nGlobal Bar-1 medians:")

    for feature, median in medians.items():
        print(
            f"  {feature}: "
            f"{median:.6f}"
        )

    return df, medians


# ============================================================
# FAMILY ANALYSIS
# ============================================================

def analyze_family(
    df: pd.DataFrame,
    family_name: str,
    config: Dict,
):
    state_column = config[
        "bar2_state"
    ]

    results = []

    outcome_columns = [
        "outcome_final_r",
    ]

    if "outcome_mfe_r" in df.columns:
        outcome_columns.append(
            "outcome_mfe_r"
        )

    if "outcome_mae_r" in df.columns:
        outcome_columns.append(
            "outcome_mae_r"
        )

    # --------------------------------------------------------
    # Primary outcome metrics
    # --------------------------------------------------------

    for outcome_column in outcome_columns:

        stats = outcome_statistics(
            df,
            state_column,
            outcome_column,
        )

        event_stats = event_rate_test(
            df,
            state_column,
        )

        results.append(
            {
                "family": family_name,
                "state_name": config[
                    "state_name"
                ],
                "outcome": outcome_column,
                **stats,
                **event_stats,
            }
        )

    return results


# ============================================================
# PER-TRADE RESEARCH TABLE
# ============================================================

def build_per_trade(
    df: pd.DataFrame,
):
    columns = [
        column
        for column in [
            "trade_id",
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            "bar1_net_excursion_r",
            "bar1_cumulative_mfe_r",
            "bar1_cumulative_mae_r",
            "bar2_net_deteriorated",
            "bar2_favorable_failed_to_improve",
            "A_net_path_bar1_state",
            "B_favorable_bar1_state",
            "C_adverse_bar1_state",
            "outcome_final_r",
            "outcome_mfe_r",
            "outcome_mae_r",
        ]
        if column in df.columns
    ]

    return df[columns].copy()


# ============================================================
# INTEGRITY
# ============================================================

def build_integrity(
    df: pd.DataFrame,
    primary_df: pd.DataFrame,
):
    checks = {}

    checks["trade_count"] = (
        len(df) == EXPECTED_TRADES
    )

    checks["event_count"] = (
        int(df["frozen_event"].sum())
        == EXPECTED_EVENTS
    )

    checks["retained_count"] = (
        int(
            (df["frozen_event"] == 0).sum()
        )
        == EXPECTED_RETAINED
    )

    checks["final_r_present"] = (
        "outcome_final_r" in df.columns
        and df["outcome_final_r"].notna().all()
    )

    checks["primary_rows_positive"] = (
        len(primary_df) >= len(FAMILIES)
    )

    checks["family_coverage"] = (
        set(primary_df["family"])
        == set(FAMILIES.keys())
    )

    checks["no_duplicate_trajectory_rows"] = (
        not df.duplicated(
            subset=[
                "symbol",
                "entry_date",
                "direction",
            ]
        ).any()
    )

    checks["bar2_states_binary"] = True

    for family_name, config in FAMILIES.items():

        column = config[
            "bar2_state"
        ]

        values = set(
            df[column]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not values.issubset({0, 1}):
            checks[
                "bar2_states_binary"
            ] = False

    checks["PASS"] = all(
        bool(value)
        for key, value in checks.items()
        if key != "PASS"
    )

    return checks


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "TRADE SENSE-AI — "
        "BAR 1→2 OUTCOME SEPARATION FORENSIC ANALYSIS"
    )

    print("=" * 72)

    ensure_output_dir()

    # --------------------------------------------------------
    # 1. Load validated trajectory
    # --------------------------------------------------------

    trajectory, trajectory_path = (
        load_trajectory()
    )

    print(
        f"Frozen population: "
        f"{len(trajectory)} / "
        f"{int(trajectory['frozen_event'].sum())} / "
        f"{int((trajectory['frozen_event'] == 0).sum())}"
    )

    # --------------------------------------------------------
    # 2. Load production outcomes
    # --------------------------------------------------------

    (
        outcomes,
        merge_type,
        final_r_column,
        mfe_column,
        mae_column,
    ) = load_trade_outcomes()

    # --------------------------------------------------------
    # 3. Merge
    # --------------------------------------------------------

    df = merge_data(
        trajectory,
        outcomes,
        merge_type,
    )

    print(
        f"\nOutcome merge: PASS "
        f"({merge_type})"
    )

    # --------------------------------------------------------
    # 4. Fixed global Bar-1 states
    # --------------------------------------------------------

    df, global_medians = (
        add_global_states(df)
    )

    # --------------------------------------------------------
    # 5. Family analysis
    # --------------------------------------------------------

    primary_rows = []

    for family_name, config in FAMILIES.items():

        family_rows = analyze_family(
            df,
            family_name,
            config,
        )

        primary_rows.extend(
            family_rows
        )

    primary_df = pd.DataFrame(
        primary_rows
    )

    # --------------------------------------------------------
    # 6. Per-trade output
    # --------------------------------------------------------

    per_trade_df = build_per_trade(
        df
    )

    per_trade_path = (
        OUTPUT_DIR
        / "bar12_outcome_separation_per_trade.csv"
    )

    per_trade_df.to_csv(
        per_trade_path,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Primary output
    # --------------------------------------------------------

    primary_path = (
        OUTPUT_DIR
        / "bar12_outcome_separation_primary.csv"
    )

    primary_df.to_csv(
        primary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Family summary
    # --------------------------------------------------------

    family_summary_rows = []

    for family_name in FAMILIES:

        subset = primary_df[
            primary_df["family"]
            == family_name
        ].copy()

        final_row = subset[
            subset["outcome"]
            == "outcome_final_r"
        ]

        if final_row.empty:
            family_summary_rows.append(
                {
                    "family": family_name,
                    "final_r_rows": 0,
                    "final_r_median_difference": np.nan,
                    "final_r_mw_p": np.nan,
                    "final_r_ci_low": np.nan,
                    "final_r_ci_high": np.nan,
                    "directional_economic_separation": False,
                }
            )
            continue

        row = final_row.iloc[0]

        median_difference = safe_float(
            row[
                "median_difference_state1_minus_state0"
            ]
        )

        mw_p = safe_float(
            row["mw_p"]
        )

        ci_low = safe_float(
            row["bootstrap_ci_low"]
        )

        ci_high = safe_float(
            row["bootstrap_ci_high"]
        )

        # State=1 is interpreted as the adverse Bar-2 state.
        # Therefore, economically worse final R means a
        # negative median difference.
        directional_separation = (
            np.isfinite(median_difference)
            and median_difference < 0
        )

        family_summary_rows.append(
            {
                "family": family_name,
                "final_r_rows": 1,
                "final_r_median_difference": (
                    median_difference
                ),
                "final_r_mw_p": mw_p,
                "final_r_ci_low": ci_low,
                "final_r_ci_high": ci_high,
                "directional_economic_separation": bool(
                    directional_separation
                ),
            }
        )

    family_summary_df = pd.DataFrame(
        family_summary_rows
    )

    family_summary_path = (
        OUTPUT_DIR
        / "bar12_outcome_separation_family_summary.csv"
    )

    family_summary_df.to_csv(
        family_summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # 9. Integrity
    # --------------------------------------------------------

    integrity = build_integrity(
        df,
        primary_df,
    )

    integrity_df = pd.DataFrame(
        [
            {
                "check": key,
                "passed": value,
            }
            for key, value in integrity.items()
        ]
    )

    integrity_path = (
        OUTPUT_DIR
        / "bar12_outcome_separation_integrity.csv"
    )

    integrity_df.to_csv(
        integrity_path,
        index=False,
    )

    # --------------------------------------------------------
    # 10. Summary JSON
    # --------------------------------------------------------

    final_r_rows = primary_df[
        primary_df["outcome"]
        == "outcome_final_r"
    ].copy()

    directional_family_count = int(
        family_summary_df[
            "directional_economic_separation"
        ].sum()
    )

    summary = {
        "trajectory_file": str(
            trajectory_path
        ),
        "trade_file": str(
            TRADE_FILE
        ),
        "frozen_population": {
            "trades": EXPECTED_TRADES,
            "events": EXPECTED_EVENTS,
            "retained": EXPECTED_RETAINED,
        },
        "merge_type": merge_type,
        "resolved_outcome_columns": {
            "final_r": final_r_column,
            "mfe_r": mfe_column,
            "mae_r": mae_column,
        },
        "global_bar1_medians": {
            key: safe_float(value)
            for key, value
            in global_medians.items()
        },
        "families": list(
            FAMILIES.keys()
        ),
        "final_r_directional_separation_count": (
            directional_family_count
        ),
        "final_r_family_count": len(
            FAMILIES
        ),
        "integrity_pass": bool(
            integrity["PASS"]
        ),
        "interpretation": (
            "Outcome separation tests whether Bar-2 "
            "states are associated with eventual trade "
            "outcomes. A negative final-R difference for "
            "the adverse Bar-2 state is directionally "
            "consistent with worse later economic outcome. "
            "This remains association and does not "
            "establish causality, independent OOS "
            "predictive power, economic value after "
            "implementation costs, or a deployable rule."
        ),
        "production_modified": False,
        "oos_modified": False,
    }

    summary_path = (
        OUTPUT_DIR
        / "bar12_outcome_separation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # 11. Console report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("OUTCOME SEPARATION RESULTS")
    print("=" * 72)

    for _, row in primary_df.iterrows():

        print(
            f"\n{row['family']} | "
            f"{row['outcome']}"
        )

        print(
            f"  State 1 n: "
            f"{int(row['n_state1'])}"
        )

        print(
            f"  State 0 n: "
            f"{int(row['n_state0'])}"
        )

        print(
            f"  State 1 median: "
            f"{row['median_state1']:.6f}"
            if pd.notna(
                row["median_state1"]
            )
            else "  State 1 median: NaN"
        )

        print(
            f"  State 0 median: "
            f"{row['median_state0']:.6f}"
            if pd.notna(
                row["median_state0"]
            )
            else "  State 0 median: NaN"
        )

        print(
            f"  Median difference: "
            f"{row['median_difference_state1_minus_state0']:.6f}"
            if pd.notna(
                row[
                    "median_difference_state1_minus_state0"
                ]
            )
            else "  Median difference: NaN"
        )

        print(
            f"  Bootstrap 95% CI: "
            f"[{row['bootstrap_ci_low']:.6f}, "
            f"{row['bootstrap_ci_high']:.6f}]"
            if pd.notna(
                row["bootstrap_ci_low"]
            )
            and pd.notna(
                row["bootstrap_ci_high"]
            )
            else "  Bootstrap 95% CI: NaN"
        )

        print(
            f"  Mann-Whitney p: "
            f"{row['mw_p']:.6f}"
            if pd.notna(
                row["mw_p"]
            )
            else "  Mann-Whitney p: NaN"
        )

        print(
            f"  Rank-biserial: "
            f"{row['rank_biserial']:.6f}"
            if pd.notna(
                row["rank_biserial"]
            )
            else "  Rank-biserial: NaN"
        )

    print("\n" + "=" * 72)
    print("FINAL-R FAMILY SUMMARY")
    print("=" * 72)

    for _, row in family_summary_df.iterrows():

        print(
            f"\n{row['family']}"
        )

        print(
            f"  Median final-R difference "
            f"(Bar2 state1 - state0): "
            f"{row['final_r_median_difference']:.6f}"
            if pd.notna(
                row[
                    "final_r_median_difference"
                ]
            )
            else
            "  Median final-R difference: NaN"
        )

        print(
            f"  Bootstrap 95% CI: "
            f"[{row['final_r_ci_low']:.6f}, "
            f"{row['final_r_ci_high']:.6f}]"
            if pd.notna(
                row["final_r_ci_low"]
            )
            and pd.notna(
                row["final_r_ci_high"]
            )
            else "  Bootstrap 95% CI: NaN"
        )

        print(
            f"  Mann-Whitney p: "
            f"{row['final_r_mw_p']:.6f}"
            if pd.notna(
                row["final_r_mw_p"]
            )
            else "  Mann-Whitney p: NaN"
        )

        print(
            f"  Directionally worse final-R outcome: "
            f"{bool(row['directional_economic_separation'])}"
        )

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"Final-R directional separation: "
        f"{directional_family_count}/{len(FAMILIES)} families"
    )

    print(
        f"Integrity PASS: "
        f"{bool(integrity['PASS'])}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis is an outcome-separation forensic test."
        "\nIt does NOT establish causality, "
        "independent OOS predictive power, "
        "economic value after trading costs, "
        "or a deployable trading rule."
        "\nProduction strategy logic and OOS data remain untouched."
    )

    print("\nOutputs:")

    print(
        per_trade_path
    )

    print(
        primary_path
    )

    print(
        family_summary_path
    )

    print(
        integrity_path
    )

    print(
        summary_path
    )

    if not integrity["PASS"]:
        raise RuntimeError(
            "INTEGRITY CHECK FAILED."
        )


if __name__ == "__main__":
    main()