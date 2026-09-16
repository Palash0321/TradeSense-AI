"""
TradeSense-AI — Bar 1→2 Conditional Robustness Forensic Analysis

PURPOSE
-------
Stress-test the already validated conditional Bar-1 → Bar-2 relationships
across:

1. Calendar year
2. Symbol
3. Setup
4. Direction

This is RESEARCH ONLY.

FROZEN POPULATION
-----------------
106 total trades
60 frozen early-adverse events
46 retained

FROZEN EVENT LABEL
------------------
The existing frozen 5-bar / 0.25R event label from the validated
Bar 1→2 trajectory dataset is used unchanged.

CONDITIONAL FAMILIES
--------------------
A. Bar-1 net path state
   Bar-1 net_excursion_r median split
   -> Bar-2 net deterioration

B. Bar-1 favorable excursion state
   Bar-1 cumulative_mfe_r median split
   -> Bar-2 favorable failure

C. Bar-1 adverse excursion state
   Bar-1 cumulative_mae_r median split
   -> Bar-2 net deterioration

METHODOLOGICAL RULES
--------------------
- Bar-1 median split is GLOBAL and fixed.
- No subgroup-specific threshold optimization.
- No machine learning.
- No threshold hunting.
- No production strategy changes.
- No OOS changes.
- Subgroup significance is descriptive because some groups are small.
- Primary robustness quantity is direction consistency.
- LOO analysis removes one subgroup at a time.
- Manual CMH implementation is used so statsmodels is not required.

PASS CRITERIA
-------------
For every family × robustness dimension:

- At least 2 valid subgroups
- At least 75% subgroup direction consistency
- At least 75% LOO direction consistency

Family-level robustness PASS requires all four dimensions:

- year
- symbol
- setup
- direction

to PASS.

OUTPUTS
-------
tests/output/research/early_adverse_bar12_conditional_robustness/
    bar12_conditional_robustness_subgroups.csv
    bar12_conditional_robustness_loo.csv
    bar12_conditional_robustness_stability.csv
    bar12_conditional_robustness_dimension_results.csv
    bar12_conditional_robustness_family_summary.csv
    bar12_conditional_robustness_integrity.csv
    bar12_conditional_robustness_summary.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_conditional_robustness"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

MIN_SUBGROUP_TRADES = 8
MIN_VALID_SUBGROUPS = 2
MIN_DIRECTION_CONSISTENCY = 0.75

ALPHA = 0.05

FAMILIES = {
    "A_net_path": {
        "bar1_feature": "bar1_net_excursion_r",
        "bar2_state": "bar2_net_deteriorated",
        "state_name": "Bar-2 net deterioration",
        "dimension": "net_path",
    },
    "B_favorable": {
        "bar1_feature": "bar1_cumulative_mfe_r",
        "bar2_state": "bar2_favorable_failed_to_improve",
        "state_name": "Bar-2 favorable failure",
        "dimension": "favorable",
    },
    "C_adverse": {
        "bar1_feature": "bar1_cumulative_mae_r",
        "bar2_state": "bar2_net_deteriorated",
        "state_name": "Bar-2 net deterioration",
        "dimension": "adverse",
    },
}

DIMENSIONS = {
    "year": "calendar_year",
    "symbol": "symbol",
    "setup": "setup",
    "direction": "direction",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    if value is None:
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def bool_series(series: pd.Series) -> pd.Series:
    """
    Robust conversion of stored boolean-like columns.
    """
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

    lowered = series.astype(str).str.strip().str.lower()

    return lowered.map(mapping).fillna(False).astype(bool)


def percentile_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.median())


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    """
    Wilson 95% confidence interval for a binomial proportion.
    """
    if total <= 0:
        return np.nan, np.nan

    p = successes / total

    denominator = 1.0 + (z ** 2) / total
    center = (
        p
        + (z ** 2) / (2.0 * total)
    ) / denominator

    half_width = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / total
                + (z ** 2) / (4.0 * total ** 2)
            )
        )
        / denominator
    )

    return center - half_width, center + half_width


def risk_difference(
    event_state: pd.Series,
    state_mask: pd.Series,
    event_label: pd.Series,
):
    """
    Event-rate difference:

        P(event | state=1) - P(event | state=0)

    Returns percentage-point difference.
    """
    state1 = state_mask == 1
    state0 = state_mask == 0

    n1 = int(state1.sum())
    n0 = int(state0.sum())

    e1 = int(event_label[state1].sum())
    e0 = int(event_label[state0].sum())

    if n1 == 0 or n0 == 0:
        return np.nan, n1, n0, e1, e0

    rate1 = e1 / n1
    rate0 = e0 / n0

    return (rate1 - rate0) * 100.0, n1, n0, e1, e0


def fisher_from_binary(
    event_label: pd.Series,
    state_mask: pd.Series,
):
    """
    Fisher exact test for:

             state=1   state=0

    event        a         b
    no event     c         d
    """
    state1 = state_mask == 1
    state0 = state_mask == 0

    a = int((event_label[state1] == 1).sum())
    b = int((event_label[state0] == 1).sum())
    c = int((event_label[state1] == 0).sum())
    d = int((event_label[state0] == 0).sum())

    table = np.array(
        [
            [a, b],
            [c, d],
        ],
        dtype=int,
    )

    try:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    except Exception:
        odds_ratio = np.nan
        p_value = np.nan

    return (
        safe_float(odds_ratio),
        safe_float(p_value),
        a,
        b,
        c,
        d,
    )


# ============================================================
# MANUAL CMH
# ============================================================

def cmh_manual(
    strata_tables: List[Tuple[int, int, int, int]]
):
    """
    Manual Cochran-Mantel-Haenszel calculation.

    Each tuple is:

        (a, b, c, d)

    where:

        a = event & state=1
        b = event & state=0
        c = no-event & state=1
        d = no-event & state=0

    MH pooled odds ratio:

        sum(a*d/n) / sum(b*c/n)

    CMH chi-square uses the standard continuity correction:

        E[a] = ((a+b)*(a+c))/n

        Var[a] =
            ((a+b)*(c+d)*(a+c)*(b+d))
            /
            (n^2*(n-1))

        X² =
            (|sum(a-E[a])|-0.5)^2
            /
            sum(Var[a])

    Returns:
        pooled_or,
        chi_square,
        p_value,
        valid_strata
    """
    valid = []

    for a, b, c, d in strata_tables:
        n = a + b + c + d

        if n <= 1:
            continue

        if (a + b) == 0:
            continue

        if (c + d) == 0:
            continue

        if (a + c) == 0:
            continue

        if (b + d) == 0:
            continue

        valid.append((a, b, c, d))

    if not valid:
        return np.nan, np.nan, np.nan, 0

    numerator = 0.0
    denominator = 0.0

    observed_minus_expected = 0.0
    variance_sum = 0.0

    for a, b, c, d in valid:
        n = a + b + c + d

        numerator += (a * d) / n
        denominator += (b * c) / n

        expected_a = (
            ((a + b) * (a + c))
            / n
        )

        variance_a = (
            (a + b)
            * (c + d)
            * (a + c)
            * (b + d)
        ) / (
            (n ** 2) * (n - 1)
        )

        observed_minus_expected += a - expected_a
        variance_sum += variance_a

    if denominator == 0:
        pooled_or = np.inf if numerator > 0 else np.nan
    else:
        pooled_or = numerator / denominator

    if variance_sum <= 0:
        return (
            safe_float(pooled_or),
            np.nan,
            np.nan,
            len(valid),
        )

    chi_square = (
        (abs(observed_minus_expected) - 0.5) ** 2
    ) / variance_sum

    if chi_square < 0:
        chi_square = 0.0

    p_value = chi2.sf(chi_square, 1)

    return (
        safe_float(pooled_or),
        safe_float(chi_square),
        safe_float(p_value),
        len(valid),
    )


# ============================================================
# CMH STRATA CONSTRUCTION
# ============================================================

def build_2x2(
    frame: pd.DataFrame,
    state_column: str,
    event_column: str,
):
    state = frame[state_column].astype(int)
    event = frame[event_column].astype(int)

    a = int(((state == 1) & (event == 1)).sum())
    b = int(((state == 0) & (event == 1)).sum())
    c = int(((state == 1) & (event == 0)).sum())
    d = int(((state == 0) & (event == 0)).sum())

    return a, b, c, d


def conditional_cmh(
    frame: pd.DataFrame,
    bar1_state_column: str,
    bar2_state_column: str,
    event_column: str,
):
    """
    CMH across Bar-1 LOW/HIGH strata.

    The Bar-1 state is the stratification variable.

    Within each Bar-1 stratum:

        Bar-2 state=1 vs state=0

    predicts the frozen event label.
    """
    strata_tables = []

    for stratum_value in [0, 1]:
        subset = frame[
            frame[bar1_state_column] == stratum_value
        ].copy()

        if subset.empty:
            continue

        table = build_2x2(
            subset,
            bar2_state_column,
            event_column,
        )

        strata_tables.append(table)

    return cmh_manual(strata_tables)


# ============================================================
# DATA PREPARATION
# ============================================================

def validate_input_columns(df: pd.DataFrame) -> None:
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

    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def prepare_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input rows: {len(df)}, columns: {len(df.columns)}")

    validate_input_columns(df)

    if len(df) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} trade rows, "
            f"found {len(df)}."
        )

    df = df.copy()

    # --------------------------------------------------------
    # Frozen event label
    # --------------------------------------------------------

    df["frozen_event"] = bool_series(
        df["frozen_event"]
    ).astype(int)

    # --------------------------------------------------------
    # Numeric Bar-1 features
    # --------------------------------------------------------

    numeric_columns = [
        "bar1_net_excursion_r",
        "bar1_cumulative_mfe_r",
        "bar1_cumulative_mae_r",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Bar-2 state columns
    # --------------------------------------------------------

    state_columns = [
        "bar2_net_deteriorated",
        "bar2_favorable_failed_to_improve",
    ]

    for column in state_columns:
        df[column] = bool_series(
            df[column]
        ).astype(int)

    # --------------------------------------------------------
    # Calendar year
    # --------------------------------------------------------

    df["entry_date"] = pd.to_datetime(
        df["entry_date"],
        errors="coerce",
    )

    if df["entry_date"].isna().any():
        raise RuntimeError(
            "Invalid entry_date values found."
        )

    df["calendar_year"] = (
        df["entry_date"]
        .dt.year
        .astype(int)
    )

    # --------------------------------------------------------
    # Fixed GLOBAL median splits
    # --------------------------------------------------------

    global_medians = {}

    for family_name, config in FAMILIES.items():
        feature = config["bar1_feature"]

        median_value = percentile_median(
            df[feature]
        )

        if not np.isfinite(median_value):
            raise RuntimeError(
                f"Could not calculate median for {feature}."
            )

        global_medians[feature] = median_value

        state_column = (
            f"{family_name}_bar1_state"
        )

        df[state_column] = (
            df[feature] >= median_value
        ).astype(int)

    # --------------------------------------------------------
    # Basic frozen population check
    # --------------------------------------------------------

    event_count = int(
        df["frozen_event"].sum()
    )

    retained_count = int(
        (df["frozen_event"] == 0).sum()
    )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} frozen events, "
            f"found {event_count}."
        )

    if retained_count != EXPECTED_RETAINED:
        raise RuntimeError(
            f"Expected {EXPECTED_RETAINED} retained trades, "
            f"found {retained_count}."
        )

    print(
        f"Frozen population: "
        f"{len(df)} / {event_count} / {retained_count}"
    )

    print("\nGlobal Bar-1 median splits:")

    for feature, median_value in global_medians.items():
        print(
            f"  {feature}: {median_value:.6f}"
        )

    return df


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================

def analyze_subgroup(
    frame: pd.DataFrame,
    family_name: str,
    dimension: str,
    subgroup: str,
    family_config: Dict,
):
    """
    Analyze one robustness subgroup.

    The subgroup itself is NOT used to define the Bar-1 median.
    The state threshold was already defined globally.
    """
    bar1_state_column = (
        f"{family_name}_bar1_state"
    )

    bar2_state_column = (
        family_config["bar2_state"]
    )

    event_column = "frozen_event"

    n = len(frame)

    if n < MIN_SUBGROUP_TRADES:
        return None

    # Require both Bar-1 strata to exist.
    strata_counts = (
        frame[bar1_state_column]
        .value_counts()
        .to_dict()
    )

    if 0 not in strata_counts or 1 not in strata_counts:
        return None

    # --------------------------------------------------------
    # Overall subgroup event-state association
    # --------------------------------------------------------

    rd, n1, n0, e1, e0 = risk_difference(
        frame[event_column],
        frame[bar2_state_column],
        frame[event_column],
    )

    odds_ratio, fisher_p, a, b, c, d = fisher_from_binary(
        frame[event_column],
        frame[bar2_state_column],
    )

    # --------------------------------------------------------
    # Conditional CMH within Bar-1 strata
    # --------------------------------------------------------

    pooled_or, chi_square, cmh_p, valid_strata = (
        conditional_cmh(
            frame,
            bar1_state_column,
            bar2_state_column,
            event_column,
        )
    )

    direction_positive = (
        np.isfinite(rd)
        and rd > 0
    )

    return {
        "family": family_name,
        "dimension": dimension,
        "subgroup": str(subgroup),
        "n": int(n),
        "events": int(frame[event_column].sum()),
        "retained": int((frame[event_column] == 0).sum()),
        "bar1_low_n": int(
            (frame[bar1_state_column] == 0).sum()
        ),
        "bar1_high_n": int(
            (frame[bar1_state_column] == 1).sum()
        ),
        "bar2_state_1_n": int(
            (frame[bar2_state_column] == 1).sum()
        ),
        "bar2_state_0_n": int(
            (frame[bar2_state_column] == 0).sum()
        ),
        "state1_events": int(e1),
        "state0_events": int(e0),
        "state1_event_rate_pct": (
            (e1 / n1) * 100.0
            if n1 > 0
            else np.nan
        ),
        "state0_event_rate_pct": (
            (e0 / n0) * 100.0
            if n0 > 0
            else np.nan
        ),
        "risk_difference_pp": safe_float(rd),
        "fisher_odds_ratio": safe_float(odds_ratio),
        "fisher_p": safe_float(fisher_p),
        "cmh_pooled_or": safe_float(pooled_or),
        "cmh_chi_square": safe_float(chi_square),
        "cmh_p_value": safe_float(cmh_p),
        "valid_cmh_strata": int(valid_strata),
        "direction_positive": bool(direction_positive),
    }


def run_subgroup_analysis(
    df: pd.DataFrame,
):
    rows = []

    for family_name, family_config in FAMILIES.items():
        for dimension, dimension_column in DIMENSIONS.items():

            grouped = df.groupby(
                dimension_column,
                dropna=False,
            )

            for subgroup, subgroup_frame in grouped:
                result = analyze_subgroup(
                    subgroup_frame,
                    family_name,
                    dimension,
                    subgroup,
                    family_config,
                )

                if result is not None:
                    rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# LOO CONDITIONAL ANALYSIS
# ============================================================

def analyze_loo(
    df: pd.DataFrame,
    family_name: str,
    dimension: str,
    dimension_column: str,
    family_config: Dict,
):
    """
    Leave-one-subgroup-out conditional CMH.

    The Bar-1 state threshold remains the GLOBAL median.

    Each LOO result removes exactly one subgroup and then
    recalculates the conditional CMH over the remaining
    Bar-1 LOW/HIGH strata.
    """
    bar1_state_column = (
        f"{family_name}_bar1_state"
    )

    bar2_state_column = (
        family_config["bar2_state"]
    )

    event_column = "frozen_event"

    subgroups = (
        df[dimension_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    results = []

    for excluded in sorted(subgroups):

        mask = (
            df[dimension_column]
            .astype(str)
            != excluded
        )

        subset = df.loc[mask].copy()

        if len(subset) < MIN_SUBGROUP_TRADES:
            continue

        pooled_or, chi_square, cmh_p, valid_strata = (
            conditional_cmh(
                subset,
                bar1_state_column,
                bar2_state_column,
                event_column,
            )
        )

        # Overall direction from Bar-2 state vs event.
        rd, n1, n0, e1, e0 = risk_difference(
            subset[event_column],
            subset[bar2_state_column],
            subset[event_column],
        )

        direction_positive = (
            np.isfinite(rd)
            and rd > 0
        )

        results.append(
            {
                "family": family_name,
                "dimension": dimension,
                "excluded_subgroup": excluded,
                "remaining_n": int(len(subset)),
                "remaining_events": int(
                    subset[event_column].sum()
                ),
                "remaining_retained": int(
                    (subset[event_column] == 0).sum()
                ),
                "bar2_state1_n": int(n1),
                "bar2_state0_n": int(n0),
                "risk_difference_pp": safe_float(rd),
                "cmh_pooled_or": safe_float(pooled_or),
                "cmh_chi_square": safe_float(chi_square),
                "cmh_p_value": safe_float(cmh_p),
                "valid_cmh_strata": int(valid_strata),
                "direction_positive": bool(
                    direction_positive
                ),
            }
        )

    return results


def run_loo_analysis(
    df: pd.DataFrame,
):
    rows = []

    for family_name, family_config in FAMILIES.items():
        for dimension, dimension_column in DIMENSIONS.items():

            rows.extend(
                analyze_loo(
                    df,
                    family_name,
                    dimension,
                    dimension_column,
                    family_config,
                )
            )

    return pd.DataFrame(rows)


# ============================================================
# STABILITY SUMMARY
# ============================================================

def calculate_stability(
    subgroup_df: pd.DataFrame,
    loo_df: pd.DataFrame,
):
    rows = []

    for family_name in FAMILIES:

        for dimension in DIMENSIONS:

            subgroup_subset = subgroup_df[
                (subgroup_df["family"] == family_name)
                & (subgroup_df["dimension"] == dimension)
            ].copy()

            loo_subset = loo_df[
                (loo_df["family"] == family_name)
                & (loo_df["dimension"] == dimension)
            ].copy()

            valid_subgroups = len(
                subgroup_subset
            )

            valid_loo = len(
                loo_subset
            )

            if valid_subgroups > 0:
                subgroup_positive_fraction = (
                    subgroup_subset[
                        "direction_positive"
                    ].mean()
                )
            else:
                subgroup_positive_fraction = np.nan

            if valid_loo > 0:
                loo_positive_fraction = (
                    loo_subset[
                        "direction_positive"
                    ].mean()
                )
            else:
                loo_positive_fraction = np.nan

            subgroup_direction_pass = (
                valid_subgroups >= MIN_VALID_SUBGROUPS
                and subgroup_positive_fraction
                >= MIN_DIRECTION_CONSISTENCY
            )

            loo_direction_pass = (
                valid_loo >= MIN_VALID_SUBGROUPS
                and loo_positive_fraction
                >= MIN_DIRECTION_CONSISTENCY
            )

            dimension_pass = (
                subgroup_direction_pass
                and loo_direction_pass
            )

            rd_values = pd.to_numeric(
                subgroup_subset[
                    "risk_difference_pp"
                ],
                errors="coerce",
            ).dropna()

            loo_rd_values = pd.to_numeric(
                loo_subset[
                    "risk_difference_pp"
                ],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "family": family_name,
                    "dimension": dimension,
                    "valid_subgroups": int(
                        valid_subgroups
                    ),
                    "subgroup_positive_fraction": safe_float(
                        subgroup_positive_fraction
                    ),
                    "subgroup_direction_pct": safe_float(
                        subgroup_positive_fraction * 100.0
                        if np.isfinite(
                            subgroup_positive_fraction
                        )
                        else np.nan
                    ),
                    "valid_loo_results": int(
                        valid_loo
                    ),
                    "loo_positive_fraction": safe_float(
                        loo_positive_fraction
                    ),
                    "loo_direction_pct": safe_float(
                        loo_positive_fraction * 100.0
                        if np.isfinite(
                            loo_positive_fraction
                        )
                        else np.nan
                    ),
                    "subgroup_rd_median_pp": safe_float(
                        rd_values.median()
                        if not rd_values.empty
                        else np.nan
                    ),
                    "subgroup_rd_min_pp": safe_float(
                        rd_values.min()
                        if not rd_values.empty
                        else np.nan
                    ),
                    "subgroup_rd_max_pp": safe_float(
                        rd_values.max()
                        if not rd_values.empty
                        else np.nan
                    ),
                    "loo_rd_median_pp": safe_float(
                        loo_rd_values.median()
                        if not loo_rd_values.empty
                        else np.nan
                    ),
                    "loo_rd_min_pp": safe_float(
                        loo_rd_values.min()
                        if not loo_rd_values.empty
                        else np.nan
                    ),
                    "loo_rd_max_pp": safe_float(
                        loo_rd_values.max()
                        if not loo_rd_values.empty
                        else np.nan
                    ),
                    "subgroup_direction_pass": bool(
                        subgroup_direction_pass
                    ),
                    "loo_direction_pass": bool(
                        loo_direction_pass
                    ),
                    "dimension_pass": bool(
                        dimension_pass
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# DIMENSION RESULTS
# ============================================================

def build_dimension_results(
    stability_df: pd.DataFrame,
):
    rows = []

    for family_name in FAMILIES:

        family_subset = stability_df[
            stability_df["family"] == family_name
        ]

        for dimension in DIMENSIONS:

            row = family_subset[
                family_subset["dimension"] == dimension
            ]

            if row.empty:
                rows.append(
                    {
                        "family": family_name,
                        "dimension": dimension,
                        "dimension_pass": False,
                        "reason": "MISSING_RESULT",
                    }
                )
                continue

            record = row.iloc[0]

            rows.append(
                {
                    "family": family_name,
                    "dimension": dimension,
                    "valid_subgroups": int(
                        record["valid_subgroups"]
                    ),
                    "subgroup_direction_pct": safe_float(
                        record[
                            "subgroup_direction_pct"
                        ]
                    ),
                    "valid_loo_results": int(
                        record["valid_loo_results"]
                    ),
                    "loo_direction_pct": safe_float(
                        record["loo_direction_pct"]
                    ),
                    "subgroup_direction_pass": bool(
                        record[
                            "subgroup_direction_pass"
                        ]
                    ),
                    "loo_direction_pass": bool(
                        record[
                            "loo_direction_pass"
                        ]
                    ),
                    "dimension_pass": bool(
                        record["dimension_pass"]
                    ),
                    "reason": (
                        "PASS"
                        if bool(record["dimension_pass"])
                        else "ROBUSTNESS_CRITERIA_NOT_MET"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# FAMILY SUMMARY
# ============================================================

def build_family_summary(
    dimension_results: pd.DataFrame,
):
    rows = []

    for family_name in FAMILIES:

        subset = dimension_results[
            dimension_results["family"] == family_name
        ].copy()

        dimensions_present = set(
            subset["dimension"]
        )

        required_dimensions = set(
            DIMENSIONS.keys()
        )

        all_dimensions_present = (
            dimensions_present
            == required_dimensions
        )

        dimension_pass_count = int(
            subset["dimension_pass"].sum()
        )

        family_pass = (
            all_dimensions_present
            and dimension_pass_count
            == len(DIMENSIONS)
        )

        rows.append(
            {
                "family": family_name,
                "dimension_count": int(
                    len(subset)
                ),
                "required_dimension_count": int(
                    len(DIMENSIONS)
                ),
                "dimension_pass_count": (
                    dimension_pass_count
                ),
                "all_dimensions_present": bool(
                    all_dimensions_present
                ),
                "family_robustness_pass": bool(
                    family_pass
                ),
                "reason": (
                    "PASS"
                    if family_pass
                    else "FAMILY_ROBUSTNESS_CRITERIA_NOT_MET"
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# INTEGRITY
# ============================================================

def build_integrity(
    df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    dimension_results: pd.DataFrame,
    family_summary: pd.DataFrame,
):
    expected_subgroup_min = (
        len(FAMILIES)
        * len(DIMENSIONS)
        * MIN_VALID_SUBGROUPS
    )

    expected_stability_rows = (
        len(FAMILIES)
        * len(DIMENSIONS)
    )

    expected_dimension_rows = (
        len(FAMILIES)
        * len(DIMENSIONS)
    )

    integrity = {
        "trade_count": (
            len(df) == EXPECTED_TRADES
        ),
        "event_count": (
            int(df["frozen_event"].sum())
            == EXPECTED_EVENTS
        ),
        "retained_count": (
            int(
                (df["frozen_event"] == 0).sum()
            )
            == EXPECTED_RETAINED
        ),
        "input_rows": (
            len(df) == EXPECTED_TRADES
        ),
        "subgroup_rows_positive": (
            len(subgroup_df) >=
            expected_subgroup_min
        ),
        "loo_rows_positive": (
            len(loo_df) >=
            expected_subgroup_min
        ),
        "stability_rows": (
            len(stability_df)
            == expected_stability_rows
        ),
        "dimension_result_rows": (
            len(dimension_results)
            == expected_dimension_rows
        ),
        "family_summary_rows": (
            len(family_summary)
            == len(FAMILIES)
        ),
        "all_required_bar1_states_present": True,
        "no_missing_symbol": (
            df["symbol"].notna().all()
        ),
        "no_missing_setup": (
            df["setup"].notna().all()
        ),
        "no_missing_direction": (
            df["direction"].notna().all()
        ),
    }

    # Verify every family state column is binary.
    for family_name in FAMILIES:
        column = (
            f"{family_name}_bar1_state"
        )

        values = set(
            df[column]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        if not values.issubset({0, 1}):
            integrity[
                "all_required_bar1_states_present"
            ] = False

    integrity["PASS"] = all(
        bool(value)
        for key, value in integrity.items()
        if key != "PASS"
    )

    return integrity


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "TRADE SENSE-AI — "
        "BAR 1→2 CONDITIONAL ROBUSTNESS FORENSIC ANALYSIS"
    )

    print("=" * 72)

    ensure_output_dir()

    # --------------------------------------------------------
    # 1. Load and freeze population
    # --------------------------------------------------------

    df = prepare_data()

    # --------------------------------------------------------
    # 2. Subgroup robustness
    # --------------------------------------------------------

    print("\nRunning subgroup robustness...")

    subgroup_df = run_subgroup_analysis(df)

    subgroup_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_subgroups.csv"
    )

    subgroup_df.to_csv(
        subgroup_path,
        index=False,
    )

    print(
        f"Subgroup rows: {len(subgroup_df)}"
    )

    # --------------------------------------------------------
    # 3. Leave-one-group-out robustness
    # --------------------------------------------------------

    print("\nRunning leave-one-group-out robustness...")

    loo_df = run_loo_analysis(df)

    loo_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_loo.csv"
    )

    loo_df.to_csv(
        loo_path,
        index=False,
    )

    print(
        f"LOO rows: {len(loo_df)}"
    )

    # --------------------------------------------------------
    # 4. Stability
    # --------------------------------------------------------

    stability_df = calculate_stability(
        subgroup_df,
        loo_df,
    )

    stability_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_stability.csv"
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Dimension results
    # --------------------------------------------------------

    dimension_results = build_dimension_results(
        stability_df
    )

    dimension_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_dimension_results.csv"
    )

    dimension_results.to_csv(
        dimension_path,
        index=False,
    )

    # --------------------------------------------------------
    # 6. Family summary
    # --------------------------------------------------------

    family_summary = build_family_summary(
        dimension_results
    )

    family_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_family_summary.csv"
    )

    family_summary.to_csv(
        family_path,
        index=False,
    )

    # --------------------------------------------------------
    # 7. Integrity
    # --------------------------------------------------------

    integrity = build_integrity(
        df,
        subgroup_df,
        loo_df,
        stability_df,
        dimension_results,
        family_summary,
    )

    integrity_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_integrity.csv"
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

    integrity_df.to_csv(
        integrity_path,
        index=False,
    )

    # --------------------------------------------------------
    # 8. Human-readable summary
    # --------------------------------------------------------

    family_pass_count = int(
        family_summary[
            "family_robustness_pass"
        ].sum()
    )

    total_families = len(FAMILIES)

    dimension_pass_count = int(
        dimension_results[
            "dimension_pass"
        ].sum()
    )

    total_dimensions = len(
        dimension_results
    )

    overall_pass = (
        integrity["PASS"]
        and family_pass_count
        == total_families
    )

    summary = {
        "input_file": str(INPUT_FILE),
        "output_dir": str(OUTPUT_DIR),
        "frozen_population": {
            "trades": EXPECTED_TRADES,
            "events": EXPECTED_EVENTS,
            "retained": EXPECTED_RETAINED,
        },
        "methodology": {
            "minimum_subgroup_trades": MIN_SUBGROUP_TRADES,
            "minimum_valid_subgroups": MIN_VALID_SUBGROUPS,
            "minimum_direction_consistency": (
                MIN_DIRECTION_CONSISTENCY
            ),
            "bar1_state_definition": (
                "GLOBAL median split"
            ),
            "subgroup_threshold_optimization": False,
            "machine_learning": False,
            "production_modified": False,
            "oos_modified": False,
        },
        "families": {
            name: {
                "bar1_feature": config[
                    "bar1_feature"
                ],
                "bar2_state": config[
                    "bar2_state"
                ],
                "state_name": config[
                    "state_name"
                ],
            }
            for name, config in FAMILIES.items()
        },
        "dimension_pass_count": dimension_pass_count,
        "dimension_total": total_dimensions,
        "family_pass_count": family_pass_count,
        "family_total": total_families,
        "integrity_pass": bool(
            integrity["PASS"]
        ),
        "overall_pass": bool(
            overall_pass
        ),
        "interpretation": (
            "Conditional robustness only. "
            "This remains association with the frozen "
            "5-bar / 0.25R event label. "
            "It does not establish causality, "
            "independent OOS predictive power, "
            "economic value, or a deployable rule."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "bar12_conditional_robustness_summary.json"
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
    # 9. Console report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("CONDITIONAL ROBUSTNESS RESULTS")
    print("=" * 72)

    for _, row in stability_df.iterrows():

        print(
            f"\n{row['family']} | "
            f"{row['dimension']}"
        )

        print(
            f"  Subgroups: "
            f"{int(row['valid_subgroups'])}"
        )

        print(
            f"  Subgroup direction: "
            f"{row['subgroup_direction_pct']:.2f}%"
            if pd.notna(
                row["subgroup_direction_pct"]
            )
            else "  Subgroup direction: NaN"
        )

        print(
            f"  LOO results: "
            f"{int(row['valid_loo_results'])}"
        )

        print(
            f"  LOO direction: "
            f"{row['loo_direction_pct']:.2f}%"
            if pd.notna(
                row["loo_direction_pct"]
            )
            else "  LOO direction: NaN"
        )

        print(
            f"  Dimension PASS: "
            f"{bool(row['dimension_pass'])}"
        )

    print("\n" + "=" * 72)
    print("FAMILY RESULTS")
    print("=" * 72)

    for _, row in family_summary.iterrows():

        print(
            f"\n{row['family']}"
        )

        print(
            f"  Dimensions passed: "
            f"{int(row['dimension_pass_count'])}"
            f"/{int(row['required_dimension_count'])}"
        )

        print(
            f"  FAMILY ROBUSTNESS PASS: "
            f"{bool(row['family_robustness_pass'])}"
        )

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"Dimension PASS count: "
        f"{dimension_pass_count}/{total_dimensions}"
    )

    print(
        f"Family PASS count: "
        f"{family_pass_count}/{total_families}"
    )

    print(
        f"Integrity PASS: "
        f"{bool(integrity['PASS'])}"
    )

    print(
        f"CONDITIONAL ROBUSTNESS: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print(
        "\nIMPORTANT:"
        "\nThis remains association with the frozen "
        "5-bar / 0.25R event label."
        "\nIt does NOT establish causality, "
        "independent OOS predictive power, "
        "economic value, or a deployable trading rule."
        "\nProduction strategy logic and OOS data remain untouched."
    )

    print("\nOutputs:")

    print(
        subgroup_path
    )

    print(
        loo_path
    )

    print(
        stability_path
    )

    print(
        dimension_path
    )

    print(
        family_path
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