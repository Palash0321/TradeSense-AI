"""
TradeSense-AI
FORENSIC VOLUME × EARLY-ADVERSE INDEPENDENT-EDGE AUDIT

Frozen research population
--------------------------
106 trades
60 frozen early-adverse events
46 retained

Frozen event definition
-----------------------
early_adverse_r_bar_5 >= 0.25R

Volume definition
-----------------
Signal-day Volume /
20-period rolling mean Volume through Signal_Date

This script is research-only.

Safety:
- frozen trade dataset READ-ONLY
- volume lineage output READ-ONLY
- production code unchanged
- OOS ledgers untouched
- no parameter optimization
- no strategy modification
- no deployment rule
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

FROZEN_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

VOLUME_LINEAGE_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_lineage"
    / "volume_lineage_per_trade.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_early_adverse"
)

TRADE_OUTPUT = (
    OUTPUT_DIR
    / "volume_early_adverse_trade_level.csv"
)

CATEGORY_OUTPUT = (
    OUTPUT_DIR
    / "volume_category_event_rates.csv"
)

CONTINUOUS_OUTPUT = (
    OUTPUT_DIR
    / "volume_continuous_effects.csv"
)

ROBUSTNESS_OUTPUT = (
    OUTPUT_DIR
    / "volume_robustness.csv"
)

INDEPENDENCE_OUTPUT = (
    OUTPUT_DIR
    / "volume_independence.csv"
)

INTEGRITY_OUTPUT = (
    OUTPUT_DIR
    / "volume_early_adverse_integrity.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "volume_early_adverse_summary.json"
)


# Frozen constants.
EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

EARLY_ADVERSE_THRESHOLD = 0.25

LINEAGE_STATUS_REQUIRED = "PASS"


# ============================================================================
# VOLUME CATEGORIES
# ============================================================================

# These categories were frozen before this analysis.
# They are descriptive and are NOT optimized here.

CATEGORY_ORDER = [
    "LOW",
    "NORMAL",
    "HIGH",
    "VERY_HIGH",
]


def classify_volume(
    ratio: float,
) -> str:

    if ratio < 1.0:
        return "LOW"

    if ratio < 1.5:
        return "NORMAL"

    if ratio < 2.0:
        return "HIGH"

    return "VERY_HIGH"


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(
    value: Any,
) -> Optional[float]:

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def normalize_symbol(
    value: Any,
) -> str:

    text = str(
        value
    ).strip().upper()

    aliases = {
        "NSE:RELIANCE": "RELIANCE.NS",
        "NSE:TCS": "TCS.NS",
        "NSE:HDFCBANK": "HDFCBANK.NS",
        "NSE:ICICIBANK": "ICICIBANK.NS",
        "NSE:INFY": "INFY.NS",
        "NSE:SBIN": "SBIN.NS",
    }

    return aliases.get(
        text,
        text,
    )


def normalize_date(
    value: Any,
) -> Optional[str]:

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    return timestamp.strftime(
        "%Y-%m-%d"
    )


def normalise_direction(
    value: Any,
) -> str:

    return (
        str(value)
        .strip()
        .upper()
    )


# ============================================================================
# STATISTICS
# ============================================================================

def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
):

    if total == 0:
        return (
            np.nan,
            np.nan,
        )

    p = successes / total

    denominator = (
        1
        + z**2 / total
    )

    centre = (
        p
        + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1 - p)
                + z**2 / (4 * total)
            )
            / total
        )
        / denominator
    )

    return (
        max(
            0.0,
            centre - margin,
        ),
        min(
            1.0,
            centre + margin,
        ),
    )


def risk_difference_ci(
    event_a: int,
    n_a: int,
    event_b: int,
    n_b: int,
):

    if (
        n_a == 0
        or n_b == 0
    ):

        return (
            np.nan,
            np.nan,
        )

    p_a = event_a / n_a
    p_b = event_b / n_b

    se = math.sqrt(
        (
            p_a * (1 - p_a) / n_a
        )
        + (
            p_b * (1 - p_b) / n_b
        )
    )

    rd = p_a - p_b

    return (
        rd - 1.959963984540054 * se,
        rd + 1.959963984540054 * se,
    )


def mann_whitney(
    x: pd.Series,
    y: pd.Series,
):

    try:

        from scipy.stats import mannwhitneyu

        result = mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        return float(
            result.statistic
        ), float(
            result.pvalue
        )

    except Exception:

        return (
            np.nan,
            np.nan,
        )


def fisher_exact(
    a: int,
    b: int,
    c: int,
    d: int,
):

    try:

        from scipy.stats import fisher_exact

        odds_ratio, p_value = (
            fisher_exact(
                [
                    [a, b],
                    [c, d],
                ]
            )
        )

        return (
            float(odds_ratio),
            float(p_value),
        )

    except Exception:

        return (
            np.nan,
            np.nan,
        )


# ============================================================================
# LOAD FROZEN TRADES
# ============================================================================

def load_frozen() -> pd.DataFrame:

    trades = pd.read_csv(
        FROZEN_FILE,
        low_memory=False,
    )

    if len(trades) != EXPECTED_TRADES:

        raise RuntimeError(
            "Frozen population mismatch: "
            f"expected {EXPECTED_TRADES}, "
            f"observed {len(trades)}"
        )

    required = [
        "_symbol",
        "entry_date",
        "Signal_Date",
        "direction",
        "early_adverse_r_bar_5",
    ]

    missing = [
        column
        for column in required
        if column not in trades.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing frozen columns: "
            + ", ".join(missing)
        )

    result = trades.copy()

    result[
        "_symbol_norm"
    ] = result[
        "_symbol"
    ].map(
        normalize_symbol
    )

    result[
        "_entry_date_norm"
    ] = result[
        "entry_date"
    ].map(
        normalize_date
    )

    result[
        "_signal_date_norm"
    ] = result[
        "Signal_Date"
    ].map(
        normalize_date
    )

    result[
        "_direction_norm"
    ] = result[
        "direction"
    ].map(
        normalise_direction
    )

    result[
        "frozen_event"
    ] = (
        pd.to_numeric(
            result[
                "early_adverse_r_bar_5"
            ],
            errors="coerce",
        )
        >= EARLY_ADVERSE_THRESHOLD
    ).astype(int)

    result[
        "trade_key"
    ] = (
        result[
            "_symbol_norm"
        ]
        + "|"
        + result[
            "_direction_norm"
        ]
        + "|"
        + result[
            "_entry_date_norm"
        ]
    )

    return result


# ============================================================================
# LOAD VOLUME LINEAGE
# ============================================================================

def load_volume_lineage() -> pd.DataFrame:

    if not VOLUME_LINEAGE_FILE.exists():

        raise RuntimeError(
            "Volume lineage file missing:\n"
            f"{VOLUME_LINEAGE_FILE}"
        )

    lineage = pd.read_csv(
        VOLUME_LINEAGE_FILE,
        low_memory=False,
    )

    required = [
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
        for column in required
        if column not in lineage.columns
    ]

    if missing:

        raise RuntimeError(
            "Volume lineage schema mismatch: "
            + ", ".join(missing)
        )

    lineage = lineage.copy()

    lineage[
        "trade_key"
    ] = lineage[
        "trade_key"
    ].astype(str)

    lineage[
        "volume_ratio"
    ] = pd.to_numeric(
        lineage[
            "volume_ratio"
        ],
        errors="coerce",
    )

    return lineage


# ============================================================================
# MERGE
# ============================================================================

def build_trade_level() -> pd.DataFrame:

    frozen = load_frozen()

    lineage = load_volume_lineage()

    if (
        lineage[
            "lineage_status"
        ]
        != LINEAGE_STATUS_REQUIRED
    ).any():

        raise RuntimeError(
            "Volume lineage is not PASS for all rows."
        )

    if (
        lineage[
            "trade_key"
        ].duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate volume lineage trade keys."
        )

    merged = frozen[
        [
            "trade_key",
            "_symbol_norm",
            "_direction_norm",
            "_entry_date_norm",
            "_signal_date_norm",
            "frozen_event",
        ]
    ].merge(
        lineage[
            [
                "trade_key",
                "volume_ratio",
                "volume_category",
                "lineage_status",
            ]
        ],
        on="trade_key",
        how="left",
        validate="one_to_one",
    )

    merged.rename(
        columns={
            "_symbol_norm": "symbol",
            "_direction_norm": "direction",
            "_entry_date_norm": "entry_date",
            "_signal_date_norm": "signal_date",
        },
        inplace=True,
    )

    if merged[
        "volume_ratio"
    ].isna().any():

        raise RuntimeError(
            "Missing Volume Ratio after frozen-trade merge."
        )

    merged[
        "volume_category"
    ] = merged[
        "volume_ratio"
    ].map(
        classify_volume
    )

    return merged


# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================

def category_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for category in CATEGORY_ORDER:

        subset = trades[
            trades[
                "volume_category"
            ]
            == category
        ]

        n = len(subset)

        events = int(
            subset[
                "frozen_event"
            ].sum()
        )

        retained = n - events

        event_rate = (
            events / n
            if n
            else np.nan
        )

        ci_low, ci_high = (
            wilson_interval(
                events,
                n,
            )
        )

        rows.append(
            {
                "category": category,
                "n": n,
                "events": events,
                "retained": retained,
                "event_rate_pct": (
                    event_rate * 100
                    if n
                    else np.nan
                ),
                "event_rate_ci_low_pct": (
                    ci_low * 100
                    if n
                    else np.nan
                ),
                "event_rate_ci_high_pct": (
                    ci_high * 100
                    if n
                    else np.nan
                ),
                "mean_volume_ratio": (
                    subset[
                        "volume_ratio"
                    ].mean()
                    if n
                    else np.nan
                ),
                "median_volume_ratio": (
                    subset[
                        "volume_ratio"
                    ].median()
                    if n
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# CONTINUOUS VOLUME ANALYSIS
# ============================================================================

def continuous_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    event = trades.loc[
        trades[
            "frozen_event"
        ] == 1,
        "volume_ratio",
    ]

    retained = trades.loc[
        trades[
            "frozen_event"
        ] == 0,
        "volume_ratio",
    ]

    u_stat, p_value = (
        mann_whitney(
            event,
            retained,
        )
    )

    correlation_pearson = (
        trades[
            [
                "volume_ratio",
                "frozen_event",
            ]
        ]
        .corr(
            method="pearson"
        )
        .iloc[0, 1]
    )

    correlation_spearman = (
        trades[
            [
                "volume_ratio",
                "frozen_event",
            ]
        ]
        .corr(
            method="spearman"
        )
        .iloc[0, 1]
    )

    event_mean = event.mean()
    retained_mean = retained.mean()

    event_median = event.median()
    retained_median = retained.median()

    return pd.DataFrame(
        [
            {
                "metric": "mean_volume_ratio_event",
                "value": event_mean,
            },
            {
                "metric": "mean_volume_ratio_retained",
                "value": retained_mean,
            },
            {
                "metric": "mean_difference_event_minus_retained",
                "value": event_mean - retained_mean,
            },
            {
                "metric": "median_volume_ratio_event",
                "value": event_median,
            },
            {
                "metric": "median_volume_ratio_retained",
                "value": retained_median,
            },
            {
                "metric": "median_difference_event_minus_retained",
                "value": event_median - retained_median,
            },
            {
                "metric": "pearson_correlation",
                "value": correlation_pearson,
            },
            {
                "metric": "spearman_correlation",
                "value": correlation_spearman,
            },
            {
                "metric": "mann_whitney_u",
                "value": u_stat,
            },
            {
                "metric": "mann_whitney_p_value",
                "value": p_value,
            },
        ]
    )


# ============================================================================
# MEDIAN-SPLIT EFFECT
# ============================================================================

def median_split(
    trades: pd.DataFrame,
) -> Dict[str, Any]:

    median = trades[
        "volume_ratio"
    ].median()

    low = trades[
        "volume_ratio"
    ] <= median

    high = trades[
        "volume_ratio"
    ] > median

    low_events = int(
        trades.loc[
            low,
            "frozen_event",
        ].sum()
    )

    high_events = int(
        trades.loc[
            high,
            "frozen_event",
        ].sum()
    )

    low_n = int(
        low.sum()
    )

    high_n = int(
        high.sum()
    )

    rd = (
        high_events / high_n
        -
        low_events / low_n
    )

    ci_low, ci_high = (
        risk_difference_ci(
            high_events,
            high_n,
            low_events,
            low_n,
        )
    )

    odds_ratio, p_value = (
        fisher_exact(
            high_events,
            high_n - high_events,
            low_events,
            low_n - low_events,
        )
    )

    return {
        "median_volume_ratio": median,
        "low_n": low_n,
        "low_events": low_events,
        "low_event_rate_pct": (
            low_events
            / low_n
            * 100
        ),
        "high_n": high_n,
        "high_events": high_events,
        "high_event_rate_pct": (
            high_events
            / high_n
            * 100
        ),
        "risk_difference_high_minus_low_pp": (
            rd * 100
        ),
        "rd_ci_low_pp": (
            ci_low * 100
        ),
        "rd_ci_high_pp": (
            ci_high * 100
        ),
        "odds_ratio_high_vs_low": odds_ratio,
        "fisher_p_value": p_value,
    }


# ============================================================================
# ROBUSTNESS
# ============================================================================

def subgroup_effect(
    subset: pd.DataFrame,
) -> Optional[Dict[str, Any]]:

    if len(subset) < 8:

        return None

    if (
        subset[
            "volume_ratio"
        ].nunique()
        < 2
    ):

        return None

    median = subset[
        "volume_ratio"
    ].median()

    low = subset[
        "volume_ratio"
    ] <= median

    high = subset[
        "volume_ratio"
    ] > median

    n_low = int(
        low.sum()
    )

    n_high = int(
        high.sum()
    )

    if n_low == 0 or n_high == 0:
        return None

    events_low = int(
        subset.loc[
            low,
            "frozen_event",
        ].sum()
    )

    events_high = int(
        subset.loc[
            high,
            "frozen_event",
        ].sum()
    )

    rd = (
        events_high / n_high
        -
        events_low / n_low
    )

    ci_low, ci_high = (
        risk_difference_ci(
            events_high,
            n_high,
            events_low,
            n_low,
        )
    )

    return {
        "n": len(subset),
        "low_n": n_low,
        "high_n": n_high,
        "low_event_rate_pct": (
            events_low
            / n_low
            * 100
        ),
        "high_event_rate_pct": (
            events_high
            / n_high
            * 100
        ),
        "risk_difference_pp": (
            rd * 100
        ),
        "rd_ci_low_pp": (
            ci_low * 100
        ),
        "rd_ci_high_pp": (
            ci_high * 100
        ),
    }


def robustness_analysis(
    trades: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    year = pd.to_datetime(
        trades[
            "entry_date"
        ],
        errors="coerce",
    ).dt.year

    trades = trades.copy()

    trades[
        "year"
    ] = year

    dimensions = {
        "year": "year",
        "symbol": "symbol",
        "direction": "direction",
        "volume_category": "volume_category",
    }

    for dimension, column in dimensions.items():

        for value, subset in trades.groupby(
            column,
            dropna=False,
        ):

            result = subgroup_effect(
                subset
            )

            if result is None:
                continue

            rows.append(
                {
                    "dimension": dimension,
                    "group": str(value),
                    **result,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SETUP-CONTROLLED ANALYSIS
# ============================================================================

def setup_controlled_analysis(
    frozen: pd.DataFrame,
    lineage: pd.DataFrame,
) -> pd.DataFrame:

    setup_column = None

    for candidate in [
        "setup",
        "Setup",
        "Setup_Normalized",
    ]:

        if candidate in frozen.columns:

            setup_column = candidate
            break

    if setup_column is None:

        return pd.DataFrame(
            [
                {
                    "status": "SETUP_FIELD_UNAVAILABLE",
                }
            ]
        )

    setup = frozen[
        [
            "trade_key",
            setup_column,
        ]
    ].copy()

    setup.rename(
        columns={
            setup_column: "setup"
        },
        inplace=True,
    )

    setup[
        "setup"
    ] = (
        setup[
            "setup"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    merged = pd.read_csv(
        TRADE_OUTPUT,
        low_memory=False,
    )

    merged = merged.merge(
        setup,
        on="trade_key",
        how="left",
        validate="one_to_one",
    )

    rows = []

    for value, subset in merged.groupby(
        "setup",
        dropna=False,
    ):

        result = subgroup_effect(
            subset
        )

        if result is None:
            continue

        rows.append(
            {
                "dimension": "setup",
                "group": str(value),
                **result,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# INDEPENDENCE — EXISTING CONTEXT
# ============================================================================

def context_independence(
    frozen: pd.DataFrame,
    volume_trade: pd.DataFrame,
) -> pd.DataFrame:

    context_columns = [
        "entry_trend_strength",
        "entry_trend_slope",
        "entry_atr",
        "Setup_Confidence",
        "Liquidity_Sweep",
        "Structure_Bias",
        "BOS",
        "CHOCH",
    ]

    rows = []

    merged = frozen[
        [
            "trade_key",
            "frozen_event",
        ]
        + [
            column
            for column in context_columns
            if column in frozen.columns
        ]
    ].merge(
        volume_trade[
            [
                "trade_key",
                "volume_ratio",
            ]
        ],
        on="trade_key",
        how="inner",
        validate="one_to_one",
    )

    for column in context_columns:

        if column not in merged.columns:
            continue

        series = pd.to_numeric(
            merged[column],
            errors="coerce",
        )

        valid = (
            series.notna()
            & merged[
                "volume_ratio"
            ].notna()
        )

        if valid.sum() < 10:
            continue

        volume_corr = (
            merged.loc[
                valid,
                [
                    "volume_ratio",
                    "frozen_event",
                ],
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        )

        context_corr = (
            merged.loc[
                valid,
                [
                    column,
                    "frozen_event",
                ],
            ]
            .corr(
                method="spearman"
            )
            .iloc[0, 1]
        )

        rows.append(
            {
                "context_field": column,
                "n": int(valid.sum()),
                "volume_vs_event_spearman": (
                    volume_corr
                ),
                "context_vs_event_spearman": (
                    context_corr
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )
    print(
        "TRADESENSE-AI — VOLUME × EARLY-ADVERSE "
        "INDEPENDENT-EDGE AUDIT"
    )
    print(
        "=" * 100
    )

    print()
    print(
        "FROZEN HYPOTHESIS:"
    )

    print(
        "Early adverse >= 0.25R "
        "within first 5 entry bars"
    )

    print()
    print(
        "FROZEN POPULATION:"
    )

    print(
        "106 trades | 60 events | 46 retained"
    )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    frozen = load_frozen()

    lineage = load_volume_lineage()

    trades = build_trade_level()

    # ------------------------------------------------------------------------
    # Frozen event integrity
    # ------------------------------------------------------------------------

    event_count = int(
        trades[
            "frozen_event"
        ].sum()
    )

    retained_count = (
        len(trades)
        - event_count
    )

    if event_count != EXPECTED_EVENTS:

        raise RuntimeError(
            "Frozen event count mismatch: "
            f"expected {EXPECTED_EVENTS}, "
            f"observed {event_count}"
        )

    if retained_count != EXPECTED_RETAINED:

        raise RuntimeError(
            "Frozen retained count mismatch: "
            f"expected {EXPECTED_RETAINED}, "
            f"observed {retained_count}"
        )

    # ------------------------------------------------------------------------
    # Trade-level output
    # ------------------------------------------------------------------------

    trades[
        [
            "trade_key",
            "symbol",
            "direction",
            "signal_date",
            "entry_date",
            "volume_ratio",
            "volume_category",
            "frozen_event",
        ]
    ].to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Category analysis
    # ------------------------------------------------------------------------

    categories = category_analysis(
        trades
    )

    categories.to_csv(
        CATEGORY_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Continuous analysis
    # ------------------------------------------------------------------------

    continuous = continuous_analysis(
        trades
    )

    continuous.to_csv(
        CONTINUOUS_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Median split
    # ------------------------------------------------------------------------

    split = median_split(
        trades
    )

    # ------------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------------

    robustness = robustness_analysis(
        trades
    )

    robustness.to_csv(
        ROBUSTNESS_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Setup-controlled analysis
    # ------------------------------------------------------------------------

    setup_results = setup_controlled_analysis(
        frozen,
        lineage,
    )

    setup_output = (
        OUTPUT_DIR
        / "volume_setup_controlled.csv"
    )

    setup_results.to_csv(
        setup_output,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Context independence
    # ------------------------------------------------------------------------

    independence = context_independence(
        frozen,
        trades,
    )

    independence.to_csv(
        INDEPENDENCE_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------------

    duplicate_keys = int(
        trades[
            "trade_key"
        ]
        .duplicated()
        .sum()
    )

    missing_volume = int(
        trades[
            "volume_ratio"
        ]
        .isna()
        .sum()
    )

    invalid_category = int(
        (
            ~trades[
                "volume_category"
            ].isin(
                CATEGORY_ORDER
            )
        ).sum()
    )

    integrity_rows = [
        {
            "check": "FROZEN_TRADE_COUNT",
            "actual": len(trades),
            "expected": EXPECTED_TRADES,
            "status": (
                "PASS"
                if len(trades)
                == EXPECTED_TRADES
                else "FAIL"
            ),
        },
        {
            "check": "FROZEN_EVENT_COUNT",
            "actual": event_count,
            "expected": EXPECTED_EVENTS,
            "status": (
                "PASS"
                if event_count
                == EXPECTED_EVENTS
                else "FAIL"
            ),
        },
        {
            "check": "FROZEN_RETAINED_COUNT",
            "actual": retained_count,
            "expected": EXPECTED_RETAINED,
            "status": (
                "PASS"
                if retained_count
                == EXPECTED_RETAINED
                else "FAIL"
            ),
        },
        {
            "check": "DUPLICATE_TRADE_KEYS",
            "actual": duplicate_keys,
            "expected": 0,
            "status": (
                "PASS"
                if duplicate_keys == 0
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
            "check": "INVALID_VOLUME_CATEGORY",
            "actual": invalid_category,
            "expected": 0,
            "status": (
                "PASS"
                if invalid_category == 0
                else "FAIL"
            ),
        },
    ]

    integrity = pd.DataFrame(
        integrity_rows
    )

    integrity.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    continuous_dict = dict(
        zip(
            continuous[
                "metric"
            ],
            continuous[
                "value"
            ],
        )
    )

    summary = {
        "status": "PASS",
        "frozen_trades": EXPECTED_TRADES,
        "frozen_events": EXPECTED_EVENTS,
        "frozen_retained": EXPECTED_RETAINED,
        "volume_definition": (
            "Signal-day Volume / "
            "20-period rolling mean Volume"
        ),
        "event_definition": (
            "early_adverse_r_bar_5 >= 0.25R"
        ),
        "threshold_optimization": "NONE",
        "strategy_modification": "NONE",
        "oos_modification": "NONE",
        "profitability_rule": "NOT_CREATED",
        "category_counts": (
            categories.set_index(
                "category"
            )["n"].to_dict()
        ),
        "category_event_rates_pct": (
            categories.set_index(
                "category"
            )["event_rate_pct"].to_dict()
        ),
        "continuous": continuous_dict,
        "median_split": split,
        "integrity_failures": int(
            (
                integrity[
                    "status"
                ]
                == "FAIL"
            ).sum()
        ),
    }

    with SUMMARY_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "VOLUME CATEGORY EVENT RATES"
    )
    print(
        "=" * 100
    )

    print(
        categories.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "CONTINUOUS VOLUME EFFECT"
    )
    print(
        "=" * 100
    )

    print(
        continuous.to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 100
    )
    print(
        "MEDIAN-SPLIT VOLUME EFFECT"
    )
    print(
        "=" * 100
    )

    for key, value in split.items():

        print(
            f"{key:<45}: {value}"
        )

    print()
    print(
        "=" * 100
    )
    print(
        "ROBUSTNESS"
    )
    print(
        "=" * 100
    )

    if robustness.empty:

        print(
            "No valid robustness strata."
        )

    else:

        print(
            robustness.to_string(
                index=False
            )
        )

    print()
    print(
        "=" * 100
    )
    print(
        "INTEGRITY"
    )
    print(
        "=" * 100
    )

    print(
        integrity.to_string(
            index=False
        )
    )

    print()
    print(
        "OUTPUTS:"
    )

    print(
        f"Trade level       : {TRADE_OUTPUT}"
    )

    print(
        f"Categories        : {CATEGORY_OUTPUT}"
    )

    print(
        f"Continuous        : {CONTINUOUS_OUTPUT}"
    )

    print(
        f"Robustness        : {ROBUSTNESS_OUTPUT}"
    )

    print(
        f"Independence      : {INDEPENDENCE_OUTPUT}"
    )

    print(
        f"Integrity         : {INTEGRITY_OUTPUT}"
    )

    print(
        f"Summary           : {SUMMARY_OUTPUT}"
    )

    print()
    print(
        "=" * 100
    )

    if (
        summary[
            "integrity_failures"
        ]
        == 0
    ):

        print(
            "STATUS: VOLUME × EARLY-ADVERSE "
            "FORENSIC AUDIT PASS"
        )

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "This establishes association evidence only."
        )

        print(
            "No Volume deployment rule was created."
        )

        print(
            "Production and OOS remain unchanged."
        )

        return 0

    print(
        "STATUS: VOLUME × EARLY-ADVERSE "
        "FORENSIC AUDIT FAIL"
    )

    return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )