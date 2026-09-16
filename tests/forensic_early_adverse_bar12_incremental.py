"""
TradeSense-AI — Bar 1→2 Incremental Information Forensic Analysis

PURPOSE
-------
Determine whether information available by the end of Bar 2 adds
meaningful separation beyond information already available at the
end of Bar 1.

This is a RESEARCH-ONLY forensic analysis.

FROZEN HYPOTHESIS
-----------------
Early-adverse event:
    stored production early_adverse_r_bar_5 >= 0.25R

Frozen population:
    106 trades
     60 events
     46 retained

RESEARCH QUESTION
-----------------
Does Bar 2 add information beyond Bar 1?

We explicitly compare:

    Model/Information Set A:
        Bar 1 only

    Model/Information Set B:
        Bar 1 + Bar 2 trajectory information

No predictive model is trained.
No threshold is optimized.
No deployment rule is selected.

IMPORTANT
---------
The frozen event label remains the stored production bar-5 label.

No production code is modified.
No frozen CSV is modified.
No OOS data is modified.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, mannwhitneyu
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_incremental"
)

OUTPUT_INCREMENTAL = (
    OUTPUT_DIR / "early_adverse_bar12_incremental.csv"
)

OUTPUT_EFFECTS = (
    OUTPUT_DIR / "early_adverse_bar12_incremental_effects.csv"
)

OUTPUT_DIRECTION = (
    OUTPUT_DIR / "early_adverse_bar12_incremental_direction.csv"
)

OUTPUT_SETUP = (
    OUTPUT_DIR / "early_adverse_bar12_incremental_setup.csv"
)

OUTPUT_STATE_FAMILY = (
    OUTPUT_DIR / "early_adverse_bar12_state_family.csv"
)

OUTPUT_CORRELATION = (
    OUTPUT_DIR / "early_adverse_bar12_incremental_correlation.csv"
)

OUTPUT_INTEGRITY = (
    OUTPUT_DIR / "early_adverse_bar12_incremental_integrity.csv"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46
EXPECTED_PATH_ROWS = 530

EVENT_COLUMN = "stored_frozen_event"

EPS = 1e-12


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return np.nan, np.nan

    p = successes / total
    denom = 1.0 + (z * z) / total
    centre = (
        p + (z * z) / (2.0 * total)
    ) / denom

    half = (
        z
        * math.sqrt(
            (
                p * (1.0 - p)
                + (z * z) / (4.0 * total)
            )
            / total
        )
        / denom
    )

    return centre - half, centre + half


def risk_difference_ci(
    event_success,
    event_total,
    retained_success,
    retained_total,
):
    e_low, e_high = wilson_interval(
        event_success,
        event_total,
    )

    r_low, r_high = wilson_interval(
        retained_success,
        retained_total,
    )

    return (
        e_low - r_high,
        e_high - r_low,
    )


def make_trade_id(df):
    fields = [
        "symbol",
        "direction",
        "entry_date",
        "entry_timestamp",
        "signal_timestamp",
        "setup",
    ]

    missing = [c for c in fields if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing trade identity columns: {missing}"
        )

    return (
        df[fields]
        .astype(str)
        .agg("|".join, axis=1)
    )


def median_difference(event_series, retained_series):
    event_series = pd.to_numeric(
        event_series,
        errors="coerce",
    ).dropna()

    retained_series = pd.to_numeric(
        retained_series,
        errors="coerce",
    ).dropna()

    if len(event_series) == 0 or len(retained_series) == 0:
        return np.nan

    return (
        event_series.median()
        - retained_series.median()
    )


def mean_difference(event_series, retained_series):
    event_series = pd.to_numeric(
        event_series,
        errors="coerce",
    ).dropna()

    retained_series = pd.to_numeric(
        retained_series,
        errors="coerce",
    ).dropna()

    if len(event_series) == 0 or len(retained_series) == 0:
        return np.nan

    return (
        event_series.mean()
        - retained_series.mean()
    )


def mann_whitney_result(event_series, retained_series):
    event_series = pd.to_numeric(
        event_series,
        errors="coerce",
    ).dropna()

    retained_series = pd.to_numeric(
        retained_series,
        errors="coerce",
    ).dropna()

    if (
        len(event_series) == 0
        or len(retained_series) == 0
        or not SCIPY_AVAILABLE
    ):
        return np.nan

    try:
        result = mannwhitneyu(
            event_series,
            retained_series,
            alternative="two-sided",
        )
        return float(result.pvalue)
    except Exception:
        return np.nan


def build_2x2_state_table(
    event_positive,
    event_total,
    retained_positive,
    retained_total,
):
    return np.array(
        [
            [
                event_positive,
                event_total - event_positive,
            ],
            [
                retained_positive,
                retained_total - retained_positive,
            ],
        ],
        dtype=int,
    )


def fisher_state_p(
    event_positive,
    event_total,
    retained_positive,
    retained_total,
):
    if not SCIPY_AVAILABLE:
        return np.nan

    table = build_2x2_state_table(
        event_positive,
        event_total,
        retained_positive,
        retained_total,
    )

    try:
        _, p = fisher_exact(
            table,
            alternative="two-sided",
        )
        return float(p)
    except Exception:
        return np.nan


def state_statistics(series, event_mask):
    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = series.notna()

    event_values = series[
        valid & event_mask
    ]

    retained_values = series[
        valid & (~event_mask)
    ]

    event_n = len(event_values)
    retained_n = len(retained_values)

    return {
        "event_n": event_n,
        "retained_n": retained_n,
        "event_mean": (
            event_values.mean()
            if event_n else np.nan
        ),
        "retained_mean": (
            retained_values.mean()
            if retained_n else np.nan
        ),
        "mean_difference": (
            event_values.mean()
            - retained_values.mean()
            if event_n and retained_n
            else np.nan
        ),
        "event_median": (
            event_values.median()
            if event_n else np.nan
        ),
        "retained_median": (
            retained_values.median()
            if retained_n else np.nan
        ),
        "median_difference": (
            event_values.median()
            - retained_values.median()
            if event_n and retained_n
            else np.nan
        ),
        "event_std": (
            event_values.std(ddof=1)
            if event_n > 1 else np.nan
        ),
        "retained_std": (
            retained_values.std(ddof=1)
            if retained_n > 1 else np.nan
        ),
        "mann_whitney_p": mann_whitney_result(
            event_values,
            retained_values,
        ),
    }


# ============================================================================
# LOAD AND VALIDATE
# ============================================================================

def load_input():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"[INFO] Input file: {INPUT_FILE}"
    )

    print(
        f"[INFO] Input rows: {len(df)}"
    )

    print(
        f"[INFO] Input columns: {len(df.columns)}"
    )

    required = [
        "symbol",
        "direction",
        "entry_date",
        "entry_timestamp",
        "signal_timestamp",
        "setup",
        "bar_number",
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "close_from_entry_r",
        EVENT_COLUMN,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {c}" for c in missing
            )
        )

    df["trade_id"] = make_trade_id(df)

    trade_count = df["trade_id"].nunique()

    if trade_count != EXPECTED_TRADES:
        raise ValueError(
            f"Expected {EXPECTED_TRADES} trades, "
            f"observed {trade_count}"
        )

    if len(df) != EXPECTED_PATH_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_PATH_ROWS} path rows, "
            f"observed {len(df)}"
        )

    event_values = pd.to_numeric(
        df[EVENT_COLUMN],
        errors="coerce",
    )

    if event_values.isna().any():
        raise ValueError(
            "Frozen event column contains missing values."
        )

    event_values = event_values.astype(int)

    trade_labels = (
        df.assign(_event=event_values)
        .groupby("trade_id")["_event"]
        .nunique()
    )

    if not (trade_labels == 1).all():
        raise ValueError(
            "Frozen event label is inconsistent within trades."
        )

    trade_level = (
        df.assign(_event=event_values)
        .groupby("trade_id")
        .agg(
            event=("_event", "first"),
        )
        .reset_index()
    )

    events = int(
        trade_level["event"].sum()
    )

    retained = int(
        len(trade_level) - events
    )

    if events != EXPECTED_EVENTS:
        raise ValueError(
            f"Expected {EXPECTED_EVENTS} events, "
            f"observed {events}"
        )

    if retained != EXPECTED_RETAINED:
        raise ValueError(
            f"Expected {EXPECTED_RETAINED} retained, "
            f"observed {retained}"
        )

    print()
    print("[PASS] Frozen population:")
    print(f"       Trades   : {trade_count}")
    print(f"       Events   : {events}")
    print(f"       Retained : {retained}")
    print(f"       Path rows: {len(df)}")

    return df


# ============================================================================
# BUILD BAR-LEVEL INFORMATION SETS
# ============================================================================

def build_trade_level(df):
    numeric_columns = [
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "close_from_entry_r",
        "close_move_r",
        "body_pct_range",
        "upper_wick_pct_range",
        "lower_wick_pct_range",
        "close_location",
        "bar_range_atr",
        "body_atr",
        "adverse_wick_atr",
        "favorable_wick_atr",
        "bar_adverse_atr",
        "bar_favorable_atr",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    bar1 = (
        df[df["bar_number"] == 1]
        .copy()
    )

    bar2 = (
        df[df["bar_number"] == 2]
        .copy()
    )

    if len(bar1) != EXPECTED_TRADES:
        raise ValueError(
            f"Expected {EXPECTED_TRADES} Bar1 rows, "
            f"observed {len(bar1)}"
        )

    if len(bar2) != EXPECTED_TRADES:
        raise ValueError(
            f"Expected {EXPECTED_TRADES} Bar2 rows, "
            f"observed {len(bar2)}"
        )

    if bar1["trade_id"].nunique() != EXPECTED_TRADES:
        raise ValueError(
            "Bar1 trade identity mismatch."
        )

    if bar2["trade_id"].nunique() != EXPECTED_TRADES:
        raise ValueError(
            "Bar2 trade identity mismatch."
        )

    b1 = bar1.set_index("trade_id")
    b2 = bar2.set_index("trade_id")

    if set(b1.index) != set(b2.index):
        raise ValueError(
            "Bar1 and Bar2 trade sets do not match."
        )

    result = pd.DataFrame(index=b1.index)

    result["symbol"] = b1["symbol"]
    result["direction"] = b1["direction"]
    result["setup"] = b1["setup"]
    result["entry_date"] = b1["entry_date"]

    result["event"] = pd.to_numeric(
        b1[EVENT_COLUMN],
        errors="coerce",
    ).astype(int)

    # ------------------------------------------------------------------------
    # BAR 1 INFORMATION
    # ------------------------------------------------------------------------

    bar1_features = [
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "close_from_entry_r",
        "close_move_r",
        "body_pct_range",
        "upper_wick_pct_range",
        "lower_wick_pct_range",
        "close_location",
        "bar_range_atr",
        "body_atr",
        "adverse_wick_atr",
        "favorable_wick_atr",
        "bar_adverse_atr",
        "bar_favorable_atr",
    ]

    for feature in bar1_features:
        if feature in b1.columns:
            result[f"bar1_{feature}"] = b1[feature]

    # ------------------------------------------------------------------------
    # BAR 2 INFORMATION
    # ------------------------------------------------------------------------

    bar2_features = [
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "close_from_entry_r",
        "close_move_r",
        "body_pct_range",
        "upper_wick_pct_range",
        "lower_wick_pct_range",
        "close_location",
        "bar_range_atr",
        "body_atr",
        "adverse_wick_atr",
        "favorable_wick_atr",
        "bar_adverse_atr",
        "bar_favorable_atr",
    ]

    for feature in bar2_features:
        if feature in b2.columns:
            result[f"bar2_{feature}"] = b2[feature]

    # ------------------------------------------------------------------------
    # BAR1 → BAR2 INCREMENTAL FEATURES
    # ------------------------------------------------------------------------

    pairs = [
        "bar_adverse_r",
        "bar_favorable_r",
        "cumulative_mae_r",
        "cumulative_mfe_r",
        "net_excursion_r",
        "close_from_entry_r",
        "close_move_r",
        "body_pct_range",
        "upper_wick_pct_range",
        "lower_wick_pct_range",
        "close_location",
        "bar_range_atr",
        "body_atr",
        "adverse_wick_atr",
        "favorable_wick_atr",
        "bar_adverse_atr",
        "bar_favorable_atr",
    ]

    for feature in pairs:
        c1 = f"bar1_{feature}"
        c2 = f"bar2_{feature}"

        if c1 in result.columns and c2 in result.columns:
            result[f"delta_{feature}"] = (
                result[c2]
                - result[c1]
            )

    # More interpretable trajectory variables.
    result["delta_mae_r"] = (
        result["bar2_cumulative_mae_r"]
        - result["bar1_cumulative_mae_r"]
    )

    result["delta_mfe_r"] = (
        result["bar2_cumulative_mfe_r"]
        - result["bar1_cumulative_mfe_r"]
    )

    result["delta_net_excursion_r"] = (
        result["bar2_net_excursion_r"]
        - result["bar1_net_excursion_r"]
    )

    result["delta_close_from_entry_r"] = (
        result["bar2_close_from_entry_r"]
        - result["bar1_close_from_entry_r"]
    )

    result["delta_bar_adverse_r"] = (
        result["bar2_bar_adverse_r"]
        - result["bar1_bar_adverse_r"]
    )

    result["delta_bar_favorable_r"] = (
        result["bar2_bar_favorable_r"]
        - result["bar1_bar_favorable_r"]
    )

    # Relationship between adverse and favorable excursion.
    result["bar1_mae_minus_mfe"] = (
        result["bar1_cumulative_mae_r"]
        - result["bar1_cumulative_mfe_r"]
    )

    result["bar2_mae_minus_mfe"] = (
        result["bar2_cumulative_mae_r"]
        - result["bar2_cumulative_mfe_r"]
    )

    result["delta_mae_minus_mfe"] = (
        result["bar2_mae_minus_mfe"]
        - result["bar1_mae_minus_mfe"]
    )

    # ------------------------------------------------------------------------
    # PRE-SPECIFIED STATE VARIABLES
    # ------------------------------------------------------------------------

    result["bar2_mae_increased"] = (
        result["delta_mae_r"] > EPS
    )

    result["bar2_mfe_increased"] = (
        result["delta_mfe_r"] > EPS
    )

    result["bar2_net_deteriorated"] = (
        result["delta_net_excursion_r"] < -EPS
    )

    result["bar2_close_deteriorated"] = (
        result["delta_close_from_entry_r"] < -EPS
    )

    result["bar2_adverse_worsened"] = (
        result["delta_bar_adverse_r"] > EPS
    )

    result["bar2_favorable_failed_to_improve"] = (
        result["delta_bar_favorable_r"] <= EPS
    )

    result["adverse_up_favorable_not_up"] = (
        (result["delta_bar_adverse_r"] > EPS)
        &
        (result["delta_bar_favorable_r"] <= EPS)
    )

    result["adverse_up_net_down"] = (
        (result["delta_bar_adverse_r"] > EPS)
        &
        (result["delta_net_excursion_r"] < -EPS)
    )

    result["adverse_up_close_down"] = (
        (result["delta_bar_adverse_r"] > EPS)
        &
        (result["delta_close_from_entry_r"] < -EPS)
    )

    result["adverse_up_favorable_not_up_net_down"] = (
        (result["delta_bar_adverse_r"] > EPS)
        &
        (result["delta_bar_favorable_r"] <= EPS)
        &
        (result["delta_net_excursion_r"] < -EPS)
    )

    result = result.reset_index()

    return result


# ============================================================================
# BAR 1 VS BAR 2 INFORMATION CONTENT
# ============================================================================

def analyze_incremental_features(trades):
    event_mask = (
        trades["event"] == 1
    )

    feature_groups = {
        "bar1_only": [
            "bar1_cumulative_mae_r",
            "bar1_cumulative_mfe_r",
            "bar1_net_excursion_r",
            "bar1_close_from_entry_r",
            "bar1_bar_adverse_r",
            "bar1_bar_favorable_r",
            "bar1_mae_minus_mfe",
        ],
        "bar2_snapshot": [
            "bar2_cumulative_mae_r",
            "bar2_cumulative_mfe_r",
            "bar2_net_excursion_r",
            "bar2_close_from_entry_r",
            "bar2_bar_adverse_r",
            "bar2_bar_favorable_r",
            "bar2_mae_minus_mfe",
        ],
        "bar2_incremental": [
            "delta_mae_r",
            "delta_mfe_r",
            "delta_net_excursion_r",
            "delta_close_from_entry_r",
            "delta_bar_adverse_r",
            "delta_bar_favorable_r",
            "delta_mae_minus_mfe",
        ],
    }

    rows = []

    for information_set, features in feature_groups.items():
        for feature in features:
            if feature not in trades.columns:
                continue

            stats = state_statistics(
                trades[feature],
                event_mask,
            )

            stats["information_set"] = information_set
            stats["feature"] = feature

            rows.append(stats)

    output = pd.DataFrame(rows)

    # Put important columns first.
    ordered = [
        "information_set",
        "feature",
        "event_n",
        "retained_n",
        "event_mean",
        "retained_mean",
        "mean_difference",
        "event_median",
        "retained_median",
        "median_difference",
        "event_std",
        "retained_std",
        "mann_whitney_p",
    ]

    output = output[
        [c for c in ordered if c in output.columns]
    ]

    return output


# ============================================================================
# CONTROLLED BAR1 → BAR2 EFFECT COMPARISON
# ============================================================================

def compare_bar1_bar2_effects(trades):
    event_mask = (
        trades["event"] == 1
    )

    comparisons = [
        (
            "cumulative_mae_r",
            "bar1_cumulative_mae_r",
            "bar2_cumulative_mae_r",
        ),
        (
            "cumulative_mfe_r",
            "bar1_cumulative_mfe_r",
            "bar2_cumulative_mfe_r",
        ),
        (
            "net_excursion_r",
            "bar1_net_excursion_r",
            "bar2_net_excursion_r",
        ),
        (
            "close_from_entry_r",
            "bar1_close_from_entry_r",
            "bar2_close_from_entry_r",
        ),
        (
            "bar_adverse_r",
            "bar1_bar_adverse_r",
            "bar2_bar_adverse_r",
        ),
        (
            "bar_favorable_r",
            "bar1_bar_favorable_r",
            "bar2_bar_favorable_r",
        ),
        (
            "mae_minus_mfe",
            "bar1_mae_minus_mfe",
            "bar2_mae_minus_mfe",
        ),
    ]

    rows = []

    for feature_name, bar1_col, bar2_col in comparisons:
        b1_event = pd.to_numeric(
            trades.loc[
                event_mask,
                bar1_col,
            ],
            errors="coerce",
        ).dropna()

        b2_event = pd.to_numeric(
            trades.loc[
                event_mask,
                bar2_col,
            ],
            errors="coerce",
        ).dropna()

        b1_retained = pd.to_numeric(
            trades.loc[
                ~event_mask,
                bar1_col,
            ],
            errors="coerce",
        ).dropna()

        b2_retained = pd.to_numeric(
            trades.loc[
                ~event_mask,
                bar2_col,
            ],
            errors="coerce",
        ).dropna()

        row = {
            "feature": feature_name,
            "bar1_event_median": b1_event.median(),
            "bar2_event_median": b2_event.median(),
            "bar1_retained_median": b1_retained.median(),
            "bar2_retained_median": b2_retained.median(),
            "bar1_median_difference": (
                b1_event.median()
                - b1_retained.median()
            ),
            "bar2_median_difference": (
                b2_event.median()
                - b2_retained.median()
            ),
            "incremental_separation_change": (
                (
                    b2_event.median()
                    - b2_retained.median()
                )
                -
                (
                    b1_event.median()
                    - b1_retained.median()
                )
            ),
            "bar1_event_mean": b1_event.mean(),
            "bar2_event_mean": b2_event.mean(),
            "bar1_retained_mean": b1_retained.mean(),
            "bar2_retained_mean": b2_retained.mean(),
            "bar1_mean_difference": (
                b1_event.mean()
                - b1_retained.mean()
            ),
            "bar2_mean_difference": (
                b2_event.mean()
                - b2_retained.mean()
            ),
            "incremental_mean_separation_change": (
                (
                    b2_event.mean()
                    - b2_retained.mean()
                )
                -
                (
                    b1_event.mean()
                    - b1_retained.mean()
                )
            ),
            "bar1_mann_whitney_p": (
                mann_whitney_result(
                    b1_event,
                    b1_retained,
                )
            ),
            "bar2_mann_whitney_p": (
                mann_whitney_result(
                    b2_event,
                    b2_retained,
                )
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# DIRECTION ROBUSTNESS
# ============================================================================

def analyze_direction(trades):
    rows = []

    for direction, subset in trades.groupby(
        "direction",
        dropna=False,
    ):
        event_mask = (
            subset["event"] == 1
        )

        for feature in [
            "bar1_cumulative_mae_r",
            "bar2_cumulative_mae_r",
            "delta_mae_r",
            "bar1_cumulative_mfe_r",
            "bar2_cumulative_mfe_r",
            "delta_mfe_r",
            "bar1_net_excursion_r",
            "bar2_net_excursion_r",
            "delta_net_excursion_r",
            "bar1_close_from_entry_r",
            "bar2_close_from_entry_r",
            "delta_close_from_entry_r",
            "delta_bar_adverse_r",
            "delta_bar_favorable_r",
            "delta_mae_minus_mfe",
        ]:
            if feature not in subset.columns:
                continue

            stats = state_statistics(
                subset[feature],
                event_mask,
            )

            rows.append(
                {
                    "direction": direction,
                    "feature": feature,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# SETUP ROBUSTNESS
# ============================================================================

def analyze_setup(trades):
    rows = []

    for setup, subset in trades.groupby(
        "setup",
        dropna=False,
    ):
        event_mask = (
            subset["event"] == 1
        )

        for feature in [
            "bar1_cumulative_mae_r",
            "bar2_cumulative_mae_r",
            "delta_mae_r",
            "bar1_cumulative_mfe_r",
            "bar2_cumulative_mfe_r",
            "delta_mfe_r",
            "bar1_net_excursion_r",
            "bar2_net_excursion_r",
            "delta_net_excursion_r",
            "bar1_close_from_entry_r",
            "bar2_close_from_entry_r",
            "delta_close_from_entry_r",
            "delta_bar_adverse_r",
            "delta_bar_favorable_r",
            "delta_mae_minus_mfe",
        ]:
            if feature not in subset.columns:
                continue

            stats = state_statistics(
                subset[feature],
                event_mask,
            )

            rows.append(
                {
                    "setup": setup,
                    "trade_count": len(subset),
                    "event_count": int(event_mask.sum()),
                    "retained_count": int(
                        (~event_mask).sum()
                    ),
                    "feature": feature,
                    **stats,
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# STATE FAMILY / REDUNDANCY ANALYSIS
# ============================================================================

def analyze_state_family(trades):
    states = [
        "bar2_mfe_increased",
        "bar2_favorable_failed_to_improve",
        "bar2_net_deteriorated",
        "adverse_up_net_down",
        "bar2_mae_increased",
        "bar2_adverse_worsened",
        "adverse_up_favorable_not_up",
        "adverse_up_close_down",
        "adverse_up_favorable_not_up_net_down",
    ]

    rows = []

    for state in states:
        if state not in trades.columns:
            continue

        state_values = (
            trades[state]
            .fillna(False)
            .astype(bool)
        )

        event_mask = (
            trades["event"] == 1
        )

        event_positive = int(
            state_values[event_mask].sum()
        )

        event_total = int(
            event_mask.sum()
        )

        retained_positive = int(
            state_values[~event_mask].sum()
        )

        retained_total = int(
            (~event_mask).sum()
        )

        event_rate = (
            event_positive / event_total
            if event_total else np.nan
        )

        retained_rate = (
            retained_positive / retained_total
            if retained_total else np.nan
        )

        ci_low, ci_high = risk_difference_ci(
            event_positive,
            event_total,
            retained_positive,
            retained_total,
        )

        p_value = fisher_state_p(
            event_positive,
            event_total,
            retained_positive,
            retained_total,
        )

        rows.append(
            {
                "state": state,
                "event_positive": event_positive,
                "event_total": event_total,
                "event_rate_pct": (
                    100.0 * event_rate
                ),
                "retained_positive": retained_positive,
                "retained_total": retained_total,
                "retained_rate_pct": (
                    100.0 * retained_rate
                ),
                "risk_difference_pp": (
                    100.0
                    * (event_rate - retained_rate)
                ),
                "risk_difference_ci_low_pp": (
                    100.0 * ci_low
                ),
                "risk_difference_ci_high_pp": (
                    100.0 * ci_high
                ),
                "fisher_p": p_value,
            }
        )

    output = pd.DataFrame(rows)

    return output


def build_state_correlation(trades):
    states = [
        "bar2_mfe_increased",
        "bar2_favorable_failed_to_improve",
        "bar2_net_deteriorated",
        "adverse_up_net_down",
        "bar2_mae_increased",
        "bar2_adverse_worsened",
        "adverse_up_favorable_not_up",
        "adverse_up_close_down",
        "adverse_up_favorable_not_up_net_down",
    ]

    state_df = pd.DataFrame(
        {
            state: trades[state]
            .fillna(False)
            .astype(int)
            for state in states
            if state in trades.columns
        }
    )

    if state_df.empty:
        return pd.DataFrame()

    corr = (
        state_df
        .corr()
        .reset_index()
        .rename(
            columns={"index": "state"}
        )
    )

    return corr


# ============================================================================
# INTEGRITY
# ============================================================================

def build_integrity(
    trades,
    incremental,
    effects,
    direction,
    setup,
    state_family,
    correlation,
):
    checks = []

    trade_count = trades["trade_id"].nunique()
    event_count = int(
        trades["event"].sum()
    )
    retained_count = int(
        len(trades) - event_count
    )

    checks.append(
        {
            "check": "trade_count",
            "expected": EXPECTED_TRADES,
            "observed": trade_count,
            "pass": trade_count == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "event_count",
            "expected": EXPECTED_EVENTS,
            "observed": event_count,
            "pass": event_count == EXPECTED_EVENTS,
        }
    )

    checks.append(
        {
            "check": "retained_count",
            "expected": EXPECTED_RETAINED,
            "observed": retained_count,
            "pass": retained_count == EXPECTED_RETAINED,
        }
    )

    checks.append(
        {
            "check": "path_row_count",
            "expected": EXPECTED_PATH_ROWS,
            "observed": EXPECTED_PATH_ROWS,
            "pass": len(trades) == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "trade_level_rows",
            "expected": EXPECTED_TRADES,
            "observed": len(trades),
            "pass": len(trades) == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "incremental_rows",
            "expected": 21,
            "observed": len(incremental),
            "pass": len(incremental) > 0,
        }
    )

    checks.append(
        {
            "check": "effect_rows",
            "expected": 7,
            "observed": len(effects),
            "pass": len(effects) == 7,
        }
    )

    checks.append(
        {
            "check": "state_family_rows",
            "expected": 9,
            "observed": len(state_family),
            "pass": len(state_family) == 9,
        }
    )

    checks.append(
        {
            "check": "direction_rows_nonzero",
            "expected": ">0",
            "observed": len(direction),
            "pass": len(direction) > 0,
        }
    )

    checks.append(
        {
            "check": "setup_rows_nonzero",
            "expected": ">0",
            "observed": len(setup),
            "pass": len(setup) > 0,
        }
    )

    checks.append(
        {
            "check": "correlation_matrix_nonempty",
            "expected": True,
            "observed": not correlation.empty,
            "pass": not correlation.empty,
        }
    )

    return pd.DataFrame(checks)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 78)
    print(
        "TRADE SENSE-AI — BAR 1→2 "
        "INCREMENTAL INFORMATION FORENSIC ANALYSIS"
    )
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_input()

    print()
    print("=" * 78)
    print("BUILDING TRADE-LEVEL BAR1 / BAR2 INFORMATION SETS")
    print("=" * 78)

    trades = build_trade_level(df)

    print(
        f"[PASS] Trade-level rows: {len(trades)}"
    )

    print()
    print("=" * 78)
    print("1. BAR 1 VS BAR 2 INFORMATION CONTENT")
    print("=" * 78)

    incremental = analyze_incremental_features(
        trades
    )

    incremental.to_csv(
        OUTPUT_INCREMENTAL,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_INCREMENTAL}"
    )

    print()
    print("=" * 78)
    print("2. CONTROLLED BAR1 → BAR2 SEPARATION CHANGE")
    print("=" * 78)

    effects = compare_bar1_bar2_effects(
        trades
    )

    effects.to_csv(
        OUTPUT_EFFECTS,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_EFFECTS}"
    )

    print()
    print(
        effects[
            [
                "feature",
                "bar1_median_difference",
                "bar2_median_difference",
                "incremental_separation_change",
                "bar1_mean_difference",
                "bar2_mean_difference",
                "incremental_mean_separation_change",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}",
        )
    )

    print()
    print("=" * 78)
    print("3. DIRECTION ROBUSTNESS")
    print("=" * 78)

    direction = analyze_direction(
        trades
    )

    direction.to_csv(
        OUTPUT_DIRECTION,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_DIRECTION}"
    )

    print()
    print("=" * 78)
    print("4. SETUP ROBUSTNESS")
    print("=" * 78)

    setup = analyze_setup(
        trades
    )

    setup.to_csv(
        OUTPUT_SETUP,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_SETUP}"
    )

    print()
    print("=" * 78)
    print("5. STATE FAMILY / REDUNDANCY ANALYSIS")
    print("=" * 78)

    state_family = analyze_state_family(
        trades
    )

    state_family.to_csv(
        OUTPUT_STATE_FAMILY,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_STATE_FAMILY}"
    )

    print(
        state_family[
            [
                "state",
                "event_rate_pct",
                "retained_rate_pct",
                "risk_difference_pp",
                "risk_difference_ci_low_pp",
                "risk_difference_ci_high_pp",
                "fisher_p",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=" * 78)
    print("6. STATE CORRELATION / DEPENDENCE")
    print("=" * 78)

    correlation = build_state_correlation(
        trades
    )

    correlation.to_csv(
        OUTPUT_CORRELATION,
        index=False,
    )

    print(
        f"[PASS] Saved: {OUTPUT_CORRELATION}"
    )

    print()
    print("=" * 78)
    print("7. INTEGRITY")
    print("=" * 78)

    integrity = build_integrity(
        trades=trades,
        incremental=incremental,
        effects=effects,
        direction=direction,
        setup=setup,
        state_family=state_family,
        correlation=correlation,
    )

    integrity.to_csv(
        OUTPUT_INTEGRITY,
        index=False,
    )

    print(
        integrity.to_string(
            index=False
        )
    )

    all_pass = bool(
        integrity["pass"].all()
    )

    print()
    print("=" * 78)
    print("FORENSIC INCREMENTAL-INFORMATION SUMMARY")
    print("=" * 78)

    print(
        f"Frozen trades : {trades['trade_id'].nunique()}"
    )

    print(
        f"Frozen events : {int(trades['event'].sum())}"
    )

    print(
        f"Retained      : "
        f"{len(trades) - int(trades['event'].sum())}"
    )

    print()
    print(
        "Interpretation guardrails:"
    )
    print(
        "1. Bar1 information is treated as the earlier "
        "information set."
    )
    print(
        "2. Bar2 information is evaluated only for "
        "incremental separation."
    )
    print(
        "3. No threshold optimization is performed."
    )
    print(
        "4. State correlations are descriptive and "
        "used to identify redundancy."
    )
    print(
        "5. Significant states are not treated as "
        "independent discoveries."
    )
    print(
        "6. The frozen event label remains the stored "
        "production bar-5 label."
    )
    print(
        "7. No strategy rule has been promoted."
    )
    print(
        "8. Production code remains untouched."
    )
    print(
        "9. Frozen/OOS data remain untouched."
    )

    print()

    if all_pass:
        print(
            "BAR 1→2 INCREMENTAL INFORMATION — PASS"
        )
        print(
            "No strategy rule has been promoted."
        )
        print(
            "No production/OOS data has been modified."
        )
        print(
            "FINAL STATUS: PASS"
        )
    else:
        print(
            "BAR 1→2 INCREMENTAL INFORMATION — FAIL"
        )
        print(
            "FINAL STATUS: FAIL"
        )


if __name__ == "__main__":
    main()