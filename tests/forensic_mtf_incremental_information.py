"""
TradeSense-AI
MTF Incremental Information Forensic Test

Purpose
-------
Determine whether Full MTF alignment adds information beyond:

    1. Direction
    2. Setup
    3. Frozen 5B / 0.25R early-adverse condition

This is a forensic research script only.

It MUST NOT modify:
- production strategy
- frozen trade ledger
- frozen hypothesis
- OOS logic
- entry logic
- exit logic
- risk logic
- MTF production implementation

Primary population
------------------
The 60 authoritative frozen 5B / 0.25R early-adverse events
identified by the completed MTF retention analysis.

Primary comparison
------------------
Within each frozen setup:

    FULL_MTF_ALIGNMENT
            vs
    NOT_FULL_MTF_ALIGNMENT

Metrics
-------
- N
- wins
- win rate
- total R
- average R
- median R
- average Bar-5 adverse excursion
- aligned minus not-aligned delta

Robustness
----------
- setup-stratified permutation test
- bootstrap 95% CI
- symbol leave-one-out
- year leave-one-out

Interpretation
--------------
This script is diagnostic.

A positive MTF effect does NOT automatically justify:
- production filtering
- frozen hypothesis modification
- OOS modification
- threshold tuning

No strategy changes are made here.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRADE_LEVEL_OUTPUT = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_7b025_retention_trade_level.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "mtf_incremental_information_detail.csv"
)

SETUP_OUTPUT = (
    OUTPUT_DIR
    / "mtf_incremental_information_setup_summary.csv"
)

LOO_SYMBOL_OUTPUT = (
    OUTPUT_DIR
    / "mtf_incremental_information_symbol_loo.csv"
)

LOO_YEAR_OUTPUT = (
    OUTPUT_DIR
    / "mtf_incremental_information_year_loo.csv"
)

RANDOM_SEED = 20260915

N_PERMUTATIONS = 20000
N_BOOTSTRAPS = 20000

SETUP_ORDER = [
    "LONG_CONTINUATION",
    "LONG_REVERSAL",
    "SHORT_CONTINUATION",
    "SHORT_REVERSAL",
]


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates, required=True):
    """
    Case-insensitive column resolver.
    """

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            "\nColumn not found.\n"
            f"Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def normalize_symbol(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def normalize_setup(value):
    if pd.isna(value):
        return "UNKNOWN"

    value = str(value).strip().upper()

    return value


def normalize_direction(value):
    if pd.isna(value):
        return "UNKNOWN"

    return (
        str(value)
        .strip()
        .upper()
    )


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def wins(series):
    return int(
        (series > 0).sum()
    )


def win_rate(series):
    if len(series) == 0:
        return np.nan

    return (
        (series > 0).mean()
        * 100.0
    )


def profit_factor(series):
    positive = series[series > 0].sum()
    negative = series[series < 0].sum()

    if negative == 0:
        if positive > 0:
            return np.inf

        return np.nan

    return positive / abs(negative)


def mean_or_nan(series):
    if len(series) == 0:
        return np.nan

    return float(series.mean())


def median_or_nan(series):
    if len(series) == 0:
        return np.nan

    return float(series.median())


def print_metric_row(label, series):
    print(
        f"{label:<24}"
        f"N={len(series):>3}  "
        f"W={wins(series):>3}  "
        f"WR={win_rate(series):>6.2f}%  "
        f"AvgR={mean_or_nan(series):>+7.4f}  "
        f"MedR={median_or_nan(series):>+7.4f}  "
        f"TotalR={series.sum():>+8.2f}"
    )


# ============================================================
# LOAD
# ============================================================

def load_data():
    print()
    print("=" * 78)
    print("LOADING COMPLETED MTF RETENTION LEDGER")
    print("=" * 78)

    if not TRADE_LEVEL_OUTPUT.exists():
        raise FileNotFoundError(
            "\nRequired completed retention output was not found:\n"
            f"{TRADE_LEVEL_OUTPUT}\n\n"
            "Run the already-completed MTF retention analysis first."
        )

    df = pd.read_csv(
        TRADE_LEVEL_OUTPUT
    )

    print(
        f"Rows loaded           : {len(df)}"
    )

    print(
        f"Columns loaded        : {len(df.columns)}"
    )

    print()
    print("Columns:")
    print(
        list(df.columns)
    )

    return df


# ============================================================
# STANDARDIZE ACTUAL OUTPUT
# ============================================================

def standardize(df):
    print()
    print("=" * 78)
    print("STANDARDIZING RETENTION LEDGER")
    print("=" * 78)

    symbol_col = find_column(
        df,
        [
            "_symbol_root",
            "Research_Symbol",
            "Symbol",
            "symbol",
            "_symbol",
        ],
    )

    date_col = find_column(
        df,
        [
            "Signal_Date",
            "Research_Date",
            "date",
            "Date",
        ],
    )

    direction_col = find_column(
        df,
        [
            "Direction_Normalized",
            "Direction",
            "direction",
        ],
    )

    setup_col = find_column(
        df,
        [
            "Setup",
            "setup",
        ],
    )

    r_col = find_column(
        df,
        [
            "R_Multiple",
            "r_multiple",
        ],
    )

    bar5_col = find_column(
        df,
        [
            "Bar5_Early_Adverse_R",
            "early_adverse_r_bar_5",
        ],
    )

    full_mtf_col = find_column(
        df,
        [
            "Full_MTF_Aligned_Flag",
            "Full_MTF_Aligned",
        ],
    )

    early_col = find_column(
        df,
        [
            "Early_Adverse",
        ],
    )

    out = pd.DataFrame()

    out["Symbol"] = (
        df[symbol_col]
        .map(normalize_symbol)
    )

    out["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    out["Direction"] = (
        df[direction_col]
        .map(normalize_direction)
    )

    out["Setup"] = (
        df[setup_col]
        .map(normalize_setup)
    )

    out["R_Multiple"] = safe_numeric(
        df[r_col]
    )

    out["Bar5_Early_Adverse_R"] = safe_numeric(
        df[bar5_col]
    )

    out["Full_MTF_Aligned"] = (
        df[full_mtf_col]
        .astype(bool)
    )

    out["Early_Adverse"] = (
        df[early_col]
        .astype(bool)
    )

    out["Year"] = (
        out["Date"]
        .dt.year
        .astype("Int64")
    )

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    required_setups = set(
        SETUP_ORDER
    )

    present_setups = set(
        out["Setup"].dropna().unique()
    )

    print()
    print(
        "Setup values found:"
    )

    for setup in sorted(
        present_setups
    ):
        print(
            f"  {setup}"
        )

    unexpected = (
        present_setups
        - required_setups
    )

    if unexpected:
        print()
        print(
            "WARNING: unexpected setup values:"
        )

        for value in sorted(
            unexpected
        ):
            print(
                f"  {value}"
            )

    if out["Symbol"].isna().any():
        raise RuntimeError(
            "Invalid symbol values detected."
        )

    if out["Date"].isna().any():
        raise RuntimeError(
            "Invalid date values detected."
        )

    if out["R_Multiple"].isna().any():
        raise RuntimeError(
            "Invalid R values detected."
        )

    if out["Bar5_Early_Adverse_R"].isna().all():
        raise RuntimeError(
            "Bar-5 early-adverse field is completely missing."
        )

    return out


# ============================================================
# IDENTIFY AUTHORITATIVE EARLY-ADVERSE EVENTS
# ============================================================

def select_events(df):
    print()
    print("=" * 78)
    print("AUTHORITATIVE 5B / 0.25R EVENT POPULATION")
    print("=" * 78)

    # --------------------------------------------------------
    # Frozen authoritative rule
    #
    # Reject only when:
    #
    #     Bar-5 adverse excursion >= 0.25R
    #
    # Missing Bar-5 remains non-event.
    # --------------------------------------------------------

    event_mask = (
        df["Bar5_Early_Adverse_R"].notna()
        & (
            df["Bar5_Early_Adverse_R"]
            >= 0.25
        )
    )

    derived_count = int(
        event_mask.sum()
    )

    ledger_count = int(
        df["Early_Adverse"].sum()
    )

    print(
        f"Derived 5B/.25R events : "
        f"{derived_count}"
    )

    print(
        f"Retention ledger events : "
        f"{ledger_count}"
    )

    if derived_count != ledger_count:
        raise RuntimeError(
            "\nAuthoritative event mismatch.\n"
            f"Derived={derived_count}, "
            f"Ledger={ledger_count}"
        )

    events = (
        df.loc[
            event_mask
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(events) != 60:
        raise RuntimeError(
            "\nExpected 60 authoritative early-adverse events "
            "from the completed retention analysis, but found "
            f"{len(events)}."
        )

    print(
        "Authoritative event count: PASS (60)"
    )

    print()
    print(
        "Event population by setup:"
    )

    setup_counts = (
        events["Setup"]
        .value_counts()
        .reindex(
            SETUP_ORDER,
            fill_value=0,
        )
    )

    for setup, count in setup_counts.items():
        print(
            f"  {setup:<22} {count}"
        )

    return events


# ============================================================
# SETUP-LEVEL SUMMARY
# ============================================================

def setup_summary(events):
    print()
    print("=" * 78)
    print("MTF INCREMENTAL INFORMATION BY SETUP")
    print("=" * 78)

    rows = []

    for setup in SETUP_ORDER:

        subset = events.loc[
            events["Setup"] == setup
        ].copy()

        aligned = subset.loc[
            subset["Full_MTF_Aligned"]
        ]

        not_aligned = subset.loc[
            ~subset["Full_MTF_Aligned"]
        ]

        aligned_r = aligned[
            "R_Multiple"
        ]

        not_aligned_r = not_aligned[
            "R_Multiple"
        ]

        aligned_avg_adverse = (
            aligned[
                "Bar5_Early_Adverse_R"
            ].mean()
            if len(aligned)
            else np.nan
        )

        not_aligned_avg_adverse = (
            not_aligned[
                "Bar5_Early_Adverse_R"
            ].mean()
            if len(not_aligned)
            else np.nan
        )

        aligned_avg_r = (
            aligned_r.mean()
            if len(aligned_r)
            else np.nan
        )

        not_aligned_avg_r = (
            not_aligned_r.mean()
            if len(not_aligned_r)
            else np.nan
        )

        delta_avg_r = (
            aligned_avg_r
            - not_aligned_avg_r
        )

        print()
        print(
            f"SETUP: {setup}"
        )

        print(
            "-" * 78
        )

        print_metric_row(
            "FULL MTF ALIGNED",
            aligned_r,
        )

        print_metric_row(
            "NOT ALIGNED",
            not_aligned_r,
        )

        print()
        print(
            f"Aligned Avg Bar-5 adverse     : "
            f"{aligned_avg_adverse:+.4f}R"
        )

        print(
            f"Not aligned Avg Bar-5 adverse : "
            f"{not_aligned_avg_adverse:+.4f}R"
        )

        print(
            f"Aligned - Not aligned Avg R   : "
            f"{delta_avg_r:+.4f}R/trade"
        )

        rows.append(
            {
                "Setup": setup,
                "Aligned_N": len(aligned),
                "Aligned_Wins": wins(
                    aligned_r
                ),
                "Aligned_Win_Rate": win_rate(
                    aligned_r
                ),
                "Aligned_Total_R": aligned_r.sum(),
                "Aligned_Avg_R": aligned_avg_r,
                "Aligned_Median_R": median_or_nan(
                    aligned_r
                ),
                "Aligned_Profit_Factor": profit_factor(
                    aligned_r
                ),
                "Aligned_Avg_Bar5_Adverse_R": aligned_avg_adverse,
                "Not_Aligned_N": len(not_aligned),
                "Not_Aligned_Wins": wins(
                    not_aligned_r
                ),
                "Not_Aligned_Win_Rate": win_rate(
                    not_aligned_r
                ),
                "Not_Aligned_Total_R": not_aligned_r.sum(),
                "Not_Aligned_Avg_R": not_aligned_avg_r,
                "Not_Aligned_Median_R": median_or_nan(
                    not_aligned_r
                ),
                "Not_Aligned_Profit_Factor": profit_factor(
                    not_aligned_r
                ),
                "Not_Aligned_Avg_Bar5_Adverse_R": not_aligned_avg_adverse,
                "Delta_Avg_R": delta_avg_r,
                "Delta_Total_R": (
                    aligned_r.sum()
                    - not_aligned_r.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# OVERALL EFFECT
# ============================================================

def overall_effect(events):
    print()
    print("=" * 78)
    print("OVERALL MTF EFFECT ACROSS 60 EVENTS")
    print("=" * 78)

    aligned = events.loc[
        events["Full_MTF_Aligned"],
        "R_Multiple",
    ]

    not_aligned = events.loc[
        ~events["Full_MTF_Aligned"],
        "R_Multiple",
    ]

    print_metric_row(
        "FULL MTF ALIGNED",
        aligned,
    )

    print_metric_row(
        "NOT ALIGNED",
        not_aligned,
    )

    aligned_avg = aligned.mean()
    not_aligned_avg = not_aligned.mean()

    delta = (
        aligned_avg
        - not_aligned_avg
    )

    print()
    print(
        f"Aligned - Not aligned Δ Avg R : "
        f"{delta:+.4f}R/trade"
    )

    print(
        f"Aligned - Not aligned Δ Total R : "
        f"{aligned.sum() - not_aligned.sum():+.2f}R"
    )

    return float(delta)


# ============================================================
# SETUP-STRATIFIED PERMUTATION TEST
# ============================================================

def stratified_permutation_test(events):
    print()
    print("=" * 78)
    print("SETUP-STRATIFIED PERMUTATION TEST")
    print("=" * 78)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    observed_parts = []

    groups = []

    for setup in SETUP_ORDER:

        subset = events.loc[
            events["Setup"] == setup
        ].copy()

        if len(subset) == 0:
            continue

        aligned_values = subset.loc[
            subset["Full_MTF_Aligned"],
            "R_Multiple",
        ].to_numpy(
            dtype=float
        )

        not_aligned_values = subset.loc[
            ~subset["Full_MTF_Aligned"],
            "R_Multiple",
        ].to_numpy(
            dtype=float
        )

        if (
            len(aligned_values) == 0
            or len(not_aligned_values) == 0
        ):
            print(
                f"{setup}: skipped "
                "(one MTF group empty)"
            )
            continue

        delta = (
            aligned_values.mean()
            - not_aligned_values.mean()
        )

        observed_parts.append(
            delta
        )

        groups.append(
            {
                "setup": setup,
                "values": subset[
                    "R_Multiple"
                ].to_numpy(
                    dtype=float
                ),
                "n_aligned": len(
                    aligned_values
                ),
            }
        )

    if not groups:
        raise RuntimeError(
            "No setup had both MTF groups."
        )

    observed = float(
        np.mean(
            observed_parts
        )
    )

    null_distribution = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    for i in range(
        N_PERMUTATIONS
    ):

        permutation_deltas = []

        for group in groups:

            values = group["values"]

            shuffled = rng.permutation(
                values
            )

            n_aligned = group[
                "n_aligned"
            ]

            perm_aligned = (
                shuffled[:n_aligned]
            )

            perm_not_aligned = (
                shuffled[n_aligned:]
            )

            permutation_deltas.append(
                perm_aligned.mean()
                - perm_not_aligned.mean()
            )

        null_distribution[i] = np.mean(
            permutation_deltas
        )

    p_value = (
        np.mean(
            np.abs(
                null_distribution
            )
            >= abs(observed)
        )
    )

    print(
        f"Observed setup-stratified Δ : "
        f"{observed:+.6f}R/trade"
    )

    print(
        f"Permutations                 : "
        f"{N_PERMUTATIONS}"
    )

    print(
        f"Two-sided permutation p      : "
        f"{p_value:.6f}"
    )

    return {
        "observed_delta": observed,
        "permutation_p_value": float(
            p_value
        ),
    }


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_stratified(events):
    print()
    print("=" * 78)
    print("SETUP-STRATIFIED BOOTSTRAP")
    print("=" * 78)

    rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    groups = []

    observed_parts = []

    for setup in SETUP_ORDER:

        subset = events.loc[
            events["Setup"] == setup
        ].copy()

        aligned = subset.loc[
            subset["Full_MTF_Aligned"],
            "R_Multiple",
        ].to_numpy(
            dtype=float
        )

        not_aligned = subset.loc[
            ~subset["Full_MTF_Aligned"],
            "R_Multiple",
        ].to_numpy(
            dtype=float
        )

        if (
            len(aligned) == 0
            or len(not_aligned) == 0
        ):
            continue

        observed_parts.append(
            aligned.mean()
            - not_aligned.mean()
        )

        groups.append(
            (
                setup,
                aligned,
                not_aligned,
            )
        )

    if not groups:
        raise RuntimeError(
            "No bootstrap groups available."
        )

    observed = float(
        np.mean(
            observed_parts
        )
    )

    boot = np.empty(
        N_BOOTSTRAPS,
        dtype=float,
    )

    for i in range(
        N_BOOTSTRAPS
    ):

        deltas = []

        for (
            setup,
            aligned,
            not_aligned,
        ) in groups:

            aligned_sample = rng.choice(
                aligned,
                size=len(aligned),
                replace=True,
            )

            not_aligned_sample = rng.choice(
                not_aligned,
                size=len(not_aligned),
                replace=True,
            )

            deltas.append(
                aligned_sample.mean()
                - not_aligned_sample.mean()
            )

        boot[i] = np.mean(
            deltas
        )

    lower = float(
        np.percentile(
            boot,
            2.5
        )
    )

    upper = float(
        np.percentile(
            boot,
            97.5
        )
    )

    print(
        f"Observed Δ                 : "
        f"{observed:+.6f}R/trade"
    )

    print(
        f"Bootstrap samples           : "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"95% bootstrap CI            : "
        f"[{lower:+.6f}, {upper:+.6f}]R/trade"
    )

    return {
        "bootstrap_delta": observed,
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
    }


# ============================================================
# SYMBOL LEAVE-ONE-OUT
# ============================================================

def symbol_loo(events):
    print()
    print("=" * 78)
    print("SYMBOL LEAVE-ONE-OUT")
    print("=" * 78)

    rows = []

    symbols = sorted(
        events["Symbol"]
        .dropna()
        .unique()
    )

    for symbol in symbols:

        subset = events.loc[
            events["Symbol"] != symbol
        ]

        aligned = subset.loc[
            subset["Full_MTF_Aligned"],
            "R_Multiple",
        ]

        not_aligned = subset.loc[
            ~subset["Full_MTF_Aligned"],
            "R_Multiple",
        ]

        if (
            len(aligned) == 0
            or len(not_aligned) == 0
        ):
            continue

        delta = (
            aligned.mean()
            - not_aligned.mean()
        )

        rows.append(
            {
                "Excluded_Symbol": symbol,
                "Remaining_N": len(subset),
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Avg_R": aligned.mean(),
                "Not_Aligned_Avg_R": not_aligned.mean(),
                "Delta_Avg_R": delta,
                "Aligned_Total_R": aligned.sum(),
                "Not_Aligned_Total_R": not_aligned.sum(),
            }
        )

        print(
            f"{symbol:<12} "
            f"Δ={delta:+.4f}R/trade"
        )

    return pd.DataFrame(rows)


# ============================================================
# YEAR LEAVE-ONE-OUT
# ============================================================

def year_loo(events):
    print()
    print("=" * 78)
    print("YEAR LEAVE-ONE-OUT")
    print("=" * 78)

    rows = []

    years = sorted(
        events["Year"]
        .dropna()
        .unique()
    )

    for year in years:

        subset = events.loc[
            events["Year"] != year
        ]

        aligned = subset.loc[
            subset["Full_MTF_Aligned"],
            "R_Multiple",
        ]

        not_aligned = subset.loc[
            ~subset["Full_MTF_Aligned"],
            "R_Multiple",
        ]

        if (
            len(aligned) == 0
            or len(not_aligned) == 0
        ):
            continue

        delta = (
            aligned.mean()
            - not_aligned.mean()
        )

        rows.append(
            {
                "Excluded_Year": int(year),
                "Remaining_N": len(subset),
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Avg_R": aligned.mean(),
                "Not_Aligned_Avg_R": not_aligned.mean(),
                "Delta_Avg_R": delta,
                "Aligned_Total_R": aligned.sum(),
                "Not_Aligned_Total_R": not_aligned.sum(),
            }
        )

        print(
            f"{int(year)} "
            f"Δ={delta:+.4f}R/trade"
        )

    return pd.DataFrame(rows)


# ============================================================
# DIRECTION × SETUP DIAGNOSTIC
# ============================================================

def direction_setup_diagnostic(events):
    print()
    print("=" * 78)
    print("DIRECTION × SETUP × MTF DIAGNOSTIC")
    print("=" * 78)

    rows = []

    grouped = (
        events
        .groupby(
            [
                "Direction",
                "Setup",
                "Full_MTF_Aligned",
            ],
            dropna=False,
        )
    )

    for (
        direction,
        setup,
        aligned_flag,
    ), group in grouped:

        r = group[
            "R_Multiple"
        ]

        rows.append(
            {
                "Direction": direction,
                "Setup": setup,
                "Full_MTF_Aligned": bool(
                    aligned_flag
                ),
                "N": len(group),
                "Wins": wins(r),
                "Win_Rate": win_rate(r),
                "Total_R": r.sum(),
                "Avg_R": r.mean(),
                "Median_R": r.median(),
                "Avg_Bar5_Adverse_R": group[
                    "Bar5_Early_Adverse_R"
                ].mean(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result):
        print(
            result.to_string(
                index=False
            )
        )

    return result


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    events,
    setup_df,
    symbol_df,
    year_df,
    direction_df,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    setup_df.to_csv(
        SETUP_OUTPUT,
        index=False,
    )

    symbol_df.to_csv(
        LOO_SYMBOL_OUTPUT,
        index=False,
    )

    year_df.to_csv(
        LOO_YEAR_OUTPUT,
        index=False,
    )

    direction_df.to_csv(
        OUTPUT_DIR
        / "mtf_incremental_information_direction_setup.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("OUTPUTS SAVED")
    print("=" * 78)

    print(
        DETAIL_OUTPUT
    )

    print(
        SETUP_OUTPUT
    )

    print(
        LOO_SYMBOL_OUTPUT
    )

    print(
        LOO_YEAR_OUTPUT
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

def final_interpretation(
    delta,
    permutation,
    bootstrap,
    setup_df,
):
    print()
    print("=" * 78)
    print("FINAL FORENSIC INTERPRETATION")
    print("=" * 78)

    lower = bootstrap[
        "bootstrap_ci_lower"
    ]

    upper = bootstrap[
        "bootstrap_ci_upper"
    ]

    p_value = permutation[
        "permutation_p_value"
    ]

    positive_setup_count = int(
        (
            setup_df[
                "Delta_Avg_R"
            ]
            > 0
        ).sum()
    )

    total_setup_count = len(
        setup_df
    )

    print()
    print(
        f"Overall MTF Δ Avg R/trade : "
        f"{delta:+.4f}R"
    )

    print(
        f"Permutation p-value       : "
        f"{p_value:.6f}"
    )

    print(
        f"Bootstrap 95% CI          : "
        f"[{lower:+.4f}, {upper:+.4f}]R"
    )

    print(
        f"Setups with positive Δ    : "
        f"{positive_setup_count}/"
        f"{total_setup_count}"
    )

    print()

    # --------------------------------------------------------
    # Conservative interpretation
    # --------------------------------------------------------

    if (
        delta > 0
        and lower > 0
        and p_value < 0.05
        and positive_setup_count
        == total_setup_count
    ):
        print(
            "RESULT: STRONGER INCREMENTAL EVIDENCE"
        )

        print(
            "Full MTF alignment shows a positive "
            "setup-stratified effect with the "
            "bootstrap interval above zero and "
            "permutation evidence below 0.05."
        )

        print(
            "This remains forensic evidence only."
        )

    elif (
        delta > 0
        and lower > 0
    ):
        print(
            "RESULT: POSITIVE BUT LIMITED EVIDENCE"
        )

        print(
            "The observed MTF effect is positive "
            "and the bootstrap interval is above "
            "zero, but the evidence is not strong "
            "enough by itself to justify a strategy "
            "change."
        )

    elif delta > 0:
        print(
            "RESULT: PROMISING BUT NOT CONFIRMED"
        )

        print(
            "Full MTF alignment improves average "
            "R within the event population, but "
            "the statistical/robustness evidence "
            "does not establish an independent edge."
        )

    else:
        print(
            "RESULT: NO POSITIVE INCREMENTAL MTF EDGE"
        )

        print(
            "Full MTF alignment does not improve "
            "the frozen early-adverse event "
            "population on average."
        )

    print()
    print(
        "NO PRODUCTION CHANGE."
    )

    print(
        "NO FROZEN HYPOTHESIS CHANGE."
    )

    print(
        "NO OOS CHANGE."
    )

    print(
        "NO THRESHOLD TUNING."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("TRADESENSE-AI")
    print("MTF INCREMENTAL INFORMATION FORENSIC TEST")
    print("=" * 78)

    print()
    print(
        "Frozen condition:"
    )

    print(
        "    Bar-5 adverse excursion >= 0.25R"
    )

    print()
    print(
        "Primary question:"
    )

    print(
        "    Does Full MTF alignment add information "
        "beyond Direction + Setup + frozen early-adverse?"
    )

    df = load_data()

    df = standardize(
        df
    )

    events = select_events(
        df
    )

    setup_df = setup_summary(
        events
    )

    delta = overall_effect(
        events
    )

    permutation = (
        stratified_permutation_test(
            events
        )
    )

    bootstrap = (
        bootstrap_stratified(
            events
        )
    )

    symbol_df = symbol_loo(
        events
    )

    year_df = year_loo(
        events
    )

    direction_df = (
        direction_setup_diagnostic(
            events
        )
    )

    save_outputs(
        events,
        setup_df,
        symbol_df,
        year_df,
        direction_df,
    )

    final_interpretation(
        delta,
        permutation,
        bootstrap,
        setup_df,
    )

    print()
    print("=" * 78)
    print("MTF INCREMENTAL INFORMATION TEST COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()