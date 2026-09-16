from __future__ import annotations

from pathlib import Path
import random
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FROZEN_PATH = ROOT / "tests/output/tradesense_trades.csv"

MTF_PATH = (
    ROOT
    / "tests/output/reconciliation/mtf_daily_weekly/"
    / "mtf_pre_entry_timing_trade_level.csv"
)

OUTPUT_DIR = ROOT / "tests/output/reconciliation/mtf_daily_weekly"

TRADE_OUTPUT = OUTPUT_DIR / "mtf_early_adverse_prediction_trade_level.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mtf_early_adverse_prediction_summary.csv"


# ============================================================
# FROZEN CONSTANTS
# ============================================================

EARLY_ADVERSE_BARS = 7
EARLY_ADVERSE_THRESHOLD_R = 0.25

N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 10_000

RNG_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def normalize_direction(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if value in {"LONG", "BUY"}:
        return "LONG"

    if value in {"SHORT", "SELL"}:
        return "SHORT"

    return value


def normalize_binary_event(series):
    """
    Convert an early-adverse indicator into 0/1.

    Accepted representations:
    TRUE/FALSE
    1/0
    YES/NO
    TRIGGERED/NOT_TRIGGERED
    """

    result = []

    for value in series:
        if pd.isna(value):
            result.append(np.nan)
            continue

        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
            continue

        text = str(value).strip().upper()

        if text in {
            "TRUE",
            "1",
            "YES",
            "Y",
            "TRIGGERED",
            "EVENT",
        }:
            result.append(1)
        elif text in {
            "FALSE",
            "0",
            "NO",
            "N",
            "NOT_TRIGGERED",
            "NO_EVENT",
        }:
            result.append(0)
        else:
            try:
                result.append(int(float(value)))
            except Exception:
                result.append(np.nan)

    return pd.Series(result, index=series.index, dtype="float64")


def safe_mean(values):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def event_rate(values):
    values = pd.Series(values).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def pct(value):
    if pd.isna(value):
        return "NA"

    return f"{value * 100:.2f}%"


def fmt(value):
    if pd.isna(value):
        return "NA"

    return f"{value:.4f}"


def print_header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# DATA LOADING
# ============================================================

def load_frozen():
    df = pd.read_csv(FROZEN_PATH)

    required = [
        "_symbol",
        "Signal_Date",
        "direction",
        "r_multiple",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"Frozen ledger missing required columns: {missing}"
        )

    df = df.copy()

    df["Symbol_Normalized"] = df["_symbol"].map(normalize_symbol)
    df["Signal_Date"] = pd.to_datetime(
        df["Signal_Date"],
        errors="coerce",
    )
    df["Direction_Normalized"] = df["direction"].map(
        normalize_direction
    )

    df["r_multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    )

    return df


def load_mtf():
    df = pd.read_csv(MTF_PATH)

    required = [
        "_symbol",
        "Signal_Date",
        "direction",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Pre_Entry_MTF_Score",
        "Early_Adverse_Group",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"MTF file missing required columns: {missing}"
        )

    df = df.copy()

    df["Symbol_Normalized"] = df["_symbol"].map(normalize_symbol)

    df["Signal_Date"] = pd.to_datetime(
        df["Signal_Date"],
        errors="coerce",
    )

    df["Direction_Normalized"] = df["direction"].map(
        normalize_direction
    )

    df["Pre_Entry_MTF_Score"] = pd.to_numeric(
        df["Pre_Entry_MTF_Score"],
        errors="coerce",
    )

    return df


# ============================================================
# KEY / MERGE
# ============================================================

def add_trade_key(df):
    df = df.copy()

    df["Trade_Key"] = (
        df["Symbol_Normalized"].astype(str)
        + "|"
        + df["Signal_Date"].dt.strftime("%Y-%m-%d")
        + "|"
        + df["Direction_Normalized"].astype(str)
    )

    return df


def reconcile_and_merge(frozen, mtf):
    frozen = add_trade_key(frozen)
    mtf = add_trade_key(mtf)

    frozen_duplicates = int(
        frozen["Trade_Key"].duplicated().sum()
    )

    mtf_duplicates = int(
        mtf["Trade_Key"].duplicated().sum()
    )

    print(f"Frozen duplicate keys : {frozen_duplicates}")
    print(f"MTF duplicate keys    : {mtf_duplicates}")

    if frozen_duplicates:
        raise RuntimeError(
            "Frozen ledger contains duplicate Trade_Key values."
        )

    if mtf_duplicates:
        raise RuntimeError(
            "MTF ledger contains duplicate Trade_Key values."
        )

    mtf_keep = [
        "Trade_Key",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Pre_Entry_MTF_Score",
        "Early_Adverse_Group",
    ]

    merged = frozen.merge(
        mtf[mtf_keep],
        on="Trade_Key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_MTF"),
    )

    return merged


# ============================================================
# EARLY ADVERSE EVENT
# ============================================================

def derive_early_adverse_event(df):
    """
    Primary authoritative event indicator comes from the already
    classified MTF forensic ledger.

    We deliberately do NOT reconstruct the event from final R.

    Early_Adverse_Group is already generated under the frozen
    7B / 0.25R hypothesis.
    """

    df = df.copy()

    raw = df["Early_Adverse_Group"].astype(str).str.upper()

    df["Early_Adverse_Event"] = np.where(
        raw.eq("TRIGGERED"),
        1,
        np.where(
            raw.eq("NOT_TRIGGERED"),
            0,
            np.nan,
        ),
    )

    df["Early_Adverse_Event"] = pd.to_numeric(
        df["Early_Adverse_Event"],
        errors="coerce",
    )

    return df


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize_group(df, group_columns):
    rows = []

    grouped = df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for group_key, g in grouped:

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {}

        for col, value in zip(group_columns, group_key):
            row[col] = value

        n = len(g)

        valid_event = g["Early_Adverse_Event"].dropna()

        events = int(valid_event.sum()) if len(valid_event) else 0

        rate = (
            float(valid_event.mean())
            if len(valid_event)
            else np.nan
        )

        row["N"] = n
        row["Event_N"] = events
        row["Event_Rate"] = rate
        row["Non_Event_N"] = (
            int((valid_event == 0).sum())
            if len(valid_event)
            else 0
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# TWO-GROUP EFFECT
# ============================================================

def two_group_effect(df, group_column, positive_group):
    """
    Compare event probability:

        positive_group
        vs everything else

    Effect = P(event | positive) - P(event | other)
    """

    work = df[
        df[group_column].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    positive = work[
        work[group_column].astype(str) == str(positive_group)
    ]["Early_Adverse_Event"].to_numpy()

    other = work[
        work[group_column].astype(str) != str(positive_group)
    ]["Early_Adverse_Event"].to_numpy()

    if len(positive) == 0 or len(other) == 0:
        return {
            "Group_Column": group_column,
            "Positive_Group": positive_group,
            "N_Positive": len(positive),
            "N_Other": len(other),
            "Positive_Event_Rate": np.nan,
            "Other_Event_Rate": np.nan,
            "Event_Rate_Difference": np.nan,
            "Risk_Ratio": np.nan,
        }

    p1 = float(np.mean(positive))
    p0 = float(np.mean(other))

    rr = (
        p1 / p0
        if p0 > 0
        else np.nan
    )

    return {
        "Group_Column": group_column,
        "Positive_Group": positive_group,
        "N_Positive": len(positive),
        "N_Other": len(other),
        "Positive_Event_Rate": p1,
        "Other_Event_Rate": p0,
        "Event_Rate_Difference": p1 - p0,
        "Risk_Ratio": rr,
    }


# ============================================================
# MTF SCORE ANALYSIS
# ============================================================

def score_analysis(df):
    work = df[
        df["Pre_Entry_MTF_Score"].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    rows = []

    for score, g in work.groupby(
        "Pre_Entry_MTF_Score",
        sort=True,
    ):
        events = int(g["Early_Adverse_Event"].sum())
        n = len(g)

        rows.append(
            {
                "Score": score,
                "N": n,
                "Event_N": events,
                "Event_Rate": (
                    events / n
                    if n
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DIRECTION ANALYSIS
# ============================================================

def direction_effects(df):
    rows = []

    for direction in ["LONG", "SHORT"]:

        subset = df[
            (df["Direction_Normalized"] == direction)
            & df["Early_Adverse_Event"].notna()
            & df["Pre_Entry_MTF_State"].notna()
        ].copy()

        aligned = subset[
            subset["Pre_Entry_MTF_State"] == "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        not_aligned = subset[
            subset["Pre_Entry_MTF_State"] != "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        if len(aligned) == 0 or len(not_aligned) == 0:
            continue

        p1 = float(np.mean(aligned))
        p0 = float(np.mean(not_aligned))

        rows.append(
            {
                "Direction": direction,
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Event_Rate": p1,
                "Not_Aligned_Event_Rate": p0,
                "Event_Rate_Difference": p1 - p0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SETUP ANALYSIS
# ============================================================

def setup_effects(df):
    if "setup" not in df.columns:
        return pd.DataFrame()

    rows = []

    for setup in sorted(
        df["setup"].dropna().astype(str).unique()
    ):

        subset = df[
            (df["setup"].astype(str) == setup)
            & df["Early_Adverse_Event"].notna()
            & df["Pre_Entry_MTF_State"].notna()
        ]

        aligned = subset[
            subset["Pre_Entry_MTF_State"] == "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        not_aligned = subset[
            subset["Pre_Entry_MTF_State"] != "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        if len(aligned) < 3 or len(not_aligned) < 3:
            continue

        p1 = float(np.mean(aligned))
        p0 = float(np.mean(not_aligned))

        rows.append(
            {
                "Setup": setup,
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Event_Rate": p1,
                "Not_Aligned_Event_Rate": p0,
                "Event_Rate_Difference": p1 - p0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CONFLICT ANALYSIS
# ============================================================

def conflict_analysis(df):
    work = df[
        df["Pre_Entry_MTF_State"].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    rows = []

    for state in [
        "ALIGNED",
        "CONFLICT",
        "NEUTRAL",
        "UNAVAILABLE",
    ]:

        g = work[
            work["Pre_Entry_MTF_State"] == state
        ]

        if len(g) == 0:
            continue

        events = int(g["Early_Adverse_Event"].sum())

        rows.append(
            {
                "MTF_State": state,
                "N": len(g),
                "Event_N": events,
                "Event_Rate": events / len(g),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SYMBOL / YEAR ROBUSTNESS
# ============================================================

def robustness_by_symbol(df):
    work = df[
        df["Pre_Entry_MTF_State"].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    rows = []

    for symbol in sorted(
        work["Symbol_Normalized"].dropna().unique()
    ):

        g = work[
            work["Symbol_Normalized"] == symbol
        ]

        aligned = g[
            g["Pre_Entry_MTF_State"] == "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        not_aligned = g[
            g["Pre_Entry_MTF_State"] != "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        if len(aligned) < 3 or len(not_aligned) < 3:
            continue

        p1 = float(np.mean(aligned))
        p0 = float(np.mean(not_aligned))

        rows.append(
            {
                "Symbol": symbol,
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Event_Rate": p1,
                "Not_Aligned_Event_Rate": p0,
                "Event_Rate_Difference": p1 - p0,
            }
        )

    return pd.DataFrame(rows)


def robustness_by_year(df):
    work = df[
        df["Pre_Entry_MTF_State"].notna()
        & df["Early_Adverse_Event"].notna()
        & df["Signal_Date"].notna()
    ].copy()

    work["Signal_Year"] = work["Signal_Date"].dt.year

    rows = []

    for year in sorted(work["Signal_Year"].dropna().unique()):

        g = work[
            work["Signal_Year"] == year
        ]

        aligned = g[
            g["Pre_Entry_MTF_State"] == "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        not_aligned = g[
            g["Pre_Entry_MTF_State"] != "ALIGNED"
        ]["Early_Adverse_Event"].to_numpy()

        if len(aligned) < 3 or len(not_aligned) < 3:
            continue

        p1 = float(np.mean(aligned))
        p0 = float(np.mean(not_aligned))

        rows.append(
            {
                "Year": int(year),
                "Aligned_N": len(aligned),
                "Not_Aligned_N": len(not_aligned),
                "Aligned_Event_Rate": p1,
                "Not_Aligned_Event_Rate": p0,
                "Event_Rate_Difference": p1 - p0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test_aligned_event_rate(
    df,
    n_permutations=N_PERMUTATIONS,
    seed=RNG_SEED,
):
    """
    Null hypothesis:

        MTF alignment has no relationship with
        early-adverse event occurrence.

    Preserve:
      - number of aligned observations
      - number of non-aligned observations
      - total number of events

    Randomly permute event labels.

    Two-sided empirical p-value.
    """

    work = df[
        df["Pre_Entry_MTF_State"].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    aligned_mask = (
        work["Pre_Entry_MTF_State"] == "ALIGNED"
    ).to_numpy()

    events = work["Early_Adverse_Event"].to_numpy()

    observed = (
        events[aligned_mask].mean()
        - events[~aligned_mask].mean()
    )

    rng = np.random.default_rng(seed)

    count = 0

    for _ in range(n_permutations):

        shuffled = rng.permutation(events)

        stat = (
            shuffled[aligned_mask].mean()
            - shuffled[~aligned_mask].mean()
        )

        if abs(stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return observed, p_value


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_ci_aligned_event_rate(
    df,
    n_bootstraps=N_BOOTSTRAPS,
    seed=RNG_SEED,
):
    """
    Bootstrap the difference:

        event rate aligned
        -
        event rate not aligned
    """

    work = df[
        df["Pre_Entry_MTF_State"].notna()
        & df["Early_Adverse_Event"].notna()
    ].copy()

    aligned = work[
        work["Pre_Entry_MTF_State"] == "ALIGNED"
    ]["Early_Adverse_Event"].to_numpy()

    not_aligned = work[
        work["Pre_Entry_MTF_State"] != "ALIGNED"
    ]["Early_Adverse_Event"].to_numpy()

    if len(aligned) == 0 or len(not_aligned) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    stats = np.empty(n_bootstraps)

    for i in range(n_bootstraps):

        sample_a = rng.choice(
            aligned,
            size=len(aligned),
            replace=True,
        )

        sample_b = rng.choice(
            not_aligned,
            size=len(not_aligned),
            replace=True,
        )

        stats[i] = (
            sample_a.mean()
            - sample_b.mean()
        )

    observed = (
        aligned.mean()
        - not_aligned.mean()
    )

    lower = float(np.percentile(stats, 2.5))
    upper = float(np.percentile(stats, 97.5))

    return observed, lower, upper


# ============================================================
# MTF SCORE MONOTONICITY
# ============================================================

def score_monotonicity(df):
    score_df = score_analysis(df)

    if len(score_df) < 2:
        return {
            "Spearman_Rho": np.nan,
            "Monotonicity": "INSUFFICIENT",
        }

    rho = score_df[
        ["Score", "Event_Rate"]
    ].corr(method="spearman").iloc[0, 1]

    if pd.isna(rho):
        label = "INSUFFICIENT"
    elif rho < -0.5:
        label = "NEGATIVE"
    elif rho > 0.5:
        label = "POSITIVE"
    else:
        label = "WEAK/NON_MONOTONIC"

    return {
        "Spearman_Rho": float(rho),
        "Monotonicity": label,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_header("MTF × EARLY-ADVERSE PREDICTION FORENSIC")

    print(
        f"Frozen hypothesis: "
        f"{EARLY_ADVERSE_BARS} bars / "
        f"{EARLY_ADVERSE_THRESHOLD_R:.2f}R"
    )

    print(f"Frozen ledger : {FROZEN_PATH}")
    print(f"MTF ledger    : {MTF_PATH}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    frozen = load_frozen()
    mtf = load_mtf()

    print()
    print(f"Frozen trade rows : {len(frozen)}")
    print(f"MTF trade rows    : {len(mtf)}")

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = reconcile_and_merge(
        frozen,
        mtf,
    )

    print(f"Analyzed rows     : {len(merged)}")

    missing_mtf = int(
        merged["Pre_Entry_MTF_Class"].isna().sum()
    )

    print(f"Missing MTF rows  : {missing_mtf}")

    if missing_mtf:
        raise RuntimeError(
            "Some frozen trades could not be matched to MTF rows."
        )

    # --------------------------------------------------------
    # Event
    # --------------------------------------------------------

    merged = derive_early_adverse_event(merged)

    missing_event = int(
        merged["Early_Adverse_Event"].isna().sum()
    )

    if missing_event:
        raise RuntimeError(
            "Early_Adverse_Group contains unsupported/missing values."
        )

    # --------------------------------------------------------
    # Signal year
    # --------------------------------------------------------

    merged["Signal_Year"] = (
        merged["Signal_Date"].dt.year
    )

    # --------------------------------------------------------
    # Core reconciliation
    # --------------------------------------------------------

    frozen_total_r = float(
        frozen["r_multiple"].sum()
    )

    analyzed_total_r = float(
        merged["r_multiple"].sum()
    )

    trade_count_pass = (
        len(frozen) == len(merged)
    )

    r_pass = (
        abs(
            frozen_total_r
            - analyzed_total_r
        ) < 1e-9
    )

    print_header("FINAL LEDGER RECONCILIATION")

    print(f"Frozen rows       : {len(frozen)}")
    print(f"Analyzed rows     : {len(merged)}")
    print(f"Frozen total R    : {frozen_total_r:.2f}")
    print(f"Analyzed total R  : {analyzed_total_r:.2f}")
    print(
        f"Trade-count PASS  : "
        f"{'PASS' if trade_count_pass else 'FAIL'}"
    )
    print(
        f"R reconciliation  : "
        f"{'PASS' if r_pass else 'FAIL'}"
    )

    if not trade_count_pass or not r_pass:
        raise RuntimeError(
            "FATAL: frozen ledger reconciliation failed."
        )

    # ========================================================
    # BASIC EVENT RATE
    # ========================================================

    print_header("1. BASELINE EARLY-ADVERSE EVENT")

    total_n = len(merged)
    total_events = int(
        merged["Early_Adverse_Event"].sum()
    )

    total_rate = total_events / total_n

    print(f"Total trades      : {total_n}")
    print(f"Early-adverse N   : {total_events}")
    print(f"Early-adverse rate: {pct(total_rate)}")

    # ========================================================
    # MTF CLASS
    # ========================================================

    print_header("2. MTF CLASS × EARLY-ADVERSE EVENT")

    class_summary = summarize_group(
        merged,
        ["Pre_Entry_MTF_Class"],
    )

    print(
        class_summary.to_string(index=False)
    )

    # ========================================================
    # MTF STATE
    # ========================================================

    print_header("3. MTF STATE × EARLY-ADVERSE EVENT")

    state_summary = summarize_group(
        merged,
        ["Pre_Entry_MTF_State"],
    )

    print(
        state_summary.to_string(index=False)
    )

    # ========================================================
    # MTF SCORE
    # ========================================================

    print_header("4. MTF SCORE × EARLY-ADVERSE EVENT")

    score_df = score_analysis(merged)

    if len(score_df):
        print(
            score_df.to_string(index=False)
        )

    score_result = score_monotonicity(merged)

    print(
        f"Spearman rho : "
        f"{fmt(score_result['Spearman_Rho'])}"
    )

    print(
        f"Monotonicity : "
        f"{score_result['Monotonicity']}"
    )

    # ========================================================
    # ALIGNMENT EFFECT
    # ========================================================

    print_header("5. MTF ALIGNMENT EFFECT")

    alignment_effect = two_group_effect(
        merged,
        "Pre_Entry_MTF_State",
        "ALIGNED",
    )

    for key, value in alignment_effect.items():
        print(
            f"{key:28s}: "
            f"{value if not isinstance(value, float) else fmt(value)}"
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    print_header("6. DIRECTION × MTF ALIGNMENT")

    direction_df = direction_effects(
        merged
    )

    if len(direction_df):
        print(
            direction_df.to_string(index=False)
        )

    # ========================================================
    # SETUP
    # ========================================================

    print_header("7. SETUP × MTF ALIGNMENT")

    setup_df = setup_effects(
        merged
    )

    if len(setup_df):
        print(
            setup_df.to_string(index=False)
        )
    else:
        print(
            "No setup cells met the minimum sample requirement."
        )

    # ========================================================
    # CONFLICT
    # ========================================================

    print_header("8. MTF CONFLICT / STATE BREAKDOWN")

    conflict_df = conflict_analysis(
        merged
    )

    if len(conflict_df):
        print(
            conflict_df.to_string(index=False)
        )

    # ========================================================
    # SYMBOL ROBUSTNESS
    # ========================================================

    print_header("9. SYMBOL ROBUSTNESS")

    symbol_df = robustness_by_symbol(
        merged
    )

    if len(symbol_df):
        print(
            symbol_df.to_string(index=False)
        )

        positive_symbols = int(
            (
                symbol_df["Event_Rate_Difference"] < 0
            ).sum()
        )

        print()
        print(
            "Symbols where ALIGNED reduced "
            "early-adverse probability: "
            f"{positive_symbols}/{len(symbol_df)}"
        )
    else:
        positive_symbols = 0
        print(
            "No symbol cells met the minimum sample requirement."
        )

    # ========================================================
    # YEAR ROBUSTNESS
    # ========================================================

    print_header("10. YEAR ROBUSTNESS")

    year_df = robustness_by_year(
        merged
    )

    if len(year_df):
        print(
            year_df.to_string(index=False)
        )

        positive_years = int(
            (
                year_df["Event_Rate_Difference"] < 0
            ).sum()
        )

        print()
        print(
            "Years where ALIGNED reduced "
            "early-adverse probability: "
            f"{positive_years}/{len(year_df)}"
        )
    else:
        positive_years = 0
        print(
            "No year cells met the minimum sample requirement."
        )

    # ========================================================
    # PERMUTATION
    # ========================================================

    print_header("11. PERMUTATION SIGNIFICANCE")

    observed_perm, permutation_p = (
        permutation_test_aligned_event_rate(
            merged,
            n_permutations=N_PERMUTATIONS,
            seed=RNG_SEED,
        )
    )

    print(
        f"Observed aligned-vs-not event-rate "
        f"difference : {observed_perm:.6f}"
    )

    print(
        f"Permutations : {N_PERMUTATIONS}"
    )

    print(
        f"Empirical p  : {permutation_p:.6f}"
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print_header("12. BOOTSTRAP CONFIDENCE INTERVAL")

    observed_boot, ci_lower, ci_upper = (
        bootstrap_ci_aligned_event_rate(
            merged,
            n_bootstraps=N_BOOTSTRAPS,
            seed=RNG_SEED,
        )
    )

    print(
        f"Observed difference : {observed_boot:.6f}"
    )

    print(
        f"95% bootstrap CI    : "
        f"[{ci_lower:.6f}, {ci_upper:.6f}]"
    )

    # ========================================================
    # EVENT-RATE DIRECTION
    # ========================================================

    alignment_reduces_event = (
        observed_perm < 0
    )

    statistically_significant = (
        permutation_p < 0.05
    )

    ci_excludes_zero = (
        ci_lower > 0
        or ci_upper < 0
    )

    # ========================================================
    # FINAL VERDICT
    # ========================================================

    print_header("13. FINAL VERDICT")

    print(
        f"Observed aligned-vs-not event-rate delta: "
        f"{observed_perm:.6f}"
    )

    print(
        f"Permutation p-value: "
        f"{permutation_p:.6f}"
    )

    print(
        f"Bootstrap 95% CI: "
        f"[{ci_lower:.6f}, {ci_upper:.6f}]"
    )

    print(
        f"Positive symbol robustness: "
        f"{positive_symbols}/{len(symbol_df) if len(symbol_df) else 0}"
    )

    print(
        f"Positive year robustness: "
        f"{positive_years}/{len(year_df) if len(year_df) else 0}"
    )

    if (
        alignment_reduces_event
        and statistically_significant
        and ci_upper < 0
    ):
        verdict = (
            "STRONGER EVIDENCE: PRE-ENTRY MTF ALIGNMENT "
            "REDUCES THE 7B/0.25R EARLY-ADVERSE EVENT. "
            "OOS VALIDATION STILL REQUIRED."
        )

    elif (
        alignment_reduces_event
        and positive_symbols >= 4
        and positive_years >= 4
    ):
        verdict = (
            "PROMISING MTF EVENT-PREDICTION SIGNAL — "
            "NOT STATISTICALLY CONFIRMED."
        )

    elif alignment_reduces_event:
        verdict = (
            "WEAK/PRELIMINARY MTF EVENT-PREDICTION SIGNAL — "
            "INSUFFICIENT ROBUSTNESS."
        )

    else:
        verdict = (
            "NO RELIABLE EVIDENCE THAT MTF ALIGNMENT "
            "REDUCES THE EARLY-ADVERSE EVENT."
        )

    print()
    print(f"VERDICT: {verdict}")

    print()
    print(
        "Standalone MTF entry filter: "
        "NOT JUSTIFIED BY THIS FORENSIC TEST"
    )

    print(
        "Production MTF modification: "
        "NOT JUSTIFIED"
    )

    print(
        "OOS modification: "
        "NONE"
    )

    print(
        "Frozen strategy modification: "
        "NONE"
    )

    # ========================================================
    # TRADE-LEVEL OUTPUT
    # ========================================================

    output_columns = [
        "_symbol",
        "Signal_Date",
        "direction",
        "r_multiple",
        "Symbol_Normalized",
        "Direction_Normalized",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Pre_Entry_MTF_Score",
        "Early_Adverse_Group",
        "Early_Adverse_Event",
        "Signal_Year",
        "Trade_Key",
    ]

    output_columns = [
        c for c in output_columns
        if c in merged.columns
    ]

    merged[output_columns].to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    # ========================================================
    # SUMMARY OUTPUT
    # ========================================================

    summary_rows = []

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "Frozen_Trades",
            "Value": len(frozen),
        }
    )

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "Analyzed_Trades",
            "Value": len(merged),
        }
    )

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "Frozen_Total_R",
            "Value": frozen_total_r,
        }
    )

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "Analyzed_Total_R",
            "Value": analyzed_total_r,
        }
    )

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "Trade_Count_Reconciliation",
            "Value": "PASS" if trade_count_pass else "FAIL",
        }
    )

    summary_rows.append(
        {
            "Section": "Integrity",
            "Metric": "R_Reconciliation",
            "Value": "PASS" if r_pass else "FAIL",
        }
    )

    summary_rows.append(
        {
            "Section": "Event",
            "Metric": "Early_Adverse_Bars",
            "Value": EARLY_ADVERSE_BARS,
        }
    )

    summary_rows.append(
        {
            "Section": "Event",
            "Metric": "Early_Adverse_Threshold_R",
            "Value": EARLY_ADVERSE_THRESHOLD_R,
        }
    )

    summary_rows.append(
        {
            "Section": "Event",
            "Metric": "Total_Event_Rate",
            "Value": total_rate,
        }
    )

    summary_rows.append(
        {
            "Section": "Inference",
            "Metric": "Aligned_vs_Not_Event_Rate_Delta",
            "Value": observed_perm,
        }
    )

    summary_rows.append(
        {
            "Section": "Inference",
            "Metric": "Permutation_P_Value",
            "Value": permutation_p,
        }
    )

    summary_rows.append(
        {
            "Section": "Inference",
            "Metric": "Bootstrap_CI_Lower",
            "Value": ci_lower,
        }
    )

    summary_rows.append(
        {
            "Section": "Inference",
            "Metric": "Bootstrap_CI_Upper",
            "Value": ci_upper,
        }
    )

    summary_rows.append(
        {
            "Section": "Robustness",
            "Metric": "Positive_Symbol_Deltas",
            "Value": positive_symbols,
        }
    )

    summary_rows.append(
        {
            "Section": "Robustness",
            "Metric": "Eligible_Symbols",
            "Value": len(symbol_df),
        }
    )

    summary_rows.append(
        {
            "Section": "Robustness",
            "Metric": "Positive_Year_Deltas",
            "Value": positive_years,
        }
    )

    summary_rows.append(
        {
            "Section": "Robustness",
            "Metric": "Eligible_Years",
            "Value": len(year_df),
        }
    )

    summary_rows.append(
        {
            "Section": "Verdict",
            "Metric": "Final_Verdict",
            "Value": verdict,
        }
    )

    summary_rows.append(
        {
            "Section": "Governance",
            "Metric": "Production_MTF_Modification",
            "Value": "NONE",
        }
    )

    summary_rows.append(
        {
            "Section": "Governance",
            "Metric": "OOS_Modification",
            "Value": "NONE",
        }
    )

    summary_rows.append(
        {
            "Section": "Governance",
            "Metric": "Frozen_Strategy_Modification",
            "Value": "NONE",
        }
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header("OUTPUTS")

    print(f"Trade-level : {TRADE_OUTPUT}")
    print(f"Summary     : {SUMMARY_OUTPUT}")

    print()
    print(
        "Production unchanged."
    )

    print(
        "OOS unchanged."
    )

    print(
        "Frozen strategy unchanged."
    )

    print(
        "Status: "
        "MTF_EARLY_ADVERSE_PREDICTION_COMPLETE"
    )


if __name__ == "__main__":
    main()