from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# TRADESENSE-AI — MTF INTERACTION SIGNIFICANCE FORENSIC
# =============================================================================
#
# PURPOSE
# -------
# Determine whether the observed pre-entry MTF × early-adverse interaction
# contains evidence beyond simple population differences.
#
# RESEARCH ONLY
# -------------
# - Does NOT modify production strategy.
# - Does NOT modify OOS data.
# - Does NOT modify frozen trade ledger.
# - Does NOT create an executable trading rule.
#
# AUTHORITATIVE OUTCOME
# ---------------------
# Frozen ledger:
#   tests/output/tradesense_trades.csv
#
# MTF forensic classification:
#   tests/output/reconciliation/mtf_daily_weekly/
#       mtf_pre_entry_timing_trade_level.csv
#
# =============================================================================


ROOT = Path(__file__).resolve().parents[1]

FROZEN_PATH = ROOT / "tests" / "output" / "tradesense_trades.csv"

MTF_PATH = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_pre_entry_timing_trade_level.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

TRADE_OUTPUT = OUTPUT_DIR / "mtf_interaction_significance_trade_level.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mtf_interaction_significance_summary.csv"

RANDOM_SEED = 42

N_PERMUTATIONS = 10000
N_BOOTSTRAPS = 10000


# =============================================================================
# HELPERS
# =============================================================================

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

    if value in {"LONG", "BUY", "1"}:
        return "LONG"

    if value in {"SHORT", "SELL", "-1"}:
        return "SHORT"

    return value


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def profit_factor(values):
    values = pd.Series(values).dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return 0.0

    return gross_profit / gross_loss


def summarize(group, label=None):
    r = pd.to_numeric(group["r_multiple"], errors="coerce").dropna()

    trades = len(r)
    wins = int((r > 0).sum())
    losses = int((r < 0).sum())

    total_r = float(r.sum()) if trades else 0.0
    average_r = float(r.mean()) if trades else 0.0

    win_rate = (
        wins / trades * 100
        if trades
        else 0.0
    )

    pf = profit_factor(r)

    return {
        "Group": label if label is not None else "",
        "Trades": trades,
        "Wins": wins,
        "Losses": losses,
        "Win_Rate_Pct": win_rate,
        "Profit_Factor": pf,
        "Total_R": total_r,
        "Average_R": average_r,
    }


def mean_difference(df, mask_a, mask_b, value_col):
    a = pd.to_numeric(
        df.loc[mask_a, value_col],
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        df.loc[mask_b, value_col],
        errors="coerce",
    ).dropna()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    return float(a.mean() - b.mean())


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 100)
    print("TRADESENSE-AI — MTF INTERACTION SIGNIFICANCE FORENSIC")
    print("=" * 100)

    if not FROZEN_PATH.exists():
        raise FileNotFoundError(
            f"Frozen ledger not found:\n{FROZEN_PATH}"
        )

    if not MTF_PATH.exists():
        raise FileNotFoundError(
            f"MTF trade-level file not found:\n{MTF_PATH}"
        )

    frozen = pd.read_csv(FROZEN_PATH)
    mtf = pd.read_csv(MTF_PATH)

    print()
    print(f"Frozen rows : {len(frozen)}")
    print(f"MTF rows    : {len(mtf)}")

    # -------------------------------------------------------------------------
    # Frozen authoritative fields
    # -------------------------------------------------------------------------

    required_frozen = [
        "_symbol",
        "Signal_Date",
        "direction",
        "r_multiple",
    ]

    missing = [
        c for c in required_frozen
        if c not in frozen.columns
    ]

    if missing:
        raise RuntimeError(
            f"Frozen ledger missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # MTF required fields
    # -------------------------------------------------------------------------

    required_mtf = [
        "_symbol",
        "Signal_Date",
        "direction",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Early_Adverse_Group",
    ]

    missing = [
        c for c in required_mtf
        if c not in mtf.columns
    ]

    if missing:
        raise RuntimeError(
            f"MTF file missing required columns: {missing}"
        )

    # -------------------------------------------------------------------------
    # Normalize keys
    # -------------------------------------------------------------------------

    frozen = frozen.copy()
    mtf = mtf.copy()

    frozen["Symbol_Normalized"] = (
        frozen["_symbol"]
        .map(normalize_symbol)
    )

    mtf["Symbol_Normalized"] = (
        mtf["_symbol"]
        .map(normalize_symbol)
    )

    frozen["Signal_Date_Normalized"] = pd.to_datetime(
        frozen["Signal_Date"],
        errors="coerce",
    ).dt.normalize()

    mtf["Signal_Date_Normalized"] = pd.to_datetime(
        mtf["Signal_Date"],
        errors="coerce",
    ).dt.normalize()

    frozen["Direction_Normalized"] = (
        frozen["direction"]
        .map(normalize_direction)
    )

    mtf["Direction_Normalized"] = (
        mtf["direction"]
        .map(normalize_direction)
    )

    frozen["r_multiple"] = clean_numeric(
        frozen["r_multiple"]
    )

    # -------------------------------------------------------------------------
    # Unique trade key
    # -------------------------------------------------------------------------

    frozen["Trade_Key"] = (
        frozen["Symbol_Normalized"].astype(str)
        + "|"
        + frozen["Signal_Date_Normalized"].astype(str)
        + "|"
        + frozen["Direction_Normalized"].astype(str)
    )

    mtf["Trade_Key"] = (
        mtf["Symbol_Normalized"].astype(str)
        + "|"
        + mtf["Signal_Date_Normalized"].astype(str)
        + "|"
        + mtf["Direction_Normalized"].astype(str)
    )

    return frozen, mtf


# =============================================================================
# RECONCILE / MERGE
# =============================================================================

def merge_data(frozen, mtf):

    print()
    print("=" * 100)
    print("TRADE KEY VALIDATION")
    print("=" * 100)

    frozen_dupes = frozen["Trade_Key"].duplicated().sum()
    mtf_dupes = mtf["Trade_Key"].duplicated().sum()

    print(f"Frozen duplicate rows : {frozen_dupes}")
    print(f"MTF duplicate rows    : {mtf_dupes}")

    if frozen_dupes:
        raise RuntimeError(
            "Frozen ledger contains duplicate trade keys."
        )

    if mtf_dupes:
        raise RuntimeError(
            "MTF file contains duplicate trade keys."
        )

    mtf_fields = [
        "Trade_Key",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Early_Adverse_Group",
    ]

    # Optional fields if present.
    optional_fields = [
        "Pre_Entry_MTF_Score",
        "Early_Adverse_Trigger_Bar",
        "max_adverse_r",
        "mae_percent",
        "bars_in_trade",
    ]

    for column in optional_fields:
        if column in mtf.columns:
            mtf_fields.append(column)

    mtf_small = mtf[mtf_fields].copy()

    merged = frozen.merge(
        mtf_small,
        on="Trade_Key",
        how="left",
        validate="one_to_one",
        suffixes=("", "_MTF"),
    )

    missing = merged["Pre_Entry_MTF_Class"].isna().sum()

    frozen_total_r = merged["r_multiple"].sum()

    print()
    print("=" * 100)
    print("DATA RECONCILIATION")
    print("=" * 100)

    print(f"Frozen trades       : {len(frozen)}")
    print(f"Analyzed trades     : {len(merged)}")
    print(f"Missing MTF matches : {missing}")
    print(f"Frozen total R      : {frozen_total_r:.4f}")

    if missing:
        raise RuntimeError(
            f"Missing {missing} MTF trade matches."
        )

    if len(frozen) != len(merged):
        raise RuntimeError(
            "Trade count reconciliation failed."
        )

    if not np.isclose(
        frozen["r_multiple"].sum(),
        merged["r_multiple"].sum(),
        atol=1e-9,
    ):
        raise RuntimeError(
            "R reconciliation failed."
        )

    print("Trade-count reconciliation : PASS")
    print("R reconciliation           : PASS")

    # -------------------------------------------------------------------------
    # Derived fields
    # -------------------------------------------------------------------------

    merged["MTF_State"] = (
        merged["Pre_Entry_MTF_State"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    merged["MTF_Class"] = (
        merged["Pre_Entry_MTF_Class"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    merged["Early_Group"] = (
        merged["Early_Adverse_Group"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    merged["Triggered"] = (
        merged["Early_Group"] == "TRIGGERED"
    )

    merged["MTF_Aligned"] = (
        merged["MTF_State"] == "ALIGNED"
    )

    merged["MTF_Conflict"] = (
        merged["MTF_State"] == "CONFLICT"
    )

    merged["Valid_MTF"] = ~merged["MTF_State"].eq(
        "UNAVAILABLE"
    )

    return merged


# =============================================================================
# BASIC INTERACTION
# =============================================================================

def run_basic_interaction(df):

    print()
    print("=" * 100)
    print("1. BASIC MTF × EARLY-ADVERSE INTERACTION")
    print("=" * 100)

    rows = []

    valid = df["Valid_MTF"]

    for early_label, early_mask in [
        ("TRIGGERED", df["Triggered"]),
        ("NOT_TRIGGERED", ~df["Triggered"]),
    ]:

        aligned_mask = valid & early_mask & df["MTF_Aligned"]
        not_aligned_mask = valid & early_mask & ~df["MTF_Aligned"]

        aligned = summarize(
            df.loc[aligned_mask],
            f"{early_label}_ALIGNED",
        )

        not_aligned = summarize(
            df.loc[not_aligned_mask],
            f"{early_label}_NOT_ALIGNED",
        )

        delta = (
            aligned["Average_R"]
            - not_aligned["Average_R"]
        )

        print()
        print(early_label)
        print(
            f"  ALIGNED     : "
            f"N={aligned['Trades']:3d} "
            f"AvgR={aligned['Average_R']:+.4f} "
            f"TotalR={aligned['Total_R']:+.2f}"
        )

        print(
            f"  NOT_ALIGNED : "
            f"N={not_aligned['Trades']:3d} "
            f"AvgR={not_aligned['Average_R']:+.4f} "
            f"TotalR={not_aligned['Total_R']:+.2f}"
        )

        print(
            f"  DELTA       : "
            f"{delta:+.4f}R/trade"
        )

        rows.append({
            "Early_Group": early_label,
            "Aligned_Trades": aligned["Trades"],
            "Aligned_Average_R": aligned["Average_R"],
            "Not_Aligned_Trades": not_aligned["Trades"],
            "Not_Aligned_Average_R": not_aligned["Average_R"],
            "MTF_Delta_Average_R": delta,
        })

    return rows


# =============================================================================
# DIFFERENCE IN DIFFERENCES
# =============================================================================

def calculate_did(df):

    valid = df["Valid_MTF"]

    triggered_aligned = (
        valid
        & df["Triggered"]
        & df["MTF_Aligned"]
    )

    triggered_not = (
        valid
        & df["Triggered"]
        & ~df["MTF_Aligned"]
    )

    not_triggered_aligned = (
        valid
        & ~df["Triggered"]
        & df["MTF_Aligned"]
    )

    not_triggered_not = (
        valid
        & ~df["Triggered"]
        & ~df["MTF_Aligned"]
    )

    triggered_effect = mean_difference(
        df,
        triggered_aligned,
        triggered_not,
        "r_multiple",
    )

    not_triggered_effect = mean_difference(
        df,
        not_triggered_aligned,
        not_triggered_not,
        "r_multiple",
    )

    did = (
        triggered_effect
        - not_triggered_effect
    )

    print()
    print("=" * 100)
    print("2. DIFFERENCE-IN-DIFFERENCES")
    print("=" * 100)

    print(
        f"Triggered MTF effect       : "
        f"{triggered_effect:+.4f}R"
    )

    print(
        f"Not-triggered MTF effect   : "
        f"{not_triggered_effect:+.4f}R"
    )

    print(
        f"Interaction / DiD          : "
        f"{did:+.4f}R"
    )

    return triggered_effect, not_triggered_effect, did


# =============================================================================
# PERMUTATION TEST
# =============================================================================

def permutation_test(df):

    print()
    print("=" * 100)
    print("3. PERMUTATION TEST — MTF × EARLY-ADVERSE INTERACTION")
    print("=" * 100)

    valid_df = df[df["Valid_MTF"]].copy()

    observed_triggered = (
        valid_df["Triggered"].values
    )

    observed_aligned = (
        valid_df["MTF_Aligned"].values
    )

    r_values = (
        pd.to_numeric(
            valid_df["r_multiple"],
            errors="coerce",
        )
        .values
    )

    good = np.isfinite(r_values)

    observed_triggered = observed_triggered[good]
    observed_aligned = observed_aligned[good]
    r_values = r_values[good]

    def calculate_interaction(
        triggered,
        aligned,
        returns,
    ):

        ta = triggered & aligned
        tn = triggered & ~aligned
        na = ~triggered & aligned
        nn = ~triggered & ~aligned

        if (
            ta.sum() == 0
            or tn.sum() == 0
            or na.sum() == 0
            or nn.sum() == 0
        ):
            return np.nan

        triggered_effect = (
            returns[ta].mean()
            - returns[tn].mean()
        )

        not_triggered_effect = (
            returns[na].mean()
            - returns[nn].mean()
        )

        return (
            triggered_effect
            - not_triggered_effect
        )

    observed = calculate_interaction(
        observed_triggered,
        observed_aligned,
        r_values,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    null_values = np.empty(N_PERMUTATIONS)

    # Preserve the early-adverse population while randomly
    # permuting MTF alignment labels.
    for i in range(N_PERMUTATIONS):

        shuffled_aligned = rng.permutation(
            observed_aligned
        )

        null_values[i] = calculate_interaction(
            observed_triggered,
            shuffled_aligned,
            r_values,
        )

    null_values = null_values[
        np.isfinite(null_values)
    ]

    p_value = (
        np.mean(
            np.abs(null_values)
            >= abs(observed)
        )
        if len(null_values)
        else np.nan
    )

    print(
        f"Observed interaction : "
        f"{observed:+.4f}R"
    )

    print(
        f"Permutations         : "
        f"{len(null_values)}"
    )

    print(
        f"Permutation p-value  : "
        f"{p_value:.6f}"
    )

    print(
        "Interpretation       : "
        "descriptive significance test; "
        "not proof of deployable edge."
    )

    return {
        "Observed_Interaction_R": observed,
        "Permutation_Count": len(null_values),
        "Permutation_P_Value": p_value,
    }


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================

def bootstrap_interaction(df):

    print()
    print("=" * 100)
    print("4. BOOTSTRAP CONFIDENCE INTERVAL — INTERACTION")
    print("=" * 100)

    valid_df = df[df["Valid_MTF"]].copy()

    rng = np.random.default_rng(RANDOM_SEED + 1)

    n = len(valid_df)

    if n == 0:
        raise RuntimeError(
            "No valid MTF trades available."
        )

    data = valid_df.reset_index(drop=True)

    def interaction_from_sample(sample):

        triggered = (
            sample["Triggered"].to_numpy()
        )

        aligned = (
            sample["MTF_Aligned"].to_numpy()
        )

        returns = (
            pd.to_numeric(
                sample["r_multiple"],
                errors="coerce",
            )
            .to_numpy()
        )

        ta = triggered & aligned
        tn = triggered & ~aligned
        na = ~triggered & aligned
        nn = ~triggered & ~aligned

        if (
            ta.sum() == 0
            or tn.sum() == 0
            or na.sum() == 0
            or nn.sum() == 0
        ):
            return np.nan

        return (
            (
                returns[ta].mean()
                - returns[tn].mean()
            )
            -
            (
                returns[na].mean()
                - returns[nn].mean()
            )
        )

    values = []

    for _ in range(N_BOOTSTRAPS):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sample = data.iloc[indices]

        value = interaction_from_sample(sample)

        if np.isfinite(value):
            values.append(value)

    values = np.asarray(values)

    observed = interaction_from_sample(data)

    if len(values) == 0:
        lower = np.nan
        upper = np.nan
    else:
        lower, upper = np.percentile(
            values,
            [2.5, 97.5],
        )

    print(
        f"Observed interaction : "
        f"{observed:+.4f}R"
    )

    print(
        f"Bootstrap samples     : "
        f"{len(values)}"
    )

    print(
        f"95% bootstrap CI      : "
        f"[{lower:+.4f}, {upper:+.4f}]R"
    )

    return {
        "Bootstrap_Count": len(values),
        "Bootstrap_Observed_R": observed,
        "Bootstrap_CI_Lower_R": lower,
        "Bootstrap_CI_Upper_R": upper,
    }


# =============================================================================
# EARLY-ADVERSE SEVERITY
# =============================================================================

def severity_analysis(df):

    print()
    print("=" * 100)
    print("5. MTF × EARLY-ADVERSE SEVERITY")
    print("=" * 100)

    triggered = df[
        df["Triggered"]
        & df["Valid_MTF"]
    ].copy()

    print()
    print(
        f"Triggered + valid MTF trades : "
        f"{len(triggered)}"
    )

    if len(triggered) == 0:
        return []

    # -------------------------------------------------------------------------
    # Look for available severity columns.
    # -------------------------------------------------------------------------

    severity_candidates = [
        "max_adverse_r",
        "mae_percent",
        "Early_Adverse_Trigger_Bar",
        "early_adverse_trigger_bar",
    ]

    available = [
        c
        for c in severity_candidates
        if c in triggered.columns
    ]

    print(
        "Available severity fields : "
        + ", ".join(available)
    )

    rows = []

    for field in available:

        triggered[field] = clean_numeric(
            triggered[field]
        )

        aligned = triggered[
            triggered["MTF_Aligned"]
        ][field].dropna()

        not_aligned = triggered[
            ~triggered["MTF_Aligned"]
        ][field].dropna()

        if len(aligned) == 0 or len(not_aligned) == 0:
            continue

        print()
        print(field)

        print(
            f"  ALIGNED     N={len(aligned):2d} "
            f"Mean={aligned.mean():+.4f} "
            f"Median={aligned.median():+.4f}"
        )

        print(
            f"  NOT_ALIGNED N={len(not_aligned):2d} "
            f"Mean={not_aligned.mean():+.4f} "
            f"Median={not_aligned.median():+.4f}"
        )

        print(
            f"  Difference   : "
            f"{aligned.mean() - not_aligned.mean():+.4f}"
        )

        rows.append({
            "Severity_Field": field,
            "Aligned_N": len(aligned),
            "Aligned_Mean": aligned.mean(),
            "Aligned_Median": aligned.median(),
            "Not_Aligned_N": len(not_aligned),
            "Not_Aligned_Mean": not_aligned.mean(),
            "Not_Aligned_Median": not_aligned.median(),
            "Mean_Difference": (
                aligned.mean()
                - not_aligned.mean()
            ),
        })

    return rows


# =============================================================================
# CONFLICT ANALYSIS
# =============================================================================

def conflict_analysis(df):

    print()
    print("=" * 100)
    print("6. MTF CONFLICT-SPECIFIC FORENSIC")
    print("=" * 100)

    valid = df["Valid_MTF"]

    for early_label, early_mask in [
        ("TRIGGERED", df["Triggered"]),
        ("NOT_TRIGGERED", ~df["Triggered"]),
    ]:

        print()
        print(early_label)

        for state in [
            "ALIGNED",
            "CONFLICT",
            "NEUTRAL",
        ]:

            mask = (
                valid
                & early_mask
                & df["MTF_State"].eq(state)
            )

            if mask.sum() == 0:
                continue

            s = summarize(
                df.loc[mask]
            )

            print(
                f"  {state:10s} "
                f"N={s['Trades']:2d} "
                f"Win={s['Win_Rate_Pct']:6.2f}% "
                f"PF={s['Profit_Factor']:6.3f} "
                f"R={s['Total_R']:+7.2f} "
                f"AvgR={s['Average_R']:+.4f}"
            )


# =============================================================================
# DIRECTION ANALYSIS
# =============================================================================

def direction_analysis(df):

    print()
    print("=" * 100)
    print("7. DIRECTION-SPECIFIC MTF INTERACTION")
    print("=" * 100)

    rows = []

    valid = df["Valid_MTF"]

    for direction in sorted(
        df["Direction_Normalized"]
        .dropna()
        .unique()
    ):

        direction_mask = (
            valid
            & df["Direction_Normalized"].eq(direction)
        )

        for early_label, early_mask in [
            ("TRIGGERED", df["Triggered"]),
            ("NOT_TRIGGERED", ~df["Triggered"]),
        ]:

            base = (
                direction_mask
                & early_mask
            )

            aligned = base & df["MTF_Aligned"]
            not_aligned = base & ~df["MTF_Aligned"]

            if (
                aligned.sum() < 3
                or not_aligned.sum() < 3
            ):
                continue

            a = summarize(df.loc[aligned])
            n = summarize(df.loc[not_aligned])

            delta = (
                a["Average_R"]
                - n["Average_R"]
            )

            print(
                f"{direction:5s} "
                f"{early_label:13s} "
                f"Aligned={a['Trades']:2d} "
                f"AvgR={a['Average_R']:+.4f} | "
                f"NotAligned={n['Trades']:2d} "
                f"AvgR={n['Average_R']:+.4f} | "
                f"Delta={delta:+.4f}"
            )

            rows.append({
                "Direction": direction,
                "Early_Group": early_label,
                "Aligned_Trades": a["Trades"],
                "Aligned_Average_R": a["Average_R"],
                "Not_Aligned_Trades": n["Trades"],
                "Not_Aligned_Average_R": n["Average_R"],
                "MTF_Delta_Average_R": delta,
            })

    return rows


# =============================================================================
# SYMBOL ROBUSTNESS
# =============================================================================

def symbol_robustness(df):

    print()
    print("=" * 100)
    print("8. SYMBOL ROBUSTNESS — TRIGGERED GROUP")
    print("=" * 100)

    rows = []

    valid = df["Valid_MTF"] & df["Triggered"]

    symbols = sorted(
        df.loc[valid, "Symbol_Normalized"]
        .dropna()
        .unique()
    )

    for symbol in symbols:

        mask = valid & df["Symbol_Normalized"].eq(symbol)

        aligned = mask & df["MTF_Aligned"]
        not_aligned = mask & ~df["MTF_Aligned"]

        if (
            aligned.sum() < 2
            or not_aligned.sum() < 2
        ):
            continue

        a = df.loc[aligned, "r_multiple"].mean()
        n = df.loc[not_aligned, "r_multiple"].mean()

        delta = a - n

        print(
            f"{symbol:12s} "
            f"A={aligned.sum():2d} "
            f"NA={not_aligned.sum():2d} "
            f"Delta={delta:+.4f}R"
        )

        rows.append({
            "Symbol": symbol,
            "Aligned_Trades": aligned.sum(),
            "Not_Aligned_Trades": not_aligned.sum(),
            "MTF_Delta_Average_R": delta,
        })

    return rows


# =============================================================================
# YEAR ROBUSTNESS
# =============================================================================

def year_robustness(df):

    print()
    print("=" * 100)
    print("9. YEAR ROBUSTNESS — TRIGGERED GROUP")
    print("=" * 100)

    rows = []

    valid = df["Valid_MTF"] & df["Triggered"]

    df = df.copy()

    df["Signal_Year"] = pd.to_datetime(
        df["Signal_Date"],
        errors="coerce",
    ).dt.year

    years = sorted(
        df.loc[valid, "Signal_Year"]
        .dropna()
        .unique()
    )

    for year in years:

        mask = (
            valid
            & df["Signal_Year"].eq(year)
        )

        aligned = mask & df["MTF_Aligned"]
        not_aligned = mask & ~df["MTF_Aligned"]

        if (
            aligned.sum() < 2
            or not_aligned.sum() < 2
        ):
            continue

        a = df.loc[aligned, "r_multiple"].mean()
        n = df.loc[not_aligned, "r_multiple"].mean()

        delta = a - n

        print(
            f"{int(year)} "
            f"A={aligned.sum():2d} "
            f"NA={not_aligned.sum():2d} "
            f"Delta={delta:+.4f}R"
        )

        rows.append({
            "Year": int(year),
            "Aligned_Trades": aligned.sum(),
            "Not_Aligned_Trades": not_aligned.sum(),
            "MTF_Delta_Average_R": delta,
        })

    return rows


# =============================================================================
# SCORE ANALYSIS
# =============================================================================

def score_analysis(df):

    print()
    print("=" * 100)
    print("10. PRE-ENTRY MTF SCORE — TRIGGERED GROUP")
    print("=" * 100)

    if "Pre_Entry_MTF_Score" not in df.columns:
        print("MTF score unavailable.")
        return []

    subset = df[
        df["Valid_MTF"]
        & df["Triggered"]
    ].copy()

    subset["Pre_Entry_MTF_Score"] = clean_numeric(
        subset["Pre_Entry_MTF_Score"]
    )

    rows = []

    for score, group in subset.groupby(
        "Pre_Entry_MTF_Score",
        dropna=True,
    ):

        s = summarize(group)

        print(
            f"Score={score:+.0f} "
            f"N={s['Trades']:2d} "
            f"Win={s['Win_Rate_Pct']:6.2f}% "
            f"PF={s['Profit_Factor']:6.3f} "
            f"R={s['Total_R']:+7.2f} "
            f"AvgR={s['Average_R']:+.4f}"
        )

        rows.append({
            "MTF_Score": score,
            **s,
        })

    return rows


# =============================================================================
# FINAL VERDICT
# =============================================================================

def determine_verdict(
    permutation,
    bootstrap,
    direction_rows,
    symbol_rows,
    year_rows,
):

    observed = permutation["Observed_Interaction_R"]
    p_value = permutation["Permutation_P_Value"]

    lower = bootstrap["Bootstrap_CI_Lower_R"]
    upper = bootstrap["Bootstrap_CI_Upper_R"]

    positive_symbols = sum(
        r["MTF_Delta_Average_R"] > 0
        for r in symbol_rows
    )

    total_symbols = len(symbol_rows)

    positive_years = sum(
        r["MTF_Delta_Average_R"] > 0
        for r in year_rows
    )

    total_years = len(year_rows)

    print()
    print("=" * 100)
    print("11. FINAL RESEARCH VERDICT")
    print("=" * 100)

    print(
        f"Observed interaction      : "
        f"{observed:+.4f}R"
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
        f"Positive symbol deltas    : "
        f"{positive_symbols}/{total_symbols}"
    )

    print(
        f"Positive year deltas      : "
        f"{positive_years}/{total_years}"
    )

    print()

    if (
        np.isfinite(p_value)
        and p_value < 0.05
        and np.isfinite(lower)
        and lower > 0
    ):
        verdict = (
            "STRONGER EVIDENCE OF MTF INTERACTION — "
            "STILL REQUIRES OUT-OF-SAMPLE VALIDATION"
        )

    elif (
        observed > 0
        and positive_symbols >= 4
        and positive_years >= 4
    ):
        verdict = (
            "PROMISING MTF INTERACTION — "
            "NOT STATISTICALLY CONFIRMED"
        )

    elif observed > 0:
        verdict = (
            "WEAK/PRELIMINARY MTF INTERACTION — "
            "INSUFFICIENT EVIDENCE FOR DEPLOYMENT"
        )

    else:
        verdict = (
            "NO RELIABLE MTF INTERACTION EVIDENCE"
        )

    print(f"VERDICT: {verdict}")

    print()
    print("RULE STATUS:")
    print("  Standalone MTF filter       : NOT JUSTIFIED")
    print("  Production MTF modification : NOT JUSTIFIED")
    print("  OOS modification            : NONE")
    print("  Frozen strategy modification: NONE")

    return verdict


# =============================================================================
# MAIN
# =============================================================================

def main():

    frozen, mtf = load_data()

    df = merge_data(
        frozen,
        mtf,
    )

    interaction_rows = run_basic_interaction(df)

    (
        triggered_effect,
        not_triggered_effect,
        did,
    ) = calculate_did(df)

    permutation = permutation_test(df)

    bootstrap = bootstrap_interaction(df)

    severity_rows = severity_analysis(df)

    conflict_analysis(df)

    direction_rows = direction_analysis(df)

    symbol_rows = symbol_robustness(df)

    year_rows = year_robustness(df)

    score_rows = score_analysis(df)

    verdict = determine_verdict(
        permutation,
        bootstrap,
        direction_rows,
        symbol_rows,
        year_rows,
    )

    # -------------------------------------------------------------------------
    # Trade-level output
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary output
    # -------------------------------------------------------------------------

    summary_rows = []

    summary_rows.append({
        "Section": "INTERACTION",
        "Metric": "Triggered_MTF_Effect_Average_R",
        "Value": triggered_effect,
    })

    summary_rows.append({
        "Section": "INTERACTION",
        "Metric": "Not_Triggered_MTF_Effect_Average_R",
        "Value": not_triggered_effect,
    })

    summary_rows.append({
        "Section": "INTERACTION",
        "Metric": "Difference_in_Differences_R",
        "Value": did,
    })

    for key, value in permutation.items():
        summary_rows.append({
            "Section": "PERMUTATION",
            "Metric": key,
            "Value": value,
        })

    for key, value in bootstrap.items():
        summary_rows.append({
            "Section": "BOOTSTRAP",
            "Metric": key,
            "Value": value,
        })

    for row in severity_rows:
        for key, value in row.items():
            summary_rows.append({
                "Section": "SEVERITY_" + row["Severity_Field"],
                "Metric": key,
                "Value": value,
            })

    for row in interaction_rows:
        for key, value in row.items():
            summary_rows.append({
                "Section": "INTERACTION_TABLE",
                "Metric": key,
                "Value": value,
            })

    summary_rows.append({
        "Section": "VERDICT",
        "Metric": "Verdict",
        "Value": verdict,
    })

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final integrity
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL INTEGRITY")
    print("=" * 100)

    frozen_count = len(frozen)
    analyzed_count = len(df)

    frozen_r = frozen["r_multiple"].sum()
    analyzed_r = df["r_multiple"].sum()

    print(
        f"Frozen trades              : "
        f"{frozen_count}"
    )

    print(
        f"Analyzed trades            : "
        f"{analyzed_count}"
    )

    print(
        f"Frozen total R             : "
        f"{frozen_r:.4f}"
    )

    print(
        f"Analyzed total R           : "
        f"{analyzed_r:.4f}"
    )

    print(
        f"Reconciliation delta       : "
        f"{analyzed_r - frozen_r:.4f}"
    )

    if (
        frozen_count != analyzed_count
        or not np.isclose(
            frozen_r,
            analyzed_r,
            atol=1e-9,
        )
    ):
        raise RuntimeError(
            "FINAL RECONCILIATION FAILED."
        )

    print()
    print("Trade-count reconciliation : PASS")
    print("R reconciliation           : PASS")
    print()
    print("Production strategy        : UNCHANGED")
    print("Production MTF logic       : UNCHANGED")
    print("OOS data                   : UNCHANGED")
    print()
    print(
        "STATUS: "
        "MTF_INTERACTION_SIGNIFICANCE_COMPLETE"
    )

    print()
    print(
        f"Trade-level output : {TRADE_OUTPUT}"
    )

    print(
        f"Summary output     : {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()