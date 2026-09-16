"""
TradeSense-AI
MTF × Frozen 5B / 0.25R Retention Economics

Purpose
-------
Evaluate whether MTF alignment can improve the frozen early-adverse
retention hypothesis without changing the frozen trading strategy.

Authoritative frozen rule
-------------------------
Reject a trade only when:

    early_adverse_r_bar_5 >= 0.25R

Missing Bar-5 values are retained.

Important
---------
This script is forensic/research only.

It MUST NOT modify:
- production strategy
- frozen trade ledger
- OOS logic
- entry logic
- exit logic
- risk logic
- frozen hypothesis definition

Historical +7.10R research figure
---------------------------------
The historical research record contains a +7.10R figure for the
early-adverse hypothesis.

The current frozen trade CSV is independently classified here using
the authoritative 5B / 0.25R rule. Therefore this script does NOT
hard-code +7.10R as an acceptance gate.

If the current ledger produces a different value, that discrepancy
must be treated as a historical ledger/version reconciliation issue,
not silently corrected.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FROZEN_TRADES = Path(
    "tests/output/tradesense_trades.csv"
)

MTF_COMPONENTS = Path(
    "tests/output/reconciliation/mtf_daily_weekly/"
    "mtf_component_contribution_trade_level.csv"
)

TRADE_OUTPUT = Path(
    "tests/output/reconciliation/mtf_daily_weekly/"
    "mtf_7b025_retention_trade_level.csv"
)

SUMMARY_OUTPUT = Path(
    "tests/output/reconciliation/mtf_daily_weekly/"
    "mtf_7b025_retention_summary.csv"
)

THRESHOLD_R = 0.25

def find_column(df, candidates, required=True):
    """
    Find a dataframe column using case-insensitive matching.
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


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(value):
    """
    Normalize symbols so that:
        RELIANCE
        RELIANCE.NS

    become the same research key.
    """
    if pd.isna(value):
        return value

    value = str(value).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def normalize_date(value):
    """
    Normalize dates to YYYY-MM-DD strings.
    """
    return pd.to_datetime(value).dt.strftime("%Y-%m-%d")


def require_columns(df, columns, dataframe_name):
    """
    Fail loudly if expected research columns are missing.
    """
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def safe_numeric(df, columns):
    """
    Convert selected columns to numeric.
    """
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD FROZEN TRADES
# ============================================================

def load_frozen():
    print()
    print("Loading frozen trade ledger...")

    if not FROZEN_TRADES.exists():
        raise FileNotFoundError(
            f"Frozen trade ledger not found: {FROZEN_TRADES}"
        )

    df = pd.read_csv(FROZEN_TRADES)

    print(f"Frozen trades loaded : {len(df)}")

    require_columns(
        df,
        [
            "r_multiple",
            "early_adverse_r_bar_5",
        ],
        "Frozen trade ledger"
    )

    # --------------------------------------------------------
    # Resolve symbol column
    # --------------------------------------------------------

    symbol_candidates = [
        "_symbol",
        "symbol",
        "Symbol",
    ]

    symbol_col = None

    for candidate in symbol_candidates:
        if candidate in df.columns:
            symbol_col = candidate
            break

    if symbol_col is None:
        raise RuntimeError(
            "Could not identify frozen trade symbol column."
        )

    df["Research_Symbol"] = df[symbol_col].apply(
        normalize_symbol
    )

    df["_symbol_root"] = df["Research_Symbol"]

    # --------------------------------------------------------
    # Resolve signal/trade date column
    # --------------------------------------------------------

    date_candidates = [
        "signal_date",
        "Signal_Date",
        "date",
        "Date",
    ]

    date_col = None

    for candidate in date_candidates:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise RuntimeError(
            "Could not identify frozen trade date column."
        )

    df["Research_Date"] = normalize_date(
        pd.to_datetime(df[date_col], errors="coerce")
    )

    df["Signal_Date"] = pd.to_datetime(
    df[date_col],
    errors="coerce"
    )

    if df["Research_Date"].isna().any():
        raise RuntimeError(
            "Frozen ledger contains invalid trade dates."
        )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    direction_candidates = [
        "direction",
        "Direction",
    ]

    direction_col = None

    for candidate in direction_candidates:
        if candidate in df.columns:
            direction_col = candidate
            break

    if direction_col is not None:
        df["Direction"] = (
            df[direction_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        df["Direction"] = "UNKNOWN"

    df["Direction_Normalized"] = df["Direction"]

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    setup_candidates = [
        "Setup",
        "setup",
    ]

    setup_col = None

    for candidate in setup_candidates:
        if candidate in df.columns:
            setup_col = candidate
            break

    if setup_col is not None:
        df["Setup"] = (
            df[setup_col]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        df["Setup"] = "UNKNOWN"

    # --------------------------------------------------------
    # R multiple
    # --------------------------------------------------------

    df["R_Multiple"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce"
    )

    if df["R_Multiple"].isna().any():
        raise RuntimeError(
            "Frozen ledger contains invalid r_multiple values."
        )

    # --------------------------------------------------------
    # Bar-5 early adverse value
    # --------------------------------------------------------

    df["Bar5_Early_Adverse_R"] = pd.to_numeric(
        df["early_adverse_r_bar_5"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Research key
    #
    # MTF component file is symbol|date.
    # Frozen trade uniqueness is additionally checked using
    # symbol|date|direction.
    # --------------------------------------------------------

    df["MTF_Key"] = (
        df["Research_Symbol"]
        + "|"
        + df["Research_Date"]
    )

    df["Trade_Key"] = (
        df["Research_Symbol"]
        + "|"
        + df["Research_Date"]
        + "|"
        + df["Direction"]
    )

    # --------------------------------------------------------
    # Duplicate validation
    # --------------------------------------------------------

    duplicate_count = df["Trade_Key"].duplicated().sum()

    print(f"Frozen duplicate keys: {duplicate_count}")

    if duplicate_count:
        duplicates = df.loc[
            df["Trade_Key"].duplicated(keep=False),
            [
                "Trade_Key",
                "Research_Symbol",
                "Research_Date",
                "Direction",
            ],
        ]

        print()
        print("Duplicate frozen trade keys:")
        print(duplicates.to_string(index=False))

        raise RuntimeError(
            "Frozen trade ledger contains duplicate trade keys."
        )

    return df


# ============================================================
# LOAD MTF COMPONENTS
# ============================================================

def load_mtf():
    print()
    print("Loading MTF component contribution ledger...")

    if not MTF_COMPONENTS.exists():
        raise FileNotFoundError(
            f"MTF component ledger not found: {MTF_COMPONENTS}"
        )

    mtf = pd.read_csv(
        MTF_COMPONENTS
    )

    print(f"MTF component rows   : {len(mtf)}")

    # --------------------------------------------------------
    # Resolve actual MTF ledger columns
    # --------------------------------------------------------

    symbol_col = find_column(
        mtf,
        [
            "Symbol",
            "symbol",
            "_symbol_root",
            "_symbol",
        ],
    )

    date_col = find_column(
        mtf,
        [
            "Signal_Date",
            "signal_date",
            "SignalDate",
        ],
    )

    daily_col = find_column(
        mtf,
        [
            "Daily_Relation",
        ],
    )

    weekly_col = find_column(
        mtf,
        [
            "Weekly_Relation",
        ],
    )

    combined_col = find_column(
        mtf,
        [
            "Combined_MTF_Class",
        ],
    )

    weekly_timestamp_col = find_column(
        mtf,
        [
            "Completed_Weekly_Date",
        ],
        required=False,
    )

    # --------------------------------------------------------
    # Normalize symbol
    # --------------------------------------------------------

    mtf["_symbol_root"] = (
        mtf[symbol_col]
        .map(normalize_symbol)
    )

    # --------------------------------------------------------
    # Normalize signal date
    # --------------------------------------------------------

    mtf["Signal_Date"] = pd.to_datetime(
        mtf[date_col],
        errors="coerce",
    )

    # --------------------------------------------------------
    # MTF merge key
    #
    # IMPORTANT:
    # symbol + signal date only
    # --------------------------------------------------------

    mtf["_mtf_key"] = (
        mtf["_symbol_root"].astype(str)
        + "|"
        + mtf["Signal_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # Daily relation
    # --------------------------------------------------------

    mtf["Daily_Relation"] = (
        mtf[daily_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Weekly relation
    # --------------------------------------------------------

    mtf["Weekly_Relation"] = (
        mtf[weekly_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Combined MTF classification
    # --------------------------------------------------------

    mtf["Combined_MTF_Class"] = (
        mtf[combined_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Completed weekly timestamp
    # --------------------------------------------------------

    if weekly_timestamp_col is not None:

        mtf["Completed_Weekly_Date"] = pd.to_datetime(
            mtf[weekly_timestamp_col],
            errors="coerce",
        )

    else:

        mtf["Completed_Weekly_Date"] = pd.NaT

    return mtf


# ============================================================
# MERGE
# ============================================================

def merge_data(trades, mtf):
    print()
    print("Merging frozen trades with MTF components...")

    # --------------------------------------------------------
    # Frozen trade MTF key
    #
    # MTF component ledger is keyed by:
    # symbol + signal date
    #
    # Direction is NOT part of this key.
    # --------------------------------------------------------

    trades["_mtf_key"] = (
        trades["_symbol_root"].astype(str)
        + "|"
        + trades["Signal_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    frozen_duplicates = (
        trades["_mtf_key"]
        .duplicated()
        .sum()
    )

    mtf_duplicates = (
        mtf["_mtf_key"]
        .duplicated()
        .sum()
    )

    print(
        f"Frozen duplicate keys: {frozen_duplicates}"
    )

    print(
        f"MTF duplicate keys   : {mtf_duplicates}"
    )

    if frozen_duplicates:
        raise RuntimeError(
            "Frozen MTF merge key is not unique."
        )

    if mtf_duplicates:
        raise RuntimeError(
            "MTF merge key is not unique."
        )

    # --------------------------------------------------------
    # One-to-one merge
    # --------------------------------------------------------

    analysis = trades.merge(
        mtf[
            [
                "_mtf_key",
                "Daily_Relation",
                "Weekly_Relation",
                "Combined_MTF_Class",
                "Completed_Weekly_Date",
            ]
        ],
        on="_mtf_key",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Missing MTF validation
    # --------------------------------------------------------

    missing = (
        analysis["Daily_Relation"]
        .isna()
    )

    print(
        f"Missing MTF rows     : "
        f"{missing.sum()}"
    )

    if missing.any():

        missing_rows = analysis.loc[
            missing,
            [
                "_symbol_root",
                "Signal_Date",
                "Direction_Normalized",
            ],
        ]

        print()
        print("Missing MTF rows:")
        print(
            missing_rows.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "MTF merge incomplete."
        )

    return analysis


# ============================================================
# CAUSALITY
# ============================================================

def validate_causality(df):
    print()
    print("Validating weekly MTF causality...")

    signal_dates = pd.to_datetime(
        df["Research_Date"],
        errors="coerce"
    )

    completed_weekly = pd.to_datetime(
        df["Completed_Weekly_Date"],
        errors="coerce"
    )

    valid = (
        completed_weekly.notna()
        & signal_dates.notna()
        & (completed_weekly < signal_dates)
    )

    pass_count = int(valid.sum())
    total = len(df)

    print(
        f"Weekly causal PASS   : {pass_count}/{total}"
    )

    if pass_count != total:
        bad = df.loc[
            ~valid,
            [
                "Research_Symbol",
                "Research_Date",
                "Completed_Weekly_Date",
            ],
        ]

        print()
        print("Weekly causality failures:")
        print(bad.to_string(index=False))

        raise RuntimeError(
            "Weekly MTF causality validation FAILED."
        )

    print("Weekly timestamp causality: PASS")


# ============================================================
# AUTHORITATIVE FROZEN EARLY-ADVERSE RULE
# ============================================================

def build_early_adverse(df):
    """
    Authoritative frozen research rule:

        Reject only when Bar-5 adverse excursion >= 0.25R.

    Missing Bar-5 values are retained.

    This intentionally uses the actual Bar-5 field rather than
    the separate historical telemetry field
    `early_adverse_triggered`.
    """

    df["Early_Adverse"] = (
        df["Bar5_Early_Adverse_R"].notna()
        & (
            df["Bar5_Early_Adverse_R"]
            >= THRESHOLD_R
        )
    )

    return df


# ============================================================
# FROZEN HYPOTHESIS
# ============================================================

def build_hypothesis(df):
    """
    Counterfactual frozen hypothesis:

        Early adverse trade -> reject -> 0R
        Otherwise           -> retain original R
    """

    df["Hypothesis_R"] = np.where(
        df["Early_Adverse"],
        0.0,
        df["R_Multiple"],
    )

    return df


# ============================================================
# MTF FLAGS
# ============================================================

def build_mtf_flags(df):
    """
    Build explicit boolean MTF flags from the authoritative
    MTF component classifications.

    Daily_Relation:
        ALIGNED / CONFLICT / ...

    Weekly_Relation:
        ALIGNED / CONFLICT / ...

    Combined_MTF_Class:
        FULL_ALIGNMENT / FULL_CONFLICT / MIXED / UNAVAILABLE
    """

    df["Daily_MTF_Aligned"] = (
        df["Daily_Relation"]
        == "ALIGNED"
    )

    df["Weekly_MTF_Aligned"] = (
        df["Weekly_Relation"]
        == "ALIGNED"
    )

    df["Full_MTF_Aligned_Flag"] = (
        df["Combined_MTF_Class"]
        == "FULL_ALIGNMENT"
    )

    return df


# ============================================================
# EARLY-ADVERSE RECONCILIATION
# ============================================================

def print_early_adverse_reconciliation(df):
    print()
    print("RECONCILIATION")

    baseline_r = df["R_Multiple"].sum()

    event_count = int(
        df["Early_Adverse"].sum()
    )

    retained_count = int(
        (~df["Early_Adverse"]).sum()
    )

    hypothesis_r = df["Hypothesis_R"].sum()

    expected_from_classification = (
        df.loc[
            ~df["Early_Adverse"],
            "R_Multiple",
        ].sum()
    )

    print(
        f"Frozen total R       : {baseline_r:+.2f}"
    )

    print(
        f"Authoritative 5B/.25R events : "
        f"{event_count}"
    )

    print(
        f"Authoritative retained trades : "
        f"{retained_count}"
    )

    print()
    print("EARLY-ADVERSE")

    print(
        f"5B/.25R event count   : {event_count}"
    )

    print(
        f"5B/.25R retained      : "
        f"{retained_count}"
    )

    print()
    print("5B/.25R RECONCILIATION")

    print(
        f"Reconstructed frozen 5B/.25R R : "
        f"{hypothesis_r:+.2f}R"
    )

    # --------------------------------------------------------
    # Internal accounting check
    #
    # Do NOT hard-code +7.10R here.
    # --------------------------------------------------------

    if not np.isclose(
        hypothesis_r,
        expected_from_classification,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Frozen 5B/.25R internal R "
            "reconciliation FAILED."
        )

    # Additional conservation check.
    removed_r = df.loc[
        df["Early_Adverse"],
        "R_Multiple",
    ].sum()

    reconstructed_sum = (
        hypothesis_r + removed_r
    )

    if not np.isclose(
        reconstructed_sum,
        baseline_r,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Frozen 5B/.25R R conservation "
            "check FAILED."
        )

    print(
        "Frozen 5B/.25R internal reconciliation: PASS"
    )

    print(
        "Frozen 5B/.25R R conservation: PASS"
    )

    return {
        "baseline_r": baseline_r,
        "event_count": event_count,
        "retained_count": retained_count,
        "hypothesis_r": hypothesis_r,
        "removed_r": removed_r,
    }


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_rule(
    df,
    alignment_column,
    keep_aligned=True,
):
    """
    Evaluate:

        - retain every non-event trade
        - for early-adverse events:
              keep if MTF condition matches requested rule

    keep_aligned=True:
        retain early-adverse events when aligned

    keep_aligned=False:
        retain early-adverse events when NOT aligned
    """

    event = df["Early_Adverse"]

    aligned = df[alignment_column].astype(bool)

    if keep_aligned:
        retained = (
            (~event)
            | (event & aligned)
        )

    else:
        retained = (
            (~event)
            | (event & ~aligned)
        )

    retained_r = df.loc[
        retained,
        "R_Multiple",
    ].sum()

    retained_count = int(retained.sum())

    total_count = len(df)

    event_count = int(event.sum())

    retained_events = int(
        (event & retained).sum()
    )

    rejected_events = int(
        (event & ~retained).sum()
    )

    delta_vs_baseline = (
        retained_r
        - df["R_Multiple"].sum()
    )

    return {
        "rule": (
            "KEEP_ALIGNED"
            if keep_aligned
            else "KEEP_NOT_ALIGNED"
        ),
        "alignment_column": alignment_column,
        "total_trades": total_count,
        "early_adverse_events": event_count,
        "retained_trades": retained_count,
        "retained_events": retained_events,
        "rejected_events": rejected_events,
        "retained_r": retained_r,
        "delta_vs_baseline": delta_vs_baseline,
    }


# ============================================================
# CORE MTF RULE TESTS
# ============================================================

def run_core_tests(df):
    print()
    print("=" * 72)
    print("CORE MTF RETENTION TESTS")
    print("=" * 72)

    tests = [
        (
            "Daily MTF",
            "Daily_MTF_Aligned",
        ),
        (
            "Weekly MTF",
            "Weekly_MTF_Aligned",
        ),
        (
            "Full MTF",
            "Full_MTF_Aligned_Flag",
        ),
    ]

    rows = []

    baseline_r = df["R_Multiple"].sum()
    hypothesis_r = df["Hypothesis_R"].sum()

    for name, column in tests:

        for keep_aligned in [
            True,
            False,
        ]:

            result = evaluate_rule(
                df,
                column,
                keep_aligned=keep_aligned,
            )

            result["test_name"] = name

            result["baseline_r"] = baseline_r

            result["frozen_hypothesis_r"] = (
                hypothesis_r
            )

            result["delta_vs_frozen_hypothesis"] = (
                result["retained_r"]
                - hypothesis_r
            )

            rows.append(result)

            label = (
                "KEEP ALIGNED"
                if keep_aligned
                else "KEEP NOT ALIGNED"
            )

            print()
            print(
                f"{name} — {label}"
            )

            print(
                f"  Retained trades : "
                f"{result['retained_trades']}"
            )

            print(
                f"  Retained events : "
                f"{result['retained_events']}"
            )

            print(
                f"  Rejected events : "
                f"{result['rejected_events']}"
            )

            print(
                f"  Retained R      : "
                f"{result['retained_r']:+.2f}R"
            )

            print(
                f"  Δ baseline      : "
                f"{result['delta_vs_baseline']:+.2f}R"
            )

            print(
                f"  Δ frozen hypo   : "
                f"{result['delta_vs_frozen_hypothesis']:+.2f}R"
            )

    return pd.DataFrame(rows)


# ============================================================
# EVENT ECONOMICS
# ============================================================

def event_economics(df):
    print()
    print("=" * 72)
    print("EARLY-ADVERSE EVENT ECONOMICS")
    print("=" * 72)

    events = df.loc[
        df["Early_Adverse"]
    ].copy()

    if events.empty:
        print("No early-adverse events.")
        return pd.DataFrame()

    rows = []

    for label, mask in [
        (
            "FULL_MTF_ALIGNED",
            events["Full_MTF_Aligned_Flag"],
        ),
        (
            "NOT_FULL_MTF_ALIGNED",
            ~events["Full_MTF_Aligned_Flag"],
        ),
        (
            "DAILY_ALIGNED",
            events["Daily_MTF_Aligned"],
        ),
        (
            "NOT_DAILY_ALIGNED",
            ~events["Daily_MTF_Aligned"],
        ),
        (
            "WEEKLY_ALIGNED",
            events["Weekly_MTF_Aligned"],
        ),
        (
            "NOT_WEEKLY_ALIGNED",
            ~events["Weekly_MTF_Aligned"],
        ),
    ]:

        subset = events.loc[mask]

        if subset.empty:
            continue

        r_sum = subset["R_Multiple"].sum()

        avg_r = subset["R_Multiple"].mean()

        wins = int(
            (subset["R_Multiple"] > 0).sum()
        )

        count = len(subset)

        win_rate = (
            wins / count * 100
            if count
            else np.nan
        )

        rows.append(
            {
                "group": label,
                "event_count": count,
                "wins": wins,
                "win_rate_pct": win_rate,
                "event_r": r_sum,
                "avg_event_r": avg_r,
            }
        )

        print()
        print(label)

        print(
            f"  Events    : {count}"
        )

        print(
            f"  Wins      : {wins}"
        )

        print(
            f"  Win rate  : {win_rate:.2f}%"
        )

        print(
            f"  Event R   : {r_sum:+.2f}R"
        )

        print(
            f"  Avg event : {avg_r:+.4f}R"
        )

    return pd.DataFrame(rows)


# ============================================================
# SETUP CONDITIONAL ANALYSIS
# ============================================================

def setup_conditional(df):
    print()
    print("=" * 72)
    print("SETUP-CONDITIONAL MTF RETENTION")
    print("=" * 72)

    rows = []

    for setup, group in df.groupby(
        "Setup",
        dropna=False,
    ):

        if len(group) < 3:
            continue

        baseline_r = group["R_Multiple"].sum()

        hypothesis_r = group["Hypothesis_R"].sum()

        event_count = int(
            group["Early_Adverse"].sum()
        )

        for name, column in [
            (
                "DAILY",
                "Daily_MTF_Aligned",
            ),
            (
                "WEEKLY",
                "Weekly_MTF_Aligned",
            ),
            (
                "FULL",
                "Full_MTF_Aligned_Flag",
            ),
        ]:

            for keep_aligned in [
                True,
                False,
            ]:

                result = evaluate_rule(
                    group,
                    column,
                    keep_aligned=keep_aligned,
                )

                rows.append(
                    {
                        "Setup": setup,
                        "MTF": name,
                        "Rule": result["rule"],
                        "Trades": len(group),
                        "Events": event_count,
                        "Baseline_R": baseline_r,
                        "Frozen_Hypothesis_R": hypothesis_r,
                        "Retained_R": result["retained_r"],
                        "Delta_vs_Baseline": (
                            result["delta_vs_baseline"]
                        ),
                        "Delta_vs_Hypothesis": (
                            result["retained_r"]
                            - hypothesis_r
                        ),
                    }
                )

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        print(
            result_df.to_string(index=False)
        )

    return result_df


# ============================================================
# SYMBOL LEAVE-ONE-OUT
# ============================================================

def symbol_leave_one_out(df):
    print()
    print("=" * 72)
    print("SYMBOL LEAVE-ONE-OUT MTF RETENTION")
    print("=" * 72)

    rows = []

    symbols = sorted(
        df["Research_Symbol"].dropna().unique()
    )

    for symbol in symbols:

        subset = df.loc[
            df["Research_Symbol"] != symbol
        ].copy()

        if subset.empty:
            continue

        hypothesis_r = subset[
            "Hypothesis_R"
        ].sum()

        for name, column in [
            (
                "DAILY",
                "Daily_MTF_Aligned",
            ),
            (
                "WEEKLY",
                "Weekly_MTF_Aligned",
            ),
            (
                "FULL",
                "Full_MTF_Aligned_Flag",
            ),
        ]:

            for keep_aligned in [
                True,
                False,
            ]:

                result = evaluate_rule(
                    subset,
                    column,
                    keep_aligned=keep_aligned,
                )

                rows.append(
                    {
                        "Left_Out_Symbol": symbol,
                        "MTF": name,
                        "Rule": result["rule"],
                        "Trades": len(subset),
                        "Retained_R": result["retained_r"],
                        "Frozen_Hypothesis_R": hypothesis_r,
                        "Delta_vs_Hypothesis": (
                            result["retained_r"]
                            - hypothesis_r
                        ),
                    }
                )

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        print(
            result_df.to_string(index=False)
        )

    return result_df


# ============================================================
# YEAR LEAVE-ONE-OUT
# ============================================================

def year_leave_one_out(df):
    print()
    print("=" * 72)
    print("YEAR LEAVE-ONE-OUT MTF RETENTION")
    print("=" * 72)

    temp = df.copy()

    temp["Year"] = pd.to_datetime(
        temp["Research_Date"]
    ).dt.year

    rows = []

    years = sorted(
        temp["Year"].dropna().unique()
    )

    for year in years:

        subset = temp.loc[
            temp["Year"] != year
        ].copy()

        if subset.empty:
            continue

        hypothesis_r = subset[
            "Hypothesis_R"
        ].sum()

        for name, column in [
            (
                "DAILY",
                "Daily_MTF_Aligned",
            ),
            (
                "WEEKLY",
                "Weekly_MTF_Aligned",
            ),
            (
                "FULL",
                "Full_MTF_Aligned_Flag",
            ),
        ]:

            for keep_aligned in [
                True,
                False,
            ]:

                result = evaluate_rule(
                    subset,
                    column,
                    keep_aligned=keep_aligned,
                )

                rows.append(
                    {
                        "Left_Out_Year": int(year),
                        "MTF": name,
                        "Rule": result["rule"],
                        "Trades": len(subset),
                        "Retained_R": result["retained_r"],
                        "Frozen_Hypothesis_R": hypothesis_r,
                        "Delta_vs_Hypothesis": (
                            result["retained_r"]
                            - hypothesis_r
                        ),
                    }
                )

    result_df = pd.DataFrame(rows)

    if not result_df.empty:
        print(
            result_df.to_string(index=False)
        )

    return result_df


# ============================================================
# TRADE-LEVEL OUTPUT
# ============================================================

def build_trade_output(df):
    """
    Build a compact trade-level research output.
    """

    out = df.copy()

    out["7B025_R"] = out["Hypothesis_R"]

    out["Frozen_5B025_Early_Adverse"] = (
        out["Early_Adverse"]
    )

    out["Daily_MTF_Keep_Aligned"] = (
        ~out["Early_Adverse"]
        | (
            out["Early_Adverse"]
            & out["Daily_MTF_Aligned"]
        )
    )

    out["Weekly_MTF_Keep_Aligned"] = (
        ~out["Early_Adverse"]
        | (
            out["Early_Adverse"]
            & out["Weekly_MTF_Aligned"]
        )
    )

    out["Full_MTF_Keep_Aligned"] = (
        ~out["Early_Adverse"]
        | (
            out["Early_Adverse"]
            & out["Full_MTF_Aligned_Flag"]
        )
    )

    columns = [
        "Trade_Key",
        "Research_Symbol",
        "Research_Date",
        "Direction",
        "Setup",
        "R_Multiple",
        "Bar5_Early_Adverse_R",
        "Early_Adverse",
        "Hypothesis_R",
        "7B025_R",
        "Daily_Aligned",
        "Weekly_Aligned",
        "Full_MTF_Aligned",
        "Completed_Weekly_Date",
        "Daily_MTF_Aligned",
        "Weekly_MTF_Aligned",
        "Full_MTF_Aligned_Flag",
        "Daily_MTF_Keep_Aligned",
        "Weekly_MTF_Keep_Aligned",
        "Full_MTF_Keep_Aligned",
    ]

    columns = [
        c for c in columns
        if c in out.columns
    ]

    return out[columns].copy()


# ============================================================
# SUMMARY OUTPUT
# ============================================================

def build_summary(
    reconciliation,
    core_results,
    event_results,
):
    rows = []

    rows.append(
        {
            "section": "FROZEN_RECONCILIATION",
            "metric": "baseline_r",
            "value": reconciliation[
                "baseline_r"
            ],
        }
    )

    rows.append(
        {
            "section": "FROZEN_RECONCILIATION",
            "metric": "5B025_event_count",
            "value": reconciliation[
                "event_count"
            ],
        }
    )

    rows.append(
        {
            "section": "FROZEN_RECONCILIATION",
            "metric": "5B025_retained_count",
            "value": reconciliation[
                "retained_count"
            ],
        }
    )

    rows.append(
        {
            "section": "FROZEN_RECONCILIATION",
            "metric": "5B025_hypothesis_r",
            "value": reconciliation[
                "hypothesis_r"
            ],
        }
    )

    rows.append(
        {
            "section": "FROZEN_RECONCILIATION",
            "metric": "removed_r",
            "value": reconciliation[
                "removed_r"
            ],
        }
    )

    if not core_results.empty:

        for _, row in core_results.iterrows():

            rows.append(
                {
                    "section": "CORE_MTF",
                    "metric": (
                        f"{row['test_name']} "
                        f"{row['rule']} "
                        f"retained_r"
                    ),
                    "value": row[
                        "retained_r"
                    ],
                }
            )

            rows.append(
                {
                    "section": "CORE_MTF",
                    "metric": (
                        f"{row['test_name']} "
                        f"{row['rule']} "
                        f"delta_vs_frozen_hypothesis"
                    ),
                    "value": row[
                        "delta_vs_frozen_hypothesis"
                    ],
                }
            )

    if not event_results.empty:

        for _, row in event_results.iterrows():

            rows.append(
                {
                    "section": "EVENT_ECONOMICS",
                    "metric": (
                        f"{row['group']} "
                        f"event_r"
                    ),
                    "value": row[
                        "event_r"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("MTF × FROZEN 5B/.25R RETENTION ECONOMICS")
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    frozen = load_frozen()

    mtf = load_mtf()

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    df = merge_data(
        frozen,
        mtf,
    )

    # --------------------------------------------------------
    # CAUSALITY
    # --------------------------------------------------------

    validate_causality(df)

    # --------------------------------------------------------
    # BUILD AUTHORITATIVE FROZEN RULE
    # --------------------------------------------------------

    df = build_early_adverse(df)

    df = build_hypothesis(df)

    df = build_mtf_flags(df)

    # --------------------------------------------------------
    # RECONCILIATION
    # --------------------------------------------------------

    reconciliation = (
        print_early_adverse_reconciliation(df)
    )

    # --------------------------------------------------------
    # CORE TESTS
    # --------------------------------------------------------

    core_results = run_core_tests(df)

    # --------------------------------------------------------
    # EVENT ECONOMICS
    # --------------------------------------------------------

    event_results = event_economics(df)

    # --------------------------------------------------------
    # SETUP CONDITIONAL
    # --------------------------------------------------------

    setup_results = setup_conditional(df)

    # --------------------------------------------------------
    # SYMBOL LOO
    # --------------------------------------------------------

    symbol_loo_results = symbol_leave_one_out(
        df
    )

    # --------------------------------------------------------
    # YEAR LOO
    # --------------------------------------------------------

    year_loo_results = year_leave_one_out(
        df
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    TRADE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRADE-LEVEL OUTPUT
    # --------------------------------------------------------

    trade_output = build_trade_output(df)

    trade_output.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY OUTPUT
    # --------------------------------------------------------

    summary = build_summary(
        reconciliation,
        core_results,
        event_results,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # ADDITIONAL FORENSIC OUTPUTS
    # --------------------------------------------------------

    setup_output = (
        TRADE_OUTPUT.parent
        / "mtf_7b025_retention_setup_conditional.csv"
    )

    symbol_output = (
        TRADE_OUTPUT.parent
        / "mtf_7b025_retention_symbol_loo.csv"
    )

    year_output = (
        TRADE_OUTPUT.parent
        / "mtf_7b025_retention_year_loo.csv"
    )

    event_output = (
        TRADE_OUTPUT.parent
        / "mtf_7b025_retention_event_economics.csv"
    )

    setup_results.to_csv(
        setup_output,
        index=False,
    )

    symbol_loo_results.to_csv(
        symbol_output,
        index=False,
    )

    year_loo_results.to_csv(
        year_output,
        index=False,
    )

    event_results.to_csv(
        event_output,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL ACCOUNTING CHECKS
    # --------------------------------------------------------

    baseline_r = df["R_Multiple"].sum()

    hypothesis_r = df["Hypothesis_R"].sum()

    event_r = df.loc[
        df["Early_Adverse"],
        "R_Multiple",
    ].sum()

    retained_r = df.loc[
        ~df["Early_Adverse"],
        "R_Multiple",
    ].sum()

    # Hypothesis should equal retained trades.
    if not np.isclose(
        hypothesis_r,
        retained_r,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Final frozen hypothesis accounting FAILED."
        )

    # Baseline must equal retained + rejected event R.
    if not np.isclose(
        baseline_r,
        retained_r + event_r,
        atol=1e-9,
    ):
        raise RuntimeError(
            "Final R conservation check FAILED."
        )

    print()
    print("=" * 72)
    print("FINAL ACCOUNTING")
    print("=" * 72)

    print(
        f"Baseline R                 : "
        f"{baseline_r:+.2f}R"
    )

    print(
        f"5B/.25R rejected-event R   : "
        f"{event_r:+.2f}R"
    )

    print(
        f"5B/.25R retained R         : "
        f"{retained_r:+.2f}R"
    )

    print(
        f"5B/.25R hypothesis R       : "
        f"{hypothesis_r:+.2f}R"
    )

    print()
    print(
        "Frozen hypothesis accounting: PASS"
    )

    print(
        "R conservation: PASS"
    )

    # --------------------------------------------------------
    # HISTORICAL +7.10 NOTE
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("HISTORICAL RESEARCH-VALUE NOTE")
    print("=" * 72)

    print(
        "Historical research records contain "
        "+7.10R for an early-adverse hypothesis."
    )

    print(
        "This script does NOT force that value."
    )

    print(
        "The current frozen CSV is classified "
        "directly using the authoritative "
        "Bar-5 >= 0.25R rule."
    )

    print(
        "Any difference from +7.10R is a "
        "historical ledger/version reconciliation "
        "issue and must be investigated separately."
    )

    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(
        f"Trade-level : {TRADE_OUTPUT}"
    )

    print(
        f"Summary     : {SUMMARY_OUTPUT}"
    )

    print(
        f"Setup       : {setup_output}"
    )

    print(
        f"Symbol LOO  : {symbol_output}"
    )

    print(
        f"Year LOO    : {year_output}"
    )

    print(
        f"Events      : {event_output}"
    )

    print()
    print("=" * 72)
    print("MTF_5B025_RETENTION_COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()