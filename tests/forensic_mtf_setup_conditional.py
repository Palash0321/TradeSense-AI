"""
TradeSense-AI
MTF Setup-Conditional Forensic

Question
--------
Does causally reconstructed Daily + completed Weekly MTF information
add useful information CONDITIONAL on the existing TradeSense setup?

Authoritative outcome
---------------------
R_Multiple comes directly from the frozen trade ledger.

Historical MTF components
-------------------------
Only:
    - Daily
    - Completed Weekly

We do NOT fabricate historical 15m / 1h / 4h data.

Causal contract
---------------
Signal_Date = strategy signal date.

Daily:
    Uses the completed daily candle on Signal_Date.
    The trade enters on the next bar, so the Signal_Date close
    is available before execution.

Weekly:
    Uses the latest completed W-FRI candle STRICTLY BEFORE Signal_Date.
    Therefore the current developing week is never used.

No production files are modified.
No frozen trade results are recomputed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRICE_DIR = ROOT / "data" / "nifty50_prices"

FROZEN_TRADES = (
    ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

MTF_COMPONENT_FILE = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_component_contribution_trade_level.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRADE_OUTPUT = (
    OUTPUT_DIR
    / "mtf_setup_conditional_trade_level.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "mtf_setup_conditional_summary.csv"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_symbol(symbol):
    if pd.isna(symbol):
        return None

    value = str(symbol).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def profit_factor(values):
    values = pd.Series(values).dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return gross_profit / gross_loss


def load_frozen_trades():

    trades = pd.read_csv(FROZEN_TRADES)

    symbol_column = None

    for candidate in [
        "_symbol",
        "Symbol",
        "symbol",
    ]:
        if candidate in trades.columns:
            symbol_column = candidate
            break

    if symbol_column is None:
        raise ValueError(
            "Frozen ledger has no recognized symbol column."
        )

    date_column = None

    for candidate in [
        "Signal_Date",
        "signal_date",
    ]:
        if candidate in trades.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Frozen ledger has no Signal_Date column."
        )

    r_column = None

    for candidate in [
        "R_Multiple",
        "r_multiple",
        "R",
    ]:
        if candidate in trades.columns:
            r_column = candidate
            break

    if r_column is None:
        raise ValueError(
            "Frozen ledger has no authoritative R_Multiple column."
        )

    direction_column = None

    for candidate in [
        "Direction",
        "direction",
    ]:
        if candidate in trades.columns:
            direction_column = candidate
            break

    if direction_column is None:
        raise ValueError(
            "Frozen ledger has no Direction column."
        )

    trades = trades.copy()

    trades["_symbol_root"] = (
        trades[symbol_column]
        .map(normalize_symbol)
    )

    trades["Signal_Date"] = pd.to_datetime(
        trades[date_column],
        errors="coerce",
    )

    trades["R_Multiple_Authoritative"] = pd.to_numeric(
        trades[r_column],
        errors="coerce",
    )

    trades["_direction"] = (
        trades[direction_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Setup identification
    # --------------------------------------------------------
    #
    # The frozen ledger normally contains the setup directly.
    # We support the known setup column variants rather than
    # reconstructing setup logic.
    # --------------------------------------------------------

    setup_column = None

    for candidate in [
        "Setup",
        "setup",
        "Setup_Type",
        "setup_type",
        "SetupType",
        "setup_name",
    ]:
        if candidate in trades.columns:
            setup_column = candidate
            break

    if setup_column is None:

        # Some frozen ledgers may encode setup through the
        # known Setup/Direction naming convention.
        #
        # Do NOT silently invent setup classifications.
        raise ValueError(
            "Frozen ledger does not contain a recognized setup column. "
            "Expected one of: Setup, setup, Setup_Type, setup_type, "
            "SetupType, setup_name."
        )

    trades["_setup"] = (
        trades[setup_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    trades["_trade_key"] = (
        trades["_symbol_root"].astype(str)
        + "|"
        + trades["Signal_Date"].dt.strftime("%Y-%m-%d")
        + "|"
        + trades["_direction"]
    )

    return trades


# ============================================================
# LOAD EXISTING CAUSALLY VALID MTF COMPONENT DATA
# ============================================================

def load_mtf_components():

    if not MTF_COMPONENT_FILE.exists():
        raise FileNotFoundError(
            "Expected component forensic output does not exist:\n"
            f"{MTF_COMPONENT_FILE}\n\n"
            "Run forensic_mtf_component_contribution.py first."
        )

    mtf = pd.read_csv(
        MTF_COMPONENT_FILE
    )

    required = [
        "Symbol",
        "Signal_Date",
        "Daily_State",
        "Weekly_State",
        "Completed_Weekly_Date",
        "Daily_Relation",
        "Weekly_Relation",
        "Combined_MTF_Class",
    ]

    missing = [
        column
        for column in required
        if column not in mtf.columns
    ]

    if missing:
        raise ValueError(
            "MTF component file is missing columns: "
            + ", ".join(missing)
        )

    mtf = mtf.copy()

    mtf["Symbol"] = (
        mtf["Symbol"]
        .map(normalize_symbol)
    )

    mtf["Signal_Date"] = pd.to_datetime(
        mtf["Signal_Date"],
        errors="coerce",
    )

    mtf["Completed_Weekly_Date"] = pd.to_datetime(
        mtf["Completed_Weekly_Date"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # The component forensic produces one MTF state per
    # Symbol + Signal_Date.
    #
    # The frozen trade ledger additionally contains Direction.
    # Direction is therefore NOT part of the MTF join key.
    # --------------------------------------------------------

    mtf["_mtf_key"] = (
        mtf["Symbol"].astype(str)
        + "|"
        + mtf["Signal_Date"].dt.strftime("%Y-%m-%d")
    )

    duplicate_keys = (
        mtf["_mtf_key"]
        .duplicated()
        .sum()
    )

    if duplicate_keys:
        raise ValueError(
            "Duplicate Symbol + Signal_Date keys found "
            "in MTF component file."
        )

    return mtf


# ============================================================
# EARLY-ADVERSE EVENT
# ============================================================

def detect_early_adverse_event(row):
    """
    Reconstruct only the frozen 7-bar / 0.25R event if the
    required frozen diagnostic columns are available.

    IMPORTANT:
    This function uses the already-authoritative frozen
    trade-level diagnostic columns where available.

    We do NOT recalculate trade outcomes.
    """

    # --------------------------------------------------------
    # Preferred frozen diagnostic:
    # early_adverse_trigger_bar
    # --------------------------------------------------------

    candidates = [
        "early_adverse_trigger_bar",
        "Early_Adverse_Trigger_Bar",
        "early_adverse_bar",
        "Early_Adverse_Bar",
    ]

    for column in candidates:
        if column in row.index:

            value = row[column]

            if pd.isna(value):
                return False

            try:
                value = float(value)

                return (
                    value >= 1
                    and value <= 7
                )

            except (TypeError, ValueError):
                return False

    # --------------------------------------------------------
    # Secondary diagnostic:
    # explicit trigger boolean
    # --------------------------------------------------------

    for column in [
        "early_adverse_triggered",
        "Early_Adverse_Triggered",
        "early_adverse_event",
        "Early_Adverse_Event",
    ]:

        if column in row.index:

            value = row[column]

            if pd.isna(value):
                return False

            if isinstance(value, bool):
                return value

            text = str(value).strip().upper()

            if text in [
                "TRUE",
                "1",
                "YES",
                "Y",
                "TRIGGERED",
            ]:
                return True

            if text in [
                "FALSE",
                "0",
                "NO",
                "N",
                "NOT_TRIGGERED",
            ]:
                return False

    return np.nan


# ============================================================
# GROUP SUMMARY
# ============================================================

def summarize(
    df,
    dimension,
    group_column,
):

    rows = []

    for group_value, group in df.groupby(
        group_column,
        dropna=False,
    ):

        r = group[
            "R_Multiple_Authoritative"
        ].dropna()

        n = len(r)

        if n == 0:
            continue

        wins = int(
            (r > 0).sum()
        )

        losses = int(
            (r < 0).sum()
        )

        rows.append(
            {
                "Dimension": dimension,
                "Group": str(group_value),
                "Trades": n,
                "Wins": wins,
                "Losses": losses,
                "Win_Rate_%": round(
                    wins / n * 100,
                    2,
                ),
                "Profit_Factor": round(
                    profit_factor(r),
                    4,
                ),
                "Total_R": round(
                    r.sum(),
                    4,
                ),
                "Avg_R": round(
                    r.mean(),
                    4,
                ),
            }
        )

    return rows


# ============================================================
# CONDITIONAL COMPARISON
# ============================================================

def print_binary_comparison(
    df,
    label,
    group_column,
):

    print()
    print(label)

    groups = {}

    for group_value, group in df.groupby(
        group_column,
        dropna=False,
    ):

        r = group[
            "R_Multiple_Authoritative"
        ].dropna()

        if len(r) == 0:
            continue

        groups[str(group_value)] = {
            "n": len(r),
            "win": (r > 0).mean() * 100,
            "pf": profit_factor(r),
            "r": r.sum(),
            "avg": r.mean(),
        }

        print(
            f"  {str(group_value):18s} "
            f"N={len(r):3d} "
            f"Win={((r > 0).mean() * 100):6.2f}% "
            f"PF={profit_factor(r):6.3f} "
            f"R={r.sum():8.2f} "
            f"Avg={r.mean():7.4f}"
        )

    if (
        "ALIGNED" in groups
        and "CONFLICT" in groups
    ):

        aligned = groups["ALIGNED"]
        conflict = groups["CONFLICT"]

        print(
            f"  DELTA ALIGNED-CONFLICT: "
            f"Win={aligned['win'] - conflict['win']:+.2f}pp "
            f"R={aligned['r'] - conflict['r']:+.2f} "
            f"Avg={aligned['avg'] - conflict['avg']:+.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MTF SETUP-CONDITIONAL FORENSIC")
    print("=" * 70)

    trades = load_frozen_trades()

    mtf = load_mtf_components()

    print(
        f"Frozen trades loaded : {len(trades)}"
    )

    print(
        f"MTF component rows   : {len(mtf)}"
    )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    frozen_duplicates = (
        trades["_trade_key"]
        .duplicated()
        .sum()
    )

    mtf_duplicates = (
        mtf["_trade_key"]
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
        raise ValueError(
            "Duplicate frozen trade keys detected."
        )

    if mtf_duplicates:
        raise ValueError(
            "Duplicate MTF trade keys detected."
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    # --------------------------------------------------------
    # MTF join key
    # --------------------------------------------------------
    #
    # MTF state is defined by:
    #     Symbol + Signal_Date
    #
    # Frozen trade identity additionally includes Direction.
    #
    # Therefore Direction must NOT be part of the MTF lookup key.
    # --------------------------------------------------------

    trades["_mtf_key"] = (
        trades["_symbol_root"].astype(str)
        + "|"
        + trades["Signal_Date"].dt.strftime("%Y-%m-%d")
    )

    analysis = trades.merge(
        mtf[
            [
                "_mtf_key",
                "Daily_State",
                "Weekly_State",
                "Completed_Weekly_Date",
                "Daily_Relation",
                "Weekly_Relation",
                "Combined_MTF_Class",
            ]
        ],
        on="_mtf_key",
        how="left",
        validate="one_to_one",
    )

    missing_mtf = (
        analysis["Daily_State"].isna()
        | analysis["Weekly_State"].isna()
    )

    print(
        f"Missing MTF rows     : "
        f"{int(missing_mtf.sum())}"
    )

    # --------------------------------------------------------
    # Authoritative reconciliation
    # --------------------------------------------------------

    frozen_total_r = (
        trades["R_Multiple_Authoritative"]
        .sum()
    )

    analyzed_total_r = (
        analysis["R_Multiple_Authoritative"]
        .sum()
    )

    print()
    print("=" * 70)
    print("RECONCILIATION")
    print("=" * 70)

    print(
        f"Frozen total R       : {frozen_total_r:.2f}"
    )

    print(
        f"Analyzed total R     : {analyzed_total_r:.2f}"
    )

    print(
        f"R delta              : "
        f"{analyzed_total_r - frozen_total_r:.10f}"
    )

    if abs(
        analyzed_total_r - frozen_total_r
    ) < 1e-9:

        print(
            "R reconciliation     : PASS"
        )

    else:

        print(
            "R reconciliation     : FAIL"
        )

    # --------------------------------------------------------
    # Causal timestamp verification
    # --------------------------------------------------------

    analysis[
        "Weekly_Causal_PASS"
    ] = (
        analysis["Completed_Weekly_Date"].notna()
        & (
            analysis["Completed_Weekly_Date"]
            < analysis["Signal_Date"]
        )
    )

    print()
    print("=" * 70)
    print("CAUSAL TIMESTAMP")
    print("=" * 70)

    print(
        "Weekly causal PASS   : "
        f"{int(analysis['Weekly_Causal_PASS'].sum())}"
        f"/{len(analysis)}"
    )

    if (
        analysis["Weekly_Causal_PASS"].all()
    ):
        print(
            "Weekly timestamp causality: PASS"
        )
    else:
        print(
            "Weekly timestamp causality: FAIL"
        )

    # --------------------------------------------------------
    # Early-adverse classification
    # --------------------------------------------------------

    analysis[
        "Early_Adverse_Event"
    ] = analysis.apply(
        detect_early_adverse_event,
        axis=1,
    )

    known_event = (
        analysis["Early_Adverse_Event"]
        .notna()
    )

    print()
    print("=" * 70)
    print("EARLY-ADVERSE AVAILABILITY")
    print("=" * 70)

    print(
        f"Known event state    : "
        f"{int(known_event.sum())}/{len(analysis)}"
    )

    if known_event.any():

        event_rate = (
            analysis.loc[
                known_event,
                "Early_Adverse_Event"
            ]
            .mean()
            * 100
        )

        print(
            f"Event rate           : "
            f"{event_rate:.2f}%"
        )

    else:

        print(
            "WARNING: no usable frozen early-adverse "
            "diagnostic column was found."
        )

    # --------------------------------------------------------
    # Valid MTF sample
    # --------------------------------------------------------

    valid = analysis[
        ~missing_mtf
    ].copy()

    print()
    print("=" * 70)
    print("VALID MTF SAMPLE")
    print("=" * 70)

    print(
        f"Valid MTF trades     : {len(valid)}"
    )

    # --------------------------------------------------------
    # Component contribution
    # --------------------------------------------------------

    summary_rows = []

    # Daily conditional on setup
    for setup, setup_group in valid.groupby(
        "_setup"
    ):

        subset = setup_group[
            setup_group["Daily_Relation"].isin(
                [
                    "ALIGNED",
                    "CONFLICT",
                    "NEUTRAL",
                ]
            )
        ]

        if len(subset):

            summary_rows.extend(
                summarize(
                    subset,
                    f"SETUP={setup}|DAILY",
                    "Daily_Relation",
                )
            )

    # Weekly conditional on setup
    for setup, setup_group in valid.groupby(
        "_setup"
    ):

        subset = setup_group[
            setup_group["Weekly_Relation"].isin(
                [
                    "ALIGNED",
                    "CONFLICT",
                    "NEUTRAL",
                ]
            )
        ]

        if len(subset):

            summary_rows.extend(
                summarize(
                    subset,
                    f"SETUP={setup}|WEEKLY",
                    "Weekly_Relation",
                )
            )

    # Daily × Weekly conditional on setup
    for setup, setup_group in valid.groupby(
        "_setup"
    ):

        subset = setup_group[
            setup_group["Combined_MTF_Class"].notna()
        ]

        if len(subset):

            summary_rows.extend(
                summarize(
                    subset,
                    f"SETUP={setup}|DAILY_WEEKLY",
                    "Combined_MTF_Class",
                )
            )

    # --------------------------------------------------------
    # Print setup-specific results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SETUP-CONDITIONAL COMPONENT RESULTS")
    print("=" * 70)

    for setup in sorted(
        valid["_setup"].dropna().unique()
    ):

        setup_group = valid[
            valid["_setup"] == setup
        ]

        print()
        print(
            f"SETUP: {setup} "
            f"(N={len(setup_group)})"
        )

        print_binary_comparison(
            setup_group,
            "  DAILY",
            "Daily_Relation",
        )

        print_binary_comparison(
            setup_group,
            "  WEEKLY",
            "Weekly_Relation",
        )

        print_binary_comparison(
            setup_group,
            "  DAILY × WEEKLY",
            "Combined_MTF_Class",
        )

    # --------------------------------------------------------
    # Direction × Setup × MTF
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DIRECTION × SETUP × MTF")
    print("=" * 70)

    direction_setup_rows = []

    for (
        direction,
        setup,
    ), group in valid.groupby(
        [
            "_direction",
            "_setup",
        ]
    ):

        for (
            daily_relation,
            weekly_relation,
        ), subgroup in group.groupby(
            [
                "Daily_Relation",
                "Weekly_Relation",
            ]
        ):

            r = subgroup[
                "R_Multiple_Authoritative"
            ].dropna()

            if len(r) == 0:
                continue

            direction_setup_rows.append(
                {
                    "Direction": direction,
                    "Setup": setup,
                    "Daily_Relation": daily_relation,
                    "Weekly_Relation": weekly_relation,
                    "Trades": len(r),
                    "Win_Rate_%": round(
                        (r > 0).mean() * 100,
                        2,
                    ),
                    "Profit_Factor": round(
                        profit_factor(r),
                        4,
                    ),
                    "Total_R": round(
                        r.sum(),
                        4,
                    ),
                    "Avg_R": round(
                        r.mean(),
                        4,
                    ),
                }
            )

            print(
                f"  {direction:5s} "
                f"{setup:24s} "
                f"D={daily_relation:9s} "
                f"W={weekly_relation:9s} "
                f"N={len(r):3d} "
                f"Win={((r > 0).mean() * 100):6.2f}% "
                f"PF={profit_factor(r):6.3f} "
                f"R={r.sum():8.2f} "
                f"Avg={r.mean():7.4f}"
            )

    # --------------------------------------------------------
    # Early-adverse conditional analysis
    # --------------------------------------------------------

    if known_event.any():

        event_df = valid[
            valid["Early_Adverse_Event"].notna()
        ].copy()

        print()
        print("=" * 70)
        print("SETUP × MTF × EARLY-ADVERSE")
        print("=" * 70)

        event_summary_rows = []

        for (
            setup,
            daily_relation,
            weekly_relation,
        ), group in event_df.groupby(
            [
                "_setup",
                "Daily_Relation",
                "Weekly_Relation",
            ]
        ):

            event_rate = (
                group[
                    "Early_Adverse_Event"
                ].mean()
                * 100
            )

            r = group[
                "R_Multiple_Authoritative"
            ]

            row = {
                "Setup": setup,
                "Daily_Relation": daily_relation,
                "Weekly_Relation": weekly_relation,
                "Trades": len(group),
                "Early_Adverse_Count": int(
                    group[
                        "Early_Adverse_Event"
                    ].sum()
                ),
                "Early_Adverse_Rate_%": round(
                    event_rate,
                    2,
                ),
                "Total_R": round(
                    r.sum(),
                    4,
                ),
                "Avg_R": round(
                    r.mean(),
                    4,
                ),
                "Profit_Factor": round(
                    profit_factor(r),
                    4,
                ),
            }

            event_summary_rows.append(
                row
            )

            print(
                f"  {setup:24s} "
                f"D={daily_relation:9s} "
                f"W={weekly_relation:9s} "
                f"N={len(group):3d} "
                f"Event={event_rate:6.2f}% "
                f"R={r.sum():8.2f} "
                f"Avg={r.mean():7.4f}"
            )

        # ----------------------------------------------------
        # Triggered vs not-triggered within each MTF state
        # ----------------------------------------------------

        print()
        print(
            "MTF STATE × EARLY-ADVERSE EVENT"
        )

        for state_column in [
            "Combined_MTF_Class",
            "Daily_Relation",
            "Weekly_Relation",
        ]:

            print()
            print(
                f"  {state_column}"
            )

            for (
                state,
                event_state,
            ), group in event_df.groupby(
                [
                    state_column,
                    "Early_Adverse_Event",
                ]
            ):

                r = group[
                    "R_Multiple_Authoritative"
                ]

                print(
                    f"    {str(state):24s} "
                    f"Event={str(event_state):5s} "
                    f"N={len(group):3d} "
                    f"Win={((r > 0).mean() * 100):6.2f}% "
                    f"PF={profit_factor(r):6.3f} "
                    f"R={r.sum():8.2f} "
                    f"Avg={r.mean():7.4f}"
                )

    else:

        event_summary_rows = []

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    analysis.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if direction_setup_rows:

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    direction_setup_rows
                ),
            ],
            ignore_index=True,
        )

    if event_summary_rows:

        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    event_summary_rows
                ),
            ],
            ignore_index=True,
        )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(
        f"Trade-level : {TRADE_OUTPUT}"
    )

    print(
        f"Summary     : {SUMMARY_OUTPUT}"
    )

    print()
    print(
        "STATUS: MTF_SETUP_CONDITIONAL_COMPLETE"
    )


if __name__ == "__main__":
    main()