"""
TradeSense-AI
MTF × Early-Adverse Interaction Forensic

PURPOSE
-------
Determine whether Daily + Weekly MTF context adds predictive information
BEYOND the already-investigated early-adverse behavior.

Research only.
No production changes.
No frozen-data changes.
No OOS changes.

Questions tested
----------------
1. Does MTF differentiate outcomes inside Early-Adverse TRIGGERED?
2. Does MTF differentiate outcomes inside Early-Adverse NOT_TRIGGERED?
3. Does MTF add information after conditioning on direction?
4. Does MTF add information after conditioning on setup?
5. Does the apparent relationship survive symbol/year leave-one-out checks?
6. Is the apparent MTF effect actually concentrated in one problematic setup?
7. Does directional MTF score behave monotonically?
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from forensic_mtf_daily_weekly import (
    load_frozen_trades,
    build_symbol_mtf,
    classify_trades,
)


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_TRADES = 106
EXPECTED_TOTAL_R = -29.15
TOLERANCE = 0.05


# ============================================================================
# PERFORMANCE
# ============================================================================

def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win_Rate_Pct": 0.0,
            "Gross_Profit_R": 0.0,
            "Gross_Loss_R": 0.0,
            "Profit_Factor": 0.0,
            "Total_R": 0.0,
            "Average_R": 0.0,
        }

    r = pd.to_numeric(
        df["R_Multiple"],
        errors="coerce",
    ).dropna()

    trades = len(r)
    wins = int((r > 0).sum())
    losses = int((r <= 0).sum())

    gross_profit = float(
        r[r > 0].sum()
    )

    gross_loss = float(
        r[r < 0].sum()
    )

    if abs(gross_loss) > 1e-12:
        pf = gross_profit / abs(gross_loss)
    else:
        pf = np.inf if gross_profit > 0 else 0.0

    return {
        "Trades": trades,
        "Wins": wins,
        "Losses": losses,
        "Win_Rate_Pct": round(
            100 * wins / trades,
            2,
        ) if trades else 0.0,
        "Gross_Profit_R": round(
            gross_profit,
            2,
        ),
        "Gross_Loss_R": round(
            gross_loss,
            2,
        ),
        "Profit_Factor": (
            round(pf, 4)
            if math.isfinite(pf)
            else np.inf
        ),
        "Total_R": round(
            float(r.sum()),
            2,
        ),
        "Average_R": round(
            float(r.mean()),
            4,
        ) if trades else 0.0,
    }


def grouped_summary(
    df: pd.DataFrame,
    group_columns,
) -> pd.DataFrame:

    rows = []

    for keys, group in df.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            col: value
            for col, value
            in zip(group_columns, keys)
        }

        row.update(
            summarize(group)
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=list(group_columns)
            + [
                "Trades",
                "Wins",
                "Losses",
                "Win_Rate_Pct",
                "Gross_Profit_R",
                "Gross_Loss_R",
                "Profit_Factor",
                "Total_R",
                "Average_R",
            ]
        )

    return pd.DataFrame(rows)


# ============================================================================
# EARLY-ADVERSE NORMALIZATION
# ============================================================================

def normalize_early_adverse(value):
    if pd.isna(value):
        return "UNKNOWN"

    if isinstance(value, bool):
        return (
            "TRIGGERED"
            if value
            else "NOT_TRIGGERED"
        )

    text = str(value).strip().upper()

    if text in {
        "TRUE",
        "1",
        "YES",
        "Y",
        "TRIGGERED",
    }:
        return "TRIGGERED"

    if text in {
        "FALSE",
        "0",
        "NO",
        "N",
        "NOT_TRIGGERED",
    }:
        return "NOT_TRIGGERED"

    return "UNKNOWN"


# ============================================================================
# DATA BUILD
# ============================================================================

def build_dataset():

    print("=" * 90)
    print("TRADESENSE-AI — MTF × EARLY-ADVERSE INTERACTION FORENSIC")
    print("=" * 90)

    trades = load_frozen_trades()

    if len(trades) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} frozen trades, "
            f"found {len(trades)}."
        )

    print(
        f"Frozen trades loaded : {len(trades)}"
    )

    symbols = sorted(
        trades["Symbol_Normalized"]
        .dropna()
        .unique()
    )

    mtf_by_symbol = {}

    for symbol in symbols:

        display_symbol = (
            symbol + ".NS"
            if not symbol.endswith(".NS")
            else symbol
        )

        mtf_by_symbol[symbol] = (
            build_symbol_mtf(display_symbol)
        )

    classified = classify_trades(
        trades,
        mtf_by_symbol,
    )

    if len(classified) != EXPECTED_TRADES:
        raise RuntimeError(
            "Trade count changed during MTF classification."
        )

    frozen_total_r = float(
        classified["R_Multiple"].sum()
    )

    if abs(
        frozen_total_r - EXPECTED_TOTAL_R
    ) > TOLERANCE:

        raise RuntimeError(
            "Frozen R reconciliation failed.\n"
            f"Expected: {EXPECTED_TOTAL_R}\n"
            f"Actual:   {frozen_total_r}"
        )

    valid = classified[
        classified["MTF_Availability"]
        == "AVAILABLE"
    ].copy()

    print(
        f"MTF-valid trades      : {len(valid)}"
    )

    print(
        f"MTF-unavailable       : "
        f"{len(classified) - len(valid)}"
    )

    # ------------------------------------------------------------------------
    # EARLY ADVERSE
    # ------------------------------------------------------------------------

    if "early_adverse_triggered" not in valid.columns:
        raise RuntimeError(
            "Frozen ledger does not contain "
            "'early_adverse_triggered'."
        )

    valid["Early_Adverse_Group"] = (
        valid["early_adverse_triggered"]
        .map(normalize_early_adverse)
    )

    valid["Signal_Year"] = (
        pd.to_datetime(
            valid["Signal_Date"]
        )
        .dt.year
    )

    return classified, valid


# ============================================================================
# PRINT
# ============================================================================

def print_section(
    title: str,
    df: pd.DataFrame,
):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    if df.empty:
        print("No rows.")
        return

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    classified, valid = build_dataset()

    # ========================================================================
    # 1. EARLY-ADVERSE BASELINE
    # ========================================================================

    early_summary = grouped_summary(
        valid,
        ["Early_Adverse_Group"],
    )

    print_section(
        "1. EARLY-ADVERSE BASELINE — MTF AVAILABLE",
        early_summary,
    )

    # ========================================================================
    # 2. EARLY-ADVERSE × MTF
    # ========================================================================

    early_mtf = grouped_summary(
        valid,
        [
            "Early_Adverse_Group",
            "MTF_Class",
        ],
    )

    print_section(
        "2. EARLY-ADVERSE × MTF CLASS",
        early_mtf,
    )

    # ========================================================================
    # 3. EARLY-ADVERSE × MTF SCORE
    # ========================================================================

    score = grouped_summary(
        valid,
        [
            "Early_Adverse_Group",
            "MTF_Score",
        ],
    )

    print_section(
        "3. EARLY-ADVERSE × DIRECTIONAL MTF SCORE",
        score,
    )

    # ========================================================================
    # 4. EARLY-ADVERSE × DIRECTION × MTF
    # ========================================================================

    direction = grouped_summary(
        valid,
        [
            "Early_Adverse_Group",
            "Direction_Normalized",
            "MTF_Class",
        ],
    )

    print_section(
        "4. EARLY-ADVERSE × DIRECTION × MTF",
        direction,
    )

    # ========================================================================
    # 5. EARLY-ADVERSE × SETUP × MTF
    # ========================================================================

    setup = grouped_summary(
        valid,
        [
            "Early_Adverse_Group",
            "Setup_Normalized",
            "MTF_Class",
        ],
    )

    print_section(
        "5. EARLY-ADVERSE × SETUP × MTF",
        setup,
    )

    # ========================================================================
    # 6. YEAR × EARLY-ADVERSE × MTF
    # ========================================================================

    year = grouped_summary(
        valid,
        [
            "Signal_Year",
            "Early_Adverse_Group",
            "MTF_Class",
        ],
    )

    print_section(
        "6. YEAR × EARLY-ADVERSE × MTF",
        year,
    )

    # ========================================================================
    # 7. SYMBOL × EARLY-ADVERSE × MTF
    # ========================================================================

    symbol = grouped_summary(
        valid,
        [
            "Symbol_Normalized",
            "Early_Adverse_Group",
            "MTF_Class",
        ],
    )

    print_section(
        "7. SYMBOL × EARLY-ADVERSE × MTF",
        symbol,
    )

    # ========================================================================
    # 8. SIMPLE MTF GROUPS
    #
    # Positive score = at least one timeframe aligned and none conflicting.
    # Zero/negative = not positively aligned.
    # ========================================================================

    valid["MTF_Positive_Alignment"] = np.select(
        [
            valid["MTF_Score"] == 2,
            valid["MTF_Score"] == 1,
        ],
        [
            "FULL_ALIGNMENT",
            "PARTIAL_ALIGNMENT",
        ],
        default="NOT_POSITIVELY_ALIGNED",
    )

    positive = grouped_summary(
        valid,
        [
            "Early_Adverse_Group",
            "MTF_Positive_Alignment",
        ],
    )

    print_section(
        "8. EARLY-ADVERSE × POSITIVE MTF ALIGNMENT",
        positive,
    )

    # ========================================================================
    # 9. INCREMENTAL EFFECT
    #
    # Compare:
    #   early-adverse alone
    # versus
    #   early-adverse + positive MTF alignment
    #
    # This is descriptive, not a production recommendation.
    # ========================================================================

    incremental_rows = []

    for ea in [
        "TRIGGERED",
        "NOT_TRIGGERED",
    ]:

        ea_group = valid[
            valid["Early_Adverse_Group"]
            == ea
        ]

        aligned = ea_group[
            ea_group["MTF_Positive_Alignment"]
            != "NOT_POSITIVELY_ALIGNED"
        ]

        not_aligned = ea_group[
            ea_group["MTF_Positive_Alignment"]
            == "NOT_POSITIVELY_ALIGNED"
        ]

        a = summarize(aligned)
        n = summarize(not_aligned)

        incremental_rows.append({
            "Early_Adverse_Group": ea,

            "Aligned_Trades": a["Trades"],
            "Aligned_Total_R": a["Total_R"],
            "Aligned_Average_R": a["Average_R"],
            "Aligned_PF": a["Profit_Factor"],

            "Not_Aligned_Trades": n["Trades"],
            "Not_Aligned_Total_R": n["Total_R"],
            "Not_Aligned_Average_R": n["Average_R"],
            "Not_Aligned_PF": n["Profit_Factor"],

            "Delta_Total_R": round(
                a["Total_R"] - n["Total_R"],
                2,
            ),

            "Delta_Average_R": round(
                a["Average_R"]
                - n["Average_R"],
                4,
            ),
        })

    incremental = pd.DataFrame(
        incremental_rows
    )

    print_section(
        "9. INCREMENTAL MTF EFFECT WITHIN EARLY-ADVERSE GROUP",
        incremental,
    )

    # ========================================================================
    # 10. LEAVE-ONE-SYMBOL-OUT
    #
    # Focus on the two major Early-Adverse groups.
    # ========================================================================

    loo_symbol_rows = []

    for symbol_name in sorted(
        valid["Symbol_Normalized"]
        .unique()
    ):

        remaining = valid[
            valid["Symbol_Normalized"]
            != symbol_name
        ]

        for ea in [
            "TRIGGERED",
            "NOT_TRIGGERED",
        ]:

            subset = remaining[
                remaining["Early_Adverse_Group"]
                == ea
            ]

            aligned = subset[
                subset["MTF_Positive_Alignment"]
                != "NOT_POSITIVELY_ALIGNED"
            ]

            not_aligned = subset[
                subset["MTF_Positive_Alignment"]
                == "NOT_POSITIVELY_ALIGNED"
            ]

            a = summarize(aligned)
            n = summarize(not_aligned)

            loo_symbol_rows.append({
                "Excluded_Symbol": symbol_name,
                "Early_Adverse_Group": ea,
                "Aligned_Trades": a["Trades"],
                "Aligned_Total_R": a["Total_R"],
                "Not_Aligned_Trades": n["Trades"],
                "Not_Aligned_Total_R": n["Total_R"],
                "Delta_Total_R": round(
                    a["Total_R"]
                    - n["Total_R"],
                    2,
                ),
            })

    loo_symbol = pd.DataFrame(
        loo_symbol_rows
    )

    print_section(
        "10. LEAVE-ONE-SYMBOL-OUT — EARLY-ADVERSE × MTF",
        loo_symbol,
    )

    # ========================================================================
    # 11. LEAVE-ONE-YEAR-OUT
    # ========================================================================

    loo_year_rows = []

    for year_value in sorted(
        valid["Signal_Year"]
        .dropna()
        .unique()
    ):

        remaining = valid[
            valid["Signal_Year"]
            != year_value
        ]

        for ea in [
            "TRIGGERED",
            "NOT_TRIGGERED",
        ]:

            subset = remaining[
                remaining["Early_Adverse_Group"]
                == ea
            ]

            aligned = subset[
                subset["MTF_Positive_Alignment"]
                != "NOT_POSITIVELY_ALIGNED"
            ]

            not_aligned = subset[
                subset["MTF_Positive_Alignment"]
                == "NOT_POSITIVELY_ALIGNED"
            ]

            a = summarize(aligned)
            n = summarize(not_aligned)

            loo_year_rows.append({
                "Excluded_Year": int(year_value),
                "Early_Adverse_Group": ea,
                "Aligned_Trades": a["Trades"],
                "Aligned_Total_R": a["Total_R"],
                "Not_Aligned_Trades": n["Trades"],
                "Not_Aligned_Total_R": n["Total_R"],
                "Delta_Total_R": round(
                    a["Total_R"]
                    - n["Total_R"],
                    2,
                ),
            })

    loo_year = pd.DataFrame(
        loo_year_rows
    )

    print_section(
        "11. LEAVE-ONE-YEAR-OUT — EARLY-ADVERSE × MTF",
        loo_year,
    )

    # ========================================================================
    # 12. MTF SCORE MONOTONICITY
    # ========================================================================

    score_only = grouped_summary(
        valid,
        ["MTF_Score"],
    )

    print_section(
        "12. MTF SCORE — MONOTONICITY CHECK",
        score_only,
    )

    # ========================================================================
    # 13. IDENTIFY CANDIDATE INTERACTIONS
    # ========================================================================

    candidates = early_mtf[
        early_mtf["Trades"] >= 5
    ].copy()

    candidates = candidates.sort_values(
        "Total_R"
    )

    print_section(
        "13. MTF INTERACTION CELLS WITH AT LEAST 5 TRADES",
        candidates,
    )

    # ========================================================================
    # 14. TRADE-LEVEL OUTPUT
    # ========================================================================

    trade_output = valid[
        [
            "_Frozen_Row",
            "Symbol_Normalized",
            "Signal_Date",
            "Signal_Year",
            "Direction_Normalized",
            "Setup_Normalized",
            "R_Multiple",
            "early_adverse_triggered",
            "Early_Adverse_Group",
            "Daily_State",
            "Weekly_State",
            "Weekly_State_Date",
            "MTF_Class",
            "MTF_Score",
            "MTF_Positive_Alignment",
        ]
    ].copy()

    trade_output = trade_output.sort_values(
        "_Frozen_Row"
    )

    trade_path = (
        OUTPUT_DIR
        / "mtf_early_adverse_trade_level.csv"
    )

    trade_output.to_csv(
        trade_path,
        index=False,
    )

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    files = {
        "early_adverse_mtf": (
            early_mtf,
            "mtf_early_adverse_summary.csv",
        ),
        "score": (
            score,
            "mtf_early_adverse_score.csv",
        ),
        "direction": (
            direction,
            "mtf_early_adverse_direction.csv",
        ),
        "setup": (
            setup,
            "mtf_early_adverse_setup.csv",
        ),
        "year": (
            year,
            "mtf_early_adverse_year.csv",
        ),
        "symbol": (
            symbol,
            "mtf_early_adverse_symbol.csv",
        ),
        "positive": (
            positive,
            "mtf_positive_alignment_summary.csv",
        ),
        "incremental": (
            incremental,
            "mtf_incremental_effect.csv",
        ),
        "loo_symbol": (
            loo_symbol,
            "mtf_loo_symbol.csv",
        ),
        "loo_year": (
            loo_year,
            "mtf_loo_year.csv",
        ),
        "score_only": (
            score_only,
            "mtf_score_summary.csv",
        ),
        "candidates": (
            candidates,
            "mtf_candidate_interactions.csv",
        ),
    }

    for _, (dataframe, filename) in files.items():

        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    # ========================================================================
    # FINAL INTEGRITY
    # ========================================================================

    classified_total_r = float(
        classified["R_Multiple"].sum()
    )

    valid_total_r = float(
        valid["R_Multiple"].sum()
    )

    unavailable_total_r = (
        classified_total_r
        - valid_total_r
    )

    reconciled = (
        valid_total_r
        + unavailable_total_r
    )

    delta = (
        reconciled
        - EXPECTED_TOTAL_R
    )

    print()
    print("=" * 90)
    print("FINAL INTEGRITY")
    print("=" * 90)

    print(
        f"Frozen trades           : "
        f"{len(classified)}"
    )

    print(
        f"MTF-valid trades        : "
        f"{len(valid)}"
    )

    print(
        f"MTF-unavailable trades  : "
        f"{len(classified) - len(valid)}"
    )

    print(
        f"Frozen total R          : "
        f"{classified_total_r:.4f}"
    )

    print(
        f"Valid MTF total R       : "
        f"{valid_total_r:.4f}"
    )

    print(
        f"Unavailable R           : "
        f"{unavailable_total_r:.4f}"
    )

    print(
        f"Reconciled R            : "
        f"{reconciled:.4f}"
    )

    print(
        f"Reconciliation delta    : "
        f"{delta:.4f}"
    )

    if (
        len(classified) != EXPECTED_TRADES
        or abs(delta) > TOLERANCE
    ):
        raise RuntimeError(
            "FINAL INTEGRITY CHECK FAILED."
        )

    print()
    print(
        "Trade-count reconciliation : PASS"
    )

    print(
        "R reconciliation            : PASS"
    )

    print(
        "Production strategy         : UNCHANGED"
    )

    print(
        "MTF production logic        : UNCHANGED"
    )

    print(
        "STATUS: MTF_EARLY_ADVERSE_INTERACTION_COMPLETE"
    )

    print()
    print(
        f"Trade-level output : {trade_path}"
    )

    print(
        f"Output directory   : {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()