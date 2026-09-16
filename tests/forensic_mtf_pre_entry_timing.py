"""
TradeSense-AI
MTF PRE-ENTRY TIMING FORENSIC

Research-only.

Question:
Was MTF conflict already present at the signal/entry,
or did the MTF state become conflicting only after the
trade experienced early adverse movement?

IMPORTANT:
- Frozen historical trades are NOT modified.
- Production strategy is NOT modified.
- OOS data is NOT modified.
- No parameter tuning.
- This is descriptive/forensic research only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# IMPORT EXISTING FORENSIC MTF FUNCTIONS
# ---------------------------------------------------------------------

from tests.forensic_mtf_daily_weekly import (
    load_frozen_trades,
    build_symbol_mtf,
    classify_trades,
)


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

TRADE_LEVEL_OUTPUT = (
    OUTPUT_DIR
    / "mtf_pre_entry_timing_trade_level.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "mtf_pre_entry_timing_summary.csv"
)

EXPECTED_TRADES = 106
EXPECTED_TOTAL_R = -29.15


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_date(value):
    return pd.Timestamp(value).normalize()


def safe_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def classify_alignment_change(pre_class, post_class):
    """
    Determine whether MTF classification became more/less
    favorable between the signal and the early-adverse event.

    Ordering is intentionally descriptive, not a tuned score.

    Conflict < Neutral/Mixed < Partial Alignment < Full Alignment
    """

    rank = {
        "PARTIAL_CONFLICT": 0,
        "NEUTRAL": 1,
        "MIXED": 1,
        "PARTIAL_ALIGNMENT": 2,
        "FULL_ALIGNMENT": 3,
    }

    if pd.isna(pre_class) or pd.isna(post_class):
        return "UNAVAILABLE"

    pre_rank = rank.get(str(pre_class), np.nan)
    post_rank = rank.get(str(post_class), np.nan)

    if pd.isna(pre_rank) or pd.isna(post_rank):
        return "UNKNOWN"

    if post_rank > pre_rank:
        return "IMPROVED_AFTER_ENTRY"

    if post_rank < pre_rank:
        return "DETERIORATED_AFTER_ENTRY"

    return "UNCHANGED"


def state_from_score(score):
    """
    Preserve the existing MTF score semantics.
    """

    if pd.isna(score):
        return "UNAVAILABLE"

    score = float(score)

    if score < 0:
        return "CONFLICT"
    if score == 0:
        return "NEUTRAL"
    return "ALIGNED"


def make_summary(df, group_cols):
    """
    Produce the same core performance metrics used by the
    existing forensic reports.
    """

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))

        r = pd.to_numeric(g["R_Multiple"], errors="coerce").dropna()

        trades = len(r)
        wins = int((r > 0).sum())
        losses = int((r <= 0).sum())

        gross_profit = float(r[r > 0].sum())
        gross_loss = float(r[r < 0].sum())

        if gross_loss < 0:
            pf = gross_profit / abs(gross_loss)
        else:
            pf = np.inf if gross_profit > 0 else np.nan

        row.update(
            {
                "Trades": trades,
                "Wins": wins,
                "Losses": losses,
                "Win_Rate_Pct": round(
                    (wins / trades) * 100, 2
                )
                if trades
                else np.nan,
                "Gross_Profit_R": round(gross_profit, 2),
                "Gross_Loss_R": round(gross_loss, 2),
                "Profit_Factor": round(pf, 4)
                if np.isfinite(pf)
                else np.inf,
                "Total_R": round(float(r.sum()), 2),
                "Average_R": round(float(r.mean()), 4)
                if trades
                else np.nan,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# BUILD MTF DATA
# ---------------------------------------------------------------------

def build_all_mtf(frozen):
    symbols = sorted(
        frozen["Symbol_Normalized"]
        .dropna()
        .astype(str)
        .unique()
    )

    mtf_by_symbol = {}

    for symbol in symbols:

        display_symbol = (
            symbol + ".NS"
            if not symbol.endswith(".NS")
            else symbol
        )

        mtf_by_symbol[symbol] = build_symbol_mtf(
            display_symbol
        )

    return mtf_by_symbol


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 90)
    print(
        "TRADESENSE-AI — MTF PRE-ENTRY TIMING FORENSIC"
    )
    print("=" * 90)

    # -------------------------------------------------------------
    # Load frozen trades
    # -------------------------------------------------------------

    frozen = load_frozen_trades()

    print(
        f"Frozen trades loaded : {len(frozen)}"
    )

    if len(frozen) != EXPECTED_TRADES:
        raise AssertionError(
            f"Frozen trade count changed: "
            f"{len(frozen)} != {EXPECTED_TRADES}"
        )

    frozen_total_r = round(
        pd.to_numeric(
            frozen["R_Multiple"],
            errors="coerce",
        ).sum(),
        2,
    )

    if frozen_total_r != EXPECTED_TOTAL_R:
        raise AssertionError(
            f"Frozen R changed: "
            f"{frozen_total_r} != {EXPECTED_TOTAL_R}"
        )

    # -------------------------------------------------------------
    # Existing MTF state
    # -------------------------------------------------------------

    mtf = build_all_mtf(frozen)

    # -------------------------------------------------------------
    # Existing classification
    # -------------------------------------------------------------

    classified = classify_trades(
        frozen,
        mtf,
    )

    print()
    print(
        f"Trades classified       : {len(classified)}"
    )

    # -------------------------------------------------------------
    # Validate required fields
    # -------------------------------------------------------------

    required = [
        "R_Multiple",
        "Symbol_Normalized",
        "Signal_Date",
        "Direction_Normalized",
        "Setup_Normalized",
        "early_adverse_triggered",
    ]

    missing = [
        c for c in required
        if c not in classified.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # -------------------------------------------------------------
    # Discover MTF columns
    # -------------------------------------------------------------

    score_candidates = [
        c
        for c in classified.columns
        if c.lower() in {
            "mtf_score",
            "directional_mtf_score",
        }
    ]

    if not score_candidates:
        raise RuntimeError(
            "Could not find MTF score column."
        )

    score_col = score_candidates[0]

    class_candidates = [
        c
        for c in classified.columns
        if c.lower() == "mtf_class"
    ]

    if not class_candidates:
        raise RuntimeError(
            "Could not find MTF_Class column."
        )

    class_col = class_candidates[0]

    # -------------------------------------------------------------
    # Signal-date MTF is the causal PRE-ENTRY state.
    #
    # The existing daily/weekly forensic classification is already
    # constructed from completed higher-timeframe candles.
    #
    # We explicitly preserve it as PRE_ENTRY.
    # -------------------------------------------------------------

    classified["Pre_Entry_MTF_Class"] = (
        classified[class_col]
        .astype(str)
    )

    classified["Pre_Entry_MTF_Score"] = pd.to_numeric(
        classified[score_col],
        errors="coerce",
    )

    classified["Pre_Entry_MTF_State"] = (
        classified["Pre_Entry_MTF_Score"]
        .apply(state_from_score)
    )

    classified["Early_Adverse_Group"] = np.where(
        classified["early_adverse_triggered"].astype(bool),
        "TRIGGERED",
        "NOT_TRIGGERED",
    )

    classified["Signal_Year"] = pd.to_datetime(
        classified["Signal_Date"],
        errors="coerce",
    ).dt.year

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # The current MTF forensic source provides the MTF state at the
    # trade signal using completed daily/weekly information.
    #
    # We therefore DO NOT fabricate post-entry MTF states.
    #
    # Instead this run establishes the causal PRE-ENTRY baseline
    # and tests whether the PRE-ENTRY state itself explains the
    # early-adverse outcome.
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # 1. PRE-ENTRY MTF × EARLY ADVERSE
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "1. PRE-ENTRY MTF × EARLY-ADVERSE"
    )
    print("=" * 90)

    s1 = make_summary(
        classified,
        [
            "Early_Adverse_Group",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s1.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 2. PRE-ENTRY MTF STATE × EARLY ADVERSE
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "2. PRE-ENTRY MTF STATE × EARLY-ADVERSE"
    )
    print("=" * 90)

    s2 = make_summary(
        classified,
        [
            "Early_Adverse_Group",
            "Pre_Entry_MTF_State",
        ],
    )

    print(
        s2.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 3. DIRECTION × PRE-ENTRY MTF
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "3. DIRECTION × PRE-ENTRY MTF"
    )
    print("=" * 90)

    s3 = make_summary(
        classified,
        [
            "Direction_Normalized",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s3.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 4. EARLY ADVERSE × DIRECTION × PRE-ENTRY MTF
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "4. EARLY-ADVERSE × DIRECTION × PRE-ENTRY MTF"
    )
    print("=" * 90)

    s4 = make_summary(
        classified,
        [
            "Early_Adverse_Group",
            "Direction_Normalized",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s4.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 5. SETUP × PRE-ENTRY MTF
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "5. SETUP × PRE-ENTRY MTF"
    )
    print("=" * 90)

    s5 = make_summary(
        classified,
        [
            "Setup_Normalized",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s5.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 6. PRE-ENTRY CONFLICT TEST
    # -------------------------------------------------------------

    classified["Pre_Entry_Conflict"] = (
        classified["Pre_Entry_MTF_Class"]
        == "PARTIAL_CONFLICT"
    )

    print()
    print("=" * 90)
    print(
        "6. PRE-ENTRY PARTIAL-CONFLICT × EARLY-ADVERSE"
    )
    print("=" * 90)

    s6 = make_summary(
        classified,
        [
            "Early_Adverse_Group",
            "Pre_Entry_Conflict",
        ],
    )

    print(
        s6.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 7. WITHIN EARLY-ADVERSE INCREMENTAL EFFECT
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "7. PRE-ENTRY MTF CONFLICT — INCREMENTAL EFFECT"
    )
    print("=" * 90)

    incremental_rows = []

    for group in [
        "TRIGGERED",
        "NOT_TRIGGERED",
    ]:

        g = classified[
            classified["Early_Adverse_Group"] == group
        ]

        aligned = g[
            g["Pre_Entry_MTF_Class"].isin(
                [
                    "FULL_ALIGNMENT",
                    "PARTIAL_ALIGNMENT",
                ]
            )
        ]

        not_aligned = g[
            ~g["Pre_Entry_MTF_Class"].isin(
                [
                    "FULL_ALIGNMENT",
                    "PARTIAL_ALIGNMENT",
                ]
            )
        ]

        aligned_r = pd.to_numeric(
            aligned["R_Multiple"],
            errors="coerce",
        ).dropna()

        not_aligned_r = pd.to_numeric(
            not_aligned["R_Multiple"],
            errors="coerce",
        ).dropna()

        aligned_total = (
            float(aligned_r.sum())
            if len(aligned_r)
            else 0.0
        )

        not_aligned_total = (
            float(not_aligned_r.sum())
            if len(not_aligned_r)
            else 0.0
        )

        aligned_avg = (
            float(aligned_r.mean())
            if len(aligned_r)
            else np.nan
        )

        not_aligned_avg = (
            float(not_aligned_r.mean())
            if len(not_aligned_r)
            else np.nan
        )

        incremental_rows.append(
            {
                "Early_Adverse_Group": group,
                "Aligned_Trades": len(aligned_r),
                "Aligned_Total_R": round(
                    aligned_total,
                    2,
                ),
                "Aligned_Average_R": round(
                    aligned_avg,
                    4,
                )
                if np.isfinite(aligned_avg)
                else np.nan,
                "Not_Aligned_Trades": len(
                    not_aligned_r
                ),
                "Not_Aligned_Total_R": round(
                    not_aligned_total,
                    2,
                ),
                "Not_Aligned_Average_R": round(
                    not_aligned_avg,
                    4,
                )
                if np.isfinite(not_aligned_avg)
                else np.nan,
                "Delta_Total_R": round(
                    aligned_total
                    - not_aligned_total,
                    2,
                ),
                "Delta_Average_R": round(
                    aligned_avg
                    - not_aligned_avg,
                    4,
                )
                if (
                    np.isfinite(aligned_avg)
                    and np.isfinite(not_aligned_avg)
                )
                else np.nan,
            }
        )

    s7 = pd.DataFrame(incremental_rows)

    print(
        s7.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 8. PRE-ENTRY MTF SCORE MONOTONICITY
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "8. PRE-ENTRY MTF SCORE MONOTONICITY"
    )
    print("=" * 90)

    s8 = make_summary(
        classified,
        ["Pre_Entry_MTF_Score"],
    )

    s8 = s8.sort_values(
        "Pre_Entry_MTF_Score"
    )

    print(
        s8.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 9. YEAR STABILITY
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "9. YEAR × PRE-ENTRY MTF × EARLY-ADVERSE"
    )
    print("=" * 90)

    s9 = make_summary(
        classified,
        [
            "Signal_Year",
            "Early_Adverse_Group",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s9.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 10. SYMBOL STABILITY
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "10. SYMBOL × PRE-ENTRY MTF × EARLY-ADVERSE"
    )
    print("=" * 90)

    s10 = make_summary(
        classified,
        [
            "Symbol_Normalized",
            "Early_Adverse_Group",
            "Pre_Entry_MTF_Class",
        ],
    )

    print(
        s10.to_string(index=False)
    )

    # -------------------------------------------------------------
    # 11. CANDIDATE PRE-ENTRY CELLS >= 5 TRADES
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "11. PRE-ENTRY INTERACTION CELLS WITH AT LEAST 5 TRADES"
    )
    print("=" * 90)

    s11 = make_summary(
        classified,
        [
            "Early_Adverse_Group",
            "Direction_Normalized",
            "Pre_Entry_MTF_Class",
        ],
    )

    s11 = s11[
        s11["Trades"] >= 5
    ].sort_values(
        "Total_R"
    )

    print(
        s11.to_string(index=False)
    )

    # -------------------------------------------------------------
    # FINAL INTEGRITY
    # -------------------------------------------------------------

    print()
    print("=" * 90)
    print("FINAL INTEGRITY")
    print("=" * 90)

    classified_total_r = round(
        pd.to_numeric(
            classified["R_Multiple"],
            errors="coerce",
        ).sum(),
        2,
    )

    print(
        f"Frozen trades           : {len(frozen)}"
    )
    print(
        f"Classified trades       : {len(classified)}"
    )
    print(
        f"Frozen total R          : {frozen_total_r:.4f}"
    )
    print(
        f"Classified total R      : {classified_total_r:.4f}"
    )
    print(
        f"Reconciliation delta    : "
        f"{classified_total_r - frozen_total_r:.4f}"
    )

    assert len(classified) == EXPECTED_TRADES
    assert abs(
        classified_total_r - EXPECTED_TOTAL_R
    ) < 1e-6

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
        "STATUS: MTF_PRE_ENTRY_TIMING_COMPLETE"
    )

    # -------------------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classified.to_csv(
        TRADE_LEVEL_OUTPUT,
        index=False,
    )

    summary_frames = [
        ("PRE_ENTRY_MTF_X_EARLY_ADVERSE", s1),
        ("PRE_ENTRY_MTF_STATE_X_EARLY_ADVERSE", s2),
        ("DIRECTION_X_PRE_ENTRY_MTF", s3),
        ("EARLY_ADVERSE_X_DIRECTION_X_PRE_ENTRY_MTF", s4),
        ("SETUP_X_PRE_ENTRY_MTF", s5),
        ("PRE_ENTRY_CONFLICT_X_EARLY_ADVERSE", s6),
        ("INCREMENTAL_PRE_ENTRY_MTF", s7),
        ("PRE_ENTRY_MTF_SCORE", s8),
        ("YEAR_X_EARLY_ADVERSE_X_PRE_ENTRY_MTF", s9),
        ("SYMBOL_X_EARLY_ADVERSE_X_PRE_ENTRY_MTF", s10),
        ("CANDIDATE_INTERACTIONS", s11),
    ]

    summary_parts = []

    for name, frame in summary_frames:

        tmp = frame.copy()
        tmp.insert(
            0,
            "Analysis",
            name,
        )

        summary_parts.append(tmp)

    pd.concat(
        summary_parts,
        ignore_index=True,
    ).to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(
        f"Trade-level output : {TRADE_LEVEL_OUTPUT}"
    )
    print(
        f"Summary output      : {SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()