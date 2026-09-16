"""
TradeSense-AI
MTF Incremental Counterfactual Forensic

RESEARCH-ONLY.

Purpose
-------
Test whether PRE-ENTRY Daily + Weekly MTF information adds
incremental information beyond the frozen 7-bar / 0.25R
early-adverse hypothesis.

This script does NOT:
    - modify production
    - modify the frozen ledger
    - modify OOS data
    - tune the 7B / 0.25R hypothesis
    - create a deployable trading rule

Important:
    Early-adverse is a POST-ENTRY event.
    Therefore combined early-adverse × MTF results are
    diagnostic/counterfactual research, not a pre-entry filter.

Authoritative outcome:
    frozen tradesense_trades.csv -> r_multiple

Authoritative MTF classification:
    mtf_pre_entry_timing_trade_level.csv
"""


from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MTF_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_pre_entry_timing_trade_level.csv"
)

FROZEN_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "mtf_incremental_counterfactual_summary.csv"
)

TRADE_OUTPUT_FILE = (
    OUTPUT_DIR
    / "mtf_incremental_counterfactual_trade_level.csv"
)

MIN_CELL_TRADES = 5


# ============================================================================
# HELPERS
# ============================================================================

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

    return str(value).strip().upper()


def safe_pf(series):

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    gross_profit = series[
        series > 0
    ].sum()

    gross_loss = -series[
        series < 0
    ].sum()

    if gross_loss == 0:

        if gross_profit > 0:
            return np.inf

        return np.nan

    return gross_profit / gross_loss


def classify_alignment(mtf_state):

    if pd.isna(mtf_state):
        return "UNAVAILABLE"

    state = (
        str(mtf_state)
        .strip()
        .upper()
    )

    if state == "ALIGNED":
        return "ALIGNED"

    if state in {
        "CONFLICT",
        "NEUTRAL",
    }:
        return "NOT_ALIGNED"

    return "UNAVAILABLE"


def summarize(df, group_columns):

    if df.empty:
        return pd.DataFrame()

    rows = []

    grouped = df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for keys, group in grouped:

        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}

        for column, value in zip(
            group_columns,
            keys,
        ):
            row[column] = value

        r = pd.to_numeric(
            group["r_multiple"],
            errors="coerce",
        ).dropna()

        trades = len(r)

        wins = int(
            (r > 0).sum()
        )

        losses = int(
            (r <= 0).sum()
        )

        total_r = float(
            r.sum()
        )

        row.update(
            {
                "Trades": trades,
                "Wins": wins,
                "Losses": losses,
                "Win_Rate_Pct": (
                    wins / trades * 100
                    if trades
                    else np.nan
                ),
                "Profit_Factor": safe_pf(r),
                "Total_R": total_r,
                "Average_R": (
                    total_r / trades
                    if trades
                    else np.nan
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def print_table(title, df):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if df.empty:
        print("NO DATA")
        return

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# LOAD
# ============================================================================

print("=" * 100)
print("TRADESENSE-AI — MTF INCREMENTAL COUNTERFACTUAL FORENSIC")
print("=" * 100)

if not MTF_FILE.exists():

    print()
    print(
        "ERROR: MTF trade-level file not found:"
    )
    print(MTF_FILE)

    sys.exit(1)


if not FROZEN_FILE.exists():

    print()
    print(
        "ERROR: Frozen trade ledger not found:"
    )
    print(FROZEN_FILE)

    sys.exit(1)


mtf = pd.read_csv(
    MTF_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

print()
print(
    f"MTF trade-level rows : {len(mtf)}"
)

print(
    f"Frozen trade rows    : {len(frozen)}"
)


# ============================================================================
# EXACT REQUIRED SCHEMA
# ============================================================================

required_frozen = [
    "_symbol",
    "Signal_Date",
    "direction",
    "r_multiple",
]

required_mtf = [
    "_symbol",
    "Signal_Date",
    "direction",
    "Pre_Entry_MTF_Class",
    "Pre_Entry_MTF_State",
    "Early_Adverse_Group",
]


missing_frozen = [
    c for c in required_frozen
    if c not in frozen.columns
]

missing_mtf = [
    c for c in required_mtf
    if c not in mtf.columns
]


if missing_frozen:

    print()
    print(
        "ERROR: Missing frozen columns:"
    )
    print(missing_frozen)

    sys.exit(1)


if missing_mtf:

    print()
    print(
        "ERROR: Missing MTF columns:"
    )
    print(missing_mtf)

    sys.exit(1)


# ============================================================================
# NORMALIZE FROZEN
# ============================================================================

frozen["Symbol_Normalized"] = (
    frozen["_symbol"]
    .apply(normalize_symbol)
)

frozen["Signal_Date_Normalized"] = (
    pd.to_datetime(
        frozen["Signal_Date"],
        errors="coerce",
    ).dt.normalize()
)

frozen["Direction_Normalized"] = (
    frozen["direction"]
    .apply(normalize_direction)
)

frozen["r_multiple"] = pd.to_numeric(
    frozen["r_multiple"],
    errors="coerce",
)


# ============================================================================
# NORMALIZE MTF
# ============================================================================

mtf["Symbol_Normalized"] = (
    mtf["_symbol"]
    .apply(normalize_symbol)
)

mtf["Signal_Date_Normalized"] = (
    pd.to_datetime(
        mtf["Signal_Date"],
        errors="coerce",
    ).dt.normalize()
)

mtf["Direction_Normalized"] = (
    mtf["direction"]
    .apply(normalize_direction)
)

mtf["Pre_Entry_MTF_Class"] = (
    mtf["Pre_Entry_MTF_Class"]
    .astype(str)
    .str.strip()
    .str.upper()
)

mtf["Pre_Entry_MTF_State"] = (
    mtf["Pre_Entry_MTF_State"]
    .astype(str)
    .str.strip()
    .str.upper()
)

mtf["Early_Adverse_Group"] = (
    mtf["Early_Adverse_Group"]
    .astype(str)
    .str.strip()
    .str.upper()
)

mtf["Pre_Entry_MTF_Alignment"] = (
    mtf["Pre_Entry_MTF_State"]
    .apply(classify_alignment)
)


# ============================================================================
# TRADE KEY
# ============================================================================

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


# ============================================================================
# DUPLICATE CHECK
# ============================================================================

frozen_duplicates = (
    frozen["Trade_Key"]
    .duplicated(keep=False)
)

mtf_duplicates = (
    mtf["Trade_Key"]
    .duplicated(keep=False)
)

print()
print("=" * 100)
print("TRADE KEY VALIDATION")
print("=" * 100)

print(
    f"Frozen duplicate rows : "
    f"{int(frozen_duplicates.sum())}"
)

print(
    f"MTF duplicate rows    : "
    f"{int(mtf_duplicates.sum())}"
)


if frozen_duplicates.any():

    print()
    print(
        "ERROR: Duplicate frozen trade keys."
    )

    print(
        frozen.loc[
            frozen_duplicates,
            [
                "Trade_Key",
                "_symbol",
                "Signal_Date",
                "direction",
            ],
        ].to_string(index=False)
    )

    sys.exit(1)


if mtf_duplicates.any():

    print()
    print(
        "ERROR: Duplicate MTF trade keys."
    )

    print(
        mtf.loc[
            mtf_duplicates,
            [
                "Trade_Key",
                "_symbol",
                "Signal_Date",
                "direction",
            ],
        ].to_string(index=False)
    )

    sys.exit(1)


# ============================================================================
# MERGE
# ============================================================================

mtf_subset = mtf[
    [
        "Trade_Key",
        "Pre_Entry_MTF_Class",
        "Pre_Entry_MTF_State",
        "Pre_Entry_MTF_Alignment",
        "Early_Adverse_Group",
    ]
].copy()


analysis = frozen.merge(
    mtf_subset,
    on="Trade_Key",
    how="left",
)


# ============================================================================
# RECONCILIATION
# ============================================================================

print()
print("=" * 100)
print("DATA RECONCILIATION")
print("=" * 100)

print(
    f"Frozen trades       : {len(frozen)}"
)

print(
    f"Analyzed trades     : {len(analysis)}"
)

missing_mtf_rows = analysis[
    analysis["Pre_Entry_MTF_Class"].isna()
]

print(
    f"Missing MTF matches : "
    f"{len(missing_mtf_rows)}"
)

frozen_total_r = float(
    frozen["r_multiple"].sum()
)

analysis_total_r = float(
    analysis["r_multiple"].sum()
)

print(
    f"Frozen total R      : "
    f"{frozen_total_r:.4f}"
)

print(
    f"Analyzed total R    : "
    f"{analysis_total_r:.4f}"
)

print(
    f"R delta             : "
    f"{analysis_total_r - frozen_total_r:.4f}"
)


if len(analysis) != len(frozen):

    print()
    print(
        "ERROR: Trade-count reconciliation FAILED."
    )

    sys.exit(1)


if len(missing_mtf_rows) > 0:

    print()
    print(
        "ERROR: MTF reconciliation FAILED."
    )

    print(
        missing_mtf_rows[
            [
                "_symbol",
                "Signal_Date",
                "direction",
            ]
        ].to_string(index=False)
    )

    sys.exit(1)


if abs(
    analysis_total_r - frozen_total_r
) > 1e-9:

    print()
    print(
        "ERROR: R reconciliation FAILED."
    )

    sys.exit(1)


print()
print(
    "Trade-count reconciliation : PASS"
)

print(
    "R reconciliation           : PASS"
)


# ============================================================================
# 1. BASELINE
# ============================================================================

baseline = pd.DataFrame(
    [
        {
            "Model": "FROZEN_BASELINE",
            "Trades": len(analysis),
            "Wins": int(
                (analysis["r_multiple"] > 0).sum()
            ),
            "Losses": int(
                (analysis["r_multiple"] <= 0).sum()
            ),
            "Win_Rate_Pct": (
                (
                    analysis["r_multiple"] > 0
                ).mean()
                * 100
            ),
            "Profit_Factor": safe_pf(
                analysis["r_multiple"]
            ),
            "Total_R": float(
                analysis["r_multiple"].sum()
            ),
            "Average_R": float(
                analysis["r_multiple"].mean()
            ),
        }
    ]
)

print_table(
    "1. FROZEN BASELINE",
    baseline,
)


# ============================================================================
# 2. EARLY ADVERSE
# ============================================================================

early_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
    ],
)

print_table(
    "2. EARLY-ADVERSE ONLY",
    early_summary,
)


# ============================================================================
# 3. PRE-ENTRY MTF
# ============================================================================

mtf_summary = summarize(
    analysis,
    [
        "Pre_Entry_MTF_Alignment",
    ],
)

print_table(
    "3. PRE-ENTRY MTF ALIGNMENT ONLY",
    mtf_summary,
)


# ============================================================================
# 4. EARLY ADVERSE × MTF
# ============================================================================

combined_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
        "Pre_Entry_MTF_Alignment",
    ],
)

print_table(
    "4. EARLY-ADVERSE × PRE-ENTRY MTF",
    combined_summary,
)


# ============================================================================
# 5. INCREMENTAL MTF EFFECT
# ============================================================================

incremental_rows = []


for early_state in [
    "TRIGGERED",
    "NOT_TRIGGERED",
]:

    subset = analysis[
        analysis["Early_Adverse_Group"]
        == early_state
    ]

    aligned = subset[
        subset["Pre_Entry_MTF_Alignment"]
        == "ALIGNED"
    ]["r_multiple"].dropna()

    not_aligned = subset[
        subset["Pre_Entry_MTF_Alignment"]
        == "NOT_ALIGNED"
    ]["r_multiple"].dropna()

    if len(aligned) == 0:
        continue

    if len(not_aligned) == 0:
        continue

    aligned_total = float(
        aligned.sum()
    )

    not_aligned_total = float(
        not_aligned.sum()
    )

    aligned_avg = float(
        aligned.mean()
    )

    not_aligned_avg = float(
        not_aligned.mean()
    )

    incremental_rows.append(
        {
            "Early_Adverse_Group": early_state,

            "Aligned_Trades": len(aligned),
            "Aligned_Total_R": aligned_total,
            "Aligned_Average_R": aligned_avg,

            "Not_Aligned_Trades": len(
                not_aligned
            ),
            "Not_Aligned_Total_R": (
                not_aligned_total
            ),
            "Not_Aligned_Average_R": (
                not_aligned_avg
            ),

            "Delta_Total_R": (
                aligned_total
                - not_aligned_total
            ),

            "Delta_Average_R": (
                aligned_avg
                - not_aligned_avg
            ),
        }
    )


incremental_summary = pd.DataFrame(
    incremental_rows
)

print_table(
    "5. INCREMENTAL MTF EFFECT WITHIN EARLY-ADVERSE GROUP",
    incremental_summary,
)


# ============================================================================
# 6. DIRECTION × EARLY ADVERSE × MTF
# ============================================================================

direction_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
        "Direction_Normalized",
        "Pre_Entry_MTF_Alignment",
    ],
)

direction_filtered = direction_summary[
    direction_summary["Trades"]
    >= MIN_CELL_TRADES
]

print_table(
    "6. DIRECTION × EARLY-ADVERSE × MTF — CELLS >= 5",
    direction_filtered,
)


# ============================================================================
# 7. SETUP × EARLY ADVERSE × MTF
# ============================================================================

analysis["Setup_Normalized"] = (
    analysis["Setup"]
    .astype(str)
    .str.strip()
    .str.upper()
)

setup_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
        "Setup_Normalized",
        "Pre_Entry_MTF_Alignment",
    ],
)

setup_filtered = setup_summary[
    setup_summary["Trades"]
    >= MIN_CELL_TRADES
]

print_table(
    "7. SETUP × EARLY-ADVERSE × MTF — CELLS >= 5",
    setup_filtered,
)


# ============================================================================
# 8. SYMBOL LEAVE-ONE-OUT
# ============================================================================

loo_symbol_rows = []

for symbol in sorted(
    analysis["Symbol_Normalized"]
    .dropna()
    .unique()
):

    subset = analysis[
        analysis["Symbol_Normalized"]
        != symbol
    ]

    triggered = subset[
        subset["Early_Adverse_Group"]
        == "TRIGGERED"
    ]

    aligned = triggered[
        triggered["Pre_Entry_MTF_Alignment"]
        == "ALIGNED"
    ]["r_multiple"].dropna()

    not_aligned = triggered[
        triggered["Pre_Entry_MTF_Alignment"]
        == "NOT_ALIGNED"
    ]["r_multiple"].dropna()

    if len(aligned) < 2:
        continue

    if len(not_aligned) < 2:
        continue

    loo_symbol_rows.append(
        {
            "Excluded_Symbol": symbol,
            "Aligned_Trades": len(aligned),
            "Aligned_Average_R": float(
                aligned.mean()
            ),
            "Not_Aligned_Trades": len(
                not_aligned
            ),
            "Not_Aligned_Average_R": float(
                not_aligned.mean()
            ),
            "MTF_Delta_Average_R": (
                float(aligned.mean())
                - float(not_aligned.mean())
            ),
        }
    )


loo_symbol = pd.DataFrame(
    loo_symbol_rows
)

print_table(
    "8. SYMBOL LEAVE-ONE-OUT — TRIGGERED GROUP",
    loo_symbol,
)


# ============================================================================
# 9. YEAR LEAVE-ONE-OUT
# ============================================================================

analysis["Signal_Year"] = (
    analysis["Signal_Date_Normalized"]
    .dt.year
)

loo_year_rows = []

for year in sorted(
    analysis["Signal_Year"]
    .dropna()
    .unique()
):

    subset = analysis[
        analysis["Signal_Year"]
        != year
    ]

    triggered = subset[
        subset["Early_Adverse_Group"]
        == "TRIGGERED"
    ]

    aligned = triggered[
        triggered["Pre_Entry_MTF_Alignment"]
        == "ALIGNED"
    ]["r_multiple"].dropna()

    not_aligned = triggered[
        triggered["Pre_Entry_MTF_Alignment"]
        == "NOT_ALIGNED"
    ]["r_multiple"].dropna()

    if len(aligned) < 2:
        continue

    if len(not_aligned) < 2:
        continue

    loo_year_rows.append(
        {
            "Excluded_Year": int(year),
            "Aligned_Trades": len(aligned),
            "Aligned_Average_R": float(
                aligned.mean()
            ),
            "Not_Aligned_Trades": len(
                not_aligned
            ),
            "Not_Aligned_Average_R": float(
                not_aligned.mean()
            ),
            "MTF_Delta_Average_R": (
                float(aligned.mean())
                - float(not_aligned.mean())
            ),
        }
    )


loo_year = pd.DataFrame(
    loo_year_rows
)

print_table(
    "9. YEAR LEAVE-ONE-OUT — TRIGGERED GROUP",
    loo_year,
)


# ============================================================================
# 10. RAW MTF CLASS
# ============================================================================

class_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
        "Pre_Entry_MTF_Class",
    ],
)

class_filtered = class_summary[
    class_summary["Trades"]
    >= MIN_CELL_TRADES
]

print_table(
    "10. RAW MTF CLASS × EARLY-ADVERSE — CELLS >= 5",
    class_filtered,
)


# ============================================================================
# 11. DIFFERENCE-IN-DIFFERENCES
# ============================================================================

triggered = analysis[
    analysis["Early_Adverse_Group"]
    == "TRIGGERED"
]

not_triggered = analysis[
    analysis["Early_Adverse_Group"]
    == "NOT_TRIGGERED"
]


triggered_aligned = triggered[
    triggered["Pre_Entry_MTF_Alignment"]
    == "ALIGNED"
]["r_multiple"].dropna()

triggered_not_aligned = triggered[
    triggered["Pre_Entry_MTF_Alignment"]
    == "NOT_ALIGNED"
]["r_multiple"].dropna()

not_triggered_aligned = not_triggered[
    not_triggered["Pre_Entry_MTF_Alignment"]
    == "ALIGNED"
]["r_multiple"].dropna()

not_triggered_not_aligned = not_triggered[
    not_triggered["Pre_Entry_MTF_Alignment"]
    == "NOT_ALIGNED"
]["r_multiple"].dropna()


triggered_effect = (
    triggered_aligned.mean()
    - triggered_not_aligned.mean()
)

not_triggered_effect = (
    not_triggered_aligned.mean()
    - not_triggered_not_aligned.mean()
)

difference_in_differences = (
    triggered_effect
    - not_triggered_effect
)


interaction = pd.DataFrame(
    [
        {
            "Triggered_MTF_Effect_Average_R":
                triggered_effect,

            "Not_Triggered_MTF_Effect_Average_R":
                not_triggered_effect,

            "Difference_in_Differences_R":
                difference_in_differences,
        }
    ]
)

print_table(
    "11. MTF × EARLY-ADVERSE INTERACTION",
    interaction,
)


# ============================================================================
# 12. MTF CONFLICT SEVERITY
# ============================================================================

conflict_summary = summarize(
    analysis,
    [
        "Early_Adverse_Group",
        "Pre_Entry_MTF_State",
    ],
)

print_table(
    "12. RAW PRE-ENTRY MTF STATE × EARLY-ADVERSE",
    conflict_summary,
)


# ============================================================================
# 13. DESCRIPTIVE COUNTERFACTUAL RETENTION
# ============================================================================

valid_mtf = analysis[
    analysis["Pre_Entry_MTF_Alignment"]
    != "UNAVAILABLE"
].copy()


aligned_only = valid_mtf[
    valid_mtf["Pre_Entry_MTF_Alignment"]
    == "ALIGNED"
]

not_aligned_only = valid_mtf[
    valid_mtf["Pre_Entry_MTF_Alignment"]
    == "NOT_ALIGNED"
]


print()
print("=" * 100)
print("13. DESCRIPTIVE COUNTERFACTUAL RETENTION")
print("=" * 100)

print()
print(
    "VALID MTF SAMPLE"
)

print(
    f"Trades : {len(valid_mtf)}"
)

print(
    f"Total R: "
    f"{valid_mtf['r_multiple'].sum():.4f}"
)

print()
print(
    "KEEP ALIGNED ONLY — DESCRIPTIVE"
)

print(
    f"Trades retained : {len(aligned_only)}"
)

print(
    f"Reduction       : "
    f"{(
        1
        - len(aligned_only)
        / len(valid_mtf)
    ) * 100:.2f}%"
)

print(
    f"Retained R      : "
    f"{aligned_only['r_multiple'].sum():.4f}"
)

print(
    f"Average R       : "
    f"{aligned_only['r_multiple'].mean():.4f}"
)

print()
print(
    "KEEP NOT-ALIGNED ONLY — DESCRIPTIVE"
)

print(
    f"Trades retained : {len(not_aligned_only)}"
)

print(
    f"Reduction       : "
    f"{(
        1
        - len(not_aligned_only)
        / len(valid_mtf)
    ) * 100:.2f}%"
)

print(
    f"Retained R      : "
    f"{not_aligned_only['r_multiple'].sum():.4f}"
)

print(
    f"Average R       : "
    f"{not_aligned_only['r_multiple'].mean():.4f}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "These are descriptive selections, NOT executable backtests."
)

print(
    "Early-adverse is observed after entry and therefore cannot "
    "be used as a pre-entry filter."
)


# ============================================================================
# SAVE TRADE-LEVEL OUTPUT
# ============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

analysis.to_csv(
    TRADE_OUTPUT_FILE,
    index=False,
)


# ============================================================================
# SAVE SUMMARY OUTPUT
# ============================================================================

summary_blocks = []


baseline_out = baseline.copy()
baseline_out.insert(
    0,
    "Section",
    "BASELINE",
)

summary_blocks.append(
    baseline_out
)


early_out = early_summary.copy()
early_out.insert(
    0,
    "Section",
    "EARLY_ADVERSE",
)

summary_blocks.append(
    early_out
)


mtf_out = mtf_summary.copy()
mtf_out.insert(
    0,
    "Section",
    "MTF_ALIGNMENT",
)

summary_blocks.append(
    mtf_out
)


combined_out = combined_summary.copy()
combined_out.insert(
    0,
    "Section",
    "COMBINED",
)

summary_blocks.append(
    combined_out
)


incremental_out = incremental_summary.copy()
incremental_out.insert(
    0,
    "Section",
    "INCREMENTAL",
)

summary_blocks.append(
    incremental_out
)


interaction_out = interaction.copy()
interaction_out.insert(
    0,
    "Section",
    "INTERACTION",
)

summary_blocks.append(
    interaction_out
)


summary_output = pd.concat(
    summary_blocks,
    ignore_index=True,
    sort=False,
)


summary_output.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================================
# FINAL STATUS
# ============================================================================

print()
print("=" * 100)
print("FINAL INTEGRITY")
print("=" * 100)

print(
    f"Frozen trades              : "
    f"{len(frozen)}"
)

print(
    f"Analyzed trades            : "
    f"{len(analysis)}"
)

print(
    f"Frozen total R             : "
    f"{frozen_total_r:.4f}"
)

print(
    f"Analyzed total R           : "
    f"{analysis_total_r:.4f}"
)

print(
    f"Reconciliation delta       : "
    f"{analysis_total_r - frozen_total_r:.4f}"
)

print()
print(
    "Trade-count reconciliation : PASS"
)

print(
    "R reconciliation           : PASS"
)

print()
print(
    "Production strategy        : UNCHANGED"
)

print(
    "Production MTF logic       : UNCHANGED"
)

print(
    "OOS data                   : UNCHANGED"
)

print()
print(
    "STATUS: "
    "MTF_INCREMENTAL_COUNTERFACTUAL_COMPLETE"
)

print()
print(
    f"Trade-level output        : "
    f"{TRADE_OUTPUT_FILE}"
)

print(
    f"Summary output             : "
    f"{SUMMARY_FILE}"
)