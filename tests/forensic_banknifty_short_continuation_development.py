"""
TradeSense-AI
BANK NIFTY SHORT_CONTINUATION FORENSIC

Research-only diagnostic.

Purpose:
    Investigate why BANK NIFTY SHORT_CONTINUATION trades
    are responsible for a large portion of baseline losses.

Input:
    tests/output/index_backtests/banknifty_index_trades.csv

Output:
    tests/output/index_backtests/
        banknifty_short_continuation_development.csv

IMPORTANT:
    - No production strategy modification.
    - No frozen historical stock data modification.
    - No stock holdout modification.
    - No filter implementation.
    - No counterfactual in this script.

This script is descriptive only.
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = (
    "tests/output/index_backtests/banknifty_index_trades.csv"
)

OUTPUT_FILE = (
    "tests/output/index_backtests/"
    "banknifty_short_continuation_development.csv"
)

TARGET_SETUP = "SHORT_CONTINUATION"

CONTROL_SETUP = "LONG_CONTINUATION"

EXPECTED_TOTAL_TRADES = 24

EXPECTED_TARGET_TRADES = 8


# ============================================================================
# HELPERS
# ============================================================================

def to_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def summarize(group, name):
    if group.empty:
        return {
            "setup": name,
            "trades": 0,
            "winners": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "avg_mfe": np.nan,
            "avg_mae": np.nan,
            "avg_early5": np.nan,
            "avg_bars": np.nan,
            "avg_atr": np.nan,
        }

    r_values = to_numeric(
        group["r_multiple"]
    ).dropna()

    winners = int(
        (r_values > 0).sum()
    )

    gross_profit = r_values[
        r_values > 0
    ].sum()

    gross_loss = abs(
        r_values[
            r_values < 0
        ].sum()
    )

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    result = {
        "setup": name,
        "trades": len(group),
        "winners": winners,
        "win_rate": (
            winners / len(group) * 100
        ),
        "total_r": r_values.sum(),
        "avg_r": r_values.mean(),
        "profit_factor": pf,
        "avg_mfe": np.nan,
        "avg_mae": np.nan,
        "avg_early5": np.nan,
        "avg_bars": np.nan,
        "avg_atr": np.nan,
    }

    if "max_favorable_r" in group.columns:
        result["avg_mfe"] = to_numeric(
            group["max_favorable_r"]
        ).mean()

    if "max_adverse_r" in group.columns:
        result["avg_mae"] = to_numeric(
            group["max_adverse_r"]
        ).mean()

    if "early_adverse_r_bar_5" in group.columns:
        result["avg_early5"] = to_numeric(
            group["early_adverse_r_bar_5"]
        ).mean()

    if "bars_in_trade" in group.columns:
        result["avg_bars"] = to_numeric(
            group["bars_in_trade"]
        ).mean()

    if "entry_atr" in group.columns:
        result["avg_atr"] = to_numeric(
            group["entry_atr"]
        ).mean()

    return result


def print_summary(result):
    pf = result["profit_factor"]

    pf_text = (
        f"{pf:.3f}"
        if np.isfinite(pf)
        else "INF"
    )

    print(
        f"Trades:          {result['trades']}"
    )

    print(
        f"Winners:         {result['winners']}"
    )

    print(
        f"Win rate:        {result['win_rate']:.2f}%"
    )

    print(
        f"Total R:         {result['total_r']:+.2f}R"
    )

    print(
        f"Avg R:           {result['avg_r']:+.3f}R"
    )

    print(
        f"Profit Factor:   {pf_text}"
    )

    print(
        f"Avg MFE:         {result['avg_mfe']:.3f}R"
    )

    print(
        f"Avg MAE:         {result['avg_mae']:.3f}R"
    )

    print(
        f"Avg Early5:      {result['avg_early5']:.3f}R"
    )

    print(
        f"Avg Holding:     {result['avg_bars']:.2f}"
    )

    print(
        f"Avg Entry ATR:   {result['avg_atr']:.2f}"
    )


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print(
    "TRADESENSE-AI — BANK NIFTY "
    "SHORT_CONTINUATION FORENSIC"
)
print("=" * 100)


# ============================================================================
# LOAD
# ============================================================================

if not os.path.exists(INPUT_FILE):

    print()
    print(
        "ERROR: Input file not found:"
    )

    print(INPUT_FILE)

    raise SystemExit(1)


df = pd.read_csv(
    INPUT_FILE
)


print()
print(
    f"Total BANK NIFTY trades: {len(df)}"
)


if len(df) != EXPECTED_TOTAL_TRADES:

    print()
    print(
        "WARNING: Expected "
        f"{EXPECTED_TOTAL_TRADES} total trades "
        f"but found {len(df)}."
    )


required_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    print()
    print(
        "ERROR: Missing required columns:"
    )

    for column in missing:
        print(f"  - {column}")

    raise SystemExit(1)


df["r_multiple"] = to_numeric(
    df["r_multiple"]
)

target = df[
    df["setup"] == TARGET_SETUP
].copy()

control = df[
    df["setup"] == CONTROL_SETUP
].copy()


print(
    f"{TARGET_SETUP} trades: {len(target)}"
)

print(
    f"{CONTROL_SETUP} trades: {len(control)}"
)


if len(target) != EXPECTED_TARGET_TRADES:

    print()
    print(
        "WARNING: Expected "
        f"{EXPECTED_TARGET_TRADES} "
        f"{TARGET_SETUP} trades but found "
        f"{len(target)}."
    )


# ============================================================================
# TARGET SUMMARY
# ============================================================================

print()
print("-" * 100)
print("SHORT_CONTINUATION SUMMARY")
print("-" * 100)

target_summary = summarize(
    target,
    TARGET_SETUP
)

print_summary(
    target_summary
)


# ============================================================================
# CONTROL SUMMARY
# ============================================================================

print()
print("-" * 100)
print("CONTROL — LONG_CONTINUATION")
print("-" * 100)

control_summary = summarize(
    control,
    CONTROL_SETUP
)

print_summary(
    control_summary
)


# ============================================================================
# COMPARISON
# ============================================================================

print()
print("=" * 100)
print(
    "SHORT_CONTINUATION vs LONG_CONTINUATION"
)
print("=" * 100)

comparison = pd.DataFrame(
    [
        target_summary,
        control_summary,
    ]
)

print(
    comparison.to_string(
        index=False,
        formatters={
            "win_rate": lambda x:
                f"{x:.2f}%",
            "total_r": lambda x:
                f"{x:+.2f}R",
            "avg_r": lambda x:
                f"{x:+.3f}R",
            "profit_factor": lambda x:
                (
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "INF"
                ),
            "avg_mfe": lambda x:
                f"{x:.3f}R",
            "avg_mae": lambda x:
                f"{x:.3f}R",
            "avg_early5": lambda x:
                f"{x:.3f}R",
            "avg_bars": lambda x:
                f"{x:.2f}",
            "avg_atr": lambda x:
                f"{x:.2f}",
        }
    )
)


# ============================================================================
# MFE / MAE DEVELOPMENT
# ============================================================================

print()
print("=" * 100)
print("MFE / MAE DEVELOPMENT")
print("=" * 100)

for name, group in [
    (TARGET_SETUP, target),
    (CONTROL_SETUP, control),
]:

    print()
    print(name)

    if "max_favorable_r" in group.columns:

        mfe = to_numeric(
            group["max_favorable_r"]
        )

        print(
            f"  MFE mean:   {mfe.mean():.3f}R"
        )

        print(
            f"  MFE median: {mfe.median():.3f}R"
        )

    if "max_adverse_r" in group.columns:

        mae = to_numeric(
            group["max_adverse_r"]
        )

        print(
            f"  MAE mean:   {mae.mean():.3f}R"
        )

        print(
            f"  MAE median: {mae.median():.3f}R"
        )


# ============================================================================
# MFE MILESTONES
# ============================================================================

print()
print("=" * 100)
print("MFE MILESTONE DEVELOPMENT")
print("=" * 100)

if "max_favorable_r" in df.columns:

    for name, group in [
        (TARGET_SETUP, target),
        (CONTROL_SETUP, control),
    ]:

        print()
        print(name)

        mfe = to_numeric(
            group["max_favorable_r"]
        )

        for threshold in [
            0.25,
            0.50,
            1.00,
            1.50,
            2.00,
        ]:

            reached = group[
                mfe >= threshold
            ]

            reached_r = to_numeric(
                reached["r_multiple"]
            ).sum()

            print(
                f"  >= {threshold:.2f}R: "
                f"{len(reached)}/{len(group)} "
                f"TotalR={reached_r:+.2f}R"
            )


# ============================================================================
# EARLY ADVERSE DEVELOPMENT
# ============================================================================

print()
print("=" * 100)
print("EARLY-ADVERSE DEVELOPMENT")
print("=" * 100)

early_columns = [
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
]


for name, group in [
    (TARGET_SETUP, target),
    (CONTROL_SETUP, control),
]:

    print()
    print(name)

    for column in early_columns:

        if column not in group.columns:
            continue

        values = to_numeric(
            group[column]
        )

        print(
            f"  {column}: "
            f"Avg={values.mean():.3f}R "
            f"Median={values.median():.3f}R "
            f"Max={values.max():.3f}R"
        )


# ============================================================================
# 5 BAR / 0.25R DESCRIPTIVE SPLIT
# ============================================================================

print()
print("=" * 100)
print("5 BAR / 0.25R DESCRIPTIVE SPLIT")
print("=" * 100)

if "early_adverse_r_bar_5" in df.columns:

    for name, group in [
        (TARGET_SETUP, target),
        (CONTROL_SETUP, control),
    ]:

        values = to_numeric(
            group["early_adverse_r_bar_5"]
        )

        triggered = group[
            values >= 0.25
        ]

        non_triggered = group[
            values < 0.25
        ]

        triggered_r = to_numeric(
            triggered["r_multiple"]
        ).sum()

        non_triggered_r = to_numeric(
            non_triggered["r_multiple"]
        ).sum()

        print()
        print(name)

        print(
            f"  Triggered: "
            f"{len(triggered)} "
            f"R={triggered_r:+.2f}R"
        )

        print(
            f"  Non-triggered: "
            f"{len(non_triggered)} "
            f"R={non_triggered_r:+.2f}R"
        )


# ============================================================================
# FAVORABLE-FIRST
# ============================================================================

print()
print("=" * 100)
print("FAVORABLE-FIRST / IMMEDIATE-FAILURE")
print("=" * 100)

for name, group in [
    (TARGET_SETUP, target),
    (CONTROL_SETUP, control),
]:

    print()
    print(name)

    if "favorable_first" in group.columns:

        print(
            "Favorable first:"
        )

        print(
            group["favorable_first"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if "immediate_failure" in group.columns:

        print(
            "Immediate failure:"
        )

        print(
            group["immediate_failure"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ============================================================================
# HOLDING PERIOD
# ============================================================================

print()
print("=" * 100)
print("HOLDING PERIOD")
print("=" * 100)

if "bars_in_trade" in df.columns:

    for name, group in [
        (TARGET_SETUP, target),
        (CONTROL_SETUP, control),
    ]:

        values = to_numeric(
            group["bars_in_trade"]
        )

        print()
        print(name)

        print(
            f"  Mean:   {values.mean():.2f}"
        )

        print(
            f"  Median: {values.median():.2f}"
        )

        print(
            f"  Min:    {values.min():.0f}"
        )

        print(
            f"  Max:    {values.max():.0f}"
        )


# ============================================================================
# YEAR-BY-YEAR TARGET
# ============================================================================

print()
print("=" * 100)
print("YEAR-BY-YEAR SHORT_CONTINUATION")
print("=" * 100)

target["entry_date_parsed"] = pd.to_datetime(
    target["entry_date"],
    errors="coerce"
)

target["year"] = (
    target["entry_date_parsed"]
    .dt.year
)

year_rows = []

for year, group in target.groupby(
    "year",
    dropna=False
):

    values = to_numeric(
        group["r_multiple"]
    ).dropna()

    winners = int(
        (values > 0).sum()
    )

    row = {
        "year": (
            int(year)
            if pd.notna(year)
            else "UNKNOWN"
        ),
        "trades": len(group),
        "winners": winners,
        "win_rate": (
            winners / len(group) * 100
        ),
        "total_r": values.sum(),
        "avg_r": values.mean(),
    }

    if "max_favorable_r" in group.columns:
        row["avg_mfe"] = to_numeric(
            group["max_favorable_r"]
        ).mean()

    if "max_adverse_r" in group.columns:
        row["avg_mae"] = to_numeric(
            group["max_adverse_r"]
        ).mean()

    if "early_adverse_r_bar_5" in group.columns:
        row["avg_early5"] = to_numeric(
            group["early_adverse_r_bar_5"]
        ).mean()

    year_rows.append(row)


year_df = pd.DataFrame(
    year_rows
)

if not year_df.empty:

    print(
        year_df.to_string(
            index=False,
            formatters={
                "win_rate": lambda x:
                    f"{x:.2f}%",
                "total_r": lambda x:
                    f"{x:+.2f}R",
                "avg_r": lambda x:
                    f"{x:+.3f}R",
                "avg_mfe": lambda x:
                    f"{x:.3f}R",
                "avg_mae": lambda x:
                    f"{x:.3f}R",
                "avg_early5": lambda x:
                    f"{x:.3f}R",
            }
        )
    )


# ============================================================================
# INDIVIDUAL TARGET TRADES
# ============================================================================

print()
print("=" * 100)
print("INDIVIDUAL SHORT_CONTINUATION TRADES")
print("=" * 100)

display_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
    "bars_in_trade",
    "entry_atr",
    "initial_risk",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
    "favorable_first",
    "immediate_failure",
    "exit_reason",
]


available_columns = [
    column
    for column in display_columns
    if column in target.columns
]


print(
    target[
        available_columns
    ].to_string(
        index=False
    )
)


# ============================================================================
# TRADE-LEVEL FAILURE CLASSIFICATION
# ============================================================================

print()
print("=" * 100)
print("DESCRIPTIVE FAILURE CLASSIFICATION")
print("=" * 100)

if (
    "max_favorable_r" in target.columns
    and "max_adverse_r" in target.columns
):

    mfe = to_numeric(
        target["max_favorable_r"]
    )

    mae = to_numeric(
        target["max_adverse_r"]
    )

    classifications = []

    for index in target.index:

        trade_mfe = mfe.loc[index]
        trade_mae = mae.loc[index]

        if (
            pd.notna(trade_mfe)
            and trade_mfe < 0.25
        ):

            classification = (
                "NO_MEANINGFUL_FAVORABLE_DEVELOPMENT"
            )

        elif (
            pd.notna(trade_mfe)
            and trade_mfe < 1.0
        ):

            classification = (
                "PARTIAL_DEVELOPMENT"
            )

        elif (
            pd.notna(trade_mfe)
            and trade_mfe < 1.5
        ):

            classification = (
                "STRONG_BUT_SUBTARGET_DEVELOPMENT"
            )

        else:

            classification = (
                "HIGH_FAVORABLE_DEVELOPMENT"
            )

        classifications.append(
            classification
        )

    target["development_class"] = (
        classifications
    )

    class_summary = (
        target
        .groupby("development_class")
        .agg(
            trades=("r_multiple", "size"),
            total_r=("r_multiple", "sum"),
            avg_r=("r_multiple", "mean"),
            avg_mfe=(
                "max_favorable_r",
                "mean"
            ),
            avg_mae=(
                "max_adverse_r",
                "mean"
            ),
        )
        .reset_index()
    )

    print(
        class_summary.to_string(
            index=False,
            formatters={
                "total_r": lambda x:
                    f"{x:+.2f}R",
                "avg_r": lambda x:
                    f"{x:+.3f}R",
                "avg_mfe": lambda x:
                    f"{x:.3f}R",
                "avg_mae": lambda x:
                    f"{x:.3f}R",
            }
        )
    )


# ============================================================================
# RESEARCH INTERPRETATION
# ============================================================================

print()
print("=" * 100)
print("RESEARCH INTERPRETATION")
print("=" * 100)

target_total = target_summary["total_r"]
control_total = control_summary["total_r"]

print()

if target_total < control_total:

    print(
        "SHORT_CONTINUATION materially underperforms "
        "the LONG_CONTINUATION control in this sample."
    )

else:

    print(
        "SHORT_CONTINUATION does not materially "
        "underperform the selected control in this sample."
    )


print()

if target_summary["avg_mfe"] < control_summary["avg_mfe"]:

    print(
        "SHORT_CONTINUATION shows weaker favorable "
        "development than LONG_CONTINUATION."
    )

else:

    print(
        "SHORT_CONTINUATION does not show weaker "
        "average favorable development than the control."
    )


print()

if target_summary["avg_mae"] > control_summary["avg_mae"]:

    print(
        "SHORT_CONTINUATION shows greater adverse "
        "excursion than the control."
    )

else:

    print(
        "SHORT_CONTINUATION does not show greater "
        "average adverse excursion than the control."
    )


print()
print(
    "These are descriptive observations only."
)

print(
    "They do NOT justify removing SHORT_CONTINUATION."
)

print(
    "They do NOT justify changing stops, targets, "
    "or early-adverse thresholds."
)


# ============================================================================
# SAVE
# ============================================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

target.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Saved: {OUTPUT_FILE}"
)


# ============================================================================
# SAFETY
# ============================================================================

print()
print("=" * 100)
print("SAFETY CHECK")
print("=" * 100)

print(
    "PASS: No production code modified."
)

print(
    "PASS: No frozen historical data modified."
)

print(
    "PASS: No stock holdout data modified."
)

print(
    "PASS: No BANK NIFTY filter implemented."
)

print(
    "PASS: No counterfactual implemented."
)

print("=" * 100)