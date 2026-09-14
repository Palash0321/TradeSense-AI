"""
TradeSense-AI
BANK NIFTY Trade Diagnostics

Research-only forensic analysis of the BANK NIFTY baseline trades.

Input:
    tests/output/index_backtests/banknifty_index_trades.csv

Output:
    tests/output/index_backtests/banknifty_trade_diagnostics.csv

No production code is modified.
No frozen stock historical data is modified.
No holdout data is modified.
No strategy filter is implemented.
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = (
    "tests/output/index_backtests/banknifty_index_trades.csv"
)

OUTPUT_FILE = (
    "tests/output/index_backtests/banknifty_trade_diagnostics.csv"
)

EXPECTED_TRADES = 24


# ============================================================================
# HELPERS
# ============================================================================

def pct(value):
    return f"{value:.2f}%"


def r(value):
    return f"{value:.2f}R"


def profit_factor(values):
    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    gross_profit = values[
        values > 0
    ].sum()

    gross_loss = abs(
        values[
            values < 0
        ].sum()
    )

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


def summarize_group(df, group_name):
    if df.empty:
        return {
            "group": group_name,
            "trades": 0,
            "winners": 0,
            "win_rate": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
        }

    values = pd.to_numeric(
        df["r_multiple"],
        errors="coerce"
    ).fillna(0.0)

    winners = int(
        (values > 0).sum()
    )

    gross_profit = values[
        values > 0
    ].sum()

    gross_loss = abs(
        values[
            values < 0
        ].sum()
    )

    return {
        "group": group_name,
        "trades": len(df),
        "winners": winners,
        "win_rate": (
            winners / len(df) * 100
        ),
        "gross_profit_r": gross_profit,
        "gross_loss_r": -gross_loss,
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        ),
        "total_r": values.sum(),
        "avg_r": values.mean(),
    }


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print("TRADESENSE-AI — BANK NIFTY TRADE FORENSIC")
print("=" * 100)


# ============================================================================
# LOAD
# ============================================================================

if not os.path.exists(INPUT_FILE):
    print()
    print("ERROR: Input file not found:")
    print(INPUT_FILE)
    raise SystemExit(1)


df = pd.read_csv(
    INPUT_FILE
)


print()
print(
    f"Historical BANK NIFTY trades loaded: {len(df)}"
)


if len(df) != EXPECTED_TRADES:
    print()
    print(
        f"WARNING: Expected {EXPECTED_TRADES} trades "
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
    print("ERROR: Missing required columns:")
    for column in missing:
        print(f"  - {column}")
    raise SystemExit(1)


df["r_multiple"] = pd.to_numeric(
    df["r_multiple"],
    errors="coerce"
)


# ============================================================================
# BASELINE
# ============================================================================

baseline = summarize_group(
    df,
    "ALL"
)

print()
print("-" * 100)
print("BASELINE")
print("-" * 100)

print(
    f"Trades:          {baseline['trades']}"
)

print(
    f"Winners:         {baseline['winners']}"
)

print(
    f"Win rate:        {pct(baseline['win_rate'])}"
)

print(
    f"Gross Profit:    {r(baseline['gross_profit_r'])}"
)

print(
    f"Gross Loss:      {r(baseline['gross_loss_r'])}"
)

print(
    f"Profit Factor:   {baseline['profit_factor']:.3f}"
)

print(
    f"Total R:         {r(baseline['total_r'])}"
)

print(
    f"Average R:       {r(baseline['avg_r'])}"
)


# ============================================================================
# SETUP ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("SETUP ANALYSIS")
print("=" * 100)

setup_rows = []

for setup, group in df.groupby(
    "setup",
    dropna=False
):

    result = summarize_group(
        group,
        str(setup)
    )

    setup_rows.append(result)

setup_df = pd.DataFrame(
    setup_rows
)

print(
    setup_df.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: pct(x),
            "gross_profit_r": lambda x: r(x),
            "gross_loss_r": lambda x: r(x),
            "profit_factor": lambda x: (
                f"{x:.3f}"
                if np.isfinite(x)
                else "INF"
            ),
            "total_r": lambda x: r(x),
            "avg_r": lambda x: r(x),
        }
    )
)


# ============================================================================
# DIRECTION ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("DIRECTION ANALYSIS")
print("=" * 100)

direction_rows = []

for direction, group in df.groupby(
    "direction",
    dropna=False
):

    direction_rows.append(
        summarize_group(
            group,
            str(direction)
        )
    )

direction_df = pd.DataFrame(
    direction_rows
)

print(
    direction_df.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: pct(x),
            "gross_profit_r": lambda x: r(x),
            "gross_loss_r": lambda x: r(x),
            "profit_factor": lambda x: (
                f"{x:.3f}"
                if np.isfinite(x)
                else "INF"
            ),
            "total_r": lambda x: r(x),
            "avg_r": lambda x: r(x),
        }
    )
)


# ============================================================================
# SETUP × DIRECTION
# ============================================================================

print()
print("=" * 100)
print("SETUP × DIRECTION")
print("=" * 100)

cross_rows = []

for (
    setup,
    direction
), group in df.groupby(
    ["setup", "direction"],
    dropna=False
):

    result = summarize_group(
        group,
        f"{setup} × {direction}"
    )

    cross_rows.append(result)

cross_df = pd.DataFrame(
    cross_rows
)

print(
    cross_df.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: pct(x),
            "gross_profit_r": lambda x: r(x),
            "gross_loss_r": lambda x: r(x),
            "profit_factor": lambda x: (
                f"{x:.3f}"
                if np.isfinite(x)
                else "INF"
            ),
            "total_r": lambda x: r(x),
            "avg_r": lambda x: r(x),
        }
    )
)


# ============================================================================
# EXIT ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("EXIT ANALYSIS")
print("=" * 100)

if "exit_reason" in df.columns:

    exit_rows = []

    for exit_reason, group in df.groupby(
        "exit_reason",
        dropna=False
    ):

        exit_rows.append(
            summarize_group(
                group,
                str(exit_reason)
            )
        )

    exit_df = pd.DataFrame(
        exit_rows
    )

    print(
        exit_df.to_string(
            index=False,
            formatters={
                "win_rate": lambda x: pct(x),
                "gross_profit_r": lambda x: r(x),
                "gross_loss_r": lambda x: r(x),
                "profit_factor": lambda x: (
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "INF"
                ),
                "total_r": lambda x: r(x),
                "avg_r": lambda x: r(x),
            }
        )
    )

else:

    print(
        "exit_reason column unavailable."
    )


# ============================================================================
# HOLDING PERIOD
# ============================================================================

print()
print("=" * 100)
print("HOLDING PERIOD ANALYSIS")
print("=" * 100)

if "bars_in_trade" in df.columns:

    df["bars_in_trade"] = pd.to_numeric(
        df["bars_in_trade"],
        errors="coerce"
    )

    bins = [
        0,
        5,
        10,
        20,
        50,
        100,
        np.inf,
    ]

    labels = [
        "1-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        ">100",
    ]

    df["holding_bucket"] = pd.cut(
        df["bars_in_trade"],
        bins=bins,
        labels=labels,
        right=True
    )

    holding_rows = []

    for bucket, group in df.groupby(
        "holding_bucket",
        observed=False
    ):

        if len(group) == 0:
            continue

        result = summarize_group(
            group,
            str(bucket)
        )

        holding_rows.append(result)

    holding_df = pd.DataFrame(
        holding_rows
    )

    print(
        holding_df.to_string(
            index=False,
            formatters={
                "win_rate": lambda x: pct(x),
                "gross_profit_r": lambda x: r(x),
                "gross_loss_r": lambda x: r(x),
                "profit_factor": lambda x: (
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "INF"
                ),
                "total_r": lambda x: r(x),
                "avg_r": lambda x: r(x),
            }
        )
    )

else:

    print(
        "bars_in_trade column unavailable."
    )


# ============================================================================
# MFE / MAE
# ============================================================================

print()
print("=" * 100)
print("MFE / MAE ANALYSIS")
print("=" * 100)

mfe_col = "max_favorable_r"
mae_col = "max_adverse_r"

if mfe_col in df.columns:

    df[mfe_col] = pd.to_numeric(
        df[mfe_col],
        errors="coerce"
    )

    print()
    print(
        f"Average MFE:   "
        f"{df[mfe_col].mean():.3f}R"
    )

    print(
        f"Median MFE:    "
        f"{df[mfe_col].median():.3f}R"
    )

if mae_col in df.columns:

    df[mae_col] = pd.to_numeric(
        df[mae_col],
        errors="coerce"
    )

    print(
        f"Average MAE:   "
        f"{df[mae_col].mean():.3f}R"
    )

    print(
        f"Median MAE:    "
        f"{df[mae_col].median():.3f}R"
    )


# ============================================================================
# WINNERS VS LOSERS
# ============================================================================

print()
print("=" * 100)
print("WINNERS VS LOSERS")
print("=" * 100)

winners_df = df[
    df["r_multiple"] > 0
]

losers_df = df[
    df["r_multiple"] <= 0
]

print()
print(
    f"Winners: {len(winners_df)}"
)

print(
    f"Losers:  {len(losers_df)}"
)

for column, label in [
    ("max_favorable_r", "MFE"),
    ("max_adverse_r", "MAE"),
    ("bars_in_trade", "Holding Bars"),
    ("entry_atr", "Entry ATR"),
    ("initial_risk", "Initial Risk"),
]:

    if column not in df.columns:
        continue

    winner_mean = pd.to_numeric(
        winners_df[column],
        errors="coerce"
    ).mean()

    loser_mean = pd.to_numeric(
        losers_df[column],
        errors="coerce"
    ).mean()

    print()
    print(
        f"{label}:"
    )

    print(
        f"  Winners avg: {winner_mean:.3f}"
    )

    print(
        f"  Losers avg:  {loser_mean:.3f}"
    )


# ============================================================================
# MFE MILESTONES
# ============================================================================

print()
print("=" * 100)
print("MFE MILESTONE ANALYSIS")
print("=" * 100)

if mfe_col in df.columns:

    for threshold in [
        0.25,
        0.50,
        1.00,
        1.50,
        2.00,
    ]:

        reached = df[
            df[mfe_col] >= threshold
        ]

        reached_r = reached[
            "r_multiple"
        ].sum()

        print(
            f"Reached {threshold:.2f}R: "
            f"{len(reached)}/{len(df)} "
            f"TotalR={reached_r:+.2f}R"
        )


# ============================================================================
# EARLY-ADVERSE
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

available_early = []

for column in early_columns:

    if column not in df.columns:
        continue

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    available_early.append(
        column
    )

    print(
        f"{column}: "
        f"Avg={df[column].mean():.3f}R "
        f"Median={df[column].median():.3f}R "
        f"Max={df[column].max():.3f}R"
    )


# ============================================================================
# EARLY-ADVERSE 5B / 0.25R
# ============================================================================

if "early_adverse_r_bar_5" in df.columns:

    print()
    print(
        "-" * 100
    )

    print(
        "5 BAR / 0.25R DIAGNOSTIC"
    )

    triggered = df[
        df["early_adverse_r_bar_5"] >= 0.25
    ]

    non_triggered = df[
        df["early_adverse_r_bar_5"] < 0.25
    ]

    print()

    print(
        f"Triggered:     {len(triggered)}"
    )

    print(
        f"Triggered R:   "
        f"{triggered['r_multiple'].sum():+.2f}R"
    )

    print(
        f"Non-triggered: {len(non_triggered)}"
    )

    print(
        f"Non-triggered R: "
        f"{non_triggered['r_multiple'].sum():+.2f}R"
    )

    filtered_r = non_triggered[
        "r_multiple"
    ].sum()

    baseline_r = df[
        "r_multiple"
    ].sum()

    print()

    print(
        f"Baseline R:    {baseline_r:+.2f}R"
    )

    print(
        f"Filtered R:    {filtered_r:+.2f}R"
    )

    print(
        f"Delta:         "
        f"{filtered_r - baseline_r:+.2f}R"
    )


# ============================================================================
# FAVORABLE-FIRST / IMMEDIATE FAILURE
# ============================================================================

print()
print("=" * 100)
print("FAVORABLE-FIRST / IMMEDIATE-FAILURE")
print("=" * 100)

if "favorable_first" in df.columns:

    print(
        "Favorable first counts:"
    )

    print(
        df["favorable_first"]
        .value_counts(dropna=False)
        .to_string()
    )

else:

    print(
        "favorable_first column unavailable."
    )


if "immediate_failure" in df.columns:

    print()
    print(
        "Immediate failure counts:"
    )

    print(
        df["immediate_failure"]
        .value_counts(dropna=False)
        .to_string()
    )

else:

    print(
        "immediate_failure column unavailable."
    )


# ============================================================================
# YEAR ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("YEAR-BY-YEAR BANK NIFTY")
print("=" * 100)

df["entry_date_parsed"] = pd.to_datetime(
    df["entry_date"],
    errors="coerce"
)

df["year"] = (
    df["entry_date_parsed"]
    .dt.year
)

year_rows = []

for year, group in df.groupby(
    "year",
    dropna=False
):

    result = summarize_group(
        group,
        str(int(year))
        if pd.notna(year)
        else "UNKNOWN"
    )

    if mfe_col in group.columns:

        result["avg_mfe"] = (
            group[mfe_col].mean()
        )

    if mae_col in group.columns:

        result["avg_mae"] = (
            group[mae_col].mean()
        )

    year_rows.append(result)


year_df = pd.DataFrame(
    year_rows
)

print(
    year_df.to_string(
        index=False,
        formatters={
            "win_rate": lambda x: pct(x),
            "gross_profit_r": lambda x: r(x),
            "gross_loss_r": lambda x: r(x),
            "profit_factor": lambda x: (
                f"{x:.3f}"
                if np.isfinite(x)
                else "INF"
            ),
            "total_r": lambda x: r(x),
            "avg_r": lambda x: r(x),
            "avg_mfe": lambda x: (
                f"{x:.3f}R"
            ),
            "avg_mae": lambda x: (
                f"{x:.3f}R"
            ),
        }
    )
)


# ============================================================================
# INDIVIDUAL TRADES
# ============================================================================

print()
print("=" * 100)
print("INDIVIDUAL BANK NIFTY TRADES")
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
    "exit_reason",
]

available_display = [
    column
    for column in display_columns
    if column in df.columns
]

print(
    df[
        available_display
    ].to_string(
        index=False
    )
)


# ============================================================================
# SAVE
# ============================================================================

output_rows = []

for _, row in df.iterrows():

    record = {
        "entry_date": row.get(
            "entry_date"
        ),
        "exit_date": row.get(
            "exit_date"
        ),
        "direction": row.get(
            "direction"
        ),
        "setup": row.get(
            "setup"
        ),
        "r_multiple": row.get(
            "r_multiple"
        ),
        "bars_in_trade": row.get(
            "bars_in_trade",
            np.nan
        ),
        "max_favorable_r": row.get(
            "max_favorable_r",
            np.nan
        ),
        "max_adverse_r": row.get(
            "max_adverse_r",
            np.nan
        ),
        "early_adverse_r_bar_5": row.get(
            "early_adverse_r_bar_5",
            np.nan
        ),
    }

    output_rows.append(
        record
    )

diagnostic_output = pd.DataFrame(
    output_rows
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

diagnostic_output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================================
# FINAL
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)

print(
    "This analysis is descriptive."
)

print(
    "No BANK NIFTY filter is being implemented."
)

print(
    "No production code was modified."
)

print(
    "No frozen historical data was modified."
)

print(
    "No stock holdout data was modified."
)

print(
    "Any apparent edge requires controlled counterfactual "
    "and independent OOS validation."
)

print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print("=" * 100)