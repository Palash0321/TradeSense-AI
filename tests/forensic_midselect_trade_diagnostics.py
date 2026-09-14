"""
TradeSense-AI
NIFTY MIDCAP SELECT Trade Outcome Forensic

Research-only diagnostic.

Purpose:
- Analyze the 13-trade MIDSELECT baseline.
- Identify structural loss/win patterns.
- Examine setup, direction, MFE, MAE,
  early development, early adverse behaviour,
  exits, holding periods, and years.
- No production changes.
- No historical stock dataset changes.
- No holdout changes.
- No filter is implemented.
"""

import os

import numpy as np
import pandas as pd


TRADE_FILE = (
    "tests/output/index_backtests/"
    "midselect_index_trades.csv"
)


EARLY_ADVERSE_THRESHOLD = 0.25


# ============================================================================
# HELPERS
# ============================================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def print_section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def profile_group(
    df,
    group_column,
):
    if df.empty:
        return

    print()
    print(
        f"BY {group_column.upper()}"
    )

    grouped = df.groupby(
        group_column,
        dropna=False,
    )

    for name, group in grouped:

        r = numeric(
            group["r_multiple"]
        ).fillna(0)

        winners = int(
            (r > 0).sum()
        )

        losers = int(
            (r <= 0).sum()
        )

        gp = r[
            r > 0
        ].sum()

        gl = abs(
            r[
                r < 0
            ].sum()
        )

        pf = (
            gp / gl
            if gl > 0
            else np.inf
        )

        win_rate = (
            winners
            / len(group)
            * 100
            if len(group)
            else 0
        )

        print()
        print(
            f"{name}:"
        )

        print(
            f"  Trades       : {len(group)}"
        )

        print(
            f"  Winners      : {winners}"
        )

        print(
            f"  Losers       : {losers}"
        )

        print(
            f"  Win rate     : {win_rate:.2f}%"
        )

        print(
            f"  Gross profit : {gp:+.2f}R"
        )

        print(
            f"  Gross loss   : {-gl:+.2f}R"
        )

        print(
            f"  Total R      : {r.sum():+.2f}R"
        )

        print(
            f"  Profit factor: "
            f"{pf:.3f}"
            if np.isfinite(pf)
            else
            f"  Profit factor: INF"
        )


# ============================================================================
# LOAD
# ============================================================================

print_section(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT TRADE OUTCOME FORENSIC"
)

print(
    f"Trade file: {TRADE_FILE}"
)


if not os.path.exists(
    TRADE_FILE
):

    print()
    print(
        "ERROR: Trade file not found."
    )

    raise SystemExit(1)


df = pd.read_csv(
    TRADE_FILE
)


if df.empty:

    print()
    print(
        "ERROR: Trade file is empty."
    )

    raise SystemExit(1)


print(
    f"Total trades: {len(df)}"
)


# ============================================================================
# REQUIRED COLUMNS
# ============================================================================

required_columns = [
    "direction",
    "setup",
    "entry_trend",
    "entry_trend_strength",
    "entry_trend_slope",
    "entry_price_position",
    "entry_date",
    "exit_date",
    "entry_atr",
    "initial_risk",
    "r_multiple",
    "mae_percent",
    "mfe_percent",
    "bars_in_trade",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
    "reached_0_5r",
    "reached_1r",
    "reached_1_5r",
    "bars_to_0_5r",
    "bars_to_1r",
    "bars_to_1_5r",
    "favorable_first",
    "immediate_failure",
    "exit_reason",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    print()
    print(
        "ERROR: Required columns missing:"
    )

    for column in missing:
        print(
            f"  - {column}"
        )

    raise SystemExit(1)


# ============================================================================
# NORMALIZATION
# ============================================================================

for column in [
    "entry_atr",
    "initial_risk",
    "r_multiple",
    "mae_percent",
    "mfe_percent",
    "bars_in_trade",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
    "bars_to_0_5r",
    "bars_to_1r",
    "bars_to_1_5r",
    "entry_trend_strength",
    "entry_trend_slope",
]:

    df[column] = numeric(
        df[column]
    )


df["entry_date"] = pd.to_datetime(
    df["entry_date"],
    errors="coerce",
)


df["exit_date"] = pd.to_datetime(
    df["exit_date"],
    errors="coerce",
)


# ============================================================================
# BASELINE
# ============================================================================

r = df["r_multiple"].fillna(0)

winners = int(
    (r > 0).sum()
)

losers = int(
    (r <= 0).sum()
)

gross_profit = r[
    r > 0
].sum()

gross_loss = abs(
    r[
        r < 0
    ].sum()
)

profit_factor = (
    gross_profit
    / gross_loss
    if gross_loss > 0
    else np.inf
)

win_rate = (
    winners
    / len(df)
    * 100
)

print_section(
    "BASELINE"
)

print(
    f"Trades       : {len(df)}"
)

print(
    f"Winners      : {winners}"
)

print(
    f"Losers       : {losers}"
)

print(
    f"Win rate     : {win_rate:.2f}%"
)

print(
    f"Gross profit : {gross_profit:+.2f}R"
)

print(
    f"Gross loss   : {-gross_loss:+.2f}R"
)

print(
    f"Profit factor: {profit_factor:.3f}"
)

print(
    f"Total R      : {r.sum():+.2f}R"
)

print(
    f"Average R    : {r.mean():+.3f}R"
)


# ============================================================================
# SETUP
# ============================================================================

print_section(
    "SETUP PROFILE"
)

profile_group(
    df,
    "setup",
)


# ============================================================================
# DIRECTION
# ============================================================================

print_section(
    "DIRECTION PROFILE"
)

profile_group(
    df,
    "direction",
)


# ============================================================================
# EXIT
# ============================================================================

print_section(
    "EXIT PROFILE"
)

profile_group(
    df,
    "exit_reason",
)


# ============================================================================
# TREND
# ============================================================================

print_section(
    "ENTRY TREND PROFILE"
)

profile_group(
    df,
    "entry_trend",
)


# ============================================================================
# PRICE POSITION
# ============================================================================

print_section(
    "ENTRY PRICE POSITION PROFILE"
)

profile_group(
    df,
    "entry_price_position",
)


# ============================================================================
# MFE
# ============================================================================

print_section(
    "MFE PROFILE"
)

mfe_bins = [
    ("<0.25R", 0.0, 0.25),
    ("0.25-0.50R", 0.25, 0.50),
    ("0.50-1.00R", 0.50, 1.00),
    ("1.00-1.50R", 1.00, 1.50),
    ("1.50-2.00R", 1.50, 2.00),
    (">=2.00R", 2.00, np.inf),
]


for label, lower, upper in mfe_bins:

    mask = (
        df["max_favorable_r"]
        >= lower
    )

    if np.isfinite(upper):

        mask &= (
            df["max_favorable_r"]
            < upper
        )

    group = df.loc[
        mask
    ]

    group_r = group[
        "r_multiple"
    ].fillna(0)

    print(
        f"{label:<15}: "
        f"{len(group):>3} trades | "
        f"{(group_r > 0).sum():>2}W | "
        f"{group_r.sum():+.2f}R"
    )


# ============================================================================
# MFE MILESTONES
# ============================================================================

print_section(
    "MFE MILESTONES"
)

for column, label in [
    ("reached_0_5r", ">= 0.50R"),
    ("reached_1r", ">= 1.00R"),
    ("reached_1_5r", ">= 1.50R"),
]:

    mask = (
        df[column]
        .astype(bool)
    )

    group = df.loc[
        mask
    ]

    group_r = group[
        "r_multiple"
    ].fillna(0)

    print(
        f"{label:<10}: "
        f"{len(group):>3}/{len(df)} trades | "
        f"outcome {group_r.sum():+.2f}R"
    )


# ============================================================================
# EARLY DEVELOPMENT
# ============================================================================

print_section(
    "EARLY DEVELOPMENT"
)

conditions = [
    (
        "NEVER_0_5R",
        ~df["reached_0_5r"].astype(bool),
    ),
    (
        "0_5_TO_1R",
        df["reached_0_5r"].astype(bool)
        & ~df["reached_1r"].astype(bool),
    ),
    (
        "1_TO_1_5R",
        df["reached_1r"].astype(bool)
        & ~df["reached_1_5r"].astype(bool),
    ),
    (
        "GE_1_5R",
        df["reached_1_5r"].astype(bool),
    ),
]


for label, mask in conditions:

    group = df.loc[
        mask
    ]

    group_r = group[
        "r_multiple"
    ].fillna(0)

    print(
        f"{label:<14}: "
        f"{len(group):>3} trades | "
        f"{group_r.sum():+.2f}R"
    )


# ============================================================================
# MAE PROFILE
# ============================================================================

print_section(
    "MAE PROFILE"
)

mae_bins = [
    ("0-0.25R", 0.0, 0.25),
    ("0.25-0.50R", 0.25, 0.50),
    ("0.50-0.75R", 0.50, 0.75),
    ("0.75-1.00R", 0.75, 1.00),
    ("1.00-1.50R", 1.00, 1.50),
    (">=1.50R", 1.50, np.inf),
]


for label, lower, upper in mae_bins:

    mask = (
        df["max_adverse_r"]
        >= lower
    )

    if np.isfinite(upper):

        mask &= (
            df["max_adverse_r"]
            < upper
        )

    group = df.loc[
        mask
    ]

    group_r = group[
        "r_multiple"
    ].fillna(0)

    print(
        f"{label:<15}: "
        f"{len(group):>3} trades | "
        f"{(group_r > 0).sum():>2}W | "
        f"{group_r.sum():+.2f}R"
    )


# ============================================================================
# EARLY ADVERSE
# ============================================================================

print_section(
    "EARLY ADVERSE DEVELOPMENT"
)

early_columns = [
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
]


for column in early_columns:

    values = df[column].dropna()

    if values.empty:
        continue

    print(
        f"{column:<28}: "
        f"avg={values.mean():.3f}R | "
        f"median={values.median():.3f}R | "
        f"max={values.max():.3f}R"
    )


early5 = df[
    "early_adverse_r_bar_5"
]


trigger_mask = (
    early5 >= EARLY_ADVERSE_THRESHOLD
)


triggered = df.loc[
    trigger_mask
]

not_triggered = df.loc[
    ~trigger_mask
]


triggered_r = triggered[
    "r_multiple"
].fillna(0)

not_triggered_r = not_triggered[
    "r_multiple"
].fillna(0)


print()
print(
    f"5B >= {EARLY_ADVERSE_THRESHOLD:.2f}R:"
)

print()
print("Triggered:")

print(
    f"  Trades       : {len(triggered)}"
)

print(
    f"  Winners      : "
    f"{(triggered_r > 0).sum()}"
)

print(
    f"  Losers       : "
    f"{(triggered_r <= 0).sum()}"
)

print(
    f"  Win rate     : "
    f"{((triggered_r > 0).mean() * 100):.2f}%"
    if len(triggered)
    else
    "  Win rate     : 0.00%"
)

print(
    f"  Total R      : "
    f"{triggered_r.sum():+.2f}R"
)

print()
print("Not triggered:")

print(
    f"  Trades       : {len(not_triggered)}"
)

print(
    f"  Winners      : "
    f"{(not_triggered_r > 0).sum()}"
)

print(
    f"  Losers       : "
    f"{(not_triggered_r <= 0).sum()}"
)

print(
    f"  Win rate     : "
    f"{((not_triggered_r > 0).mean() * 100):.2f}%"
    if len(not_triggered)
    else
    "  Win rate     : 0.00%"
)

print(
    f"  Total R      : "
    f"{not_triggered_r.sum():+.2f}R"
)

print()
print(
    "Diagnostic filtered result "
    "(retain non-triggered only):"
)

print(
    f"  Trades       : {len(not_triggered)}"
)

print(
    f"  Total R      : "
    f"{not_triggered_r.sum():+.2f}R"
)

print(
    f"  Delta vs baseline: "
    f"{not_triggered_r.sum() - r.sum():+.2f}R"
)


# ============================================================================
# FAVORABLE / ADVERSE ORDER
# ============================================================================

print_section(
    "FAVORABLE / ADVERSE ORDER"
)

profile_group(
    df,
    "favorable_first",
)


# ============================================================================
# IMMEDIATE FAILURE
# ============================================================================

print_section(
    "IMMEDIATE FAILURE"
)

profile_group(
    df,
    "immediate_failure",
)


# ============================================================================
# HOLDING PERIOD
# ============================================================================

print_section(
    "HOLDING PERIOD"
)

holding_bins = [
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    (">100", 101, np.inf),
]


for label, lower, upper in holding_bins:

    mask = (
        df["bars_in_trade"]
        >= lower
    )

    if np.isfinite(upper):

        mask &= (
            df["bars_in_trade"]
            <= upper
        )

    group = df.loc[
        mask
    ]

    group_r = group[
        "r_multiple"
    ].fillna(0)

    print(
        f"{label:<8}: "
        f"{len(group):>3} trades | "
        f"{group_r.sum():+.2f}R"
    )


# ============================================================================
# WINNERS VS LOSERS
# ============================================================================

print_section(
    "WINNERS VS LOSERS"
)

winner_df = df[
    df["r_multiple"] > 0
]

loser_df = df[
    df["r_multiple"] <= 0
]


comparison_columns = [
    (
        "entry_atr",
        "ATR",
    ),
    (
        "initial_risk",
        "Initial risk",
    ),
    (
        "max_favorable_r",
        "MFE",
    ),
    (
        "max_adverse_r",
        "MAE",
    ),
    (
        "bars_in_trade",
        "Holding bars",
    ),
    (
        "early_adverse_r_bar_5",
        "Early adverse bar 5",
    ),
]


for column, label in comparison_columns:

    winner_values = winner_df[
        column
    ].dropna()

    loser_values = loser_df[
        column
    ].dropna()

    print()
    print(
        f"{label}:"
    )

    print(
        f"  Winners: "
        f"{winner_values.mean():.3f}"
        if not winner_values.empty
        else
        "  Winners: N/A"
    )

    print(
        f"  Losers : "
        f"{loser_values.mean():.3f}"
        if not loser_values.empty
        else
        "  Losers : N/A"
    )


# ============================================================================
# YEAR PROFILE
# ============================================================================

print_section(
    "YEAR PROFILE"
)

df["year"] = (
    df["entry_date"]
    .dt.year
)


profile_group(
    df,
    "year",
)


# ============================================================================
# INDIVIDUAL TRADES
# ============================================================================

print_section(
    "INDIVIDUAL MIDSELECT TRADES"
)

display_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_5",
    "bars_in_trade",
    "exit_reason",
]


for _, row in (
    df[
        display_columns
    ]
    .sort_values("entry_date")
    .iterrows()
):

    print(
        f"{row['entry_date'].date()} | "
        f"{row['exit_date'].date()} | "
        f"{str(row['direction']):<5} | "
        f"{str(row['setup']):<20} | "
        f"R={row['r_multiple']:+.2f} | "
        f"MFE={row['max_favorable_r']:.2f} | "
        f"MAE={row['max_adverse_r']:.2f} | "
        f"Early5="
        f"{row['early_adverse_r_bar_5']:.2f} | "
        f"bars={int(row['bars_in_trade'])} | "
        f"{row['exit_reason']}"
    )


# ============================================================================
# DATA INTEGRITY
# ============================================================================

print_section(
    "FORENSIC DATA INTEGRITY"
)

duplicate_keys = int(
    df.duplicated(
        subset=[
            "entry_date",
            "direction",
            "setup",
        ]
    ).sum()
)


missing_r = int(
    df["r_multiple"]
    .isna()
    .sum()
)


print(
    f"Duplicate trade identities: "
    f"{duplicate_keys}"
)

print(
    f"Missing r_multiple values:  "
    f"{missing_r}"
)


if (
    duplicate_keys == 0
    and missing_r == 0
):

    print(
        "INTEGRITY STATUS: PASS"
    )

else:

    print(
        "INTEGRITY STATUS: ATTENTION"
    )


# ============================================================================
# FINAL VERDICT
# ============================================================================

print_section(
    "RESEARCH VERDICT"
)

print(
    f"MIDSELECT baseline trades : {len(df)}"
)

print(
    f"MIDSELECT baseline Total R: "
    f"{r.sum():+.2f}R"
)

print(
    f"MIDSELECT baseline PF     : "
    f"{profit_factor:.3f}"
)

print()

if len(df) < 20:

    print(
        "VERDICT: INSUFFICIENT_SAMPLE"
    )

    print()
    print(
        "The MIDSELECT baseline contains "
        f"only {len(df)} trades."
    )

    print(
        "Any setup/filter/counterfactual "
        "pattern must be treated as descriptive "
        "until independently validated."
    )

else:

    print(
        "VERDICT: FORENSIC_COMPLETE"
    )


print()
print(
    "This is a research-only diagnostic."
)

print(
    "No production, frozen, or holdout "
    "dataset was modified."
)