"""
TradeSense-AI
SENSEX Index Trade Diagnostics

Purpose:
    Forensic analysis of the SENSEX baseline index backtest.

IMPORTANT:
    - Research only.
    - Does NOT modify production code.
    - Does NOT modify frozen historical trades.
    - Does NOT modify holdout trades.
    - Does NOT optimize or select a trading rule.
"""

import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRADES_FILE = (
    "tests/output/index_backtests/"
    "sensex_index_trades.csv"
)

INDEX_NAME = "SENSEX"


# ============================================================
# HELPERS
# ============================================================

def load_trades():
    if not os.path.exists(TRADES_FILE):
        raise FileNotFoundError(
            f"Trade file not found: {TRADES_FILE}"
        )

    df = pd.read_csv(TRADES_FILE)

    if df.empty:
        raise RuntimeError(
            "SENSEX trade file is empty."
        )

    return df


def numeric(df, column):
    if column not in df.columns:
        return pd.Series(
            [float("nan")] * len(df),
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def print_separator(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def print_group_analysis(
    df,
    group_column,
    title
):
    print_separator(title)

    if group_column not in df.columns:
        print(
            f"Column '{group_column}' not found."
        )
        return

    grouped = []

    for value, group in df.groupby(
        group_column,
        dropna=False
    ):
        r = numeric(group, "r_multiple")

        winners = int(
            (r > 0).sum()
        )

        losers = int(
            (r <= 0).sum()
        )

        total_r = float(
            r.sum()
        )

        gross_profit = float(
            r[r > 0].sum()
        )

        gross_loss = float(
            r[r < 0].sum()
        )

        if gross_loss < 0:
            profit_factor = (
                gross_profit / abs(gross_loss)
            )
        else:
            profit_factor = float("inf")

        win_rate = (
            winners / len(group) * 100
            if len(group) > 0
            else 0
        )

        grouped.append(
            {
                "group": value,
                "trades": len(group),
                "winners": winners,
                "losers": losers,
                "win_rate": win_rate,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor,
                "total_r": total_r,
            }
        )

    result = pd.DataFrame(grouped)

    result = result.sort_values(
        "total_r",
        ascending=False
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}%".format,
                "gross_profit": "{:.2f}R".format,
                "gross_loss": "{:.2f}R".format,
                "profit_factor": "{:.3f}".format,
                "total_r": "{:.2f}R".format,
            }
        )
    )


# ============================================================
# BASIC METRICS
# ============================================================

def basic_metrics(df):
    print_separator(
        "SENSEX BASELINE TRADE DIAGNOSTICS"
    )

    r = numeric(df, "r_multiple")

    winners = int(
        (r > 0).sum()
    )

    losers = int(
        (r <= 0).sum()
    )

    gross_profit = float(
        r[r > 0].sum()
    )

    gross_loss = float(
        r[r < 0].sum()
    )

    total_r = float(
        r.sum()
    )

    if gross_loss < 0:
        profit_factor = (
            gross_profit / abs(gross_loss)
        )
    else:
        profit_factor = float("inf")

    win_rate = (
        winners / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(f"Index:              {INDEX_NAME}")
    print(f"Trades:             {len(df)}")
    print(f"Winners:            {winners}")
    print(f"Losers:             {losers}")
    print(f"Win rate:           {win_rate:.2f}%")
    print(f"Gross profit:       {gross_profit:.2f}R")
    print(f"Gross loss:         {gross_loss:.2f}R")
    print(f"Profit factor:      {profit_factor:.3f}")
    print(f"Total R:            {total_r:.2f}R")
    print(
        f"Average R/trade:    "
        f"{total_r / len(df):.3f}R"
    )


# ============================================================
# EARLY-ADVERSE ANALYSIS
# ============================================================

def early_adverse_analysis(df):
    print_separator(
        "EARLY-ADVERSE 5-BAR / 0.25R DIAGNOSTIC"
    )

    early = numeric(
        df,
        "early_adverse_r_bar_5"
    )

    valid = early.notna()

    if not valid.any():
        print(
            "No valid early-adverse bar-5 data."
        )
        return

    triggered = df.loc[
        valid & (early >= 0.25)
    ]

    non_triggered = df.loc[
        valid & (early < 0.25)
    ]

    def summarize(label, group):
        r = numeric(group, "r_multiple")

        winners = int(
            (r > 0).sum()
        )

        total_r = float(
            r.sum()
        )

        gross_profit = float(
            r[r > 0].sum()
        )

        gross_loss = float(
            r[r < 0].sum()
        )

        pf = (
            gross_profit / abs(gross_loss)
            if gross_loss < 0
            else float("inf")
        )

        win_rate = (
            winners / len(group) * 100
            if len(group) > 0
            else 0
        )

        avg_early = float(
            numeric(
                group,
                "early_adverse_r_bar_5"
            ).mean()
        )

        print(
            f"{label:<15}"
            f"N={len(group):>3}  "
            f"AvgEarly={avg_early:>6.3f}R  "
            f"Win%={win_rate:>6.2f}%  "
            f"PF={pf:>6.3f}  "
            f"TotalR={total_r:>7.2f}R"
        )

    summarize(
        "TRIGGERED",
        triggered
    )

    summarize(
        "NON-TRIGGERED",
        non_triggered
    )

    filtered = non_triggered

    if len(filtered) > 0:
        filtered_r = numeric(
            filtered,
            "r_multiple"
        )

        baseline_r = float(
            numeric(df, "r_multiple").sum()
        )

        filtered_total_r = float(
            filtered_r.sum()
        )

        print()
        print(
            f"Baseline Total R:      "
            f"{baseline_r:.2f}R"
        )

        print(
            f"Without Triggered:     "
            f"{filtered_total_r:.2f}R"
        )

        print(
            f"Counterfactual Delta:   "
            f"{filtered_total_r - baseline_r:+.2f}R"
        )

    print()
    print(
        "IMPORTANT: This is descriptive only."
    )
    print(
        "No early-adverse rule is being selected."
    )


# ============================================================
# MFE / MAE
# ============================================================

def mfe_mae_analysis(df):
    print_separator(
        "MFE / MAE DEVELOPMENT DIAGNOSTIC"
    )

    mfe = numeric(
        df,
        "max_favorable_r"
    )

    mae = numeric(
        df,
        "max_adverse_r"
    )

    r = numeric(
        df,
        "r_multiple"
    )

    print(
        f"Average MFE:       {mfe.mean():.3f}R"
    )

    print(
        f"Median MFE:        {mfe.median():.3f}R"
    )

    print(
        f"Average MAE:       {mae.mean():.3f}R"
    )

    print(
        f"Median MAE:        {mae.median():.3f}R"
    )

    print()
    print("WINNERS vs LOSERS")
    print("-" * 100)

    winners = df.loc[
        r > 0
    ]

    losers = df.loc[
        r <= 0
    ]

    print(
        f"Winners: N={len(winners):>3}  "
        f"Avg MFE={numeric(winners, 'max_favorable_r').mean():.3f}R  "
        f"Avg MAE={numeric(winners, 'max_adverse_r').mean():.3f}R"
    )

    print(
        f"Losers:  N={len(losers):>3}  "
        f"Avg MFE={numeric(losers, 'max_favorable_r').mean():.3f}R  "
        f"Avg MAE={numeric(losers, 'max_adverse_r').mean():.3f}R"
    )


# ============================================================
# MFE MILESTONES
# ============================================================

def mfe_milestone_analysis(df):
    print_separator(
        "MFE MILESTONE DEVELOPMENT"
    )

    mfe = numeric(
        df,
        "max_favorable_r"
    )

    r = numeric(
        df,
        "r_multiple"
    )

    thresholds = [
        0.25,
        0.50,
        1.00,
        1.50,
        2.00,
    ]

    for threshold in thresholds:
        reached = (
            mfe >= threshold
        )

        count = int(
            reached.sum()
        )

        pct = (
            count / len(df) * 100
            if len(df) > 0
            else 0
        )

        reached_r = r.loc[
            reached
        ]

        print(
            f"Reached {threshold:.2f}R: "
            f"{count:>3}/{len(df)} "
            f"({pct:>6.2f}%)  "
            f"Total R={reached_r.sum():>7.2f}R"
        )


# ============================================================
# HOLDING PERIOD
# ============================================================

def holding_period_analysis(df):
    print_separator(
        "HOLDING PERIOD DIAGNOSTIC"
    )

    bars = numeric(
        df,
        "bars_in_trade"
    )

    r = numeric(
        df,
        "r_multiple"
    )

    bins = [
        -float("inf"),
        5,
        10,
        20,
        50,
        100,
        float("inf"),
    ]

    labels = [
        "1-5",
        "6-10",
        "11-20",
        "21-50",
        "51-100",
        ">100",
    ]

    bucket = pd.cut(
        bars,
        bins=bins,
        labels=labels
    )

    temp = df.copy()
    temp["_holding_bucket"] = bucket

    for label, group in temp.groupby(
        "_holding_bucket",
        observed=False
    ):
        if len(group) == 0:
            continue

        group_r = numeric(
            group,
            "r_multiple"
        )

        print(
            f"{str(label):<8}"
            f"N={len(group):>3}  "
            f"Win%="
            f"{(group_r > 0).mean() * 100:>6.2f}%  "
            f"TotalR="
            f"{group_r.sum():>7.2f}R"
        )


# ============================================================
# YEARLY ANALYSIS
# ============================================================

def yearly_analysis(df):
    print_separator(
        "YEAR-BY-YEAR PERFORMANCE"
    )

    if "entry_date" not in df.columns:
        print(
            "entry_date column not found."
        )
        return

    dates = pd.to_datetime(
        df["entry_date"],
        errors="coerce"
    )

    temp = df.copy()

    temp["_year"] = dates.dt.year

    for year, group in temp.groupby(
        "_year",
        dropna=True
    ):
        r = numeric(
            group,
            "r_multiple"
        )

        winners = int(
            (r > 0).sum()
        )

        win_rate = (
            winners / len(group) * 100
            if len(group) > 0
            else 0
        )

        print(
            f"{int(year)}: "
            f"N={len(group):>3}  "
            f"W={winners:>2}  "
            f"Win%={win_rate:>6.2f}%  "
            f"TotalR={r.sum():>7.2f}R"
        )


# ============================================================
# DIRECTION + SETUP + EXIT
# ============================================================

def structural_breakdowns(df):
    print_group_analysis(
        df,
        "direction",
        "DIRECTION PERFORMANCE"
    )

    print_group_analysis(
        df,
        "setup",
        "SETUP PERFORMANCE"
    )

    print_group_analysis(
        df,
        "exit_reason",
        "EXIT REASON PERFORMANCE"
    )


# ============================================================
# EARLY DEVELOPMENT
# ============================================================

def early_development_analysis(df):
    print_separator(
        "EARLY TRADE DEVELOPMENT"
    )

    mfe = numeric(
        df,
        "max_favorable_r"
    )

    r = numeric(
        df,
        "r_multiple"
    )

    buckets = {
        "NEVER_0_5R": mfe < 0.5,
        "0_5_TO_UNDER_1R": (
            (mfe >= 0.5)
            & (mfe < 1.0)
        ),
        "1_TO_UNDER_1_5R": (
            (mfe >= 1.0)
            & (mfe < 1.5)
        ),
        "1_5R_PLUS": (
            mfe >= 1.5
        ),
    }

    for name, mask in buckets.items():
        group = df.loc[
            mask
        ]

        group_r = r.loc[
            mask
        ]

        if len(group) == 0:
            continue

        print(
            f"{name:<20}"
            f"N={len(group):>3}  "
            f"Win%="
            f"{(group_r > 0).mean() * 100:>6.2f}%  "
            f"TotalR="
            f"{group_r.sum():>7.2f}R"
        )


# ============================================================
# INDIVIDUAL TRADES
# ============================================================

def individual_trade_table(df):
    print_separator(
        "INDIVIDUAL SENSEX TRADES"
    )

    columns = [
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

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    table = df[available].copy()

    if "entry_date" in table.columns:
        table = table.sort_values(
            "entry_date"
        )

    print(
        table.to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print("TRADESENSE-AI — SENSEX FORENSIC TRADE DIAGNOSTICS")
    print("=" * 100)

    df = load_trades()

    print()
    print(
        f"Loaded trades: {len(df)}"
    )

    basic_metrics(df)

    structural_breakdowns(df)

    holding_period_analysis(df)

    yearly_analysis(df)

    mfe_mae_analysis(df)

    mfe_milestone_analysis(df)

    early_development_analysis(df)

    early_adverse_analysis(df)

    individual_trade_table(df)

    print()
    print("=" * 100)
    print("RESEARCH INTERPRETATION")
    print("=" * 100)

    print(
        "This diagnostic is descriptive only."
    )

    print(
        "It does NOT select a production filter."
    )

    print(
        "It does NOT modify the TradeSense strategy."
    )

    print(
        "It does NOT modify frozen historical data."
    )

    print(
        "It does NOT modify the holdout dataset."
    )

    print(
        "Any apparent edge must be validated through "
        "controlled counterfactual and out-of-sample testing."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()