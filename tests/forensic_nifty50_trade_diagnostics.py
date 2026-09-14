"""
TradeSense-AI
NIFTY 50 Trade Diagnostics

Purpose:
    Analyze the completed NIFTY 50 research backtest.

IMPORTANT:
    - Research only.
    - Does NOT modify production code.
    - Does NOT modify tradesense_trades.csv.
    - Does NOT modify tradesense_holdout_trades.csv.
    - Does NOT modify the NIFTY backtest itself.
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "tests/output/index_backtests/"
    "nifty50_index_trades.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

TRADE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nifty50_trade_diagnostics.csv"
)

REPORT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nifty50_trade_diagnostics_report.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_trades():
    print()
    print("=" * 100)
    print("TRADESENSE-AI — NIFTY 50 TRADE DIAGNOSTICS")
    print("=" * 100)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"NIFTY 50 trade file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise RuntimeError(
            "NIFTY 50 trade file is empty."
        )

    print()
    print(f"Input file:       {INPUT_FILE}")
    print(f"Trades loaded:    {len(df)}")

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_data(df):

    date_columns = [
        "entry_date",
        "exit_date"
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    numeric_columns = [
        "profit",
        "r_multiple",
        "mfe_r",
        "mae_r",
        "bars_in_trade",
        "entry_atr",
        "initial_risk",
        "entry_price",
        "exit_price",
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    if "r_multiple" not in df.columns:
        raise RuntimeError(
            "Required field 'r_multiple' is missing."
        )

    df["winner"] = (
        df["r_multiple"] > 0
    )

    df["loser"] = (
        df["r_multiple"] <= 0
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def print_summary(df):

    total_trades = len(df)
    winners = int(df["winner"].sum())
    losers = int(df["loser"].sum())

    total_r = df["r_multiple"].sum()

    win_rate = (
        winners / total_trades * 100
        if total_trades
        else 0
    )

    print()
    print("=" * 100)
    print("BASELINE SUMMARY")
    print("=" * 100)

    print(
        f"Trades:              {total_trades}"
    )

    print(
        f"Winners:             {winners}"
    )

    print(
        f"Losers:              {losers}"
    )

    print(
        f"Win rate:            {win_rate:.2f}%"
    )

    print(
        f"Total R:             {total_r:.2f}R"
    )

    if "profit" in df.columns:
        print(
            f"Total P&L:           "
            f"₹{df['profit'].sum():,.2f}"
        )


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_analysis(
    df,
    column
):

    if column not in df.columns:
        return None

    grouped = (
        df.groupby(
            column,
            dropna=False
        )
        .agg(
            trades=("r_multiple", "size"),
            winners=("winner", "sum"),
            total_r=("r_multiple", "sum"),
            avg_r=("r_multiple", "mean"),
        )
        .reset_index()
    )

    grouped["win_rate_pct"] = (
        grouped["winners"]
        / grouped["trades"]
        * 100
    )

    return grouped


def print_group_analysis(
    df,
    column,
    title
):

    result = group_analysis(
        df,
        column
    )

    if result is None:
        return

    print()
    print("-" * 100)
    print(title)
    print("-" * 100)

    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# MFE / MAE BUCKETS
# ============================================================

def add_path_buckets(df):

    if "mfe_r" in df.columns:

        mfe_bins = [
            -np.inf,
            0.25,
            0.50,
            1.00,
            1.50,
            2.00,
            np.inf
        ]

        mfe_labels = [
            "<0.25R",
            "0.25-0.50R",
            "0.50-1R",
            "1-1.5R",
            "1.5-2R",
            ">=2R"
        ]

        df["mfe_bucket"] = pd.cut(
            df["mfe_r"],
            bins=mfe_bins,
            labels=mfe_labels
        )

    if "mae_r" in df.columns:

        mae_bins = [
            -np.inf,
            0.25,
            0.50,
            0.75,
            1.00,
            1.50,
            np.inf
        ]

        mae_labels = [
            "<0.25R",
            "0.25-0.50R",
            "0.50-0.75R",
            "0.75-1R",
            "1-1.5R",
            ">=1.5R"
        ]

        df["mae_bucket"] = pd.cut(
            df["mae_r"],
            bins=mae_bins,
            labels=mae_labels
        )

    if "bars_in_trade" in df.columns:

        holding_bins = [
            0,
            5,
            10,
            20,
            50,
            100,
            np.inf
        ]

        holding_labels = [
            "1-5",
            "6-10",
            "11-20",
            "21-50",
            "51-100",
            ">100"
        ]

        df["holding_bucket"] = pd.cut(
            df["bars_in_trade"],
            bins=holding_bins,
            labels=holding_labels
        )

    return df


# ============================================================
# EARLY DEVELOPMENT
# ============================================================

def add_early_development(df):

    if "mfe_r" not in df.columns:
        return df

    conditions = [
        df["mfe_r"] < 0.50,
        (
            (df["mfe_r"] >= 0.50)
            & (df["mfe_r"] < 1.00)
        ),
        (
            (df["mfe_r"] >= 1.00)
            & (df["mfe_r"] < 1.50)
        ),
        df["mfe_r"] >= 1.50
    ]

    choices = [
        "NEVER_0.5R",
        "0.5R_TO_1R",
        "1R_TO_1.5R",
        ">=1.5R"
    ]

    df["early_development"] = np.select(
        conditions,
        choices,
        default="UNKNOWN"
    )

    return df


# ============================================================
# EARLY ADVERSE ANALYSIS
# ============================================================

def add_early_adverse(df):

    early_columns = [
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5"
    ]

    available = [
        column
        for column in early_columns
        if column in df.columns
    ]

    if not available:
        return df

    df["early_adverse_5bar_r"] = (
        df[available]
        .max(
            axis=1,
            skipna=True
        )
    )

    df["early_adverse_025_trigger"] = (
        df["early_adverse_5bar_r"]
        >= 0.25
    )

    return df


def print_early_adverse_analysis(df):

    if "early_adverse_025_trigger" not in df.columns:
        return

    result = (
        df.groupby(
            "early_adverse_025_trigger",
            dropna=False
        )
        .agg(
            trades=("r_multiple", "size"),
            winners=("winner", "sum"),
            total_r=("r_multiple", "sum"),
            avg_r=("r_multiple", "mean"),
        )
        .reset_index()
    )

    result["win_rate_pct"] = (
        result["winners"]
        / result["trades"]
        * 100
    )

    print()
    print("-" * 100)
    print("EARLY-ADVERSE 5-BAR / 0.25R DIAGNOSTIC")
    print("-" * 100)

    print(
        result.to_string(
            index=False
        )
    )

    baseline_r = df["r_multiple"].sum()

    filtered_r = df.loc[
        ~df["early_adverse_025_trigger"],
        "r_multiple"
    ].sum()

    delta = filtered_r - baseline_r

    print()
    print(
        f"Baseline R:          {baseline_r:.2f}R"
    )

    print(
        f"Filtered R:          {filtered_r:.2f}R"
    )

    print(
        f"Delta R:             {delta:+.2f}R"
    )

    print()
    print(
        "STATUS: DIAGNOSTIC ONLY"
    )


# ============================================================
# YEAR ANALYSIS
# ============================================================

def print_year_analysis(df):

    if "entry_date" not in df.columns:
        return

    working = df.dropna(
        subset=["entry_date"]
    ).copy()

    if working.empty:
        return

    working["year"] = (
        working["entry_date"]
        .dt.year
    )

    result = (
        working.groupby("year")
        .agg(
            trades=("r_multiple", "size"),
            winners=("winner", "sum"),
            total_r=("r_multiple", "sum"),
            avg_r=("r_multiple", "mean"),
        )
        .reset_index()
    )

    result["win_rate_pct"] = (
        result["winners"]
        / result["trades"]
        * 100
    )

    print()
    print("-" * 100)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("-" * 100)

    print(
        result.to_string(
            index=False
        )
    )


# ============================================================
# WINNERS VS LOSERS
# ============================================================

def print_winners_vs_losers(df):

    print()
    print("-" * 100)
    print("WINNERS VS LOSERS")
    print("-" * 100)

    metrics = []

    for column in [
        "mfe_r",
        "mae_r",
        "bars_in_trade",
        "entry_atr",
        "initial_risk"
    ]:

        if column not in df.columns:
            continue

        winners = df.loc[
            df["winner"],
            column
        ].mean()

        losers = df.loc[
            df["loser"],
            column
        ].mean()

        metrics.append(
            {
                "metric": column,
                "winners": winners,
                "losers": losers,
                "difference": winners - losers
            }
        )

    if metrics:
        result = pd.DataFrame(
            metrics
        )

        print(
            result.to_string(
                index=False
            )
        )


# ============================================================
# BEST / WORST TRADES
# ============================================================

def print_best_worst(df):

    columns = [
        column
        for column in [
            "entry_date",
            "exit_date",
            "direction",
            "setup",
            "r_multiple",
            "mfe_r",
            "mae_r",
            "bars_in_trade",
            "exit_reason"
        ]
        if column in df.columns
    ]

    print()
    print("-" * 100)
    print("TOP 10 TRADES")
    print("-" * 100)

    print(
        df.sort_values(
            "r_multiple",
            ascending=False
        )
        .head(10)[columns]
        .to_string(index=False)
    )

    print()
    print("-" * 100)
    print("BOTTOM 10 TRADES")
    print("-" * 100)

    print(
        df.sort_values(
            "r_multiple",
            ascending=True
        )
        .head(10)[columns]
        .to_string(index=False)
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        TRADE_OUTPUT,
        index=False
    )

    report_lines = []

    report_lines.append(
        "TRADESENSE-AI — NIFTY 50 TRADE DIAGNOSTICS"
    )

    report_lines.append(
        "=" * 100
    )

    report_lines.append(
        f"Trades: {len(df)}"
    )

    report_lines.append(
        f"Total R: {df['r_multiple'].sum():.2f}R"
    )

    report_lines.append(
        f"Win rate: "
        f"{df['winner'].mean() * 100:.2f}%"
    )

    report_lines.append("")

    report_lines.append(
        "RESEARCH STATUS:"
    )

    report_lines.append(
        "DIAGNOSTIC ONLY — NO PRODUCTION CHANGES"
    )

    report_lines.append(
        "NO FROZEN DATASET CHANGES"
    )

    report_lines.append(
        "NO HOLDOUT CHANGES"
    )

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(
                report_lines
            )
        )

    print()
    print("=" * 100)
    print("OUTPUT")
    print("=" * 100)

    print(
        f"Trade diagnostics: "
        f"{TRADE_OUTPUT}"
    )

    print(
        f"Report:            "
        f"{REPORT_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_trades()

    df = normalize_data(
        df
    )

    df = add_path_buckets(
        df
    )

    df = add_early_development(
        df
    )

    df = add_early_adverse(
        df
    )

    print_summary(
        df
    )

    print_group_analysis(
        df,
        "setup",
        "SETUP PERFORMANCE"
    )

    print_group_analysis(
        df,
        "direction",
        "DIRECTION PERFORMANCE"
    )

    print_group_analysis(
        df,
        "trend",
        "TREND CONTEXT"
    )

    print_group_analysis(
        df,
        "structure_bias",
        "STRUCTURE BIAS"
    )

    print_group_analysis(
        df,
        "liquidity",
        "LIQUIDITY"
    )

    print_group_analysis(
        df,
        "price_position",
        "PRICE POSITION"
    )

    print_group_analysis(
        df,
        "break_age",
        "BREAK AGE"
    )

    print_group_analysis(
        df,
        "exit_reason",
        "EXIT REASONS"
    )

    print_group_analysis(
        df,
        "mfe_bucket",
        "MFE DISTRIBUTION"
    )

    print_group_analysis(
        df,
        "mae_bucket",
        "MAE DISTRIBUTION"
    )

    print_group_analysis(
        df,
        "holding_bucket",
        "HOLDING PERIOD"
    )

    print_group_analysis(
        df,
        "early_development",
        "EARLY DEVELOPMENT"
    )

    print_early_adverse_analysis(
        df
    )

    print_winners_vs_losers(
        df
    )

    print_year_analysis(
        df
    )

    print_best_worst(
        df
    )

    save_outputs(
        df
    )

    print()
    print("=" * 100)
    print("NIFTY 50 DIAGNOSTICS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()