"""
TradeSense-AI
NIFTY 50 Directional Counterfactual Research

Purpose
-------
Research-only analysis of the NIFTY 50 backtest to determine whether
the negative baseline result is primarily driven by short trades.

This script:
1. Reads the existing NIFTY 50 trade output.
2. Compares:
      - All trades
      - Long-only
      - Short-only
3. Breaks results down by:
      - Year
      - Setup
      - Direction
4. Builds a chronological closed-trade equity curve for:
      - All trades
      - Long-only
      - Short-only
5. Calculates closed-trade drawdown for each.
6. Performs leave-one-year-out analysis for the long-only hypothesis.
7. Does NOT modify:
      - Production code
      - Frozen historical trades
      - Holdout trades
      - Strategy parameters
      - NIFTY backtest output

This is a research diagnostic only.
"""

import csv
import os
from collections import defaultdict


INPUT_FILE = (
    "tests/output/index_backtests/"
    "nifty50_index_trades.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_directional_counterfactual.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_directional_counterfactual_report.txt"
)

INITIAL_CAPITAL = 100000.0


def load_trades():
    """Load the existing NIFTY 50 trade output."""

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)
        trades = list(reader)

    if not trades:
        raise ValueError(
            "NIFTY 50 trade file contains no trades."
        )

    return trades


def normalize_trade(trade):
    """Normalize numeric fields used by this diagnostic."""

    normalized = dict(trade)

    normalized["profit"] = float(
        trade.get("profit", 0.0) or 0.0
    )

    normalized["r_multiple"] = float(
        trade.get("r_multiple", 0.0) or 0.0
    )

    normalized["entry_date"] = str(
        trade.get("entry_date", "")
    )

    normalized["exit_date"] = str(
        trade.get("exit_date", "")
    )

    normalized["direction"] = str(
        trade.get("direction", "")
    ).upper()

    normalized["setup_type"] = str(
        trade.get(
            "setup",
            ""
        )
    )

    return normalized


def calculate_summary(trades):
    """Calculate core performance statistics."""

    count = len(trades)

    if count == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "pnl": 0.0,
        }

    wins = sum(
        1
        for trade in trades
        if trade["profit"] > 0
    )

    losses = sum(
        1
        for trade in trades
        if trade["profit"] < 0
    )

    gross_profit_r = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] > 0
    )

    gross_loss_r = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] < 0
    )

    total_r = sum(
        trade["r_multiple"]
        for trade in trades
    )

    pnl = sum(
        trade["profit"]
        for trade in trades
    )

    if gross_loss_r < 0:
        profit_factor = (
            gross_profit_r
            / abs(gross_loss_r)
        )
    else:
        profit_factor = (
            float("inf")
            if gross_profit_r > 0
            else 0.0
        )

    return {
        "trades": count,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / count * 100.0
        ),
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "profit_factor": profit_factor,
        "total_r": total_r,
        "avg_r": total_r / count,
        "pnl": pnl,
    }


def calculate_closed_trade_drawdown(
    trades,
    initial_capital=INITIAL_CAPITAL
):
    """
    Research-only closed-trade realized equity drawdown.

    This is intentionally separate from the production engine.
    """

    if not trades:
        return 0.0

    ordered = sorted(
        trades,
        key=lambda trade: (
            trade["exit_date"],
            trade["entry_date"]
        )
    )

    equity = float(initial_capital)
    peak_equity = equity
    max_drawdown = 0.0

    for trade in ordered:
        equity += trade["profit"]

        if equity > peak_equity:
            peak_equity = equity

        if peak_equity > 0:
            drawdown = (
                (peak_equity - equity)
                / peak_equity
            ) * 100.0

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

    return max_drawdown


def build_equity_curve(
    trades,
    initial_capital=INITIAL_CAPITAL
):
    """
    Build a chronological closed-trade equity curve.

    Returns:
        list of dictionaries containing:
        trade number, exit date, direction,
        trade R, P&L, equity, peak, drawdown.
    """

    ordered = sorted(
        trades,
        key=lambda trade: (
            trade["exit_date"],
            trade["entry_date"]
        )
    )

    equity = float(initial_capital)
    peak_equity = equity

    curve = []

    for number, trade in enumerate(
        ordered,
        start=1
    ):
        equity += trade["profit"]

        if equity > peak_equity:
            peak_equity = equity

        if peak_equity > 0:
            drawdown = (
                (peak_equity - equity)
                / peak_equity
            ) * 100.0
        else:
            drawdown = 0.0

        curve.append(
            {
                "trade_number": number,
                "exit_date": trade["exit_date"],
                "direction": trade["direction"],
                "setup_type": trade["setup_type"],
                "trade_r": trade["r_multiple"],
                "profit": trade["profit"],
                "equity": equity,
                "peak_equity": peak_equity,
                "drawdown_pct": drawdown,
            }
        )

    return curve


def print_summary(
    label,
    trades
):
    """Print one performance summary."""

    summary = calculate_summary(trades)

    drawdown = calculate_closed_trade_drawdown(
        trades
    )

    pf = summary["profit_factor"]

    if pf == float("inf"):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    print(
        f"Trades:               "
        f"{summary['trades']}"
    )

    print(
        f"Winners:              "
        f"{summary['wins']}"
    )

    print(
        f"Losers:               "
        f"{summary['losses']}"
    )

    print(
        f"Win Rate:             "
        f"{summary['win_rate']:.2f}%"
    )

    print(
        f"Gross Profit R:       "
        f"{summary['gross_profit_r']:.2f}R"
    )

    print(
        f"Gross Loss R:         "
        f"{summary['gross_loss_r']:.2f}R"
    )

    print(
        f"Profit Factor:        "
        f"{pf_text}"
    )

    print(
        f"Total R:              "
        f"{summary['total_r']:.2f}R"
    )

    print(
        f"Average R:            "
        f"{summary['avg_r']:.3f}R"
    )

    print(
        f"P&L:                  "
        f"₹{summary['pnl']:.2f}"
    )

    print(
        f"Closed-trade DD:      "
        f"{drawdown:.2f}%"
    )


def print_year_direction_analysis(
    trades
):
    """Print year × direction performance."""

    groups = defaultdict(list)

    for trade in trades:
        year = trade["entry_date"][:4]

        groups[
            (year, trade["direction"])
        ].append(trade)

    print()
    print("=" * 100)
    print("YEAR × DIRECTION")
    print("=" * 100)

    print(
        f"{'Year':<8}"
        f"{'Direction':<12}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>12}"
        f"{'P&L':>16}"
    )

    for key in sorted(groups):
        year, direction = key
        subset = groups[key]

        summary = calculate_summary(
            subset
        )

        print(
            f"{year:<8}"
            f"{direction:<12}"
            f"{summary['trades']:>8}"
            f"{summary['wins']:>8}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>11.2f}R"
            f"{summary['pnl']:>15.2f}"
        )


def print_setup_direction_analysis(
    trades
):
    """Print setup × direction performance."""

    groups = defaultdict(list)

    for trade in trades:
        groups[
            (
                trade["setup_type"],
                trade["direction"]
            )
        ].append(trade)

    print()
    print("=" * 100)
    print("SETUP × DIRECTION")
    print("=" * 100)

    print(
        f"{'Setup':<25}"
        f"{'Direction':<12}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>12}"
        f"{'P&L':>16}"
    )

    for key in sorted(groups):
        setup, direction = key
        subset = groups[key]

        summary = calculate_summary(
            subset
        )

        print(
            f"{setup:<25}"
            f"{direction:<12}"
            f"{summary['trades']:>8}"
            f"{summary['wins']:>8}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>11.2f}R"
            f"{summary['pnl']:>15.2f}"
        )


def print_long_only_year_analysis(
    trades
):
    """
    Analyze long-only performance by year.

    Also performs leave-one-year-out analysis.
    """

    long_trades = [
        trade
        for trade in trades
        if trade["direction"] == "LONG"
    ]

    years = sorted(
        {
            trade["entry_date"][:4]
            for trade in long_trades
        }
    )

    print()
    print("=" * 100)
    print("LONG-ONLY YEAR ANALYSIS")
    print("=" * 100)

    print(
        f"{'Year':<8}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>12}"
        f"{'P&L':>16}"
    )

    for year in years:
        subset = [
            trade
            for trade in long_trades
            if trade["entry_date"][:4] == year
        ]

        summary = calculate_summary(
            subset
        )

        print(
            f"{year:<8}"
            f"{summary['trades']:>8}"
            f"{summary['wins']:>8}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>11.2f}R"
            f"{summary['pnl']:>15.2f}"
        )

    print()
    print("=" * 100)
    print("LONG-ONLY LEAVE-ONE-YEAR-OUT")
    print("=" * 100)

    print(
        f"{'Excluded Year':<18}"
        f"{'Trades':>10}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
        f"{'P&L':>16}"
        f"{'DD%':>12}"
    )

    for excluded_year in years:
        subset = [
            trade
            for trade in long_trades
            if trade["entry_date"][:4]
            != excluded_year
        ]

        summary = calculate_summary(
            subset
        )

        drawdown = calculate_closed_trade_drawdown(
            subset
        )

        print(
            f"{excluded_year:<18}"
            f"{summary['trades']:>10}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>13.2f}R"
            f"{summary['pnl']:>15.2f}"
            f"{drawdown:>11.2f}%"
        )


def print_equity_curve_summary(
    label,
    trades
):
    """Print chronological equity-curve summary."""

    curve = build_equity_curve(
        trades
    )

    if not curve:
        return

    final_equity = curve[-1]["equity"]

    max_drawdown = max(
        row["drawdown_pct"]
        for row in curve
    )

    largest_r_loss = min(
        row["trade_r"]
        for row in curve
    )

    largest_r_gain = max(
        row["trade_r"]
        for row in curve
    )

    print()
    print("=" * 100)
    print(f"{label} EQUITY CURVE")
    print("=" * 100)

    print(
        f"Final Equity:         "
        f"₹{final_equity:.2f}"
    )

    print(
        f"Net P&L:              "
        f"₹{final_equity - INITIAL_CAPITAL:.2f}"
    )

    print(
        f"Max Closed DD:        "
        f"{max_drawdown:.2f}%"
    )

    print(
        f"Largest Winning R:    "
        f"{largest_r_gain:.2f}R"
    )

    print(
        f"Largest Losing R:     "
        f"{largest_r_loss:.2f}R"
    )


def save_results(
    all_trades,
    long_trades,
    short_trades
):
    """Save the counterfactual summary."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    rows = []

    datasets = [
        ("ALL", all_trades),
        ("LONG_ONLY", long_trades),
        ("SHORT_ONLY", short_trades),
    ]

    for label, trades in datasets:
        summary = calculate_summary(
            trades
        )

        drawdown = calculate_closed_trade_drawdown(
            trades
        )

        rows.append(
            {
                "scenario": label,
                "trades": summary["trades"],
                "wins": summary["wins"],
                "losses": summary["losses"],
                "win_rate": round(
                    summary["win_rate"],
                    4
                ),
                "gross_profit_r": round(
                    summary["gross_profit_r"],
                    4
                ),
                "gross_loss_r": round(
                    summary["gross_loss_r"],
                    4
                ),
                "profit_factor": (
                    "INF"
                    if summary["profit_factor"]
                    == float("inf")
                    else round(
                        summary["profit_factor"],
                        4
                    )
                ),
                "total_r": round(
                    summary["total_r"],
                    4
                ),
                "avg_r": round(
                    summary["avg_r"],
                    4
                ),
                "pnl": round(
                    summary["pnl"],
                    2
                ),
                "closed_trade_drawdown_pct":
                    round(
                        drawdown,
                        4
                    ),
            }
        )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fieldnames = list(
            rows[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Saved counterfactual summary: "
        f"{OUTPUT_FILE}"
    )


def write_report(
    all_trades,
    long_trades,
    short_trades
):
    """Write a compact text research report."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    scenarios = [
        ("ALL", all_trades),
        ("LONG_ONLY", long_trades),
        ("SHORT_ONLY", short_trades),
    ]

    lines = []

    lines.append(
        "TradeSense-AI NIFTY 50 "
        "Directional Counterfactual"
    )

    lines.append(
        "=" * 80
    )

    lines.append(
        "Research-only diagnostic."
    )

    lines.append(
        "No production or frozen datasets modified."
    )

    lines.append("")

    for label, trades in scenarios:
        summary = calculate_summary(
            trades
        )

        dd = calculate_closed_trade_drawdown(
            trades
        )

        lines.append(label)
        lines.append("-" * 40)

        lines.append(
            f"Trades: {summary['trades']}"
        )

        lines.append(
            f"Wins: {summary['wins']}"
        )

        lines.append(
            f"Losses: {summary['losses']}"
        )

        lines.append(
            f"Win Rate: "
            f"{summary['win_rate']:.2f}%"
        )

        lines.append(
            f"Total R: "
            f"{summary['total_r']:.2f}R"
        )

        lines.append(
            f"Average R: "
            f"{summary['avg_r']:.3f}R"
        )

        lines.append(
            f"P&L: "
            f"₹{summary['pnl']:.2f}"
        )

        lines.append(
            f"Closed-trade DD: "
            f"{dd:.2f}%"
        )

        lines.append("")

    long_summary = calculate_summary(
        long_trades
    )

    all_summary = calculate_summary(
        all_trades
    )

    short_summary = calculate_summary(
        short_trades
    )

    lines.append(
        "COUNTERFACTUAL DIFFERENCES"
    )

    lines.append("-" * 40)

    lines.append(
        "Long-only vs all-trades:"
    )

    lines.append(
        f"Trade count change: "
        f"{long_summary['trades'] - all_summary['trades']}"
    )

    lines.append(
        f"Total R change: "
        f"{long_summary['total_r'] - all_summary['total_r']:.2f}R"
    )

    lines.append(
        f"P&L change: "
        f"₹{long_summary['pnl'] - all_summary['pnl']:.2f}"
    )

    lines.append("")

    lines.append(
        "Short-only contribution:"
    )

    lines.append(
        f"Trades: "
        f"{short_summary['trades']}"
    )

    lines.append(
        f"Total R: "
        f"{short_summary['total_r']:.2f}R"
    )

    lines.append(
        f"P&L: "
        f"₹{short_summary['pnl']:.2f}"
    )

    lines.append("")

    lines.append(
        "Interpretation rule:"
    )

    lines.append(
        "A positive long-only result is NOT sufficient "
        "to deploy a long-only NIFTY strategy."
    )

    lines.append(
        "It must subsequently survive chronological, "
        "year-level, setup-level and unseen-data validation."
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines)
        )

    print(
        f"Saved report: {REPORT_FILE}"
    )


def main():
    print()
    print("=" * 100)
    print("TRADESENSE-AI — NIFTY 50 DIRECTIONAL COUNTERFACTUAL")
    print("=" * 100)

    trades = [
        normalize_trade(trade)
        for trade in load_trades()
    ]

    all_trades = trades

    long_trades = [
        trade
        for trade in trades
        if trade["direction"] == "LONG"
    ]

    short_trades = [
        trade
        for trade in trades
        if trade["direction"] == "SHORT"
    ]

    print(
        f"Input file:           {INPUT_FILE}"
    )

    print(
        f"Total loaded trades:  "
        f"{len(all_trades)}"
    )

    print_summary(
        "ALL TRADES",
        all_trades
    )

    print_summary(
        "LONG ONLY",
        long_trades
    )

    print_summary(
        "SHORT ONLY",
        short_trades
    )

    print_year_direction_analysis(
        all_trades
    )

    print_setup_direction_analysis(
        all_trades
    )

    print_long_only_year_analysis(
        all_trades
    )

    print_equity_curve_summary(
        "ALL TRADES",
        all_trades
    )

    print_equity_curve_summary(
        "LONG ONLY",
        long_trades
    )

    print_equity_curve_summary(
        "SHORT ONLY",
        short_trades
    )

    save_results(
        all_trades,
        long_trades,
        short_trades
    )

    write_report(
        all_trades,
        long_trades,
        short_trades
    )

    print()
    print("=" * 100)
    print("DIRECTIONAL COUNTERFACTUAL COMPLETE")
    print("=" * 100)

    print(
        "Production code:       UNCHANGED"
    )

    print(
        "Frozen stock dataset:  UNCHANGED"
    )

    print(
        "Holdout dataset:       UNCHANGED"
    )


if __name__ == "__main__":
    main()