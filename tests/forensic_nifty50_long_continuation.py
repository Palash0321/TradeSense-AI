"""
TradeSense-AI — NIFTY 50 LONG-CONTINUATION FORENSIC

Research-only diagnostic.

Purpose:
    Compare:
        1. ALL TRADES
        2. LONG ONLY
        3. LONG_CONTINUATION ONLY

Then evaluate LONG_CONTINUATION chronologically and by
leave-one-year-out robustness.

This script MUST NOT modify:
    - production code
    - frozen stock dataset
    - holdout dataset

Input:
    tests/output/index_backtests/nifty50_index_trades.csv

Output:
    tests/output/index_backtests/nifty50_long_continuation.csv
    tests/output/index_backtests/nifty50_long_continuation_report.txt
"""

import csv
import os
from collections import defaultdict


INPUT_FILE = "tests/output/index_backtests/nifty50_index_trades.csv"

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "nifty50_long_continuation.csv"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "nifty50_long_continuation_report.txt"
)

INITIAL_CAPITAL = 100000.0


def load_trades():
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
        rows = list(reader)

    trades = []

    for row in rows:
        try:
            r_multiple = float(
                row.get("r_multiple", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            r_multiple = 0.0

        try:
            profit = float(
                row.get("profit", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            profit = 0.0

        setup = str(
            row.get("setup", "")
        ).strip()

        direction = str(
            row.get("direction", "")
        ).strip()

        entry_date = str(
            row.get("entry_date", "")
        ).strip()

        exit_date = str(
            row.get("exit_date", "")
        ).strip()

        exit_reason = str(
            row.get("exit_reason", "")
        ).strip()

        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "setup": setup,
                "r_multiple": r_multiple,
                "profit": profit,
                "exit_reason": exit_reason,
            }
        )

    return trades


def calculate_profit_factor(trades):
    gross_profit = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] > 0
    )

    gross_loss = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] < 0
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return gross_profit / abs(gross_loss)


def calculate_drawdown(trades):
    ordered = sorted(
        trades,
        key=lambda trade: trade["exit_date"]
    )

    equity = INITIAL_CAPITAL
    peak = equity
    max_drawdown = 0.0

    for trade in ordered:
        equity += trade["profit"]

        if equity > peak:
            peak = equity

        if peak > 0:
            drawdown = (
                (peak - equity) / peak
            ) * 100.0

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

    return max_drawdown


def summarize(trades):
    count = len(trades)

    winners = sum(
        1
        for trade in trades
        if trade["r_multiple"] > 0
    )

    losers = sum(
        1
        for trade in trades
        if trade["r_multiple"] < 0
    )

    gross_profit = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] > 0
    )

    gross_loss = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] < 0
    )

    total_r = sum(
        trade["r_multiple"]
        for trade in trades
    )

    total_pnl = sum(
        trade["profit"]
        for trade in trades
    )

    win_rate = (
        winners / count * 100
        if count
        else 0.0
    )

    average_r = (
        total_r / count
        if count
        else 0.0
    )

    profit_factor = calculate_profit_factor(
        trades
    )

    drawdown = calculate_drawdown(
        trades
    )

    return {
        "trades": count,
        "winners": winners,
        "losers": losers,
        "win_rate": win_rate,
        "gross_profit_r": gross_profit,
        "gross_loss_r": gross_loss,
        "profit_factor": profit_factor,
        "total_r": total_r,
        "average_r": average_r,
        "pnl": total_pnl,
        "drawdown": drawdown,
    }


def print_summary(title, trades):
    summary = summarize(trades)

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    pf = summary["profit_factor"]

    if pf == float("inf"):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"Trades:               "
        f"{summary['trades']}"
    )

    print(
        f"Winners:              "
        f"{summary['winners']}"
    )

    print(
        f"Losers:               "
        f"{summary['losers']}"
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
        f"{summary['average_r']:.3f}R"
    )

    print(
        f"P&L:                  "
        f"₹{summary['pnl']:.2f}"
    )

    print(
        f"Closed-trade DD:      "
        f"{summary['drawdown']:.2f}%"
    )

    return summary


def year_analysis(trades):
    grouped = defaultdict(list)

    for trade in trades:
        year = trade["entry_date"][:4]

        if year:
            grouped[year].append(trade)

    print()
    print("=" * 100)
    print("YEAR ANALYSIS")
    print("=" * 100)

    print(
        f"{'Year':<8}"
        f"{'Trades':>8}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
        f"{'P&L':>16}"
    )

    results = []

    for year in sorted(grouped):
        summary = summarize(grouped[year])

        print(
            f"{year:<8}"
            f"{summary['trades']:>8}"
            f"{summary['winners']:>8}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>13.2f}R"
            f"{summary['pnl']:>16.2f}"
        )

        results.append(
            {
                "analysis": "YEAR",
                "group": year,
                **summary,
            }
        )

    return results


def leave_one_year_out(trades):
    years = sorted(
        {
            trade["entry_date"][:4]
            for trade in trades
            if trade["entry_date"]
        }
    )

    print()
    print("=" * 100)
    print("LEAVE-ONE-YEAR-OUT")
    print("=" * 100)

    print(
        f"{'Excluded Year':<18}"
        f"{'Trades':>10}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
        f"{'P&L':>16}"
        f"{'DD%':>10}"
    )

    results = []

    for excluded_year in years:
        remaining = [
            trade
            for trade in trades
            if trade["entry_date"][:4]
            != excluded_year
        ]

        summary = summarize(remaining)

        print(
            f"{excluded_year:<18}"
            f"{summary['trades']:>10}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>13.2f}R"
            f"{summary['pnl']:>16.2f}"
            f"{summary['drawdown']:>9.2f}%"
        )

        results.append(
            {
                "analysis": "LEAVE_ONE_YEAR_OUT",
                "group": excluded_year,
                **summary,
            }
        )

    return results


def exit_analysis(trades):
    grouped = defaultdict(list)

    for trade in trades:
        grouped[
            trade["exit_reason"]
        ].append(trade)

    print()
    print("=" * 100)
    print("EXIT REASON ANALYSIS")
    print("=" * 100)

    print(
        f"{'Exit Reason':<22}"
        f"{'Trades':>10}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
    )

    results = []

    for exit_reason in sorted(grouped):
        summary = summarize(
            grouped[exit_reason]
        )

        print(
            f"{exit_reason:<22}"
            f"{summary['trades']:>10}"
            f"{summary['winners']:>8}"
            f"{summary['win_rate']:>9.2f}%"
            f"{summary['total_r']:>13.2f}R"
        )

        results.append(
            {
                "analysis": "EXIT",
                "group": exit_reason,
                **summary,
            }
        )

    return results


def chronological_equity(trades):
    ordered = sorted(
        trades,
        key=lambda trade: (
            trade["exit_date"],
            trade["entry_date"]
        )
    )

    equity = INITIAL_CAPITAL
    peak = equity
    max_drawdown = 0.0

    rows = []

    for index, trade in enumerate(
        ordered,
        start=1
    ):
        equity += trade["profit"]

        if equity > peak:
            peak = equity

        drawdown = (
            ((peak - equity) / peak) * 100
            if peak > 0
            else 0.0
        )

        max_drawdown = max(
            max_drawdown,
            drawdown
        )

        rows.append(
            {
                "trade_number": index,
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "r_multiple": trade["r_multiple"],
                "profit": trade["profit"],
                "equity": equity,
                "drawdown": drawdown,
            }
        )

    print()
    print("=" * 100)
    print("CHRONOLOGICAL EQUITY")
    print("=" * 100)

    if not rows:
        print("No trades.")
        return rows

    print(
        f"First trade:          "
        f"{rows[0]['entry_date']}"
    )

    print(
        f"Last trade:           "
        f"{rows[-1]['entry_date']}"
    )

    print(
        f"Final Equity:         "
        f"₹{rows[-1]['equity']:.2f}"
    )

    print(
        f"Max Closed DD:        "
        f"{max_drawdown:.2f}%"
    )

    print()
    print(
        "Trade-by-trade equity:"
    )

    for row in rows:
        print(
            f"{row['trade_number']:>3}  "
            f"{row['entry_date']}  "
            f"{row['r_multiple']:>7.2f}R  "
            f"Equity ₹{row['equity']:>10.2f}  "
            f"DD {row['drawdown']:>6.2f}%"
        )

    return rows


def save_results(results):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    fieldnames = [
        "analysis",
        "group",
        "trades",
        "winners",
        "losers",
        "win_rate",
        "gross_profit_r",
        "gross_loss_r",
        "profit_factor",
        "total_r",
        "average_r",
        "pnl",
        "drawdown",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                result
            )


def save_report(
    all_summary,
    long_summary,
    lc_summary,
    year_results,
    loo_results,
    exit_results,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "TRADESENSE-AI — NIFTY 50 "
            "LONG-CONTINUATION FORENSIC\n"
        )

        file.write(
            "=" * 100 + "\n\n"
        )

        for name, summary in [
            ("ALL", all_summary),
            ("LONG ONLY", long_summary),
            (
                "LONG_CONTINUATION ONLY",
                lc_summary
            ),
        ]:
            file.write(
                f"{name}\n"
            )

            file.write(
                f"Trades: {summary['trades']}\n"
            )

            file.write(
                f"Winners: {summary['winners']}\n"
            )

            file.write(
                f"Losers: {summary['losers']}\n"
            )

            file.write(
                f"Win Rate: "
                f"{summary['win_rate']:.2f}%\n"
            )

            file.write(
                f"Total R: "
                f"{summary['total_r']:.2f}R\n"
            )

            file.write(
                f"Profit Factor: "
                f"{summary['profit_factor']:.3f}\n"
            )

            file.write(
                f"P&L: "
                f"₹{summary['pnl']:.2f}\n"
            )

            file.write(
                f"DD: "
                f"{summary['drawdown']:.2f}%\n\n"
            )

        file.write(
            "\nYEAR RESULTS\n"
        )
        file.write(
            "=" * 100 + "\n"
        )

        for row in year_results:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nLEAVE-ONE-YEAR-OUT\n"
        )
        file.write(
            "=" * 100 + "\n"
        )

        for row in loo_results:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nEXIT RESULTS\n"
        )
        file.write(
            "=" * 100 + "\n"
        )

        for row in exit_results:
            file.write(
                f"{row}\n"
            )


def main():
    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — "
        "NIFTY 50 LONG-CONTINUATION FORENSIC"
    )
    print("=" * 100)

    trades = load_trades()

    print(
        f"Input file:           "
        f"{INPUT_FILE}"
    )

    print(
        f"Total loaded trades:  "
        f"{len(trades)}"
    )

    all_trades = trades

    long_trades = [
        trade
        for trade in trades
        if trade["direction"] == "LONG"
    ]

    long_continuation_trades = [
        trade
        for trade in trades
        if (
            trade["direction"] == "LONG"
            and
            trade["setup"]
            == "LONG_CONTINUATION"
        )
    ]

    all_summary = print_summary(
        "ALL TRADES",
        all_trades
    )

    long_summary = print_summary(
        "LONG ONLY",
        long_trades
    )

    lc_summary = print_summary(
        "LONG_CONTINUATION ONLY",
        long_continuation_trades
    )

    year_results = year_analysis(
        long_continuation_trades
    )

    loo_results = leave_one_year_out(
        long_continuation_trades
    )

    exit_results = exit_analysis(
        long_continuation_trades
    )

    chronological_equity(
        long_continuation_trades
    )

    results = [
        {
            "analysis": "ALL",
            "group": "ALL",
            **all_summary,
        },
        {
            "analysis": "LONG_ONLY",
            "group": "LONG",
            **long_summary,
        },
        {
            "analysis": "LONG_CONTINUATION",
            "group": "LONG_CONTINUATION",
            **lc_summary,
        },
    ]

    results.extend(
        year_results
    )

    results.extend(
        loo_results
    )

    results.extend(
        exit_results
    )

    save_results(results)

    save_report(
        all_summary,
        long_summary,
        lc_summary,
        year_results,
        loo_results,
        exit_results,
    )

    print()
    print("=" * 100)
    print("LONG-CONTINUATION FORENSIC COMPLETE")
    print("=" * 100)

    print(
        f"Saved summary:        "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Saved report:         "
        f"{OUTPUT_REPORT}"
    )

    print()
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