"""
TradeSense-AI — NIFTY 50 LONG_CONTINUATION TRADE-DEVELOPMENT FORENSIC

Research-only diagnostic.

Purpose:
    Study the internal behaviour of NIFTY 50 LONG_CONTINUATION trades
    without changing production code or any frozen dataset.

Questions:
    1. How do winners differ from losers?
    2. How does early adverse movement behave?
    3. How does MFE behave?
    4. How does MAE behave?
    5. How long do winners/losers hold?
    6. What are the entry ATR / risk characteristics?
    7. Did the 2021-2023 winners and 2024-2026 failures behave differently?

IMPORTANT:
    This script is descriptive.
    It does NOT optimize or select a trading rule.

Input:
    tests/output/index_backtests/nifty50_index_trades.csv

Output:
    tests/output/index_backtests/nifty50_lc_trade_development.csv
    tests/output/index_backtests/nifty50_lc_trade_development_report.txt
"""

import csv
import os
from collections import defaultdict


INPUT_FILE = (
    "tests/output/index_backtests/"
    "nifty50_index_trades.csv"
)

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

OUTPUT_CSV = (
    "tests/output/index_backtests/"
    "nifty50_lc_trade_development.csv"
)

OUTPUT_REPORT = (
    "tests/output/index_backtests/"
    "nifty50_lc_trade_development_report.txt"
)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_trades():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames or []

        print()
        print("=" * 100)
        print("INPUT FIELD AUDIT")
        print("=" * 100)

        print(
            f"Fields available:     "
            f"{len(fieldnames)}"
        )

        print()

        for field in fieldnames:
            print(field)

        rows = list(reader)

    trades = []

    for row in rows:
        setup = str(
            row.get("setup", "")
        ).strip()

        direction = str(
            row.get("direction", "")
        ).strip()

        if (
            setup != "LONG_CONTINUATION"
            or direction != "LONG"
        ):
            continue

        trade = dict(row)

        trade["_r"] = safe_float(
            row.get("r_multiple")
        )

        trade["_profit"] = safe_float(
            row.get("profit")
        )

        trade["_bars"] = safe_int(
            row.get("bars_in_trade")
        )

        trade["_entry_atr"] = safe_float(
            row.get("entry_atr")
        )

        trade["_initial_risk"] = safe_float(
            row.get("initial_risk")
        )

        trade["_mfe_r"] = safe_float(
            row.get("max_favorable_r")
        )

        trade["_mae_r"] = safe_float(
            row.get("max_adverse_r")
        )

        for bar in range(1, 6):
            trade[
                f"_early_adverse_r_bar_{bar}"
            ] = safe_float(
                row.get(
                    f"early_adverse_r_bar_{bar}"
                )
            )

        trades.append(trade)

    return trades, fieldnames


def average(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def minimum(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return min(values)


def maximum(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return max(values)


def median(values):
    values = sorted(
        value
        for value in values
        if value is not None
    )

    if not values:
        return None

    middle = len(values) // 2

    if len(values) % 2:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def fmt(value, suffix=""):
    if value is None:
        return "N/A"

    return f"{value:.2f}{suffix}"


def print_distribution(
    title,
    trades,
    field,
    suffix="",
):
    values = [
        trade.get(field)
        for trade in trades
        if trade.get(field) is not None
    ]

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"Available:            "
        f"{len(values)}/{len(trades)}"
    )

    print(
        f"Minimum:              "
        f"{fmt(minimum(values), suffix)}"
    )

    print(
        f"Median:               "
        f"{fmt(median(values), suffix)}"
    )

    print(
        f"Average:              "
        f"{fmt(average(values), suffix)}"
    )

    print(
        f"Maximum:              "
        f"{fmt(maximum(values), suffix)}"
    )


def print_group_comparison(
    title,
    winners,
    losers,
    field,
    suffix="",
):
    winner_values = [
        trade.get(field)
        for trade in winners
        if trade.get(field) is not None
    ]

    loser_values = [
        trade.get(field)
        for trade in losers
        if trade.get(field) is not None
    ]

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'Metric':<22}"
        f"{'Winners':>16}"
        f"{'Losers':>16}"
        f"{'Difference':>18}"
    )

    winner_avg = average(
        winner_values
    )

    loser_avg = average(
        loser_values
    )

    difference = None

    if (
        winner_avg is not None
        and loser_avg is not None
    ):
        difference = (
            winner_avg - loser_avg
        )

    print(
        f"{'Count':<22}"
        f"{len(winner_values):>16}"
        f"{len(loser_values):>16}"
        f"{'':>18}"
    )

    print(
        f"{'Average':<22}"
        f"{fmt(winner_avg, suffix):>16}"
        f"{fmt(loser_avg, suffix):>16}"
        f"{fmt(difference, suffix):>18}"
    )

    winner_median = median(
        winner_values
    )

    loser_median = median(
        loser_values
    )

    print(
        f"{'Median':<22}"
        f"{fmt(winner_median, suffix):>16}"
        f"{fmt(loser_median, suffix):>16}"
        f"{'':>18}"
    )

    print(
        f"{'Minimum':<22}"
        f"{fmt(minimum(winner_values), suffix):>16}"
        f"{fmt(minimum(loser_values), suffix):>16}"
        f"{'':>18}"
    )

    print(
        f"{'Maximum':<22}"
        f"{fmt(maximum(winner_values), suffix):>16}"
        f"{fmt(maximum(loser_values), suffix):>16}"
        f"{'':>18}"
    )


def print_basic_summary(trades):
    winners = [
        trade
        for trade in trades
        if (
            trade.get("_r") is not None
            and trade["_r"] > 0
        )
    ]

    losers = [
        trade
        for trade in trades
        if (
            trade.get("_r") is not None
            and trade["_r"] < 0
        )
    ]

    total_r = sum(
        trade["_r"]
        for trade in trades
        if trade.get("_r") is not None
    )

    total_pnl = sum(
        trade["_profit"]
        for trade in trades
        if trade.get("_profit") is not None
    )

    print()
    print("=" * 100)
    print("LONG_CONTINUATION BASIC SUMMARY")
    print("=" * 100)

    print(
        f"Trades:               "
        f"{len(trades)}"
    )

    print(
        f"Winners:              "
        f"{len(winners)}"
    )

    print(
        f"Losers:               "
        f"{len(losers)}"
    )

    print(
        f"Win Rate:             "
        f"{(
            len(winners) / len(trades) * 100
            if trades else 0.0
        ):.2f}%"
    )

    print(
        f"Total R:              "
        f"{total_r:.2f}R"
    )

    print(
        f"P&L:                  "
        f"₹{total_pnl:.2f}"
    )

    return winners, losers


def early_adverse_table(trades):
    print()
    print("=" * 100)
    print("EARLY ADVERSE DEVELOPMENT")
    print("=" * 100)

    print(
        f"{'Bar':<10}"
        f"{'Available':>12}"
        f"{'Average R':>16}"
        f"{'Median R':>16}"
        f"{'Min R':>16}"
        f"{'Max R':>16}"
    )

    results = []

    for bar in range(1, 6):
        field = (
            f"_early_adverse_r_bar_{bar}"
        )

        values = [
            trade.get(field)
            for trade in trades
            if trade.get(field) is not None
        ]

        row = {
            "bar": bar,
            "available": len(values),
            "average": average(values),
            "median": median(values),
            "minimum": minimum(values),
            "maximum": maximum(values),
        }

        results.append(row)

        print(
            f"{bar:<10}"
            f"{len(values):>12}"
            f"{fmt(average(values), 'R'):>16}"
            f"{fmt(median(values), 'R'):>16}"
            f"{fmt(minimum(values), 'R'):>16}"
            f"{fmt(maximum(values), 'R'):>16}"
        )

    return results


def winner_loser_early_comparison(
    winners,
    losers,
):
    print()
    print("=" * 100)
    print("WINNER VS LOSER — EARLY ADVERSE")
    print("=" * 100)

    print(
        f"{'Bar':<10}"
        f"{'Winner Avg':>16}"
        f"{'Loser Avg':>16}"
        f"{'Difference':>16}"
    )

    results = []

    for bar in range(1, 6):
        field = (
            f"_early_adverse_r_bar_{bar}"
        )

        winner_values = [
            trade.get(field)
            for trade in winners
            if trade.get(field) is not None
        ]

        loser_values = [
            trade.get(field)
            for trade in losers
            if trade.get(field) is not None
        ]

        winner_avg = average(
            winner_values
        )

        loser_avg = average(
            loser_values
        )

        difference = None

        if (
            winner_avg is not None
            and loser_avg is not None
        ):
            difference = (
                loser_avg - winner_avg
            )

        print(
            f"{bar:<10}"
            f"{fmt(winner_avg, 'R'):>16}"
            f"{fmt(loser_avg, 'R'):>16}"
            f"{fmt(difference, 'R'):>16}"
        )

        results.append(
            {
                "bar": bar,
                "winner_avg": winner_avg,
                "loser_avg": loser_avg,
                "difference_loser_minus_winner": (
                    difference
                ),
            }
        )

    return results


def year_regime_comparison(trades):
    grouped = defaultdict(list)

    for trade in trades:
        year = str(
            trade.get("entry_date", "")
        )[:4]

        if year:
            grouped[year].append(trade)

    print()
    print("=" * 100)
    print("YEAR / REGIME COMPARISON")
    print("=" * 100)

    print(
        f"{'Year':<10}"
        f"{'Trades':>10}"
        f"{'Wins':>8}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
        f"{'Avg Early5':>16}"
        f"{'Avg MFE':>14}"
        f"{'Avg MAE':>14}"
    )

    results = []

    for year in sorted(grouped):
        year_trades = grouped[year]

        winners = [
            trade
            for trade in year_trades
            if (
                trade.get("_r") is not None
                and trade["_r"] > 0
            )
        ]

        total_r = sum(
            trade["_r"]
            for trade in year_trades
            if trade.get("_r") is not None
        )

        early5 = [
            trade.get(
                "_early_adverse_r_bar_5"
            )
            for trade in year_trades
            if trade.get(
                "_early_adverse_r_bar_5"
            ) is not None
        ]

        mfe = [
            trade.get("_mfe_r")
            for trade in year_trades
            if trade.get("_mfe_r") is not None
        ]

        mae = [
            trade.get("_mae_r")
            for trade in year_trades
            if trade.get("_mae_r") is not None
        ]

        win_rate = (
            len(winners)
            / len(year_trades)
            * 100
            if year_trades
            else 0.0
        )

        print(
            f"{year:<10}"
            f"{len(year_trades):>10}"
            f"{len(winners):>8}"
            f"{win_rate:>9.2f}%"
            f"{total_r:>13.2f}R"
            f"{fmt(average(early5), 'R'):>16}"
            f"{fmt(average(mfe), 'R'):>14}"
            f"{fmt(average(mae), 'R'):>14}"
        )

        results.append(
            {
                "year": year,
                "trades": len(year_trades),
                "wins": len(winners),
                "win_rate": win_rate,
                "total_r": total_r,
                "avg_early5": average(early5),
                "avg_mfe": average(mfe),
                "avg_mae": average(mae),
            }
        )

    return results


def individual_trade_table(trades):
    print()
    print("=" * 100)
    print("INDIVIDUAL LONG_CONTINUATION TRADES")
    print("=" * 100)

    print(
        f"{'#':<4}"
        f"{'Entry':<13}"
        f"{'Exit':<13}"
        f"{'Outcome':<10}"
        f"{'R':>9}"
        f"{'Bars':>8}"
        f"{'ATR':>12}"
        f"{'Early5':>12}"
        f"{'MFE':>10}"
        f"{'MAE':>10}"
    )

    ordered = sorted(
        trades,
        key=lambda trade: (
            str(
                trade.get(
                    "entry_date",
                    ""
                )
            )
        )
    )

    results = []

    for index, trade in enumerate(
        ordered,
        start=1
    ):
        r = trade.get("_r")

        outcome = (
            "WIN"
            if r is not None and r > 0
            else "LOSS"
            if r is not None and r < 0
            else "FLAT"
        )

        print(
            f"{index:<4}"
            f"{str(trade.get('entry_date', ''))[:12]:<13}"
            f"{str(trade.get('exit_date', ''))[:12]:<13}"
            f"{outcome:<10}"
            f"{fmt(r, 'R'):>9}"
            f"{str(trade.get('_bars') or 'N/A'):>8}"
            f"{fmt(trade.get('_entry_atr')):>12}"
            f"{fmt(trade.get('_early_adverse_r_bar_5'), 'R'):>12}"
            f"{fmt(trade.get('_mfe_r'), 'R'):>10}"
            f"{fmt(trade.get('_mae_r'), 'R'):>10}"
        )

        results.append(
            {
                "entry_date": trade.get(
                    "entry_date",
                    ""
                ),
                "exit_date": trade.get(
                    "exit_date",
                    ""
                ),
                "r_multiple": r,
                "bars": trade.get("_bars"),
                "entry_atr": trade.get(
                    "_entry_atr"
                ),
                "early_adverse_r_bar_5": trade.get(
                    "_early_adverse_r_bar_5"
                ),
                "mfe_r": trade.get("_mfe_r"),
                "mae_r": trade.get("_mae_r"),
            }
        )

    return results


def save_csv(
    trade_rows,
    year_rows,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "row_type",
                "entry_date",
                "exit_date",
                "year",
                "outcome",
                "r_multiple",
                "bars",
                "entry_atr",
                "early_adverse_r_bar_5",
                "mfe_r",
                "mae_r",
                "year_trades",
                "year_wins",
                "year_win_rate",
                "year_total_r",
                "year_avg_early5",
                "year_avg_mfe",
                "year_avg_mae",
            ]
        )

        for row in trade_rows:
            r = row["r_multiple"]

            outcome = (
                "WIN"
                if r is not None and r > 0
                else "LOSS"
                if r is not None and r < 0
                else "FLAT"
            )

            writer.writerow(
                [
                    "TRADE",
                    row["entry_date"],
                    row["exit_date"],
                    row["entry_date"][:4],
                    outcome,
                    row["r_multiple"],
                    row["bars"],
                    row["entry_atr"],
                    row[
                        "early_adverse_r_bar_5"
                    ],
                    row["mfe_r"],
                    row["mae_r"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for row in year_rows:
            writer.writerow(
                [
                    "YEAR",
                    "",
                    "",
                    row["year"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    row["trades"],
                    row["wins"],
                    row["win_rate"],
                    row["total_r"],
                    row["avg_early5"],
                    row["avg_mfe"],
                    row["avg_mae"],
                ]
            )


def save_report(
    trades,
    winners,
    losers,
    year_rows,
):
    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TRADESENSE-AI — NIFTY 50 "
            "LONG_CONTINUATION TRADE-DEVELOPMENT FORENSIC\n"
        )

        file.write("=" * 100 + "\n\n")

        file.write(
            f"Trades: {len(trades)}\n"
        )

        file.write(
            f"Winners: {len(winners)}\n"
        )

        file.write(
            f"Losers: {len(losers)}\n\n"
        )

        for title, group in [
            ("WINNERS", winners),
            ("LOSERS", losers),
        ]:
            file.write(
                f"{title}\n"
            )

            file.write(
                f"Count: {len(group)}\n"
            )

            for field, label in [
                (
                    "_entry_atr",
                    "Entry ATR"
                ),
                (
                    "_initial_risk",
                    "Initial Risk"
                ),
                (
                    "_bars",
                    "Bars"
                ),
                (
                    "_early_adverse_r_bar_5",
                    "Early Adverse Bar 5"
                ),
                (
                    "_mfe_r",
                    "MFE"
                ),
                (
                    "_mae_r",
                    "MAE"
                ),
                (
                    "_r",
                    "R Multiple"
                ),
            ]:
                values = [
                    trade.get(field)
                    for trade in group
                    if trade.get(field)
                    is not None
                ]

                file.write(
                    f"{label}: "
                    f"avg={fmt(average(values))}, "
                    f"median={fmt(median(values))}, "
                    f"min={fmt(minimum(values))}, "
                    f"max={fmt(maximum(values))}\n"
                )

            file.write("\n")

        file.write(
            "YEAR RESULTS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in year_rows:
            file.write(
                f"{row}\n"
            )


def main():
    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — "
        "NIFTY 50 LONG_CONTINUATION "
        "TRADE-DEVELOPMENT FORENSIC"
    )
    print("=" * 100)

    trades, fieldnames = load_trades()

    print()
    print("=" * 100)
    print("FILTERED TRADE SET")
    print("=" * 100)

    print(
        f"LONG_CONTINUATION trades: "
        f"{len(trades)}"
    )

    if not trades:
        print(
            "No LONG_CONTINUATION trades found."
        )
        return

    winners, losers = (
        print_basic_summary(trades)
    )

    print_group_comparison(
        "ENTRY ATR — WINNER VS LOSER",
        winners,
        losers,
        "_entry_atr",
    )

    print_group_comparison(
        "INITIAL RISK — WINNER VS LOSER",
        winners,
        losers,
        "_initial_risk",
    )

    print_group_comparison(
        "HOLDING PERIOD — WINNER VS LOSER",
        winners,
        losers,
        "_bars",
    )

    print_group_comparison(
        "MFE — WINNER VS LOSER",
        winners,
        losers,
        "_mfe_r",
        "R",
    )

    print_group_comparison(
        "MAE — WINNER VS LOSER",
        winners,
        losers,
        "_mae_r",
        "R",
    )

    early_rows = (
        early_adverse_table(trades)
    )

    early_comparison = (
        winner_loser_early_comparison(
            winners,
            losers,
        )
    )

    year_rows = (
        year_regime_comparison(
            trades
        )
    )

    trade_rows = (
        individual_trade_table(
            trades
        )
    )

    save_csv(
        trade_rows,
        year_rows,
    )

    save_report(
        trades,
        winners,
        losers,
        year_rows,
    )

    print()
    print("=" * 100)
    print("FORENSIC COMPLETE")
    print("=" * 100)

    print(
        f"Saved CSV:            "
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

    print()
    print(
        "IMPORTANT: This analysis is "
        "descriptive only."
    )

    print(
        "No trading filter was selected "
        "or optimized."
    )


if __name__ == "__main__":
    main()