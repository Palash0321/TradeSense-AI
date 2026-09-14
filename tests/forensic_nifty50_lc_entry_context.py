"""
TradeSense-AI — NIFTY 50 LONG_CONTINUATION ENTRY-CONTEXT FORENSIC

Research-only diagnostic.

Purpose:
    Examine whether NIFTY 50 LONG_CONTINUATION winners and losers
    entered under materially different market-context conditions.

Fields examined:
    - entry_trend
    - entry_trend_strength
    - entry_trend_slope
    - entry_price_position
    - entry_price_position_value
    - entry_atr
    - initial_risk
    - early_adverse_r_bar_5
    - max_favorable_r
    - max_adverse_r

Comparisons:
    - Winners vs losers
    - 2021-2023 vs 2024-2026
    - Context × outcome
    - Individual trade context

IMPORTANT:
    This script is descriptive only.

    It does NOT:
        - optimize thresholds
        - select a trading filter
        - alter production code
        - alter frozen stock data
        - alter holdout data
"""

import csv
import os
from collections import Counter, defaultdict


INPUT_FILE = (
    "tests/output/index_backtests/"
    "nifty50_index_trades.csv"
)

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

OUTPUT_CSV = (
    "tests/output/index_backtests/"
    "nifty50_lc_entry_context.csv"
)

OUTPUT_REPORT = (
    "tests/output/index_backtests/"
    "nifty50_lc_entry_context_report.txt"
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


def average(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def median(values):
    values = sorted(
        value
        for value in values
        if value is not None
    )

    if not values:
        return None

    middle = len(values) // 2

    if len(values) % 2 == 1:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def fmt(value, suffix=""):
    if value is None:
        return "N/A"

    return f"{value:.2f}{suffix}"


def pct(count, total):
    if total == 0:
        return 0.0

    return (
        count / total
    ) * 100.0


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

        rows = list(reader)

    trades = []

    for row in rows:

        if str(
            row.get("setup", "")
        ).strip() != "LONG_CONTINUATION":
            continue

        if str(
            row.get("direction", "")
        ).strip() != "LONG":
            continue

        r_multiple = safe_float(
            row.get("r_multiple")
        )

        trade = {
            "entry_date": row.get(
                "entry_date",
                "",
            ),
            "exit_date": row.get(
                "exit_date",
                "",
            ),
            "direction": row.get(
                "direction",
                "",
            ),
            "setup": row.get(
                "setup",
                "",
            ),
            "entry_trend": row.get(
                "entry_trend",
                "",
            ),
            "entry_trend_strength": row.get(
                "entry_trend_strength",
                "",
            ),
            "entry_trend_slope": row.get(
                "entry_trend_slope",
                "",
            ),
            "entry_price_position": row.get(
                "entry_price_position",
                "",
            ),
            "entry_price_position_value": safe_float(
                row.get(
                    "entry_price_position_value"
                )
            ),
            "entry_atr": safe_float(
                row.get("entry_atr")
            ),
            "initial_risk": safe_float(
                row.get("initial_risk")
            ),
            "early5": safe_float(
                row.get(
                    "early_adverse_r_bar_5"
                )
            ),
            "mfe": safe_float(
                row.get("max_favorable_r")
            ),
            "mae": safe_float(
                row.get("max_adverse_r")
            ),
            "bars": safe_int(
                row.get("bars_in_trade")
            ),
            "r_multiple": r_multiple,
            "profit": safe_float(
                row.get("profit")
            ),
            "exit_reason": row.get(
                "exit_reason",
                "",
            ),
        }

        trade["outcome"] = (
            "WIN"
            if r_multiple is not None
            and r_multiple > 0
            else "LOSS"
            if r_multiple is not None
            and r_multiple < 0
            else "FLAT"
        )

        trade["period"] = (
            "2021-2023"
            if trade["entry_date"][:4]
            in {
                "2021",
                "2022",
                "2023",
            }
            else "2024-2026"
        )

        trades.append(trade)

    return trades


def print_basic_summary(trades):
    print()
    print("=" * 100)
    print("FILTERED SET SUMMARY")
    print("=" * 100)

    winners = [
        trade
        for trade in trades
        if trade["outcome"] == "WIN"
    ]

    losers = [
        trade
        for trade in trades
        if trade["outcome"] == "LOSS"
    ]

    total_r = sum(
        trade["r_multiple"]
        for trade in trades
        if trade["r_multiple"] is not None
    )

    print(
        f"Trades:               {len(trades)}"
    )

    print(
        f"Winners:              {len(winners)}"
    )

    print(
        f"Losers:               {len(losers)}"
    )

    print(
        f"Win Rate:             "
        f"{pct(len(winners), len(trades)):.2f}%"
    )

    print(
        f"Total R:              "
        f"{total_r:.2f}R"
    )

    return winners, losers


def print_numeric_comparison(
    winners,
    losers,
):
    print()
    print("=" * 100)
    print("WINNER VS LOSER — NUMERIC ENTRY CONTEXT")
    print("=" * 100)

    print(
        f"{'Metric':<30}"
        f"{'Winners':>16}"
        f"{'Losers':>16}"
        f"{'Difference':>18}"
    )

    fields = [
        (
            "Entry ATR",
            "entry_atr",
            "",
        ),
        (
            "Initial Risk",
            "initial_risk",
            "",
        ),
        (
            "Price Position Value",
            "entry_price_position_value",
            "",
        ),
        (
            "Early Adverse Bar 5",
            "early5",
            "R",
        ),
        (
            "Maximum Favorable Excursion",
            "mfe",
            "R",
        ),
        (
            "Maximum Adverse Excursion",
            "mae",
            "R",
        ),
        (
            "Bars in Trade",
            "bars",
            "",
        ),
    ]

    results = []

    for label, field, suffix in fields:

        winner_values = [
            trade[field]
            for trade in winners
            if trade[field] is not None
        ]

        loser_values = [
            trade[field]
            for trade in losers
            if trade[field] is not None
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
                winner_avg
                - loser_avg
            )

        print(
            f"{label:<30}"
            f"{fmt(winner_avg, suffix):>16}"
            f"{fmt(loser_avg, suffix):>16}"
            f"{fmt(difference, suffix):>18}"
        )

        results.append(
            {
                "metric": label,
                "winner_avg": winner_avg,
                "loser_avg": loser_avg,
                "difference": difference,
            }
        )

    return results


def print_median_comparison(
    winners,
    losers,
):
    print()
    print("=" * 100)
    print("WINNER VS LOSER — NUMERIC ENTRY CONTEXT MEDIANS")
    print("=" * 100)

    print(
        f"{'Metric':<30}"
        f"{'Winners':>16}"
        f"{'Losers':>16}"
        f"{'Difference':>18}"
    )

    fields = [
        (
            "Entry ATR",
            "entry_atr",
            "",
        ),
        (
            "Initial Risk",
            "initial_risk",
            "",
        ),
        (
            "Price Position Value",
            "entry_price_position_value",
            "",
        ),
        (
            "Early Adverse Bar 5",
            "early5",
            "R",
        ),
        (
            "MFE",
            "mfe",
            "R",
        ),
        (
            "MAE",
            "mae",
            "R",
        ),
        (
            "Bars in Trade",
            "bars",
            "",
        ),
    ]

    results = []

    for label, field, suffix in fields:

        winner_values = [
            trade[field]
            for trade in winners
            if trade[field] is not None
        ]

        loser_values = [
            trade[field]
            for trade in losers
            if trade[field] is not None
        ]

        winner_median = median(
            winner_values
        )

        loser_median = median(
            loser_values
        )

        difference = None

        if (
            winner_median is not None
            and loser_median is not None
        ):
            difference = (
                winner_median
                - loser_median
            )

        print(
            f"{label:<30}"
            f"{fmt(winner_median, suffix):>16}"
            f"{fmt(loser_median, suffix):>16}"
            f"{fmt(difference, suffix):>18}"
        )

        results.append(
            {
                "metric": label,
                "winner_median": winner_median,
                "loser_median": loser_median,
                "difference": difference,
            }
        )

    return results


def categorical_analysis(
    trades,
    field,
    title,
):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    groups = defaultdict(list)

    for trade in trades:

        value = str(
            trade.get(field, "")
        ).strip()

        if not value:
            value = "MISSING"

        groups[value].append(
            trade
        )

    print(
        f"{'Category':<30}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Losses':>10}"
        f"{'Win%':>12}"
        f"{'Total R':>14}"
    )

    results = []

    for category in sorted(
        groups.keys()
    ):

        group = groups[category]

        wins = [
            trade
            for trade in group
            if trade["outcome"] == "WIN"
        ]

        total_r = sum(
            trade["r_multiple"]
            for trade in group
            if trade["r_multiple"] is not None
        )

        print(
            f"{category:<30}"
            f"{len(group):>10}"
            f"{len(wins):>10}"
            f"{len(group) - len(wins):>10}"
            f"{pct(len(wins), len(group)):>11.2f}%"
            f"{total_r:>13.2f}R"
        )

        results.append(
            {
                "field": field,
                "category": category,
                "trades": len(group),
                "wins": len(wins),
                "losses": (
                    len(group)
                    - len(wins)
                ),
                "win_rate": pct(
                    len(wins),
                    len(group),
                ),
                "total_r": total_r,
            }
        )

    return results


def print_period_context(
    trades,
):
    print()
    print("=" * 100)
    print(
        "ENTRY CONTEXT — 2021-2023 VS 2024-2026"
    )
    print("=" * 100)

    periods = [
        "2021-2023",
        "2024-2026",
    ]

    print(
        f"{'Period':<15}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Win%':>10}"
        f"{'ATR Avg':>14}"
        f"{'ATR Median':>16}"
        f"{'Risk Avg':>14}"
        f"{'Early5 Avg':>16}"
        f"{'MFE Avg':>14}"
        f"{'MAE Avg':>14}"
    )

    results = []

    for period in periods:

        group = [
            trade
            for trade in trades
            if trade["period"] == period
        ]

        wins = [
            trade
            for trade in group
            if trade["outcome"] == "WIN"
        ]

        atr = [
            trade["entry_atr"]
            for trade in group
            if trade["entry_atr"] is not None
        ]

        risk = [
            trade["initial_risk"]
            for trade in group
            if trade["initial_risk"] is not None
        ]

        early5 = [
            trade["early5"]
            for trade in group
            if trade["early5"] is not None
        ]

        mfe = [
            trade["mfe"]
            for trade in group
            if trade["mfe"] is not None
        ]

        mae = [
            trade["mae"]
            for trade in group
            if trade["mae"] is not None
        ]

        print(
            f"{period:<15}"
            f"{len(group):>10}"
            f"{len(wins):>10}"
            f"{pct(len(wins), len(group)):>9.2f}%"
            f"{fmt(average(atr)):>14}"
            f"{fmt(median(atr)):>16}"
            f"{fmt(average(risk)):>14}"
            f"{fmt(average(early5), 'R'):>16}"
            f"{fmt(average(mfe), 'R'):>14}"
            f"{fmt(average(mae), 'R'):>14}"
        )

        results.append(
            {
                "period": period,
                "trades": len(group),
                "wins": len(wins),
                "win_rate": pct(
                    len(wins),
                    len(group),
                ),
                "avg_atr": average(atr),
                "median_atr": median(atr),
                "avg_risk": average(risk),
                "avg_early5": average(early5),
                "avg_mfe": average(mfe),
                "avg_mae": average(mae),
            }
        )

    return results


def print_cross_context(
    trades,
):
    print()
    print("=" * 100)
    print(
        "CONTEXT × OUTCOME — COMBINED ENTRY STATE"
    )
    print("=" * 100)

    print(
        "The following tables are descriptive cross-tabs."
    )

    print(
        "No category or combination is selected as a filter."
    )

    trend_strength_results = categorical_analysis(
        trades,
        "entry_trend_strength",
        "TREND STRENGTH × OUTCOME",
    )

    trend_results = categorical_analysis(
        trades,
        "entry_trend",
        "ENTRY TREND × OUTCOME",
    )

    slope_results = categorical_analysis(
        trades,
        "entry_trend_slope",
        "TREND SLOPE × OUTCOME",
    )

    price_position_results = categorical_analysis(
        trades,
        "entry_price_position",
        "ENTRY PRICE POSITION × OUTCOME",
    )

    return (
        trend_strength_results,
        trend_results,
        slope_results,
        price_position_results,
    )


def print_individual_context(
    trades,
):
    print()
    print("=" * 100)
    print("INDIVIDUAL LONG_CONTINUATION ENTRY CONTEXT")
    print("=" * 100)

    print(
        f"{'#':<4}"
        f"{'Entry':<13}"
        f"{'Outcome':<8}"
        f"{'R':>8}"
        f"{'Trend':<16}"
        f"{'Strength':<14}"
        f"{'Slope':<14}"
        f"{'Position':<18}"
        f"{'PosVal':>10}"
        f"{'ATR':>10}"
        f"{'Early5':>10}"
        f"{'MFE':>10}"
        f"{'MAE':>10}"
    )

    ordered = sorted(
        trades,
        key=lambda trade: (
            str(
                trade.get(
                    "entry_date",
                    "",
                )
            )
        ),
    )

    results = []

    for number, trade in enumerate(
        ordered,
        start=1,
    ):

        print(
            f"{number:<4}"
            f"{str(trade['entry_date'])[:12]:<13}"
            f"{trade['outcome']:<8}"
            f"{fmt(trade['r_multiple'], 'R'):>8}"
            f"{str(trade['entry_trend'])[:15]:<16}"
            f"{str(trade['entry_trend_strength'])[:13]:<14}"
            f"{str(trade['entry_trend_slope'])[:13]:<14}"
            f"{str(trade['entry_price_position'])[:17]:<18}"
            f"{fmt(trade['entry_price_position_value']):>10}"
            f"{fmt(trade['entry_atr']):>10}"
            f"{fmt(trade['early5'], 'R'):>10}"
            f"{fmt(trade['mfe'], 'R'):>10}"
            f"{fmt(trade['mae'], 'R'):>10}"
        )

        results.append(trade)

    return results


def save_csv(
    numeric_rows,
    median_rows,
    period_rows,
    categorical_rows,
    trade_rows,
):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
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
                "field",
                "category",
                "period",
                "entry_date",
                "exit_date",
                "outcome",
                "r_multiple",
                "winner_avg",
                "loser_avg",
                "difference",
                "winner_median",
                "loser_median",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "total_r",
                "avg_atr",
                "median_atr",
                "avg_risk",
                "avg_early5",
                "avg_mfe",
                "avg_mae",
                "entry_trend",
                "entry_trend_strength",
                "entry_trend_slope",
                "entry_price_position",
                "entry_price_position_value",
                "entry_atr",
                "initial_risk",
                "early5",
                "mfe",
                "mae",
            ]
        )

        for row in numeric_rows:

            writer.writerow(
                [
                    "NUMERIC_AVG",
                    row["metric"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    row["winner_avg"],
                    row["loser_avg"],
                    row["difference"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for row in median_rows:

            writer.writerow(
                [
                    "NUMERIC_MEDIAN",
                    row["metric"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    row["winner_median"],
                    row["loser_median"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for row in period_rows:

            writer.writerow(
                [
                    "PERIOD",
                    "",
                    "",
                    row["period"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    row["trades"],
                    row["wins"],
                    (
                        row["trades"]
                        - row["wins"]
                    ),
                    row["win_rate"],
                    "",
                    row["avg_atr"],
                    row["median_atr"],
                    row["avg_risk"],
                    row["avg_early5"],
                    row["avg_mfe"],
                    row["avg_mae"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for row in categorical_rows:

            writer.writerow(
                [
                    "CATEGORY",
                    row["field"],
                    row["category"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    row["trades"],
                    row["wins"],
                    row["losses"],
                    row["win_rate"],
                    row["total_r"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for trade in trade_rows:

            writer.writerow(
                [
                    "TRADE",
                    "",
                    "",
                    trade["period"],
                    trade["entry_date"],
                    trade["exit_date"],
                    trade["outcome"],
                    trade["r_multiple"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    trade["entry_trend"],
                    trade["entry_trend_strength"],
                    trade["entry_trend_slope"],
                    trade["entry_price_position"],
                    trade["entry_price_position_value"],
                    trade["entry_atr"],
                    trade["initial_risk"],
                    trade["early5"],
                    trade["mfe"],
                    trade["mae"],
                ]
            )


def save_report(
    trades,
    winners,
    losers,
    numeric_rows,
    median_rows,
    period_rows,
    categorical_rows,
):
    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TRADESENSE-AI — NIFTY 50 "
            "LONG_CONTINUATION ENTRY-CONTEXT FORENSIC\n"
        )

        file.write(
            "=" * 100 + "\n\n"
        )

        file.write(
            f"LONG_CONTINUATION trades: {len(trades)}\n"
        )

        file.write(
            f"Winners: {len(winners)}\n"
        )

        file.write(
            f"Losers: {len(losers)}\n\n"
        )

        file.write(
            "This is a descriptive forensic analysis.\n"
        )

        file.write(
            "No threshold was optimized.\n"
        )

        file.write(
            "No trading filter was selected.\n\n"
        )

        file.write(
            "NUMERIC WINNER VS LOSER ANALYSIS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in numeric_rows:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nMEDIAN ANALYSIS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in median_rows:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nPERIOD ANALYSIS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in period_rows:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nCATEGORICAL ANALYSIS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in categorical_rows:
            file.write(
                f"{row}\n"
            )


def main():

    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — "
        "NIFTY 50 LONG_CONTINUATION "
        "ENTRY-CONTEXT FORENSIC"
    )
    print("=" * 100)

    trades = load_trades()

    print(
        f"Input file:           "
        f"{INPUT_FILE}"
    )

    print(
        f"LONG_CONTINUATION:    "
        f"{len(trades)}"
    )

    if not trades:
        print(
            "No LONG_CONTINUATION trades found."
        )
        return

    winners, losers = (
        print_basic_summary(
            trades
        )
    )

    numeric_rows = (
        print_numeric_comparison(
            winners,
            losers,
        )
    )

    median_rows = (
        print_median_comparison(
            winners,
            losers,
        )
    )

    period_rows = (
        print_period_context(
            trades
        )
    )

    (
        trend_strength_rows,
        trend_rows,
        slope_rows,
        price_position_rows,
    ) = print_cross_context(
        trades
    )

    categorical_rows = (
        trend_strength_rows
        + trend_rows
        + slope_rows
        + price_position_rows
    )

    trade_rows = (
        print_individual_context(
            trades
        )
    )

    save_csv(
        numeric_rows,
        median_rows,
        period_rows,
        categorical_rows,
        trade_rows,
    )

    save_report(
        trades,
        winners,
        losers,
        numeric_rows,
        median_rows,
        period_rows,
        categorical_rows,
    )

    print()
    print("=" * 100)
    print(
        "ENTRY-CONTEXT FORENSIC COMPLETE"
    )
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
        "IMPORTANT:"
    )

    print(
        "No trading filter was selected."
    )

    print(
        "No threshold was optimized."
    )


if __name__ == "__main__":
    main()