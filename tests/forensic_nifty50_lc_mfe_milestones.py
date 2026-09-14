"""
TradeSense-AI — NIFTY 50 LONG_CONTINUATION MFE MILESTONE FORENSIC

Research-only diagnostic.

Purpose:
    Determine whether NIFTY 50 LONG_CONTINUATION losing trades
    actually developed favorably before eventually losing.

Milestones:
    0.5R
    1.0R
    1.5R

Also examines:
    - bars_to_0_5r
    - bars_to_1r
    - bars_to_1_5r
    - favorable_first
    - immediate_failure
    - winner vs loser behaviour
    - 2021-2023 vs 2024-2026 regime behaviour
    - individual trade progression

IMPORTANT:
    This script is descriptive only.
    It does NOT optimize thresholds.
    It does NOT change production code.
    It does NOT modify frozen or holdout datasets.

Input:
    tests/output/index_backtests/nifty50_index_trades.csv

Output:
    tests/output/index_backtests/nifty50_lc_mfe_milestones.csv
    tests/output/index_backtests/nifty50_lc_mfe_milestones_report.txt
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
    "nifty50_lc_mfe_milestones.csv"
)

OUTPUT_REPORT = (
    "tests/output/index_backtests/"
    "nifty50_lc_mfe_milestones_report.txt"
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


def safe_bool(value):
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

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

        trade["_mfe"] = safe_float(
            row.get("max_favorable_r")
        )

        trade["_mae"] = safe_float(
            row.get("max_adverse_r")
        )

        trade["_bars"] = safe_int(
            row.get("bars_in_trade")
        )

        trade["_bars_0_5"] = safe_int(
            row.get("bars_to_0_5r")
        )

        trade["_bars_1"] = safe_int(
            row.get("bars_to_1r")
        )

        trade["_bars_1_5"] = safe_int(
            row.get("bars_to_1_5r")
        )

        trade["_reached_0_5"] = safe_bool(
            row.get("reached_0_5r")
        )

        trade["_reached_1"] = safe_bool(
            row.get("reached_1r")
        )

        trade["_reached_1_5"] = safe_bool(
            row.get("reached_1_5r")
        )

        trade["_favorable_first"] = safe_bool(
            row.get("favorable_first")
        )

        trade["_immediate_failure"] = safe_bool(
            row.get("immediate_failure")
        )

        trade["_early5"] = safe_float(
            row.get("early_adverse_r_bar_5")
        )

        trades.append(trade)

    return trades


def percentage(count, total):
    if total == 0:
        return 0.0

    return (
        count / total
    ) * 100.0


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


def yes_count(
    trades,
    field,
):
    return sum(
        1
        for trade in trades
        if trade.get(field) is True
    )


def print_milestone_table(
    trades,
):
    print()
    print("=" * 100)
    print("MFE MILESTONE REACH ANALYSIS")
    print("=" * 100)

    print(
        f"{'Milestone':<15}"
        f"{'Reached':>12}"
        f"{'Not Reached':>15}"
        f"{'Reach %':>12}"
        f"{'Avg Bars':>15}"
        f"{'Median Bars':>16}"
    )

    results = []

    milestones = [
        (
            "0.5R",
            "_reached_0_5",
            "_bars_0_5",
        ),
        (
            "1.0R",
            "_reached_1",
            "_bars_1",
        ),
        (
            "1.5R",
            "_reached_1_5",
            "_bars_1_5",
        ),
    ]

    for name, reached_field, bars_field in milestones:

        reached = [
            trade
            for trade in trades
            if trade.get(reached_field) is True
        ]

        not_reached = [
            trade
            for trade in trades
            if trade.get(reached_field) is False
        ]

        bars = [
            trade.get(bars_field)
            for trade in reached
            if trade.get(bars_field) is not None
        ]

        reach_pct = percentage(
            len(reached),
            len(trades),
        )

        print(
            f"{name:<15}"
            f"{len(reached):>12}"
            f"{len(not_reached):>15}"
            f"{reach_pct:>11.2f}%"
            f"{fmt(average(bars)):>15}"
            f"{fmt(median(bars)):>16}"
        )

        results.append(
            {
                "milestone": name,
                "reached": len(reached),
                "not_reached": len(not_reached),
                "reach_pct": reach_pct,
                "avg_bars": average(bars),
                "median_bars": median(bars),
            }
        )

    return results


def print_winner_loser_milestones(
    winners,
    losers,
):
    print()
    print("=" * 100)
    print("WINNER VS LOSER — MFE MILESTONES")
    print("=" * 100)

    print(
        f"{'Metric':<22}"
        f"{'Winners':>16}"
        f"{'Losers':>16}"
        f"{'Winner % / Avg':>20}"
        f"{'Loser % / Avg':>20}"
    )

    milestone_fields = [
        (
            "0.5R reached",
            "_reached_0_5",
            None,
        ),
        (
            "1.0R reached",
            "_reached_1",
            None,
        ),
        (
            "1.5R reached",
            "_reached_1_5",
            None,
        ),
        (
            "Bars to 0.5R",
            "_reached_0_5",
            "_bars_0_5",
        ),
        (
            "Bars to 1.0R",
            "_reached_1",
            "_bars_1",
        ),
        (
            "Bars to 1.5R",
            "_reached_1_5",
            "_bars_1_5",
        ),
    ]

    results = []

    for label, reached_field, bars_field in milestone_fields:

        winner_reached = yes_count(
            winners,
            reached_field,
        )

        loser_reached = yes_count(
            losers,
            reached_field,
        )

        winner_pct = percentage(
            winner_reached,
            len(winners),
        )

        loser_pct = percentage(
            loser_reached,
            len(losers),
        )

        if bars_field is None:

            print(
                f"{label:<22}"
                f"{winner_reached:>16}"
                f"{loser_reached:>16}"
                f"{winner_pct:>19.2f}%"
                f"{loser_pct:>19.2f}%"
            )

            results.append(
                {
                    "metric": label,
                    "winner_count": winner_reached,
                    "loser_count": loser_reached,
                    "winner_value": winner_pct,
                    "loser_value": loser_pct,
                }
            )

        else:

            winner_bars = [
                trade.get(bars_field)
                for trade in winners
                if trade.get(
                    reached_field
                ) is True
                and trade.get(
                    bars_field
                ) is not None
            ]

            loser_bars = [
                trade.get(bars_field)
                for trade in losers
                if trade.get(
                    reached_field
                ) is True
                and trade.get(
                    bars_field
                ) is not None
            ]

            winner_avg = average(
                winner_bars
            )

            loser_avg = average(
                loser_bars
            )

            print(
                f"{label:<22}"
                f"{len(winner_bars):>16}"
                f"{len(loser_bars):>16}"
                f"{fmt(winner_avg):>20}"
                f"{fmt(loser_avg):>20}"
            )

            results.append(
                {
                    "metric": label,
                    "winner_count": len(
                        winner_bars
                    ),
                    "loser_count": len(
                        loser_bars
                    ),
                    "winner_value": winner_avg,
                    "loser_value": loser_avg,
                }
            )

    return results


def print_behaviour_comparison(
    winners,
    losers,
):
    print()
    print("=" * 100)
    print("WINNER VS LOSER — TRADE BEHAVIOUR")
    print("=" * 100)

    print(
        f"{'Metric':<25}"
        f"{'Winners':>16}"
        f"{'Losers':>16}"
        f"{'Difference':>18}"
    )

    metrics = [
        (
            "MFE",
            "_mfe",
        ),
        (
            "MAE",
            "_mae",
        ),
        (
            "Early Adverse Bar 5",
            "_early5",
        ),
        (
            "Bars in Trade",
            "_bars",
        ),
    ]

    results = []

    for label, field in metrics:

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
                winner_avg - loser_avg
            )

        print(
            f"{label:<25}"
            f"{fmt(winner_avg, 'R'):>16}"
            f"{fmt(loser_avg, 'R'):>16}"
            f"{fmt(difference, 'R'):>18}"
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


def print_behaviour_flags(
    trades,
):
    print()
    print("=" * 100)
    print("BEHAVIOUR FLAGS")
    print("=" * 100)

    print(
        f"{'Flag':<25}"
        f"{'Count':>12}"
        f"{'Percentage':>15}"
    )

    fields = [
        (
            "Favorable First",
            "_favorable_first",
        ),
        (
            "Immediate Failure",
            "_immediate_failure",
        ),
    ]

    results = []

    for label, field in fields:

        count = yes_count(
            trades,
            field,
        )

        pct = percentage(
            count,
            len(trades),
        )

        print(
            f"{label:<25}"
            f"{count:>12}"
            f"{pct:>14.2f}%"
        )

        results.append(
            {
                "flag": label,
                "count": count,
                "percentage": pct,
            }
        )

    return results


def print_regime_comparison(
    trades,
):
    print()
    print("=" * 100)
    print("REGIME COMPARISON — 2021-2023 VS 2024-2026")
    print("=" * 100)

    groups = {
        "2021-2023": [
            trade
            for trade in trades
            if "2021"
            <= trade.get(
                "entry_date",
                "",
            )[:4]
            <= "2023"
        ],
        "2024-2026": [
            trade
            for trade in trades
            if "2024"
            <= trade.get(
                "entry_date",
                "",
            )[:4]
            <= "2026"
        ],
    }

    print(
        f"{'Period':<15}"
        f"{'Trades':>10}"
        f"{'Wins':>10}"
        f"{'Win%':>10}"
        f"{'Total R':>14}"
        f"{'MFE Avg':>14}"
        f"{'MAE Avg':>14}"
        f"{'Early5 Avg':>16}"
    )

    results = []

    for period, group in groups.items():

        winners = [
            trade
            for trade in group
            if (
                trade.get("_r") is not None
                and trade["_r"] > 0
            )
        ]

        total_r = sum(
            trade["_r"]
            for trade in group
            if trade.get("_r") is not None
        )

        mfe = [
            trade.get("_mfe")
            for trade in group
            if trade.get("_mfe") is not None
        ]

        mae = [
            trade.get("_mae")
            for trade in group
            if trade.get("_mae") is not None
        ]

        early5 = [
            trade.get("_early5")
            for trade in group
            if trade.get("_early5") is not None
        ]

        win_pct = percentage(
            len(winners),
            len(group),
        )

        print(
            f"{period:<15}"
            f"{len(group):>10}"
            f"{len(winners):>10}"
            f"{win_pct:>9.2f}%"
            f"{total_r:>13.2f}R"
            f"{fmt(average(mfe), 'R'):>14}"
            f"{fmt(average(mae), 'R'):>14}"
            f"{fmt(average(early5), 'R'):>16}"
        )

        results.append(
            {
                "period": period,
                "trades": len(group),
                "wins": len(winners),
                "win_rate": win_pct,
                "total_r": total_r,
                "avg_mfe": average(mfe),
                "avg_mae": average(mae),
                "avg_early5": average(early5),
            }
        )

    return results


def print_individual_trades(
    trades,
):
    print()
    print("=" * 100)
    print("INDIVIDUAL MFE MILESTONE PROFILE")
    print("=" * 100)

    print(
        f"{'#':<4}"
        f"{'Entry':<13}"
        f"{'Outcome':<9}"
        f"{'R':>9}"
        f"{'0.5R':>8}"
        f"{'1R':>8}"
        f"{'1.5R':>8}"
        f"{'B0.5':>9}"
        f"{'B1':>9}"
        f"{'B1.5':>9}"
        f"{'MFE':>9}"
        f"{'MAE':>9}"
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

    for index, trade in enumerate(
        ordered,
        start=1,
    ):
        r = trade.get("_r")

        outcome = (
            "WIN"
            if r is not None and r > 0
            else "LOSS"
            if r is not None and r < 0
            else "FLAT"
        )

        def flag(field):
            value = trade.get(field)

            if value is True:
                return "YES"

            if value is False:
                return "NO"

            return "N/A"

        print(
            f"{index:<4}"
            f"{str(trade.get('entry_date', ''))[:12]:<13}"
            f"{outcome:<9}"
            f"{fmt(r, 'R'):>9}"
            f"{flag('_reached_0_5'):>8}"
            f"{flag('_reached_1'):>8}"
            f"{flag('_reached_1_5'):>8}"
            f"{str(trade.get('_bars_0_5') or 'N/A'):>9}"
            f"{str(trade.get('_bars_1') or 'N/A'):>9}"
            f"{str(trade.get('_bars_1_5') or 'N/A'):>9}"
            f"{fmt(trade.get('_mfe'), 'R'):>9}"
            f"{fmt(trade.get('_mae'), 'R'):>9}"
        )

        results.append(
            {
                "entry_date": trade.get(
                    "entry_date",
                    "",
                ),
                "exit_date": trade.get(
                    "exit_date",
                    "",
                ),
                "outcome": outcome,
                "r_multiple": r,
                "reached_0_5r": trade.get(
                    "_reached_0_5"
                ),
                "reached_1r": trade.get(
                    "_reached_1"
                ),
                "reached_1_5r": trade.get(
                    "_reached_1_5"
                ),
                "bars_to_0_5r": trade.get(
                    "_bars_0_5"
                ),
                "bars_to_1r": trade.get(
                    "_bars_1"
                ),
                "bars_to_1_5r": trade.get(
                    "_bars_1_5"
                ),
                "mfe_r": trade.get(
                    "_mfe"
                ),
                "mae_r": trade.get(
                    "_mae"
                ),
            }
        )

    return results


def save_csv(
    milestone_rows,
    regime_rows,
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
                "group",
                "entry_date",
                "exit_date",
                "outcome",
                "r_multiple",
                "reached_0_5r",
                "reached_1r",
                "reached_1_5r",
                "bars_to_0_5r",
                "bars_to_1r",
                "bars_to_1_5r",
                "mfe_r",
                "mae_r",
                "trades",
                "wins",
                "win_rate",
                "total_r",
                "avg_mfe",
                "avg_mae",
                "avg_early5",
                "milestone",
                "reached",
                "not_reached",
                "reach_pct",
                "avg_bars",
                "median_bars",
            ]
        )

        for row in milestone_rows:
            writer.writerow(
                [
                    "MILESTONE",
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
                    row["milestone"],
                    row["reached"],
                    row["not_reached"],
                    row["reach_pct"],
                    row["avg_bars"],
                    row["median_bars"],
                ]
            )

        for row in regime_rows:
            writer.writerow(
                [
                    "REGIME",
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
                    "",
                    "",
                    "",
                    row["trades"],
                    row["wins"],
                    row["win_rate"],
                    row["total_r"],
                    row["avg_mfe"],
                    row["avg_mae"],
                    row["avg_early5"],
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for row in trade_rows:
            writer.writerow(
                [
                    "TRADE",
                    "",
                    row["entry_date"],
                    row["exit_date"],
                    row["outcome"],
                    row["r_multiple"],
                    row["reached_0_5r"],
                    row["reached_1r"],
                    row["reached_1_5r"],
                    row["bars_to_0_5r"],
                    row["bars_to_1r"],
                    row["bars_to_1_5r"],
                    row["mfe_r"],
                    row["mae_r"],
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


def save_report(
    trades,
    winners,
    losers,
    milestone_rows,
    regime_rows,
):
    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "TRADESENSE-AI — NIFTY 50 "
            "LONG_CONTINUATION MFE MILESTONE FORENSIC\n"
        )

        file.write(
            "=" * 100 + "\n\n"
        )

        file.write(
            f"Trades: {len(trades)}\n"
        )

        file.write(
            f"Winners: {len(winners)}\n"
        )

        file.write(
            f"Losers: {len(losers)}\n\n"
        )

        file.write(
            "This report is descriptive only.\n"
        )

        file.write(
            "No threshold or trading rule was selected.\n\n"
        )

        file.write(
            "MILESTONES\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in milestone_rows:
            file.write(
                f"{row}\n"
            )

        file.write(
            "\nREGIME RESULTS\n"
        )

        file.write(
            "=" * 100 + "\n"
        )

        for row in regime_rows:
            file.write(
                f"{row}\n"
            )


def main():
    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — "
        "NIFTY 50 LONG_CONTINUATION "
        "MFE MILESTONE FORENSIC"
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

    milestone_rows = (
        print_milestone_table(
            trades
        )
    )

    print_winner_loser_milestones(
        winners,
        losers,
    )

    print_behaviour_comparison(
        winners,
        losers,
    )

    print_behaviour_flags(
        trades,
    )

    regime_rows = (
        print_regime_comparison(
            trades
        )
    )

    trade_rows = (
        print_individual_trades(
            trades
        )
    )

    save_csv(
        milestone_rows,
        regime_rows,
        trade_rows,
    )

    save_report(
        trades,
        winners,
        losers,
        milestone_rows,
        regime_rows,
    )

    print()
    print("=" * 100)
    print("MFE MILESTONE FORENSIC COMPLETE")
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
        "No filter was selected."
    )

    print(
        "No parameter was optimized."
    )


if __name__ == "__main__":
    main()