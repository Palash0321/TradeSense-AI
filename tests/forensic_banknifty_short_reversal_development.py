import csv
import os
from collections import defaultdict


TRADES_FILE = "tests/output/index_backtests/banknifty_index_trades.csv"

TARGET_SETUP = "SHORT_REVERSAL"


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_setup(value):
    return str(value).strip().upper()


def load_trades():
    if not os.path.exists(TRADES_FILE):
        raise FileNotFoundError(
            f"Trade file not found: {TRADES_FILE}"
        )

    with open(
        TRADES_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def trade_r(trade):
    return to_float(trade.get("r_multiple"))


def trade_year(trade):
    entry_date = str(trade.get("entry_date", ""))
    return entry_date[:4]


def print_header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def summarize(label, trades):
    count = len(trades)

    if count == 0:
        print(f"{label}: 0 trades")
        return

    winners = [
        trade for trade in trades
        if trade_r(trade) > 0
    ]

    losers = [
        trade for trade in trades
        if trade_r(trade) <= 0
    ]

    gross_profit = sum(
        trade_r(trade)
        for trade in winners
    )

    gross_loss = sum(
        trade_r(trade)
        for trade in losers
    )

    total_r = sum(
        trade_r(trade)
        for trade in trades
    )

    win_rate = (
        len(winners) / count * 100
        if count
        else 0
    )

    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    else:
        profit_factor = float("inf")

    print(f"{label}:")
    print(f"  Trades       : {count}")
    print(f"  Winners      : {len(winners)}")
    print(f"  Losers       : {len(losers)}")
    print(f"  Win rate     : {win_rate:.2f}%")
    print(f"  Gross profit : {gross_profit:+.2f}R")
    print(f"  Gross loss   : {gross_loss:+.2f}R")
    print(f"  Total R      : {total_r:+.2f}R")
    print(
        f"  Profit factor: "
        f"{profit_factor:.3f}"
        if profit_factor != float("inf")
        else "  Profit factor: INF"
    )


def average(trades, field):
    values = [
        to_float(trade.get(field))
        for trade in trades
        if str(trade.get(field, "")).strip() != ""
    ]

    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0

    values = sorted(values)
    middle = len(values) // 2

    if len(values) % 2:
        return values[middle]

    return (values[middle - 1] + values[middle]) / 2


def print_development(trades):
    print_header("DEVELOPMENT PROFILE")

    if not trades:
        print("No SHORT_REVERSAL trades found.")
        return

    mfe_values = [
        to_float(trade.get("mfe_r"))
        for trade in trades
    ]

    mae_values = [
        to_float(trade.get("mae_r"))
        for trade in trades
    ]

    bars_values = [
        to_int(trade.get("bars_in_trade"))
        for trade in trades
    ]

    atr_values = [
        to_float(trade.get("entry_atr"))
        for trade in trades
    ]

    risk_values = [
        to_float(trade.get("initial_risk"))
        for trade in trades
    ]

    print(f"Average ATR          : {average(trades, 'entry_atr'):.2f}")
    print(f"Average initial risk : {average(trades, 'initial_risk'):.2f}")
    print(f"Average MFE          : {average(trades, 'mfe_r'):.3f}R")
    print(f"Median MFE           : {median(mfe_values):.3f}R")
    print(f"Average MAE          : {average(trades, 'mae_r'):.3f}R")
    print(f"Median MAE           : {median(mae_values):.3f}R")
    print(
        f"Average holding     : "
        f"{sum(bars_values) / len(bars_values):.2f} bars"
    )


def print_mfe_milestones(trades):
    print_header("MFE MILESTONES")

    thresholds = [
        0.25,
        0.50,
        1.00,
        1.50,
        2.00,
    ]

    for threshold in thresholds:
        reached = [
            trade
            for trade in trades
            if to_float(trade.get("mfe_r")) >= threshold
        ]

        total_r = sum(
            trade_r(trade)
            for trade in reached
        )

        print(
            f">= {threshold:.2f}R : "
            f"{len(reached):2d}/{len(trades)} trades | "
            f"outcome {total_r:+.2f}R"
        )


def print_early_development(trades):
    print_header("EARLY DEVELOPMENT")

    buckets = {
        "NEVER_0_5R": [],
        "0_5_TO_1R": [],
        "1_TO_1_5R": [],
        "GE_1_5R": [],
    }

    for trade in trades:
        mfe = to_float(trade.get("mfe_r"))

        if mfe < 0.50:
            buckets["NEVER_0_5R"].append(trade)
        elif mfe < 1.00:
            buckets["0_5_TO_1R"].append(trade)
        elif mfe < 1.50:
            buckets["1_TO_1_5R"].append(trade)
        else:
            buckets["GE_1_5R"].append(trade)

    for name, bucket in buckets.items():
        total_r = sum(
            trade_r(trade)
            for trade in bucket
        )

        print(
            f"{name:15s}: "
            f"{len(bucket):2d} trades | "
            f"{total_r:+.2f}R"
        )


def print_early_adverse(trades):
    print_header("EARLY ADVERSE DEVELOPMENT")

    fields = [
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5",
    ]

    for field in fields:
        values = [
            to_float(trade.get(field))
            for trade in trades
            if str(trade.get(field, "")).strip() != ""
        ]

        if not values:
            continue

        print(
            f"{field:26s}: "
            f"avg={sum(values) / len(values):.3f}R | "
            f"max={max(values):.3f}R"
        )

    threshold = 0.25

    triggered = []
    not_triggered = []

    for trade in trades:
        value = to_float(
            trade.get("early_adverse_r_bar_5")
        )

        if value >= threshold:
            triggered.append(trade)
        else:
            not_triggered.append(trade)

    print()
    print(
        f"5B >= {threshold:.2f}R:"
    )

    summarize(
        "  Triggered",
        triggered,
    )

    summarize(
        "  Not triggered",
        not_triggered,
    )


def print_favorable_adverse_order(trades):
    print_header("FAVORABLE / ADVERSE ORDER")

    counts = defaultdict(list)

    for trade in trades:
        favorable_first = str(
            trade.get("favorable_first", "")
        ).strip().upper()

        if not favorable_first:
            favorable_first = "UNKNOWN"

        counts[favorable_first].append(trade)

    for label, bucket in sorted(counts.items()):
        total_r = sum(
            trade_r(trade)
            for trade in bucket
        )

        print(
            f"{label:20s}: "
            f"{len(bucket):2d} trades | "
            f"{total_r:+.2f}R"
        )


def print_exit_profile(trades):
    print_header("EXIT PROFILE")

    grouped = defaultdict(list)

    for trade in trades:
        exit_reason = str(
            trade.get("exit_reason", "")
        ).strip().upper()

        grouped[exit_reason].append(trade)

    for reason, bucket in sorted(grouped.items()):
        summarize(
            f"  {reason}",
            bucket,
        )


def print_year_profile(trades):
    print_header("YEAR PROFILE")

    grouped = defaultdict(list)

    for trade in trades:
        grouped[trade_year(trade)].append(trade)

    for year in sorted(grouped):
        bucket = grouped[year]

        total_r = sum(
            trade_r(trade)
            for trade in bucket
        )

        winners = sum(
            1
            for trade in bucket
            if trade_r(trade) > 0
        )

        win_rate = (
            winners / len(bucket) * 100
            if bucket
            else 0
        )

        print(
            f"{year}: "
            f"{len(bucket):2d} trades | "
            f"{win_rate:6.2f}% | "
            f"{total_r:+.2f}R"
        )


def print_individual_trades(trades):
    print_header("INDIVIDUAL SHORT_REVERSAL TRADES")

    sorted_trades = sorted(
        trades,
        key=lambda trade: (
            str(trade.get("entry_date", "")),
            str(trade.get("symbol", "")),
        ),
    )

    for trade in sorted_trades:
        print(
            f"{str(trade.get('entry_date', ''))[:10]} | "
            f"{str(trade.get('exit_date', ''))[:10]} | "
            f"{str(trade.get('symbol', '')):14s} | "
            f"R={trade_r(trade):+.2f} | "
            f"MFE={to_float(trade.get('mfe_r')):.2f} | "
            f"MAE={to_float(trade.get('mae_r')):.2f} | "
            f"bars={to_int(trade.get('bars_in_trade'))}"
        )


def compare_other_setups(all_trades):
    print_header("SETUP COMPARISON")

    grouped = defaultdict(list)

    for trade in all_trades:
        setup = normalize_setup(
            trade.get("setup")
        )

        grouped[setup].append(trade)

    for setup in sorted(grouped):
        summarize(
            f"{setup}",
            grouped[setup],
        )


def main():
    print_header(
        "BANK NIFTY SHORT_REVERSAL DEVELOPMENT FORENSIC"
    )

    all_trades = load_trades()

    print(f"Trade file      : {TRADES_FILE}")
    print(f"Total trades    : {len(all_trades)}")

    target_trades = [
        trade
        for trade in all_trades
        if normalize_setup(
            trade.get("setup")
        ) == TARGET_SETUP
    ]

    print(
        f"Target setup    : {TARGET_SETUP}"
    )
    print(
        f"Target trades   : {len(target_trades)}"
    )

    if not target_trades:
        print()
        print(
            "No SHORT_REVERSAL trades found."
        )
        print(
            "FORENSIC STATUS: NO_TARGET_TRADES"
        )
        return

    summarize(
        "SHORT_REVERSAL BASELINE",
        target_trades,
    )

    print_development(target_trades)
    print_mfe_milestones(target_trades)
    print_early_development(target_trades)
    print_early_adverse(target_trades)
    print_favorable_adverse_order(target_trades)
    print_exit_profile(target_trades)
    print_year_profile(target_trades)
    print_individual_trades(target_trades)
    compare_other_setups(all_trades)

    print_header("RESEARCH VERDICT")

    total_r = sum(
        trade_r(trade)
        for trade in target_trades
    )

    avg_mfe = average(
        target_trades,
        "mfe_r",
    )

    avg_mae = average(
        target_trades,
        "mae_r",
    )

    print(
        f"SHORT_REVERSAL trades : {len(target_trades)}"
    )
    print(
        f"Total R               : {total_r:+.2f}R"
    )
    print(
        f"Average MFE           : {avg_mfe:.3f}R"
    )
    print(
        f"Average MAE           : {avg_mae:.3f}R"
    )

    if len(target_trades) < 10:
        verdict = "INSUFFICIENT_SAMPLE"
    elif total_r < 0 and avg_mfe < 1.0:
        verdict = "WEAK_DEVELOPMENT_PROFILE"
    else:
        verdict = "REQUIRES_FURTHER_FORENSICS"

    print()
    print(f"VERDICT: {verdict}")
    print()
    print(
        "This is a research-only diagnostic."
    )
    print(
        "No production, frozen, or forward-holdout "
        "dataset is modified."
    )


if __name__ == "__main__":
    main()