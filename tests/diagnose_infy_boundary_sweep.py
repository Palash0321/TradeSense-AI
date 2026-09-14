import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

# Diagnostic range only.
# We are characterizing how the historical data boundary affects
# signal generation and execution.
START_DATES = pd.date_range(
    start="2021-08-23",
    end="2021-09-08",
    freq="D",
)

END_DATE = "2022-02-01"

SWING_WINDOW = 3

# Historical trades that disappeared from the fresh 103-trade dataset.
TARGET_TRADES = {
    "2021-10-04": {
        "direction": "SHORT",
        "label": "2021-10-04 SHORT",
        "old_entry": 1670.83,
        "old_r": -1.06,
    },
    "2021-11-01": {
        "direction": "SHORT",
        "label": "2021-11-01 SHORT",
        "old_entry": 1675.82,
        "old_r": -1.05,
    },
    "2022-01-17": {
        "direction": "LONG",
        "label": "2022-01-17 LONG",
        "old_entry": 1940.49,
        "old_r": -1.06,
    },
}


def normalize_date(value):
    if value is None:
        return None

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def get_trade_entry_date(trade):
    value = trade.get("entry_date")

    if value is None:
        value = trade.get("Entry_Date")

    if value is None:
        value = trade.get("date")

    if value is None:
        value = trade.get("Date")

    return normalize_date(value)


def get_trade_direction(trade):
    value = trade.get("direction")

    if value is None:
        value = trade.get("Direction")

    if value is None:
        value = trade.get("side")

    if value is None:
        value = trade.get("Side")

    if value is None:
        return None

    return str(value).strip().upper()


def get_entry_price(trade):
    value = trade.get("entry_price")

    if value is None:
        value = trade.get("Entry_Price")

    if value is None:
        value = trade.get("entry")

    if value is None:
        value = trade.get("Entry")

    try:
        return float(value)
    except Exception:
        return None


def get_r_multiple(trade):
    value = trade.get("r_multiple")

    if value is None:
        value = trade.get("R_Multiple")

    if value is None:
        value = trade.get("R")

    try:
        return float(value)
    except Exception:
        return None


def find_target_trade(trades, target_date, target_direction):
    for trade in trades:
        entry_date = get_trade_entry_date(trade)
        direction = get_trade_direction(trade)

        if (
            entry_date == target_date
            and direction == target_direction
        ):
            return trade

    return None


def main():
    summary_rows = []

    print("=" * 100)
    print("INFY HISTORICAL DATA-BOUNDARY SWEEP")
    print("=" * 100)

    print()
    print("PURPOSE:")
    print(
        "Characterize deterministic boundary sensitivity in signal generation "
        "and execution."
    )
    print(
        "This is diagnostic only. It does NOT select a production data start "
        "date or optimize the strategy."
    )

    print()
    print("TARGET HISTORICAL TRADES:")
    for target_date, target in TARGET_TRADES.items():
        print(
            f"  {target['label']} | "
            f"old entry={target['old_entry']:.2f} | "
            f"old R={target['old_r']:.2f}"
        )

    print()
    print("=" * 100)

    for start_date in START_DATES:
        start_str = start_date.strftime("%Y-%m-%d")

        print()
        print("-" * 100)
        print(f"DATA START = {start_str}")
        print("-" * 100)

        try:
            service = BacktestService(
                symbol=SYMBOL,
                strategy="tradesense",
            )

            # IMPORTANT:
            # Explicitly load the requested historical boundary.
            data = service.load_data(
                start_date=start_str,
                end_date=END_DATE,
            )

            if data is None or data.empty:
                print("NO DATA")
                continue

            print(f"Rows       : {len(data)}")
            print(
                f"First date : "
                f"{pd.Timestamp(data.index.min()).strftime('%Y-%m-%d')}"
            )
            print(
                f"Last date  : "
                f"{pd.Timestamp(data.index.max()).strftime('%Y-%m-%d')}"
            )

            # IMPORTANT:
            # Pass the already-loaded boundary-specific dataframe.
            signals = service.generate_tradesense_signals(
                df=data,
                swing_window=SWING_WINDOW,
            )

            backtest_result = service.run_directional_backtest(
                signals
            )

            trades = backtest_result.get("trades", [])

            total_r = sum(
                get_r_multiple(trade) or 0.0
                for trade in trades
            )

            print(f"Total trades: {len(trades)}")
            print(f"Total R     : {total_r:.2f}")

            found_flags = {}

            print()
            print("TARGET TRADE CHECKS:")

            for target_date, target in TARGET_TRADES.items():
                trade = find_target_trade(
                    trades,
                    target_date,
                    target["direction"],
                )

                found = trade is not None
                found_flags[target_date] = found

                if found:
                    entry_price = get_entry_price(trade)
                    r_multiple = get_r_multiple(trade)

                    print(
                        f"  FOUND    {target['label']} | "
                        f"entry={entry_price:.2f} | "
                        f"R={r_multiple:.2f}"
                    )
                else:
                    print(
                        f"  MISSING  {target['label']}"
                    )

            target_count = sum(found_flags.values())

            summary_rows.append(
                {
                    "data_start": start_str,
                    "rows": len(data),
                    "first_date": pd.Timestamp(
                        data.index.min()
                    ).strftime("%Y-%m-%d"),
                    "total_trades": len(trades),
                    "total_R": round(total_r, 2),
                    "10_04_SHORT": "YES"
                    if found_flags.get("2021-10-04", False)
                    else "NO",
                    "11_01_SHORT": "YES"
                    if found_flags.get("2021-11-01", False)
                    else "NO",
                    "01_17_LONG": "YES"
                    if found_flags.get("2022-01-17", False)
                    else "NO",
                    "target_count": target_count,
                }
            )

        except Exception as exc:
            print()
            print(f"ERROR for start {start_str}:")
            print(f"{type(exc).__name__}: {exc}")

            summary_rows.append(
                {
                    "data_start": start_str,
                    "rows": None,
                    "first_date": None,
                    "total_trades": None,
                    "total_R": None,
                    "10_04_SHORT": "ERROR",
                    "11_01_SHORT": "ERROR",
                    "01_17_LONG": "ERROR",
                    "target_count": None,
                }
            )

    print()
    print()
    print("=" * 100)
    print("BOUNDARY SWEEP SUMMARY")
    print("=" * 100)

    if not summary_rows:
        print("No results.")
        return

    summary_df = pd.DataFrame(summary_rows)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    successful = summary_df[
        summary_df["target_count"].notna()
    ]

    if successful.empty:
        print("No successful boundary runs.")
        return

    max_targets = successful["target_count"].max()

    print(
        f"Maximum historical target trades reproduced: "
        f"{int(max_targets)}/3"
    )

    best_rows = successful[
        successful["target_count"] == max_targets
    ]

    print()
    print("Starts achieving the maximum target count:")
    for _, row in best_rows.iterrows():
        print(
            f"  {row['data_start']} -> "
            f"{int(row['target_count'])}/3 targets"
        )

    print()
    print(
        "IMPORTANT: These starts are NOT being selected as the "
        "correct production boundary."
    )
    print(
        "The sweep is only being used to understand why the "
        "historical reconstruction changes when the input boundary changes."
    )

    print()
    print(
        "Do not modify the strategy, frozen 5B_0.25R rule, or production "
        "data-loading behavior based on this diagnostic."
    )


if __name__ == "__main__":
    main()