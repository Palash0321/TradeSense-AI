import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"
DATA_START = "2021-09-07"
END_DATE = "2022-02-01"
SWING_WINDOW = 3


def normalize_date(value):
    if value is None:
        return None

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def get_value(row, *names):
    for name in names:
        if isinstance(row, dict) and name in row:
            return row[name]

        if hasattr(row, "get"):
            value = row.get(name)
            if value is not None:
                return value

    return None


def print_signal_row(signals, date_str):
    target_date = pd.Timestamp(date_str)

    matching = signals[
        pd.to_datetime(signals.index).normalize()
        == target_date
    ]

    if matching.empty:
        print(f"\n{date_str}: NO ROW")
        return

    row = matching.iloc[0]

    print(f"\n{'=' * 90}")
    print(f"SIGNAL DATE: {date_str}")
    print(f"{'=' * 90}")

    fields = [
        "Signal",
        "Setup",
        "Setup_Direction",
        "Setup_Confidence",
        "Structure_Bias",
        "Structure",
        "BOS",
        "CHOCH",
        "Break_Direction",
        "Break_Confirmed",
        "Break_Age",
        "Break_Level",
        "Trend_Strength",
        "Trend_Slope",
    ]

    for field in fields:
        if field in signals.columns:
            print(f"{field:25}: {row[field]}")


def print_execution_window(signals):
    print()
    print("=" * 90)
    print("EXECUTION WINDOW")
    print("=" * 90)

    start = pd.Timestamp("2022-01-10")
    end = pd.Timestamp("2022-01-25")

    window = signals[
        (pd.to_datetime(signals.index) >= start)
        & (pd.to_datetime(signals.index) <= end)
    ]

    columns = [
        "Signal",
        "Setup",
        "Setup_Direction",
        "Structure_Bias",
        "Structure",
        "BOS",
        "CHOCH",
        "Break_Direction",
        "Break_Confirmed",
        "Break_Age",
    ]

    available = [
        column
        for column in columns
        if column in window.columns
    ]

    if window.empty:
        print("No rows in execution window.")
        return

    print(
        window[available].to_string()
    )


def print_backtest_trades(trades):
    print()
    print("=" * 90)
    print("BACKTEST TRADES")
    print("=" * 90)

    if not trades:
        print("No trades.")
        return

    for i, trade in enumerate(trades, start=1):
        print()
        print(f"TRADE #{i}")

        for key, value in trade.items():
            print(f"  {key}: {value}")


def main():
    print("=" * 90)
    print("INFY JANUARY 2022 EXECUTION TRACE")
    print("=" * 90)

    print(f"Symbol      : {SYMBOL}")
    print(f"Data start  : {DATA_START}")
    print(f"End date    : {END_DATE}")
    print(f"Swing window: {SWING_WINDOW}")

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
    )

    # Explicit boundary-specific data.
    data = service.load_data(
        start_date=DATA_START,
        end_date=END_DATE,
    )

    print()
    print(f"Rows       : {len(data)}")
    print(
        f"First date : "
        f"{pd.Timestamp(data.index.min()).strftime('%Y-%m-%d')}"
    )
    print(
        f"Last date  : "
        f"{pd.Timestamp(data.index.max()).strftime('%Y-%m-%d')}"
    )

    # Generate signals from the exact dataframe.
    signals = service.generate_tradesense_signals(
        df=data,
        swing_window=SWING_WINDOW,
    )

    print()
    print("TARGET SIGNALS")

    print_signal_row(signals, "2022-01-14")
    print_signal_row(signals, "2022-01-17")

    print_execution_window(signals)

    # Run the actual execution layer.
    result = service.run_directional_backtest(signals)

    trades = result.get("trades", [])

    print_backtest_trades(trades)

    print()
    print("=" * 90)
    print("TARGET TRADE SEARCH")
    print("=" * 90)

    for target_date, target_direction in [
        ("2021-11-01", "SHORT"),
        ("2022-01-17", "LONG"),
    ]:
        found = []

        for trade in trades:
            entry_date = normalize_date(
                get_value(
                    trade,
                    "entry_date",
                    "Entry_Date",
                    "date",
                    "Date",
                )
            )

            direction = get_value(
                trade,
                "direction",
                "Direction",
                "side",
                "Side",
            )

            if direction is not None:
                direction = str(direction).strip().upper()

            if (
                entry_date == target_date
                and direction == target_direction
            ):
                found.append(trade)

        print()

        if found:
            print(
                f"FOUND {target_date} {target_direction}"
            )
            for trade in found:
                print(trade)
        else:
            print(
                f"MISSING {target_date} {target_direction}"
            )


if __name__ == "__main__":
    main()