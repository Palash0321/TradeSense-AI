"""
Diagnostic only.

Compare TradeSense signal generation + directional backtest execution
for INFY when the historical data starts on 2021-09-07 vs 2021-09-08.

Purpose:
    Determine whether the 2021-10-01 signal difference causes a different
    position/execution path, which then changes later executed trades.

This does NOT modify strategy code.
"""

import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

DATA_STARTS = [
    "2021-09-07",
    "2021-09-08",
]

END_DATE = "2022-02-01"

CHECK_DATES = [
    "2021-10-01",
    "2021-10-04",
    "2021-10-18",
    "2021-10-29",
    "2021-11-01",
    "2021-12-17",
    "2021-12-23",
    "2022-01-14",
    "2022-01-17",
    "2022-01-21",
]


def normalize_columns(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        flattened = []

        for col in df.columns:
            if isinstance(col, tuple):
                flattened.append(col[0])
            else:
                flattened.append(col)

        df.columns = flattened

    return df


def get_date_rows(df, date):
    ts = pd.Timestamp(date)

    return df.loc[
        pd.to_datetime(df.index).normalize() == ts.normalize()
    ]


def print_signal_path(signals, start):
    print("\n")
    print("=" * 110)
    print(f"SIGNAL PATH — DATA START {start}")
    print("=" * 110)

    for date in CHECK_DATES:
        matching = get_date_rows(signals, date)

        if matching.empty:
            print(f"\n{date}: NO ROW")
            continue

        row = matching.iloc[-1]

        print(f"\nDATE = {date}")

        for field in [
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
        ]:
            if field in row.index:
                print(f"{field:22s}: {row[field]}")


def print_backtest_summary(result, start):
    print("\n")
    print("=" * 110)
    print(f"BACKTEST RESULT — DATA START {start}")
    print("=" * 110)

    if result is None:
        print("Backtest result is None.")
        return

    if isinstance(result, dict):
        print("\nResult keys:")
        print(list(result.keys()))

        for key in [
            "total_trades",
            "total_r",
            "win_rate",
            "profit_factor",
            "final_capital",
        ]:
            if key in result:
                print(f"{key:22s}: {result[key]}")

        trades = result.get("trades", [])

    else:
        print(f"Result type: {type(result)}")
        trades = []

    print(f"\nTrades returned: {len(trades)}")

    if not trades:
        return

    trades_df = pd.DataFrame(trades)

    print("\nTRADE COLUMNS")
    print("-" * 110)
    print(list(trades_df.columns))

    print("\nTRADES IN CRITICAL PERIOD")
    print("-" * 110)

    date_column = None

    for candidate in [
        "Entry Date",
        "entry_date",
        "EntryDate",
        "date",
        "Date",
    ]:
        if candidate in trades_df.columns:
            date_column = candidate
            break

    if date_column is None:
        print("Could not identify entry-date column.")
        print(trades_df.to_string())
        return

    entry_dates = pd.to_datetime(
        trades_df[date_column],
        errors="coerce",
    )

    mask = (
        (entry_dates >= pd.Timestamp("2021-09-01"))
        & (entry_dates <= pd.Timestamp("2022-02-01"))
    )

    critical = trades_df.loc[mask].copy()

    if critical.empty:
        print("No trades in critical period.")
    else:
        print(critical.to_string(index=False))


def compare_trade_entries(all_results):
    print("\n")
    print("=" * 110)
    print("DIRECT TRADE-ENTRY COMPARISON")
    print("=" * 110)

    for start, result in all_results.items():
        print(f"\nDATA START = {start}")

        if not isinstance(result, dict):
            print("Invalid result.")
            continue

        trades = result.get("trades", [])

        if not trades:
            print("No trades.")
            continue

        trades_df = pd.DataFrame(trades)

        date_column = None

        for candidate in [
            "Entry Date",
            "entry_date",
            "EntryDate",
            "date",
            "Date",
        ]:
            if candidate in trades_df.columns:
                date_column = candidate
                break

        if date_column is None:
            print("Entry-date column not found.")
            continue

        entry_dates = pd.to_datetime(
            trades_df[date_column],
            errors="coerce",
        )

        mask = (
            (entry_dates >= pd.Timestamp("2021-09-01"))
            & (entry_dates <= pd.Timestamp("2022-02-01"))
        )

        critical = trades_df.loc[mask]

        if critical.empty:
            print("No trades.")
        else:
            print(
                critical.to_string(index=False)
            )


def run_one(start):
    print("\n")
    print("#" * 110)
    print(f"RUNNING DATA START = {start}")
    print("#" * 110)

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    # ------------------------------------------------------------
    # STEP 1: Explicitly load boundary-specific data
    # ------------------------------------------------------------

    data = service.load_data(
        start_date=start,
        end_date=END_DATE,
    )

    if data is None or data.empty:
        print("ERROR: No data loaded.")
        return None, None

    data = normalize_columns(data)

    print("\nDATA")
    print("-" * 110)
    print(f"Rows       : {len(data)}")
    print(f"First date : {data.index.min()}")
    print(f"Last date  : {data.index.max()}")

    # ------------------------------------------------------------
    # STEP 2: Generate signals from THAT exact DataFrame
    # ------------------------------------------------------------

    signals = service.generate_tradesense_signals(
        df=data,
        swing_window=3,
    )

    if signals is None or signals.empty:
        print("ERROR: No signals generated.")
        return None, None

    print_signal_path(signals, start)

    # ------------------------------------------------------------
    # STEP 3: Run directional backtest
    #
    # IMPORTANT:
    # Inspect the method signature first if this fails.
    # ------------------------------------------------------------

    result = service.run_directional_backtest(
        signals,
        signal_column="Signal",
        initial_stop_atr=3.5,
        target_r=2.0,
        risk_per_trade=0.01,
    )

    print_backtest_summary(result, start)

    return signals, result


def main():
    all_signals = {}
    all_results = {}

    for start in DATA_STARTS:
        signals, result = run_one(start)

        all_signals[start] = signals
        all_results[start] = result

    compare_trade_entries(all_results)

    print("\n")
    print("=" * 110)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 110)
    print("No strategy files were modified.")


if __name__ == "__main__":
    main()