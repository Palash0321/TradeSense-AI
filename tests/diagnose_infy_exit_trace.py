from pathlib import Path
import pandas as pd
import inspect

from app.services.backtest_service import BacktestService


DATA_START = "2021-09-07"
DATA_END = "2022-02-01"

OLD_FILE = Path("tests/output/tradesense_trades_OLD_REFERENCE.csv")


def pick_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"Could not find {label}. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def print_trade(label, trade):
    print(f"\n--- {label} ---")

    if trade is None:
        print("NOT FOUND")
        return

    preferred = [
        "_symbol",
        "symbol",
        "Symbol",
        "ticker",
        "Ticker",
        "Entry_Date",
        "entry_date",
        "Entry_Price",
        "entry_price",
        "Entry_ATR",
        "entry_atr",
        "ATR",
        "Initial_Stop",
        "initial_stop",
        "Stop_Loss",
        "stop_loss",
        "Risk_Per_Share",
        "risk_per_share",
        "Initial_Risk",
        "initial_risk",
        "Shares",
        "shares",
        "Exit_Date",
        "exit_date",
        "Exit_Price",
        "exit_price",
        "Exit_Reason",
        "exit_reason",
        "R_Multiple",
        "r_multiple",
        "Final_R",
        "final_r",
    ]

    for col in preferred:
        if col in trade.index:
            print(f"{col}: {trade[col]}")


def find_old_trade(old, entry_date, direction):
    date_col = pick_column(
        old,
        ["Entry_Date", "entry_date", "Entry Date"],
        "old entry date",
    )

    direction_col = pick_column(
        old,
        ["Direction", "direction"],
        "old direction",
    )

    dates = pd.to_datetime(old[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    matches = old[
        (dates == entry_date)
        & (old[direction_col].astype(str).str.upper() == direction.upper())
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def find_current_trade(current_trades, entry_date, direction):
    for trade in current_trades:
        trade_date = str(
            trade.get("Entry_Date")
            or trade.get("entry_date")
            or trade.get("entryDate")
            or ""
        )[:10]

        trade_direction = str(
            trade.get("Direction")
            or trade.get("direction")
            or ""
        ).upper()

        if trade_date == entry_date and trade_direction == direction.upper():
            return trade

    return None


def print_candle_window(data, start_date, end_date):
    print(f"\n{'=' * 90}")
    print(f"CANDLE WINDOW: {start_date} -> {end_date}")
    print(f"{'=' * 90}")

    window = data.loc[start_date:end_date]

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "Signal",
        "Setup",
    ]

    available = [c for c in cols if c in window.columns]

    print(window[available].to_string())


def main():

    print("=" * 90)
    print("INFY OLD vs CURRENT EXECUTION SEMANTICS DIAGNOSTIC")
    print("=" * 90)

    # ------------------------------------------------------------------
    # OLD REFERENCE
    # ------------------------------------------------------------------

    print("\nLoading OLD reference...")

    old = pd.read_csv(OLD_FILE)

    print(f"Old reference rows: {len(old)}")
    print(f"Old columns:")
    print(list(old.columns))

    old_1004 = find_old_trade(old, "2021-10-04", "SHORT")
    old_1101 = find_old_trade(old, "2021-11-01", "SHORT")
    old_0117 = find_old_trade(old, "2022-01-17", "LONG")

    print_trade("OLD 2021-10-04 SHORT", old_1004)
    print_trade("OLD 2021-11-01 SHORT", old_1101)
    print_trade("OLD 2022-01-17 LONG", old_0117)

    # ------------------------------------------------------------------
    # CURRENT DATA
    # ------------------------------------------------------------------

    print("\nLoading CURRENT boundary-specific data...")

    service = BacktestService(
        symbol="INFY.NS",
        strategy="tradesense",
    )

    data = service.load_data(
        start_date=DATA_START,
        end_date=DATA_END,
    )

    print(f"Current rows: {len(data)}")
    print(f"Current first date: {data.index.min()}")
    print(f"Current last date: {data.index.max()}")

    # ------------------------------------------------------------------
    # SIGNAL GENERATION
    # ------------------------------------------------------------------

    signals = service.generate_tradesense_signals(
        df=data,
        swing_window=3
    )

    print("\nSignal rows around relevant entries:")

    signal_dates = [
        "2021-10-01",
        "2021-10-04",
        "2021-10-18",
        "2021-10-29",
        "2021-11-01",
        "2021-11-12",
        "2021-12-17",
        "2021-12-20",
        "2021-12-23",
        "2022-01-14",
        "2022-01-17",
        "2022-01-21",
        "2022-01-25",
    ]

    for d in signal_dates:
        if d in signals.index:
            row = signals.loc[d]

            print(
                d,
                "Signal=", row.get("Signal"),
                "Setup=", row.get("Setup"),
                "ATR=", row.get("ATR"),
                "Structure=", row.get("Structure"),
                "Break_Direction=", row.get("Break_Direction"),
                "Break_Confirmed=", row.get("Break_Confirmed"),
                "Break_Age=", row.get("Break_Age"),
            )

    # ------------------------------------------------------------------
    # CURRENT BACKTEST
    # ------------------------------------------------------------------

    current_result = service.run_directional_backtest(signals)

    print("\nCURRENT BACKTEST RETURN TYPE:")
    print(type(current_result))

    print("\nCURRENT BACKTEST RETURN:")
    print(current_result)

    if isinstance(current_result, dict):
        print("\nCURRENT BACKTEST DICT KEYS:")
        print(list(current_result.keys()))

        current_trades = current_result.get("trades", [])

    elif isinstance(current_result, list):
        current_trades = current_result

    else:
        raise TypeError(
            f"Unexpected run_directional_backtest return type: "
            f"{type(current_result)}"
        )

    print(f"\nCurrent executed trades: {len(current_trades)}")

    if current_trades:
        print("\nFIRST CURRENT TRADE TYPE:")
        print(type(current_trades[0]))

        print("\nFIRST CURRENT TRADE:")
        print(current_trades[0])

    for trade in current_trades:
        if not isinstance(trade, dict):
            print("\nWARNING: CURRENT TRADE IS NOT A DICT:")
            print(type(trade))
            print(trade)
            continue

        entry_date = str(
            trade.get("Entry_Date")
            or trade.get("entry_date")
            or ""
        )[:10]

        if entry_date in {
            "2021-10-04",
            "2021-10-19",
            "2021-11-01",
            "2021-12-20",
        }:
            print_trade(
                f"CURRENT {entry_date}",
                pd.Series(trade)
            )

    # ------------------------------------------------------------------
    # SIDE-BY-SIDE COMPARISON
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("SIDE-BY-SIDE OLD vs CURRENT")
    print("=" * 90)

    comparisons = [
        ("2021-10-04", "SHORT"),
        ("2021-11-01", "SHORT"),
        ("2022-01-17", "LONG"),
    ]

    for entry_date, direction in comparisons:

        old_trade = find_old_trade(
            old,
            entry_date,
            direction,
        )

        current_trade = find_current_trade(
            current_trades,
            entry_date,
            direction,
        )

        print(f"\n{'#' * 80}")
        print(f"{entry_date} {direction}")
        print(f"{'#' * 80}")

        print_trade("OLD", old_trade)

        if current_trade is None:
            print("\n--- CURRENT ---")
            print("NOT FOUND")
        else:
            print_trade("CURRENT", pd.Series(current_trade))

    # ------------------------------------------------------------------
    # EXACT CURRENT BACKTEST SOURCE
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("CURRENT run_directional_backtest SOURCE")
    print("=" * 90)

    source = inspect.getsource(
        BacktestService.run_directional_backtest
    )

    print(source)

    # ------------------------------------------------------------------
    # CANDLE WINDOWS
    # ------------------------------------------------------------------

    print_candle_window(
        data,
        "2021-10-04",
        "2021-10-19",
    )

    print_candle_window(
        data,
        "2021-11-01",
        "2021-12-24",
    )

    print_candle_window(
        data,
        "2021-12-20",
        "2022-01-26",
    )

    print_candle_window(
        data,
        "2022-01-14",
        "2022-01-22",
    )

    print("\n")
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()