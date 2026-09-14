import yfinance as yf
import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

STARTS = [
    "2021-09-01",
    "2021-09-07",
    "2021-09-08",
]

WINDOW_START = "2021-09-20"
WINDOW_END = "2021-10-05"


def download_data(start_date):
    print("\n" + "=" * 100)
    print(f"DATA START: {start_date}")
    print("=" * 100)

    df = yf.download(
        SYMBOL,
        start=start_date,
        end="2022-02-01",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for start={start_date}")

    # Flatten yfinance MultiIndex columns if necessary.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)

    print(f"Rows: {len(df)}")
    print(f"First: {df.index[0].date()}")
    print(f"Last : {df.index[-1].date()}")

    return df


def inspect_start(start_date):
    df = download_data(start_date)

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    signals = service.generate_tradesense_signals(
        df=df,
        swing_window=3,
    )

    signals.index = pd.to_datetime(signals.index)

    mask = (
        (signals.index >= WINDOW_START)
        & (signals.index <= WINDOW_END)
    )

    cols = [
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
        "Break_Date",
    ]

    available = [c for c in cols if c in signals.columns]

    print("\nSIGNAL / STRUCTURE WINDOW")
    print("-" * 100)

    window = signals.loc[mask, available].copy()

    if window.empty:
        print("No rows in requested window.")
    else:
        print(window.to_string())

    print("\nNON-ZERO SIGNALS")
    print("-" * 100)

    nonzero = signals.loc[
        mask & signals["Signal"].fillna(0).astype(int).ne(0)
    ]

    if nonzero.empty:
        print("None")
    else:
        print(
            nonzero[
                [c for c in available if c in [
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
                    "Break_Date",
                ]]
            ].to_string()
        )

    # Explicitly inspect 2021-10-01.
    target = pd.Timestamp("2021-10-01")

    print("\nTARGET: 2021-10-01")
    print("-" * 100)

    if target in signals.index:
        row = signals.loc[target]

        for col in available:
            print(f"{col:25s}: {row[col]}")
    else:
        print("2021-10-01 not present.")

    return signals


def main():
    print("=" * 100)
    print("INFY HISTORICAL DATA-BOUNDARY DIAGNOSTIC")
    print("=" * 100)
    print()
    print("Purpose:")
    print("Compare TradeSense market-structure reconstruction under")
    print("different historical data starts.")
    print()
    print("IMPORTANT: This script is diagnostic only.")
    print("It does NOT modify production logic.")
    print("It does NOT modify the frozen 5B_0.25R rule.")

    all_signals = {}

    for start_date in STARTS:
        all_signals[start_date] = inspect_start(start_date)

    print("\n" + "=" * 100)
    print("DIRECT 2021-10-01 COMPARISON")
    print("=" * 100)

    comparison_cols = [
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
        "Break_Date",
    ]

    target = pd.Timestamp("2021-10-01")

    comparison = []

    for start_date, signals in all_signals.items():
        if target not in signals.index:
            continue

        row = signals.loc[target]

        record = {
            "data_start": start_date,
        }

        for col in comparison_cols:
            if col in signals.columns:
                record[col] = row[col]

        comparison.append(record)

    if comparison:
        result = pd.DataFrame(comparison)
        print(result.to_string(index=False))

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()