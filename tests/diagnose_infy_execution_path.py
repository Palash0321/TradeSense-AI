import pandas as pd
import yfinance as yf

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

STARTS = [
    "2021-09-07",
    "2021-09-08",
]

END_DATE = "2022-02-01"


def main():
    print("=" * 110)
    print("INFY EXECUTION-PATH DIAGNOSTIC")
    print("=" * 110)
    print()
    print("This script ONLY diagnoses execution-path differences.")
    print("It does NOT modify production code.")
    print("It does NOT modify the trading strategy.")
    print("It does NOT modify the frozen 5B_0.25R rule.")
    print()

    for start in STARTS:
        print("=" * 110)
        print(f"DATA START = {start}")
        print("=" * 110)

        service = BacktestService(
            symbol=SYMBOL,
            strategy="tradesense",
            initial_capital=1_000_000,
        )

        data = service.load_data(
            start_date=start,
            end_date=END_DATE,
        )

        signals = service.generate_tradesense_signals()

        df = signals.copy()
        df.index = pd.to_datetime(df.index)

        print(f"Rows: {len(df)}")
        print(f"First: {df.index[0].date()}")
        print(f"Last : {df.index[-1].date()}")
        print()

        print("-" * 110)
        print("SIGNALS AROUND TARGET DATES")
        print("-" * 110)

        target_start = pd.Timestamp("2021-09-20")
        target_end = pd.Timestamp("2022-01-25")

        subset = df.loc[
            (df.index >= target_start)
            & (df.index <= target_end)
        ]

        signal_columns = [
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
        ]

        available = [c for c in signal_columns if c in subset.columns]

        print(
            subset[available].to_string()
        )

        print()
        print("-" * 110)
        print("TARGET SIGNALS")
        print("-" * 110)

        for target in [
            "2021-10-01",
            "2021-10-18",
            "2021-10-29",
            "2021-11-01",
            "2021-12-17",
            "2021-12-23",
            "2022-01-14",
            "2022-01-17",
        ]:
            ts = pd.Timestamp(target)

            if ts not in df.index:
                print(f"{target}: NOT AVAILABLE")
                continue

            row = df.loc[ts]

            print(
                f"{target} | "
                f"Signal={row.get('Signal')} | "
                f"Setup={row.get('Setup')} | "
                f"Direction={row.get('Setup_Direction')} | "
                f"Confidence={row.get('Setup_Confidence')}"
            )

        print()

    print("=" * 110)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()