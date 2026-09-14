"""
Diagnostic only:
Compare the actual TradeSense signal-generation path for INFY
when the historical data starts on 2021-09-07 vs 2021-09-08.

This does NOT modify strategy code.
This does NOT modify production behavior.
"""

from pathlib import Path

import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

DATA_STARTS = [
    "2021-09-07",
    "2021-09-08",
]

END_DATE = "2022-02-01"

CHECK_DATES = [
    "2021-09-13",
    "2021-09-20",
    "2021-09-24",
    "2021-10-01",
    "2021-10-29",
    "2021-11-01",
    "2021-12-17",
    "2021-12-23",
    "2022-01-14",
    "2022-01-17",
]


def normalize_columns(df):
    """
    Flatten MultiIndex columns if yfinance returned them.
    """
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


def run_diagnostic(data_start):
    print("\n" + "=" * 100)
    print(f"DATA START = {data_start}")
    print("=" * 100)

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    # IMPORTANT:
    # Explicitly load the requested historical boundary.
    data = service.load_data(
        start_date=data_start,
        end_date=END_DATE,
    )

    if data is None or data.empty:
        print("ERROR: No data returned.")
        return None

    data = normalize_columns(data)

    print(f"Rows loaded : {len(data)}")
    print(f"First date  : {data.index.min()}")
    print(f"Last date   : {data.index.max()}")

    # IMPORTANT:
    # Pass the explicitly loaded DataFrame into signal generation.
    signals = service.generate_tradesense_signals(
        df=data,
        swing_window=3,
    )

    if signals is None or signals.empty:
        print("ERROR: No signals returned.")
        return None

    signals = signals.copy()

    print(f"Signal rows : {len(signals)}")

    # ------------------------------------------------------------------
    # Show actual generated signal records
    # ------------------------------------------------------------------

    available_dates = []

    for date in CHECK_DATES:
        try:
            ts = pd.Timestamp(date)

            matching = signals.loc[
                signals.index.normalize() == ts.normalize()
            ]

            if not matching.empty:
                available_dates.append((date, matching))
        except Exception:
            pass

    print("\nTARGET-DATE SIGNAL OUTPUT")
    print("-" * 100)

    for date, matching in available_dates:
        print(f"\nDATE = {date}")

        row = matching.iloc[-1]

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
        ]

        for field in fields:
            if field in row.index:
                print(f"{field:22s}: {row[field]}")

    # ------------------------------------------------------------------
    # Show every non-WAIT signal around the critical period
    # ------------------------------------------------------------------

    print("\n\nNON-WAIT SIGNALS")
    print("-" * 100)

    temp = signals.copy()

    temp_dates = pd.to_datetime(temp.index)

    mask = (
        (temp_dates >= pd.Timestamp("2021-09-01"))
        & (temp_dates <= pd.Timestamp("2022-02-01"))
    )

    period = temp.loc[mask]

    if "Signal" in period.columns:
        non_wait = period[period["Signal"] != 0]

        if non_wait.empty:
            print("No non-WAIT signals.")
        else:
            display_cols = [
                c
                for c in [
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
                if c in non_wait.columns
            ]

            print(non_wait[display_cols].to_string())

    # ------------------------------------------------------------------
    # Save diagnostic output dataframe for optional inspection
    # ------------------------------------------------------------------

    return signals


def main():
    results = {}

    for start in DATA_STARTS:
        results[start] = run_diagnostic(start)

    # ------------------------------------------------------------------
    # Direct comparison of the critical dates
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("DIRECT COMPARISON: 2021-09-07 vs 2021-09-08")
    print("=" * 100)

    rows = []

    for date in CHECK_DATES:
        row_data = {
            "date": date,
            "start_2021_09_07": {},
            "start_2021_09_08": {},
        }

        for start in DATA_STARTS:
            signals = results.get(start)

            if signals is None or signals.empty:
                continue

            ts = pd.Timestamp(date)

            matching = signals.loc[
                signals.index.normalize() == ts.normalize()
            ]

            if matching.empty:
                continue

            row = matching.iloc[-1]

            row_data[f"start_{start.replace('-', '_')}"] = {
                "Signal": row.get("Signal"),
                "Setup": row.get("Setup"),
                "Structure_Bias": row.get("Structure_Bias"),
                "Structure": row.get("Structure"),
                "BOS": row.get("BOS"),
                "CHOCH": row.get("CHOCH"),
                "Break_Direction": row.get("Break_Direction"),
                "Break_Confirmed": row.get("Break_Confirmed"),
                "Break_Age": row.get("Break_Age"),
            }

        rows.append(row_data)

    print("\n")

    for item in rows:
        print(f"DATE = {item['date']}")
        print(
            "  START 2021-09-07:",
            item["start_2021_09_07"],
        )
        print(
            "  START 2021-09-08:",
            item["start_2021_09_08"],
        )
        print()

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)
    print("No strategy files were modified.")


if __name__ == "__main__":
    main()