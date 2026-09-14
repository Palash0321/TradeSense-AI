import yfinance as yf
import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

# Keep the market endpoint identical across all runs.
END_DATE = "2026-07-29"

START_DATES = [
    "2021-08-23",
    "2021-08-24",
    "2021-08-25",
    "2021-08-26",
    "2021-08-27",
    "2021-08-30",
    "2021-08-31",
    "2021-09-01",
    "2021-09-02",
    "2021-09-03",
    "2021-09-06",
    "2021-09-07",
    "2021-09-08",
    "2021-09-09",
]

COMPARE_FROM = pd.Timestamp("2021-10-01")


def download_data(start_date):
    data = yf.download(
        SYMBOL,
        start=start_date,
        end=END_DATE,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=15,
    )

    if data.empty:
        return data

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.copy()
    data.index = pd.to_datetime(data.index).normalize()

    return data


def generate_signals(start_date):
    data = download_data(start_date)

    if data.empty:
        return None

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    signals = service.generate_tradesense_signals(df=data)

    signals = signals.copy()
    signals.index = pd.to_datetime(signals.index).normalize()

    return signals


def extract_signal_rows(signals):
    result = signals.loc[
        signals.index >= COMPARE_FROM
    ].copy()

    rows = []

    for idx, row in result.iterrows():
        signal = row.get("Signal", 0)

        if pd.isna(signal):
            signal = 0

        rows.append(
            {
                "date": idx,
                "signal": int(signal),
                "setup": row.get("Setup", None),
                "entry_trend": row.get("Entry_Trend", None),
                "trend_strength": row.get("Trend_Strength", None),
                "break_direction": row.get(
                    "Break_Direction",
                    None,
                ),
                "break_confirmed": row.get(
                    "Break_Confirmed",
                    None,
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("TRADE SENSE-AI — SIGNAL-LEVEL BOUNDARY INVARIANCE TEST")
    print("=" * 100)
    print(f"Symbol: {SYMBOL}")
    print(f"Compare from: {COMPARE_FROM.date()}")
    print(f"Common end date: {END_DATE}")
    print()

    all_runs = {}

    for start_date in START_DATES:
        print("-" * 100)
        print(f"GENERATING SIGNALS — START: {start_date}")

        signals = generate_signals(start_date)

        if signals is None:
            print("NO DATA")
            continue

        extracted = extract_signal_rows(signals)

        all_runs[start_date] = extracted

        nonzero = extracted[
            extracted["signal"] != 0
        ]

        print(f"Rows compared: {len(extracted)}")
        print(f"Non-zero signals: {len(nonzero)}")

        for _, row in nonzero.iterrows():
            print(
                f"  {row['date'].date()} | "
                f"Signal={row['signal']} | "
                f"Setup={row['setup']}"
            )

    print()
    print("=" * 100)
    print("PAIRWISE INVARIANCE AGAINST 2021-08-23 BASELINE")
    print("=" * 100)

    baseline_start = START_DATES[0]
    baseline = all_runs.get(baseline_start)

    if baseline is None:
        print("Baseline unavailable.")
        return

    baseline = baseline.set_index("date")

    print(
        f"{'START':<14}"
        f"{'COMMON DATES':<15}"
        f"{'SIGNAL DIFF':<15}"
        f"{'SETUP DIFF':<15}"
        f"{'IDENTICAL?':<12}"
    )

    print("-" * 100)

    for start_date in START_DATES:
        current = all_runs.get(start_date)

        if current is None:
            continue

        current = current.set_index("date")

        common_dates = baseline.index.intersection(
            current.index
        )

        signal_diff = (
            baseline.loc[common_dates, "signal"]
            != current.loc[common_dates, "signal"]
        ).sum()

        setup_diff = (
            baseline.loc[common_dates, "setup"].astype(str)
            != current.loc[common_dates, "setup"].astype(str)
        ).sum()

        identical = (
            signal_diff == 0
            and setup_diff == 0
        )

        print(
            f"{start_date:<14}"
            f"{len(common_dates):<15}"
            f"{signal_diff:<15}"
            f"{setup_diff:<15}"
            f"{str(identical):<12}"
        )

    print()
    print("=" * 100)
    print("FIRST SIGNAL DIFFERENCES VS BASELINE")
    print("=" * 100)

    for start_date in START_DATES:
        if start_date == baseline_start:
            continue

        current = all_runs.get(start_date)

        if current is None:
            continue

        current = current.set_index("date")

        common_dates = baseline.index.intersection(
            current.index
        )

        differences = []

        for date in common_dates:
            base_signal = baseline.loc[
                date, "signal"
            ]
            current_signal = current.loc[
                date, "signal"
            ]

            base_setup = str(
                baseline.loc[date, "setup"]
            )
            current_setup = str(
                current.loc[date, "setup"]
            )

            if (
                base_signal != current_signal
                or base_setup != current_setup
            ):
                differences.append(
                    (
                        date,
                        base_signal,
                        current_signal,
                        base_setup,
                        current_setup,
                    )
                )

        print()
        print(f"START {start_date}")

        if not differences:
            print("  No differences vs baseline.")
            continue

        for (
            date,
            base_signal,
            current_signal,
            base_setup,
            current_setup,
        ) in differences[:20]:
            print(
                f"  {date.date()} | "
                f"baseline Signal={base_signal}, "
                f"current Signal={current_signal} | "
                f"baseline Setup={base_setup}, "
                f"current Setup={current_setup}"
            )

        if len(differences) > 20:
            print(
                f"  ... {len(differences) - 20} more differences"
            )

    print()
    print("=" * 100)
    print("SIGNAL-LEVEL INVARIANCE TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()