import pandas as pd
import yfinance as yf

from app.core.levels.market_structure import calculate_market_structure


SYMBOL = "INFY.NS"

DATA_STARTS = [
    "2021-09-01",
    "2021-09-07",
    "2021-09-08",
]

CHECK_DATE = "2021-10-01"

WINDOW_START = "2021-09-15"
WINDOW_END = "2021-10-01"


def load_data(start_date):
    print()
    print("=" * 120)
    print(f"DATA START: {start_date}")
    print("=" * 120)

    data = yf.download(
        SYMBOL,
        start=start_date,
        end="2021-10-02",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=15,
    )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.sort_index()

    print(
        f"Rows: {len(data)} | "
        f"First: {data.index[0].date()} | "
        f"Last: {data.index[-1].date()}"
    )

    return data


def print_price_window(data):
    window = data.loc[
        (data.index >= WINDOW_START)
        & (data.index <= WINDOW_END)
    ][["Open", "High", "Low", "Close"]].copy()

    print()
    print("PRICE WINDOW")
    print("-" * 120)

    for index, row in window.iterrows():
        print(
            f"{index.date()} "
            f"O={float(row['Open']):10.2f} "
            f"H={float(row['High']):10.2f} "
            f"L={float(row['Low']):10.2f} "
            f"C={float(row['Close']):10.2f}"
        )


def print_structure(start_date, data):
    history = data.loc[:CHECK_DATE].copy()

    result = calculate_market_structure(
        history,
        swing_window=3,
    )

    print()
    print("STRUCTURE RESULT AT 2021-10-01")
    print("-" * 120)

    fields = [
        "bias",
        "structure",
        "last_swing_high",
        "last_swing_low",
        "bos",
        "choch",
        "break_direction",
        "break_level",
        "break_date",
        "break_confirmed",
        "break_age",
        "strength",
    ]

    for field in fields:
        print(
            f"{field:<22}: {result.get(field)}"
        )

    print()
    print("SWING HIGHS")
    print("-" * 120)

    for swing in result.get("swing_highs", []):
        index = swing["index"]

        if (
            str(index.date()) >= WINDOW_START
            and str(index.date()) <= CHECK_DATE
        ):
            print(
                f"{index.date()} "
                f"price={swing['price']:.2f}"
            )

    print()
    print("SWING LOWS")
    print("-" * 120)

    for swing in result.get("swing_lows", []):
        index = swing["index"]

        if (
            str(index.date()) >= WINDOW_START
            and str(index.date()) <= CHECK_DATE
        ):
            print(
                f"{index.date()} "
                f"price={swing['price']:.2f}"
            )

    print()
    print("LAST 5 SWING HIGHS")
    print("-" * 120)

    for swing in result.get("swing_highs", [])[-5:]:
        print(
            f"{swing['index'].date()} "
            f"price={swing['price']:.2f}"
        )

    print()
    print("LAST 5 SWING LOWS")
    print("-" * 120)

    for swing in result.get("swing_lows", [])[-5:]:
        print(
            f"{swing['index'].date()} "
            f"price={swing['price']:.2f}"
        )

    return result


def compare_results(results):
    print()
    print("=" * 120)
    print("DIRECT SWING / STRUCTURE COMPARISON")
    print("=" * 120)

    fields = [
        "bias",
        "structure",
        "last_swing_high",
        "last_swing_low",
        "bos",
        "choch",
        "break_direction",
        "break_level",
        "break_date",
        "break_confirmed",
        "break_age",
        "strength",
    ]

    rows = []

    for start_date, result in results.items():
        row = {
            "data_start": start_date,
        }

        for field in fields:
            row[field] = result.get(field)

        row["swing_high_count"] = len(
            result.get("swing_highs", [])
        )

        row["swing_low_count"] = len(
            result.get("swing_lows", [])
        )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    print(comparison.to_string(index=False))


def main():
    results = {}

    for start_date in DATA_STARTS:
        data = load_data(start_date)

        print_price_window(data)

        result = print_structure(
            start_date,
            data,
        )

        results[start_date] = result

    compare_results(results)

    print()
    print("=" * 120)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 120)
    print()
    print("NO PRODUCTION CODE WAS MODIFIED.")
    print("NO STRATEGY RULE WAS MODIFIED.")
    print("NO FROZEN 5B_0.25R RULE WAS MODIFIED.")


if __name__ == "__main__":
    main()