import pandas as pd
import yfinance as yf


SYMBOL = "INFY.NS"

DATA_STARTS = [
    "2021-09-01",
    "2021-09-07",
    "2021-09-08",
]

CHECK_DATES = [
    "2021-09-07",
    "2021-09-13",
    "2021-09-20",
    "2021-09-24",
]

SWING_WINDOW = 3


def load_data(start_date):
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

    return data


def print_candle(data, date):
    if date not in data.index:
        print(
            f"    {date}: NOT AVAILABLE"
        )
        return

    row = data.loc[date]

    print(
        f"    {date} | "
        f"H={float(row['High']):.2f} "
        f"L={float(row['Low']):.2f} "
        f"C={float(row['Close']):.2f}"
    )


def inspect_candidate(data, candidate_date):
    print()
    print("-" * 120)
    print(f"CANDIDATE: {candidate_date}")
    print("-" * 120)

    if candidate_date not in data.index:
        print("Candidate candle is NOT available.")
        return

    candidate_position = data.index.get_loc(candidate_date)

    candidate_high = float(
        data.loc[candidate_date, "High"]
    )

    candidate_low = float(
        data.loc[candidate_date, "Low"]
    )

    print(
        f"Candidate High = {candidate_high:.2f}"
    )

    print(
        f"Candidate Low  = {candidate_low:.2f}"
    )

    print()

    # ----------------------------------------------------------
    # LEFT SIDE
    # ----------------------------------------------------------

    left_start = max(
        0,
        candidate_position - SWING_WINDOW
    )

    left_end = candidate_position

    left_data = data.iloc[
        left_start:left_end
    ]

    # ----------------------------------------------------------
    # RIGHT SIDE
    # ----------------------------------------------------------

    right_start = candidate_position + 1

    right_end = (
        candidate_position
        + SWING_WINDOW
        + 1
    )

    right_data = data.iloc[
        right_start:right_end
    ]

    print("LEFT-SIDE CANDLES")
    print()

    if len(left_data) < SWING_WINDOW:
        print(
            f"    NOT ENOUGH DATA "
            f"({len(left_data)}/{SWING_WINDOW})"
        )
    else:
        for index, row in left_data.iterrows():
            print(
                f"    {index.date()} | "
                f"H={float(row['High']):.2f} "
                f"L={float(row['Low']):.2f}"
            )

    print()

    print("RIGHT-SIDE CANDLES")
    print()

    if len(right_data) < SWING_WINDOW:
        print(
            f"    NOT ENOUGH DATA "
            f"({len(right_data)}/{SWING_WINDOW})"
        )
    else:
        for index, row in right_data.iterrows():
            print(
                f"    {index.date()} | "
                f"H={float(row['High']):.2f} "
                f"L={float(row['Low']):.2f}"
            )

    print()

    # ----------------------------------------------------------
    # SWING HIGH TEST
    # ----------------------------------------------------------

    print("SWING HIGH TEST")
    print()

    if len(left_data) < SWING_WINDOW:
        print(
            "    FAIL: insufficient left-side candles"
        )

        swing_high = False

    elif len(right_data) < SWING_WINDOW:
        print(
            "    FAIL: insufficient right-side candles"
        )

        swing_high = False

    else:
        left_max = float(
            left_data["High"].max()
        )

        right_max = float(
            right_data["High"].max()
        )

        left_condition = (
            candidate_high > left_max
        )

        right_condition = (
            candidate_high > right_max
        )

        print(
            f"    Candidate High : {candidate_high:.2f}"
        )

        print(
            f"    Left Max High  : {left_max:.2f}"
        )

        print(
            f"    Right Max High : {right_max:.2f}"
        )

        print()

        print(
            f"    Candidate > Left Max  : "
            f"{left_condition}"
        )

        print(
            f"    Candidate > Right Max : "
            f"{right_condition}"
        )

        swing_high = (
            left_condition
            and right_condition
        )

        print()

        print(
            f"    FINAL SWING HIGH: "
            f"{swing_high}"
        )

    # ----------------------------------------------------------
    # SWING LOW TEST
    # ----------------------------------------------------------

    print()
    print("SWING LOW TEST")
    print()

    if len(left_data) < SWING_WINDOW:
        print(
            "    FAIL: insufficient left-side candles"
        )

        swing_low = False

    elif len(right_data) < SWING_WINDOW:
        print(
            "    FAIL: insufficient right-side candles"
        )

        swing_low = False

    else:
        left_min = float(
            left_data["Low"].min()
        )

        right_min = float(
            right_data["Low"].min()
        )

        left_condition = (
            candidate_low < left_min
        )

        right_condition = (
            candidate_low < right_min
        )

        print(
            f"    Candidate Low  : {candidate_low:.2f}"
        )

        print(
            f"    Left Min Low   : {left_min:.2f}"
        )

        print(
            f"    Right Min Low  : {right_min:.2f}"
        )

        print()

        print(
            f"    Candidate < Left Min  : "
            f"{left_condition}"
        )

        print(
            f"    Candidate < Right Min : "
            f"{right_condition}"
        )

        swing_low = (
            left_condition
            and right_condition
        )

        print()

        print(
            f"    FINAL SWING LOW: "
            f"{swing_low}"
        )


def inspect_start(start_date):
    print()
    print("=" * 120)
    print(
        f"DATA START = {start_date}"
    )
    print("=" * 120)

    data = load_data(start_date)

    print(
        f"Rows: {len(data)} | "
        f"First: {data.index[0].date()} | "
        f"Last: {data.index[-1].date()}"
    )

    for candidate_date in CHECK_DATES:
        inspect_candidate(
            data,
            candidate_date
        )


def main():

    print()
    print("=" * 120)
    print("INFY SWING-CONDITION DIAGNOSTIC")
    print("=" * 120)

    print()
    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Swing window: {SWING_WINDOW}"
    )

    print()
    print(
        "This script ONLY diagnoses swing conditions."
    )

    print(
        "It does NOT modify production code."
    )

    print(
        "It does NOT modify the trading strategy."
    )

    print(
        "It does NOT modify the frozen 5B_0.25R rule."
    )

    for start_date in DATA_STARTS:
        inspect_start(start_date)

    print()
    print("=" * 120)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()