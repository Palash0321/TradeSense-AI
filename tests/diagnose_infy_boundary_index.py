import pandas as pd
import yfinance as yf


SYMBOL = "INFY.NS"

DATA_STARTS = [
    "2021-09-01",
    "2021-09-07",
    "2021-09-08",
]

END_DATE = "2021-10-02"
CANDIDATE_DATE = pd.Timestamp("2021-09-13")


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def main():
    print("=" * 110)
    print("INFY BOUNDARY INDEX DIAGNOSTIC")
    print("=" * 110)
    print()
    print("This script ONLY inspects dataframe positional indexing.")
    print("It does NOT modify production code.")
    print("It does NOT modify the trading strategy.")
    print("It does NOT modify the frozen 5B_0.25R rule.")
    print()

    for start in DATA_STARTS:
        print("=" * 110)
        print(f"DATA START = {start}")
        print("=" * 110)

        data = yf.download(
            SYMBOL,
            start=start,
            end=END_DATE,
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=15,
        )

        data = flatten_columns(data)
        data.index = pd.to_datetime(data.index)

        print(f"Rows: {len(data)}")
        print(f"First index: {data.index[0]}")
        print(f"Last index : {data.index[-1]}")
        print()

        if CANDIDATE_DATE not in data.index:
            print(f"{CANDIDATE_DATE.date()} NOT FOUND")
            print()
            continue

        candidate_pos = data.index.get_loc(CANDIDATE_DATE)

        print(f"Candidate date: {CANDIDATE_DATE.date()}")
        print(f"Candidate positional index: {candidate_pos}")
        print()

        print("INDEXES AROUND CANDIDATE")
        print("-" * 110)

        start_pos = max(0, candidate_pos - 5)
        end_pos = min(len(data), candidate_pos + 6)

        for pos in range(start_pos, end_pos):
            marker = "  <-- CANDIDATE" if pos == candidate_pos else ""

            row = data.iloc[pos]

            print(
                f"position={pos:>3} | "
                f"date={data.index[pos].date()} | "
                f"H={float(row['High']):.2f} | "
                f"L={float(row['Low']):.2f}"
                f"{marker}"
            )

        print()

        print("3-LEFT / 3-RIGHT POSITION CHECK")
        print("-" * 110)

        left_positions = [
            candidate_pos - 3,
            candidate_pos - 2,
            candidate_pos - 1,
        ]

        right_positions = [
            candidate_pos + 1,
            candidate_pos + 2,
            candidate_pos + 3,
        ]

        print("LEFT POSITIONS:")

        for pos in left_positions:
            if pos < 0 or pos >= len(data):
                print(f"  position={pos}: OUT OF RANGE")
            else:
                print(
                    f"  position={pos} | "
                    f"date={data.index[pos].date()} | "
                    f"Low={float(data.iloc[pos]['Low']):.2f}"
                )

        print()

        print("RIGHT POSITIONS:")

        for pos in right_positions:
            if pos < 0 or pos >= len(data):
                print(f"  position={pos}: OUT OF RANGE")
            else:
                print(
                    f"  position={pos} | "
                    f"date={data.index[pos].date()} | "
                    f"Low={float(data.iloc[pos]['Low']):.2f}"
                )

        print()

        print("LEFT COUNT:", len(left_positions))
        print("RIGHT COUNT:", len(right_positions))

        left_available = all(
            0 <= pos < len(data)
            for pos in left_positions
        )

        right_available = all(
            0 <= pos < len(data)
            for pos in right_positions
        )

        print("3 LEFT AVAILABLE :", left_available)
        print("3 RIGHT AVAILABLE:", right_available)

        print()

        candidate_low = float(data.loc[CANDIDATE_DATE, "Low"])

        left_lows = [
            float(data.iloc[pos]["Low"])
            for pos in left_positions
            if 0 <= pos < len(data)
        ]

        right_lows = [
            float(data.iloc[pos]["Low"])
            for pos in right_positions
            if 0 <= pos < len(data)
        ]

        print("MANUAL SWING-LOW CALCULATION")
        print("-" * 110)

        print(f"Candidate Low: {candidate_low:.2f}")

        if len(left_lows) == 3:
            left_min = min(left_lows)
            print(f"Left Min Low: {left_min:.2f}")
            print(f"Candidate < Left Min: {candidate_low < left_min}")
        else:
            print(f"Left Min Low: NOT AVAILABLE ({len(left_lows)}/3)")

        if len(right_lows) == 3:
            right_min = min(right_lows)
            print(f"Right Min Low: {right_min:.2f}")
            print(f"Candidate < Right Min: {candidate_low < right_min}")
        else:
            print(f"Right Min Low: NOT AVAILABLE ({len(right_lows)}/3)")

        if len(left_lows) == 3 and len(right_lows) == 3:
            swing_low = (
                candidate_low < min(left_lows)
                and candidate_low < min(right_lows)
            )
            print(f"FINAL MANUAL SWING LOW: {swing_low}")
        else:
            print("FINAL MANUAL SWING LOW: CANNOT TEST")

        print()

    print("=" * 110)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()