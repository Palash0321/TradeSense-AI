from datetime import datetime

import yfinance as yf

from app.core.levels.market_structure import calculate_market_structure


SYMBOL = "INFY.NS"

# We intentionally keep the same end date so every test sees
# exactly the same market endpoint.
END_DATE = "2021-10-15"

# Test progressively earlier starting points.
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

    # Handle yfinance MultiIndex columns.
    if hasattr(data.columns, "levels"):
        if len(data.columns.levels) > 1:
            data.columns = data.columns.get_level_values(0)

    data = data.copy()
    data.index = data.index.normalize()

    return data


def get_structure_at_date(df, target_date):
    history = df.loc[df.index <= target_date].copy()

    if history.empty:
        return None

    return calculate_market_structure(
        history,
        swing_window=3,
        break_validity_bars=5,
    )


def print_swing_list(title, swings):
    print(title)

    for swing in swings[-10:]:
        print(
            f"  {swing['index'].date()} -> "
            f"{swing['price']:.2f}"
        )


def main():
    target_date = datetime(2021, 10, 1)

    print("=" * 90)
    print("TRADE SENSE-AI — MARKET STRUCTURE WARM-UP INVARIANCE TEST")
    print("=" * 90)
    print(f"Symbol: {SYMBOL}")
    print(f"Structure target date: {target_date.date()}")
    print(f"Swing window: 3")
    print(f"End date: {END_DATE}")
    print()

    results = []

    for start_date in START_DATES:
        print("=" * 90)
        print(f"START DATE: {start_date}")
        print("=" * 90)

        df = download_data(start_date)

        if df.empty:
            print("NO DATA")
            continue

        structure = get_structure_at_date(
            df,
            target_date,
        )

        if structure is None:
            print("NO STRUCTURE")
            continue

        print(f"Rows through target: {len(df.loc[df.index <= target_date])}")
        print(f"First available date: {df.index.min().date()}")
        print(f"Last available date: {df.index.max().date()}")
        print()

        print("STRUCTURE:")
        print(f"  Bias: {structure.get('bias')}")
        print(f"  Structure: {structure.get('structure')}")
        print(f"  BOS: {structure.get('bos')}")
        print(f"  CHOCH: {structure.get('choch')}")
        print(f"  Break direction: {structure.get('break_direction')}")
        print(f"  Break level: {structure.get('break_level')}")
        print(f"  Break date: {structure.get('break_date')}")
        print(f"  Break confirmed: {structure.get('break_confirmed')}")
        print(f"  Break age: {structure.get('break_age')}")

        print()

        print_swing_list(
            "LAST SWING HIGHS:",
            structure.get("swing_highs", []),
        )

        print_swing_list(
            "LAST SWING LOWS:",
            structure.get("swing_lows", []),
        )

        results.append(
            {
                "start_date": start_date,
                "bias": structure.get("bias"),
                "structure": structure.get("structure"),
                "bos": structure.get("bos"),
                "choch": structure.get("choch"),
                "break_direction": structure.get("break_direction"),
                "break_level": structure.get("break_level"),
                "break_confirmed": structure.get("break_confirmed"),
                "break_age": structure.get("break_age"),
                "swing_high_count": len(
                    structure.get("swing_highs", [])
                ),
                "swing_low_count": len(
                    structure.get("swing_lows", [])
                ),
            }
        )

    print()
    print("=" * 90)
    print("COMPACT COMPARISON")
    print("=" * 90)

    print(
        f"{'START':<12}"
        f"{'BIAS':<12}"
        f"{'STRUCTURE':<20}"
        f"{'CHOCH':<12}"
        f"{'BREAK':<12}"
        f"{'LOWS':<8}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['start_date']:<12}"
            f"{str(result['bias']):<12}"
            f"{str(result['structure']):<20}"
            f"{str(result['choch']):<12}"
            f"{str(result['break_direction']):<12}"
            f"{result['swing_low_count']:<8}"
        )


if __name__ == "__main__":
    main()