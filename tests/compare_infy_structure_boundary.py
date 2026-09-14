import yfinance as yf
from app.core.levels.market_structure import calculate_market_structure


def get_structure(start_date):
    df = yf.download(
        "INFY.NS",
        start=start_date,
        end="2021-10-02",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
        timeout=15,
    )

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    structure = calculate_market_structure(
        df,
        swing_window=3
    )

    return df, structure


for start_date in ["2021-09-06", "2021-09-07", "2021-09-08"]:

    df, structure = get_structure(start_date)

    print("\n" + "=" * 80)
    print(f"START DATE: {start_date}")
    print("=" * 80)

    print("Rows:", len(df))
    print("First:", df.index[0])
    print("Last:", df.index[-1])

    print("\nSTRUCTURE:")
    print("Bias:", structure["bias"])
    print("Structure:", structure["structure"])
    print("BOS:", structure["bos"])
    print("CHOCH:", structure["choch"])
    print("Break direction:", structure["break_direction"])
    print("Break level:", structure["break_level"])
    print("Break date:", structure["break_date"])
    print("Break confirmed:", structure["break_confirmed"])
    print("Break age:", structure["break_age"])

    print("\nLAST 10 SWING HIGHS:")
    for x in structure["swing_highs"][-10:]:
        print(x)

    print("\nLAST 10 SWING LOWS:")
    for x in structure["swing_lows"][-10:]:
        print(x)
