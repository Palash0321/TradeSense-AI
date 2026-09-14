import os
import time

import pandas as pd
import yfinance as yf


OUTPUT_DIR = "data/nifty50_prices"

SPECIAL_YAHOO = {
    "ZOMATO": {
        "source_symbol": "ETERNAL.NS",
        "start": "2021-01-29",
        "end": "2026-09-11",
    },
}


def download_yahoo_source(normalized_symbol, source_symbol, start, end):
    print("=" * 70)
    print(f"DOWNLOADING {normalized_symbol}")
    print("=" * 70)
    print(f"Normalized symbol : {normalized_symbol}")
    print(f"Yahoo source      : {source_symbol}")
    print(f"Requested period  : {start} -> {end}")
    print()

    data = yf.download(
        source_symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        timeout=30,
    )

    if data.empty:
        raise ValueError(
            f"{source_symbol}: Yahoo returned no data."
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{source_symbol}: Missing columns: {sorted(missing)}"
        )

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index = data.index.normalize()

    data = data.reset_index()

    if "Date" not in data.columns:
        raise ValueError(
            f"{source_symbol}: Date column missing."
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    output = pd.DataFrame(
        {
            "date": data["Date"],
            "open": data["Open"],
            "high": data["High"],
            "low": data["Low"],
            "close": data["Close"],
            "adj_close": data["Adj Close"],
            "volume": data["Volume"],
            "dividends": data["Dividends"],
            "stock_splits": data["Stock Splits"],
        }
    )

    output = output.sort_values("date")
    output = output.drop_duplicates(
        subset=["date"],
        keep="last",
    )

    if output.empty:
        raise ValueError(
            f"{normalized_symbol}: No rows after normalization."
        )

    if output[
        ["open", "high", "low", "close"]
    ].isna().any().any():
        raise ValueError(
            f"{normalized_symbol}: Missing OHLC values."
        )

    if (
        output[
            ["open", "high", "low", "close"]
        ] <= 0
    ).any().any():
        raise ValueError(
            f"{normalized_symbol}: Non-positive OHLC value."
        )

    if (output["volume"] < 0).any():
        raise ValueError(
            f"{normalized_symbol}: Negative volume."
        )

    first_date = output["date"].min()
    last_date = output["date"].max()

    print(f"Rows              : {len(output)}")
    print(f"Actual coverage   : {first_date} -> {last_date}")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{normalized_symbol}.csv",
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved             : {output_file}")
    print()
    print("STATUS: PASS")


def main():
    for normalized_symbol, config in SPECIAL_YAHOO.items():
        download_yahoo_source(
            normalized_symbol=normalized_symbol,
            source_symbol=config["source_symbol"],
            start=config["start"],
            end=config["end"],
        )
        time.sleep(1)


if __name__ == "__main__":
    main()