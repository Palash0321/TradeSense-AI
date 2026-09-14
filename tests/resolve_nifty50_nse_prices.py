import io
import os
import time

import pandas as pd
import requests


OUTPUT_DIR = "data/nifty50_prices"

SPECIAL_NSE = {
    "HDFC": {
        "source_symbol": "HDFC",
        "start": "01-01-2021",
        "end": "12-07-2023",
    },
    "LTIM": {
        "source_symbol": "LTIM",
        "start": "13-07-2023",
        "end": "27-09-2024",
    },
    "TATAMOTORS": {
        "source_symbol": "TATAMOTORS",
        "start": "01-01-2021",
        "end": "23-10-2025",
    },
}

NSE_PAGE = "https://www.nseindia.com/report-detail/eq_security"

NSE_API = (
    "https://www.nseindia.com/"
    "api/historicalOR/generateSecurityWiseHistoricalData"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,*/*;q=0.8"
    ),
    "Referer": NSE_PAGE,
}


def create_session():
    session = requests.Session()

    response = session.get(
        NSE_PAGE,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    return session


def parse_number(value):
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if not text or text == "-":
        return float("nan")

    text = (
        text.replace(",", "")
        .replace("₹", "")
        .replace("â‚¹", "")
        .strip()
    )

    return float(text)


def normalize_nse_data(symbol, response):
    if not response.content:
        raise ValueError(
            f"{symbol}: NSE returned empty response."
        )

    try:
        data = pd.read_csv(
            io.BytesIO(response.content),
            encoding="utf-8-sig",
        )
    except Exception as error:
        raise ValueError(
            f"{symbol}: Could not parse NSE CSV: {error}"
        ) from error

    data.columns = [
        str(column).strip()
        for column in data.columns
    ]

    required_columns = {
        "Symbol",
        "Series",
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{symbol}: Missing NSE columns: {sorted(missing)}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        format="%d-%b-%Y",
        errors="raise",
    )

    data["Series"] = (
        data["Series"]
        .astype(str)
        .str.strip()
    )

    data = data[
        data["Series"] == "EQ"
    ].copy()

    if data.empty:
        raise ValueError(
            f"{symbol}: No EQ-series rows returned."
        )

    for column in [
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    ]:
        data[column] = data[column].apply(
            parse_number
        )

    data = data.sort_values("Date")

    data = data.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    if data[
        [
            "Open Price",
            "High Price",
            "Low Price",
            "Close Price",
        ]
    ].isna().any().any():
        raise ValueError(
            f"{symbol}: Missing OHLC values detected."
        )

    if (
        data[
            [
                "Open Price",
                "High Price",
                "Low Price",
                "Close Price",
            ]
        ] <= 0
    ).any().any():
        raise ValueError(
            f"{symbol}: Non-positive OHLC value detected."
        )

    if (
        data["Total Traded Quantity"] < 0
    ).any():
        raise ValueError(
            f"{symbol}: Negative volume detected."
        )

    output = pd.DataFrame(
        {
            "date": data["Date"].dt.strftime(
                "%Y-%m-%d"
            ),
            "open": data["Open Price"],
            "high": data["High Price"],
            "low": data["Low Price"],
            "close": data["Close Price"],
            "adj_close": float("nan"),
            "volume": data[
                "Total Traded Quantity"
            ],
            "dividends": float("nan"),
            "stock_splits": float("nan"),
        }
    )

    return output


def download_symbol(
    normalized_symbol,
    source_symbol,
    start,
    end,
):
    print("=" * 70)
    print(
        f"DOWNLOADING {normalized_symbol}"
    )
    print("=" * 70)
    print(
        f"Normalized symbol : {normalized_symbol}"
    )
    print(
        f"NSE source        : {source_symbol}"
    )
    print(
        f"Requested period  : {start} -> {end}"
    )
    print()

    session = create_session()

    params = {
        "from": start,
        "to": end,
        "symbol": source_symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    response = session.get(
        NSE_API,
        params=params,
        headers=HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if "csv" not in content_type:
        raise ValueError(
            f"{normalized_symbol}: Unexpected "
            f"content type: {content_type}"
        )

    output = normalize_nse_data(
        normalized_symbol,
        response,
    )

    if output.empty:
        raise ValueError(
            f"{normalized_symbol}: No rows."
        )

    first_date = output["date"].min()
    last_date = output["date"].max()

    print(
        f"Rows              : {len(output)}"
    )
    print(
        f"Actual coverage   : "
        f"{first_date} -> {last_date}"
    )

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

    print(
        f"Saved             : {output_file}"
    )
    print()
    print("STATUS: PASS")


def main():
    print("=" * 70)
    print(
        "NIFTY 50 NSE SPECIAL-SECURITY "
        "PRICE RESOLVER"
    )
    print("=" * 70)

    for normalized_symbol, config in SPECIAL_NSE.items():
        try:
            download_symbol(
                normalized_symbol=normalized_symbol,
                source_symbol=config[
                    "source_symbol"
                ],
                start=config["start"],
                end=config["end"],
            )
        except Exception as error:
            print()
            print(
                f"[{normalized_symbol}] FAILED: "
                f"{error}"
            )
            raise

        time.sleep(1)

    print()
    print("=" * 70)
    print("ALL NSE SPECIAL SECURITIES COMPLETE")
    print("=" * 70)
    print("STATUS: PASS")


if __name__ == "__main__":
    main()