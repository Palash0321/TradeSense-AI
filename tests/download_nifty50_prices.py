import csv
import io
import os
import time
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf


MEMBERSHIP_FILE = "data/nifty50_constituent_membership.csv"
OUTPUT_DIR = "data/nifty50_prices"

START_DATE = "2021-01-29"
END_DATE = "2026-09-11"  # yfinance end date is exclusive

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 3

NSE_HISTORY_URL = (
    "https://www.nseindia.com/api/"
    "historicalOR/generateSecurityWiseHistoricalData"
)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/json",
    "Referer": "https://www.nseindia.com/report-detail/eq_security",
}

# Yahoo Finance acquisition ticker overrides.
# Keep historical NIFTY symbols unchanged in the research dataset.
# Historical Yahoo ticker mappings.
# The normalized research symbol remains the NIFTY membership symbol.
YAHOO_TICKER_OVERRIDES = {
    "ZOMATO": "ETERNAL",
}

# Symbols that are part of historical NIFTY 50 index membership
# but are not ordinary tradable securities for which we should
# download OHLCV data.
#
# DUMMYTATAM represented the demerged commercial-vehicle entity
# during a temporary index-adjustment period.
SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM",
}
SPECIAL_NO_PRICE_PERIODS = {
    "JIOFIN": [
        (
            pd.Timestamp("2023-07-20"),
            pd.Timestamp("2023-08-18"),
        ),
    ],
}

# Symbols whose Yahoo ticker changed during the requested history.
# Each tuple is:
# (Yahoo ticker, start_date, end_date)
HISTORICAL_TICKER_RANGES = {
    "LTIM": [
        ("LTIM", "2021-01-29", "2026-02-27"),
        ("LTM", "2026-02-27", "2026-09-11"),
    ],
}
DIRECT_NSE_SYMBOLS = {
    "HDFC",
    "LTIM",
    "TATAMOTORS",
}

DIRECT_NSE_DATE_RANGES = {
    "HDFC": (
        "2021-01-29",
        "2023-07-12",
    ),
    "LTIM": (
        "2023-07-13",
        "2024-09-27",
    ),
    "TATAMOTORS": (
        "2021-01-29",
        "2025-10-23",
    ),
}


def load_membership():
    if not os.path.exists(MEMBERSHIP_FILE):
        raise FileNotFoundError(MEMBERSHIP_FILE)

    membership = pd.read_csv(MEMBERSHIP_FILE)

    required_columns = {
        "date",
        "symbol",
        "is_constituent",
    }

    missing = required_columns - set(membership.columns)

    if missing:
        raise ValueError(
            f"Membership file missing columns: {sorted(missing)}"
        )

    membership["date"] = pd.to_datetime(
        membership["date"],
        errors="raise",
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
    )

    membership["is_constituent"] = (
        membership["is_constituent"]
        .astype(int)
    )

    return membership


def get_symbols(membership):
    symbols = sorted(
        membership.loc[
            membership["is_constituent"] == 1,
            "symbol",
        ].unique()
    )

    if not symbols:
        raise ValueError("No NIFTY 50 constituents found.")

    return symbols


def get_price_symbols(symbols):
    price_symbols = [
        symbol
        for symbol in symbols
        if symbol not in SPECIAL_INDEX_ENTITIES
    ]

    if not price_symbols:
        raise ValueError(
            "No ordinary securities found for price download."
        )

    return price_symbols


def validate_price_data(symbol, data):
    if data.empty:
        raise ValueError(
            f"{symbol}: Yahoo returned no price data."
        )

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"{symbol}: Missing columns: {sorted(missing)}"
        )

    data = data.copy()

    # Normalize timezone-aware Yahoo dates to calendar dates.
    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data.index = data.index.normalize()

    # Duplicate dates
    if data.index.duplicated().any():
        duplicates = data.index[data.index.duplicated()].tolist()

        raise ValueError(
            f"{symbol}: Duplicate dates found: {duplicates[:5]}"
        )

    # Required OHLC values must not be missing.
    ohlc = ["Open", "High", "Low", "Close"]

    missing_ohlc = data[ohlc].isna().any(axis=1)

    if missing_ohlc.any():
        bad_dates = data.index[missing_ohlc].tolist()

        raise ValueError(
            f"{symbol}: Missing OHLC values on "
            f"{bad_dates[:5]}"
        )

    # Basic OHLC sanity checks.
    invalid_high = (
        data["High"]
        < data[["Open", "Close", "Low"]].max(axis=1)
    )

    invalid_low = (
        data["Low"]
        > data[["Open", "High", "Close"]].min(axis=1)
    )

    if invalid_high.any():
        raise ValueError(
            f"{symbol}: Invalid High values detected."
        )

    if invalid_low.any():
        raise ValueError(
            f"{symbol}: Invalid Low values detected."
        )

    # Prices must be positive.
    if (
        data[ohlc] <= 0
    ).any().any():
        raise ValueError(
            f"{symbol}: Non-positive OHLC value detected."
        )

    # Volume should not be negative.
    if (data["Volume"] < 0).any():
        raise ValueError(
            f"{symbol}: Negative volume detected."
        )

    return data


def repair_missing_required_fields_from_nse(symbol, data):
    """
    Repair incomplete or missing Yahoo OHLCV observations using
    complete NSE security-wise historical price/volume rows.

    Yahoo remains the primary source.

    If a Yahoo row has one or more missing required OHLCV fields,
    the COMPLETE OHLCV row is replaced with the corresponding NSE
    observation.

    This avoids creating synthetic market bars by mixing Yahoo and
    NSE values from different observations.

    NSE does not provide Yahoo's adjusted-close series, so Adj Close
    is set to NaN for rows replaced from NSE.
    """

    data = data.copy()

    required_fields = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    # Find rows where at least one required OHLCV field is missing.
    missing_mask = data[required_fields].isna().any(axis=1)

    if not missing_mask.any():
        return data

    missing_dates = data.index[missing_mask]

    first_missing = missing_dates.min()
    last_missing = missing_dates.max()

    from_date = first_missing.strftime("%d-%m-%Y")
    to_date = last_missing.strftime("%d-%m-%Y")

    print(
        f"[{symbol}] NSE FALLBACK: "
        f"replacing incomplete/missing OHLCV rows "
        f"from {from_date} -> {to_date}"
    )

    params = {
        "from": from_date,
        "to": to_date,
        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    try:
        response = requests.get(
            NSE_HISTORY_URL,
            params=params,
            headers=NSE_HEADERS,
            timeout=30,
        )

        response.raise_for_status()

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: NSE fallback request failed: {error}"
        ) from error

    if not response.text.strip():
        raise RuntimeError(
            f"{symbol}: NSE fallback returned empty response."
        )

    try:
        rows = list(
            csv.reader(
                io.StringIO(
                    response.text.lstrip("\ufeff")
                )
            )
        )

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: Failed to parse NSE response: {error}"
        ) from error

    if len(rows) < 2:
        raise RuntimeError(
            f"{symbol}: NSE fallback returned no data rows."
        )

    header = [
        str(column).strip()
        for column in rows[0]
    ]

    header_map = {
        name: index
        for index, name in enumerate(header)
    }

    required_nse_columns = {
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    missing_nse_columns = (
        required_nse_columns - set(header_map)
    )

    if missing_nse_columns:
        raise RuntimeError(
            f"{symbol}: NSE response missing columns: "
            f"{sorted(missing_nse_columns)}"
        )

    nse_data = {}

    for row in rows[1:]:
        if not row:
            continue

        try:
            raw_date = row[header_map["Date"]].strip()

            date = pd.to_datetime(
                raw_date,
                format="%d-%b-%Y",
            ).normalize()

            def parse_number(value):
                value = str(value).strip()

                if not value:
                    return None

                return float(
                    value.replace(",", "")
                )

            nse_data[date] = {
                "Open": parse_number(
                    row[header_map["Open Price"]]
                ),
                "High": parse_number(
                    row[header_map["High Price"]]
                ),
                "Low": parse_number(
                    row[header_map["Low Price"]]
                ),
                "Close": parse_number(
                    row[header_map["Close Price"]]
                ),
                "Volume": parse_number(
                    row[header_map["Total Traded Quantity"]]
                ),
            }

        except Exception as error:
            raise RuntimeError(
                f"{symbol}: Failed to parse NSE row: "
                f"{row}. Error: {error}"
            ) from error

    replaced_rows = []

    for date in missing_dates:
        if date not in nse_data:
            raise RuntimeError(
                f"{symbol}: NSE has no observation for "
                f"missing Yahoo date "
                f"{date.strftime('%Y-%m-%d')}."
            )

        nse_row = nse_data[date]

        # --------------------------------------------------------------
        # Replace the COMPLETE OHLCV observation.
        #
        # We deliberately do NOT mix Yahoo and NSE values within
        # the same market bar.
        # --------------------------------------------------------------

        for field in required_fields:
            nse_value = nse_row.get(field)

            if nse_value is None or pd.isna(nse_value):
                raise RuntimeError(
                    f"{symbol}: NSE missing {field} for "
                    f"{date.strftime('%Y-%m-%d')}."
                )

            data.at[date, field] = nse_value

        # NSE security-wise price/volume data does not provide
        # Yahoo's adjusted-close series.
        data.at[date, "Adj Close"] = float("nan")

        replaced_rows.append(
            date.strftime("%Y-%m-%d")
        )

    if replaced_rows:
        print(
            f"[{symbol}] NSE FALLBACK REPLACED "
            f"{len(replaced_rows)} complete OHLCV row(s):"
        )

        for date in replaced_rows:
            print(
                f"    {date}: Open/High/Low/Close/Volume "
                f"replaced from NSE"
            )

    return data

def repair_missing_dates_from_nse(symbol, data, membership):
    """
    Repair completely absent Yahoo trading dates using NSE.

    Yahoo remains the primary source.

    A date is considered missing only when:
    1. It is a genuine NIFTY trading date,
    2. The symbol was a NIFTY 50 constituent on that date,
    3. The date is not an explicitly documented special
       no-price period, and
    4. Yahoo has no row for that date.

    When a date is missing, the COMPLETE OHLCV observation
    is obtained from NSE.

    No synthetic bars are created and Yahoo/NSE values are
    never mixed within the same observation.
    """

    data = data.copy()

    if data.empty:
        return data

    # --------------------------------------------------------------
    # Normalize Yahoo dates.
    # --------------------------------------------------------------
    if isinstance(data.index, pd.DatetimeIndex):
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data.index = data.index.normalize()

    # --------------------------------------------------------------
    # Determine the period during which this symbol was a
    # genuine NIFTY 50 constituent.
    # --------------------------------------------------------------
    symbol_membership = membership[
        (membership["symbol"] == symbol)
        & (membership["is_constituent"] == 1)
    ].copy()

    if symbol_membership.empty:
        return data

    membership_start = symbol_membership["date"].min()
    membership_end = symbol_membership["date"].max()

    # --------------------------------------------------------------
    # Download the NIFTY 50 trading calendar.
    #
    # The NIFTY index itself defines the actual market sessions
    # relevant to the point-in-time panel.
    # --------------------------------------------------------------
    print(
        f"[{symbol}] Checking Yahoo coverage against "
        f"the NIFTY 50 trading calendar..."
    )

    nifty_data = yf.download(
        "^NSEI",
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if nifty_data.empty:
        raise RuntimeError(
            f"{symbol}: Unable to download NIFTY 50 trading calendar."
        )

    if isinstance(nifty_data.columns, pd.MultiIndex):
        nifty_data.columns = (
            nifty_data.columns.get_level_values(0)
        )

    if not isinstance(nifty_data.index, pd.DatetimeIndex):
        raise RuntimeError(
            f"{symbol}: NIFTY 50 trading calendar has "
            f"an invalid index."
        )

    if nifty_data.index.tz is not None:
        nifty_data.index = nifty_data.index.tz_localize(None)

    nifty_dates = pd.DatetimeIndex(
        nifty_data.index.normalize()
    )

    # --------------------------------------------------------------
    # Restrict expected dates to the period where this symbol
    # was actually a NIFTY 50 constituent.
    # --------------------------------------------------------------
    expected_dates = nifty_dates[
        (nifty_dates >= membership_start)
        & (nifty_dates <= membership_end)
    ]

    # --------------------------------------------------------------
    # Remove explicitly documented periods where an ordinary
    # OHLCV observation must NOT exist.
    #
    # Example:
    # JIOFIN was temporarily included in the index before its
    # actual listing, so we must not fabricate ordinary prices.
    # --------------------------------------------------------------
    special_periods = SPECIAL_NO_PRICE_PERIODS.get(
        symbol,
        [],
    )

    if special_periods:
        keep_mask = pd.Series(
            True,
            index=expected_dates,
        )

        for period_start, period_end in special_periods:
            keep_mask &= ~(
                (expected_dates >= period_start)
                & (expected_dates <= period_end)
            )

        expected_dates = expected_dates[
            keep_mask.to_numpy()
        ]

    # --------------------------------------------------------------
    # Find dates that should have an ordinary price observation
    # but are completely absent from Yahoo.
    # --------------------------------------------------------------
    yahoo_dates = pd.DatetimeIndex(data.index)

    missing_dates = expected_dates.difference(
        yahoo_dates
    )

    if len(missing_dates) == 0:
        return data

    print(
        f"[{symbol}] NSE DATE FALLBACK: "
        f"{len(missing_dates)} completely missing "
        f"trading date(s)."
    )

    for date in missing_dates:
        print(
            f"    {date.strftime('%Y-%m-%d')}: "
            f"missing from Yahoo"
        )

    # --------------------------------------------------------------
    # Query NSE only for the missing-date range.
    # --------------------------------------------------------------
    first_missing = missing_dates.min()
    last_missing = missing_dates.max()

    from_date = first_missing.strftime("%d-%m-%Y")
    to_date = last_missing.strftime("%d-%m-%Y")

    params = {
        "from": from_date,
        "to": to_date,
        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    try:
        response = requests.get(
            NSE_HISTORY_URL,
            params=params,
            headers=NSE_HEADERS,
            timeout=30,
        )

        response.raise_for_status()

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: NSE missing-date request failed: "
            f"{error}"
        ) from error

    if not response.text.strip():
        raise RuntimeError(
            f"{symbol}: NSE missing-date response is empty."
        )

    try:
        rows = list(
            csv.reader(
                io.StringIO(
                    response.text.lstrip("\ufeff")
                )
            )
        )

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: Failed to parse NSE missing-date "
            f"response: {error}"
        ) from error

    if len(rows) < 2:
        raise RuntimeError(
            f"{symbol}: NSE returned no rows for missing dates."
        )

    header = [
        str(column).strip()
        for column in rows[0]
    ]

    header_map = {
        name: index
        for index, name in enumerate(header)
    }

    required_nse_columns = {
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    missing_nse_columns = (
        required_nse_columns - set(header_map)
    )

    if missing_nse_columns:
        raise RuntimeError(
            f"{symbol}: NSE missing-date response missing "
            f"columns: {sorted(missing_nse_columns)}"
        )

    nse_data = {}

    for row in rows[1:]:
        if not row:
            continue

        try:
            raw_date = row[
                header_map["Date"]
            ].strip()

            date = pd.to_datetime(
                raw_date,
                format="%d-%b-%Y",
            ).normalize()

            def parse_number(value):
                value = str(value).strip()

                if not value:
                    return None

                return float(
                    value.replace(",", "")
                )

            nse_data[date] = {
                "Open": parse_number(
                    row[
                        header_map["Open Price"]
                    ]
                ),
                "High": parse_number(
                    row[
                        header_map["High Price"]
                    ]
                ),
                "Low": parse_number(
                    row[
                        header_map["Low Price"]
                    ]
                ),
                "Close": parse_number(
                    row[
                        header_map["Close Price"]
                    ]
                ),
                "Volume": parse_number(
                    row[
                        header_map[
                            "Total Traded Quantity"
                        ]
                    ]
                ),
            }

        except Exception as error:
            raise RuntimeError(
                f"{symbol}: Failed to parse NSE missing-date "
                f"row: {row}. Error: {error}"
            ) from error

    # --------------------------------------------------------------
    # Insert COMPLETE NSE observations.
    # --------------------------------------------------------------
    inserted_rows = []

    required_fields = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for date in missing_dates:
        if date not in nse_data:
            raise RuntimeError(
                f"{symbol}: NSE has no observation for "
                f"missing Yahoo date "
                f"{date.strftime('%Y-%m-%d')}."
            )

        nse_row = nse_data[date]

        for field in required_fields:
            nse_value = nse_row.get(field)

            if (
                nse_value is None
                or pd.isna(nse_value)
            ):
                raise RuntimeError(
                    f"{symbol}: NSE missing {field} for "
                    f"{date.strftime('%Y-%m-%d')}."
                )

        # Build a complete new observation.
        new_row = {}

        for column in data.columns:
            new_row[column] = float("nan")

        new_row["Open"] = nse_row["Open"]
        new_row["High"] = nse_row["High"]
        new_row["Low"] = nse_row["Low"]
        new_row["Close"] = nse_row["Close"]
        new_row["Volume"] = nse_row["Volume"]

        # NSE does not provide Yahoo adjusted close,
        # dividends, or split history in this endpoint.
        if "Adj Close" in data.columns:
            new_row["Adj Close"] = float("nan")

        if "Dividends" in data.columns:
            new_row["Dividends"] = float("nan")

        if "Stock Splits" in data.columns:
            new_row["Stock Splits"] = float("nan")

        data.loc[date] = new_row

        inserted_rows.append(
            date.strftime("%Y-%m-%d")
        )

    # --------------------------------------------------------------
    # Sort after inserting the missing observations.
    # --------------------------------------------------------------
    data = data.sort_index()

    print(
        f"[{symbol}] NSE DATE FALLBACK INSERTED "
        f"{len(inserted_rows)} complete OHLCV row(s):"
    )

    for date in inserted_rows:
        print(
            f"    {date}: Open/High/Low/Close/Volume "
            f"inserted from NSE"
        )

    return data


def download_symbol_from_nse(symbol):
    """
    Download a historical security-wise price/volume series directly
    from NSE for symbols where Yahoo does not provide usable history.

    NSE is used as the complete source for these symbols.
    We do not fabricate adjusted-close, dividend, or split data.
    """

    nse_start_date, nse_end_date = (
        DIRECT_NSE_DATE_RANGES[symbol]
    )

    print(
        f"[{symbol}] DIRECT NSE DOWNLOAD: "
        f"{nse_start_date} -> {nse_end_date}"
    )


    params = {
        "from": pd.Timestamp(
            nse_start_date
        ).strftime("%d-%m-%Y"),

        "to": pd.Timestamp(
            nse_end_date
        ).strftime("%d-%m-%Y"),

        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    try:
        response = requests.get(
            NSE_HISTORY_URL,
            params=params,
            headers=NSE_HEADERS,
            timeout=30,
        )

        response.raise_for_status()

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: Direct NSE request failed: {error}"
        ) from error

    if not response.text.strip():
        raise RuntimeError(
            f"{symbol}: Direct NSE response is empty."
        )

    try:
        rows = list(
            csv.reader(
                io.StringIO(
                    response.text.lstrip("\ufeff")
                )
            )
        )

    except Exception as error:
        raise RuntimeError(
            f"{symbol}: Failed to parse NSE response: {error}"
        ) from error

    if len(rows) < 2:
        raise RuntimeError(
            f"{symbol}: NSE returned no data rows."
        )

    header = [
        str(column).strip()
        for column in rows[0]
    ]

    header_map = {
        name: index
        for index, name in enumerate(header)
    }

    required_columns = {
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    missing_columns = (
        required_columns - set(header_map)
    )

    if missing_columns:
        raise RuntimeError(
            f"{symbol}: NSE response missing columns: "
            f"{sorted(missing_columns)}"
        )

    records = []

    for row in rows[1:]:

        if not row:
            continue

        try:

            raw_date = row[header_map["Date"]].strip()

            date = pd.to_datetime(
                raw_date,
                format="%d-%b-%Y",
            ).normalize()

            def parse_number(value):

                value = str(value).strip()

                if not value:
                    return float("nan")

                return float(
                    value.replace(",", "")
                )

            records.append(
                {
                    "Date": date,
                    "Open": parse_number(
                        row[header_map["Open Price"]]
                    ),
                    "High": parse_number(
                        row[header_map["High Price"]]
                    ),
                    "Low": parse_number(
                        row[header_map["Low Price"]]
                    ),
                    "Close": parse_number(
                        row[header_map["Close Price"]]
                    ),
                    "Adj Close": float("nan"),
                    "Volume": parse_number(
                        row[
                            header_map[
                                "Total Traded Quantity"
                            ]
                        ]
                    ),
                    "Dividends": float("nan"),
                    "Stock Splits": float("nan"),
                }
            )

        except Exception as error:

            raise RuntimeError(
                f"{symbol}: Failed to parse NSE row: "
                f"{row}. Error: {error}"
            ) from error

    if not records:
        raise RuntimeError(
            f"{symbol}: NSE returned no usable observations."
        )

    data = pd.DataFrame(records)

    data = data.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    data = data.set_index("Date")

    data = data.sort_index()

    validate_price_data(
        symbol,
        data,
    )

    print(
        f"[{symbol}] DIRECT NSE COVERAGE: "
        f"{data.index.min().strftime('%Y-%m-%d')} -> "
        f"{data.index.max().strftime('%Y-%m-%d')} "
        f"({len(data)} rows)"
    )

    data = data.reset_index()

    data = data.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
    )

    data["date"] = pd.to_datetime(
        data["date"]
    ).dt.strftime("%Y-%m-%d")

    data = data[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
        ]
    ]

    return data

def download_symbol(symbol, membership):

    if symbol in DIRECT_NSE_SYMBOLS:

        data = download_symbol_from_nse(
            symbol
        )

        return data
    yahoo_symbol = YAHOO_TICKER_OVERRIDES.get(symbol, symbol)
    ticker = f"{yahoo_symbol}.NS"

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"\n[{symbol}] Downloading attempt {attempt}/{MAX_RETRIES} "
                f"(Yahoo ticker: {ticker})"
            )

            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=False,
                timeout=30,
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Normalize Yahoo dates before any repair.
            if isinstance(data.index, pd.DatetimeIndex):
                if data.index.tz is not None:
                    data.index = data.index.tz_localize(None)

                data.index = data.index.normalize()

            # Yahoo is the primary source.
            #
            # First repair completely absent trading dates.
            data = repair_missing_dates_from_nse(
                symbol,
                data,
                membership,
            )

            # Then repair Yahoo rows that exist but have
            # incomplete required OHLCV fields.
            data = repair_missing_required_fields_from_nse(
                symbol,
                data,
            )

            # Validate only after all permitted repairs.
            data = validate_price_data(
                symbol,
                data,
            )

            data = data.reset_index()

            if "Date" not in data.columns:
                raise ValueError(
                    f"{symbol}: Date column missing after reset."
                )

            data["Date"] = pd.to_datetime(
                data["Date"]
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

            return output

        except Exception as error:
            last_error = error

            print(
                f"[{symbol}] ERROR: {error}"
            )

            if attempt < MAX_RETRIES:
                print(
                    f"[{symbol}] Retrying in "
                    f"{RETRY_DELAY_SECONDS} seconds..."
                )

                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"{symbol}: Failed after "
        f"{MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def save_symbol(symbol, data):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{symbol}.csv",
    )

    data.to_csv(
        output_file,
        index=False,
    )

    return output_file


def validate_date_coverage(symbol, data):
    first_date = data["date"].min()
    last_date = data["date"].max()

    if first_date > START_DATE:
        print(
            f"[{symbol}] WARNING: "
            f"first available date is {first_date}"
        )

    print(
        f"[{symbol}] Coverage: "
        f"{first_date} -> {last_date} "
        f"({len(data)} rows)"
    )


def main():
    print("=" * 70)
    print("NIFTY 50 HISTORICAL PRICE DOWNLOADER")
    print("=" * 70)

    membership = load_membership()

    symbols = get_symbols(membership)
    price_symbols = get_price_symbols(symbols)

    print(
        f"Membership rows: {len(membership)}"
    )

    print(
        f"Unique historical membership symbols: "
        f"{len(symbols)}"
    )

    print(
        f"Ordinary price symbols: "
        f"{len(price_symbols)}"
    )

    print(
        f"Special index entities excluded from price download: "
        f"{sorted(set(symbols) - set(price_symbols))}"
    )

    print(
        f"Requested period: "
        f"{START_DATE} -> {END_DATE}"
    )

    print()

    successful = []
    failed = []

    for symbol in price_symbols:

        try:
            data = download_symbol(symbol, membership)

            validate_date_coverage(
                symbol,
                data,
            )

            output_file = save_symbol(
                symbol,
                data,
            )

            successful.append(symbol)

            print(
                f"[{symbol}] SAVED: "
                f"{output_file}"
            )

        except Exception as error:
            failed.append(
                (symbol, str(error))
            )

            print(
                f"[{symbol}] FAILED: {error}"
            )

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed:     {len(failed)}"
    )

    if failed:
        print()
        print("FAILED SYMBOLS")
        print("-" * 70)

        for symbol, error in failed:
            print(
                f"{symbol}: {error}"
            )

        raise RuntimeError(
            "Price download completed with failures."
        )

    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()