from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PRICE_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
)

NIFTY_OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
    / "NIFTY_INDEX.csv"
)

REFRESH_START_DATE = "2026-09-01"

NIFTY_SYMBOL = "^NSEI"

YAHOO_TICKER_OVERRIDES = {
    "ZOMATO": "ETERNAL.NS",
}


# =========================================================
# HELPERS
# =========================================================

def normalize_yahoo_data(data):
    """
    Normalize Yahoo Finance daily data into:

    date
    open
    high
    low
    close
    volume

    Handles yfinance MultiIndex output safely.
    """

    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    # -----------------------------------------------------
    # yfinance may return MultiIndex columns:
    #
    # Price      Open        High ...
    # Ticker     RELIANCE.NS RELIANCE.NS ...
    #
    # We explicitly select the first ticker level rather
    # than relying on generic column flattening.
    # -----------------------------------------------------

    if isinstance(data.columns, pd.MultiIndex):

        # First level contains:
        # Open, High, Low, Close, Adj Close, Volume
        fields = data.columns.get_level_values(0)

        data.columns = fields

        # Remove accidental duplicate columns if any.
        data = data.loc[
            :,
            ~data.columns.duplicated()
        ]

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            "Yahoo data missing columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Reset date index AFTER column normalization.
    # -----------------------------------------------------

    data = data.reset_index()

    # yfinance normally calls this Date.
    # Handle lowercase date defensively too.
    if "Date" in data.columns:

        date_column = "Date"

    elif "date" in data.columns:

        date_column = "date"

    else:

        raise RuntimeError(
            "Yahoo data does not contain a Date column."
        )

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="raise",
    ).dt.normalize()

    # -----------------------------------------------------
    # Rename into our canonical schema.
    # -----------------------------------------------------

    data = data.rename(
        columns={
            date_column: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    data = data[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    # -----------------------------------------------------
    # Force numeric types.
    # -----------------------------------------------------

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # -----------------------------------------------------
    # Remove rows with missing required OHLCV.
    # -----------------------------------------------------

    data = data.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # -----------------------------------------------------
    # Sort and deduplicate.
    # -----------------------------------------------------

    data = (
        data
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Diagnostic sanity check BEFORE returning.
    # -----------------------------------------------------

    if data.empty:
        raise RuntimeError(
            "Yahoo normalization produced zero rows."
        )

    if (
        data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        <= 0
    ).any().any():

        bad_columns = (
            data[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]
            .columns[
                (
                    data[
                        [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                        ]
                    ]
                    <= 0
                ).any()
            ]
            .tolist()
        )

        raise RuntimeError(
            "Yahoo normalization produced "
            f"non-positive values in: {bad_columns}"
        )

    return data


# =========================================================
# REFRESH ONE YAHOO PRICE FILE
# =========================================================

def refresh_symbol(symbol):
    yahoo_ticker = YAHOO_TICKER_OVERRIDES.get(
        symbol,
        f"{symbol}.NS",
    )

    print()
    print("-" * 70)
    print(f"Refreshing {symbol}")
    print(f"Yahoo ticker: {yahoo_ticker}")
    print("-" * 70)

    price_file = (
        PRICE_DIR
        / f"{symbol}.csv"
    )

    # -----------------------------------------------------
    # Existing historical data
    # -----------------------------------------------------

    if price_file.exists():

        existing = pd.read_csv(
            price_file
        )

        existing["date"] = pd.to_datetime(
            existing["date"],
            errors="raise",
        ).dt.normalize()

        existing = (
            existing
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        existing_last_date = existing["date"].max()

        print(
            f"Existing coverage: "
            f"{existing['date'].min().date()} "
            f"-> "
            f"{existing_last_date.date()}"
        )

    else:

        existing = pd.DataFrame()

        print("Existing file: NONE")

    # -----------------------------------------------------
    # Download only the prospective refresh window
    # -----------------------------------------------------

    today = pd.Timestamp.today().normalize()

    download_end = (
        today
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    data = yf.download(
        yahoo_ticker,
        start=REFRESH_START_DATE,
        end=download_end,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    data = normalize_yahoo_data(data)

    if data.empty:
        print("No Yahoo rows returned.")
        return False

    print(
        f"Yahoo refresh rows: {len(data)}"
    )

    print(
        f"Yahoo refresh coverage: "
        f"{data['date'].min().date()} "
        f"-> "
        f"{data['date'].max().date()}"
    )

    # -----------------------------------------------------
    # Merge
    #
    # New Yahoo observations are added.
    # Existing historical rows are retained.
    # -----------------------------------------------------

    if existing.empty:

        combined = data.copy()

    else:

        # Preserve all existing columns.
        # Replace only dates returned by the refresh.
        historical_part = existing[
            ~existing["date"].isin(
                data["date"]
            )
        ].copy()

        combined = pd.concat(
            [
                historical_part,
                data,
            ],
            ignore_index=True,
        )

    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Required integrity checks
    # -----------------------------------------------------

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required_columns
        - set(combined.columns)
    )

    if missing:
        raise RuntimeError(
            f"{symbol}: missing columns after "
            f"refresh: {sorted(missing)}"
        )

    if combined["date"].duplicated().any():
        raise RuntimeError(
            f"{symbol}: duplicate dates after refresh."
        )

    if combined[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].isna().any().any():

        raise RuntimeError(
            f"{symbol}: missing required OHLCV "
            "values after refresh."
        )

    if (
        combined[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        <= 0
    ).any().any():

        raise RuntimeError(
            f"{symbol}: non-positive OHLCV "
            "value detected."
        )

    if (
        combined["high"]
        < combined[
            [
                "open",
                "close",
                "low",
            ]
        ].max(axis=1)
    ).any():

        raise RuntimeError(
            f"{symbol}: invalid High detected."
        )

    if (
        combined["low"]
        > combined[
            [
                "open",
                "close",
                "high",
            ]
        ].min(axis=1)
    ).any():

        raise RuntimeError(
            f"{symbol}: invalid Low detected."
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    price_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        price_file,
        index=False,
    )

    print(
        f"Final coverage: "
        f"{combined['date'].min().date()} "
        f"-> "
        f"{combined['date'].max().date()}"
    )

    print(
        f"Final rows: {len(combined)}"
    )

    print("STATUS: PASS")

    return True


# =========================================================
# REFRESH NIFTY INDEX
# =========================================================

def refresh_nifty_index():

    print()
    print("=" * 70)
    print("REFRESHING NIFTY 50 INDEX")
    print("=" * 70)

    data = yf.download(
        NIFTY_SYMBOL,
        start=REFRESH_START_DATE,
        end=(
            pd.Timestamp.today().normalize()
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    data = normalize_yahoo_data(data)

    if data.empty:
        raise RuntimeError(
            "No NIFTY index data returned."
        )

    data = data.rename(
        columns={
            "open": "nifty_open",
            "high": "nifty_high",
            "low": "nifty_low",
            "close": "nifty_close",
            "volume": "nifty_volume",
        }
    )

    data = data[
        [
            "date",
            "nifty_open",
            "nifty_high",
            "nifty_low",
            "nifty_close",
            "nifty_volume",
        ]
    ]

    NIFTY_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Merge with existing local NIFTY index file
    # -----------------------------------------------------

    if NIFTY_OUTPUT_FILE.exists():

        existing = pd.read_csv(
            NIFTY_OUTPUT_FILE
        )

        existing["date"] = pd.to_datetime(
            existing["date"],
            errors="raise",
        ).dt.normalize()

        existing = existing[
            ~existing["date"].isin(
                data["date"]
            )
        ].copy()

        combined = pd.concat(
            [
                existing,
                data,
            ],
            ignore_index=True,
        )

    else:

        combined = data.copy()

    combined = (
        combined
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if combined[
        [
            "nifty_open",
            "nifty_high",
            "nifty_low",
            "nifty_close",
            "nifty_volume",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "NIFTY index contains missing "
            "required values."
        )

    combined.to_csv(
        NIFTY_OUTPUT_FILE,
        index=False,
    )

    print(
        f"NIFTY coverage: "
        f"{combined['date'].min().date()} "
        f"-> "
        f"{combined['date'].max().date()}"
    )

    print(
        f"NIFTY rows: {len(combined)}"
    )

    print("NIFTY STATUS: PASS")


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print("NIFTY 50 PROSPECTIVE DATA REFRESH")
    print("=" * 70)

    print(
        f"Refresh start: {REFRESH_START_DATE}"
    )

    print(
        f"Run date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    PRICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # NIFTY index
    # -----------------------------------------------------

    refresh_nifty_index()

    # -----------------------------------------------------
    # Discover existing ordinary constituent files
    #
    # We intentionally do NOT create DUMMYTATAM.
    # -----------------------------------------------------

    price_files = sorted(
        PRICE_DIR.glob("*.csv")
    )

    symbols = []

    for price_file in price_files:

        symbol = price_file.stem.upper()

        if symbol == "NIFTY_INDEX":
            continue

        if symbol == "DUMMYTATAM":
            continue

        symbols.append(symbol)

    symbols = sorted(
        set(symbols)
    )

    if not symbols:
        raise RuntimeError(
            "No constituent price files found."
        )

    print()
    print(
        f"Constituent price files found: "
        f"{len(symbols)}"
    )

    # -----------------------------------------------------
    # Refresh constituents
    # -----------------------------------------------------

    successful = []
    failed = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(symbols)}]"
        )

        try:

            if refresh_symbol(symbol):
                successful.append(symbol)

        except Exception as exc:

            print(
                f"FAILED: {symbol}"
            )

            print(
                f"Reason: {exc}"
            )

            failed.append(
                symbol
            )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("PROSPECTIVE DATA REFRESH SUMMARY")
    print("=" * 70)

    print(
        f"Symbols discovered: {len(symbols)}"
    )

    print(
        f"Successful: {len(successful)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    if failed:

        print(
            "Failed symbols:"
        )

        for symbol in failed:
            print(
                f"  - {symbol}"
            )

        print()
        print("STATUS: FAIL")

        raise RuntimeError(
            "One or more prospective "
            "price refreshes failed."
        )

    print()
    print("STATUS: PASS")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()