from pathlib import Path
import io
import shutil
import time

import pandas as pd
import requests


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PRICE_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
)

AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
    / "nse_repair_audit.csv"
)

BACKUP_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
    / "_pre_nse_repair_backup"
)

NSE_URL = (
    "https://www.nseindia.com/api/"
    "historicalOR/generateSecurityWiseHistoricalData"
)

NSE_PAGE = "https://www.nseindia.com/"


SPECIAL_SYMBOLS = {
    "DUMMYTATAM",
    "NIFTY_INDEX",
}


# =========================================================
# NSE SESSION
# =========================================================

def create_nse_session():

    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/153.0.0.0 "
                "Safari/537.36"
            ),
            "Accept": (
                "application/json,"
                "text/plain,*/*"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_PAGE,
            "Connection": "keep-alive",
        }
    )

    # Establish NSE cookies/session.
    response = session.get(
        NSE_PAGE,
        timeout=30,
    )

    response.raise_for_status()

    return session


# =========================================================
# HELPERS
# =========================================================

def clean_column_name(column):

    return (
        str(column)
        .replace("\ufeff", "")
        .strip()
    )


def parse_numeric(value):

    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("₹", "")
        .replace("â¹", "")
    )

    if text in {
        "",
        "-",
        "nan",
        "None",
        "NA",
    }:
        return None

    return float(text)


def find_bad_rows():

    bad_rows = []

    price_files = sorted(
        PRICE_DIR.glob("*.csv")
    )

    for price_file in price_files:

        # Ignore audit artifacts created by data-quality scripts.
        if price_file.name in {
            AUDIT_FILE.name,
            "nse_holiday_cleanup_audit.csv",
        }:
            continue

        symbol = price_file.stem.upper()

        if symbol in SPECIAL_SYMBOLS:
            continue

        data = pd.read_csv(
            price_file
        )

        required = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = (
            required
            - set(data.columns)
        )

        if missing:
            raise RuntimeError(
                f"{symbol}: missing columns "
                f"{sorted(missing)}"
            )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        ).dt.normalize()

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        for column in numeric_columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        bad = data[
            data["volume"] <= 0
        ].copy()

        for row in bad.itertuples(
            index=False
        ):

            bad_rows.append(
                {
                    "symbol": symbol,
                    "date": row.date,
                    "old_open": row.open,
                    "old_high": row.high,
                    "old_low": row.low,
                    "old_close": row.close,
                    "old_volume": row.volume,
                }
            )

    return pd.DataFrame(
        bad_rows
    )


# =========================================================
# NSE DOWNLOAD
# =========================================================

def fetch_nse_history(
    session,
    symbol,
    start_date,
    end_date,
):

    params = {
        "from": start_date.strftime(
            "%d-%m-%Y"
        ),
        "to": end_date.strftime(
            "%d-%m-%Y"
        ),
        "symbol": symbol,
        "type": "priceVolumeDeliverable",
        "series": "EQ",
        "csv": "true",
    }

    response = session.get(
        NSE_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    text = response.content.decode(
        "utf-8-sig",
        errors="replace",
    )

    if not text.strip():
        raise RuntimeError(
            f"{symbol}: NSE returned empty response."
        )

    # NSE response is CSV.
    data = pd.read_csv(
        io.StringIO(text)
    )

    # Normalize column names.
    data.columns = [
        clean_column_name(c)
        for c in data.columns
    ]

    # Normalize expected names.
    rename_map = {}

    for column in data.columns:

        normalized = (
            column
            .replace("₹", "")
            .replace("â¹", "")
            .strip()
        )

        if normalized == "Symbol":
            rename_map[column] = "Symbol"

        elif normalized == "Series":
            rename_map[column] = "Series"

        elif normalized == "Date":
            rename_map[column] = "Date"

        elif normalized == "Open Price":
            rename_map[column] = "Open Price"

        elif normalized == "High Price":
            rename_map[column] = "High Price"

        elif normalized == "Low Price":
            rename_map[column] = "Low Price"

        elif normalized == "Close Price":
            rename_map[column] = "Close Price"

        elif normalized == "Total Traded Quantity":
            rename_map[column] = (
                "Total Traded Quantity"
            )

    data = data.rename(
        columns=rename_map
    )

    required = {
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise RuntimeError(
            f"{symbol}: NSE response missing "
            f"columns {sorted(missing)}"
        )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        dayfirst=True,
    ).dt.normalize()

    for column in [
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    ]:

        data[column] = data[column].apply(
            parse_numeric
        )

    data = data.dropna(
        subset=[
            "Date",
            "Open Price",
            "High Price",
            "Low Price",
            "Close Price",
            "Total Traded Quantity",
        ]
    )

    data = data[
        (
            data["Date"]
            >= start_date
        )
        & (
            data["Date"]
            <= end_date
        )
    ].copy()

    data = (
        data
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return data


# =========================================================
# VALIDATE ONE NSE ROW
# =========================================================

def validate_nse_row(row):
    """
    Validate one NSE historical price-volume observation.

    The row is expected to be a pandas Series returned by
    DataFrame.iterrows(), preserving the actual NSE column names.
    """

    required_columns = {
        "Date",
        "Open Price",
        "High Price",
        "Low Price",
        "Close Price",
        "Total Traded Quantity",
    }

    if not required_columns.issubset(row.index):
        return False

    try:
        open_price = parse_numeric(
            row["Open Price"]
        )

        high_price = parse_numeric(
            row["High Price"]
        )

        low_price = parse_numeric(
            row["Low Price"]
        )

        close_price = parse_numeric(
            row["Close Price"]
        )

        volume = parse_numeric(
            row["Total Traded Quantity"]
        )

    except (TypeError, ValueError):
        return False

    values = [
        open_price,
        high_price,
        low_price,
        close_price,
        volume,
    ]

    if any(
        value is None
        or pd.isna(value)
        or value <= 0
        for value in values
    ):
        return False

    # OHLC integrity.
    if high_price < max(
        open_price,
        close_price,
        low_price,
    ):
        return False

    if low_price > min(
        open_price,
        close_price,
        high_price,
    ):
        return False

    return True


# =========================================================
# MAIN REPAIR
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "NIFTY 50 — NSE HISTORICAL "
        "BAD-VOLUME REPAIR"
    )
    print("=" * 70)

    print()
    print(
        "This script repairs only rows where "
        "existing volume <= 0."
    )

    print(
        "Entire OHLCV rows are replaced from NSE."
    )

    print(
        "Model 10 files are NOT touched."
    )

    print()

    bad_rows = find_bad_rows()

    if bad_rows.empty:

        print(
            "No zero/non-positive volume rows found."
        )

        print(
            "STATUS: NOTHING TO REPAIR"
        )

        return

    print(
        f"Symbols affected: "
        f"{bad_rows['symbol'].nunique()}"
    )

    print(
        f"Rows requiring repair: "
        f"{len(bad_rows)}"
    )

    print()

    # -----------------------------------------------------
    # Create backup directory.
    # -----------------------------------------------------

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    affected_symbols = sorted(
        bad_rows["symbol"].unique()
    )

    print(
        "Creating backups..."
    )

    for symbol in affected_symbols:

        source = (
            PRICE_DIR
            / f"{symbol}.csv"
        )

        destination = (
            BACKUP_DIR
            / f"{symbol}.csv"
        )

        shutil.copy2(
            source,
            destination,
        )

    print(
        f"Backups created: "
        f"{len(affected_symbols)}"
    )

    print()

    # -----------------------------------------------------
    # NSE session
    # -----------------------------------------------------

    print(
        "Connecting to NSE..."
    )

    session = create_nse_session()

    print(
        "NSE session: PASS"
    )

    print()

    audit_rows = []

    unrepaired = []

    repaired_count = 0

    # -----------------------------------------------------
    # Process each symbol.
    # -----------------------------------------------------

    for symbol in affected_symbols:

        symbol_bad = bad_rows[
            bad_rows["symbol"]
            == symbol
        ].copy()

        start_date = (
            symbol_bad["date"].min()
        )

        end_date = (
            symbol_bad["date"].max()
        )

        print("-" * 70)

        print(
            f"{symbol}: "
            f"{len(symbol_bad)} rows"
        )

        print(
            f"NSE range: "
            f"{start_date.date()} "
            f"-> "
            f"{end_date.date()}"
        )

        try:

            nse_data = fetch_nse_history(
                session,
                symbol,
                start_date,
                end_date,
            )

        except Exception as exc:

            print(
                f"ERROR fetching NSE data: {exc}"
            )

            for row in symbol_bad.itertuples(
                index=False
            ):

                audit_rows.append(
                    {
                        "symbol": symbol,
                        "date": row.date,
                        "status": "FAILED_NSE_FETCH",
                        "old_open": row.old_open,
                        "old_high": row.old_high,
                        "old_low": row.old_low,
                        "old_close": row.old_close,
                        "old_volume": row.old_volume,
                    }
                )

                unrepaired.append(
                    (
                        symbol,
                        row.date,
                    )
                )

            continue

        nse_lookup = {
            row["Date"]: row
            for _, row in nse_data.iterrows()
        }

        price_file = (
            PRICE_DIR
            / f"{symbol}.csv"
        )

        existing = pd.read_csv(
            price_file
        )

        existing["date"] = pd.to_datetime(
            existing["date"],
            errors="raise",
        ).dt.normalize()

        symbol_repaired = 0

        for row in symbol_bad.itertuples(
            index=False
        ):

            repair_date = row.date

            nse_row = nse_lookup.get(
                repair_date
            )

            if nse_row is None:

                print(
                    f"  {repair_date.date()}: "
                    "NO NSE OBSERVATION"
                )

                audit_rows.append(
                    {
                        "symbol": symbol,
                        "date": repair_date,
                        "status": "FAILED_NO_NSE_ROW",
                        "old_open": row.old_open,
                        "old_high": row.old_high,
                        "old_low": row.old_low,
                        "old_close": row.old_close,
                        "old_volume": row.old_volume,
                    }
                )

                unrepaired.append(
                    (
                        symbol,
                        repair_date,
                    )
                )

                continue

            if not validate_nse_row(
                nse_row
            ):

                print(
                    f"  {repair_date.date()}: "
                    "INVALID NSE ROW"
                )

                audit_rows.append(
                    {
                        "symbol": symbol,
                        "date": repair_date,
                        "status": "FAILED_INVALID_NSE_ROW",
                        "old_open": row.old_open,
                        "old_high": row.old_high,
                        "old_low": row.old_low,
                        "old_close": row.old_close,
                        "old_volume": row.old_volume,
                    }
                )

                unrepaired.append(
                    (
                        symbol,
                        repair_date,
                    )
                )

                continue

            new_open = parse_numeric(
                nse_row["Open Price"]
            )

            new_high = parse_numeric(
                nse_row["High Price"]
            )

            new_low = parse_numeric(
                nse_row["Low Price"]
            )

            new_close = parse_numeric(
                nse_row["Close Price"]
            )

            new_volume = parse_numeric(
                nse_row["Total Traded Quantity"]
            )

            mask = (
                existing["date"]
                == repair_date
            )

            matching_rows = int(
                mask.sum()
            )

            if matching_rows != 1:

                print(
                    f"  {repair_date.date()}: "
                    f"EXPECTED 1 LOCAL ROW, "
                    f"FOUND {matching_rows}"
                )

                audit_rows.append(
                    {
                        "symbol": symbol,
                        "date": repair_date,
                        "status": "FAILED_LOCAL_ROW_COUNT",
                        "old_open": row.old_open,
                        "old_high": row.old_high,
                        "old_low": row.old_low,
                        "old_close": row.old_close,
                        "old_volume": row.old_volume,
                    }
                )

                unrepaired.append(
                    (
                        symbol,
                        repair_date,
                    )
                )

                continue

            # -------------------------------------------------
            # Replace COMPLETE OHLCV row.
            #
            # Do NOT fabricate adjusted close,
            # dividends, or splits.
            # -------------------------------------------------

            existing.loc[
                mask,
                "open"
            ] = new_open

            existing.loc[
                mask,
                "high"
            ] = new_high

            existing.loc[
                mask,
                "low"
            ] = new_low

            existing.loc[
                mask,
                "close"
            ] = new_close

            existing.loc[
                mask,
                "volume"
            ] = new_volume

            audit_rows.append(
                {
                    "symbol": symbol,
                    "date": repair_date,
                    "status": "REPAIRED_FROM_NSE",
                    "old_open": row.old_open,
                    "old_high": row.old_high,
                    "old_low": row.old_low,
                    "old_close": row.old_close,
                    "old_volume": row.old_volume,
                    "new_open": new_open,
                    "new_high": new_high,
                    "new_low": new_low,
                    "new_close": new_close,
                    "new_volume": new_volume,
                }
            )

            repaired_count += 1
            symbol_repaired += 1

            print(
                f"  {repair_date.date()}: "
                f"REPAIRED "
                f"volume {row.old_volume} "
                f"-> {new_volume}"
            )

        # -----------------------------------------------------
        # Save symbol only if every bad row for that symbol
        # was successfully repaired.
        # -----------------------------------------------------

        symbol_failures = [
            x
            for x in unrepaired
            if x[0] == symbol
        ]

        if symbol_failures:

            print(
                f"{symbol}: "
                f"NOT SAVED — "
                f"{len(symbol_failures)} "
                "repair(s) failed."
            )

        else:

            existing = (
                existing
                .sort_values("date")
                .drop_duplicates(
                    "date",
                    keep="last",
                )
                .reset_index(drop=True)
            )

            existing.to_csv(
                price_file,
                index=False,
            )

            print(
                f"{symbol}: "
                f"SAVED — "
                f"{symbol_repaired} repaired"
            )

        # Avoid hammering NSE.
        time.sleep(0.25)

    # -----------------------------------------------------
    # Audit output
    # -----------------------------------------------------

    audit = pd.DataFrame(
        audit_rows
    )

    if not audit.empty:

        audit = (
            audit
            .sort_values(
                ["symbol", "date"]
            )
            .reset_index(drop=True)
        )

        AUDIT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit.to_csv(
            AUDIT_FILE,
            index=False,
        )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("REPAIR SUMMARY")
    print("=" * 70)

    print(
        f"Original suspicious rows: "
        f"{len(bad_rows)}"
    )

    print(
        f"Successfully repaired: "
        f"{repaired_count}"
    )

    print(
        f"Unrepaired: "
        f"{len(unrepaired)}"
    )

    print(
        f"Audit file: "
        f"{AUDIT_FILE}"
    )

    print(
        f"Backup directory: "
        f"{BACKUP_DIR}"
    )

    if unrepaired:

        print()
        print(
            "UNREPAIRED ROWS:"
        )

        for symbol, date in unrepaired:

            print(
                f"  {symbol} "
                f"{date.date()}"
            )

        print()
        print(
            "STATUS: FAIL"
        )

        raise RuntimeError(
            "Not all suspicious rows "
            "were successfully repaired."
        )

    # -----------------------------------------------------
    # Re-scan after repair.
    # -----------------------------------------------------

    remaining = find_bad_rows()

    print()

    print(
        f"Remaining volume <= 0 rows: "
        f"{len(remaining)}"
    )

    if not remaining.empty:

        print(
            "STATUS: FAIL"
        )

        raise RuntimeError(
            "Bad volume rows remain "
            "after repair."
        )

    print()
    print(
        "All suspicious rows were "
        "successfully repaired from NSE."
    )

    print(
        "STATUS: PASS"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()