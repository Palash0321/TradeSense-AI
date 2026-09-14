import os
import time
from datetime import datetime

import pandas as pd
import requests


INDEX_NAME = "NIFTY MIDCAP SELECT"

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

OUTPUT_DIR = "tests/output/index_backtests"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_nse_ohlc.csv",
)

NSE_HOME = "https://www.nseindia.com/"
NSE_ENDPOINT = (
    "https://www.nseindia.com/api/historical/indicesHistory"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "application/json,text/plain,*/*"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "Connection": "keep-alive",
}


def create_session():
    session = requests.Session()

    session.headers.update(HEADERS)

    # Establish NSE session/cookies first.
    response = session.get(
        NSE_HOME,
        timeout=30,
    )

    response.raise_for_status()

    time.sleep(1)

    return session


def fetch_chunk(
    session,
    start_date,
    end_date,
    max_attempts=5,
):
    params = {
        "indexType": INDEX_NAME,
        "from": datetime.strptime(
            start_date,
            "%Y-%m-%d",
        ).strftime("%d-%m-%Y"),
        "to": datetime.strptime(
            end_date,
            "%Y-%m-%d",
        ).strftime("%d-%m-%Y"),
    }

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(
                NSE_ENDPOINT,
                params=params,
                timeout=30,
            )

            print(
                f"Request attempt {attempt}/{max_attempts}: "
                f"{start_date} -> {end_date}"
            )

            print(
                f"HTTP status: {response.status_code}"
            )

            if response.status_code in {
                429,
                500,
                502,
                503,
                504,
            }:
                last_error = RuntimeError(
                    f"NSE temporary HTTP "
                    f"{response.status_code}"
                )

                if attempt < max_attempts:
                    wait_seconds = 2 ** (attempt - 1)

                    print(
                        f"Temporary NSE response. "
                        f"Waiting {wait_seconds}s..."
                    )

                    time.sleep(wait_seconds)
                    continue

                raise last_error

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise RuntimeError(
                    "Unexpected NSE response format."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = exc

            if attempt < max_attempts:
                wait_seconds = 2 ** (attempt - 1)

                print(
                    f"Request failed: {exc}"
                )

                print(
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)
            else:
                raise

    raise RuntimeError(
        f"Unable to retrieve NSE data: {last_error}"
    )


def extract_records(payload):
    data = payload.get("data", {})

    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    candidate_keys = [
        "indexCloseOnlineRecords",
        "indexCloseRecords",
        "records",
    ]

    for key in candidate_keys:
        records = data.get(key)

        if isinstance(records, list):
            return records

    # Some NSE responses may expose nested data.
    for value in data.values():
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                return value

    return []


def find_value(record, candidates):
    for key in candidates:
        if key in record:
            return record[key]

    return None


def normalize_records(records):
    normalized = []

    for record in records:
        date_value = find_value(
            record,
            [
                "EOD_TIMESTAMP",
                "TIMESTAMP",
                "TIMESTAMP",
                "Date",
                "date",
            ],
        )

        open_value = find_value(
            record,
            [
                "EOD_OPEN_INDEX_VAL",
                "OPEN_INDEX_VAL",
                "Open",
                "open",
            ],
        )

        high_value = find_value(
            record,
            [
                "EOD_HIGH_INDEX_VAL",
                "HIGH_INDEX_VAL",
                "High",
                "high",
            ],
        )

        low_value = find_value(
            record,
            [
                "EOD_LOW_INDEX_VAL",
                "LOW_INDEX_VAL",
                "Low",
                "low",
            ],
        )

        close_value = find_value(
            record,
            [
                "EOD_CLOSE_INDEX_VAL",
                "CLOSE_INDEX_VAL",
                "Close",
                "close",
            ],
        )

        if (
            date_value is None
            or open_value is None
            or high_value is None
            or low_value is None
            or close_value is None
        ):
            continue

        normalized.append(
            {
                "Date": date_value,
                "Open": open_value,
                "High": high_value,
                "Low": low_value,
                "Close": close_value,
            }
        )

    return normalized


def clean_dataframe(records):
    df = pd.DataFrame(records)

    if df.empty:
        return df

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True,
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df = (
        df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
            ]
        ]
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def validate_ohlc(df):
    print()
    print("=" * 100)
    print("MIDSELECT NSE OHLC VALIDATION")
    print("=" * 100)

    if df.empty:
        print("No rows available.")
        return False

    duplicate_dates = int(
        df["Date"].duplicated().sum()
    )

    invalid_ohlc = int(
        (
            (df["Open"] <= 0)
            | (df["High"] <= 0)
            | (df["Low"] <= 0)
            | (df["Close"] <= 0)
            | (df["High"] < df["Low"])
            | (df["High"] < df["Open"])
            | (df["High"] < df["Close"])
            | (df["Low"] > df["Open"])
            | (df["Low"] > df["Close"])
        ).sum()
    )

    missing_values = int(
        df[
            [
                "Open",
                "High",
                "Low",
                "Close",
            ]
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        f"Rows:              {len(df)}"
    )

    print(
        f"Start:             "
        f"{df['Date'].min()}"
    )

    print(
        f"End:               "
        f"{df['Date'].max()}"
    )

    print(
        f"Duplicate dates:   "
        f"{duplicate_dates}"
    )

    print(
        f"Invalid OHLC rows: "
        f"{invalid_ohlc}"
    )

    print(
        f"Missing OHLC rows:  "
        f"{missing_values}"
    )

    print()
    print("Latest rows:")
    print(
        df.tail(5).to_string(
            index=False
        )
    )

    valid = (
        duplicate_dates == 0
        and invalid_ohlc == 0
        and missing_values == 0
    )

    print()

    if valid:
        print("OHLC VALIDATION: PASS")
    else:
        print("OHLC VALIDATION: FAIL")

    return valid


def validate_date_coverage(df):
    print()
    print("=" * 100)
    print("MIDSELECT DATE COVERAGE")
    print("=" * 100)

    if df.empty:
        print("No rows available.")
        return False

    expected_start = pd.Timestamp(
        START_DATE
    )

    expected_end = pd.Timestamp(
        "2026-09-11"
    )

    actual_start = df["Date"].min()
    actual_end = df["Date"].max()

    print(
        f"Requested start : {expected_start.date()}"
    )

    print(
        f"Actual start    : {actual_start.date()}"
    )

    print(
        f"Requested end   : {expected_end.date()}"
    )

    print(
        f"Actual end      : {actual_end.date()}"
    )

    start_ok = actual_start <= expected_start
    end_ok = actual_end >= expected_end

    if start_ok and end_ok:
        print(
            "Coverage boundary: PASS"
        )
        return True

    print(
        "Coverage boundary: ATTENTION"
    )

    return False


def main():
    print("=" * 100)
    print("TRADESENSE-AI — MIDSELECT NSE DATA SOURCE AUDIT")
    print("=" * 100)

    print(
        f"Index: {INDEX_NAME}"
    )

    print(
        f"Requested range: "
        f"{START_DATE} -> {END_DATE}"
    )

    print()
    print(
        "Source: NSE historical index data"
    )

    session = create_session()

    # Retrieve the history in smaller yearly chunks.
    # This reduces the chance of NSE rejecting a very large request.
    all_records = []

    start_year = int(
        START_DATE[:4]
    )

    end_year = int(
        END_DATE[:4]
    )

    for year in range(
        start_year,
        end_year + 1,
    ):
        chunk_start = (
            START_DATE
            if year == start_year
            else f"{year}-01-01"
        )

        chunk_end = (
            END_DATE
            if year == end_year
            else f"{year}-12-31"
        )

        print()
        print(
            "-" * 100
        )

        print(
            f"Fetching NSE MIDSELECT: "
            f"{chunk_start} -> {chunk_end}"
        )

        try:
            payload = fetch_chunk(
                session,
                chunk_start,
                chunk_end,
            )

            chunk_records = extract_records(
                payload
            )

            print(
                f"Records returned: "
                f"{len(chunk_records)}"
            )

            all_records.extend(
                chunk_records
            )

        except Exception as exc:
            print()
            print(
                f"CHUNK FAILED: "
                f"{chunk_start} -> {chunk_end}"
            )

            print(
                f"Reason: {exc}"
            )

            print(
                "MIDSELECT retrieval is incomplete."
            )

            print(
                "Do NOT use the dataset for backtesting."
            )

            return

        # Give NSE a little breathing room between requests.
        time.sleep(2)

    print()
    print(
        f"Total raw NSE records: "
        f"{len(all_records)}"
    )

    normalized = normalize_records(
        all_records
    )

    print(
        f"Normalized OHLC records: "
        f"{len(normalized)}"
    )

    print(
        f"Normalized OHLC records: "
        f"{len(normalized)}"
    )

    df = clean_dataframe(
        normalized
    )

    if df.empty:
        print()
        print(
            "STATUS: NO_DATA"
        )
        return

    df = df[
        (df["Date"] >= pd.Timestamp(START_DATE))
        & (
            df["Date"]
            <= pd.Timestamp("2026-09-11")
        )
    ].copy()

    df = df.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    ohlc_pass = validate_ohlc(
        df
    )

    coverage_pass = validate_date_coverage(
        df
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("OUTPUT")
    print("=" * 100)

    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print(
        f"Rows saved: {len(df)}"
    )

    print()
    print("=" * 100)
    print("FINAL STATUS")
    print("=" * 100)

    if ohlc_pass and coverage_pass:
        print(
            "STATUS: PASS"
        )
        print(
            "MIDSELECT NSE dataset is ready "
            "for the next independent backtest gate."
        )
    else:
        print(
            "STATUS: ATTENTION REQUIRED"
        )
        print(
            "Do NOT begin the MIDSELECT backtest yet."
        )


if __name__ == "__main__":
    main()