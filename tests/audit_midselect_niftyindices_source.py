import os
import ast
import time
from datetime import datetime

import pandas as pd
import requests


INDEX_NAME = "Nifty Midcap Select"

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

OUTPUT_DIR = "tests/output/index_backtests"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_niftyindices_ohlc.csv",
)

BASE_URL = "https://niftyindices.com"
ENDPOINT = (
    f"{BASE_URL}/BackPage/"
    "getHistoricaldatatabletoString"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "application/json, text/plain, */*"
    ),
    "Content-Type": "application/json",
    "Origin": BASE_URL,
    "Referer": (
        f"{BASE_URL}/"
    ),
}


def create_session():
    session = requests.Session()
    session.headers.update(HEADERS)

    response = session.get(
        BASE_URL,
        timeout=30,
    )

    print(
        f"Nifty Indices homepage HTTP: "
        f"{response.status_code}"
    )

    response.raise_for_status()

    time.sleep(1)

    return session


def parse_date_for_api(date_string):
    return datetime.strptime(
        date_string,
        "%Y-%m-%d",
    ).strftime("%d-%b-%Y")


def fetch_history(
    session,
    start_date,
    end_date,
):
    cinfo = {
        "name": INDEX_NAME,
        "startDate": parse_date_for_api(
            start_date
        ),
        "endDate": parse_date_for_api(
            end_date
        ),
        "indexName": INDEX_NAME,
    }

    params = {
        "cinfo": str(cinfo).replace(
            '"',
            "'",
        )
    }

    print()
    print(
        "Request endpoint:"
    )
    print(ENDPOINT)

    print()
    print(
        "Requested index:"
    )
    print(INDEX_NAME)

    print()
    print(
        "Requested range:"
    )
    print(
        f"{start_date} -> {end_date}"
    )

    response = session.post(
        ENDPOINT,
        json=params,
        timeout=60,
    )

    print()
    print(
        f"HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    return response.json()


def extract_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in [
            "data",
            "d",
            "records",
            "result",
        ]:
            value = payload.get(key)

            if isinstance(value, list):
                return value

            if isinstance(value, str):
                try:
                    parsed = ast.literal_eval(
                        value
                    )

                    if isinstance(
                        parsed,
                        list,
                    ):
                        return parsed

                except Exception:
                    pass

    if isinstance(payload, str):
        try:
            parsed = ast.literal_eval(
                payload
            )

            if isinstance(
                parsed,
                list,
            ):
                return parsed

        except Exception:
            pass

    return []


def find_value(record, candidates):
    for candidate in candidates:
        if candidate in record:
            return record[candidate]

    return None


def normalize_records(records):
    normalized = []

    for record in records:
        if not isinstance(record, dict):
            continue

        date_value = find_value(
            record,
            [
                "Date",
                "date",
                "DATE",
                "HistoricalDate",
                "historicalDate",
            ],
        )

        open_value = find_value(
            record,
            [
                "Open",
                "open",
                "OPEN",
                "OpenIndex",
            ],
        )

        high_value = find_value(
            record,
            [
                "High",
                "high",
                "HIGH",
                "HighIndex",
            ],
        )

        low_value = find_value(
            record,
            [
                "Low",
                "low",
                "LOW",
                "LowIndex",
            ],
        )

        close_value = find_value(
            record,
            [
                "Close",
                "close",
                "CLOSE",
                "CloseIndex",
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
    print("MIDSELECT NIFTY INDICES OHLC VALIDATION")
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
        print(
            "OHLC VALIDATION: PASS"
        )
    else:
        print(
            "OHLC VALIDATION: FAIL"
        )

    return valid


def validate_coverage(df):
    print()
    print("=" * 100)
    print("MIDSELECT DATE COVERAGE")
    print("=" * 100)

    if df.empty:
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
        f"Expected start: "
        f"{expected_start.date()}"
    )

    print(
        f"Actual start:   "
        f"{actual_start.date()}"
    )

    print(
        f"Expected end:   "
        f"{expected_end.date()}"
    )

    print(
        f"Actual end:     "
        f"{actual_end.date()}"
    )

    start_ok = (
        actual_start <= expected_start
    )

    end_ok = (
        actual_end >= expected_end
    )

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
    print(
        "TRADESENSE-AI — MIDSELECT "
        "NIFTY INDICES DATA SOURCE AUDIT"
    )
    print("=" * 100)

    session = create_session()

    payload = fetch_history(
        session,
        START_DATE,
        END_DATE,
    )

    print()
    print(
        "Response type:",
        type(payload).__name__,
    )

    records = extract_records(
        payload
    )

    print(
        f"Raw records: {len(records)}"
    )

    if not records:
        print()
        print(
            "STATUS: NO_RECORDS"
        )
        return

    normalized = normalize_records(
        records
    )

    print(
        f"Normalized records: "
        f"{len(normalized)}"
    )

    # Inspect the complete source boundaries BEFORE
    # applying our requested research-date filter.
    raw_boundary_df = clean_dataframe(
        normalized
    )

    if not raw_boundary_df.empty:
        print()
        print("=" * 100)
        print("RAW SOURCE BOUNDARY CHECK")
        print("=" * 100)

        print(
            f"Raw source start: "
            f"{raw_boundary_df['Date'].min()}"
        )

        print(
            f"Raw source end:   "
            f"{raw_boundary_df['Date'].max()}"
        )

        print()
        print("Earliest 10 source rows:")
        print(
            raw_boundary_df.head(10).to_string(
                index=False
            )
        )

        print()
        print("Rows around 2022-01-10:")
        print(
            raw_boundary_df[
                raw_boundary_df["Date"]
                <= pd.Timestamp("2022-01-20")
            ].head(20).to_string(
                index=False
            )
        )

    df = raw_boundary_df

    if df.empty:
        print()
        print(
            "STATUS: NO_VALID_OHLC"
        )
        return

    df = df[
        (df["Date"] >= pd.Timestamp(
            START_DATE
        ))
        & (
            df["Date"]
            <= pd.Timestamp(
                "2026-09-11"
            )
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

    coverage_pass = validate_coverage(
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

    if (
        ohlc_pass
        and coverage_pass
    ):
        print(
            "STATUS: PASS"
        )

        print(
            "MIDSELECT dataset is ready "
            "for independent backtesting."
        )

    else:
        print(
            "STATUS: ATTENTION REQUIRED"
        )

        print(
            "Do NOT begin the MIDSELECT "
            "backtest yet."
        )


if __name__ == "__main__":
    main()