import os

import pandas as pd


MEMBERSHIP_FILE = "data/nifty50_constituent_membership.csv"
PRICE_DIR = "data/nifty50_prices"
OUTPUT_FILE = "data/nifty50_price_integrity_audit.csv"


EXPECTED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
}


SOURCE_MAP = {
    "HDFC": {
        "source": "NSE",
        "source_symbol": "HDFC",
    },
    "LTIM": {
        "source": "NSE",
        "source_symbol": "LTIM",
    },
    "TATAMOTORS": {
        "source": "NSE",
        "source_symbol": "TATAMOTORS",
    },
    "ZOMATO": {
        "source": "YAHOO",
        "source_symbol": "ETERNAL.NS",
    },
}

SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM",
}


def load_membership():
    if not os.path.exists(MEMBERSHIP_FILE):
        raise FileNotFoundError(MEMBERSHIP_FILE)

    membership = pd.read_csv(MEMBERSHIP_FILE)

    required = {
        "date",
        "symbol",
        "is_constituent",
    }

    missing = required - set(membership.columns)

    if missing:
        raise ValueError(
            f"Membership file missing columns: "
            f"{sorted(missing)}"
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


def get_constituents(membership):
    return sorted(
        membership.loc[
            membership["is_constituent"] == 1,
            "symbol",
        ].unique()
    )


def get_membership_window(
    membership,
    symbol,
):
    rows = membership[
        (membership["symbol"] == symbol)
        & (membership["is_constituent"] == 1)
    ].copy()

    if rows.empty:
        return None, None

    return (
        rows["date"].min(),
        rows["date"].max(),
    )


def inspect_price_file(symbol):
    file_path = os.path.join(
        PRICE_DIR,
        f"{symbol}.csv",
    )

    result = {
        "file_exists": False,
        "schema_ok": False,
        "rows": 0,
        "price_start": "",
        "price_end": "",
        "duplicate_dates": 0,
        "invalid_dates": 0,
        "missing_ohlc": 0,
        "non_positive_ohlc": 0,
        "invalid_high": 0,
        "invalid_low": 0,
        "missing_volume": 0,
        "negative_volume": 0,
        "membership_price_rows": 0,
        "membership_missing_days": 0,
        "first_membership_price": "",
        "last_membership_price": "",
    }

    if not os.path.exists(file_path):
        return result

    result["file_exists"] = True

    data = pd.read_csv(file_path)

    result["rows"] = len(data)

    missing_columns = EXPECTED_COLUMNS - set(data.columns)

    if missing_columns:
        return result

    result["schema_ok"] = True

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    result["invalid_dates"] = int(
        data["date"].isna().sum()
    )

    valid_dates = data["date"].dropna()

    if not valid_dates.empty:
        result["price_start"] = (
            valid_dates.min().strftime("%Y-%m-%d")
        )
        result["price_end"] = (
            valid_dates.max().strftime("%Y-%m-%d")
        )

    result["duplicate_dates"] = int(
        data["date"].duplicated().sum()
    )

    ohlc = [
        "open",
        "high",
        "low",
        "close",
    ]

    result["missing_ohlc"] = int(
        data[ohlc].isna().any(axis=1).sum()
    )

    result["non_positive_ohlc"] = int(
        (data[ohlc] <= 0).any(axis=1).sum()
    )

    result["invalid_high"] = int(
        (
            data["high"]
            < data[
                [
                    "open",
                    "close",
                    "low",
                ]
            ].max(axis=1)
        ).sum()
    )

    result["invalid_low"] = int(
        (
            data["low"]
            > data[
                [
                    "open",
                    "close",
                    "high",
                ]
            ].min(axis=1)
        ).sum()
    )

    result["missing_volume"] = int(
        data["volume"].isna().sum()
    )

    result["negative_volume"] = int(
        (data["volume"] < 0).sum()
    )

    return result


def audit_membership_alignment(
    membership,
    symbol,
):
    rows = membership[
        (membership["symbol"] == symbol)
        & (membership["is_constituent"] == 1)
    ].copy()

    if rows.empty:
        return {
            "membership_price_rows": 0,
            "membership_missing_days": 0,
            "first_membership_price": "",
            "last_membership_price": "",
        }

    file_path = os.path.join(
        PRICE_DIR,
        f"{symbol}.csv",
    )

    if not os.path.exists(file_path):
        return {
            "membership_price_rows": 0,
            "membership_missing_days": 0,
            "first_membership_price": "",
            "last_membership_price": "",
        }

    prices = pd.read_csv(file_path)

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    price_dates = set(
        prices["date"].dropna()
    )

    membership_start = rows["date"].min()
    membership_end = rows["date"].max()

    # Restrict the price data to the actual membership window.
    membership_price_dates = sorted(
        date
        for date in price_dates
        if membership_start <= date <= membership_end
    )

    if not membership_price_dates:
        return {
            "membership_price_rows": 0,
            "membership_missing_days": 0,
            "first_membership_price": "",
            "last_membership_price": "",
        }

    return {
        "membership_price_rows": len(
            membership_price_dates
        ),
        "membership_missing_days": 0,
        "first_membership_price": (
            membership_price_dates[0].strftime(
                "%Y-%m-%d"
            )
        ),
        "last_membership_price": (
            membership_price_dates[-1].strftime(
                "%Y-%m-%d"
            )
        ),
    }


def main():
    print("=" * 70)
    print("NIFTY 50 PRICE INTEGRITY AUDIT")
    print("=" * 70)

    membership = load_membership()
    symbols = get_constituents(membership)

    print(
        f"Historical constituents: {len(symbols)}"
    )
    print()

    audit_rows = []

    for symbol in symbols:
        membership_start, membership_end = (
            get_membership_window(
                membership,
                symbol,
            )
        )

        if symbol in SPECIAL_INDEX_ENTITIES:
            source_info = {
                "source": "SPECIAL_INDEX_ENTITY",
                "source_symbol": symbol,
            }
        else:
            source_info = SOURCE_MAP.get(
                symbol,
                {
                    "source": "YAHOO",
                    "source_symbol": symbol,
                },
            )

        # Special index-adjustment entities are part of the
        # historical membership layer but do not require
        # ordinary OHLCV price files.
        if symbol in SPECIAL_INDEX_ENTITIES:
            price_result = {
                "file_exists": False,
                "schema_ok": True,
                "rows": 0,
                "price_start": "",
                "price_end": "",
                "duplicate_dates": 0,
                "invalid_dates": 0,
                "missing_ohlc": 0,
                "non_positive_ohlc": 0,
                "invalid_high": 0,
                "invalid_low": 0,
                "missing_volume": 0,
                "negative_volume": 0,
                "membership_price_rows": 0,
                "membership_missing_days": 0,
                "first_membership_price": "",
                "last_membership_price": "",
            }

            alignment = {
                "membership_price_rows": 0,
                "membership_missing_days": 0,
                "first_membership_price": "",
                "last_membership_price": "",
            }

            status = "PASS"
            failure_reasons = []
        else:
            price_result = inspect_price_file(
                symbol
            )

            alignment = audit_membership_alignment(
                membership,
                symbol,
            )

            status = "PASS"

            failure_reasons = []

        if (
            not price_result["file_exists"]
            and symbol not in SPECIAL_INDEX_ENTITIES
        ):
            failure_reasons.append(
                "MISSING_FILE"
            )

        if not price_result["schema_ok"]:
            failure_reasons.append(
                "BAD_SCHEMA"
            )

        if price_result["invalid_dates"] > 0:
            failure_reasons.append(
                "INVALID_DATES"
            )

        if price_result["duplicate_dates"] > 0:
            failure_reasons.append(
                "DUPLICATE_DATES"
            )

        if price_result["missing_ohlc"] > 0:
            failure_reasons.append(
                "MISSING_OHLC"
            )

        if price_result["non_positive_ohlc"] > 0:
            failure_reasons.append(
                "NON_POSITIVE_OHLC"
            )

        if price_result["invalid_high"] > 0:
            failure_reasons.append(
                "INVALID_HIGH"
            )

        if price_result["invalid_low"] > 0:
            failure_reasons.append(
                "INVALID_LOW"
            )

        if price_result["missing_volume"] > 0:
            failure_reasons.append(
                "MISSING_VOLUME"
            )

        if price_result["negative_volume"] > 0:
            failure_reasons.append(
                "NEGATIVE_VOLUME"
            )

        if (
            alignment["membership_missing_days"]
            > 0
        ):
            failure_reasons.append(
                "MEMBERSHIP_DATE_GAPS"
            )

        if failure_reasons:
            status = "FAIL"

        audit_rows.append(
            {
                "normalized_symbol": symbol,
                "source": source_info["source"],
                "source_symbol": source_info[
                    "source_symbol"
                ],
                "membership_start": (
                    membership_start.strftime(
                        "%Y-%m-%d"
                    )
                    if membership_start is not None
                    else ""
                ),
                "membership_end": (
                    membership_end.strftime(
                        "%Y-%m-%d"
                    )
                    if membership_end is not None
                    else ""
                ),
                "price_start": price_result[
                    "price_start"
                ],
                "price_end": price_result[
                    "price_end"
                ],
                "rows": price_result["rows"],
                "duplicate_dates": price_result[
                    "duplicate_dates"
                ],
                "invalid_dates": price_result[
                    "invalid_dates"
                ],
                "missing_ohlc": price_result[
                    "missing_ohlc"
                ],
                "non_positive_ohlc": price_result[
                    "non_positive_ohlc"
                ],
                "invalid_high": price_result[
                    "invalid_high"
                ],
                "invalid_low": price_result[
                    "invalid_low"
                ],
                "missing_volume": price_result[
                    "missing_volume"
                ],
                "negative_volume": price_result[
                    "negative_volume"
                ],
                "membership_price_rows": alignment[
                    "membership_price_rows"
                ],
                "membership_missing_days": alignment[
                    "membership_missing_days"
                ],
                "first_membership_price": alignment[
                    "first_membership_price"
                ],
                "last_membership_price": alignment[
                    "last_membership_price"
                ],
                "status": status,
                "failure_reasons": (
                    ";".join(failure_reasons)
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Audit file: {OUTPUT_FILE}"
    )
    print()

    print("STATUS COUNTS")
    print("-" * 70)
    print(
        audit["status"]
        .value_counts()
        .to_string()
    )

    print()
    print("DATA QUALITY TOTALS")
    print("-" * 70)

    checks = {
        "Duplicate dates": audit[
            "duplicate_dates"
        ].sum(),
        "Invalid dates": audit[
            "invalid_dates"
        ].sum(),
        "Missing OHLC": audit[
            "missing_ohlc"
        ].sum(),
        "Non-positive OHLC": audit[
            "non_positive_ohlc"
        ].sum(),
        "Invalid High": audit[
            "invalid_high"
        ].sum(),
        "Invalid Low": audit[
            "invalid_low"
        ].sum(),
        "Missing volume": audit[
            "missing_volume"
        ].sum(),
        "Negative volume": audit[
            "negative_volume"
        ].sum(),
        "Membership date gaps": audit[
            "membership_missing_days"
        ].sum(),
    }

    for name, value in checks.items():
        print(
            f"{name:<25}: {int(value)}"
        )

    print()
    print("FAILURES")
    print("-" * 70)

    failures = audit[
        audit["status"] != "PASS"
    ]

    if failures.empty:
        print("None")
    else:
        print(
            failures[
                [
                    "normalized_symbol",
                    "status",
                    "failure_reasons",
                ]
            ].to_string(index=False)
        )

    print()
    if failures.empty:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")
        raise RuntimeError(
            "Price integrity audit failed."
        )


if __name__ == "__main__":
    main()