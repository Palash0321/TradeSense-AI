import os
import pandas as pd


MEMBERSHIP_FILE = "data/nifty50_constituent_membership.csv"
PRICE_DIR = "data/nifty50_prices"
AUDIT_FILE = "data/nifty50_price_source_audit.csv"

SOURCE_POLICY = {
    "HDFC": {
        "source": "NSE",
        "source_symbol": "HDFC",
        "required_start": "2021-01-29",
        "required_end": "2023-07-12",
        "reason": "Historical HDFC Ltd security; merged into HDFC Bank.",
    },
    "LTIM": {
        "source": "NSE",
        "source_symbol": "LTIM",
        "required_start": "2023-07-13",
        "required_end": "2024-09-27",
        "reason": "Historical LTIMindtree NIFTY constituent.",
    },
    "TATAMOTORS": {
        "source": "NSE",
        "source_symbol": "TATAMOTORS",
        "required_start": "2021-01-29",
        "required_end": "2025-10-23",
        "reason": "Historical Tata Motors security before 2025 restructuring.",
    },
}

SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM": {
        "reason": (
            "Temporary NIFTY 50 index-adjustment entity during the "
            "Tata Motors demerger; no ordinary OHLCV series required."
        ),
    },
}


def load_membership():
    if not os.path.exists(MEMBERSHIP_FILE):
        raise FileNotFoundError(MEMBERSHIP_FILE)

    membership = pd.read_csv(MEMBERSHIP_FILE)

    required = {"date", "symbol", "is_constituent"}
    missing = required - set(membership.columns)

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


def get_required_constituents(membership):
    return sorted(
        membership.loc[
            membership["is_constituent"] == 1,
            "symbol",
        ].unique()
    )


def get_membership_date_range(membership, symbol):
    rows = membership[
        (membership["symbol"] == symbol)
        & (membership["is_constituent"] == 1)
    ]

    if rows.empty:
        return None, None

    return (
        rows["date"].min().strftime("%Y-%m-%d"),
        rows["date"].max().strftime("%Y-%m-%d"),
    )


def inspect_local_file(symbol):
    file_path = os.path.join(
        PRICE_DIR,
        f"{symbol}.csv",
    )

    if not os.path.exists(file_path):
        return {
            "exists": False,
            "actual_start": "",
            "actual_end": "",
            "rows": 0,
            "columns": "",
            "ohlcv_complete": False,
        }

    data = pd.read_csv(file_path)

    required_columns = {
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

    missing_columns = required_columns - set(data.columns)

    if "date" in data.columns and not data.empty:
        dates = pd.to_datetime(
            data["date"],
            errors="coerce",
        )

        actual_start = (
            dates.min().strftime("%Y-%m-%d")
            if dates.notna().any()
            else ""
        )

        actual_end = (
            dates.max().strftime("%Y-%m-%d")
            if dates.notna().any()
            else ""
        )
    else:
        actual_start = ""
        actual_end = ""

    return {
        "exists": True,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "rows": len(data),
        "columns": ",".join(data.columns),
        "ohlcv_complete": not missing_columns,
    }


def main():
    print("=" * 70)
    print("NIFTY 50 PRICE SOURCE AUDIT")
    print("=" * 70)

    membership = load_membership()
    symbols = get_required_constituents(membership)

    print(f"Historical constituents: {len(symbols)}")
    print()

    audit_rows = []

    for symbol in symbols:
        membership_start, membership_end = (
            get_membership_date_range(
                membership,
                symbol,
            )
        )

        local = inspect_local_file(symbol)

        if symbol in SPECIAL_INDEX_ENTITIES:
            special = SPECIAL_INDEX_ENTITIES[symbol]

            source = "SPECIAL_INDEX_ENTITY"
            source_symbol = ""
            required_start = membership_start
            required_end = membership_end
            notes = special["reason"]

            status = "SPECIAL_NO_PRICE_REQUIRED"

        elif symbol in SOURCE_POLICY:
            policy = SOURCE_POLICY[symbol]

            source = policy["source"]
            source_symbol = policy["source_symbol"]
            required_start = policy["required_start"]
            required_end = policy["required_end"]
            notes = policy["reason"]

            status = "PASS"

            if not local["exists"]:
                status = "MISSING_LOCAL_DATA"

            elif not local["ohlcv_complete"]:
                status = "INCOMPLETE_LOCAL_SCHEMA"

            elif (
                local["actual_start"] > required_start
                or local["actual_end"] < required_end
            ):
                status = "INSUFFICIENT_DATE_COVERAGE"

        else:
            source = "YAHOO"
            source_symbol = symbol
            required_start = membership_start
            required_end = membership_end
            notes = "Yahoo primary source."

            status = "PASS"

            if not local["exists"]:
                status = "MISSING_LOCAL_DATA"

            elif not local["ohlcv_complete"]:
                status = "INCOMPLETE_LOCAL_SCHEMA"

            elif (
                local["actual_start"] > required_start
                or local["actual_end"] < required_end
            ):
                status = "INSUFFICIENT_DATE_COVERAGE"

        audit_rows.append(
            {
                "normalized_symbol": symbol,
                "source": source,
                "source_symbol": source_symbol,
                "membership_start": membership_start,
                "membership_end": membership_end,
                "required_start": required_start,
                "required_end": required_end,
                "actual_start": local["actual_start"],
                "actual_end": local["actual_end"],
                "rows": local["rows"],
                "ohlcv_complete": local["ohlcv_complete"],
                "corporate_action_issue": (
                    "YES"
                    if symbol in SOURCE_POLICY
                    else "NO"
                ),
                "source_verified": (
                    "SPECIAL"
                    if symbol in SPECIAL_INDEX_ENTITIES
                    else (
                        "PENDING"
                        if symbol in SOURCE_POLICY
                        else "YES"
                    )
                ),
                "status": status,
                "notes": notes,
            }
        )

    audit = pd.DataFrame(audit_rows)

    os.makedirs(
        os.path.dirname(AUDIT_FILE),
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    print(f"Audit file: {AUDIT_FILE}")
    print()

    print("SOURCE COUNTS")
    print("-" * 70)
    print(audit["source"].value_counts().to_string())

    print()
    print("STATUS COUNTS")
    print("-" * 70)
    print(audit["status"].value_counts().to_string())

    print()
    print("MISSING / INCOMPLETE")
    print("-" * 70)

    problem_rows = audit[
        ~audit["status"].isin(
            ["PASS", "SPECIAL_NO_PRICE_REQUIRED"]
        )
    ]

    if problem_rows.empty:
        print("None")
    else:
        print(
            problem_rows[
                [
                    "normalized_symbol",
                    "source",
                    "source_symbol",
                    "required_start",
                    "required_end",
                    "status",
                ]
            ].to_string(index=False)
        )

    print()
    print("STATUS: AUDIT COMPLETE")


if __name__ == "__main__":
    main()