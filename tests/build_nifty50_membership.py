import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta


BASELINE_FILE = "data/nifty50_constituent_baseline_2021_01_29.csv"
EVENTS_FILE = "data/nifty50_constituent_events.csv"
OUTPUT_FILE = "data/nifty50_constituent_membership.csv"


BASELINE_DATE = "2021-01-29"
END_DATE = "2026-09-10"


# Documented temporary 51-member periods caused by corporate actions.
#
# JIOFIN:
# 2023-07-20 through 2023-09-06
#
# Tata Motors demerger:
# 2025-10-14 through 2025-11-16
#
# The Tata Motors period includes DUMMYTATAM, which represented the
# demerged commercial-vehicle entity for index-adjustment purposes.
TEMPORARY_51_PERIODS = [
    ("2023-07-20", "2023-09-06"),
    ("2025-10-14", "2025-11-16"),
]


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_files():
    if not os.path.exists(BASELINE_FILE):
        raise FileNotFoundError(BASELINE_FILE)

    if not os.path.exists(EVENTS_FILE):
        raise FileNotFoundError(EVENTS_FILE)


def load_baseline():
    rows = read_csv(BASELINE_FILE)

    symbols = []

    for row in rows:
        symbol = row["symbol"].strip()

        if not symbol:
            raise ValueError("Baseline contains an empty symbol.")

        symbols.append(symbol)

    if len(symbols) != 50:
        raise ValueError(
            f"Baseline must contain exactly 50 stocks. Found {len(symbols)}."
        )

    if len(set(symbols)) != 50:
        raise ValueError("Baseline contains duplicate symbols.")

    for row in rows:
        if row["baseline_date"] != BASELINE_DATE:
            raise ValueError(
                f"Unexpected baseline date: {row['baseline_date']}"
            )

    return set(symbols)


def load_events():
    rows = read_csv(EVENTS_FILE)

    required_columns = {
        "effective_date",
        "symbol",
        "company_name",
        "action",
        "event_type",
        "source_date",
        "source_reference",
        "notes",
    }

    if not rows:
        raise ValueError("Event ledger is empty.")

    missing = required_columns - set(rows[0].keys())

    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")

    for row in rows:
        datetime.strptime(row["effective_date"], "%Y-%m-%d")
        datetime.strptime(row["source_date"], "%Y-%m-%d")

        row["symbol"] = row["symbol"].strip()
        row["action"] = row["action"].strip().upper()

        if row["action"] not in {"INCLUSION", "EXCLUSION"}:
            raise ValueError(
                f"Invalid action for {row['symbol']}: {row['action']}"
            )

    rows.sort(key=lambda x: (x["effective_date"], x["symbol"], x["action"]))

    return rows


def daterange(start_date, end_date):
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def expected_count_for_date(date_str):
    for start_date, end_date in TEMPORARY_51_PERIODS:
        if start_date <= date_str <= end_date:
            return 51
    return 50


def build_membership():
    validate_files()

    constituents = load_baseline()
    events = load_events()

    events_by_date = defaultdict(list)

    for event in events:
        events_by_date[event["effective_date"]].append(event)

    output_rows = []

    start = datetime.strptime(BASELINE_DATE, "%Y-%m-%d").date()
    end = datetime.strptime(END_DATE, "%Y-%m-%d").date()

    event_dates = set(events_by_date.keys())

    checkpoint_results = []

    for current_date in daterange(start, end):
        date_str = current_date.isoformat()

        if date_str in events_by_date:
            before_count = len(constituents)

            for event in events_by_date[date_str]:
                symbol = event["symbol"]
                action = event["action"]

                if action == "INCLUSION":
                    if symbol in constituents:
                        raise ValueError(
                            f"Invalid inclusion on {date_str}: "
                            f"{symbol} already exists."
                        )

                    constituents.add(symbol)

                elif action == "EXCLUSION":
                    if symbol not in constituents:
                        raise ValueError(
                            f"Invalid exclusion on {date_str}: "
                            f"{symbol} is not currently a constituent."
                        )

                    constituents.remove(symbol)

            after_count = len(constituents)

            expected_count = expected_count_for_date(date_str)

            if after_count != expected_count:
                raise ValueError(
                    f"Invalid constituent count on {date_str}: "
                    f"before={before_count}, after={after_count}, "
                    f"expected={expected_count}"
                )

            checkpoint_results.append(
                (
                    date_str,
                    before_count,
                    after_count,
                    expected_count,
                )
            )

        expected_count = expected_count_for_date(date_str)

        if date_str >= BASELINE_DATE:
            if len(constituents) != expected_count:
                raise ValueError(
                    f"Daily membership count failure on {date_str}: "
                    f"{len(constituents)} instead of {expected_count}"
                )

            for symbol in sorted(constituents):
                output_rows.append(
                    {
                        "date": date_str,
                        "symbol": symbol,
                        "is_constituent": "1",
                    }
                )

    validate_output(output_rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "date",
                "symbol",
                "is_constituent",
            ],
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 60)
    print("NIFTY 50 MEMBERSHIP BUILDER")
    print("=" * 60)

    print(f"Baseline date: {BASELINE_DATE}")
    print(f"End date:      {END_DATE}")
    print(f"Output file:   {OUTPUT_FILE}")
    print()

    print("EVENT CHECKPOINTS")
    print("-" * 60)

    for date_str, before_count, after_count, expected_count in checkpoint_results:
        print(
            f"{date_str}: "
            f"{before_count} -> {after_count} "
            f"(expected {expected_count})"
        )

    print()
    print("VALIDATION")
    print("-" * 60)
    print("Baseline: PASS")
    print("Event actions: PASS")
    print("Daily constituent counts: PASS")
    print("Duplicate membership rows: PASS")
    print("Output uniqueness: PASS")

    print()
    print("STATUS: PASS")
    print(f"Rows written: {len(output_rows)}")


def validate_output(rows):
    seen = set()

    counts_by_date = defaultdict(int)

    for row in rows:
        key = (row["date"], row["symbol"])

        if key in seen:
            raise ValueError(
                f"Duplicate membership row: {key}"
            )

        seen.add(key)

        counts_by_date[row["date"]] += 1

        if row["is_constituent"] != "1":
            raise ValueError(
                f"Invalid is_constituent value: {row}"
            )

    for date_str, count in counts_by_date.items():
        expected = expected_count_for_date(date_str)

        if count != expected:
            raise ValueError(
                f"Output count mismatch on {date_str}: "
                f"{count} vs expected {expected}"
            )


if __name__ == "__main__":
    build_membership()