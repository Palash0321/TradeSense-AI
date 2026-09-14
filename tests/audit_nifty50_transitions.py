import csv
from collections import defaultdict


EVENTS_FILE = "data/nifty50_constituent_events.csv"
MEMBERSHIP_FILE = "data/nifty50_constituent_membership.csv"


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_events():
    return read_csv(EVENTS_FILE)


def load_membership():
    return read_csv(MEMBERSHIP_FILE)


def get_members_for_date(rows, date):
    return {
        row["symbol"]
        for row in rows
        if row["date"] == date and row["is_constituent"] == "1"
    }


def audit_transitions():
    events = load_events()
    membership = load_membership()

    events_by_date = defaultdict(list)

    for event in events:
        events_by_date[event["effective_date"]].append(event)

    print("=" * 70)
    print("NIFTY 50 CONSTITUENT TRANSITION AUDIT")
    print("=" * 70)
    print()

    previous_date = None

    for event_date in sorted(events_by_date):

        before_members = get_members_for_date(
            membership,
            event_date,
        )

        # Find the last membership date before event date.
        dates = sorted(
            {
                row["date"]
                for row in membership
                if row["date"] < event_date
            }
        )

        if not dates:
            raise ValueError(
                f"No pre-event membership available for {event_date}"
            )

        previous_date = dates[-1]

        before_members = get_members_for_date(
            membership,
            previous_date,
        )

        after_members = get_members_for_date(
            membership,
            event_date,
        )

        actual_out = sorted(before_members - after_members)
        actual_in = sorted(after_members - before_members)

        expected_out = sorted(
            event["symbol"]
            for event in events_by_date[event_date]
            if event["action"] == "EXCLUSION"
        )

        expected_in = sorted(
            event["symbol"]
            for event in events_by_date[event_date]
            if event["action"] == "INCLUSION"
        )

        print("-" * 70)
        print(f"DATE: {event_date}")
        print(f"Previous membership date: {previous_date}")

        print()
        print("EXPECTED OUT:")
        for symbol in expected_out:
            print(f"  {symbol}")

        print("ACTUAL OUT:")
        for symbol in actual_out:
            print(f"  {symbol}")

        print()
        print("EXPECTED IN:")
        for symbol in expected_in:
            print(f"  {symbol}")

        print("ACTUAL IN:")
        for symbol in actual_in:
            print(f"  {symbol}")

        if actual_out != expected_out:
            raise ValueError(
                f"OUT mismatch on {event_date}: "
                f"expected={expected_out}, actual={actual_out}"
            )

        if actual_in != expected_in:
            raise ValueError(
                f"IN mismatch on {event_date}: "
                f"expected={expected_in}, actual={actual_in}"
            )

        print()
        print("TRANSITION: PASS")

    print()
    print("=" * 70)
    print("OVERALL TRANSITION AUDIT: PASS")
    print("=" * 70)


if __name__ == "__main__":
    audit_transitions()