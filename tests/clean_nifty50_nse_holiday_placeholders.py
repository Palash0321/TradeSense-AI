from pathlib import Path
import shutil

import pandas as pd


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
    PRICE_DIR
    / "nse_holiday_cleanup_audit.csv"
)

BACKUP_DIR = (
    PRICE_DIR
    / "_pre_holiday_cleanup_backup"
)

SPECIAL_SYMBOLS = {
    "DUMMYTATAM",
    "NIFTY_INDEX",
}


# =========================================================
# CONFIRMED NSE EQUITY HOLIDAYS — 2026
# =========================================================
#
# Source:
# NSE India — Market Timings & Holidays
#
# We only remove a row when ALL of the following are true:
#
# 1. Date is an official NSE equity-market holiday.
# 2. Existing volume <= 0.
# 3. OHLC are identical.
# 4. OHLC equals the previous trading row's close.
#
# This prevents us from deleting a legitimate anomalous
# trading row merely because its date appears in the holiday
# calendar.
# =========================================================

NSE_EQUITY_HOLIDAYS_2026 = {
    pd.Timestamp("2026-01-15"),
    pd.Timestamp("2026-01-26"),
    pd.Timestamp("2026-02-19"),
    pd.Timestamp("2026-03-03"),
    pd.Timestamp("2026-03-19"),
    pd.Timestamp("2026-03-26"),
    pd.Timestamp("2026-03-31"),
    pd.Timestamp("2026-04-01"),
    pd.Timestamp("2026-04-03"),
    pd.Timestamp("2026-04-14"),
    pd.Timestamp("2026-05-01"),
    pd.Timestamp("2026-05-28"),
    pd.Timestamp("2026-06-26"),
    pd.Timestamp("2026-08-26"),
    pd.Timestamp("2026-09-14"),
    pd.Timestamp("2026-10-02"),
    pd.Timestamp("2026-10-20"),
    pd.Timestamp("2026-11-10"),
    pd.Timestamp("2026-11-24"),
    pd.Timestamp("2026-12-25"),
}


# =========================================================
# HELPERS
# =========================================================

def values_match(left, right, tolerance=1e-8):
    """
    Compare numeric values safely.
    """
    if pd.isna(left) or pd.isna(right):
        return False

    return abs(float(left) - float(right)) <= tolerance


def is_holiday_placeholder(data, row_index):
    """
    Confirm that a zero-volume holiday row is a synthetic
    carry-forward placeholder.

    Required signature:

        volume <= 0
        open == high == low == close
        current close == previous row close
    """

    row = data.loc[row_index]

    repair_date = pd.Timestamp(row["date"]).normalize()

    if repair_date not in NSE_EQUITY_HOLIDAYS_2026:
        return False

    if pd.isna(row["volume"]) or row["volume"] > 0:
        return False

    o = row["open"]
    h = row["high"]
    l = row["low"]
    c = row["close"]

    if not all(
        values_match(value, c)
        for value in [o, h, l]
    ):
        return False

    # Find the immediately preceding row.
    position = data.index.get_loc(row_index)

    if position == 0:
        return False

    previous_index = data.index[position - 1]
    previous_close = data.loc[
        previous_index,
        "close",
    ]

    if not values_match(c, previous_close):
        return False

    return True


# =========================================================
# MAIN CLEANUP
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "NIFTY 50 — NSE HOLIDAY PLACEHOLDER CLEANUP"
    )
    print("=" * 70)

    print()
    print(
        "This script removes only confirmed NSE holiday "
        "placeholder rows."
    )

    print(
        "It does NOT repair trading-day volume."
    )

    print(
        "It does NOT touch Model 10 files."
    )

    print()

    price_files = sorted(
        PRICE_DIR.glob("*.csv")
    )

    candidates = []

    for price_file in price_files:

        if price_file.name in {
            AUDIT_FILE.name,
            "nse_repair_audit.csv",
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

        for row_index in data.index:

            if is_holiday_placeholder(
                data,
                row_index,
            ):
                row = data.loc[row_index]

                candidates.append(
                    {
                        "symbol": symbol,
                        "date": row["date"],
                        "old_open": row["open"],
                        "old_high": row["high"],
                        "old_low": row["low"],
                        "old_close": row["close"],
                        "old_volume": row["volume"],
                    }
                )

    print(
        f"Holiday placeholders identified: "
        f"{len(candidates)}"
    )

    print(
        f"Affected symbols: "
        f"{len(set(row['symbol'] for row in candidates))}"
    )

    print()

    if not candidates:
        print(
            "No confirmed holiday placeholders found."
        )
        print()
        print("STATUS: NOTHING TO CLEAN")
        return

    # -----------------------------------------------------
    # Backup
    # -----------------------------------------------------

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    affected_symbols = sorted(
        set(
            row["symbol"]
            for row in candidates
        )
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

        if not destination.exists():

            shutil.copy2(
                source,
                destination,
            )

    print(
        f"Backups available: "
        f"{len(affected_symbols)}"
    )

    print()

    # -----------------------------------------------------
    # Remove placeholders
    # -----------------------------------------------------

    audit_rows = []

    removed_count = 0

    candidates_by_symbol = {}

    for candidate in candidates:

        candidates_by_symbol.setdefault(
            candidate["symbol"],
            [],
        ).append(candidate)

    for symbol in affected_symbols:

        price_file = (
            PRICE_DIR
            / f"{symbol}.csv"
        )

        data = pd.read_csv(
            price_file
        )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="raise",
        ).dt.normalize()

        candidate_dates = {
            candidate["date"]
            for candidate in
            candidates_by_symbol[symbol]
        }

        remove_mask = (
            data["date"].isin(
                candidate_dates
            )
        )

        removed_rows = data[
            remove_mask
        ].copy()

        data = data[
            ~remove_mask
        ].copy()

        data = (
            data
            .sort_values("date")
            .drop_duplicates(
                "date",
                keep="last",
            )
            .reset_index(drop=True)
        )

        data.to_csv(
            price_file,
            index=False,
        )

        for candidate in candidates_by_symbol[symbol]:

            audit_rows.append(
                {
                    "symbol": symbol,
                    "date": candidate["date"],
                    "status":
                        "REMOVED_NSE_HOLIDAY_PLACEHOLDER",
                    "old_open":
                        candidate["old_open"],
                    "old_high":
                        candidate["old_high"],
                    "old_low":
                        candidate["old_low"],
                    "old_close":
                        candidate["old_close"],
                    "old_volume":
                        candidate["old_volume"],
                }
            )

            removed_count += 1

        print(
            f"{symbol}: removed "
            f"{len(removed_rows)} holiday row(s)"
        )

    # -----------------------------------------------------
    # Audit
    # -----------------------------------------------------

    audit = pd.DataFrame(
        audit_rows
    )

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
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("HOLIDAY CLEANUP SUMMARY")
    print("=" * 70)

    print(
        f"Holiday placeholders removed: "
        f"{removed_count}"
    )

    print(
        f"Affected symbols: "
        f"{len(affected_symbols)}"
    )

    print(
        f"Audit file: "
        f"{AUDIT_FILE}"
    )

    print(
        f"Backup directory: "
        f"{BACKUP_DIR}"
    )

    print()

    print(
        "STATUS: PASS"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()