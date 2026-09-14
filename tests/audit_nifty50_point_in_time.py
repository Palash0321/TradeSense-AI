from pathlib import Path

import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MEMBERSHIP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_constituent_membership.csv"
)

PRICE_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_point_in_time_audit.csv"
)

NIFTY_SYMBOL = "^NSEI"

START_DATE = "2021-01-29"

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
    "DUMMYTATAM": [
        (
            pd.Timestamp("2025-10-14"),
            pd.Timestamp("2025-11-16"),
        ),
    ],
}

# yfinance end date is exclusive.
# Add one day beyond the desired current date.
END_DATE = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")


# =========================================================
# HELPERS
# =========================================================

def load_membership():
    """
    Load the frozen historical NIFTY 50 membership file.
    """

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    required_columns = {
        "date",
        "symbol",
        "is_constituent",
    }

    missing = required_columns - set(
        membership.columns
    )

    if missing:
        raise ValueError(
            f"Membership file missing columns: {sorted(missing)}"
        )

    membership["date"] = pd.to_datetime(
        membership["date"]
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    membership["is_constituent"] = (
        pd.to_numeric(
            membership["is_constituent"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    return membership


def load_nifty_trading_calendar():
    """
    Download NIFTY 50 index history.

    The NIFTY trading dates are used as the canonical
    market-session calendar.
    """

    print()
    print("=" * 70)
    print("DOWNLOADING NIFTY 50 TRADING CALENDAR")
    print("=" * 70)

    nifty = yf.download(
        NIFTY_SYMBOL,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
        progress=False,
    )

    if nifty.empty:
        raise RuntimeError(
            "Unable to download NIFTY 50 index data."
        )

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = (
            nifty.columns
            .get_level_values(0)
        )

    if "Close" not in nifty.columns:
        raise RuntimeError(
            "NIFTY 50 data does not contain Close."
        )

    nifty = nifty.dropna(
        subset=["Close"]
    ).copy()

    nifty.index = pd.to_datetime(
        nifty.index
    )

    trading_dates = pd.DatetimeIndex(
        nifty.index.normalize()
    ).unique()

    trading_dates = trading_dates.sort_values()

    print(
        f"NIFTY trading dates: "
        f"{len(trading_dates)}"
    )

    print(
        f"Coverage: "
        f"{trading_dates.min().date()} "
        f"-> "
        f"{trading_dates.max().date()}"
    )

    return trading_dates


def load_price_dates(symbol):
    """
    Load only the dates from a constituent price file.
    """

    price_file = (
        PRICE_DIR
        / f"{symbol}.csv"
    )

    if not price_file.exists():
        return None

    prices = pd.read_csv(
        price_file
    )

    if "date" not in prices.columns:
        raise ValueError(
            f"{symbol}: missing date column."
        )

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    prices = prices.dropna(
        subset=["date"]
    )

    return pd.DatetimeIndex(
        prices["date"]
        .dt.normalize()
        .unique()
    ).sort_values()


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print()
    print("=" * 70)
    print("NIFTY 50 POINT-IN-TIME PANEL AUDIT")
    print("=" * 70)

    # -----------------------------------------------------
    # Load membership
    # -----------------------------------------------------

    membership = load_membership()

    print(
        f"Membership rows: "
        f"{len(membership)}"
    )

    symbols = sorted(
        membership.loc[
            membership["is_constituent"],
            "symbol"
        ].unique()
    )

    print(
        f"Historical constituents: "
        f"{len(symbols)}"
    )

    # -----------------------------------------------------
    # Trading calendar
    # -----------------------------------------------------

    trading_dates = (
        load_nifty_trading_calendar()
    )

    # -----------------------------------------------------
    # Build audit
    # -----------------------------------------------------

    audit_rows = []

    for symbol in symbols:

        symbol_membership = (
            membership[
                membership["symbol"] == symbol
            ]
            .copy()
        )

        member_dates = set(
            symbol_membership.loc[
                symbol_membership["is_constituent"],
                "date"
            ].dt.normalize()
        )

        # Only actual NIFTY trading sessions.
        expected_dates = {
            date
            for date in trading_dates
            if date in member_dates
        }

        # Some historical constituents were temporarily present
        # in the index for corporate-action/index-adjustment
        # purposes without requiring ordinary tradable OHLCV data.
        special_periods = SPECIAL_NO_PRICE_PERIODS.get(
            symbol,
            [],
        )

        for period_start, period_end in special_periods:
            expected_dates = {
                date
                for date in expected_dates
                if not (
                    period_start
                    <= date
                    <= period_end
                )
            }

        price_dates = load_price_dates(
            symbol
        )

        if price_dates is None:

            if symbol in SPECIAL_INDEX_ENTITIES:
                audit_rows.append({
                    "symbol": symbol,
                    "membership_start": (
                        min(member_dates)
                        if member_dates
                        else pd.NaT
                    ),
                    "membership_end": (
                        max(member_dates)
                        if member_dates
                        else pd.NaT
                    ),
                    "expected_trading_days": 0,
                    "price_days": 0,
                    "covered_expected_days": 0,
                    "missing_expected_days": 0,
                    "first_price_date": pd.NaT,
                    "last_price_date": pd.NaT,
                    "status": "PASS",
                })

            else:
                audit_rows.append({
                    "symbol": symbol,
                    "membership_start": (
                        min(member_dates)
                        if member_dates
                        else pd.NaT
                    ),
                    "membership_end": (
                        max(member_dates)
                        if member_dates
                        else pd.NaT
                    ),
                    "expected_trading_days": (
                        len(expected_dates)
                    ),
                    "price_days": 0,
                    "covered_expected_days": 0,
                    "missing_expected_days": (
                        len(expected_dates)
                    ),
                    "first_price_date": pd.NaT,
                    "last_price_date": pd.NaT,
                    "status": "FAIL_NO_PRICE_FILE",
                })

            continue

        price_dates_set = set(
            price_dates
        )

        missing_dates = (
            expected_dates
            - price_dates_set
        )

        covered_dates = (
            expected_dates
            & price_dates_set
        )

        first_price_date = (
            price_dates.min()
            if len(price_dates) > 0
            else pd.NaT
        )

        last_price_date = (
            price_dates.max()
            if len(price_dates) > 0
            else pd.NaT
        )

        # -------------------------------------------------
        # Diagnose missing observations
        # -------------------------------------------------

        if not missing_dates:

            status = "PASS"

        else:

            # Missing dates that occur before the first
            # available price observation.
            pre_price_missing = {
                d
                for d in missing_dates
                if (
                    pd.notna(first_price_date)
                    and d < first_price_date
                )
            }

            # Missing dates after first price but before
            # last price are much more suspicious.
            internal_missing = {
                d
                for d in missing_dates
                if (
                    pd.notna(first_price_date)
                    and pd.notna(last_price_date)
                    and first_price_date
                    <= d
                    <= last_price_date
                )
            }

            post_price_missing = {
                d
                for d in missing_dates
                if (
                    pd.notna(last_price_date)
                    and d > last_price_date
                )
            }

            if internal_missing:

                status = (
                    "FAIL_INTERNAL_MISSING"
                )

            elif (
                len(pre_price_missing)
                == len(missing_dates)
            ):

                status = (
                    "WARN_BEFORE_FIRST_PRICE"
                )

            elif (
                len(post_price_missing)
                == len(missing_dates)
            ):

                status = (
                    "WARN_AFTER_LAST_PRICE"
                )

            else:

                status = (
                    "WARN_MIXED_MISSING"
                )

        audit_rows.append({
            "symbol": symbol,
            "membership_start": (
                min(member_dates)
                if member_dates
                else pd.NaT
            ),
            "membership_end": (
                max(member_dates)
                if member_dates
                else pd.NaT
            ),
            "expected_trading_days": (
                len(expected_dates)
            ),
            "price_days": len(price_dates),
            "covered_expected_days": (
                len(covered_dates)
            ),
            "missing_expected_days": (
                len(missing_dates)
            ),
            "first_price_date": first_price_date,
            "last_price_date": last_price_date,
            "status": status,
        })

    # -----------------------------------------------------
    # Save audit
    # -----------------------------------------------------

    audit = pd.DataFrame(
        audit_rows
    )

    audit = audit.sort_values(
        "symbol"
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("POINT-IN-TIME AUDIT SUMMARY")
    print("=" * 70)

    print()
    print("STATUS COUNTS")
    print("-" * 70)

    print(
        audit["status"]
        .value_counts()
        .sort_index()
    )

    print()
    print("TOTALS")
    print("-" * 70)

    print(
        "Expected member × trading days :",
        int(
            audit[
                "expected_trading_days"
            ].sum()
        )
    )

    print(
        "Covered member × trading days  :",
        int(
            audit[
                "covered_expected_days"
            ].sum()
        )
    )

    print(
        "Missing expected observations  :",
        int(
            audit[
                "missing_expected_days"
            ].sum()
        )
    )

    print()
    print("NON-PASS SYMBOLS")
    print("-" * 70)

    non_pass = audit[
        audit["status"] != "PASS"
    ]

    if non_pass.empty:

        print("None")

    else:

        print(
            non_pass[
                [
                    "symbol",
                    "membership_start",
                    "membership_end",
                    "expected_trading_days",
                    "covered_expected_days",
                    "missing_expected_days",
                    "first_price_date",
                    "last_price_date",
                    "status",
                ]
            ].to_string(index=False)
        )

    print()
    print("=" * 70)

    # -----------------------------------------------------
    # Important:
    # This audit is diagnostic.
    #
    # WARN cases require investigation.
    # FAIL_INTERNAL_MISSING is a hard failure.
    # -----------------------------------------------------

    if (
        audit["status"]
        .eq("FAIL_INTERNAL_MISSING")
        .any()
    ):

        print(
            "STATUS: FAIL"
        )

        raise SystemExit(1)

    elif (
        audit["status"]
        .ne("PASS")
        .any()
    ):

        print(
            "STATUS: REVIEW REQUIRED"
        )

    else:

        print(
            "STATUS: PASS"
        )


if __name__ == "__main__":
    main()