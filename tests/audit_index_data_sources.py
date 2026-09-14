"""
TradeSense-AI
Index Data Source Audit

Purpose:
    Verify whether the planned Indian index instruments have usable
    daily OHLC data before building the dedicated index backtest layer.

IMPORTANT:
    This is an audit only.
    It does NOT modify production code.
    It does NOT modify historical trades.
    It does NOT modify holdout data.
"""

import sys

import pandas as pd
import yfinance as yf


INDEXES = {
    "NIFTY_50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK_NIFTY": "^NSEBANK",
    "MIDSELECT": "NIFTYMIDSELECT",
}


START_DATE = "2021-01-01"
END_DATE = "2026-09-12"


def audit_index(name, ticker):
    print()
    print("=" * 100)
    print(f"{name} | {ticker}")
    print("=" * 100)

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        print(f"DOWNLOAD ERROR: {exc}")
        return {
            "index": name,
            "ticker": ticker,
            "status": "DOWNLOAD_ERROR",
            "rows": 0,
            "start": None,
            "end": None,
        }

    if df is None or df.empty:
        print("STATUS: NO_DATA")
        return {
            "index": name,
            "ticker": ticker,
            "status": "NO_DATA",
            "rows": 0,
            "start": None,
            "end": None,
        }

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        print(f"STATUS: MISSING_COLUMNS {missing}")
        return {
            "index": name,
            "ticker": ticker,
            "status": "MISSING_COLUMNS",
            "rows": len(df),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        }

    df = df.dropna(subset=required).copy()

    invalid_ohlc = (
        (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Close"] <= 0)
        | (df["High"] < df["Low"])
    )

    invalid_count = int(invalid_ohlc.sum())

    duplicate_dates = int(df.index.duplicated().sum())

    print(f"Rows:              {len(df)}")
    print(f"Start:             {df.index.min()}")
    print(f"End:               {df.index.max()}")
    print(f"Invalid OHLC rows: {invalid_count}")
    print(f"Duplicate dates:   {duplicate_dates}")

    print()
    print("Latest rows:")
    print(
        df[
            ["Open", "High", "Low", "Close"]
        ].tail(5).to_string()
    )

    if invalid_count == 0 and duplicate_dates == 0:
        status = "PASS"
    else:
        status = "FAIL"

    print()
    print(f"STATUS: {status}")

    return {
        "index": name,
        "ticker": ticker,
        "status": status,
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
    }


def main():
    print("=" * 100)
    print("TRADESENSE-AI — INDIAN INDEX DATA SOURCE AUDIT")
    print("=" * 100)
    print(f"Date range: {START_DATE} -> {END_DATE}")

    results = []

    for name, ticker in INDEXES.items():
        results.append(
            audit_index(name, ticker)
        )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    summary = pd.DataFrame(results)

    print(summary.to_string(index=False))

    print()
    print("=" * 100)

    failed = summary[
        summary["status"] != "PASS"
    ]

    if failed.empty:
        print("OVERALL STATUS: PASS")
        print()
        print(
            "All four planned index data sources returned "
            "usable OHLC data."
        )
    else:
        print("OVERALL STATUS: ATTENTION REQUIRED")
        print()
        print(
            "One or more index data sources require a "
            "dedicated data-source solution before backtesting."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()