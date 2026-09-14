"""
TradeSense-AI
EARLY-ADVERSE OHLC REPLAY FORENSIC

RESEARCH ONLY.
DO NOT DEPLOY.

Purpose
-------
Reconstruct early adverse excursion directly from historical OHLCV
bars for the frozen 106-trade dataset.

Tests:
    Windows   : 3, 5, 7, 10 bars
    Thresholds: 0.15R ... 0.40R

Critical integrity checks:
    1. Reconstructed bars 1-5 must agree with stored trade fields.
    2. Only post-entry bars are used.
    3. No final MFE/MAE or exit information is used to classify a trade.
    4. Frozen historical trade dataset is READ ONLY.
    5. Production code is NOT modified.
    6. Holdout dataset is NOT modified.
"""

import csv
import math
import os
from collections import defaultdict

import pandas as pd

from app.services.backtest_service import BacktestService


# =====================================================================
# CONFIGURATION
# =====================================================================

HISTORICAL_TRADES_FILE = "tests/output/tradesense_trades.csv"

OUTPUT_DIR = "tests/output/reconciliation"

MATRIX_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_ohlc_replay_matrix.csv"
)

BAR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_ohlc_replay_bars.csv"
)

VALIDATION_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_ohlc_replay_validation.csv"
)

WINDOWS = [3, 5, 7, 10]

THRESHOLDS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

EXPECTED_TRADE_COUNT = 106

TOLERANCE = 1e-6

# Stored early-adverse values are rounded to 2 decimals
# by production before being written to the trade dataset.
VALIDATION_TOLERANCE = 0.005001


# =====================================================================
# HELPERS
# =====================================================================

def safe_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_date(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text[:10]


def normalize_symbol(value):
    if value is None:
        return None

    return str(value).strip()


def almost_equal(a, b, tolerance=TOLERANCE):
    if a is None or b is None:
        return False

    return abs(float(a) - float(b)) <= tolerance


def calculate_profit_factor(trades):
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:

        r = safe_float(
            trade.get("r_multiple")
        )

        if r is None:
            continue

        if r > 0:
            gross_profit += r

        elif r < 0:
            gross_loss += abs(r)

    if gross_loss == 0:

        if gross_profit > 0:
            return float("inf")

        return 0.0

    return gross_profit / gross_loss


def calculate_metrics(trades):

    if not trades:

        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
        }

    r_values = []

    for trade in trades:

        r = safe_float(
            trade.get("r_multiple")
        )

        if r is not None:
            r_values.append(r)

    if not r_values:

        return {
            "trades": len(trades),
            "wins": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
        }

    wins = sum(
        1
        for r in r_values
        if r > 0
    )

    total_r = sum(r_values)

    return {
        "trades": len(r_values),
        "wins": wins,
        "win_rate": wins / len(r_values) * 100.0,
        "total_r": total_r,
        "avg_r": total_r / len(r_values),
        "profit_factor": calculate_profit_factor(trades),
    }


def get_trade_key(trade):

    return (
        normalize_symbol(trade.get("_symbol")),
        normalize_date(trade.get("entry_date")),
        str(trade.get("direction", "")).strip().upper(),
    )


def calculate_production_atr(df, columns):
    """
    Reconstruct the exact ATR calculation used by production.

    Production uses a 14-period simple rolling mean of True Range:

        previous_close = Close.shift(1)

        true_range = max(
            High - Low,
            abs(High - previous_close),
            abs(Low - previous_close)
        )

        ATR = true_range.rolling(14).mean()
    """

    previous_close = df[columns["Close"]].shift(1)

    true_range = pd.concat(
        [
            df[columns["High"]] - df[columns["Low"]],
            (
                df[columns["High"]]
                - previous_close
            ).abs(),
            (
                df[columns["Low"]]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(14).mean()


def adverse_r_from_bar(
    direction,
    entry_price,
    risk_per_unit,
    low_price,
    high_price,
):
    """
    Calculate adverse excursion for ONE post-entry OHLC bar.

    LONG:
        adverse movement is entry - bar low.

    SHORT:
        adverse movement is bar high - entry.

    Values below zero are clipped to zero because favorable movement
    is not adverse excursion.
    """

    direction = direction.upper()

    if direction == "LONG":

        adverse_price = entry_price - low_price

    elif direction == "SHORT":

        adverse_price = high_price - entry_price

    else:

        raise ValueError(
            f"Unsupported direction: {direction}"
        )

    if adverse_price < 0:
        adverse_price = 0.0

    return adverse_price / risk_per_unit


def get_ohlc_columns(df):

    mapping = {}

    for column in df.columns:

        text = str(column).strip().lower()

        if text == "open":
            mapping["Open"] = column

        elif text == "high":
            mapping["High"] = column

        elif text == "low":
            mapping["Low"] = column

        elif text == "close":
            mapping["Close"] = column

        elif text == "volume":
            mapping["Volume"] = column

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in mapping
    ]

    if missing:

        raise RuntimeError(
            f"Missing OHLC columns: {missing}"
        )

    return mapping


# =====================================================================
# LOAD FROZEN TRADES
# =====================================================================

print()
print("=" * 115)
print("TRADESENSE-AI — EARLY-ADVERSE OHLC REPLAY FORENSIC")
print("=" * 115)
print()
print("RESEARCH ONLY — DO NOT DEPLOY")
print()

with open(
    HISTORICAL_TRADES_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    trades = list(reader)


print(
    f"Frozen trades loaded: {len(trades)}"
)

if len(trades) != EXPECTED_TRADE_COUNT:

    raise RuntimeError(
        "Frozen historical trade count is not 106. "
        "Stopping to protect research integrity."
    )


# =====================================================================
# BUILD SYMBOL → TRADE LIST
# =====================================================================

trades_by_symbol = defaultdict(list)

for trade in trades:

    symbol = normalize_symbol(
        trade.get("_symbol")
    )

    if not symbol:
        raise RuntimeError(
            "Trade has no _symbol."
        )

    trades_by_symbol[symbol].append(trade)


print()
print("Symbols:")

for symbol in sorted(trades_by_symbol):

    print(
        f"  {symbol}: "
        f"{len(trades_by_symbol[symbol])} trades"
    )


# =====================================================================
# LOAD DATA WITH ORIGINAL FROZEN REFERENCE CONTEXT
# =====================================================================
#
# The frozen historical dataset was reproduced exactly using:
#
#   start = 2021-09-07
#   end   = 2026-09-08
#
# yfinance treats end as exclusive, so the final market bar is
# 2026-09-07.
#
# For OHLC replay we need bars BEFORE the first trade as well as
# all bars after each entry.
#
# We therefore load the same reference period plus enough historical
# context to safely locate every entry.
# =====================================================================

data_by_symbol = {}

print()
print("=" * 115)
print("LOADING OHLC DATA")
print("=" * 115)

for symbol in sorted(trades_by_symbol):

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    # Use the exact frozen reference boundary.
    data = service.load_data(
        start_date="2021-09-07",
        end_date="2026-09-08",
    )

    if data is None or data.empty:

        raise RuntimeError(
            f"No data returned for {symbol}"
        )

    data = data.copy()

    # Normalize index to date.
    data.index = pd.to_datetime(
        data.index
    ).tz_localize(None)

    data = data.sort_index()

    data_by_symbol[symbol] = data

    print(
        f"{symbol}: "
        f"{data.index.min().date()} → "
        f"{data.index.max().date()} | "
        f"{len(data)} rows"
    )


# =====================================================================
# RECONSTRUCT EARLY ADVERSE BARS
# =====================================================================

print()
print("=" * 115)
print("REPLAYING POST-ENTRY BARS")
print("=" * 115)

replayed_trade_data = {}

bar_rows = []

validation_rows = []

fatal_validation_errors = []


for symbol in sorted(trades_by_symbol):

    df = data_by_symbol[symbol]

    columns = get_ohlc_columns(df)

    # -------------------------------------------------------------
    # Reconstruct the exact production ATR series.
    #
    # Production calculates ATR from the OHLC history using:
    #   True Range
    #   14-period simple rolling mean
    # -------------------------------------------------------------

    df = df.copy()

    df["_production_atr"] = calculate_production_atr(
        df,
        columns,
    )

    # Create date → positional row lookup.
    date_to_position = {}

    for position, timestamp in enumerate(df.index):

        date_to_position[
            timestamp.strftime("%Y-%m-%d")
        ] = position

    print()
    print(f"Processing {symbol}...")

    for trade_index, trade in enumerate(
        trades_by_symbol[symbol],
        start=1
    ):

        key = get_trade_key(trade)

        entry_date = normalize_date(
            trade.get("entry_date")
        )

        direction = str(
            trade.get("direction", "")
        ).strip().upper()


        if not entry_date:
            fatal_validation_errors.append(
                f"{key}: missing entry_date"
            )
            continue

        if direction not in {"LONG", "SHORT"}:
            fatal_validation_errors.append(
                f"{key}: invalid direction {direction}"
            )
            continue


        if entry_date not in date_to_position:

            fatal_validation_errors.append(
                f"{key}: entry date not found in OHLC data"
            )

            continue

        entry_position = date_to_position[
            entry_date
        ]

        # -------------------------------------------------------------
        # Reconstruct the exact production entry price and risk.
        #
        # The frozen CSV stores rounded values. Production itself uses
        # the full-precision execution price and full-precision ATR.
        # -------------------------------------------------------------

        entry_row = df.iloc[entry_position]

        entry_open = safe_float(
            entry_row[columns["Open"]]
        )

        if entry_open is None:
            fatal_validation_errors.append(
                f"{key}: missing entry-date OPEN"
            )
            continue

        # Production default slippage = 0.10%.
        slippage_rate = 0.10 / 100.0

        if direction == "LONG":
            entry_price = (
                entry_open
                * (1.0 + slippage_rate)
            )
        else:
            entry_price = (
                entry_open
                * (1.0 - slippage_rate)
            )

        # Production takes ATR from the signal bar.
        #
        # Because entry_date is the next bar after the signal bar:
        #
        #     signal_position = entry_position - 1
        #
        signal_position = entry_position - 1

        if signal_position < 0:
            fatal_validation_errors.append(
                f"{key}: no signal bar available before entry"
            )
            continue

        entry_atr_exact = safe_float(
            df.iloc[signal_position]["_production_atr"]
        )

        if entry_atr_exact is None:
            fatal_validation_errors.append(
                f"{key}: unable to reconstruct production ATR"
            )
            continue

        if entry_atr_exact <= 0:
            fatal_validation_errors.append(
                f"{key}: reconstructed ATR is non-positive"
            )
            continue

        # Frozen historical trades were generated with
        # initial_stop_atr = 3.5.
        initial_stop_atr = 3.5

        risk_per_unit = (
            initial_stop_atr
            * entry_atr_exact
        )

        # -------------------------------------------------------------
        # Replay bars managed after entry.
        #
        # The production engine enters at the OPEN of entry_date
        # (the next bar after the signal bar).
        #
        # On the next loop iteration, that same entry-date bar is
        # already an open position and bars_in_trade becomes 1.
        #
        # Therefore:
        #   bar_1 = entry_position
        #   bar_2 = entry_position + 1
        #   bar_3 = entry_position + 2
        #   ...
        # -------------------------------------------------------------

        per_bar = {}

        max_adverse = 0.0

        for bar_number in range(1, 11):

            position = entry_position + bar_number - 1

            if position >= len(df):

                break

            row = df.iloc[position]

            high_price = safe_float(
                row[columns["High"]]
            )

            low_price = safe_float(
                row[columns["Low"]]
            )

            if high_price is None or low_price is None:

                fatal_validation_errors.append(
                    f"{key}: missing OHLC at bar {bar_number}"
                )

                break

            adverse_r = adverse_r_from_bar(
                direction=direction,
                entry_price=entry_price,
                risk_per_unit=risk_per_unit,
                low_price=low_price,
                high_price=high_price,
            )

            max_adverse = max(
                max_adverse,
                adverse_r
            )

            bar_date = row.name.strftime(
                "%Y-%m-%d"
            )

            per_bar[
                bar_number
            ] = max_adverse

            bar_rows.append(
                {
                    "_symbol": symbol,
                    "entry_date": entry_date,
                    "direction": direction,
                    "bar_number": bar_number,
                    "bar_date": bar_date,
                    "entry_price": entry_price,
                    "risk_per_unit": risk_per_unit,
                    "bar_high": high_price,
                    "bar_low": low_price,
                    "bar_adverse_r": adverse_r,
                    "cumulative_max_adverse_r": max_adverse,
                }
            )

        replayed_trade_data[key] = {
            "trade": trade,
            "per_bar": per_bar,
        }

        # -------------------------------------------------------------
        # Validate stored production fields for bars 1-5.
        # -------------------------------------------------------------

        for bar_number in range(1, 6):

            stored_field = (
                f"early_adverse_r_bar_{bar_number}"
            )

            stored_value = safe_float(
                trade.get(stored_field)
            )

            replay_value = per_bar.get(
                bar_number
            )

            if stored_value is None:

                validation_rows.append(
                    {
                        "_symbol": symbol,
                        "entry_date": entry_date,
                        "direction": direction,
                        "bar_number": bar_number,
                        "stored_value": None,
                        "replayed_value": replay_value,
                        "difference": None,
                        "status": "MISSING_STORED_VALUE",
                    }
                )

                continue

            if replay_value is None:

                validation_rows.append(
                    {
                        "_symbol": symbol,
                        "entry_date": entry_date,
                        "direction": direction,
                        "bar_number": bar_number,
                        "stored_value": stored_value,
                        "replayed_value": None,
                        "difference": None,
                        "status": "MISSING_REPLAY_VALUE",
                    }
                )

                continue

            difference = (
                replay_value
                - stored_value
            )

            if (
                abs(
                    replay_value
                    - stored_value
                )
                <= VALIDATION_TOLERANCE
            ):

                status = "MATCH"

            else:

                status = "MISMATCH"

                fatal_validation_errors.append(
                    f"{key} "
                    f"bar {bar_number}: "
                    f"stored={stored_value:.8f}, "
                    f"replayed={replay_value:.8f}, "
                    f"diff={difference:.8f}"
                )

            validation_rows.append(
                {
                    "_symbol": symbol,
                    "entry_date": entry_date,
                    "direction": direction,
                    "bar_number": bar_number,
                    "stored_value": stored_value,
                    "replayed_value": replay_value,
                    "difference": difference,
                    "status": status,
                }
            )


# =====================================================================
# VALIDATION RESULT
# =====================================================================

print()
print("=" * 115)
print("BAR 1-5 INTEGRITY VALIDATION")
print("=" * 115)

match_count = sum(
    1
    for row in validation_rows
    if row["status"] == "MATCH"
)

mismatch_count = sum(
    1
    for row in validation_rows
    if row["status"] == "MISMATCH"
)

missing_count = sum(
    1
    for row in validation_rows
    if row["status"] != "MATCH"
    and row["status"] != "MISMATCH"
)

print(
    f"Validation rows : {len(validation_rows)}"
)

print(
    f"Matches         : {match_count}"
)

print(
    f"Mismatches      : {mismatch_count}"
)

print(
    f"Missing         : {missing_count}"
)

expected_validation_rows = (
    len(trades) * 5
)

expected_replay_trades = (
    len(trades)
)

print()
print(
    f"Expected replay trades    : "
    f"{expected_replay_trades}"
)

print(
    f"Actual replay trades      : "
    f"{len(replayed_trade_data)}"
)

print(
    f"Expected validation rows  : "
    f"{expected_validation_rows}"
)

print(
    f"Actual validation rows    : "
    f"{len(validation_rows)}"
)

# -------------------------------------------------------------
# HARD INTEGRITY GUARDS
# -------------------------------------------------------------

if len(replayed_trade_data) != expected_replay_trades:

    raise RuntimeError(
        "CRITICAL REPLAY FAILURE: "
        f"expected {expected_replay_trades} reconstructed trades, "
        f"got {len(replayed_trade_data)}."
    )

if len(validation_rows) != expected_validation_rows:

    raise RuntimeError(
        "CRITICAL VALIDATION FAILURE: "
        f"expected {expected_validation_rows} validation rows, "
        f"got {len(validation_rows)}."
    )

if len(bar_rows) == 0:

    raise RuntimeError(
        "CRITICAL REPLAY FAILURE: "
        "zero OHLC replay bars were generated."
    )

if mismatch_count > 0:

    # -------------------------------------------------------------
    # FORENSIC FAILURE DUMP
    #
    # Save the complete in-memory validation table BEFORE raising.
    # This is diagnostic only. It does not alter validation logic,
    # replay logic, or any production/historical dataset.
    # -------------------------------------------------------------

    validation_output_path = os.path.abspath(
        VALIDATION_OUTPUT
    )

    print()
    print("=" * 115)
    print("FORENSIC VALIDATION FAILURE DETAILS")
    print("=" * 115)

    print(
        f"Validation output: {validation_output_path}"
    )

    print(
        f"Rows being written: {len(validation_rows)}"
    )

    with open(
        VALIDATION_OUTPUT,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        validation_writer = csv.DictWriter(
            file,
            fieldnames=[
                "_symbol",
                "entry_date",
                "direction",
                "bar_number",
                "stored_value",
                "replayed_value",
                "difference",
                "status",
            ]
        )

        validation_writer.writeheader()
        validation_writer.writerows(
            validation_rows
        )

    print(
        f"Validation CSV written: {os.path.getsize(validation_output_path)} bytes"
    )

    print()
    print("NON-MATCHING VALIDATION ROWS")
    print("-" * 115)

    for row in validation_rows:

        if row["status"] != "MATCH":

            print(
                f"{row['_symbol']} | "
                f"{row['entry_date']} | "
                f"{row['direction']} | "
                f"bar={row['bar_number']} | "
                f"stored={row['stored_value']} | "
                f"replayed={row['replayed_value']} | "
                f"diff={row['difference']} | "
                f"status={row['status']}"
            )

    print()
    print(
        "Stopping before matrix analysis."
    )

    raise RuntimeError(
        "CRITICAL VALIDATION FAILURE: "
        f"{mismatch_count} stored/replayed early-adverse values differ."
    )

if validation_rows:

    print(
        f"Expected checks: "
        f"{len(trades) * 5}"
    )

if mismatch_count > 0:

    print()
    print(
        "!!! CRITICAL: STORED EARLY-ADVERSE VALUES "
        "DO NOT MATCH OHLC REPLAY !!!"
    )

    print(
        "Stopping before matrix analysis."
    )

    raise RuntimeError(
        "Early-adverse OHLC replay integrity validation failed."
    )


# =====================================================================
# CHECK FOR ENOUGH BARS
# =====================================================================

insufficient_10_bar_trades = []

for key, payload in replayed_trade_data.items():

    per_bar = payload["per_bar"]

    if len(per_bar) < 10:

        insufficient_10_bar_trades.append(
            key
        )


print()

print(
    f"Trades with fewer than 10 post-entry bars: "
    f"{len(insufficient_10_bar_trades)}"
)

if insufficient_10_bar_trades:

    print(
        "These trades will be excluded ONLY from "
        "7/10-bar cells when their required bars "
        "are unavailable."
    )


# =====================================================================
# BASELINE
# =====================================================================

baseline_metrics = calculate_metrics(
    trades
)

baseline_total_r = baseline_metrics[
    "total_r"
]


# =====================================================================
# MATRIX
# =====================================================================

matrix_rows = []

print()
print("=" * 115)
print("FULL OHLC REPLAY MATRIX")
print("=" * 115)
print()

print(
    f"{'Window':>8} "
    f"{'Threshold':>11} "
    f"{'Flagged':>9} "
    f"{'Retained':>10} "
    f"{'Win%':>9} "
    f"{'TotalR':>11} "
    f"{'DeltaR':>11} "
    f"{'PF':>8} "
    f"{'Years+':>8} "
    f"{'Years-':>8}"
)

print("-" * 115)


for window in WINDOWS:

    for threshold in THRESHOLDS:

        retained = []
        flagged = []

        excluded_for_insufficient_data = []

        for trade in trades:

            key = get_trade_key(trade)

            payload = replayed_trade_data.get(
                key
            )

            if payload is None:

                excluded_for_insufficient_data.append(
                    trade
                )

                continue

            per_bar = payload["per_bar"]

            if len(per_bar) < window:

                excluded_for_insufficient_data.append(
                    trade
                )

                continue

            early_adverse = per_bar[
                window
            ]

            if early_adverse >= threshold:

                flagged.append(trade)

            else:

                retained.append(trade)

        metrics = calculate_metrics(
            retained
        )

        delta_r = (
            metrics["total_r"]
            - baseline_total_r
        )

        # -------------------------------------------------------------
        # Yearly robustness
        # -------------------------------------------------------------

        yearly = defaultdict(
            lambda: {
                "baseline": 0.0,
                "filtered": 0.0,
            }
        )

        for trade in trades:

            year = normalize_date(
                trade.get("entry_date")
            )

            if not year:
                continue

            year = int(year[:4])

            r = safe_float(
                trade.get("r_multiple")
            ) or 0.0

            yearly[year]["baseline"] += r

        for trade in retained:

            year = normalize_date(
                trade.get("entry_date")
            )

            if not year:
                continue

            year = int(year[:4])

            r = safe_float(
                trade.get("r_multiple")
            ) or 0.0

            yearly[year]["filtered"] += r

        improved_years = 0
        worsened_years = 0
        unchanged_years = 0

        for year in sorted(yearly):

            yearly_delta = (
                yearly[year]["filtered"]
                - yearly[year]["baseline"]
            )

            if yearly_delta > TOLERANCE:

                improved_years += 1

            elif yearly_delta < -TOLERANCE:

                worsened_years += 1

            else:

                unchanged_years += 1

        if delta_r > TOLERANCE:

            status = "IMPROVED"

        elif delta_r < -TOLERANCE:

            status = "WORSE"

        else:

            status = "UNCHANGED"

        row = {
            "window_bars": window,
            "threshold_r": threshold,
            "flagged_trades": len(flagged),
            "retained_trades": len(retained),
            "excluded_insufficient_bars": len(
                excluded_for_insufficient_data
            ),
            "wins": metrics["wins"],
            "win_rate": metrics["win_rate"],
            "total_r": metrics["total_r"],
            "avg_r": metrics["avg_r"],
            "profit_factor": metrics["profit_factor"],
            "delta_r": delta_r,
            "improved_years": improved_years,
            "worsened_years": worsened_years,
            "unchanged_years": unchanged_years,
            "status": status,
        }

        matrix_rows.append(row)

        pf_text = (
            "INF"
            if math.isinf(
                metrics["profit_factor"]
            )
            else f"{metrics['profit_factor']:.3f}"
        )

        print(
            f"{window:>8} "
            f"{threshold:>11.2f} "
            f"{len(flagged):>9} "
            f"{len(retained):>10} "
            f"{metrics['win_rate']:>8.2f}% "
            f"{metrics['total_r']:>10.2f} "
            f"{delta_r:>10.2f} "
            f"{pf_text:>8} "
            f"{improved_years:>8} "
            f"{worsened_years:>8}"
        )


# =====================================================================
# YEAR × PARAMETER DETAILED TABLE
# =====================================================================

print()
print("=" * 115)
print("YEAR-BY-YEAR PARAMETER ROBUSTNESS")
print("=" * 115)

year_detail_rows = []

years = sorted(
    {
        int(
            normalize_date(
                trade.get("entry_date")
            )[:4]
        )
        for trade in trades
        if normalize_date(
            trade.get("entry_date")
        )
    }
)

for window in WINDOWS:

    for threshold in THRESHOLDS:

        retained = []

        for trade in trades:

            key = get_trade_key(trade)

            payload = replayed_trade_data.get(
                key
            )

            if payload is None:
                continue

            per_bar = payload["per_bar"]

            if len(per_bar) < window:
                continue

            if per_bar[window] < threshold:

                retained.append(trade)

        for year in years:

            baseline_year = [
                trade
                for trade in trades
                if int(
                    normalize_date(
                        trade.get("entry_date")
                    )[:4]
                ) == year
            ]

            filtered_year = [
                trade
                for trade in retained
                if int(
                    normalize_date(
                        trade.get("entry_date")
                    )[:4]
                ) == year
            ]

            baseline_r = sum(
                safe_float(
                    trade.get("r_multiple")
                ) or 0.0
                for trade in baseline_year
            )

            filtered_r = sum(
                safe_float(
                    trade.get("r_multiple")
                ) or 0.0
                for trade in filtered_year
            )

            delta = (
                filtered_r
                - baseline_r
            )

            year_detail_rows.append(
                {
                    "window_bars": window,
                    "threshold_r": threshold,
                    "year": year,
                    "baseline_trades": len(
                        baseline_year
                    ),
                    "filtered_trades": len(
                        filtered_year
                    ),
                    "baseline_r": baseline_r,
                    "filtered_r": filtered_r,
                    "delta_r": delta,
                    "improved": (
                        delta > TOLERANCE
                    ),
                }
            )


# =====================================================================
# SAVE OUTPUTS
# =====================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


matrix_fields = [
    "window_bars",
    "threshold_r",
    "flagged_trades",
    "retained_trades",
    "excluded_insufficient_bars",
    "wins",
    "win_rate",
    "total_r",
    "avg_r",
    "profit_factor",
    "delta_r",
    "improved_years",
    "worsened_years",
    "unchanged_years",
    "status",
]

with open(
    MATRIX_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=matrix_fields
    )

    writer.writeheader()

    writer.writerows(
        matrix_rows
    )


bar_fields = [
    "_symbol",
    "entry_date",
    "direction",
    "bar_number",
    "bar_date",
    "entry_price",
    "risk_per_unit",
    "bar_high",
    "bar_low",
    "bar_adverse_r",
    "cumulative_max_adverse_r",
]

with open(
    BAR_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=bar_fields
    )

    writer.writeheader()

    writer.writerows(
        bar_rows
    )


validation_fields = [
    "_symbol",
    "entry_date",
    "direction",
    "bar_number",
    "stored_value",
    "replayed_value",
    "difference",
    "status",
]

with open(
    VALIDATION_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=validation_fields
    )

    writer.writeheader()

    writer.writerows(
        validation_rows
    )


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print()
print("=" * 115)
print("FINAL FORENSIC SUMMARY")
print("=" * 115)

print()

print(
    f"Frozen trades                    : {len(trades)}"
)

print(
    f"Bar replay records               : {len(bar_rows)}"
)

print(
    f"Bar 1-5 validation matches       : {match_count}"
)

print(
    f"Bar 1-5 validation mismatches    : {mismatch_count}"
)

print(
    f"Matrix combinations tested       : {len(matrix_rows)}"
)

print()

if (
    len(replayed_trade_data) == expected_replay_trades
    and len(validation_rows) == expected_validation_rows
    and len(bar_rows) > 0
    and mismatch_count == 0
):

    print(
        "BAR REPLAY INTEGRITY: PASS"
    )

else:

    raise RuntimeError(
        "BAR REPLAY INTEGRITY: FAIL"
    )


print()
print(
    "Outputs:"
)

print(
    f"  Matrix     : {MATRIX_OUTPUT}"
)

print(
    f"  Bar replay : {BAR_OUTPUT}"
)

print(
    f"  Validation : {VALIDATION_OUTPUT}"
)

print()
print(
    "Production code modified         : NO"
)

print(
    "Frozen historical dataset changed: NO"
)

print(
    "Holdout dataset changed          : NO"
)

print()
print("=" * 115)
print("IMPORTANT")
print("=" * 115)
print()
print(
    "Do NOT select a deployment threshold from this output yet."
)
print(
    "The purpose of this experiment is to establish whether the"
)
print(
    "early-adverse relationship survives direct OHLC reconstruction"
)
print(
    "and whether a stable parameter region exists."
)
print()