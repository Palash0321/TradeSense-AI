"""
TradeSense-AI
SENSEX Early-Adverse Robustness Research

Research-only analysis.

Tests the validated early-adverse concept across the complete SENSEX
21-trade baseline using the same OHLC replay methodology.

No production code is modified.
No frozen stock historical data is modified.
No holdout data is modified.
"""

import os
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf


warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG
# ============================================================================

INDEX_NAME = "SENSEX"
TICKER = "^BSESN"

HISTORICAL_TRADES_FILE = (
    "tests/output/index_backtests/sensex_index_trades.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

ROBUSTNESS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sensex_early_adverse_robustness.csv",
)

YEAR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sensex_early_adverse_year_robustness.csv",
)

LOO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "sensex_early_adverse_leave_one_out.csv",
)

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

INITIAL_STOP_ATR = 3.5
SLIPPAGE_PERCENT = 0.10

WINDOWS = [3, 5, 7, 10]
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

EXPECTED_TRADE_COUNT = 21

TOLERANCE = 0.005001


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    print()
    print("=" * 100)
    print("FAILURE")
    print("=" * 100)
    print(message)
    print("=" * 100)
    raise SystemExit(1)


def safe_float(value):
    try:
        value = float(value)
        if math.isnan(value):
            return np.nan
        return value
    except (TypeError, ValueError):
        return np.nan


def normalize_date(value):
    if pd.isna(value):
        return None

    text = str(value)

    if " " in text:
        text = text.split(" ")[0]

    if "T" in text:
        text = text.split("T")[0]

    return text[:10]


def calculate_atr(df, period=14):
    """
    Exact production ATR reconstruction.

    TR =
        max(
            High-Low,
            abs(High-previous Close),
            abs(Low-previous Close)
        )

    ATR = simple rolling mean over 14 periods.
    """

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def reconstruct_entry_price(direction, open_price):
    slippage_rate = SLIPPAGE_PERCENT / 100.0

    if direction == "LONG":
        return open_price * (1.0 + slippage_rate)

    if direction == "SHORT":
        return open_price * (1.0 - slippage_rate)

    raise ValueError(f"Unsupported direction: {direction}")


def favorable_r(direction, entry_price, high, low, risk_per_share):
    """
    Maximum favorable excursion for one OHLC bar.
    """

    if risk_per_share <= 0:
        return np.nan

    if direction == "LONG":
        favorable_high = (
            high - entry_price
        ) / risk_per_share

        favorable_low = (
            low - entry_price
        ) / risk_per_share

    else:
        favorable_high = (
            entry_price - high
        ) / risk_per_share

        favorable_low = (
            entry_price - low
        ) / risk_per_share

    return max(
        favorable_high,
        favorable_low,
    )


def adverse_r(direction, entry_price, high, low, risk_per_share):
    """
    Maximum adverse excursion for one OHLC bar.
    """

    if risk_per_share <= 0:
        return np.nan

    if direction == "LONG":
        adverse_high = (
            entry_price - high
        ) / risk_per_share

        adverse_low = (
            entry_price - low
        ) / risk_per_share

    else:
        adverse_high = (
            high - entry_price
        ) / risk_per_share

        adverse_low = (
            low - entry_price
        ) / risk_per_share

    return max(
        adverse_high,
        adverse_low,
    )


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print("TRADESENSE-AI — SENSEX EARLY-ADVERSE ROBUSTNESS")
print("=" * 100)


# ============================================================================
# LOAD TRADES
# ============================================================================

if not os.path.exists(HISTORICAL_TRADES_FILE):
    fail(
        f"Historical SENSEX trade file not found:\n"
        f"{HISTORICAL_TRADES_FILE}"
    )

trades = pd.read_csv(
    HISTORICAL_TRADES_FILE
)

if trades.empty:
    fail("Historical SENSEX trade file is empty.")

print(
    f"Historical SENSEX trades: {len(trades)}"
)

if len(trades) != EXPECTED_TRADE_COUNT:
    fail(
        "Unexpected SENSEX trade count.\n"
        f"Expected: {EXPECTED_TRADE_COUNT}\n"
        f"Actual:   {len(trades)}"
    )


required_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "r_multiple",
    "initial_risk",
    "shares",
]

missing = [
    c for c in required_columns
    if c not in trades.columns
]

if missing:
    fail(
        "Missing required trade columns:\n"
        + "\n".join(missing)
    )

trades["entry_date_norm"] = (
    trades["entry_date"]
    .apply(normalize_date)
)

trades["exit_date_norm"] = (
    trades["exit_date"]
    .apply(normalize_date)
)


# ============================================================================
# DOWNLOAD OHLC
# ============================================================================

print()
print("-" * 100)
print("DOWNLOADING SENSEX OHLC")
print("-" * 100)

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False,
)

if data is None or data.empty:
    fail("No SENSEX data returned by Yahoo Finance.")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

for column in [
    "Open",
    "High",
    "Low",
    "Close",
]:
    if column not in data.columns:
        fail(
            f"Missing SENSEX OHLC column: {column}"
        )

data = data.copy()

data.index = pd.to_datetime(data.index)

if getattr(data.index, "tz", None) is not None:
    data.index = data.index.tz_localize(None)

data = data.sort_index()

for column in [
    "Open",
    "High",
    "Low",
    "Close",
]:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data["ATR"] = calculate_atr(data)

data = data.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
    ]
).copy()

data["Date"] = data.index.strftime(
    "%Y-%m-%d"
)

print(
    f"SENSEX rows: {len(data)}"
)

print(
    f"Date range: "
    f"{data['Date'].iloc[0]} -> "
    f"{data['Date'].iloc[-1]}"
)

print(
    "ATR: 14-period simple rolling True Range mean"
)

print(
    "Entry slippage: 0.10%"
)


# ============================================================================
# RECONSTRUCT EARLY-ADVERSE VALUES
# ============================================================================

print()
print("=" * 100)
print("RECONSTRUCTING EARLY-ADVERSE DEVELOPMENT")
print("=" * 100)

records = []

replay_failures = []

for _, trade in trades.iterrows():

    entry_date = trade["entry_date_norm"]
    exit_date = trade["exit_date_norm"]

    direction = str(
        trade["direction"]
    ).upper()

    lifetime = data[
        (data["Date"] >= entry_date)
        & (data["Date"] <= exit_date)
    ].copy()

    lifetime.reset_index(drop=True, inplace=True)

    if lifetime.empty:
        replay_failures.append(
            f"{entry_date}: no lifetime bars"
        )
        continue

    entry_row = lifetime.iloc[0]

    entry_open = safe_float(
        entry_row["Open"]
    )

    entry_atr = safe_float(
        entry_row["ATR"]
    )

    if (
        not np.isfinite(entry_open)
        or not np.isfinite(entry_atr)
        or entry_atr <= 0
    ):
        replay_failures.append(
            f"{entry_date}: invalid entry Open/ATR"
        )
        continue

    entry_price = reconstruct_entry_price(
        direction,
        entry_open,
    )

    risk_per_share = (
        INITIAL_STOP_ATR
        * entry_atr
    )

    max_adverse = {
        1: 0.0,
        2: 0.0,
        3: 0.0,
        4: 0.0,
        5: 0.0,
        7: 0.0,
        10: 0.0,
    }

    max_favorable = 0.0

    for position, bar in lifetime.iterrows():

        high = safe_float(bar["High"])
        low = safe_float(bar["Low"])

        if (
            not np.isfinite(high)
            or not np.isfinite(low)
        ):
            continue

        bar_number = position + 1

        current_adverse = adverse_r(
            direction,
            entry_price,
            high,
            low,
            risk_per_share,
        )

        current_favorable = favorable_r(
            direction,
            entry_price,
            high,
            low,
            risk_per_share,
        )

        if np.isfinite(current_adverse):

            for window in max_adverse:
                if bar_number <= window:
                    max_adverse[window] = max(
                        max_adverse[window],
                        current_adverse,
                    )

        if np.isfinite(current_favorable):
            max_favorable = max(
                max_favorable,
                current_favorable,
            )

    actual_r = safe_float(
        trade["r_multiple"]
    )

    records.append(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": direction,
            "setup": str(
                trade.get(
                    "setup",
                    ""
                )
            ),

            "actual_r": actual_r,

            "reconstructed_mfe_r": max_favorable,

            "early_adverse_r_bar_3": max_adverse[3],
            "early_adverse_r_bar_5": max_adverse[5],
            "early_adverse_r_bar_7": max_adverse[7],
            "early_adverse_r_bar_10": max_adverse[10],
        }
    )


if replay_failures:
    fail(
        "Replay failures:\n"
        + "\n".join(replay_failures)
    )

replay_df = pd.DataFrame(records)

if len(replay_df) != EXPECTED_TRADE_COUNT:
    fail(
        "Replay trade count mismatch.\n"
        f"Expected: {EXPECTED_TRADE_COUNT}\n"
        f"Actual: {len(replay_df)}"
    )

print(
    f"Reconstructed trades: {len(replay_df)}"
)

print("OHLC replay: PASS")


# ============================================================================
# BASELINE RECONCILIATION
# ============================================================================

frozen_total_r = trades[
    "r_multiple"
].astype(float).sum()

replay_total_r = replay_df[
    "actual_r"
].sum()

baseline_difference = (
    replay_total_r
    - frozen_total_r
)

print()
print("=" * 100)
print("BASELINE RECONCILIATION")
print("=" * 100)

print(
    f"Frozen Total R:       {frozen_total_r:.2f}R"
)

print(
    f"Replay Total R:       {replay_total_r:.2f}R"
)

print(
    f"Difference:           {baseline_difference:.6f}R"
)

if abs(baseline_difference) > TOLERANCE:
    fail(
        "Baseline R reconciliation FAILED."
    )

print("BASELINE RECONCILIATION: PASS")


# ============================================================================
# BASELINE METRICS
# ============================================================================

def profit_factor(r_values):
    gross_profit = r_values[
        r_values > 0
    ].sum()

    gross_loss = abs(
        r_values[
            r_values < 0
        ].sum()
    )

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


baseline_win_rate = (
    (
        replay_df["actual_r"] > 0
    ).mean()
    * 100
)

baseline_pf = profit_factor(
    replay_df["actual_r"]
)

print()
print("=" * 100)
print("BASELINE")
print("=" * 100)

print(
    f"Trades:        {len(replay_df)}"
)

print(
    f"Win rate:      {baseline_win_rate:.2f}%"
)

print(
    f"Profit factor: {baseline_pf:.3f}"
)

print(
    f"Total R:       {frozen_total_r:.2f}R"
)


# ============================================================================
# PARAMETER MATRIX
# ============================================================================

print()
print("=" * 100)
print("EARLY-ADVERSE PARAMETER MATRIX")
print("=" * 100)

matrix_rows = []

for window in WINDOWS:

    column = (
        f"early_adverse_r_bar_{window}"
    )

    for threshold in THRESHOLDS:

        flagged = (
            replay_df[column]
            >= threshold
        )

        retained = replay_df[
            ~flagged
        ].copy()

        retained_r = (
            retained["actual_r"].sum()
        )

        delta_r = (
            retained_r
            - frozen_total_r
        )

        retained_count = len(retained)

        retained_winners = int(
            (
                retained["actual_r"] > 0
            ).sum()
        )

        retained_win_rate = (
            (
                retained["actual_r"] > 0
            ).mean()
            * 100
            if retained_count > 0
            else 0.0
        )

        retained_pf = profit_factor(
            retained["actual_r"]
        ) if retained_count > 0 else 0.0

        matrix_rows.append(
            {
                "window": window,
                "threshold_r": threshold,
                "flagged_trades": int(
                    flagged.sum()
                ),
                "retained_trades": retained_count,
                "retained_winners": retained_winners,
                "retained_win_rate": retained_win_rate,
                "retained_r": retained_r,
                "delta_r": delta_r,
                "profit_factor": retained_pf,
            }
        )


matrix_df = pd.DataFrame(
    matrix_rows
)

print(
    matrix_df.to_string(
        index=False,
        formatters={
            "threshold_r": "{:.2f}".format,
            "retained_win_rate": "{:.2f}%".format,
            "retained_r": "{:.2f}".format,
            "delta_r": "{:+.2f}".format,
            "profit_factor": "{:.3f}".format,
        },
    )
)


# ============================================================================
# MATRIX ROBUSTNESS COUNTS
# ============================================================================

improved_cells = int(
    (
        matrix_df["delta_r"]
        > TOLERANCE
    ).sum()
)

worsened_cells = int(
    (
        matrix_df["delta_r"]
        < -TOLERANCE
    ).sum()
)

unchanged_cells = (
    len(matrix_df)
    - improved_cells
    - worsened_cells
)

print()
print(
    f"Parameter cells:  {len(matrix_df)}"
)

print(
    f"Improved:         {improved_cells}"
)

print(
    f"Worsened:         {worsened_cells}"
)

print(
    f"Unchanged:        {unchanged_cells}"
)


# ============================================================================
# ANCHOR RESULT
# ============================================================================

anchor = matrix_df[
    (matrix_df["window"] == 5)
    & (
        matrix_df["threshold_r"]
        == 0.25
    )
].iloc[0]

print()
print("=" * 100)
print("ANCHOR — 5 BAR / 0.25R")
print("=" * 100)

print(
    f"Flagged:          "
    f"{int(anchor['flagged_trades'])}"
)

print(
    f"Retained:         "
    f"{int(anchor['retained_trades'])}"
)

print(
    f"Win rate:         "
    f"{anchor['retained_win_rate']:.2f}%"
)

print(
    f"Total R:          "
    f"{anchor['retained_r']:.2f}R"
)

print(
    f"Delta:            "
    f"{anchor['delta_r']:+.2f}R"
)

print(
    f"Profit factor:    "
    f"{anchor['profit_factor']:.3f}"
)


# ============================================================================
# YEAR ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("YEAR ROBUSTNESS — 5 BAR / 0.25R")
print("=" * 100)

replay_df["year"] = (
    replay_df["entry_date"]
    .astype(str)
    .str[:4]
)

anchor_column = (
    "early_adverse_r_bar_5"
)

year_rows = []

for year, group in replay_df.groupby(
    "year"
):

    baseline_r = group[
        "actual_r"
    ].sum()

    retained = group[
        group[anchor_column] < 0.25
    ]

    filtered_r = retained[
        "actual_r"
    ].sum()

    delta_r = (
        filtered_r
        - baseline_r
    )

    year_rows.append(
        {
            "year": year,
            "trades": len(group),
            "retained": len(retained),
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "delta_r": delta_r,
        }
    )

year_df = pd.DataFrame(
    year_rows
)

print(
    year_df.to_string(
        index=False,
        formatters={
            "baseline_r": "{:.2f}".format,
            "filtered_r": "{:.2f}".format,
            "delta_r": "{:+.2f}".format,
        },
    )
)


# ============================================================================
# LEAVE-ONE-OUT
# ============================================================================

print()
print("=" * 100)
print("LEAVE-ONE-OUT ROBUSTNESS — 5 BAR / 0.25R")
print("=" * 100)

loo_rows = []

for excluded_index, excluded_trade in replay_df.iterrows():

    remaining = replay_df.drop(
        index=excluded_index
    )

    remaining_baseline = (
        remaining["actual_r"].sum()
    )

    remaining_filtered = remaining[
        remaining[anchor_column] < 0.25
    ]

    remaining_filtered_r = (
        remaining_filtered["actual_r"].sum()
    )

    remaining_delta = (
        remaining_filtered_r
        - remaining_baseline
    )

    loo_rows.append(
        {
            "excluded_entry_date":
                excluded_trade["entry_date"],

            "excluded_r":
                excluded_trade["actual_r"],

            "excluded_early5":
                excluded_trade[
                    anchor_column
                ],

            "remaining_baseline_r":
                remaining_baseline,

            "remaining_filtered_r":
                remaining_filtered_r,

            "remaining_delta_r":
                remaining_delta,
        }
    )

loo_df = pd.DataFrame(
    loo_rows
)

print(
    loo_df.to_string(
        index=False,
        formatters={
            "excluded_r": "{:.2f}".format,
            "excluded_early5": "{:.2f}".format,
            "remaining_baseline_r": "{:.2f}".format,
            "remaining_filtered_r": "{:.2f}".format,
            "remaining_delta_r": "{:+.2f}".format,
        },
    )
)

loo_positive = int(
    (
        loo_df["remaining_delta_r"]
        > TOLERANCE
    ).sum()
)

loo_negative = int(
    (
        loo_df["remaining_delta_r"]
        < -TOLERANCE
    ).sum()
)

loo_unchanged = (
    len(loo_df)
    - loo_positive
    - loo_negative
)

print()
print(
    f"LOO positive:  {loo_positive}"
)

print(
    f"LOO negative:  {loo_negative}"
)

print(
    f"LOO unchanged: {loo_unchanged}"
)


# ============================================================================
# SETUP-SPECIFIC ANCHOR ANALYSIS
# ============================================================================

print()
print("=" * 100)
print("SETUP-SPECIFIC 5 BAR / 0.25R")
print("=" * 100)

setup_rows = []

for setup, group in replay_df.groupby(
    "setup"
):

    baseline_r = group[
        "actual_r"
    ].sum()

    retained = group[
        group[anchor_column] < 0.25
    ]

    filtered_r = retained[
        "actual_r"
    ].sum()

    setup_rows.append(
        {
            "setup": setup,
            "trades": len(group),
            "flagged": (
                group[anchor_column] >= 0.25
            ).sum(),
            "retained": len(retained),
            "baseline_r": baseline_r,
            "filtered_r": filtered_r,
            "delta_r": (
                filtered_r
                - baseline_r
            ),
        }
    )

setup_df = pd.DataFrame(
    setup_rows
)

print(
    setup_df.to_string(
        index=False,
        formatters={
            "baseline_r": "{:.2f}".format,
            "filtered_r": "{:.2f}".format,
            "delta_r": "{:+.2f}".format,
        },
    )
)


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

matrix_df.to_csv(
    ROBUSTNESS_OUTPUT,
    index=False,
)

year_df.to_csv(
    YEAR_OUTPUT,
    index=False,
)

loo_df.to_csv(
    LOO_OUTPUT,
    index=False,
)

print()
print("=" * 100)
print("OUTPUTS")
print("=" * 100)

print(
    f"Parameter matrix: {ROBUSTNESS_OUTPUT}"
)

print(
    f"Year robustness:  {YEAR_OUTPUT}"
)

print(
    f"LOO robustness:   {LOO_OUTPUT}"
)


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)

if improved_cells == 0:

    print(
        "NO ROBUST EARLY-ADVERSE EDGE DETECTED"
    )

elif (
    improved_cells == len(matrix_df)
    and worsened_cells == 0
):

    if (
        loo_negative == 0
        and len(year_df) >= 3
    ):

        print(
            "PROMISING ROBUSTNESS SIGNAL"
        )

    else:

        print(
            "PROMISING PARAMETER ROBUSTNESS — "
            "YEAR/LOO INSUFFICIENT"
        )

else:

    print(
        "MIXED EARLY-ADVERSE RESPONSE — "
        "NO DEPLOYMENT CASE"
    )

print()
print(
    "This is in-sample SENSEX research only."
)

print(
    "No production code was modified."
)

print(
    "No frozen historical data was modified."
)

print(
    "No holdout data was modified."
)

print(
    "No strategy filter is being deployed."
)

print(
    "Any promising result requires independent OOS validation."
)

print("=" * 100)