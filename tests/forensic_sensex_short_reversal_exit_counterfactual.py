"""
TradeSense-AI
SENSEX SHORT_REVERSAL Exit-Path Counterfactual

Research-only forensic analysis.

Purpose:
    Test whether SENSEX SHORT_REVERSAL trades that first reach
    +1.5R favorable excursion could benefit from:

        +1.5R trigger
        -> +1.0R protection
        -> +2.0R target

Important:
    - Uses only the actual lifetime of each frozen research trade.
    - Does NOT look beyond the actual production exit date.
    - Trigger is observed on a completed bar.
    - Hypothetical protection/target become active from the NEXT bar.
    - Trades that never reach +1.5R retain their actual R outcome.
    - No production code is modified.
    - No frozen historical dataset is modified.
    - No holdout dataset is modified.

This is a counterfactual research test only.
"""


import os
import sys
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "SENSEX"
TICKER = "^BSESN"

HISTORICAL_TRADES_FILE = (
    "tests/output/index_backtests/sensex_index_trades.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sensex_short_reversal_exit_counterfactual.csv"
)

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

# Must match frozen SENSEX research generation.
INITIAL_STOP_ATR = 3.5
SLIPPAGE_PERCENT = 0.10

# Counterfactual controls.
TRIGGER_R = 1.5
PROTECTION_R = 1.0
TARGET_R = 2.0

# Numerical tolerance.
TOLERANCE = 0.005001

EXPECTED_TOTAL_SENSEX_TRADES = 21
EXPECTED_SHORT_REVERSAL_TRADES = 7


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
    sys.exit(1)


def safe_float(value):
    try:
        value = float(value)
        if math.isnan(value):
            return np.nan
        return value
    except (TypeError, ValueError):
        return np.nan


def calculate_atr(df, period=14):
    """
    Exact production-style ATR reconstruction.

    Production logic:

        previous_close = Close.shift(1)

        TR = max(
            High - Low,
            abs(High - previous_close),
            abs(Low - previous_close)
        )

        ATR = rolling(14).mean()
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


def normalize_date(value):
    """
    Convert trade date into YYYY-MM-DD.
    """
    if pd.isna(value):
        return None

    text = str(value)

    if " " in text:
        text = text.split(" ")[0]

    if "T" in text:
        text = text.split("T")[0]

    return text[:10]


def reconstruct_entry_price(direction, open_price):
    """
    Exact production slippage convention.

    Long:
        Open * (1 + 0.10%)

    Short:
        Open * (1 - 0.10%)
    """

    slippage_rate = SLIPPAGE_PERCENT / 100.0

    if direction == "SHORT":
        return open_price * (1.0 - slippage_rate)

    if direction == "LONG":
        return open_price * (1.0 + slippage_rate)

    raise ValueError(f"Unsupported direction: {direction}")


def reconstruct_trade_risk(entry_price, atr, direction):
    """
    Exact frozen research risk reconstruction.

    risk/share = 3.5 * ATR

    Short stop:
        entry + risk/share

    Long stop:
        entry - risk/share
    """

    risk_per_share = INITIAL_STOP_ATR * atr

    if direction == "SHORT":
        stop_price = entry_price + risk_per_share
    else:
        stop_price = entry_price - risk_per_share

    return risk_per_share, stop_price


def adverse_or_favorable_r(
    direction,
    entry_price,
    price,
    risk_per_share,
):
    """
    Return signed favorable R from entry to a price.

    SHORT:
        favorable movement = entry - price

    LONG:
        favorable movement = price - entry
    """

    if risk_per_share <= 0:
        return np.nan

    if direction == "SHORT":
        return (entry_price - price) / risk_per_share

    return (price - entry_price) / risk_per_share


def reconstruct_production_r(
    direction,
    entry_price,
    risk_per_share,
    exit_price,
):
    """
    Reconstruct the price-movement R before transaction-cost effects.

    This is used only as a diagnostic cross-check.

    The frozen CSV's actual r_multiple is treated as the authoritative
    production result for the baseline.
    """

    if risk_per_share <= 0:
        return np.nan

    if direction == "SHORT":
        return (entry_price - exit_price) / risk_per_share

    return (exit_price - entry_price) / risk_per_share


# ============================================================================
# LOAD HISTORICAL TRADES
# ============================================================================

print()
print("=" * 100)
print("TRADESENSE-AI — SENSEX SHORT_REVERSAL EXIT COUNTERFACTUAL")
print("=" * 100)

if not os.path.exists(HISTORICAL_TRADES_FILE):
    fail(
        f"Historical SENSEX trade file not found:\n"
        f"{HISTORICAL_TRADES_FILE}"
    )

trades_df = pd.read_csv(HISTORICAL_TRADES_FILE)

if trades_df.empty:
    fail("Historical SENSEX trade file is empty.")

print(f"Historical SENSEX trades loaded: {len(trades_df)}")

if len(trades_df) != EXPECTED_TOTAL_SENSEX_TRADES:
    fail(
        "Unexpected SENSEX trade count.\n"
        f"Expected: {EXPECTED_TOTAL_SENSEX_TRADES}\n"
        f"Actual:   {len(trades_df)}"
    )

required_trade_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
    "entry_atr",
    "initial_risk",
    "max_favorable_r",
    "max_adverse_r",
]

missing_trade_columns = [
    column
    for column in required_trade_columns
    if column not in trades_df.columns
]

if missing_trade_columns:
    fail(
        "Historical trade file is missing required columns:\n"
        + "\n".join(missing_trade_columns)
    )


# ============================================================================
# ISOLATE SHORT_REVERSAL
# ============================================================================

sr_df = trades_df[
    trades_df["setup"].astype(str).str.upper() == "SHORT_REVERSAL"
].copy()

sr_df.reset_index(drop=True, inplace=True)

print(f"SHORT_REVERSAL trades: {len(sr_df)}")

if len(sr_df) != EXPECTED_SHORT_REVERSAL_TRADES:
    fail(
        "Unexpected SHORT_REVERSAL trade count.\n"
        f"Expected: {EXPECTED_SHORT_REVERSAL_TRADES}\n"
        f"Actual:   {len(sr_df)}"
    )

sr_df["entry_date_norm"] = sr_df["entry_date"].apply(normalize_date)
sr_df["exit_date_norm"] = sr_df["exit_date"].apply(normalize_date)

if sr_df["entry_date_norm"].isna().any():
    fail("At least one SHORT_REVERSAL trade has an invalid entry date.")

if sr_df["exit_date_norm"].isna().any():
    fail("At least one SHORT_REVERSAL trade has an invalid exit date.")


# ============================================================================
# DOWNLOAD SENSEX OHLC DATA
# ============================================================================

print()
print("-" * 100)
print("DOWNLOADING SENSEX OHLC DATA")
print("-" * 100)

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False,
)

if data is None or data.empty:
    fail("Yahoo Finance returned no SENSEX data.")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

required_price_columns = [
    "Open",
    "High",
    "Low",
    "Close",
]

missing_price_columns = [
    column
    for column in required_price_columns
    if column not in data.columns
]

if missing_price_columns:
    fail(
        "Downloaded SENSEX data is missing columns:\n"
        + "\n".join(missing_price_columns)
    )

data = data.copy()

data.index = pd.to_datetime(data.index)

if getattr(data.index, "tz", None) is not None:
    data.index = data.index.tz_localize(None)

data = data.sort_index()

for column in required_price_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

data["ATR"] = calculate_atr(data)

data = data.dropna(
    subset=required_price_columns
).copy()

data["Date"] = data.index.strftime("%Y-%m-%d")

data_by_date = {
    row["Date"]: row
    for _, row in data.iterrows()
}

print(
    f"SENSEX rows: {len(data)}"
)

print(
    f"Date range: "
    f"{data.index.min().strftime('%Y-%m-%d')} "
    f"-> "
    f"{data.index.max().strftime('%Y-%m-%d')}"
)

print("ATR reconstruction: 14-period rolling True Range mean")
print("Slippage reconstruction: 0.10%")


# ============================================================================
# VALIDATE TRADE DATES
# ============================================================================

print()
print("-" * 100)
print("VALIDATING TRADE DATES")
print("-" * 100)

date_failures = []

for _, trade in sr_df.iterrows():

    entry_date = trade["entry_date_norm"]
    exit_date = trade["exit_date_norm"]

    if entry_date not in data_by_date:
        date_failures.append(
            f"Missing entry date: {entry_date}"
        )

    if exit_date not in data_by_date:
        date_failures.append(
            f"Missing exit date: {exit_date}"
        )

    if (
        entry_date in data_by_date
        and exit_date in data_by_date
        and entry_date > exit_date
    ):
        date_failures.append(
            f"Entry after exit: {entry_date} -> {exit_date}"
        )

if date_failures:
    fail(
        "Trade-date validation failed:\n"
        + "\n".join(date_failures)
    )

print("Trade-date validation: PASS")


# ============================================================================
# COUNTERFACTUAL REPLAY
# ============================================================================

print()
print("=" * 100)
print("COUNTERFACTUAL REPLAY")
print("=" * 100)

results = []

replay_failures = []

for trade_index, trade in sr_df.iterrows():

    entry_date = trade["entry_date_norm"]
    exit_date = trade["exit_date_norm"]

    direction = str(trade["direction"]).upper()

    if direction not in {"SHORT", "LONG"}:
        replay_failures.append(
            f"{entry_date}: unsupported direction {direction}"
        )
        continue

    # ------------------------------------------------------------------------
    # Build actual trade lifetime.
    #
    # IMPORTANT:
    # We use the entry-date through actual-exit-date inclusive.
    # We never inspect bars after the production exit.
    # ------------------------------------------------------------------------

    lifetime = data[
        (data["Date"] >= entry_date)
        & (data["Date"] <= exit_date)
    ].copy()

    lifetime.reset_index(drop=True, inplace=True)

    if lifetime.empty:
        replay_failures.append(
            f"{entry_date}: no OHLC bars in actual lifetime"
        )
        continue

    # ------------------------------------------------------------------------
    # Production entry reconstruction.
    #
    # Production enters on the entry-date open with slippage.
    # ------------------------------------------------------------------------

    entry_row = lifetime.iloc[0]

    entry_open = safe_float(entry_row["Open"])
    entry_atr = safe_float(entry_row["ATR"])

    if not np.isfinite(entry_open):
        replay_failures.append(
            f"{entry_date}: invalid entry Open"
        )
        continue

    if not np.isfinite(entry_atr) or entry_atr <= 0:
        replay_failures.append(
            f"{entry_date}: invalid entry ATR"
        )
        continue

    reconstructed_entry = reconstruct_entry_price(
        direction,
        entry_open,
    )

    risk_per_share, initial_stop_price = reconstruct_trade_risk(
        reconstructed_entry,
        entry_atr,
        direction,
    )

    if risk_per_share <= 0:
        replay_failures.append(
            f"{entry_date}: invalid reconstructed risk"
        )
        continue

    # ------------------------------------------------------------------------
    # Baseline frozen values.
    # ------------------------------------------------------------------------

    actual_r = safe_float(trade["r_multiple"])
    actual_mfe = safe_float(trade["max_favorable_r"])
    actual_mae = safe_float(trade["max_adverse_r"])

    # ------------------------------------------------------------------------
    # Determine the first completed bar that reaches +1.5R.
    #
    # IMPORTANT:
    # The trigger itself is NOT allowed to retroactively change the
    # trigger bar's outcome.
    #
    # Protection and target therefore become active starting on the
    # NEXT bar.
    # ------------------------------------------------------------------------

    trigger_bar_number = None
    trigger_date = None

    for bar_position in range(len(lifetime)):

        bar = lifetime.iloc[bar_position]

        high = safe_float(bar["High"])
        low = safe_float(bar["Low"])

        if not np.isfinite(high) or not np.isfinite(low):
            continue

        favorable_high_r = adverse_or_favorable_r(
            direction,
            reconstructed_entry,
            high,
            risk_per_share,
        )

        favorable_low_r = adverse_or_favorable_r(
            direction,
            reconstructed_entry,
            low,
            risk_per_share,
        )

        bar_favorable_r = max(
            favorable_high_r,
            favorable_low_r,
        )

        if (
            np.isfinite(bar_favorable_r)
            and bar_favorable_r >= TRIGGER_R
        ):
            trigger_bar_number = bar_position + 1
            trigger_date = str(bar["Date"])
            break

    # ------------------------------------------------------------------------
    # No trigger:
    # retain actual production outcome.
    # ------------------------------------------------------------------------

    if trigger_bar_number is None:

        results.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "setup": "SHORT_REVERSAL",

                "actual_r": actual_r,
                "counterfactual_r": actual_r,
                "delta_r": 0.0,

                "actual_mfe_r": actual_mfe,
                "actual_mae_r": actual_mae,

                "triggered_1_5r": False,
                "trigger_bar_number": np.nan,
                "trigger_date": "",

                "counterfactual_exit_date": exit_date,
                "counterfactual_exit_reason": "ACTUAL_NO_TRIGGER",

                "entry_open": entry_open,
                "reconstructed_entry_price": reconstructed_entry,
                "entry_atr_reconstructed": entry_atr,
                "risk_per_share_reconstructed": risk_per_share,
                "initial_stop_price": initial_stop_price,

                "lifetime_bars": len(lifetime),
            }
        )

        continue

    # ------------------------------------------------------------------------
    # Trigger occurred.
    #
    # Protection:
    #     +1R favorable from entry
    #
    # Target:
    #     +2R favorable from entry
    #
    # For SHORT:
    #     +1R = entry - risk
    #     +2R = entry - 2*risk
    #
    # For LONG:
    #     +1R = entry + risk
    #     +2R = entry + 2*risk
    # ------------------------------------------------------------------------

    if direction == "SHORT":

        protection_price = (
            reconstructed_entry
            - PROTECTION_R * risk_per_share
        )

        target_price = (
            reconstructed_entry
            - TARGET_R * risk_per_share
        )

    else:

        protection_price = (
            reconstructed_entry
            + PROTECTION_R * risk_per_share
        )

        target_price = (
            reconstructed_entry
            + TARGET_R * risk_per_share
        )

    counterfactual_r = actual_r
    counterfactual_exit_date = exit_date
    counterfactual_exit_reason = "ACTUAL_NO_COUNTERFACTUAL_EXIT"

    # ------------------------------------------------------------------------
    # Start from the bar AFTER the trigger.
    # ------------------------------------------------------------------------

    trigger_position_zero_based = trigger_bar_number - 1

    counterfactual_resolved = False

    for bar_position in range(
        trigger_position_zero_based + 1,
        len(lifetime),
    ):

        bar = lifetime.iloc[bar_position]

        high = safe_float(bar["High"])
        low = safe_float(bar["Low"])

        bar_date = str(bar["Date"])

        if not np.isfinite(high) or not np.isfinite(low):
            continue

        # ====================================================================
        # SHORT
        # ====================================================================

        if direction == "SHORT":

            target_hit = low <= target_price
            protection_hit = high >= protection_price

            # Both could theoretically occur in one OHLC bar.
            #
            # We cannot know the intrabar ordering from daily OHLC alone.
            # Therefore use a conservative ambiguity treatment:
            # if both are touched, count the protective exit.
            #
            # This prevents the counterfactual from assuming the favorable
            # target happened first.
            if target_hit and protection_hit:

                counterfactual_r = PROTECTION_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = (
                    "PROTECTED_1R_AMBIGUOUS_BAR"
                )
                counterfactual_resolved = True
                break

            if target_hit:

                counterfactual_r = TARGET_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = "TARGET_2R"
                counterfactual_resolved = True
                break

            if protection_hit:

                counterfactual_r = PROTECTION_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = "PROTECTED_1R"
                counterfactual_resolved = True
                break

        # ====================================================================
        # LONG
        # ====================================================================

        else:

            target_hit = high >= target_price
            protection_hit = low <= protection_price

            if target_hit and protection_hit:

                counterfactual_r = PROTECTION_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = (
                    "PROTECTED_1R_AMBIGUOUS_BAR"
                )
                counterfactual_resolved = True
                break

            if target_hit:

                counterfactual_r = TARGET_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = "TARGET_2R"
                counterfactual_resolved = True
                break

            if protection_hit:

                counterfactual_r = PROTECTION_R
                counterfactual_exit_date = bar_date
                counterfactual_exit_reason = "PROTECTED_1R"
                counterfactual_resolved = True
                break

    # ------------------------------------------------------------------------
    # If the trigger occurred but neither hypothetical exit was reached
    # before the actual production exit, retain actual R.
    # ------------------------------------------------------------------------

    if not counterfactual_resolved:

        counterfactual_r = actual_r
        counterfactual_exit_date = exit_date
        counterfactual_exit_reason = (
            "TRIGGERED_NO_COUNTERFACTUAL_EXIT_BEFORE_ACTUAL_EXIT"
        )

    delta_r = counterfactual_r - actual_r

    results.append(
        {
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": direction,
            "setup": "SHORT_REVERSAL",

            "actual_r": actual_r,
            "counterfactual_r": counterfactual_r,
            "delta_r": delta_r,

            "actual_mfe_r": actual_mfe,
            "actual_mae_r": actual_mae,

            "triggered_1_5r": True,
            "trigger_bar_number": trigger_bar_number,
            "trigger_date": trigger_date,

            "counterfactual_exit_date": counterfactual_exit_date,
            "counterfactual_exit_reason": counterfactual_exit_reason,

            "entry_open": entry_open,
            "reconstructed_entry_price": reconstructed_entry,
            "entry_atr_reconstructed": entry_atr,
            "risk_per_share_reconstructed": risk_per_share,
            "initial_stop_price": initial_stop_price,

            "protection_price": protection_price,
            "target_price": target_price,

            "lifetime_bars": len(lifetime),
        }
    )


# ============================================================================
# REPLAY VALIDATION
# ============================================================================

if replay_failures:
    fail(
        "Replay validation failed:\n"
        + "\n".join(replay_failures)
    )

results_df = pd.DataFrame(results)

if len(results_df) != EXPECTED_SHORT_REVERSAL_TRADES:
    fail(
        "Counterfactual result count mismatch.\n"
        f"Expected: {EXPECTED_SHORT_REVERSAL_TRADES}\n"
        f"Actual:   {len(results_df)}"
    )

print()
print("Replay records generated:", len(results_df))
print("Replay validation: PASS")


# ============================================================================
# BASELINE RECONCILIATION
# ============================================================================

baseline_actual_r = results_df["actual_r"].sum()
baseline_trade_r = sr_df["r_multiple"].astype(float).sum()

baseline_delta = baseline_actual_r - baseline_trade_r

print()
print("=" * 100)
print("BASELINE RECONCILIATION")
print("=" * 100)

print(
    f"Frozen SHORT_REVERSAL Total R: "
    f"{baseline_trade_r:.2f}R"
)

print(
    f"Counterfactual input Actual R: "
    f"{baseline_actual_r:.2f}R"
)

print(
    f"Difference: "
    f"{baseline_delta:.6f}R"
)

if abs(baseline_delta) > TOLERANCE:
    fail(
        "BASELINE RECONCILIATION FAILED.\n"
        "Counterfactual cannot proceed because actual R values "
        "do not reconcile with the frozen trade dataset."
    )

print("BASELINE RECONCILIATION: PASS")


# ============================================================================
# COUNTERFACTUAL SUMMARY
# ============================================================================

triggered = results_df[
    results_df["triggered_1_5r"] == True
].copy()

non_triggered = results_df[
    results_df["triggered_1_5r"] == False
].copy()

counterfactual_total_r = results_df[
    "counterfactual_r"
].sum()

delta_total_r = (
    counterfactual_total_r
    - baseline_actual_r
)

print()
print("=" * 100)
print("COUNTERFACTUAL SUMMARY")
print("=" * 100)

print(
    f"Baseline trades:                 "
    f"{len(results_df)}"
)

print(
    f"Reached +{TRIGGER_R:.1f}R:                  "
    f"{len(triggered)}"
)

print(
    f"Did not reach +{TRIGGER_R:.1f}R:              "
    f"{len(non_triggered)}"
)

print(
    f"Baseline Total R:                "
    f"{baseline_actual_r:.2f}R"
)

print(
    f"Counterfactual Total R:          "
    f"{counterfactual_total_r:.2f}R"
)

print(
    f"Delta:                            "
    f"{delta_total_r:+.2f}R"
)

if len(results_df) > 0:

    baseline_win_rate = (
        (results_df["actual_r"] > 0).mean()
        * 100.0
    )

    counterfactual_win_rate = (
        (results_df["counterfactual_r"] > 0).mean()
        * 100.0
    )

    print(
        f"Baseline Win Rate:               "
        f"{baseline_win_rate:.2f}%"
    )

    print(
        f"Counterfactual Win Rate:         "
        f"{counterfactual_win_rate:.2f}%"
    )


# ============================================================================
# TRIGGERED TRADE SUMMARY
# ============================================================================

print()
print("=" * 100)
print("TRIGGERED TRADE OUTCOMES")
print("=" * 100)

if triggered.empty:

    print("No SHORT_REVERSAL trade reached +1.5R.")
    print("No exit-path counterfactual was triggered.")

else:

    triggered_r = triggered["counterfactual_r"].sum()
    triggered_actual_r = triggered["actual_r"].sum()
    triggered_delta = triggered["delta_r"].sum()

    print(
        f"Triggered trades:                "
        f"{len(triggered)}"
    )

    print(
        f"Triggered baseline R:            "
        f"{triggered_actual_r:.2f}R"
    )

    print(
        f"Triggered counterfactual R:      "
        f"{triggered_r:.2f}R"
    )

    print(
        f"Triggered delta:                 "
        f"{triggered_delta:+.2f}R"
    )

    print()
    print("Counterfactual exit reasons:")

    print(
        triggered[
            "counterfactual_exit_reason"
        ].value_counts().to_string()
    )


# ============================================================================
# YEAR-BY-YEAR ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("YEAR-BY-YEAR COUNTERFACTUAL")
print("=" * 100)

results_df["year"] = (
    results_df["entry_date"]
    .astype(str)
    .str[:4]
)

year_rows = []

for year, group in results_df.groupby("year"):

    baseline_r = group["actual_r"].sum()
    counterfactual_r = group["counterfactual_r"].sum()
    delta_r = counterfactual_r - baseline_r

    triggered_count = int(
        group["triggered_1_5r"].sum()
    )

    year_rows.append(
        {
            "year": year,
            "trades": len(group),
            "triggered_1_5r": triggered_count,
            "baseline_r": baseline_r,
            "counterfactual_r": counterfactual_r,
            "delta_r": delta_r,
        }
    )

year_df = pd.DataFrame(year_rows)

if not year_df.empty:

    print(
        year_df.to_string(
            index=False,
            formatters={
                "baseline_r": "{:.2f}".format,
                "counterfactual_r": "{:.2f}".format,
                "delta_r": "{:+.2f}".format,
            },
        )
    )


# ============================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("LEAVE-ONE-OUT COUNTERFACTUAL")
print("=" * 100)

loo_rows = []

for excluded_index, excluded_trade in results_df.iterrows():

    remaining = results_df.drop(
        index=excluded_index
    )

    remaining_baseline = remaining["actual_r"].sum()
    remaining_counterfactual = (
        remaining["counterfactual_r"].sum()
    )

    remaining_delta = (
        remaining_counterfactual
        - remaining_baseline
    )

    loo_rows.append(
        {
            "excluded_entry_date": excluded_trade["entry_date"],
            "excluded_r": excluded_trade["actual_r"],
            "excluded_delta_r": excluded_trade["delta_r"],
            "remaining_baseline_r": remaining_baseline,
            "remaining_counterfactual_r": remaining_counterfactual,
            "remaining_delta_r": remaining_delta,
        }
    )

loo_df = pd.DataFrame(loo_rows)

if not loo_df.empty:

    print(
        loo_df.to_string(
            index=False,
            formatters={
                "excluded_r": "{:.2f}".format,
                "excluded_delta_r": "{:+.2f}".format,
                "remaining_baseline_r": "{:.2f}".format,
                "remaining_counterfactual_r": "{:.2f}".format,
                "remaining_delta_r": "{:+.2f}".format,
            },
        )
    )

positive_loo = int(
    (
        loo_df["remaining_delta_r"]
        > TOLERANCE
    ).sum()
)

negative_loo = int(
    (
        loo_df["remaining_delta_r"]
        < -TOLERANCE
    ).sum()
)

unchanged_loo = len(loo_df) - positive_loo - negative_loo

print()
print(
    f"LOO positive:  {positive_loo}"
)

print(
    f"LOO negative:  {negative_loo}"
)

print(
    f"LOO unchanged: {unchanged_loo}"
)


# ============================================================================
# INDIVIDUAL TRADE TABLE
# ============================================================================

print()
print("=" * 100)
print("INDIVIDUAL COUNTERFACTUAL TRADES")
print("=" * 100)

display_columns = [
    "entry_date",
    "exit_date",
    "actual_r",
    "counterfactual_r",
    "delta_r",
    "actual_mfe_r",
    "triggered_1_5r",
    "trigger_bar_number",
    "trigger_date",
    "counterfactual_exit_date",
    "counterfactual_exit_reason",
    "lifetime_bars",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        formatters={
            "actual_r": "{:.2f}".format,
            "counterfactual_r": "{:.2f}".format,
            "delta_r": "{:+.2f}".format,
            "actual_mfe_r": "{:.2f}".format,
        },
    )
)


# ============================================================================
# SAVE RESEARCH OUTPUT
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

results_df.to_csv(
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


# ============================================================================
# INTEGRITY CHECKS
# ============================================================================

print()
print("=" * 100)
print("INTEGRITY CHECKS")
print("=" * 100)

integrity_pass = True

# Trade count.
if len(results_df) != EXPECTED_SHORT_REVERSAL_TRADES:
    print("FAIL: trade count mismatch")
    integrity_pass = False
else:
    print("PASS: trade count reconciliation")

# Baseline R.
if abs(baseline_delta) > TOLERANCE:
    print("FAIL: baseline R mismatch")
    integrity_pass = False
else:
    print("PASS: baseline R reconciliation")

# No result may have a counterfactual exit after actual exit.
for _, row in results_df.iterrows():

    cf_exit = normalize_date(
        row["counterfactual_exit_date"]
    )

    actual_exit = normalize_date(
        row["exit_date"]
    )

    if cf_exit > actual_exit:

        print(
            "FAIL: counterfactual exit occurs after "
            f"actual exit for {row['entry_date']}"
        )

        integrity_pass = False

if integrity_pass:
    print(
        "PASS: no counterfactual exit exceeds actual trade lifetime"
    )

# Trigger consistency.
for _, row in results_df.iterrows():

    if row["triggered_1_5r"]:

        if not np.isfinite(
            safe_float(row["trigger_bar_number"])
        ):
            print(
                "FAIL: triggered trade missing trigger bar"
            )
            integrity_pass = False

        if not row["trigger_date"]:
            print(
                "FAIL: triggered trade missing trigger date"
            )
            integrity_pass = False

if integrity_pass:
    print(
        "PASS: trigger metadata integrity"
    )

# No-trigger trades must have zero delta.
for _, row in results_df.iterrows():

    if not row["triggered_1_5r"]:

        if abs(
            safe_float(row["delta_r"])
        ) > TOLERANCE:

            print(
                "FAIL: no-trigger trade changed outcome "
                f"for {row['entry_date']}"
            )

            integrity_pass = False

if integrity_pass:
    print(
        "PASS: no-trigger trades retain actual outcome"
    )


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)

if not integrity_pass:

    print(
        "INTEGRITY FAILURE — DO NOT USE COUNTERFACTUAL RESULT"
    )

elif delta_total_r <= TOLERANCE:

    print(
        "NO POSITIVE EXIT-PATH EDGE DETECTED"
    )

elif len(triggered) < 3:

    print(
        "PROMISING BUT INSUFFICIENT TRIGGER SAMPLE"
    )

else:

    print(
        "POSITIVE COUNTERFACTUAL SIGNAL — "
        "ROBUSTNESS TEST REQUIRED"
    )

print()
print(
    "This analysis is descriptive/counterfactual research only."
)

print(
    "SHORT_REVERSAL exclusion is NOT being implemented."
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
    "Any apparent exit-management edge requires "
    "independent robustness and OOS validation."
)

print("=" * 100)