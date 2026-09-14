"""
TradeSense-AI
NIFTY MIDCAP SELECT Exit-Path Counterfactual

Research-only.

Hypothesis:
If a trade reaches +1.5R, protect +1R thereafter while
retaining the original +2R target.

This script does NOT modify production.

The counterfactual:
- Uses the actual historical trade lifetime.
- Finds the first bar where the trade reaches +1.5R.
- After that trigger, if price subsequently reaches +1R
  in the adverse direction, exit at +1R.
- If +2R is reached first, retain +2R target.
- If neither happens before the actual production exit,
  retain the actual production outcome.

Important:
- No look-ahead beyond the actual trade lifetime.
- No extension of trade lifetime.
- No production changes.
- No historical dataset changes.
- No holdout changes.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

TRADE_FILE = (
    "tests/output/index_backtests/"
    "midselect_index_trades.csv"
)

PRICE_FILE = (
    "tests/output/index_backtests/"
    "midselect_niftyindices_ohlc.csv"
)

START_DATE = "2022-01-10"
END_DATE = "2026-09-12"

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0

SLIPPAGE_PERCENT = 0.10

TRIGGER_R = 1.5
PROTECTION_R = 1.0

VALIDATION_TOLERANCE = 0.005001


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    return pd.Timestamp(value).normalize()


def calculate_r_levels(
    direction,
    entry_price,
    risk_per_share,
):
    if direction == "LONG":

        return {
            "target": (
                entry_price
                + TARGET_R * risk_per_share
            ),
            "trigger": (
                entry_price
                + TRIGGER_R * risk_per_share
            ),
            "protection": (
                entry_price
                + PROTECTION_R * risk_per_share
            ),
        }

    return {
        "target": (
            entry_price
            - TARGET_R * risk_per_share
        ),
        "trigger": (
            entry_price
            - TRIGGER_R * risk_per_share
        ),
        "protection": (
            entry_price
            - PROTECTION_R * risk_per_share
        ),
    }


def r_from_price(
    direction,
    entry_price,
    price,
    risk_per_share,
):
    if risk_per_share <= 0:
        return np.nan

    if direction == "LONG":

        return (
            price - entry_price
        ) / risk_per_share

    return (
        entry_price - price
    ) / risk_per_share


def calculate_metrics(r_values):
    values = pd.Series(
        r_values,
        dtype=float,
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
        }

    wins = int(
        (values > 0).sum()
    )

    losses = int(
        (values <= 0).sum()
    )

    gross_profit = float(
        values[
            values > 0
        ].sum()
    )

    gross_loss = float(
        values[
            values < 0
        ].sum()
    )

    abs_loss = abs(
        gross_loss
    )

    profit_factor = (
        gross_profit / abs_loss
        if abs_loss > 0
        else np.inf
    )

    return {
        "trades": len(values),
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / len(values) * 100
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_r": float(values.sum()),
        "average_r": float(values.mean()),
    }


def print_metrics(
    label,
    r_values,
):
    metrics = calculate_metrics(
        r_values
    )

    print()
    print(label)
    print("-" * 70)

    print(
        f"Trades       : "
        f"{metrics['trades']}"
    )

    print(
        f"Winners      : "
        f"{metrics['wins']}"
    )

    print(
        f"Losers       : "
        f"{metrics['losses']}"
    )

    print(
        f"Win rate     : "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"Gross profit : "
        f"{metrics['gross_profit']:+.2f}R"
    )

    print(
        f"Gross loss   : "
        f"{metrics['gross_loss']:+.2f}R"
    )

    if np.isfinite(
        metrics["profit_factor"]
    ):
        print(
            f"Profit factor: "
            f"{metrics['profit_factor']:.3f}"
        )
    else:
        print(
            "Profit factor: INF"
        )

    print(
        f"Total R      : "
        f"{metrics['total_r']:+.2f}R"
    )

    print(
        f"Average R    : "
        f"{metrics['average_r']:+.3f}R"
    )

    return metrics


# ============================================================================
# LOAD HISTORICAL TRADES
# ============================================================================

print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT EXIT-PATH COUNTERFACTUAL"
)
print("=" * 100)

print(
    f"Trade file: {TRADE_FILE}"
)

if not os.path.exists(
    TRADE_FILE
):

    print(
        "ERROR: Trade file not found."
    )

    raise SystemExit(1)


trades = pd.read_csv(
    TRADE_FILE
)


if trades.empty:

    print(
        "ERROR: Trade file is empty."
    )

    raise SystemExit(1)


required_columns = [
    "direction",
    "setup",
    "entry_date",
    "exit_date",
    "entry_price",
    "entry_atr",
    "initial_risk",
    "r_multiple",
    "max_favorable_r",
]


missing = [
    column
    for column in required_columns
    if column not in trades.columns
]


if missing:

    print(
        "ERROR: Missing required columns:"
    )

    for column in missing:
        print(
            f"  - {column}"
        )

    raise SystemExit(1)


trades["entry_date"] = pd.to_datetime(
    trades["entry_date"],
    errors="coerce",
)

trades["exit_date"] = pd.to_datetime(
    trades["exit_date"],
    errors="coerce",
)

trades["entry_price"] = pd.to_numeric(
    trades["entry_price"],
    errors="coerce",
)

trades["entry_atr"] = pd.to_numeric(
    trades["entry_atr"],
    errors="coerce",
)

trades["initial_risk"] = pd.to_numeric(
    trades["initial_risk"],
    errors="coerce",
)

trades["r_multiple"] = pd.to_numeric(
    trades["r_multiple"],
    errors="coerce",
)

trades["max_favorable_r"] = pd.to_numeric(
    trades["max_favorable_r"],
    errors="coerce",
)


print(
    f"Historical trades: {len(trades)}"
)


# ============================================================================
# LOAD VALIDATED MIDSELECT PRICE DATA
# ============================================================================

print()
print("=" * 100)
print("PRICE DATA")
print("=" * 100)

print(
    f"Price file: {PRICE_FILE}"
)

print(
    f"Period    : {START_DATE} -> {END_DATE}"
)


if not os.path.exists(
    PRICE_FILE
):

    print(
        "ERROR: Validated MIDSELECT price file "
        "does not exist."
    )

    raise SystemExit(1)


data = pd.read_csv(
    PRICE_FILE
)


if data.empty:

    print(
        "ERROR: Validated MIDSELECT price file "
        "is empty."
    )

    raise SystemExit(1)


# --------------------------------------------------------------------------
# Normalize date column.
# The validated source is expected to contain Date.
# --------------------------------------------------------------------------

if "Date" in data.columns:

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.set_index(
        "Date"
    )

elif "date" in data.columns:

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.set_index(
        "date"
    )

else:

    print(
        "ERROR: Could not find Date column "
        "in validated MIDSELECT price file."
    )

    print(
        "Available columns:"
    )

    for column in data.columns:
        print(
            f"  - {column}"
        )

    raise SystemExit(1)


data.index = pd.to_datetime(
    data.index
).normalize()


data = data[
    ~data.index.duplicated(
        keep="first"
    )
].copy()


# --------------------------------------------------------------------------
# Normalize OHLC column names.
# --------------------------------------------------------------------------

column_map = {}

for column in data.columns:

    normalized = str(
        column
    ).strip().lower()

    if normalized == "open":
        column_map[column] = "Open"

    elif normalized == "high":
        column_map[column] = "High"

    elif normalized == "low":
        column_map[column] = "Low"

    elif normalized == "close":
        column_map[column] = "Close"

    elif normalized == "volume":
        column_map[column] = "Volume"


data = data.rename(
    columns=column_map
)


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

    print(
        "ERROR: Missing required price columns:"
    )

    for column in missing_price_columns:
        print(
            f"  - {column}"
        )

    print()
    print(
        "Available columns:"
    )

    for column in data.columns:
        print(
            f"  - {column}"
        )

    raise SystemExit(1)


# --------------------------------------------------------------------------
# Numeric validation.
# --------------------------------------------------------------------------

for column in required_price_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


invalid_ohlc = int(
    (
        data[required_price_columns]
        .isna()
        .any(axis=1)
    ).sum()
)


if invalid_ohlc:

    print(
        f"ERROR: Invalid OHLC rows: "
        f"{invalid_ohlc}"
    )

    raise SystemExit(1)


if (
    (
        data["Open"] <= 0
    )
    | (
        data["High"] <= 0
    )
    | (
        data["Low"] <= 0
    )
    | (
        data["Close"] <= 0
    )
).any():

    print(
        "ERROR: Non-positive OHLC values found."
    )

    raise SystemExit(1)


if (
    data["High"]
    < data["Low"]
).any():

    print(
        "ERROR: High < Low detected."
    )

    raise SystemExit(1)


print(
    f"Rows loaded : {len(data)}"
)

print(
    f"Date range  : "
    f"{data.index.min().date()} "
    f"-> "
    f"{data.index.max().date()}"
)

print(
    f"Invalid OHLC: {invalid_ohlc}"
)

print(
    "PRICE DATA VALIDATION: PASS"
)

# ============================================================================
# TRADE-DATE VALIDATION
# ============================================================================

print()
print("=" * 100)
print("TRADE-DATE VALIDATION")
print("=" * 100)

missing_entry_dates = []

missing_exit_dates = []

for _, trade in trades.iterrows():

    entry_date = normalize_date(
        trade["entry_date"]
    )

    exit_date = normalize_date(
        trade["exit_date"]
    )

    if entry_date not in data.index:
        missing_entry_dates.append(
            entry_date
        )

    if exit_date not in data.index:
        missing_exit_dates.append(
            exit_date
        )


if missing_entry_dates:

    print(
        "ERROR: Missing entry dates:"
    )

    for date in sorted(
        set(missing_entry_dates)
    ):
        print(
            f"  {date.date()}"
        )

    raise SystemExit(1)


if missing_exit_dates:

    print(
        "ERROR: Missing exit dates:"
    )

    for date in sorted(
        set(missing_exit_dates)
    ):
        print(
            f"  {date.date()}"
        )

    raise SystemExit(1)


print(
    "TRADE-DATE VALIDATION: PASS"
)


# ============================================================================
# REPLAY
# ============================================================================

print()
print("=" * 100)
print("EXIT-PATH REPLAY")
print("=" * 100)

results = []

triggered_count = 0
resolved_count = 0
no_trigger_count = 0

for index, trade in trades.iterrows():

    direction = (
        str(
            trade["direction"]
        )
        .strip()
        .upper()
    )

    setup = str(
        trade["setup"]
    )

    entry_date = normalize_date(
        trade["entry_date"]
    )

    exit_date = normalize_date(
        trade["exit_date"]
    )

    entry_price = float(
        trade["entry_price"]
    )

    entry_atr = float(
        trade["entry_atr"]
    )

    initial_risk_total = float(
        trade["initial_risk"]
    )

    r_multiple_actual = float(
        trade["r_multiple"]
    )

    shares = (
        initial_risk_total
        / (
            INITIAL_STOP_ATR
            * entry_atr
        )
        if entry_atr > 0
        else np.nan
    )

    risk_per_share = (
        initial_risk_total
        / shares
        if shares > 0
        else np.nan
    )

    # Production risk/share should reconstruct to:
    # INITIAL_STOP_ATR * entry_atr
    expected_risk_per_share = (
        INITIAL_STOP_ATR
        * entry_atr
    )

    if not np.isfinite(
        risk_per_share
    ):

        results.append(
            {
                "index": index,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "setup": setup,
                "actual_r": r_multiple_actual,
                "cf_r": r_multiple_actual,
                "triggered": False,
                "trigger_bar": None,
                "cf_exit_bar": None,
                "cf_exit_date": None,
                "cf_exit_reason": (
                    "RECONSTRUCTION_ERROR"
                ),
                "trigger_to_exit_bars": None,
                "lifetime_bars": None,
            }
        )

        continue


    # ------------------------------------------------------------------------
    # Determine actual trade lifetime.
    # ------------------------------------------------------------------------

    lifetime_rows = data.loc[
        (data.index >= entry_date)
        & (data.index <= exit_date)
    ].copy()


    if lifetime_rows.empty:

        results.append(
            {
                "index": index,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "setup": setup,
                "actual_r": r_multiple_actual,
                "cf_r": r_multiple_actual,
                "triggered": False,
                "trigger_bar": None,
                "cf_exit_bar": None,
                "cf_exit_date": None,
                "cf_exit_reason": (
                    "NO_LIFETIME_DATA"
                ),
                "trigger_to_exit_bars": None,
                "lifetime_bars": 0,
            }
        )

        continue


    # ------------------------------------------------------------------------
    # Build price levels.
    # ------------------------------------------------------------------------

    levels = calculate_r_levels(
        direction=direction,
        entry_price=entry_price,
        risk_per_share=expected_risk_per_share,
    )

    target_price = levels[
        "target"
    ]

    trigger_price = levels[
        "trigger"
    ]

    protection_price = levels[
        "protection"
    ]


    # ------------------------------------------------------------------------
    # Scan the actual lifetime.
    #
    # Bar 1 is the entry-date OHLC bar.
    # This matches the production early-development
    # convention already established for TradeSense.
    # ------------------------------------------------------------------------

    trigger_found = False

    trigger_bar = None

    trigger_date = None

    trigger_position = None

    cf_exit_bar = None

    cf_exit_date = None

    cf_exit_reason = None

    cf_r = r_multiple_actual


    for bar_number, (
        bar_date,
        bar,
    ) in enumerate(
        lifetime_rows.iterrows(),
        start=1,
    ):

        high = float(
            bar["High"]
        )

        low = float(
            bar["Low"]
        )


        # ------------------------------------------------------------
        # Before trigger:
        # detect first +1.5R favorable touch.
        # ------------------------------------------------------------

        if not trigger_found:

            if direction == "LONG":

                trigger_hit = (
                    high
                    >= trigger_price
                )

            else:

                trigger_hit = (
                    low
                    <= trigger_price
                )


            if trigger_hit:

                trigger_found = True

                trigger_bar = bar_number

                trigger_date = bar_date

                trigger_position = (
                    bar_number
                )

                triggered_count += 1

                # IMPORTANT:
                # If +2R is also reached on the same bar,
                # the target has priority because the trade
                # would already have reached the production
                # target before a hypothetical protection exit
                # can be considered.

                if direction == "LONG":

                    target_hit = (
                        high
                        >= target_price
                    )

                else:

                    target_hit = (
                        low
                        <= target_price
                    )


                if target_hit:

                    cf_exit_bar = (
                        bar_number
                    )

                    cf_exit_date = (
                        bar_date
                    )

                    cf_exit_reason = (
                        "TARGET_2R"
                    )

                    cf_r = (
                        TARGET_R
                    )

                    resolved_count += 1

                    break

                # Otherwise protection becomes active
                # starting with subsequent bars.
                continue


        # ------------------------------------------------------------
        # After +1.5R trigger:
        #
        # We use SUBSEQUENT bars for the protection test.
        #
        # This avoids assuming an intra-bar sequence when both
        # protection and trigger levels are touched on the same bar.
        # ------------------------------------------------------------

        if trigger_found:

            if bar_number <= trigger_position:

                continue


            if direction == "LONG":

                target_hit = (
                    high
                    >= target_price
                )

                protection_hit = (
                    low
                    <= protection_price
                )

            else:

                target_hit = (
                    low
                    <= target_price
                )

                protection_hit = (
                    high
                    >= protection_price
                )


            # If both occur in the same subsequent bar,
            # OHLC alone cannot establish intrabar ordering.
            # We conservatively classify this as ambiguous and
            # retain the actual production result rather than
            # manufacturing an outcome.

            if (
                target_hit
                and protection_hit
            ):

                cf_exit_bar = (
                    bar_number
                )

                cf_exit_date = (
                    bar_date
                )

                cf_exit_reason = (
                    "AMBIGUOUS_TARGET_PROTECTION"
                )

                cf_r = r_multiple_actual

                resolved_count += 1

                break


            if target_hit:

                cf_exit_bar = (
                    bar_number
                )

                cf_exit_date = (
                    bar_date
                )

                cf_exit_reason = (
                    "TARGET_2R"
                )

                cf_r = TARGET_R

                resolved_count += 1

                break


            if protection_hit:

                cf_exit_bar = (
                    bar_number
                )

                cf_exit_date = (
                    bar_date
                )

                cf_exit_reason = (
                    "PROTECTED_1R"
                )

                cf_r = PROTECTION_R

                resolved_count += 1

                break


    # ------------------------------------------------------------------------
    # No counterfactual exit before actual production exit.
    # ------------------------------------------------------------------------

    if trigger_found and cf_exit_reason is None:

        no_trigger_count += 1

        cf_exit_reason = (
            "RETAIN_ACTUAL_EXIT"
        )

        cf_r = r_multiple_actual


    elif not trigger_found:

        no_trigger_count += 1

        cf_exit_reason = (
            "NO_TRIGGER_RETAIN_ACTUAL"
        )

        cf_r = r_multiple_actual


    trigger_to_exit_bars = None

    if (
        trigger_found
        and cf_exit_bar is not None
    ):

        trigger_to_exit_bars = (
            cf_exit_bar
            - trigger_bar
        )


    results.append(
        {
            "index": index,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": direction,
            "setup": setup,
            "actual_r": r_multiple_actual,
            "cf_r": cf_r,
            "delta_r": (
                cf_r
                - r_multiple_actual
            ),
            "triggered": trigger_found,
            "trigger_bar": trigger_bar,
            "trigger_date": trigger_date,
            "cf_exit_bar": cf_exit_bar,
            "cf_exit_date": cf_exit_date,
            "cf_exit_reason": cf_exit_reason,
            "trigger_to_exit_bars": (
                trigger_to_exit_bars
            ),
            "lifetime_bars": len(
                lifetime_rows
            ),
            "max_favorable_r_saved": float(
                trade["max_favorable_r"]
            ),
        }
    )


results_df = pd.DataFrame(
    results
)


# ============================================================================
# RECONCILIATION
# ============================================================================

print()
print("=" * 100)
print("BASELINE RECONCILIATION")
print("=" * 100)

actual_r_sum = (
    results_df["actual_r"]
    .sum()
)

frozen_r_sum = (
    trades["r_multiple"]
    .sum()
)

print(
    f"Frozen trade R : "
    f"{frozen_r_sum:+.2f}R"
)

print(
    f"Replay trade R : "
    f"{actual_r_sum:+.2f}R"
)

print(
    f"Difference     : "
    f"{actual_r_sum - frozen_r_sum:+.4f}R"
)


if abs(
    actual_r_sum
    - frozen_r_sum
) <= VALIDATION_TOLERANCE:

    print(
        "BASELINE RECONCILIATION: PASS"
    )

else:

    print(
        "BASELINE RECONCILIATION: FAIL"
    )

    raise SystemExit(1)


# ============================================================================
# TRIGGER VALIDATION
# ============================================================================

print()
print("=" * 100)
print("TRIGGER VALIDATION")
print("=" * 100)

trigger_validation = 0

trigger_mismatch = 0

for _, result in results_df.iterrows():

    saved_mfe = float(
        result[
            "max_favorable_r_saved"
        ]
    )

    expected_trigger = (
        saved_mfe
        >= TRIGGER_R
        - VALIDATION_TOLERANCE
    )

    actual_trigger = bool(
        result["triggered"]
    )

    if (
        expected_trigger
        == actual_trigger
    ):

        trigger_validation += 1

    else:

        trigger_mismatch += 1

        print(
            "MISMATCH:",
            result["entry_date"],
            result["direction"],
            result["setup"],
            "saved_mfe=",
            saved_mfe,
            "triggered=",
            actual_trigger,
        )


print(
    f"Validated trades : "
    f"{trigger_validation}"
)

print(
    f"Mismatches       : "
    f"{trigger_mismatch}"
)


if trigger_mismatch == 0:

    print(
        "TRIGGER VALIDATION: PASS"
    )

else:

    print(
        "TRIGGER VALIDATION: FAIL"
    )

    raise SystemExit(1)


# ============================================================================
# BASELINE VS COUNTERFACTUAL
# ============================================================================

print()
print("=" * 100)
print("BASELINE VS COUNTERFACTUAL")
print("=" * 100)

baseline_values = (
    results_df["actual_r"]
    .tolist()
)

counterfactual_values = (
    results_df["cf_r"]
    .tolist()
)

baseline_metrics = print_metrics(
    "ACTUAL PRODUCTION EXIT",
    baseline_values,
)

cf_metrics = print_metrics(
    "COUNTERFACTUAL",
    counterfactual_values,
)

delta_total = (
    cf_metrics["total_r"]
    - baseline_metrics["total_r"]
)

print()
print(
    f"Total R delta: "
    f"{delta_total:+.2f}R"
)


# ============================================================================
# TRIGGERED TRADES
# ============================================================================

print()
print("=" * 100)
print("TRIGGERED TRADES")
print("=" * 100)

triggered_df = results_df[
    results_df["triggered"]
].copy()


print(
    f"Triggered trades: "
    f"{len(triggered_df)}"
)


if not triggered_df.empty:

    print()

    triggered_actual = (
        triggered_df[
            "actual_r"
        ]
    )

    triggered_cf = (
        triggered_df[
            "cf_r"
        ]
    )

    print_metrics(
        "TRIGGERED — ACTUAL",
        triggered_actual,
    )

    print_metrics(
        "TRIGGERED — COUNTERFACTUAL",
        triggered_cf,
    )

    print()

    for _, row in (
        triggered_df
        .sort_values("entry_date")
        .iterrows()
    ):

        print(
            f"{row['entry_date'].date()} | "
            f"{row['exit_date'].date()} | "
            f"{row['direction']:<5} | "
            f"{row['setup']:<20} | "
            f"actual={row['actual_r']:+.2f}R | "
            f"CF={row['cf_r']:+.2f}R | "
            f"delta={row['delta_r']:+.2f}R | "
            f"trigger_bar="
            f"{int(row['trigger_bar'])} | "
            f"CF={row['cf_exit_reason']}"
        )


# ============================================================================
# COUNTERFACTUAL EXIT REASONS
# ============================================================================

print()
print("=" * 100)
print("COUNTERFACTUAL EXIT REASONS")
print("=" * 100)

if results_df.empty:

    print(
        "No results."
    )

else:

    reason_groups = (
        results_df
        .groupby(
            "cf_exit_reason"
        )["cf_r"]
        .agg(
            [
                "count",
                "sum",
            ]
        )
        .sort_index()
    )

    for reason, row in (
        reason_groups.iterrows()
    ):

        print(
            f"{reason:<35}: "
            f"{int(row['count']):>3} trades | "
            f"{row['sum']:+.2f}R"
        )


# ============================================================================
# YEAR ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("YEAR ROBUSTNESS")
print("=" * 100)

results_df["year"] = (
    results_df["entry_date"]
    .dt.year
)


year_results = []

for year in sorted(
    results_df["year"]
    .dropna()
    .unique()
):

    group = results_df[
        results_df["year"] == year
    ]

    actual = group[
        "actual_r"
    ].sum()

    counterfactual = group[
        "cf_r"
    ].sum()

    delta = (
        counterfactual
        - actual
    )

    year_results.append(
        {
            "year": int(year),
            "trades": len(group),
            "actual_r": actual,
            "counterfactual_r": counterfactual,
            "delta_r": delta,
        }
    )

    print(
        f"{int(year)}: "
        f"{len(group)} trades | "
        f"actual={actual:+.2f}R | "
        f"CF={counterfactual:+.2f}R | "
        f"delta={delta:+.2f}R"
    )


# ============================================================================
# LEAVE-ONE-OUT TRADE ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("LEAVE-ONE-OUT ROBUSTNESS")
print("=" * 100)

positive_loo = 0
negative_loo = 0
unchanged_loo = 0

loo_rows = []

for excluded_index in results_df.index:

    remaining = results_df[
        results_df.index
        != excluded_index
    ]

    actual = (
        remaining["actual_r"]
        .sum()
    )

    counterfactual = (
        remaining["cf_r"]
        .sum()
    )

    delta = (
        counterfactual
        - actual
    )

    if delta > VALIDATION_TOLERANCE:

        positive_loo += 1

    elif delta < -VALIDATION_TOLERANCE:

        negative_loo += 1

    else:

        unchanged_loo += 1

    row = results_df.loc[
        excluded_index
    ]

    loo_rows.append(
        {
            "excluded_entry_date": (
                row["entry_date"]
            ),
            "excluded_direction": (
                row["direction"]
            ),
            "excluded_setup": (
                row["setup"]
            ),
            "remaining_trades": (
                len(remaining)
            ),
            "actual_r": actual,
            "counterfactual_r": (
                counterfactual
            ),
            "delta_r": delta,
        }
    )


print(
    f"Positive LOO : "
    f"{positive_loo}"
)

print(
    f"Negative LOO : "
    f"{negative_loo}"
)

print(
    f"Unchanged LOO: "
    f"{unchanged_loo}"
)


# ============================================================================
# OUTPUT
# ============================================================================

output_dir = (
    "tests/output/index_backtests"
)

os.makedirs(
    output_dir,
    exist_ok=True,
)


results_file = os.path.join(
    output_dir,
    "midselect_exit_counterfactual.csv",
)


year_file = os.path.join(
    output_dir,
    "midselect_exit_counterfactual_years.csv",
)


loo_file = os.path.join(
    output_dir,
    "midselect_exit_counterfactual_loo.csv",
)


results_df.to_csv(
    results_file,
    index=False,
)


pd.DataFrame(
    year_results
).to_csv(
    year_file,
    index=False,
)


pd.DataFrame(
    loo_rows
).to_csv(
    loo_file,
    index=False,
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Trade results : {results_file}"
)

print(
    f"Year results  : {year_file}"
)

print(
    f"LOO results   : {loo_file}"
)


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)

print(
    f"Baseline R : "
    f"{baseline_metrics['total_r']:+.2f}R"
)

print(
    f"CF R       : "
    f"{cf_metrics['total_r']:+.2f}R"
)

print(
    f"Delta      : "
    f"{delta_total:+.2f}R"
)

print(
    f"Triggered  : "
    f"{len(triggered_df)}"
)

print(
    f"Positive LOO : "
    f"{positive_loo}"
)

print(
    f"Negative LOO : "
    f"{negative_loo}"
)

print(
    f"Unchanged LOO: "
    f"{unchanged_loo}"
)

print()

if len(trades) < 20:

    print(
        "VERDICT: INSUFFICIENT_SAMPLE"
    )

    print()
    print(
        "The counterfactual may reveal a useful "
        "research signal, but MIDSELECT has only "
        f"{len(trades)} historical trades."
    )

    print(
        "No exit rule should be implemented from "
        "this sample alone."
    )

else:

    if (
        delta_total > 0
        and negative_loo == 0
    ):

        print(
            "VERDICT: PROMISING_ROBUSTNESS_SIGNAL"
        )

    else:

        print(
            "VERDICT: ROBUSTNESS_NOT_ESTABLISHED"
        )


print()
print(
    "RESEARCH ONLY."
)

print(
    "Production strategy was NOT modified."
)

print(
    "Frozen historical stock dataset was NOT modified."
)

print(
    "Holdout dataset was NOT modified."
)