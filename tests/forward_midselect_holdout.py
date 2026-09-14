"""
TradeSense-AI
NIFTY MIDCAP SELECT — Genuine Forward / OOS Holdout Collector

RESEARCH-ONLY.

Frozen forward hypothesis:
    For genuinely unseen MIDSELECT trades, record whether the trade
    reaches 1.25R, 1.50R, or 1.75R during its actual production lifetime.

This script DOES NOT:
    - modify production code
    - modify the frozen historical stock dataset
    - modify the historical MIDSELECT backtest dataset
    - modify any existing stock holdout
    - select an exit parameter
    - alter trade outcomes
    - use future candles to generate signals

Forward start:
    2026-09-14

Historical price data is allowed as SIGNAL CONTEXT.
Only trades whose actual ENTRY DATE >= 2026-09-14
are admitted to the forward ledger.

The validated MIDSELECT source is:
    tests/output/index_backtests/midselect_niftyindices_ohlc.csv
"""

import csv
import os
import sys

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "NIFTY MIDCAP SELECT"

FORWARD_START_DATE = "2026-09-14"

PRICE_FILE = (
    "tests/output/index_backtests/"
    "midselect_niftyindices_ohlc.csv"
)

HISTORICAL_TRADES_FILE = (
    "tests/output/index_backtests/"
    "midselect_index_trades.csv"
)

FORWARD_HOLDOUT_FILE = (
    "tests/output/index_backtests/"
    "midselect_forward_holdout.csv"
)

INITIAL_CAPITAL = 1_000_000

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0
RISK_PER_TRADE = 0.01
SWING_WINDOW = 3

THRESHOLDS = [
    1.25,
    1.50,
    1.75,
]

EXPECTED_HISTORICAL_TRADES = 13


# ============================================================================
# IMPORT PRODUCTION SERVICE
# ============================================================================

try:

    from app.services.backtest_service import (
        BacktestService,
    )

except Exception as exc:

    print()
    print("=" * 100)
    print("ERROR IMPORTING BACKTEST SERVICE")
    print("=" * 100)
    print(exc)
    sys.exit(1)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):

    return pd.Timestamp(
        value
    ).normalize()


def make_trade_key(
    trade,
):

    direction = str(
        trade.get(
            "direction",
            "",
        )
    ).strip().upper()

    entry_date = str(
        trade.get(
            "entry_date",
            "",
        )
    )[:10]

    return (
        entry_date,
        direction,
    )


def get_threshold_price(
    direction,
    entry_price,
    risk_per_share,
    threshold_r,
):

    if direction == "LONG":

        return (
            entry_price
            + threshold_r
            * risk_per_share
        )

    return (
        entry_price
        - threshold_r
        * risk_per_share
    )


def calculate_threshold_paths(
    price_df,
    trade,
):

    """
    Analyze favorable development during the ACTUAL
    production lifetime of the trade.

    No lifetime extension is performed.
    """

    direction = str(
        trade.get(
            "direction",
            "",
        )
    ).strip().upper()

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

    risk_per_share = (
        INITIAL_STOP_ATR
        * entry_atr
    )

    lifetime = price_df.loc[
        (
            price_df.index
            >= entry_date
        )
        & (
            price_df.index
            <= exit_date
        )
    ].copy()

    results = {}

    for threshold in THRESHOLDS:

        results[
            threshold
        ] = {
            "reached": False,
            "bar": "",
            "date": "",
        }


    for bar_number, (
        bar_date,
        bar,
    ) in enumerate(
        lifetime.iterrows(),
        start=1,
    ):

        high = float(
            bar["High"]
        )

        low = float(
            bar["Low"]
        )


        for threshold in THRESHOLDS:

            result = results[
                threshold
            ]

            if result["reached"]:

                continue


            trigger_price = (
                get_threshold_price(
                    direction=direction,
                    entry_price=entry_price,
                    risk_per_share=(
                        risk_per_share
                    ),
                    threshold_r=threshold,
                )
            )


            if direction == "LONG":

                reached = (
                    high
                    >= trigger_price
                )

            else:

                reached = (
                    low
                    <= trigger_price
                )


            if reached:

                result[
                    "reached"
                ] = True

                result[
                    "bar"
                ] = bar_number

                result[
                    "date"
                ] = (
                    bar_date.strftime(
                        "%Y-%m-%d"
                    )
                )


    output = {}

    for threshold in THRESHOLDS:

        suffix = (
            str(
                threshold
            )
            .replace(
                ".",
                "_",
            )
        )

        output[
            f"reached_{suffix}r"
        ] = results[
            threshold
        ][
            "reached"
        ]

        output[
            f"trigger_bar_{suffix}r"
        ] = results[
            threshold
        ][
            "bar"
        ]

        output[
            f"trigger_date_{suffix}r"
        ] = results[
            threshold
        ][
            "date"
        ]


    return output


# ============================================================================
# HEADER
# ============================================================================

print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT FORWARD / OOS HOLDOUT"
)
print("=" * 100)

print(
    f"Index               : {INDEX_NAME}"
)

print(
    f"Forward start       : {FORWARD_START_DATE}"
)

print(
    f"Validated price file: {PRICE_FILE}"
)

print(
    f"Historical file     : {HISTORICAL_TRADES_FILE}"
)

print(
    f"Forward ledger      : {FORWARD_HOLDOUT_FILE}"
)


# ============================================================================
# LOAD VALIDATED MIDSELECT DATA
# ============================================================================

print()
print("=" * 100)
print("LOAD VALIDATED MIDSELECT PRICE DATA")
print("=" * 100)


if not os.path.exists(
    PRICE_FILE
):

    print(
        "ERROR: Validated MIDSELECT "
        "price file does not exist."
    )

    sys.exit(1)


raw_df = pd.read_csv(
    PRICE_FILE
)


if raw_df.empty:

    print(
        "ERROR: Validated MIDSELECT "
        "price file is empty."
    )

    sys.exit(1)


# ---------------------------------------------------------------------------
# Identify date column.
# ---------------------------------------------------------------------------

date_column = None

for candidate in [
    "Date",
    "date",
]:

    if candidate in raw_df.columns:

        date_column = candidate
        break


if date_column is None:

    print(
        "ERROR: No Date column found."
    )

    sys.exit(1)


raw_df[date_column] = pd.to_datetime(
    raw_df[date_column],
    errors="coerce",
)


raw_df = raw_df.dropna(
    subset=[
        date_column
    ]
)


raw_df = raw_df.set_index(
    date_column
)


raw_df.index = (
    pd.to_datetime(
        raw_df.index
    )
    .normalize()
)


# ---------------------------------------------------------------------------
# Normalize OHLC names.
# ---------------------------------------------------------------------------

rename_map = {}

for column in raw_df.columns:

    normalized = (
        str(column)
        .strip()
        .lower()
    )

    if normalized == "open":

        rename_map[column] = "Open"

    elif normalized == "high":

        rename_map[column] = "High"

    elif normalized == "low":

        rename_map[column] = "Low"

    elif normalized == "close":

        rename_map[column] = "Close"


price_df = raw_df.rename(
    columns=rename_map
).copy()


required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
]


missing_columns = [
    column
    for column in required_columns
    if column not in price_df.columns
]


if missing_columns:

    print(
        "ERROR: Missing OHLC columns:"
    )

    for column in missing_columns:

        print(
            f"  - {column}"
        )

    sys.exit(1)


for column in required_columns:

    price_df[column] = pd.to_numeric(
        price_df[column],
        errors="coerce",
    )


invalid_ohlc = int(
    price_df[
        required_columns
    ]
    .isna()
    .any(axis=1)
    .sum()
)


if invalid_ohlc:

    print(
        f"ERROR: Invalid OHLC rows: "
        f"{invalid_ohlc}"
    )

    sys.exit(1)


non_positive = (
    (
        price_df["Open"] <= 0
    )
    | (
        price_df["High"] <= 0
    )
    | (
        price_df["Low"] <= 0
    )
    | (
        price_df["Close"] <= 0
    )
)


if non_positive.any():

    print(
        "ERROR: Non-positive OHLC detected."
    )

    sys.exit(1)


if (
    price_df["High"]
    < price_df["Low"]
).any():

    print(
        "ERROR: High < Low detected."
    )

    sys.exit(1)


duplicate_dates = int(
    price_df.index.duplicated(
        keep=False
    ).sum()
)


if duplicate_dates:

    print(
        f"ERROR: Duplicate dates: "
        f"{duplicate_dates}"
    )

    sys.exit(1)


price_df = price_df.sort_index()


print(
    f"Rows available : "
    f"{len(price_df)}"
)

print(
    f"Date range     : "
    f"{price_df.index.min().date()} "
    f"-> "
    f"{price_df.index.max().date()}"
)

print(
    "PRICE DATA VALIDATION: PASS"
)


# ============================================================================
# LOAD BACKTEST SERVICE
# ============================================================================

print()
print("=" * 100)
print("GENERATE CURRENT SIGNAL CONTEXT")
print("=" * 100)


service = BacktestService(
    symbol="NIFTY_MIDCAP_SELECT",
    strategy="tradesense",
    initial_capital=INITIAL_CAPITAL,
)


# IMPORTANT:
# MIDSELECT does not have a usable Yahoo Finance ticker.
#
# The validated Nifty Indices OHLC source was already loaded above.
# Therefore use that validated dataframe directly as the signal context.
#
# This keeps the forward collector independent of Yahoo's unavailable
# NIFTY_MIDCAP_SELECT symbol while still using the production
# BacktestService signal/backtest logic.
#
# Historical data is context only.
# The forward boundary is enforced AFTER trade generation using entry_date.

service_df = price_df.copy()


if service_df is None or service_df.empty:

    print(
        "ERROR: Validated MIDSELECT price "
        "data is empty."
    )

    sys.exit(1)


print(
    f"Signal context rows: "
    f"{len(service_df)}"
)

print(
    f"Signal context date range: "
    f"{service_df.index.min().date()} "
    f"-> "
    f"{service_df.index.max().date()}"
)

print(
    "SIGNAL CONTEXT SOURCE: "
    "VALIDATED NIFTY INDICES DATA"
)


# ============================================================================
# GENERATE SIGNALS
# ============================================================================

signals_df = service.generate_tradesense_signals(
    df=service_df,
    swing_window=SWING_WINDOW,
)


if signals_df is None or signals_df.empty:

    print(
        "ERROR: Signal generation "
        "returned no data."
    )

    sys.exit(1)


print(
    f"Signal rows: "
    f"{len(signals_df)}"
)


if "Signal" not in signals_df.columns:

    print(
        "ERROR: Signal column missing."
    )

    sys.exit(1)


signal_counts = (
    signals_df["Signal"]
    .value_counts()
    .sort_index()
)


print(
    "Signal distribution:"
)

for value, count in signal_counts.items():

    print(
        f"  {value:+}: {count}"
    )


# ============================================================================
# RUN PRODUCTION-MATCHED BACKTEST
# ============================================================================

print()
print("=" * 100)
print("RUN FORWARD-CANDIDATE BACKTEST")
print("=" * 100)


backtest_result = (
    service.run_directional_backtest(
        signals_df,
        signal_column="Signal",
        initial_stop_atr=INITIAL_STOP_ATR,
        target_r=TARGET_R,
        risk_per_trade=RISK_PER_TRADE,
    )
)


if not isinstance(
    backtest_result,
    dict,
):

    print(
        "ERROR: Unexpected backtest "
        "result type."
    )

    sys.exit(1)


all_generated_trades = (
    backtest_result.get(
        "trades",
        [],
    )
)


print(
    f"Total generated trades: "
    f"{len(all_generated_trades)}"
)


# ============================================================================
# FILTER GENUINE OOS TRADES
# ============================================================================

forward_start = normalize_date(
    FORWARD_START_DATE
)


forward_trades = []


for trade in all_generated_trades:

    if not trade.get(
        "entry_date"
    ):

        continue


    entry_date = normalize_date(
        trade["entry_date"]
    )


    # ------------------------------------------------------------------------
    # The ONLY criterion for forward admission:
    #
    # actual trade entry date >= frozen forward start.
    # ------------------------------------------------------------------------

    if entry_date < forward_start:

        continue


    # FINAL_LIQUIDATION is not a genuine new signal/trade.
    if (
        str(
            trade.get(
                "exit_reason",
                "",
            )
        )
        .strip()
        .upper()
        == "FINAL_LIQUIDATION"
    ):

        continue


    forward_trade = dict(
        trade
    )


    # ------------------------------------------------------------------------
    # Add frozen-hypothesis path information.
    # ------------------------------------------------------------------------

    path_info = calculate_threshold_paths(
        price_df=price_df,
        trade=forward_trade,
    )


    forward_trade.update(
        path_info
    )


    forward_trade[
        "forward_start_date"
    ] = FORWARD_START_DATE


    forward_trade[
        "forward_status"
    ] = "GENUINE_OOS"


    forward_trades.append(
        forward_trade
    )


print()
print(
    f"Forward trades generated: "
    f"{len(forward_trades)}"
)


# ============================================================================
# LOAD EXISTING FORWARD LEDGER
# ============================================================================

print()
print("=" * 100)
print("LOAD EXISTING FORWARD LEDGER")
print("=" * 100)


existing_forward_trades = []


if os.path.exists(
    FORWARD_HOLDOUT_FILE
):

    try:

        existing_df = pd.read_csv(
            FORWARD_HOLDOUT_FILE
        )

        if not existing_df.empty:

            existing_forward_trades = (
                existing_df
                .to_dict(
                    orient="records"
                )
            )

    except Exception as exc:

        print(
            "ERROR reading existing "
            "forward ledger:"
        )

        print(
            exc
        )

        sys.exit(1)


print(
    f"Existing forward trades: "
    f"{len(existing_forward_trades)}"
)


# ============================================================================
# DEDUPLICATE / MERGE
# ============================================================================

combined = []


seen_keys = set()


# Existing ledger first.
for trade in existing_forward_trades:

    key = make_trade_key(
        trade
    )

    if key in seen_keys:

        continue

    seen_keys.add(
        key
    )

    combined.append(
        trade
    )


new_trade_count = 0


# New generated forward trades.
for trade in forward_trades:

    key = make_trade_key(
        trade
    )

    if key in seen_keys:

        continue

    seen_keys.add(
        key
    )

    combined.append(
        trade
    )

    new_trade_count += 1


print(
    f"New unique forward trades: "
    f"{new_trade_count}"
)

print(
    f"Combined forward trades: "
    f"{len(combined)}"
)


# ============================================================================
# SAFETY: FORWARD LEDGER MUST CONTAIN ONLY OOS TRADES
# ============================================================================

invalid_forward_entries = []


for trade in combined:

    entry_date = normalize_date(
        trade["entry_date"]
    )

    if entry_date < forward_start:

        invalid_forward_entries.append(
            (
                entry_date,
                trade.get(
                    "direction",
                    "",
                ),
            )
        )


if invalid_forward_entries:

    print()
    print(
        "ERROR: Forward ledger contains "
        "pre-forward trades."
    )

    for item in invalid_forward_entries:

        print(
            f"  - {item}"
        )

    sys.exit(1)


# ============================================================================
# PROTECT AGAINST ACCIDENTAL HISTORICAL FILE USE
# ============================================================================

print()
print("=" * 100)
print("DATASET SAFETY CHECK")
print("=" * 100)


if os.path.exists(
    HISTORICAL_TRADES_FILE
):

    historical_df = pd.read_csv(
        HISTORICAL_TRADES_FILE
    )

    historical_count = len(
        historical_df
    )

    print(
        f"Historical MIDSELECT trades: "
        f"{historical_count}"
    )

    if (
        historical_count
        != EXPECTED_HISTORICAL_TRADES
    ):

        print(
            "WARNING: Historical trade "
            "count differs from expected "
            f"{EXPECTED_HISTORICAL_TRADES}."
        )

else:

    print(
        "WARNING: Historical MIDSELECT "
        "trade file not found."
    )

print(
    "Historical dataset is read-only."
)

print(
    "Production code is read-only."
)

print(
    "Stock holdout is untouched."
)


# ============================================================================
# SAVE FORWARD LEDGER
# ============================================================================

if combined:

    fieldnames = sorted(
        {
            key
            for trade in combined
            for key in trade.keys()
        }
    )


    os.makedirs(
        os.path.dirname(
            FORWARD_HOLDOUT_FILE
        ),
        exist_ok=True,
    )


    with open(
        FORWARD_HOLDOUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            combined
        )


    print()
    print(
        f"Forward ledger saved: "
        f"{FORWARD_HOLDOUT_FILE}"
    )

else:

    print()
    print(
        "No genuine OOS trades exist."
    )

    print(
        "Forward ledger not created/modified."
    )


# ============================================================================
# FORWARD SUMMARY
# ============================================================================

print()
print("=" * 100)
print("FORWARD OOS SUMMARY")
print("=" * 100)


if not combined:

    print(
        "Genuine OOS trades: 0"
    )

    print(
        "No conclusion can be drawn."
    )

else:

    combined_df = pd.DataFrame(
        combined
    )


    combined_df[
        "r_multiple"
    ] = pd.to_numeric(
        combined_df[
            "r_multiple"
        ],
        errors="coerce",
    )


    total_r = float(
        combined_df[
            "r_multiple"
        ].sum()
    )


    winners = int(
        (
            combined_df[
                "r_multiple"
            ]
            > 0
        ).sum()
    )


    losers = int(
        (
            combined_df[
                "r_multiple"
            ]
            < 0
        ).sum()
    )


    print(
        f"Trades       : "
        f"{len(combined_df)}"
    )

    print(
        f"Winners      : "
        f"{winners}"
    )

    print(
        f"Losers       : "
        f"{losers}"
    )

    print(
        f"Win rate     : "
        f"{winners / len(combined_df) * 100:.2f}%"
    )

    print(
        f"Total R      : "
        f"{total_r:+.2f}R"
    )


    for threshold in THRESHOLDS:

        suffix = (
            str(
                threshold
            )
            .replace(
                ".",
                "_",
            )
        )


        column = (
            f"reached_{suffix}r"
        )


        if column not in combined_df:

            continue


        reached = (
            combined_df[
                column
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
        )


        count = int(
            reached.sum()
        )


        reached_r = float(
            combined_df.loc[
                reached,
                "r_multiple"
            ].sum()
        )


        print(
            f"{threshold:.2f}R reached: "
            f"{count} trades | "
            f"{reached_r:+.2f}R"
        )


# ============================================================================
# CURRENT OOS STATUS
# ============================================================================

print()
print("=" * 100)
print("FORWARD VALIDATION STATUS")
print("=" * 100)


genuine_oos_count = len(
    combined
)


if genuine_oos_count == 0:

    print(
        "STATUS: WAITING_FOR_GENUINE_OOS_TRADES"
    )

    print(
        "The current completed price data "
        "does not contain a post-"
        f"{FORWARD_START_DATE} trade."
    )

elif genuine_oos_count < 5:

    print(
        "STATUS: EARLY_OOS_SAMPLE"
    )

    print(
        "Insufficient evidence for "
        "strategy or exit conclusions."
    )

else:

    print(
        "STATUS: OOS_DATA_ACCUMULATING"
    )

    print(
        "Do not tune parameters from "
        "this early sample."
    )


print()
print("=" * 100)
print("FINAL RESEARCH VERDICT")
print("=" * 100)

print(
    "The forward ledger is isolated "
    "from the historical MIDSELECT dataset."
)

print(
    "Only entry dates on/after "
    f"{FORWARD_START_DATE} are admitted."
)

print(
    "Threshold path measurements are "
    "observational only."
)

print(
    "No production rule has been changed."
)

print(
    "No historical dataset has been changed."
)

print(
    "No stock holdout has been changed."
)

print()
print(
    "RESEARCH ONLY."
)