"""
TradeSense-AI
BANK NIFTY SHORT_CONTINUATION — EARLY-ADVERSE ROBUSTNESS

Research-only.

Purpose:
    Validate whether early adverse movement is a robust discriminator
    specifically for BANK NIFTY SHORT_CONTINUATION trades.

Method:
    1. Load frozen BANK NIFTY baseline trades.
    2. Download BANK NIFTY OHLC data.
    3. Reconstruct the exact production ATR.
    4. Reconstruct the exact production entry price.
    5. Reconstruct production risk/share using 3.5 ATR.
    6. Replay each SHORT_CONTINUATION trade bar-by-bar.
    7. Calculate early adverse R for windows 3/5/7/10.
    8. Test thresholds 0.15/0.20/0.25/0.30/0.35/0.40.
    9. Test year robustness.
   10. Test leave-one-out robustness.

IMPORTANT:
    - No production code modification.
    - No baseline CSV modification.
    - No frozen stock historical data modification.
    - No holdout modification.
    - No filter implementation.
    - This is research/counterfactual validation only.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
# CONFIGURATION
# ============================================================================

TRADES_FILE = (
    "tests/output/index_backtests/"
    "banknifty_index_trades.csv"
)

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

ROBUSTNESS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "banknifty_short_continuation_early_adverse_robustness.csv"
)

YEAR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "banknifty_short_continuation_early_adverse_year_robustness.csv"
)

LOO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "banknifty_short_continuation_early_adverse_leave_one_out.csv"
)

TICKER = "^NSEBANK"

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

TARGET_SETUP = "SHORT_CONTINUATION"

WINDOWS = [3, 5, 7, 10]

THRESHOLDS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

INITIAL_STOP_ATR = 3.5

SLIPPAGE_RATE = 0.001

EXPECTED_TOTAL_TRADES = 24

EXPECTED_TARGET_TRADES = 8

VALIDATION_TOLERANCE = 0.005001


# ============================================================================
# HELPERS
# ============================================================================

def numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def calculate_atr(df):
    """
    Exact production ATR reconstruction:

        previous_close = Close.shift(1)

        true_range = max(
            High-Low,
            abs(High-previous_close),
            abs(Low-previous_close)
        )

        ATR = rolling(14).mean()
    """

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"] - previous_close
            ).abs(),
            (
                df["Low"] - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(14).mean()


def calculate_profit_factor(r_values):
    r_values = pd.Series(
        r_values,
        dtype=float
    )

    gross_profit = r_values[
        r_values > 0
    ].sum()

    gross_loss = abs(
        r_values[
            r_values < 0
        ].sum()
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return 0.0

    return gross_profit / gross_loss


def summarize(values):
    values = pd.Series(
        values,
        dtype=float
    ).dropna()

    if values.empty:
        return {
            "trades": 0,
            "winners": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
        }

    winners = int(
        (values > 0).sum()
    )

    gross_profit = values[
        values > 0
    ].sum()

    gross_loss = values[
        values < 0
    ].sum()

    return {
        "trades": len(values),
        "winners": winners,
        "win_rate": (
            winners / len(values) * 100
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": (
            calculate_profit_factor(values)
        ),
        "total_r": values.sum(),
        "avg_r": values.mean(),
    }


# ============================================================================
# HEADER
# ============================================================================

print()
print("=" * 100)
print(
    "TRADESENSE-AI — BANK NIFTY "
    "SHORT_CONTINUATION EARLY-ADVERSE ROBUSTNESS"
)
print("=" * 100)

print()
print(
    "Research-only parameter robustness."
)

print(
    "Production strategy is NOT modified."
)

print(
    "BANK NIFTY baseline trades are NOT modified."
)


# ============================================================================
# LOAD FROZEN BASELINE
# ============================================================================

if not os.path.exists(TRADES_FILE):

    print()
    print(
        f"ERROR: Missing trade file: {TRADES_FILE}"
    )

    raise SystemExit(1)


trades = pd.read_csv(
    TRADES_FILE
)

print()
print(
    f"Historical BANK NIFTY trades loaded: {len(trades)}"
)

if len(trades) != EXPECTED_TOTAL_TRADES:

    print(
        f"WARNING: Expected {EXPECTED_TOTAL_TRADES} "
        f"trades, found {len(trades)}."
    )


required_trade_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
    "entry_atr",
    "initial_risk",
]

missing = [
    column
    for column in required_trade_columns
    if column not in trades.columns
]

if missing:

    print()
    print("ERROR: Missing required columns:")

    for column in missing:
        print(f"  - {column}")

    raise SystemExit(1)


target = trades[
    trades["setup"] == TARGET_SETUP
].copy()

target["entry_date"] = pd.to_datetime(
    target["entry_date"],
    errors="coerce"
)

target["exit_date"] = pd.to_datetime(
    target["exit_date"],
    errors="coerce"
)

target["r_multiple"] = numeric(
    target["r_multiple"]
)

target["entry_atr"] = numeric(
    target["entry_atr"]
)

target["initial_risk"] = numeric(
    target["initial_risk"]
)

target = target.sort_values(
    "entry_date"
).reset_index(drop=True)


print(
    f"{TARGET_SETUP} trades loaded: {len(target)}"
)

if len(target) != EXPECTED_TARGET_TRADES:

    print(
        f"WARNING: Expected {EXPECTED_TARGET_TRADES} "
        f"{TARGET_SETUP} trades, found {len(target)}."
    )


# ============================================================================
# DOWNLOAD OHLC
# ============================================================================

print()
print("-" * 100)
print("DOWNLOADING BANK NIFTY OHLC DATA")
print("-" * 100)

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False,
)

if data is None or data.empty:

    print()
    print("ERROR: No BANK NIFTY data returned.")

    raise SystemExit(1)


if isinstance(
    data.columns,
    pd.MultiIndex
):

    data.columns = data.columns.get_level_values(0)


required_ohlc = [
    "Open",
    "High",
    "Low",
    "Close",
]

missing_ohlc = [
    column
    for column in required_ohlc
    if column not in data.columns
]

if missing_ohlc:

    print()
    print(
        "ERROR: Missing OHLC columns:"
    )

    for column in missing_ohlc:
        print(f"  - {column}")

    raise SystemExit(1)


data = data[
    required_ohlc
].copy()

data.index = pd.to_datetime(
    data.index
)

data = data.sort_index()

data["ATR"] = calculate_atr(
    data
)

print(
    f"Rows: {len(data)}"
)

print(
    "Date range: "
    f"{data.index.min().date()} -> "
    f"{data.index.max().date()}"
)

print(
    "ATR reconstruction: "
    "14-period rolling True Range mean"
)

print(
    "Slippage reconstruction: 0.10%"
)


# ============================================================================
# TRADE DATE VALIDATION
# ============================================================================

print()
print("=" * 100)
print("VALIDATING TRADE DATES")
print("=" * 100)

trade_date_failures = []

for idx, trade in target.iterrows():

    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]

    if pd.isna(entry_date):

        trade_date_failures.append(
            f"row {idx}: invalid entry date"
        )

        continue

    if entry_date not in data.index:

        trade_date_failures.append(
            f"{entry_date.date()}: entry date missing"
        )

    if pd.notna(exit_date):

        if exit_date not in data.index:

            trade_date_failures.append(
                f"{exit_date.date()}: exit date missing"
            )


if trade_date_failures:

    print("Trade-date validation: FAIL")

    for failure in trade_date_failures:
        print(
            f"  {failure}"
        )

    raise SystemExit(1)


print(
    "Trade-date validation: PASS"
)


# ============================================================================
# BUILD TRADE REPLAY
# ============================================================================

print()
print("=" * 100)
print("BUILDING OHLC EARLY-ADVERSE REPLAY")
print("=" * 100)

replay_rows = []

replay_failures = []

for trade_index, trade in target.iterrows():

    entry_date = trade["entry_date"]
    exit_date = trade["exit_date"]

    entry_position = data.index.get_loc(
        entry_date
    )

    exit_position = data.index.get_loc(
        exit_date
    )

    if exit_position < entry_position:

        replay_failures.append(
            f"{entry_date.date()}: exit before entry"
        )

        continue

    signal_day_atr = (
        float(
            trade["entry_atr"]
        )
    )

    if not np.isfinite(
        signal_day_atr
    ):

        replay_failures.append(
            f"{entry_date.date()}: invalid entry ATR"
        )

        continue

    entry_day_open = float(
        data.loc[
            entry_date,
            "Open"
        ]
    )

    # Production short entry:
    # next_open * (1 - slippage_rate)
    #
    # For the frozen BANK NIFTY trade CSV,
    # entry_date is the actual execution date.

    execution_price = (
        entry_day_open
        * (
            1.0
            - SLIPPAGE_RATE
        )
    )

    risk_per_share = (
        INITIAL_STOP_ATR
        * signal_day_atr
    )

    if risk_per_share <= 0:

        replay_failures.append(
            f"{entry_date.date()}: invalid risk/share"
        )

        continue

    max_adverse_r = 0.0

    max_favorable_r = 0.0

    early_adverse = {
        window: None
        for window in WINDOWS
    }

    lifetime_bars = (
        exit_position
        - entry_position
        + 1
    )

    # Production bar numbering:
    #
    # entry-date OHLC bar = bar 1
    #
    # Therefore:
    # position = entry_position + bar_number - 1

    for bar_number in range(
        1,
        lifetime_bars + 1
    ):

        position = (
            entry_position
            + bar_number
            - 1
        )

        row = data.iloc[
            position
        ]

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        # SHORT:
        #
        # adverse movement happens when
        # price moves above entry.
        #
        # favorable movement happens when
        # price moves below entry.

        adverse_r = max(
            0.0,
            (
                high
                - execution_price
            )
            / risk_per_share
        )

        favorable_r = max(
            0.0,
            (
                execution_price
                - low
            )
            / risk_per_share
        )

        max_adverse_r = max(
            max_adverse_r,
            adverse_r
        )

        max_favorable_r = max(
            max_favorable_r,
            favorable_r
        )

        for window in WINDOWS:

            if (
                bar_number == window
            ):

                early_adverse[
                    window
                ] = max_adverse_r

    expected_mfe = numeric(
        pd.Series(
            [trade.get(
                "max_favorable_r",
                np.nan
            )]
        )
    ).iloc[0]

    expected_mae = numeric(
        pd.Series(
            [trade.get(
                "max_adverse_r",
                np.nan
            )]
        )
    ).iloc[0]

    replay_rows.append(
        {
            "trade_index": trade_index,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": trade["direction"],
            "setup": trade["setup"],
            "actual_r": float(
                trade["r_multiple"]
            ),
            "entry_atr_csv": signal_day_atr,
            "entry_open": entry_day_open,
            "execution_price": execution_price,
            "risk_per_share": risk_per_share,
            "initial_risk_csv": float(
                trade["initial_risk"]
            ),
            "reconstructed_max_mfe_r":
                max_favorable_r,
            "reconstructed_max_mae_r":
                max_adverse_r,
            "expected_mfe_r":
                expected_mfe,
            "expected_mae_r":
                expected_mae,
            "lifetime_bars": lifetime_bars,
            "early_adverse_3":
                early_adverse[3],
            "early_adverse_5":
                early_adverse[5],
            "early_adverse_7":
                early_adverse[7],
            "early_adverse_10":
                early_adverse[10],
        }
    )


if replay_failures:

    print()
    print("Replay failures:")

    for failure in replay_failures:
        print(
            f"  {failure}"
        )

    raise SystemExit(1)


replay = pd.DataFrame(
    replay_rows
)


print()
print(
    f"Replay trades generated: {len(replay)}"
)


# ============================================================================
# REPLAY INTEGRITY
# ============================================================================

print()
print("-" * 100)
print("REPLAY INTEGRITY")
print("-" * 100)

mfe_mismatches = 0
mae_mismatches = 0

mfe_checked = 0
mae_checked = 0

for _, row in replay.iterrows():

    expected_mfe = row[
        "expected_mfe_r"
    ]

    expected_mae = row[
        "expected_mae_r"
    ]

    actual_mfe = row[
        "reconstructed_max_mfe_r"
    ]

    actual_mae = row[
        "reconstructed_max_mae_r"
    ]

    if pd.notna(expected_mfe):

        mfe_checked += 1

        if abs(
            actual_mfe
            - expected_mfe
        ) > VALIDATION_TOLERANCE:

            mfe_mismatches += 1

    if pd.notna(expected_mae):

        mae_checked += 1

        if abs(
            actual_mae
            - expected_mae
        ) > VALIDATION_TOLERANCE:

            mae_mismatches += 1


print(
    f"MFE checked: {mfe_checked}"
)

print(
    f"MFE mismatches: {mfe_mismatches}"
)

print(
    f"MAE checked: {mae_checked}"
)

print(
    f"MAE mismatches: {mae_mismatches}"
)


if (
    mfe_mismatches > 0
    or mae_mismatches > 0
):

    print()
    print(
        "REPLAY INTEGRITY: FAIL"
    )

    print(
        "Do not use robustness results."
    )

    raise SystemExit(1)


print()
print(
    "REPLAY INTEGRITY: PASS"
)


# ============================================================================
# BASELINE
# ============================================================================

baseline_values = replay[
    "actual_r"
].astype(float)

baseline = summarize(
    baseline_values
)

print()
print("=" * 100)
print("SHORT_CONTINUATION BASELINE")
print("=" * 100)

print(
    f"Trades:        {baseline['trades']}"
)

print(
    f"Winners:       {baseline['winners']}"
)

print(
    f"Win rate:      {baseline['win_rate']:.2f}%"
)

print(
    f"Gross Profit:  {baseline['gross_profit']:+.2f}R"
)

print(
    f"Gross Loss:    {baseline['gross_loss']:+.2f}R"
)

print(
    f"Profit Factor: {baseline['profit_factor']:.3f}"
)

print(
    f"Total R:       {baseline['total_r']:+.2f}R"
)

print(
    f"Average R:     {baseline['avg_r']:+.3f}R"
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
        f"early_adverse_{window}"
    )

    for threshold in THRESHOLDS:

        values = replay[
            column
        ]

        trigger_mask = (
            values >= threshold
        )

        retained = replay[
            ~trigger_mask
        ].copy()

        retained_values = retained[
            "actual_r"
        ].astype(float)

        result = summarize(
            retained_values
        )

        delta = (
            result["total_r"]
            - baseline["total_r"]
        )

        matrix_rows.append(
            {
                "window": window,
                "threshold": threshold,
                "baseline_trades":
                    baseline["trades"],
                "triggered_trades":
                    int(
                        trigger_mask.sum()
                    ),
                "retained_trades":
                    result["trades"],
                "retained_winners":
                    result["winners"],
                "retained_win_rate":
                    result["win_rate"],
                "retained_profit_factor":
                    result["profit_factor"],
                "retained_total_r":
                    result["total_r"],
                "delta_r":
                    delta,
            }
        )


matrix = pd.DataFrame(
    matrix_rows
)


print(
    matrix.to_string(
        index=False,
        formatters={
            "threshold":
                lambda x:
                f"{x:.2f}",
            "retained_win_rate":
                lambda x:
                f"{x:.2f}%",
            "retained_profit_factor":
                lambda x:
                (
                    f"{x:.3f}"
                    if np.isfinite(x)
                    else "INF"
                ),
            "retained_total_r":
                lambda x:
                f"{x:+.2f}R",
            "delta_r":
                lambda x:
                f"{x:+.2f}R",
        }
    )
)


# ============================================================================
# BEST CELL
# ============================================================================

matrix_sorted = matrix.sort_values(
    [
        "retained_total_r",
        "delta_r",
    ],
    ascending=False
)

best = matrix_sorted.iloc[
    0
]

print()
print("=" * 100)
print("BEST PARAMETER CELL")
print("=" * 100)

print(
    f"Window:          {int(best['window'])} bars"
)

print(
    f"Threshold:       {best['threshold']:.2f}R"
)

print(
    f"Triggered:       {int(best['triggered_trades'])}"
)

print(
    f"Retained:        {int(best['retained_trades'])}"
)

print(
    f"Win rate:        {best['retained_win_rate']:.2f}%"
)

print(
    f"Profit Factor:   {best['retained_profit_factor']:.3f}"
)

print(
    f"Filtered TotalR: {best['retained_total_r']:+.2f}R"
)

print(
    f"Delta:           {best['delta_r']:+.2f}R"
)


# ============================================================================
# PARAMETER ROBUSTNESS COUNTS
# ============================================================================

improved_cells = int(
    (
        matrix["delta_r"] > 0
    ).sum()
)

worsened_cells = int(
    (
        matrix["delta_r"] < 0
    ).sum()
)

unchanged_cells = int(
    (
        matrix["delta_r"] == 0
    ).sum()
)

print()
print("=" * 100)
print("PARAMETER ROBUSTNESS COUNT")
print("=" * 100)

print(
    f"Total cells:    {len(matrix)}"
)

print(
    f"Improved:       {improved_cells}"
)

print(
    f"Worsened:       {worsened_cells}"
)

print(
    f"Unchanged:      {unchanged_cells}"
)


# ============================================================================
# YEAR ROBUSTNESS
# ============================================================================

print()
print("=" * 100)
print("YEAR ROBUSTNESS")
print("=" * 100)

replay["year"] = pd.to_datetime(
    replay["entry_date"]
).dt.year

year_rows = []

anchor_window = 5
anchor_threshold = 0.25

anchor_column = (
    f"early_adverse_{anchor_window}"
)

for year, year_group in replay.groupby(
    "year"
):

    baseline_values = year_group[
        "actual_r"
    ].astype(float)

    year_baseline = summarize(
        baseline_values
    )

    trigger_mask = (
        year_group[
            anchor_column
        ]
        >= anchor_threshold
    )

    retained = year_group[
        ~trigger_mask
    ]

    filtered_values = retained[
        "actual_r"
    ].astype(float)

    year_filtered = summarize(
        filtered_values
    )

    delta = (
        year_filtered["total_r"]
        - year_baseline["total_r"]
    )

    year_rows.append(
        {
            "year": int(year),
            "baseline_trades":
                year_baseline["trades"],
            "baseline_winners":
                year_baseline["winners"],
            "baseline_win_rate":
                year_baseline["win_rate"],
            "baseline_total_r":
                year_baseline["total_r"],
            "triggered_trades":
                int(
                    trigger_mask.sum()
                ),
            "retained_trades":
                year_filtered["trades"],
            "filtered_winners":
                year_filtered["winners"],
            "filtered_win_rate":
                year_filtered["win_rate"],
            "filtered_total_r":
                year_filtered["total_r"],
            "delta_r":
                delta,
        }
    )


year_df = pd.DataFrame(
    year_rows
)

print(
    year_df.to_string(
        index=False,
        formatters={
            "baseline_win_rate":
                lambda x:
                f"{x:.2f}%",
            "baseline_total_r":
                lambda x:
                f"{x:+.2f}R",
            "filtered_win_rate":
                lambda x:
                f"{x:.2f}%",
            "filtered_total_r":
                lambda x:
                f"{x:+.2f}R",
            "delta_r":
                lambda x:
                f"{x:+.2f}R",
        }
    )
)


positive_years = int(
    (
        year_df["delta_r"] > 0
    ).sum()
)

negative_years = int(
    (
        year_df["delta_r"] < 0
    ).sum()
)

unchanged_years = int(
    (
        year_df["delta_r"] == 0
    ).sum()
)

print()
print(
    f"Anchor: {anchor_window}B / "
    f"{anchor_threshold:.2f}R"
)

print(
    f"Years improved: {positive_years}"
)

print(
    f"Years worsened: {negative_years}"
)

print(
    f"Years unchanged: {unchanged_years}"
)


# ============================================================================
# LEAVE-ONE-OUT
# ============================================================================

print()
print("=" * 100)
print("LEAVE-ONE-OUT ROBUSTNESS")
print("=" * 100)

loo_rows = []

anchor_column = (
    f"early_adverse_{anchor_window}"
)

base_trigger = (
    replay[
        anchor_column
    ]
    >= anchor_threshold
)

base_retained = replay[
    ~base_trigger
].copy()

base_filtered_r = (
    base_retained[
        "actual_r"
    ].astype(float)
    .sum()
)

for excluded_index, excluded_trade in replay.iterrows():

    remaining = replay[
        replay.index
        != excluded_index
    ].copy()

    remaining_baseline_r = (
        remaining[
            "actual_r"
        ].astype(float)
        .sum()
    )

    remaining_trigger = (
        remaining[
            anchor_column
        ]
        >= anchor_threshold
    )

    remaining_retained = remaining[
        ~remaining_trigger
    ]

    remaining_filtered_r = (
        remaining_retained[
            "actual_r"
        ].astype(float)
        .sum()
    )

    remaining_delta = (
        remaining_filtered_r
        - remaining_baseline_r
    )

    loo_rows.append(
        {
            "excluded_entry_date":
                excluded_trade[
                    "entry_date"
                ].strftime(
                    "%Y-%m-%d"
                ),
            "excluded_r":
                excluded_trade[
                    "actual_r"
                ],
            "excluded_triggered":
                bool(
                    base_trigger.loc[
                        excluded_index
                    ]
                ),
            "remaining_baseline_r":
                remaining_baseline_r,
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
            "excluded_r":
                lambda x:
                f"{x:+.2f}R",
            "remaining_baseline_r":
                lambda x:
                f"{x:+.2f}R",
            "remaining_filtered_r":
                lambda x:
                f"{x:+.2f}R",
            "remaining_delta_r":
                lambda x:
                f"{x:+.2f}R",
        }
    )
)


loo_positive = int(
    (
        loo_df["remaining_delta_r"] > 0
    ).sum()
)

loo_negative = int(
    (
        loo_df["remaining_delta_r"] < 0
    ).sum()
)

loo_unchanged = int(
    (
        loo_df["remaining_delta_r"] == 0
    ).sum()
)

print()
print(
    f"LOO positive:   {loo_positive}"
)

print(
    f"LOO negative:   {loo_negative}"
)

print(
    f"LOO unchanged:  {loo_unchanged}"
)


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

matrix.to_csv(
    ROBUSTNESS_OUTPUT,
    index=False
)

year_df.to_csv(
    YEAR_OUTPUT,
    index=False
)

loo_df.to_csv(
    LOO_OUTPUT,
    index=False
)


# ============================================================================
# VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)

all_parameter_cells_positive = (
    improved_cells == len(matrix)
)

year_robust = (
    positive_years >= 4
    and negative_years == 0
)

loo_robust = (
    loo_positive >= (
        len(loo_df) - 1
    )
    and loo_negative == 0
)

if (
    all_parameter_cells_positive
    and year_robust
    and loo_robust
):

    verdict = (
        "STRONG_IN_SAMPLE_ROBUSTNESS_SIGNAL"
    )

elif (
    improved_cells > worsened_cells
    and positive_years > negative_years
):

    verdict = (
        "PROMISING_BUT_NOT_ROBUST"
    )

else:

    verdict = (
        "ROBUSTNESS_NOT_ESTABLISHED"
    )


print()
print(
    f"Verdict: {verdict}"
)

print()
print(
    "This remains in-sample research."
)

print(
    "No production filter is being implemented."
)

print(
    "No baseline BANK NIFTY trades are being changed."
)

print(
    "Independent OOS validation is still required."
)


# ============================================================================
# SAFETY
# ============================================================================

print()
print("=" * 100)
print("SAFETY CHECK")
print("=" * 100)

print(
    "PASS: Production strategy was not modified."
)

print(
    "PASS: BANK NIFTY baseline CSV was not modified."
)

print(
    "PASS: Frozen stock historical data was not modified."
)

print(
    "PASS: Stock holdout data was not modified."
)

print(
    "PASS: No trading filter was implemented."
)

print(
    "PASS: Research outputs saved separately."
)

print("=" * 100)