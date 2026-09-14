"""
TradeSense-AI
NIFTY MIDCAP SELECT — Trigger Threshold Trade-Path Forensic

Research-only.

Purpose:
    Compare trade behaviour at favorable-development thresholds:

        +1.25R
        +1.50R
        +1.75R

For every frozen MIDSELECT trade, determine:

    1. Whether each threshold was reached.
    2. First bar/date reaching each threshold.
    3. Whether 1.25R was reached but 1.50R was not.
    4. Whether 1.50R was reached but 1.75R was not.
    5. Maximum favorable excursion.
    6. Maximum adverse excursion.
    7. Early adverse behaviour.
    8. Setup.
    9. Direction.
    10. Year.
    11. Actual production outcome.

Rules:
    - Same validated MIDSELECT OHLC source as baseline.
    - Same frozen 13 trades.
    - Same actual trade lifetime.
    - No trade lifetime extension.
    - No production changes.
    - No historical dataset changes.
    - No holdout changes.
    - No parameter selection.
"""

import os

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

THRESHOLDS = [
    1.25,
    1.50,
    1.75,
]

EXPECTED_TRADE_COUNT = 13

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

TRADE_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_trigger_threshold_trade_paths.csv"
)

SUMMARY_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_trigger_threshold_summary.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):

    return pd.Timestamp(
        value
    ).normalize()


def threshold_price(
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


def favorable_r(
    direction,
    high,
    low,
    entry_price,
    risk_per_share,
):

    if direction == "LONG":

        return (
            high
            - entry_price
        ) / risk_per_share

    return (
        entry_price
        - low
    ) / risk_per_share


def adverse_r(
    direction,
    high,
    low,
    entry_price,
    risk_per_share,
):

    if direction == "LONG":

        return (
            entry_price
            - low
        ) / risk_per_share

    return (
        high
        - entry_price
    ) / risk_per_share


# ============================================================================
# HEADER
# ============================================================================

print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT TRIGGER THRESHOLD FORENSIC"
)
print("=" * 100)

print(
    f"Trade file : {TRADE_FILE}"
)

print(
    f"Price file : {PRICE_FILE}"
)

print(
    f"Thresholds : {THRESHOLDS}"
)


# ============================================================================
# LOAD TRADES
# ============================================================================

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


required_trade_columns = [
    "direction",
    "setup",
    "entry_date",
    "exit_date",
    "entry_price",
    "entry_atr",
    "initial_risk",
    "r_multiple",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
    "bars_in_trade",
]


missing_trade_columns = [
    column
    for column in required_trade_columns
    if column not in trades.columns
]


if missing_trade_columns:

    print(
        "ERROR: Missing trade columns:"
    )

    for column in missing_trade_columns:

        print(
            f"  - {column}"
        )

    raise SystemExit(1)


if len(trades) != EXPECTED_TRADE_COUNT:

    print(
        "WARNING: Expected "
        f"{EXPECTED_TRADE_COUNT} trades but found "
        f"{len(trades)}."
    )


trades["entry_date"] = pd.to_datetime(
    trades["entry_date"],
    errors="coerce",
)

trades["exit_date"] = pd.to_datetime(
    trades["exit_date"],
    errors="coerce",
)


numeric_columns = [
    "entry_price",
    "entry_atr",
    "initial_risk",
    "r_multiple",
    "max_favorable_r",
    "max_adverse_r",
    "early_adverse_r_bar_1",
    "early_adverse_r_bar_2",
    "early_adverse_r_bar_3",
    "early_adverse_r_bar_4",
    "early_adverse_r_bar_5",
    "bars_in_trade",
]


for column in numeric_columns:

    trades[column] = pd.to_numeric(
        trades[column],
        errors="coerce",
    )


print()
print(
    f"Historical trades loaded: "
    f"{len(trades)}"
)


# ============================================================================
# LOAD VALIDATED PRICE DATA
# ============================================================================

print()
print("=" * 100)
print("VALIDATED PRICE DATA")
print("=" * 100)


if not os.path.exists(
    PRICE_FILE
):

    print(
        "ERROR: Price file not found."
    )

    raise SystemExit(1)


data = pd.read_csv(
    PRICE_FILE
)


if data.empty:

    print(
        "ERROR: Price file is empty."
    )

    raise SystemExit(1)


if "Date" in data.columns:

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    data = data.set_index(
        "Date"
    )

elif "date" in data.columns:

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data.set_index(
        "date"
    )

else:

    print(
        "ERROR: Date column not found."
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
        "ERROR: Missing price columns:"
    )

    for column in missing_price_columns:

        print(
            f"  - {column}"
        )

    raise SystemExit(1)


for column in required_price_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )


invalid_rows = int(
    data[
        required_price_columns
    ]
    .isna()
    .any(axis=1)
    .sum()
)


if invalid_rows:

    print(
        f"ERROR: Invalid OHLC rows: "
        f"{invalid_rows}"
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
        "ERROR: Non-positive OHLC found."
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
    "PRICE DATA VALIDATION: PASS"
)


# ============================================================================
# TRADE-DATE VALIDATION
# ============================================================================

print()
print("=" * 100)
print("TRADE-DATE VALIDATION")
print("=" * 100)


missing_dates = []


for _, trade in trades.iterrows():

    entry_date = normalize_date(
        trade["entry_date"]
    )

    exit_date = normalize_date(
        trade["exit_date"]
    )

    if entry_date not in data.index:

        missing_dates.append(
            entry_date
        )

    if exit_date not in data.index:

        missing_dates.append(
            exit_date
        )


if missing_dates:

    print(
        "ERROR: Missing trade dates:"
    )

    for date in sorted(
        set(missing_dates)
    ):

        print(
            f"  - {date.date()}"
        )

    raise SystemExit(1)


print(
    "TRADE-DATE VALIDATION: PASS"
)


# ============================================================================
# TRADE REPLAY
# ============================================================================

print()
print("=" * 100)
print("TRIGGER THRESHOLD TRADE-PATH REPLAY")
print("=" * 100)


trade_rows = []


for trade_index, trade in trades.iterrows():

    entry_date = normalize_date(
        trade["entry_date"]
    )

    exit_date = normalize_date(
        trade["exit_date"]
    )

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


    entry_price = float(
        trade["entry_price"]
    )

    entry_atr = float(
        trade["entry_atr"]
    )

    risk_per_share = (
        3.5
        * entry_atr
    )


    lifetime = data.loc[
        (
            data.index
            >= entry_date
        )
        & (
            data.index
            <= exit_date
        )
    ].copy()


    if lifetime.empty:

        print(
            "ERROR: Empty lifetime for "
            f"trade {trade_index}."
        )

        raise SystemExit(1)


    threshold_results = {}


    # ------------------------------------------------------------------------
    # Prepare threshold results.
    # ------------------------------------------------------------------------

    for threshold in THRESHOLDS:

        threshold_results[
            threshold
        ] = {
            "reached": False,
            "bar": None,
            "date": None,
            "favorable_r_on_bar": None,
        }


    max_favorable_r_replay = -np.inf

    max_adverse_r_replay = -np.inf

    max_favorable_bar = None

    max_favorable_date = None

    max_adverse_bar = None

    max_adverse_date = None


    # ------------------------------------------------------------------------
    # Replay every bar in the actual lifetime.
    # ------------------------------------------------------------------------

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


        fav_r = favorable_r(
            direction=direction,
            high=high,
            low=low,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
        )


        adv_r = adverse_r(
            direction=direction,
            high=high,
            low=low,
            entry_price=entry_price,
            risk_per_share=risk_per_share,
        )


        if fav_r > max_favorable_r_replay:

            max_favorable_r_replay = fav_r

            max_favorable_bar = (
                bar_number
            )

            max_favorable_date = (
                bar_date
            )


        if adv_r > max_adverse_r_replay:

            max_adverse_r_replay = adv_r

            max_adverse_bar = (
                bar_number
            )

            max_adverse_date = (
                bar_date
            )


        # --------------------------------------------------------------------
        # Determine which thresholds were reached.
        # --------------------------------------------------------------------

        for threshold in THRESHOLDS:

            result = threshold_results[
                threshold
            ]


            if result["reached"]:

                continue


            threshold_price_value = (
                threshold_price(
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
                    >= threshold_price_value
                )

            else:

                reached = (
                    low
                    <= threshold_price_value
                )


            if reached:

                result["reached"] = True

                result["bar"] = (
                    bar_number
                )

                result["date"] = (
                    bar_date
                )

                result[
                    "favorable_r_on_bar"
                ] = fav_r


    # ------------------------------------------------------------------------
    # Extract threshold information.
    # ------------------------------------------------------------------------

    reached_125 = threshold_results[
        1.25
    ]["reached"]

    reached_150 = threshold_results[
        1.50
    ]["reached"]

    reached_175 = threshold_results[
        1.75
    ]["reached"]


    # ------------------------------------------------------------------------
    # Cohort classification.
    # ------------------------------------------------------------------------

    if reached_125 and not reached_150:

        threshold_cohort = (
            "1.25_ONLY"
        )

    elif reached_150 and not reached_175:

        threshold_cohort = (
            "1.50_ONLY"
        )

    elif reached_175:

        threshold_cohort = (
            "1.75_REACHED"
        )

    elif not reached_125:

        threshold_cohort = (
            "NONE_REACHED"
        )

    else:

        threshold_cohort = (
            "OTHER"
        )


    # ------------------------------------------------------------------------
    # Early-adverse values.
    # ------------------------------------------------------------------------

    early_values = [
        trade[
            f"early_adverse_r_bar_{bar}"
        ]
        for bar in range(
            1,
            6,
        )
        if pd.notna(
            trade[
                f"early_adverse_r_bar_{bar}"
            ]
        )
    ]


    if early_values:

        early5_replay = max(
            early_values
        )

    else:

        early5_replay = np.nan


    # ------------------------------------------------------------------------
    # Production values.
    # ------------------------------------------------------------------------

    actual_r = float(
        trade["r_multiple"]
    )

    actual_mfe = float(
        trade["max_favorable_r"]
    )

    actual_mae = float(
        trade["max_adverse_r"]
    )

    bars_in_trade = int(
        trade["bars_in_trade"]
    )


    row = {
        "trade_index": trade_index,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": direction,
        "setup": setup,
        "year": entry_date.year,
        "entry_price": entry_price,
        "entry_atr": entry_atr,
        "risk_per_share": risk_per_share,
        "actual_r": actual_r,
        "actual_mfe_r": actual_mfe,
        "actual_mae_r": actual_mae,
        "replay_mfe_r": max_favorable_r_replay,
        "replay_mae_r": max_adverse_r_replay,
        "mfe_bar": max_favorable_bar,
        "mfe_date": max_favorable_date,
        "mae_bar": max_adverse_bar,
        "mae_date": max_adverse_date,
        "bars_in_trade": bars_in_trade,
        "early5_r": early5_replay,
        "threshold_cohort": threshold_cohort,
    }


    for threshold in THRESHOLDS:

        result = threshold_results[
            threshold
        ]

        suffix = (
            str(
                threshold
            )
            .replace(
                ".",
                "_",
            )
        )


        row[
            f"reached_{suffix}r"
        ] = result[
            "reached"
        ]

        row[
            f"bar_{suffix}r"
        ] = result[
            "bar"
        ]

        row[
            f"date_{suffix}r"
        ] = result[
            "date"
        ]

        row[
            f"favorable_on_{suffix}r_bar"
        ] = result[
            "favorable_r_on_bar"
        ]


    # ------------------------------------------------------------------------
    # Explicit transition flags.
    # ------------------------------------------------------------------------

    row[
        "reached_1_25_not_1_50"
    ] = (
        reached_125
        and not reached_150
    )

    row[
        "reached_1_50_not_1_75"
    ] = (
        reached_150
        and not reached_175
    )

    row[
        "reached_1_75"
    ] = reached_175


    trade_rows.append(
        row
    )


# ============================================================================
# DATAFRAME
# ============================================================================

result_df = pd.DataFrame(
    trade_rows
)


# ============================================================================
# REPLAY INTEGRITY
# ============================================================================

print()
print("=" * 100)
print("REPLAY INTEGRITY")
print("=" * 100)


mfe_difference = (
    result_df["replay_mfe_r"]
    - result_df["actual_mfe_r"]
)

mae_difference = (
    result_df["replay_mae_r"]
    - result_df["actual_mae_r"]
)


mfe_mismatches = int(
    (
        mfe_difference.abs()
        > 0.005001
    ).sum()
)


mae_mismatches = int(
    (
        mae_difference.abs()
        > 0.005001
    ).sum()
)


print(
    f"Trades replayed : "
    f"{len(result_df)}"
)

print(
    f"MFE mismatches  : "
    f"{mfe_mismatches}"
)

print(
    f"MAE mismatches  : "
    f"{mae_mismatches}"
)


if (
    mfe_mismatches == 0
    and mae_mismatches == 0
):

    print(
        "REPLAY INTEGRITY: PASS"
    )

else:

    print(
        "REPLAY INTEGRITY: FAIL"
    )

    raise SystemExit(1)


# ============================================================================
# THRESHOLD SUMMARY
# ============================================================================

print()
print("=" * 100)
print("THRESHOLD SUMMARY")
print("=" * 100)


summary_rows = []


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


    mask = result_df[
        f"reached_{suffix}r"
    ].astype(bool)


    reached = result_df[
        mask
    ]


    not_reached = result_df[
        ~mask
    ]


    reached_count = len(
        reached
    )

    not_reached_count = len(
        not_reached
    )


    reached_r = float(
        reached["actual_r"].sum()
    ) if reached_count else 0.0


    not_reached_r = float(
        not_reached["actual_r"].sum()
    ) if not_reached_count else 0.0


    reached_win_rate = (
        float(
            (
                reached["actual_r"]
                > 0
            ).mean()
            * 100
        )
        if reached_count
        else 0.0
    )


    not_reached_win_rate = (
        float(
            (
                not_reached["actual_r"]
                > 0
            ).mean()
            * 100
        )
        if not_reached_count
        else 0.0
    )


    median_bar = (
        float(
            reached[
                f"bar_{suffix}r"
            ].median()
        )
        if reached_count
        else np.nan
    )


    summary_rows.append(
        {
            "threshold_r": threshold,
            "reached_trades": reached_count,
            "not_reached_trades": (
                not_reached_count
            ),
            "reached_actual_r": (
                reached_r
            ),
            "not_reached_actual_r": (
                not_reached_r
            ),
            "reached_win_rate": (
                reached_win_rate
            ),
            "not_reached_win_rate": (
                not_reached_win_rate
            ),
            "median_bars_to_threshold": (
                median_bar
            ),
        }
    )


    print()
    print(
        f"{threshold:.2f}R"
    )

    print(
        f"  Reached     : "
        f"{reached_count} trades | "
        f"{reached_win_rate:.2f}% win | "
        f"{reached_r:+.2f}R"
    )

    print(
        f"  Not reached : "
        f"{not_reached_count} trades | "
        f"{not_reached_win_rate:.2f}% win | "
        f"{not_reached_r:+.2f}R"
    )

    print(
        f"  Median bars : "
        f"{median_bar:.1f}"
        if not np.isnan(
            median_bar
        )
        else
        "  Median bars : N/A"
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================================
# TRANSITION COHORTS
# ============================================================================

print()
print("=" * 100)
print("THRESHOLD TRANSITION COHORTS")
print("=" * 100)


cohort_order = [
    "NONE_REACHED",
    "1.25_ONLY",
    "1.50_ONLY",
    "1.75_REACHED",
]


for cohort in cohort_order:

    cohort_df = result_df[
        result_df[
            "threshold_cohort"
        ]
        == cohort
    ]


    count = len(
        cohort_df
    )


    total_r = float(
        cohort_df[
            "actual_r"
        ].sum()
    ) if count else 0.0


    wins = int(
        (
            cohort_df[
                "actual_r"
            ]
            > 0
        ).sum()
    ) if count else 0


    win_rate = (
        wins
        / count
        * 100
        if count
        else 0.0
    )


    print()
    print(
        f"{cohort}"
    )

    print(
        f"  Trades   : {count}"
    )

    print(
        f"  Winners  : {wins}"
    )

    print(
        f"  Win rate : {win_rate:.2f}%"
    )

    print(
        f"  Total R  : {total_r:+.2f}R"
    )


# ============================================================================
# INDIVIDUAL TRADE PATHS
# ============================================================================

print()
print("=" * 100)
print("INDIVIDUAL TRADE PATHS")
print("=" * 100)


display_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "actual_r",
    "actual_mfe_r",
    "actual_mae_r",
    "early5_r",
    "threshold_cohort",
]


print()

print(
    result_df[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "THRESHOLD ARRIVAL DETAILS"
)
print(
    "-" * 100
)


for _, row in result_df.iterrows():

    print()

    print(
        f"{row['entry_date'].date()} | "
        f"{row['direction']:<5} | "
        f"{row['setup']:<20} | "
        f"Actual={row['actual_r']:+.2f}R | "
        f"MFE={row['actual_mfe_r']:.2f}R | "
        f"Cohort={row['threshold_cohort']}"
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


        reached = row[
            f"reached_{suffix}r"
        ]


        if reached:

            bar = row[
                f"bar_{suffix}r"
            ]

            date = row[
                f"date_{suffix}r"
            ]

            print(
                f"    {threshold:.2f}R "
                f"-> bar {int(bar)} "
                f"-> {pd.Timestamp(date).date()}"
            )

        else:

            print(
                f"    {threshold:.2f}R "
                f"-> NOT REACHED"
            )


# ============================================================================
# YEAR BREAKDOWN
# ============================================================================

print()
print("=" * 100)
print("YEAR × THRESHOLD PROFILE")
print("=" * 100)


for year in sorted(
    result_df["year"].unique()
):

    year_df = result_df[
        result_df["year"]
        == year
    ]


    print()
    print(
        f"{year}"
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


        reached = year_df[
            year_df[
                f"reached_{suffix}r"
            ]
            .astype(bool)
        ]


        not_reached = year_df[
            ~year_df[
                f"reached_{suffix}r"
            ].astype(bool)
        ]


        print(
            f"  {threshold:.2f}R | "
            f"reached={len(reached)} "
            f"({reached['actual_r'].sum():+.2f}R) | "
            f"not={len(not_reached)} "
            f"({not_reached['actual_r'].sum():+.2f}R)"
        )


# ============================================================================
# SETUP BREAKDOWN
# ============================================================================

print()
print("=" * 100)
print("SETUP × THRESHOLD PROFILE")
print("=" * 100)


for setup in sorted(
    result_df["setup"].unique()
):

    setup_df = result_df[
        result_df["setup"]
        == setup
    ]


    print()
    print(
        setup
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


        reached = setup_df[
            setup_df[
                f"reached_{suffix}r"
            ]
            .astype(bool)
        ]


        print(
            f"  {threshold:.2f}R "
            f"reached: {len(reached)} trades | "
            f"{reached['actual_r'].sum():+.2f}R"
        )


# ============================================================================
# DIRECTION BREAKDOWN
# ============================================================================

print()
print("=" * 100)
print("DIRECTION × THRESHOLD PROFILE")
print("=" * 100)


for direction in sorted(
    result_df["direction"].unique()
):

    direction_df = result_df[
        result_df["direction"]
        == direction
    ]


    print()
    print(
        direction
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


        reached = direction_df[
            direction_df[
                f"reached_{suffix}r"
            ]
            .astype(bool)
        ]


        print(
            f"  {threshold:.2f}R "
            f"reached: {len(reached)} trades | "
            f"{reached['actual_r'].sum():+.2f}R"
        )


# ============================================================================
# SAVE OUTPUT
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


result_df.to_csv(
    TRADE_OUTPUT_FILE,
    index=False,
)


summary_df.to_csv(
    SUMMARY_OUTPUT_FILE,
    index=False,
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Trade paths : "
    f"{TRADE_OUTPUT_FILE}"
)

print(
    f"Summary     : "
    f"{SUMMARY_OUTPUT_FILE}"
)


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)


print(
    "This test describes the favorable-development "
    "structure of the 13 MIDSELECT trades."
)

print(
    "It does NOT select or optimize an exit parameter."
)

print(
    "It does NOT modify production."
)

print(
    "It does NOT modify the frozen historical dataset."
)

print(
    "It does NOT modify the holdout dataset."
)

print()

print(
    "VERDICT: RESEARCH_FORENSIC_COMPLETE"
)

print(
    "Parameter selection remains deferred "
    "until sufficient independent/OOS evidence exists."
)

print()
print(
    "RESEARCH ONLY."
)