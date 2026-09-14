"""
TradeSense-AI
NIFTY MIDCAP SELECT Exit-Parameter Sensitivity

Research-only.

Tests a small pre-declared neighbourhood around the anchor:

    Trigger +1.50R
    Protection +1.00R
    Target +2.00R

Sensitivity matrix:

Trigger:
    1.25R
    1.50R
    1.75R

Protection:
    0.75R
    1.00R
    1.25R

Total: 9 combinations.

Rules:
- Same validated MIDSELECT OHLC source as baseline.
- Same frozen 13 historical trades.
- Same actual production trade lifetime.
- No trade lifetime extension.
- No production changes.
- No frozen stock dataset changes.
- No holdout changes.
- No parameter optimization after seeing results.

Important:
If trigger and protection are touched on the same bar,
OHLC cannot determine intrabar ordering. Such cases are
classified as ambiguous and retain the actual outcome.

If target and protection are both touched on the same
post-trigger bar, that is also ambiguous and retains the
actual outcome.
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

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0

TRIGGER_LEVELS = [
    1.25,
    1.50,
    1.75,
]

PROTECTION_LEVELS = [
    0.75,
    1.00,
    1.25,
]

VALIDATION_TOLERANCE = 0.005001

EXPECTED_TRADE_COUNT = 13

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_exit_sensitivity.csv"
)

TRADE_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "midselect_exit_sensitivity_trades.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    return pd.Timestamp(
        value
    ).normalize()


def calculate_metrics(values):

    series = pd.Series(
        values,
        dtype=float,
    ).dropna()

    if series.empty:

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
        (series > 0).sum()
    )

    losses = int(
        (series <= 0).sum()
    )

    gross_profit = float(
        series[
            series > 0
        ].sum()
    )

    gross_loss = float(
        series[
            series < 0
        ].sum()
    )

    absolute_loss = abs(
        gross_loss
    )

    if absolute_loss > 0:

        profit_factor = (
            gross_profit
            / absolute_loss
        )

    else:

        profit_factor = np.inf

    return {
        "trades": len(series),
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins
            / len(series)
            * 100
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_r": float(
            series.sum()
        ),
        "average_r": float(
            series.mean()
        ),
    }


def calculate_levels(
    direction,
    entry_price,
    risk_per_share,
    trigger_r,
    protection_r,
):

    if direction == "LONG":

        return {
            "trigger": (
                entry_price
                + trigger_r
                * risk_per_share
            ),
            "protection": (
                entry_price
                + protection_r
                * risk_per_share
            ),
            "target": (
                entry_price
                + TARGET_R
                * risk_per_share
            ),
        }

    return {
        "trigger": (
            entry_price
            - trigger_r
            * risk_per_share
        ),
        "protection": (
            entry_price
            - protection_r
            * risk_per_share
        ),
        "target": (
            entry_price
            - TARGET_R
            * risk_per_share
        ),
    }


def print_metrics(
    label,
    values,
):

    metrics = calculate_metrics(
        values
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
# HEADER
# ============================================================================

print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT EXIT PARAMETER SENSITIVITY"
)
print("=" * 100)

print(
    f"Trade file : {TRADE_FILE}"
)

print(
    f"Price file : {PRICE_FILE}"
)

print(
    f"Trigger levels    : {TRIGGER_LEVELS}"
)

print(
    f"Protection levels : {PROTECTION_LEVELS}"
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
        "ERROR: Validated price file not found."
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
        "ERROR: No Date column found."
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
    f"Rows loaded: {len(data)}"
)

print(
    f"Date range: "
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
# BASELINE RECONSTRUCTION
# ============================================================================

print()
print("=" * 100)
print("BASELINE RECONCILIATION")
print("=" * 100)


frozen_total_r = float(
    trades["r_multiple"].sum()
)


print(
    f"Frozen baseline R: "
    f"{frozen_total_r:+.2f}R"
)


# ============================================================================
# PREPARE TRADE LIFETIMES
# ============================================================================

trade_lifetimes = []


for index, trade in trades.iterrows():

    entry_date = normalize_date(
        trade["entry_date"]
    )

    exit_date = normalize_date(
        trade["exit_date"]
    )

    lifetime = data.loc[
        (data.index >= entry_date)
        & (data.index <= exit_date)
    ].copy()


    if lifetime.empty:

        print(
            "ERROR: Empty lifetime for "
            f"trade {index}."
        )

        raise SystemExit(1)


    direction = (
        str(
            trade["direction"]
        )
        .strip()
        .upper()
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


    if entry_atr <= 0:

        print(
            "ERROR: Invalid entry ATR "
            f"for trade {index}."
        )

        raise SystemExit(1)


    risk_per_share = (
        INITIAL_STOP_ATR
        * entry_atr
    )


    trade_lifetimes.append(
        {
            "index": index,
            "direction": direction,
            "setup": str(
                trade["setup"]
            ),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "entry_atr": entry_atr,
            "initial_risk_total": (
                initial_risk_total
            ),
            "risk_per_share": (
                risk_per_share
            ),
            "actual_r": float(
                trade["r_multiple"]
            ),
            "max_favorable_r": float(
                trade["max_favorable_r"]
            ),
            "lifetime": lifetime,
        }
    )


# ============================================================================
# RUN SENSITIVITY MATRIX
# ============================================================================

print()
print("=" * 100)
print("9-CELL EXIT SENSITIVITY MATRIX")
print("=" * 100)


matrix_rows = []

trade_rows = []


for trigger_r in TRIGGER_LEVELS:

    for protection_r in PROTECTION_LEVELS:

        print()
        print(
            "-" * 100
        )

        print(
            f"TRIGGER {trigger_r:.2f}R "
            f"-> PROTECTION {protection_r:.2f}R "
            f"-> TARGET {TARGET_R:.2f}R"
        )

        counterfactual_values = []

        triggered_count = 0

        protected_count = 0

        target_count = 0

        ambiguous_count = 0

        retained_actual_count = 0


        for trade in trade_lifetimes:

            direction = trade[
                "direction"
            ]

            entry_price = trade[
                "entry_price"
            ]

            risk_per_share = trade[
                "risk_per_share"
            ]

            actual_r = trade[
                "actual_r"
            ]

            lifetime = trade[
                "lifetime"
            ]


            levels = calculate_levels(
                direction=direction,
                entry_price=entry_price,
                risk_per_share=(
                    risk_per_share
                ),
                trigger_r=trigger_r,
                protection_r=protection_r,
            )


            trigger_price = levels[
                "trigger"
            ]

            protection_price = levels[
                "protection"
            ]

            target_price = levels[
                "target"
            ]


            trigger_found = False

            trigger_bar = None

            trigger_date = None

            cf_exit_bar = None

            cf_exit_date = None

            cf_exit_reason = (
                "NO_TRIGGER_RETAIN_ACTUAL"
            )

            cf_r = actual_r


            # --------------------------------------------------------------
            # Scan trade lifetime.
            # --------------------------------------------------------------

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


                # ----------------------------------------------------------
                # Before trigger.
                # ----------------------------------------------------------

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

                        trigger_bar = (
                            bar_number
                        )

                        trigger_date = (
                            bar_date
                        )

                        triggered_count += 1


                        # --------------------------------------------------
                        # Same-bar target after trigger.
                        #
                        # If trigger and target are both touched on
                        # this first trigger bar, we cannot know whether
                        # target was touched before or after trigger.
                        #
                        # Treat as target only when trigger < target
                        # structurally, but retain actual if ambiguity
                        # cannot be resolved.
                        # --------------------------------------------------

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

                            # The trigger is below +2R for long
                            # and above -2R for short. A same-bar
                            # target touch after a trigger touch is
                            # intrabar ambiguous.

                            ambiguous_count += 1

                            cf_exit_reason = (
                                "AMBIGUOUS_TARGET_TRIGGER"
                            )

                            cf_r = actual_r

                            break


                        continue


                # ----------------------------------------------------------
                # Protection/target active only AFTER trigger bar.
                # ----------------------------------------------------------

                if trigger_found:

                    if (
                        bar_number
                        <= trigger_bar
                    ):

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


                    # ------------------------------------------------------
                    # Same-bar target + protection ambiguity.
                    # ------------------------------------------------------

                    if (
                        target_hit
                        and protection_hit
                    ):

                        ambiguous_count += 1

                        cf_exit_bar = (
                            bar_number
                        )

                        cf_exit_date = (
                            bar_date
                        )

                        cf_exit_reason = (
                            "AMBIGUOUS_TARGET_PROTECTION"
                        )

                        cf_r = actual_r

                        break


                    # ------------------------------------------------------
                    # Target.
                    # ------------------------------------------------------

                    if target_hit:

                        target_count += 1

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

                        break


                    # ------------------------------------------------------
                    # Protection.
                    # ------------------------------------------------------

                    if protection_hit:

                        protected_count += 1

                        cf_exit_bar = (
                            bar_number
                        )

                        cf_exit_date = (
                            bar_date
                        )

                        cf_exit_reason = (
                            "PROTECTED"
                        )

                        cf_r = protection_r

                        break


            if (
                trigger_found
                and cf_exit_bar is None
            ):

                retained_actual_count += 1

                cf_exit_reason = (
                    "TRIGGER_NO_CF_EXIT_RETAIN_ACTUAL"
                )

                cf_r = actual_r


            if not trigger_found:

                retained_actual_count += 1


            counterfactual_values.append(
                cf_r
            )


            trade_rows.append(
                {
                    "trade_index": trade[
                        "index"
                    ],
                    "entry_date": trade[
                        "entry_date"
                    ],
                    "exit_date": trade[
                        "exit_date"
                    ],
                    "direction": direction,
                    "setup": trade[
                        "setup"
                    ],
                    "trigger_r": trigger_r,
                    "protection_r": protection_r,
                    "target_r": TARGET_R,
                    "actual_r": actual_r,
                    "counterfactual_r": cf_r,
                    "delta_r": (
                        cf_r
                        - actual_r
                    ),
                    "triggered": (
                        trigger_found
                    ),
                    "trigger_bar": (
                        trigger_bar
                    ),
                    "trigger_date": (
                        trigger_date
                    ),
                    "cf_exit_bar": (
                        cf_exit_bar
                    ),
                    "cf_exit_date": (
                        cf_exit_date
                    ),
                    "cf_exit_reason": (
                        cf_exit_reason
                    ),
                }
            )


        metrics = calculate_metrics(
            counterfactual_values
        )


        delta_r = (
            metrics["total_r"]
            - frozen_total_r
        )


        matrix_rows.append(
            {
                "trigger_r": trigger_r,
                "protection_r": protection_r,
                "trades": metrics[
                    "trades"
                ],
                "wins": metrics[
                    "wins"
                ],
                "losses": metrics[
                    "losses"
                ],
                "win_rate": metrics[
                    "win_rate"
                ],
                "gross_profit": metrics[
                    "gross_profit"
                ],
                "gross_loss": metrics[
                    "gross_loss"
                ],
                "profit_factor": metrics[
                    "profit_factor"
                ],
                "total_r": metrics[
                    "total_r"
                ],
                "average_r": metrics[
                    "average_r"
                ],
                "delta_r": delta_r,
                "triggered": triggered_count,
                "protected": protected_count,
                "targeted": target_count,
                "ambiguous": ambiguous_count,
                "retained_actual": (
                    retained_actual_count
                ),
            }
        )


        pf_text = (
            f"{metrics['profit_factor']:.3f}"
            if np.isfinite(
                metrics["profit_factor"]
            )
            else "INF"
        )


        print(
            f"Trades={metrics['trades']} | "
            f"Win={metrics['win_rate']:.2f}% | "
            f"PF={pf_text} | "
            f"CF={metrics['total_r']:+.2f}R | "
            f"Delta={delta_r:+.2f}R | "
            f"Triggered={triggered_count} | "
            f"Protected={protected_count} | "
            f"Target={target_count} | "
            f"Ambiguous={ambiguous_count}"
        )


# ============================================================================
# MATRIX SUMMARY
# ============================================================================

matrix_df = pd.DataFrame(
    matrix_rows
)


print()
print("=" * 100)
print("SENSITIVITY SUMMARY")
print("=" * 100)


print()

print(
    "Trigger | Protection | CF R | Delta R | "
    "Win Rate | PF | Triggered"
)

print(
    "-" * 85
)


for _, row in (
    matrix_df
    .sort_values(
        [
            "trigger_r",
            "protection_r",
        ]
    )
    .iterrows()
):

    pf_text = (
        f"{row['profit_factor']:.3f}"
        if np.isfinite(
            row["profit_factor"]
        )
        else "INF"
    )

    print(
        f"{row['trigger_r']:>7.2f} | "
        f"{row['protection_r']:>10.2f} | "
        f"{row['total_r']:>+5.2f} | "
        f"{row['delta_r']:>+7.2f} | "
        f"{row['win_rate']:>7.2f}% | "
        f"{pf_text:>5} | "
        f"{int(row['triggered']):>9}"
    )


# ============================================================================
# ANCHOR IDENTIFICATION
# ============================================================================

anchor_row = matrix_df[
    (
        matrix_df["trigger_r"]
        == 1.50
    )
    & (
        matrix_df["protection_r"]
        == 1.00
    )
].iloc[0]


print()
print("=" * 100)
print("ANCHOR RESULT")
print("=" * 100)

print(
    f"Anchor: "
    f"+{anchor_row['trigger_r']:.2f}R "
    f"-> +{anchor_row['protection_r']:.2f}R "
    f"-> +{TARGET_R:.2f}R"
)

print(
    f"Anchor CF R    : "
    f"{anchor_row['total_r']:+.2f}R"
)

print(
    f"Anchor delta R : "
    f"{anchor_row['delta_r']:+.2f}R"
)

print(
    f"Anchor triggered: "
    f"{int(anchor_row['triggered'])}"
)


# ============================================================================
# POSITIVE / NEGATIVE CELLS
# ============================================================================

positive_cells = int(
    (
        matrix_df["delta_r"]
        > VALIDATION_TOLERANCE
    ).sum()
)


negative_cells = int(
    (
        matrix_df["delta_r"]
        < -VALIDATION_TOLERANCE
    ).sum()
)


unchanged_cells = int(
    (
        matrix_df["delta_r"].abs()
        <= VALIDATION_TOLERANCE
    ).sum()
)


print()
print("=" * 100)
print("CELL ROBUSTNESS")
print("=" * 100)

print(
    f"Positive cells : "
    f"{positive_cells}/9"
)

print(
    f"Negative cells : "
    f"{negative_cells}/9"
)

print(
    f"Unchanged cells: "
    f"{unchanged_cells}/9"
)


# ============================================================================
# OUTPUT
# ============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


matrix_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


pd.DataFrame(
    trade_rows
).to_csv(
    TRADE_OUTPUT_FILE,
    index=False,
)


print()
print("=" * 100)
print("OUTPUT")
print("=" * 100)

print(
    f"Matrix file : {OUTPUT_FILE}"
)

print(
    f"Trade file  : {TRADE_OUTPUT_FILE}"
)


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)


print(
    f"Baseline R: "
    f"{frozen_total_r:+.2f}R"
)

print(
    f"Anchor CF R: "
    f"{anchor_row['total_r']:+.2f}R"
)

print(
    f"Anchor delta: "
    f"{anchor_row['delta_r']:+.2f}R"
)

print(
    f"Positive cells: "
    f"{positive_cells}/9"
)

print(
    f"Negative cells: "
    f"{negative_cells}/9"
)


if len(trades) < 20:

    print()
    print(
        "VERDICT: INSUFFICIENT_SAMPLE"
    )

    print()
    print(
        "The sensitivity matrix is useful for "
        "research, but MIDSELECT has only "
        f"{len(trades)} historical trades."
    )

    print(
        "No exit parameter should be promoted "
        "to production from this dataset."
    )

else:

    if (
        positive_cells >= 7
        and negative_cells == 0
    ):

        print()
        print(
            "VERDICT: PROMISING_PARAMETER_ROBUSTNESS"
        )

    else:

        print()
        print(
            "VERDICT: PARAMETER_ROBUSTNESS_NOT_ESTABLISHED"
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