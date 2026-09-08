"""
TradeSense-AI
Exact-Path Early-Adverse-R Walk-Forward Research

Research-only.

This script:
- preserves the original trade universe
- reconstructs the OHLC path
- does not modify BacktestService
- does not create replacement trades
- selects rules using PRIOR YEARS ONLY
- evaluates the selected rule on the NEXT year
"""

import os
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT
# ============================================================

from app.services.backtest_service import (
    BacktestService
)


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0
RISK_PER_TRADE = 0.01

RULES = [
    ("NO_FILTER", None, None),
    ("3B_0.25R", 3, 0.25),
    ("4B_0.25R", 4, 0.25),
    ("5B_0.25R", 5, 0.25),
    ("5B_0.35R", 5, 0.35),
    ("5B_0.50R", 5, 0.50),
]


# ============================================================
# BUILD ORIGINAL TRADE UNIVERSE
# ============================================================

def build_trade_universe():

    print("=" * 100)
    print(
        "BUILDING ORIGINAL TRADE UNIVERSE"
    )
    print("=" * 100)

    all_trades = []
    symbol_data = {}

    for symbol in SYMBOLS:

        print(
            f"\nGenerating baseline trades: {symbol}"
        )

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        signals = (
            service.generate_tradesense_signals()
        )

        result = (
            service.run_directional_backtest(
                signals,
                signal_column="Signal",
                initial_stop_atr=INITIAL_STOP_ATR,
                target_r=TARGET_R,
                risk_per_trade=RISK_PER_TRADE,
            )
        )

        trades = result.get(
            "trades",
            []
        )

        symbol_data[symbol] = {
            "service": service,
            "signals": signals,
        }

        for trade in trades:

            trade_copy = dict(
                trade
            )

            trade_copy["symbol"] = (
                symbol
            )

            all_trades.append(
                trade_copy
            )

        print(
            f"  Trades: {len(trades)}"
        )

    print()
    print(
        f"Original trade universe: "
        f"{len(all_trades)}"
    )

    return (
        all_trades,
        symbol_data
    )


# ============================================================
# CALCULATE COUNTERFACTUAL EXIT
# ============================================================

def calculate_counterfactual(
    trade,
    df,
    threshold_bar,
    threshold_r,
    brokerage,
    slippage_rate
):

    baseline_r = float(
        trade["r_multiple"]
    )

    # --------------------------------------------------------
    # NO FILTER
    # --------------------------------------------------------

    if threshold_bar is None:

        return {
            "r_multiple": baseline_r,
            "counterfactual": False,
            "exit_date": trade[
                "exit_date"
            ],
            "exit_reason": "NO_FILTER",
        }

    direction = trade["direction"]

    entry_date = pd.Timestamp(
        trade["entry_date"]
    )

    original_exit_date = pd.Timestamp(
        trade["exit_date"]
    )

    entry_price = float(
        trade["entry_price"]
    )

    shares = int(
        trade["shares"]
    )

    initial_stop = float(
        trade["initial_stop"]
    )

    initial_risk = float(
        trade["initial_risk"]
    )

    if shares <= 0:
        return {
            "r_multiple": baseline_r,
            "counterfactual": False,
            "exit_date": original_exit_date,
            "exit_reason": "INVALID_SHARES",
        }

    risk_per_share = (
        initial_risk / shares
    )

    # --------------------------------------------------------
    # Original target
    # --------------------------------------------------------

    if direction == "LONG":

        target_price = (
            entry_price
            + risk_per_share
            * TARGET_R
        )

        trigger_price = (
            entry_price
            - risk_per_share
            * threshold_r
        )

    else:

        target_price = (
            entry_price
            - risk_per_share
            * TARGET_R
        )

        trigger_price = (
            entry_price
            + risk_per_share
            * threshold_r
        )

    # --------------------------------------------------------
    # Datetime-normalized path
    # --------------------------------------------------------

    working_df = df.copy()

    working_df.index = pd.to_datetime(
        working_df.index
    )

    working_df = working_df.sort_index()

    path = working_df[
        (
            working_df.index
            > entry_date
        )
        &
        (
            working_df.index
            <= original_exit_date
        )
    ]

    bars_seen = 0

    # --------------------------------------------------------
    # Scan exact OHLC path
    # --------------------------------------------------------

    for index, row in path.iterrows():

        bars_seen += 1

        high_price = float(
            row["High"]
        )

        low_price = float(
            row["Low"]
        )

        # ====================================================
        # ORIGINAL ENGINE PRIORITY
        # ====================================================

        if direction == "LONG":

            # STOP FIRST
            if low_price <= initial_stop:

                return {
                    "r_multiple": baseline_r,
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason":
                        "ORIGINAL_STOP_LOSS",
                }

            # TARGET SECOND
            if high_price >= target_price:

                return {
                    "r_multiple": baseline_r,
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason":
                        "ORIGINAL_TARGET",
                }

        else:

            # STOP FIRST
            if high_price >= initial_stop:

                return {
                    "r_multiple": baseline_r,
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason":
                        "ORIGINAL_STOP_LOSS",
                }

            # TARGET SECOND
            if low_price <= target_price:

                return {
                    "r_multiple": baseline_r,
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason":
                        "ORIGINAL_TARGET",
                }

        # ====================================================
        # EARLY-ADVERSE THRESHOLD
        # ====================================================

        if bars_seen < threshold_bar:
            continue

        if direction == "LONG":

            threshold_hit = (
                low_price <= trigger_price
            )

        else:

            threshold_hit = (
                high_price >= trigger_price
            )

        if not threshold_hit:
            continue

        # ----------------------------------------------------
        # Exit-side slippage
        # ----------------------------------------------------

        if direction == "LONG":

            exit_price = (
                trigger_price
                * (
                    1
                    - slippage_rate
                )
            )

        else:

            exit_price = (
                trigger_price
                * (
                    1
                    + slippage_rate
                )
            )

        # ----------------------------------------------------
        # P&L
        # ----------------------------------------------------

        entry_value = (
            entry_price * shares
        )

        exit_value = (
            exit_price * shares
        )

        entry_cost = float(
            brokerage
        )

        exit_cost = float(
            brokerage
        )

        if direction == "LONG":

            profit = (
                exit_value
                - exit_cost
                - entry_value
                - entry_cost
            )

        else:

            profit = (
                entry_value
                - exit_value
                - entry_cost
                - exit_cost
            )

        if initial_risk > 0:

            r_multiple = (
                profit / initial_risk
            )

        else:

            r_multiple = 0.0

        return {
            "r_multiple": r_multiple,
            "counterfactual": True,
            "exit_date": index,
            "exit_reason":
                f"EARLY_ADVERSE_{threshold_bar}B",
        }

    # --------------------------------------------------------
    # Threshold never hit
    # --------------------------------------------------------

    return {
        "r_multiple": baseline_r,
        "counterfactual": False,
        "exit_date": original_exit_date,
        "exit_reason":
            "ORIGINAL_EXIT",
    }


# ============================================================
# EVALUATE ALL RULES
# ============================================================

def evaluate_all_rules(
    trades,
    symbol_data
):

    outputs = {}

    for (
        rule_name,
        threshold_bar,
        threshold_r
    ) in RULES:

        results = []

        for trade in trades:

            symbol = trade["symbol"]

            service = symbol_data[
                symbol
            ]["service"]

            signals = symbol_data[
                symbol
            ]["signals"]

            result = calculate_counterfactual(
                trade=trade,
                df=signals,
                threshold_bar=threshold_bar,
                threshold_r=threshold_r,
                brokerage=service.brokerage,
                slippage_rate=(
                    service.slippage / 100
                ),
            )

            results.append({
                "symbol": symbol,
                "entry_date": pd.Timestamp(
                    trade["entry_date"]
                ),
                "direction":
                    trade["direction"],
                "baseline_r":
                    float(
                        trade["r_multiple"]
                    ),
                "filtered_r":
                    float(
                        result["r_multiple"]
                    ),
                "counterfactual":
                    result[
                        "counterfactual"
                    ],
                "exit_reason":
                    result[
                        "exit_reason"
                    ],
                "counterfactual_exit_date":
                    result[
                        "exit_date"
                    ],
            })

        outputs[
            rule_name
        ] = results

    return outputs


# ============================================================
# YEAR R
# ============================================================

def year_r(
    results,
    year,
    field
):

    return sum(
        float(row[field])
        for row in results
        if row["entry_date"].year == year
    )


# ============================================================
# YEAR TRADE COUNT
# ============================================================

def year_trade_count(
    results,
    year
):

    return sum(
        1
        for row in results
        if row["entry_date"].year == year
    )


# ============================================================
# YEAR THRESHOLD EXITS
# ============================================================

def year_threshold_exits(
    results,
    year
):

    return sum(
        1
        for row in results
        if (
            row["entry_date"].year == year
            and row["counterfactual"]
        )
    )


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(
    trades,
    outputs
):

    years = sorted(
        set(
            pd.Timestamp(
                trade["entry_date"]
            ).year
            for trade in trades
        )
    )

    print()
    print("=" * 100)
    print(
        "EXACT-PATH WALK-FORWARD VALIDATION"
    )
    print("=" * 100)

    print()
    print(
        "Rule selection uses PRIOR YEARS ONLY."
    )

    print(
        "Selected rule is then frozen for "
        "the following validation year."
    )

    validation_results = []

    # --------------------------------------------------------
    # First year cannot be validated because
    # there is no previous training period.
    # --------------------------------------------------------

    for index in range(1, len(years)):

        validation_year = years[index]

        training_years = years[:index]

        print()
        print("-" * 100)

        print(
            f"TRAINING YEARS: "
            f"{training_years}"
        )

        print(
            f"VALIDATION YEAR: "
            f"{validation_year}"
        )

        print("-" * 100)

        # ====================================================
        # SELECT BEST RULE FROM PRIOR YEARS
        # ====================================================

        training_scores = []

        for (
            rule_name,
            _,
            _
        ) in RULES:

            results = outputs[
                rule_name
            ]

            training_r = sum(
                float(
                    row["filtered_r"]
                )
                for row in results
                if row[
                    "entry_date"
                ].year in training_years
            )

            training_baseline = sum(
                float(
                    row["baseline_r"]
                )
                for row in results
                if row[
                    "entry_date"
                ].year in training_years
            )

            training_delta = (
                training_r
                - training_baseline
            )

            training_scores.append({
                "rule":
                    rule_name,
                "filtered_r":
                    training_r,
                "baseline_r":
                    training_baseline,
                "delta_r":
                    training_delta,
            })

        # ----------------------------------------------------
        # Highest training Delta R wins.
        # NO validation-year information is used.
        # ----------------------------------------------------

        selected = max(
            training_scores,
            key=lambda x:
                x["delta_r"]
        )

        selected_rule = selected[
            "rule"
        ]

        print(
            f"\nSELECTED RULE: "
            f"{selected_rule}"
        )

        print(
            f"Training baseline: "
            f"{selected['baseline_r']:+.2f}R"
        )

        print(
            f"Training filtered: "
            f"{selected['filtered_r']:+.2f}R"
        )

        print(
            f"Training Delta: "
            f"{selected['delta_r']:+.2f}R"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        selected_results = outputs[
            selected_rule
        ]

        validation_baseline = (
            year_r(
                selected_results,
                validation_year,
                "baseline_r"
            )
        )

        validation_filtered = (
            year_r(
                selected_results,
                validation_year,
                "filtered_r"
            )
        )

        validation_delta = (
            validation_filtered
            - validation_baseline
        )

        validation_count = (
            year_trade_count(
                selected_results,
                validation_year
            )
        )

        threshold_exits = (
            year_threshold_exits(
                selected_results,
                validation_year
            )
        )

        print()
        print(
            f"Validation trades: "
            f"{validation_count}"
        )

        print(
            f"Validation baseline: "
            f"{validation_baseline:+.2f}R"
        )

        print(
            f"Validation filtered: "
            f"{validation_filtered:+.2f}R"
        )

        print(
            f"Validation Delta: "
            f"{validation_delta:+.2f}R"
        )

        print(
            f"Threshold exits: "
            f"{threshold_exits}"
        )

        if validation_delta > 0:

            print(
                "Validation result: POSITIVE"
            )

        elif validation_delta < 0:

            print(
                "Validation result: NEGATIVE"
            )

        else:

            print(
                "Validation result: FLAT"
            )

        validation_results.append({
            "training_years":
                training_years,
            "validation_year":
                validation_year,
            "selected_rule":
                selected_rule,
            "training_delta":
                selected[
                    "delta_r"
                ],
            "validation_baseline":
                validation_baseline,
            "validation_filtered":
                validation_filtered,
            "validation_delta":
                validation_delta,
            "validation_trades":
                validation_count,
            "threshold_exits":
                threshold_exits,
        })

    return validation_results


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    validation_results
):

    print()
    print("=" * 100)
    print(
        "WALK-FORWARD SUMMARY"
    )
    print("=" * 100)

    positive = 0
    negative = 0
    flat = 0

    total_baseline = 0.0
    total_filtered = 0.0

    for row in validation_results:

        delta = row[
            "validation_delta"
        ]

        total_baseline += (
            row[
                "validation_baseline"
            ]
        )

        total_filtered += (
            row[
                "validation_filtered"
            ]
        )

        print()
        print(
            f"{row['validation_year']}: "
            f"Rule={row['selected_rule']} | "
            f"Baseline="
            f"{row['validation_baseline']:+.2f}R | "
            f"Filtered="
            f"{row['validation_filtered']:+.2f}R | "
            f"Delta="
            f"{delta:+.2f}R"
        )

        if delta > 0:

            positive += 1

        elif delta < 0:

            negative += 1

        else:

            flat += 1

    total_delta = (
        total_filtered
        - total_baseline
    )

    print()
    print("-" * 100)

    print(
        f"Positive validation years: "
        f"{positive}"
    )

    print(
        f"Negative validation years: "
        f"{negative}"
    )

    print(
        f"Flat validation years: "
        f"{flat}"
    )

    print(
        f"Walk-forward baseline: "
        f"{total_baseline:+.2f}R"
    )

    print(
        f"Walk-forward filtered: "
        f"{total_filtered:+.2f}R"
    )

    print(
        f"Walk-forward Delta: "
        f"{total_delta:+.2f}R"
    )

    print()
    print("=" * 100)
    print(
        "FINAL RESEARCH VERDICT"
    )
    print("=" * 100)

    if positive >= 3:

        print(
            "PROMISING WALK-FORWARD SIGNAL."
        )

        print(
            "The Early-Adverse-R hypothesis "
            "improved at least 3 validation years "
            "using rules selected only from "
            "previous years."
        )

    else:

        print(
            "WEAK / UNSTABLE WALK-FORWARD SIGNAL."
        )

        print(
            "The Early-Adverse-R hypothesis did "
            "not produce positive validation "
            "performance in at least 3 years."
        )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY"
    )

    print()
    print(
        "No production engine changes were made."
    )

    print(
        "The experiment preserves the "
        "original trade universe."
    )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    (
        trades,
        symbol_data
    ) = build_trade_universe()

    if not trades:

        raise RuntimeError(
            "No baseline trades generated."
        )

    print()
    print(
        f"Trade universe: {len(trades)}"
    )

    outputs = evaluate_all_rules(
        trades,
        symbol_data
    )

    # --------------------------------------------------------
    # Invariant check
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "TRADE-COUNT INVARIANT"
    )
    print("=" * 100)

    invariant_pass = True

    for (
        rule_name,
        results
    ) in outputs.items():

        count = len(results)

        print(
            f"{rule_name}: "
            f"{count}"
        )

        if count != len(trades):

            invariant_pass = False

    if not invariant_pass:

        raise RuntimeError(
            "Trade-count invariant FAILED."
        )

    print(
        "\nSTATUS: PASS"
    )

    # --------------------------------------------------------
    # Walk forward
    # --------------------------------------------------------

    validation_results = (
        walk_forward(
            trades,
            outputs
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        validation_results
    )


if __name__ == "__main__":
    main()