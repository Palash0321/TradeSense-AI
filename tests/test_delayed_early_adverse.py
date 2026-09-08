"""
TradeSense-AI
Research-only exact-path Early Adverse-R counterfactual.

IMPORTANT:
- Does NOT modify BacktestService.
- Does NOT create replacement trades.
- Preserves the original baseline trade universe.
- Reconstructs the original OHLC path for each trade.
- Respects original STOP_LOSS / TARGET priority.
- Applies the early-adverse threshold only after the original
  stop/target checks for each bar.
- This is research-only. DO NOT deploy.
"""

import os
import sys

import pandas as pd


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# TradeSense imports
# ============================================================

from app.services.backtest_service import (
    BacktestService
)


# ============================================================
# Configuration
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
    ("3B_0.25R", 3, 0.25),
    ("4B_0.25R", 4, 0.25),
    ("5B_0.25R", 5, 0.25),
    ("5B_0.35R", 5, 0.35),
    ("5B_0.50R", 5, 0.50),
]


# ============================================================
# Helpers
# ============================================================

def normalize_date(value):
    """
    Convert trade/index dates into pandas timestamps.
    """
    return pd.Timestamp(value)


def calculate_exit_profit(
    trade,
    exit_price,
    brokerage
):
    """
    Reconstruct the exact P&L formula used by the directional
    backtest for an early counterfactual exit.
    """

    direction = trade["direction"]

    entry_price = float(
        trade["entry_price"]
    )

    shares = int(
        trade["shares"]
    )

    entry_value = (
        entry_price * shares
    )

    entry_cost = float(
        brokerage
    )

    exit_value = (
        exit_price * shares
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

    initial_risk = float(
        trade["initial_risk"]
    )

    if initial_risk > 0:

        r_multiple = (
            profit / initial_risk
        )

    else:

        r_multiple = 0.0

    return (
        profit,
        r_multiple
    )


def threshold_price(
    trade,
    threshold_r
):
    """
    Calculate the theoretical market price at which the
    position reaches the requested adverse-R threshold.

    LONG:
        entry - threshold * risk_per_share

    SHORT:
        entry + threshold * risk_per_share
    """

    entry_price = float(
        trade["entry_price"]
    )

    shares = int(
        trade["shares"]
    )

    initial_risk = float(
        trade["initial_risk"]
    )

    if shares <= 0:
        return None

    risk_per_share = (
        initial_risk / shares
    )

    if trade["direction"] == "LONG":

        return (
            entry_price
            - threshold_r
            * risk_per_share
        )

    return (
        entry_price
        + threshold_r
        * risk_per_share
    )


def reconstruct_trade(
    trade,
    df,
    threshold_bar,
    threshold_r,
    brokerage,
    slippage_rate
):
    """
    Reconstruct one original trade using the actual OHLC path.

    The original trade remains exactly one trade.

    We scan only from the first bar AFTER entry through the
    original exit date.

    For every bar:
        1. Original stop-loss check
        2. Original target check
        3. Early-adverse threshold check

    Therefore the engine's original exit priority is preserved.
    """

    direction = trade["direction"]

    entry_date = normalize_date(
        trade["entry_date"]
    )

    original_exit_date = normalize_date(
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
            "r_multiple": float(
                trade["r_multiple"]
            ),
            "counterfactual": False,
            "exit_date": original_exit_date,
            "exit_reason": trade.get(
                "exit_reason",
                "ORIGINAL"
            ),
        }

    risk_per_share = (
        initial_risk / shares
    )

    # Reconstruct target using the original
    # directional backtest's target-R configuration.
    if direction == "LONG":

        target_price = (
            entry_price
            + risk_per_share
            * TARGET_R
        )

    else:

        target_price = (
            entry_price
            - risk_per_share
            * TARGET_R
        )

    trigger_price = threshold_price(
        trade,
        threshold_r
    )

    if trigger_price is None:
        return {
            "r_multiple": float(
                trade["r_multiple"]
            ),
            "counterfactual": False,
            "exit_date": original_exit_date,
            "exit_reason": trade.get(
                "exit_reason",
                "ORIGINAL"
            ),
        }

    # --------------------------------------------------------
    # Make sure the dataframe index is datetime.
    # --------------------------------------------------------

    working_df = df.copy()

    working_df.index = pd.to_datetime(
        working_df.index
    )

    working_df = working_df.sort_index()

    # --------------------------------------------------------
    # IMPORTANT:
    # The engine enters on next bar open.
    #
    # Therefore the first diagnostic bar is AFTER the
    # entry bar.
    # --------------------------------------------------------

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
    # Scan original trade path.
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
        # ORIGINAL STOP / TARGET PRIORITY
        # ====================================================

        if direction == "LONG":

            # Stop first
            if low_price <= initial_stop:

                return {
                    "r_multiple": float(
                        trade["r_multiple"]
                    ),
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason": "ORIGINAL_STOP_LOSS",
                }

            # Target second
            if high_price >= target_price:

                return {
                    "r_multiple": float(
                        trade["r_multiple"]
                    ),
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason": "ORIGINAL_TARGET",
                }

        else:

            # Stop first
            if high_price >= initial_stop:

                return {
                    "r_multiple": float(
                        trade["r_multiple"]
                    ),
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason": "ORIGINAL_STOP_LOSS",
                }

            # Target second
            if low_price <= target_price:

                return {
                    "r_multiple": float(
                        trade["r_multiple"]
                    ),
                    "counterfactual": False,
                    "exit_date": index,
                    "exit_reason": "ORIGINAL_TARGET",
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
        # Exact threshold execution with exit-side slippage.
        #
        # LONG -> selling
        # SHORT -> buying to cover
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

        profit, r_multiple = (
            calculate_exit_profit(
                trade,
                exit_price,
                brokerage
            )
        )

        return {
            "r_multiple": r_multiple,
            "profit": profit,
            "counterfactual": True,
            "exit_date": index,
            "exit_reason": (
                f"EARLY_ADVERSE_{threshold_bar}B"
            ),
        }

    # --------------------------------------------------------
    # No counterfactual threshold hit before original exit.
    # --------------------------------------------------------

    return {
        "r_multiple": float(
            trade["r_multiple"]
        ),
        "counterfactual": False,
        "exit_date": original_exit_date,
        "exit_reason": trade.get(
            "exit_reason",
            "ORIGINAL"
        ),
    }


# ============================================================
# Build original baseline trade universe
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
            f"\nGenerating baseline trades: "
            f"{symbol}"
        )

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        signals = (
            service.generate_tradesense_signals()
        )

        backtest_result = (
            service.run_directional_backtest(
                signals,
                signal_column="Signal",
                initial_stop_atr=INITIAL_STOP_ATR,
                target_r=TARGET_R,
                risk_per_trade=RISK_PER_TRADE,
            )
        )

        trades = (
            backtest_result.get(
                "trades",
                []
            )
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
# Evaluate one fixed rule
# ============================================================

def evaluate_rule(
    trades,
    symbol_data,
    threshold_bar,
    threshold_r
):

    baseline_r = 0.0
    counterfactual_r = 0.0

    threshold_exits = 0

    results = []

    for trade in trades:

        baseline_value = float(
            trade["r_multiple"]
        )

        baseline_r += (
            baseline_value
        )

        symbol = trade["symbol"]

        service = symbol_data[
            symbol
        ]["service"]

        signals = symbol_data[
            symbol
        ]["signals"]

        result = reconstruct_trade(
            trade=trade,
            df=signals,
            threshold_bar=threshold_bar,
            threshold_r=threshold_r,
            brokerage=service.brokerage,
            slippage_rate=(
                service.slippage / 100
            ),
        )

        counterfactual_value = float(
            result["r_multiple"]
        )

        counterfactual_r += (
            counterfactual_value
        )

        if result["counterfactual"]:

            threshold_exits += 1

        results.append({
            "symbol": symbol,
            "entry_date": trade[
                "entry_date"
            ],
            "exit_date": trade[
                "exit_date"
            ],
            "direction": trade[
                "direction"
            ],
            "setup": trade.get(
                "setup"
            ),
            "baseline_r": baseline_value,
            "counterfactual_r":
                counterfactual_value,
            "counterfactual":
                result[
                    "counterfactual"
                ],
            "counterfactual_exit_date":
                result[
                    "exit_date"
                ],
            "exit_reason":
                result[
                    "exit_reason"
                ],
        })

    delta_r = (
        counterfactual_r
        - baseline_r
    )

    return {
        "baseline_r": baseline_r,
        "counterfactual_r":
            counterfactual_r,
        "delta_r": delta_r,
        "threshold_exits":
            threshold_exits,
        "trade_count": len(results),
        "results": results,
    }


# ============================================================
# Year analysis
# ============================================================

def print_year_analysis(
    rule_name,
    results,
    baseline_by_year
):

    print()
    print(
        f"YEAR-BY-YEAR: {rule_name}"
    )
    print("-" * 100)

    years = sorted(
        baseline_by_year.keys()
    )

    for year in years:

        baseline_r = (
            baseline_by_year[year]
        )

        filtered_r = 0.0
        count = 0
        exits = 0

        for row in results:

            date = pd.Timestamp(
                row["entry_date"]
            )

            if date.year != year:
                continue

            filtered_r += float(
                row["counterfactual_r"]
            )

            count += 1

            if row["counterfactual"]:
                exits += 1

        delta = (
            filtered_r
            - baseline_r
        )

        print(
            f"{year}: "
            f"Trades={count} | "
            f"Baseline={baseline_r:+.2f}R | "
            f"Filtered={filtered_r:+.2f}R | "
            f"Delta={delta:+.2f}R | "
            f"Threshold exits={exits}"
        )


# ============================================================
# Main
# ============================================================

def main():

    (
        trades,
        symbol_data
    ) = build_trade_universe()

    if not trades:

        print(
            "\nERROR: No trades generated."
        )

        return

    original_count = len(
        trades
    )

    baseline_r = sum(
        float(
            trade["r_multiple"]
        )
        for trade in trades
    )

    baseline_by_year = {}

    for trade in trades:

        year = pd.Timestamp(
            trade["entry_date"]
        ).year

        baseline_by_year.setdefault(
            year,
            0.0
        )

        baseline_by_year[
            year
        ] += float(
            trade["r_multiple"]
        )

    # ========================================================
    # Header
    # ========================================================

    print()
    print("=" * 100)
    print(
        "EXACT-PATH TRADE-PRESERVING "
        "EARLY-ADVERSE-R RESEARCH"
    )
    print("=" * 100)

    print(
        f"Original trade universe: "
        f"{original_count}"
    )

    print(
        f"Initial Stop ATR: "
        f"{INITIAL_STOP_ATR}"
    )

    print(
        f"Target R: "
        f"{TARGET_R}"
    )

    print()

    # ========================================================
    # Baseline
    # ========================================================

    print("-" * 100)
    print("BASELINE")
    print("-" * 100)

    print(
        f"Trades   : {original_count}"
    )

    print(
        f"Total R  : {baseline_r:+.2f}"
    )

    # ========================================================
    # Rules
    # ========================================================

    rule_outputs = []

    print()
    print("-" * 100)
    print(
        "EXACT-PATH COUNTERFACTUAL RULES"
    )
    print("-" * 100)

    for (
        rule_name,
        threshold_bar,
        threshold_r
    ) in RULES:

        output = evaluate_rule(
            trades=trades,
            symbol_data=symbol_data,
            threshold_bar=threshold_bar,
            threshold_r=threshold_r,
        )

        rule_outputs.append(
            (
                rule_name,
                output
            )
        )

        print()
        print(
            f"RULE: {rule_name}"
        )

        print(
            f"Trades   : "
            f"{output['trade_count']}"
        )

        print(
            f"Total R  : "
            f"{output['counterfactual_r']:+.2f}"
        )

        print(
            f"Delta R  : "
            f"{output['delta_r']:+.2f}"
        )

        print(
            f"Threshold exits: "
            f"{output['threshold_exits']}"
        )

        print_year_analysis(
            rule_name,
            output["results"],
            baseline_by_year
        )

    # ========================================================
    # Trade-count invariant
    # ========================================================

    print()
    print("=" * 100)
    print(
        "TRADE-COUNT INVARIANT CHECK"
    )
    print("=" * 100)

    print(
        f"Original trades: "
        f"{original_count}"
    )

    invariant_pass = True

    for (
        rule_name,
        output
    ) in rule_outputs:

        print(
            f"{rule_name}: "
            f"{output['trade_count']}"
        )

        if (
            output["trade_count"]
            != original_count
        ):
            invariant_pass = False

    if invariant_pass:

        print(
            "\nSTATUS: PASS"
        )

        print(
            "Every rule evaluated the "
            "exact same original trade universe."
        )

    else:

        print(
            "\nSTATUS: FAIL"
        )

        print(
            "Trade universe changed."
        )

        raise RuntimeError(
            "Trade-count invariant failed."
        )

    # ========================================================
    # Best rule
    # ========================================================

    best_rule = max(
        rule_outputs,
        key=lambda item:
            item[1]["delta_r"]
    )

    best_name = (
        best_rule[0]
    )

    best_output = (
        best_rule[1]
    )

    print()
    print("=" * 100)
    print(
        "BEST EXACT-PATH "
        "TRADE-PRESERVING RULE"
    )
    print("=" * 100)

    print(
        f"Rule       : {best_name}"
    )

    print(
        f"Delta R    : "
        f"{best_output['delta_r']:+.2f}"
    )

    print(
        f"Total R    : "
        f"{best_output['counterfactual_r']:+.2f}"
    )

    print(
        f"Trades     : "
        f"{best_output['trade_count']}"
    )

    print(
        f"Threshold exits: "
        f"{best_output['threshold_exits']}"
    )

    # ========================================================
    # Research verdict
    # ========================================================

    print()
    print("=" * 100)
    print(
        "RESEARCH VERDICT"
    )
    print("=" * 100)

    if best_output["delta_r"] > 0:

        print(
            "POSITIVE EXACT-PATH "
            "COUNTERFACTUAL SIGNAL."
        )

        print(
            "At least one fixed early-adverse "
            "threshold improves the original "
            "trade universe without creating "
            "replacement trades."
        )

    else:

        print(
            "NO POSITIVE EXACT-PATH "
            "COUNTERFACTUAL SIGNAL."
        )

    print()
    print(
        "RESEARCH STATUS: DO NOT DEPLOY"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is a research reconstruction "
        "using the original trade universe."
    )

    print(
        "It does not modify the production "
        "BacktestService."
    )

    print(
        "It does not generate replacement "
        "entries after a counterfactual exit."
    )

    print(
        "The threshold crossing is evaluated "
        "from the historical OHLC bar path."
    )

    print(
        "If stop/target is hit on a bar, the "
        "original engine's stop-before-target "
        "priority is retained."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()