"""
TradeSense-AI evaluation/backtest regression.

Purpose
-------
Compare two calculation architectures using the SAME evaluation period:

A) Full calculation context
   - Load from 2021-01-01
   - Generate signals over the complete history
   - Evaluate/backtest only from 2021-09-13 onward

B) Rolling evaluation context
   - Load from 2021-09-13
   - Generate signals from that boundary
   - Backtest from the same evaluation period

This test is diagnostic only.

It does NOT modify:
- production code
- frozen historical trades
- holdout trades
"""

import csv
import os

import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

CALCULATION_START = "2021-01-01"
EVALUATION_START = "2021-09-13"

FROZEN_TRADES_FILE = (
    "tests/output/tradesense_trades.csv"
)


def normalize_value(value):
    """
    Normalize values for safe comparison.
    """

    if pd.isna(value):
        return ""

    return str(value)


def trade_identity(trade):
    """
    Identity used by the existing reconciliation logic.
    """

    symbol = normalize_value(
        trade.get("_symbol", "")
    )

    entry_date = normalize_value(
        trade.get("entry_date", "")
    )

    direction = normalize_value(
        trade.get("direction", "")
    )

    return (
        symbol,
        entry_date[:10],
        direction
    )


def load_trades(result):
    """
    Extract trades from the backtest result.

    run_directional_backtest() returns a structure containing
    the trade list. This helper handles the expected dictionary
    format without changing production behavior.
    """

    if isinstance(result, dict):

        if isinstance(
            result.get("trades"),
            list
        ):
            return result["trades"]

        if isinstance(
            result.get("trade_log"),
            list
        ):
            return result["trade_log"]

    if isinstance(result, list):
        return result

    raise RuntimeError(
        "Could not locate trade list in backtest result."
    )


def run_backtest(
    service,
    signals
):
    """
    Run the exact production directional backtest
    on a supplied signal dataframe.
    """

    evaluation_signals = signals.loc[
        signals.index >= EVALUATION_START
    ].copy()

    result = service.run_directional_backtest(
        df=evaluation_signals
    )

    trades = load_trades(result)

    return evaluation_signals, trades, result


def summarize_trades(trades):

    total_r = 0.0
    wins = 0
    losses = 0

    for trade in trades:

        if normalize_value(
            trade.get("exit_reason", "")
        ) == "FINAL_LIQUIDATION":
            continue

        r_value = trade.get("r_multiple")

        if r_value is None:
            r_value = trade.get("R")

        if r_value is None:
            continue

        try:
            r_value = float(r_value)
        except (TypeError, ValueError):
            continue

        total_r += r_value

        if r_value > 0:
            wins += 1
        elif r_value < 0:
            losses += 1

    return {
        "total_r": total_r,
        "wins": wins,
        "losses": losses,
    }


def compare_trade_sets(
    full_trades,
    rolling_trades
):
    """
    Compare trade identities and R values.
    """

    full_map = {
        trade_identity(trade): trade
        for trade in full_trades
        if trade_identity(trade)[1]
    }

    rolling_map = {
        trade_identity(trade): trade
        for trade in rolling_trades
        if trade_identity(trade)[1]
    }

    full_ids = set(full_map)
    rolling_ids = set(rolling_map)

    common_ids = (
        full_ids & rolling_ids
    )

    full_only = (
        full_ids - rolling_ids
    )

    rolling_only = (
        rolling_ids - full_ids
    )

    common_r_delta = 0.0

    r_differences = 0

    for identity in common_ids:

        full_trade = full_map[identity]
        rolling_trade = rolling_map[identity]

        full_r = full_trade.get(
            "r_multiple",
            full_trade.get("R")
        )

        rolling_r = rolling_trade.get(
            "r_multiple",
            rolling_trade.get("R")
        )

        try:
            full_r = float(full_r)
            rolling_r = float(rolling_r)

            delta = (
                rolling_r - full_r
            )

            common_r_delta += delta

            if abs(delta) > 1e-9:
                r_differences += 1

        except (
            TypeError,
            ValueError
        ):
            pass

    return {
        "full_count": len(full_ids),
        "rolling_count": len(rolling_ids),
        "common_count": len(common_ids),
        "full_only": len(full_only),
        "rolling_only": len(rolling_only),
        "common_r_delta": common_r_delta,
        "r_differences": r_differences,
        "full_only_ids": sorted(full_only),
        "rolling_only_ids": sorted(rolling_only),
    }


def load_frozen_trades():

    if not os.path.exists(
        FROZEN_TRADES_FILE
    ):
        return []

    with open(
        FROZEN_TRADES_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


def main():

    print()
    print("=" * 100)
    print("TRADESENSE-AI EVALUATION/BACKTEST REGRESSION")
    print("=" * 100)

    print(
        f"Calculation context : {CALCULATION_START}"
    )

    print(
        f"Evaluation period   : {EVALUATION_START} onward"
    )

    print()
    print(
        "IMPORTANT: frozen historical dataset is READ ONLY."
    )

    print()

    all_full_trades = []
    all_rolling_trades = []

    for symbol in SYMBOLS:

        print()
        print("-" * 100)
        print(symbol)
        print("-" * 100)

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        # ==========================================================
        # A) FULL CALCULATION CONTEXT
        # ==========================================================

        print()
        print(
            "A) FULL CALCULATION CONTEXT"
        )

        full_data = service.load_data(
            start_date=CALCULATION_START
        )

        full_signals = (
            service.generate_tradesense_signals(
                df=full_data
            )
        )

        (
            full_evaluation_signals,
            full_trades,
            full_result
        ) = run_backtest(
            service,
            full_signals
        )

        for trade in full_trades:

            trade["_symbol"] = symbol

        full_summary = summarize_trades(
            full_trades
        )

        print(
            f"Full signal rows: "
            f"{len(full_signals)}"
        )

        print(
            f"Evaluation signal rows: "
            f"{len(full_evaluation_signals)}"
        )

        print(
            f"Trades: "
            f"{len(full_trades)}"
        )

        print(
            f"Total R: "
            f"{full_summary['total_r']:.2f}R"
        )

        # ==========================================================
        # B) ROLLING CALCULATION CONTEXT
        # ==========================================================

        print()
        print(
            "B) ROLLING CALCULATION CONTEXT"
        )

        rolling_data = service.load_data(
            start_date=EVALUATION_START
        )

        rolling_signals = (
            service.generate_tradesense_signals(
                df=rolling_data
            )
        )

        (
            rolling_evaluation_signals,
            rolling_trades,
            rolling_result
        ) = run_backtest(
            service,
            rolling_signals
        )

        for trade in rolling_trades:

            trade["_symbol"] = symbol

        rolling_summary = summarize_trades(
            rolling_trades
        )

        print(
            f"Rolling signal rows: "
            f"{len(rolling_signals)}"
        )

        print(
            f"Trades: "
            f"{len(rolling_trades)}"
        )

        print(
            f"Total R: "
            f"{rolling_summary['total_r']:.2f}R"
        )

        # ==========================================================
        # TRADE COMPARISON
        # ==========================================================

        comparison = compare_trade_sets(
            full_trades,
            rolling_trades
        )

        print()
        print(
            "TRADE IDENTITY COMPARISON"
        )

        print(
            f"Full-context trades : "
            f"{comparison['full_count']}"
        )

        print(
            f"Rolling-context trades: "
            f"{comparison['rolling_count']}"
        )

        print(
            f"Common trades       : "
            f"{comparison['common_count']}"
        )

        print(
            f"Full-context only   : "
            f"{comparison['full_only']}"
        )

        print(
            f"Rolling-context only: "
            f"{comparison['rolling_only']}"
        )

        print(
            f"Common-trade R delta: "
            f"{comparison['common_r_delta']:.2f}R"
        )

        print(
            f"Common trades with R differences: "
            f"{comparison['r_differences']}"
        )

        # ==========================================================
        # SHOW IDENTITY DIFFERENCES
        # ==========================================================

        if comparison["full_only_ids"]:

            print()
            print(
                "FULL-CONTEXT ONLY:"
            )

            for identity in comparison[
                "full_only_ids"
            ]:

                print(
                    f"  {identity}"
                )

        if comparison["rolling_only_ids"]:

            print()
            print(
                "ROLLING-CONTEXT ONLY:"
            )

            for identity in comparison[
                "rolling_only_ids"
            ]:

                print(
                    f"  {identity}"
                )

        all_full_trades.extend(
            full_trades
        )

        all_rolling_trades.extend(
            rolling_trades
        )

    # ==============================================================
    # GLOBAL COMPARISON
    # ==============================================================

    print()
    print("=" * 100)
    print("GLOBAL COMPARISON")
    print("=" * 100)

    global_comparison = compare_trade_sets(
        all_full_trades,
        all_rolling_trades
    )

    full_summary = summarize_trades(
        all_full_trades
    )

    rolling_summary = summarize_trades(
        all_rolling_trades
    )

    print()
    print(
        f"Full-context trades : "
        f"{global_comparison['full_count']}"
    )

    print(
        f"Rolling-context trades: "
        f"{global_comparison['rolling_count']}"
    )

    print(
        f"Common trades       : "
        f"{global_comparison['common_count']}"
    )

    print(
        f"Full-context only   : "
        f"{global_comparison['full_only']}"
    )

    print(
        f"Rolling-context only: "
        f"{global_comparison['rolling_only']}"
    )

    print()
    print(
        f"Full-context total R: "
        f"{full_summary['total_r']:.2f}R"
    )

    print(
        f"Rolling-context total R: "
        f"{rolling_summary['total_r']:.2f}R"
    )

    print(
        f"Total-R difference: "
        f"{rolling_summary['total_r'] - full_summary['total_r']:.2f}R"
    )

    print(
        f"Common-trade R delta: "
        f"{global_comparison['common_r_delta']:.2f}R"
    )

    print(
        f"Common trades with R differences: "
        f"{global_comparison['r_differences']}"
    )

    # ==============================================================
    # FROZEN DATASET COMPARISON
    # ==============================================================

    frozen_trades = load_frozen_trades()

    frozen_map = {
        trade_identity(trade): trade
        for trade in frozen_trades
    }

    full_map = {
        trade_identity(trade): trade
        for trade in all_full_trades
    }

    frozen_ids = set(
        frozen_map
    )

    full_ids = set(
        full_map
    )

    print()
    print("=" * 100)
    print("FROZEN DATASET COMPARISON")
    print("=" * 100)

    print(
        f"Frozen historical trades: "
        f"{len(frozen_ids)}"
    )

    print(
        f"Full-context trades: "
        f"{len(full_ids)}"
    )

    print(
        f"Common with frozen: "
        f"{len(frozen_ids & full_ids)}"
    )

    print(
        f"Frozen-only: "
        f"{len(frozen_ids - full_ids)}"
    )

    print(
        f"Full-context-only: "
        f"{len(full_ids - frozen_ids)}"
    )

    print()
    print(
        "NO FILES WERE MODIFIED."
    )

    print()
    print("=" * 100)
    print("DIAGNOSTIC RESULT")
    print("=" * 100)

    if global_comparison["full_count"] > 0:

        print(
            "PASS: Full-context backtest completed."
        )

    else:

        print(
            "FAIL: Full-context produced no trades."
        )

    print()
    print(
        "Rolling-context differences are diagnostic only."
    )

    print(
        "Do NOT replace the frozen dataset from this test."
    )


if __name__ == "__main__":
    main()