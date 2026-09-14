"""
SENSEX Index Baseline Backtest
================================

Research-only backtest.

Purpose:
    Apply the existing TradeSense signal-generation and directional
    backtest engine to the SENSEX index.

This script does NOT modify:
    - production strategy
    - frozen historical trades
    - holdout trades

Index:
    SENSEX
    Yahoo ticker: ^BSESN

This is an independent index validation.
No NIFTY-specific filters or conclusions are applied.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
# PROJECT IMPORT
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from app.services.backtest_service import BacktestService

def calculate_trade_drawdown(
    trades: list[dict],
    initial_capital: float,
) -> float:
    """
    Calculate maximum drawdown from closed-trade realized equity.

    This is a research-side helper.
    It does not modify the production backtest engine.
    """

    if not trades:
        return 0.0

    equity = float(initial_capital)
    peak_equity = equity
    max_drawdown_pct = 0.0

    for trade in trades:

        pnl = trade.get("pnl", 0.0)

        try:
            pnl = float(pnl or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0

        equity += pnl

        if equity > peak_equity:
            peak_equity = equity

        if peak_equity > 0:

            drawdown_pct = (
                (peak_equity - equity)
                / peak_equity
                * 100.0
            )

            max_drawdown_pct = max(
                max_drawdown_pct,
                drawdown_pct,
            )

    return max_drawdown_pct

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "SENSEX"

TICKER = "^BSESN"

START_DATE = "2021-01-01"

END_DATE = "2026-09-12"

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sensex_index_trades.csv"
)

INITIAL_CAPITAL = 1000000

INITIAL_STOP_ATR = 3.5

TARGET_R = 2.0

RISK_PER_TRADE = 0.01


# ============================================================================
# DATA LOADING
# ============================================================================

def load_sensex_data() -> pd.DataFrame:

    print("=" * 100)
    print("SENSEX INDEX BASELINE BACKTEST")
    print("=" * 100)

    print()
    print("Downloading SENSEX data...")
    print(
        f"Ticker:     {TICKER}"
    )
    print(
        f"Start:      {START_DATE}"
    )
    print(
        f"End:        {END_DATE}"
    )

    data = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            "SENSEX data download returned no data."
        )

    # ------------------------------------------------------------------------
    # Handle yfinance MultiIndex columns.
    # ------------------------------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = [
            column[0]
            for column in data.columns
        ]

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required OHLC columns: {missing}"
        )

    data = data[
        required_columns
        + (
            ["Volume"]
            if "Volume" in data.columns
            else []
        )
    ].copy()

    # ------------------------------------------------------------------------
    # Numeric conversion.
    # ------------------------------------------------------------------------

    for column in data.columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    )

    # ------------------------------------------------------------------------
    # OHLC integrity.
    # ------------------------------------------------------------------------

    invalid_ohlc = (
        (data["High"] < data["Low"])
        |
        (data["High"] < data["Open"])
        |
        (data["High"] < data["Close"])
        |
        (data["Low"] > data["Open"])
        |
        (data["Low"] > data["Close"])
    )

    invalid_count = int(
        invalid_ohlc.sum()
    )

    duplicate_count = int(
        data.index.duplicated().sum()
    )

    print()
    print("=" * 100)
    print("SENSEX DATA VALIDATION")
    print("=" * 100)

    print(
        f"Rows:                 {len(data)}"
    )

    print(
        f"Date range:           "
        f"{data.index.min().date()} → "
        f"{data.index.max().date()}"
    )

    print(
        f"Invalid OHLC rows:    {invalid_count}"
    )

    print(
        f"Duplicate dates:      {duplicate_count}"
    )

    if invalid_count != 0:
        raise RuntimeError(
            "Invalid OHLC rows detected."
        )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate dates detected."
        )

    print(
        "STATUS:               PASS"
    )

    return data


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(
    data: pd.DataFrame,
) -> tuple[BacktestService, pd.DataFrame]:

    print()
    print("=" * 100)
    print("GENERATING TRADESENSE SIGNALS")
    print("=" * 100)

    service = BacktestService(
        symbol=TICKER,
        strategy="tradesense",
        initial_capital=INITIAL_CAPITAL,
    )

    signals = service.generate_tradesense_signals(
        df=data.copy(),
        swing_window=3,
    )

    if signals is None or signals.empty:
        raise RuntimeError(
            "Signal generation returned no rows."
        )

    print(
        f"Signal rows:          {len(signals)}"
    )

    if "Signal" in signals.columns:

        signal_counts = (
            signals["Signal"]
            .value_counts()
            .sort_index()
        )

        print()
        print("Signal distribution:")

        for value, count in signal_counts.items():

            print(
                f"  {int(value):>2}: "
                f"{int(count)}"
            )

    return service, signals


# ============================================================================
# BACKTEST
# ============================================================================

def run_backtest(
    service: BacktestService,
    signals: pd.DataFrame,
) -> list[dict]:

    print()
    print("=" * 100)
    print("RUNNING DIRECTIONAL BACKTEST")
    print("=" * 100)

    result = service.run_directional_backtest(
        signals.copy(),
        signal_column="Signal",
        initial_stop_atr=INITIAL_STOP_ATR,
        target_r=TARGET_R,
        risk_per_trade=RISK_PER_TRADE,
    )

    if result is None:
        raise RuntimeError(
            "Backtest returned None."
        )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Unexpected backtest result type: "
            f"{type(result)}"
        )

    # ------------------------------------------------------------------------
    # The production TradeSense backtest returns a dictionary.
    # The actual trade records are stored under the "trades" key.
    # ------------------------------------------------------------------------

    trades = result.get(
        "trades",
        []
    )

    if trades is None:
        trades = []

    if isinstance(
        trades,
        pd.DataFrame
    ):
        trades = trades.to_dict(
            "records"
        )

    elif isinstance(
        trades,
        list
    ):
        trades = trades

    else:
        raise RuntimeError(
            "Unexpected 'trades' value type: "
            f"{type(trades)}"
        )

    print(
        f"Executed trades:      {len(trades)}"
    )

    return trades


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    trades: list[dict],
) -> dict:

    if not trades:

        return {
            "trade_count": 0,
            "winners": 0,
            "losers": 0,
            "win_rate": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "pnl": 0.0,
        }

    trade_df = pd.DataFrame(
        trades
    )

    if "r_multiple" not in trade_df.columns:
        raise RuntimeError(
            "Backtest trades do not contain "
            "'r_multiple'."
        )

    trade_df["r_multiple"] = pd.to_numeric(
        trade_df["r_multiple"],
        errors="coerce"
    )

    trade_df = trade_df[
        trade_df["r_multiple"].notna()
    ].copy()

    r_values = (
        trade_df["r_multiple"]
        .astype(float)
    )

    winners = int(
        (r_values > 0).sum()
    )

    losers = int(
        (r_values <= 0).sum()
    )

    trade_count = len(
        trade_df
    )

    win_rate = (
        winners
        / trade_count
        * 100.0
        if trade_count
        else 0.0
    )

    gross_profit_r = float(
        r_values[
            r_values > 0
        ].sum()
    )

    gross_loss_r = float(
        r_values[
            r_values < 0
        ].sum()
    )

    if gross_loss_r < 0:

        profit_factor = (
            gross_profit_r
            / abs(gross_loss_r)
        )

    else:

        profit_factor = float(
            "inf"
        )

    total_r = float(
        r_values.sum()
    )

    avg_r = (
        total_r
        / trade_count
        if trade_count
        else 0.0
    )

    pnl = float(
        trade_df.get(
            "pnl",
            pd.Series(
                [0.0] * trade_count
            )
        ).sum()
    )

    return {
        "trade_count": trade_count,
        "winners": winners,
        "losers": losers,
        "win_rate": win_rate,
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "profit_factor": profit_factor,
        "total_r": total_r,
        "avg_r": avg_r,
        "pnl": pnl,
    }


# ============================================================================
# TRADE OUTPUT
# ============================================================================

def save_trades(
    trades: list[dict],
) -> None:

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not trades:
        print()
        print(
            "No trades generated; "
            "nothing to save."
        )
        return

    trade_df = pd.DataFrame(
        trades
    )

    trade_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved trades: "
        f"{OUTPUT_FILE}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    data = load_sensex_data()

    service, signals = generate_signals(
        data
    )

    trades = run_backtest(
        service,
        signals
    )

    metrics = calculate_metrics(
        trades
    )

    print()
    print("=" * 100)
    print("SENSEX BASELINE RESULTS")
    print("=" * 100)

    print(
        f"Index:                 {INDEX_NAME}"
    )

    print(
        f"Ticker:                {TICKER}"
    )

    print(
        f"Trades:                "
        f"{metrics['trade_count']}"
    )

    print(
        f"Winners:               "
        f"{metrics['winners']}"
    )

    print(
        f"Losers:                "
        f"{metrics['losers']}"
    )

    print(
        f"Win rate:              "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"Gross profit:          "
        f"{metrics['gross_profit_r']:.2f}R"
    )

    print(
        f"Gross loss:            "
        f"{metrics['gross_loss_r']:.2f}R"
    )

    print(
        f"Profit factor:         "
        f"{metrics['profit_factor']:.3f}"
    )

    print(
        f"Total R:               "
        f"{metrics['total_r']:.2f}R"
    )

    print(
        f"Average R/trade:       "
        f"{metrics['avg_r']:.3f}R"
    )

    print(
        f"P&L:                   "
        f"₹{metrics['pnl']:.2f}"
    )

    # ------------------------------------------------------------------------
    # Exit breakdown.
    # ------------------------------------------------------------------------

    trade_df = pd.DataFrame(
        trades
    )

    if not trade_df.empty:

        if "exit_reason" in trade_df.columns:

            print()
            print("Exit reasons:")

            print(
                trade_df[
                    "exit_reason"
                ]
                .value_counts()
                .to_string()
            )

        if "direction" in trade_df.columns:

            print()
            print("Direction:")

            print(
                trade_df[
                    "direction"
                ]
                .value_counts()
                .to_string()
            )

        if "setup" in trade_df.columns:

            print()
            print("Setup:")

            print(
                trade_df[
                    "setup"
                ]
                .value_counts()
                .to_string()
            )

        # --------------------------------------------------------------------
        # Research drawdown using the existing project helper.
        # --------------------------------------------------------------------

        research_drawdown = calculate_trade_drawdown(
            trade_df.to_dict(
                "records"
            ),
            initial_capital=INITIAL_CAPITAL,
        )

        print()
        print(
            f"Max Drawdown:         "
            f"{research_drawdown:.2f}%"
        )

        print(
            "Drawdown Method:      "
            "Closed-trade realized equity"
        )

    save_trades(
        trades
    )

    # ------------------------------------------------------------------------
    # Integrity statement.
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("INTEGRITY")
    print("=" * 100)

    print(
        "Production strategy modified:     NO"
    )

    print(
        "Frozen historical modified:       NO"
    )

    print(
        "Holdout modified:                 NO"
    )

    print(
        "NIFTY-specific filters applied:   NO"
    )

    print(
        "Status:                            RESEARCH ONLY"
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()