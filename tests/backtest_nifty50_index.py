"""
TradeSense-AI
NIFTY 50 Index Backtest

Purpose:
    Apply the existing TradeSense signal engine and directional
    backtest engine to the NIFTY 50 index itself.

IMPORTANT:
    - Does NOT modify production code.
    - Does NOT modify tradesense_trades.csv.
    - Does NOT modify tradesense_holdout_trades.csv.
    - Research/backtest only.
"""

import os
import sys

import pandas as pd
import yfinance as yf

from app.services.backtest_service import BacktestService


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_NAME = "NIFTY 50"
TICKER = "^NSEI"

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

OUTPUT_DIR = "tests/output/index_backtests"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_index_trades.csv"
)

INITIAL_CAPITAL = 100000

# IMPORTANT:
# Use the same historical stop/target assumptions that were used
# for the frozen TradeSense research generation.
INITIAL_STOP_ATR = 3.5
TARGET_R = 2.0
RISK_PER_TRADE = 0.01


# ============================================================
# DATA DOWNLOAD
# ============================================================

def load_nifty50_data():
    print()
    print("=" * 100)
    print("NIFTY 50 INDEX DATA")
    print("=" * 100)

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            "NIFTY 50 data download returned no data."
        )

    # yfinance can return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"NIFTY 50 data is missing columns: {missing}"
        )

    df = df[required_columns].copy()

    df = df.dropna(
        subset=required_columns
    )

    # Remove duplicate dates defensively.
    df = df[
        ~df.index.duplicated(
            keep="first"
        )
    ].copy()

    # Ensure positive OHLC.
    invalid_ohlc = (
        (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Close"] <= 0)
        | (df["High"] < df["Low"])
    )

    invalid_count = int(
        invalid_ohlc.sum()
    )

    if invalid_count > 0:
        raise RuntimeError(
            f"NIFTY 50 contains {invalid_count} invalid OHLC rows."
        )

    print(f"Ticker:       {TICKER}")
    print(f"Rows:         {len(df)}")
    print(f"Start:        {df.index.min()}")
    print(f"End:          {df.index.max()}")

    print()
    print("Latest 5 rows:")
    print(
        df.tail(5).to_string()
    )

    print()
    print("DATA STATUS: PASS")

    return df


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signals(df):
    print()
    print("=" * 100)
    print("TRADESENSE SIGNAL GENERATION")
    print("=" * 100)

    service = BacktestService(
        symbol=TICKER,
        strategy="tradesense",
        initial_capital=INITIAL_CAPITAL,
    )

    signals = service.generate_tradesense_signals(
        df=df.copy(),
        swing_window=3,
    )

    if signals is None or signals.empty:
        raise RuntimeError(
            "TradeSense signal engine returned no data."
        )

    print(f"Signal rows: {len(signals)}")

    if "Signal" not in signals.columns:
        raise RuntimeError(
            "Signal column was not generated."
        )

    signal_counts = (
        signals["Signal"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Signal distribution:")
    print(signal_counts.to_string())

    long_count = int(
        (signals["Signal"] == 1).sum()
    )

    short_count = int(
        (signals["Signal"] == -1).sum()
    )

    wait_count = int(
        (signals["Signal"] == 0).sum()
    )

    print()
    print(f"LONG signals:  {long_count}")
    print(f"SHORT signals: {short_count}")
    print(f"WAIT signals:  {wait_count}")

    return service, signals


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(service, signals):
    print()
    print("=" * 100)
    print("NIFTY 50 DIRECTIONAL BACKTEST")
    print("=" * 100)

    print(
        f"Initial capital:       ₹{INITIAL_CAPITAL:,.2f}"
    )
    print(
        f"Initial stop ATR:      {INITIAL_STOP_ATR}"
    )
    print(
        f"Target R:              {TARGET_R}"
    )
    print(
        f"Risk per trade:        {RISK_PER_TRADE:.2%}"
    )

    result = service.run_directional_backtest(
        signals.copy(),
        signal_column="Signal",
        initial_stop_atr=INITIAL_STOP_ATR,
        target_r=TARGET_R,
        risk_per_trade=RISK_PER_TRADE,
    )

    trades = result.get(
        "trades",
        []
    )

    if trades is None:
        trades = []

    print()
    print(f"Executed trades: {len(trades)}")

    return result, trades


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

def calculate_metrics(trades):
    print()
    print("=" * 100)
    print("NIFTY 50 BACKTEST RESULTS")
    print("=" * 100)

    if not trades:
        print("NO TRADES GENERATED.")
        return

    trade_df = pd.DataFrame(trades)

    # --------------------------------------------------------
    # R MULTIPLE
    # --------------------------------------------------------

    if "r_multiple" in trade_df.columns:
        r_series = pd.to_numeric(
            trade_df["r_multiple"],
            errors="coerce"
        ).fillna(0.0)
    else:
        r_series = pd.Series(
            [0.0] * len(trade_df)
        )

    total_r = float(
        r_series.sum()
    )

    avg_r = float(
        r_series.mean()
    )

    winners = r_series[
        r_series > 0
    ]

    losers = r_series[
        r_series < 0
    ]

    winning_trades = len(winners)
    losing_trades = len(losers)

    win_rate = (
        winning_trades
        / len(r_series)
        * 100
    )

    gross_profit_r = float(
        winners.sum()
    )

    gross_loss_r = abs(
        float(losers.sum())
    )

    if gross_loss_r > 0:
        profit_factor = (
            gross_profit_r
            / gross_loss_r
        )
    else:
        profit_factor = float("inf")

    # --------------------------------------------------------
    # P&L
    # --------------------------------------------------------

    if "profit" in trade_df.columns:
        profit_series = pd.to_numeric(
            trade_df["profit"],
            errors="coerce"
        ).fillna(0.0)

        total_profit = float(
            profit_series.sum()
        )
    else:
        total_profit = 0.0

    # --------------------------------------------------------
    # EXIT REASONS
    # --------------------------------------------------------

    if "exit_reason" in trade_df.columns:
        exit_counts = (
            trade_df["exit_reason"]
            .value_counts()
        )
    else:
        exit_counts = pd.Series(
            dtype="int64"
        )

    # --------------------------------------------------------
    # DIRECTIONS
    # --------------------------------------------------------

    if "direction" in trade_df.columns:
        direction_counts = (
            trade_df["direction"]
            .value_counts()
        )
    else:
        direction_counts = pd.Series(
            dtype="int64"
        )

    # --------------------------------------------------------
    # SETUPS
    # --------------------------------------------------------

    if "setup" in trade_df.columns:
        setup_counts = (
            trade_df["setup"]
            .value_counts()
        )
    else:
        setup_counts = pd.Series(
            dtype="int64"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        f"Trades:              {len(trade_df)}"
    )
    print(
        f"Winners:             {winning_trades}"
    )
    print(
        f"Losers:              {losing_trades}"
    )
    print(
        f"Win rate:            {win_rate:.2f}%"
    )
    print(
        f"Gross profit R:      {gross_profit_r:.2f}R"
    )
    print(
        f"Gross loss R:        -{gross_loss_r:.2f}R"
    )

    if profit_factor == float("inf"):
        print(
            "Profit Factor:       INF"
        )
    else:
        print(
            f"Profit Factor:       {profit_factor:.3f}"
        )

    print(
        f"Total R:              {total_r:.2f}R"
    )
    print(
        f"Average R:            {avg_r:.3f}R"
    )
    print(
        f"Total P&L:            ₹{total_profit:,.2f}"
    )

    print()
    print(
        f"Max Drawdown:         "
        f"{calculate_trade_drawdown(trade_df.to_dict("records"), INITIAL_CAPITAL):.2f}%"
    )

    print()
    print("-" * 100)
    print("EXIT REASONS")
    print("-" * 100)

    if exit_counts.empty:
        print("No exit information.")
    else:
        print(
            exit_counts.to_string()
        )

    print()
    print("-" * 100)
    print("DIRECTION")
    print("-" * 100)

    if direction_counts.empty:
        print("No direction information.")
    else:
        print(
            direction_counts.to_string()
        )

    print()
    print("-" * 100)
    print("SETUPS")
    print("-" * 100)

    if setup_counts.empty:
        print("No setup information.")
    else:
        print(
            setup_counts.to_string()
        )

    print()
    print("-" * 100)
    print("FIRST 10 TRADES")
    print("-" * 100)

    display_columns = [
        column
        for column in [
            "_symbol",
            "entry_date",
            "direction",
            "setup",
            "entry_price",
            "exit_date",
            "exit_reason",
            "profit",
            "r_multiple",
        ]
        if column in trade_df.columns
    ]

    if display_columns:
        print(
            trade_df[
                display_columns
            ]
            .head(10)
            .to_string(index=False)
        )

    return trade_df


# ============================================================
# SAVE RESEARCH OUTPUT
# ============================================================

def save_results(trades):
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not trades:
        print()
        print(
            "No trades to save."
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
    print("=" * 100)
    print("RESEARCH OUTPUT")
    print("=" * 100)
    print(
        f"Saved NIFTY 50 research trades:"
    )
    print(
        OUTPUT_FILE
    )
    print(
        f"Rows saved: {len(trade_df)}"
    )

def calculate_trade_drawdown(trades, initial_capital=100000.0):
    """
    Research-only closed-trade equity drawdown.

    Uses the actual realized trade P&L in chronological order.
    Does not modify the production backtest engine.
    """

    if not trades:
        return 0.0

    ordered_trades = sorted(
        trades,
        key=lambda trade: str(
            trade.get("exit_date", "")
        )
    )

    equity = float(initial_capital)
    peak_equity = equity
    max_drawdown = 0.0

    for trade in ordered_trades:
        profit = float(
            trade.get("profit", 0.0) or 0.0
        )

        equity += profit

        if equity > peak_equity:
            peak_equity = equity

        if peak_equity > 0:
            drawdown = (
                (peak_equity - equity)
                / peak_equity
            ) * 100.0

            max_drawdown = max(
                max_drawdown,
                drawdown
            )

    return round(max_drawdown, 2)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("TRADESENSE-AI — NIFTY 50 INDEX BACKTEST")
    print("=" * 100)

    print()
    print(
        "IMPORTANT: This is an independent research backtest."
    )
    print(
        "Frozen historical and genuine holdout datasets will NOT be modified."
    )

    df = load_nifty50_data()

    service, signals = generate_signals(
        df
    )

    result, trades = run_backtest(
        service,
        signals
    )

    trade_df = calculate_metrics(
        trades
    )

    save_results(
        trades
    )

    print()
    print("=" * 100)
    print("BACKTEST COMPLETE")
    print("=" * 100)

    print()
    print(
        "Production code:      UNCHANGED"
    )
    print(
        "Frozen historical:    UNCHANGED"
    )
    print(
        "Future holdout:       UNCHANGED"
    )
    print(
        "NIFTY 50 research:    COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()