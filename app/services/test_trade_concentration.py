from collections import defaultdict

from app.services.walk_forward_engine import WalkForwardEngine


symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "M&M.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "SUNPHARMA.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
]


windows = {
    2024: {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },

    2025: {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
}


def get_trades(year):

    window = windows[year]

    all_trades = []

    excluded = []

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

        engine = WalkForwardEngine(
            symbol,
            initial_capital=100000,
            brokerage=20,
            slippage=0.10,
        )

        result = engine.run(
            windows=[window],

            adx_values=[15],

            ema_gap_values=[0.0],

            trailing_atr_values=[2.5],

            trailing_activation_atr=2.5,

            momentum_min=None,

            use_market_regime=False,

            initial_stop_atr=3.5,

            risk_per_trade=0.01,
        )

        window_result = result["windows"][0]

        testing = window_result["testing"]

        if testing is None:

            excluded.append(symbol)

            continue

        parameters = (
            window_result["best_parameters"]
        )

        trade_result = engine.test_period(
            start_date=window["test_start"],
            end_date=window["test_end"],
            parameters=parameters,
            use_market_regime=False,
            initial_stop_atr=3.5,
            risk_per_trade=0.01,
            return_trades=True,
        )

        for trade in trade_result.get(
            "trades",
            []
        ):

            trade["symbol"] = symbol

            all_trades.append(trade)

    return all_trades, excluded


for year in [2024, 2025]:

    print()
    print("=" * 100)
    print(
        f"TRADE CONCENTRATION DIAGNOSTIC — {year}"
    )
    print("=" * 100)

    trades, excluded = get_trades(year)

    if not trades:

        print("No trades found.")
        continue

    # =================================
    # Stock-level aggregation
    # =================================

    stock_profit = defaultdict(float)
    stock_trades = defaultdict(int)

    for trade in trades:

        symbol = trade["symbol"]

        stock_profit[symbol] += float(
            trade.get("profit", 0)
        )

        stock_trades[symbol] += 1

    # =================================
    # Trade-level sorting
    # =================================

    winning_trades = sorted(
        [
            t
            for t in trades
            if float(t.get("profit", 0)) > 0
        ],
        key=lambda t: float(
            t.get("profit", 0)
        ),
        reverse=True,
    )

    losing_trades = sorted(
        [
            t
            for t in trades
            if float(t.get("profit", 0)) < 0
        ],
        key=lambda t: float(
            t.get("profit", 0)
        ),
    )

    total_profit = sum(
        float(t.get("profit", 0))
        for t in trades
    )

    # =================================
    # Stock ranking
    # =================================

    ranked_stocks = sorted(
        stock_profit.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print()
    print("PORTFOLIO SUMMARY")
    print("-" * 100)

    print(
        f"Total Trades       : {len(trades)}"
    )

    print(
        f"Total Profit       : "
        f"{total_profit:.2f}"
    )

    print(
        f"Excluded Stocks    : "
        f"{len(excluded)}"
    )

    # =================================
    # Top stocks
    # =================================

    print()
    print("TOP 10 STOCKS BY PROFIT")
    print("-" * 100)

    print(
        "Rank | Stock              | Trades | Profit"
    )

    print("-" * 55)

    for rank, (
        symbol,
        profit
    ) in enumerate(
        ranked_stocks[:10],
        start=1
    ):

        print(
            f"{rank:>4} | "
            f"{symbol:<18} | "
            f"{stock_trades[symbol]:>6} | "
            f"{profit:>10.2f}"
        )

    # =================================
    # Bottom stocks
    # =================================

    print()
    print("BOTTOM 10 STOCKS BY PROFIT")
    print("-" * 100)

    print(
        "Rank | Stock              | Trades | Profit"
    )

    print("-" * 55)

    for rank, (
        symbol,
        profit
    ) in enumerate(
        ranked_stocks[-10:],
        start=1
    ):

        print(
            f"{rank:>4} | "
            f"{symbol:<18} | "
            f"{stock_trades[symbol]:>6} | "
            f"{profit:>10.2f}"
        )

    # =================================
    # Top trade concentration
    # =================================

    print()
    print("TOP 10 WINNING TRADES")
    print("-" * 100)

    print(
        "Rank | Stock              | Profit | "
        "Return | Exit"
    )

    print("-" * 75)

    for rank, trade in enumerate(
        winning_trades[:10],
        start=1
    ):

        print(
            f"{rank:>4} | "
            f"{trade['symbol']:<18} | "
            f"{float(trade.get('profit', 0)):>8.2f} | "
            f"{float(trade.get('return_percent', 0)):>7.2f}% | "
            f"{trade.get('exit_reason', 'UNKNOWN')}"
        )

    # =================================
    # Profit concentration
    # =================================

    top_5_stock_profit = sum(
        profit
        for _, profit in ranked_stocks[:5]
    )

    top_10_stock_profit = sum(
        profit
        for _, profit in ranked_stocks[:10]
    )

    top_5_trade_profit = sum(
        float(t.get("profit", 0))
        for t in winning_trades[:5]
    )

    top_10_trade_profit = sum(
        float(t.get("profit", 0))
        for t in winning_trades[:10]
    )

    print()
    print("PROFIT CONCENTRATION")
    print("-" * 100)

    print(
        f"Top 5 Stocks Profit      : "
        f"{top_5_stock_profit:.2f}"
    )

    print(
        f"Top 5 Stocks Share       : "
        f"{(
            top_5_stock_profit / total_profit * 100
            if total_profit != 0
            else 0
        ):.2f}%"
    )

    print(
        f"Top 10 Stocks Profit     : "
        f"{top_10_stock_profit:.2f}"
    )

    print(
        f"Top 10 Stocks Share      : "
        f"{(
            top_10_stock_profit / total_profit * 100
            if total_profit != 0
            else 0
        ):.2f}%"
    )

    print(
        f"Top 5 Trades Profit      : "
        f"{top_5_trade_profit:.2f}"
    )

    print(
        f"Top 5 Trades Share       : "
        f"{(
            top_5_trade_profit / total_profit * 100
            if total_profit != 0
            else 0
        ):.2f}%"
    )

    print(
        f"Top 10 Trades Profit     : "
        f"{top_10_trade_profit:.2f}"
    )

    print(
        f"Top 10 Trades Share      : "
        f"{(
            top_10_trade_profit / total_profit * 100
            if total_profit != 0
            else 0
        ):.2f}%"
    )

    # =================================
    # Exit reason distribution
    # =================================

    exit_counts = defaultdict(int)
    exit_profit = defaultdict(float)

    for trade in trades:

        reason = trade.get(
            "exit_reason",
            "UNKNOWN"
        )

        exit_counts[reason] += 1

        exit_profit[reason] += float(
            trade.get("profit", 0)
        )

    print()
    print("EXIT REASON DISTRIBUTION")
    print("-" * 100)

    for reason in sorted(exit_counts):

        print(
            f"{reason:<20} | "
            f"Trades: {exit_counts[reason]:>4} | "
            f"Profit: {exit_profit[reason]:>10.2f}"
        )


print()
print("=" * 100)
print("TRADE CONCENTRATION DIAGNOSTIC COMPLETE")
print("=" * 100)