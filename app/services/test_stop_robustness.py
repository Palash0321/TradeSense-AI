from app.services.strategy_robustness import StrategyRobustness


symbols = [
    'RELIANCE.NS',
    'TCS.NS',
    'INFY.NS',
    'HDFCBANK.NS',
    'ICICIBANK.NS',
    'SBIN.NS',
    'LT.NS',
    'ITC.NS',
    'BHARTIARTL.NS',
    'AXISBANK.NS',
    'HCLTECH.NS',
    'WIPRO.NS',
    'TECHM.NS',
    'KOTAKBANK.NS',
    'BAJFINANCE.NS',
    'MARUTI.NS',
    'M&M.NS',
    'TATASTEEL.NS',
    'JSWSTEEL.NS',
    'SUNPHARMA.NS',
    'HINDUNILVR.NS',
    'ASIANPAINT.NS',
    'TITAN.NS',
    'NTPC.NS',
    'POWERGRID.NS',
    'ONGC.NS',
    'COALINDIA.NS',
    'ADANIENT.NS',
    'ADANIPORTS.NS'
]


stop_values = [3.0, 3.5]


print("=" * 90)
print("INITIAL STOP ATR — ROBUSTNESS COMPARISON")
print("=" * 90)
print()

for stop_atr in stop_values:

    print("-" * 90)
    print(f"TESTING INITIAL STOP ATR = {stop_atr}")
    print("-" * 90)

    robustness = StrategyRobustness(
        symbols=symbols,
        brokerage=20,
        slippage=0.10
    )

    results = []

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

        from app.services.backtest_service import BacktestService

        service = BacktestService(
            symbol=symbol,
            strategy="ema_atr",
            brokerage=20,
            slippage=0.10,
            initial_capital=100000
        )

        metrics = service.performance_metrics(
            adx_min=15,
            ema_gap_min=0.0,
            trailing_atr=2.5,
            trailing_activation_atr=2.5,
            initial_stop_atr=stop_atr,
            momentum_min=None,
            use_market_regime=False
        )

        results.append({
            "symbol": symbol,

            "total_trades":
                metrics["total_trades"],

            "net_profit":
                metrics["net_profit"],

            "total_return":
                metrics["total_return"],

            "profit_factor":
                metrics["profit_factor"],

            "win_rate":
                metrics["win_rate"],

            "max_drawdown":
                metrics["max_drawdown"],

            "sharpe_ratio":
                metrics["sharpe_ratio"]
        })

    score = robustness.score_configuration(
        results
    )

    print()
    print(f"INITIAL STOP ATR : {stop_atr}")
    print()
    print(
        f"Average Return      : "
        f"{score['average_return']:.2f}%"
    )

    print(
        f"Median Return       : "
        f"{score['median_return']:.2f}%"
    )

    print(
        f"Average Profit Factor: "
        f"{score['average_profit_factor']:.2f}"
    )

    print(
        f"Average Drawdown    : "
        f"{score['average_drawdown']:.2f}%"
    )

    print(
        f"Profitable Stocks   : "
        f"{score['profitable_stocks']}"
        f"/"
        f"{score['total_stocks']}"
    )

    print(
        f"Positive Ratio      : "
        f"{score['positive_stock_ratio']:.2f}%"
    )

    print(
        f"Robustness Score    : "
        f"{score['robustness_score']:.4f}"
    )

    print()


print("=" * 90)
print("STOP ROBUSTNESS COMPARISON COMPLETE")
print("=" * 90)