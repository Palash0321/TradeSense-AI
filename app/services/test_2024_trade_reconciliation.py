from app.services.backtest_service import BacktestService


SYMBOLS = [
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


def run():

    print("=" * 95)
    print("2024 TRADE RECONCILIATION DIAGNOSTIC")
    print("=" * 95)

    print()
    print("Frozen Configuration")
    print("-" * 95)
    print("ADX                  : 15")
    print("EMA Gap              : 0.0")
    print("Trailing ATR         : 2.5")
    print("Activation ATR       : 2.5")
    print("Initial Stop ATR     : 3.5")
    print("Risk Per Trade       : 1.00%")
    print("Momentum             : OFF")
    print("Market Regime        : OFF")

    print()
    print("=" * 95)
    print("DIRECT 2024 TRADE EXTRACTION")
    print("=" * 95)

    all_trades = []

    for symbol in SYMBOLS:

        print(
            f"Running {symbol}...",
            flush=True
        )

        service = BacktestService(
            symbol=symbol,
            strategy="ema_atr",
            brokerage=20,
            slippage=0.10,
        )

        metrics = service.performance_metrics(
            start_date="2024-01-01",
            end_date="2024-12-31",
            ema_gap_min=0.0,
            adx_min=15,
            trailing_atr=2.5,
            trailing_activation_atr=2.5,
            momentum_min=None,
            use_market_regime=False,
            initial_stop_atr=3.5,
            risk_per_trade=0.01,
        )

        trades = metrics.get(
            "trades",
            []
        )

        for trade in trades:

            all_trades.append({
                "symbol": symbol,
                "buy_date": trade.get("buy_date"),
                "sell_date": trade.get("sell_date"),
                "profit": trade.get("profit", 0),
                "exit_reason": trade.get("exit_reason"),
            })

    print()
    print("=" * 95)
    print("DIRECT EXTRACTION SUMMARY")
    print("=" * 95)

    print(
        f"Total Trades : {len(all_trades)}"
    )

    total_profit = sum(
        trade["profit"]
        for trade in all_trades
    )

    print(
        f"Total Profit : {total_profit:.2f}"
    )

    print()
    print("=" * 95)
    print("ALL 2024 TRADES")
    print("=" * 95)

    for trade in all_trades:

        print(
            f"{trade['symbol']:<16} | "
            f"{str(trade['buy_date']):<12} | "
            f"{str(trade['sell_date']):<12} | "
            f"Profit={trade['profit']:>10.2f} | "
            f"Exit={trade['exit_reason']}"
        )

    print()
    print("=" * 95)
    print("RECONCILIATION TARGET")
    print("=" * 95)

    print(
        "Walk-forward validation:"
    )

    print(
        "Trades = 147"
    )

    print(
        "Profit = 743.82"
    )

    print()

    print(
        "Direct extraction:"
    )

    print(
        f"Trades = {len(all_trades)}"
    )

    print(
        f"Profit = {total_profit:.2f}"
    )

    print()

    print(
        "Trade count difference : "
        f"{len(all_trades) - 147}"
    )

    print(
        "Profit difference      : "
        f"{total_profit - 743.82:.2f}"
    )

    print()
    print("=" * 95)
    print("2024 TRADE RECONCILIATION DIAGNOSTIC COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    run()