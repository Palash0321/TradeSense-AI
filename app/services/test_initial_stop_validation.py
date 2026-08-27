from app.services.walk_forward_engine import WalkForwardEngine


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


windows = [
    {
        'train_start': '2021-01-01',
        'train_end': '2023-12-31',
        'test_start': '2024-01-01',
        'test_end': '2024-12-31'
    },
    {
        'train_start': '2021-01-01',
        'train_end': '2024-12-31',
        'test_start': '2025-01-01',
        'test_end': '2025-12-31'
    }
]


initial_stop_values = [
    2.5,
    3.0,
    3.5
]


print("=" * 95)
print("INITIAL STOP ATR — 2024 vs 2025 VALIDATION")
print("=" * 95)
print()

print(
    "Stop ATR | Year | Profit | Trades | Profitable | "
    "Pos% | Avg Return | Avg DD | Avg Sharpe"
)

print("-" * 95)


for initial_stop_atr in initial_stop_values:

    for window_index, year in [
        (0, 2024),
        (1, 2025)
    ]:

        profits = []
        trades = []
        returns = []
        drawdowns = []
        sharpes = []

        profitable_stocks = 0
        valid_stocks = 0

        print(
            f"Running Stop ATR={initial_stop_atr} | "
            f"Year={year}...",
            flush=True
        )

        for symbol in symbols:

            engine = WalkForwardEngine(
                symbol,
                brokerage=20,
                slippage=0.10
            )

            result = engine.run(
                windows=windows,

                adx_values=[15],

                ema_gap_values=[0.0],

                trailing_atr_values=[2.5],

                trailing_activation_atr=2.5,

                momentum_min=None,

                use_market_regime=False,

                initial_stop_atr=initial_stop_atr
            )

            testing = (
                result["windows"][window_index]["testing"]
            )

            if testing is None:
                continue

            valid_stocks += 1

            profit = testing["net_profit"]

            profits.append(profit)

            trades.append(
                testing["total_trades"]
            )

            returns.append(
                testing["total_return"]
            )

            drawdowns.append(
                testing["max_drawdown"]
            )

            sharpes.append(
                testing["sharpe_ratio"]
            )

            if profit > 0:
                profitable_stocks += 1

        if valid_stocks == 0:
            continue

        total_profit = sum(profits)

        total_trades = sum(trades)

        positive_ratio = (
            profitable_stocks
            / valid_stocks
            * 100
        )

        average_return = (
            sum(returns)
            / valid_stocks
        )

        average_dd = (
            sum(drawdowns)
            / valid_stocks
        )

        average_sharpe = (
            sum(sharpes)
            / valid_stocks
        )

        print(
            f"{initial_stop_atr:>8.1f} | "
            f"{year} | "
            f"{total_profit:>8.2f} | "
            f"{total_trades:>6} | "
            f"{profitable_stocks:>2}/{valid_stocks:<2} | "
            f"{positive_ratio:>5.2f}% | "
            f"{average_return:>8.2f}% | "
            f"{average_dd:>6.2f}% | "
            f"{average_sharpe:>8.2f}"
        )


print()
print("=" * 95)
print("INITIAL STOP VALIDATION COMPLETE")
print("=" * 95)