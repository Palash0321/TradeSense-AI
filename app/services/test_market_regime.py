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


configs = [
    {
        'name': 'BASELINE',
        'use_market_regime': False
    },
    {
        'name': 'MARKET REGIME',
        'use_market_regime': True
    }
]


for config in configs:

    print()
    print("=" * 80)
    print(config['name'])
    print("=" * 80)

    profits = []
    trades = []
    returns = []
    drawdowns = []
    sharpes = []

    profitable_stocks = 0
    valid_stocks = 0

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

            use_market_regime=config['use_market_regime'],

            initial_stop_atr=3.0
        )

        testing = result['windows'][1]['testing']

        if testing is None:
            continue

        valid_stocks += 1

        profits.append(testing['net_profit'])
        trades.append(testing['total_trades'])
        returns.append(testing['total_return'])
        drawdowns.append(testing['max_drawdown'])
        sharpes.append(testing['sharpe_ratio'])

        if testing['net_profit'] > 0:
            profitable_stocks += 1

    if valid_stocks == 0:
        print("No valid stocks.")
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

    print()
    print(f"Stocks           : {valid_stocks}")
    print(f"Total Profit     : {total_profit:.2f}")
    print(f"Total Trades     : {total_trades}")
    print(
        f"Profitable Stocks: "
        f"{profitable_stocks}/{valid_stocks}"
    )
    print(f"Positive Ratio   : {positive_ratio:.2f}%")
    print(f"Average Return   : {average_return:.2f}%")
    print(f"Average DD       : {average_dd:.2f}%")
    print(f"Average Sharpe   : {average_sharpe:.2f}")


print()
print("=" * 80)
print("MARKET REGIME TEST COMPLETE")
print("=" * 80)