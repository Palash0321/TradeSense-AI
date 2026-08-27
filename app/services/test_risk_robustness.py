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


risk_values = [
    0.0075,
    0.0100,
    0.0125,
    0.0150
]


windows = [
    {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31"
    },
    {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31"
    }
]


print("=" * 95)
print("RISK PER TRADE — ROBUSTNESS COMPARISON")
print("=" * 95)


for risk_per_trade in risk_values:

    print()
    print("-" * 95)
    print(
        f"TESTING RISK PER TRADE = "
        f"{risk_per_trade * 100:.2f}%"
    )
    print("-" * 95)

    returns = []
    profit_factors = []
    drawdowns = []
    profits = []

    profitable_stocks = 0
    valid_stocks = 0

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

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

            initial_stop_atr=3.5,

            risk_per_trade=risk_per_trade
        )

        valid_tests = [
            window["testing"]
            for window in result["windows"]
            if window["testing"] is not None
        ]

        if not valid_tests:
            continue

        for testing in valid_tests:

            valid_stocks += 1

            profit = testing["net_profit"]

            profits.append(profit)

            returns.append(
                testing["total_return"]
            )

            profit_factors.append(
                testing["profit_factor"]
            )

            drawdowns.append(
                testing["max_drawdown"]
            )

        # Count stock as profitable if its
        # aggregate out-of-sample profit
        # is positive.

        stock_profit = sum(
            testing["net_profit"]
            for testing in valid_tests
        )

        if stock_profit > 0:
            profitable_stocks += 1

    if valid_stocks == 0:
        continue

    average_return = (
        sum(returns)
        / valid_stocks
    )

    sorted_returns = sorted(returns)

    median_return = sorted_returns[
        len(sorted_returns) // 2
    ]

    finite_profit_factors = [
        pf
        for pf in profit_factors
        if pf != float("inf")
        and pf != float("-inf")
    ]

    if finite_profit_factors:

        average_pf = (
            sum(finite_profit_factors)
            / len(finite_profit_factors)
        )

    else:

        average_pf = 0

    average_dd = (
        sum(drawdowns)
        / valid_stocks
    )

    total_profit = sum(profits)

    positive_ratio = (
        profitable_stocks
        / len(symbols)
        * 100
    )

    # =================================
    # Robustness score
    # =================================

    if (
        average_dd > 0
        and average_pf > 0
    ):

        robustness_score = (
            average_return
            * average_pf
            / average_dd
        )

    else:

        robustness_score = 0

    print()

    print(
        f"Risk Per Trade      : "
        f"{risk_per_trade * 100:.2f}%"
    )

    print(
        f"Average Return      : "
        f"{average_return:.2f}%"
    )

    print(
        f"Median Return       : "
        f"{median_return:.2f}%"
    )

    print(
        f"Average Profit Factor: "
        f"{average_pf:.2f}"
    )

    print(
        f"Average Drawdown    : "
        f"{average_dd:.2f}%"
    )

    print(
        f"Profitable Stocks   : "
        f"{profitable_stocks}/{len(symbols)}"
    )

    print(
        f"Positive Ratio      : "
        f"{positive_ratio:.2f}%"
    )

    print(
        f"Total Profit        : "
        f"{total_profit:.2f}"
    )

    print(
        f"Robustness Score    : "
        f"{robustness_score:.4f}"
    )


print()
print("=" * 95)
print("RISK ROBUSTNESS COMPARISON COMPLETE")
print("=" * 95)