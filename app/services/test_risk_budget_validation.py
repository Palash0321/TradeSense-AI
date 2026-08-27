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
    0.0125
]


years = [
    {
        "year": 2024,
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31"
    },
    {
        "year": 2025,
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31"
    }
]


print("=" * 105)
print("RISK PER TRADE — 2024 vs 2025 VALIDATION")
print("=" * 105)
print()

print(
    "Risk | Year | Profit | Trades | Profitable | "
    "Pos% | Avg Return | Median Return | Avg DD | Avg Sharpe"
)

print("-" * 105)


for risk_per_trade in risk_values:

    for year_config in years:

        year = year_config["year"]

        profits = []
        trades = []
        returns = []
        drawdowns = []
        sharpes = []

        profitable_stocks = 0
        valid_stocks = 0

        print(
            f"Running Risk={risk_per_trade * 100:.2f}% | "
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
                windows=[
                    {
                        "train_start":
                            year_config["train_start"],

                        "train_end":
                            year_config["train_end"],

                        "test_start":
                            year_config["test_start"],

                        "test_end":
                            year_config["test_end"]
                    }
                ],

                adx_values=[15],

                ema_gap_values=[0.0],

                trailing_atr_values=[2.5],

                trailing_activation_atr=2.5,

                momentum_min=None,

                use_market_regime=False,

                initial_stop_atr=3.5,

                risk_per_trade=risk_per_trade
            )

            testing = (
                result["windows"][0]["testing"]
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

        sorted_returns = sorted(returns)

        n = len(sorted_returns)

        if n % 2 == 1:

            median_return = (
                sorted_returns[n // 2]
            )

        else:

            median_return = (
                (
                    sorted_returns[
                        n // 2 - 1
                    ]
                    +
                    sorted_returns[
                        n // 2
                    ]
                )
                / 2
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
            f"{risk_per_trade * 100:>5.2f}% | "
            f"{year} | "
            f"{total_profit:>8.2f} | "
            f"{total_trades:>6} | "
            f"{profitable_stocks:>2}/{valid_stocks:<2} | "
            f"{positive_ratio:>5.2f}% | "
            f"{average_return:>9.2f}% | "
            f"{median_return:>13.2f}% | "
            f"{average_dd:>6.2f}% | "
            f"{average_sharpe:>9.2f}"
        )


print()
print("=" * 105)
print("RISK BUDGET VALIDATION COMPLETE")
print("=" * 105)