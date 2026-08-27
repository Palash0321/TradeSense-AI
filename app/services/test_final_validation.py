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


windows = [
    {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
]


def run_period(test_index):

    profits = []
    trades = []
    returns = []
    drawdowns = []
    sharpes = []

    profitable = 0
    valid = 0

    for symbol in symbols:

        engine = WalkForwardEngine(
            symbol,
            brokerage=20,
            slippage=0.10,
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
            risk_per_trade=0.01,
        )

        testing = result["windows"][test_index]["testing"]

        if testing is None:
            continue

        valid += 1

        profit = testing["net_profit"]

        profits.append(profit)
        trades.append(testing["total_trades"])
        returns.append(testing["total_return"])
        drawdowns.append(testing["max_drawdown"])
        sharpes.append(testing["sharpe_ratio"])

        if profit > 0:
            profitable += 1

    return {
        "profit": sum(profits),
        "trades": sum(trades),
        "profitable": profitable,
        "stocks": valid,
        "positive_ratio": (
            profitable / valid * 100
            if valid
            else 0
        ),
        "average_return": (
            sum(returns) / valid
            if valid
            else 0
        ),
        "average_dd": (
            sum(drawdowns) / valid
            if valid
            else 0
        ),
        "average_sharpe": (
            sum(sharpes) / valid
            if valid
            else 0
        ),
    }


print()
print("=" * 80)
print("FINAL FROZEN CONFIGURATION VALIDATION")
print("=" * 80)

print()
print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Momentum             : OFF")
print("Market Regime        : OFF")
print()

period_2024 = run_period(0)
period_2025 = run_period(1)

for year, data in [
    (2024, period_2024),
    (2025, period_2025),
]:

    print("=" * 80)
    print(f"OUT-OF-SAMPLE {year}")
    print("=" * 80)

    print(
        f"Stocks           : {data['stocks']}"
    )

    print(
        f"Total Profit     : "
        f"{data['profit']:.2f}"
    )

    print(
        f"Total Trades     : "
        f"{data['trades']}"
    )

    print(
        f"Profitable Stocks: "
        f"{data['profitable']}/{data['stocks']}"
    )

    print(
        f"Positive Ratio   : "
        f"{data['positive_ratio']:.2f}%"
    )

    print(
        f"Average Return   : "
        f"{data['average_return']:.2f}%"
    )

    print(
        f"Average DD       : "
        f"{data['average_dd']:.2f}%"
    )

    print(
        f"Average Sharpe   : "
        f"{data['average_sharpe']:.2f}"
    )

    print()


print("=" * 80)
print("FINAL VALIDATION COMPLETE")
print("=" * 80)