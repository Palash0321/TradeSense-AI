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


def run_year(test_index):

    results = []

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

        testing = (
            result["windows"][test_index]["testing"]
        )
        if testing is None:

            excluded.append({
                "symbol": symbol,
                "reason":
                    result["windows"][test_index].get(
                        "exclusion_reason",
                        "UNKNOWN"
                    )
            })

            continue

        results.append(testing)

    return results, excluded


print("=" * 95)
print("PORTFOLIO-LEVEL VALIDATION — FROZEN CONFIGURATION")
print("=" * 95)

print()
print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Risk Per Trade       : 1.00%")
print("Momentum             : OFF")
print("Market Regime        : OFF")


for year, test_index in [
    (2024, 0),
    (2025, 1),
]:

    print()
    print("=" * 95)
    print(f"OUT-OF-SAMPLE PORTFOLIO — {year}")
    print("=" * 95)

    testing_results, excluded_stocks = (
        run_year(test_index)
    )

    if not testing_results:
        print("No valid results.")
        continue

    print()
    print(
        f"Total Universe      : "
        f"{len(symbols)}"
    )

    print(
        f"Validated Stocks    : "
        f"{len(testing_results)}"
    )

    print(
        f"Excluded Stocks     : "
        f"{len(excluded_stocks)}"
    )

    if excluded_stocks:

        print()
        print("Excluded Stock Details")
        print("-" * 60)

        for item in excluded_stocks:

            print(
                f"{item['symbol']:<20} | "
                f"{item['reason']}"
            )

    total_profit = sum(
        r["net_profit"]
        for r in testing_results
    )

    total_trades = sum(
        r["total_trades"]
        for r in testing_results
    )

    profitable_stocks = sum(
        1
        for r in testing_results
        if r["net_profit"] > 0
    )

    total_stocks = len(testing_results)

    average_return = (
        sum(
            r["total_return"]
            for r in testing_results
        )
        / total_stocks
    )

    average_dd = (
        sum(
            r["max_drawdown"]
            for r in testing_results
        )
        / total_stocks
    )

    average_sharpe = (
        sum(
            r["sharpe_ratio"]
            for r in testing_results
        )
        / total_stocks
    )

    returns = sorted(
        r["total_return"]
        for r in testing_results
    )

    median_return = returns[
        len(returns) // 2
    ]

    print()
    print(
        f"Stocks              : "
        f"{total_stocks}"
    )

    print(
        f"Total Profit        : "
        f"{total_profit:.2f}"
    )

    print(
        f"Total Trades        : "
        f"{total_trades}"
    )

    print(
        f"Profitable Stocks   : "
        f"{profitable_stocks}/{total_stocks}"
    )

    print(
        f"Positive Ratio      : "
        f"{(
            profitable_stocks
            / total_stocks
            * 100
        ):.2f}%"
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
        f"Average Drawdown    : "
        f"{average_dd:.2f}%"
    )

    print(
        f"Average Sharpe      : "
        f"{average_sharpe:.2f}"
    )


print()
print("=" * 95)
print("PORTFOLIO VALIDATION COMPLETE")
print("=" * 95)