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


def run_year(
    train_start,
    train_end,
    test_start,
    test_end
):

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
            windows=[
                {
                    "train_start":
                        train_start,

                    "train_end":
                        train_end,

                    "test_start":
                        test_start,

                    "test_end":
                        test_end,
                }
            ],

            adx_values=[15],

            ema_gap_values=[0.0],

            trailing_atr_values=[2.5],

            trailing_activation_atr=2.5,

            momentum_min=None,

            use_market_regime=False,

            initial_stop_atr=3.5,

            risk_per_trade=0.01,
        )

        window_result = (
            result["windows"][0]
        )

        testing = (
            window_result["testing"]
        )

        if testing is None:

            excluded.append({
                "symbol":
                    symbol,

                "reason":
                    window_result.get(
                        "exclusion_reason",
                        "UNKNOWN"
                    )
            })

            continue

        # ---------------------------------
        # Obtain actual OOS trades using
        # the same selected parameters.
        # ---------------------------------

        trade_result = (
            engine.test_period(
                start_date=test_start,
                end_date=test_end,
                parameters=window_result[
                    "best_parameters"
                ],
                use_market_regime=False,
                initial_stop_atr=3.5,
                risk_per_trade=0.01,
                return_trades=True
            )
        )

        all_trades.extend(
            trade_result.get(
                "trades",
                []
            )
        )

    return all_trades, excluded


print("=" * 95)
print("MONTHLY PERFORMANCE — FROZEN CONFIGURATION")
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


for year, train_start, train_end, test_start, test_end in [

    (
        2024,
        "2021-01-01",
        "2023-12-31",
        "2024-01-01",
        "2024-12-31",
    ),

    (
        2025,
        "2021-01-01",
        "2024-12-31",
        "2025-01-01",
        "2025-12-31",
    ),

]:

    print()
    print("=" * 95)
    print(f"MONTHLY ANALYSIS — {year}")
    print("=" * 95)

    trades, excluded_stocks = run_year(
        train_start,
        train_end,
        test_start,
        test_end
    )

    print()
    print(
        f"Total Universe      : "
        f"{len(symbols)}"
    )

    print(
        f"Validated Stocks    : "
        f"{len(symbols) - len(excluded_stocks)}"
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

    monthly_profit = defaultdict(float)
    monthly_trades = defaultdict(int)

    for trade in trades:

        exit_date = trade.get("sell_date")

        if exit_date is None:

            exit_date = trade.get(
                "exit_date"
            )

        if exit_date is None:
            continue

        if hasattr(
            exit_date,
            "strftime"
        ):

            month = exit_date.strftime(
                "%Y-%m"
            )

        else:

            month = str(
                exit_date
            )[:7]

        monthly_profit[month] += float(
            trade.get("profit", 0)
        )

        monthly_trades[month] += 1

    print()
    print(
        "Month      | Profit      | Trades"
    )

    print("-" * 45)

    months = sorted(
        monthly_profit.keys()
    )

    cumulative_profit = 0.0
    peak_profit = 0.0
    max_drawdown = 0.0

    positive_months = 0
    negative_months = 0
    zero_months = 0

    for month in months:

        profit = monthly_profit[month]

        trade_count = monthly_trades[month]

        cumulative_profit += profit

        if cumulative_profit > peak_profit:

            peak_profit = cumulative_profit

        drawdown = (
            peak_profit
            - cumulative_profit
        )

        if drawdown > max_drawdown:

            max_drawdown = drawdown

        if profit > 0:

            positive_months += 1

        elif profit < 0:

            negative_months += 1

        else:

            zero_months += 1

        print(
            f"{month}     | "
            f"{profit:>10.2f} | "
            f"{trade_count:>6}"
        )

    print()

    print(
        f"Total Trades          : "
        f"{len(trades)}"
    )

    print(
        f"Positive Months       : "
        f"{positive_months}"
    )

    print(
        f"Negative Months       : "
        f"{negative_months}"
    )

    print(
        f"Zero Months           : "
        f"{zero_months}"
    )

    print(
        f"Maximum Monthly DD    : "
        f"{max_drawdown:.2f}"
    )

    print(
        f"Final Cumulative P&L  : "
        f"{cumulative_profit:.2f}"
    )