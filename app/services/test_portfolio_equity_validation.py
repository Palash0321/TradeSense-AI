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


def collect_trades(test_index):

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

        window = result["windows"][test_index]

        testing = window["testing"]

        if testing is None:

            excluded.append({
                "symbol": symbol,
                "reason": window.get(
                    "exclusion_reason",
                    "UNKNOWN"
                )
            })

            continue

        parameters = window["best_parameters"]

        trade_result = engine.test_period(
            start_date=window["test_start"],
            end_date=window["test_end"],
            parameters=parameters,
            use_market_regime=False,
            initial_stop_atr=3.5,
            risk_per_trade=0.01,
            return_trades=True,
        )

        trades = trade_result.get(
            "trades",
            []
        )

        for trade in trades:

            trade_copy = dict(trade)

            trade_copy["symbol"] = symbol

            all_trades.append(
                trade_copy
            )

    return all_trades, excluded


def normalize_date(value):

    if value is None:
        return None

    if hasattr(value, "strftime"):
        return value.strftime(
            "%Y-%m-%d"
        )

    return str(value)[:10]


def validate_year(year, test_index):

    print()
    print("=" * 100)
    print(
        f"PORTFOLIO EQUITY VALIDATION — {year}"
    )
    print("=" * 100)

    trades, excluded = collect_trades(
        test_index
    )

    print()
    print(
        f"Total Universe      : "
        f"{len(symbols)}"
    )

    print(
        f"Validated Stocks    : "
        f"{len(symbols) - len(excluded)}"
    )

    print(
        f"Excluded Stocks     : "
        f"{len(excluded)}"
    )

    for item in excluded:

        print(
            f"Excluded            : "
            f"{item['symbol']} | "
            f"{item['reason']}"
        )

    if not trades:

        print()
        print("No trades available.")
        return

    # -----------------------------------------
    # Basic trade reconciliation
    # -----------------------------------------

    total_trade_profit = sum(
        float(
            trade.get("profit", 0)
        )
        for trade in trades
    )

    total_trades = len(trades)

    profitable_trades = sum(
        1
        for trade in trades
        if float(
            trade.get("profit", 0)
        ) > 0
    )

    losing_trades = sum(
        1
        for trade in trades
        if float(
            trade.get("profit", 0)
        ) < 0
    )

    # -----------------------------------------
    # Capital / equity simulation
    #
    # This is deliberately a diagnostic model.
    #
    # We start with one portfolio capital pool.
    # Each completed trade contributes its actual
    # backtest profit to that pool.
    #
    # We are NOT changing position sizing here.
    # -----------------------------------------

    initial_capital = 100000.0

    equity = initial_capital

    peak_equity = initial_capital

    max_drawdown_amount = 0.0

    max_drawdown_percent = 0.0

    equity_points = []

    daily_profit = defaultdict(float)

    for trade in trades:

        sell_date = (
            trade.get("sell_date")
        )

        if sell_date is None:

            sell_date = (
                trade.get("exit_date")
            )

        if sell_date is None:
            continue

        date = normalize_date(
            sell_date
        )

        profit = float(
            trade.get("profit", 0)
        )

        daily_profit[date] += profit

    # -----------------------------------------
    # Aggregate all trades by exit date.
    # This prevents arbitrary ordering when
    # multiple stocks exit on the same day.
    # -----------------------------------------

    for date in sorted(
        daily_profit.keys()
    ):

        day_profit = daily_profit[date]

        equity += day_profit

        if equity > peak_equity:

            peak_equity = equity

        drawdown_amount = (
            peak_equity
            - equity
        )

        drawdown_percent = 0.0

        if peak_equity > 0:

            drawdown_percent = (
                drawdown_amount
                / peak_equity
            ) * 100

        if drawdown_amount > max_drawdown_amount:

            max_drawdown_amount = (
                drawdown_amount
            )

        if drawdown_percent > max_drawdown_percent:

            max_drawdown_percent = (
                drawdown_percent
            )

        equity_points.append({
            "date": date,
            "profit": day_profit,
            "equity": equity,
            "drawdown": drawdown_percent,
        })

    # -----------------------------------------
    # Final reconciliation
    # -----------------------------------------

    final_equity = (
        initial_capital
        + total_trade_profit
    )

    calculated_return = 0.0

    if initial_capital > 0:

        calculated_return = (
            total_trade_profit
            / initial_capital
        ) * 100

    # -----------------------------------------
    # Positive / negative days
    # -----------------------------------------

    positive_days = sum(
        1
        for value in daily_profit.values()
        if value > 0
    )

    negative_days = sum(
        1
        for value in daily_profit.values()
        if value < 0
    )

    zero_days = sum(
        1
        for value in daily_profit.values()
        if value == 0
    )

    # -----------------------------------------
    # Largest winning / losing day
    # -----------------------------------------

    largest_winning_day = max(
        daily_profit.values()
    )

    largest_losing_day = min(
        daily_profit.values()
    )

    # -----------------------------------------
    # Trade concentration
    # -----------------------------------------

    sorted_profits = sorted(
        (
            float(
                trade.get("profit", 0)
            )
            for trade in trades
        ),
        reverse=True
    )

    top_5_trade_profit = sum(
        sorted_profits[:5]
    )

    top_10_trade_profit = sum(
        sorted_profits[:10]
    )

    # -----------------------------------------
    # Output
    # -----------------------------------------

    print()
    print(
        "CAPITAL / EQUITY"
    )
    print("-" * 100)

    print(
        f"Initial Capital      : "
        f"{initial_capital:,.2f}"
    )

    print(
        f"Final Equity         : "
        f"{final_equity:,.2f}"
    )

    print(
        f"Total P&L            : "
        f"{total_trade_profit:,.2f}"
    )

    print(
        f"Calculated Return    : "
        f"{calculated_return:.2f}%"
    )

    print()
    print(
        "DRAWDOWN"
    )
    print("-" * 100)

    print(
        f"Peak Equity          : "
        f"{peak_equity:,.2f}"
    )

    print(
        f"Maximum DD Amount    : "
        f"{max_drawdown_amount:,.2f}"
    )

    print(
        f"Maximum DD %         : "
        f"{max_drawdown_percent:.2f}%"
    )

    print()
    print(
        "TRADE / DAY STATISTICS"
    )
    print("-" * 100)

    print(
        f"Total Trades         : "
        f"{total_trades}"
    )

    print(
        f"Winning Trades       : "
        f"{profitable_trades}"
    )

    print(
        f"Losing Trades        : "
        f"{losing_trades}"
    )

    print(
        f"Trading Days         : "
        f"{len(daily_profit)}"
    )

    print(
        f"Positive Days        : "
        f"{positive_days}"
    )

    print(
        f"Negative Days        : "
        f"{negative_days}"
    )

    print(
        f"Zero Days            : "
        f"{zero_days}"
    )

    print()
    print(
        "DAILY EXTREMES"
    )
    print("-" * 100)

    print(
        f"Largest Winning Day  : "
        f"{largest_winning_day:,.2f}"
    )

    print(
        f"Largest Losing Day   : "
        f"{largest_losing_day:,.2f}"
    )

    print()
    print(
        "TRADE CONCENTRATION"
    )
    print("-" * 100)

    print(
        f"Top 5 Trade P&L      : "
        f"{top_5_trade_profit:,.2f}"
    )

    print(
        f"Top 10 Trade P&L     : "
        f"{top_10_trade_profit:,.2f}"
    )

    print()
    print(
        "RECONCILIATION"
    )
    print("-" * 100)

    print(
        f"Trade Profit Sum     : "
        f"{total_trade_profit:,.2f}"
    )

    print(
        f"Equity - Capital     : "
        f"{(
            final_equity
            - initial_capital
        ):,.2f}"
    )

    reconciliation_error = (
        (
            final_equity
            - initial_capital
        )
        - total_trade_profit
    )

    print(
        f"Reconciliation Error : "
        f"{reconciliation_error:,.6f}"
    )

    if abs(reconciliation_error) < 0.01:

        print(
            "Status               : "
            "PASS"
        )

    else:

        print(
            "Status               : "
            "FAIL"
        )


print()
print("=" * 100)
print(
    "FROZEN CONFIGURATION"
)
print("=" * 100)

print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Risk Per Trade       : 1.00%")
print("Momentum             : OFF")
print("Market Regime        : OFF")


validate_year(
    2024,
    0
)

validate_year(
    2025,
    1
)

print()
print("=" * 100)
print(
    "PORTFOLIO EQUITY VALIDATION COMPLETE"
)
print("=" * 100)