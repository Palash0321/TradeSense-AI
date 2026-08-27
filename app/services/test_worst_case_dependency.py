from collections import defaultdict

from app.services.backtest_service import BacktestService


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


def get_trades(
    train_start,
    train_end,
    test_start,
    test_end,
):

    all_trades = []
    excluded = []

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

        service = BacktestService(
            symbol=symbol,
            strategy="ema_atr",
            initial_capital=100000,
            brokerage=20,
            slippage=0.10,
        )

        result = service.run_backtest_v3(
            start_date=test_start,
            end_date=test_end,
            ema_gap_min=0.0,
            adx_min=15,
            trailing_atr=2.5,
            trailing_activation_atr=2.5,
            momentum_min=None,
            use_market_regime=False,
            initial_stop_atr=3.5,
            risk_per_trade=0.01,
        )

        trades = result.get(
            "trades",
            []
        )

        if not trades:
            excluded.append(symbol)
            continue

        # =================================
        # Attach symbol to every trade
        # =================================

        for trade in trades:

            trade["symbol"] = symbol

        all_trades.extend(
            trades
        )

    return all_trades, excluded


def calculate_metrics(trades):

    profits = [
        float(
            trade.get("profit", 0)
        )
        for trade in trades
    ]

    if not profits:
        return {
            "trades": 0,
            "net_profit": 0,
            "gross_profit": 0,
            "gross_loss": 0,
            "profit_factor": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "average_win": 0,
            "average_loss": 0,
        }

    winning = [
        p for p in profits
        if p > 0
    ]

    losing = [
        p for p in profits
        if p < 0
    ]

    gross_profit = sum(
        winning
    )

    gross_loss = abs(
        sum(losing)
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = float("inf")

    average_win = (
        gross_profit / len(winning)
        if winning
        else 0
    )

    average_loss = (
        gross_loss / len(losing)
        if losing
        else 0
    )

    return {
        "trades": len(profits),

        "net_profit":
            sum(profits),

        "gross_profit":
            gross_profit,

        "gross_loss":
            gross_loss,

        "profit_factor":
            profit_factor,

        "wins":
            len(winning),

        "losses":
            len(losing),

        "win_rate":
            (
                len(winning)
                / len(profits)
                * 100
            ),

        "average_win":
            average_win,

        "average_loss":
            average_loss,
    }


def print_metrics(
    label,
    trades
):

    metrics = calculate_metrics(
        trades
    )

    pf = metrics[
        "profit_factor"
    ]

    if pf == float("inf"):

        pf_text = "INF"

    else:

        pf_text = f"{pf:.2f}"

    print()
    print(label)
    print("-" * 70)

    print(
        f"Trades              : "
        f"{metrics['trades']}"
    )

    print(
        f"Net Profit          : "
        f"{metrics['net_profit']:.2f}"
    )

    print(
        f"Gross Profit        : "
        f"{metrics['gross_profit']:.2f}"
    )

    print(
        f"Gross Loss          : "
        f"{metrics['gross_loss']:.2f}"
    )

    print(
        f"Profit Factor       : "
        f"{pf_text}"
    )

    print(
        f"Wins                : "
        f"{metrics['wins']}"
    )

    print(
        f"Losses              : "
        f"{metrics['losses']}"
    )

    print(
        f"Win Rate            : "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"Average Win         : "
        f"{metrics['average_win']:.2f}"
    )

    print(
        f"Average Loss        : "
        f"{metrics['average_loss']:.2f}"
    )


def remove_top_stocks(
    trades,
    count
):

    stock_profit = defaultdict(
        float
    )

    for trade in trades:

        symbol = (
            trade.get("symbol")
            or trade.get("ticker")
        )

        if symbol is None:
            continue

        stock_profit[symbol] += (
            float(
                trade.get("profit", 0)
            )
        )

    ranked = sorted(
        stock_profit.items(),
        key=lambda x: x[1],
        reverse=True
    )

    excluded_symbols = {
        symbol
        for symbol, profit
        in ranked[:count]
    }

    remaining = [
        trade
        for trade in trades
        if (
            trade.get("symbol")
            or trade.get("ticker")
        ) not in excluded_symbols
    ]

    return (
        remaining,
        ranked[:count]
    )


def remove_top_trades(
    trades,
    count
):

    ranked = sorted(
        trades,
        key=lambda trade:
            float(
                trade.get(
                    "profit",
                    0
                )
            ),
        reverse=True
    )

    return (
        ranked[count:],
        ranked[:count]
    )


def run_year(
    year,
    test_start,
    test_end
):

    print()
    print("=" * 100)
    print(
        f"WORST-CASE DEPENDENCY — {year}"
    )
    print("=" * 100)

    trades, excluded = get_trades(
        test_start,
        test_start,
        test_start,
        test_end,
    )

    print()
    print(
        f"Total Universe      : "
        f"{len(symbols)}"
    )

    print(
        f"Stocks with Trades  : "
        f"{len(symbols) - len(excluded)}"
    )

    print(
        f"Stocks without Trades: "
        f"{len(excluded)}"
    )

    print_metrics(
        "FULL PORTFOLIO",
        trades
    )

    # =================================
    # Remove top stocks
    # =================================

    print()
    print("=" * 100)
    print("STOCK DEPENDENCY TEST")
    print("=" * 100)

    for count in [1, 3, 5]:

        remaining, removed = (
            remove_top_stocks(
                trades,
                count
            )
        )

        print()
        print(
            f"REMOVE TOP {count} "
            f"PROFITABLE STOCK(S)"
        )

        print(
            "Removed:"
        )

        for symbol, profit in removed:

            print(
                f"  {symbol:<20}"
                f" {profit:>12.2f}"
            )

        print_metrics(
            f"AFTER REMOVING TOP {count} STOCK(S)",
            remaining
        )

    # =================================
    # Remove top individual trades
    # =================================

    print()
    print("=" * 100)
    print("INDIVIDUAL TRADE DEPENDENCY TEST")
    print("=" * 100)

    for count in [1, 3, 5]:

        remaining, removed = (
            remove_top_trades(
                trades,
                count
            )
        )

        print()
        print(
            f"REMOVE TOP {count} "
            f"WINNING TRADE(S)"
        )

        for rank, trade in enumerate(
            removed,
            start=1
        ):

            symbol = (
                trade.get("symbol")
                or trade.get("ticker")
                or "UNKNOWN"
            )

            profit = float(
                trade.get(
                    "profit",
                    0
                )
            )

            return_pct = float(
                trade.get(
                    "return_percent",
                    0
                )
            )

            print(
                f"  #{rank} "
                f"{symbol:<20}"
                f" Profit: {profit:>10.2f}"
                f" Return: {return_pct:>8.2f}%"
            )

        print_metrics(
            f"AFTER REMOVING TOP {count} TRADE(S)",
            remaining
        )


print("=" * 100)
print(
    "WORST-CASE DEPENDENCY ANALYSIS"
)
print("=" * 100)

print()
print("FROZEN CONFIGURATION")
print("-" * 70)

print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Risk Per Trade       : 1.00%")
print("Momentum             : OFF")
print("Market Regime        : OFF")


run_year(
    2024,
    "2024-01-01",
    "2024-12-31"
)

run_year(
    2025,
    "2025-01-01",
    "2025-12-31"
)

print()
print("=" * 100)
print(
    "WORST-CASE DEPENDENCY ANALYSIS COMPLETE"
)
print("=" * 100)