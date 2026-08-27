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


CONFIG = {
    "adx_min": 15,
    "ema_gap_min": 0.0,
    "trailing_atr": 2.5,
    "trailing_activation_atr": 2.5,
    "initial_stop_atr": 3.0,
    "momentum_min": None,
    "use_market_regime": False,
}


START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


all_trades = []


print()
print("=" * 90)
print("R-MULTIPLE TRADE DIAGNOSTIC — 2025 HOLDOUT")
print("=" * 90)

print()
print("Configuration:")
print(f"ADX              : {CONFIG['adx_min']}")
print(f"EMA Gap          : {CONFIG['ema_gap_min']}")
print(f"Trailing ATR     : {CONFIG['trailing_atr']}")
print(f"Activation ATR   : {CONFIG['trailing_activation_atr']}")
print(f"Initial Stop ATR : {CONFIG['initial_stop_atr']}")
print("Momentum         : OFF")
print("Market Regime    : OFF")

print()
print("Running trade extraction...")


for symbol in symbols:

    service = BacktestService(
        symbol=symbol,
        strategy="ema_atr",
        brokerage=20,
        slippage=0.10,
        initial_capital=100000,
    )

    metrics = service.performance_metrics(
        start_date=START_DATE,
        end_date=END_DATE,
        adx_min=CONFIG["adx_min"],
        ema_gap_min=CONFIG["ema_gap_min"],
        trailing_atr=CONFIG["trailing_atr"],
        trailing_activation_atr=CONFIG["trailing_activation_atr"],
        momentum_min=CONFIG["momentum_min"],
        use_market_regime=CONFIG["use_market_regime"],
        initial_stop_atr=CONFIG["initial_stop_atr"],
    )

    for trade in metrics["trades"]:

        trade_copy = dict(trade)

        trade_copy["symbol"] = symbol

        all_trades.append(trade_copy)


print()
print("=" * 90)
print("TRADE SUMMARY")
print("=" * 90)

total_trades = len(all_trades)

print(f"Total Trades : {total_trades}")


if total_trades == 0:

    print("No trades found.")
    raise SystemExit


# --------------------------------------------------
# R MULTIPLE DISTRIBUTION
# --------------------------------------------------

r_values = [
    float(t["r_multiple"])
    for t in all_trades
    if t.get("r_multiple") is not None
]


print()
print("=" * 90)
print("R-MULTIPLE DISTRIBUTION")
print("=" * 90)

print(f"Trades with R data : {len(r_values)}")


if r_values:

    print(
        f"Average R         : "
        f"{sum(r_values) / len(r_values):.2f}"
    )

    print(
        f"Best R            : "
        f"{max(r_values):.2f}"
    )

    print(
        f"Worst R           : "
        f"{min(r_values):.2f}"
    )

    buckets = {
        "< -1R": 0,
        "-1R to 0R": 0,
        "0R to +1R": 0,
        "+1R to +2R": 0,
        "+2R to +3R": 0,
        "> +3R": 0,
    }

    for r in r_values:

        if r < -1:
            buckets["< -1R"] += 1

        elif r < 0:
            buckets["-1R to 0R"] += 1

        elif r < 1:
            buckets["0R to +1R"] += 1

        elif r < 2:
            buckets["+1R to +2R"] += 1

        elif r < 3:
            buckets["+2R to +3R"] += 1

        else:
            buckets["> +3R"] += 1


    print()

    for bucket, count in buckets.items():

        percentage = (
            count
            / len(r_values)
            * 100
        )

        print(
            f"{bucket:>12} : "
            f"{count:>4} "
            f"({percentage:>6.2f}%)"
        )


# --------------------------------------------------
# EXIT REASON ANALYSIS
# --------------------------------------------------

print()
print("=" * 90)
print("EXIT REASON ANALYSIS")
print("=" * 90)


exit_reasons = {}


for trade in all_trades:

    reason = trade.get(
        "exit_reason",
        "UNKNOWN"
    )

    if reason not in exit_reasons:

        exit_reasons[reason] = []

    exit_reasons[reason].append(trade)


for reason, trades in sorted(
    exit_reasons.items(),
    key=lambda x: len(x[1]),
    reverse=True,
):

    profits = [
        float(t["profit"])
        for t in trades
    ]

    rs = [
        float(t["r_multiple"])
        for t in trades
        if t.get("r_multiple") is not None
    ]

    wins = sum(
        p > 0
        for p in profits
    )

    total_profit = sum(profits)

    average_profit = (
        total_profit
        / len(profits)
    )

    average_r = (
        sum(rs) / len(rs)
        if rs
        else 0
    )

    win_rate = (
        wins
        / len(profits)
        * 100
    )

    print()
    print(f"Exit Reason : {reason}")
    print(f"Trades      : {len(trades)}")
    print(f"Wins        : {wins}")
    print(f"Win Rate    : {win_rate:.2f}%")
    print(f"Profit      : {total_profit:.2f}")
    print(f"Avg Profit  : {average_profit:.2f}")
    print(f"Avg R       : {average_r:.2f}")


# --------------------------------------------------
# EXTREME WINNERS / LOSERS
# --------------------------------------------------

print()
print("=" * 90)
print("BEST / WORST R-MULTIPLE TRADES")
print("=" * 90)


sorted_trades = sorted(
    all_trades,
    key=lambda t: float(t.get("r_multiple", 0))
)


print()
print("WORST 10:")

for trade in sorted_trades[:10]:

    print(
        f"{trade['symbol']:16} | "
        f"R={float(trade.get('r_multiple', 0)):>6.2f} | "
        f"Profit={float(trade['profit']):>9.2f} | "
        f"Exit={trade.get('exit_reason', 'UNKNOWN')}"
    )


print()
print("BEST 10:")

for trade in sorted_trades[-10:][::-1]:

    print(
        f"{trade['symbol']:16} | "
        f"R={float(trade.get('r_multiple', 0)):>6.2f} | "
        f"Profit={float(trade['profit']):>9.2f} | "
        f"Exit={trade.get('exit_reason', 'UNKNOWN')}"
    )


print()
print("=" * 90)
print("R-MULTIPLE DIAGNOSTIC COMPLETE")
print("=" * 90)