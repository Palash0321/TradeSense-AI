from app.services.strategy_robustness import StrategyRobustness


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
    "ADANIPORTS.NS"
]


robustness = StrategyRobustness(
    symbols,
    initial_capital=100000,
    brokerage=20,
    slippage=0.10
)


results = robustness.test_configuration(
    adx_min=15,
    ema_gap_min=0.0,
    trailing_atr=2.5,
    trailing_activation_atr=2.5,
    initial_stop_atr=3.0,
    momentum_min=None,
    use_market_regime=False
)


score = robustness.score_configuration(results)


print()
print("=== ROBUSTNESS — SELECTED CONFIGURATION ===")
print()
print("ADX              :", 15)
print("EMA Gap          :", 0.0)
print("Trailing ATR     :", 2.5)
print("Activation ATR   :", 2.5)
print("Initial Stop ATR :", 3.0)
print("Momentum         :", None)
print("Market Regime    :", False)
print()

print("=== AGGREGATE ===")
print("Average Return   :", score["average_return"], "%")
print("Median Return    :", score["median_return"], "%")
print("Average PF       :", score["average_profit_factor"])
print("Average DD       :", score["average_drawdown"], "%")
print(
    "Profitable Stocks:",
    score["profitable_stocks"],
    "/",
    score["total_stocks"]
)
print(
    "Positive Ratio   :",
    score["positive_stock_ratio"],
    "%"
)
print(
    "Robustness Score :",
    score["robustness_score"]
)

print()
print("=== STOCK RESULTS ===")

for r in results:
    print(
        r["symbol"],
        "| Profit:",
        round(r["net_profit"], 2),
        "| Return:",
        round(r["total_return"], 2),
        "%",
        "| Trades:",
        r["total_trades"],
        "| PF:",
        r["profit_factor"],
        "| DD:",
        round(r["max_drawdown"], 2),
        "%",
        "| Sharpe:",
        round(r["sharpe_ratio"], 2)
    )