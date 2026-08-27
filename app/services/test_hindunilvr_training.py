from app.services.backtest_service import BacktestService


service = BacktestService(
    symbol="HINDUNILVR.NS",
    strategy="ema_atr",
    brokerage=20,
    slippage=0.10,
)


result = service.performance_metrics(
    start_date="2021-01-01",
    end_date="2023-12-31",
    ema_gap_min=0.0,
    adx_min=15,
    trailing_atr=2.5,
    trailing_activation_atr=2.5,
    momentum_min=None,
    use_market_regime=False,
    initial_stop_atr=3.5,
    risk_per_trade=0.01,
)


print("=" * 80)
print("HINDUNILVR.NS — TRAINING PERIOD DIAGNOSTIC")
print("=" * 80)

print()
print(f"Training Trades : {result['total_trades']}")
print(f"Training Profit : {result['net_profit']:.2f}")
print(f"Training Return : {result['total_return']:.2f}%")
print(f"Profit Factor   : {result['profit_factor']}")
print(f"Drawdown        : {result['max_drawdown']:.2f}%")
print(f"Sharpe          : {result['sharpe_ratio']:.2f}")

print()
print("=" * 80)
print("MINIMUM TRADE REQUIREMENT")
print("=" * 80)

print("Required : 10")
print(
    f"Actual   : {result['total_trades']}"
)

if result["total_trades"] >= 10:
    print("Status   : PASS")
else:
    print("Status   : FAIL — excluded by optimizer")

print()
print("=" * 80)