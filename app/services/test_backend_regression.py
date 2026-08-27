from app.services.backtest_service import BacktestService
from app.services.strategy_optimizer import StrategyOptimizer
from app.services.walk_forward_engine import WalkForwardEngine


SYMBOL = "RELIANCE.NS"

INITIAL_CAPITAL = 100000
BROKERAGE = 20
SLIPPAGE = 0.10

ADX = 15
EMA_GAP = 0.0
TRAILING_ATR = 2.5
ACTIVATION_ATR = 2.5
INITIAL_STOP_ATR = 3.5
RISK_PER_TRADE = 0.01
MOMENTUM = None
MARKET_REGIME = False


def assert_close(a, b, tolerance=0.000001):
    assert abs(a - b) <= tolerance, (
        f"Values differ: {a} != {b}"
    )


print("=" * 95)
print("TRADESENSE-AI — FINAL BACKEND REGRESSION GATE")
print("=" * 95)


# ============================================================
# TEST 1 — BACKTEST ENGINE
# ============================================================

print()
print("=" * 95)
print("TEST 1 — BACKTEST ENGINE")
print("=" * 95)

service = BacktestService(
    symbol=SYMBOL,
    strategy="ema_atr",
    initial_capital=INITIAL_CAPITAL,
    brokerage=BROKERAGE,
    slippage=SLIPPAGE,
)

backtest = service.run_backtest_v3(
    start_date="2024-01-01",
    end_date="2024-12-31",
    ema_gap_min=EMA_GAP,
    adx_min=ADX,
    trailing_atr=TRAILING_ATR,
    trailing_activation_atr=ACTIVATION_ATR,
    momentum_min=MOMENTUM,
    use_market_regime=MARKET_REGIME,
    initial_stop_atr=INITIAL_STOP_ATR,
    risk_per_trade=RISK_PER_TRADE,
)

trades = backtest.get("trades", [])

print(f"Trades                 : {len(trades)}")
print(
    f"Trade Profit           : "
    f"{sum(t.get('profit', 0) for t in trades):.2f}"
)

assert len(trades) >= 0

for trade in trades:
    assert trade.get("buy_date") is not None
    assert trade.get("sell_date") is not None
    assert trade.get("buy_price", 0) > 0
    assert trade.get("sell_price", 0) > 0
    assert trade.get("shares", 0) > 0

print("Status                 : PASS")


# ============================================================
# TEST 2 — PERFORMANCE METRICS
# ============================================================

print()
print("=" * 95)
print("TEST 2 — PERFORMANCE METRICS")
print("=" * 95)

metrics = service.performance_metrics(
    start_date="2024-01-01",
    end_date="2024-12-31",
    ema_gap_min=EMA_GAP,
    adx_min=ADX,
    trailing_atr=TRAILING_ATR,
    trailing_activation_atr=ACTIVATION_ATR,
    momentum_min=MOMENTUM,
    use_market_regime=MARKET_REGIME,
    initial_stop_atr=INITIAL_STOP_ATR,
    risk_per_trade=RISK_PER_TRADE,
)

print(
    f"Total Trades           : "
    f"{metrics['total_trades']}"
)

print(
    f"Net Profit             : "
    f"{metrics['net_profit']:.2f}"
)

print(
    f"Return                 : "
    f"{metrics['total_return']:.2f}%"
)

print(
    f"Profit Factor          : "
    f"{metrics['profit_factor']:.4f}"
)

print(
    f"Max Drawdown           : "
    f"{metrics['max_drawdown']:.2f}%"
)

assert metrics["total_trades"] == len(trades)

trade_profit = sum(
    t.get("profit", 0)
    for t in trades
)

assert_close(
    metrics["net_profit"],
    trade_profit,
)

print("Trade/Metrics Match    : PASS")


# ============================================================
# TEST 3 — OPTIMIZER
# ============================================================

print()
print("=" * 95)
print("TEST 3 — STRATEGY OPTIMIZER")
print("=" * 95)

optimizer = StrategyOptimizer(
    symbol=SYMBOL,
    initial_capital=INITIAL_CAPITAL,
    brokerage=BROKERAGE,
    slippage=SLIPPAGE,
)

optimization_results = optimizer.optimize(
    adx_values=[ADX],
    ema_gap_values=[EMA_GAP],
    trailing_atr_values=[TRAILING_ATR],
    start_date="2024-01-01",
    end_date="2024-12-31",
    trailing_activation_atr=ACTIVATION_ATR,
    momentum_min=MOMENTUM,
    use_market_regime=MARKET_REGIME,
    initial_stop_atr=INITIAL_STOP_ATR,
    risk_per_trade=RISK_PER_TRADE,
)

print(
    f"Optimization Results  : "
    f"{len(optimization_results)}"
)

assert len(optimization_results) > 0

ranked = optimizer.rank_results(
    optimization_results,
    min_trades=1,
)

print(
    f"Ranked Results        : "
    f"{len(ranked)}"
)

assert len(ranked) > 0

best = ranked[0]

assert best["adx_min"] == ADX
assert best["ema_gap_min"] == EMA_GAP
assert best["trailing_atr"] == TRAILING_ATR

print("Status                 : PASS")


# ============================================================
# TEST 4 — WALK-FORWARD ENGINE
# ============================================================

print()
print("=" * 95)
print("TEST 4 — WALK-FORWARD ENGINE")
print("=" * 95)

engine = WalkForwardEngine(
    SYMBOL,
    initial_capital=INITIAL_CAPITAL,
    brokerage=BROKERAGE,
    slippage=SLIPPAGE,
)

parameters = {
    "adx_min": ADX,
    "ema_gap_min": EMA_GAP,
    "trailing_atr": TRAILING_ATR,
    "trailing_activation_atr": ACTIVATION_ATR,
    "momentum_min": MOMENTUM,
}

walk_forward = engine.test_period(
    start_date="2024-01-01",
    end_date="2024-12-31",
    parameters=parameters,
    use_market_regime=MARKET_REGIME,
    initial_stop_atr=INITIAL_STOP_ATR,
    risk_per_trade=RISK_PER_TRADE,
    return_trades=True,
)

print(
    f"Trades                 : "
    f"{walk_forward['total_trades']}"
)

print(
    f"Profit                 : "
    f"{walk_forward['net_profit']:.2f}"
)

assert (
    walk_forward["total_trades"]
    ==
    len(walk_forward["trades"])
)

walk_forward_profit = sum(
    t.get("profit", 0)
    for t in walk_forward["trades"]
)

assert_close(
    walk_forward["net_profit"],
    walk_forward_profit,
)

print("Status                 : PASS")


# ============================================================
# TEST 5 — WALK-FORWARD / BACKTEST CONSISTENCY
# ============================================================

print()
print("=" * 95)
print("TEST 5 — WALK-FORWARD / BACKTEST CONSISTENCY")
print("=" * 95)

assert (
    walk_forward["total_trades"]
    ==
    metrics["total_trades"]
)

assert_close(
    walk_forward["net_profit"],
    metrics["net_profit"],
)

assert_close(
    walk_forward["total_return"],
    metrics["total_return"],
)

print(
    f"Trade Count Match      : "
    f"{walk_forward['total_trades']} == "
    f"{metrics['total_trades']}"
)

print(
    f"Profit Match           : "
    f"{walk_forward['net_profit']:.2f}"
)

print(
    f"Return Match           : "
    f"{walk_forward['total_return']:.2f}%"
)

print("Status                 : PASS")


# ============================================================
# TEST 6 — REPEATED BACKTEST REPRODUCIBILITY
# ============================================================

print()
print("=" * 95)
print("TEST 6 — REPRODUCIBILITY")
print("=" * 95)

repeat = service.run_backtest_v3(
    start_date="2024-01-01",
    end_date="2024-12-31",
    ema_gap_min=EMA_GAP,
    adx_min=ADX,
    trailing_atr=TRAILING_ATR,
    trailing_activation_atr=ACTIVATION_ATR,
    momentum_min=MOMENTUM,
    use_market_regime=MARKET_REGIME,
    initial_stop_atr=INITIAL_STOP_ATR,
    risk_per_trade=RISK_PER_TRADE,
)

repeat_trades = repeat.get("trades", [])

assert len(repeat_trades) == len(trades)

repeat_profit = sum(
    t.get("profit", 0)
    for t in repeat_trades
)

assert_close(
    repeat_profit,
    trade_profit,
)

for first, second in zip(
    trades,
    repeat_trades,
):
    assert (
        first["buy_date"]
        ==
        second["buy_date"]
    )

    assert (
        first["sell_date"]
        ==
        second["sell_date"]
    )

    assert_close(
        first["buy_price"],
        second["buy_price"],
    )

    assert_close(
        first["sell_price"],
        second["sell_price"],
    )

    assert_close(
        first["profit"],
        second["profit"],
    )

print(
    f"First Run Trades       : "
    f"{len(trades)}"
)

print(
    f"Second Run Trades      : "
    f"{len(repeat_trades)}"
)

print(
    f"First Run Profit       : "
    f"{trade_profit:.2f}"
)

print(
    f"Second Run Profit      : "
    f"{repeat_profit:.2f}"
)

print("Status                 : PASS")


# ============================================================
# FINAL REGRESSION GATE
# ============================================================

print()
print("=" * 95)
print("FINAL BACKEND REGRESSION GATE COMPLETE")
print("=" * 95)
print()
print("BACKTEST ENGINE         : PASS")
print("PERFORMANCE METRICS     : PASS")
print("STRATEGY OPTIMIZER      : PASS")
print("WALK-FORWARD ENGINE     : PASS")
print("CROSS-LAYER CONSISTENCY : PASS")
print("REPRODUCIBILITY         : PASS")
print()
print("ALL BACKEND REGRESSION CHECKS PASSED")
print()
print("BACKEND READY FOR FINAL FREEZE")