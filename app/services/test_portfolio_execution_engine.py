from app.services.portfolio_execution_engine import (
    PortfolioExecutionEngine,
)


def print_result(title, result):

    print()
    print("=" * 95)
    print(title)
    print("=" * 95)

    print(
        f"Initial Capital        : "
        f"{result['initial_capital']:,.2f}"
    )

    print(
        f"Final Equity           : "
        f"{result['final_equity']:,.2f}"
    )

    print(
        f"Realized P&L           : "
        f"{result['realized_pnl']:,.2f}"
    )

    print(
        f"Return                 : "
        f"{result['return_percent']:.2f}%"
    )

    print(
        f"Maximum Exposure       : "
        f"{result['max_exposure']:,.2f}"
    )

    print(
        f"Maximum Exposure %     : "
        f"{result['max_exposure_percent']:.2f}%"
    )

    print(
        f"Maximum Open Positions : "
        f"{result['max_open_positions']}"
    )

    print(
        f"Maximum Drawdown       : "
        f"{result['max_drawdown']:,.2f}"
    )

    print(
        f"Maximum Drawdown %     : "
        f"{result['max_drawdown_percent']:.2f}%"
    )

    print(
        f"Original Signals       : "
        f"{result['original_signals']}"
    )

    print(
        f"Executed Trades        : "
        f"{result['executed_trades']}"
    )

    print(
        f"Scaled Trades          : "
        f"{result['scaled_trades']}"
    )

    print(
        f"Skipped Trades         : "
        f"{result['skipped_trades']}"
    )

    print(
        f"Reconciliation Error   : "
        f"{result['reconciliation_error']:.10f}"
    )

    print(
        f"Reconciliation Status  : "
        f"{'PASS' if result['reconciliation_passed'] else 'FAIL'}"
    )


# ============================================================
# BASIC EXECUTION TEST
# ============================================================

trades = [

    {
        "symbol": "TEST_A",
        "buy_date": "2025-01-02",
        "sell_date": "2025-01-03",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 500,
        "profit": 5000,
    },

    {
        "symbol": "TEST_B",
        "buy_date": "2025-01-02",
        "sell_date": "2025-01-04",
        "buy_price": 100,
        "sell_price": 90,
        "shares": 500,
        "profit": -5000,
    },

]


engine = PortfolioExecutionEngine(
    initial_capital=100000,
    max_portfolio_exposure=1.0,
    brokerage=20,
)

result = engine.execute(trades)

print_result(
    "PORTFOLIO EXECUTION ENGINE — BASIC TEST",
    result,
)


# ============================================================
# ASSERTIONS
# ============================================================

assert result["original_signals"] == 2

assert result["executed_trades"] == 2

assert result["skipped_trades"] == 0

assert result["scaled_trades"] == 1

assert (
    result["max_exposure"]
    <= 100000.0 + 1e-9
)

assert (
    result["max_exposure_percent"]
    <= 100.0 + 1e-9
)

assert (
    abs(result["reconciliation_error"])
    < 1e-9
)

assert result["reconciliation_passed"]


# ============================================================
# CAPITAL CONSTRAINT TEST
# ============================================================

large_trades = [

    {
        "symbol": "A",
        "buy_date": "2025-02-03",
        "sell_date": "2025-02-04",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 700,
        "profit": 7000,
    },

    {
        "symbol": "B",
        "buy_date": "2025-02-03",
        "sell_date": "2025-02-05",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 700,
        "profit": 7000,
    },

    {
        "symbol": "C",
        "buy_date": "2025-02-03",
        "sell_date": "2025-02-06",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 700,
        "profit": 7000,
    },

]


constrained_engine = PortfolioExecutionEngine(
    initial_capital=100000,
    max_portfolio_exposure=1.0,
    brokerage=20,
)

constrained_result = (
    constrained_engine.execute(
        large_trades
    )
)

print_result(
    "PORTFOLIO EXECUTION ENGINE — CAPITAL CONSTRAINT TEST",
    constrained_result,
)


assert (
    constrained_result["max_exposure"]
    <= 100000.0 + 1e-9
)

assert (
    constrained_result["scaled_trades"]
    > 0
    or constrained_result["skipped_trades"]
    > 0
)


# ============================================================
# EXIT-BEFORE-ENTRY TEST
# ============================================================

same_day_trades = [

    {
        "symbol": "OLD",
        "buy_date": "2025-03-03",
        "sell_date": "2025-03-04",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 1000,
        "profit": 10000,
    },

    {
        "symbol": "NEW",
        "buy_date": "2025-03-04",
        "sell_date": "2025-03-05",
        "buy_price": 100,
        "sell_price": 110,
        "shares": 1000,
        "profit": 10000,
    },

]


same_day_engine = PortfolioExecutionEngine(
    initial_capital=100000,
    max_portfolio_exposure=1.0,
    brokerage=20,
)

same_day_result = (
    same_day_engine.execute(
        same_day_trades
    )
)

print_result(
    "PORTFOLIO EXECUTION ENGINE — EXIT BEFORE ENTRY TEST",
    same_day_result,
)


assert (
    same_day_result["executed_trades"]
    == 2
)

assert (
    same_day_result["skipped_trades"]
    == 0
)

assert (
    same_day_result["max_exposure"]
    <= 100000.0 + 1e-9
)


print()
print("=" * 95)
print(
    "PORTFOLIO EXECUTION ENGINE VALIDATION COMPLETE"
)
print("=" * 95)