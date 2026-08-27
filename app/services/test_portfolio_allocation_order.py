from app.services.portfolio_execution_engine import (
    PortfolioExecutionEngine
)


def make_trade(
    symbol,
    buy_date,
    sell_date,
    buy_price,
    sell_price,
    shares,
    profit,
):
    return {
        "symbol": symbol,
        "buy_date": buy_date,
        "sell_date": sell_date,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "shares": shares,
        "profit": profit,
    }


def run_test(name, trades):
    print()
    print("=" * 95)
    print(name)
    print("=" * 95)

    engine = PortfolioExecutionEngine(
        initial_capital=100000,
        max_portfolio_exposure=1.0,
        brokerage=20,
    )

    result = engine.execute(trades)

    print(
        f"Initial Capital        : "
        f"{result['initial_capital']:.2f}"
    )

    print(
        f"Final Equity           : "
        f"{result['final_equity']:.2f}"
    )

    print(
        f"Realized P&L           : "
        f"{result['realized_pnl']:.2f}"
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
        f"Maximum Exposure       : "
        f"{result['max_exposure']:.2f}"
    )

    print(
        f"Maximum Open Positions : "
        f"{result['max_open_positions']}"
    )

    print(
        f"Reconciliation Error   : "
        f"{result['reconciliation_error']:.10f}"
    )

    assert (
        result["original_signals"]
        ==
        result["executed_trades"]
        +
        result["skipped_trades"]
    ), "Signal accounting failed"

    assert (
        result["max_exposure"]
        <= 100000.000001
    ), "Exposure exceeded 100%"

    assert (
        result["reconciliation_error"]
        < 0.000001
        and
        result["reconciliation_error"]
        > -0.000001
    ), "P&L reconciliation failed"

    return result


# ============================================================
# TEST 1 — SAME-DAY ORDER MUST BE DETERMINISTIC
# ============================================================

trades_a = [
    make_trade(
        "AAA",
        "2024-01-02",
        "2024-01-10",
        50000,
        55000,
        100,
        5000,
    ),
    make_trade(
        "BBB",
        "2024-01-02",
        "2024-01-10",
        50000,
        40000,
        100,
        -10000,
    ),
]


trades_b = list(reversed(trades_a))


result_a = run_test(
    "TEST 1 — SAME-DAY ORDER A",
    trades_a,
)

result_b = run_test(
    "TEST 1 — SAME-DAY ORDER B",
    trades_b,
)


assert (
    result_a["final_equity"]
    ==
    result_b["final_equity"]
), "Results depend on input order"

assert (
    result_a["realized_pnl"]
    ==
    result_b["realized_pnl"]
), "P&L depends on input order"


# ============================================================
# TEST 2 — EXIT MUST RELEASE CAPITAL BEFORE ENTRY
# ============================================================

trades = [
    make_trade(
        "AAA",
        "2024-01-02",
        "2024-01-05",
        100000,
        110000,
        1,
        10000,
    ),
    make_trade(
        "BBB",
        "2024-01-05",
        "2024-01-10",
        100000,
        105000,
        1,
        5000,
    ),
]


result = run_test(
    "TEST 2 — EXIT BEFORE SAME-DAY ENTRY",
    trades,
)


assert (
    result["executed_trades"] == 2
), "Capital was not released before entry"

assert (
    result["skipped_trades"] == 0
), "Same-day replacement trade was incorrectly skipped"


# ============================================================
# TEST 3 — EXPOSURE MUST NEVER EXCEED 100%
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-01-02",
        "2024-01-10",
        40000,
        42000,
        1,
        2000,
    ),
    make_trade(
        "B",
        "2024-01-02",
        "2024-01-10",
        40000,
        42000,
        1,
        2000,
    ),
    make_trade(
        "C",
        "2024-01-02",
        "2024-01-10",
        40000,
        42000,
        1,
        2000,
    ),
]


result = run_test(
    "TEST 3 — HARD EXPOSURE CAP",
    trades,
)


assert (
    result["max_exposure"]
    <= 100000.000001
), "Portfolio exposure exceeded capital"


# ============================================================
# TEST 4 — ALL SIGNALS MUST HAVE A FINAL STATE
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-01-02",
        "2024-01-10",
        60000,
        62000,
        1,
        2000,
    ),
    make_trade(
        "B",
        "2024-01-02",
        "2024-01-10",
        60000,
        58000,
        1,
        -2000,
    ),
    make_trade(
        "C",
        "2024-01-02",
        "2024-01-10",
        60000,
        61000,
        1,
        1000,
    ),
]


result = run_test(
    "TEST 4 — COMPLETE SIGNAL ACCOUNTING",
    trades,
)


assert (
    result["original_signals"]
    ==
    result["executed_trades"]
    +
    result["skipped_trades"]
), "Signals disappeared from accounting"


# ============================================================
# TEST 5 — REPEATED EXECUTION MUST BE IDENTICAL
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-01-02",
        "2024-01-08",
        30000,
        33000,
        1,
        3000,
    ),
    make_trade(
        "B",
        "2024-01-03",
        "2024-01-09",
        30000,
        27000,
        1,
        -3000,
    ),
    make_trade(
        "C",
        "2024-01-04",
        "2024-01-10",
        50000,
        55000,
        1,
        5000,
    ),
]


result_1 = run_test(
    "TEST 5 — REPRODUCIBILITY RUN 1",
    trades,
)

result_2 = run_test(
    "TEST 5 — REPRODUCIBILITY RUN 2",
    trades,
)


assert (
    result_1["final_equity"]
    ==
    result_2["final_equity"]
), "Repeated execution changed final equity"

assert (
    result_1["realized_pnl"]
    ==
    result_2["realized_pnl"]
), "Repeated execution changed P&L"

assert (
    result_1["executed_trades"]
    ==
    result_2["executed_trades"]
), "Repeated execution changed trade count"


print()
print("=" * 95)
print("PORTFOLIO ALLOCATION / ORDER-BIAS VALIDATION COMPLETE")
print("=" * 95)
print()
print("ALL TESTS PASSED")