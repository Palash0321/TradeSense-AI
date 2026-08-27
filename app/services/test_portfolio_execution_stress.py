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


def execute_test(name, trades):
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
        f"Maximum Exposure %     : "
        f"{result['max_exposure_percent']:.2f}%"
    )

    print(
        f"Maximum Open Positions : "
        f"{result['max_open_positions']}"
    )

    print(
        f"Maximum Drawdown       : "
        f"{result['max_drawdown']:.2f}"
    )

    print(
        f"Maximum Drawdown %     : "
        f"{result['max_drawdown_percent']:.2f}%"
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
        result["final_equity"] >= 0
    ), "Final equity became negative"

    assert (
        result["reconciliation_error"]
        <= 0.000001
        and
        result["reconciliation_error"]
        >= -0.000001
    ), "P&L reconciliation failed"

    return result


# ============================================================
# TEST 1
# MULTIPLE EXITS + MULTIPLE ENTRIES ON SAME DAY
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-01-02",
        "2024-01-05",
        30000,
        33000,
        1,
        3000,
    ),
    make_trade(
        "B",
        "2024-01-02",
        "2024-01-05",
        30000,
        27000,
        1,
        -3000,
    ),
    make_trade(
        "C",
        "2024-01-05",
        "2024-01-10",
        40000,
        44000,
        1,
        4000,
    ),
    make_trade(
        "D",
        "2024-01-05",
        "2024-01-10",
        40000,
        36000,
        1,
        -4000,
    ),
]


result = execute_test(
    "TEST 1 — MULTIPLE EXITS + SAME-DAY ENTRIES",
    trades,
)

assert (
    result["executed_trades"] >= 2
), "Expected capital recycling to occur"


# ============================================================
# TEST 2
# MANY SIGNALS COMPETING FOR CAPITAL
# ============================================================

trades = []

for i in range(10):

    trades.append(
        make_trade(
            f"STOCK_{i}",
            "2024-02-01",
            "2024-02-10",
            20000,
            21000,
            1,
            1000,
        )
    )


result = execute_test(
    "TEST 2 — TEN SIMULTANEOUS CAPITAL COMPETITORS",
    trades,
)

assert (
    result["max_exposure"]
    <= 100000.000001
), "Capital competition exceeded exposure limit"

assert (
    result["executed_trades"]
    +
    result["skipped_trades"]
    ==
    10
), "Signal accounting failed"


# ============================================================
# TEST 3
# TINY RESIDUAL CAPITAL
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-03-01",
        "2024-03-05",
        99900,
        100000,
        1,
        100,
    ),
    make_trade(
        "B",
        "2024-03-01",
        "2024-03-05",
        1000,
        1100,
        1,
        100,
    ),
]


result = execute_test(
    "TEST 3 — TINY RESIDUAL CAPITAL",
    trades,
)

assert (
    result["max_exposure"]
    <= 100000.000001
), "Tiny residual capital breached exposure"


# ============================================================
# TEST 4
# EXPOSURE EXACTLY AT LIMIT
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-04-01",
        "2024-04-10",
        50000,
        51000,
        1,
        1000,
    ),
    make_trade(
        "B",
        "2024-04-01",
        "2024-04-10",
        50000,
        49000,
        1,
        -1000,
    ),
]


result = execute_test(
    "TEST 4 — EXACT 100% EXPOSURE",
    trades,
)

assert (
    result["max_exposure"]
    <= 100000.000001
), "Exact exposure limit was exceeded"


# ============================================================
# TEST 5
# JUST ABOVE EXPOSURE LIMIT
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-05-01",
        "2024-05-10",
        60000,
        62000,
        1,
        2000,
    ),
    make_trade(
        "B",
        "2024-05-01",
        "2024-05-10",
        60000,
        58000,
        1,
        -2000,
    ),
]


result = execute_test(
    "TEST 5 — ABOVE EXPOSURE LIMIT",
    trades,
)

assert (
    result["max_exposure"]
    <= 100000.000001
), "Above-limit exposure was not constrained"


# ============================================================
# TEST 6
# CONSECUTIVE CAPITAL RECYCLING
# ============================================================

trades = [
    make_trade(
        "A",
        "2024-06-01",
        "2024-06-02",
        90000,
        91000,
        1,
        1000,
    ),
    make_trade(
        "B",
        "2024-06-02",
        "2024-06-03",
        90000,
        92000,
        1,
        2000,
    ),
    make_trade(
        "C",
        "2024-06-03",
        "2024-06-04",
        90000,
        88000,
        1,
        -2000,
    ),
    make_trade(
        "D",
        "2024-06-04",
        "2024-06-05",
        90000,
        95000,
        1,
        5000,
    ),
]


result = execute_test(
    "TEST 6 — CONSECUTIVE CAPITAL RECYCLING",
    trades,
)

assert (
    result["executed_trades"] == 4
), "Capital was not correctly recycled"


# ============================================================
# TEST 7
# MANY POSITIONS ACROSS MANY DAYS
# ============================================================

trades = []

for i in range(30):

    day = (
        f"2024-07-{(i % 9) + 1:02d}"
    )

    exit_day = (
        f"2024-07-{(i % 9) + 10:02d}"
    )

    trades.append(
        make_trade(
            f"SYMBOL_{i:02d}",
            day,
            exit_day,
            10000,
            10100,
            1,
            100,
        )
    )


result = execute_test(
    "TEST 7 — HIGH SIGNAL VOLUME",
    trades,
)

assert (
    result["original_signals"] == 30
), "Incorrect original signal count"

assert (
    result["executed_trades"]
    +
    result["skipped_trades"]
    ==
    30
), "High-volume signal accounting failed"

assert (
    result["max_exposure"]
    <= 100000.000001
), "High-volume exposure breached limit"


# ============================================================
# TEST 8
# INVALID RECORDS MUST NOT CRASH ENGINE
# ============================================================

trades = [
    make_trade(
        "VALID",
        "2024-08-01",
        "2024-08-05",
        50000,
        51000,
        1,
        1000,
    ),
    {
        "symbol": "INVALID_ZERO_SHARES",
        "buy_date": "2024-08-01",
        "sell_date": "2024-08-05",
        "buy_price": 50000,
        "sell_price": 51000,
        "shares": 0,
        "profit": 0,
    },
    {
        "symbol": "INVALID_PRICE",
        "buy_date": "2024-08-01",
        "sell_date": "2024-08-05",
        "buy_price": 0,
        "sell_price": 51000,
        "shares": 1,
        "profit": 0,
    },
]


result = execute_test(
    "TEST 8 — INVALID TRADE RECORDS",
    trades,
)

assert (
    result["final_equity"] >= 0
), "Invalid records caused negative equity"


# ============================================================
# TEST 9
# EXTREME LOSS SCENARIO
# ============================================================

trades = [
    make_trade(
        "LOSS_A",
        "2024-09-01",
        "2024-09-02",
        50000,
        10000,
        1,
        -40000,
    ),
    make_trade(
        "LOSS_B",
        "2024-09-01",
        "2024-09-02",
        40000,
        5000,
        1,
        -35000,
    ),
]


result = execute_test(
    "TEST 9 — EXTREME LOSS SCENARIO",
    trades,
)

assert (
    result["final_equity"] >= 0
), "Extreme loss produced negative equity"

assert (
    result["reconciliation_error"]
    <= 0.000001
    and
    result["reconciliation_error"]
    >= -0.000001
), "Extreme loss reconciliation failed"


# ============================================================
# TEST 10
# INPUT ORDER REPRODUCIBILITY UNDER HEAVY COMPETITION
# ============================================================

trades = []

for i in range(20):

    trades.append(
        make_trade(
            f"ORDER_{i:02d}",
            "2024-10-01",
            "2024-10-10",
            15000,
            16000 if i % 2 == 0 else 14000,
            1,
            1000 if i % 2 == 0 else -1000,
        )
    )


engine_1 = PortfolioExecutionEngine(
    initial_capital=100000,
    max_portfolio_exposure=1.0,
    brokerage=20,
)

engine_2 = PortfolioExecutionEngine(
    initial_capital=100000,
    max_portfolio_exposure=1.0,
    brokerage=20,
)

result_1 = engine_1.execute(trades)
result_2 = engine_2.execute(
    list(reversed(trades))
)

print()
print("=" * 95)
print("TEST 10 — HEAVY COMPETITION ORDER REVERSAL")
print("=" * 95)

print(
    f"Run 1 Final Equity      : "
    f"{result_1['final_equity']:.2f}"
)

print(
    f"Run 2 Final Equity      : "
    f"{result_2['final_equity']:.2f}"
)

print(
    f"Run 1 Executed          : "
    f"{result_1['executed_trades']}"
)

print(
    f"Run 2 Executed          : "
    f"{result_2['executed_trades']}"
)

print(
    f"Run 1 Skipped           : "
    f"{result_1['skipped_trades']}"
)

print(
    f"Run 2 Skipped           : "
    f"{result_2['skipped_trades']}"
)

assert (
    result_1["final_equity"]
    ==
    result_2["final_equity"]
), "Heavy competition depends on input order"

assert (
    result_1["executed_trades"]
    ==
    result_2["executed_trades"]
), "Executed count depends on input order"

assert (
    result_1["skipped_trades"]
    ==
    result_2["skipped_trades"]
), "Skipped count depends on input order"


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 95)
print("PORTFOLIO EXECUTION STRESS VALIDATION COMPLETE")
print("=" * 95)
print()
print("ALL STRESS TESTS PASSED")