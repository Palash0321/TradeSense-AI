from app.services.walk_forward_engine import WalkForwardEngine
from app.services.portfolio_execution_engine import (
    PortfolioExecutionEngine,
)


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


FROZEN = {
    "adx_values": [15],
    "ema_gap_values": [0.0],
    "trailing_atr_values": [2.5],
    "trailing_activation_atr": 2.5,
    "momentum_min": None,
    "use_market_regime": False,
    "initial_stop_atr": 3.5,
    "risk_per_trade": 0.01,
}


def collect_trades(
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
            flush=True,
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
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            ],
            **FROZEN,
        )

        window = result["windows"][0]

        testing = window["testing"]

        if testing is None:

            excluded.append({
                "symbol": symbol,
                "reason": window.get(
                    "exclusion_reason",
                    "UNKNOWN",
                ),
            })

            continue

        parameters = (
            window["best_parameters"]
        )

        test_result = engine.test_period(
            start_date=test_start,
            end_date=test_end,
            parameters=parameters,
            use_market_regime=False,
            initial_stop_atr=3.5,
            risk_per_trade=0.01,
            return_trades=True,
        )

        trades = test_result.get(
            "trades",
            [],
        )

        for trade in trades:

            trade["symbol"] = symbol

        all_trades.extend(trades)

    return all_trades, excluded


def validate_year(
    year,
    train_start,
    train_end,
    test_start,
    test_end,
):

    print()
    print("=" * 100)
    print(
        f"PORTFOLIO EXECUTION — REAL DATA — {year}"
    )
    print("=" * 100)

    trades, excluded = collect_trades(
        train_start,
        train_end,
        test_start,
        test_end,
    )

    print()
    print("SIGNAL COLLECTION")
    print("-" * 100)

    print(
        f"Total Universe       : "
        f"{len(symbols)}"
    )

    print(
        f"Validated Stocks     : "
        f"{len(symbols) - len(excluded)}"
    )

    print(
        f"Excluded Stocks      : "
        f"{len(excluded)}"
    )

    for item in excluded:

        print(
            f"Excluded             : "
            f"{item['symbol']} | "
            f"{item['reason']}"
        )

    print(
        f"Original Trade Signals: "
        f"{len(trades)}"
    )

    # ==================================================
    # Portfolio execution
    # ==================================================

    portfolio = PortfolioExecutionEngine(
        initial_capital=100000,
        max_portfolio_exposure=1.0,
        brokerage=20,
    )

    result = portfolio.execute(
        trades
    )

    print()
    print("PORTFOLIO EXECUTION")
    print("-" * 100)

    print(
        f"Initial Capital       : "
        f"{result['initial_capital']:,.2f}"
    )

    print(
        f"Final Equity          : "
        f"{result['final_equity']:,.2f}"
    )

    print(
        f"Realized P&L          : "
        f"{result['realized_pnl']:,.2f}"
    )

    print(
        f"Return                : "
        f"{result['return_percent']:.2f}%"
    )

    print(
        f"Maximum Exposure      : "
        f"{result['max_exposure']:,.2f}"
    )

    print(
        f"Maximum Exposure %    : "
        f"{result['max_exposure_percent']:.2f}%"
    )

    print(
        f"Maximum Open Positions: "
        f"{result['max_open_positions']}"
    )

    print(
        f"Maximum Drawdown      : "
        f"{result['max_drawdown']:,.2f}"
    )

    print(
        f"Maximum Drawdown %    : "
        f"{result['max_drawdown_percent']:.2f}%"
    )

    print()
    print("EXECUTION CONSTRAINTS")
    print("-" * 100)

    print(
        f"Original Signals      : "
        f"{result['original_signals']}"
    )

    print(
        f"Executed Trades       : "
        f"{result['executed_trades']}"
    )

    print(
        f"Scaled Trades         : "
        f"{result['scaled_trades']}"
    )

    print(
        f"Skipped Trades        : "
        f"{result['skipped_trades']}"
    )

    print()
    print("RECONCILIATION")
    print("-" * 100)

    print(
        f"Reconciliation Error   : "
        f"{result['reconciliation_error']:.10f}"
    )

    print(
        f"Reconciliation Status  : "
        f"{'PASS' if result['reconciliation_passed'] else 'FAIL'}"
    )

    # ==================================================
    # Safety assertions
    # ==================================================

    assert (
        result["max_exposure"]
        <= 100000.0 + 1e-9
    ), (
        "Portfolio exposure exceeded "
        "the 100% capital limit."
    )

    assert (
        result["max_exposure_percent"]
        <= 100.0 + 1e-9
    ), (
        "Portfolio exposure percentage "
        "exceeded 100%."
    )

    assert (
        result["final_equity"]
        >= 0
    ), (
        "Portfolio equity became negative."
    )

    assert (
        result["reconciliation_passed"]
    ), (
        "Portfolio P&L reconciliation failed."
    )

    assert (
        result["executed_trades"]
        + result["skipped_trades"]
        == result["original_signals"]
    ), (
        "Execution accounting does not "
        "reconcile."
    )

    return result


# ============================================================
# Frozen configuration
# ============================================================

print("=" * 100)
print(
    "REAL-DATA PORTFOLIO EXECUTION VALIDATION"
)
print("=" * 100)

print()
print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Risk Per Trade       : 1.00%")
print("Momentum             : OFF")
print("Market Regime        : OFF")
print("Maximum Exposure     : 100.00%")


results = {}

results[2024] = validate_year(
    2024,
    "2021-01-01",
    "2023-12-31",
    "2024-01-01",
    "2024-12-31",
)

results[2025] = validate_year(
    2025,
    "2021-01-01",
    "2024-12-31",
    "2025-01-01",
    "2025-12-31",
)


# ============================================================
# Final comparison
# ============================================================

print()
print("=" * 100)
print("REAL-DATA PORTFOLIO EXECUTION VALIDATION COMPLETE")
print("=" * 100)

print()
print(
    "Year | Signals | Executed | Scaled | "
    "Skipped | Final Equity | P&L | Max Exposure"
)

print("-" * 100)

for year in [2024, 2025]:

    result = results[year]

    print(
        f"{year} | "
        f"{result['original_signals']:>7} | "
        f"{result['executed_trades']:>8} | "
        f"{result['scaled_trades']:>6} | "
        f"{result['skipped_trades']:>7} | "
        f"{result['final_equity']:>12.2f} | "
        f"{result['realized_pnl']:>10.2f} | "
        f"{result['max_exposure_percent']:>10.2f}%"
    )

print()
print(
    "ALL REAL-DATA PORTFOLIO SAFETY CHECKS PASSED"
)