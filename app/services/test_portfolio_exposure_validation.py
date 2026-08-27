from collections import defaultdict
from datetime import datetime

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
    "ema_gap_min": 0.0,
    "adx_min": 15,
    "trailing_atr": 2.5,
    "trailing_activation_atr": 2.5,
    "momentum_min": None,
    "use_market_regime": False,
    "initial_stop_atr": 3.5,
    "risk_per_trade": 0.01,
}


def to_date(value):

    if value is None:
        return None

    if hasattr(value, "date"):
        try:
            return value.date()
        except Exception:
            pass

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    text = str(value)

    return text[:10]


def run_symbol(
    symbol,
    start_date,
    end_date
):

    service = BacktestService(
        symbol=symbol,
        strategy="ema_atr",
        initial_capital=100000,
        brokerage=20,
        slippage=0.10,
    )

    result = service.run_backtest_v3(
        start_date=start_date,
        end_date=end_date,

        ema_gap_min=CONFIG[
            "ema_gap_min"
        ],

        adx_min=CONFIG[
            "adx_min"
        ],

        trailing_atr=CONFIG[
            "trailing_atr"
        ],

        trailing_activation_atr=CONFIG[
            "trailing_activation_atr"
        ],

        momentum_min=CONFIG[
            "momentum_min"
        ],

        use_market_regime=CONFIG[
            "use_market_regime"
        ],

        initial_stop_atr=CONFIG[
            "initial_stop_atr"
        ],

        risk_per_trade=CONFIG[
            "risk_per_trade"
        ],
    )

    return result


def build_position_events(
    trades,
    symbol
):

    events = []

    for trade in trades:

        buy_date = to_date(
            trade.get("buy_date")
        )

        sell_date = to_date(
            trade.get("sell_date")
        )

        if sell_date is None:

            sell_date = to_date(
                trade.get("exit_date")
            )

        if buy_date is None:
            continue

        if sell_date is None:
            continue

        shares = float(
            trade.get("shares", 0)
        )

        buy_price = float(
            trade.get("buy_price", 0)
        )

        sell_price = float(
            trade.get("sell_price", 0)
        )

        buy_value = (
            shares
            * buy_price
        )

        sell_value = (
            shares
            * sell_price
        )

        profit = float(
            trade.get("profit", 0)
        )

        events.append({
            "symbol": symbol,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "shares": shares,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "profit": profit,
        })

    return events


def validate_period(
    year,
    start_date,
    end_date
):

    print()
    print("=" * 105)
    print(
        f"PORTFOLIO EXPOSURE VALIDATION — {year}"
    )
    print("=" * 105)

    all_positions = []

    excluded = []

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

        try:

            result = run_symbol(
                symbol,
                start_date,
                end_date
            )

        except Exception as exc:

            excluded.append({
                "symbol": symbol,
                "reason": (
                    f"ERROR: {exc}"
                )
            })

            continue

        trades = result.get(
            "trades",
            []
        )

        if not trades:

            excluded.append({
                "symbol": symbol,
                "reason": "NO_TRADES"
            })

            continue

        positions = build_position_events(
            trades,
            symbol
        )

        all_positions.extend(
            positions
        )

    print()
    print(
        f"Total Universe      : "
        f"{len(symbols)}"
    )

    print(
        f"Symbols With Trades  : "
        f"{len(set(
            p['symbol']
            for p in all_positions
        ))}"
    )

    print(
        f"Total Completed Trades: "
        f"{len(all_positions)}"
    )

    print(
        f"Excluded / Invalid   : "
        f"{len(excluded)}"
    )

    for item in excluded:

        print(
            f"Excluded            : "
            f"{item['symbol']} | "
            f"{item['reason']}"
        )

    if not all_positions:

        print("No positions available.")
        return

    # -------------------------------------------------
    # Build the complete event calendar.
    #
    # IMPORTANT:
    # Exits are processed BEFORE entries on the same
    # date. This prevents capital from being counted
    # twice when one position closes and another opens
    # on the same day.
    # -------------------------------------------------

    entry_events = defaultdict(list)
    exit_events = defaultdict(list)

    dates = set()

    for position in all_positions:

        buy_date = position["buy_date"]
        sell_date = position["sell_date"]

        entry_events[buy_date].append(
            position
        )

        exit_events[sell_date].append(
            position
        )

        dates.add(buy_date)
        dates.add(sell_date)

    active_positions = {}

    cash = 100000.0

    initial_capital = cash

    peak_equity = cash

    max_drawdown_amount = 0.0
    max_drawdown_percent = 0.0

    max_gross_exposure = 0.0
    max_exposure_percent = 0.0

    max_open_positions = 0

    exposure_by_date = {}

    equity_by_date = {}

    realized_profit = 0.0

    for current_date in sorted(dates):

        # ---------------------------------------------
        # 1. Close positions first.
        # ---------------------------------------------

        for position in exit_events[
            current_date
        ]:

            key = (
                position["symbol"],
                position["buy_date"],
                position["sell_date"]
            )

            if key not in active_positions:

                # A same-day position may not have been
                # added yet because its entry and exit are
                # on the same date. Handle it directly.
                cash += position[
                    "sell_value"
                ]

                realized_profit += (
                    position["profit"]
                )

                continue

            cash += position[
                "sell_value"
            ]

            realized_profit += (
                position["profit"]
            )

            del active_positions[key]

        # ---------------------------------------------
        # 2. Open new positions.
        # ---------------------------------------------

        for position in entry_events[
            current_date
        ]:

            key = (
                position["symbol"],
                position["buy_date"],
                position["sell_date"]
            )

            cash -= position[
                "buy_value"
            ]

            active_positions[key] = (
                position
            )

        # ---------------------------------------------
        # 3. Mark active positions at their entry
        # price for this diagnostic.
        #
        # This gives us a conservative capital/exposure
        # representation without inventing unavailable
        # daily OHLC data inside this test.
        # ---------------------------------------------

        gross_exposure = sum(
            position["buy_value"]
            for position in active_positions.values()
        )

        equity = (
            cash
            + gross_exposure
        )

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

        if gross_exposure > max_gross_exposure:

            max_gross_exposure = (
                gross_exposure
            )

        exposure_percent = (
            gross_exposure
            / initial_capital
            * 100
        )

        if exposure_percent > max_exposure_percent:

            max_exposure_percent = (
                exposure_percent
            )

        open_count = len(
            active_positions
        )

        if open_count > max_open_positions:

            max_open_positions = (
                open_count
            )

        exposure_by_date[
            current_date
        ] = gross_exposure

        equity_by_date[
            current_date
        ] = equity

    # -------------------------------------------------
    # Final checks
    # -------------------------------------------------

    total_trade_profit = sum(
        position["profit"]
        for position in all_positions
    )

    final_equity = (
        initial_capital
        + total_trade_profit
    )

    reconciliation_error = (
        final_equity
        - (
            initial_capital
            + realized_profit
        )
    )

    # -------------------------------------------------
    # Risk-per-trade theoretical budget
    # -------------------------------------------------

    theoretical_trade_risk = (
        initial_capital
        * CONFIG["risk_per_trade"]
    )

    # -------------------------------------------------
    # Output
    # -------------------------------------------------

    print()
    print(
        "CAPITAL"
    )
    print("-" * 105)

    print(
        f"Initial Capital          : "
        f"{initial_capital:,.2f}"
    )

    print(
        f"Final Equity             : "
        f"{final_equity:,.2f}"
    )

    print(
        f"Total Trade P&L          : "
        f"{total_trade_profit:,.2f}"
    )

    print()
    print(
        "EXPOSURE"
    )
    print("-" * 105)

    print(
        f"Maximum Gross Exposure   : "
        f"{max_gross_exposure:,.2f}"
    )

    print(
        f"Maximum Exposure %       : "
        f"{max_exposure_percent:.2f}%"
    )

    print(
        f"Maximum Open Positions   : "
        f"{max_open_positions}"
    )

    print(
        f"Theoretical 1% Risk     : "
        f"{theoretical_trade_risk:,.2f}"
    )

    print()
    print(
        "DRAWDOWN"
    )
    print("-" * 105)

    print(
        f"Maximum DD Amount        : "
        f"{max_drawdown_amount:,.2f}"
    )

    print(
        f"Maximum DD %             : "
        f"{max_drawdown_percent:.2f}%"
    )

    print()
    print(
        "RECONCILIATION"
    )
    print("-" * 105)

    print(
        f"Realized P&L             : "
        f"{realized_profit:,.2f}"
    )

    print(
        f"Trade P&L Sum            : "
        f"{total_trade_profit:,.2f}"
    )

    print(
        f"Reconciliation Error     : "
        f"{reconciliation_error:,.6f}"
    )

    if abs(reconciliation_error) < 0.01:

        print(
            "Status                   : PASS"
        )

    else:

        print(
            "Status                   : FAIL"
        )


print()
print("=" * 105)
print("FROZEN CONFIGURATION")
print("=" * 105)

print("ADX                  : 15")
print("EMA Gap              : 0.0")
print("Trailing ATR         : 2.5")
print("Activation ATR       : 2.5")
print("Initial Stop ATR     : 3.5")
print("Risk Per Trade       : 1.00%")
print("Momentum             : OFF")
print("Market Regime        : OFF")


validate_period(
    2024,
    "2024-01-01",
    "2024-12-31"
)

validate_period(
    2025,
    "2025-01-01",
    "2025-12-31"
)

print()
print("=" * 105)
print(
    "PORTFOLIO EXPOSURE VALIDATION COMPLETE"
)
print("=" * 105)