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


INITIAL_CAPITAL = 100000.0

RISK_PER_TRADE = 0.01

MAX_PORTFOLIO_EXPOSURE = 1.00

CONFIG = {
    "ema_gap_min": 0.0,
    "adx_min": 15,
    "trailing_atr": 2.5,
    "trailing_activation_atr": 2.5,
    "momentum_min": None,
    "use_market_regime": False,
    "initial_stop_atr": 3.5,
    "risk_per_trade": RISK_PER_TRADE,
}


def normalize_date(value):

    if value is None:
        return None

    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")

    return str(value)[:10]


def load_trades(
    symbol,
    start_date,
    end_date
):

    service = BacktestService(
        symbol=symbol,
        strategy="ema_atr",
        initial_capital=INITIAL_CAPITAL,
        brokerage=20,
        slippage=0.10,
    )

    result = service.run_backtest_v3(
        start_date=start_date,
        end_date=end_date,
        ema_gap_min=CONFIG["ema_gap_min"],
        adx_min=CONFIG["adx_min"],
        trailing_atr=CONFIG["trailing_atr"],
        trailing_activation_atr=CONFIG[
            "trailing_activation_atr"
        ],
        momentum_min=CONFIG["momentum_min"],
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

    return result.get("trades", [])


def prepare_trades(
    trades,
    symbol
):

    prepared = []

    for trade in trades:

        buy_date = normalize_date(
            trade.get("buy_date")
        )

        sell_date = normalize_date(
            trade.get("sell_date")
        )

        if sell_date is None:

            sell_date = normalize_date(
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

        profit = float(
            trade.get("profit", 0)
        )

        buy_value = (
            shares
            * buy_price
        )

        sell_value = (
            shares
            * sell_price
        )

        if shares <= 0:
            continue

        if buy_price <= 0:
            continue

        prepared.append({
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

    return prepared


def collect_all_trades(
    start_date,
    end_date
):

    all_trades = []

    for symbol in symbols:

        print(
            f"Running {symbol}...",
            flush=True
        )

        trades = load_trades(
            symbol,
            start_date,
            end_date
        )

        all_trades.extend(
            prepare_trades(
                trades,
                symbol
            )
        )

    return all_trades


def simulate(
    trades,
    year
):

    print()
    print("=" * 105)
    print(
        f"CAPITAL-CONSTRAINED SIMULATION — {year}"
    )
    print("=" * 105)

    if not trades:

        print("No trades.")
        return

    # --------------------------------------------------
    # The original independent-stock backtest gives us
    # the strategy's intended trade sizes.
    #
    # Here we apply ONE shared portfolio capital pool.
    #
    # When a new position would exceed the portfolio
    # exposure limit, its position is scaled down.
    # --------------------------------------------------

    entry_events = defaultdict(list)
    exit_events = defaultdict(list)

    dates = set()

    for trade in trades:

        entry_events[
            trade["buy_date"]
        ].append(trade)

        exit_events[
            trade["sell_date"]
        ].append(trade)

        dates.add(trade["buy_date"])
        dates.add(trade["sell_date"])

    cash = INITIAL_CAPITAL

    active = {}

    peak_equity = INITIAL_CAPITAL

    max_drawdown = 0.0

    max_drawdown_percent = 0.0

    max_exposure = 0.0

    max_open_positions = 0

    executed_trades = 0

    skipped_trades = 0

    scaled_trades = 0

    total_original_profit = sum(
        trade["profit"]
        for trade in trades
    )

    total_realized_profit = 0.0

    daily_profit = defaultdict(float)

    # --------------------------------------------------
    # Process dates chronologically.
    # Exits happen before entries.
    # --------------------------------------------------

    for current_date in sorted(dates):

        # ----------------------------------------------
        # EXIT EXISTING POSITIONS
        # ----------------------------------------------

        for trade_id, position in list(
            active.items()
        ):

            if position["sell_date"] != current_date:
                continue

            sell_value = (
                position["executed_shares"]
                * position["sell_price"]
            )

            cash += sell_value

            realized_profit = (
                (
                    position["sell_price"]
                    - position["buy_price"]
                )
                * position["executed_shares"]
            )

            realized_profit -= (
                position["entry_cost"]
            )

            realized_profit -= 20

            total_realized_profit += (
                realized_profit
            )

            daily_profit[
                current_date
            ] += realized_profit

            del active[trade_id]

        # ----------------------------------------------
        # ENTRY NEW POSITIONS
        # ----------------------------------------------

        for sequence, trade in enumerate(
            entry_events[current_date]
        ):

            trade_id = (
                current_date,
                sequence,
                trade["symbol"],
                trade["sell_date"]
            )

            current_exposure = sum(
                position[
                    "executed_buy_value"
                ]
                for position in active.values()
            )

            available_exposure = max(
                (
                    INITIAL_CAPITAL
                    * MAX_PORTFOLIO_EXPOSURE
                )
                - current_exposure,
                0
            )

            intended_value = trade[
                "buy_value"
            ]

            if available_exposure <= 0:

                skipped_trades += 1

                continue

            executed_value = min(
                intended_value,
                available_exposure
            )

            scale_factor = (
                executed_value
                / intended_value
            )

            if scale_factor < 0.999999:

                scaled_trades += 1

            executed_shares = (
                trade["shares"]
                * scale_factor
            )

            entry_cost = 20.0

            total_entry_cash = (
                executed_value
                + entry_cost
            )

            if total_entry_cash > cash:

                affordable_value = max(
                    cash - entry_cost,
                    0
                )

                if affordable_value <= 0:

                    skipped_trades += 1

                    continue

                executed_value = min(
                    executed_value,
                    affordable_value
                )

                scale_factor = (
                    executed_value
                    / intended_value
                )

                executed_shares = (
                    trade["shares"]
                    * scale_factor
                )

                if scale_factor < 0.999999:

                    scaled_trades += 1

                total_entry_cash = (
                    executed_value
                    + entry_cost
                )

            if executed_shares <= 0:

                skipped_trades += 1

                continue

            cash -= total_entry_cash

            active[trade_id] = {
                "symbol": trade["symbol"],
                "buy_date": trade["buy_date"],
                "sell_date": trade["sell_date"],
                "buy_price": trade["buy_price"],
                "sell_price": trade["sell_price"],
                "executed_shares": executed_shares,
                "executed_buy_value": executed_value,
                "entry_cost": entry_cost,
            }

            executed_trades += 1

        # ----------------------------------------------
        # PORTFOLIO EQUITY
        #
        # Since this test only has completed trade
        # prices, active positions are marked at entry
        # price. This avoids fabricating daily prices.
        # ----------------------------------------------

        active_value = sum(
            position[
                "executed_buy_value"
            ]
            for position in active.values()
        )

        equity = (
            cash
            + active_value
        )

        exposure = active_value

        if exposure > max_exposure:

            max_exposure = exposure

        open_positions = len(active)

        if open_positions > max_open_positions:

            max_open_positions = (
                open_positions
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

        if drawdown_amount > max_drawdown:

            max_drawdown = drawdown_amount

        if drawdown_percent > max_drawdown_percent:

            max_drawdown_percent = (
                drawdown_percent
            )

    # --------------------------------------------------
    # Final equity
    # --------------------------------------------------

    final_equity = (
        INITIAL_CAPITAL
        + total_realized_profit
    )

    final_return = (
        total_realized_profit
        / INITIAL_CAPITAL
        * 100
    )

    print()
    print(
        "CAPITAL-CONSTRAINED RESULTS"
    )
    print("-" * 105)

    print(
        f"Initial Capital          : "
        f"{INITIAL_CAPITAL:,.2f}"
    )

    print(
        f"Final Equity             : "
        f"{final_equity:,.2f}"
    )

    print(
        f"Realized P&L             : "
        f"{total_realized_profit:,.2f}"
    )

    print(
        f"Return                   : "
        f"{final_return:.2f}%"
    )

    print()
    print(
        "EXPOSURE"
    )
    print("-" * 105)

    print(
        f"Maximum Exposure         : "
        f"{max_exposure:,.2f}"
    )

    print(
        f"Maximum Exposure %       : "
        f"{(
            max_exposure
            / INITIAL_CAPITAL
            * 100
        ):.2f}%"
    )

    print(
        f"Maximum Open Positions   : "
        f"{max_open_positions}"
    )

    print()
    print(
        "EXECUTION CONSTRAINTS"
    )
    print("-" * 105)

    print(
        f"Original Signals         : "
        f"{len(trades)}"
    )

    print(
        f"Executed Trades          : "
        f"{executed_trades}"
    )

    print(
        f"Scaled Trades            : "
        f"{scaled_trades}"
    )

    print(
        f"Skipped Trades           : "
        f"{skipped_trades}"
    )

    print()
    print(
        "DRAWDOWN"
    )
    print("-" * 105)

    print(
        f"Maximum DD Amount        : "
        f"{max_drawdown:,.2f}"
    )

    print(
        f"Maximum DD %             : "
        f"{max_drawdown_percent:.2f}%"
    )

    print()
    print(
        "REFERENCE"
    )
    print("-" * 105)

    print(
        f"Independent Backtest P&L: "
        f"{total_original_profit:,.2f}"
    )

    difference = (
        total_realized_profit
        - total_original_profit
    )

    print(
        f"Constraint Impact       : "
        f"{difference:,.2f}"
    )


for year, start_date, end_date in [
    (
        2024,
        "2024-01-01",
        "2024-12-31",
    ),
    (
        2025,
        "2025-01-01",
        "2025-12-31",
    ),
]:

    trades = collect_all_trades(
        start_date,
        end_date
    )

    simulate(
        trades,
        year
    )


print()
print("=" * 105)
print(
    "CAPITAL-CONSTRAINED PORTFOLIO VALIDATION COMPLETE"
)
print("=" * 105)