from collections import defaultdict


class PortfolioExecutionEngine:
    """
    Portfolio-level execution engine.

    Responsibilities:
    - Maintain one shared portfolio cash pool.
    - Process exits before entries on each trading date.
    - Enforce maximum gross exposure.
    - Scale trades when partial capital is available.
    - Skip trades when no capital is available.
    - Track realized P&L.
    - Track portfolio equity and drawdown.
    - Preserve the original strategy trade information.
    """

    def __init__(
        self,
        initial_capital=100000.0,
        max_portfolio_exposure=1.0,
        brokerage=20.0,
    ):
        if initial_capital <= 0:
            raise ValueError(
                "initial_capital must be greater than zero"
            )

        if not 0 < max_portfolio_exposure <= 1.0:
            raise ValueError(
                "max_portfolio_exposure must be between 0 and 1"
            )

        if brokerage < 0:
            raise ValueError(
                "brokerage cannot be negative"
            )

        self.initial_capital = float(
            initial_capital
        )

        self.max_portfolio_exposure = float(
            max_portfolio_exposure
        )

        self.brokerage = float(
            brokerage
        )

    # ==================================================
    # Public API
    # ==================================================

    def execute(self, trades):
        """
        Execute independently-generated strategy trades
        through a shared portfolio capital pool.

        Returns a complete portfolio execution report.
        """

        prepared, preparation_skips = self._prepare_trades(
            trades
        )

        if not prepared and not preparation_skips:
            return self._empty_result()

        entry_events = defaultdict(list)
        exit_events = defaultdict(list)
        dates = set()

        for sequence, trade in enumerate(prepared):

            trade_id = (
                trade["buy_date"],
                sequence,
                trade["symbol"],
                trade["sell_date"],
            )

            trade["_trade_id"] = trade_id

            entry_events[
                trade["buy_date"]
            ].append(trade)

            exit_events[
                trade["sell_date"]
            ].append(trade)

            dates.add(trade["buy_date"])
            dates.add(trade["sell_date"])

        cash = self.initial_capital

        active = {}

        executed_trades = []

        skipped_trades = list(
            preparation_skips
        )

        peak_equity = self.initial_capital
        max_drawdown = 0.0
        max_drawdown_percent = 0.0

        max_exposure = 0.0
        max_open_positions = 0

        daily_realized_pnl = defaultdict(float)

        total_realized_profit = 0.0

        for current_date in sorted(dates):

            # ==========================================
            # 1. EXIT POSITIONS FIRST
            # ==========================================

            for trade_id, position in list(
                active.items()
            ):

                if (
                    position["sell_date"]
                    != current_date
                ):
                    continue

                sell_value = (
                    position["executed_shares"]
                    * position["sell_price"]
                )

                exit_cost = self.brokerage

                cash += (
                    sell_value
                    - exit_cost
                )

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

                realized_profit -= (
                    exit_cost
                )

                total_realized_profit += (
                    realized_profit
                )

                daily_realized_pnl[
                    current_date
                ] += realized_profit

                position["exit_cost"] = (
                    exit_cost
                )

                position["realized_profit"] = (
                    realized_profit
                )

                executed_trades.append(
                    position
                )

                del active[trade_id]

            # ==========================================
            # 2. ENTER NEW POSITIONS
            # ==========================================

            for trade in entry_events[
                current_date
            ]:

                intended_value = (
                    trade["buy_value"]
                )

                if intended_value <= 0:
                    skipped_trades.append(
                        self._skip_record(
                            trade,
                            "INVALID_INTENDED_VALUE",
                        )
                    )
                    continue

                current_exposure = sum(
                    position[
                        "executed_buy_value"
                    ]
                    for position in active.values()
                )

                maximum_exposure = (
                    self.initial_capital
                    * self.max_portfolio_exposure
                )

                available_exposure = max(
                    maximum_exposure
                    - current_exposure,
                    0.0,
                )

                if available_exposure <= 0:
                    skipped_trades.append(
                        self._skip_record(
                            trade,
                            "EXPOSURE_LIMIT_REACHED",
                        )
                    )
                    continue

                executed_value = min(
                    intended_value,
                    available_exposure,
                )

                # --------------------------------------
                # Capital constraint
                # --------------------------------------

                entry_cost = self.brokerage

                affordable_value = max(
                    cash - entry_cost,
                    0.0,
                )

                executed_value = min(
                    executed_value,
                    affordable_value,
                )

                if executed_value <= 0:
                    skipped_trades.append(
                        self._skip_record(
                            trade,
                            "INSUFFICIENT_CASH",
                        )
                    )
                    continue

                scale_factor = (
                    executed_value
                    / intended_value
                )

                executed_shares = (
                    trade["shares"]
                    * scale_factor
                )

                if executed_shares <= 0:
                    skipped_trades.append(
                        self._skip_record(
                            trade,
                            "ZERO_EXECUTED_SHARES",
                        )
                    )
                    continue

                total_entry_cash = (
                    executed_value
                    + entry_cost
                )

                cash -= total_entry_cash

                position = {
                    "trade_id":
                        trade["_trade_id"],

                    "symbol":
                        trade["symbol"],

                    "buy_date":
                        trade["buy_date"],

                    "sell_date":
                        trade["sell_date"],

                    "buy_price":
                        trade["buy_price"],

                    "sell_price":
                        trade["sell_price"],

                    "original_shares":
                        trade["shares"],

                    "executed_shares":
                        executed_shares,

                    "intended_buy_value":
                        intended_value,

                    "executed_buy_value":
                        executed_value,

                    "scale_factor":
                        scale_factor,

                    "entry_cost":
                        entry_cost,

                    "exit_cost":
                        0.0,

                    "original_profit":
                        trade["profit"],

                    "realized_profit":
                        None,
                }

                active[
                    trade["_trade_id"]
                ] = position

                                # ======================================
                # SAME-DAY ENTRY / EXIT
                # ======================================
                #
                # If a strategy trade opens and closes
                # on the same trading date, the exit phase
                # has already occurred. Therefore execute
                # the exit immediately after the entry.
                #
                if (
                    trade["sell_date"]
                    == current_date
                ):

                    sell_value = (
                        position["executed_shares"]
                        * position["sell_price"]
                    )

                    exit_cost = self.brokerage

                    cash += (
                        sell_value
                        - exit_cost
                    )

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

                    realized_profit -= (
                        exit_cost
                    )

                    total_realized_profit += (
                        realized_profit
                    )

                    daily_realized_pnl[
                        current_date
                    ] += realized_profit

                    position["exit_cost"] = (
                        exit_cost
                    )

                    position["realized_profit"] = (
                        realized_profit
                    )

                    executed_trades.append(
                        position
                    )

                    del active[
                        trade["_trade_id"]
                    ]

            # ==========================================
            # 3. PORTFOLIO MARKING
            # ==========================================

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

            max_exposure = max(
                max_exposure,
                exposure,
            )

            max_open_positions = max(
                max_open_positions,
                len(active),
            )

            peak_equity = max(
                peak_equity,
                equity,
            )

            drawdown = (
                peak_equity
                - equity
            )

            drawdown_percent = (
                (
                    drawdown
                    / peak_equity
                )
                * 100
                if peak_equity > 0
                else 0.0
            )

            max_drawdown = max(
                max_drawdown,
                drawdown,
            )

            max_drawdown_percent = max(
                max_drawdown_percent,
                drawdown_percent,
            )

        # ==========================================
        # 4. FINAL RECONCILIATION
        # ==========================================

        final_equity = (
            self.initial_capital
            + total_realized_profit
        )

        expected_profit = sum(
            trade["realized_profit"]
            for trade in executed_trades
            if trade["realized_profit"]
            is not None
        )

        reconciliation_error = (
            total_realized_profit
            - expected_profit
        )

        return {
            "initial_capital":
                self.initial_capital,

            "final_equity":
                final_equity,

            "realized_pnl":
                total_realized_profit,

            "return_percent": (
                total_realized_profit
                / self.initial_capital
                * 100
            ),

            "max_exposure":
                max_exposure,

            "max_exposure_percent": (
                max_exposure
                / self.initial_capital
                * 100
            ),

            "max_open_positions":
                max_open_positions,

            "max_drawdown":
                max_drawdown,

            "max_drawdown_percent":
                max_drawdown_percent,

            "original_signals":
                len(prepared)
                + len(preparation_skips),

            "executed_trades":
                len(executed_trades),

            "scaled_trades":
                sum(
                    1
                    for trade in executed_trades
                    if trade["scale_factor"]
                    < 0.999999
                ),

            "skipped_trades":
                len(skipped_trades),

            "executed_trade_details":
                executed_trades,

            "skipped_trade_details":
                skipped_trades,

            "daily_realized_pnl":
                dict(daily_realized_pnl),

            "reconciliation_error":
                reconciliation_error,

            "reconciliation_passed":
                abs(reconciliation_error)
                < 1e-9,
        }

    # ==================================================
    # Trade Preparation
    # ==================================================

    def _prepare_trades(self, trades):

        prepared = []

        preparation_skips = []

        for trade in trades:

            buy_date = self._normalize_date(
                trade.get("buy_date")
            )

            sell_date = self._normalize_date(
                trade.get("sell_date")
            )

            if sell_date is None:
                sell_date = self._normalize_date(
                    trade.get("exit_date")
                )

            if buy_date is None:
                preparation_skips.append(
                    {
                        "symbol":
                            trade.get(
                                "symbol",
                                "UNKNOWN"
                            ),
                        "buy_date":
                            None,
                        "sell_date":
                            sell_date,
                        "reason":
                            "INVALID_BUY_DATE",
                        "intended_buy_value":
                            0.0,
                    }
                )
                continue

            if sell_date is None:
                preparation_skips.append(
                    {
                        "symbol":
                            trade.get(
                                "symbol",
                                "UNKNOWN"
                            ),
                        "buy_date":
                            buy_date,
                        "sell_date":
                            None,
                        "reason":
                            "INVALID_SELL_DATE",
                        "intended_buy_value":
                            0.0,
                    }
                )
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

            if shares <= 0:
                preparation_skips.append(
                    {
                        "symbol":
                            trade.get(
                                "symbol",
                                "UNKNOWN"
                            ),
                        "buy_date":
                            buy_date,
                        "sell_date":
                            sell_date,
                        "reason":
                            "INVALID_SHARES",
                        "intended_buy_value":
                            0.0,
                    }
                )
                continue

            if buy_price <= 0:
                preparation_skips.append(
                    {
                        "symbol":
                            trade.get(
                                "symbol",
                                "UNKNOWN"
                            ),
                        "buy_date":
                            buy_date,
                        "sell_date":
                            sell_date,
                        "reason":
                            "INVALID_BUY_PRICE",
                        "intended_buy_value":
                            0.0,
                    }
                )
                continue

            if sell_price <= 0:
                preparation_skips.append(
                    {
                        "symbol":
                            trade.get(
                                "symbol",
                                "UNKNOWN"
                            ),
                        "buy_date":
                            buy_date,
                        "sell_date":
                            sell_date,
                        "reason":
                            "INVALID_SELL_PRICE",
                        "intended_buy_value":
                            0.0,
                    }
                )
                continue

            prepared.append({
                "symbol":
                    trade.get(
                        "symbol",
                        "UNKNOWN",
                    ),

                "buy_date":
                    buy_date,

                "sell_date":
                    sell_date,

                "shares":
                    shares,

                "buy_price":
                    buy_price,

                "sell_price":
                    sell_price,

                "buy_value":
                    shares
                    * buy_price,

                "sell_value":
                    shares
                    * sell_price,

                "profit":
                    profit,
            })

        prepared.sort(
            key=lambda trade: (
                trade["buy_date"],
                trade["sell_date"],
                trade["symbol"],
            )
        )

        return prepared, preparation_skips

    # ==================================================
    # Helpers
    # ==================================================

    @staticmethod
    def _normalize_date(value):

        if value is None:
            return None

        if hasattr(value, "strftime"):
            return value.strftime(
                "%Y-%m-%d"
            )

        return str(value)[:10]

    @staticmethod
    def _skip_record(
        trade,
        reason,
    ):

        return {
            "symbol":
                trade["symbol"],

            "buy_date":
                trade["buy_date"],

            "sell_date":
                trade["sell_date"],

            "reason":
                reason,

            "intended_buy_value":
                trade["buy_value"],
        }

    def _empty_result(self):

        return {
            "initial_capital":
                self.initial_capital,

            "final_equity":
                self.initial_capital,

            "realized_pnl":
                0.0,

            "return_percent":
                0.0,

            "max_exposure":
                0.0,

            "max_exposure_percent":
                0.0,

            "max_open_positions":
                0,

            "max_drawdown":
                0.0,

            "max_drawdown_percent":
                0.0,

            "original_signals":
                0,

            "executed_trades":
                0,

            "scaled_trades":
                0,

            "skipped_trades":
                0,

            "executed_trade_details":
                [],

            "skipped_trade_details":
                [],

            "daily_realized_pnl":
                {},

            "reconciliation_error":
                0.0,

            "reconciliation_passed":
                True,
        }