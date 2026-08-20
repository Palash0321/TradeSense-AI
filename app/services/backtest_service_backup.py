import pandas as pd
import yfinance as yf


class BacktestService:

    def __init__(
        self,
        symbol: str,
        brokerage: float = 20,
        slippage: float = 0.10,
        initial_capital: float = 100000
    ):
        self.symbol = symbol
        self.brokerage = brokerage
        self.slippage = slippage
        self.initial_capital = initial_capital

    # =====================================
    # Load Historical Data
    # =====================================

    def load_data(self):

        data = yf.download(
            self.symbol,
            period="5y",
            interval="1d",
            progress=False
        )

        if data.empty:
            return data

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data.dropna(inplace=True)

        return data

    # =====================================
    # EMA Strategy
    # =====================================

    def ema_strategy(self):

        df = self.load_data()

        if df.empty:
            return df

        df["EMA20"] = (
            df["Close"]
            .ewm(
                span=20,
                adjust=False
            )
            .mean()
        )

        df["EMA50"] = (
            df["Close"]
            .ewm(
                span=50,
                adjust=False
            )
            .mean()
        )

        df["Signal"] = 0

        df.loc[
            df["EMA20"] > df["EMA50"],
            "Signal"
        ] = 1

        df.loc[
            df["EMA20"] < df["EMA50"],
            "Signal"
        ] = -1

        df["Position"] = df["Signal"].diff()

        return df

    # =====================================
    # Execute Backtest
    # =====================================

    def run_backtest(self):

        df = self.ema_strategy()

        if df.empty or len(df) < 2:

            return {
                "trades": [],
                "equity_curve": [],
                "max_drawdown": 0,
                "peak_capital": self.initial_capital
            }

        trades = []

        equity_curve = []

        cash = float(self.initial_capital)

        peak_equity = cash

        max_drawdown = 0

        in_position = False

        buy_price = None

        buy_date = None

        shares = 0

        buy_value = 0

        buy_cost = 0

        # ---------------------------------
        # Slippage
        # ---------------------------------

        slippage_rate = (
            self.slippage / 100
        )

        # ---------------------------------
        # Iterate through historical bars
        # ---------------------------------

        rows = list(df.iterrows())

        for i in range(len(rows)):

            index, row = rows[i]

            close_price = float(row["Close"])

            # ---------------------------------
            # 1. Mark existing position
            # ---------------------------------

            if in_position:

                current_equity = (
                    cash
                    + (
                        shares
                        * close_price
                    )
                )

            else:

                current_equity = cash

            # ---------------------------------
            # Update peak equity
            # ---------------------------------

            if current_equity > peak_equity:

                peak_equity = current_equity

            # ---------------------------------
            # Calculate drawdown
            # ---------------------------------

            if peak_equity > 0:

                drawdown = (
                    (
                        peak_equity
                        - current_equity
                    )
                    / peak_equity
                ) * 100

                if drawdown > max_drawdown:

                    max_drawdown = drawdown

            equity_curve.append({

                "date":
                    index.strftime("%Y-%m-%d"),

                "capital":
                    round(
                        current_equity,
                        2
                    )

            })

            # ---------------------------------
            # No next bar available
            # ---------------------------------

            if i >= len(rows) - 1:
                continue

            next_index, next_row = rows[i + 1]

            next_open = float(
                next_row["Open"]
            )

            position = row["Position"]

            if pd.isna(position):
                continue

            # =================================
            # BUY
            # Signal generated on day T
            # Execute on day T+1 OPEN
            # =================================

            if (
                position == 2
                and not in_position
            ):

                execution_price = (
                    next_open
                    * (1 + slippage_rate)
                )

                # Maximum affordable shares
                shares = int(
                    max(
                        0,
                        (
                            cash
                            - self.brokerage
                        )
                        // execution_price
                    )
                )

                if shares <= 0:
                    continue

                buy_price = execution_price

                buy_value = (
                    buy_price
                    * shares
                )

                buy_cost = self.brokerage

                total_buy_cost = (
                    buy_value
                    + buy_cost
                )

                if total_buy_cost > cash:

                    shares = int(
                        max(
                            0,
                            (
                                cash
                                - self.brokerage
                            )
                            // buy_price
                        )
                    )

                    buy_value = (
                        buy_price
                        * shares
                    )

                    total_buy_cost = (
                        buy_value
                        + self.brokerage
                    )

                if shares <= 0:
                    continue

                cash -= total_buy_cost

                buy_date = next_index

                in_position = True

            # =================================
            # SELL
            # Signal generated on day T
            # Execute on day T+1 OPEN
            # =================================

            elif (
                position == -2
                and in_position
            ):

                execution_price = (
                    next_open
                    * (1 - slippage_rate)
                )

                sell_date = next_index

                sell_value = (
                    execution_price
                    * shares
                )

                sell_cost = self.brokerage

                net_sell_value = (
                    sell_value
                    - sell_cost
                )

                # ---------------------------------
                # Profit
                # ---------------------------------

                profit = (
                    net_sell_value
                    - buy_value
                    - buy_cost
                )

                cash += net_sell_value

                invested_capital = (
                    buy_value
                    + buy_cost
                )

                if invested_capital > 0:

                    return_percent = (
                        profit
                        / invested_capital
                    ) * 100

                else:

                    return_percent = 0

                trades.append({

                    "buy_date":
                        buy_date,

                    "sell_date":
                        sell_date,

                    "buy_price":
                        round(
                            buy_price,
                            2
                        ),

                    "sell_price":
                        round(
                            execution_price,
                            2
                        ),

                    "shares":
                        shares,

                    "profit":
                        round(
                            profit,
                            2
                        ),

                    "return_percent":
                        round(
                            return_percent,
                            2
                        )

                })

                in_position = False

                buy_price = None

                buy_date = None

                shares = 0

                buy_value = 0

                buy_cost = 0

        # =====================================
        # Force-close final open position
        # =====================================

        if in_position:

            final_index = df.index[-1]

            final_close = float(
                df.iloc[-1]["Close"]
            )

            # Final liquidation at available close
            sell_price = (
                final_close
                * (1 - slippage_rate)
            )

            sell_value = (
                sell_price
                * shares
            )

            sell_cost = self.brokerage

            net_sell_value = (
                sell_value
                - sell_cost
            )

            profit = (
                net_sell_value
                - buy_value
                - buy_cost
            )

            cash += net_sell_value

            invested_capital = (
                buy_value
                + buy_cost
            )

            if invested_capital > 0:

                return_percent = (
                    profit
                    / invested_capital
                ) * 100

            else:

                return_percent = 0

            trades.append({

                "buy_date":
                    buy_date,

                "sell_date":
                    final_index,

                "buy_price":
                    round(
                        buy_price,
                        2
                    ),

                "sell_price":
                    round(
                        sell_price,
                        2
                    ),

                "shares":
                    shares,

                "profit":
                    round(
                        profit,
                        2
                    ),

                "return_percent":
                    round(
                        return_percent,
                        2
                    )

            })

            # Final equity after liquidation
            final_equity = cash

            if final_equity > peak_equity:

                peak_equity = final_equity

            if peak_equity > 0:

                drawdown = (
                    (
                        peak_equity
                        - final_equity
                    )
                    / peak_equity
                ) * 100

                if drawdown > max_drawdown:

                    max_drawdown = drawdown

            # Avoid duplicate final date
            if (
                not equity_curve
                or equity_curve[-1]["date"]
                != final_index.strftime("%Y-%m-%d")
            ):

                equity_curve.append({

                    "date":
                        final_index.strftime(
                            "%Y-%m-%d"
                        ),

                    "capital":
                        round(
                            final_equity,
                            2
                        )

                })

            else:

                equity_curve[-1]["capital"] = round(
                    final_equity,
                    2
                )

        # =====================================
        # Return Backtest Result
        # =====================================

        return {

            "trades":
                trades,

            "equity_curve":
                equity_curve,

            "max_drawdown":
                round(
                    max_drawdown,
                    2
                ),

            "peak_capital":
                round(
                    peak_equity,
                    2
                )

        }

    # =====================================
    # Performance Metrics
    # =====================================

    def performance_metrics(self):

        result = self.run_backtest()

        trades = result["trades"]

        equity_curve = result["equity_curve"]

        total_trades = len(trades)

        winning_trades = [
            t
            for t in trades
            if t["profit"] > 0
        ]

        losing_trades = [
            t
            for t in trades
            if t["profit"] <= 0
        ]

        # =====================================
        # Profit
        # =====================================

        net_profit = sum(
            t["profit"]
            for t in trades
        )

        # =====================================
        # Win Rate
        # =====================================

        win_rate = 0

        if total_trades > 0:

            win_rate = (
                len(winning_trades)
                / total_trades
            ) * 100

        # =====================================
        # Gross Profit / Loss
        # =====================================

        gross_profit = sum(
            t["profit"]
            for t in trades
            if t["profit"] > 0
        )

        gross_loss = abs(
            sum(
                t["profit"]
                for t in trades
                if t["profit"] < 0
            )
        )

        profit_factor = 0

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        # =====================================
        # Best / Worst Trade
        # =====================================

        best_trade = 0

        worst_trade = 0

        if trades:

            best_trade = max(
                t["profit"]
                for t in trades
            )

            worst_trade = min(
                t["profit"]
                for t in trades
            )

        # =====================================
        # Average Holding Period
        # =====================================

        holding_days = []

        for trade in trades:

            days = (
                trade["sell_date"]
                - trade["buy_date"]
            ).days

            holding_days.append(days)

        average_holding = 0

        if holding_days:

            average_holding = (
                sum(holding_days)
                / len(holding_days)
            )

        # =====================================
        # Daily Sharpe Ratio
        # =====================================

        sharpe_ratio = 0

        if len(equity_curve) > 1:

            equity_series = pd.Series(
                [
                    x["capital"]
                    for x in equity_curve
                ],
                dtype=float
            )

            daily_returns = (
                equity_series
                .pct_change()
                .dropna()
            )

            if (
                len(daily_returns) > 1
                and daily_returns.std() != 0
            ):

                sharpe_ratio = (
                    daily_returns.mean()
                    / daily_returns.std()
                ) * (
                    252 ** 0.5
                )

        # =====================================
        # Final Capital
        # =====================================

        final_capital = (
            self.initial_capital
            + net_profit
        )

        if equity_curve:

            final_capital = float(
                equity_curve[-1]["capital"]
            )

        # =====================================
        # Total Return
        # =====================================

        total_return = 0

        if self.initial_capital > 0:

            total_return = (
                (
                    final_capital
                    - self.initial_capital
                )
                / self.initial_capital
            ) * 100

        # =====================================
        # CAGR
        # =====================================

        cagr = 0

        if (
            equity_curve
            and final_capital > 0
        ):

            start_date = pd.to_datetime(
                equity_curve[0]["date"]
            )

            end_date = pd.to_datetime(
                equity_curve[-1]["date"]
            )

            years = (
                end_date
                - start_date
            ).days / 365.25

            if years > 0:

                cagr = (
                    (
                        final_capital
                        / self.initial_capital
                    )
                    ** (1 / years)
                    - 1
                ) * 100

        # =====================================
        # Average Win / Loss
        # =====================================

        average_win = 0

        if winning_trades:

            average_win = (
                sum(
                    t["profit"]
                    for t in winning_trades
                )
                / len(winning_trades)
            )

        average_loss = 0

        if losing_trades:

            average_loss = (
                sum(
                    t["profit"]
                    for t in losing_trades
                )
                / len(losing_trades)
            )

        # =====================================
        # Expectancy
        # =====================================

        expectancy = 0

        if total_trades > 0:

            expectancy = (
                net_profit
                / total_trades
            )

        # =====================================
        # Final Result
        # =====================================

        return {

            "total_trades":
                total_trades,

            "winning_trades":
                len(winning_trades),

            "losing_trades":
                len(losing_trades),

            "win_rate":
                round(
                    win_rate,
                    2
                ),

            "net_profit":
                round(
                    net_profit,
                    2
                ),

            "final_capital":
                round(
                    final_capital,
                    2
                ),

            "total_return":
                round(
                    total_return,
                    2
                ),

            "cagr":
                round(
                    cagr,
                    2
                ),

            "best_trade":
                round(
                    best_trade,
                    2
                ),

            "worst_trade":
                round(
                    worst_trade,
                    2
                ),

            "average_win":
                round(
                    average_win,
                    2
                ),

            "average_loss":
                round(
                    average_loss,
                    2
                ),

            "expectancy":
                round(
                    expectancy,
                    2
                ),

            "sharpe_ratio":
                round(
                    float(sharpe_ratio),
                    2
                ),

            "profit_factor":
                round(
                    profit_factor,
                    2
                ),

            "average_holding_days":
                round(
                    average_holding,
                    2
                ),

            "max_drawdown":
                round(
                    result["max_drawdown"],
                    2
                ),

            "peak_capital":
                round(
                    result["peak_capital"],
                    2
                ),

            "trades":
                trades,

            "equity_curve":
                equity_curve

        }