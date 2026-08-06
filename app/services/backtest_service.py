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

        data.dropna(inplace=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    # =====================================
    # EMA Strategy
    # =====================================

    def ema_strategy(self):

        df = self.load_data()

        df["EMA20"] = (
            df["Close"]
            .ewm(span=20, adjust=False)
            .mean()
        )

        df["EMA50"] = (
            df["Close"]
            .ewm(span=50, adjust=False)
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

        trades = []

        equity_curve = []

        capital = self.initial_capital

        peak_capital = capital

        max_drawdown = 0

        in_position = False

        buy_price = None

        buy_date = None

        shares = 0

        for index, row in df.iterrows():

            position = row["Position"]

            if pd.isna(position):
                continue

            # BUY
            if position == 2 and not in_position:

                buy_price = float(row["Close"])

                shares = int(capital // buy_price)

                if shares == 0:
                    continue

                buy_date = index

                in_position = True

            # SELL
            elif position == -2 and in_position:

                sell_price = float(row["Close"])

                sell_date = index

                brokerage_cost = self.brokerage * 2

                slippage_cost = (
                    buy_price
                    * self.slippage
                    / 100
                )

                profit_per_share = (
                    sell_price
                    - buy_price
                    - brokerage_cost
                    - slippage_cost
                )

                profit = profit_per_share * shares

                capital += profit

                if capital > peak_capital:
                    peak_capital = capital

                drawdown = (
                    (peak_capital - capital)
                    / peak_capital
                ) * 100

                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                equity_curve.append({

                    "date": sell_date.strftime("%Y-%m-%d"),

                    "capital": round(capital, 2)

                })

                percent = (
                    profit
                    /
                    (buy_price * shares)
                ) * 100

                trades.append({

                    "buy_date": buy_date,

                    "sell_date": sell_date,

                    "buy_price": round(
                        buy_price,
                        2
                    ),

                    "sell_price": round(
                        sell_price,
                        2
                    ),

                    "shares": shares,

                    "profit": round(
                        profit,
                        2
                    ),

                    "return_percent": round(
                        percent,
                        2
                    )

                })

                in_position = False

        return {

            "trades": trades,

            "equity_curve": equity_curve,

            "max_drawdown": round(
                max_drawdown,
                2
            ),

            "peak_capital": round(
                peak_capital,
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
            t for t in trades
            if t["profit"] > 0
        ]

        losing_trades = [
            t for t in trades
            if t["profit"] <= 0
        ]

        net_profit = sum(
            t["profit"]
            for t in trades
        )

        returns = [
            t["return_percent"]
            for t in trades
        ]

        sharpe_ratio = 0

        if len(returns) > 1:

            returns_series = pd.Series(
                returns
            )

            std = returns_series.std()

            if std != 0:

                sharpe_ratio = (
                    returns_series.mean()
                    / std
                ) * (252 ** 0.5)

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

        if gross_loss != 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )

        holding_days = []

        for trade in trades:

            days = (
                trade["sell_date"]
                -
                trade["buy_date"]
            ).days

            holding_days.append(days)

        average_holding = 0

        if holding_days:

            average_holding = (
                sum(holding_days)
                /
                len(holding_days)
            )

        win_rate = 0

        if total_trades > 0:

            win_rate = (
                len(winning_trades)
                / total_trades
            ) * 100

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

        return {

            "total_trades": total_trades,

            "winning_trades": len(
                winning_trades
            ),

            "losing_trades": len(
                losing_trades
            ),

            "win_rate": round(
                win_rate,
                2
            ),

            "net_profit": round(
                net_profit,
                2
            ),

            "best_trade": round(
                best_trade,
                2
            ),

            "worst_trade": round(
                worst_trade,
                2
            ),

            "sharpe_ratio": round(
                sharpe_ratio,
                2
            ),

            "profit_factor": round(
                profit_factor,
                2
            ),

            "average_holding_days": round(
                average_holding,
                1
            ),

            "max_drawdown": result[
                "max_drawdown"
            ],

            "peak_capital": result[
                "peak_capital"
            ],

            "trades": trades,

            "equity_curve": equity_curve

        }