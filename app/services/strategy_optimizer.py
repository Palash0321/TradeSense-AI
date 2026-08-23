from itertools import product

from app.services.backtest_service import BacktestService


class StrategyOptimizer:

    def __init__(
        self,
        symbol: str,
        initial_capital: float = 100000,
        brokerage: float = 20,
        slippage: float = 0.10
    ):

        self.symbol = symbol

        self.initial_capital = (
            initial_capital
        )

        self.brokerage = brokerage

        self.slippage = slippage

    # =====================================
    # Optimization
    # =====================================

    def optimize(
        self,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        start_date=None,
        end_date=None,
        trailing_activation_atr=1.0
    ):

        if adx_values is None:

            adx_values = [
                10,
                15,
                20,
                25,
                30
            ]

        if ema_gap_values is None:

            ema_gap_values = [
                0.00,
                0.25,
                0.50,
                0.75,
                1.00
            ]

        if trailing_atr_values is None:

            trailing_atr_values = [
                2.0,
                3.0,
                4.0,
                5.0
            ]

        results = []

        combinations = product(
            adx_values,
            ema_gap_values,
            trailing_atr_values
        )

        for (
            adx_min,
            ema_gap_min,
            trailing_atr
        ) in combinations:

            service = BacktestService(

                symbol=self.symbol,

                strategy="ema_atr",

                brokerage=self.brokerage,

                slippage=self.slippage,

                initial_capital=self.initial_capital

            )

            metrics = service.performance_metrics(

                start_date=start_date,

                end_date=end_date,

                adx_min=adx_min,

                ema_gap_min=ema_gap_min,

                trailing_atr=trailing_atr,

                trailing_activation_atr=trailing_activation_atr

            )

            results.append({

                "adx_min":
                    adx_min,

                "ema_gap_min":
                    ema_gap_min,

                "trailing_atr":
                    trailing_atr,

                "trailing_activation_atr":
                    trailing_activation_atr,

                "total_trades":
                    metrics["total_trades"],

                "net_profit":
                    metrics["net_profit"],

                "total_return":
                    metrics["total_return"],

                "profit_factor":
                    metrics["profit_factor"],

                "win_rate":
                    metrics["win_rate"],

                "max_drawdown":
                    metrics["max_drawdown"],

                "sharpe_ratio":
                    metrics["sharpe_ratio"]

            })

        return results

    # =====================================
    # Rank Results
    # =====================================

    def rank_results(
        self,
        results,
        min_trades=10
    ):

        filtered = [

            result

            for result in results

            if result["total_trades"]
            >= min_trades

        ]

        for result in filtered:

            profit_factor = max(
                result["profit_factor"],
                0
            )

            total_return = (
                result["total_return"]
            )

            max_drawdown = max(
                result["max_drawdown"],
                0.01
            )

            sharpe = (
                result["sharpe_ratio"]
            )

            result["optimization_score"] = round(

                (
                    total_return
                    * 0.35
                )

                +

                (
                    profit_factor
                    * 20
                    * 0.30
                )

                +

                (
                    sharpe
                    * 10
                    * 0.20
                )

                -

                (
                    max_drawdown
                    * 0.15
                ),

                4
            )

        filtered.sort(

            key=lambda x:
                x["optimization_score"],

            reverse=True

        )

        return filtered

    # =====================================
    # Complete Optimization
    # =====================================

    def run(
        self,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        min_trades=10,
        start_date=None,
        end_date=None,
        trailing_activation_atr=1.0
    ):

        results = self.optimize(

            adx_values=adx_values,

            ema_gap_values=ema_gap_values,

            trailing_atr_values=trailing_atr_values,

            start_date=start_date,

            end_date=end_date,

            trailing_activation_atr=trailing_activation_atr


        )

        ranked = self.rank_results(

            results,

            min_trades=min_trades

        )

        return {

            "symbol":
                self.symbol,

            "total_combinations":
                len(results),

            "valid_combinations":
                len(ranked),

            "best":
                ranked[0]
                if ranked
                else None,

            "results":
                ranked

        }