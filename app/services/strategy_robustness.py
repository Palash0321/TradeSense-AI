from app.services.backtest_service import BacktestService


class StrategyRobustness:

    def __init__(
        self,
        symbols,
        initial_capital=100000,
        brokerage=20,
        slippage=0.10
    ):

        self.symbols = symbols

        self.initial_capital = (
            initial_capital
        )

        self.brokerage = brokerage

        self.slippage = slippage

    # =====================================
    # Test One Configuration
    # =====================================

    def test_configuration(
        self,
        adx_min,
        ema_gap_min,
        trailing_atr,
        trailing_activation_atr=1.0,
        initial_stop_atr=2.0,
        momentum_min=None,
        use_market_regime=True
    ):

        results = []

        for symbol in self.symbols:

            service = BacktestService(

                symbol=symbol,

                strategy="ema_atr",

                brokerage=self.brokerage,

                slippage=self.slippage,

                initial_capital=self.initial_capital

            )

            metrics = service.performance_metrics(

                adx_min=adx_min,

                ema_gap_min=ema_gap_min,

                trailing_atr=trailing_atr,

                trailing_activation_atr=
                    trailing_activation_atr,

                initial_stop_atr=
                    initial_stop_atr,

                momentum_min=
                    momentum_min,

                use_market_regime=
                    use_market_regime

            )

            results.append({

                "symbol":
                    symbol,

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
    # Aggregate Configuration
    # =====================================

    def score_configuration(
        self,
        results
    ):

        if not results:

            return None

        returns = [
            r["total_return"]
            for r in results
        ]

        profit_factors = [
            r["profit_factor"]
            for r in results
        ]

        drawdowns = [
            r["max_drawdown"]
            for r in results
        ]

        positive_returns = [
            r
            for r in returns
            if r > 0
        ]

        profitable_stocks = len(
            positive_returns
        )

        average_return = (
            sum(returns)
            / len(returns)
        )

        average_pf = (
            sum(profit_factors)
            / len(profit_factors)
        )

        average_drawdown = (
            sum(drawdowns)
            / len(drawdowns)
        )

        median_return = sorted(
            returns
        )[len(returns) // 2]

        # ---------------------------------
        # Robustness Score
        # ---------------------------------

        score = (

            average_return
            * 0.35

            +

            average_pf
            * 20
            * 0.30

            +

            (
                profitable_stocks
                / len(results)
            )
            * 20
            * 0.20

            -

            average_drawdown
            * 0.15

        )

        return {

            "average_return":
                round(
                    average_return,
                    2
                ),

            "median_return":
                round(
                    median_return,
                    2
                ),

            "average_profit_factor":
                round(
                    average_pf,
                    2
                ),

            "average_drawdown":
                round(
                    average_drawdown,
                    2
                ),

            "profitable_stocks":
                profitable_stocks,

            "total_stocks":
                len(results),

            "positive_stock_ratio":
                round(
                    (
                        profitable_stocks
                        / len(results)
                    ) * 100,
                    2
                ),

            "robustness_score":
                round(
                    score,
                    4
                )

        }

    # =====================================
    # Compare Configurations
    # =====================================

    def compare(
        self,
        configurations
    ):

        candidates = []

        for configuration in configurations:

            adx_min = (
                configuration["adx_min"]
            )

            ema_gap_min = (
                configuration["ema_gap_min"]
            )

            trailing_atr = (
                configuration["trailing_atr"]
            )

            results = (
                self.test_configuration(

                    adx_min=adx_min,

                    ema_gap_min=ema_gap_min,

                    trailing_atr=trailing_atr

                )
            )

            score = (
                self.score_configuration(
                    results
                )
            )

            candidates.append({

                "adx_min":
                    adx_min,

                "ema_gap_min":
                    ema_gap_min,

                "trailing_atr":
                    trailing_atr,

                "summary":
                    score,

                "stocks":
                    results

            })

        candidates.sort(

            key=lambda x:
                x["summary"]
                ["robustness_score"],

            reverse=True

        )

        return candidates