from app.services.walk_forward_engine import WalkForwardEngine


class ParameterRobustness:

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
        config
    ):

        stock_results = []

        for symbol in self.symbols:

            engine = WalkForwardEngine(

                symbol=symbol,

                initial_capital=self.initial_capital,

                brokerage=self.brokerage,

                slippage=self.slippage

            )

            result = engine.run(

                adx_values=[
                    config["adx_min"]
                ],

                ema_gap_values=[
                    config["ema_gap_min"]
                ],

                trailing_atr_values=[
                    config["trailing_atr"]
                ]

            )

            oos = result[
                "out_of_sample"
            ]

            stock_results.append({

                "symbol":
                    symbol,

                "total_profit":
                    oos["total_profit"],

                "average_return":
                    oos["average_return"],

                "total_trades":
                    oos["total_trades"],

                "profitable_windows":
                    oos["profitable_windows"],

                "total_windows":
                    oos["windows"],

                "positive_window_ratio":
                    oos[
                        "positive_window_ratio"
                    ]

            })

        # =================================
        # Aggregate
        # =================================

        total_profit = sum(

            x["total_profit"]

            for x in stock_results

        )

        total_trades = sum(

            x["total_trades"]

            for x in stock_results

        )

        profitable_stocks = sum(

            1

            for x in stock_results

            if x["total_profit"] > 0

        )

        positive_windows = sum(

            x["profitable_windows"]

            for x in stock_results

        )

        total_windows = sum(

            x["total_windows"]

            for x in stock_results

        )

        average_return = 0

        if stock_results:

            average_return = (

                sum(
                    x["average_return"]
                    for x in stock_results
                )
                / len(stock_results)

            )

        positive_window_ratio = 0

        if total_windows > 0:

            positive_window_ratio = (

                positive_windows
                / total_windows

            ) * 100

        return {

            "configuration":
                config,

            "stocks":
                stock_results,

            "summary": {

                "total_profit":
                    round(
                        total_profit,
                        2
                    ),

                "total_trades":
                    total_trades,

                "profitable_stocks":
                    profitable_stocks,

                "stock_profit_ratio":
                    round(

                        (
                            profitable_stocks
                            / len(stock_results)
                        ) * 100,

                        2

                    )
                    if stock_results
                    else 0,

                "average_return":
                    round(
                        average_return,
                        2
                    ),

                "positive_windows":
                    positive_windows,

                "total_windows":
                    total_windows,

                "positive_window_ratio":
                    round(
                        positive_window_ratio,
                        2
                    )

            }

        }

    # =====================================
    # Compare Configurations
    # =====================================

    def compare(
        self,
        configurations
    ):

        results = []

        for config in configurations:

            result = (
                self.test_configuration(
                    config
                )
            )

            results.append(result)

        # =================================
        # Rank by OOS robustness
        # =================================

        for result in results:

            summary = result[
                "summary"
            ]

            score = (

                summary[
                    "average_return"
                ]
                * 0.30

                +

                summary[
                    "stock_profit_ratio"
                ]
                * 0.20

                +

                summary[
                    "positive_window_ratio"
                ]
                * 0.30

                +

                (
                    max(
                        summary[
                            "total_profit"
                        ],
                        0
                    )
                    / self.initial_capital
                )
                * 100
                * 0.20

            )

            result[
                "robustness_score"
            ] = round(
                score,
                4
            )

        results.sort(

            key=lambda x:
                x["robustness_score"],

            reverse=True

        )

        return results