from app.services.walk_forward_engine import WalkForwardEngine


class WalkForwardRobustness:

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
    # Run One Symbol
    # =====================================

    def run_symbol(
        self,
        symbol,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        trailing_activation_atr=1.0,
        momentum_min=None
    ):

        engine = WalkForwardEngine(

            symbol=symbol,

            initial_capital=self.initial_capital,

            brokerage=self.brokerage,

            slippage=self.slippage

        )

        result = engine.run(

            adx_values=adx_values,

            ema_gap_values=ema_gap_values,

            trailing_atr_values=trailing_atr_values,

            trailing_activation_atr=trailing_activation_atr,

            momentum_min=momentum_min

        )

        return result

    # =====================================
    # Run All Symbols
    # =====================================

    def run(
        self,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        trailing_activation_atr=1.0,
        momentum_min=None
    ):

        results = []

        for symbol in self.symbols:

            result = self.run_symbol(

                symbol=symbol,

                adx_values=adx_values,

                ema_gap_values=ema_gap_values,

                trailing_atr_values=trailing_atr_values,

                trailing_activation_atr=trailing_activation_atr,

                momentum_min=momentum_min

            )

            oos = result["out_of_sample"]

            results.append({

                "symbol":
                    symbol,

                "oos_trades":
                    oos["total_trades"],

                "oos_profit":
                    oos["total_profit"],

                "average_oos_return":
                    oos["average_return"],

                "profitable_windows":
                    oos["profitable_windows"],

                "total_windows":
                    len(result["windows"]),

                "positive_window_ratio":
                    oos["positive_window_ratio"],

                "windows":
                    result["windows"]

            })

        # =================================
        # Aggregate
        # =================================

        total_profit = sum(

            x["oos_profit"]

            for x in results

        )

        total_trades = sum(

            x["oos_trades"]

            for x in results

        )

        average_return = 0

        if results:

            average_return = (

                sum(
                    x["average_oos_return"]
                    for x in results
                )

                / len(results)

            )

        profitable_stocks = sum(

            1

            for x in results

            if x["oos_profit"] > 0

        )

        positive_windows = sum(

            x["profitable_windows"]

            for x in results

        )

        total_windows = sum(

            x["total_windows"]

            for x in results

        )

        positive_window_ratio = 0

        if total_windows > 0:

            positive_window_ratio = (

                positive_windows
                / total_windows
            ) * 100

        return {

            "symbols":
                results,

            "summary": {

                "total_stocks":
                    len(results),

                "profitable_stocks":
                    profitable_stocks,

                "stock_profit_ratio":
                    round(

                        (
                            profitable_stocks
                            / len(results)
                        ) * 100,

                        2

                    )
                    if results
                    else 0,

                "total_oos_trades":
                    total_trades,

                "total_oos_profit":
                    round(
                        total_profit,
                        2
                    ),

                "average_oos_return":
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