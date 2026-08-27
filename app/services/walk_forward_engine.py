from app.services.strategy_optimizer import StrategyOptimizer
from app.services.backtest_service import BacktestService


class WalkForwardEngine:

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
    # Optimize Training Period
    # =====================================

    def optimize_training_period(
        self,
        start_date,
        end_date,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        trailing_activation_atr=1.0,
        momentum_min=None,
        use_market_regime=True,
        initial_stop_atr=2.0,
        risk_per_trade=0.01
    ):

        optimizer = StrategyOptimizer(

            symbol=self.symbol,

            initial_capital=self.initial_capital,

            brokerage=self.brokerage,

            slippage=self.slippage

        )

        # =================================
        # IMPORTANT:
        # Optimize ONLY on training data
        # =================================

        results = optimizer.optimize(

            adx_values=adx_values,

            ema_gap_values=ema_gap_values,

            trailing_atr_values=trailing_atr_values,

            start_date=start_date,

            end_date=end_date,

            trailing_activation_atr=trailing_activation_atr,

            momentum_min=momentum_min,

            use_market_regime=use_market_regime,

            initial_stop_atr=initial_stop_atr,

            risk_per_trade=risk_per_trade

        )

        # =================================
        # Rank training-period results
        # =================================

        ranked = optimizer.rank_results(

            results,

            min_trades=10

        )

        return ranked

    # =====================================
    # Test Selected Parameters
    # =====================================

    def test_period(
        self,
        start_date,
        end_date,
        parameters,
        use_market_regime=True,
        initial_stop_atr=2.0,
        risk_per_trade=0.01,
        return_trades=False
    ):

        service = BacktestService(

            symbol=self.symbol,

            strategy="ema_atr",

            initial_capital=self.initial_capital,

            brokerage=self.brokerage,

            slippage=self.slippage

        )

        metrics = service.performance_metrics(

            start_date=start_date,

            end_date=end_date,

            adx_min=parameters["adx_min"],

            ema_gap_min=parameters["ema_gap_min"],

            trailing_atr=parameters["trailing_atr"],

            trailing_activation_atr=parameters["trailing_activation_atr"],

            momentum_min=parameters["momentum_min"],

            use_market_regime=use_market_regime,

            initial_stop_atr=initial_stop_atr,

            risk_per_trade=risk_per_trade

        )

        result = {

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

        }

        if return_trades:

            backtest = service.run_backtest_v3(

                start_date=start_date,

                end_date=end_date,

                adx_min=parameters["adx_min"],

                ema_gap_min=parameters["ema_gap_min"],

                trailing_atr=parameters["trailing_atr"],

                trailing_activation_atr=(
                    parameters[
                        "trailing_activation_atr"
                    ]
                ),

                momentum_min=parameters[
                    "momentum_min"
                ],

                use_market_regime=use_market_regime,

                initial_stop_atr=initial_stop_atr,

                risk_per_trade=risk_per_trade

            )

            result["trades"] = (
                backtest.get("trades", [])
            )

        return result

    # =====================================
    # Single Walk-Forward Window
    # =====================================

    def run_window(
        self,
        train_start,
        train_end,
        test_start,
        test_end,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        trailing_activation_atr=1.0,
        momentum_min=None,
        use_market_regime=True,
        initial_stop_atr=2.0,
        risk_per_trade=0.01
    ):

        training_results = (
            self.optimize_training_period(

                start_date=train_start,

                end_date=train_end,

                adx_values=adx_values,

                ema_gap_values=ema_gap_values,

                trailing_atr_values=trailing_atr_values,

                trailing_activation_atr=trailing_activation_atr,

                momentum_min=momentum_min,

                use_market_regime=use_market_regime,

                initial_stop_atr=initial_stop_atr,

                risk_per_trade=risk_per_trade

            )
        )

        if not training_results:

            return {

                "train_start":
                    train_start,

                "train_end":
                    train_end,

                "test_start":
                    test_start,

                "test_end":
                    test_end,

                "best_parameters":
                    None,

                "training":
                    None,

                "testing":
                    None,

                "status":
                    "EXCLUDED",

                "exclusion_reason":
                    "MINIMUM_TRADE_REQUIREMENT_NOT_MET"

            }

        best = training_results[0]

        parameters = {

            "adx_min":
                best["adx_min"],

            "ema_gap_min":
                best["ema_gap_min"],

            "trailing_atr":
                best["trailing_atr"],

            "trailing_activation_atr":
                trailing_activation_atr,

            "momentum_min":
                best["momentum_min"]

        }

        testing = self.test_period(

            start_date=test_start,

            end_date=test_end,

            parameters=parameters,

            use_market_regime=use_market_regime,

            initial_stop_atr=initial_stop_atr,

            risk_per_trade=risk_per_trade

        )

        return {

            "train_start":
                train_start,

            "train_end":
                train_end,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "best_parameters":
                parameters,

            "training": {

                "trades":
                    best["total_trades"],

                "profit":
                    best["net_profit"],

                "return":
                    best["total_return"],

                "profit_factor":
                    best["profit_factor"],

                "drawdown":
                    best["max_drawdown"]

            },

            "testing":
                testing

        }

    # =====================================
    # Complete Walk-Forward Test
    # =====================================

    def run(
        self,
        windows=None,
        adx_values=None,
        ema_gap_values=None,
        trailing_atr_values=None,
        trailing_activation_atr=1.0,
        momentum_min=None,
        use_market_regime=True,
        initial_stop_atr=2.0,
        risk_per_trade=0.01
    ):

        if windows is None:

            windows = [

                {

                    "train_start":
                        "2021-01-01",

                    "train_end":
                        "2023-12-31",

                    "test_start":
                        "2024-01-01",

                    "test_end":
                        "2024-12-31"

                },

                {

                    "train_start":
                        "2022-01-01",

                    "train_end":
                        "2024-12-31",

                    "test_start":
                        "2025-01-01",

                    "test_end":
                        "2025-12-31"

                },

                {

                    "train_start":
                        "2023-01-01",

                    "train_end":
                        "2025-12-31",

                    "test_start":
                        "2026-01-01",

                    "test_end":
                        "2026-12-31"

                }

            ]

        results = []

        for window in windows:

            result = self.run_window(

                train_start=
                    window["train_start"],

                train_end=
                    window["train_end"],

                test_start=
                    window["test_start"],

                test_end=
                    window["test_end"],

                adx_values=
                    adx_values,

                ema_gap_values=
                    ema_gap_values,

                trailing_atr_values=
                    trailing_atr_values,

                trailing_activation_atr=trailing_activation_atr,

                momentum_min=
                    momentum_min,

                use_market_regime=
                    use_market_regime,

                initial_stop_atr=
                    initial_stop_atr,

                risk_per_trade=
                    risk_per_trade

            )

            results.append(result)

        # =================================
        # Aggregate Out-of-Sample Results
        # =================================

        valid_tests = [

            result["testing"]

            for result in results

            if result["testing"] is not None

        ]

        total_profit = sum(

            result["net_profit"]

            for result in valid_tests

        )

        total_trades = sum(

            result["total_trades"]

            for result in valid_tests

        )

        average_return = 0

        if valid_tests:

            average_return = (

                sum(
                    result["total_return"]
                    for result in valid_tests
                )

                / len(valid_tests)

            )

        profitable_windows = sum(

            1

            for result in valid_tests

            if result["net_profit"] > 0

        )

        return {

            "symbol":
                self.symbol,

            "windows":
                results,

            "out_of_sample": {

                "windows":
                    len(valid_tests),

                "total_trades":
                    total_trades,

                "total_profit":
                    round(
                        total_profit,
                        2
                    ),

                "average_return":
                    round(
                        average_return,
                        2
                    ),

                "profitable_windows":
                    profitable_windows,

                "positive_window_ratio":
                    round(

                        (
                            profitable_windows
                            / len(valid_tests)
                        ) * 100,

                        2

                    )
                    if valid_tests
                    else 0

            }

        }