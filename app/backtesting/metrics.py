from dataclasses import dataclass


@dataclass
class Metrics:

    total_trades: int

    winning_trades: int

    losing_trades: int

    win_rate: float

    total_profit: float

    total_loss: float

    profit_factor: float

    max_drawdown: float

    expectancy: float

    average_rr: float