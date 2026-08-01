from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:

    symbol: str

    entry_time: datetime

    entry_price: float

    stop_loss: float

    target1: float

    target2: float

    signal: str

    confidence: int

    trend_score: int

    momentum_score: int

    option_score: int

    exit_time: datetime | None = None

    exit_price: float | None = None

    pnl: float = 0

    status: str = "OPEN"

    outcome: str = "UNKNOWN"

    risk_reward: float = 0