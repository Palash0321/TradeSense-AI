from dataclasses import dataclass

from .metrics import Metrics

from .trade import Trade


@dataclass
class BacktestReport:

    metrics: Metrics

    trades: list[Trade]