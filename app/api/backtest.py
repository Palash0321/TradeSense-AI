from fastapi import APIRouter

from app.services.backtest_service import BacktestService

router = APIRouter(
    prefix="/api/backtest",
    tags=["Backtesting"]
)


@router.get("/")
def run_backtest(

    symbol: str,

    brokerage: float = 20,

    slippage: float = 0.10,

    capital: float = 100000

):

    service = BacktestService(

        symbol=symbol,

        brokerage=brokerage,

        slippage=slippage,

        initial_capital=capital

    )

    return service.performance_metrics()