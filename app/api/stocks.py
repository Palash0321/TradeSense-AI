from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.services.signal_service import generate_signal

from app.services.stock_service import (
    get_stock_price,
    get_stock_history,
)

from app.services.chart_service import create_stock_chart

router = APIRouter()


@router.get("/stock/{symbol}")
def stock(symbol: str):
    return get_stock_price(symbol)


@router.get("/history/{symbol}")
def history(symbol: str, period: str = "6mo"):
    return get_stock_history(symbol, period)


@router.get("/chart/{symbol}", response_class=HTMLResponse)
def chart(symbol: str, period: str = "6mo"):
    return create_stock_chart(symbol, period)

@router.get("/signal/{symbol}")
def signal(symbol: str, period: str = "6mo"):
    return generate_signal(symbol, period)