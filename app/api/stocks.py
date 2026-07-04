from fastapi import APIRouter
from app.services.stock_service import get_stock_price

router = APIRouter()

@router.get("/stock/{symbol}")
def stock(symbol: str):
    return get_stock_price(symbol)