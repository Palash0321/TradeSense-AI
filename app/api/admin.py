from fastapi import APIRouter
from app.services.scan_engine import scan_market

router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)

@router.get("/scan")
def run_scan(
    market: str = "india",
    limit: int = 20
):

    scan_market(market, limit)

    return {
        "status": "success",
        "message": f"{market} scan completed",
        "stocks_scanned": limit
    }