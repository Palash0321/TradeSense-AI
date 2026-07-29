from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.market_service import get_market_indices

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


@router.get("/market")
async def market_page(request: Request):

    indices = get_market_indices()

    return templates.TemplateResponse(
        request=request,
        name="market.html",
        context={
            "request": request,
            "indices": indices
        }
    )