from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.paper_portfolio import PaperPortfolio

from app.services.market_price_service import get_live_price

router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"],
)


@router.get("/analytics")
def analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    holdings = (
        db.query(PaperPortfolio)
        .filter(
            PaperPortfolio.user_id == current_user.id
        )
        .all()
    )

    best_stock = None
    worst_stock = None

    best_return = float("-inf")
    worst_return = float("inf")

    total_profit = 0

    for stock in holdings:

        current = get_live_price(stock.symbol)

        investment = (
            stock.average_price
            * stock.quantity
        )

        current_value = (
            current
            * stock.quantity
        )

        pnl = current_value - investment

        total_profit += pnl

        if pnl > best_return:

            best_return = pnl
            best_stock = stock.symbol

        if pnl < worst_return:

            worst_return = pnl
            worst_stock = stock.symbol

    return {

        "best_stock": best_stock,

        "best_profit": round(best_return, 2)
        if holdings else 0,

        "worst_stock": worst_stock,

        "worst_profit": round(worst_return, 2)
        if holdings else 0,

        "total_profit": round(total_profit, 2),

    }