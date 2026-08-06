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


@router.get("/portfolio")
def paper_portfolio(
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

    result = []

    for stock in holdings:

        current = get_live_price(stock.symbol)

        stock.current_price = current

        investment = (
            stock.average_price
            * stock.quantity
        )

        value = (
            current
            * stock.quantity
        )

        result.append({

            "symbol": stock.symbol,

            "quantity": stock.quantity,

            "average_price": round(
                stock.average_price,
                2,
            ),

            "current_price": round(
                current,
                2,
            ),

            "investment": round(
                investment,
                2,
            ),

            "current_value": round(
                value,
                2,
            ),

            "profit_loss": round(
                value - investment,
                2,
            ),

        })

    db.commit()

    return result