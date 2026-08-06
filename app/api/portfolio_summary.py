from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.portfolio import Portfolio

router = APIRouter(
    prefix="/api/portfolio",
    tags=["Portfolio"]
)


@router.get("/summary")
def portfolio_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    holdings = (
        db.query(Portfolio)
        .filter(
            Portfolio.user_id == current_user.id
        )
        .all()
    )

    total_investment = 0
    total_current = 0

    for holding in holdings:

        investment = (
            holding.quantity
            * holding.buy_price
        )

        current = (
            holding.quantity
            * holding.current_price
        )

        total_investment += investment
        total_current += current

    total_profit = (
        total_current
        - total_investment
    )

    if total_investment == 0:

        total_return = 0

    else:

        total_return = (
            total_profit
            / total_investment
        ) * 100

    return {

        "total_investment": round(
            total_investment,
            2,
        ),

        "current_value": round(
            total_current,
            2,
        ),

        "profit_loss": round(
            total_profit,
            2,
        ),

        "return_percent": round(
            total_return,
            2,
        ),

        "holdings": len(holdings),
    }