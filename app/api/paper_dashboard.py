from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio

from app.services.market_price_service import get_live_price

router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"],
)


@router.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    account = (
        db.query(PaperAccount)
        .filter(
            PaperAccount.user_id == current_user.id
        )
        .first()
    )

    holdings = (
        db.query(PaperPortfolio)
        .filter(
            PaperPortfolio.user_id == current_user.id
        )
        .all()
    )

    investment = 0
    value = 0

    best_stock = None
    best_return = -999999

    for stock in holdings:

        current = get_live_price(stock.symbol)

        stock.current_price = current

        inv = stock.average_price * stock.quantity
        val = current * stock.quantity

        investment += inv
        value += val

        pnl = val - inv

        if pnl > best_return:

            best_return = pnl
            best_stock = stock.symbol

    db.commit()

    return {

        "cash_balance": round(
            account.balance,
            2,
        ),

        "investment": round(
            investment,
            2,
        ),

        "portfolio_value": round(
            value,
            2,
        ),

        "total_value": round(
            value + account.balance,
            2,
        ),

        "profit_loss": round(
            value - investment,
            2,
        ),

        "best_stock": best_stock,

        "holdings": len(holdings),

    }