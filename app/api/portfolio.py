from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.portfolio import Portfolio
from app.services.portfolio_service import (
    calculate_portfolio_metrics,
)
from app.services.market_price_service import (
    get_live_price,
)

router = APIRouter(
    prefix="/api/portfolio",
    tags=["Portfolio"]
)


class PortfolioCreate(BaseModel):
    symbol: str
    quantity: float
    buy_price: float



@router.get("/")
def get_portfolio(
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

    results = []

    for holding in holdings:

        current_price = get_live_price(
            holding.symbol
        )

        holding.current_price = current_price

        db.commit()

        db.refresh(holding)

        results.append(
            calculate_portfolio_metrics(
                holding
            )
        )

    return results


@router.post("/")
def add_portfolio(
    holding: PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    stock = Portfolio(
        user_id=current_user.id,
        symbol=holding.symbol.upper(),
        quantity=holding.quantity,
        buy_price=holding.buy_price,
    )

    db.add(stock)
    db.commit()
    db.refresh(stock)

    return {
        "message": "Portfolio holding added successfully",
        "id": stock.id,
    }


@router.delete("/{holding_id}")
def delete_portfolio(
    holding_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    stock = (
        db.query(Portfolio)
        .filter(
            Portfolio.id == holding_id,
            Portfolio.user_id == current_user.id,
        )
        .first()
    )

    if stock is None:
        raise HTTPException(
            status_code=404,
            detail="Holding not found",
        )

    db.delete(stock)
    db.commit()

    return {
        "message": "Holding deleted successfully"
    }