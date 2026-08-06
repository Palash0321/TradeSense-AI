from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio

from app.services.market_price_service import get_live_price
from app.models.paper_transaction import PaperTransaction


router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"],
)


class PaperBuy(BaseModel):

    symbol: str
    quantity: float


@router.post("/buy")
def paper_buy(
    order: PaperBuy,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    symbol = order.symbol.upper()

    price = get_live_price(symbol)

    if price <= 0:

        raise HTTPException(
            status_code=400,
            detail="Unable to fetch live price",
        )

    account = (
        db.query(PaperAccount)
        .filter(
            PaperAccount.user_id == current_user.id
        )
        .first()
    )

    if account is None:

        raise HTTPException(
            status_code=404,
            detail="Paper account not found",
        )

    total_cost = price * order.quantity

    if total_cost > account.balance:

        raise HTTPException(
            status_code=400,
            detail="Insufficient paper balance",
        )

    account.balance -= total_cost

    paper_tx = PaperTransaction(
    user_id=current_user.id,
    symbol=symbol,
    transaction_type="BUY",
    quantity=order.quantity,
    price=price,
    total_amount=total_cost,
)

    db.add(paper_tx)

    holding = (
        db.query(PaperPortfolio)
        .filter(
            PaperPortfolio.user_id == current_user.id,
            PaperPortfolio.symbol == symbol,
        )
        .first()
    )

    if holding:

        total_qty = (
            holding.quantity
            + order.quantity
        )

        total_value = (
            holding.quantity
            * holding.average_price
        ) + total_cost

        holding.quantity = total_qty

        holding.average_price = (
            total_value / total_qty
        )

        holding.current_price = price

    else:

        holding = PaperPortfolio(

            user_id=current_user.id,

            symbol=symbol,

            quantity=order.quantity,

            average_price=price,

            current_price=price,

        )

        db.add(holding)

    db.commit()

    return {

        "message": "Paper trade executed",

        "balance": round(account.balance, 2),

        "price": round(price, 2),

    }

@router.post("/sell")
def paper_sell(
    order: PaperBuy,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    symbol = order.symbol.upper()

    price = get_live_price(symbol)

    if price <= 0:

        raise HTTPException(
            status_code=400,
            detail="Unable to fetch live price",
        )

    account = (
        db.query(PaperAccount)
        .filter(
            PaperAccount.user_id == current_user.id
        )
        .first()
    )

    if account is None:

        raise HTTPException(
            status_code=404,
            detail="Paper account not found",
        )

    holding = (
        db.query(PaperPortfolio)
        .filter(
            PaperPortfolio.user_id == current_user.id,
            PaperPortfolio.symbol == symbol,
        )
        .first()
    )

    if holding is None:

        raise HTTPException(
            status_code=404,
            detail="Stock not found",
        )

    if order.quantity > holding.quantity:

        raise HTTPException(
            status_code=400,
            detail="Not enough quantity",
        )

    sale_value = price * order.quantity

    account.balance += sale_value

    paper_tx = PaperTransaction(
    user_id=current_user.id,
    symbol=symbol,
    transaction_type="SELL",
    quantity=order.quantity,
    price=price,
    total_amount=sale_value,
)

    db.add(paper_tx)

    holding.quantity -= order.quantity

    holding.current_price = price

    if holding.quantity == 0:

        db.delete(holding)

    db.commit()

    return {

        "message": "Paper sell executed",

        "balance": round(
            account.balance,
            2,
        ),

        "sell_price": round(
            price,
            2,
        ),

    }