from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.transaction import Transaction
from app.models.portfolio import Portfolio


router = APIRouter(
    prefix="/api/transactions",
    tags=["Transactions"]
)


class TransactionCreate(BaseModel):
    symbol: str
    transaction_type: str
    quantity: float
    price: float


@router.post("/")
def add_transaction(
    transaction: TransactionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    symbol = transaction.symbol.upper()

    tx_type = transaction.transaction_type.upper()

    new_transaction = Transaction(
        user_id=current_user.id,
        symbol=symbol,
        transaction_type=tx_type,
        quantity=transaction.quantity,
        price=transaction.price,
    )

    db.add(new_transaction)

    holding = (
        db.query(Portfolio)
        .filter(
            Portfolio.user_id == current_user.id,
            Portfolio.symbol == symbol,
        )
        .first()
    )

    # ==========================
    # BUY
    # ==========================

    if tx_type == "BUY":

        if holding:

            total_cost = (
                holding.quantity * holding.buy_price
            ) + (
                transaction.quantity * transaction.price
            )

            total_quantity = (
                holding.quantity
                + transaction.quantity
            )

            holding.buy_price = (
                total_cost / total_quantity
            )

            holding.quantity = total_quantity

        else:

            holding = Portfolio(

                user_id=current_user.id,

                symbol=symbol,

                quantity=transaction.quantity,

                buy_price=transaction.price,

            )

            db.add(holding)

    # ==========================
    # SELL
    # ==========================

    elif tx_type == "SELL":

        if holding is None:

            db.rollback()

            return {

                "error":
                "No holdings available"

            }

        if transaction.quantity > holding.quantity:

            db.rollback()

            return {

                "error":
                "Not enough quantity"

            }

        average_price = holding.buy_price

    realized_profit = (
        transaction.price
        - average_price
    ) * transaction.quantity

    new_transaction.realized_profit = realized_profit

    holding.quantity -= transaction.quantity

    if holding.quantity == 0:

        db.delete(holding)

    else:

        db.rollback()

        return {

            "error":
            "Transaction type must be BUY or SELL"

        }

    db.commit()

    return {

        "message":
        "Transaction completed successfully"

    }


@router.get("/")
def get_transactions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    transactions = (
        db.query(Transaction)
        .filter(
            Transaction.user_id == current_user.id
        )
        .order_by(
            Transaction.transaction_date.desc()
        )
        .all()
    )

    return [

        {

            "id": t.id,

            "symbol": t.symbol,

            "type": t.transaction_type,

            "quantity": t.quantity,

            "price": t.price,

            "realized_profit": round(
    t.realized_profit or 0,
    2,
),

            "date": t.transaction_date,

        }

        for t in transactions

    ]