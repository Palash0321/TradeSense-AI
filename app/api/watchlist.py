from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user
from app.models.user import User
from app.models.watchlist import Watchlist

router = APIRouter(
    prefix="/api/watchlist",
    tags=["Watchlist"]
)


@router.get("/")
def get_watchlist(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    stocks = (
        db.query(Watchlist)
        .filter(Watchlist.user_id == current_user.id)
        .all()
    )

    return stocks


@router.post("/{symbol}")
def add_stock(
    symbol: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    symbol = symbol.upper()

    existing = (
        db.query(Watchlist)
        .filter(
            Watchlist.user_id == current_user.id,
            Watchlist.symbol == symbol,
        )
        .first()
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Stock already exists",
        )

    stock = Watchlist(
        user_id=current_user.id,
        symbol=symbol,
    )

    db.add(stock)
    db.commit()

    return {
        "message": "Stock added successfully"
    }


@router.delete("/{symbol}")
def delete_stock(
    symbol: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    stock = (
        db.query(Watchlist)
        .filter(
            Watchlist.user_id == current_user.id,
            Watchlist.symbol == symbol.upper(),
        )
        .first()
    )

    if stock is None:
        raise HTTPException(
            status_code=404,
            detail="Stock not found",
        )

    db.delete(stock)
    db.commit()

    return {
        "message": "Stock removed successfully"
    }