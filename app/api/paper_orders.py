from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User

from app.services.paper_execution_service import PaperExecutionService


router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"],
)


class PaperBuy(BaseModel):
    symbol: str
    quantity: int


@router.post("/buy")
def paper_buy(
    order: PaperBuy,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PaperExecutionService(
        db=db,
        user_id=current_user.id,
    )

    result = service.execute_manual_buy(
        symbol=order.symbol,
        quantity=order.quantity,
    )

    if result["status"] == "REJECT":
        reason = result.get("reason", "Paper BUY rejected")

        if reason == "Paper account not found":
            raise HTTPException(
                status_code=404,
                detail=reason,
            )

        if reason == "Unable to fetch live price":
            raise HTTPException(
                status_code=400,
                detail=reason,
            )

        if reason == "Insufficient paper balance":
            raise HTTPException(
                status_code=400,
                detail=reason,
            )

        raise HTTPException(
            status_code=400,
            detail=reason,
        )

    return {
        "message": result["message"],
        "balance": result["remaining_balance"],
        "price": result["execution_price"],
    }


@router.post("/sell")
def paper_sell(
    order: PaperBuy,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PaperExecutionService(
        db=db,
        user_id=current_user.id,
    )

    result = service.execute_manual_sell(
        symbol=order.symbol,
        quantity=order.quantity,
    )

    if result["status"] == "REJECT":
        reason = result.get("reason", "Paper SELL rejected")

        if reason in (
            "Paper account not found",
            "Stock not found",
        ):
            raise HTTPException(
                status_code=404,
                detail=reason,
            )

        if reason == "Unable to fetch live price":
            raise HTTPException(
                status_code=400,
                detail=reason,
            )

        if reason == "Not enough quantity":
            raise HTTPException(
                status_code=400,
                detail=reason,
            )

        raise HTTPException(
            status_code=400,
            detail=reason,
        )

    return {
        "message": result["message"],
        "balance": result["remaining_balance"],
        "sell_price": result["execution_price"],
    }