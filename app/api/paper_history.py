from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.paper_transaction import PaperTransaction

router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"],
)


@router.get("/history")
def history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):

    trades = (
        db.query(PaperTransaction)
        .filter(
            PaperTransaction.user_id == current_user.id
        )
        .order_by(
            PaperTransaction.created_at.desc()
        )
        .all()
    )

    return [
        {
            "symbol": t.symbol,
            "type": t.transaction_type,
            "quantity": t.quantity,
            "price": round(t.price, 2),
            "total": round(t.total_amount, 2),
            "date": t.created_at,
        }
        for t in trades
    ]