from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.auth.dependencies import get_current_user

from app.models.user import User
from app.models.paper_account import PaperAccount

router = APIRouter(
    prefix="/api/paper",
    tags=["Paper Trading"]
)


@router.get("/account")
def get_account(
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

    if account is None:

        account = PaperAccount(
            user_id=current_user.id,
            balance=1000000.0,
        )

        db.add(account)
        db.commit()
        db.refresh(account)

    return {

        "balance": account.balance

    }