from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.auth.jwt_handler import verify_access_token
from app.models.user import User

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):

    token = credentials.credentials

    print("=" * 60)
    print("TOKEN RECEIVED:")
    print(token)
    print("=" * 60)

    payload = verify_access_token(token)

    print("PAYLOAD:", payload)


    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    db: Session = SessionLocal()

    try:
        user = (
            db.query(User)
            .filter(User.id == int(payload["sub"]))
            .first()
        )

        if user is None:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return user

    finally:
        db.close()