from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.user import User

from app.auth.schemas import UserRegister
from app.auth.hashing import hash_password

from app.auth.schemas import UserLogin
from app.auth.hashing import verify_password
from app.auth.jwt_handler import create_access_token

from fastapi import Depends
from app.auth.dependencies import get_current_user

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post("/register")
def register(user: UserRegister):

    db: Session = SessionLocal()

    existing = db.query(User).filter(User.email == user.email).first()

    if existing:
        db.close()
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_user = User(
        full_name=user.full_name,
        email=user.email,
        password_hash=hash_password(user.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    db.close()

    return {
        "message": "Registration Successful",
        "user_id": new_user.id
    }

@router.post("/login")
def login(user: UserLogin):

    db: Session = SessionLocal()

    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user:
        db.close()
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(user.password, db_user.password_hash):
        db.close()
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        data={
            "sub": str(db_user.id),
            "email": db_user.email
        }
    )

    db.close()

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": db_user.id,
            "full_name": db_user.full_name,
            "email": db_user.email
        }
    }



@router.get("/me")
def me(current_user=Depends(get_current_user)):

    return {
        "id": current_user.id,
        "full_name": current_user.full_name,
        "email": current_user.email
    }