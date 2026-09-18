import pytest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.paper_orders import router
from app.auth.dependencies import get_current_user
from app.core.database import Base, get_db
from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio
from app.models.paper_transaction import PaperTransaction
from app.models.user import User


TEST_DATABASE_URL = "sqlite://"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)


app = FastAPI()
app.include_router(router)



def create_test_session():
    return TestingSessionLocal()


def override_get_db():
    db = TestingSessionLocal()

    try:
        yield db
    finally:
        db.close()


def create_test_user(db):
    user = User(
        full_name="Paper API Test User",
        email="paper-api-test@example.com",
        password_hash="test-password",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def override_get_current_user():
    db = TestingSessionLocal()

    try:
        return db.query(User).filter(
            User.email == "paper-api-test@example.com"
        ).first()
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

@pytest.fixture(autouse=True)
def reset_database():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    yield

    Base.metadata.drop_all(bind=engine)


def test_api_buy_executes_successfully():
    

    db = create_test_session()

    try:
        user = create_test_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=100000.0,
        )

        db.add(account)
        db.commit()

    finally:
        db.close()

    with patch(
        "app.services.paper_execution_service.get_live_price",
        return_value=1500.0,
    ):
        client = TestClient(app)

        response = client.post(
            "/api/paper/buy",
            json={
                "symbol": "RELIANCE.NS",
                "quantity": 10,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Paper BUY executed"
    assert data["balance"] == 85000.0
    assert data["price"] == 1500.0

    db = create_test_session()

    try:
        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == user.id,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 10
        assert holding.average_price == 1500.0
        assert holding.current_price == 1500.0

        transaction = db.query(PaperTransaction).filter(
            PaperTransaction.user_id == user.id,
            PaperTransaction.symbol == "RELIANCE.NS",
            PaperTransaction.transaction_type == "BUY",
        ).first()

        assert transaction is not None
        assert transaction.quantity == 10
        assert transaction.price == 1500.0
        assert transaction.total_amount == 15000.0

    finally:
        db.close()


def test_api_sell_executes_successfully():
    

    db = create_test_session()

    try:
        user = create_test_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=85000.0,
        )

        holding = PaperPortfolio(
            user_id=user.id,
            symbol="RELIANCE.NS",
            quantity=10,
            average_price=1500.0,
            current_price=1500.0,
        )

        db.add(account)
        db.add(holding)
        db.commit()

    finally:
        db.close()

    with patch(
        "app.services.paper_execution_service.get_live_price",
        return_value=1600.0,
    ):
        client = TestClient(app)

        response = client.post(
            "/api/paper/sell",
            json={
                "symbol": "RELIANCE.NS",
                "quantity": 5,
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Paper SELL executed"
    assert data["balance"] == 93000.0
    assert data["sell_price"] == 1600.0

    db = create_test_session()

    try:
        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == user.id,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 5
        assert holding.average_price == 1500.0
        assert holding.current_price == 1600.0

        transaction = db.query(PaperTransaction).filter(
            PaperTransaction.user_id == user.id,
            PaperTransaction.symbol == "RELIANCE.NS",
            PaperTransaction.transaction_type == "SELL",
        ).first()

        assert transaction is not None
        assert transaction.quantity == 5
        assert transaction.price == 1600.0
        assert transaction.total_amount == 8000.0

    finally:
        db.close()
        


def test_api_buy_rejects_insufficient_balance():
    

    db = create_test_session()

    try:
        user = create_test_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=10000.0,
        )

        db.add(account)
        db.commit()

    finally:
        db.close()

    with patch(
        "app.services.paper_execution_service.get_live_price",
        return_value=1500.0,
    ):
        client = TestClient(app)

        response = client.post(
            "/api/paper/buy",
            json={
                "symbol": "RELIANCE.NS",
                "quantity": 10,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Insufficient paper balance"

    db = create_test_session()

    try:
        account = db.query(PaperAccount).filter(
            PaperAccount.user_id == user.id,
        ).first()

        assert account is not None
        assert account.balance == 10000.0

        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == user.id,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is None

    finally:
        db.close()


def test_api_sell_rejects_insufficient_quantity():
    

    db = create_test_session()

    try:
        user = create_test_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=85000.0,
        )

        holding = PaperPortfolio(
            user_id=user.id,
            symbol="RELIANCE.NS",
            quantity=5,
            average_price=1500.0,
            current_price=1500.0,
        )

        db.add(account)
        db.add(holding)
        db.commit()

    finally:
        db.close()

    with patch(
        "app.services.paper_execution_service.get_live_price",
        return_value=1600.0,
    ):
        client = TestClient(app)

        response = client.post(
            "/api/paper/sell",
            json={
                "symbol": "RELIANCE.NS",
                "quantity": 10,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Not enough quantity"

    db = create_test_session()

    try:
        account = db.query(PaperAccount).filter(
            PaperAccount.user_id == user.id,
        ).first()

        assert account is not None
        assert account.balance == 85000.0

        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == user.id,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 5

    finally:
        db.close()
        