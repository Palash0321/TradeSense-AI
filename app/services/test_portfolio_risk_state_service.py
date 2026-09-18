from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.user import User
from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio

from app.services.portfolio_risk_state_service import (
    PortfolioRiskStateService,
)


def create_test_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    return SessionLocal()


def create_user(db):
    user = User(
        email="portfolio-risk-test@example.com",
        password_hash="test-password",
        full_name="Portfolio Risk Test",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def test_account_only_state():
    db = create_test_session()

    try:
        user = create_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=1000000.0,
        )

        db.add(account)
        db.commit()

        service = PortfolioRiskStateService(
            db=db,
            user_id=user.id,
        )

        result = service.get_state()

        assert result["status"] == "PASS"
        assert result["passed"] is True

        assert result["cash_balance"] == 1000000.0
        assert result["holdings_market_value"] == 0.0
        assert result["gross_exposure"] == 0.0
        assert result["equity"] == 1000000.0
        assert result["holdings_count"] == 0

        assert result["open_risk_known"] is False
        assert result["open_risk"] is None

        assert result["risk_budget_policy_defined"] is False
        assert result["risk_budget"] is None

    finally:
        db.close()


def test_portfolio_exposure_and_equity():
    db = create_test_session()

    try:
        user = create_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=900000.0,
        )

        holding = PaperPortfolio(
            user_id=user.id,
            symbol="RELIANCE.NS",
            quantity=100,
            average_price=900.0,
            current_price=1000.0,
        )

        db.add(account)
        db.add(holding)
        db.commit()

        service = PortfolioRiskStateService(
            db=db,
            user_id=user.id,
        )

        result = service.get_state()

        assert result["status"] == "PASS"

        assert result["cash_balance"] == 900000.0
        assert result["holdings_market_value"] == 100000.0
        assert result["gross_exposure"] == 100000.0
        assert result["equity"] == 1000000.0
        assert result["holdings_count"] == 1

    finally:
        db.close()


def test_multiple_holdings_are_aggregated():
    db = create_test_session()

    try:
        user = create_user(db)

        account = PaperAccount(
            user_id=user.id,
            balance=700000.0,
        )

        holding_one = PaperPortfolio(
            user_id=user.id,
            symbol="RELIANCE.NS",
            quantity=100,
            average_price=900.0,
            current_price=1000.0,
        )

        holding_two = PaperPortfolio(
            user_id=user.id,
            symbol="TCS.NS",
            quantity=50,
            average_price=3000.0,
            current_price=3200.0,
        )

        db.add(account)
        db.add(holding_one)
        db.add(holding_two)
        db.commit()

        service = PortfolioRiskStateService(
            db=db,
            user_id=user.id,
        )

        result = service.get_state()

        assert result["status"] == "PASS"

        assert result["holdings_market_value"] == 260000.0
        assert result["gross_exposure"] == 260000.0
        assert result["equity"] == 960000.0
        assert result["holdings_count"] == 2

    finally:
        db.close()


def test_missing_account_rejects():
    db = create_test_session()

    try:
        user = create_user(db)

        service = PortfolioRiskStateService(
            db=db,
            user_id=user.id,
        )

        result = service.get_state()

        assert result["status"] == "REJECT"
        assert result["passed"] is False

        assert result["cash_balance"] is None
        assert result["equity"] is None

        assert result["reasons"] == [
            "Paper account not found."
        ]

    finally:
        db.close()