from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.user import User
from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio
from app.models.paper_transaction import PaperTransaction

from app.services.paper_execution_service import (
    PaperExecutionService,
)


TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def setup_database():
    Base.metadata.create_all(bind=engine)


def teardown_database():
    Base.metadata.drop_all(bind=engine)


def create_test_session():
    setup_database()
    return TestingSessionLocal()


def create_test_user(db):
    user = User(
        id=1,
        full_name="Test User",
        email="test@example.com",
        password_hash="test-password",
    )

    db.add(user)
    db.commit()

    return user


def test_market_long_buy_executes():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=100000.0,
        )

        db.add(account)
        db.commit()

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1000.0,
        ):
            result = PaperExecutionService(
                db=db,
                user_id=1,
            ).execute(order_intent)

        assert result["status"] == "PASS"
        assert result["executed"] is True
        assert result["symbol"] == "RELIANCE.NS"
        assert result["direction"] == "LONG"
        assert result["order_type"] == "MARKET"
        assert result["quantity"] == 10
        assert result["execution_price"] == 1000.0
        assert result["total_amount"] == 10000.0
        assert result["remaining_balance"] == 90000.0

        holding = (
            db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == 1,
                PaperPortfolio.symbol == "RELIANCE.NS",
            )
            .first()
        )

        assert holding is not None
        assert holding.quantity == 10
        assert holding.average_price == 1000.0
        assert holding.current_price == 1000.0

        transaction = (
            db.query(PaperTransaction)
            .filter(
                PaperTransaction.user_id == 1,
                PaperTransaction.symbol == "RELIANCE.NS",
            )
            .first()
        )

        assert transaction is not None
        assert transaction.transaction_type == "BUY"
        assert transaction.quantity == 10
        assert transaction.price == 1000.0
        assert transaction.total_amount == 10000.0

    finally:
        db.close()
        teardown_database()


def test_insufficient_balance_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=5000.0,
        )

        db.add(account)
        db.commit()

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1000.0,
        ):
            result = PaperExecutionService(
                db=db,
                user_id=1,
            ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == "Insufficient paper balance"

        account = (
            db.query(PaperAccount)
            .filter(PaperAccount.user_id == 1)
            .first()
        )

        assert account.balance == 5000.0

        holding = (
            db.query(PaperPortfolio)
            .filter(PaperPortfolio.user_id == 1)
            .first()
        )

        assert holding is None

    finally:
        db.close()
        teardown_database()


def test_stop_order_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "STOP",
            "quantity": 10,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert (
            result["reason"]
            == "Paper execution currently supports MARKET orders only"
        )

    finally:
        db.close()
        teardown_database()


def test_short_order_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "SHORT",
            "order_type": "MARKET",
            "quantity": 10,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert (
            result["reason"]
            == "SHORT execution is not supported by the current paper account"
        )

    finally:
        db.close()
        teardown_database()


def test_zero_quantity_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 0,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == "Quantity must be greater than zero"

    finally:
        db.close()
        teardown_database()

def test_second_buy_updates_average_price():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=100000.0,
        )

        db.add(account)

        holding = PaperPortfolio(
            user_id=1,
            symbol="RELIANCE.NS",
            quantity=10,
            average_price=1000.0,
            current_price=1000.0,
        )

        db.add(holding)
        db.commit()

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1200.0,
        ):
            result = PaperExecutionService(
                db=db,
                user_id=1,
            ).execute(order_intent)

        assert result["status"] == "PASS"
        assert result["executed"] is True
        assert result["quantity"] == 10
        assert result["execution_price"] == 1200.0
        assert result["total_amount"] == 12000.0
        assert result["remaining_balance"] == 88000.0

        holding = (
            db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == 1,
                PaperPortfolio.symbol == "RELIANCE.NS",
            )
            .first()
        )

        assert holding is not None
        assert holding.quantity == 20
        assert holding.average_price == 1100.0
        assert holding.current_price == 1200.0

    finally:
        db.close()
        teardown_database()

def test_manual_buy_executes():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=100000.0,
        )
        db.add(account)
        db.commit()

        service = PaperExecutionService(
            db=db,
            user_id=1,
        )

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1500.0,
        ):
            result = service.execute_manual_buy(
                symbol="RELIANCE.NS",
                quantity=10,
            )

        assert result["status"] == "PASS"
        assert result["executed"] is True
        assert result["message"] == "Paper BUY executed"
        assert result["symbol"] == "RELIANCE.NS"
        assert result["quantity"] == 10
        assert result["execution_price"] == 1500.0
        assert result["total_amount"] == 15000.0
        assert result["remaining_balance"] == 85000.0

        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == 1,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 10
        assert holding.average_price == 1500.0
        assert holding.current_price == 1500.0

        transaction = db.query(PaperTransaction).filter(
            PaperTransaction.user_id == 1,
            PaperTransaction.symbol == "RELIANCE.NS",
            PaperTransaction.transaction_type == "BUY",
        ).first()

        assert transaction is not None
        assert transaction.quantity == 10
        assert transaction.price == 1500.0
        assert transaction.total_amount == 15000.0

    finally:
        db.close()
        teardown_database()


def test_manual_sell_executes():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=85000.0,
        )
        db.add(account)

        holding = PaperPortfolio(
            user_id=1,
            symbol="RELIANCE.NS",
            quantity=10,
            average_price=1500.0,
            current_price=1500.0,
        )
        db.add(holding)
        db.commit()

        service = PaperExecutionService(
            db=db,
            user_id=1,
        )

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1600.0,
        ):
            result = service.execute_manual_sell(
                symbol="RELIANCE.NS",
                quantity=5,
            )

        assert result["status"] == "PASS"
        assert result["executed"] is True
        assert result["message"] == "Paper SELL executed"
        assert result["symbol"] == "RELIANCE.NS"
        assert result["quantity"] == 5
        assert result["execution_price"] == 1600.0
        assert result["total_amount"] == 8000.0
        assert result["remaining_balance"] == 93000.0

        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == 1,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 5
        assert holding.average_price == 1500.0
        assert holding.current_price == 1600.0

        transaction = db.query(PaperTransaction).filter(
            PaperTransaction.user_id == 1,
            PaperTransaction.symbol == "RELIANCE.NS",
            PaperTransaction.transaction_type == "SELL",
        ).first()

        assert transaction is not None
        assert transaction.quantity == 5
        assert transaction.price == 1600.0
        assert transaction.total_amount == 8000.0

    finally:
        db.close()
        teardown_database()


def test_manual_sell_rejects_insufficient_quantity():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=85000.0,
        )
        db.add(account)

        holding = PaperPortfolio(
            user_id=1,
            symbol="RELIANCE.NS",
            quantity=5,
            average_price=1500.0,
            current_price=1500.0,
        )
        db.add(holding)
        db.commit()

        service = PaperExecutionService(
            db=db,
            user_id=1,
        )

        with patch(
            "app.services.paper_execution_service.get_live_price",
            return_value=1600.0,
        ):
            result = service.execute_manual_sell(
                symbol="RELIANCE.NS",
                quantity=10,
            )

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == "Not enough quantity"

        account = db.query(PaperAccount).filter(
            PaperAccount.user_id == 1,
        ).first()

        assert account.balance == 85000.0

        holding = db.query(PaperPortfolio).filter(
            PaperPortfolio.user_id == 1,
            PaperPortfolio.symbol == "RELIANCE.NS",
        ).first()

        assert holding is not None
        assert holding.quantity == 5

    finally:
        db.close()
        teardown_database()


def test_manual_buy_rejects_zero_quantity():
    db = create_test_session()

    try:
        create_test_user(db)

        account = PaperAccount(
            user_id=1,
            balance=100000.0,
        )
        db.add(account)
        db.commit()

        service = PaperExecutionService(
            db=db,
            user_id=1,
        )

        result = service.execute_manual_buy(
            symbol="RELIANCE.NS",
            quantity=0,
        )

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == "Quantity must be greater than zero"

    finally:
        db.close()
        teardown_database()
def test_missing_intent_decision_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "ready_for_execution": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == (
            "Order intent is not approved"
        )

    finally:
        db.close()
        teardown_database()

def test_not_ready_for_execution_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "intent_decision": "PASS",
            "ready_for_execution": False,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == (
            "Order intent is not execution-ready"
        )

    finally:
        db.close()
        teardown_database()

def test_legacy_ready_field_does_not_authorize_execution():
    db = create_test_session()

    try:
        create_test_user(db)

        order_intent = {
            "intent_decision": "PASS",
            "ready": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        result = PaperExecutionService(
            db=db,
            user_id=1,
        ).execute(order_intent)

        assert result["status"] == "REJECT"
        assert result["executed"] is False
        assert result["reason"] == (
            "Order intent is not execution-ready"
        )

    finally:
        db.close()
        teardown_database()