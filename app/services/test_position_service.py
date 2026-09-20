from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.user import User
from app.models.position import Position

from app.services.position_service import PositionService


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
        email="position-test@example.com",
        password_hash="test-password",
    )

    db.add(user)
    db.commit()

    return user


def test_create_position():
    db = create_test_session()

    try:
        create_test_user(db)

        result = PositionService(
            db=db,
            user_id=1,
        ).create(
            symbol="RELIANCE.NS",
            direction="LONG",
            quantity=100,
            average_entry_price=2500.0,
            current_price=2510.0,
            stop_loss=2450.0,
            target1=2600.0,
            target2=2700.0,
            target3=2800.0,
            position_id="POSITION-001",
        )

        assert result["status"] == "PASS"
        assert result["position_id"] == "POSITION-001"
        assert result["state"] == "OPEN"
        assert result["quantity"] == 100
        assert result["direction"] == "LONG"

        position = (
            db.query(Position)
            .filter(
                Position.position_id == "POSITION-001"
            )
            .first()
        )

        assert position is not None
        assert position.quantity == 100
        assert position.state == "OPEN"
        assert position.average_entry_price == 2500.0

    finally:
        db.close()
        teardown_database()


def test_duplicate_position_id_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        first = service.create(
            symbol="RELIANCE.NS",
            direction="LONG",
            quantity=100,
            average_entry_price=2500.0,
            position_id="POSITION-002",
        )

        assert first["status"] == "PASS"

        second = service.create(
            symbol="TCS.NS",
            direction="LONG",
            quantity=50,
            average_entry_price=3000.0,
            position_id="POSITION-002",
        )

        assert second["status"] == "REJECT"
        assert second["reason"] == (
            "A position with this position_id already exists."
        )

    finally:
        db.close()
        teardown_database()


def test_invalid_position_creation_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        result = PositionService(
            db=db,
            user_id=1,
        ).create(
            symbol="",
            direction="LONG",
            quantity=100,
            average_entry_price=2500.0,
        )

        assert result["status"] == "REJECT"

    finally:
        db.close()
        teardown_database()


def test_get_position():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="INFY.NS",
            direction="LONG",
            quantity=25,
            average_entry_price=1500.0,
            position_id="POSITION-003",
        )

        result = service.get(
            "POSITION-003"
        )

        assert result["status"] == "PASS"
        assert result["position_id"] == "POSITION-003"
        assert result["symbol"] == "INFY.NS"
        assert result["quantity"] == 25
        assert result["state"] == "OPEN"

    finally:
        db.close()
        teardown_database()


def test_missing_position_rejects():
    db = create_test_session()

    try:
        create_test_user(db)

        result = PositionService(
            db=db,
            user_id=1,
        ).get("DOES-NOT-EXIST")

        assert result["status"] == "REJECT"
        assert result["reason"] == (
            "Position not found."
        )

    finally:
        db.close()
        teardown_database()


def test_update_current_price():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="SBIN.NS",
            direction="LONG",
            quantity=10,
            average_entry_price=800.0,
            position_id="POSITION-004",
        )

        result = service.update_current_price(
            position_id="POSITION-004",
            current_price=815.0,
        )

        assert result["status"] == "PASS"
        assert result["current_price"] == 815.0
        assert result["state"] == "OPEN"

    finally:
        db.close()
        teardown_database()


def test_partial_close_reduces_quantity_and_changes_state():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="HDFCBANK.NS",
            direction="LONG",
            quantity=100,
            average_entry_price=1700.0,
            position_id="POSITION-005",
        )

        result = service.partially_close(
            position_id="POSITION-005",
            quantity=40,
            current_price=1750.0,
        )

        assert result["status"] == "PASS"
        assert result["quantity"] == 60
        assert result["quantity_closed"] == 40
        assert result["state"] == "PARTIALLY_CLOSED"
        assert result["current_price"] == 1750.0

    finally:
        db.close()
        teardown_database()


def test_partial_close_cannot_close_entire_position():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="TCS.NS",
            direction="LONG",
            quantity=100,
            average_entry_price=3500.0,
            position_id="POSITION-006",
        )

        result = service.partially_close(
            position_id="POSITION-006",
            quantity=100,
        )

        assert result["status"] == "REJECT"
        assert result["reason"] == (
            "Partial close quantity must be less than "
            "the current open quantity."
        )

    finally:
        db.close()
        teardown_database()


def test_close_sets_quantity_zero_and_state_closed():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="ICICIBANK.NS",
            direction="LONG",
            quantity=50,
            average_entry_price=1200.0,
            position_id="POSITION-007",
        )

        result = service.close(
            position_id="POSITION-007",
            current_price=1250.0,
        )

        assert result["status"] == "PASS"
        assert result["quantity"] == 0
        assert result["state"] == "CLOSED"
        assert result["current_price"] == 1250.0
        assert result["closed_at"] is not None

    finally:
        db.close()
        teardown_database()


def test_closed_position_cannot_be_closed_again():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="SBIN.NS",
            direction="LONG",
            quantity=10,
            average_entry_price=800.0,
            position_id="POSITION-008",
        )

        first = service.close(
            position_id="POSITION-008",
            current_price=810.0,
        )

        assert first["status"] == "PASS"

        second = service.close(
            position_id="POSITION-008",
            current_price=820.0,
        )

        assert second["status"] == "REJECT"
        assert second["reason"] == (
            "Position is already closed."
        )

    finally:
        db.close()
        teardown_database()


def test_partial_position_can_be_fully_closed():
    db = create_test_session()

    try:
        create_test_user(db)

        service = PositionService(
            db=db,
            user_id=1,
        )

        service.create(
            symbol="RELIANCE.NS",
            direction="LONG",
            quantity=100,
            average_entry_price=2500.0,
            position_id="POSITION-009",
        )

        partial = service.partially_close(
            position_id="POSITION-009",
            quantity=40,
        )

        assert partial["status"] == "PASS"
        assert partial["quantity"] == 60
        assert partial["state"] == "PARTIALLY_CLOSED"

        closed = service.close(
            position_id="POSITION-009",
            current_price=2525.0,
        )

        assert closed["status"] == "PASS"
        assert closed["quantity"] == 0
        assert closed["state"] == "CLOSED"
        assert closed["closed_at"] is not None

    finally:
        db.close()
        teardown_database()