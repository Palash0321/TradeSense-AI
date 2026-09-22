from datetime import datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.execution_fill import ExecutionFill
from app.models.execution_record import ExecutionRecord
from app.models.position import Position
from app.models.position_fill_allocation import PositionFillAllocation
from app.models.user import User
from app.services.execution_position_integration_service import (
    ExecutionPositionIntegrationService,
)
from app.services.position_service import PositionService


# ---------------------------------------------------------------------------
# Isolated test database
# ---------------------------------------------------------------------------

TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestSessionLocal = sessionmaker(
    bind=TEST_ENGINE,
    autoflush=False,
    autocommit=False,
)


@event.listens_for(TEST_ENGINE, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture(scope="module", autouse=True)
def setup_test_database():
    Base.metadata.create_all(
        TEST_ENGINE,
        tables=[
            User.__table__,
            ExecutionRecord.__table__,
            ExecutionFill.__table__,
            Position.__table__,
            PositionFillAllocation.__table__,
        ],
    )

    yield

    Base.metadata.drop_all(
        TEST_ENGINE,
        tables=[
            PositionFillAllocation.__table__,
            ExecutionFill.__table__,
            Position.__table__,
            ExecutionRecord.__table__,
            User.__table__,
        ],
    )


@pytest.fixture()
def db():
    # Each test gets a completely clean isolated SQLite schema.
    # This is necessary because individual tests intentionally call
    # db.commit(), so a final rollback cannot clean committed rows.
    Base.metadata.drop_all(
        TEST_ENGINE,
        tables=[
            PositionFillAllocation.__table__,
            ExecutionFill.__table__,
            Position.__table__,
            ExecutionRecord.__table__,
            User.__table__,
        ],
    )

    Base.metadata.create_all(
        TEST_ENGINE,
        tables=[
            User.__table__,
            ExecutionRecord.__table__,
            ExecutionFill.__table__,
            Position.__table__,
            PositionFillAllocation.__table__,
        ],
    )

    session = TestSessionLocal()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_user(db, user_id=1):
    user = User(
        id=user_id,
        full_name=f"Test User {user_id}",
        email=f"test{user_id}@example.com",
        password_hash="test-password",
    )
    db.add(user)
    db.flush()
    return user


def create_execution(
    db,
    *,
    execution_id="EXEC-001",
    user_id=1,
    symbol="RELIANCE.NS",
    direction="LONG",
    quantity=100,
    current_state="FILLED",
):
    execution = ExecutionRecord(
        execution_id=execution_id,
        user_id=user_id,
        symbol=symbol,
        direction=direction,
        order_type="MARKET",
        quantity=quantity,
        entry_price=None,
        current_state=current_state,
        broker_status="FILLED",
        broker_order_id=f"BROKER-{execution_id}",
        failure_reason=None,
    )

    db.add(execution)
    db.flush()

    return execution


def create_fill(
    db,
    *,
    fill_id="FILL-001",
    execution_id="EXEC-001",
    fill_sequence=1,
    fill_quantity=40,
    fill_price=2500.0,
):
    fill = ExecutionFill(
        fill_id=fill_id,
        execution_id=execution_id,
        broker_order_id=f"BROKER-{execution_id}",
        fill_sequence=fill_sequence,
        fill_quantity=fill_quantity,
        fill_price=fill_price,
        filled_at=datetime.utcnow(),
    )

    db.add(fill)
    db.flush()

    return fill


def create_position(
    db,
    *,
    position_id="POS-001",
    user_id=1,
    symbol="RELIANCE.NS",
    direction="LONG",
    quantity=40,
    average_entry_price=2500.0,
    current_price=2500.0,
    state="OPEN",
):
    position = Position(
        position_id=position_id,
        user_id=user_id,
        symbol=symbol,
        direction=direction,
        quantity=quantity,
        average_entry_price=average_entry_price,
        current_price=current_price,
        stop_loss=None,
        target1=None,
        target2=None,
        target3=None,
        state=state,
        opened_at=datetime.utcnow(),
        closed_at=None,
    )

    db.add(position)
    db.flush()

    return position


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_first_reconciled_fill_creates_position(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-001",
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-001",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=40,
        fill_price=2510.5,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id="EXEC-001",
        fill_id="FILL-001",
    )

    assert result["status"] == "PASS"
    assert result["integration_decision"] == "POSITION_CREATED"
    assert result["position_id"]

    position = (
        db.query(Position)
        .filter(Position.position_id == result["position_id"])
        .one()
    )

    assert position.user_id == 1
    assert position.symbol == "RELIANCE.NS"
    assert position.direction == "LONG"
    assert position.quantity == 40
    assert position.average_entry_price == pytest.approx(2510.5)
    assert position.current_price == pytest.approx(2510.5)

    allocation = (
        db.query(PositionFillAllocation)
        .filter(PositionFillAllocation.fill_id == "FILL-001")
        .one()
    )

    assert allocation.position_id == position.position_id
    assert allocation.execution_id == "EXEC-001"
    assert allocation.fill_quantity == 40
    assert allocation.fill_price == pytest.approx(2510.5)


def test_second_fill_increases_existing_position(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-002",
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-002-A",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=40,
        fill_price=2500.0,
    )

    create_fill(
        db,
        fill_id="FILL-002-B",
        execution_id=execution.execution_id,
        fill_sequence=2,
        fill_quantity=60,
        fill_price=2600.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    first_result = service.apply_reconciled_fill(
        execution_id="EXEC-002",
        fill_id="FILL-002-A",
    )

    assert first_result["status"] == "PASS"
    assert first_result["integration_decision"] == "POSITION_CREATED"

    position_id = first_result["position_id"]

    second_result = service.apply_reconciled_fill(
        execution_id="EXEC-002",
        fill_id="FILL-002-B",
    )

    assert second_result["status"] == "PASS"
    assert second_result["integration_decision"] == "POSITION_INCREASED"
    assert second_result["position_id"] == position_id

    position = (
        db.query(Position)
        .filter(Position.position_id == position_id)
        .one()
    )

    assert position.quantity == 100

    # Weighted average:
    # (40 * 2500 + 60 * 2600) / 100 = 2560
    assert position.average_entry_price == pytest.approx(2560.0)

    assert position.current_price == pytest.approx(2600.0)

    allocations = (
        db.query(PositionFillAllocation)
        .filter(
            PositionFillAllocation.position_id == position_id,
        )
        .all()
    )

    assert len(allocations) == 2


def test_same_fill_is_idempotent(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-003",
        user_id=1,
        symbol="INFY.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-003",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=25,
        fill_price=1500.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    first_result = service.apply_reconciled_fill(
        execution_id="EXEC-003",
        fill_id="FILL-003",
    )

    assert first_result["status"] == "PASS"
    assert first_result["integration_decision"] == "POSITION_CREATED"

    position_id = first_result["position_id"]

    second_result = service.apply_reconciled_fill(
        execution_id="EXEC-003",
        fill_id="FILL-003",
    )

    assert second_result["status"] == "PASS"
    assert second_result["integration_decision"] == "ALREADY_APPLIED"
    assert second_result["position_id"] == position_id

    position = (
        db.query(Position)
        .filter(Position.position_id == position_id)
        .one()
    )

    assert position.quantity == 25
    assert position.average_entry_price == pytest.approx(1500.0)

    allocations = (
        db.query(PositionFillAllocation)
        .filter(PositionFillAllocation.fill_id == "FILL-003")
        .all()
    )

    assert len(allocations) == 1


@pytest.mark.parametrize(
    "execution_state",
    [
        "CREATED",
        "ACKNOWLEDGED",
    ],
)
def test_unreconciled_execution_is_rejected(db, execution_state):
    create_user(db)

    execution = create_execution(
        db,
        execution_id=f"EXEC-UNREC-{execution_state}",
        user_id=1,
        symbol="TCS.NS",
        direction="LONG",
        quantity=100,
        current_state=execution_state,
    )

    create_fill(
        db,
        fill_id=f"FILL-UNREC-{execution_state}",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=10,
        fill_price=3000.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id=execution.execution_id,
        fill_id=f"FILL-UNREC-{execution_state}",
    )

    assert result["status"] == "REJECT"

    assert (
        "reconciled" in result["reason"].lower()
        or "state" in result["reason"].lower()
    )

    assert db.query(Position).count() == 0
    assert db.query(PositionFillAllocation).count() == 0


def test_missing_fill_is_rejected(db):
    create_user(db)

    create_execution(
        db,
        execution_id="EXEC-MISSING-FILL",
        user_id=1,
        symbol="SBIN.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id="EXEC-MISSING-FILL",
        fill_id="DOES-NOT-EXIST",
    )

    assert result["status"] == "REJECT"
    assert "fill" in result["reason"].lower()

    assert db.query(Position).count() == 0
    assert db.query(PositionFillAllocation).count() == 0


def test_fill_execution_mismatch_is_rejected(db):
    create_user(db)

    create_execution(
        db,
        execution_id="EXEC-MISMATCH-A",
        user_id=1,
        symbol="HDFCBANK.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_execution(
        db,
        execution_id="EXEC-MISMATCH-B",
        user_id=1,
        symbol="HDFCBANK.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-MISMATCH",
        execution_id="EXEC-MISMATCH-B",
        fill_sequence=1,
        fill_quantity=20,
        fill_price=1700.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id="EXEC-MISMATCH-A",
        fill_id="FILL-MISMATCH",
    )

    assert result["status"] == "REJECT"
    assert "execution" in result["reason"].lower()

    assert db.query(Position).count() == 0
    assert db.query(PositionFillAllocation).count() == 0


@pytest.mark.parametrize(
    "field,position_value",
    [
        ("user_id", 2),
        ("symbol", "ICICIBANK.NS"),
        ("direction", "SHORT"),
    ],
)
def test_explicit_position_identity_mismatch_is_rejected(
    db,
    field,
    position_value,
):
    create_user(db, 1)

    if field == "user_id":
        create_user(db, 2)

    execution = create_execution(
        db,
        execution_id=f"EXEC-MISMATCH-{field}",
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    if field == "user_id":
        position_user_id = position_value
        position_symbol = "RELIANCE.NS"
        position_direction = "LONG"

    elif field == "symbol":
        position_user_id = 1
        position_symbol = position_value
        position_direction = "LONG"

    else:
        position_user_id = 1
        position_symbol = "RELIANCE.NS"
        position_direction = position_value

    position = create_position(
        db,
        position_id=f"POS-MISMATCH-{field}",
        user_id=position_user_id,
        symbol=position_symbol,
        direction=position_direction,
        quantity=10,
        average_entry_price=100.0,
        current_price=100.0,
    )

    fill = create_fill(
        db,
        execution_id=execution.execution_id,
        fill_id=f"FILL-MISMATCH-{field}",
        fill_quantity=10,
        fill_price=101.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id=execution.execution_id,
        fill_id=fill.fill_id,
        position_id=position.position_id,
    )

    assert result["status"] == "REJECT"
    assert "match" in result["reason"].lower()
    assert db.query(PositionFillAllocation).count() == 0


def test_explicit_valid_position_receives_fill(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-EXPLICIT-POS",
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-EXPLICIT-POS",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=30,
        fill_price=2550.0,
    )

    position = create_position(
        db,
        position_id="POS-EXPLICIT-POS",
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        quantity=20,
        average_entry_price=2500.0,
        current_price=2500.0,
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id=execution.execution_id,
        fill_id="FILL-EXPLICIT-POS",
        position_id=position.position_id,
    )

    assert result["status"] == "PASS"
    assert result["integration_decision"] == "POSITION_INCREASED"
    assert result["position_id"] == position.position_id

    db.refresh(position)

    assert position.quantity == 50

    # (20 * 2500 + 30 * 2550) / 50 = 2530
    assert position.average_entry_price == pytest.approx(2530.0)
    assert position.current_price == pytest.approx(2550.0)


def test_multiple_prior_positions_require_explicit_position_id(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-AMBIGUOUS",
        user_id=1,
        symbol="TCS.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-AMBIGUOUS",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=20,
        fill_price=3500.0,
    )

    position_a = create_position(
        db,
        position_id="POS-AMBIGUOUS-A",
        user_id=1,
        symbol="TCS.NS",
        direction="LONG",
        quantity=20,
        average_entry_price=3450.0,
        current_price=3450.0,
    )

    position_b = create_position(
        db,
        position_id="POS-AMBIGUOUS-B",
        user_id=1,
        symbol="TCS.NS",
        direction="LONG",
        quantity=30,
        average_entry_price=3550.0,
        current_price=3550.0,
    )

    # Create the prior fills first because PositionFillAllocation.fill_id
    # has a foreign-key relationship to ExecutionFill.fill_id.
    create_fill(
        db,
        fill_id="PRIOR-FILL-A",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=10,
        fill_price=3450.0,
    )

    create_fill(
        db,
        fill_id="PRIOR-FILL-B",
        execution_id=execution.execution_id,
        fill_sequence=2,
        fill_quantity=10,
        fill_price=3550.0,
    )

    # Now make both positions known as allocations for the same execution,
    # creating an intentionally ambiguous routing situation.
    db.add(
        PositionFillAllocation(
            position_id=position_a.position_id,
            fill_id="PRIOR-FILL-A",
            execution_id=execution.execution_id,
            fill_sequence=1,
            fill_quantity=10,
            fill_price=3450.0,
        )
    )

    db.add(
        PositionFillAllocation(
            position_id=position_b.position_id,
            fill_id="PRIOR-FILL-B",
            execution_id=execution.execution_id,
            fill_sequence=2,
            fill_quantity=10,
            fill_price=3550.0,
        )
    )

    db.flush()

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id=execution.execution_id,
        fill_id="FILL-AMBIGUOUS",
    )

    assert result["status"] == "REJECT"

    assert (
        "position" in result["reason"].lower()
        or "ambiguous" in result["reason"].lower()
    )

    assert (
        db.query(PositionFillAllocation)
        .filter(PositionFillAllocation.fill_id == "FILL-AMBIGUOUS")
        .count()
        == 0
    )


def test_invalid_input_is_rejected(db):
    create_user(db)

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id="",
        fill_id="",
    )

    assert result["status"] == "REJECT"
    assert "execution_id" in result["reason"].lower()


def test_closed_position_cannot_receive_fill(db):
    create_user(db)

    execution = create_execution(
        db,
        execution_id="EXEC-CLOSED-POS",
        user_id=1,
        symbol="SBIN.NS",
        direction="LONG",
        quantity=100,
        current_state="FILLED",
    )

    create_fill(
        db,
        fill_id="FILL-CLOSED-POS",
        execution_id=execution.execution_id,
        fill_sequence=1,
        fill_quantity=20,
        fill_price=900.0,
    )

    position = create_position(
        db,
        position_id="POS-CLOSED-POS",
        user_id=1,
        symbol="SBIN.NS",
        direction="LONG",
        quantity=20,
        average_entry_price=890.0,
        current_price=890.0,
        state="CLOSED",
    )

    db.commit()

    service = ExecutionPositionIntegrationService(db)

    result = service.apply_reconciled_fill(
        execution_id=execution.execution_id,
        fill_id="FILL-CLOSED-POS",
        position_id=position.position_id,
    )

    assert result["status"] == "REJECT"

    db.refresh(position)

    assert position.quantity == 20
    assert position.average_entry_price == pytest.approx(890.0)

    assert db.query(PositionFillAllocation).count() == 0