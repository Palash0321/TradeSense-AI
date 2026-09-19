import uuid

from app.core.database import SessionLocal
from app.models.execution_record import ExecutionRecord
from app.services.execution_ledger_service import ExecutionLedgerService


def _create_ledger(db, execution_id=None):
    return ExecutionLedgerService(
        db=db,
        execution_id=execution_id or str(uuid.uuid4()),
        user_id=1,
        symbol="RELIANCE.NS",
        direction="LONG",
        order_type="MARKET",
        quantity=10,
        entry_price=2500.0,
    )


def test_create_persists_execution_record():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        result = ledger.create()

        assert result["status"] == "PASS"
        assert result["execution_id"] == execution_id
        assert result["state"] == "CREATED"

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.user_id == 1
        assert record.symbol == "RELIANCE.NS"
        assert record.direction == "LONG"
        assert record.order_type == "MARKET"
        assert record.quantity == 10
        assert record.entry_price == 2500.0
        assert record.current_state == "CREATED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_duplicate_execution_id_is_rejected():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        first_result = ledger.create()

        assert first_result["status"] == "PASS"

        duplicate_ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        duplicate_result = duplicate_ledger.create()

        assert duplicate_result["status"] == "REJECT"
        assert (
            "already exists"
            in duplicate_result["reason"]
        )

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_state_transition_is_persisted():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        create_result = ledger.create()

        assert create_result["status"] == "PASS"

        validated_result = ledger.transition(
            new_state="VALIDATED",
            reason="Execution validation completed.",
        )

        assert validated_result["status"] == "PASS"
        assert validated_result["previous_state"] == "CREATED"
        assert validated_result["state"] == "VALIDATED"

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.current_state == "VALIDATED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_full_execution_lifecycle_is_persisted():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        lifecycle = [
            "VALIDATED",
            "SUBMITTED",
            "ACKNOWLEDGED",
            "FILLED",
            "COMPLETED",
        ]

        for state in lifecycle:
            result = ledger.transition(
                new_state=state,
                reason=f"Transitioned to {state}.",
            )

            assert result["status"] == "PASS"
            assert result["state"] == state

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.current_state == "COMPLETED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_invalid_transition_is_not_persisted():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        result = ledger.transition(
            new_state="FILLED",
            reason="Invalid direct transition.",
        )

        assert result["status"] == "REJECT"

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.current_state == "CREATED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_terminal_state_cannot_transition():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        assert (
            ledger.transition("REJECTED")["status"]
            == "PASS"
        )

        result = ledger.transition(
            new_state="VALIDATED",
            reason="Attempted transition after rejection.",
        )

        assert result["status"] == "REJECT"

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.current_state == "REJECTED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_broker_state_is_persisted():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        result = ledger.update_broker_state(
            broker_status="ACKNOWLEDGED",
            broker_order_id="BROKER-12345",
        )

        assert result["status"] == "PASS"
        assert result["broker_status"] == "ACKNOWLEDGED"
        assert result["broker_order_id"] == "BROKER-12345"

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.broker_status == "ACKNOWLEDGED"
        assert record.broker_order_id == "BROKER-12345"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_failure_reason_is_persisted():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        result = ledger.update_broker_state(
            broker_status="REJECTED",
            failure_reason="Broker rejected the order.",
        )

        assert result["status"] == "PASS"
        assert result["broker_status"] == "REJECTED"
        assert (
            result["failure_reason"]
            == "Broker rejected the order."
        )

        db.expire_all()

        record = (
            db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id
            )
            .first()
        )

        assert record is not None
        assert record.broker_status == "REJECTED"
        assert (
            record.failure_reason
            == "Broker rejected the order."
        )

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_get_record_returns_persisted_execution():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        assert ledger.create()["status"] == "PASS"

        result = ledger.get_record()

        assert result["status"] == "PASS"
        assert result["execution_id"] == execution_id
        assert result["symbol"] == "RELIANCE.NS"
        assert result["direction"] == "LONG"
        assert result["order_type"] == "MARKET"
        assert result["quantity"] == 10
        assert result["entry_price"] == 2500.0
        assert result["current_state"] == "CREATED"

    finally:
        db.query(ExecutionRecord).filter(
            ExecutionRecord.execution_id == execution_id
        ).delete(
            synchronize_session=False
        )
        db.commit()
        db.close()


def test_missing_record_rejects_transition():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        result = ledger.transition(
            new_state="VALIDATED",
            reason="No persistent record exists.",
        )

        assert result["status"] == "REJECT"
        assert (
            "does not exist"
            in result["reason"]
        )

    finally:
        db.close()


def test_missing_record_rejects_broker_update():
    db = SessionLocal()

    execution_id = str(uuid.uuid4())

    try:
        ledger = _create_ledger(
            db=db,
            execution_id=execution_id,
        )

        result = ledger.update_broker_state(
            broker_status="ACKNOWLEDGED",
            broker_order_id="BROKER-999",
        )

        assert result["status"] == "REJECT"
        assert (
            "does not exist"
            in result["reason"]
        )

    finally:
        db.close()