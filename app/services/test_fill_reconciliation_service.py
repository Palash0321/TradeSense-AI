from datetime import datetime
from types import SimpleNamespace

import pytest

from app.models.execution_fill import ExecutionFill

from app.services.execution_state_service import ExecutionStateService
from app.services.fill_reconciliation_service import (
    FillReconciliationService,
)


class FakeQuery:
    def __init__(
        self,
        *,
        first_result=None,
        scalar_result=0,
        records=None,
    ):
        self.first_result = first_result
        self.scalar_result = scalar_result
        self.records = records

    def filter(self, *criteria, **kwargs):
        if self.records is None:
            return self

        filtered_records = list(self.records)

        for criterion in criteria:
            left = getattr(criterion, "left", None)
            right = getattr(criterion, "right", None)

            column_name = getattr(left, "name", None)
            expected_value = getattr(right, "value", right)

            if column_name is None:
                continue

            filtered_records = [
                record
                for record in filtered_records
                if getattr(record, column_name, None)
                == expected_value
            ]

        self.first_result = (
            filtered_records[0]
            if filtered_records
            else None
        )

        return self

    def first(self):
        return self.first_result

    def scalar(self):
        return self.scalar_result


class FakeDB:
    def __init__(self):
        self.added = []
        self.rollback_called = False
        self.fill_records = []

    def query(self, model_expression):
        if model_expression is ExecutionFill:
            return FakeQuery(
                records=self.fill_records,
            )

        expression = str(model_expression)

        if "execution_fills.fill_quantity" in expression:
            return FakeQuery(
                scalar_result=sum(
                    record.fill_quantity
                    for record in self.fill_records
                )
            )

        if "execution_fills.fill_sequence" in expression:
            return FakeQuery(
                scalar_result=max(
                    (
                        record.fill_sequence
                        for record in self.fill_records
                    ),
                    default=0,
                )
            )

        return FakeQuery()

    def add(self, record):
        self.added.append(record)

        self.fill_records.append(record)

    def rollback(self):
        self.rollback_called = True


class FakeLedgerService:
    def __init__(
        self,
        *,
        execution_id="EXEC-001",
        quantity=10,
        current_state="ACKNOWLEDGED",
    ):
        self.execution_id = execution_id
        self.transition_calls = []

        self.record = SimpleNamespace(
            execution_id=execution_id,
            quantity=quantity,
            current_state=current_state,
        )

        self.state_service = ExecutionStateService(execution_id)

        if current_state != "CREATED":
            setup_states = [
                "VALIDATED",
                "SUBMITTED",
                "ACKNOWLEDGED",
            ]

            if current_state != "ACKNOWLEDGED":
                setup_states.append(current_state)

            for setup_state in setup_states:
                result = self.state_service.transition(
                    setup_state,
                    reason="Fixture setup.",
                    metadata={},
                )
                assert result["status"] == "PASS"

    def get_record(self):
        return self.record

    def transition(self, **kwargs):
        self.transition_calls.append(kwargs)

        new_state = kwargs["new_state"]

        result = self.state_service.transition(
            new_state=new_state,
            reason=kwargs.get("reason"),
            metadata=kwargs.get("metadata"),
        )

        if result["status"] != "PASS":
            return result

        self.record.current_state = new_state

        return {
            "status": "PASS",
            "execution_id": self.execution_id,
            "previous_state": result["previous_state"],
            "state": new_state,
            "terminal": new_state in ExecutionStateService.TERMINAL_STATES,
            "source": "FakeLedgerService",
        }


def build_service(
    *,
    quantity=10,
    current_state="ACKNOWLEDGED",
):
    db = FakeDB()

    ledger = FakeLedgerService(
        quantity=quantity,
        current_state=current_state,
    )

    service = FillReconciliationService(
        db=db,
        ledger_service=ledger,
    )

    return service, db, ledger


def test_partial_fill_passes_and_persists():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
        broker_order_id="BROKER-001",
    )

    assert result["reconciliation_decision"] == "PASS"
    assert result["reconciled"] is True
    assert result["state"] == "PARTIALLY_FILLED"
    assert result["fill_quantity"] == 4
    assert result["cumulative_filled_quantity"] == 4
    assert result["remaining_quantity"] == 6

    assert len(db.fill_records) == 1
    assert db.fill_records[0].fill_id == "FILL-001"
    assert db.fill_records[0].fill_quantity == 4
    assert db.fill_records[0].fill_price == 2501.0
    assert db.fill_records[0].fill_sequence == 1

    assert len(ledger.transition_calls) == 1
    assert ledger.transition_calls[0]["new_state"] == "PARTIALLY_FILLED"


def test_second_partial_fill_accumulates_quantity():
    service, db, ledger = build_service(quantity=10)

    first = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert first["reconciliation_decision"] == "PASS"

    second = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-002",
        state="PARTIALLY_FILLED",
        fill_quantity=3,
        fill_price=2502.0,
    )

    assert second["reconciliation_decision"] == "PASS"
    assert second["cumulative_filled_quantity"] == 7
    assert second["remaining_quantity"] == 3

    assert len(db.fill_records) == 2
    assert db.fill_records[1].fill_sequence == 2

    assert len(ledger.transition_calls) == 2


def test_final_fill_transitions_to_filled():
    service, db, ledger = build_service(quantity=10)

    first = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert first["reconciliation_decision"] == "PASS"

    second = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-002",
        state="FILLED",
        fill_quantity=6,
        fill_price=2503.0,
    )

    assert second["reconciliation_decision"] == "PASS"
    assert second["state"] == "FILLED"
    assert second["cumulative_filled_quantity"] == 10
    assert second["remaining_quantity"] == 0

    assert ledger.record.current_state == "FILLED"
    assert len(db.fill_records) == 2
    assert len(ledger.transition_calls) == 2


def test_filled_requires_exact_order_quantity():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "FILLED requires cumulative" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_partial_fill_cannot_complete_order():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=10,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "PARTIALLY_FILLED requires cumulative" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_overfill_is_rejected():
    service, db, ledger = build_service(quantity=10)

    first = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=7,
        fill_price=2501.0,
    )

    assert first["reconciliation_decision"] == "PASS"

    second = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-002",
        state="FILLED",
        fill_quantity=4,
        fill_price=2502.0,
    )

    assert second["reconciliation_decision"] == "REJECT"
    assert second["reconciled"] is False
    assert "exceed the execution order quantity" in second["reason"]

    assert len(db.fill_records) == 1
    assert len(ledger.transition_calls) == 1


def test_duplicate_fill_id_is_rejected():
    service, db, ledger = build_service(quantity=10)

    first = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert first["reconciliation_decision"] == "PASS"

    duplicate = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert duplicate["reconciliation_decision"] == "REJECT"
    assert duplicate["reconciled"] is False
    assert "already been reconciled" in duplicate["reason"]

    assert len(db.fill_records) == 1
    assert len(ledger.transition_calls) == 1


def test_invalid_fill_state_is_rejected():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="COMPLETED",
        fill_quantity=10,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "PARTIALLY_FILLED or FILLED" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_invalid_fill_quantity_is_rejected():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=0,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "fill_quantity must be a positive integer" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_invalid_fill_price_is_rejected():
    service, db, ledger = build_service(quantity=10)

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "fill_price must be a positive number" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_missing_execution_is_rejected():
    db = FakeDB()
    ledger = FakeLedgerService(
        execution_id="OTHER-EXECUTION",
        quantity=10,
        current_state="ACKNOWLEDGED",
    )

    service = FillReconciliationService(
        db=db,
        ledger_service=ledger,
    )

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "Execution ID mismatch" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_fill_after_filled_is_rejected():
    service, db, ledger = build_service(
        quantity=10,
        current_state="FILLED",
    )

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-002",
        state="FILLED",
        fill_quantity=1,
        fill_price=2504.0,
    )

    assert result["reconciliation_decision"] == "REJECT"
    assert result["reconciled"] is False
    assert "cannot accept a fill" in result["reason"]

    assert len(db.fill_records) == 0
    assert len(ledger.transition_calls) == 0


def test_fill_is_explicit_and_not_inferred_from_broker_status():
    service, db, ledger = build_service(quantity=10)

    ledger.record.broker_status = "EXECUTED"

    assert ledger.record.current_state == "ACKNOWLEDGED"

    assert len(db.fill_records) == 0

    result = service.reconcile(
        execution_id="EXEC-001",
        fill_id="FILL-001",
        state="PARTIALLY_FILLED",
        fill_quantity=4,
        fill_price=2501.0,
    )

    assert result["reconciliation_decision"] == "PASS"
    assert ledger.record.current_state == "PARTIALLY_FILLED"