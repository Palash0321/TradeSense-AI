from typing import Any

from sqlalchemy.orm import Session

from app.models.execution_record import ExecutionRecord
from app.services.execution_state_service import ExecutionStateService


class ExecutionLedgerService:
    """
    Persists execution lifecycle state into the execution_records table.

    This service connects the in-memory ExecutionStateService with the
    persistent ExecutionRecord database model.

    It does NOT:
    - generate trading signals
    - approve or reject trading strategy decisions
    - calculate risk
    - calculate position size
    - place broker orders
    - modify strategy logic
    - infer broker state
    """

    def __init__(
        self,
        db: Session,
        execution_id: str,
        user_id: int,
        symbol: str,
        direction: str,
        order_type: str,
        quantity: int,
        entry_price: float | None = None,
        state_service: ExecutionStateService | None = None,
    ):
        self.db = db
        self.execution_id = execution_id
        self.user_id = user_id
        self.symbol = symbol
        self.direction = direction
        self.order_type = order_type
        self.quantity = quantity
        self.entry_price = entry_price

        self.state_service = state_service or ExecutionStateService(
            execution_id=execution_id
        )

    def create(self):
        """
        Create the initial persistent execution record.

        The initial state is always CREATED because this service does not
        independently approve or validate an execution.
        """

        existing = (
            self.db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == self.execution_id
            )
            .first()
        )

        if existing is not None:
            return {
                "status": "REJECT",
                "execution_id": self.execution_id,
                "state": existing.current_state,
                "reason": (
                    "An execution record with this execution_id "
                    "already exists."
                ),
                "source": "ExecutionLedgerService",
            }

        record = ExecutionRecord(
            execution_id=self.execution_id,
            user_id=self.user_id,
            symbol=self.symbol,
            direction=self.direction,
            order_type=self.order_type,
            quantity=self.quantity,
            entry_price=self.entry_price,
            current_state=self.state_service.state,
        )

        self.db.add(record)

        try:
            self.db.commit()
            self.db.refresh(record)
        except Exception:
            self.db.rollback()
            raise

        return {
            "status": "PASS",
            "execution_id": record.execution_id,
            "record_id": record.id,
            "state": record.current_state,
            "source": "ExecutionLedgerService",
        }

    def transition(
        self,
        new_state: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Transition the execution state and persist the new state.

        The in-memory state is restored if persistence cannot be completed,
        ensuring that the in-memory lifecycle and persistent ledger do not
        diverge.
        """

        previous_state = self.state_service.state
        previous_history = list(self.state_service.history)

        transition_result = self.state_service.transition(
            new_state=new_state,
            reason=reason,
            metadata=metadata,
        )

        if transition_result.get("status") != "PASS":
            return transition_result

        record = (
            self.db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == self.execution_id
            )
            .first()
        )

        if record is None:
            self.state_service.state = previous_state
            self.state_service.history = previous_history

            return {
                "status": "REJECT",
                "execution_id": self.execution_id,
                "state": self.state_service.state,
                "reason": (
                    "Persistent execution record does not exist."
                ),
                "source": "ExecutionLedgerService",
            }

        record.current_state = self.state_service.state

        try:
            self.db.commit()
        except Exception:
            self.db.rollback()

            self.state_service.state = previous_state
            self.state_service.history = previous_history

            raise

        return {
            "status": "PASS",
            "execution_id": self.execution_id,
            "record_id": record.id,
            "previous_state": transition_result.get(
                "previous_state"
            ),
            "state": record.current_state,
            "terminal": record.current_state
            in ExecutionStateService.TERMINAL_STATES,
            "source": "ExecutionLedgerService",
        }

    def update_broker_state(
        self,
        broker_status: str | None = None,
        broker_order_id: str | None = None,
        failure_reason: str | None = None,
    ):
        """
        Persist explicit broker/execution information.

        This method does not infer broker state. It only stores values
        explicitly supplied by the caller.
        """

        record = (
            self.db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == self.execution_id
            )
            .first()
        )

        if record is None:
            return {
                "status": "REJECT",
                "execution_id": self.execution_id,
                "reason": (
                    "Persistent execution record does not exist."
                ),
                "source": "ExecutionLedgerService",
            }

        if broker_status is not None:
            record.broker_status = broker_status

        if broker_order_id is not None:
            record.broker_order_id = broker_order_id

        if failure_reason is not None:
            record.failure_reason = failure_reason

        try:
            self.db.commit()
            self.db.refresh(record)
        except Exception:
            self.db.rollback()
            raise

        return {
            "status": "PASS",
            "execution_id": self.execution_id,
            "record_id": record.id,
            "state": record.current_state,
            "broker_status": record.broker_status,
            "broker_order_id": record.broker_order_id,
            "failure_reason": record.failure_reason,
            "source": "ExecutionLedgerService",
        }

    def get_record(self):
        """
        Retrieve the persistent execution record.
        """

        record = (
            self.db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == self.execution_id
            )
            .first()
        )

        if record is None:
            return {
                "status": "REJECT",
                "execution_id": self.execution_id,
                "reason": (
                    "Persistent execution record does not exist."
                ),
                "source": "ExecutionLedgerService",
            }

        return {
            "status": "PASS",
            "execution_id": record.execution_id,
            "record_id": record.id,
            "user_id": record.user_id,
            "symbol": record.symbol,
            "direction": record.direction,
            "order_type": record.order_type,
            "quantity": record.quantity,
            "entry_price": record.entry_price,
            "current_state": record.current_state,
            "broker_status": record.broker_status,
            "broker_order_id": record.broker_order_id,
            "failure_reason": record.failure_reason,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "source": "ExecutionLedgerService",
        }