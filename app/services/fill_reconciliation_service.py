from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.execution_fill import ExecutionFill
from app.services.execution_ledger_service import ExecutionLedgerService


class FillReconciliationService:
    """
    Reconciles explicit broker fill events against an execution.

    Responsibilities:
    - validate explicit fill events
    - verify the execution exists
    - prevent duplicate fills
    - prevent overfilling
    - enforce PARTIALLY_FILLED / FILLED consistency
    - persist individual fill records
    - transition execution lifecycle through ExecutionLedgerService

    Does NOT:
    - place broker orders
    - infer fills from broker status
    - infer fills from executed=True
    - calculate risk
    - calculate position size
    - create or modify positions
    - calculate P&L
    - modify strategy logic
    """

    FILL_STATES = {
        "PARTIALLY_FILLED",
        "FILLED",
    }

    RECONCILABLE_EXECUTION_STATES = {
        "ACKNOWLEDGED",
        "PARTIALLY_FILLED",
    }

    def __init__(
        self,
        db: Session,
        ledger_service: ExecutionLedgerService,
    ):
        self.db = db
        self.ledger_service = ledger_service

    def reconcile(
        self,
        *,
        execution_id: str,
        fill_id: str,
        state: str,
        fill_quantity: int,
        fill_price: float,
        broker_order_id: str | None = None,
        filled_at: datetime | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Reconcile one explicit broker fill event.

        The caller must explicitly provide:
        - execution_id
        - fill_id
        - state
        - fill_quantity
        - fill_price

        The service calculates cumulative filled quantity from
        persisted ExecutionFill records.
        """

        validation = self._validate_input(
            execution_id=execution_id,
            fill_id=fill_id,
            state=state,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            broker_order_id=broker_order_id,
        )

        if not validation["valid"]:
            return self._reject(
                execution_id=execution_id,
                reason=validation["reason"],
                checks=validation["checks"],
            )

        record = self.ledger_service.get_record()

        if record is None:
            return self._reject(
                execution_id=execution_id,
                reason="Persistent execution record does not exist.",
                checks={
                    "execution_exists": False,
                },
            )

        if record.execution_id != execution_id:
            return self._reject(
                execution_id=execution_id,
                reason="Execution ID mismatch.",
                checks={
                    "execution_exists": True,
                    "execution_id_match": False,
                },
            )

        if record.current_state not in self.RECONCILABLE_EXECUTION_STATES:
            return self._reject(
                execution_id=execution_id,
                reason=(
                    f"Execution state {record.current_state} "
                    "cannot accept a fill."
                ),
                checks={
                    "execution_exists": True,
                    "execution_id_match": True,
                    "execution_state_reconcilable": False,
                },
            )

        existing_fill = (
            self.db.query(ExecutionFill)
            .filter(ExecutionFill.fill_id == fill_id)
            .first()
        )

        if existing_fill is not None:
            return self._reject(
                execution_id=execution_id,
                reason=(
                    f"Fill with fill_id {fill_id} "
                    "has already been reconciled."
                ),
                checks={
                    "execution_exists": True,
                    "execution_state_reconcilable": True,
                    "duplicate_fill": True,
                },
            )

        cumulative_before = (
            self.db.query(
                func.coalesce(
                    func.sum(ExecutionFill.fill_quantity),
                    0,
                )
            )
            .filter(
                ExecutionFill.execution_id == execution_id
            )
            .scalar()
        )

        cumulative_before = int(cumulative_before or 0)
        cumulative_after = cumulative_before + fill_quantity

        if cumulative_after > record.quantity:
            return self._reject(
                execution_id=execution_id,
                reason=(
                    "Fill would exceed the execution order quantity."
                ),
                checks={
                    "execution_exists": True,
                    "execution_state_reconcilable": True,
                    "duplicate_fill": False,
                    "quantity_not_exceeded": False,
                },
                extra={
                    "order_quantity": record.quantity,
                    "cumulative_before": cumulative_before,
                    "fill_quantity": fill_quantity,
                    "cumulative_after": cumulative_after,
                },
            )

        if state == "PARTIALLY_FILLED":
            if cumulative_after >= record.quantity:
                return self._reject(
                    execution_id=execution_id,
                    reason=(
                        "PARTIALLY_FILLED requires cumulative "
                        "filled quantity to remain below order quantity."
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconcilable": True,
                        "duplicate_fill": False,
                        "quantity_not_exceeded": True,
                        "state_quantity_consistent": False,
                    },
                    extra={
                        "order_quantity": record.quantity,
                        "cumulative_before": cumulative_before,
                        "fill_quantity": fill_quantity,
                        "cumulative_after": cumulative_after,
                    },
                )

        if state == "FILLED":
            if cumulative_after != record.quantity:
                return self._reject(
                    execution_id=execution_id,
                    reason=(
                        "FILLED requires cumulative filled quantity "
                        "to equal the order quantity."
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconcilable": True,
                        "duplicate_fill": False,
                        "quantity_not_exceeded": True,
                        "state_quantity_consistent": False,
                    },
                    extra={
                        "order_quantity": record.quantity,
                        "cumulative_before": cumulative_before,
                        "fill_quantity": fill_quantity,
                        "cumulative_after": cumulative_after,
                    },
                )

        next_sequence = (
            self.db.query(
                func.coalesce(
                    func.max(ExecutionFill.fill_sequence),
                    0,
                )
            )
            .filter(
                ExecutionFill.execution_id == execution_id
            )
            .scalar()
        )

        next_sequence = int(next_sequence or 0) + 1

        fill_record = ExecutionFill(
            fill_id=fill_id,
            execution_id=execution_id,
            broker_order_id=broker_order_id,
            fill_sequence=next_sequence,
            fill_quantity=fill_quantity,
            fill_price=fill_price,
            filled_at=filled_at or datetime.now(timezone.utc).replace(
                tzinfo=None
            ),
            created_at=datetime.now(timezone.utc).replace(
                tzinfo=None
            ),
        )

        self.db.add(fill_record)

        transition_result = self.ledger_service.transition(
            new_state=state,
            reason=reason or "Explicit broker fill reconciled.",
            metadata={
                "fill_id": fill_id,
                "fill_sequence": next_sequence,
                "fill_quantity": fill_quantity,
                "fill_price": fill_price,
                "cumulative_filled_quantity": cumulative_after,
                "remaining_quantity": (
                    record.quantity - cumulative_after
                ),
                "broker_order_id": broker_order_id,
                "filled_at": (
                    (
                        filled_at
                        or datetime.now(timezone.utc).replace(tzinfo=None)
                    ).isoformat()
                ),
            },
        )

        if transition_result.get("status") != "PASS":
            self.db.rollback()

            return self._reject(
                execution_id=execution_id,
                reason=transition_result.get(
                    "reason",
                    "Execution state transition failed.",
                ),
                checks={
                    "execution_exists": True,
                    "execution_state_reconcilable": True,
                    "duplicate_fill": False,
                    "quantity_not_exceeded": True,
                    "state_quantity_consistent": True,
                    "state_transition": False,
                    "fill_persistence": False,
                },
            )

        return {
            "reconciliation_decision": "PASS",
            "reconciled": True,
            "execution_id": execution_id,
            "fill_id": fill_id,
            "fill_sequence": next_sequence,
            "state": state,
            "fill_quantity": fill_quantity,
            "fill_price": fill_price,
            "cumulative_filled_quantity": cumulative_after,
            "remaining_quantity": (
                record.quantity - cumulative_after
            ),
            "broker_order_id": broker_order_id,
            "checks": {
                "execution_exists": True,
                "execution_id_match": True,
                "execution_state_reconcilable": True,
                "duplicate_fill": False,
                "quantity_not_exceeded": True,
                "state_quantity_consistent": True,
                "state_transition": True,
                "fill_persistence": True,
            },
            "source": "FillReconciliationService",
        }

    def _validate_input(
        self,
        *,
        execution_id: str,
        fill_id: str,
        state: str,
        fill_quantity: int,
        fill_price: float,
        broker_order_id: str | None,
    ) -> dict[str, Any]:
        checks = {
            "execution_id_valid": False,
            "fill_id_valid": False,
            "state_valid": False,
            "fill_quantity_valid": False,
            "fill_price_valid": False,
            "broker_order_id_valid": True,
        }

        if not isinstance(execution_id, str) or not execution_id.strip():
            return {
                "valid": False,
                "reason": "execution_id must be a non-empty string.",
                "checks": checks,
            }

        checks["execution_id_valid"] = True

        if not isinstance(fill_id, str) or not fill_id.strip():
            return {
                "valid": False,
                "reason": "fill_id must be a non-empty string.",
                "checks": checks,
            }

        checks["fill_id_valid"] = True

        if state not in self.FILL_STATES:
            return {
                "valid": False,
                "reason": (
                    "state must be PARTIALLY_FILLED or FILLED."
                ),
                "checks": checks,
            }

        checks["state_valid"] = True

        if (
            isinstance(fill_quantity, bool)
            or not isinstance(fill_quantity, int)
            or fill_quantity <= 0
        ):
            return {
                "valid": False,
                "reason": (
                    "fill_quantity must be a positive integer."
                ),
                "checks": checks,
            }

        checks["fill_quantity_valid"] = True

        if (
            isinstance(fill_price, bool)
            or not isinstance(fill_price, (int, float))
            or fill_price <= 0
        ):
            return {
                "valid": False,
                "reason": "fill_price must be a positive number.",
                "checks": checks,
            }

        checks["fill_price_valid"] = True

        if broker_order_id is not None:
            if (
                not isinstance(broker_order_id, str)
                or not broker_order_id.strip()
            ):
                checks["broker_order_id_valid"] = False

        if not checks["broker_order_id_valid"]:
            return {
                "valid": False,
                "reason": (
                    "broker_order_id must be a non-empty string "
                    "when supplied."
                ),
                "checks": checks,
            }

        return {
            "valid": True,
            "reason": None,
            "checks": checks,
        }

    @staticmethod
    def _reject(
        *,
        execution_id: str,
        reason: str,
        checks: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "reconciliation_decision": "REJECT",
            "reconciled": False,
            "execution_id": execution_id,
            "reason": reason,
            "checks": checks,
            "source": "FillReconciliationService",
        }

        if extra:
            result.update(extra)

        return result