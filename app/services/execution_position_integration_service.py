from __future__ import annotations

from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.execution_fill import ExecutionFill
from app.models.execution_record import ExecutionRecord
from app.models.position import Position
from app.models.position_fill_allocation import PositionFillAllocation
from app.services.position_service import PositionService


class ExecutionPositionIntegrationService:
    """
    Integrates already-reconciled execution fills into persistent positions.

    Responsibilities:
    - consume persisted ExecutionFill records
    - verify execution/fill consistency
    - provide fill-level idempotency
    - resolve the target position
    - create a new position from the first fill
    - increase an existing position from subsequent fills
    - persist PositionFillAllocation
    - keep position mutation and fill allocation atomic

    This service does NOT:
    - reconcile broker fills
    - create ExecutionFill records
    - place broker orders
    - generate signals
    - calculate risk
    - calculate position size
    - modify execution lifecycle state
    - calculate P&L
    - modify strategy logic
    """

    RECONCILED_EXECUTION_STATES = {
        "PARTIALLY_FILLED",
        "FILLED",
    }

    def __init__(self, db: Session):
        self.db = db

    def apply_reconciled_fill(
        self,
        *,
        execution_id: str,
        fill_id: str,
        position_id: str | None = None,
        stop_loss: float | None = None,
        target1: float | None = None,
        target2: float | None = None,
        target3: float | None = None,
    ) -> dict[str, Any]:
        """
        Apply one already-reconciled ExecutionFill to a Position.

        Position creation uses the actual ExecutionFill.fill_price.

        Subsequent fills increase the existing position using
        PositionService.increase_position().

        PositionFillAllocation provides durable fill-level idempotency
        and traceability.

        position_id is optional:
        - supplied -> explicitly target that position
        - omitted + one existing position for execution -> continue it
        - omitted + multiple positions -> reject as ambiguous
        - omitted + no existing position -> create a new position
        """

        validation = self._validate_input(
            execution_id=execution_id,
            fill_id=fill_id,
            position_id=position_id,
        )

        if not validation["valid"]:
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason=validation["reason"],
                checks=validation["checks"],
            )

        # Lock the execution row for the duration of this integration
        # transaction where the database supports row-level locking.
        execution = (
            self.db.query(ExecutionRecord)
            .filter(
                ExecutionRecord.execution_id == execution_id,
            )
            .with_for_update()
            .first()
        )

        if execution is None:
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason="Persistent execution record does not exist.",
                checks={
                    "execution_exists": False,
                },
            )

        if execution.current_state not in self.RECONCILED_EXECUTION_STATES:
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason=(
                    f"Execution state {execution.current_state} "
                    "is not eligible for position integration."
                ),
                checks={
                    "execution_exists": True,
                    "execution_state_reconciled": False,
                },
            )

        # The fill must already exist because this service consumes
        # reconciled fills; it must never create one itself.
        fill = (
            self.db.query(ExecutionFill)
            .filter(
                ExecutionFill.fill_id == fill_id,
            )
            .first()
        )

        if fill is None:
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason=(
                    "ExecutionFill does not exist. "
                    "Only already-reconciled fills can be integrated."
                ),
                checks={
                    "execution_exists": True,
                    "execution_state_reconciled": True,
                    "fill_exists": False,
                },
            )

        if fill.execution_id != execution_id:
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason="ExecutionFill belongs to a different execution.",
                checks={
                    "execution_exists": True,
                    "execution_state_reconciled": True,
                    "fill_exists": True,
                    "fill_execution_match": False,
                },
            )

        # First idempotency boundary:
        # the same fill must never increase a position twice.
        existing_allocation = (
            self.db.query(PositionFillAllocation)
            .filter(
                PositionFillAllocation.fill_id == fill_id,
            )
            .first()
        )

        if existing_allocation is not None:
            if (
                position_id is not None
                and existing_allocation.position_id != position_id
            ):
                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=(
                        "Fill has already been allocated to a different "
                        "position_id."
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconciled": True,
                        "fill_exists": True,
                        "fill_execution_match": True,
                        "duplicate_allocation": True,
                        "position_id_consistent": False,
                    },
                )

            position = (
                self.db.query(Position)
                .filter(
                    Position.position_id == existing_allocation.position_id,
                )
                .first()
            )

            if position is None:
                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=(
                        "Existing fill allocation points to a missing "
                        "position."
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconciled": True,
                        "fill_exists": True,
                        "fill_execution_match": True,
                        "duplicate_allocation": True,
                        "allocated_position_exists": False,
                    },
                )

            return self._idempotent_result(
                execution=execution,
                fill=fill,
                allocation=existing_allocation,
                position=position,
            )

        # Resolve which position should receive this fill.
        resolution = self._resolve_position(
            execution=execution,
            fill=fill,
            position_id=position_id,
        )

        if resolution["status"] != "PASS":
            return self._reject(
                execution_id=execution_id,
                fill_id=fill_id,
                reason=resolution["reason"],
                checks=resolution.get("checks", {}),
            )

        position = resolution.get("position")

        position_service = PositionService(
            db=self.db,
            user_id=execution.user_id,
        )

        if position is None:
            # First fill for this execution:
            # create the position using ACTUAL fill price.
            position_result = position_service.create(
                symbol=execution.symbol,
                direction=execution.direction,
                quantity=fill.fill_quantity,
                average_entry_price=fill.fill_price,
                current_price=fill.fill_price,
                stop_loss=stop_loss,
                target1=target1,
                target2=target2,
                target3=target3,
                position_id=position_id,
                commit=False,
            )

            if position_result.get("status") != "PASS":
                self.db.rollback()

                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=position_result.get(
                        "reason",
                        "Position creation failed.",
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconciled": True,
                        "fill_exists": True,
                        "fill_execution_match": True,
                        "duplicate_allocation": False,
                        "position_creation": False,
                    },
                )

            target_position_id = position_result["position_id"]

            position = (
                self.db.query(Position)
                .filter(
                    Position.position_id == target_position_id,
                )
                .first()
            )

            if position is None:
                self.db.rollback()

                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=(
                        "Position was created but could not be retrieved "
                        "inside the integration transaction."
                    ),
                    checks={
                        "position_creation": True,
                        "position_retrieval": False,
                    },
                )

            integration_action = "POSITION_CREATED"

        else:
            # Subsequent fill:
            # PositionService owns the actual position mutation.
            position_result = position_service.increase_position(
                position_id=position.position_id,
                fill_quantity=fill.fill_quantity,
                fill_price=fill.fill_price,
                commit=False,
            )

            if position_result.get("status") != "PASS":
                self.db.rollback()

                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=position_result.get(
                        "reason",
                        "Position increase failed.",
                    ),
                    checks={
                        "execution_exists": True,
                        "execution_state_reconciled": True,
                        "fill_exists": True,
                        "fill_execution_match": True,
                        "duplicate_allocation": False,
                        "target_position_exists": True,
                        "position_mutation": False,
                    },
                )

            position = (
                self.db.query(Position)
                .filter(
                    Position.position_id == position.position_id,
                )
                .first()
            )

            if position is None:
                self.db.rollback()

                return self._reject(
                    execution_id=execution_id,
                    fill_id=fill_id,
                    reason=(
                        "Position was updated but could not be retrieved "
                        "inside the integration transaction."
                    ),
                    checks={
                        "position_mutation": True,
                        "position_retrieval": False,
                    },
                )

            integration_action = "POSITION_INCREASED"

        allocation = PositionFillAllocation(
            position_id=position.position_id,
            fill_id=fill.fill_id,
            execution_id=fill.execution_id,
            fill_sequence=fill.fill_sequence,
            fill_quantity=fill.fill_quantity,
            fill_price=fill.fill_price,
        )

        self.db.add(allocation)

        try:
            self.db.commit()
            self.db.refresh(allocation)
            self.db.refresh(position)
        except IntegrityError:
            # The unique fill_id allocation is the authoritative
            # idempotency boundary for concurrent/repeated application.
            self.db.rollback()

            existing_allocation = (
                self.db.query(PositionFillAllocation)
                .filter(
                    PositionFillAllocation.fill_id == fill_id,
                )
                .first()
            )

            if existing_allocation is not None:
                existing_position = (
                    self.db.query(Position)
                    .filter(
                        Position.position_id
                        == existing_allocation.position_id,
                    )
                    .first()
                )

                if existing_position is not None:
                    return self._idempotent_result(
                        execution=execution,
                        fill=fill,
                        allocation=existing_allocation,
                        position=existing_position,
                    )

            raise

        except Exception:
            self.db.rollback()
            raise

        return {
            "status": "PASS",
            "integration_decision": integration_action,
            "idempotent": False,
            "execution_id": execution.execution_id,
            "fill_id": fill.fill_id,
            "fill_sequence": fill.fill_sequence,
            "fill_quantity": fill.fill_quantity,
            "fill_price": fill.fill_price,
            "position_id": position.position_id,
            "position": self._position_snapshot(position),
            "allocation_id": allocation.id,
            "allocation": self._allocation_snapshot(allocation),
            "checks": {
                "execution_exists": True,
                "execution_state_reconciled": True,
                "fill_exists": True,
                "fill_execution_match": True,
                "duplicate_allocation": False,
                "position_resolved": True,
                "position_mutation": True,
                "allocation_persisted": True,
                "atomic_commit": True,
            },
            "source": "ExecutionPositionIntegrationService",
        }

    def _resolve_position(
        self,
        *,
        execution: ExecutionRecord,
        fill: ExecutionFill,
        position_id: str | None,
    ) -> dict[str, Any]:
        """
        Resolve the target position.

        Explicit position_id always wins, subject to identity validation.

        Without an explicit position_id:
        - one prior position for this execution -> continue it
        - multiple prior positions -> reject as ambiguous
        - no prior position -> create a new position
        """

        if position_id is not None:
            position = (
                self.db.query(Position)
                .filter(
                    Position.position_id == position_id,
                )
                .with_for_update()
                .first()
            )

            if position is None:
                return {
                    "status": "REJECT",
                    "reason": "Explicit position_id does not exist.",
                    "checks": {
                        "explicit_position_exists": False,
                    },
                }

            identity_check = self._validate_position_identity(
                execution=execution,
                position=position,
            )

            if not identity_check["valid"]:
                return {
                    "status": "REJECT",
                    "reason": identity_check["reason"],
                    "checks": identity_check["checks"],
                }

            return {
                "status": "PASS",
                "position": position,
                "resolution": "EXPLICIT_POSITION",
            }

        # Find all positions that have already received fills from this
        # execution.
        allocations = (
            self.db.query(PositionFillAllocation)
            .filter(
                PositionFillAllocation.execution_id == execution.execution_id,
            )
            .all()
        )

        position_ids = {
            allocation.position_id
            for allocation in allocations
        }

        if len(position_ids) > 1:
            return {
                "status": "REJECT",
                "reason": (
                    "Multiple positions are already associated with this "
                    "execution. position_id must be supplied explicitly."
                ),
                "checks": {
                    "prior_execution_positions": len(position_ids),
                    "position_resolution_ambiguous": True,
                },
            }

        if len(position_ids) == 1:
            target_position_id = next(iter(position_ids))

            position = (
                self.db.query(Position)
                .filter(
                    Position.position_id == target_position_id,
                )
                .with_for_update()
                .first()
            )

            if position is None:
                return {
                    "status": "REJECT",
                    "reason": (
                        "Prior allocation references a position that "
                        "does not exist."
                    ),
                    "checks": {
                        "prior_execution_positions": 1,
                        "prior_position_exists": False,
                    },
                }

            identity_check = self._validate_position_identity(
                execution=execution,
                position=position,
            )

            if not identity_check["valid"]:
                return {
                    "status": "REJECT",
                    "reason": identity_check["reason"],
                    "checks": identity_check["checks"],
                }

            return {
                "status": "PASS",
                "position": position,
                "resolution": "PRIOR_EXECUTION_POSITION",
            }

        # No position has received a fill from this execution yet.
        return {
            "status": "PASS",
            "position": None,
            "resolution": "NEW_POSITION",
        }

    @staticmethod
    def _validate_position_identity(
        *,
        execution: ExecutionRecord,
        position: Position,
    ) -> dict[str, Any]:
        checks = {
            "position_user_match": position.user_id == execution.user_id,
            "position_symbol_match": (
                position.symbol.strip().upper()
                == execution.symbol.strip().upper()
            ),
            "position_direction_match": (
                position.direction == execution.direction
            ),
        }

        if not checks["position_user_match"]:
            return {
                "valid": False,
                "reason": "Position user_id does not match execution user_id.",
                "checks": checks,
            }

        if not checks["position_symbol_match"]:
            return {
                "valid": False,
                "reason": "Position symbol does not match execution symbol.",
                "checks": checks,
            }

        if not checks["position_direction_match"]:
            return {
                "valid": False,
                "reason": (
                    "Position direction does not match execution direction."
                ),
                "checks": checks,
            }

        return {
            "valid": True,
            "reason": None,
            "checks": checks,
        }

    @staticmethod
    def _position_snapshot(position: Position) -> dict[str, Any]:
        return {
            "position_id": position.position_id,
            "user_id": position.user_id,
            "symbol": position.symbol,
            "direction": position.direction,
            "quantity": position.quantity,
            "average_entry_price": position.average_entry_price,
            "current_price": position.current_price,
            "stop_loss": position.stop_loss,
            "target1": position.target1,
            "target2": position.target2,
            "target3": position.target3,
            "state": position.state,
            "opened_at": position.opened_at,
            "closed_at": position.closed_at,
        }

    @staticmethod
    def _allocation_snapshot(
        allocation: PositionFillAllocation,
    ) -> dict[str, Any]:
        return {
            "allocation_id": allocation.id,
            "position_id": allocation.position_id,
            "fill_id": allocation.fill_id,
            "execution_id": allocation.execution_id,
            "fill_sequence": allocation.fill_sequence,
            "fill_quantity": allocation.fill_quantity,
            "fill_price": allocation.fill_price,
            "applied_at": allocation.applied_at,
            "created_at": allocation.created_at,
        }

    @staticmethod
    def _idempotent_result(
        *,
        execution: ExecutionRecord,
        fill: ExecutionFill,
        allocation: PositionFillAllocation,
        position: Position,
    ) -> dict[str, Any]:
        return {
            "status": "PASS",
            "integration_decision": "ALREADY_APPLIED",
            "idempotent": True,
            "execution_id": execution.execution_id,
            "fill_id": fill.fill_id,
            "fill_sequence": fill.fill_sequence,
            "fill_quantity": fill.fill_quantity,
            "fill_price": fill.fill_price,
            "position_id": position.position_id,
            "position": ExecutionPositionIntegrationService._position_snapshot(
                position
            ),
            "allocation_id": allocation.id,
            "allocation": (
                ExecutionPositionIntegrationService._allocation_snapshot(
                    allocation
                )
            ),
            "checks": {
                "execution_exists": True,
                "execution_state_reconciled": True,
                "fill_exists": True,
                "fill_execution_match": True,
                "duplicate_allocation": True,
                "position_resolved": True,
                "position_mutation": False,
                "allocation_persisted": True,
                "idempotency_protected": True,
            },
            "source": "ExecutionPositionIntegrationService",
        }

    @staticmethod
    def _validate_input(
        *,
        execution_id: str,
        fill_id: str,
        position_id: str | None,
    ) -> dict[str, Any]:
        checks = {
            "execution_id_valid": False,
            "fill_id_valid": False,
            "position_id_valid": True,
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

        if position_id is not None:
            if (
                not isinstance(position_id, str)
                or not position_id.strip()
            ):
                checks["position_id_valid"] = False

                return {
                    "valid": False,
                    "reason": (
                        "position_id must be a non-empty string "
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
        fill_id: str,
        reason: str,
        checks: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "REJECT",
            "integration_decision": "REJECTED",
            "idempotent": False,
            "execution_id": execution_id,
            "fill_id": fill_id,
            "reason": reason,
            "checks": checks,
            "source": "ExecutionPositionIntegrationService",
        }