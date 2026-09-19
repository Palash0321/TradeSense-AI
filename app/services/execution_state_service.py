from datetime import datetime, timezone
from typing import Any


class ExecutionStateService:
    """
    Tracks the lifecycle of an execution attempt in memory.

    This is an execution-state/audit boundary only.

    It does NOT:
    - generate trading signals
    - approve trades
    - calculate risk
    - calculate position size
    - place broker orders
    - modify strategy logic
    - infer broker state without an explicit update
    """

    VALID_STATES = {
        "CREATED",
        "VALIDATED",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "PARTIALLY_FILLED",
        "FILLED",
        "COMPLETED",
        "REJECTED",
        "FAILED",
        "CANCELLED",
    }

    TERMINAL_STATES = {
        "COMPLETED",
        "REJECTED",
        "FAILED",
        "CANCELLED",
    }

    ALLOWED_TRANSITIONS = {
        "CREATED": {
            "VALIDATED",
            "REJECTED",
            "FAILED",
            "CANCELLED",
        },
        "VALIDATED": {
            "SUBMITTED",
            "REJECTED",
            "FAILED",
            "CANCELLED",
        },
        "SUBMITTED": {
            "ACKNOWLEDGED",
            "REJECTED",
            "FAILED",
            "CANCELLED",
        },
        "ACKNOWLEDGED": {
            "PARTIALLY_FILLED",
            "FILLED",
            "REJECTED",
            "FAILED",
            "CANCELLED",
        },
        "PARTIALLY_FILLED": {
            "PARTIALLY_FILLED",
            "FILLED",
            "REJECTED",
            "FAILED",
            "CANCELLED",
        },
        "FILLED": {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        },
        "COMPLETED": set(),
        "REJECTED": set(),
        "FAILED": set(),
        "CANCELLED": set(),
    }

    def __init__(self, execution_id: str):
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution_id must be a non-empty string.")

        self.execution_id = execution_id
        self.state = "CREATED"
        self.history = []

        self._record_event(
            state="CREATED",
            reason="Execution lifecycle created.",
            metadata={},
        )

    def transition(
        self,
        new_state: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if new_state not in self.VALID_STATES:
            return self._reject_transition(
                new_state=new_state,
                reason="Invalid execution state.",
            )

        if self.state in self.TERMINAL_STATES:
            return self._reject_transition(
                new_state=new_state,
                reason=(
                    f"Execution is already in terminal state "
                    f"{self.state}."
                ),
            )

        allowed_states = self.ALLOWED_TRANSITIONS.get(
            self.state,
            set(),
        )

        if new_state not in allowed_states:
            return self._reject_transition(
                new_state=new_state,
                reason=(
                    f"Transition from {self.state} "
                    f"to {new_state} is not allowed."
                ),
            )

        previous_state = self.state
        self.state = new_state

        self._record_event(
            state=new_state,
            reason=reason,
            metadata=metadata or {},
        )

        return {
            "status": "PASS",
            "execution_id": self.execution_id,
            "previous_state": previous_state,
            "state": self.state,
            "terminal": self.state in self.TERMINAL_STATES,
            "source": "ExecutionStateService",
        }

    def get_state(self):
        return {
            "status": "PASS",
            "execution_id": self.execution_id,
            "state": self.state,
            "terminal": self.state in self.TERMINAL_STATES,
            "history": list(self.history),
            "source": "ExecutionStateService",
        }

    def _record_event(
        self,
        state: str,
        reason: str | None,
        metadata: dict[str, Any],
    ):
        self.history.append(
            {
                "execution_id": self.execution_id,
                "state": state,
                "reason": reason,
                "metadata": metadata,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _reject_transition(
        self,
        new_state: str,
        reason: str,
    ):
        return {
            "status": "REJECT",
            "execution_id": self.execution_id,
            "state": self.state,
            "requested_state": new_state,
            "reason": reason,
            "source": "ExecutionStateService",
        }