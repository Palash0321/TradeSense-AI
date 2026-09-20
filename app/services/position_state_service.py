from datetime import datetime, timezone
from typing import Any


class PositionStateService:
    """
    Tracks the lifecycle of a trading position in memory.

    This service is a position-state boundary only.

    It does NOT:
    - generate trading signals
    - approve trades
    - calculate risk
    - calculate position size
    - place broker orders
    - execute trades
    - calculate P&L
    - modify strategy logic
    """

    VALID_STATES = {
        "OPEN",
        "PARTIALLY_CLOSED",
        "CLOSED",
    }

    TERMINAL_STATES = {
        "CLOSED",
    }

    ALLOWED_TRANSITIONS = {
        "OPEN": {
            "PARTIALLY_CLOSED",
            "CLOSED",
        },
        "PARTIALLY_CLOSED": {
            "PARTIALLY_CLOSED",
            "CLOSED",
        },
        "CLOSED": set(),
    }

    def __init__(
        self,
        position_id: str,
        initial_state: str = "OPEN",
        record_initial_event: bool = True,
    ):
        if not isinstance(position_id, str) or not position_id.strip():
            raise ValueError(
                "position_id must be a non-empty string."
            )

        if initial_state not in self.VALID_STATES:
            raise ValueError(
                "initial_state must be a valid position state."
            )

        self.position_id = position_id
        self.state = initial_state
        self.history = []

        if record_initial_event:
            self._record_event(
                state=initial_state,
                reason=(
                    "Position lifecycle created."
                    if initial_state == "OPEN"
                    else "Position lifecycle initialized."
                ),
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
                reason="Invalid position state.",
            )

        if self.state in self.TERMINAL_STATES:
            return self._reject_transition(
                new_state=new_state,
                reason=(
                    f"Position is already in terminal state "
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
            "position_id": self.position_id,
            "previous_state": previous_state,
            "state": self.state,
            "terminal": (
                self.state in self.TERMINAL_STATES
            ),
            "source": "PositionStateService",
        }

    def get_state(self):
        return {
            "status": "PASS",
            "position_id": self.position_id,
            "state": self.state,
            "terminal": (
                self.state in self.TERMINAL_STATES
            ),
            "history": list(self.history),
            "source": "PositionStateService",
        }

    def _record_event(
        self,
        state: str,
        reason: str | None,
        metadata: dict[str, Any],
    ):
        self.history.append(
            {
                "position_id": self.position_id,
                "state": state,
                "reason": reason,
                "metadata": metadata,
                "timestamp": (
                    datetime.now(timezone.utc).isoformat()
                ),
            }
        )

    def _reject_transition(
        self,
        new_state: str,
        reason: str,
    ):
        return {
            "status": "REJECT",
            "position_id": self.position_id,
            "state": self.state,
            "requested_state": new_state,
            "reason": reason,
            "source": "PositionStateService",
        }