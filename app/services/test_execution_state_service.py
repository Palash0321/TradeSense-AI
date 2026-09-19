from app.services.execution_state_service import ExecutionStateService


def test_execution_starts_in_created_state():
    service = ExecutionStateService("EXEC-001")

    result = service.get_state()

    assert result["status"] == "PASS"
    assert result["execution_id"] == "EXEC-001"
    assert result["state"] == "CREATED"
    assert result["terminal"] is False
    assert len(result["history"]) == 1


def test_valid_execution_lifecycle():
    service = ExecutionStateService("EXEC-002")

    states = [
        "VALIDATED",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "FILLED",
        "COMPLETED",
    ]

    for state in states:
        result = service.transition(
            state,
            reason=f"Moved to {state}.",
        )

        assert result["status"] == "PASS"
        assert result["state"] == state

    final_state = service.get_state()

    assert final_state["state"] == "COMPLETED"
    assert final_state["terminal"] is True
    assert len(final_state["history"]) == 6


def test_partial_fill_lifecycle():
    service = ExecutionStateService("EXEC-003")

    assert service.transition("VALIDATED")["status"] == "PASS"
    assert service.transition("SUBMITTED")["status"] == "PASS"
    assert service.transition("ACKNOWLEDGED")["status"] == "PASS"

    result = service.transition(
        "PARTIALLY_FILLED",
        metadata={"filled_quantity": 40},
    )

    assert result["status"] == "PASS"
    assert result["state"] == "PARTIALLY_FILLED"

    result = service.transition(
        "PARTIALLY_FILLED",
        metadata={"filled_quantity": 70},
    )

    assert result["status"] == "PASS"
    assert result["state"] == "PARTIALLY_FILLED"

    result = service.transition(
        "FILLED",
        metadata={"filled_quantity": 100},
    )

    assert result["status"] == "PASS"
    assert result["state"] == "FILLED"

    result = service.transition("COMPLETED")

    assert result["status"] == "PASS"
    assert result["state"] == "COMPLETED"


def test_rejection_is_terminal():
    service = ExecutionStateService("EXEC-004")

    result = service.transition(
        "REJECTED",
        reason="Broker rejected order.",
    )

    assert result["status"] == "PASS"
    assert result["state"] == "REJECTED"
    assert result["terminal"] is True

    state = service.get_state()

    assert state["state"] == "REJECTED"
    assert state["terminal"] is True


def test_failed_execution_is_terminal():
    service = ExecutionStateService("EXEC-005")

    service.transition("VALIDATED")

    result = service.transition(
        "FAILED",
        reason="Execution infrastructure failure.",
    )

    assert result["status"] == "PASS"
    assert result["state"] == "FAILED"
    assert result["terminal"] is True


def test_cancelled_execution_is_terminal():
    service = ExecutionStateService("EXEC-006")

    service.transition("VALIDATED")

    result = service.transition(
        "CANCELLED",
        reason="Execution cancelled.",
    )

    assert result["status"] == "PASS"
    assert result["state"] == "CANCELLED"
    assert result["terminal"] is True


def test_invalid_state_is_rejected():
    service = ExecutionStateService("EXEC-007")

    result = service.transition("UNKNOWN_STATE")

    assert result["status"] == "REJECT"
    assert result["state"] == "CREATED"
    assert result["requested_state"] == "UNKNOWN_STATE"


def test_invalid_transition_is_rejected():
    service = ExecutionStateService("EXEC-008")

    result = service.transition("FILLED")

    assert result["status"] == "REJECT"
    assert result["state"] == "CREATED"
    assert result["requested_state"] == "FILLED"


def test_terminal_state_cannot_transition():
    service = ExecutionStateService("EXEC-009")

    service.transition("REJECTED")

    result = service.transition("VALIDATED")

    assert result["status"] == "REJECT"
    assert result["state"] == "REJECTED"
    assert result["requested_state"] == "VALIDATED"


def test_history_preserves_transition_sequence():
    service = ExecutionStateService("EXEC-010")

    service.transition("VALIDATED")
    service.transition("SUBMITTED")
    service.transition("ACKNOWLEDGED")
    service.transition("FILLED")
    service.transition("COMPLETED")

    history = service.get_state()["history"]

    assert [event["state"] for event in history] == [
        "CREATED",
        "VALIDATED",
        "SUBMITTED",
        "ACKNOWLEDGED",
        "FILLED",
        "COMPLETED",
    ]


def test_transition_metadata_is_preserved():
    service = ExecutionStateService("EXEC-011")

    service.transition(
        "VALIDATED",
        reason="Risk and sizing checks passed.",
        metadata={
            "risk_decision": "PASS",
            "sizing_decision": "PASS",
        },
    )

    event = service.get_state()["history"][-1]

    assert event["state"] == "VALIDATED"
    assert event["reason"] == "Risk and sizing checks passed."
    assert event["metadata"]["risk_decision"] == "PASS"
    assert event["metadata"]["sizing_decision"] == "PASS"


def test_empty_execution_id_is_rejected():
    try:
        ExecutionStateService("")
        assert False
    except ValueError as exc:
        assert str(exc) == "execution_id must be a non-empty string."