from app.services.position_state_service import (
    PositionStateService,
)


def test_position_starts_open():
    service = PositionStateService(
        position_id="POSITION-001"
    )

    result = service.get_state()

    assert result["status"] == "PASS"
    assert result["position_id"] == "POSITION-001"
    assert result["state"] == "OPEN"
    assert result["terminal"] is False
    assert len(result["history"]) == 1


def test_open_to_partial_close():
    service = PositionStateService(
        position_id="POSITION-002"
    )

    result = service.transition(
        new_state="PARTIALLY_CLOSED"
    )

    assert result["status"] == "PASS"
    assert result["previous_state"] == "OPEN"
    assert result["state"] == "PARTIALLY_CLOSED"
    assert result["terminal"] is False


def test_open_to_closed():
    service = PositionStateService(
        position_id="POSITION-003"
    )

    result = service.transition(
        new_state="CLOSED"
    )

    assert result["status"] == "PASS"
    assert result["previous_state"] == "OPEN"
    assert result["state"] == "CLOSED"
    assert result["terminal"] is True


def test_partial_close_can_remain_partial():
    service = PositionStateService(
        position_id="POSITION-004"
    )

    assert (
        service.transition(
            "PARTIALLY_CLOSED"
        )["status"]
        == "PASS"
    )

    result = service.transition(
        "PARTIALLY_CLOSED"
    )

    assert result["status"] == "PASS"
    assert result["previous_state"] == "PARTIALLY_CLOSED"
    assert result["state"] == "PARTIALLY_CLOSED"


def test_partial_close_to_closed():
    service = PositionStateService(
        position_id="POSITION-005"
    )

    service.transition("PARTIALLY_CLOSED")

    result = service.transition("CLOSED")

    assert result["status"] == "PASS"
    assert result["state"] == "CLOSED"
    assert result["terminal"] is True


def test_closed_position_is_terminal():
    service = PositionStateService(
        position_id="POSITION-006"
    )

    service.transition("CLOSED")

    result = service.transition(
        "PARTIALLY_CLOSED"
    )

    assert result["status"] == "REJECT"
    assert result["state"] == "CLOSED"


def test_invalid_state_is_rejected():
    service = PositionStateService(
        position_id="POSITION-007"
    )

    result = service.transition("INVALID")

    assert result["status"] == "REJECT"
    assert result["state"] == "OPEN"
    assert result["requested_state"] == "INVALID"


def test_invalid_position_id_is_rejected_at_construction():
    try:
        PositionStateService(position_id="")
    except ValueError as exc:
        assert str(exc) == (
            "position_id must be a non-empty string."
        )
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )


def test_invalid_initial_state_is_rejected():
    try:
        PositionStateService(
            position_id="POSITION-008",
            initial_state="INVALID",
        )
    except ValueError as exc:
        assert str(exc) == (
            "initial_state must be a valid position state."
        )
    else:
        raise AssertionError(
            "Expected ValueError was not raised."
        )