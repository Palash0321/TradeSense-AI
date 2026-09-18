from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.auth.dependencies import get_current_user


client = TestClient(app)


def _authenticated_user():
    return SimpleNamespace(
        id=1,
        full_name="Test User",
        email="test@example.com",
    )


def _authenticated_client():
    app.dependency_overrides[get_current_user] = _authenticated_user
    return client


def _clear_auth_override():
    app.dependency_overrides.pop(get_current_user, None)


def _valid_trade_candidate():
    return {
        "symbol": "RELIANCE.NS",
        "direction": "LONG",
        "decision": "BUY",
        "actionable": True,
        "setup_type": "PULLBACK",
        "preferred_setup": "PULLBACK",
        "validation_status": "PASS",
        "entry": 100.0,
        "entry_low": None,
        "entry_high": None,
        "stop_loss": 98.0,
        "target1": 103.0,
        "target2": 105.0,
        "target3": 107.0,
        "risk_reward": 1.5,
        "ai_confidence": 75.0,
        "breakout_trigger": False,
        "breakout_level": None,
    }


def test_prepare_requires_authentication():
    _clear_auth_override()

    response = client.post(
        "/api/trade-execution/prepare",
        json={
            "trade_candidate": _valid_trade_candidate(),
            "risk_budget": 1000.0,
        },
    )

    assert response.status_code in (401, 403)


@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_prepare_missing_risk_budget(mock_runtime_service):
    _authenticated_client()

    try:
        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": "risk_budget",
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }

        response = client.post(
            "/api/trade-execution/prepare",
            json={
                "trade_candidate": _valid_trade_candidate(),
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["runtime_decision"] == "REJECT"
        assert data["ready_for_execution"] is False
        assert data["failed_stage"] == "risk_budget"

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_prepare_invalid_risk_budget(mock_runtime_service):
    _authenticated_client()

    try:
        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": "risk_budget",
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }

        response = client.post(
            "/api/trade-execution/prepare",
            json={
                "trade_candidate": _valid_trade_candidate(),
                "risk_budget": -100.0,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["runtime_decision"] == "REJECT"
        assert data["ready_for_execution"] is False
        assert data["failed_stage"] == "risk_budget"

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_prepare_missing_risk_budget(mock_runtime_service):
    _authenticated_client()

    try:
        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": "risk_budget",
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }

        response = client.post(
            "/api/trade-execution/prepare",
            json={
                "trade_candidate": _valid_trade_candidate(),
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["user_id"] == 1
        assert data["runtime_decision"] == "REJECT"
        assert data["ready_for_execution"] is False
        assert data["failed_stage"] == "risk_budget"

        mock_runtime.prepare.assert_called_once()

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_prepare_invalid_risk_budget(mock_runtime_service):
    _authenticated_client()

    try:
        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": "risk_budget",
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }

        response = client.post(
            "/api/trade-execution/prepare",
            json={
                "trade_candidate": _valid_trade_candidate(),
                "risk_budget": -100.0,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["user_id"] == 1
        assert data["runtime_decision"] == "REJECT"
        assert data["ready_for_execution"] is False
        assert data["failed_stage"] == "risk_budget"

        mock_runtime.prepare.assert_called_once()

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.PaperExecutionService")
@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_paper_execute_stops_when_runtime_not_ready(
    mock_runtime_service,
    mock_paper_execution_service,
):
    _authenticated_client()

    try:
        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "REJECT",
            "ready_for_execution": False,
            "failed_stage": "position_sizing",
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": None,
            "source": "TradeExecutionRuntimeService",
        }

        response = client.post(
            "/api/trade-execution/paper-execute",
            json={
                "trade_candidate": _valid_trade_candidate(),
                "risk_budget": 1000.0,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["execution_status"] == "REJECTED"
        assert data["runtime"]["runtime_decision"] == "REJECT"
        assert data["runtime"]["ready_for_execution"] is False
        assert data["runtime"]["failed_stage"] == "position_sizing"
        assert data["paper_execution"] is None

        mock_paper_execution_service.assert_not_called()

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.PaperExecutionService")
@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_paper_execute_success(
    mock_runtime_service,
    mock_paper_execution_service,
):
    _authenticated_client()

    try:
        order_intent = {
            "status": "PASS",
            "ready": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "PASS",
            "ready_for_execution": True,
            "failed_stage": None,
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": {
                "execution_decision": "PASS",
                "ready_for_execution": True,
                "order_intent": order_intent,
            },
            "source": "TradeExecutionRuntimeService",
        }

        mock_paper = mock_paper_execution_service.return_value

        mock_paper.execute.return_value = {
            "status": "PASS",
            "executed": True,
            "message": "Paper BUY executed",
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
            "execution_price": 100.0,
            "total_amount": 1000.0,
            "remaining_balance": 999000.0,
        }

        response = client.post(
            "/api/trade-execution/paper-execute",
            json={
                "trade_candidate": _valid_trade_candidate(),
                "risk_budget": 1000.0,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["execution_status"] == "EXECUTED"
        assert data["runtime"]["runtime_decision"] == "PASS"
        assert data["runtime"]["ready_for_execution"] is True

        assert data["paper_execution"]["status"] == "PASS"
        assert data["paper_execution"]["executed"] is True
        assert data["paper_execution"]["quantity"] == 10

        mock_paper.execute.assert_called_once_with(order_intent)

    finally:
        _clear_auth_override()


@patch("app.api.trade_execution.PaperExecutionService")
@patch("app.api.trade_execution.TradeExecutionRuntimeService")
def test_paper_execute_propagates_paper_rejection(
    mock_runtime_service,
    mock_paper_execution_service,
):
    _authenticated_client()

    try:
        order_intent = {
            "status": "PASS",
            "ready": True,
            "symbol": "RELIANCE.NS",
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": 10,
        }

        mock_runtime = mock_runtime_service.return_value

        mock_runtime.prepare.return_value = {
            "runtime_decision": "PASS",
            "ready_for_execution": True,
            "failed_stage": None,
            "portfolio_risk_state": {
                "status": "PASS",
                "user_id": 1,
            },
            "execution": {
                "execution_decision": "PASS",
                "ready_for_execution": True,
                "order_intent": order_intent,
            },
            "source": "TradeExecutionRuntimeService",
        }

        mock_paper = mock_paper_execution_service.return_value

        mock_paper.execute.return_value = {
            "status": "REJECT",
            "executed": False,
            "reason": "Insufficient paper balance",
        }

        response = client.post(
            "/api/trade-execution/paper-execute",
            json={
                "trade_candidate": _valid_trade_candidate(),
                "risk_budget": 1000.0,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["execution_status"] == "REJECTED"
        assert data["runtime"]["runtime_decision"] == "PASS"
        assert data["runtime"]["ready_for_execution"] is True

        assert data["paper_execution"]["status"] == "REJECT"
        assert data["paper_execution"]["executed"] is False
        assert data["paper_execution"]["reason"] == "Insufficient paper balance"

    finally:
        _clear_auth_override()