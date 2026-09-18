from app.services.risk_budget_service import RiskBudgetService


def valid_portfolio_state():
    return {
        "status": "PASS",
        "passed": True,
        "user_id": 1,
        "cash_balance": 900000.0,
        "holdings_market_value": 100000.0,
        "gross_exposure": 100000.0,
        "equity": 1000000.0,
        "holdings_count": 2,
        "open_risk_known": False,
        "open_risk": None,
        "risk_budget_policy_defined": False,
        "risk_budget": None,
        "checks": {
            "account_exists": True,
            "cash_balance_valid": True,
            "equity_valid": True,
            "portfolio_exposure_valid": True,
        },
        "reasons": [],
        "source": "PortfolioRiskStateService",
    }


def test_valid_explicit_risk_budget_passes():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget=5000,
    ).validate()

    assert result["status"] == "PASS"
    assert result["passed"] is True
    assert result["risk_budget"] == 5000.0
    assert result["checks"]["portfolio_state_valid"] is True
    assert result["checks"]["risk_budget_present"] is True
    assert result["checks"]["risk_budget_numeric"] is True
    assert result["checks"]["risk_budget_positive"] is True
    assert result["reasons"] == []


def test_missing_risk_budget_rejects():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget=None,
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None
    assert result["checks"]["risk_budget_present"] is False
    assert "Risk budget was not supplied." in result["reasons"]


def test_zero_risk_budget_rejects():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget=0,
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None
    assert result["checks"]["risk_budget_positive"] is False
    assert "Risk budget must be greater than zero." in result["reasons"]


def test_negative_risk_budget_rejects():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget=-100,
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None
    assert result["checks"]["risk_budget_positive"] is False


def test_non_numeric_risk_budget_rejects():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget="5000",
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None
    assert result["checks"]["risk_budget_numeric"] is False
    assert "Risk budget must be numeric." in result["reasons"]


def test_invalid_portfolio_state_rejects():
    invalid_state = valid_portfolio_state()
    invalid_state["status"] = "REJECT"
    invalid_state["passed"] = False

    result = RiskBudgetService(
        portfolio_state=invalid_state,
        risk_budget=5000,
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None
    assert result["checks"]["portfolio_state_valid"] is False
    assert "Portfolio risk state is invalid." in result["reasons"]


def test_boolean_risk_budget_rejects():
    result = RiskBudgetService(
        portfolio_state=valid_portfolio_state(),
        risk_budget=True,
    ).validate()

    assert result["status"] == "REJECT"
    assert result["passed"] is False
    assert result["risk_budget"] is None