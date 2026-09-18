class RiskBudgetService:
    """
    Validates and transports an explicitly supplied monetary risk budget.

    This service does NOT:
    - choose a risk-per-trade percentage
    - invent a risk policy
    - calculate a risk budget from equity
    - calculate position size
    - modify strategy signals
    - approve or reject trade candidates
    - place or execute orders

    It only validates that:
    1. portfolio state is valid
    2. an explicit monetary risk budget was supplied
    3. the supplied risk budget is numeric
    4. the supplied risk budget is strictly positive
    """

    def __init__(self, portfolio_state, risk_budget):
        self.portfolio_state = portfolio_state or {}
        self.risk_budget = risk_budget

    def validate(self):
        checks = {
            "portfolio_state_valid": False,
            "risk_budget_present": False,
            "risk_budget_numeric": False,
            "risk_budget_positive": False,
        }

        reasons = []

        if self.portfolio_state.get("status") != "PASS":
            reasons.append("Portfolio risk state is invalid.")
        else:
            checks["portfolio_state_valid"] = True

        if self.risk_budget is None:
            reasons.append("Risk budget was not supplied.")
        else:
            checks["risk_budget_present"] = True

        numeric = self._is_numeric(self.risk_budget)
        if self.risk_budget is not None and not numeric:
            reasons.append("Risk budget must be numeric.")
        elif numeric:
            checks["risk_budget_numeric"] = True

        if numeric:
            try:
                positive = float(self.risk_budget) > 0
            except (TypeError, ValueError):
                positive = False

            if positive:
                checks["risk_budget_positive"] = True
            else:
                reasons.append("Risk budget must be greater than zero.")

        passed = not reasons

        if passed:
            validated_budget = float(self.risk_budget)
        else:
            validated_budget = None

        return {
            "status": "PASS" if passed else "REJECT",
            "passed": passed,
            "user_id": self.portfolio_state.get("user_id"),
            "equity": self.portfolio_state.get("equity"),
            "cash_balance": self.portfolio_state.get("cash_balance"),
            "gross_exposure": self.portfolio_state.get("gross_exposure"),
            "risk_budget": validated_budget,
            "risk_budget_policy_defined": False,
            "checks": checks,
            "reasons": reasons,
            "source": "RiskBudgetService",
        }

    @staticmethod
    def _is_numeric(value):
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        try:
            return value == value
        except Exception:
            return False