from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio


class PortfolioRiskStateService:
    """
    Builds the current portfolio/account state required by the
    trading risk architecture.

    This service does NOT:
    - choose a risk-per-trade percentage
    - calculate a risk budget policy
    - approve or reject a trade candidate
    - calculate position size
    - modify strategy signals
    - place orders
    - execute trades
    - modify portfolio holdings

    It only reports the currently observable portfolio state.
    """

    def __init__(self, db, user_id):
        self.db = db
        self.user_id = user_id

    def get_state(self):
        account = (
            self.db.query(PaperAccount)
            .filter(
                PaperAccount.user_id == self.user_id
            )
            .first()
        )

        if account is None:
            return self._reject(
                "Paper account not found."
            )

        holdings = (
            self.db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == self.user_id
            )
            .all()
        )

        cash_balance = self._safe_number(
            account.balance
        )

        gross_exposure = 0.0
        holdings_market_value = 0.0
        holdings_count = 0

        for holding in holdings:
            quantity = self._safe_number(
                holding.quantity
            )

            current_price = self._safe_number(
                holding.current_price
            )

            if quantity <= 0:
                continue

            if current_price <= 0:
                continue

            market_value = quantity * current_price

            holdings_market_value += market_value
            gross_exposure += market_value
            holdings_count += 1

        equity = cash_balance + holdings_market_value

        return {
            "status": "PASS",
            "passed": True,
            "user_id": self.user_id,

            "cash_balance": cash_balance,

            "holdings_market_value": (
                holdings_market_value
            ),

            "gross_exposure": gross_exposure,

            "equity": equity,

            "holdings_count": holdings_count,

            "open_risk_known": False,
            "open_risk": None,

            "risk_budget_policy_defined": False,
            "risk_budget": None,

            "checks": {
                "account_exists": True,
                "cash_balance_valid": cash_balance >= 0,
                "equity_valid": equity >= 0,
                "portfolio_exposure_valid": (
                    gross_exposure >= 0
                ),
            },

            "reasons": [],

            "source": "PortfolioRiskStateService",
        }

    @staticmethod
    def _safe_number(value):
        if isinstance(value, bool):
            return 0.0

        if isinstance(value, (int, float)):
            return float(value)

        return 0.0

    @staticmethod
    def _reject(reason):
        return {
            "status": "REJECT",
            "passed": False,

            "user_id": None,

            "cash_balance": None,
            "holdings_market_value": None,
            "gross_exposure": None,
            "equity": None,
            "holdings_count": None,

            "open_risk_known": False,
            "open_risk": None,

            "risk_budget_policy_defined": False,
            "risk_budget": None,

            "checks": {
                "account_exists": False,
            },

            "reasons": [reason],

            "source": "PortfolioRiskStateService",
        }