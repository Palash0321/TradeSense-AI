from sqlalchemy.orm import Session

from app.models.paper_account import PaperAccount
from app.models.paper_portfolio import PaperPortfolio
from app.models.paper_transaction import PaperTransaction
from app.services.market_price_service import get_live_price


class PaperExecutionService:
    """
    Executes an already-approved OrderIntent against the existing
    paper-trading account.

    Responsibilities:
    - validate the execution contract
    - fetch the current market price
    - validate paper-account cash/holdings
    - execute paper BUY/SELL
    - update PaperAccount
    - update PaperPortfolio
    - create PaperTransaction

    This service does NOT:
    - generate trading signals
    - calculate entries
    - calculate stop losses
    - calculate targets
    - approve risk
    - calculate position size
    - modify strategy logic
    - communicate with a live broker
    """

    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id

    def execute(self, order_intent: dict) -> dict:
        """
        Execute a validated OrderIntent in the paper account.

        Current paper execution supports:
        - MARKET orders
        - LONG direction for BUY execution
        - existing LONG holdings for SELL execution

        STOP/LIMIT execution and true short selling are intentionally
        not simulated here because the existing paper infrastructure
        does not currently implement those mechanics.
        """

        if not isinstance(order_intent, dict):
            return self._reject(
                "Invalid order intent",
            )

        if order_intent.get("intent_decision") != "PASS":
            return self._reject(
                "Order intent is not approved",
            )

        if order_intent.get("ready_for_execution") is not True:
            return self._reject(
                "Order intent is not execution-ready",
            )

        symbol = order_intent.get("symbol")

        if not isinstance(symbol, str) or not symbol.strip():
            return self._reject(
                "Invalid symbol",
            )

        symbol = symbol.strip().upper()

        direction = order_intent.get("direction")

        if direction not in ("LONG", "SHORT"):
            return self._reject(
                "Invalid direction",
            )

        order_type = order_intent.get("order_type")

        if order_type != "MARKET":
            return self._reject(
                "Paper execution currently supports MARKET orders only",
            )

        quantity = order_intent.get("quantity")

        if isinstance(quantity, bool) or not isinstance(
            quantity,
            int,
        ):
            return self._reject(
                "Quantity must be a positive integer",
            )

        if quantity <= 0:
            return self._reject(
                "Quantity must be greater than zero",
            )

        # The current paper infrastructure does not support
        # true short selling. A SHORT signal therefore cannot
        # be silently converted into a SELL transaction.
        if direction == "SHORT":
            return self._reject(
                "SHORT execution is not supported by the current paper account",
            )

        account = (
            self.db.query(PaperAccount)
            .filter(
                PaperAccount.user_id == self.user_id,
            )
            .first()
        )

        if account is None:
            return self._reject(
                "Paper account not found",
            )

        price = get_live_price(symbol)

        if price <= 0:
            return self._reject(
                "Unable to fetch live price",
            )

        total_cost = price * quantity

        if total_cost > account.balance:
            return self._reject(
                "Insufficient paper balance",
            )

        account.balance -= total_cost

        paper_transaction = PaperTransaction(
            user_id=self.user_id,
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            price=price,
            total_amount=total_cost,
        )

        self.db.add(paper_transaction)

        holding = (
            self.db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == self.user_id,
                PaperPortfolio.symbol == symbol,
            )
            .first()
        )

        if holding is not None:
            total_quantity = holding.quantity + quantity

            total_value = (
                holding.quantity * holding.average_price
            ) + total_cost

            holding.quantity = total_quantity
            holding.average_price = (
                total_value / total_quantity
            )
            holding.current_price = price

        else:
            holding = PaperPortfolio(
                user_id=self.user_id,
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price,
            )

            self.db.add(holding)

        self.db.commit()

        return {
            "status": "PASS",
            "executed": True,
            "message": "Paper BUY executed",
            "symbol": symbol,
            "direction": direction,
            "order_type": order_type,
            "quantity": quantity,
            "execution_price": round(price, 2),
            "total_amount": round(total_cost, 2),
            "remaining_balance": round(
                account.balance,
                2,
            ),
        }

    def execute_manual_buy(self, symbol: str, quantity: int) -> dict:
        """
        Execute a manual paper BUY using the existing paper-account
        infrastructure.

        This method is intentionally separate from execute(order_intent)
        because manual paper orders are not AI-generated OrderIntents.

        It does NOT:
        - generate trading signals
        - calculate entries
        - calculate stop losses
        - calculate targets
        - approve risk
        - calculate position size
        - modify strategy logic
        - communicate with a live broker
        """
        if not isinstance(symbol, str) or not symbol.strip():
            return self._reject("Invalid symbol")

        if isinstance(quantity, bool) or not isinstance(quantity, int):
            return self._reject("Quantity must be a positive integer")

        if quantity <= 0:
            return self._reject("Quantity must be greater than zero")

        symbol = symbol.strip().upper()

        account = (
            self.db.query(PaperAccount)
            .filter(PaperAccount.user_id == self.user_id)
            .first()
        )

        if account is None:
            return self._reject("Paper account not found")

        price = get_live_price(symbol)

        if price <= 0:
            return self._reject("Unable to fetch live price")

        total_cost = price * quantity

        if total_cost > account.balance:
            return self._reject("Insufficient paper balance")

        account.balance -= total_cost

        paper_transaction = PaperTransaction(
            user_id=self.user_id,
            symbol=symbol,
            transaction_type="BUY",
            quantity=quantity,
            price=price,
            total_amount=total_cost,
        )

        self.db.add(paper_transaction)

        holding = (
            self.db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == self.user_id,
                PaperPortfolio.symbol == symbol,
            )
            .first()
        )

        if holding is not None:
            total_quantity = holding.quantity + quantity

            total_value = (
                holding.quantity * holding.average_price
            ) + total_cost

            holding.quantity = total_quantity
            holding.average_price = total_value / total_quantity
            holding.current_price = price

        else:
            holding = PaperPortfolio(
                user_id=self.user_id,
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                current_price=price,
            )

            self.db.add(holding)

        self.db.commit()

        return {
            "status": "PASS",
            "executed": True,
            "message": "Paper BUY executed",
            "symbol": symbol,
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": quantity,
            "execution_price": round(price, 2),
            "total_amount": round(total_cost, 2),
            "remaining_balance": round(account.balance, 2),
        }

    def execute_manual_sell(self, symbol: str, quantity: int) -> dict:
        """
        Execute a manual paper SELL against an existing LONG holding.

        This method preserves the current paper-trading behavior:
        - sell only an existing holding
        - reject insufficient quantity
        - credit the paper account
        - record the transaction
        - reduce the holding
        - remove the holding when quantity reaches zero

        It does NOT:
        - generate trading signals
        - calculate entries
        - calculate stop losses
        - calculate targets
        - approve risk
        - calculate position size
        - modify strategy logic
        - communicate with a live broker
        """
        if not isinstance(symbol, str) or not symbol.strip():
            return self._reject("Invalid symbol")

        if isinstance(quantity, bool) or not isinstance(quantity, int):
            return self._reject("Quantity must be a positive integer")

        if quantity <= 0:
            return self._reject("Quantity must be greater than zero")

        symbol = symbol.strip().upper()

        account = (
            self.db.query(PaperAccount)
            .filter(PaperAccount.user_id == self.user_id)
            .first()
        )

        if account is None:
            return self._reject("Paper account not found")

        holding = (
            self.db.query(PaperPortfolio)
            .filter(
                PaperPortfolio.user_id == self.user_id,
                PaperPortfolio.symbol == symbol,
            )
            .first()
        )

        if holding is None:
            return self._reject("Stock not found")

        if quantity > holding.quantity:
            return self._reject("Not enough quantity")

        price = get_live_price(symbol)

        if price <= 0:
            return self._reject("Unable to fetch live price")

        sale_value = price * quantity

        account.balance += sale_value

        paper_transaction = PaperTransaction(
            user_id=self.user_id,
            symbol=symbol,
            transaction_type="SELL",
            quantity=quantity,
            price=price,
            total_amount=sale_value,
        )

        self.db.add(paper_transaction)

        holding.quantity -= quantity
        holding.current_price = price

        if holding.quantity == 0:
            self.db.delete(holding)

        self.db.commit()

        return {
            "status": "PASS",
            "executed": True,
            "message": "Paper SELL executed",
            "symbol": symbol,
            "direction": "LONG",
            "order_type": "MARKET",
            "quantity": quantity,
            "execution_price": round(price, 2),
            "total_amount": round(sale_value, 2),
            "remaining_balance": round(account.balance, 2),
        }

    def _reject(self, reason: str) -> dict:
        return {
            "status": "REJECT",
            "executed": False,
            "reason": reason,
        }