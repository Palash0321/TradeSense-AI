from datetime import datetime
import uuid

from sqlalchemy.orm import Session

from app.models.position import Position
from app.services.position_state_service import (
    PositionStateService,
)


class PositionService:
    """
    Persistent position lifecycle service.

    Responsibilities:
    - create positions
    - retrieve positions
    - update current market price
    - partially close positions
    - close positions
    - persist position lifecycle state

    This service does NOT:
    - generate signals
    - approve risk
    - calculate position size
    - generate order intents
    - communicate with brokers
    - execute trades
    - calculate strategy decisions
    """

    def __init__(
        self,
        db: Session,
        user_id: int,
    ):
        self.db = db
        self.user_id = user_id

    def create(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        average_entry_price: float,
        current_price: float | None = None,
        stop_loss: float | None = None,
        target1: float | None = None,
        target2: float | None = None,
        target3: float | None = None,
        position_id: str | None = None,
    ) -> dict:
        validation = self._validate_creation(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            average_entry_price=average_entry_price,
            stop_loss=stop_loss,
            target1=target1,
            target2=target2,
            target3=target3,
        )

        if validation["status"] != "PASS":
            return validation

        symbol = symbol.strip().upper()

        if position_id is None:
            position_id = str(uuid.uuid4())
        elif (
            not isinstance(position_id, str)
            or not position_id.strip()
        ):
            return self._reject(
                "position_id must be a non-empty string."
            )

        existing = (
            self.db.query(Position)
            .filter(
                Position.position_id == position_id,
            )
            .first()
        )

        if existing is not None:
            return self._reject(
                "A position with this position_id already exists."
            )

        position = Position(
            position_id=position_id,
            user_id=self.user_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            average_entry_price=float(
                average_entry_price
            ),
            current_price=(
                float(current_price)
                if current_price is not None
                else float(average_entry_price)
            ),
            stop_loss=(
                float(stop_loss)
                if stop_loss is not None
                else None
            ),
            target1=(
                float(target1)
                if target1 is not None
                else None
            ),
            target2=(
                float(target2)
                if target2 is not None
                else None
            ),
            target3=(
                float(target3)
                if target3 is not None
                else None
            ),
            state="OPEN",
        )

        self.db.add(position)

        try:
            self.db.commit()
            self.db.refresh(position)
        except Exception:
            self.db.rollback()
            raise

        return {
            "status": "PASS",
            "position_id": position.position_id,
            "user_id": position.user_id,
            "symbol": position.symbol,
            "direction": position.direction,
            "quantity": position.quantity,
            "average_entry_price": (
                position.average_entry_price
            ),
            "current_price": position.current_price,
            "state": position.state,
            "opened_at": position.opened_at,
            "closed_at": position.closed_at,
            "source": "PositionService",
        }

    def get(self, position_id: str) -> dict:
        position = self._get(position_id)

        if position is None:
            return self._reject(
                "Position not found."
            )

        return {
            "status": "PASS",
            "position_id": position.position_id,
            "user_id": position.user_id,
            "symbol": position.symbol,
            "direction": position.direction,
            "quantity": position.quantity,
            "average_entry_price": (
                position.average_entry_price
            ),
            "current_price": position.current_price,
            "stop_loss": position.stop_loss,
            "target1": position.target1,
            "target2": position.target2,
            "target3": position.target3,
            "state": position.state,
            "opened_at": position.opened_at,
            "closed_at": position.closed_at,
            "source": "PositionService",
        }

    def update_current_price(
        self,
        position_id: str,
        current_price: float,
    ) -> dict:
        position = self._get(position_id)

        if position is None:
            return self._reject(
                "Position not found."
            )

        if position.state == "CLOSED":
            return self._reject(
                "Closed positions cannot be marked with a new price."
            )

        if not self._is_positive_number(current_price):
            return self._reject(
                "current_price must be a positive numeric value."
            )

        position.current_price = float(current_price)

        try:
            self.db.commit()
            self.db.refresh(position)
        except Exception:
            self.db.rollback()
            raise

        return self.get(position_id)

    def partially_close(
        self,
        position_id: str,
        quantity: int,
        current_price: float | None = None,
        reason: str | None = None,
    ) -> dict:
        position = self._get(position_id)

        if position is None:
            return self._reject(
                "Position not found."
            )

        if position.state == "CLOSED":
            return self._reject(
                "Closed positions cannot be partially closed."
            )

        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            return self._reject(
                "Quantity must be a positive integer."
            )

        if quantity >= position.quantity:
            return self._reject(
                "Partial close quantity must be less than "
                "the current open quantity."
            )

        if (
            current_price is not None
            and not self._is_positive_number(
                current_price
            )
        ):
            return self._reject(
                "current_price must be a positive numeric value."
            )

        state_service = PositionStateService(
            position_id=position.position_id,
            initial_state=position.state,
            record_initial_event=False,
        )

        transition_result = state_service.transition(
            new_state="PARTIALLY_CLOSED",
            reason=(
                reason
                or "Position partially closed."
            ),
            metadata={
                "quantity_closed": quantity,
            },
        )

        if transition_result["status"] != "PASS":
            return transition_result

        position.quantity -= quantity

        if current_price is not None:
            position.current_price = float(
                current_price
            )

        position.state = (
            transition_result["state"]
        )

        try:
            self.db.commit()
            self.db.refresh(position)
        except Exception:
            self.db.rollback()
            raise

        return {
            **self.get(position_id),
            "quantity_closed": quantity,
            "source": "PositionService",
        }

    def close(
        self,
        position_id: str,
        current_price: float | None = None,
        reason: str | None = None,
    ) -> dict:
        position = self._get(position_id)

        if position is None:
            return self._reject(
                "Position not found."
            )

        if position.state == "CLOSED":
            return self._reject(
                "Position is already closed."
            )

        if (
            current_price is not None
            and not self._is_positive_number(
                current_price
            )
        ):
            return self._reject(
                "current_price must be a positive numeric value."
            )

        state_service = PositionStateService(
            position_id=position.position_id,
            initial_state=position.state,
            record_initial_event=False,
        )

        transition_result = state_service.transition(
            new_state="CLOSED",
            reason=(
                reason
                or "Position closed."
            ),
            metadata={
                "quantity_closed": position.quantity,
            },
        )

        if transition_result["status"] != "PASS":
            return transition_result

        position.quantity = 0
        position.state = "CLOSED"
        position.closed_at = datetime.utcnow()

        if current_price is not None:
            position.current_price = float(
                current_price
            )

        try:
            self.db.commit()
            self.db.refresh(position)
        except Exception:
            self.db.rollback()
            raise

        return {
            **self.get(position_id),
            "source": "PositionService",
        }

    def _get(self, position_id: str):
        if (
            not isinstance(position_id, str)
            or not position_id.strip()
        ):
            return None

        return (
            self.db.query(Position)
            .filter(
                Position.position_id == position_id
            )
            .first()
        )

    @staticmethod
    def _validate_creation(
        symbol,
        direction,
        quantity,
        average_entry_price,
        stop_loss,
        target1,
        target2,
        target3,
    ):
        if (
            not isinstance(symbol, str)
            or not symbol.strip()
        ):
            return {
                "status": "REJECT",
                "reason": (
                    "symbol must be a non-empty string."
                ),
                "source": "PositionService",
            }

        if direction not in ("LONG", "SHORT"):
            return {
                "status": "REJECT",
                "reason": (
                    "direction must be LONG or SHORT."
                ),
                "source": "PositionService",
            }

        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, int)
            or quantity <= 0
        ):
            return {
                "status": "REJECT",
                "reason": (
                    "quantity must be a positive integer."
                ),
                "source": "PositionService",
            }

        if not PositionService._is_positive_number(
            average_entry_price
        ):
            return {
                "status": "REJECT",
                "reason": (
                    "average_entry_price must be "
                    "a positive numeric value."
                ),
                "source": "PositionService",
            }

        optional_prices = {
            "stop_loss": stop_loss,
            "target1": target1,
            "target2": target2,
            "target3": target3,
        }

        for field_name, value in optional_prices.items():
            if (
                value is not None
                and not PositionService._is_positive_number(
                    value
                )
            ):
                return {
                    "status": "REJECT",
                    "reason": (
                        f"{field_name} must be a positive "
                        "numeric value when supplied."
                    ),
                    "source": "PositionService",
                }

        return {
            "status": "PASS",
            "source": "PositionService",
        }

    @staticmethod
    def _is_positive_number(value):
        if isinstance(value, bool):
            return False

        if not isinstance(value, (int, float)):
            return False

        return value > 0

    @staticmethod
    def _reject(reason: str):
        return {
            "status": "REJECT",
            "reason": reason,
            "source": "PositionService",
        }