from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    ForeignKey,
    DateTime,
)

from sqlalchemy.orm import relationship

from app.core.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    symbol = Column(
        String(20),
        nullable=False,
    )

    transaction_type = Column(
        String(10),
        nullable=False,
    )

    quantity = Column(
        Float,
        nullable=False,
    )

    price = Column(
        Float,
        nullable=False,
    )

    realized_profit = Column(
    Float,
    nullable=False,
    default=0.0,
)

    transaction_date = Column(
        DateTime,
        default=datetime.utcnow,
    )

    user = relationship("User")