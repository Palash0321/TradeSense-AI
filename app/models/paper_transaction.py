from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    ForeignKey,
)

from sqlalchemy.orm import relationship

from app.core.database import Base


class PaperTransaction(Base):

    __tablename__ = "paper_transactions"

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

    total_amount = Column(
        Float,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    user = relationship("User")