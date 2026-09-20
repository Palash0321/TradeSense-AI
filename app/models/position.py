from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


class Position(Base):
    __tablename__ = "positions"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    position_id = Column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    symbol = Column(
        String(20),
        nullable=False,
        index=True,
    )

    direction = Column(
        String(10),
        nullable=False,
    )

    quantity = Column(
        Integer,
        nullable=False,
    )

    average_entry_price = Column(
        Float,
        nullable=False,
    )

    current_price = Column(
        Float,
        nullable=True,
    )

    stop_loss = Column(
        Float,
        nullable=True,
    )

    target1 = Column(
        Float,
        nullable=True,
    )

    target2 = Column(
        Float,
        nullable=True,
    )

    target3 = Column(
        Float,
        nullable=True,
    )

    state = Column(
        String(30),
        nullable=False,
        default="OPEN",
    )

    opened_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    closed_at = Column(
        DateTime,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship("User")