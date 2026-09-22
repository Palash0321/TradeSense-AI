from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)

from app.core.database import Base


class PositionFillAllocation(Base):
    __tablename__ = "position_fill_allocations"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    position_id = Column(
        String(100),
        ForeignKey("positions.position_id"),
        nullable=False,
        index=True,
    )

    fill_id = Column(
        String(100),
        ForeignKey("execution_fills.fill_id"),
        nullable=False,
        unique=True,
        index=True,
    )

    execution_id = Column(
        String(100),
        ForeignKey("execution_records.execution_id"),
        nullable=False,
        index=True,
    )

    fill_sequence = Column(
        Integer,
        nullable=False,
    )

    fill_quantity = Column(
        Integer,
        nullable=False,
    )

    fill_price = Column(
        Float,
        nullable=False,
    )

    applied_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )