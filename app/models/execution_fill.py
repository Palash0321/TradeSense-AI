from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from app.core.database import Base


class ExecutionFill(Base):
    __tablename__ = "execution_fills"

    id = Column(Integer, primary_key=True, index=True)

    fill_id = Column(
        String(100),
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

    broker_order_id = Column(String(100), nullable=True)

    fill_sequence = Column(Integer, nullable=False)

    fill_quantity = Column(Integer, nullable=False)

    fill_price = Column(Float, nullable=False)

    filled_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )