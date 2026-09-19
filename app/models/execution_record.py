from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class ExecutionRecord(Base):
    __tablename__ = "execution_records"

    id = Column(Integer, primary_key=True, index=True)

    execution_id = Column(String(100), nullable=False, unique=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    symbol = Column(String(20), nullable=False)

    direction = Column(String(10), nullable=False)

    order_type = Column(String(20), nullable=False)

    quantity = Column(Integer, nullable=False)

    entry_price = Column(Float, nullable=True)

    current_state = Column(String(30), nullable=False)

    broker_status = Column(String(50), nullable=True)

    broker_order_id = Column(String(100), nullable=True)

    failure_reason = Column(String(500), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship("User")