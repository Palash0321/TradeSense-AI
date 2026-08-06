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


class Portfolio(Base):
    __tablename__ = "portfolio"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
    )

    symbol = Column(
        String(20),
        nullable=False,
    )

    quantity = Column(
        Float,
        nullable=False,
    )

    buy_price = Column(
        Float,
        nullable=False,
    )

    current_price = Column(
        Float,
        default=0.0,
    )

    last_updated = Column(
        DateTime,
        default=datetime.utcnow,
    )

    user = relationship("User")