from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class PaperPortfolio(Base):
    __tablename__ = "paper_portfolio"

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

    quantity = Column(
        Float,
        nullable=False,
    )

    average_price = Column(
        Float,
        nullable=False,
    )

    current_price = Column(
        Float,
        default=0,
        nullable=False,
    )

    user = relationship("User")