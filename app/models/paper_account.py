from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=False,
        unique=True,
    )

    balance = Column(
        Float,
        default=1000000.0,
        nullable=False,
    )

    user = relationship("User")