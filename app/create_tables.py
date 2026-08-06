from app.core.database import Base, engine

# Import all models here
from app.models.user import User

from app.models.watchlist import Watchlist

from app.models.transaction import Transaction

from app.models.paper_account import PaperAccount

from app.models.paper_portfolio import PaperPortfolio

from app.models.paper_transaction import PaperTransaction

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("✅ All tables created successfully.")