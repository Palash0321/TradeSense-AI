from app.core.database import Base, engine

# Import all models here
from app.models.user import User

from app.models.watchlist import Watchlist

print("Creating database tables...")

Base.metadata.create_all(bind=engine)

print("✅ All tables created successfully.")