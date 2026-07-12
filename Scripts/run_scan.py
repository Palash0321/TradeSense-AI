import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.scan_engine import scan_market

scan_market("india", limit=100)