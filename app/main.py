from fastapi import FastAPI
from app.api.stocks import router as stock_router

app = FastAPI(
    title="TradeSense AI",
    description="AI-Powered Stock Market Analysis Platform",
    version="1.0.0"
)

app.include_router(stock_router)
@app.get("/")
def home():
    return {
        "message": "Welcome to TradeSense AI 🚀"
    }

@app.get("/health")
def health():
    return {
        "status": "Running Successfully"
    }