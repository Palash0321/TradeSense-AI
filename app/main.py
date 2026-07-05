from fastapi import FastAPI, Request
from app.api.stocks import router as stock_router
from fastapi.templating import Jinja2Templates
from app.services.signal_service import generate_signal
from app.services.chart_service import create_stock_chart

app = FastAPI(
    title="TradeSense AI",
    description="AI-Powered Stock Market Analysis Platform",
    version="1.0.0"
)
templates = Jinja2Templates(directory="app/templates")

app.include_router(stock_router)
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

@app.get("/analyze")
def analyze(request: Request, market: str, symbol: str):
    if market == "india":
        symbol = symbol.upper() + ".NS"
    else:
        symbol = symbol.upper()

    result = generate_signal(symbol)
    chart = create_stock_chart(symbol)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
        "request": request,
        "result": result,
        "chart": chart
}
    )

@app.get("/health")
def health():
    return {
        "status": "Running Successfully"
    }