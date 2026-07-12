from fastapi import FastAPI, Request
from app.api.stocks import router as stock_router
from fastapi.templating import Jinja2Templates
from app.services.signal_service import generate_signal
from app.services.chart_service import create_stock_chart
from fastapi.staticfiles import StaticFiles
from app.services.news_service import get_stock_news
from fastapi.responses import RedirectResponse
import json
from app.services.signal_service import get_live_price
from app.services.screener_service import get_top_stocks
from app.core.stock_universe import StockUniverse
from app.services.ai_screener_service import get_ai_picks
from app.services.database_service import get_connection
from app.api.admin import router as admin_router

app = FastAPI(
    title="TradeSense AI",
    description="AI-Powered Stock Market Analysis Platform",
    version="1.0.0"
)
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
    news = get_stock_news(symbol)
    chart = create_stock_chart(symbol)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
    "request": request,
    "result": result,
    "chart": chart,
    "news": news
}
    )

@app.get("/health")
def health():
    return {
        "status": "Running Successfully"
    }


@app.get("/stock/{symbol}")
def stock_redirect(symbol: str):

    symbol = symbol.upper()

    return RedirectResponse(
        url=f"/analyze?market=india&symbol={symbol}"
    )

@app.get("/api/stocks")
def stocks():

    with open("data/stocks.json", "r") as file:
        data = json.load(file)

    return data

@app.get("/api/price/{symbol}")
def price(symbol: str):

    if symbol.endswith(".NS"):

        final_symbol = symbol

    else:

        final_symbol = symbol

    return get_live_price(final_symbol)

@app.get("/api/screener")
def screener(market: str = "india"):

    stocks = get_top_stocks(market)

    return stocks

@app.get("/watchlist")
def watchlist(request: Request):

    return templates.TemplateResponse(

        request=request,

        name="watchlist.html"

    )

@app.get("/api/search")
def search_stocks(

    query: str = "",

    market: str = "india"

):

    universe = StockUniverse()

    return universe.search(

        query,

        market

    )

@app.get("/api/ai-picks")
def ai_picks(market: str = "india"):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
    """
    SELECT
        symbol,
        name,
        ai_score,
        signal,
        confidence
    FROM stocks
    WHERE
        market = ?
        AND ai_score IS NOT NULL
        AND ai_score >= 50
    ORDER BY
        ai_score DESC,
        signal DESC
    LIMIT 20
    """,
    (market,)
)
    rows = cursor.fetchall()

    conn.close()

    results = []

    for row in rows:

        symbol = row["symbol"]

        if market == "india":
            symbol += ".NS"

        results.append({

            "symbol": symbol,

            "company": row["name"],

            "ai_score": row["ai_score"],

            "signal": row["signal"],

            "confidence": row["confidence"]

        })

    return results

app.include_router(admin_router)

@app.get("/api/breakouts")
def breakout_scanner(market: str = "india"):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            symbol,
            name,
            ai_score,
            signal,
            breakout,
            trend,
            rsi
        FROM stocks
        WHERE
            market=?
            AND breakout='YES'
        ORDER BY ai_score DESC
        LIMIT 20
        """,
        (market,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]