from fastapi import FastAPI, Request
from app.api.stocks import router as stock_router
from fastapi.templating import Jinja2Templates
from app.services.signal_service import generate_signal
from fastapi.staticfiles import StaticFiles
from app.services.news_service import get_stock_news
from fastapi.responses import RedirectResponse, HTMLResponse
import json
from app.services.signal_service import get_live_price
from app.services.screener_service import get_top_stocks
from app.core.stock_universe import StockUniverse
from app.services.ai_screener_service import get_ai_picks
from app.services.database_service import (
    get_connection,
    initialize_database
)
from app.api.admin import router as admin_router
from fastapi.responses import JSONResponse
from app.ai.ranking_engine import rank_stocks
from pydantic import BaseModel
from app.services.portfolio_service import get_stock_metadata
from app.routers import market
from app.services.market_service import get_market_indices

app = FastAPI(
    title="TradeSense AI",
    description="AI-Powered Stock Market Analysis Platform",
    version="1.0.0"
)

initialize_database()

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
    print(result["risk_reward"])
    news = get_stock_news(symbol)

    return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
    "request": request,
    "result": result,
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

@app.get("/api/stock-summary")
def stock_summary(market: str, symbol: str):

    if market == "india":
        symbol = symbol.upper() + ".NS"
    else:
        symbol = symbol.upper()

    result = generate_signal(symbol)

    return {

        "symbol": result["symbol"],

        "company": result["company"],

        "price": result["price"],

        "previous_close": result["previous_close"],

        "change": result["price_change"],

        "change_percent": result["price_change_percent"],

        "is_positive": result["is_positive"],

        "market_status": result["market_status"],

        "signal": result["signal"],

        "confidence": result["confidence"],

        "ai_score": result["ai_score"],

        "target": result["prediction"]["target"],

        "risk": result["risk"]

    }

@app.get("/api/chart-data")
def chart_data(symbol: str):

    import yfinance as yf

    history = yf.Ticker(symbol).history(period="6mo")

    candles = []

    for date, row in history.iterrows():

        candles.append({

            "time": date.strftime("%Y-%m-%d"),

            "open": round(float(row["Open"]),2),

            "high": round(float(row["High"]),2),

            "low": round(float(row["Low"]),2),

            "close": round(float(row["Close"]),2)

        })

    result = generate_signal(symbol)

    return JSONResponse({

        "candles": candles,

        "support": result["support"],

        "resistance": result["resistance"],

        "target": result["prediction"]["target"],

        "stoploss": result["prediction"]["stoploss"],

        "signal": result["signal"],

        "confidence": result["confidence"],

    })

@app.post("/api/watchlist")
@app.delete("/api/watchlist/{symbol}")
def remove_watchlist(symbol: str):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM watchlist
        WHERE symbol = ?
        """,
        (symbol,)
    )

    conn.commit()

    conn.close()

    return {
        "success": True
    }
def add_to_watchlist(stock: dict):

    conn = get_connection()

    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO watchlist(symbol, company)
            VALUES(?,?)
            """,
            (
                stock["symbol"],
                stock["company"]
            )
        )

        conn.commit()

        return {
            "success": True,
            "message": "Stock added to watchlist."
        }

    except Exception:

        return {
            "success": False,
            "message": "Stock already exists."
        }

    finally:

        conn.close()

@app.get("/api/watchlist")
def get_watchlist():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

        SELECT
            id,
            symbol,
            company,
            added_on
        FROM watchlist
        ORDER BY added_on DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]

@app.get("/api/top-picks")
def get_top_picks():

    stocks = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BHARTIARTL.NS",
]

    ranked = rank_stocks(stocks)

    return ranked[:5]

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html"
    )

@app.get("/api/market-overview")
def market_overview():

    import yfinance as yf

    indices = {

        "nifty": "^NSEI",

        "sensex": "^BSESN",

        "sp500": "^GSPC"

    }

    results = {}

    for name, symbol in indices.items():

        try:

            ticker = yf.Ticker(symbol)

            history = ticker.history(period="5d")

            if history.empty:

                results[name] = {
                    "price": None,
                    "change": None,
                    "change_percent": None,
                    "is_positive": None
                }

                continue

            closes = history["Close"].dropna()

            current_price = float(closes.iloc[-1])

            if len(closes) >= 2:

                previous_close = float(closes.iloc[-2])

                change = current_price - previous_close

                change_percent = (
                    change / previous_close
                ) * 100

            else:

                change = 0
                change_percent = 0

            results[name] = {

                "price": round(current_price, 2),

                "change": round(change, 2),

                "change_percent": round(change_percent, 2),

                "is_positive": change >= 0

            }

        except Exception as error:

            print(
                f"Market overview error for {symbol}:",
                error
            )

            results[name] = {
                "price": None,
                "change": None,
                "change_percent": None,
                "is_positive": None
            }

    return results

@app.get("/api/market-movers")
def market_movers(market: str = "india"):

    import yfinance as yf

    if market == "india":

        stocks = {
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "INFY": "INFY.NS",
            "HDFCBANK": "HDFCBANK.NS",
            "ICICIBANK": "ICICIBANK.NS",
            "SBIN": "SBIN.NS",
            "ITC": "ITC.NS",
            "LT": "LT.NS",
            "AXISBANK": "AXISBANK.NS",
            "BHARTIARTL": "BHARTIARTL.NS"
        }

    else:

        stocks = {
            "AAPL": "AAPL",
            "MSFT": "MSFT",
            "NVDA": "NVDA",
            "AMZN": "AMZN",
            "GOOGL": "GOOGL",
            "META": "META",
            "TSLA": "TSLA"
        }


    movers = []

    for name, symbol in stocks.items():

        try:

            history = yf.Ticker(symbol).history(
                period="5d"
            )

            closes = history["Close"].dropna()

            if len(closes) < 2:
                continue

            current_price = float(
                closes.iloc[-1]
            )

            previous_close = float(
                closes.iloc[-2]
            )

            change = (
                current_price - previous_close
            )

            change_percent = (
                change / previous_close
            ) * 100


            movers.append({

                "symbol": name,

                "yahoo_symbol": symbol,

                "price": round(
                    current_price,
                    2
                ),

                "change": round(
                    change,
                    2
                ),

                "change_percent": round(
                    change_percent,
                    2
                )

            })

        except Exception as error:

            print(
                f"Market mover error {symbol}:",
                error
            )


    gainers = sorted(
        movers,
        key=lambda stock:
            stock["change_percent"],
        reverse=True
    )[:5]


    losers = sorted(
        movers,
        key=lambda stock:
            stock["change_percent"]
    )[:5]


    return {

        "market": market,

        "gainers": gainers,

        "losers": losers

    }

@app.get("/api/market-news")
def market_news():

    news_queries = [
        "NIFTY",
        "Indian stock market",
        "Sensex"
    ]

    combined_news = []

    seen_titles = set()

    for query in news_queries:

        try:

            articles = get_stock_news(query)

            for article in articles:

                title = article.get("title")

                if not title:
                    continue

                if title in seen_titles:
                    continue

                seen_titles.add(title)

                combined_news.append({
                    "title": title,
                    "link": article.get("link"),
                    "published": article.get("published"),
                    "source_query": query
                })

        except Exception as error:

            print(
                f"News error for {query}:",
                error
            )

    return combined_news[:8]


# =====================================================
# PORTFOLIO MODEL
# =====================================================

class PortfolioStock(BaseModel):

    symbol: str
    company: str
    quantity: float
    buy_price: float


# =====================================================
# GET PORTFOLIO
# =====================================================

@app.get("/api/portfolio")
def get_portfolio():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

    SELECT
        id,
        symbol,
        company,
        quantity,
        buy_price,
        sector,
        last_price,
        price_updated_at,
        added_on
    FROM portfolio
    ORDER BY added_on DESC

""")

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


# =====================================================
# ADD PORTFOLIO HOLDING
# =====================================================

@app.post("/api/portfolio")
def add_portfolio(stock: PortfolioStock):

    conn = get_connection()

    cursor = conn.cursor()


    metadata = get_stock_metadata(
            stock.symbol
        )


    company = stock.company.strip() \
        if stock.company.strip() \
        else metadata["company"]


    sector = metadata["sector"]


    cursor.execute(
        """
        INSERT INTO portfolio(
            symbol,
            company,
            quantity,
            buy_price,
            sector
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            stock.symbol,
            company,
            stock.quantity,
            stock.buy_price,
            sector
        )
    )


    conn.commit()

    conn.close()


    return {

        "success": True,

        "message":
            "Holding added successfully."

    }

# =====================================================
# UPDATE PORTFOLIO HOLDING
# =====================================================

@app.put("/api/portfolio/{holding_id}")
def update_portfolio_holding(
    holding_id: int,
    stock: PortfolioStock
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE portfolio
        SET
            symbol = ?,
            company = ?,
            quantity = ?,
            buy_price = ?
        WHERE id = ?
        """,
        (
            stock.symbol,
            stock.company,
            stock.quantity,
            stock.buy_price,
            holding_id
        )
    )

    updated_rows = cursor.rowcount

    conn.commit()

    conn.close()

    if updated_rows == 0:

        return {
            "success": False,
            "message": "Holding not found."
        }

    return {
        "success": True,
        "message": "Holding updated successfully."
    }

# =====================================================
# DELETE PORTFOLIO HOLDING
# =====================================================

@app.delete("/api/portfolio/{holding_id}")
def delete_portfolio_holding(holding_id: int):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM portfolio
        WHERE id = ?
        """,
        (holding_id,)
    )

    deleted_rows = cursor.rowcount

    conn.commit()

    conn.close()

    if deleted_rows == 0:

        return {
            "success": False,
            "message": "Holding not found."
        }

    return {
        "success": True,
        "message": "Holding removed successfully."
    }


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="portfolio.html"
    )

@app.get("/api/portfolio-price")
def portfolio_price(market: str, symbol: str):

    import yfinance as yf

    from datetime import datetime, timedelta


    # =================================================
    # NORMALIZE SYMBOL
    # =================================================

    if market == "india":

        clean_symbol = (
            symbol.upper()
            .replace(".NS", "")
        )

        yahoo_symbol = (
            clean_symbol + ".NS"
        )

    else:

        clean_symbol = (
            symbol.upper()
            .replace(".NS", "")
        )

        yahoo_symbol = clean_symbol


    conn = get_connection()

    cursor = conn.cursor()


    try:

        # =============================================
        # CHECK CACHED PRICE
        # =============================================

        cursor.execute(
            """
            SELECT
                last_price,
                price_updated_at
            FROM portfolio
            WHERE symbol = ?
            LIMIT 1
            """,
            (yahoo_symbol,)
        )


        cached = cursor.fetchone()


        if (
            cached
            and cached["last_price"] is not None
            and cached["price_updated_at"]
        ):

            try:

                updated_at = datetime.fromisoformat(
                        cached["price_updated_at"]
                    )


                cache_age = datetime.now() - updated_at


                # Use cached price for 2 minutes

                if cache_age < timedelta(minutes=2):

                    return {

                        "success": True,

                        "symbol": yahoo_symbol,

                        "price":
                            cached["last_price"],

                        "cached": True

                    }

            except Exception as error:

                print(
                    "Price cache timestamp error:",
                    error
                )


        # =============================================
        # FETCH FRESH PRICE
        # =============================================

        ticker = yf.Ticker(yahoo_symbol)


        history = ticker.history(
                period="5d"
            )


        if history.empty:

            return {
                "success": False,
                "price": None
            }


        closes = history["Close"].dropna()


        if closes.empty:

            return {
                "success": False,
                "price": None
            }


        price = round(
                float(closes.iloc[-1]),
                2
            )


        # =============================================
        # SAVE PRICE TO CACHE
        # =============================================

        cursor.execute(
            """
            UPDATE portfolio
            SET
                last_price = ?,
                price_updated_at = ?
            WHERE symbol = ?
            """,
            (
                price,
                datetime.now().isoformat(),
                yahoo_symbol
            )
        )


        conn.commit()


        return {

            "success": True,

            "symbol": yahoo_symbol,

            "price": price,

            "cached": False

        }


    except Exception as error:

        print(
            f"Portfolio price error for {yahoo_symbol}:",
            error
        )


        return {

            "success": False,

            "price": None

        }


    finally:

        conn.close()


app.include_router(market.router)

@app.get("/api/market-sentiment")
def market_sentiment():

    indices = get_market_indices()

    positive = sum(1 for i in indices if i["positive"])
    negative = len(indices) - positive

    if positive >= 5:
        sentiment = "🟢 Strong Bullish"

    elif positive >= 3:
        sentiment = "🟡 Moderately Bullish"

    elif positive == 2:
        sentiment = "🟠 Neutral"

    else:
        sentiment = "🔴 Bearish"

    return {

        "sentiment": sentiment,

        "positive": positive,

        "negative": negative

    }