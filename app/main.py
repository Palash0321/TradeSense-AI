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
from app.services.macro_service import get_macro_data
from app.services.sentiment_service import get_fear_greed
from fastapi import Query
from datetime import datetime, timedelta, time
from app.services.market_data.provider import provider
from app.auth.routes import router as auth_router
from app.api.watchlist import router as watchlist_router
from app.api.portfolio import router as portfolio_router
from app.api.portfolio_summary import (
    router as portfolio_summary_router
)
from app.api.dashboard import (
    router as dashboard_router
)
from app.api.transactions import (
    router as transactions_router
)
from app.api.paper_trading import (
    router as paper_router
)
from app.api.paper_orders import (
    router as paper_orders_router
)   
from app.api.paper_portfolio import (
    router as paper_portfolio_router
)
from app.api.paper_history import router as paper_history_router
from app.api.paper_dashboard import (
    router as paper_dashboard_router
)
from app.api.paper_analytics import (
    router as paper_analytics_router
)
from app.api.backtest import (
    router as backtest_router
)
app = FastAPI(
    title="TradeSense AI",
    description="AI-Powered Stock Market Analysis Platform",
    version="1.0.0"
)

app.include_router(auth_router)
app.include_router(watchlist_router)
app.include_router(portfolio_router)
app.include_router(
    portfolio_summary_router
)
app.include_router(
    dashboard_router
)
app.include_router(
    transactions_router
)
app.include_router(
    paper_router
)
app.include_router(
    paper_orders_router
)
app.include_router(
    paper_portfolio_router
)
app.include_router(paper_history_router)
app.include_router(
    paper_dashboard_router
)
app.include_router(
    paper_analytics_router
)
app.include_router(
    backtest_router
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="dashboard.html",
    context={
        "request": request
    }
)

initialize_database()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(stock_router)
from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
async def home():

    return RedirectResponse(
        url="/dashboard",
        status_code=302
    )

@app.get("/analyze")
def analyze(request: Request, market: str, symbol: str):
    symbol = symbol.upper()

    if market == "india" and not symbol.endswith(".NS"):
        symbol = symbol + ".NS"

    result = generate_signal(symbol)
    print("Signal generated:", result.get("signal"))
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

@app.get("/login")
def login_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="login.html",
    )

@app.get("/register")
def register_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="register.html",
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

    import math
    import yfinance as yf

    history = yf.Ticker(symbol).history(period="6mo")

    candles = []

    for date, row in history.iterrows():

            open_price = float(row["Open"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            close_price = float(row["Close"])

            import math

            if (
                math.isnan(open_price)
                or math.isnan(high_price)
                or math.isnan(low_price)
                or math.isnan(close_price)
            ):
                continue

            candles.append({

                "time": date.strftime("%Y-%m-%d"),

                "open": round(open_price, 2),

                "high": round(high_price, 2),

                "low": round(low_price, 2),

                "close": round(close_price, 2)

            })

    result = generate_signal(symbol)

    print("\n===================")
    print("SUPPORT =", result["support"])
    print("RESISTANCE =", result["resistance"])
    print("TARGET =", result["prediction"]["target"])
    print("STOPLOSS =", result["prediction"]["stoploss"])
    print("===================\n")

    response = {
        "candles": candles,
        "support": result["support"],
        "resistance": result["resistance"],
        "target": result["prediction"]["target"],
        "stoploss": result["prediction"]["stoploss"],
        "signal": result["signal"],
        "confidence": result["confidence"],
    }

    return response



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

def get_market_status():

    now = datetime.now()

    current = now.time()

    if now.weekday() >= 5:

        return {

            "status": "CLOSED",

            "color": "red"

        }

    if time(9, 0) <= current < time(9, 15):

        return {

            "status": "PRE OPEN",

            "color": "yellow"

        }

    if time(9, 15) <= current <= time(15, 30):

        return {

            "status": "LIVE",

            "color": "green"

        }

    return {

        "status": "CLOSED",

        "color": "red"

    }

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

    results["market_status"] = get_market_status()

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

@app.get("/api/expiries")
async def get_expiries(index: str = Query(default="NIFTY")):

    from datetime import datetime, timedelta

    today = datetime.today()

    expiries = []

    current = today

    while len(expiries) < 8:

        if current.weekday() == 3:      # Thursday

            expiries.append(
                current.strftime("%d %b %Y")
            )

        current += timedelta(days=1)

    monthly = expiries[-1]

    if index.upper() in ["NIFTY", "^NSEI"]:

        contracts = [
            "Spot",
            *expiries,
            f"Monthly ({monthly})"
        ]

    elif index.upper() in ["BANKNIFTY", "^NSEBANK"]:

        contracts = [
            "Spot",
            *expiries,
            f"Monthly ({monthly})"
        ]

    elif index.upper() in ["FINNIFTY"]:

        contracts = [
            "Spot",
            *expiries
        ]

    else:

        contracts = ["Spot"]

    return {
        "index": index,
        "contracts": contracts
    }

@app.get("/api/option-chain")
async def get_option_chain(

    index: str,

    expiry: str

):

    symbol_map = {

        "^NSEI": "^NSEI",

        "^NSEBANK": "^NSEBANK",

        "^BSESN": "^BSESN",

        "FINNIFTY": "NIFTY_FIN_SERVICE.NS"

    }

    yahoo_symbol = symbol_map.get(index, "^NSEI")

    history = provider.get_history(

        yahoo_symbol,

        period="5d",

        interval="1d"

    )

    if history.empty:

        return {

            "index": index,

            "expiry": expiry,

            "spot": 0,

            "data": []

        }

    spot = round(float(history["Close"].iloc[-1]), 2)

    base = int(round(spot / 100) * 100)

    strikes = []

    for strike in range(base - 1000, base + 1100, 100):

        distance = abs(strike - base)

        intrinsic = max(0, base - strike)

        call_ltp = max(10, 400 - distance * 0.35)

        put_ltp = max(10, 400 - distance * 0.35)

        call_oi = max(10000, 500000 - distance * 300)

        put_oi = max(10000, 500000 - distance * 300)

        strikes.append({

            "strike": strike,

            "call": {

                "ltp": round(call_ltp, 2),

                "oi": int(call_oi),

                "iv": round(12 + distance / 250, 2)

            },

            "put": {

                "ltp": round(put_ltp, 2),

                "oi": int(put_oi),

                "iv": round(12 + distance / 250, 2)

            }

        })

    max_call = max(
    strikes,
    key=lambda row: row["call"]["oi"]
)

    max_put = max(
    strikes,
    key=lambda row: row["put"]["oi"]
)

    total_call = sum(
    row["call"]["oi"]
    for row in strikes
)

    total_put = sum(
    row["put"]["oi"]
    for row in strikes
)

    pcr = round(total_put / total_call, 2)

    if pcr > 1.1:
        bias = "Bullish"
    elif pcr < 0.9:
        bias = "Bearish"
    else:
        bias = "Neutral"

    return {

    "index": index,

    "expiry": expiry,

    "spot": spot,

    "support": max_put["strike"],

    "resistance": max_call["strike"],

    "pcr": pcr,

    "bias": bias,

    "data": strikes

}
@app.get("/paper-dashboard")
def paper_dashboard(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="paper_dashboard.html",
    )

@app.get("/backtest")
def backtest_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="backtest.html",
    )