from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from app.services.market_service import (
    get_market_indices,
    get_market_breadth,
)
from app.services.macro_service import get_macro_data
from app.services.sentiment_service import get_fear_greed
from app.services.news_service import get_stock_news
from app.services.chart_service import get_index_history

router = APIRouter()

templates = Jinja2Templates(directory="app/templates")


@router.get("/market")
def market_page(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="market.html"
    )


@router.get("/api/market-sentiment")
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


@router.get("/api/sectors")
def sectors():

    return [

        {"name":"Information Technology","change":2.34},
        {"name":"Banking","change":1.65},
        {"name":"Automobile","change":0.94},
        {"name":"FMCG","change":-0.48},
        {"name":"Pharma","change":1.12},
        {"name":"Real Estate","change":-1.04},
        {"name":"Energy","change":0.73},
        {"name":"Metal","change":-0.82}

    ]


@router.get("/api/macro")
async def macro():

    return await get_macro_data()


@router.get("/api/fear-greed")
async def fear_greed():

    return await get_fear_greed()


@router.get("/api/market-breadth")
def breadth():

    return get_market_breadth()


@router.get("/api/market-news")
def market_news():

    queries = [

        "NIFTY",

        "Indian stock market",

        "Sensex"

    ]

    news = []

    seen = set()

    for query in queries:

        try:

            articles = get_stock_news(query)

            for article in articles:

                title = article.get("title")

                if not title or title in seen:

                    continue

                seen.add(title)

                news.append({

                    "title": title,

                    "link": article.get("link"),

                    "published": article.get("published"),

                    "source_query": query

                })

        except Exception:

            pass

    return news[:8]

@router.get("/api/index-history/{symbol}")
async def index_history(symbol: str, period: str = "1mo"):

    return await get_index_history(symbol, period)