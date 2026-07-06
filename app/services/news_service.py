import feedparser


def get_stock_news(symbol):

    query = symbol.replace(".NS", "")

    url = (
        f"https://news.google.com/rss/search?"
        f"q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    news = []

    for entry in feed.entries[:5]:

        news.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published
        })

    return news