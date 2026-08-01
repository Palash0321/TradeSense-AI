from app.services.market_data.provider import provider

quote = provider.get_quote("AAPL")

print(quote)