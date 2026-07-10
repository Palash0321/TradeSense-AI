import pandas as pd

# ---------------- INDIA ---------------- #

india_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

print("Downloading Indian stocks...")

india = pd.read_csv(india_url)

india = india.rename(columns={
    "SYMBOL": "symbol",
    "NAME OF COMPANY": "name"
})

india["market"] = "india"
india["exchange"] = "NSE"

india = india[["symbol", "name", "market", "exchange"]]

india.to_csv("data/india_stocks.csv", index=False)

print(f"Downloaded {len(india)} Indian stocks")


# ---------------- USA ---------------- #

nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

print("Downloading US stocks...")

usa = pd.read_csv(
    nasdaq_url,
    sep="|"
)

usa = usa.rename(columns={
    "Symbol": "symbol",
    "Security Name": "name"
})

usa["market"] = "us"
usa["exchange"] = "NASDAQ"

usa = usa[["symbol", "name", "market", "exchange"]]

usa = usa.iloc[:-1]

usa.to_csv("data/us_stocks.csv", index=False)

print(f"Downloaded {len(usa)} US stocks")