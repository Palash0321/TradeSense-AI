import pandas as pd
from nselib import capital_market

print("Downloading complete NSE equity list...")

df = capital_market.equity_list()

df = df.rename(
    columns={
        "SYMBOL": "symbol",
        "NAME OF COMPANY": "name"
    }
)

df["market"] = "india"
df["exchange"] = "NSE"

df = df[
    [
        "symbol",
        "name",
        "market",
        "exchange"
    ]
]

df.to_csv(
    "data/india_stocks.csv",
    index=False
)

print(f"Downloaded {len(df)} Indian Stocks")