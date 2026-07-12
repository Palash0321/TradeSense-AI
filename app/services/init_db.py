import sqlite3
import json

conn = sqlite3.connect("app/database/stocks.db")

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS stocks(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT UNIQUE,

    name TEXT,

    market TEXT,

    exchange TEXT,

    sector TEXT,

    industry TEXT,

    country TEXT,

    currency TEXT

)
               
CREATE INDEX IF NOT EXISTS idx_symbol
ON stocks(symbol);

CREATE INDEX IF NOT EXISTS idx_name
ON stocks(name);

CREATE INDEX IF NOT EXISTS idx_market
ON stocks(market);

""")

with open("data/stocks.json", "r") as file:

    stocks = json.load(file)

for stock in stocks:

    cursor.execute(

        """
        INSERT INTO stocks
(
symbol,
name,
market,
sector,
industry,
exchange
)

        VALUES (?, ?, ?, ?, ?, ?)
        """,
(
    stock["symbol"],
    stock["name"],
    stock["market"],
    stock["sector"],
    stock.get("industry", ""),
    stock.get("exchange", "")
)

    )

conn.commit()

conn.close()

print("Database Created Successfully")

