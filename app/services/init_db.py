import sqlite3
import json

conn = sqlite3.connect("app/database/stocks.db")

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS stocks(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,

    name TEXT NOT NULL,

    market TEXT NOT NULL,

    sector TEXT,

    industry TEXT,

    exchange TEXT

)

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