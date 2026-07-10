import sqlite3
import pandas as pd

# ---------------- CONNECT DATABASE ---------------- #

conn = sqlite3.connect("app/database/stocks.db")

cursor = conn.cursor()

# ---------------- LOAD INDIA ---------------- #

stocks = pd.read_csv("data/master_stocks.csv")

stocks["sector"] = ""
stocks["industry"] = ""

stocks["country"] = stocks["market"].map({
    "india": "India",
    "us": "USA"
})

stocks["currency"] = stocks["market"].map({
    "india": "INR",
    "us": "USD"
})

# ---------------- INSERT ---------------- #

for _, row in stocks.iterrows():

    cursor.execute(
        """
        INSERT OR IGNORE INTO stocks
        (
            symbol,
            name,
            market,
            exchange,
            sector,
            industry,
            country,
            currency
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            row["symbol"],
            row["name"],
            row["market"],
            row["exchange"],
            row["sector"],
            row["industry"],
            row["country"],
            row["currency"]
        )
    )

conn.commit()

conn.close()

print("✅ Master Database Imported Successfully!")