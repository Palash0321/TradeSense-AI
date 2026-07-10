import sqlite3

conn = sqlite3.connect("app/database/stocks.db")

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM stocks")

count = cursor.fetchone()[0]

print(f"Total Stocks = {count}")

conn.close()