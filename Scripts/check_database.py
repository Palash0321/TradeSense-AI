import sqlite3

conn = sqlite3.connect("app/database/stocks.db")
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(stocks);")

print("\n===== STOCKS TABLE STRUCTURE =====\n")

for row in cursor.fetchall():
    print(row)

conn.close()