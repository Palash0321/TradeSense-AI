import sqlite3

conn = sqlite3.connect("app/database/stocks.db")
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

cursor.execute("""
SELECT
    symbol,
    ai_score,
    signal,
    confidence,
    breakout
FROM stocks
WHERE market='india'
ORDER BY last_scanned DESC
LIMIT 20
""")

rows = cursor.fetchall()

for row in rows:
    print(dict(row))

conn.close()