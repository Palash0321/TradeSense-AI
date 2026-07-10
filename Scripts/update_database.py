import sqlite3

conn = sqlite3.connect("app/database/stocks.db")
cursor = conn.cursor()

queries = [
    "ALTER TABLE stocks ADD COLUMN ai_score INTEGER;",
    "ALTER TABLE stocks ADD COLUMN signal TEXT;",
    "ALTER TABLE stocks ADD COLUMN confidence TEXT;",
    "ALTER TABLE stocks ADD COLUMN last_scanned TEXT;"
]

for query in queries:
    try:
        cursor.execute(query)
        print(f"✅ Executed: {query}")
    except Exception as e:
        print(f"⚠️ {e}")

conn.commit()
conn.close()

print("Database updated successfully!")