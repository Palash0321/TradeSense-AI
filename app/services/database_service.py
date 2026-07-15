import sqlite3

DATABASE = "app/database/stocks.db"


def get_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn

def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS watchlist (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        symbol TEXT NOT NULL UNIQUE,

        company TEXT,

        added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)

    conn.commit()

    conn.close()