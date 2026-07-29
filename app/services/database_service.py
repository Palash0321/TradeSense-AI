import sqlite3


DATABASE = "app/database/stocks.db"


# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_connection():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# =====================================================
# INITIALIZE DATABASE
# =====================================================

def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()


    # =================================================
    # WATCHLIST TABLE
    # =================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL UNIQUE,

            company TEXT,

            added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)


    # =================================================
    # PORTFOLIO TABLE
    # =================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,

            company TEXT,

            quantity REAL NOT NULL,

            buy_price REAL NOT NULL,

            added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)


    # =================================================
    # PORTFOLIO METADATA COLUMNS
    # =================================================

    cursor.execute(
        "PRAGMA table_info(portfolio)"
    )

    portfolio_columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    # Add sector column if missing

    if "sector" not in portfolio_columns:

        cursor.execute("""
            ALTER TABLE portfolio
            ADD COLUMN sector TEXT
        """)


    # Add cached price column if missing

    if "last_price" not in portfolio_columns:

        cursor.execute("""
            ALTER TABLE portfolio
            ADD COLUMN last_price REAL
        """)


    # Add price timestamp column if missing

    if "price_updated_at" not in portfolio_columns:

        cursor.execute("""
            ALTER TABLE portfolio
            ADD COLUMN price_updated_at TIMESTAMP
        """)


    # =================================================
    # SAVE DATABASE CHANGES
    # =================================================

    conn.commit()

    conn.close()