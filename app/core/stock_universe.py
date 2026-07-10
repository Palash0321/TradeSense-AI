import sqlite3
import json

DATABASE = "app/database/stocks.db"


class StockUniverse:

    def __init__(self):

        self.conn = sqlite3.connect(DATABASE)

        self.conn.row_factory = sqlite3.Row

        self.cursor = self.conn.cursor()

    def search(self, query, market):

        # Load symbol aliases
        with open("data/symbol_alias.json", "r") as file:
            alias = json.load(file)

        # Replace old symbol with new symbol if available
        query = alias.get(query.upper(), query.upper())

        self.cursor.execute(

            """
            SELECT
                symbol,
                name,
                market,
                exchange,
                sector
            FROM stocks
            WHERE market = ?
            AND (
                symbol LIKE ?
                OR name LIKE ?
            )
            ORDER BY
                CASE
                    WHEN symbol LIKE ? THEN 1
                    WHEN name LIKE ? THEN 2
                    ELSE 3
                END,
                symbol
            LIMIT 15
            """,

            (
                market,
                f"%{query}%",
                f"%{query}%",
                f"{query}%",
                f"{query}%"
            )

        )

        return [
            dict(row)
            for row in self.cursor.fetchall()
        ]