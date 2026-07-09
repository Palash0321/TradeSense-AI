import csv
import sqlite3


def import_stocks():

    conn = sqlite3.connect("app/database/stocks.db")

    cursor = conn.cursor()

    with open("data/stock.csv", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            cursor.execute("""

            INSERT INTO stocks

            (symbol,name,market,sector,industry,exchange)

            VALUES(?,?,?,?,?,?)

            """,

            (

                row["symbol"],

                row["name"],

                row["market"],

                row["sector"],

                row["industry"],

                row["exchange"]

            ))

    conn.commit()

    conn.close()

    print("Stocks Imported Successfully")