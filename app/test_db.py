from sqlalchemy import text

from app.core.database import engine


try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))

        print("\n✅ Connected Successfully!\n")

        for row in result:
            print(row[0])

except Exception as e:
    print("\n❌ Database Connection Failed!\n")
    print(e)