from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from app.services.signal_service import generate_signal
from app.services.database_service import get_connection


def scan_single_stock(symbol, market):

    yf_symbol = symbol + ".NS" if market == "india" else symbol

    try:

        result = generate_signal(yf_symbol)

        return {

    "symbol": symbol,

    "ai_score": abs(result["score"]),

    "signal": result["signal"],

    "confidence": result["confidence"],

    "last_scanned": datetime.now().isoformat(),

    "rsi": result["RSI"],

    "macd": result["MACD"],

    "trend": "Bullish" if result["score"] > 0 else "Bearish",

    "breakout": "YES" if result["score"] >= 75 else "NO"

}

    except Exception as e:

        print(symbol, e)

        return None
def scan_market(market="india", limit=100):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT symbol
        FROM stocks
        WHERE market=?
        LIMIT ?
        """,
        (market, limit)
    )

    stocks = cursor.fetchall()

    total = len(stocks)

    print(f"\nScanning {total} {market} stocks...\n")

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        future_map = {

            executor.submit(
                scan_single_stock,
                row["symbol"],
                market
            ): row["symbol"]

            for row in stocks

        }

        completed = 0

        for future in as_completed(future_map):

            completed += 1

            symbol = future_map[future]

            try:

                result = future.result()

                if result:

                    results.append(result)

                    print(f"[{completed}/{total}] ✔ {symbol}")

            except Exception as e:

                print(f"[{completed}/{total}] ✘ {symbol} -> {e}")

    print("\nUpdating database...\n")

    cursor.executemany(

        """
        UPDATE stocks
SET
    ai_score=?,
    signal=?,
    confidence=?,
    last_scanned=?,
    breakout=?,
    trend=?,
    rsi=?,
    macd=?
        WHERE symbol=?
        """,

        [

            (

                stock["ai_score"],

                stock["signal"],

                stock["confidence"],

                stock["last_scanned"],

                stock["breakout"],

                stock["trend"],

                stock["rsi"],

                stock["macd"],

                stock["symbol"]

            )

            for stock in results

        ]

    )

    conn.commit()

    conn.close()

    print("\n✅ Scan Completed Successfully")