from app.services.signal_service import generate_signal


def get_ai_picks(symbols):

    picks = []

    for symbol in symbols:

        try:

            result = generate_signal(symbol)

            picks.append({

                "symbol": result["symbol"],

                "company": result["company"],

                "price": result["price"],

                "score": result["score"],

                "ai_score": abs(result["score"]),

                "signal": result["signal"],

                "confidence": result["confidence"],

                "sector": result["sector"],

                "market_cap": result["market_cap"]

            })

        except Exception as e:

            print(symbol, "->", e)

    picks.sort(

        key=lambda x: x["ai_score"],

        reverse=True

    )

    return picks[:10]