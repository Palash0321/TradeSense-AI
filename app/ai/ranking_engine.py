from app.services.signal_service import generate_signal

def rank_stocks(stock_list, market="india"):

    ranked = []

    for symbol in stock_list:

        try:
            result = generate_signal(symbol)
            ranked.append(result)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    ranked.sort(

        key=lambda x: x.get("ai_score", 0),

        reverse=True

    )

    return ranked