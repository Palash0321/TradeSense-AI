import yfinance as yf


# =====================================================
# GET STOCK METADATA
# =====================================================

def get_stock_metadata(symbol: str):

    try:

        ticker = yf.Ticker(symbol)

        info = ticker.info or {}

        return {

            "company":
                info.get("longName")
                or info.get("shortName")
                or symbol,

            "sector":
                info.get("sector")
                or "Unknown"

        }

    except Exception as error:

        print(
            f"Metadata error for {symbol}:",
            error
        )

        return {

            "company": symbol,

            "sector": "Unknown"

        }

    # =====================================================
# PORTFOLIO ANALYTICS
# =====================================================

def calculate_portfolio_metrics(holding):

    investment = holding.quantity * holding.buy_price

    current_value = holding.quantity * holding.current_price

    profit_loss = current_value - investment

    if investment == 0:

        profit_percent = 0

    else:

        profit_percent = (
            profit_loss / investment
        ) * 100

    return {

        "id": holding.id,

        "symbol": holding.symbol,

        "quantity": holding.quantity,

        "buy_price": round(
            holding.buy_price,
            2,
        ),

        "current_price": round(
            holding.current_price,
            2,
        ),

        "investment": round(
            investment,
            2,
        ),

        "current_value": round(
            current_value,
            2,
        ),

        "profit_loss": round(
            profit_loss,
            2,
        ),

        "profit_percent": round(
            profit_percent,
            2,
        ),
    }