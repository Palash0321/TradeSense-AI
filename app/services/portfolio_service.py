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