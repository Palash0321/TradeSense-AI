import yfinance as yf
from datetime import datetime

def format_market_cap(value):

    if not value:
        return "N/A"

    if value >= 1_000_000_000_000:
        return f"₹ {value/1_000_000_000_000:.2f} T"

    elif value >= 1_000_000_000:
        return f"₹ {value/1_000_000_000:.2f} B"

    elif value >= 1_000_000:
        return f"₹ {value/1_000_000:.2f} M"

    return f"₹ {value:,}"


def format_volume(value):

    if not value:
        return "N/A"

    if value >= 1_000_000:
        return f"{value/1_000_000:.2f} M"

    elif value >= 1_000:
        return f"{value/1_000:.2f} K"

    return str(value)


def format_price(value):

    if value is None:
        return "N/A"

    return f"{value:,.2f}"

def generate_signal(symbol: str, period: str = "6mo"):

    stock = yf.Ticker(symbol)
    info = stock.info
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose")

    price_change = None
    price_change_percent = None

    if current_price and previous_close:
        price_change = current_price - previous_close
        price_change_percent = (price_change / previous_close) * 100
        history = stock.history(period=period)

    if history.empty:
        return {
        "symbol": symbol.upper(),
        "price": "N/A",
        "signal": "HOLD",
        "confidence": "0%",
        "risk": "Unknown",
        "reasons": ["No market data available."]
    }

    history = history.dropna(subset=["Close"])

    

    # Moving Averages
    history["MA20"] = history["Close"].rolling(20).mean()
    history["MA50"] = history["Close"].rolling(50).mean()

    # RSI
    delta = history["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    history["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    history["EMA12"] = history["Close"].ewm(span=12, adjust=False).mean()
    history["EMA26"] = history["Close"].ewm(span=26, adjust=False).mean()

    history["MACD"] = history["EMA12"] - history["EMA26"]
    history["Signal"] = history["MACD"].ewm(span=9, adjust=False).mean()

    # Remove rows where Close is NaN
    history = history.dropna(subset=["Close"])

    latest = history.iloc[-1]

    score = 0
    reasons = []

        # Trend
    if latest["MA20"] > latest["MA50"]:
        score += 25
        reasons.append("MA20 is above MA50 (Bullish Trend)")
    else:
        score -= 25
        reasons.append("MA20 is below MA50 (Bearish Trend)")

        # RSI
    if latest["RSI"] < 30:
        score += 25
        reasons.append("RSI indicates Oversold (Potential Buy)")
    elif latest["RSI"] > 70:
        score -= 25
        reasons.append("RSI indicates Overbought (Potential Sell)")
    else:
        reasons.append("RSI is Neutral")

        # MACD
    # MACD
    if latest["MACD"] > latest["Signal"]:
        score += 25
        reasons.append("MACD is above Signal (Bullish Momentum)")
    else:
        score -= 25
        reasons.append("MACD is below Signal (Bearish Momentum)")

        # Price vs MA20
   # Price vs MA20
    if latest["Close"] > latest["MA20"]:
        score += 25
        reasons.append("Price is above MA20 (Positive Trend)")
    else:
        score -= 25
        reasons.append("Price is below MA20 (Negative Trend)")

    if score >= 50:
        recommendation = "BUY"

    elif score <= -50:
        recommendation = "SELL"

    else:
        recommendation = "HOLD"

    confidence = f"{abs(score)}%"

    if abs(score) == 100:
        risk = "Low"

    elif abs(score) == 75:
        risk = "Medium"

    elif abs(score) == 50:
        risk = "High"

    else:
        risk = "Very High"

    # -------- Market Status --------

    now = datetime.now()

    weekday = now.weekday()
    hour = now.hour
    minute = now.minute

    market_status = "CLOSED"

    # NSE Market Hours
    if weekday < 5:

        if (
            (hour == 9 and minute >= 15)
            or (9 < hour < 15)
            or (hour == 15 and minute <= 30)
        ):
            market_status = "OPEN"


    return {
    "symbol": symbol.upper(),
    "price": format_price(float(latest["Close"])),
    "score": score,
    "confidence": confidence,
    "risk": risk,
    "signal": recommendation,
    "reasons": reasons,
    "RSI": round(float(latest["RSI"]), 2),
    "MACD": round(float(latest["MACD"]), 2),
    "Signal": round(float(latest["Signal"]), 2),
    "open": format_price(info.get("open")),
    "high": format_price(info.get("dayHigh")),
    "low": format_price(info.get("dayLow")),
    "previous_close": format_price(info.get("previousClose")),

    "volume": format_volume(info.get("volume")),
    "market_cap": format_market_cap(info.get("marketCap")),
    "pe_ratio": round(info.get("trailingPE", 0), 2),
    "company": info.get("longName"),
    "sector": info.get("sector"),
    "industry": info.get("industry"),

    "price_change": round(price_change, 2) if price_change else 0,

    "price_change_percent": round(price_change_percent, 2) if price_change_percent else 0,

    "is_positive": price_change >= 0 if price_change is not None else True,

    "market_status": market_status,
}