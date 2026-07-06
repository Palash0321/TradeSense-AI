import yfinance as yf


def generate_signal(symbol: str, period: str = "6mo"):

    stock = yf.Ticker(symbol)
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

    latest = history.iloc[-1]

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

    return {
    "symbol": symbol.upper(),
    "price": f"{latest['Close']:,.2f}",
    "score": score,
    "confidence": confidence,
    "risk": risk,
    "signal": recommendation,
    "reasons": reasons,
    "RSI": round(float(latest["RSI"]), 2),
    "MACD": round(float(latest["MACD"]), 2),
    "Signal": round(float(latest["Signal"]), 2),
}