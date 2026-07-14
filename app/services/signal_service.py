from app.core.indicators.moving_average import calculate_moving_averages
from app.core.indicators.rsi import calculate_rsi
from app.core.indicators.macd import calculate_macd
from app.core.indicators.market_status import get_market_status
from app.core.scoring.score_engine import calculate_score
from app.core.scoring.recommendation import get_recommendation
from app.core.scoring.confidence import calculate_confidence
from app.core.scoring.risk import calculate_risk
from app.services.market_data_service import get_stock_data
from app.core.prediction.predictor import generate_prediction
from app.core.decision.decision_engine import generate_decision
from app.core.explainability.explainability import explain
from app.core.levels.support_resistance import (
    calculate_support_resistance
)
from app.core.risk_reward.risk_reward import (
    calculate_risk_reward
)
from app.core.utils.formatter import (
    format_price,
    format_volume,
    format_market_cap,
)

def generate_signal(symbol: str, period: str = "6mo"):

    data = get_stock_data(symbol, period)

    info = data["info"]
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    previous_close = info.get("previousClose")

    price_change = None
    price_change_percent = None

    history = data["history"]

    if current_price and previous_close:

        price_change = current_price - previous_close

        price_change_percent = (
            price_change / previous_close
        ) * 100

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

    levels = calculate_support_resistance(history)

    

    # Moving Averages
    history = calculate_moving_averages(history)

    # RSI
    history = calculate_rsi(history)

    # MACD
    history = calculate_macd(history)

    # Remove rows where Close is NaN
    history = history.dropna(subset=["Close"])

    latest = history.iloc[-1]

    score, reasons = calculate_score(latest)

    recommendation = get_recommendation(score)

    confidence = calculate_confidence(score)

    risk = calculate_risk(score)

    market_status = get_market_status()

    prediction = generate_prediction(latest, score)

    decision = generate_decision({
    "score": score,
    "signal": recommendation,
    "prediction": prediction,
    "price": format_price(float(latest["Close"])),
    "reasons": reasons
})
    
    risk_reward = calculate_risk_reward(

    float(latest["Close"]),

    decision["target"],

    decision["stoploss"]

)

    ai_explanation = explain(result={
    "score": score,
    "RSI": latest["RSI"],
    "MACD": latest["MACD"],
    "Signal": latest["Signal"]
})

    return {
    "symbol": symbol.upper(),
    "price": format_price(float(latest["Close"])),
    "score": score,
    "ai_score": score,
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

    "prediction": prediction,

    "decision": decision,

    "ai_explanation": ai_explanation,

    "support": levels["support"],

    "resistance": levels["resistance"],

    "risk_reward": risk_reward,

    "market_status": market_status,
}

def get_live_price(symbol: str):

    history = get_stock_history(symbol, "2d")

    if history.empty:
        return None

    latest = float(history["Close"].iloc[-1])

    previous = float(history["Close"].iloc[-2])

    change = latest - previous

    change_percent = (change / previous) * 100

    return {

        "price": round(latest,2),

        "change": round(change,2),

        "change_percent": round(change_percent,2)

    }