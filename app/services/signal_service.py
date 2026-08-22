from app.core.indicators.moving_average import calculate_moving_averages
from app.core.indicators.rsi import calculate_rsi
from app.core.indicators.macd import calculate_macd
from app.core.indicators.market_status import get_market_status
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
from app.core.indicators.atr import calculate_atr
from app.services.ai_engine import AIEngine
from app.services.trade_horizon_service import (
    TradeHorizonService
)
from app.services.stock_service import get_stock_history
from app.services.market_health_service import (
    calculate_market_health,
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
        "score": 0,
        "ai_score": 0,
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

    history = calculate_atr(history)

    # Remove rows where Close is NaN
    history = history.dropna(subset=["Close"])

    latest = history.iloc[-1]

    # =====================================
    # TREND STRENGTH
    # =====================================

    ma20 = float(latest["MA20"])

    ma50 = float(latest["MA50"])

    close = float(latest["Close"])

    if close > ma20 > ma50:

        trend_strength = "Strong Bullish"

    elif close > ma20 and ma20 < ma50:

        trend_strength = "Bullish Recovery"

    elif close < ma20 < ma50:

        trend_strength = "Strong Bearish"

    elif close < ma20 and ma20 > ma50:

        trend_strength = "Bearish Reversal"

    else:

        trend_strength = "Sideways"


    # =====================================
    # VOLATILITY LEVEL
    # =====================================

    atr = float(latest["ATR"])

    volatility_percent = (atr / close) * 100

    if volatility_percent > 4:

        volatility = "High"

    elif volatility_percent > 2:

        volatility = "Medium"

    else:

        volatility = "Low"


    # ==========================
    # RSI STATUS
    # ==========================

    rsi = round(float(latest["RSI"]),2)

    if rsi >= 70:

        rsi_status = "Overbought"

    elif rsi <= 30:

        rsi_status = "Oversold"

    else:

        rsi_status = "Neutral"


    # ==========================
    # MACD STATUS
    # ==========================

    macd = round(float(latest["MACD"]), 2)

    signal_line = round(float(latest["Signal"]), 2)

    if macd > signal_line:

        macd_status = "Bullish"

    elif macd < signal_line:

        macd_status = "Bearish"

    else:

        macd_status = "Neutral"

    # =====================================
    # MOMENTUM SCORE
    # =====================================

    momentum_score = 0

    # RSI Contribution
    if 45 <= rsi <= 65:

        momentum_score += 25

    elif 30 <= rsi < 45 or 65 < rsi <= 70:

        momentum_score += 15

    else:

        momentum_score += 5

    # MACD Contribution
    if macd_status == "Bullish":

        momentum_score += 35

    elif macd_status == "Neutral":

        momentum_score += 20

    else:

        momentum_score += 5

    # Trend Contribution
    if trend_strength == "Strong Bullish":

        momentum_score += 40

    elif trend_strength == "Bullish Recovery":

        momentum_score += 30

    elif trend_strength == "Sideways":

        momentum_score += 20

    else:

        momentum_score += 10

    momentum_score = min(momentum_score, 100)

    # =====================================
    # Basic Analysis
    # =====================================

    market_status = get_market_status()

    prediction = generate_prediction(
        latest,
        0
    )

    decision = generate_decision({

        "score": 0,

        "signal": "WAIT",

        "prediction": prediction,

        "price": format_price(
            float(latest["Close"])
        ),

        "reasons": []

    })

    risk_reward = calculate_risk_reward(

        float(latest["Close"]),

        decision["target"],

        decision["stoploss"]

    )

    # =====================================
    # AI ENGINE
    # =====================================

    ai = AIEngine(

        symbol,

        history,

        latest,

        levels,

        risk_reward

    ).analyze()

    trade_quality = ai["trade_quality"]

    trade_plan = ai["trade_plan"]

    opportunity = ai["opportunity"]

    probability = ai["probability"]

    base_confidence = ai["ai_confidence"]

    confidence_bonus = 0

    if trend_strength == "Strong Bullish":
        confidence_bonus += 5

    if macd_status == "Bullish":
        confidence_bonus += 3

    if rsi_status == "Neutral":
        confidence_bonus += 2

    ai_confidence = min(base_confidence + confidence_bonus, 100)

    multi_timeframe = ai["multi_timeframe"]

    entry_engine = ai["entry_engine"]

    candlestick_patterns = ai["candlestick_patterns"]

    volume_analysis = ai["volume_analysis"]

    trade_validation = ai["trade_validation"]

    final_decision = ai["final_decision"]

    # =====================================
    # Trade Horizon
    # =====================================

    trade_horizon = TradeHorizonService(

        preferred_setup=opportunity.get(
            "preferred_setup",
            "NO_SETUP"
        ),

        current_price=float(
            latest["Close"]
        ),

        atr=float(
            latest["ATR"]
        ),

        entry_engine=entry_engine

    ).calculate()

    # =====================================
    # Trade Signal
    # =====================================

    trade_signal = ai.get(
        "trade_signal",
        {}
    )

    trade_signal["trade_horizon"] = (
        trade_horizon
    )

    score = trade_quality["score"]

   
# =====================================
# MARKET HEALTH SCORE
# =====================================

    market_health_data = calculate_market_health()

    market_health = market_health_data["score"]

    recommendation = trade_quality["recommendation"]

    confidence = f"{ai_confidence}%"

    risk = trade_quality["risk"]

    reasons = trade_quality["reasons"]

    # =====================================
    # Final Decision
    # =====================================

    decision = generate_decision({

        "score": score,

        "signal": recommendation,

        "prediction": prediction,

        "price": format_price(
            float(latest["Close"])
        ),

        "reasons": reasons

    })

    ai_explanation = explain(result={
        "score": score,
        "RSI": latest["RSI"],
        "MACD": latest["MACD"],
        "Signal": latest["Signal"]
})

    # =====================================
    # Standardized Trade Signal
    # =====================================

    preferred_setup = final_decision.get(
        "preferred_setup",
        "NO_SETUP"
    )

    setup_details = final_decision.get(
        "setup_details"
    )

    trade_signal = {
        "decision": final_decision.get(
            "decision",
            "WAIT"
        ),

        "preferred_setup": preferred_setup,

        "trade_horizon": trade_horizon,

        "current_price": round(
            float(latest["Close"]),
            2
        ),

        "entry": None,

        "entry_low": None,

        "entry_high": None,

        "stop_loss": None,

        "target1": None,

        "target2": None,

        "target3": None,

        "risk_reward": {},

        "breakout_trigger": final_decision.get(
            "breakout_trigger"
        ),

        "breakout_level": final_decision.get(
            "breakout_level"
        ),

        "message": final_decision.get(
            "message",
            ""
        )
    }

    if setup_details:

        trade_signal["stop_loss"] = setup_details.get(
            "stop_loss"
        )

        trade_signal["target1"] = setup_details.get(
            "target1"
        )

        trade_signal["target2"] = setup_details.get(
            "target2"
        )

        trade_signal["target3"] = setup_details.get(
            "target3"
        )

        trade_signal["risk_reward"] = setup_details.get(
            "risk_reward",
            {}
        )

        if setup_details.get("type") == "BREAKOUT":

            trade_signal["entry"] = setup_details.get(
                "entry"
            )

        elif setup_details.get("type") == "PULLBACK":

            trade_signal["entry_low"] = setup_details.get(
                "entry_low"
            )

            trade_signal["entry_high"] = setup_details.get(
                "entry_high"
            )

    return {

        "symbol": symbol.upper(),

        "price": format_price(float(latest["Close"])),

        "score": score,

        "ai_score": score,

        "market_health": market_health,

        "trend_strength": trend_strength,

        "volatility": volatility,

        "momentum_score": momentum_score,

        "confidence": confidence,

        "risk": risk,

        "signal": recommendation,

        "reasons": reasons,

        "RSI": rsi,

        "rsi_status": rsi_status,

        "MACD": macd,

        "macd_status": macd_status,

        "Signal": signal_line,

        "open": format_price(info.get("open")),

        "high": format_price(info.get("dayHigh")),

        "low": format_price(info.get("dayLow")),

        "previous_close": format_price(
            info.get("previousClose")
        ),

        "volume": format_volume(
            info.get("volume")
        ),

        "market_cap": format_market_cap(
            info.get("marketCap")
        ),

        "pe_ratio": round(
            info.get("trailingPE", 0),
            2
        ),

        "company": info.get("longName"),

        "sector": info.get("sector"),

        "industry": info.get("industry"),

        "price_change": round(
            price_change,
            2
        ) if price_change else 0,

        "price_change_percent": round(
            price_change_percent,
            2
        ) if price_change_percent else 0,

        "is_positive": (
            price_change >= 0
            if price_change is not None
            else True
        ),

        # ==========================
        # AI
        # ==========================

        "prediction": prediction,

        "decision": decision,

        "ai_explanation": ai_explanation,

        # ==========================
        # Levels
        # ==========================

        "support": levels["support"],

        "resistance": levels["resistance"],

        "risk_reward": risk_reward,

        "market_status": market_status,

        # ==========================
        # AI ENGINE OUTPUT
        # ==========================

        "trade_quality": trade_quality,

        "trade_plan": trade_plan,

        "opportunity": opportunity,

        "probability": probability,

        "ai_confidence": ai_confidence,

        "multi_timeframe": multi_timeframe,

        "entry_engine": entry_engine,

        "trade_validation": trade_validation,

        "final_decision": final_decision,

        "trade_signal": trade_signal,

        "trade_horizon": trade_horizon,

        "candlestick_patterns": candlestick_patterns,

        "volume_analysis": volume_analysis,

        "bullish_signals": trade_quality["bullish_signals"],

        "bearish_signals": trade_quality["bearish_signals"],

        "neutral_signals": trade_quality["neutral_signals"],

        "total_signals": trade_quality["total_signals"],

        "rsi_status": rsi_status,

        "macd_status": macd_status,

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
        "price": round(latest, 2),
        "change": round(change, 2),
        "change_percent": round(change_percent, 2)
    }