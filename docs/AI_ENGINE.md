# 🤖 TradeSense AI Engine

# Overview

The AI Engine is the heart of TradeSense AI.

It combines multiple technical indicators, trend analysis, volatility, and explainability into a single AI recommendation.

Instead of relying on a single indicator, TradeSense AI evaluates multiple market conditions before generating BUY, HOLD, or SELL signals.

---

# AI Workflow

```
Market Data

↓

Technical Indicators

↓

AI Engine

↓

Trade Quality

↓

Confidence Score

↓

Decision Engine

↓

Explainability

↓

Dashboard
```

---

# Technical Indicators Used

## Trend

- MA20
- MA50

Purpose

Identify overall market trend.

---

## Momentum

Indicator

RSI

Purpose

Measure buying and selling pressure.

---

## Momentum Confirmation

Indicator

MACD

Purpose

Confirm trend direction.

---

## Volatility

Indicator

ATR

Purpose

Estimate market volatility.

---

## Price Levels

- Support
- Resistance

Purpose

Identify entry and exit zones.

---

# AI Components

## Trade Quality

Calculates:

- Overall Score
- Grade
- Risk
- Bullish Signals
- Bearish Signals

---

## Opportunity Engine

Determines whether the current setup provides a good trading opportunity.

---

## Confidence Engine

Generates AI Confidence Percentage based on:

- Trend
- Momentum
- Volume
- Volatility
- Risk

---

## Multi Timeframe Analysis

Evaluates higher timeframe confirmation before producing a recommendation.

---

## Decision Engine

Outputs:

- BUY
- HOLD
- SELL

---

## Explainability

Generates human-readable explanations describing why the recommendation was selected.

---

# Final Output

TradeSense AI produces:

- Recommendation
- Confidence
- AI Score
- Risk
- Trade Grade
- Trade Plan
- Support
- Resistance
- Target
- Stop Loss
- AI Explanation