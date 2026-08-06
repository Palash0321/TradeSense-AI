# 🏗 TradeSense AI Architecture

# Overview

TradeSense AI follows a modular architecture where each responsibility is separated into dedicated layers.

This keeps the application scalable, maintainable, and easy to extend.

---

# High Level Architecture

```
                    User
                      │
                      ▼
             FastAPI Application
                      │
                      ▼
                 API Endpoints
                      │
                      ▼
             Signal Service Layer
                      │
     ┌────────────────┼────────────────┐
     ▼                ▼                ▼
Market Data      AI Engine      Technical Indicators
     │                │                │
     ▼                ▼                ▼
 Yahoo Finance   Decision Engine    RSI / MACD / ATR
                      │
                      ▼
               Explainability Engine
                      │
                      ▼
              Portfolio / Watchlist
                      │
                      ▼
              Jinja2 Templates
                      │
                      ▼
                  Web Browser
```

---

# Project Structure

```
TradeSense-AI/

│
├── app/
│
├── api/
│
├── core/
│
├── models/
│
├── services/
│
├── static/
│
├── templates/
│
├── alembic/
│
├── docs/
│
└── requirements.txt
```

---

# Backend Flow

1. User searches for a stock.
2. FastAPI receives the request.
3. Signal Service downloads market data.
4. Technical indicators are calculated.
5. AI Engine generates analysis.
6. Decision Engine creates BUY / HOLD / SELL recommendation.
7. Explainability Engine prepares reasons.
8. Result is rendered using Jinja2.

---

# Core Modules

## Market Data Service

Responsible for:

- Downloading stock prices
- Company information
- Historical market data

---

## Indicator Engine

Calculates:

- MA20
- MA50
- RSI
- MACD
- ATR
- Support & Resistance

---

## AI Engine

Responsible for:

- AI Score
- Confidence
- Trade Quality
- Opportunity Detection
- Momentum
- Market Health
- Multi-timeframe Analysis

---

## Decision Engine

Generates:

- BUY
- HOLD
- SELL

using AI outputs.

---

## Explainability Engine

Explains why the AI selected the recommendation.

---

## Portfolio Module

Responsible for:

- Holdings
- Transactions
- Performance
- Portfolio Summary

---

## Paper Trading Module

Responsible for:

- Virtual Orders
- Paper Portfolio
- Analytics

---

## Frontend

Technologies

- HTML
- CSS
- JavaScript
- Lightweight Charts
- Jinja2 Templates

---

# Database

Current

- SQLite

Future

- PostgreSQL

---

# Deployment

Future Target

- Docker
- Render / Railway
- Nginx
- PostgreSQL