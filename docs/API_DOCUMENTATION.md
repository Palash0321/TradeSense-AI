# 🌐 TradeSense AI API Documentation

# Overview

This document describes all API endpoints currently available in TradeSense AI.

---

# Home

## GET /

Description

Loads the homepage where users can search for stocks.

---

# Analyze Stock

## GET /analyze

Description

Analyzes a selected stock and generates AI-based recommendations.

### Parameters

| Parameter | Description |
|-----------|-------------|
| symbol | Stock Symbol |
| market | india / us |

---

# Chart Data

## GET /api/chart-data

Description

Returns historical candlestick data for Lightweight Charts.

### Response

- OHLC Candles
- Support
- Resistance
- Target
- Stop Loss
- AI Recommendation Marker

---

# Market Overview

## GET /api/market-overview

Description

Returns overall market statistics.

---

# Market Movers

## GET /api/market-movers

Description

Returns Top Gainers and Top Losers.

### Parameter

market

---

# Option Chain

## GET /api/option-chain

Description

Returns Option Chain information.

### Parameters

index

expiry

---

# Watchlist

## GET /watchlist

Displays the watchlist page.

---

## POST /api/watchlist

Adds a stock to the user's watchlist.

---

# Portfolio

## GET /portfolio

Portfolio dashboard.

---

## GET /api/portfolio-summary

Returns portfolio summary.

---

## GET /api/transactions

Returns portfolio transactions.

---

# Paper Trading

## GET /paper-dashboard

Loads Paper Trading Dashboard.

---

## GET /paper-portfolio

Returns Paper Portfolio.

---

## GET /paper-history

Returns Paper Trading History.

---

## POST /paper-orders

Creates Paper Orders.

---

## GET /paper-analytics

Paper Trading Analytics.

---

# Backtesting

## GET /backtest

Loads Backtesting module.

---

# Authentication

## GET /login

User Login Page.

---

## GET /register

User Registration Page.

---

# Future APIs

Planned Endpoints

- AI News Sentiment
- Strategy Builder
- Portfolio Optimizer
- ML Prediction API
- Real-time Alerts