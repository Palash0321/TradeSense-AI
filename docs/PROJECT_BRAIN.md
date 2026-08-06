# 🧠 PROJECT BRAIN

## Purpose

This document captures the reasoning, vision, architecture decisions, development philosophy, and future direction of TradeSense AI.

Unlike technical documentation, this file explains **why** decisions were made, not just **what** was built.

---

# Vision

TradeSense AI is not intended to be another stock price website.

The goal is to build an AI-powered trading intelligence platform that helps users understand the market rather than simply showing indicators.

The long-term vision is to combine:

- AI
- Technical Analysis
- Portfolio Analytics
- Explainable Decisions
- Risk Management
- News Intelligence
- Strategy Testing

into one intelligent platform.

---

# Development Principles

Every feature should satisfy these principles:

- Modular
- Reusable
- Explainable
- Scalable
- Beginner Friendly
- Production Ready

---

# Coding Standards

Whenever possible:

- Keep business logic inside services.
- Keep API routes lightweight.
- Separate AI logic from UI.
- Avoid duplicate calculations.
- Prefer reusable functions.

---

# UI Philosophy

TradeSense AI should look like a professional financial platform.

Design inspiration:

- Bloomberg Terminal
- TradingView
- Zerodha Kite
- Tickertape

Focus areas:

- Clean layout
- Dark theme
- Interactive charts
- Minimal scrolling
- High information density
- Mobile responsiveness

---

# AI Philosophy

Recommendations should never rely on a single indicator.

Instead, combine multiple signals including:

- Trend
- Momentum
- Volatility
- Support / Resistance
- Volume
- Multi-timeframe confirmation

The AI must also explain *why* it generated a recommendation.

---

# Long-Term Product Vision

Future modules include:

- Authentication
- News Sentiment
- Portfolio Optimizer
- Strategy Builder
- AI Chat Assistant
- Real-time Alerts
- Broker Integration
- Mobile Application

---

# Development Workflow

Every coding session should end with:

git status

git add .

git commit -m "meaningful message"

git push

Documentation should be updated whenever a major feature is completed.

---

# Lessons Learned

- Build features incrementally.
- Commit frequently.
- Keep documentation synchronized with the code.
- Favor clarity over unnecessary complexity.
- Preserve backward compatibility whenever possible.

---

# Recovery Guide

If development is resumed after a long break:

1. Read PROJECT_PROGRESS.md.
2. Read PROJECT_BRAIN.md.
3. Review CHANGELOG.md.
4. Start from ROADMAP.md.
5. Check the latest GitHub commit before writing new code.

This process should make it possible to continue development quickly, even after months away from the project.