# 🗄 Database Documentation

# Current Database

TradeSense AI currently uses SQLite as the primary database.

ORM:
- SQLAlchemy

Migration Tool:
- Alembic

---

# Tables

## Users

Purpose

Store registered users.

Future Fields

- id
- username
- email
- password_hash
- created_at

---

## Watchlist

Purpose

Store user watchlists.

Fields

- id
- user_id
- symbol
- company

---

## Portfolio

Purpose

Store user holdings.

Fields

- id
- user_id
- symbol
- quantity
- average_price
- current_price

---

## Transactions

Purpose

Track portfolio transactions.

Fields

- Buy
- Sell
- Quantity
- Price
- Timestamp

---

## Paper Portfolio

Purpose

Virtual trading account.

---

## Paper Transactions

Purpose

Virtual buy/sell history.

---

# Future Database

Planned migration:

SQLite

↓

PostgreSQL

Reasons

- Better scalability
- Multi-user support
- Cloud deployment
- Improved performance