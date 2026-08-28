import pandas as pd

from app.core.liquidity.liquidity import calculate_liquidity


def make_history(rows):
    return pd.DataFrame(
        rows,
        index=pd.date_range(
            "2026-01-01",
            periods=len(rows),
            freq="D"
        )
    )


# ==========================================
# TEST 1 — Equal highs / equal lows
# ==========================================

history = make_history([
    {
        "High": 100,
        "Low": 90,
        "Close": 95
    },
    {
        "High": 110,
        "Low": 92,
        "Close": 105
    },
    {
        "High": 110.2,
        "Low": 91,
        "Close": 100
    },
    {
        "High": 108,
        "Low": 90.2,
        "Close": 102
    }
])

structure = {
    "swing_highs": [
        {
            "index": history.index[1],
            "price": 110
        },
        {
            "index": history.index[2],
            "price": 110.2
        }
    ],
    "swing_lows": [
        {
            "index": history.index[1],
            "price": 92
        },
        {
            "index": history.index[3],
            "price": 91.8
        }
    ]
}

result = calculate_liquidity(
    history,
    structure
)

assert len(result["equal_highs"]) == 1
assert len(result["equal_lows"]) == 1

assert result["equal_highs"][0]["touches"] == 2
assert result["equal_lows"][0]["touches"] == 2

print("TEST 1 — Equal levels: PASS")


# ==========================================
# TEST 2 — Directional classification
# ==========================================

assert all(
    item["type"] == "EQUAL_HIGH"
    for item in result["buy_side_liquidity"]
)

assert all(
    item["type"] == "EQUAL_LOW"
    for item in result["sell_side_liquidity"]
)

print("TEST 2 — Liquidity classification: PASS")


# ==========================================
# TEST 3 — Bullish sell-side sweep
# ==========================================

history = make_history([
    {
        "High": 105,
        "Low": 95,
        "Close": 100
    },
    {
        "High": 104,
        "Low": 90,
        "Close": 93
    },
    {
        "High": 106,
        "Low": 88,
        "Close": 94
    }
])

structure = {
    "swing_highs": [],

    "swing_lows": [
        {
            "index": history.index[0],
            "price": 90
        },
        {
            "index": history.index[1],
            "price": 90.1
        }
    ]
}

result = calculate_liquidity(
    history,
    structure
)

assert len(result["equal_lows"]) == 1

assert result["sweep"]["detected"] is True
assert result["sweep"]["direction"] == "SELL_SIDE"

print("TEST 3 — Sell-side sweep: PASS")


# ==========================================
# TEST 4 — No false sweep
# ==========================================

history = make_history([
    {
        "High": 105,
        "Low": 95,
        "Close": 100
    },
    {
        "High": 104,
        "Low": 91,
        "Close": 92
    },
    {
        "High": 106,
        "Low": 91,
        "Close": 92
    }
])

structure = {
    "swing_highs": [],
    "swing_lows": [
        {
            "index": history.index[0],
            "price": 90
        },
        {
            "index": history.index[1],
            "price": 91
        }
    ]
}

result = calculate_liquidity(
    history,
    structure
)

assert result["sweep"]["detected"] is False

print("TEST 4 — No false sweep: PASS")


# ==========================================
# TEST 5 — Invalid / empty history
# ==========================================

result = calculate_liquidity(
    pd.DataFrame()
)

assert result["strength"] == 0
assert result["sweep"]["detected"] is False

print("TEST 5 — Empty history: PASS")


print()
print("ALL LIQUIDITY TESTS PASSED")
