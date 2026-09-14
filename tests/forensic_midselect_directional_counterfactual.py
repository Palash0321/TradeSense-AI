"""
TradeSense-AI
NIFTY MIDCAP SELECT Directional Counterfactual

Research-only.

Tests:
1. ALL trades baseline
2. LONG ONLY counterfactual
3. SHORT ONLY counterfactual
4. Leave-one-year-out LONG ONLY
5. Leave-one-year-out SHORT ONLY

Purpose:
Determine whether the apparent LONG vs SHORT difference
in the MIDSELECT baseline is strong enough to justify
a deeper research investigation.

IMPORTANT:
- No production changes.
- No frozen stock dataset changes.
- No holdout changes.
- No strategy modification.
- This is NOT deployment approval.
"""

import os

import numpy as np
import pandas as pd


TRADE_FILE = (
    "tests/output/index_backtests/"
    "midselect_index_trades.csv"
)


EXPECTED_TRADE_COUNT = 13


# ============================================================================
# HELPERS
# ============================================================================

def calculate_metrics(df):
    if df.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_r": 0.0,
            "average_r": 0.0,
        }

    r = pd.to_numeric(
        df["r_multiple"],
        errors="coerce",
    ).fillna(0.0)

    wins = int(
        (r > 0).sum()
    )

    losses = int(
        (r <= 0).sum()
    )

    gross_profit = float(
        r[r > 0].sum()
    )

    gross_loss = float(
        r[r < 0].sum()
    )

    absolute_loss = abs(
        gross_loss
    )

    if absolute_loss > 0:
        profit_factor = (
            gross_profit
            / absolute_loss
        )
    else:
        profit_factor = np.inf

    return {
        "trades": len(df),
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / len(df) * 100
        ),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "total_r": float(r.sum()),
        "average_r": float(r.mean()),
    }


def print_metrics(label, df):
    metrics = calculate_metrics(df)

    print()
    print(label)
    print("-" * 70)

    print(
        f"Trades       : "
        f"{metrics['trades']}"
    )

    print(
        f"Winners      : "
        f"{metrics['wins']}"
    )

    print(
        f"Losers       : "
        f"{metrics['losses']}"
    )

    print(
        f"Win rate     : "
        f"{metrics['win_rate']:.2f}%"
    )

    print(
        f"Gross profit : "
        f"{metrics['gross_profit']:+.2f}R"
    )

    print(
        f"Gross loss   : "
        f"{metrics['gross_loss']:+.2f}R"
    )

    if np.isfinite(
        metrics["profit_factor"]
    ):
        print(
            f"Profit factor: "
            f"{metrics['profit_factor']:.3f}"
        )
    else:
        print(
            "Profit factor: INF"
        )

    print(
        f"Total R      : "
        f"{metrics['total_r']:+.2f}R"
    )

    print(
        f"Average R    : "
        f"{metrics['average_r']:+.3f}R"
    )

    return metrics


# ============================================================================
# HEADER
# ============================================================================

print("=" * 100)
print(
    "TRADESENSE-AI — "
    "NIFTY MIDCAP SELECT DIRECTIONAL COUNTERFACTUAL"
)
print("=" * 100)

print(
    f"Trade file: {TRADE_FILE}"
)


# ============================================================================
# LOAD
# ============================================================================

if not os.path.exists(
    TRADE_FILE
):
    print()
    print(
        "ERROR: Trade file does not exist."
    )
    raise SystemExit(1)


df = pd.read_csv(
    TRADE_FILE
)


if df.empty:
    print()
    print(
        "ERROR: Trade file is empty."
    )
    raise SystemExit(1)


print(
    f"Loaded trades: {len(df)}"
)


# ============================================================================
# VALIDATION
# ============================================================================

required_columns = [
    "entry_date",
    "exit_date",
    "direction",
    "setup",
    "r_multiple",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    print()
    print(
        "ERROR: Missing required columns:"
    )

    for column in missing:
        print(
            f"  - {column}"
        )

    raise SystemExit(1)


df["entry_date"] = pd.to_datetime(
    df["entry_date"],
    errors="coerce",
)


df["exit_date"] = pd.to_datetime(
    df["exit_date"],
    errors="coerce",
)


df["r_multiple"] = pd.to_numeric(
    df["r_multiple"],
    errors="coerce",
)


if df["r_multiple"].isna().any():

    print()
    print(
        "ERROR: Missing r_multiple values."
    )

    raise SystemExit(1)


if len(df) != EXPECTED_TRADE_COUNT:

    print()
    print(
        "WARNING: Expected "
        f"{EXPECTED_TRADE_COUNT} trades but found "
        f"{len(df)}."
    )


duplicate_count = int(
    df.duplicated(
        subset=[
            "entry_date",
            "direction",
            "setup",
        ]
    ).sum()
)


print(
    f"Duplicate identities: "
    f"{duplicate_count}"
)


if duplicate_count != 0:

    print()
    print(
        "ERROR: Duplicate trade identities."
    )

    raise SystemExit(1)


# ============================================================================
# BASELINE
# ============================================================================

print()
print("=" * 100)
print("BASELINE")
print("=" * 100)

baseline_metrics = print_metrics(
    "ALL TRADES",
    df,
)


# ============================================================================
# DIRECTION SPLIT
# ============================================================================

long_df = df[
    df["direction"]
    .astype(str)
    .str.upper()
    == "LONG"
].copy()


short_df = df[
    df["direction"]
    .astype(str)
    .str.upper()
    == "SHORT"
].copy()


print()
print("=" * 100)
print("DIRECTIONAL COUNTERFACTUAL")
print("=" * 100)

long_metrics = print_metrics(
    "LONG ONLY",
    long_df,
)


short_metrics = print_metrics(
    "SHORT ONLY",
    short_df,
)


# ============================================================================
# DELTA VS BASELINE
# ============================================================================

print()
print("=" * 100)
print("DIRECTIONAL DELTA VS BASELINE")
print("=" * 100)

print(
    f"LONG ONLY delta : "
    f"{long_metrics['total_r'] - baseline_metrics['total_r']:+.2f}R"
)

print(
    f"SHORT ONLY delta: "
    f"{short_metrics['total_r'] - baseline_metrics['total_r']:+.2f}R"
)

print()
print(
    f"Removing SHORT trades "
    f"(LONG ONLY): "
    f"{long_metrics['total_r']:+.2f}R"
)

print(
    f"Removing LONG trades "
    f"(SHORT ONLY): "
    f"{short_metrics['total_r']:+.2f}R"
)


# ============================================================================
# YEAR PROFILE
# ============================================================================

df["year"] = (
    df["entry_date"]
    .dt.year
)


print()
print("=" * 100)
print("YEAR-BY-YEAR DIRECTION PROFILE")
print("=" * 100)


years = sorted(
    df["year"]
    .dropna()
    .unique()
)


for year in years:

    year_df = df[
        df["year"] == year
    ]

    year_long = year_df[
        year_df["direction"]
        .astype(str)
        .str.upper()
        == "LONG"
    ]

    year_short = year_df[
        year_df["direction"]
        .astype(str)
        .str.upper()
        == "SHORT"
    ]

    year_metrics = calculate_metrics(
        year_df
    )

    long_year_metrics = calculate_metrics(
        year_long
    )

    short_year_metrics = calculate_metrics(
        year_short
    )

    print()
    print(
        f"{int(year)}"
    )

    print(
        f"  ALL   : "
        f"{year_metrics['trades']} trades | "
        f"{year_metrics['total_r']:+.2f}R"
    )

    print(
        f"  LONG  : "
        f"{long_year_metrics['trades']} trades | "
        f"{long_year_metrics['total_r']:+.2f}R"
    )

    print(
        f"  SHORT : "
        f"{short_year_metrics['trades']} trades | "
        f"{short_year_metrics['total_r']:+.2f}R"
    )


# ============================================================================
# LEAVE-ONE-YEAR-OUT
# ============================================================================

print()
print("=" * 100)
print("LEAVE-ONE-YEAR-OUT — LONG ONLY")
print("=" * 100)


long_loo_results = []


for year in years:

    subset = long_df[
        long_df["entry_date"].dt.year
        != year
    ]

    metrics = calculate_metrics(
        subset
    )

    long_loo_results.append(
        {
            "excluded_year": int(year),
            "trades": metrics["trades"],
            "total_r": metrics["total_r"],
            "win_rate": metrics["win_rate"],
            "profit_factor": (
                metrics["profit_factor"]
            ),
        }
    )

    pf_text = (
        f"{metrics['profit_factor']:.3f}"
        if np.isfinite(
            metrics["profit_factor"]
        )
        else "INF"
    )

    print(
        f"Exclude {int(year)}: "
        f"{metrics['trades']} trades | "
        f"{metrics['win_rate']:.2f}% | "
        f"{metrics['total_r']:+.2f}R | "
        f"PF {pf_text}"
    )


print()
print("=" * 100)
print("LEAVE-ONE-YEAR-OUT — SHORT ONLY")
print("=" * 100)


short_loo_results = []


for year in years:

    subset = short_df[
        short_df["entry_date"].dt.year
        != year
    ]

    metrics = calculate_metrics(
        subset
    )

    short_loo_results.append(
        {
            "excluded_year": int(year),
            "trades": metrics["trades"],
            "total_r": metrics["total_r"],
            "win_rate": metrics["win_rate"],
            "profit_factor": (
                metrics["profit_factor"]
            ),
        }
    )

    pf_text = (
        f"{metrics['profit_factor']:.3f}"
        if np.isfinite(
            metrics["profit_factor"]
        )
        else "INF"
    )

    print(
        f"Exclude {int(year)}: "
        f"{metrics['trades']} trades | "
        f"{metrics['win_rate']:.2f}% | "
        f"{metrics['total_r']:+.2f}R | "
        f"PF {pf_text}"
    )


# ============================================================================
# SETUP × DIRECTION
# ============================================================================

print()
print("=" * 100)
print("SETUP × DIRECTION")
print("=" * 100)


for setup in sorted(
    df["setup"]
    .dropna()
    .unique()
):

    setup_df = df[
        df["setup"] == setup
    ]

    print()
    print(
        f"{setup}"
    )

    for direction in [
        "LONG",
        "SHORT",
    ]:

        group = setup_df[
            setup_df["direction"]
            .astype(str)
            .str.upper()
            == direction
        ]

        metrics = calculate_metrics(
            group
        )

        print(
            f"  {direction:<5}: "
            f"{metrics['trades']} trades | "
            f"{metrics['wins']}W/"
            f"{metrics['losses']}L | "
            f"{metrics['total_r']:+.2f}R"
        )


# ============================================================================
# INDIVIDUAL SHORT TRADES
# ============================================================================

print()
print("=" * 100)
print("INDIVIDUAL SHORT TRADES")
print("=" * 100)


if short_df.empty:

    print(
        "No SHORT trades."
    )

else:

    short_display = [
        "entry_date",
        "exit_date",
        "setup",
        "r_multiple",
        "max_favorable_r",
        "max_adverse_r",
        "early_adverse_r_bar_5",
        "bars_in_trade",
        "exit_reason",
    ]

    available_columns = [
        column
        for column in short_display
        if column in short_df.columns
    ]

    for _, row in (
        short_df[
            available_columns
        ]
        .sort_values("entry_date")
        .iterrows()
    ):

        mfe = row.get(
            "max_favorable_r",
            np.nan,
        )

        mae = row.get(
            "max_adverse_r",
            np.nan,
        )

        early5 = row.get(
            "early_adverse_r_bar_5",
            np.nan,
        )

        bars = row.get(
            "bars_in_trade",
            np.nan,
        )

        print(
            f"{row['entry_date'].date()} | "
            f"{row['exit_date'].date()} | "
            f"{row['setup']:<20} | "
            f"R={row['r_multiple']:+.2f} | "
            f"MFE={mfe:.2f} | "
            f"MAE={mae:.2f} | "
            f"Early5={early5:.2f} | "
            f"bars={int(bars)} | "
            f"{row['exit_reason']}"
        )


# ============================================================================
# RESEARCH VERDICT
# ============================================================================

print()
print("=" * 100)
print("RESEARCH VERDICT")
print("=" * 100)


long_delta = (
    long_metrics["total_r"]
    - baseline_metrics["total_r"]
)


short_delta = (
    short_metrics["total_r"]
    - baseline_metrics["total_r"]
)


positive_long_loo = sum(
    result["total_r"] > 0
    for result in long_loo_results
    if result["trades"] > 0
)


negative_long_loo = sum(
    result["total_r"] < 0
    for result in long_loo_results
    if result["trades"] > 0
)


positive_short_loo = sum(
    result["total_r"] > 0
    for result in short_loo_results
    if result["trades"] > 0
)


negative_short_loo = sum(
    result["total_r"] < 0
    for result in short_loo_results
    if result["trades"] > 0
)


print()
print(
    f"LONG ONLY total R : "
    f"{long_metrics['total_r']:+.2f}R"
)

print(
    f"SHORT ONLY total R: "
    f"{short_metrics['total_r']:+.2f}R"
)

print(
    f"LONG-only LOO positive years : "
    f"{positive_long_loo}"
)

print(
    f"LONG-only LOO negative years : "
    f"{negative_long_loo}"
)

print(
    f"SHORT-only LOO positive years: "
    f"{positive_short_loo}"
)

print(
    f"SHORT-only LOO negative years: "
    f"{negative_short_loo}"
)

print()


if len(df) < 20:

    print(
        "VERDICT: INSUFFICIENT_SAMPLE"
    )

    print()
    print(
        "The directional difference is "
        "research-worthy, but the total sample "
        f"is only {len(df)} trades."
    )

    print(
        "Do not implement a LONG-only or "
        "SHORT-exclusion rule from this result."
    )

else:

    print(
        "VERDICT: COUNTERFACTUAL_COMPLETE"
    )


print()
print(
    "No production strategy was modified."
)

print(
    "No historical stock dataset was modified."
)

print(
    "No holdout dataset was modified."
)