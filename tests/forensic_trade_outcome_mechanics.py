"""
TradeSense-AI
Trade Outcome Mechanics Forensic

Purpose
-------
Analyze the frozen historical TradeSense-AI dataset to determine whether
poor performance is primarily caused by:

1. Immediate adverse movement
2. Failure to develop
3. Insufficient MFE
4. Stop-loss behavior
5. Target behavior
6. Long holding periods
7. Setup-specific outcome mechanics

IMPORTANT
---------
Diagnostic only.

This script MUST NOT:
- modify production code
- modify the frozen historical dataset
- modify holdout data
- change strategy parameters
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict


HISTORICAL_FILE = "tests/output/tradesense_trades.csv"

OUTPUT_DIR = "tests/output/reconciliation"

OUTPUT_FILE = (
    f"{OUTPUT_DIR}/trade_outcome_mechanics_forensic.csv"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def pct(value, total):
    if total == 0:
        return 0.0
    return value / total * 100.0


def get_setup(trade):
    return (
        trade.get("Diagnostic_Setup")
        or trade.get("setup")
        or "UNKNOWN"
    )


def get_trend(trade):
    return (
        trade.get("Diagnostic_Trend")
        or trade.get("entry_trend")
        or "UNKNOWN"
    )


def get_r(trade):
    return to_float(
        trade.get("r_multiple"),
        0.0
    )


def get_mfe(trade):
    return to_float(
        trade.get("max_favorable_r"),
        0.0
    )


def get_mae(trade):
    return to_float(
        trade.get("max_adverse_r"),
        0.0
    )


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

if not os.path.exists(HISTORICAL_FILE):
    raise FileNotFoundError(
        f"Frozen historical dataset not found: {HISTORICAL_FILE}"
    )


with open(
    HISTORICAL_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    trades = list(reader)


print()
print("=" * 110)
print("TRADESENSE-AI — TRADE OUTCOME MECHANICS FORENSIC")
print("=" * 110)

print()
print(f"Frozen historical trades : {len(trades)}")
print(f"Source                   : {HISTORICAL_FILE}")


# ---------------------------------------------------------------------
# GLOBAL OUTCOME MECHANICS
# ---------------------------------------------------------------------

total_r = sum(get_r(t) for t in trades)

wins = [
    t for t in trades
    if get_r(t) > 0
]

losses = [
    t for t in trades
    if get_r(t) <= 0
]

target_trades = [
    t for t in trades
    if str(t.get("exit_reason", "")).upper() == "TARGET"
]

stop_trades = [
    t for t in trades
    if str(t.get("exit_reason", "")).upper() == "STOP_LOSS"
]

immediate_failures = [
    t for t in trades
    if str(t.get("immediate_failure", "")).lower()
    in {"true", "1", "yes"}
]

mfe_05 = [
    t for t in trades
    if str(t.get("reached_0_5r", "")).lower()
    in {"true", "1", "yes"}
]

mfe_10 = [
    t for t in trades
    if str(t.get("reached_1r", "")).lower()
    in {"true", "1", "yes"}
]

mfe_15 = [
    t for t in trades
    if str(t.get("reached_1_5r", "")).lower()
    in {"true", "1", "yes"}
]


print()
print("=" * 110)
print("GLOBAL OUTCOME MECHANICS")
print("=" * 110)

print(f"Total R                  : {total_r:.2f}R")
print(f"Winning trades           : {len(wins)}")
print(f"Losing trades            : {len(losses)}")

print()
print(
    f"Reached 0.5R             : "
    f"{len(mfe_05):3d} "
    f"({pct(len(mfe_05), len(trades)):.2f}%)"
)

print(
    f"Reached 1.0R             : "
    f"{len(mfe_10):3d} "
    f"({pct(len(mfe_10), len(trades)):.2f}%)"
)

print(
    f"Reached 1.5R             : "
    f"{len(mfe_15):3d} "
    f"({pct(len(mfe_15), len(trades)):.2f}%)"
)

print()
print(
    f"TARGET exits             : "
    f"{len(target_trades):3d} "
    f"({pct(len(target_trades), len(trades)):.2f}%) "
    f"R={sum(get_r(t) for t in target_trades):.2f}"
)

print(
    f"STOP_LOSS exits          : "
    f"{len(stop_trades):3d} "
    f"({pct(len(stop_trades), len(trades)):.2f}%) "
    f"R={sum(get_r(t) for t in stop_trades):.2f}"
)

print(
    f"Immediate failures       : "
    f"{len(immediate_failures):3d} "
    f"({pct(len(immediate_failures), len(trades)):.2f}%) "
    f"R={sum(get_r(t) for t in immediate_failures):.2f}"
)


# ---------------------------------------------------------------------
# MFE BUCKET ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("MFE BUCKET ANALYSIS")
print("=" * 110)

mfe_buckets = [
    ("< 0.25R", float("-inf"), 0.25),
    ("0.25R - 0.50R", 0.25, 0.50),
    ("0.50R - 1.00R", 0.50, 1.00),
    ("1.00R - 1.50R", 1.00, 1.50),
    ("1.50R - 2.00R", 1.50, 2.00),
    (">= 2.00R", 2.00, float("inf")),
]


mfe_rows = []


for name, lower, upper in mfe_buckets:

    bucket = [
        t
        for t in trades
        if get_mfe(t) >= lower
        and get_mfe(t) < upper
    ]

    bucket_r = sum(get_r(t) for t in bucket)

    avg_r = safe_div(
        bucket_r,
        len(bucket)
    )

    wins_bucket = sum(
        1
        for t in bucket
        if get_r(t) > 0
    )

    print(
        f"{name:<18} "
        f"Trades={len(bucket):3d} "
        f"Win={pct(wins_bucket, len(bucket)):6.2f}% "
        f"TotalR={bucket_r:8.2f} "
        f"AvgR={avg_r:7.3f}"
    )

    mfe_rows.append(
        {
            "analysis": "MFE_BUCKET",
            "group": name,
            "trades": len(bucket),
            "wins": wins_bucket,
            "win_rate": pct(
                wins_bucket,
                len(bucket)
            ),
            "total_r": bucket_r,
            "avg_r": avg_r,
        }
    )


# ---------------------------------------------------------------------
# MAE BUCKET ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("MAE BUCKET ANALYSIS")
print("=" * 110)

mae_buckets = [
    ("0.00 - 0.25R", 0.00, 0.25),
    ("0.25 - 0.50R", 0.25, 0.50),
    ("0.50 - 0.75R", 0.50, 0.75),
    ("0.75 - 1.00R", 0.75, 1.00),
    ("1.00 - 1.50R", 1.00, 1.50),
    (">= 1.50R", 1.50, float("inf")),
]


mae_rows = []


for name, lower, upper in mae_buckets:

    bucket = [
        t
        for t in trades
        if get_mae(t) >= lower
        and get_mae(t) < upper
    ]

    bucket_r = sum(get_r(t) for t in bucket)

    avg_r = safe_div(
        bucket_r,
        len(bucket)
    )

    wins_bucket = sum(
        1
        for t in bucket
        if get_r(t) > 0
    )

    print(
        f"{name:<18} "
        f"Trades={len(bucket):3d} "
        f"Win={pct(wins_bucket, len(bucket)):6.2f}% "
        f"TotalR={bucket_r:8.2f} "
        f"AvgR={avg_r:7.3f}"
    )

    mae_rows.append(
        {
            "analysis": "MAE_BUCKET",
            "group": name,
            "trades": len(bucket),
            "wins": wins_bucket,
            "win_rate": pct(
                wins_bucket,
                len(bucket)
            ),
            "total_r": bucket_r,
            "avg_r": avg_r,
        }
    )


# ---------------------------------------------------------------------
# EXIT REASON ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("EXIT REASON ANALYSIS")
print("=" * 110)

exit_groups = defaultdict(list)

for trade in trades:

    reason = (
        trade.get("exit_reason")
        or "UNKNOWN"
    )

    exit_groups[reason].append(trade)


exit_rows = []


for reason in sorted(exit_groups):

    group = exit_groups[reason]

    group_r = sum(
        get_r(t)
        for t in group
    )

    wins_group = sum(
        1
        for t in group
        if get_r(t) > 0
    )

    avg_r = safe_div(
        group_r,
        len(group)
    )

    avg_mfe = safe_div(
        sum(get_mfe(t) for t in group),
        len(group)
    )

    avg_mae = safe_div(
        sum(get_mae(t) for t in group),
        len(group)
    )

    print(
        f"{reason:<22} "
        f"Trades={len(group):3d} "
        f"Win={pct(wins_group, len(group)):6.2f}% "
        f"TotalR={group_r:8.2f} "
        f"AvgR={avg_r:7.3f} "
        f"AvgMFE={avg_mfe:6.2f} "
        f"AvgMAE={avg_mae:6.2f}"
    )

    exit_rows.append(
        {
            "analysis": "EXIT_REASON",
            "group": reason,
            "trades": len(group),
            "wins": wins_group,
            "win_rate": pct(
                wins_group,
                len(group)
            ),
            "total_r": group_r,
            "avg_r": avg_r,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
        }
    )


# ---------------------------------------------------------------------
# SETUP ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("SETUP OUTCOME MECHANICS")
print("=" * 110)

setup_groups = defaultdict(list)

for trade in trades:
    setup_groups[
        get_setup(trade)
    ].append(trade)


setup_rows = []


for setup in sorted(setup_groups):

    group = setup_groups[setup]

    group_r = sum(
        get_r(t)
        for t in group
    )

    wins_group = sum(
        1
        for t in group
        if get_r(t) > 0
    )

    avg_mfe = safe_div(
        sum(get_mfe(t) for t in group),
        len(group)
    )

    avg_mae = safe_div(
        sum(get_mae(t) for t in group),
        len(group)
    )

    immediate = sum(
        1
        for t in group
        if str(t.get("immediate_failure", "")).lower()
        in {"true", "1", "yes"}
    )

    reached_15 = sum(
        1
        for t in group
        if str(t.get("reached_1_5r", "")).lower()
        in {"true", "1", "yes"}
    )

    print(
        f"{setup:<22} "
        f"Trades={len(group):3d} "
        f"Win={pct(wins_group, len(group)):6.2f}% "
        f"TotalR={group_r:8.2f} "
        f"AvgMFE={avg_mfe:6.2f} "
        f"AvgMAE={avg_mae:6.2f} "
        f"1.5R={pct(reached_15, len(group)):6.2f}% "
        f"Immediate={pct(immediate, len(group)):6.2f}%"
    )

    setup_rows.append(
        {
            "analysis": "SETUP",
            "group": setup,
            "trades": len(group),
            "wins": wins_group,
            "win_rate": pct(
                wins_group,
                len(group)
            ),
            "total_r": group_r,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "reached_1_5r_pct": pct(
                reached_15,
                len(group)
            ),
            "immediate_failure_pct": pct(
                immediate,
                len(group)
            ),
        }
    )


# ---------------------------------------------------------------------
# SETUP × EXIT REASON
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("SETUP × EXIT REASON")
print("=" * 110)

setup_exit_groups = defaultdict(list)

for trade in trades:

    key = (
        get_setup(trade),
        trade.get("exit_reason") or "UNKNOWN"
    )

    setup_exit_groups[key].append(trade)


setup_exit_rows = []


for (setup, reason), group in sorted(
    setup_exit_groups.items()
):

    group_r = sum(
        get_r(t)
        for t in group
    )

    print(
        f"{setup:<22} × "
        f"{reason:<22} "
        f"Trades={len(group):3d} "
        f"R={group_r:8.2f}"
    )

    setup_exit_rows.append(
        {
            "analysis": "SETUP_X_EXIT",
            "group": f"{setup} × {reason}",
            "trades": len(group),
            "total_r": group_r,
        }
    )


# ---------------------------------------------------------------------
# HOLDING-TIME ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("HOLDING-TIME ANALYSIS")
print("=" * 110)

holding_buckets = [
    ("1-5 bars", 1, 5),
    ("6-10 bars", 6, 10),
    ("11-20 bars", 11, 20),
    ("21-50 bars", 21, 50),
    ("51-100 bars", 51, 100),
    (">100 bars", 101, 10**9),
]


holding_rows = []


for name, lower, upper in holding_buckets:
    
    bucket = []

    for trade in trades:

        bars = to_int(
            trade.get("bars_in_trade")
        )

        if bars is None:
            continue

        if lower <= bars <= upper:
            bucket.append(trade)

    bucket_r = sum(
        get_r(t)
        for t in bucket
    )

    wins_bucket = sum(
        1
        for t in bucket
        if get_r(t) > 0
    )

    print(
        f"{name:<18} "
        f"Trades={len(bucket):3d} "
        f"Win={pct(wins_bucket, len(bucket)):6.2f}% "
        f"TotalR={bucket_r:8.2f} "
        f"AvgR={safe_div(bucket_r, len(bucket)):7.3f}"
    )

    holding_rows.append(
        {
            "analysis": "HOLDING_TIME",
            "group": name,
            "trades": len(bucket),
            "wins": wins_bucket,
            "win_rate": pct(
                wins_bucket,
                len(bucket)
            ),
            "total_r": bucket_r,
            "avg_r": safe_div(
                bucket_r,
                len(bucket)
            ),
        }
    )


# ---------------------------------------------------------------------
# EARLY DEVELOPMENT ANALYSIS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("EARLY DEVELOPMENT ANALYSIS")
print("=" * 110)

development_groups = [
    (
        "Never reached 0.5R",
        lambda t: str(
            t.get("reached_0_5r", "")
        ).lower() not in {"true", "1", "yes"}
    ),
    (
        "Reached 0.5R but not 1R",
        lambda t: (
            str(t.get("reached_0_5r", "")).lower()
            in {"true", "1", "yes"}
            and
            str(t.get("reached_1r", "")).lower()
            not in {"true", "1", "yes"}
        )
    ),
    (
        "Reached 1R but not 1.5R",
        lambda t: (
            str(t.get("reached_1r", "")).lower()
            in {"true", "1", "yes"}
            and
            str(t.get("reached_1_5r", "")).lower()
            not in {"true", "1", "yes"}
        )
    ),
    (
        "Reached 1.5R",
        lambda t: str(
            t.get("reached_1_5r", "")
        ).lower() in {"true", "1", "yes"}
    ),
]


development_rows = []


for name, predicate in development_groups:

    group = [
        t for t in trades
        if predicate(t)
    ]

    group_r = sum(
        get_r(t)
        for t in group
    )

    wins_group = sum(
        1
        for t in group
        if get_r(t) > 0
    )

    avg_bars = safe_div(
        sum(
            to_int(
                t.get("bars_in_trade"),
                0
            )
            for t in group
        ),
        len(group)
    )

    print(
        f"{name:<28} "
        f"Trades={len(group):3d} "
        f"Win={pct(wins_group, len(group)):6.2f}% "
        f"TotalR={group_r:8.2f} "
        f"AvgR={safe_div(group_r, len(group)):7.3f} "
        f"AvgBars={avg_bars:7.2f}"
    )

    development_rows.append(
        {
            "analysis": "EARLY_DEVELOPMENT",
            "group": name,
            "trades": len(group),
            "wins": wins_group,
            "win_rate": pct(
                wins_group,
                len(group)
            ),
            "total_r": group_r,
            "avg_r": safe_div(
                group_r,
                len(group)
            ),
            "avg_bars": avg_bars,
        }
    )


# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


all_rows = (
    mfe_rows
    + mae_rows
    + exit_rows
    + setup_rows
    + setup_exit_rows
    + holding_rows
    + development_rows
)


fieldnames = sorted(
    {
        key
        for row in all_rows
        for key in row.keys()
    }
)


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(all_rows)


print()
print("=" * 110)
print("FORENSIC COMPLETE")
print("=" * 110)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print("IMPORTANT:")
print("No production code was modified.")
print("Frozen historical dataset was not modified.")
print("Holdout dataset was not modified.")