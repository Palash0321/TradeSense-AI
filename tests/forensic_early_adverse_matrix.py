"""
TradeSense-AI
EARLY-ADVERSE THRESHOLD × WINDOW ROBUSTNESS FORENSIC

RESEARCH ONLY.
DO NOT DEPLOY.

Purpose
-------
Test whether the previously observed early-adverse edge is robust
across multiple observation windows and adverse-R thresholds.

Frozen historical dataset is READ ONLY.
Production code is NOT modified.
Holdout dataset is NOT modified.
"""

import csv
import os
from collections import defaultdict


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

HISTORICAL_TRADES_FILE = "tests/output/tradesense_trades.csv"

OUTPUT_DIR = "tests/output/reconciliation"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "early_adverse_threshold_window_matrix.csv"
)

BASELINE_TOTAL_R = -29.15
BASELINE_TRADE_COUNT = 106

WINDOWS = [3, 5, 7, 10]

THRESHOLDS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def safe_float(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def profit_factor(trades):
    gross_profit = sum(
        max(safe_float(t.get("r_multiple")) or 0.0, 0.0)
        for t in trades
    )

    gross_loss = sum(
        abs(min(safe_float(t.get("r_multiple")) or 0.0, 0.0))
        for t in trades
    )

    if gross_loss == 0:
        if gross_profit > 0:
            return float("inf")
        return 0.0

    return gross_profit / gross_loss


def calculate_metrics(trades):
    count = len(trades)

    if count == 0:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
        }

    r_values = [
        safe_float(t.get("r_multiple")) or 0.0
        for t in trades
    ]

    wins = sum(
        1
        for r in r_values
        if r > 0
    )

    total_r = sum(r_values)

    return {
        "trades": count,
        "wins": wins,
        "win_rate": (wins / count) * 100.0,
        "total_r": total_r,
        "avg_r": total_r / count,
        "profit_factor": profit_factor(trades),
    }


def early_adverse_value(trade, window):
    """
    Return the maximum adverse R observed during the requested
    first N bars after entry.

    The frozen dataset already contains:
        early_adverse_r_bar_1
        ...
        early_adverse_r_bar_5

    For windows <= 5 we use the exact stored early-adverse fields.

    For windows > 5, this diagnostic cannot reconstruct bars 6-10
    from the frozen trade-level CSV alone.

    Therefore:
        3 / 5 bars = exact frozen fields
        7 / 10 bars = NOT available from frozen dataset

    Those cells are intentionally marked unavailable rather than
    inventing information.
    """

    if window > 5:
        return None

    values = []

    for bar in range(1, window + 1):

        field = f"early_adverse_r_bar_{bar}"

        value = safe_float(
            trade.get(field)
        )

        if value is not None:
            values.append(value)

    if not values:
        return None

    return max(values)


def classify_year(entry_date):
    if not entry_date:
        return None

    text = str(entry_date)

    if len(text) < 4:
        return None

    try:
        return int(text[:4])
    except ValueError:
        return None


def fmt(value, digits=2):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------
# LOAD FROZEN DATA
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("TRADESENSE-AI — EARLY-ADVERSE THRESHOLD × WINDOW ROBUSTNESS FORENSIC")
print("=" * 110)
print()
print("RESEARCH ONLY — DO NOT DEPLOY")
print()
print(f"Frozen historical trades : {HISTORICAL_TRADES_FILE}")

with open(
    HISTORICAL_TRADES_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    trades = list(reader)


print(f"Trades loaded            : {len(trades)}")


if len(trades) != BASELINE_TRADE_COUNT:
    raise RuntimeError(
        "Frozen historical trade count changed unexpectedly."
    )


# ---------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------

baseline = calculate_metrics(trades)

print()
print("=" * 110)
print("BASELINE")
print("=" * 110)

print(
    f"Trades       : {baseline['trades']}"
)

print(
    f"Win rate     : {baseline['win_rate']:.2f}%"
)

print(
    f"Total R      : {baseline['total_r']:.2f}R"
)

print(
    f"Avg R        : {baseline['avg_r']:.3f}R"
)

print(
    f"Profit factor: {baseline['profit_factor']:.3f}"
)


# ---------------------------------------------------------------------
# MATRIX
# ---------------------------------------------------------------------

results = []

print()
print("=" * 110)
print("THRESHOLD × WINDOW MATRIX")
print("=" * 110)
print()

print(
    f"{'Window':>8} "
    f"{'Threshold':>11} "
    f"{'Flagged':>9} "
    f"{'Remain':>9} "
    f"{'Win%':>9} "
    f"{'TotalR':>11} "
    f"{'DeltaR':>11} "
    f"{'PF':>8} "
    f"{'Status':>14}"
)

print("-" * 110)


for window in WINDOWS:

    for threshold in THRESHOLDS:

        # -------------------------------------------------------------
        # Unsupported windows are explicitly recorded.
        # -------------------------------------------------------------

        if window > 5:

            row = {
                "window_bars": window,
                "threshold_r": threshold,
                "status": "UNAVAILABLE",
                "reason": (
                    "Frozen trade CSV contains early adverse data "
                    "only through bar 5."
                ),
            }

            results.append(row)

            print(
                f"{window:>8} "
                f"{threshold:>11.2f} "
                f"{'N/A':>9} "
                f"{'N/A':>9} "
                f"{'N/A':>9} "
                f"{'N/A':>11} "
                f"{'N/A':>11} "
                f"{'N/A':>8} "
                f"{'UNAVAILABLE':>14}"
            )

            continue

        flagged = []
        retained = []

        missing_early_value = 0

        for trade in trades:

            adverse = early_adverse_value(
                trade,
                window
            )

            if adverse is None:
                missing_early_value += 1
                retained.append(trade)
                continue

            if adverse >= threshold:
                flagged.append(trade)
            else:
                retained.append(trade)

        metrics = calculate_metrics(retained)

        delta_r = (
            metrics["total_r"]
            - baseline["total_r"]
        )

        if delta_r > 0:
            status = "IMPROVED"
        elif delta_r < 0:
            status = "WORSE"
        else:
            status = "UNCHANGED"

        row = {
            "window_bars": window,
            "threshold_r": threshold,
            "status": status,
            "reason": "",
            "flagged_trades": len(flagged),
            "retained_trades": metrics["trades"],
            "wins": metrics["wins"],
            "win_rate": metrics["win_rate"],
            "total_r": metrics["total_r"],
            "avg_r": metrics["avg_r"],
            "profit_factor": metrics["profit_factor"],
            "delta_r": delta_r,
            "missing_early_value": missing_early_value,
        }

        results.append(row)

        print(
            f"{window:>8} "
            f"{threshold:>11.2f} "
            f"{len(flagged):>9} "
            f"{metrics['trades']:>9} "
            f"{metrics['win_rate']:>8.2f}% "
            f"{metrics['total_r']:>10.2f} "
            f"{delta_r:>10.2f} "
            f"{metrics['profit_factor']:>8.3f} "
            f"{status:>14}"
        )


# ---------------------------------------------------------------------
# YEAR-BY-YEAR ROBUSTNESS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("YEAR-BY-YEAR ROBUSTNESS")
print("=" * 110)

years = sorted(
    {
        classify_year(t.get("entry_date"))
        for t in trades
        if classify_year(t.get("entry_date")) is not None
    }
)

print(
    "Years analysed:",
    ", ".join(str(y) for y in years)
)


year_rows = []


for window in WINDOWS:

    if window > 5:
        continue

    for threshold in THRESHOLDS:

        improved_years = 0
        worsened_years = 0
        unchanged_years = 0

        per_year = {}

        for year in years:

            year_trades = [
                t
                for t in trades
                if classify_year(t.get("entry_date")) == year
            ]

            year_retained = []

            for trade in year_trades:

                adverse = early_adverse_value(
                    trade,
                    window
                )

                if adverse is None:
                    year_retained.append(trade)
                    continue

                if adverse < threshold:
                    year_retained.append(trade)

            baseline_year_r = sum(
                safe_float(t.get("r_multiple")) or 0.0
                for t in year_trades
            )

            filtered_year_r = sum(
                safe_float(t.get("r_multiple")) or 0.0
                for t in year_retained
            )

            delta = (
                filtered_year_r
                - baseline_year_r
            )

            if delta > 0:
                improved_years += 1
            elif delta < 0:
                worsened_years += 1
            else:
                unchanged_years += 1

            per_year[year] = delta

        year_rows.append(
            {
                "window_bars": window,
                "threshold_r": threshold,
                "improved_years": improved_years,
                "worsened_years": worsened_years,
                "unchanged_years": unchanged_years,
                "year_deltas": per_year,
            }
        )


print()
print(
    f"{'Window':>8} "
    f"{'Threshold':>11} "
    f"{'Improved':>10} "
    f"{'Worsened':>10} "
    f"{'Unchanged':>10}"
)

print("-" * 65)


for row in year_rows:

    print(
        f"{row['window_bars']:>8} "
        f"{row['threshold_r']:>11.2f} "
        f"{row['improved_years']:>10} "
        f"{row['worsened_years']:>10} "
        f"{row['unchanged_years']:>10}"
    )


# ---------------------------------------------------------------------
# ROBUSTNESS REGIONS
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("ROBUSTNESS REGION SUMMARY")
print("=" * 110)

valid_results = [
    r
    for r in results
    if r.get("status") != "UNAVAILABLE"
]

positive_results = [
    r
    for r in valid_results
    if r["delta_r"] > 0
]

print(
    f"Valid matrix cells       : {len(valid_results)}"
)

print(
    f"Improving cells          : {len(positive_results)}"
)

if valid_results:

    best_delta = max(
        r["delta_r"]
        for r in valid_results
    )

    worst_delta = min(
        r["delta_r"]
        for r in valid_results
    )

    print(
        f"Best ΔR                 : {best_delta:.2f}R"
    )

    print(
        f"Worst ΔR                : {worst_delta:.2f}R"
    )


# ---------------------------------------------------------------------
# NEIGHBOUR CONSISTENCY
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("NEIGHBOUR CONSISTENCY CHECK")
print("=" * 110)

for window in [3, 5]:

    window_results = [
        r
        for r in valid_results
        if r["window_bars"] == window
    ]

    positive = sum(
        1
        for r in window_results
        if r["delta_r"] > 0
    )

    print()
    print(
        f"Window {window} bars:"
    )

    print(
        f"  Positive threshold cells : "
        f"{positive}/{len(window_results)}"
    )

    for r in window_results:

        print(
            f"  {r['threshold_r']:.2f}R "
            f"→ ΔR {r['delta_r']:+.2f}R "
            f"| {r['status']}"
        )


# ---------------------------------------------------------------------
# SAVE MATRIX
# ---------------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

fieldnames = [
    "window_bars",
    "threshold_r",
    "status",
    "reason",
    "flagged_trades",
    "retained_trades",
    "wins",
    "win_rate",
    "total_r",
    "avg_r",
    "profit_factor",
    "delta_r",
    "missing_early_value",
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in results:

        writer.writerow(row)


# ---------------------------------------------------------------------
# FINAL CONCLUSION
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("FORENSIC COMPLETE")
print("=" * 110)

print()
print(
    f"Saved to: {OUTPUT_FILE}"
)

print()
print("IMPORTANT:")
print(
    "1. This is RESEARCH ONLY."
)

print(
    "2. Production strategy was NOT modified."
)

print(
    "3. Frozen historical dataset was NOT modified."
)

print(
    "4. Holdout dataset was NOT modified."
)

print(
    "5. Windows >5 bars are intentionally unavailable because "
    "the frozen trade-level dataset stores early adverse values "
    "only through bar 5."
)

print(
    "6. No threshold was selected for deployment."
)

print()