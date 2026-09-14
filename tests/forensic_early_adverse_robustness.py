"""
TradeSense-AI
EARLY-ADVERSE ROBUSTNESS FORENSIC

RESEARCH ONLY.
DO NOT DEPLOY.

Purpose
-------
Analyze the validated OHLC-reconstructed early-adverse excursion
data produced by:

    tests.forensic_early_adverse_ohlc_replay

This script does NOT modify:
    - production code
    - frozen historical trades
    - holdout trades

Robustness tests:
    1. Parameter neighbourhood
    2. Year robustness
    3. Symbol robustness
    4. Leave-one-out robustness
"""

import csv
import os
from collections import defaultdict


# =====================================================================
# CONFIGURATION
# =====================================================================

HISTORICAL_TRADES_FILE = (
    "tests/output/tradesense_trades.csv"
)

BAR_REPLAY_FILE = (
    "tests/output/reconciliation/"
    "early_adverse_ohlc_replay_bars.csv"
)

OUTPUT_DIR = (
    "tests/output/reconciliation"
)

ROBUSTNESS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_robustness.csv",
)

SYMBOL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_symbol_robustness.csv",
)

YEAR_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_year_robustness.csv",
)

LOO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "early_adverse_leave_one_out.csv",
)

WINDOWS = [3, 5, 7, 10]

THRESHOLDS = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]

EXPECTED_TRADE_COUNT = 106

TOLERANCE = 1e-6


# =====================================================================
# HELPERS
# =====================================================================

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


def normalize_date(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text[:10]


def normalize_symbol(value):
    if value is None:
        return None

    return str(value).strip()


def trade_key(trade):
    return (
        normalize_symbol(
            trade.get("_symbol")
        ),
        normalize_date(
            trade.get("entry_date")
        ),
        str(
            trade.get("direction", "")
        ).strip().upper(),
    )


def calculate_profit_factor(trades):
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:
        r = safe_float(
            trade.get("r_multiple")
        )

        if r is None:
            continue

        if r > 0:
            gross_profit += r

        elif r < 0:
            gross_loss += abs(r)

    if gross_loss == 0:

        if gross_profit > 0:
            return float("inf")

        return 0.0

    return gross_profit / gross_loss


def calculate_metrics(trades):

    r_values = []

    for trade in trades:

        r = safe_float(
            trade.get("r_multiple")
        )

        if r is not None:
            r_values.append(r)

    if not r_values:

        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
        }

    wins = sum(
        1
        for r in r_values
        if r > 0
    )

    total_r = sum(r_values)

    return {
        "trades": len(r_values),
        "wins": wins,
        "win_rate": (
            wins
            / len(r_values)
            * 100.0
        ),
        "total_r": total_r,
        "avg_r": (
            total_r
            / len(r_values)
        ),
        "profit_factor": (
            calculate_profit_factor(
                trades
            )
        ),
    }


# =====================================================================
# LOAD DATA
# =====================================================================

print()
print("=" * 115)
print("TRADESENSE-AI — EARLY-ADVERSE ROBUSTNESS FORENSIC")
print("=" * 115)
print()

print("RESEARCH ONLY — DO NOT DEPLOY")
print()


with open(
    HISTORICAL_TRADES_FILE,
    "r",
    newline="",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    trades = list(reader)


print(
    f"Frozen trades loaded: {len(trades)}"
)


if len(trades) != EXPECTED_TRADE_COUNT:

    raise RuntimeError(
        "Frozen historical trade count "
        "is not 106. "
        "Stopping to protect integrity."
    )


with open(
    BAR_REPLAY_FILE,
    "r",
    newline="",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    bar_rows = list(reader)


print(
    f"OHLC replay rows loaded: {len(bar_rows)}"
)


if not bar_rows:

    raise RuntimeError(
        "OHLC replay file is empty."
    )


# =====================================================================
# BUILD RECONSTRUCTED EARLY-ADVERSE MAP
# =====================================================================

early_adverse_by_trade = defaultdict(dict)

for row in bar_rows:

    key = (
        normalize_symbol(
            row.get("_symbol")
        ),
        normalize_date(
            row.get("entry_date")
        ),
        str(
            row.get("direction", "")
        ).strip().upper(),
    )

    bar_number = int(
        row["bar_number"]
    )

    cumulative_max_adverse_r = (
        safe_float(
            row[
                "cumulative_max_adverse_r"
            ]
        )
    )

    if cumulative_max_adverse_r is None:
        continue

    early_adverse_by_trade[key][
        bar_number
    ] = cumulative_max_adverse_r


print(
    f"Reconstructed trade keys: "
    f"{len(early_adverse_by_trade)}"
)


if len(early_adverse_by_trade) != EXPECTED_TRADE_COUNT:

    raise RuntimeError(
        "OHLC replay does not contain "
        "all 106 reconstructed trades."
    )


# =====================================================================
# BASELINE
# =====================================================================

baseline_metrics = calculate_metrics(
    trades
)

baseline_total_r = (
    baseline_metrics["total_r"]
)


print()
print("=" * 115)
print("BASELINE")
print("=" * 115)
print()

print(
    f"Trades       : "
    f"{baseline_metrics['trades']}"
)

print(
    f"Win rate     : "
    f"{baseline_metrics['win_rate']:.2f}%"
)

print(
    f"Total R      : "
    f"{baseline_metrics['total_r']:.2f}"
)

print(
    f"Profit factor: "
    f"{baseline_metrics['profit_factor']:.3f}"
)


# =====================================================================
# PARAMETER ROBUSTNESS
# =====================================================================

print()
print("=" * 115)
print("PARAMETER NEIGHBOURHOOD ROBUSTNESS")
print("=" * 115)
print()

print(
    f"{'Window':>8} "
    f"{'Threshold':>11} "
    f"{'Flagged':>9} "
    f"{'Retained':>10} "
    f"{'Win%':>9} "
    f"{'TotalR':>11} "
    f"{'DeltaR':>11} "
    f"{'PF':>8}"
)

print("-" * 90)


parameter_rows = []


for window in WINDOWS:

    for threshold in THRESHOLDS:

        retained = []
        flagged = []

        for trade in trades:

            key = trade_key(trade)

            per_bar = (
                early_adverse_by_trade
                .get(key, {})
            )

            if window not in per_bar:

                raise RuntimeError(
                    f"Missing reconstructed "
                    f"bar {window} for {key}"
                )

            early_adverse = (
                per_bar[window]
            )

            if early_adverse >= threshold:

                flagged.append(trade)

            else:

                retained.append(trade)

        metrics = calculate_metrics(
            retained
        )

        delta_r = (
            metrics["total_r"]
            - baseline_total_r
        )

        if delta_r > TOLERANCE:
            status = "IMPROVED"

        elif delta_r < -TOLERANCE:
            status = "WORSE"

        else:
            status = "UNCHANGED"

        parameter_rows.append(
            {
                "window_bars": window,
                "threshold_r": threshold,
                "flagged_trades": len(
                    flagged
                ),
                "retained_trades": len(
                    retained
                ),
                "win_rate": metrics[
                    "win_rate"
                ],
                "total_r": metrics[
                    "total_r"
                ],
                "delta_r": delta_r,
                "profit_factor": metrics[
                    "profit_factor"
                ],
                "status": status,
            }
        )

        pf_text = (
            "INF"
            if metrics["profit_factor"]
            == float("inf")
            else f"{metrics['profit_factor']:.3f}"
        )

        print(
            f"{window:>8} "
            f"{threshold:>11.2f} "
            f"{len(flagged):>9} "
            f"{len(retained):>10} "
            f"{metrics['win_rate']:>8.2f}% "
            f"{metrics['total_r']:>10.2f} "
            f"{delta_r:>10.2f} "
            f"{pf_text:>8}"
        )


# =====================================================================
# SYMBOL ROBUSTNESS
# =====================================================================

BEST_WINDOW = 7
BEST_THRESHOLD = 0.25


print()
print("=" * 115)
print("SYMBOL ROBUSTNESS — 7 BAR / 0.25R")
print("=" * 115)
print()


symbol_rows = []

symbols = sorted(
    {
        normalize_symbol(
            trade.get("_symbol")
        )
        for trade in trades
    }
)


for symbol in symbols:

    symbol_trades = [
        trade
        for trade in trades
        if normalize_symbol(
            trade.get("_symbol")
        ) == symbol
    ]

    baseline = calculate_metrics(
        symbol_trades
    )

    retained = []

    for trade in symbol_trades:

        key = trade_key(trade)

        early_adverse = (
            early_adverse_by_trade[key][
                BEST_WINDOW
            ]
        )

        if early_adverse < BEST_THRESHOLD:

            retained.append(trade)

    filtered = calculate_metrics(
        retained
    )

    delta_r = (
        filtered["total_r"]
        - baseline["total_r"]
    )

    if delta_r > TOLERANCE:
        status = "IMPROVED"

    elif delta_r < -TOLERANCE:
        status = "WORSE"

    else:
        status = "UNCHANGED"

    symbol_rows.append(
        {
            "symbol": symbol,
            "baseline_trades": baseline[
                "trades"
            ],
            "retained_trades": filtered[
                "trades"
            ],
            "baseline_r": baseline[
                "total_r"
            ],
            "filtered_r": filtered[
                "total_r"
            ],
            "delta_r": delta_r,
            "status": status,
        }
    )

    print(
        f"{symbol:15} "
        f"baseline={baseline['total_r']:>8.2f}R "
        f"filtered={filtered['total_r']:>8.2f}R "
        f"delta={delta_r:>8.2f}R "
        f"{status}"
    )


# =====================================================================
# YEAR ROBUSTNESS
# =====================================================================

print()
print("=" * 115)
print("YEAR ROBUSTNESS — 7 BAR / 0.25R")
print("=" * 115)
print()


year_rows = []

years = sorted(
    {
        int(
            normalize_date(
                trade.get("entry_date")
            )[:4]
        )
        for trade in trades
        if normalize_date(
            trade.get("entry_date")
        )
    }
)


for year in years:

    year_trades = [
        trade
        for trade in trades
        if int(
            normalize_date(
                trade.get("entry_date")
            )[:4]
        ) == year
    ]

    baseline = calculate_metrics(
        year_trades
    )

    retained = []

    for trade in year_trades:

        key = trade_key(trade)

        early_adverse = (
            early_adverse_by_trade[key][
                BEST_WINDOW
            ]
        )

        if early_adverse < BEST_THRESHOLD:

            retained.append(trade)

    filtered = calculate_metrics(
        retained
    )

    delta_r = (
        filtered["total_r"]
        - baseline["total_r"]
    )

    if delta_r > TOLERANCE:
        status = "IMPROVED"

    elif delta_r < -TOLERANCE:
        status = "WORSE"

    else:
        status = "UNCHANGED"

    year_rows.append(
        {
            "year": year,
            "baseline_trades": baseline[
                "trades"
            ],
            "retained_trades": filtered[
                "trades"
            ],
            "baseline_r": baseline[
                "total_r"
            ],
            "filtered_r": filtered[
                "total_r"
            ],
            "delta_r": delta_r,
            "status": status,
        }
    )

    print(
        f"{year}: "
        f"baseline={baseline['total_r']:>8.2f}R "
        f"filtered={filtered['total_r']:>8.2f}R "
        f"delta={delta_r:>8.2f}R "
        f"{status}"
    )


# =====================================================================
# LEAVE-ONE-OUT ROBUSTNESS
# =====================================================================

print()
print("=" * 115)
print("LEAVE-ONE-OUT ROBUSTNESS — 7 BAR / 0.25R")
print("=" * 115)
print()


loo_rows = []

retained_base = []

for trade in trades:

    key = trade_key(trade)

    early_adverse = (
        early_adverse_by_trade[key][
            BEST_WINDOW
        ]
    )

    if early_adverse < BEST_THRESHOLD:

        retained_base.append(trade)


base_filtered_metrics = calculate_metrics(
    retained_base
)

positive_loo = 0
negative_loo = 0
unchanged_loo = 0


for removed_trade in retained_base:

    reduced = [
        trade
        for trade in retained_base
        if trade is not removed_trade
    ]

    metrics = calculate_metrics(
        reduced
    )

    delta_vs_base = (
        metrics["total_r"]
        - baseline_metrics["total_r"]
    )

    if delta_vs_base > TOLERANCE:

        status = "POSITIVE"

        positive_loo += 1

    elif delta_vs_base < -TOLERANCE:

        status = "NEGATIVE"

        negative_loo += 1

    else:

        status = "UNCHANGED"

        unchanged_loo += 1

    loo_rows.append(
        {
            "removed_symbol": (
                removed_trade.get(
                    "_symbol"
                )
            ),
            "removed_entry_date": (
                removed_trade.get(
                    "entry_date"
                )
            ),
            "removed_direction": (
                removed_trade.get(
                    "direction"
                )
            ),
            "remaining_trades": len(
                reduced
            ),
            "remaining_total_r": (
                metrics["total_r"]
            ),
            "delta_vs_full_baseline": (
                delta_vs_base
            ),
            "status": status,
        }
    )


print(
    f"Retained base trades : "
    f"{len(retained_base)}"
)

print(
    f"Positive LOO         : "
    f"{positive_loo}"
)

print(
    f"Negative LOO         : "
    f"{negative_loo}"
)

print(
    f"Unchanged LOO        : "
    f"{unchanged_loo}"
)


# =====================================================================
# OVERALL ROBUSTNESS ASSESSMENT
# =====================================================================

improved_parameter_cells = sum(
    1
    for row in parameter_rows
    if row["status"] == "IMPROVED"
)

worsened_parameter_cells = sum(
    1
    for row in parameter_rows
    if row["status"] == "WORSE"
)

improved_symbols = sum(
    1
    for row in symbol_rows
    if row["status"] == "IMPROVED"
)

worsened_symbols = sum(
    1
    for row in symbol_rows
    if row["status"] == "WORSE"
)

improved_years = sum(
    1
    for row in year_rows
    if row["status"] == "IMPROVED"
)

worsened_years = sum(
    1
    for row in year_rows
    if row["status"] == "WORSE"
)


# Conservative research classification.
#
# This does NOT mean deployment approval.
#
# We require:
#   - every parameter cell improves
#   - every symbol improves
#   - every year improves
#   - every retained-trade LOO remains above baseline
#
# Otherwise the result remains promising/research-only.

if (
    improved_parameter_cells
    == len(parameter_rows)
    and worsened_parameter_cells == 0
    and improved_symbols == len(symbol_rows)
    and worsened_symbols == 0
    and improved_years == len(year_rows)
    and worsened_years == 0
    and negative_loo == 0
):

    verdict = "STRONG_ROBUSTNESS"

else:

    if (
        improved_parameter_cells
        > worsened_parameter_cells
        and improved_years
        > worsened_years
    ):

        verdict = (
            "PROMISING_BUT_NOT_ROBUST"
        )

    else:

        verdict = "NOT_ROBUST"


# =====================================================================
# SAVE OUTPUTS
# =====================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


with open(
    ROBUSTNESS_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "window_bars",
            "threshold_r",
            "flagged_trades",
            "retained_trades",
            "win_rate",
            "total_r",
            "delta_r",
            "profit_factor",
            "status",
        ],
    )

    writer.writeheader()

    writer.writerows(
        parameter_rows
    )


with open(
    SYMBOL_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "symbol",
            "baseline_trades",
            "retained_trades",
            "baseline_r",
            "filtered_r",
            "delta_r",
            "status",
        ],
    )

    writer.writeheader()

    writer.writerows(
        symbol_rows
    )


with open(
    YEAR_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "year",
            "baseline_trades",
            "retained_trades",
            "baseline_r",
            "filtered_r",
            "delta_r",
            "status",
        ],
    )

    writer.writeheader()

    writer.writerows(
        year_rows
    )


with open(
    LOO_OUTPUT,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "removed_symbol",
            "removed_entry_date",
            "removed_direction",
            "remaining_trades",
            "remaining_total_r",
            "delta_vs_full_baseline",
            "status",
        ],
    )

    writer.writeheader()

    writer.writerows(
        loo_rows
    )


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print()
print("=" * 115)
print("FINAL ROBUSTNESS SUMMARY")
print("=" * 115)
print()

print(
    f"Baseline total R                 : "
    f"{baseline_total_r:.2f}R"
)

print(
    f"Candidate parameter              : "
    f"{BEST_WINDOW} bars / "
    f"{BEST_THRESHOLD:.2f}R"
)

print(
    f"Parameter cells improved         : "
    f"{improved_parameter_cells}/"
    f"{len(parameter_rows)}"
)

print(
    f"Parameter cells worsened         : "
    f"{worsened_parameter_cells}/"
    f"{len(parameter_rows)}"
)

print(
    f"Symbols improved                 : "
    f"{improved_symbols}/"
    f"{len(symbol_rows)}"
)

print(
    f"Symbols worsened                 : "
    f"{worsened_symbols}/"
    f"{len(symbol_rows)}"
)

print(
    f"Years improved                   : "
    f"{improved_years}/"
    f"{len(year_rows)}"
)

print(
    f"Years worsened                   : "
    f"{worsened_years}/"
    f"{len(year_rows)}"
)

print(
    f"LOO negative cases               : "
    f"{negative_loo}"
)

print()

print(
    f"RESEARCH VERDICT                 : "
    f"{verdict}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "This is an in-sample historical "
    "robustness analysis."
)

print(
    "It does NOT authorize deployment."
)

print(
    "The next independent gate is "
    "genuine unseen holdout validation."
)

print()

print(
    "Outputs:"
)

print(
    f"  Parameter: {ROBUSTNESS_OUTPUT}"
)

print(
    f"  Symbol   : {SYMBOL_OUTPUT}"
)

print(
    f"  Year     : {YEAR_OUTPUT}"
)

print(
    f"  LOO      : {LOO_OUTPUT}"
)

print()
print("=" * 115)