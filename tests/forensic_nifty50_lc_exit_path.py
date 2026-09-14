"""
NIFTY 50 LONG_CONTINUATION EXIT-PATH FORENSIC
==============================================

Purpose
-------
Descriptive forensic analysis of the 12 NIFTY 50 LONG_CONTINUATION
trades produced by the research backtest.

This script DOES NOT:
- modify production code
- modify frozen historical trades
- modify holdout trades
- optimize thresholds
- select a new exit strategy
- alter any trading logic

Research question
-----------------
For each LONG_CONTINUATION trade, reconstruct the observed path:

    Entry
      |
      +--> 0.5R
      |
      +--> 1.0R
      |
      +--> 1.5R
      |
      +--> MFE
      |
      +--> MAE
      |
      +--> Exit

The main question is:

    How many LC losers developed enough favorable movement
    that exit architecture could theoretically matter?

Milestone classification
------------------------
1. TARGET / 2R reached
2. >1.5R but <2R
3. 1R to <1.5R
4. 0.5R to <1R
5. <0.5R

Important:
---------
TARGET is determined from the actual stored exit_reason.
max_favorable_r is used for the non-target milestone classification.

This is descriptive only.
"""

import csv
import os
import math
from collections import Counter, defaultdict


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = "tests/output/index_backtests/nifty50_index_trades.csv"

OUTPUT_DIR = "tests/output/index_backtests"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_path.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_path_report.txt"
)

EXPECTED_LC_TRADES = 12

MILESTONE_TOLERANCE = 1e-9


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value, default=None):
    """
    Safely convert a CSV value to float.
    """
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    try:
        number = float(text)

        if math.isnan(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    """
    Safely convert a CSV value to int.
    """
    number = safe_float(value, default=None)

    if number is None:
        return default

    try:
        return int(number)

    except (TypeError, ValueError):
        return default


def clean_text(value):
    """
    Normalize text fields.
    """
    if value is None:
        return ""

    return str(value).strip()


def format_number(value, decimals=2):
    """
    Human-readable number formatting.
    """
    if value is None:
        return "N/A"

    return f"{value:.{decimals}f}"


def pct(value):
    """
    Format decimal percentage.
    """
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def mean(values):
    """
    Simple arithmetic mean.
    """
    if not values:
        return None

    return sum(values) / len(values)


def median(values):
    """
    Simple median.
    """
    if not values:
        return None

    ordered = sorted(values)

    n = len(ordered)

    middle = n // 2

    if n % 2 == 1:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / 2


def get_first_numeric(row, names, default=None):
    """
    Return the first usable numeric value among possible field names.
    """
    for name in names:
        if name in row:
            value = safe_float(row.get(name))

            if value is not None:
                return value

    return default


# ============================================================================
# LOAD DATA
# ============================================================================

def load_lc_trades():
    """
    Load the NIFTY research trade file and retain LONG_CONTINUATION trades.
    """

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    lc_trades = []

    for row in rows:

        setup = clean_text(
            row.get("setup", "")
        )

        if setup != "LONG_CONTINUATION":
            continue

        lc_trades.append(row)

    return lc_trades


# ============================================================================
# MILESTONE CLASSIFICATION
# ============================================================================

def classify_path(row):
    """
    Classify the maximum favorable path.

    Actual TARGET exit is authoritative for 2R achievement.

    For non-target trades:
        >=1.5R -> >1.5R but <2R
        >=1.0R -> 1R to <1.5R
        >=0.5R -> 0.5R to <1R
        else    -> <0.5R
    """

    exit_reason = clean_text(
        row.get("exit_reason", "")
    ).upper()

    mfe = get_first_numeric(
        row,
        [
            "max_favorable_r",
            "mfe"
        ]
    )

    if exit_reason == "TARGET":
        return "TARGET_2R"

    if mfe is None:
        return "UNKNOWN_MFE"

    if mfe >= 1.5 - MILESTONE_TOLERANCE:
        return "OVER_1_5R_UNDER_2R"

    if mfe >= 1.0 - MILESTONE_TOLERANCE:
        return "ONE_TO_UNDER_1_5R"

    if mfe >= 0.5 - MILESTONE_TOLERANCE:
        return "HALF_TO_UNDER_1R"

    return "UNDER_0_5R"


# ============================================================================
# BUILD FORENSIC RECORD
# ============================================================================

def build_record(row, sequence):
    """
    Convert raw trade row into a compact forensic record.
    """

    entry_date = clean_text(
        row.get("entry_date", "")
    )

    exit_date = clean_text(
        row.get("exit_date", "")
    )

    direction = clean_text(
        row.get("direction", "")
    )

    setup = clean_text(
        row.get("setup", "")
    )

    exit_reason = clean_text(
        row.get("exit_reason", "")
    )

    r_multiple = get_first_numeric(
        row,
        [
            "r_multiple"
        ],
        default=0.0
    )

    mfe = get_first_numeric(
        row,
        [
            "max_favorable_r",
            "mfe"
        ]
    )

    mae = get_first_numeric(
        row,
        [
            "max_adverse_r",
            "mae"
        ]
    )

    bars_in_trade = get_first_numeric(
        row,
        [
            "bars_in_trade"
        ]
    )

    bars_to_half = get_first_numeric(
        row,
        [
            "bars_to_0_5r",
            "bars_to_half_r"
        ]
    )

    bars_to_one = get_first_numeric(
        row,
        [
            "bars_to_1r"
        ]
    )

    bars_to_one_half = get_first_numeric(
        row,
        [
            "bars_to_1_5r"
        ]
    )

    early_adverse_5 = get_first_numeric(
        row,
        [
            "early_adverse_r_bar_5"
        ]
    )

    entry_price = get_first_numeric(
        row,
        [
            "entry_price"
        ]
    )

    initial_risk = get_first_numeric(
        row,
        [
            "initial_risk"
        ]
    )

    record = {
        "sequence": sequence,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": direction,
        "setup": setup,
        "exit_reason": exit_reason,
        "r_multiple": r_multiple,
        "max_favorable_r": mfe,
        "max_adverse_r": mae,
        "bars_in_trade": bars_in_trade,
        "bars_to_0_5r": bars_to_half,
        "bars_to_1r": bars_to_one,
        "bars_to_1_5r": bars_to_one_half,
        "early_adverse_r_bar_5": early_adverse_5,
        "entry_price": entry_price,
        "initial_risk": initial_risk,
        "path_class": classify_path(row),
    }

    return record


# ============================================================================
# SUMMARY HELPERS
# ============================================================================

def summarize_group(records):
    """
    Calculate descriptive statistics for a group.
    """

    if not records:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_r": 0.0,
            "avg_r": None,
            "avg_mfe": None,
            "avg_mae": None,
            "avg_bars": None,
        }

    r_values = [
        r["r_multiple"]
        for r in records
        if r["r_multiple"] is not None
    ]

    mfe_values = [
        r["max_favorable_r"]
        for r in records
        if r["max_favorable_r"] is not None
    ]

    mae_values = [
        r["max_adverse_r"]
        for r in records
        if r["max_adverse_r"] is not None
    ]

    bar_values = [
        r["bars_in_trade"]
        for r in records
        if r["bars_in_trade"] is not None
    ]

    wins = sum(
        1
        for r in records
        if r["r_multiple"] > 0
    )

    losses = sum(
        1
        for r in records
        if r["r_multiple"] <= 0
    )

    return {
        "count": len(records),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(records),
        "total_r": sum(r_values),
        "avg_r": mean(r_values),
        "avg_mfe": mean(mfe_values),
        "avg_mae": mean(mae_values),
        "avg_bars": mean(bar_values),
    }


# ============================================================================
# MILESTONE ANALYSIS
# ============================================================================

def milestone_counts(records):
    """
    Count actual MFE milestone achievement.
    """

    milestones = {
        "0.5R": 0,
        "1.0R": 0,
        "1.5R": 0,
        "2.0R_TARGET": 0,
    }

    for record in records:

        mfe = record["max_favorable_r"]

        exit_reason = record["exit_reason"].upper()

        if mfe is not None:

            if mfe >= 0.5:
                milestones["0.5R"] += 1

            if mfe >= 1.0:
                milestones["1.0R"] += 1

            if mfe >= 1.5:
                milestones["1.5R"] += 1

        if exit_reason == "TARGET":
            milestones["2.0R_TARGET"] += 1

    return milestones


# ============================================================================
# LOSER DEVELOPMENT ANALYSIS
# ============================================================================

def analyze_loser_development(records):
    """
    Focus specifically on LC losers that nevertheless developed favorably.
    """

    losers = [
        r
        for r in records
        if r["r_multiple"] <= 0
    ]

    categories = {
        "never_0_5R": [],
        "reached_0_5R_not_1R": [],
        "reached_1R_not_1_5R": [],
        "reached_1_5R": [],
    }

    for record in losers:

        mfe = record["max_favorable_r"]

        if mfe is None:
            continue

        if mfe >= 1.5:
            categories["reached_1_5R"].append(record)

        elif mfe >= 1.0:
            categories["reached_1R_not_1_5R"].append(record)

        elif mfe >= 0.5:
            categories["reached_0_5R_not_1R"].append(record)

        else:
            categories["never_0_5R"].append(record)

    return categories


# ============================================================================
# PRINT TRADE TABLE
# ============================================================================

def print_trade_table(records, output_lines):
    """
    Print the individual LC path table.
    """

    header = (
        "SEQ  ENTRY       EXIT        RESULT    "
        "MFE     MAE     0.5R  1R   1.5R  "
        "BARS   EXIT          PATH"
    )

    separator = "-" * len(header)

    print()
    print("INDIVIDUAL LC EXIT PATH")
    print(separator)
    print(header)
    print(separator)

    output_lines.append("")
    output_lines.append("INDIVIDUAL LC EXIT PATH")
    output_lines.append(separator)
    output_lines.append(header)
    output_lines.append(separator)

    for record in records:

        r_text = format_number(
            record["r_multiple"]
        )

        mfe_text = format_number(
            record["max_favorable_r"]
        )

        mae_text = format_number(
            record["max_adverse_r"]
        )

        half_text = (
            str(safe_int(record["bars_to_0_5r"]))
            if record["bars_to_0_5r"] is not None
            else "-"
        )

        one_text = (
            str(safe_int(record["bars_to_1r"]))
            if record["bars_to_1r"] is not None
            else "-"
        )

        one_half_text = (
            str(safe_int(record["bars_to_1_5r"]))
            if record["bars_to_1_5r"] is not None
            else "-"
        )

        bars_text = (
            str(safe_int(record["bars_in_trade"]))
            if record["bars_in_trade"] is not None
            else "-"
        )

        line = (
            f"{record['sequence']:>3}  "
            f"{record['entry_date'][:10]:<10}  "
            f"{record['exit_date'][:10]:<10}  "
            f"{r_text:>7}  "
            f"{mfe_text:>6}  "
            f"{mae_text:>6}  "
            f"{half_text:>4}  "
            f"{one_text:>3}  "
            f"{one_half_text:>5}  "
            f"{bars_text:>5}  "
            f"{record['exit_reason']:<12}  "
            f"{record['path_class']}"
        )

        print(line)
        output_lines.append(line)

    print(separator)
    output_lines.append(separator)


# ============================================================================
# MAIN
# ============================================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 100)
    print("NIFTY 50 LONG_CONTINUATION EXIT-PATH FORENSIC")
    print("=" * 100)

    lc_rows = load_lc_trades()

    print(
        f"Input file:              {INPUT_FILE}"
    )

    print(
        f"LONG_CONTINUATION trades: {len(lc_rows)}"
    )

    if len(lc_rows) != EXPECTED_LC_TRADES:

        print()
        print(
            "WARNING: Expected "
            f"{EXPECTED_LC_TRADES} LC trades, "
            f"but found {len(lc_rows)}."
        )

    records = []

    for sequence, row in enumerate(
        lc_rows,
        start=1
    ):

        records.append(
            build_record(
                row,
                sequence
            )
        )

    # Sort chronologically for the final forensic view.
    records.sort(
        key=lambda r: r["entry_date"]
    )

    # Re-number after sorting.
    for sequence, record in enumerate(
        records,
        start=1
    ):
        record["sequence"] = sequence

    # ========================================================================
    # BASIC SUMMARY
    # ========================================================================

    overall = summarize_group(records)

    print()
    print("=" * 100)
    print("BASIC LC SUMMARY")
    print("=" * 100)

    print(
        f"Trades:                 {overall['count']}"
    )

    print(
        f"Winners:                {overall['wins']}"
    )

    print(
        f"Losers:                 {overall['losses']}"
    )

    print(
        f"Win Rate:               {pct(overall['win_rate'])}"
    )

    print(
        f"Total R:                {format_number(overall['total_r'])}"
    )

    print(
        f"Average R:              {format_number(overall['avg_r'])}"
    )

    print(
        f"Average MFE:             {format_number(overall['avg_mfe'])}R"
    )

    print(
        f"Average MAE:             {format_number(overall['avg_mae'])}R"
    )

    print(
        f"Average Bars:            {format_number(overall['avg_bars'])}"
    )

    # ========================================================================
    # MILESTONES
    # ========================================================================

    milestones = milestone_counts(records)

    print()
    print("=" * 100)
    print("MFE MILESTONE ACHIEVEMENT")
    print("=" * 100)

    for name, count in milestones.items():

        print(
            f"{name:<15}: "
            f"{count:>2}/{len(records):<2} "
            f"({count / len(records) * 100:.2f}%)"
            if records
            else f"{name:<15}: 0"
        )

    # ========================================================================
    # PATH CLASSIFICATION
    # ========================================================================

    grouped = defaultdict(list)

    for record in records:
        grouped[
            record["path_class"]
        ].append(record)

    path_order = [
        "UNDER_0_5R",
        "HALF_TO_UNDER_1R",
        "ONE_TO_UNDER_1_5R",
        "OVER_1_5R_UNDER_2R",
        "TARGET_2R",
        "UNKNOWN_MFE",
    ]

    print()
    print("=" * 100)
    print("EXIT-PATH CLASSIFICATION")
    print("=" * 100)

    print(
        f"{'PATH':<28}"
        f"{'N':>5}"
        f"{'W':>5}"
        f"{'L':>5}"
        f"{'WIN%':>9}"
        f"{'TOTAL R':>12}"
        f"{'AVG MFE':>10}"
        f"{'AVG MAE':>10}"
    )

    print("-" * 100)

    for path in path_order:

        group = grouped.get(
            path,
            []
        )

        if not group:
            continue

        summary = summarize_group(
            group
        )

        print(
            f"{path:<28}"
            f"{summary['count']:>5}"
            f"{summary['wins']:>5}"
            f"{summary['losses']:>5}"
            f"{pct(summary['win_rate']):>9}"
            f"{format_number(summary['total_r']):>12}"
            f"{format_number(summary['avg_mfe']):>10}"
            f"{format_number(summary['avg_mae']):>10}"
        )

    # ========================================================================
    # LOSER DEVELOPMENT
    # ========================================================================

    loser_categories = analyze_loser_development(
        records
    )

    print()
    print("=" * 100)
    print("LOSER DEVELOPMENT")
    print("=" * 100)

    for category, group in loser_categories.items():

        summary = summarize_group(
            group
        )

        print(
            f"{category:<28}"
            f"N={summary['count']:>2}  "
            f"Total R={format_number(summary['total_r'])}  "
            f"Avg MFE={format_number(summary['avg_mfe'])}R  "
            f"Avg MAE={format_number(summary['avg_mae'])}R"
        )

    # ========================================================================
    # POTENTIAL EXIT-ARCHITECTURE CASES
    # ========================================================================

    developed_losers = [
        r
        for r in records
        if r["r_multiple"] <= 0
        and r["max_favorable_r"] is not None
        and r["max_favorable_r"] >= 1.0
    ]

    strongly_developed_losers = [
        r
        for r in records
        if r["r_multiple"] <= 0
        and r["max_favorable_r"] is not None
        and r["max_favorable_r"] >= 1.5
    ]

    print()
    print("=" * 100)
    print("EXIT-ARCHITECTURE RESEARCH FLAGS")
    print("=" * 100)

    print(
        "Losers reaching >=1.0R:   "
        f"{len(developed_losers)}"
    )

    print(
        "Losers reaching >=1.5R:   "
        f"{len(strongly_developed_losers)}"
    )

    if strongly_developed_losers:

        print()
        print(
            "Losers reaching >=1.5R:"
        )

        for record in strongly_developed_losers:

            print(
                f"  {record['entry_date'][:10]} | "
                f"R={format_number(record['r_multiple'])} | "
                f"MFE={format_number(record['max_favorable_r'])}R | "
                f"MAE={format_number(record['max_adverse_r'])}R | "
                f"Exit={record['exit_reason']}"
            )

    # ========================================================================
    # WINNERS VS LOSERS
    # ========================================================================

    winners = [
        r
        for r in records
        if r["r_multiple"] > 0
    ]

    losers = [
        r
        for r in records
        if r["r_multiple"] <= 0
    ]

    winner_summary = summarize_group(
        winners
    )

    loser_summary = summarize_group(
        losers
    )

    print()
    print("=" * 100)
    print("WINNERS VS LOSERS")
    print("=" * 100)

    print(
        f"{'Metric':<24}"
        f"{'Winners':>15}"
        f"{'Losers':>15}"
    )

    print("-" * 60)

    comparisons = [
        (
            "Count",
            winner_summary["count"],
            loser_summary["count"]
        ),
        (
            "Total R",
            winner_summary["total_r"],
            loser_summary["total_r"]
        ),
        (
            "Avg R",
            winner_summary["avg_r"],
            loser_summary["avg_r"]
        ),
        (
            "Avg MFE (R)",
            winner_summary["avg_mfe"],
            loser_summary["avg_mfe"]
        ),
        (
            "Avg MAE (R)",
            winner_summary["avg_mae"],
            loser_summary["avg_mae"]
        ),
        (
            "Avg Bars",
            winner_summary["avg_bars"],
            loser_summary["avg_bars"]
        ),
    ]

    for name, winner_value, loser_value in comparisons:

        print(
            f"{name:<24}"
            f"{format_number(winner_value):>15}"
            f"{format_number(loser_value):>15}"
        )

    # ========================================================================
    # INDIVIDUAL TABLE
    # ========================================================================

    report_lines = []

    report_lines.append(
        "NIFTY 50 LONG_CONTINUATION EXIT-PATH FORENSIC"
    )

    report_lines.append(
        "=" * 100
    )

    report_lines.append(
        f"Input: {INPUT_FILE}"
    )

    report_lines.append(
        f"LC trades: {len(records)}"
    )

    report_lines.append("")

    report_lines.append(
        "BASIC LC SUMMARY"
    )

    report_lines.append(
        f"Trades: {overall['count']}"
    )

    report_lines.append(
        f"Winners: {overall['wins']}"
    )

    report_lines.append(
        f"Losers: {overall['losses']}"
    )

    report_lines.append(
        f"Win Rate: {pct(overall['win_rate'])}"
    )

    report_lines.append(
        f"Total R: {format_number(overall['total_r'])}"
    )

    report_lines.append(
        f"Average R: {format_number(overall['avg_r'])}"
    )

    report_lines.append(
        f"Average MFE: {format_number(overall['avg_mfe'])}R"
    )

    report_lines.append(
        f"Average MAE: {format_number(overall['avg_mae'])}R"
    )

    report_lines.append(
        f"Average Bars: {format_number(overall['avg_bars'])}"
    )

    report_lines.append("")

    report_lines.append(
        "MFE MILESTONES"
    )

    for name, count in milestones.items():

        report_lines.append(
            f"{name}: {count}/{len(records)}"
        )

    print_trade_table(
        records,
        report_lines
    )

    # ========================================================================
    # RESEARCH VERDICT
    # ========================================================================

    print()
    print("=" * 100)
    print("RESEARCH VERDICT")
    print("=" * 100)

    if strongly_developed_losers:

        verdict = (
            "EXIT_PATH_WORTH_ONE_CONTROLLED_COUNTERFACTUAL"
        )

        explanation = (
            "At least one LC loser reached >=1.5R before ultimately "
            "losing. This means exit architecture is worth one "
            "controlled research counterfactual. This is NOT evidence "
            "for changing production exits."
        )

    elif developed_losers:

        verdict = (
            "LIMITED_EXIT_PATH_SIGNAL"
        )

        explanation = (
            "Some LC losers reached >=1.0R, but none reached >=1.5R. "
            "There may be limited exit-path information, but the "
            "evidence is not strong enough by itself for an exit change."
        )

    else:

        verdict = (
            "NO_MEANINGFUL_LATE_DEVELOPMENT_PATTERN"
        )

        explanation = (
            "LC losers generally failed to develop meaningful favorable "
            "movement. Exit architecture is unlikely to be the primary "
            "issue based on this small sample."
        )

    print(
        f"VERDICT: {verdict}"
    )

    print()
    print(
        explanation
    )

    print()
    print(
        "IMPORTANT: This analysis is descriptive and in-sample."
    )

    print(
        "No production strategy, frozen dataset, or holdout dataset "
        "was modified."
    )

    report_lines.append("")
    report_lines.append(
        f"RESEARCH VERDICT: {verdict}"
    )
    report_lines.append(
        explanation
    )
    report_lines.append("")
    report_lines.append(
        "IMPORTANT: This analysis is descriptive and in-sample."
    )
    report_lines.append(
        "No production strategy, frozen dataset, or holdout dataset "
        "was modified."
    )

    # ========================================================================
    # SAVE CSV
    # ========================================================================

    fieldnames = [
        "sequence",
        "entry_date",
        "exit_date",
        "direction",
        "setup",
        "exit_reason",
        "r_multiple",
        "max_favorable_r",
        "max_adverse_r",
        "bars_in_trade",
        "bars_to_0_5r",
        "bars_to_1r",
        "bars_to_1_5r",
        "early_adverse_r_bar_5",
        "entry_price",
        "initial_risk",
        "path_class",
    ]

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

        writer.writerows(records)

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        f"CSV:    {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print()
    print(
        "STATUS: PASS"
    )


if __name__ == "__main__":
    main()