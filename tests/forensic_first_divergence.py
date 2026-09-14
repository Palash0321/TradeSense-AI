"""
TradeSense-AI
First-Divergence Forensic Analysis

Purpose:
    Compare the frozen historical trade dataset against the fresh
    research candidate and identify the earliest point where the
    executed trade sequences diverge for each symbol.

This is a DIAGNOSTIC ONLY script.

It does NOT:
    - modify the frozen historical dataset
    - modify the research candidate
    - modify strategy logic
    - modify market structure logic
    - modify any backtest code

Identity:
    _symbol + entry_date + direction
"""

from pathlib import Path
import csv
from collections import defaultdict


# ==============================================================
# FILES
# ==============================================================

HISTORICAL_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

CANDIDATE_FILE = Path(
    "tests/output/tradesense_trades_research_candidate.csv"
)

OUTPUT_DIR = Path(
    "tests/output/reconciliation"
)

OUTPUT_FILE = OUTPUT_DIR / "first_divergence_forensic.csv"


# ==============================================================
# HELPERS
# ==============================================================

def load_trades(file_path):
    """Load trades from CSV and normalize important fields."""

    with open(
        file_path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        trades = []

        for row in reader:

            # Ignore empty rows.
            if not row:
                continue

            # Ignore final liquidation rows because they are not
            # actual generated strategy entries.
            if row.get("exit_reason", "").strip() == "FINAL_LIQUIDATION":
                continue

            symbol = row.get("_symbol", "").strip()
            entry_date = row.get("entry_date", "").strip()
            direction = row.get("direction", "").strip()

            if not symbol or not entry_date or not direction:
                continue

            row["_identity"] = (
                symbol,
                entry_date[:10],
                direction
            )

            row["_symbol_clean"] = symbol
            row["_entry_date_clean"] = entry_date[:10]
            row["_direction_clean"] = direction

            try:
                row["_r_value"] = float(
                    row.get("r_multiple", 0) or 0
                )
            except (TypeError, ValueError):
                row["_r_value"] = 0.0

            trades.append(row)

    return trades


def group_by_symbol(trades):
    """Group trades by symbol and sort chronologically."""

    grouped = defaultdict(list)

    for trade in trades:
        grouped[trade["_symbol_clean"]].append(trade)

    for symbol in grouped:
        grouped[symbol].sort(
            key=lambda x: (
                x["_entry_date_clean"],
                x["_direction_clean"]
            )
        )

    return grouped


def trade_summary(trade):
    """Compact representation of a trade."""

    return (
        f"{trade['_entry_date_clean']} "
        f"{trade['_direction_clean']} "
        f"{trade.get('setup', '')} "
        f"R={trade['_r_value']:.2f}"
    )


# ==============================================================
# LOAD
# ==============================================================

print()
print("=" * 110)
print("TRADESENSE-AI — FIRST DIVERGENCE FORENSIC ANALYSIS")
print("=" * 110)

print()
print(f"Historical file : {HISTORICAL_FILE}")
print(f"Candidate file  : {CANDIDATE_FILE}")

if not HISTORICAL_FILE.exists():
    raise FileNotFoundError(
        f"Historical dataset not found: {HISTORICAL_FILE}"
    )

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Research candidate not found: {CANDIDATE_FILE}"
    )


historical_trades = load_trades(HISTORICAL_FILE)
candidate_trades = load_trades(CANDIDATE_FILE)

historical_by_symbol = group_by_symbol(historical_trades)
candidate_by_symbol = group_by_symbol(candidate_trades)


symbols = sorted(
    set(historical_by_symbol)
    | set(candidate_by_symbol)
)


# ==============================================================
# ANALYSIS
# ==============================================================

results = []


for symbol in symbols:

    historical = historical_by_symbol.get(symbol, [])
    candidate = candidate_by_symbol.get(symbol, [])

    historical_ids = [
        trade["_identity"]
        for trade in historical
    ]

    candidate_ids = [
        trade["_identity"]
        for trade in candidate
    ]

    historical_id_set = set(historical_ids)
    candidate_id_set = set(candidate_ids)

    common_ids = historical_id_set & candidate_id_set
    fresh_only_ids = candidate_id_set - historical_id_set
    historical_only_ids = historical_id_set - candidate_id_set

    # ----------------------------------------------------------
    # Find earliest chronological trade identity where the
    # two sequences are no longer aligned.
    # ----------------------------------------------------------

    max_len = max(
        len(historical),
        len(candidate)
    )

    first_sequence_divergence_index = None
    first_sequence_divergence_historical = None
    first_sequence_divergence_candidate = None

    for index in range(max_len):

        historical_trade = (
            historical[index]
            if index < len(historical)
            else None
        )

        candidate_trade = (
            candidate[index]
            if index < len(candidate)
            else None
        )

        historical_identity = (
            historical_trade["_identity"]
            if historical_trade
            else None
        )

        candidate_identity = (
            candidate_trade["_identity"]
            if candidate_trade
            else None
        )

        if historical_identity != candidate_identity:

            first_sequence_divergence_index = index

            first_sequence_divergence_historical = (
                historical_trade
            )

            first_sequence_divergence_candidate = (
                candidate_trade
            )

            break

    # ----------------------------------------------------------
    # Find earliest fresh-only / historical-only trade.
    # ----------------------------------------------------------

    fresh_only_trades = [
        trade
        for trade in candidate
        if trade["_identity"] in fresh_only_ids
    ]

    historical_only_trades = [
        trade
        for trade in historical
        if trade["_identity"] in historical_only_ids
    ]

    fresh_only_trades.sort(
        key=lambda x: x["_entry_date_clean"]
    )

    historical_only_trades.sort(
        key=lambda x: x["_entry_date_clean"]
    )

    earliest_fresh_only = (
        fresh_only_trades[0]
        if fresh_only_trades
        else None
    )

    earliest_historical_only = (
        historical_only_trades[0]
        if historical_only_trades
        else None
    )

    # ----------------------------------------------------------
    # Aggregate R
    # ----------------------------------------------------------

    historical_r = sum(
        trade["_r_value"]
        for trade in historical
    )

    candidate_r = sum(
        trade["_r_value"]
        for trade in candidate
    )

    fresh_only_r = sum(
        trade["_r_value"]
        for trade in fresh_only_trades
    )

    historical_only_r = sum(
        trade["_r_value"]
        for trade in historical_only_trades
    )

    # ----------------------------------------------------------
    # Print symbol analysis
    # ----------------------------------------------------------

    print()
    print("=" * 110)
    print(f"SYMBOL: {symbol}")
    print("=" * 110)

    print(
        f"Historical trades : {len(historical)}"
    )

    print(
        f"Candidate trades  : {len(candidate)}"
    )

    print(
        f"Common trades     : {len(common_ids)}"
    )

    print(
        f"Fresh-only trades : {len(fresh_only_ids)}"
    )

    print(
        f"Historical-only   : {len(historical_only_ids)}"
    )

    print(
        f"Historical R      : {historical_r:.2f}"
    )

    print(
        f"Candidate R       : {candidate_r:.2f}"
    )

    print(
        f"Fresh-only R      : {fresh_only_r:.2f}"
    )

    print(
        f"Historical-only R : {historical_only_r:.2f}"
    )

    # ----------------------------------------------------------
    # First sequence divergence
    # ----------------------------------------------------------

    print()
    print("FIRST SEQUENCE DIVERGENCE")
    print("-" * 110)

    if first_sequence_divergence_index is None:

        print("NONE — sequences remain aligned.")

        sequence_status = "ALIGNED"

        divergence_index = ""

        historical_divergence = ""

        candidate_divergence = ""

    else:

        print(
            f"Trade sequence index: "
            f"{first_sequence_divergence_index + 1}"
        )

        if first_sequence_divergence_historical:
            print(
                "Historical:"
            )
            print(
                f"  {trade_summary(first_sequence_divergence_historical)}"
            )
        else:
            print(
                "Historical: NO TRADE"
            )

        if first_sequence_divergence_candidate:
            print(
                "Candidate:"
            )
            print(
                f"  {trade_summary(first_sequence_divergence_candidate)}"
            )
        else:
            print(
                "Candidate: NO TRADE"
            )

        sequence_status = "DIVERGED"

        divergence_index = (
            first_sequence_divergence_index + 1
        )

        historical_divergence = (
            trade_summary(first_sequence_divergence_historical)
            if first_sequence_divergence_historical
            else "NO TRADE"
        )

        candidate_divergence = (
            trade_summary(first_sequence_divergence_candidate)
            if first_sequence_divergence_candidate
            else "NO TRADE"
        )

    # ----------------------------------------------------------
    # Earliest identity differences
    # ----------------------------------------------------------

    print()
    print("EARLIEST FRESH-ONLY TRADE")
    print("-" * 110)

    if earliest_fresh_only:

        print(
            trade_summary(earliest_fresh_only)
        )

        earliest_fresh_only_date = (
            earliest_fresh_only["_entry_date_clean"]
        )

    else:

        print("NONE")

        earliest_fresh_only_date = ""

    print()
    print("EARLIEST HISTORICAL-ONLY TRADE")
    print("-" * 110)

    if earliest_historical_only:

        print(
            trade_summary(earliest_historical_only)
        )

        earliest_historical_only_date = (
            earliest_historical_only["_entry_date_clean"]
        )

    else:

        print("NONE")

        earliest_historical_only_date = ""

    # ----------------------------------------------------------
    # Show first 5 trades from each side around divergence.
    # ----------------------------------------------------------

    print()
    print("CHRONOLOGICAL TRADE SEQUENCE — FIRST 5")
    print("-" * 110)

    print("HISTORICAL:")

    for index, trade in enumerate(historical[:5], start=1):

        print(
            f"  {index:02d}. "
            f"{trade_summary(trade)}"
        )

    print()
    print("CANDIDATE:")

    for index, trade in enumerate(candidate[:5], start=1):

        print(
            f"  {index:02d}. "
            f"{trade_summary(trade)}"
        )

    # ----------------------------------------------------------
    # Save result
    # ----------------------------------------------------------

    results.append(
        {
            "symbol": symbol,
            "historical_trade_count": len(historical),
            "candidate_trade_count": len(candidate),
            "common_trade_count": len(common_ids),
            "fresh_only_count": len(fresh_only_ids),
            "historical_only_count": len(historical_only_ids),
            "historical_total_r": round(historical_r, 6),
            "candidate_total_r": round(candidate_r, 6),
            "fresh_only_r": round(fresh_only_r, 6),
            "historical_only_r": round(historical_only_r, 6),
            "sequence_status": sequence_status,
            "first_sequence_divergence_index": divergence_index,
            "historical_at_first_divergence": historical_divergence,
            "candidate_at_first_divergence": candidate_divergence,
            "earliest_fresh_only_date": earliest_fresh_only_date,
            "earliest_historical_only_date": earliest_historical_only_date,
        }
    )


# ==============================================================
# SAVE FORENSIC REPORT
# ==============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

fieldnames = [
    "symbol",
    "historical_trade_count",
    "candidate_trade_count",
    "common_trade_count",
    "fresh_only_count",
    "historical_only_count",
    "historical_total_r",
    "candidate_total_r",
    "fresh_only_r",
    "historical_only_r",
    "sequence_status",
    "first_sequence_divergence_index",
    "historical_at_first_divergence",
    "candidate_at_first_divergence",
    "earliest_fresh_only_date",
    "earliest_historical_only_date",
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
    writer.writerows(results)


# ==============================================================
# FINAL SUMMARY
# ==============================================================

print()
print("=" * 110)
print("FORENSIC SUMMARY")
print("=" * 110)

for result in results:

    print()
    print(
        f"{result['symbol']}"
    )

    print(
        f"  First sequence divergence : "
        f"{result['first_sequence_divergence_index'] or 'NONE'}"
    )

    print(
        f"  Earliest fresh-only       : "
        f"{result['earliest_fresh_only_date'] or 'NONE'}"
    )

    print(
        f"  Earliest historical-only  : "
        f"{result['earliest_historical_only_date'] or 'NONE'}"
    )

    print(
        f"  Fresh-only R              : "
        f"{result['fresh_only_r']:.2f}"
    )

    print(
        f"  Historical-only R         : "
        f"{result['historical_only_r']:.2f}"
    )

print()
print("=" * 110)
print("REPORT SAVED")
print("=" * 110)

print(
    f"File: {OUTPUT_FILE}"
)

print()
print(
    "IMPORTANT: This diagnostic does not modify either dataset."
)

print("=" * 110)