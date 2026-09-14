"""
TradeSense-AI
Full Trade Dataset Reconciliation

Purpose
-------
Compare the frozen historical research dataset against the newly generated
research candidate dataset WITHOUT modifying the frozen historical dataset.

Frozen dataset:
    tests/output/tradesense_trades.csv

Research candidate:
    tests/output/tradesense_trades_research_candidate.csv

Outputs:
    tests/output/reconciliation/summary.csv
    tests/output/reconciliation/fresh_only_trades.csv
    tests/output/reconciliation/historical_only_trades.csv
    tests/output/reconciliation/common_trade_field_changes.csv
    tests/output/reconciliation/common_trade_r_reconciliation.csv
    tests/output/reconciliation/symbol_reconciliation.csv
    tests/output/reconciliation/year_reconciliation.csv
"""

import csv
import os
from collections import defaultdict


# ---------------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------------

HISTORICAL_FILE = "tests/output/tradesense_trades.csv"

CANDIDATE_FILE = (
    "tests/output/tradesense_trades_research_candidate.csv"
)

OUTPUT_DIR = "tests/output/reconciliation"


# ---------------------------------------------------------------------
# TRADE IDENTITY
# ---------------------------------------------------------------------
#
# IMPORTANT:
#
# A trade identity is:
#
#     symbol + entry_date + direction
#
# Setup is deliberately NOT part of the identity because setup metadata
# may change while the underlying trade entry remains the same.
#
# This matches the identity logic already used by the holdout protection.
# ---------------------------------------------------------------------

IDENTITY_FIELDS = (
    "_symbol",
    "entry_date",
    "direction",
)


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def normalize(value):
    """Normalize CSV values for reliable comparison."""
    if value is None:
        return ""

    return str(value).strip()


def trade_identity(trade):
    """Return the canonical identity of a trade."""

    return (
        normalize(trade.get("_symbol")),
        normalize(trade.get("entry_date"))[:10],
        normalize(trade.get("direction")).upper(),
    )


def safe_float(value):
    """Convert a value to float when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    """Load a CSV file into a list of dictionaries."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    return rows


def write_csv(path, rows, fieldnames=None):
    """Write rows to CSV."""

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    if fieldnames is None:

        fieldnames = sorted(
            {
                key
                for row in rows
                for key in row.keys()
            }
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(rows)


# ---------------------------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("TRADESENSE-AI TRADE DATASET RECONCILIATION")
print("=" * 100)

print()
print(f"Frozen historical dataset:")
print(f"  {HISTORICAL_FILE}")

print()
print(f"Fresh research candidate:")
print(f"  {CANDIDATE_FILE}")


historical_trades = load_csv(HISTORICAL_FILE)

candidate_trades = load_csv(CANDIDATE_FILE)


print()
print("-" * 100)
print("DATASET COUNTS")
print("-" * 100)

print(
    f"Historical trades : {len(historical_trades)}"
)

print(
    f"Candidate trades  : {len(candidate_trades)}"
)

print(
    f"Net count change  : "
    f"{len(candidate_trades) - len(historical_trades):+d}"
)


# ---------------------------------------------------------------------
# BUILD IDENTITY MAPS
# ---------------------------------------------------------------------

historical_map = {}
candidate_map = {}

historical_duplicates = []
candidate_duplicates = []


for trade in historical_trades:

    identity = trade_identity(trade)

    if identity in historical_map:

        historical_duplicates.append(
            identity
        )

    else:

        historical_map[identity] = trade


for trade in candidate_trades:

    identity = trade_identity(trade)

    if identity in candidate_map:

        candidate_duplicates.append(
            identity
        )

    else:

        candidate_map[identity] = trade


# ---------------------------------------------------------------------
# DUPLICATE CHECK
# ---------------------------------------------------------------------

print()
print("-" * 100)
print("TRADE IDENTITY CHECK")
print("-" * 100)

print(
    "Identity fields:"
)

print(
    "  _symbol + entry_date + direction"
)

print(
    f"Historical duplicate identities : "
    f"{len(historical_duplicates)}"
)

print(
    f"Candidate duplicate identities  : "
    f"{len(candidate_duplicates)}"
)


if historical_duplicates:

    print()
    print("WARNING: Historical dataset contains duplicate trade identities.")

    for identity in historical_duplicates[:20]:

        print(
            f"  {identity}"
        )


if candidate_duplicates:

    print()
    print("WARNING: Candidate dataset contains duplicate trade identities.")

    for identity in candidate_duplicates[:20]:

        print(
            f"  {identity}"
        )


# ---------------------------------------------------------------------
# IDENTITY SETS
# ---------------------------------------------------------------------

historical_ids = set(
    historical_map.keys()
)

candidate_ids = set(
    candidate_map.keys()
)


common_ids = (
    historical_ids
    & candidate_ids
)

fresh_only_ids = (
    candidate_ids
    - historical_ids
)

historical_only_ids = (
    historical_ids
    - candidate_ids
)


# ---------------------------------------------------------------------
# BASIC RECONCILIATION
# ---------------------------------------------------------------------

print()
print("-" * 100)
print("TRADE IDENTITY RECONCILIATION")
print("-" * 100)

print(
    f"Historical unique trades : "
    f"{len(historical_ids)}"
)

print(
    f"Candidate unique trades  : "
    f"{len(candidate_ids)}"
)

print(
    f"Common trades            : "
    f"{len(common_ids)}"
)

print(
    f"Fresh-only trades        : "
    f"{len(fresh_only_ids)}"
)

print(
    f"Historical-only trades   : "
    f"{len(historical_only_ids)}"
)

print(
    f"Net trade-count change   : "
    f"{len(candidate_ids) - len(historical_ids):+d}"
)


# ---------------------------------------------------------------------
# FRESH-ONLY TRADES
# ---------------------------------------------------------------------

fresh_only_rows = []

for identity in sorted(fresh_only_ids):

    trade = candidate_map[identity]

    row = dict(trade)

    row["reconciliation_status"] = "FRESH_ONLY"

    fresh_only_rows.append(row)


# ---------------------------------------------------------------------
# HISTORICAL-ONLY TRADES
# ---------------------------------------------------------------------

historical_only_rows = []

for identity in sorted(historical_only_ids):

    trade = historical_map[identity]

    row = dict(trade)

    row["reconciliation_status"] = "HISTORICAL_ONLY"

    historical_only_rows.append(row)


# ---------------------------------------------------------------------
# COMMON TRADE FIELD CHANGES
# ---------------------------------------------------------------------

common_trade_field_changes = []

all_fields = sorted(
    set(
        key
        for trade in historical_trades + candidate_trades
        for key in trade.keys()
    )
)

ignored_fields = {
    "reconciliation_status",
}


for identity in sorted(common_ids):

    historical_trade = historical_map[identity]

    candidate_trade = candidate_map[identity]

    for field in all_fields:

        if field in ignored_fields:
            continue

        historical_value = normalize(
            historical_trade.get(field)
        )

        candidate_value = normalize(
            candidate_trade.get(field)
        )

        if historical_value != candidate_value:

            common_trade_field_changes.append(
                {
                    "_symbol": identity[0],
                    "entry_date": identity[1],
                    "direction": identity[2],
                    "field": field,
                    "historical_value": historical_value,
                    "candidate_value": candidate_value,
                }
            )


# ---------------------------------------------------------------------
# R-MULTIPLE RECONCILIATION
# ---------------------------------------------------------------------

common_r_reconciliation = []


for identity in sorted(common_ids):

    historical_trade = historical_map[identity]

    candidate_trade = candidate_map[identity]

    historical_r = safe_float(
        historical_trade.get("r_multiple")
    )

    candidate_r = safe_float(
        candidate_trade.get("r_multiple")
    )

    if (
        historical_r is not None
        and candidate_r is not None
    ):

        delta_r = candidate_r - historical_r

    else:

        delta_r = None

    common_r_reconciliation.append(
        {
            "_symbol": identity[0],
            "entry_date": identity[1],
            "direction": identity[2],
            "historical_r_multiple": (
                historical_r
                if historical_r is not None
                else ""
            ),
            "candidate_r_multiple": (
                candidate_r
                if candidate_r is not None
                else ""
            ),
            "delta_r": (
                delta_r
                if delta_r is not None
                else ""
            ),
            "historical_exit_date": normalize(
                historical_trade.get("exit_date")
            ),
            "candidate_exit_date": normalize(
                candidate_trade.get("exit_date")
            ),
            "historical_exit_reason": normalize(
                historical_trade.get("exit_reason")
            ),
            "candidate_exit_reason": normalize(
                candidate_trade.get("exit_reason")
            ),
        }
    )


# ---------------------------------------------------------------------
# SYMBOL-LEVEL RECONCILIATION
# ---------------------------------------------------------------------

symbol_stats = defaultdict(
    lambda: {
        "historical": 0,
        "candidate": 0,
        "common": 0,
        "fresh_only": 0,
        "historical_only": 0,
        "historical_r": 0.0,
        "candidate_r": 0.0,
    }
)


for identity, trade in historical_map.items():

    symbol = identity[0]

    symbol_stats[symbol]["historical"] += 1

    r = safe_float(
        trade.get("r_multiple")
    )

    if r is not None:

        symbol_stats[symbol]["historical_r"] += r


for identity, trade in candidate_map.items():

    symbol = identity[0]

    symbol_stats[symbol]["candidate"] += 1

    r = safe_float(
        trade.get("r_multiple")
    )

    if r is not None:

        symbol_stats[symbol]["candidate_r"] += r


for identity in common_ids:

    symbol = identity[0]

    symbol_stats[symbol]["common"] += 1


for identity in fresh_only_ids:

    symbol = identity[0]

    symbol_stats[symbol]["fresh_only"] += 1


for identity in historical_only_ids:

    symbol = identity[0]

    symbol_stats[symbol]["historical_only"] += 1


symbol_rows = []


for symbol in sorted(symbol_stats.keys()):

    stats = symbol_stats[symbol]

    symbol_rows.append(
        {
            "_symbol": symbol,
            "historical_trades": stats["historical"],
            "candidate_trades": stats["candidate"],
            "common_trades": stats["common"],
            "fresh_only_trades": stats["fresh_only"],
            "historical_only_trades": (
                stats["historical_only"]
            ),
            "net_trade_change": (
                stats["candidate"]
                - stats["historical"]
            ),
            "historical_total_r": round(
                stats["historical_r"],
                6,
            ),
            "candidate_total_r": round(
                stats["candidate_r"],
                6,
            ),
            "delta_r": round(
                stats["candidate_r"]
                - stats["historical_r"],
                6,
            ),
        }
    )


# ---------------------------------------------------------------------
# YEAR-LEVEL RECONCILIATION
# ---------------------------------------------------------------------

year_stats = defaultdict(
    lambda: {
        "historical": 0,
        "candidate": 0,
        "common": 0,
        "fresh_only": 0,
        "historical_only": 0,
        "historical_r": 0.0,
        "candidate_r": 0.0,
    }
)


def get_year(identity):
    entry_date = identity[1]

    if len(entry_date) >= 4:

        return entry_date[:4]

    return "UNKNOWN"


for identity, trade in historical_map.items():

    year = get_year(identity)

    year_stats[year]["historical"] += 1

    r = safe_float(
        trade.get("r_multiple")
    )

    if r is not None:

        year_stats[year]["historical_r"] += r


for identity, trade in candidate_map.items():

    year = get_year(identity)

    year_stats[year]["candidate"] += 1

    r = safe_float(
        trade.get("r_multiple")
    )

    if r is not None:

        year_stats[year]["candidate_r"] += r


for identity in common_ids:

    year = get_year(identity)

    year_stats[year]["common"] += 1


for identity in fresh_only_ids:

    year = get_year(identity)

    year_stats[year]["fresh_only"] += 1


for identity in historical_only_ids:

    year = get_year(identity)

    year_stats[year]["historical_only"] += 1


year_rows = []


for year in sorted(year_stats.keys()):

    stats = year_stats[year]

    year_rows.append(
        {
            "year": year,
            "historical_trades": stats["historical"],
            "candidate_trades": stats["candidate"],
            "common_trades": stats["common"],
            "fresh_only_trades": stats["fresh_only"],
            "historical_only_trades": (
                stats["historical_only"]
            ),
            "net_trade_change": (
                stats["candidate"]
                - stats["historical"]
            ),
            "historical_total_r": round(
                stats["historical_r"],
                6,
            ),
            "candidate_total_r": round(
                stats["candidate_r"],
                6,
            ),
            "delta_r": round(
                stats["candidate_r"]
                - stats["historical_r"],
                6,
            ),
        }
    )


# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------

historical_total_r = sum(
    safe_float(trade.get("r_multiple")) or 0.0
    for trade in historical_trades
)

candidate_total_r = sum(
    safe_float(trade.get("r_multiple")) or 0.0
    for trade in candidate_trades
)

common_historical_r = sum(
    safe_float(
        historical_map[identity].get("r_multiple")
    ) or 0.0
    for identity in common_ids
)

common_candidate_r = sum(
    safe_float(
        candidate_map[identity].get("r_multiple")
    ) or 0.0
    for identity in common_ids
)


summary_rows = [
    {
        "metric": "historical_trade_count",
        "value": len(historical_trades),
    },
    {
        "metric": "candidate_trade_count",
        "value": len(candidate_trades),
    },
    {
        "metric": "historical_unique_trade_count",
        "value": len(historical_ids),
    },
    {
        "metric": "candidate_unique_trade_count",
        "value": len(candidate_ids),
    },
    {
        "metric": "common_trade_count",
        "value": len(common_ids),
    },
    {
        "metric": "fresh_only_trade_count",
        "value": len(fresh_only_ids),
    },
    {
        "metric": "historical_only_trade_count",
        "value": len(historical_only_ids),
    },
    {
        "metric": "net_trade_count_change",
        "value": (
            len(candidate_ids)
            - len(historical_ids)
        ),
    },
    {
        "metric": "historical_duplicate_identity_count",
        "value": len(historical_duplicates),
    },
    {
        "metric": "candidate_duplicate_identity_count",
        "value": len(candidate_duplicates),
    },
    {
        "metric": "common_trade_field_change_count",
        "value": len(common_trade_field_changes),
    },
    {
        "metric": "historical_total_r",
        "value": round(
            historical_total_r,
            6,
        ),
    },
    {
        "metric": "candidate_total_r",
        "value": round(
            candidate_total_r,
            6,
        ),
    },
    {
        "metric": "total_r_change",
        "value": round(
            candidate_total_r
            - historical_total_r,
            6,
        ),
    },
    {
        "metric": "common_historical_total_r",
        "value": round(
            common_historical_r,
            6,
        ),
    },
    {
        "metric": "common_candidate_total_r",
        "value": round(
            common_candidate_r,
            6,
        ),
    },
    {
        "metric": "common_trade_r_change",
        "value": round(
            common_candidate_r
            - common_historical_r,
            6,
        ),
    },
]


# ---------------------------------------------------------------------
# WRITE OUTPUT FILES
# ---------------------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "summary.csv",
    ),
    summary_rows,
    fieldnames=[
        "metric",
        "value",
    ],
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "fresh_only_trades.csv",
    ),
    fresh_only_rows,
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "historical_only_trades.csv",
    ),
    historical_only_rows,
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "common_trade_field_changes.csv",
    ),
    common_trade_field_changes,
    fieldnames=[
        "_symbol",
        "entry_date",
        "direction",
        "field",
        "historical_value",
        "candidate_value",
    ],
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "common_trade_r_reconciliation.csv",
    ),
    common_r_reconciliation,
    fieldnames=[
        "_symbol",
        "entry_date",
        "direction",
        "historical_r_multiple",
        "candidate_r_multiple",
        "delta_r",
        "historical_exit_date",
        "candidate_exit_date",
        "historical_exit_reason",
        "candidate_exit_reason",
    ],
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "symbol_reconciliation.csv",
    ),
    symbol_rows,
)


write_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_reconciliation.csv",
    ),
    year_rows,
)


# ---------------------------------------------------------------------
# FINAL TERMINAL REPORT
# ---------------------------------------------------------------------

print()
print("=" * 100)
print("RECONCILIATION RESULT")
print("=" * 100)

print()
print(
    f"Historical trades       : "
    f"{len(historical_trades)}"
)

print(
    f"Candidate trades        : "
    f"{len(candidate_trades)}"
)

print(
    f"Common trades           : "
    f"{len(common_ids)}"
)

print(
    f"Fresh-only trades       : "
    f"{len(fresh_only_ids)}"
)

print(
    f"Historical-only trades  : "
    f"{len(historical_only_ids)}"
)

print(
    f"Net trade-count change  : "
    f"{len(candidate_ids) - len(historical_ids):+d}"
)

print()
print(
    f"Historical total R      : "
    f"{historical_total_r:.6f}"
)

print(
    f"Candidate total R       : "
    f"{candidate_total_r:.6f}"
)

print(
    f"Total R change          : "
    f"{candidate_total_r - historical_total_r:+.6f}"
)

print()
print(
    f"Common historical R     : "
    f"{common_historical_r:.6f}"
)

print(
    f"Common candidate R      : "
    f"{common_candidate_r:.6f}"
)

print(
    f"Common-trade R change   : "
    f"{common_candidate_r - common_historical_r:+.6f}"
)

print()
print(
    f"Common field changes    : "
    f"{len(common_trade_field_changes)}"
)

print(
    f"Historical duplicates   : "
    f"{len(historical_duplicates)}"
)

print(
    f"Candidate duplicates    : "
    f"{len(candidate_duplicates)}"
)


# ---------------------------------------------------------------------
# OUTPUT LOCATIONS
# ---------------------------------------------------------------------

print()
print("-" * 100)
print("RECONCILIATION FILES")
print("-" * 100)

print(
    f"{OUTPUT_DIR}/summary.csv"
)

print(
    f"{OUTPUT_DIR}/fresh_only_trades.csv"
)

print(
    f"{OUTPUT_DIR}/historical_only_trades.csv"
)

print(
    f"{OUTPUT_DIR}/common_trade_field_changes.csv"
)

print(
    f"{OUTPUT_DIR}/common_trade_r_reconciliation.csv"
)

print(
    f"{OUTPUT_DIR}/symbol_reconciliation.csv"
)

print(
    f"{OUTPUT_DIR}/year_reconciliation.csv"
)


# ---------------------------------------------------------------------
# FINAL SAFETY CHECK
# ---------------------------------------------------------------------

print()
print("-" * 100)
print("FROZEN DATASET SAFETY CHECK")
print("-" * 100)

if os.path.exists(HISTORICAL_FILE):

    print(
        "PASS: Frozen historical dataset still exists."
    )

else:

    print(
        "FAIL: Frozen historical dataset is missing."
    )


print()
print("=" * 100)
print("RECONCILIATION COMPLETE")
print("=" * 100)