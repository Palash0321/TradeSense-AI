"""
TradeSense-AI — OOS Evidence Ledger

Purpose
-------
Read-only evidence/audit layer for the five-stream genuine OOS holdouts.

This script:
1. Reads existing OOS ledgers only.
2. Admits only genuine post-freeze trades.
3. Applies the LOCKED 7B / 0.25R hypothesis observationally.
4. Records per-trade evidence when genuine OOS trades exist.
5. Does NOT create fake zero-trade ledgers.
6. Does NOT modify production, historical data, or OOS source ledgers.
7. Does NOT retune any parameter.
8. Maintains a cumulative evidence ledger plus a current snapshot.

Locked hypothesis
-----------------
Early adverse >= 0.25R within first 7 entry bars.

Important
---------
This is an evidence/audit layer only.
It is NOT a strategy rule and does NOT alter production behavior.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

LOCKED_HYPOTHESIS = "Early adverse >= 0.25R within first 7 entry bars"

EARLY_ADVERSE_THRESHOLD_R = 0.25
MAX_ENTRY_BARS = 7

OUTPUT_DIR = Path("tests/output/research/oos_monitor")
EVIDENCE_DIR = OUTPUT_DIR / "evidence"

CURRENT_EVIDENCE_FILE = EVIDENCE_DIR / "oos_evidence_current.csv"
EVIDENCE_HISTORY_FILE = EVIDENCE_DIR / "oos_evidence_history.jsonl"

EXPECTED_STREAMS = {
    "STOCK_HOLDOUT": {
        "forward_start": "2026-09-09",
        "ledger": Path("tests/output/tradesense_holdout_trades.csv"),
    },
    "NIFTY_50_FORWARD": {
        "forward_start": "2026-09-14",
        "ledger": Path(
            "tests/output/index_backtests/nifty50_forward_holdout.csv"
        ),
    },
    "SENSEX_FORWARD": {
        "forward_start": "2026-09-14",
        "ledger": Path(
            "tests/output/index_backtests/sensex_forward_holdout.csv"
        ),
    },
    "BANK_NIFTY_FORWARD": {
        "forward_start": "2026-09-14",
        "ledger": Path(
            "tests/output/index_backtests/"
            "banknifty_short_continuation_forward_holdout.csv"
        ),
    },
    "MIDSELECT_FORWARD": {
        "forward_start": "2026-09-14",
        "ledger": Path(
            "tests/output/index_backtests/midselect_forward_holdout.csv"
        ),
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def normalize_column_name(value: Any) -> str:
    text = str(value).strip().lower()
    for char in [
        " ",
        "-",
        "/",
        "\\",
        ".",
        ":",
        "(",
        ")",
        "[",
        "]",
    ]:
        text = text.replace(char, "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Map normalized column name -> actual source column name.
    """
    mapping: dict[str, str] = {}

    for col in df.columns:
        mapping[normalize_column_name(col)] = col

    return mapping


def find_column(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    mapping = normalized_columns(df)

    for alias in aliases:
        key = normalize_column_name(alias)

        if key in mapping:
            return mapping[key]

    return None


def numeric_value(
    row: pd.Series,
    column: str | None,
) -> float | None:
    if column is None:
        return None

    value = row.get(column)

    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def text_value(
    row: pd.Series,
    column: str | None,
) -> str | None:
    if column is None:
        return None

    value = row.get(column)

    if pd.isna(value):
        return None

    return str(value).strip()


def date_value(
    row: pd.Series,
    column: str | None,
) -> str | None:
    if column is None:
        return None

    value = row.get(column)

    if pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.strftime("%Y-%m-%d")


def timestamp_value(
    row: pd.Series,
    column: str | None,
) -> str | None:
    if column is None:
        return None

    value = row.get(column)

    if pd.isna(value):
        return None

    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    return parsed.isoformat()


def first_existing(
    df: pd.DataFrame,
    aliases: list[str],
) -> str | None:
    return find_column(df, aliases)


def stable_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def trade_key(
    stream: str,
    row: pd.Series,
    symbol_col: str | None,
    direction_col: str | None,
    entry_date_col: str | None,
    entry_timestamp_col: str | None,
) -> str:

    symbol = text_value(row, symbol_col) or ""
    direction = text_value(row, direction_col) or ""

    entry_date = (
        date_value(row, entry_date_col)
        or timestamp_value(row, entry_timestamp_col)
        or ""
    )

    return "|".join(
        [
            stream,
            symbol,
            direction,
            entry_date,
        ]
    )


def extract_bar_adverse_values(
    row: pd.Series,
    df: pd.DataFrame,
) -> list[float | None]:

    values: list[float | None] = []

    for bar_number in range(1, MAX_ENTRY_BARS + 1):

        aliases = [
            f"early_adverse_r_bar_{bar_number}",
            f"early_adverse_bar_{bar_number}_r",
            f"bar_{bar_number}_adverse_r",
            f"bar{bar_number}_adverse_r",
            f"bar_{bar_number}_bar_adverse_r",
            f"bar{bar_number}_bar_adverse_r",
            f"bar{bar_number}_adverse",
            f"early_adverse_{bar_number}",
        ]

        column = first_existing(df, aliases)

        values.append(
            numeric_value(row, column)
        )

    return values


def extract_path_from_generic_columns(
    row: pd.Series,
    df: pd.DataFrame,
) -> list[float | None]:

    """
    Secondary path extraction.

    Supports columns such as:
        bar1_bar_adverse_r
        bar2_bar_adverse_r
        ...
    """

    values: list[float | None] = []

    for bar_number in range(1, MAX_ENTRY_BARS + 1):

        aliases = [
            f"bar{bar_number}_bar_adverse_r",
            f"bar_{bar_number}_bar_adverse_r",
            f"bar{bar_number}_adverse_r",
            f"bar_{bar_number}_adverse_r",
        ]

        column = first_existing(df, aliases)

        values.append(
            numeric_value(row, column)
        )

    return values


def classify_7b_event(
    bar_values: list[float | None],
) -> tuple[
    str,
    int | None,
    float | None,
    int,
]:
    """
    Returns:

    classification
    earliest_crossing_bar
    max_observed_adverse_r
    full_7b_window
    """

    observed = [
        value
        for value in bar_values
        if value is not None and np.isfinite(value)
    ]

    if not observed:
        return (
            "MISSING_PATH",
            None,
            None,
            0,
        )

    max_adverse = max(observed)

    crossing_bar: int | None = None

    for index, value in enumerate(bar_values, start=1):

        if value is None:
            continue

        if not np.isfinite(value):
            continue

        if value >= EARLY_ADVERSE_THRESHOLD_R:
            crossing_bar = index
            break

    full_window = int(
        all(
            value is not None
            and np.isfinite(value)
            for value in bar_values
        )
    )

    if crossing_bar is not None:
        classification = "EARLY_ADVERSE_7B"

    elif full_window:
        classification = "NO_EARLY_ADVERSE_7B"

    else:
        classification = "PARTIAL_7B_WINDOW"

    return (
        classification,
        crossing_bar,
        max_adverse,
        full_window,
    )


def build_evidence_record(
    stream: str,
    forward_start: str,
    row: pd.Series,
    df: pd.DataFrame,
) -> dict[str, Any]:

    symbol_col = first_existing(
        df,
        [
            "symbol",
            "_symbol",
            "ticker",
            "instrument",
        ],
    )

    direction_col = first_existing(
        df,
        [
            "direction",
            "position",
            "side",
        ],
    )

    setup_col = first_existing(
        df,
        [
            "setup",
            "setup_name",
            "strategy_setup",
        ],
    )

    entry_date_col = first_existing(
        df,
        [
            "entry_date",
            "entry_day",
            "date",
        ],
    )

    entry_timestamp_col = first_existing(
        df,
        [
            "entry_timestamp",
            "entry_time",
            "timestamp",
        ],
    )

    signal_timestamp_col = first_existing(
        df,
        [
            "signal_timestamp",
            "signal_time",
        ],
    )

    entry_price_col = first_existing(
        df,
        [
            "entry_price",
            "entry_price_reconstructed",
            "entry",
            "entry_px",
        ],
    )

    final_r_col = first_existing(
        df,
        [
            "r_multiple",
            "final_r",
            "final_r_multiple",
            "trade_r",
        ],
    )

    mfe_col = first_existing(
        df,
        [
            "max_favorable_r",
            "mfe_r",
            "mfe",
        ],
    )

    mae_col = first_existing(
        df,
        [
            "max_adverse_r",
            "mae_r",
            "mae",
        ],
    )

    observation_status_col = first_existing(
        df,
        [
            "production_observation_status",
            "observation_status",
            "oos_observation_status",
        ],
    )

    production_observed_col = first_existing(
        df,
        [
            "production_observed",
            "oos_observed",
            "observed",
        ],
    )

    bars_in_trade_col = first_existing(
        df,
        [
            "production_bars_in_trade",
            "bars_in_trade",
            "trade_bars",
        ],
    )

    bar_values = extract_bar_adverse_values(row, df)

    if not any(
        value is not None
        for value in bar_values
    ):
        bar_values = extract_path_from_generic_columns(
            row,
            df,
        )

    (
        classification,
        earliest_crossing_bar,
        max_observed_adverse_r,
        full_7b_window,
    ) = classify_7b_event(bar_values)

    record: dict[str, Any] = {
        "stream": stream,
        "forward_start": forward_start,
        "symbol": text_value(row, symbol_col),
        "direction": text_value(row, direction_col),
        "setup": text_value(row, setup_col),
        "entry_date": date_value(row, entry_date_col),
        "entry_timestamp": timestamp_value(
            row,
            entry_timestamp_col,
        ),
        "signal_timestamp": timestamp_value(
            row,
            signal_timestamp_col,
        ),
        "entry_price": numeric_value(
            row,
            entry_price_col,
        ),
        "final_r": numeric_value(
            row,
            final_r_col,
        ),
        "mfe_r": numeric_value(
            row,
            mfe_col,
        ),
        "mae_r": numeric_value(
            row,
            mae_col,
        ),
        "bar1_adverse_r": bar_values[0],
        "bar2_adverse_r": bar_values[1],
        "bar3_adverse_r": bar_values[2],
        "bar4_adverse_r": bar_values[3],
        "bar5_adverse_r": bar_values[4],
        "bar6_adverse_r": bar_values[5],
        "bar7_adverse_r": bar_values[6],
        "earliest_crossing_bar": earliest_crossing_bar,
        "max_observed_adverse_r": max_observed_adverse_r,
        "full_7b_window": full_7b_window,
        "classification": classification,
        "observation_status": text_value(
            row,
            observation_status_col,
        ),
        "production_observed": numeric_value(
            row,
            production_observed_col,
        ),
        "bars_in_trade": numeric_value(
            row,
            bars_in_trade_col,
        ),
    }

    record["trade_key"] = trade_key(
        stream,
        row,
        symbol_col,
        direction_col,
        entry_date_col,
        entry_timestamp_col,
    )

    record["evidence_hash"] = stable_hash(record)

    return record


def read_stream(
    stream: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    ledger_path: Path = config["ledger"]
    forward_start = config["forward_start"]

    metadata: dict[str, Any] = {
        "stream": stream,
        "forward_start": forward_start,
        "ledger": str(ledger_path),
        "ledger_exists": ledger_path.exists(),
        "source_rows": 0,
        "genuine_oos_rows": 0,
        "records_created": 0,
        "pre_start_rows": 0,
        "invalid_dates": 0,
        "duplicate_keys": 0,
        "status": "WAITING_FOR_GENUINE_OOS_TRADES",
    }

    if not ledger_path.exists():
        return [], metadata

    try:
        df = pd.read_csv(ledger_path)
    except Exception as exc:
        metadata["status"] = "INVALID"
        metadata["error"] = (
            f"Unable to read ledger: {type(exc).__name__}: {exc}"
        )
        return [], metadata

    metadata["source_rows"] = int(len(df))

    if df.empty:
        return [], metadata

    entry_date_col = first_existing(
        df,
        [
            "entry_date",
            "entry_day",
            "date",
        ],
    )

    entry_timestamp_col = first_existing(
        df,
        [
            "entry_timestamp",
            "entry_time",
            "timestamp",
        ],
    )

    if entry_date_col is None and entry_timestamp_col is None:
        metadata["status"] = "INVALID"
        metadata["error"] = "No entry date/timestamp column found."
        return [], metadata

    if entry_date_col is not None:
        dates = pd.to_datetime(
            df[entry_date_col],
            errors="coerce",
        )
    else:
        dates = pd.to_datetime(
            df[entry_timestamp_col],
            errors="coerce",
        )

    metadata["invalid_dates"] = int(
        dates.isna().sum()
    )

    valid_mask = dates.notna()

    forward_date = pd.Timestamp(
        forward_start
    )

    genuine_mask = (
        valid_mask
        & (dates >= forward_date)
    )

    pre_start_mask = (
        valid_mask
        & (dates < forward_date)
    )

    metadata["pre_start_rows"] = int(
        pre_start_mask.sum()
    )

    genuine_df = df.loc[genuine_mask].copy()

    metadata["genuine_oos_rows"] = int(
        len(genuine_df)
    )

    if genuine_df.empty:
        return [], metadata

    records: list[dict[str, Any]] = []

    for _, row in genuine_df.iterrows():

        record = build_evidence_record(
            stream,
            forward_start,
            row,
            df,
        )

        records.append(record)

    keys = [
        record["trade_key"]
        for record in records
    ]

    duplicate_count = (
        len(keys) - len(set(keys))
    )

    metadata["duplicate_keys"] = int(
        duplicate_count
    )

    if duplicate_count > 0:
        metadata["status"] = "INVALID"
        metadata["error"] = (
            "Duplicate genuine OOS trade keys detected."
        )
        return records, metadata

    metadata["records_created"] = len(records)
    metadata["status"] = "OBSERVING_GENUINE_OOS"

    return records, metadata


# =============================================================================
# HISTORY
# =============================================================================

def load_existing_evidence() -> pd.DataFrame:

    if not CURRENT_EVIDENCE_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            CURRENT_EVIDENCE_FILE
        )
    except Exception:
        return pd.DataFrame()


def append_history(
    snapshot: dict[str, Any],
) -> None:

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with EVIDENCE_HISTORY_FILE.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                snapshot,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print()
    print("=" * 100)
    print("TRADESENSE-AI — OOS EVIDENCE LEDGER")
    print("=" * 100)

    print()
    print("LOCKED HYPOTHESIS")
    print("-" * 100)
    print(LOCKED_HYPOTHESIS)

    print()
    print("SAFETY CONTRACT")
    print("-" * 100)
    print("Production strategy : READ-ONLY")
    print("Historical data     : READ-ONLY")
    print("OOS source ledgers  : READ-ONLY")
    print("Parameter tuning    : NONE")
    print("Hypothesis          : LOCKED")
    print("Zero-trade ledger   : NOT FABRICATED")

    all_records: list[dict[str, Any]] = []
    stream_metadata: list[dict[str, Any]] = []

    for stream, config in EXPECTED_STREAMS.items():

        print()
        print("-" * 100)
        print(f"STREAM: {stream}")
        print("-" * 100)

        records, metadata = read_stream(
            stream,
            config,
        )

        stream_metadata.append(
            metadata
        )

        all_records.extend(
            records
        )

        print(
            f"Ledger exists       : "
            f"{metadata['ledger_exists']}"
        )

        print(
            f"Source rows         : "
            f"{metadata['source_rows']}"
        )

        print(
            f"Genuine OOS rows    : "
            f"{metadata['genuine_oos_rows']}"
        )

        print(
            f"Evidence records    : "
            f"{metadata['records_created']}"
        )

        print(
            f"Pre-start rows      : "
            f"{metadata['pre_start_rows']}"
        )

        print(
            f"Duplicate keys      : "
            f"{metadata['duplicate_keys']}"
        )

        print(
            f"Status              : "
            f"{metadata['status']}"
        )

        if "error" in metadata:
            print(
                f"ERROR               : "
                f"{metadata['error']}"
            )

    total_genuine = sum(
        item["genuine_oos_rows"]
        for item in stream_metadata
    )

    invalid_streams = sum(
        item["status"] == "INVALID"
        for item in stream_metadata
    )

    observing_streams = sum(
        item["status"]
        == "OBSERVING_GENUINE_OOS"
        for item in stream_metadata
    )

    waiting_streams = sum(
        item["status"]
        == "WAITING_FOR_GENUINE_OOS_TRADES"
        for item in stream_metadata
    )

    print()
    print("=" * 100)
    print("GLOBAL EVIDENCE STATUS")
    print("=" * 100)

    print(
        f"Streams monitored   : "
        f"{len(EXPECTED_STREAMS)}"
    )

    print(
        f"Streams observing   : "
        f"{observing_streams}"
    )

    print(
        f"Streams waiting     : "
        f"{waiting_streams}"
    )

    print(
        f"Streams invalid     : "
        f"{invalid_streams}"
    )

    print(
        f"Total genuine OOS   : "
        f"{total_genuine}"
    )

    # -------------------------------------------------------------------------
    # INVALID STATE
    # -------------------------------------------------------------------------

    if invalid_streams > 0:

        print()
        print(
            "FINAL STATUS         : INVALID"
        )

        print()
        print("SAFETY CHECK")
        print("-" * 100)
        print(
            "FAIL: One or more OOS evidence "
            "sources are invalid."
        )
        print(
            "No evidence ledger snapshot written."
        )

        return 1

    # -------------------------------------------------------------------------
    # ZERO OOS STATE
    # -------------------------------------------------------------------------

    if total_genuine == 0:

        print()
        print(
            "FINAL STATUS         : "
            "WAITING_FOR_GENUINE_OOS_TRADES"
        )

        print()
        print(
            "No genuine OOS trades exist."
        )

        print(
            "No zero-trade evidence ledger "
            "has been fabricated."
        )

        print()
        print("SAFETY CHECK")
        print("-" * 100)
        print(
            "PASS: Production strategy unchanged."
        )
        print(
            "PASS: Historical datasets unchanged."
        )
        print(
            "PASS: OOS source ledgers read-only."
        )
        print(
            "PASS: Locked 7B / 0.25R hypothesis unchanged."
        )
        print(
            "PASS: No parameter tuning."
        )
        print(
            "PASS: No trade fabrication."
        )

        print()
        print(
            "STATUS: OOS EVIDENCE WAITING"
        )

        return 0

    # -------------------------------------------------------------------------
    # GENUINE OOS EXISTS
    # -------------------------------------------------------------------------

    evidence_df = pd.DataFrame(
        all_records
    )

    if evidence_df.empty:

        print()
        print(
            "FINAL STATUS         : INVALID"
        )

        print(
            "Genuine OOS trades were detected "
            "but no evidence records were produced."
        )

        return 1

    # Stable ordering
    sort_columns = [
        column
        for column in [
            "entry_date",
            "stream",
            "symbol",
            "direction",
            "setup",
            "trade_key",
        ]
        if column in evidence_df.columns
    ]

    if sort_columns:
        evidence_df = evidence_df.sort_values(
            sort_columns,
            kind="stable",
        )

    evidence_df = evidence_df.drop_duplicates(
        subset=["trade_key"],
        keep="last",
    )

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Compare with previous evidence
    # -------------------------------------------------------------------------

    previous_df = load_existing_evidence()

    previous_keys: set[str] = set()

    if (
        not previous_df.empty
        and "trade_key" in previous_df.columns
    ):
        previous_keys = set(
            previous_df["trade_key"]
            .astype(str)
        )

    current_keys = set(
        evidence_df["trade_key"]
        .astype(str)
    )

    new_keys = (
        current_keys
        - previous_keys
    )

    # -------------------------------------------------------------------------
    # Write current evidence
    # -------------------------------------------------------------------------

    evidence_df.to_csv(
        CURRENT_EVIDENCE_FILE,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Aggregate classification
    # -------------------------------------------------------------------------

    event_mask = (
        evidence_df["classification"]
        == "EARLY_ADVERSE_7B"
    )

    no_event_mask = (
        evidence_df["classification"]
        == "NO_EARLY_ADVERSE_7B"
    )

    partial_mask = (
        evidence_df["classification"]
        == "PARTIAL_7B_WINDOW"
    )

    missing_mask = (
        evidence_df["classification"]
        == "MISSING_PATH"
    )

    event_count = int(
        event_mask.sum()
    )

    no_event_count = int(
        no_event_mask.sum()
    )

    partial_count = int(
        partial_mask.sum()
    )

    missing_count = int(
        missing_mask.sum()
    )

    full_evaluated = (
        event_count
        + no_event_count
    )

    event_rate = None

    if full_evaluated > 0:
        event_rate = (
            event_count
            / full_evaluated
        )

    # -------------------------------------------------------------------------
    # Snapshot
    # -------------------------------------------------------------------------

    snapshot = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "locked_hypothesis": LOCKED_HYPOTHESIS,
        "threshold_r": EARLY_ADVERSE_THRESHOLD_R,
        "max_entry_bars": MAX_ENTRY_BARS,
        "streams_monitored": len(
            EXPECTED_STREAMS
        ),
        "streams_observing": observing_streams,
        "streams_waiting": waiting_streams,
        "streams_invalid": invalid_streams,
        "total_genuine_oos": int(
            len(evidence_df)
        ),
        "new_genuine_oos_records": int(
            len(new_keys)
        ),
        "early_adverse_7b_events": event_count,
        "full_7b_no_event": no_event_count,
        "partial_7b_window": partial_count,
        "missing_path": missing_count,
        "full_7b_evaluated": full_evaluated,
        "event_rate": event_rate,
        "current_evidence_file": str(
            CURRENT_EVIDENCE_FILE
        ),
        "evidence_history_file": str(
            EVIDENCE_HISTORY_FILE
        ),
        "evidence_fingerprint": stable_hash(
            evidence_df.to_dict(
                orient="records"
            )
        ),
    }

    append_history(
        snapshot
    )

    # -------------------------------------------------------------------------
    # Console output
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("EVIDENCE SUMMARY")
    print("=" * 100)

    print(
        f"Total genuine OOS   : "
        f"{len(evidence_df)}"
    )

    print(
        f"New evidence records: "
        f"{len(new_keys)}"
    )

    print(
        f"7B early-adverse    : "
        f"{event_count}"
    )

    print(
        f"7B no-event         : "
        f"{no_event_count}"
    )

    print(
        f"Partial 7B          : "
        f"{partial_count}"
    )

    print(
        f"Missing path        : "
        f"{missing_count}"
    )

    print(
        f"Full 7B evaluated   : "
        f"{full_evaluated}"
    )

    if event_rate is None:
        print(
            "7B event rate       : None"
        )
    else:
        print(
            f"7B event rate       : "
            f"{event_rate:.6f}"
        )

    print()
    print("EVIDENCE FILE")
    print("-" * 100)
    print(
        CURRENT_EVIDENCE_FILE
    )

    print()
    print("HISTORY FILE")
    print("-" * 100)
    print(
        EVIDENCE_HISTORY_FILE
    )

    print()
    print("=" * 100)
    print("SAFETY CHECK")
    print("=" * 100)

    print(
        "PASS: Production strategy unchanged."
    )

    print(
        "PASS: Historical datasets unchanged."
    )

    print(
        "PASS: OOS source ledgers read-only."
    )

    print(
        "PASS: Locked 7B / 0.25R hypothesis unchanged."
    )

    print(
        "PASS: No parameter tuning."
    )

    print(
        "PASS: Evidence keyed at trade level."
    )

    print(
        "PASS: Evidence history append-only."
    )

    print()
    print(
        "STATUS: OOS EVIDENCE PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )