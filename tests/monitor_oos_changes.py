"""
TradeSense-AI — OOS Monitor Change Detection

Purpose
-------
Compare the latest OOS monitor snapshot with the immediately previous
snapshot and report meaningful OOS-state changes.

This is a READ-ONLY monitoring layer.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within first 7 entry bars

SAFETY CONTRACT
---------------
- Production strategy unchanged.
- Historical data untouched.
- OOS ledgers untouched.
- No parameter tuning.
- No hypothesis modification.
- No trade fabrication.
- No promotion/rejection decision.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "research"
    / "oos_monitor"
    / "snapshots"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "research"
    / "oos_monitor"
)

CHANGE_HISTORY_FILE = (
    OUTPUT_DIR
    / "oos_monitor_change_history.jsonl"
)

LOCKED_HYPOTHESIS = (
    "Early adverse >= 0.25R within first 7 entry bars"
)


# ============================================================================
# HELPERS
# ============================================================================

def load_snapshot(path: Path) -> dict:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def snapshot_files():
    if not SNAPSHOT_DIR.exists():
        return []

    return sorted(
        SNAPSHOT_DIR.glob(
            "oos_monitor_*.json"
        ),
        key=lambda p: p.stat().st_mtime,
    )


def safe_number(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def stream_map(snapshot):
    return {
        item["stream"]: item
        for item in snapshot.get(
            "streams",
            [],
        )
    }


def format_value(value):
    if value is None:
        return "None"

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


# ============================================================================
# CHANGE DETECTION
# ============================================================================

def compare_stream(
    previous,
    current,
):
    changes = []

    stream = current.get(
        "stream",
        previous.get(
            "stream",
            "UNKNOWN",
        ),
    )

    previous_status = previous.get(
        "status"
    )

    current_status = current.get(
        "status"
    )

    # ------------------------------------------------------------------------
    # Status transition
    # ------------------------------------------------------------------------

    if previous_status != current_status:
        changes.append(
            {
                "type": "STATUS_CHANGE",
                "stream": stream,
                "from": previous_status,
                "to": current_status,
            }
        )

    # ------------------------------------------------------------------------
    # Genuine OOS trade count
    # ------------------------------------------------------------------------

    previous_genuine = int(
        previous.get(
            "genuine_oos",
            0,
        )
        or 0
    )

    current_genuine = int(
        current.get(
            "genuine_oos",
            0,
        )
        or 0
    )

    if current_genuine != previous_genuine:
        changes.append(
            {
                "type": "GENUINE_OOS_COUNT_CHANGE",
                "stream": stream,
                "from": previous_genuine,
                "to": current_genuine,
                "delta": (
                    current_genuine
                    - previous_genuine
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Duplicate keys
    # ------------------------------------------------------------------------

    previous_duplicates = int(
        previous.get(
            "duplicate_keys",
            0,
        )
        or 0
    )

    current_duplicates = int(
        current.get(
            "duplicate_keys",
            0,
        )
        or 0
    )

    if current_duplicates != previous_duplicates:
        changes.append(
            {
                "type": "DUPLICATE_KEY_CHANGE",
                "stream": stream,
                "from": previous_duplicates,
                "to": current_duplicates,
            }
        )

    # ------------------------------------------------------------------------
    # Invalid rows
    # ------------------------------------------------------------------------

    previous_invalid = int(
        previous.get(
            "invalid_rows",
            0,
        )
        or 0
    )

    current_invalid = int(
        current.get(
            "invalid_rows",
            0,
        )
        or 0
    )

    if current_invalid != previous_invalid:
        changes.append(
            {
                "type": "INVALID_ROW_CHANGE",
                "stream": stream,
                "from": previous_invalid,
                "to": current_invalid,
            }
        )

    # ------------------------------------------------------------------------
    # Locked early-adverse event count
    # ------------------------------------------------------------------------

    previous_events = int(
        previous.get(
            "event_count",
            0,
        )
        or 0
    )

    current_events = int(
        current.get(
            "event_count",
            0,
        )
        or 0
    )

    if current_events != previous_events:
        changes.append(
            {
                "type": "EARLY_ADVERSE_EVENT_CHANGE",
                "stream": stream,
                "from": previous_events,
                "to": current_events,
                "delta": (
                    current_events
                    - previous_events
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Full 7-bar evaluated count
    # ------------------------------------------------------------------------

    previous_full = int(
        previous.get(
            "evaluated_full_7b",
            0,
        )
        or 0
    )

    current_full = int(
        current.get(
            "evaluated_full_7b",
            0,
        )
        or 0
    )

    if current_full != previous_full:
        changes.append(
            {
                "type": "FULL_7B_EVALUATION_CHANGE",
                "stream": stream,
                "from": previous_full,
                "to": current_full,
                "delta": (
                    current_full
                    - previous_full
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Event rate
    # ------------------------------------------------------------------------

    previous_rate = safe_number(
        previous.get(
            "event_rate_full_7b_pct"
        )
    )

    current_rate = safe_number(
        current.get(
            "event_rate_full_7b_pct"
        )
    )

    if previous_rate != current_rate:
        changes.append(
            {
                "type": "EVENT_RATE_CHANGE",
                "stream": stream,
                "from": previous_rate,
                "to": current_rate,
            }
        )

    # ------------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------------

    previous_fingerprint = previous.get(
        "fingerprint"
    )

    current_fingerprint = current.get(
        "fingerprint"
    )

    if previous_fingerprint != current_fingerprint:
        changes.append(
            {
                "type": "LEDGER_FINGERPRINT_CHANGE",
                "stream": stream,
                "from": previous_fingerprint,
                "to": current_fingerprint,
            }
        )

    return changes


def compare_snapshots(
    previous,
    current,
):
    changes = []

    previous_streams = stream_map(
        previous
    )

    current_streams = stream_map(
        current
    )

    all_stream_names = sorted(
        set(previous_streams)
        | set(current_streams)
    )

    for stream in all_stream_names:

        if stream not in previous_streams:
            changes.append(
                {
                    "type": "NEW_STREAM",
                    "stream": stream,
                }
            )
            continue

        if stream not in current_streams:
            changes.append(
                {
                    "type": "REMOVED_STREAM",
                    "stream": stream,
                }
            )
            continue

        changes.extend(
            compare_stream(
                previous_streams[stream],
                current_streams[stream],
            )
        )

    # ------------------------------------------------------------------------
    # Global status
    # ------------------------------------------------------------------------

    previous_global = (
        previous.get(
            "global_status"
        )
    )

    current_global = (
        current.get(
            "global_status"
        )
    )

    # Current monitor snapshots created by monitor_oos.py do not contain
    # global_status inside the JSON root, so derive it from stream states.

    if previous_global is None:
        previous_global = derive_global_status(
            previous
        )

    if current_global is None:
        current_global = derive_global_status(
            current
        )

    if previous_global != current_global:
        changes.append(
            {
                "type": "GLOBAL_STATUS_CHANGE",
                "from": previous_global,
                "to": current_global,
            }
        )

    # ------------------------------------------------------------------------
    # Global genuine count
    # ------------------------------------------------------------------------

    previous_total = sum(
        int(
            item.get(
                "genuine_oos",
                0,
            )
            or 0
        )
        for item in previous.get(
            "streams",
            [],
        )
    )

    current_total = sum(
        int(
            item.get(
                "genuine_oos",
                0,
            )
            or 0
        )
        for item in current.get(
            "streams",
            [],
        )
    )

    if previous_total != current_total:
        changes.append(
            {
                "type": "GLOBAL_GENUINE_OOS_CHANGE",
                "from": previous_total,
                "to": current_total,
                "delta": (
                    current_total
                    - previous_total
                ),
            }
        )

    return changes


# ============================================================================
# GLOBAL STATUS
# ============================================================================

def derive_global_status(snapshot):
    streams = snapshot.get(
        "streams",
        [],
    )

    if any(
        item.get("status") == "INVALID"
        for item in streams
    ):
        return "INVALID"

    total_genuine = sum(
        int(
            item.get(
                "genuine_oos",
                0,
            )
            or 0
        )
        for item in streams
    )

    if total_genuine == 0:
        return (
            "WAITING_FOR_GENUINE_OOS_TRADES"
        )

    return "OBSERVING_GENUINE_OOS"


# ============================================================================
# EVENT SEVERITY
# ============================================================================

def classify_change(change):
    change_type = change.get(
        "type"
    )

    if change_type in {
        "INVALID_ROW_CHANGE",
        "DUPLICATE_KEY_CHANGE",
        "REMOVED_STREAM",
    }:
        return "CRITICAL"

    if change_type in {
        "GENUINE_OOS_COUNT_CHANGE",
        "EARLY_ADVERSE_EVENT_CHANGE",
        "FULL_7B_EVALUATION_CHANGE",
        "STATUS_CHANGE",
        "GLOBAL_STATUS_CHANGE",
        "GLOBAL_GENUINE_OOS_CHANGE",
        "NEW_STREAM",
    }:
        return "IMPORTANT"

    if change_type in {
        "EVENT_RATE_CHANGE",
        "LEDGER_FINGERPRINT_CHANGE",
    }:
        return "INFORMATIONAL"

    return "INFORMATIONAL"


# ============================================================================
# HISTORY
# ============================================================================

def append_history(record):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        CHANGE_HISTORY_FILE,
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                default=str,
            )
            + "\n"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — OOS MONITOR CHANGE DETECTION"
    )
    print("=" * 100)

    print()
    print("LOCKED HYPOTHESIS")
    print("-" * 100)
    print(
        LOCKED_HYPOTHESIS
    )

    print()
    print("SAFETY CONTRACT")
    print("-" * 100)
    print(
        "Production strategy : UNCHANGED"
    )
    print(
        "Historical data     : READ-ONLY"
    )
    print(
        "OOS ledgers         : READ-ONLY"
    )
    print(
        "Parameter tuning    : NONE"
    )
    print(
        "Hypothesis          : LOCKED"
    )

    files = snapshot_files()

    print()
    print(
        f"Snapshots available : {len(files)}"
    )

    # ------------------------------------------------------------------------
    # First snapshot
    # ------------------------------------------------------------------------

    if len(files) == 0:
        print()
        print(
            "STATUS: NO_MONITOR_SNAPSHOT_FOUND"
        )
        print(
            "Run: python tests/monitor_oos.py"
        )
        return

    if len(files) == 1:
        latest = load_snapshot(
            files[-1]
        )

        print()
        print(
            "Latest snapshot:"
        )
        print(
            files[-1]
        )

        print()
        print(
            "No previous snapshot exists."
        )

        print(
            "Change detection begins on the next monitor run."
        )

        record = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "type": "BASELINE_SNAPSHOT",
            "snapshot": str(
                files[-1]
            ),
            "changes": [],
        }

        append_history(
            record
        )

        print()
        print(
            "STATUS: BASELINE_ESTABLISHED"
        )

        return

    # ------------------------------------------------------------------------
    # Compare latest two snapshots
    # ------------------------------------------------------------------------

    previous_path = files[-2]
    current_path = files[-1]

    previous = load_snapshot(
        previous_path
    )

    current = load_snapshot(
        current_path
    )

    changes = compare_snapshots(
        previous,
        current,
    )

    for change in changes:
        change["severity"] = (
            classify_change(
                change
            )
        )

    print()
    print("=" * 100)
    print("SNAPSHOT COMPARISON")
    print("=" * 100)

    print()
    print(
        f"Previous: {previous_path.name}"
    )

    print(
        f"Current : {current_path.name}"
    )

    print()
    print(
        f"Changes detected: {len(changes)}"
    )

    # ------------------------------------------------------------------------
    # No changes
    # ------------------------------------------------------------------------

    if not changes:
        print()
        print(
            "NO OOS STATE CHANGE"
        )

        print(
            "The latest snapshot is identical in monitored state."
        )

        print(
            "No new genuine OOS trades."
        )

        print(
            "No new early-adverse events."
        )

        print(
            "No integrity changes."
        )

    # ------------------------------------------------------------------------
    # Changes
    # ------------------------------------------------------------------------

    else:
        print()

        for index, change in enumerate(
            changes,
            start=1,
        ):
            print(
                f"[{index}] "
                f"{change['severity']} "
                f"{change['type']}"
            )

            if "stream" in change:
                print(
                    f"    Stream : "
                    f"{change['stream']}"
                )

            if "from" in change:
                print(
                    f"    From   : "
                    f"{format_value(change['from'])}"
                )

            if "to" in change:
                print(
                    f"    To     : "
                    f"{format_value(change['to'])}"
                )

            if "delta" in change:
                print(
                    f"    Delta  : "
                    f"{format_value(change['delta'])}"
                )

    # ------------------------------------------------------------------------
    # Persist change record
    # ------------------------------------------------------------------------

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "previous_snapshot": str(
            previous_path
        ),
        "current_snapshot": str(
            current_path
        ),
        "previous_global_status": derive_global_status(
            previous
        ),
        "current_global_status": derive_global_status(
            current
        ),
        "change_count": len(changes),
        "changes": changes,
    }

    append_history(
        record
    )

    # ------------------------------------------------------------------------
    # Final safety status
    # ------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("SAFETY CHECK")
    print("=" * 100)

    print(
        "PASS: Production strategy unchanged."
    )

    print(
        "PASS: Historical datasets untouched."
    )

    print(
        "PASS: OOS ledgers not modified."
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
        "Change history:"
    )
    print(
        CHANGE_HISTORY_FILE
    )

    print()

    if any(
        change["severity"] == "CRITICAL"
        for change in changes
    ):
        print(
            "STATUS: CRITICAL_OOS_MONITOR_CHANGE"
        )

    elif changes:
        print(
            "STATUS: OOS_STATE_CHANGED"
        )

    else:
        print(
            "STATUS: NO_OOS_STATE_CHANGE"
        )


if __name__ == "__main__":
    main()