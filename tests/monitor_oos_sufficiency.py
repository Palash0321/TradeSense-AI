from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# TRADE SENSE-AI
# OOS EVIDENCE SUFFICIENCY GATE
# ============================================================
#
# PURPOSE
# -------
# Operational gate determining whether enough genuine
# post-freeze OOS evidence exists to OPEN the future
# evaluation stage.
#
# IMPORTANT
# ---------
# This script is READ-ONLY with respect to:
#   - production strategy
#   - historical datasets
#   - OOS source ledgers
#
# It consumes the AUTHORITATIVE OOS MONITOR output.
#
# It does NOT independently reconstruct or reinterpret
# 7B completeness.
#
# LOCKED HYPOTHESIS
# -----------------
# Early adverse >= 0.25R within first 7 entry bars
#
# SUFFICIENCY IS NOT PERFORMANCE
# ------------------------------
# Reaching the thresholds below does NOT mean:
#   - the hypothesis is accepted
#   - the hypothesis is rejected
#   - the strategy is profitable
#   - the strategy is production-ready
#
# It only means that enough operational OOS evidence exists
# to permit a subsequent evaluation stage.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

MONITOR_FILE = Path(
    "tests/output/research/"
    "oos_monitor/oos_monitor_current.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/"
    "oos_monitor/sufficiency"
)

CURRENT_OUTPUT = (
    OUTPUT_DIR
    / "oos_sufficiency_current.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "oos_sufficiency_summary.json"
)

HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "oos_sufficiency_history.jsonl"
)


# ============================================================
# LOCKED OPERATIONAL THRESHOLDS
# ============================================================

MIN_GENUINE_OOS_TRADES = 30
MIN_COMPLETE_7B = 25
MIN_7B_COMPLETENESS = 0.80


LOCKED_WINDOW_BARS = 7
LOCKED_THRESHOLD_R = 0.25


# ============================================================
# AUTHORITATIVE MONITOR SCHEMA
# ============================================================
#
# These fields must already exist in the OOS Monitor output.
#
# We deliberately DO NOT derive these values from the source
# ledgers here.
# ============================================================

REQUIRED_COLUMNS = [
    "stream",
    "genuine_oos",
    "pre_start",
    "duplicate_keys",
    "invalid_dates",
    "window_bars",
    "threshold_r",
    "master_7b_eligible",
    "event_count",
    "no_event_full_contract",
    "partial_contract_window",
    "missing_path",
    "status",
]


# ============================================================
# HELPERS
# ============================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def numeric_column(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0.0)


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            "AUTHORITATIVE OOS MONITOR SCHEMA FAILURE.\n"
            f"Missing required columns: {missing}\n"
            "Stage 5 will not guess or reconstruct these fields."
        )


def safe_int(value: float) -> int:

    if not np.isfinite(value):
        return 0

    return int(round(float(value)))


def safe_float(value: float):

    if not np.isfinite(value):
        return None

    return float(value)


def append_history(record: dict) -> None:

    HISTORY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HISTORY_OUTPUT.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                record,
                sort_keys=True,
            )
            + "\n"
        )


# ============================================================
# START
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print("TRADE SENSE-AI — OOS EVIDENCE SUFFICIENCY GATE")
print("=" * 100)


# ============================================================
# LOCKED HYPOTHESIS
# ============================================================

print()
print("LOCKED HYPOTHESIS")
print("-" * 100)
print(
    "Early adverse >= "
    f"{LOCKED_THRESHOLD_R:.2f}R within first "
    f"{LOCKED_WINDOW_BARS} entry bars"
)


# ============================================================
# SUFFICIENCY CONTRACT
# ============================================================

print()
print("SUFFICIENCY CONTRACT")
print("-" * 100)

print(
    f"Minimum genuine OOS trades : "
    f"{MIN_GENUINE_OOS_TRADES}"
)

print(
    f"Minimum complete 7B         : "
    f"{MIN_COMPLETE_7B}"
)

print(
    f"Minimum 7B completeness     : "
    f"{MIN_7B_COMPLETENESS:.0%}"
)


# ============================================================
# SAFETY CONTRACT
# ============================================================

print()
print("SAFETY CONTRACT")
print("-" * 100)

print("Production strategy : READ-ONLY")
print("Historical data     : READ-ONLY")
print("OOS source ledgers  : READ-ONLY")
print("Monitor output      : READ-ONLY")
print("Parameter tuning    : NONE")
print("Hypothesis          : LOCKED")
print("Performance gating  : NONE")


# ============================================================
# LOAD AUTHORITATIVE MONITOR
# ============================================================

print()
print("AUTHORITATIVE SOURCE")
print("-" * 100)

print(
    f"Monitor file: "
    f"{MONITOR_FILE}"
)


if not MONITOR_FILE.exists():

    fail(
        "Authoritative OOS Monitor output does not exist.\n"
        "Run tests/monitor_oos.py before Stage 5."
    )


monitor = pd.read_csv(
    MONITOR_FILE
)


print(
    f"Monitor rows    : {len(monitor)}"
)

print(
    f"Monitor columns : {len(monitor.columns)}"
)


# ============================================================
# SCHEMA VALIDATION
# ============================================================

require_columns(
    monitor,
    REQUIRED_COLUMNS,
)


# ============================================================
# NORMALIZATION
# ============================================================

monitor = monitor.copy()

for column in [
    "genuine_oos",
    "pre_start",
    "duplicate_keys",
    "invalid_dates",
    "window_bars",
    "threshold_r",
    "master_7b_eligible",
    "event_count",
    "no_event_full_contract",
    "partial_contract_window",
    "missing_path",
]:

    monitor[column] = numeric_column(
        monitor,
        column,
    )

    monitor["master_7b_eligible"] = (
    monitor["master_7b_eligible"]
    .astype(bool)
    )

monitor["stream"] = (
    monitor["stream"]
    .astype(str)
    .str.strip()
)


# ============================================================
# AUTHORITATIVE OPERATIONAL COUNTS
# ============================================================

total_genuine_oos_all_streams = safe_int(
    monitor["genuine_oos"].sum()
)

total_pre_start = safe_int(
    monitor["pre_start"].sum()
)

total_duplicates = safe_int(
    monitor["duplicate_keys"].sum()
)

total_invalid_dates = safe_int(
    monitor["invalid_dates"].sum()
)

total_contract_events = safe_int(
    monitor["event_count"].sum()
)

total_contract_no_event = safe_int(
    monitor["no_event_full_contract"].sum()
)

total_partial_contract = safe_int(
    monitor["partial_contract_window"].sum()
)

total_missing_path_all_streams = safe_int(
    monitor["missing_path"].sum()
)


# ============================================================
# MASTER 7B ELIGIBLE POPULATION
# ============================================================

master_7b_mask = (
    monitor["master_7b_eligible"].eq(True)
    & monitor["window_bars"].eq(LOCKED_WINDOW_BARS)
    & monitor["threshold_r"].eq(LOCKED_THRESHOLD_R)
)

master_monitor = monitor.loc[
    master_7b_mask
].copy()


# ============================================================
# MASTER 7B COUNTS
# ============================================================

master_7b_genuine_oos = safe_int(
    master_monitor["genuine_oos"].sum()
)

master_7b_events = safe_int(
    master_monitor["event_count"].sum()
)

master_7b_no_event_full = safe_int(
    master_monitor["no_event_full_contract"].sum()
)

master_7b_partial = safe_int(
    master_monitor["partial_contract_window"].sum()
)

master_7b_missing_path = safe_int(
    master_monitor["missing_path"].sum()
)


# ============================================================
# AUTHORITATIVE COMPLETE 7B COUNT
# ============================================================
#
# COMPLETE 7B consists of:
#
#     full 7B event
#     +
#     full 7B no-event
#
# Therefore:
#
#     complete_7b =
#         7B event count
#         +
#         full 7B no-event
#
# This uses the OOS Monitor's own authoritative
# classification.
# ============================================================

# ============================================================
# MASTER 7B COMPLETE COUNT
# ============================================================

master_7b_complete = (
    master_7b_events
    + master_7b_no_event_full
)


# ============================================================
# 7B COMPLETENESS
# ============================================================

if master_7b_genuine_oos > 0:

    master_7b_completeness = (
        master_7b_complete
        / master_7b_genuine_oos
    )

else:

    master_7b_completeness = np.nan


# ============================================================
# INTEGRITY CHECKS
# ============================================================

no_duplicates = (
    total_duplicates == 0
)

no_pre_start_rows = (
    total_pre_start == 0
)

no_invalid_dates = (
    total_invalid_dates == 0
)

master_7b_complete_consistent = (
    master_7b_complete
    + master_7b_partial
    + master_7b_missing_path
    <= master_7b_genuine_oos
)


# ============================================================
# GLOBAL STATUS
# ============================================================

if not (
    no_duplicates
    and no_pre_start_rows
    and no_invalid_dates
    and master_7b_complete_consistent
):

    global_status = (
        "INVALID_OOS_EVIDENCE"
    )

    global_reason = (
        "One or more OOS evidence integrity checks failed."
    )

elif master_7b_genuine_oos == 0:

    global_status = (
        "NO_GENUINE_OOS_EVIDENCE"
    )

    global_reason = (
        "No genuine post-freeze OOS trades exist yet."
    )

elif (
    master_7b_genuine_oos >= MIN_GENUINE_OOS_TRADES
    and master_7b_complete >= MIN_COMPLETE_7B
    and master_7b_completeness >= MIN_7B_COMPLETENESS
):

    global_status = (
        "OOS_EVIDENCE_SUFFICIENT"
    )

    global_reason = (
        "Master-eligible 7B OOS evidence exists and the "
        "operational sufficiency thresholds have been met."
    )

else:

    global_status = (
        "OOS_EVIDENCE_ACCUMULATING"
    )

    global_reason = (
        "Genuine OOS evidence exists, but the operational "
        "sufficiency thresholds have not yet been met."
    )


# ============================================================
# PER-STREAM TABLE
# ============================================================

stream_rows = []

for _, row in monitor.iterrows():

    genuine = safe_int(
        row["genuine_oos"]
    )

    events = safe_int(
        row["event_count"]
    )

    window_bars = safe_int(
        row["window_bars"]
    )

    threshold_r = float(
        row["threshold_r"]
    )

    master_eligible = bool(
        row["master_7b_eligible"]
    )

    full_no_event = safe_int(
        row["no_event_full_contract"]
    )

    partial = safe_int(
        row["partial_contract_window"]
    )

    missing_path = safe_int(
        row["missing_path"]
    )

    complete = (
        events
        + full_no_event
    )

    if genuine > 0:

        stream_completeness = (
            complete
            / genuine
        )

    else:

        stream_completeness = np.nan

    stream_duplicate = safe_int(
        row["duplicate_keys"]
    )

    stream_pre_start = safe_int(
        row["pre_start"]
    )

    stream_invalid_dates = safe_int(
        row["invalid_dates"]
    )

    stream_integrity = (
        stream_duplicate == 0
        and stream_pre_start == 0
        and stream_invalid_dates == 0
        and (
            complete
            + partial
            + missing_path
            <= genuine
        )
    )

    if not stream_integrity:

        stream_status = (
            "INVALID_OOS_EVIDENCE"
        )

        stream_reason = (
            "Integrity issue in authoritative monitor output."
        )

    elif genuine == 0:

        stream_status = (
            "WAITING_FOR_GENUINE_OOS_TRADES"
        )

        stream_reason = (
            "No genuine post-freeze OOS trades."
        )

    elif not master_eligible:

        stream_status = (
            "OPERATIONAL_CONTRACT_ONLY"
        )

        stream_reason = (
            "This stream is monitored under its frozen "
            f"{window_bars}B contract but is excluded from "
            "master 7B sufficiency."
        )

    elif (
        master_eligible
        and window_bars == LOCKED_WINDOW_BARS
        and threshold_r == LOCKED_THRESHOLD_R
        and genuine >= MIN_GENUINE_OOS_TRADES
        and complete >= MIN_COMPLETE_7B
        and stream_completeness >= MIN_7B_COMPLETENESS
    ):

        stream_status = (
            "OOS_EVIDENCE_SUFFICIENT"
        )

        stream_reason = (
            "This master-eligible 7B stream independently meets "
            "the operational sufficiency thresholds."
        )

    else:

        stream_status = (
            "OOS_EVIDENCE_ACCUMULATING"
        )

        stream_reason = (
            "Genuine OOS exists, but this stream has not "
            "met all sufficiency thresholds."
        )

    stream_rows.append(
        {
            "stream": row["stream"],
            "genuine_oos": genuine,

            "window_bars": window_bars,
            "threshold_r": threshold_r,
            "master_7b_eligible": master_eligible,

            "event_count": events,
            "no_event_full_contract": full_no_event,
            "complete_contract": complete,
            "partial_contract_window": partial,
            "missing_path": missing_path,

            "contract_completeness": safe_float(
                stream_completeness
            ),

            "pre_start": stream_pre_start,
            "duplicate_keys": stream_duplicate,
            "invalid_dates": stream_invalid_dates,

            "monitor_status": row["status"],
            "sufficiency_status": stream_status,
            "reason": stream_reason,
        }
    )


stream_df = pd.DataFrame(
    stream_rows
)


# ============================================================
# SAVE CURRENT OUTPUT
# ============================================================

stream_df.to_csv(
    CURRENT_OUTPUT,
    index=False,
)

# ============================================================
# REMAINING MASTER 7B REQUIREMENTS
# ============================================================

additional_master_7b_genuine_oos_needed = max(
    0,
    MIN_GENUINE_OOS_TRADES
    - master_7b_genuine_oos,
)

additional_master_7b_complete_needed = max(
    0,
    MIN_COMPLETE_7B
    - master_7b_complete,
)

if np.isfinite(
    master_7b_completeness
):

    additional_master_7b_completeness_needed = max(
        0.0,
        MIN_7B_COMPLETENESS
        - master_7b_completeness,
    )

else:

    additional_master_7b_completeness_needed = None


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary = {
    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "locked_hypothesis": {
        "window_bars": LOCKED_WINDOW_BARS,
        "threshold_r": LOCKED_THRESHOLD_R,
        "status": "FROZEN",
    },

    "sufficiency_thresholds": {
        "minimum_genuine_oos_trades":
            MIN_GENUINE_OOS_TRADES,

        "minimum_complete_7b":
            MIN_COMPLETE_7B,

        "minimum_7b_completeness":
            MIN_7B_COMPLETENESS,
    },

    "global_counts": {
        "streams_monitored":
            int(len(monitor)),

        "genuine_oos_all_streams":
            total_genuine_oos_all_streams,

        "master_7b_genuine_oos":
            master_7b_genuine_oos,

        "master_7b_events":
            master_7b_events,

        "master_7b_no_event_full":
            master_7b_no_event_full,

        "master_7b_complete":
            master_7b_complete,

        "master_7b_partial":
            master_7b_partial,

        "master_7b_missing_path":
            master_7b_missing_path,

        "contract_events_all_streams":
            total_contract_events,

        "contract_no_event_all_streams":
            total_contract_no_event,

        "partial_contract_all_streams":
            total_partial_contract,

        "missing_path_all_streams":
            total_missing_path_all_streams,

        "pre_start":
            total_pre_start,

        "duplicate_keys":
            total_duplicates,

        "invalid_dates":
            total_invalid_dates,
    },

    "master_7b_completeness":
        safe_float(master_7b_completeness),

    "remaining_requirements": {
        "additional_master_7b_genuine_oos_trades":
            additional_master_7b_genuine_oos_needed,

        "additional_master_7b_complete":
            additional_master_7b_complete_needed,

        "additional_master_7b_completeness":
            additional_master_7b_completeness_needed,
    },

    "integrity": {
        "no_duplicates":
            no_duplicates,

        "no_pre_start":
            no_pre_start_rows,

        "no_invalid_dates":
            no_invalid_dates,

        "master_7b_complete_consistent":
            master_7b_complete_consistent,
    },

    "status":
        global_status,

    "reason":
        global_reason,

    "safety": {
        "production_strategy_modified":
            False,

        "historical_data_modified":
            False,

        "oos_source_ledgers_modified":
            False,

        "hypothesis_modified":
            False,

        "parameter_tuning":
            False,

        "performance_based_stopping":
            False,

        "trade_fabrication":
            False,
    },
}


with SUMMARY_OUTPUT.open(
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        summary,
        handle,
        indent=2,
        sort_keys=True,
    )


append_history(
    summary
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 100)
print("STREAM OPERATIONAL CONTRACTS")
print("=" * 100)

for _, row in stream_df.iterrows():

    print()
    print(
        f"STREAM: {row['stream']}"
    )

    print(
        f"  Genuine OOS          : "
        f"{row['genuine_oos']}"
    )

    print(
        f"  Contract window     : "
        f"{row['window_bars']}B"
    )

    print(
        f"  Threshold           : "
        f"{row['threshold_r']:.2f}R"
    )

    print(
        f"  Master 7B eligible  : "
        f"{row['master_7b_eligible']}"
    )

    print(
        f"  Contract events     : "
        f"{row['event_count']}"
    )

    print(
        f"  Contract no-event   : "
        f"{row['no_event_full_contract']}"
    )

    print(
        f"  Complete contract   : "
        f"{row['complete_contract']}"
    )

    print(
        f"  Partial contract   : "
        f"{row['partial_contract_window']}"
    )

    print(
        f"  Missing path       : "
        f"{row['missing_path']}"
    )

    if pd.isna(
        row["contract_completeness"]
    ):

        completeness_text = "None"

    else:

        completeness_text = (
            f"{row['contract_completeness']:.2%}"
        )

    print(
        f"  Contract completeness: "
        f"{completeness_text}"
    )

    print(
        f"  Sufficiency status  : "
        f"{row['sufficiency_status']}"
    )


# ============================================================
# GLOBAL REPORT
# ============================================================

print()
print("=" * 100)
print("GLOBAL OOS EVIDENCE SUFFICIENCY")
print("=" * 100)

print(
    f"Streams monitored             : "
    f"{len(monitor)}"
)

print(
    f"All-stream genuine OOS        : "
    f"{total_genuine_oos_all_streams}"
)

print()
print("MASTER 7B SUFFICIENCY")
print("-" * 100)

print(
    f"Master 7B genuine OOS         : "
    f"{master_7b_genuine_oos}"
)

print(
    f"Master 7B events              : "
    f"{master_7b_events}"
)

print(
    f"Master 7B no-event            : "
    f"{master_7b_no_event_full}"
)

print(
    f"Master 7B complete            : "
    f"{master_7b_complete}"
)

print(
    f"Master 7B partial             : "
    f"{master_7b_partial}"
)

print(
    f"Master 7B missing path        : "
    f"{master_7b_missing_path}"
)

if pd.isna(
    master_7b_completeness
):

    master_completeness_text = "None"

else:

    master_completeness_text = (
        f"{master_7b_completeness:.2%}"
    )

print(
    f"Master 7B completeness        : "
    f"{master_completeness_text}"
)

print()
print("MASTER 7B REQUIREMENTS")
print("-" * 100)

print(
    f"Minimum genuine OOS          : "
    f"{MIN_GENUINE_OOS_TRADES}"
)

print(
    f"Minimum complete 7B          : "
    f"{MIN_COMPLETE_7B}"
)

print(
    f"Minimum 7B completeness      : "
    f"{MIN_7B_COMPLETENESS:.0%}"
)

print(
    f"Additional genuine OOS       : "
    f"{additional_master_7b_genuine_oos_needed}"
)

print(
    f"Additional complete 7B       : "
    f"{additional_master_7b_complete_needed}"
)

if (
    additional_master_7b_completeness_needed
    is None
):

    additional_completeness_text = "None"

else:

    additional_completeness_text = (
        f"{additional_master_7b_completeness_needed:.2%}"
    )

print(
    f"Additional completeness      : "
    f"{additional_completeness_text}"
)

print()
print("OPERATIONAL CONTRACT TOTALS")
print("-" * 100)

print(
    f"Contract events (all streams): "
    f"{total_contract_events}"
)

print(
    f"Contract no-event (all)      : "
    f"{total_contract_no_event}"
)

print(
    f"Partial contract (all)       : "
    f"{total_partial_contract}"
)

print(
    f"Missing path (all)           : "
    f"{total_missing_path_all_streams}"
)

print()
print("INTEGRITY")
print("-" * 100)

print(
    f"No duplicates                : "
    f"{no_duplicates}"
)

print(
    f"No pre-start rows            : "
    f"{no_pre_start_rows}"
)

print(
    f"No invalid dates             : "
    f"{no_invalid_dates}"
)

print(
    f"Master 7B consistency        : "
    f"{master_7b_complete_consistent}"
)

print()
print(
    f"GLOBAL STATUS                : "
    f"{global_status}"
)

print(
    f"REASON                       : "
    f"{global_reason}"
)

print("=" * 100)


# ============================================================
# SAFETY
# ============================================================

print()
print("=" * 100)
print("SAFETY CHECK")
print("=" * 100)

print(
    "PASS: Production strategy unchanged."
)

print(
    "PASS: Historical datasets remain read-only."
)

print(
    "PASS: OOS source ledgers remain read-only."
)

print(
    "PASS: Authoritative monitor output consumed read-only."
)

print(
    "PASS: Locked 7B / 0.25R hypothesis unchanged."
)

print(
    "PASS: No parameter tuning."
)

print(
    "PASS: No performance-based stopping."
)

print(
    "PASS: No trade fabrication."
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("FINAL OOS EVIDENCE SUFFICIENCY STATUS")
print("=" * 100)

print(
    f"STATUS: {global_status}"
)

print(
    f"REASON: {global_reason}"
)


print()
print(
    "IMPORTANT:"
)

print(
    "Sufficiency is an operational evidence gate only."
)

print(
    "It does NOT accept or reject the 7B/0.25R hypothesis."
)

print(
    "It does NOT establish profitability."
)

print(
    "It does NOT authorize production deployment."
)

print(
    "Future evaluation remains separately gated."
)


print()
print(
    "Outputs:"
)

print(
    f"  Current : {CURRENT_OUTPUT}"
)

print(
    f"  Summary : {SUMMARY_OUTPUT}"
)

print(
    f"  History : {HISTORY_OUTPUT}"
)

print()
print(
    "OOS EVIDENCE SUFFICIENCY GATE: "
    f"{'PASS' if global_status != 'INVALID_OOS_EVIDENCE' else 'FAIL'}"
)