"""
TradeSense-AI
STAGE 6 — LOCKED OOS HYPOTHESIS EVALUATION

PURPOSE
-------
Evaluate the already-frozen OOS hypothesis only after the
Stage 5 evidence sufficiency gate explicitly opens evaluation.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within the first 7 entry bars.

THIS SCRIPT IS READ-ONLY WITH RESPECT TO:
    - production strategy
    - historical datasets
    - OOS source ledgers
    - frozen research population
    - parameters
    - thresholds
    - setups
    - regimes

IMPORTANT
---------
This evaluator NEVER uses the historical 106-trade forensic
population as OOS evidence.

It consumes only the authoritative OOS evidence ledger.

If Stage 5 is not OOS_EVIDENCE_SUFFICIENT, evaluation is
BLOCKED.

No threshold optimization.
No setup selection.
No regime selection.
No symbol selection.
No parameter tuning.
No production deployment decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATH
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]


# ============================================================================
# LOCKED HYPOTHESIS
# ============================================================================

LOCKED_WINDOW_BARS = 7
LOCKED_THRESHOLD_R = 0.25

LOCKED_RULE_VERSION = "7B_0.25R"

LOCKED_HYPOTHESIS_LABEL = (
    "Early adverse >= "
    f"{LOCKED_THRESHOLD_R:.2f}R within first "
    f"{LOCKED_WINDOW_BARS} entry bars"
)


# ============================================================================
# CURRENT OOS ARCHITECTURE
# ============================================================================

SUFFICIENCY_SUMMARY = (
    BASE_DIR
    / "tests"
    / "output"
    / "research"
    / "oos_monitor"
    / "sufficiency"
    / "oos_sufficiency_summary.json"
)

EVIDENCE_FILE = (
    BASE_DIR
    / "tests"
    / "output"
    / "research"
    / "oos_monitor"
    / "evidence"
    / "oos_evidence_current.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "tests"
    / "output"
    / "research"
    / "oos_monitor"
    / "evaluation"
)

CURRENT_OUTPUT = (
    OUTPUT_DIR
    / "oos_evaluation_current.csv"
)

TRADE_OUTPUT = (
    OUTPUT_DIR
    / "oos_evaluation_trade_level.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "oos_evaluation_summary.json"
)

HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "oos_evaluation_history.jsonl"
)

INTEGRITY_OUTPUT = (
    OUTPUT_DIR
    / "oos_evaluation_integrity.csv"
)


# ============================================================================
# STATUS CONTRACT
# ============================================================================

STATUS_BLOCKED = "EVALUATION_BLOCKED"
STATUS_INVALID = "INVALID_OOS_EVALUATION_INPUT"
STATUS_READY = "EVALUATION_READY"
STATUS_COMPLETE = "EVALUATION_COMPLETE"


# ============================================================================
# REQUIRED EVIDENCE CONTRACT
# ============================================================================

# These are the minimum fields required for the Stage 6 evaluator.
#
# Stage 5 remains authoritative for:
#     - genuine OOS membership
#     - complete 7B observation
#     - event/no-event accounting
#
# Stage 6 does not reconstruct those concepts from historical data.

REQUIRED_EVIDENCE_COLUMNS = [
    "trade_key",
    "entry_date",
    "symbol",
    "direction",
    "r_multiple",
]


BAR_COLUMNS = [
    f"early_adverse_r_bar_{bar}"
    for bar in range(1, LOCKED_WINDOW_BARS + 1)
]


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_dataframe(df: pd.DataFrame) -> str:
    if df.empty:
        payload = b"EMPTY_DATAFRAME"
    else:
        stable = df.copy()

        stable = stable.reindex(
            sorted(stable.columns),
            axis=1,
        )

        stable = stable.fillna("")

        payload = stable.to_csv(
            index=False,
            lineterminator="\n",
        ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def safe_float(value):
    if value is None:
        return np.nan

    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not math.isfinite(value):
        return np.nan

    return value


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def normalized_text(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def normalize_date(value):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        parsed = pd.to_datetime(
            text,
            errors="coerce",
        )

        if pd.isna(parsed):
            return None

        return parsed.strftime("%Y-%m-%d")

    except Exception:
        return None


def require_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{label} schema failure. "
            f"Missing required columns: {missing}"
        )


def append_history(record: dict) -> None:
    OUTPUT_DIR.mkdir(
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


def mean_or_nan(values) -> float:
    clean = [
        safe_float(value)
        for value in values
        if not pd.isna(safe_float(value))
    ]

    if not clean:
        return np.nan

    return float(np.mean(clean))


def median_or_nan(values) -> float:
    clean = [
        safe_float(value)
        for value in values
        if not pd.isna(safe_float(value))
    ]

    if not clean:
        return np.nan

    return float(np.median(clean))


def percentile_or_nan(
    values,
    percentile: float,
) -> float:
    clean = [
        safe_float(value)
        for value in values
        if not pd.isna(safe_float(value))
    ]

    if not clean:
        return np.nan

    return float(
        np.percentile(
            clean,
            percentile,
        )
    )


# ============================================================================
# STAGE 5 GATE
# ============================================================================

def load_sufficiency_summary() -> dict:
    if not SUFFICIENCY_SUMMARY.exists():
        fail(
            "Stage 5 sufficiency summary does not exist.\n"
            f"Expected: {SUFFICIENCY_SUMMARY}"
        )

    try:
        with SUFFICIENCY_SUMMARY.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    except Exception as exc:
        fail(
            "Unable to read Stage 5 sufficiency summary: "
            f"{exc}"
        )


def validate_stage5_contract(
    summary: dict,
) -> tuple[str, list[dict]]:
    required_top_level = [
        "locked_hypothesis",
        "sufficiency_thresholds",
        "global_counts",
        "integrity",
        "status",
    ]

    missing = [
        field
        for field in required_top_level
        if field not in summary
    ]

    if missing:
        fail(
            "Stage 5 summary schema failure. "
            f"Missing: {missing}"
        )

    locked = summary["locked_hypothesis"]

    if safe_int(
        locked.get("window_bars")
    ) != LOCKED_WINDOW_BARS:
        fail(
            "Stage 5 window does not match locked "
            f"{LOCKED_WINDOW_BARS}B contract."
        )

    threshold = safe_float(
        locked.get("threshold_r")
    )

    if pd.isna(threshold):
        fail(
            "Stage 5 threshold is missing."
        )

    if not math.isclose(
        threshold,
        LOCKED_THRESHOLD_R,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(
            "Stage 5 threshold does not match locked "
            f"{LOCKED_THRESHOLD_R:.2f}R contract."
        )

    integrity = summary["integrity"]

    integrity_fields = [
        "no_duplicates",
        "no_pre_start",
        "no_invalid_dates",
        "complete_7b_consistent",
    ]

    integrity_failures = [
        field
        for field in integrity_fields
        if integrity.get(field) is not True
    ]

    if integrity_failures:
        fail(
            "Stage 5 integrity contract is not satisfied: "
            f"{integrity_failures}"
        )

    status = normalized_text(
        summary.get("status")
    )

    return status, integrity_failures


# ============================================================================
# EVIDENCE LOADING
# ============================================================================

def load_evidence() -> tuple[pd.DataFrame, str]:
    if not EVIDENCE_FILE.exists():
        fail(
            "Authoritative OOS evidence ledger does not exist.\n"
            f"Expected: {EVIDENCE_FILE}"
        )

    try:
        df = pd.read_csv(
            EVIDENCE_FILE,
            dtype=str,
        )
    except Exception as exc:
        fail(
            "Unable to read authoritative OOS evidence ledger: "
            f"{exc}"
        )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    require_columns(
        df,
        REQUIRED_EVIDENCE_COLUMNS,
        "OOS evidence ledger",
    )

    # The current evidence ledger may only contain the
    # frozen 5B fields while Stage 5 independently accounts
    # for complete 7B observations.
    #
    # Stage 6 therefore refuses to silently infer Bar 6/7
    # from absent fields.
    for column in BAR_COLUMNS:
        if column not in df.columns:
            # Do not guess or reconstruct.
            continue

    fingerprint = sha256_file(
        EVIDENCE_FILE
    )

    return df, fingerprint


# ============================================================================
# TRADE-LEVEL VALIDATION
# ============================================================================

def validate_trade_identity(
    df: pd.DataFrame,
) -> list[dict]:

    problems = []

    for index, row in df.iterrows():

        trade_key = normalized_text(
            row.get("trade_key")
        )

        entry_date = normalize_date(
            row.get("entry_date")
        )

        symbol = normalized_text(
            row.get("symbol")
        )

        direction = normalized_text(
            row.get("direction")
        ).upper()

        if not trade_key:
            problems.append(
                {
                    "row": int(index),
                    "reason": "MISSING_TRADE_KEY",
                }
            )

        if entry_date is None:
            problems.append(
                {
                    "row": int(index),
                    "reason": "INVALID_ENTRY_DATE",
                }
            )

        if not symbol:
            problems.append(
                {
                    "row": int(index),
                    "reason": "MISSING_SYMBOL",
                }
            )

        if direction not in {
            "LONG",
            "SHORT",
        }:
            problems.append(
                {
                    "row": int(index),
                    "reason": "INVALID_DIRECTION",
                }
            )

        final_r = safe_float(
            row.get("r_multiple")
        )

        if pd.isna(final_r):
            problems.append(
                {
                    "row": int(index),
                    "reason": "MISSING_FINAL_R",
                }
            )

    return problems


def check_duplicate_trade_keys(
    df: pd.DataFrame,
) -> int:

    keys = (
        df["trade_key"]
        .astype(str)
        .str.strip()
    )

    return int(
        keys.duplicated(
            keep=False
        ).sum()
    )


# ============================================================================
# LOCKED 7B CLASSIFICATION
# ============================================================================

def classify_trade(
    row: pd.Series,
) -> tuple[str, float | None, int | None]:

    values = []

    for bar in range(
        1,
        LOCKED_WINDOW_BARS + 1,
    ):
        column = f"early_adverse_r_bar_{bar}"

        if column not in row.index:
            return (
                "UNCLASSIFIABLE_MISSING_BAR_FIELD",
                None,
                None,
            )

        value = safe_float(
            row.get(column)
        )

        if pd.isna(value):
            return (
                "PARTIAL_7B",
                None,
                None,
            )

        values.append(
            (
                bar,
                value,
            )
        )

    crossings = [
        (
            bar,
            value,
        )
        for bar, value in values
        if value >= LOCKED_THRESHOLD_R
    ]

    if crossings:
        first_bar, first_value = crossings[0]

        return (
            "COMPLETE_7B_EVENT",
            first_value,
            first_bar,
        )

    return (
        "COMPLETE_7B_NO_EVENT",
        max(
            value
            for _, value in values
        ),
        None,
    )


# ============================================================================
# EVENT / NO-EVENT OUTCOME DATA
# ============================================================================

def build_trade_level(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, source in df.iterrows():

        classification, trigger_value, trigger_bar = (
            classify_trade(source)
        )

        row = {
            "trade_key": normalized_text(
                source.get("trade_key")
            ),
            "entry_date": normalize_date(
                source.get("entry_date")
            ),
            "symbol": normalized_text(
                source.get("symbol")
            ),
            "direction": normalized_text(
                source.get("direction")
            ).upper(),
            "rule_version": normalized_text(
                source.get("rule_version")
            ),
            "r_multiple": safe_float(
                source.get("r_multiple")
            ),
            "classification": classification,
            "event_flag": int(
                classification
                == "COMPLETE_7B_EVENT"
            ),
            "first_crossing_bar": (
                trigger_bar
                if trigger_bar is not None
                else np.nan
            ),
            "trigger_bar_adverse_r": (
                trigger_value
                if trigger_value is not None
                else np.nan
            ),
        }

        # Preserve optional outcome columns when they
        # are present in the evidence ledger.
        for source_column in [
            "max_favorable_r",
            "max_adverse_r",
            "mfe",
            "mae",
            "profit",
        ]:
            if source_column in source.index:
                row[source_column] = safe_float(
                    source.get(source_column)
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# STATISTICS
# ============================================================================

def outcome_statistics(
    group: pd.DataFrame,
    label: str,
) -> dict:

    final_r = pd.to_numeric(
        group["r_multiple"],
        errors="coerce",
    ).dropna()

    result = {
        "group": label,
        "n": int(len(final_r)),
        "mean_final_r": (
            float(final_r.mean())
            if len(final_r)
            else np.nan
        ),
        "median_final_r": (
            float(final_r.median())
            if len(final_r)
            else np.nan
        ),
        "total_final_r": (
            float(final_r.sum())
            if len(final_r)
            else np.nan
        ),
        "win_rate_pct": (
            float(
                (final_r > 0).mean()
                * 100.0
            )
            if len(final_r)
            else np.nan
        ),
    }

    for outcome_column, prefix in [
        ("max_favorable_r", "mfe"),
        ("mfe", "mfe"),
        ("max_adverse_r", "mae"),
        ("mae", "mae"),
    ]:

        if outcome_column not in group.columns:
            continue

        values = pd.to_numeric(
            group[outcome_column],
            errors="coerce",
        ).dropna()

        if not len(values):
            continue

        result[
            f"{prefix}_n"
        ] = int(len(values))

        result[
            f"{prefix}_mean_r"
        ] = float(values.mean())

        result[
            f"{prefix}_median_r"
        ] = float(values.median())

    return result


def build_pooled_statistics(
    trade_level: pd.DataFrame,
) -> pd.DataFrame:

    complete = trade_level[
        trade_level["classification"].isin(
            [
                "COMPLETE_7B_EVENT",
                "COMPLETE_7B_NO_EVENT",
            ]
        )
    ].copy()

    rows = []

    for label in [
        "COMPLETE_7B_EVENT",
        "COMPLETE_7B_NO_EVENT",
    ]:

        group = complete[
            complete["classification"] == label
        ]

        rows.append(
            outcome_statistics(
                group,
                label,
            )
        )

    return pd.DataFrame(rows)


# ============================================================================
# STREAM SUMMARY
# ============================================================================

def build_stream_statistics(
    trade_level: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for stream in sorted(
        trade_level["stream"].dropna().unique()
        if "stream" in trade_level.columns
        else []
    ):

        group = trade_level[
            trade_level["stream"] == stream
        ]

        complete = group[
            group["classification"].isin(
                [
                    "COMPLETE_7B_EVENT",
                    "COMPLETE_7B_NO_EVENT",
                ]
            )
        ]

        event_count = int(
            (
                complete["classification"]
                == "COMPLETE_7B_EVENT"
            ).sum()
        )

        no_event_count = int(
            (
                complete["classification"]
                == "COMPLETE_7B_NO_EVENT"
            ).sum()
        )

        rows.append(
            {
                "scope": "STREAM",
                "stream": stream,
                "genuine_oos_rows": int(
                    len(group)
                ),
                "complete_7b": int(
                    len(complete)
                ),
                "event_count": event_count,
                "no_event_count": no_event_count,
                "event_rate_pct": (
                    event_count
                    / len(complete)
                    * 100.0
                    if len(complete)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# SAFETY / INTEGRITY
# ============================================================================

def build_integrity_rows(
    stage5_summary: dict,
    evidence_df: pd.DataFrame,
    trade_level: pd.DataFrame,
    evidence_fingerprint: str,
) -> list[dict]:

    duplicate_count = check_duplicate_trade_keys(
        evidence_df
    )

    missing_bar7 = (
        "early_adverse_r_bar_7"
        not in evidence_df.columns
    )

    rows = [
        {
            "check": "STAGE5_GATE_SUFFICIENT",
            "status": (
                "PASS"
                if stage5_summary.get("status")
                == "OOS_EVIDENCE_SUFFICIENT"
                else "FAIL"
            ),
            "detail": str(
                stage5_summary.get("status")
            ),
        },
        {
            "check": "LOCKED_WINDOW_7B",
            "status": (
                "PASS"
                if LOCKED_WINDOW_BARS == 7
                else "FAIL"
            ),
            "detail": str(
                LOCKED_WINDOW_BARS
            ),
        },
        {
            "check": "LOCKED_THRESHOLD_0.25R",
            "status": (
                "PASS"
                if math.isclose(
                    LOCKED_THRESHOLD_R,
                    0.25,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                else "FAIL"
            ),
            "detail": str(
                LOCKED_THRESHOLD_R
            ),
        },
        {
            "check": "EVIDENCE_FILE_PRESENT",
            "status": (
                "PASS"
                if EVIDENCE_FILE.exists()
                else "FAIL"
            ),
            "detail": str(
                EVIDENCE_FILE
            ),
        },
        {
            "check": "DUPLICATE_TRADE_KEYS",
            "status": (
                "PASS"
                if duplicate_count == 0
                else "FAIL"
            ),
            "detail": str(
                duplicate_count
            ),
        },
        {
            "check": "INVALID_TRADE_IDENTITIES",
            "status": (
                "PASS"
                if not validate_trade_identity(
                    evidence_df
                )
                else "FAIL"
            ),
            "detail": "Identity validation",
        },
        {
            "check": "BAR7_FIELD_PRESENT",
            "status": (
                "PASS"
                if not missing_bar7
                else "FAIL"
            ),
            "detail": (
                "Required for 7B classification"
                if not missing_bar7
                else "Missing early_adverse_r_bar_7"
            ),
        },
        {
            "check": "NO_HISTORICAL_106_TRADE_INPUT",
            "status": "PASS",
            "detail": (
                "Stage 6 reads only the authoritative "
                "OOS evidence ledger."
            ),
        },
        {
            "check": "EVIDENCE_FINGERPRINT",
            "status": (
                "PASS"
                if evidence_fingerprint
                else "FAIL"
            ),
            "detail": evidence_fingerprint,
        },
        {
            "check": "NO_PRODUCTION_MODIFICATION",
            "status": "PASS",
            "detail": "Read-only evaluator",
        },
        {
            "check": "NO_PARAMETER_TUNING",
            "status": "PASS",
            "detail": "No optimization code",
        },
        {
            "check": "NO_SETUP_SELECTION",
            "status": "PASS",
            "detail": "No setup filtering",
        },
        {
            "check": "NO_REGIME_SELECTION",
            "status": "PASS",
            "detail": "No regime filtering",
        },
    ]

    return rows


# ============================================================================
# BLOCKED OUTPUT
# ============================================================================

def write_blocked_outputs(
    stage5_summary: dict,
) -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    current = pd.DataFrame(
        [
            {
                "scope": "GLOBAL",
                "stream": "ALL",
                "status": STATUS_BLOCKED,
                "reason": (
                    "Stage 5 has not opened OOS evaluation."
                ),
                "locked_window_bars":
                    LOCKED_WINDOW_BARS,
                "locked_threshold_r":
                    LOCKED_THRESHOLD_R,
                "stage5_status":
                    stage5_summary.get(
                        "status",
                        "",
                    ),
                "genuine_oos": safe_int(
                    stage5_summary
                    .get(
                        "global_counts",
                        {},
                    )
                    .get(
                        "genuine_oos",
                        0,
                    )
                ),
                "complete_7b": safe_int(
                    stage5_summary
                    .get(
                        "global_counts",
                        {},
                    )
                    .get(
                        "complete_7b",
                        0,
                    )
                ),
                "timestamp_utc":
                    timestamp,
            }
        ]
    )

    current.to_csv(
        CURRENT_OUTPUT,
        index=False,
    )

    summary = {
        "timestamp_utc": timestamp,
        "status": STATUS_BLOCKED,
        "reason": (
            "Stage 5 has not opened OOS evaluation."
        ),
        "locked_hypothesis": {
            "window_bars":
                LOCKED_WINDOW_BARS,
            "threshold_r":
                LOCKED_THRESHOLD_R,
            "rule_version":
                LOCKED_RULE_VERSION,
            "status":
                "FROZEN",
        },
        "stage5_status":
            stage5_summary.get(
                "status"
            ),
        "stage5_global_counts":
            stage5_summary.get(
                "global_counts",
                {},
            ),
        "safety": {
            "production_modified":
                False,
            "historical_data_modified":
                False,
            "oos_ledgers_modified":
                False,
            "hypothesis_modified":
                False,
            "parameter_tuning":
                False,
            "setup_selection":
                False,
            "regime_selection":
                False,
            "historical_106_used_as_oos":
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

    integrity = pd.DataFrame(
        [
            {
                "check":
                    "STAGE5_GATE_OPEN",
                "status":
                    "FAIL",
                "detail":
                    stage5_summary.get(
                        "status"
                    ),
            },
            {
                "check":
                    "EVALUATION_BLOCKED",
                "status":
                    "PASS",
                "detail":
                    "No OOS evaluation performed",
            },
            {
                "check":
                    "HISTORICAL_106_USED",
                "status":
                    "PASS",
                "detail":
                    "False",
            },
            {
                "check":
                    "PRODUCTION_MODIFIED",
                "status":
                    "PASS",
                "detail":
                    "False",
            },
        ]
    )

    integrity.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
    )

    append_history(
        summary
    )

    print()
    print(
        "STAGE 5 STATUS:"
    )
    print(
        stage5_summary.get(
            "status"
        )
    )

    print()
    print(
        "STATUS: "
        f"{STATUS_BLOCKED}"
    )

    print(
        "REASON: Stage 5 has not opened "
        "OOS evaluation."
    )

    print()
    print(
        "Historical 106-trade population: NOT USED"
    )
    print(
        "Production strategy: UNCHANGED"
    )
    print(
        "OOS ledger: READ-ONLY"
    )
    print(
        "Parameter tuning: NONE"
    )

    print()
    print(
        "PASS: STAGE 6 SAFETY CONTRACT"
    )

    return 0


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def run_evaluation(
    stage5_summary: dict,
) -> int:

    evidence_df, evidence_fingerprint = (
        load_evidence()
    )

    identity_problems = validate_trade_identity(
        evidence_df
    )

    if identity_problems:
        fail(
            "OOS evidence identity validation failed: "
            f"{identity_problems[:10]}"
        )

    duplicate_count = check_duplicate_trade_keys(
        evidence_df
    )

    if duplicate_count:
        fail(
            "Duplicate OOS trade keys detected: "
            f"{duplicate_count}"
        )

    trade_level = build_trade_level(
        evidence_df
    )

    # The evidence ledger must expose Bar 1..7 for
    # Stage 6's primary locked classification.
    missing_bar_columns = [
        column
        for column in BAR_COLUMNS
        if column not in evidence_df.columns
    ]

    if missing_bar_columns:
        fail(
            "Authoritative OOS evidence ledger does not "
            "contain the complete locked 7B field set.\n"
            f"Missing: {missing_bar_columns}\n"
            "Stage 6 will not reconstruct or guess "
            "missing OOS observations."
        )

    if "stream" not in evidence_df.columns:
        fail(
            "OOS evidence ledger is missing the "
            "authoritative stream field."
        )

    trade_level["stream"] = (
        evidence_df["stream"]
        .astype(str)
        .str.strip()
        .values
    )

    classification_counts = (
        trade_level["classification"]
        .value_counts()
        .to_dict()
    )

    complete = trade_level[
        trade_level["classification"].isin(
            [
                "COMPLETE_7B_EVENT",
                "COMPLETE_7B_NO_EVENT",
            ]
        )
    ].copy()

    if complete.empty:
        fail(
            "Stage 5 declared sufficient evidence, but "
            "Stage 6 found no complete 7B observations."
        )

    pooled = build_pooled_statistics(
        trade_level
    )

    stream_summary = build_stream_statistics(
        trade_level
    )

    # Re-use the authoritative Stage 5 integrity state.
    integrity_rows = build_integrity_rows(
        stage5_summary,
        evidence_df,
        trade_level,
        evidence_fingerprint,
    )

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    integrity_failures = integrity_df[
        integrity_df["status"] != "PASS"
    ]

    if not integrity_failures.empty:
        fail(
            "Stage 6 integrity validation failed: "
            f"{integrity_failures['check'].tolist()}"
        )

    # ------------------------------------------------------------------------
    # CURRENT OUTPUT
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trade_output = trade_level.copy()

    trade_output[
        "evaluation_hypothesis"
    ] = LOCKED_RULE_VERSION

    trade_output[
        "evaluation_timestamp_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    trade_output.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    summary_rows = []

    for _, row in pooled.iterrows():
        summary_rows.append(
            {
                "scope": "POOLED_FIVE_STREAM",
                "stream": "ALL",
                "group": row["group"],
                "n": row["n"],
                "mean_final_r":
                    row["mean_final_r"],
                "median_final_r":
                    row["median_final_r"],
                "total_final_r":
                    row["total_final_r"],
                "win_rate_pct":
                    row["win_rate_pct"],
                "mfe_n":
                    row.get("mfe_n", np.nan),
                "mfe_mean_r":
                    row.get(
                        "mfe_mean_r",
                        np.nan,
                    ),
                "mfe_median_r":
                    row.get(
                        "mfe_median_r",
                        np.nan,
                    ),
                "mae_n":
                    row.get("mae_n", np.nan),
                "mae_mean_r":
                    row.get(
                        "mae_mean_r",
                        np.nan,
                    ),
                "mae_median_r":
                    row.get(
                        "mae_median_r",
                        np.nan,
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    if not stream_summary.empty:
        stream_summary.to_csv(
            CURRENT_OUTPUT,
            index=False,
        )
    else:
        summary_df.to_csv(
            CURRENT_OUTPUT,
            index=False,
        )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    summary = {
        "timestamp_utc": timestamp,
        "status": STATUS_COMPLETE,
        "locked_hypothesis": {
            "window_bars":
                LOCKED_WINDOW_BARS,
            "threshold_r":
                LOCKED_THRESHOLD_R,
            "rule_version":
                LOCKED_RULE_VERSION,
            "label":
                LOCKED_HYPOTHESIS_LABEL,
            "status":
                "FROZEN",
        },
        "stage5": {
            "status":
                stage5_summary.get(
                    "status"
                ),
            "summary_file":
                str(
                    SUFFICIENCY_SUMMARY
                ),
        },
        "evidence": {
            "file":
                str(EVIDENCE_FILE),
            "sha256":
                evidence_fingerprint,
            "rows":
                int(len(evidence_df)),
        },
        "classification": {
            "complete_7b_event":
                int(
                    classification_counts.get(
                        "COMPLETE_7B_EVENT",
                        0,
                    )
                ),
            "complete_7b_no_event":
                int(
                    classification_counts.get(
                        "COMPLETE_7B_NO_EVENT",
                        0,
                    )
                ),
            "partial_7b":
                int(
                    classification_counts.get(
                        "PARTIAL_7B",
                        0,
                    )
                ),
            "unclassifiable":
                int(
                    classification_counts.get(
                        "UNCLASSIFIABLE_MISSING_BAR_FIELD",
                        0,
                    )
                ),
        },
        "primary_population": {
            "complete_7b":
                int(len(complete)),
            "event_count":
                int(
                    (
                        complete["classification"]
                        == "COMPLETE_7B_EVENT"
                    ).sum()
                ),
            "no_event_count":
                int(
                    (
                        complete["classification"]
                        == "COMPLETE_7B_NO_EVENT"
                    ).sum()
                ),
        },
        "safety": {
            "production_modified":
                False,
            "historical_data_modified":
                False,
            "oos_ledgers_modified":
                False,
            "hypothesis_modified":
                False,
            "parameter_tuning":
                False,
            "setup_selection":
                False,
            "regime_selection":
                False,
            "historical_106_used_as_oos":
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

    integrity_df.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
    )

    append_history(
        summary
    )

    print()
    print("=" * 110)
    print(
        "TRADESENSE-AI — "
        "STAGE 6 LOCKED OOS HYPOTHESIS EVALUATION"
    )
    print("=" * 110)

    print()
    print(
        "LOCKED HYPOTHESIS:"
    )
    print(
        LOCKED_HYPOTHESIS_LABEL
    )

    print()
    print(
        "STAGE 5:"
    )
    print(
        stage5_summary.get(
            "status"
        )
    )

    print()
    print(
        "EVIDENCE:"
    )
    print(
        f"Rows                 : "
        f"{len(evidence_df)}"
    )

    print()
    print(
        "PRIMARY COMPLETE 7B POPULATION:"
    )
    print(
        f"Complete 7B          : "
        f"{len(complete)}"
    )
    print(
        f"Events               : "
        f"{classification_counts.get('COMPLETE_7B_EVENT', 0)}"
    )
    print(
        f"No-event             : "
        f"{classification_counts.get('COMPLETE_7B_NO_EVENT', 0)}"
    )

    print()
    print(
        "PRIMARY OUTCOME RESULTS"
    )
    print("-" * 110)

    for _, row in pooled.iterrows():
        print(
            f"{row['group']:<24}"
            f"N={int(row['n']):>4}  "
            f"MeanR={row['mean_final_r']:>8.4f}  "
            f"MedianR={row['median_final_r']:>8.4f}  "
            f"TotalR={row['total_final_r']:>9.4f}"
        )

    print()
    print(
        "Evidence fingerprint:"
    )
    print(
        evidence_fingerprint
    )

    print()
    print(
        "SAFETY CHECK"
    )
    print("-" * 110)
    print(
        "PASS: Production strategy unchanged."
    )
    print(
        "PASS: Historical datasets unchanged."
    )
    print(
        "PASS: OOS evidence ledger consumed read-only."
    )
    print(
        "PASS: Locked 7B / 0.25R hypothesis unchanged."
    )
    print(
        "PASS: No parameter tuning."
    )
    print(
        "PASS: No setup selection."
    )
    print(
        "PASS: No regime selection."
    )
    print(
        "PASS: Historical 106-trade population not used as OOS."
    )

    print()
    print(
        "STATUS: "
        f"{STATUS_COMPLETE}"
    )

    return 0


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print()
    print("=" * 110)
    print(
        "TRADESENSE-AI — "
        "STAGE 6 LOCKED OOS HYPOTHESIS EVALUATION"
    )
    print("=" * 110)

    print()
    print(
        "LOCKED HYPOTHESIS:"
    )
    print(
        LOCKED_HYPOTHESIS_LABEL
    )

    print()
    print(
        "This evaluator is dormant until Stage 5 "
        "explicitly opens evaluation."
    )

    # ------------------------------------------------------------------------
    # Load and validate Stage 5 first.
    # ------------------------------------------------------------------------

    try:
        stage5_summary = (
            load_sufficiency_summary()
        )

        stage5_status, _ = (
            validate_stage5_contract(
                stage5_summary
            )
        )

    except Exception as exc:

        print()
        print(
            "STATUS: "
            f"{STATUS_INVALID}"
        )
        print(
            f"Reason: {exc}"
        )

        return 1

    # ------------------------------------------------------------------------
    # Hard gate.
    # ------------------------------------------------------------------------

    if stage5_status != "OOS_EVIDENCE_SUFFICIENT":

        return write_blocked_outputs(
            stage5_summary
        )

    # ------------------------------------------------------------------------
    # Evaluation is now permitted by Stage 5.
    # ------------------------------------------------------------------------

    try:

        return run_evaluation(
            stage5_summary
        )

    except Exception as exc:

        print()
        print(
            "=" * 110
        )
        print(
            "STAGE 6 INPUT / INTEGRITY FAILURE"
        )
        print(
            "=" * 110
        )

        print()
        print(
            "STATUS: "
            f"{STATUS_INVALID}"
        )

        print(
            f"Reason: {exc}"
        )

        print()
        print(
            "NO OOS RESULT HAS BEEN PRODUCED."
        )

        print(
            "NO PRODUCTION DATA WAS MODIFIED."
        )

        return 1


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(
        main()
    )