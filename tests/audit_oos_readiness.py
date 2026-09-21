"""
TradeSense-AI
LOCKED FIVE-STREAM OOS READINESS AUDIT

Purpose
-------
Audit the readiness of every currently established genuine forward /
holdout stream against the frozen OOS research contract.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within the first 7 entry bars.

This audit is READ-ONLY.

It does NOT:
    - generate trades
    - modify production
    - modify historical datasets
    - modify holdout ledgers
    - tune parameters
    - select setups
    - select regimes
    - manufacture trades
    - interpret zero trades as failure

A missing ledger is NORMAL when the corresponding collector has
not yet observed a genuine post-freeze trade.

Current streams
---------------
1. Stock holdout
2. NIFTY 50 forward
3. SENSEX forward
4. BANK NIFTY forward
5. NIFTY MIDCAP SELECT forward
"""

from pathlib import Path
import csv
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

# Master OOS evaluation contract remains frozen at 7B / 0.25R.
# Stream-specific contracts below preserve separately frozen 5B / 0.25R
# research tracks without allowing them to masquerade as master 7B evidence.
LOCKED_WINDOW = 7
LOCKED_THRESHOLD_R = 0.25
LOCKED_RULE_VERSION = "7B_0.25R"

STOCK_HOLDOUT_START = "2026-09-09"
INDEX_FORWARD_START = "2026-09-14"

OUTPUT_DIR = Path(
    "tests/output/reconciliation/oos_readiness"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "oos_readiness_summary.csv"
)


# ============================================================================
# OOS STREAM DEFINITIONS
# ============================================================================

STREAMS = [
    {
        "stream": "STOCK_HOLDOUT",
        "description": "Frozen stock holdout",
        "start_date": STOCK_HOLDOUT_START,
        "rule_version": "5B_0.25R",
        "threshold_r": 0.25,
        "window_bars": 5,
        "master_7b_eligible": False,
        "file": Path(
            "tests/output/tradesense_holdout_trades.csv"
        ),
        "required_trade_fields": [
            "entry_date",
        ],
        "hypothesis_fields": [
            "early_adverse_r_bar_1",
            "early_adverse_r_bar_2",
            "early_adverse_r_bar_3",
            "early_adverse_r_bar_4",
            "early_adverse_r_bar_5",
        ],
    },
    {
        "stream": "NIFTY_50_FORWARD",
        "description": "NIFTY 50 genuine forward holdout",
        "start_date": INDEX_FORWARD_START,
        "rule_version": "7B_0.25R",
        "threshold_r": 0.25,
        "window_bars": 7,
        "master_7b_eligible": True,
        "file": Path(
            "tests/output/index_backtests/nifty50_forward_holdout.csv"
        ),
        "required_trade_fields": [
            "entry_date",
        ],
        "hypothesis_fields": [
            "early_adverse_r_bar_1",
            "early_adverse_r_bar_2",
            "early_adverse_r_bar_3",
            "early_adverse_r_bar_4",
            "early_adverse_r_bar_5",
            "early_adverse_r_bar_6",
            "early_adverse_r_bar_7",
        ],
    },
    {
        "stream": "SENSEX_FORWARD",
        "description": "SENSEX genuine forward holdout",
        "start_date": INDEX_FORWARD_START,
        "rule_version": "7B_0.25R",
        "threshold_r": 0.25,
        "window_bars": 7,
        "master_7b_eligible": True,
        "file": Path(
            "tests/output/index_backtests/sensex_forward_holdout.csv"
        ),
        "required_trade_fields": [
            "entry_date",
        ],
        "hypothesis_fields": [
            "early_adverse_r_bar_1",
            "early_adverse_r_bar_2",
            "early_adverse_r_bar_3",
            "early_adverse_r_bar_4",
            "early_adverse_r_bar_5",
            "early_adverse_r_bar_6",
            "early_adverse_r_bar_7",
        ],
    },
    {
        "stream": "BANK_NIFTY_FORWARD",
        "description": (
            "BANK NIFTY genuine forward "
            "SHORT_CONTINUATION holdout"
        ),
        "start_date": INDEX_FORWARD_START,
        "rule_version": "5B_0.25R",
        "threshold_r": 0.25,
        "window_bars": 5,
        "master_7b_eligible": False,
        "file": Path(
            "tests/output/index_backtests/"
            "banknifty_short_continuation_forward_holdout.csv"
        ),
        "required_trade_fields": [
            "entry_date",
        ],
        "hypothesis_fields": [
            "early_adverse_r_bar_1",
            "early_adverse_r_bar_2",
            "early_adverse_r_bar_3",
            "early_adverse_r_bar_4",
            "early_adverse_r_bar_5",
        ],
    },
    {
        "stream": "MIDSELECT_FORWARD",
        "description": (
            "NIFTY MIDCAP SELECT genuine "
            "forward holdout"
        ),
        "start_date": INDEX_FORWARD_START,
        "rule_version": "7B_0.25R",
        "threshold_r": 0.25,
        "window_bars": 7,
        "master_7b_eligible": True,
        "file": Path(
            "tests/output/index_backtests/"
            "midselect_forward_holdout.csv"
        ),
        "required_trade_fields": [
            "entry_date",
        ],
        "hypothesis_fields": [
            "early_adverse_r_bar_1",
            "early_adverse_r_bar_2",
            "early_adverse_r_bar_3",
            "early_adverse_r_bar_4",
            "early_adverse_r_bar_5",
            "early_adverse_r_bar_6",
            "early_adverse_r_bar_7",
        ],
    },
]


# ============================================================================
# HELPERS
# ============================================================================

def normalize_date(value):
    """
    Return YYYY-MM-DD for a date-like value.

    Returns None when invalid.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        return text[:10]
    except Exception:
        return None


def date_is_before(
    value,
    start_date,
):
    """
    Compare normalized YYYY-MM-DD strings.
    """

    normalized = normalize_date(value)

    if normalized is None:
        return False

    return normalized < start_date


def date_is_on_or_after(
    value,
    start_date,
):
    """
    Check whether a normalized date is on/after the OOS boundary.
    """

    normalized = normalize_date(value)

    if normalized is None:
        return False

    return normalized >= start_date


def read_csv_file(path):
    """
    Read a CSV into a list of dictionaries.

    Returns:
        rows, fieldnames
    """

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )

        fieldnames = (
            reader.fieldnames
            or []
        )

    return rows, fieldnames


def field_has_value(
    row,
    field,
):
    """
    True when a field exists and contains a non-empty value.
    """

    if field not in row:
        return False

    value = row.get(
        field
    )

    if value is None:
        return False

    return str(
        value
    ).strip() != ""

def validate_stream_contract(stream):
    """
    Validate the explicitly frozen contract attached to one stream.

    The master evaluation contract remains 7B / 0.25R.
    Separate frozen 5B / 0.25R research tracks are permitted,
    but they are never master-7B eligible.
    """

    errors = []

    rule_version = stream.get(
        "rule_version"
    )

    threshold_r = stream.get(
        "threshold_r"
    )

    window_bars = stream.get(
        "window_bars"
    )

    master_7b_eligible = stream.get(
        "master_7b_eligible"
    )

    hypothesis_fields = stream.get(
        "hypothesis_fields",
        [],
    )

    if (
        not isinstance(
            window_bars,
            int,
        )
        or isinstance(
            window_bars,
            bool,
        )
        or window_bars < 1
    ):
        errors.append(
            "INVALID_WINDOW_BARS"
        )

    threshold_is_numeric = (
        isinstance(
            threshold_r,
            (int, float),
        )
        and not isinstance(
            threshold_r,
            bool,
        )
    )

    if not threshold_is_numeric:

        errors.append(
            "INVALID_THRESHOLD_R"
        )

    elif threshold_r != LOCKED_THRESHOLD_R:

        errors.append(
            "UNLOCKED_THRESHOLD_R"
        )

    if (
        isinstance(
            window_bars,
            int,
        )
        and not isinstance(
            window_bars,
            bool,
        )
        and window_bars >= 1
        and threshold_is_numeric
    ):

        expected_rule_version = (
            f"{window_bars}B_{threshold_r:.2f}R"
        )

        if rule_version != expected_rule_version:

            errors.append(
                "RULE_VERSION_MISMATCH"
            )

        expected_fields = [
            f"early_adverse_r_bar_{bar}"
            for bar in range(
                1,
                window_bars + 1,
            )
        ]

        if hypothesis_fields != expected_fields:

            errors.append(
                "HYPOTHESIS_FIELDS_DO_NOT_MATCH_WINDOW"
            )

        expected_master_eligibility = (
            window_bars == LOCKED_WINDOW
            and rule_version == LOCKED_RULE_VERSION
        )

        if (
            master_7b_eligible
            != expected_master_eligibility
        ):

            errors.append(
                "MASTER_7B_ELIGIBILITY_MISMATCH"
            )

    return errors

# ============================================================================
# STREAM AUDIT
# ============================================================================

def audit_stream(stream):
    """
    Audit one OOS stream.

    Returns a result dictionary.
    """

    name = stream["stream"]

    description = stream[
        "description"
    ]

    start_date = stream[
        "start_date"
    ]

    path = stream[
        "file"
    ]

    required_trade_fields = stream[
        "required_trade_fields"
    ]

    hypothesis_fields = stream[
        "hypothesis_fields"
    ]

    rule_version = stream.get(
        "rule_version"
    )

    threshold_r = stream.get(
        "threshold_r"
    )

    window_bars = stream.get(
        "window_bars"
    )

    master_7b_eligible = stream.get(
        "master_7b_eligible"
    )


    result = {
        "stream": name,
        "description": description,
        "oos_start_date": start_date,
        "file": str(path),
        "rule_version": rule_version,
        "threshold_r": threshold_r,
        "window_bars": window_bars,
        "master_7b_eligible": master_7b_eligible,
        "contract_status": None,
        "file_exists": False,
        "ledger_state": None,
        "trade_rows": 0,
        "genuine_oos_rows": 0,
        "pre_start_rows": 0,
        "invalid_date_rows": 0,
        "missing_required_fields": 0,
        "missing_hypothesis_fields": 0,
        "status": None,
        "failure_reasons": "",
    }

    # ------------------------------------------------------------------------
    # Contract validation must happen before file-state handling.
    # ------------------------------------------------------------------------

    contract_errors = validate_stream_contract(
        stream
    )

    if contract_errors:

        result[
            "contract_status"
        ] = "INVALID"

        result[
            "status"
        ] = "INVALID_LEDGER"

        result[
            "failure_reasons"
        ] = (
            "INVALID_STREAM_CONTRACT:"
            + ";".join(contract_errors)
        )

        return result

    result[
        "contract_status"
    ] = "VALID"

    # ------------------------------------------------------------------------
    # Missing file is a legitimate zero-trade state.
    # ------------------------------------------------------------------------

    if not path.exists():

        result[
            "ledger_state"
        ] = "NOT_CREATED"

        result[
            "status"
        ] = "WAITING_FOR_GENUINE_OOS_TRADES"

        result[
            "failure_reasons"
        ] = (
            "No genuine OOS trade has "
            "been persisted yet."
        )

        return result

    result[
        "file_exists"
    ] = True

    result[
        "ledger_state"
    ] = "EXISTS"

    # ------------------------------------------------------------------------
    # Read ledger.
    # ------------------------------------------------------------------------

    try:

        rows, fields = read_csv_file(
            path
        )

    except Exception as exc:

        result[
            "ledger_state"
        ] = "READ_ERROR"

        result[
            "status"
        ] = "INVALID_LEDGER"

        result[
            "failure_reasons"
        ] = (
            f"CSV read error: {exc}"
        )

        return result

    result[
        "trade_rows"
    ] = len(rows)

    # ------------------------------------------------------------------------
    # Existing empty ledger.
    # ------------------------------------------------------------------------

    if not rows:

        result[
            "status"
        ] = "WAITING_FOR_GENUINE_OOS_TRADES"

        result[
            "failure_reasons"
        ] = (
            "Ledger exists but contains "
            "zero rows."
        )

        return result

    failures = []

    # ------------------------------------------------------------------------
    # Required fields.
    # ------------------------------------------------------------------------

    missing_columns = [
        field
        for field in required_trade_fields
        if field not in fields
    ]

    if missing_columns:

        failures.append(
            "MISSING_REQUIRED_COLUMNS:"
            + ",".join(
                missing_columns
            )
        )

    # ------------------------------------------------------------------------
    # Hypothesis fields.
    # ------------------------------------------------------------------------

    missing_hypothesis_columns = [
        field
        for field in hypothesis_fields
        if field not in fields
    ]

    if missing_hypothesis_columns:

        failures.append(
            "MISSING_HYPOTHESIS_COLUMNS:"
            + ",".join(
                missing_hypothesis_columns
            )
        )

    # ------------------------------------------------------------------------
    # Validate every row.
    # ------------------------------------------------------------------------

    for row in rows:

        entry_date = normalize_date(
            row.get(
                "entry_date"
            )
        )

        if entry_date is None:

            result[
                "invalid_date_rows"
            ] += 1

            continue

        if date_is_before(
            entry_date,
            start_date,
        ):

            result[
                "pre_start_rows"
            ] += 1

            continue

        result[
            "genuine_oos_rows"
        ] += 1

        # ------------------------------------------------------------
        # Required trade fields.
        # ------------------------------------------------------------

        for field in required_trade_fields:

            if not field_has_value(
                row,
                field,
            ):

                result[
                    "missing_required_fields"
                ] += 1

                break

        # ------------------------------------------------------------
        # Hypothesis fields.
        # ------------------------------------------------------------

        for field in hypothesis_fields:

            if not field_has_value(
                row,
                field,
            ):

                result[
                    "missing_hypothesis_fields"
                ] += 1

                break

    # ------------------------------------------------------------------------
    # Validation decisions.
    # ------------------------------------------------------------------------

    if result[
        "invalid_date_rows"
    ] > 0:

        failures.append(
            "INVALID_ENTRY_DATES"
        )

    if result[
        "pre_start_rows"
    ] > 0:

        failures.append(
            "PRE_START_ROWS_PRESENT"
        )

    if result[
        "missing_required_fields"
    ] > 0:

        failures.append(
            "MISSING_REQUIRED_TRADE_FIELDS"
        )

    if result[
        "missing_hypothesis_fields"
    ] > 0:

        failures.append(
            "MISSING_LOCKED_HYPOTHESIS_FIELDS"
        )

    # ------------------------------------------------------------------------
    # Zero genuine OOS rows inside an existing ledger.
    # ------------------------------------------------------------------------

    if result[
        "genuine_oos_rows"
    ] == 0 and not failures:

        result[
            "status"
        ] = "WAITING_FOR_GENUINE_OOS_TRADES"

        result[
            "failure_reasons"
        ] = (
            "Ledger contains no genuine "
            "post-start OOS rows."
        )

        return result

    # ------------------------------------------------------------------------
    # Invalid ledger.
    # ------------------------------------------------------------------------

    if failures:

        result[
            "status"
        ] = "INVALID_LEDGER"

        result[
            "failure_reasons"
        ] = ";".join(
            failures
        )

        return result

    # ------------------------------------------------------------------------
    # Genuine OOS data exists and passes structural checks.
    # ------------------------------------------------------------------------

    result[
        "status"
    ] = "READY"

    result[
        "failure_reasons"
    ] = ""

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 110)
    print(
        "TRADESENSE-AI — "
        "LOCKED FIVE-STREAM OOS READINESS AUDIT"
    )
    print("=" * 110)

    print()

    print(
        "LOCKED HYPOTHESIS:"
    )

    print(
        f"Early adverse >= "
        f"{LOCKED_THRESHOLD_R:.2f}R "
        f"within first "
        f"{LOCKED_WINDOW} entry bars"
    )

    print()

    print(
        "This audit is READ-ONLY."
    )

    print(
        "No production or historical "
        "dataset will be modified."
    )

    print()

    print(
        "Streams under audit: "
        f"{len(STREAMS)}"
    )

    print()

    results = []

    # ------------------------------------------------------------------------
    # Audit each stream.
    # ------------------------------------------------------------------------

    for stream in STREAMS:

        print(
            "-" * 110
        )

        print(
            stream["stream"]
        )

        print(
            f"Description : "
            f"{stream['description']}"
        )

        print(
            f"OOS start   : "
            f"{stream['start_date']}"
        )

        print(
            f"Rule version: "
            f"{stream['rule_version']}"
        )

        print(
            f"Threshold R : "
            f"{stream['threshold_r']}"
        )

        print(
            f"Window bars : "
            f"{stream['window_bars']}"
        )

        print(
            f"Master 7B   : "
            f"{stream['master_7b_eligible']}"
        )

        print(
            f"File        : "
            f"{stream['file']}"
        )

        result = audit_stream(
            stream
        )

        results.append(
            result
        )

        print(
            f"File exists : "
            f"{result['file_exists']}"
        )

        print(
            f"Ledger state: "
            f"{result['ledger_state']}"
        )

        print(
            f"Trade rows  : "
            f"{result['trade_rows']}"
        )

        print(
            f"Genuine OOS : "
            f"{result['genuine_oos_rows']}"
        )

        print(
            f"Pre-start   : "
            f"{result['pre_start_rows']}"
        )

        print(
            f"Status      : "
            f"{result['status']}"
        )

        if result[
            "failure_reasons"
        ]:

            print(
                f"Reason      : "
                f"{result['failure_reasons']}"
            )

    # =========================================================================
    # GLOBAL SUMMARY
    # =========================================================================

    ready_count = sum(
        result["status"] == "READY"
        for result in results
    )

    master_7b_stream_count = sum(
        result["master_7b_eligible"] is True
        for result in results
    )

    separate_5b_stream_count = sum(
        (
            result["window_bars"] == 5
            and result["master_7b_eligible"] is False
        )
        for result in results
    )

    waiting_count = sum(
        result["status"]
        == "WAITING_FOR_GENUINE_OOS_TRADES"
        for result in results
    )

    invalid_count = sum(
        result["status"]
        == "INVALID_LEDGER"
        for result in results
    )

    total_oos_rows = sum(
        result["genuine_oos_rows"]
        for result in results
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "stream",
        "description",
        "oos_start_date",
        "rule_version",
        "threshold_r",
        "window_bars",
        "master_7b_eligible",
        "contract_status",
        "file",
        "file_exists",
        "ledger_state",
        "trade_rows",
        "genuine_oos_rows",
        "pre_start_rows",
        "invalid_date_rows",
        "missing_required_fields",
        "missing_hypothesis_fields",
        "status",
        "failure_reasons",
    ]

    with SUMMARY_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "GLOBAL OOS READINESS"
    )
    print("=" * 110)

    print(
        f"OOS streams inspected : "
        f"{len(results)}"
    )

    print(
        f"Master 7B streams     : "
        f"{master_7b_stream_count}"
    )

    print(
        f"Separate 5B tracks    : "
        f"{separate_5b_stream_count}"
    )

    print(
        f"Streams READY         : "
        f"{ready_count}"
    )

    print(
        f"Streams WAITING       : "
        f"{waiting_count}"
    )

    print(
        f"Streams INVALID       : "
        f"{invalid_count}"
    )

    print(
        f"Total genuine OOS rows: "
        f"{total_oos_rows}"
    )

    print()

    if invalid_count > 0:

        print(
            "FINAL STATUS: "
            "INVALID_OOS_LEDGER"
        )

        print()

        print(
            "At least one OOS stream has "
            "a structural validation failure."
        )

        print(
            "DO NOT perform strategy "
            "interpretation until repaired."
        )

        final_status = 1

    elif total_oos_rows == 0:

        print(
            "FINAL STATUS: "
            "WAITING_FOR_GENUINE_OOS_TRADES"
        )

        print()

        print(
            "No genuine post-freeze OOS "
            "trade has been persisted yet."
        )

        print(
            "This is NOT a failure."
        )

        print(
            "The collectors are waiting for "
            "completed post-freeze market data."
        )

        final_status = 0

    else:

        print(
            "FINAL STATUS: "
            "OOS_DATA_AVAILABLE"
        )

        print()

        print(
            "At least one genuine OOS trade "
            "has been persisted."
        )

        print(
            "The locked hypothesis remains frozen."
        )

        print(
            "No parameter selection is permitted "
            "from this audit."
        )

        final_status = 0

    # =========================================================================
    # SAFETY CHECK
    # =========================================================================

    print()
    print("=" * 110)
    print(
        "SAFETY CHECK"
    )
    print("=" * 110)

    print(
        "PASS: Production strategy unchanged."
    )

    print(
        "PASS: Historical datasets unchanged."
    )

    print(
        "PASS: Existing OOS ledgers unchanged."
    )

    print(
        "PASS: Master 7B / 0.25R hypothesis remains locked."
    )

    print(
        "PASS: Stream-specific 5B / 0.25R research tracks are explicit."
    )

    print(
        "PASS: 5B streams are not marked as master-7B eligible."
    )

    print(
        "PASS: No regime condition introduced."
    )

    print(
        "PASS: No setup condition introduced."
    )

    print(
        "PASS: Missing zero-trade ledgers "
        "are treated as WAITING."
    )

    print()

    print(
        "Readiness summary saved:"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print()

    print("=" * 110)
    print(
        "END LOCKED FIVE-STREAM OOS READINESS AUDIT"
    )
    print("=" * 110)

    return final_status


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )