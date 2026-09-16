"""
TradeSense-AI
Frozen OOS Collection Runner

Purpose:
    Run all currently established forward/OOS collectors and then
    run the locked five-stream OOS readiness audit.

IMPORTANT:
    - No strategy tuning.
    - No regime filters.
    - No setup filters.
    - No modification of frozen historical datasets.
    - No modification of production strategy logic.
    - Zero-trade forward streams are intentionally allowed to have
      no ledger file.
    - The locked hypothesis remains:
          Early adverse >= 0.25R
          within first 7 entry bars.

Streams:
    1. Stock holdout
    2. NIFTY 50 forward
    3. SENSEX forward
    4. BANK NIFTY forward
    5. NIFTY MIDCAP SELECT forward
"""

import os
import sys
import subprocess
from datetime import datetime


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# LOCKED RESEARCH CONTRACT
# ============================================================

LOCKED_WINDOW = 7
LOCKED_THRESHOLD_R = 0.25


# ============================================================
# COLLECTORS
# ============================================================

COLLECTORS = [
    (
        "STOCK HOLDOUT",
        os.path.join(
            PROJECT_ROOT,
            "tests",
            "test_trend_pipeline.py",
        ),
    ),
    (
        "NIFTY 50 FORWARD",
        os.path.join(
            PROJECT_ROOT,
            "tests",
            "forward_nifty50_holdout.py",
        ),
    ),
    (
        "SENSEX FORWARD",
        os.path.join(
            PROJECT_ROOT,
            "tests",
            "forward_sensex_holdout.py",
        ),
    ),
    (
        "BANK NIFTY FORWARD",
        os.path.join(
            PROJECT_ROOT,
            "tests",
            "forward_banknifty_short_continuation_holdout.py",
        ),
    ),
    (
        "MIDSELECT FORWARD",
        os.path.join(
            PROJECT_ROOT,
            "tests",
            "forward_midselect_holdout.py",
        ),
    ),
]


# ============================================================
# READINESS AUDIT
# ============================================================

READINESS_AUDIT = os.path.join(
    PROJECT_ROOT,
    "tests",
    "audit_oos_readiness.py",
)


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print()
    print("=" * 78)
    print()


def run_script(label, script_path):
    """
    Run one project test/collector from the project root.

    Returns:
        True  -> process completed successfully
        False -> process failed
    """

    print_separator()

    print(f"RUNNING: {label}")
    print(f"Script  : {os.path.relpath(script_path, PROJECT_ROOT)}")

    if not os.path.exists(script_path):
        print("STATUS  : MISSING_SCRIPT")
        print("This collector has not been established locally.")
        return False

    started = datetime.now()

    try:
        result = subprocess.run(
            [
                sys.executable,
                script_path,
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except Exception as exc:
        print()
        print(f"STATUS  : EXECUTION_ERROR")
        print(f"ERROR   : {exc}")
        return False

    elapsed = datetime.now() - started

    print()
    print(f"RETURN CODE: {result.returncode}")
    print(f"ELAPSED    : {elapsed}")

    if result.returncode != 0:
        print(f"STATUS     : FAILED — {label}")
        return False

    print(f"STATUS     : PASS — {label}")
    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("TRADESENSE-AI — FROZEN OOS COLLECTION RUNNER")
    print("=" * 78)

    print()
    print("LOCKED HYPOTHESIS")
    print("-----------------")
    print(
        f"Early adverse >= {LOCKED_THRESHOLD_R:.2f}R "
        f"within first {LOCKED_WINDOW} entry bars"
    )

    print()
    print("RESEARCH CONTRACT")
    print("-----------------")
    print("Production strategy : UNCHANGED")
    print("Historical datasets : READ-ONLY")
    print("OOS hypothesis      : LOCKED")
    print("Regime filters      : NONE")
    print("Setup filters       : NONE")
    print("Parameter tuning    : NONE")
    print("Zero-trade ledger   : NOT FABRICATED")

    print()
    print("COLLECTORS")
    print("----------")

    for label, script_path in COLLECTORS:
        print(
            f"- {label}: "
            f"{os.path.relpath(script_path, PROJECT_ROOT)}"
        )

    print()
    print(
        f"- OOS READINESS AUDIT: "
        f"{os.path.relpath(READINESS_AUDIT, PROJECT_ROOT)}"
    )

    results = []

    # --------------------------------------------------------
    # RUN ALL COLLECTORS
    # --------------------------------------------------------

    for label, script_path in COLLECTORS:
        success = run_script(
            label,
            script_path,
        )

        results.append(
            (
                label,
                success,
            )
        )

        if not success:
            print()
            print(
                f"WARNING: {label} failed or is unavailable."
            )
            print(
                "Continuing so the final readiness audit can "
                "report the current state."
            )

    # --------------------------------------------------------
    # RUN READINESS AUDIT
    # --------------------------------------------------------

    audit_success = run_script(
        "FIVE-STREAM OOS READINESS AUDIT",
        READINESS_AUDIT,
    )

    results.append(
        (
            "FIVE-STREAM OOS READINESS AUDIT",
            audit_success,
        )
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_separator()

    print("=" * 78)
    print("FINAL COLLECTION RUN SUMMARY")
    print("=" * 78)

    passed = 0
    failed = 0

    for label, success in results:

        if success:
            status = "PASS"
            passed += 1
        else:
            status = "FAILED / UNAVAILABLE"
            failed += 1

        print(
            f"{status:<22} {label}"
        )

    print()
    print(f"Components passed : {passed}")
    print(f"Components failed : {failed}")

    print()
    print("IMPORTANT INTERPRETATION")
    print("------------------------")
    print(
        "A collector reporting zero genuine OOS trades is NOT a failure."
    )
    print(
        "It means completed post-freeze data has not produced a genuine"
    )
    print(
        "post-start trade yet."
    )

    print()
    print("OOS GATE")
    print("--------")
    print(
        "Do NOT evaluate, retune, reject, or accept the 7B/.25R"
    )
    print(
        "hypothesis until genuine post-freeze trades exist."
    )

    print()
    print("LOCKED STATE")
    print("------------")
    print(
        f"Window    : {LOCKED_WINDOW} bars"
    )
    print(
        f"Threshold : {LOCKED_THRESHOLD_R:.2f}R"
    )
    print(
        "Status    : FROZEN"
    )

    print()
    print("=" * 78)
    print("OOS COLLECTION RUN COMPLETE")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()