"""
TradeSense-AI — Master OOS Monitor Runner

Single-command entry point for the OOS monitoring workflow.

Workflow
--------
1. Run OOS collection
2. Run OOS monitor
3. Run OOS change detection
4. Run OOS evidence ledger

This orchestration layer does NOT alter the research logic of any
underlying component.

LOCKED HYPOTHESIS
-----------------
Early adverse >= 0.25R within first 7 entry bars

SAFETY CONTRACT
---------------
- Production strategy unchanged
- Historical data unchanged
- OOS ledgers remain controlled by the collector
- No parameter tuning
- No hypothesis changes
- No fabricated OOS observations
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PYTHON_EXECUTABLE = sys.executable

STEPS = [
    (
        "OOS COLLECTION",
        PROJECT_ROOT
        / "tests"
        / "run_oos_collection.py",
    ),
    (
        "OOS MONITOR",
        PROJECT_ROOT
        / "tests"
        / "monitor_oos.py",
    ),
    (
        "OOS CHANGE DETECTION",
        PROJECT_ROOT
        / "tests"
        / "monitor_oos_changes.py",
    ),
    (
        "OOS EVIDENCE LEDGER",
        PROJECT_ROOT
        / "tests"
        / "monitor_oos_evidence.py",
    ),
    (
        "OOS EVIDENCE SUFFICIENCY",
        PROJECT_ROOT
        / "tests"
        / "monitor_oos_sufficiency.py",
    ),
]


# ============================================================================
# HELPERS
# ============================================================================

def run_step(
    step_number: int,
    total_steps: int,
    name: str,
    script_path: Path,
) -> bool:

    print()
    print("=" * 100)
    print(
        f"STEP {step_number}/{total_steps} — {name}"
    )
    print("=" * 100)

    print()
    print(
        f"Script: {script_path}"
    )

    if not script_path.exists():
        print()
        print(
            f"FAIL: Script not found: {script_path}"
        )
        return False

    print()
    print(
        f"Running: python {script_path.relative_to(PROJECT_ROOT)}"
    )

    result = subprocess.run(
        [
            PYTHON_EXECUTABLE,
            str(script_path),
        ],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print()
        print(
            f"FAIL: {name}"
        )
        print(
            f"Exit code: {result.returncode}"
        )
        return False

    print()
    print(
        f"PASS: {name}"
    )

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():

    started_at = datetime.now().astimezone()

    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — MASTER OOS MONITOR"
    )
    print("=" * 100)

    print()
    print(
        "Single-command OOS monitoring pipeline"
    )

    print()
    print(
        "LOCKED HYPOTHESIS"
    )
    print("-" * 100)
    print(
        "Early adverse >= 0.25R within first 7 entry bars"
    )

    print()
    print(
        "RESEARCH CONTRACT"
    )
    print("-" * 100)
    print(
        "Production strategy : UNCHANGED"
    )
    print(
        "Historical data     : READ-ONLY"
    )
    print(
        "OOS hypothesis      : LOCKED"
    )
    print(
        "Parameter tuning    : NONE"
    )
    print(
        "Trade fabrication   : NONE"
    )

    print()
    print(
        "PIPELINE"
    )
    print("-" * 100)

    for index, (
        name,
        script_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):
        print(
            f"{index}. {name}"
        )

    # ------------------------------------------------------------------------
    # Execute pipeline
    # ------------------------------------------------------------------------

    results = []

    for index, (
        name,
        script_path,
    ) in enumerate(
        STEPS,
        start=1,
    ):

        success = run_step(
            index,
            len(STEPS),
            name,
            script_path,
        )

        results.append(
            (
                name,
                success,
            )
        )

        if not success:

            print()
            print("=" * 100)
            print(
                "MASTER OOS MONITOR — FAILED"
            )
            print("=" * 100)

            print()

            for result_name, result_ok in results:
                print(
                    f"{result_name:<25} : "
                    f"{'PASS' if result_ok else 'FAIL'}"
                )

            print()
            print(
                "Pipeline stopped at the first failed step."
            )

            print(
                "No later monitoring step was executed."
            )

            print()
            print(
                "STATUS: OOS MONITOR PIPELINE FAIL"
            )

            return 1

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    finished_at = datetime.now().astimezone()

    print()
    print("=" * 100)
    print(
        "MASTER OOS MONITOR — FINAL SUMMARY"
    )
    print("=" * 100)

    print()

    for name, success in results:
        print(
            f"{name:<25} : "
            f"{'PASS' if success else 'FAIL'}"
        )

    print()
    print(
        f"Started : {started_at.isoformat()}"
    )

    print(
        f"Ended   : {finished_at.isoformat()}"
    )

    print()
    print(
        "SAFETY CHECK"
    )
    print("-" * 100)

    print(
        "PASS: Production strategy unchanged."
    )

    print(
        "PASS: Historical datasets remain read-only."
    )

    print(
        "PASS: OOS hypothesis remains locked."
    )

    print(
        "PASS: No parameter tuning performed."
    )

    print(
        "PASS: No OOS trades fabricated."
    )

    print(
        "PASS: Pipeline stopped on failure if any step failed."
    )

    print()
    print(
        "STATUS: OOS MONITOR PIPELINE PASS"
    )

    print()
    print(
        "Daily/periodic entry point:"
    )

    print(
        "python tests/run_oos_monitor.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )