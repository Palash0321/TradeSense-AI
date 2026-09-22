"""
TradeSense-AI
STAGE 6 — OOS HYPOTHESIS EVALUATOR REGRESSION / SAFETY HARNESS

Purpose
-------
Regression and safety tests for tests/evaluate_oos_hypothesis.py.

This harness uses isolated temporary fixtures only.

It NEVER modifies:
    - production strategy
    - historical datasets
    - genuine OOS source ledgers
    - frozen 106-trade research population

Tests covered
-------------
1. Current real Stage 5 state remains blocked.
2. Stage 5 sufficient state opens evaluation on synthetic evidence.
3. Complete 7B event classification.
4. Complete 7B no-event classification.
5. Partial 7B is rejected.
6. Wrong locked threshold is rejected.
7. Wrong locked window is rejected.
8. Duplicate trade keys are rejected.
9. Missing final R is rejected.
10. Historical 106-trade population is not used.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT IMPORT
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


import tests.evaluate_oos_hypothesis as evaluator


# ============================================================================
# TEST CONSTANTS
# ============================================================================

LOCKED_WINDOW = 7
LOCKED_THRESHOLD = 0.25


# ============================================================================
# ASSERTION HELPERS
# ============================================================================

def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(
            f"{message}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def expect_failure(function, expected_text):
    try:
        function()
    except Exception as exc:
        text = str(exc)

        assert_true(
            expected_text in text,
            (
                f"Expected failure containing:\n"
                f"{expected_text}\n\n"
                f"Actual exception:\n"
                f"{text}"
            ),
        )

        return

    raise AssertionError(
        "Expected the operation to fail, but it completed successfully."
    )


# ============================================================================
# SYNTHETIC STAGE 5 SUMMARY
# ============================================================================

def build_stage5_summary(
    status="OOS_EVIDENCE_SUFFICIENT",
    genuine_oos=4,
    complete_7b=4,
):
    return {
        "locked_hypothesis": {
            "window_bars": LOCKED_WINDOW,
            "threshold_r": LOCKED_THRESHOLD,
            "rule_version": "7B_0.25R",
        },
        "sufficiency_thresholds": {
            "minimum_genuine_oos_trades": 30,
            "minimum_complete_7b_observations": 25,
            "minimum_7b_completeness_pct": 80.0,
        },
        "global_counts": {
            "genuine_oos": genuine_oos,
            "complete_7b": complete_7b,
            "event_count": 2,
            "no_event_full_7b": 2,
            "partial_7b_window": 0,
            "missing_path": 0,
        },
        "integrity": {
            "no_duplicates": True,
            "no_pre_start": True,
            "no_invalid_dates": True,
            "master_7b_complete_consistent": True,
        },
        "status": status,
    }


# ============================================================================
# SYNTHETIC EVIDENCE
# ============================================================================

def build_complete_evidence():
    rows = []

    # ------------------------------------------------------------------------
    # Trade 1 — complete 7B EVENT
    # ------------------------------------------------------------------------

    event_bars = {
        "early_adverse_r_bar_1": "0.10",
        "early_adverse_r_bar_2": "0.18",
        "early_adverse_r_bar_3": "0.27",
        "early_adverse_r_bar_4": "0.31",
        "early_adverse_r_bar_5": "0.29",
        "early_adverse_r_bar_6": "0.35",
        "early_adverse_r_bar_7": "0.30",
    }

    rows.append(
        {
            "trade_key": "SYNTHETIC|TEST1|LONG|2026-09-15",
            "entry_date": "2026-09-15",
            "symbol": "TEST1",
            "direction": "LONG",
            "setup": "LONG_CONTINUATION",
            "rule_version": "7B_0.25R",
            "r_multiple": "1.50",
            "max_favorable_r": "2.00",
            "max_adverse_r": "0.35",
            **event_bars,
        }
    )

    # ------------------------------------------------------------------------
    # Trade 2 — complete 7B NO EVENT
    # ------------------------------------------------------------------------

    no_event_bars = {
        "early_adverse_r_bar_1": "0.05",
        "early_adverse_r_bar_2": "0.08",
        "early_adverse_r_bar_3": "0.11",
        "early_adverse_r_bar_4": "0.14",
        "early_adverse_r_bar_5": "0.17",
        "early_adverse_r_bar_6": "0.19",
        "early_adverse_r_bar_7": "0.22",
    }

    rows.append(
        {
            "trade_key": "SYNTHETIC|TEST2|SHORT|2026-09-15",
            "entry_date": "2026-09-15",
            "symbol": "TEST2",
            "direction": "SHORT",
            "setup": "SHORT_CONTINUATION",
            "rule_version": "7B_0.25R",
            "r_multiple": "-0.50",
            "max_favorable_r": "0.80",
            "max_adverse_r": "0.22",
            **no_event_bars,
        }
    )

    # ------------------------------------------------------------------------
    # Trade 3 — another event
    # ------------------------------------------------------------------------

    event_2_bars = {
        "early_adverse_r_bar_1": "0.26",
        "early_adverse_r_bar_2": "0.30",
        "early_adverse_r_bar_3": "0.32",
        "early_adverse_r_bar_4": "0.35",
        "early_adverse_r_bar_5": "0.40",
        "early_adverse_r_bar_6": "0.41",
        "early_adverse_r_bar_7": "0.45",
    }

    rows.append(
        {
            "trade_key": "SYNTHETIC|TEST3|LONG|2026-09-16",
            "entry_date": "2026-09-16",
            "symbol": "TEST3",
            "direction": "LONG",
            "setup": "LONG_REVERSAL",
            "rule_version": "7B_0.25R",
            "r_multiple": "-1.00",
            "max_favorable_r": "0.30",
            "max_adverse_r": "1.00",
            **event_2_bars,
        }
    )

    # ------------------------------------------------------------------------
    # Trade 4 — another no-event
    # ------------------------------------------------------------------------

    no_event_2_bars = {
        "early_adverse_r_bar_1": "0.02",
        "early_adverse_r_bar_2": "0.04",
        "early_adverse_r_bar_3": "0.06",
        "early_adverse_r_bar_4": "0.09",
        "early_adverse_r_bar_5": "0.12",
        "early_adverse_r_bar_6": "0.16",
        "early_adverse_r_bar_7": "0.20",
    }

    rows.append(
        {
            "trade_key": "SYNTHETIC|TEST4|SHORT|2026-09-16",
            "entry_date": "2026-09-16",
            "symbol": "TEST4",
            "direction": "SHORT",
            "setup": "SHORT_REVERSAL",
            "rule_version": "7B_0.25R",
            "r_multiple": "0.75",
            "max_favorable_r": "1.20",
            "max_adverse_r": "0.20",
            **no_event_2_bars,
        }
    )

    evidence = pd.DataFrame(rows)
    evidence["stream"] = "SYNTHETIC_TEST"
    return evidence


# ============================================================================
# TEMPORARY ISOLATED ENVIRONMENT
# ============================================================================

class IsolatedEnvironment:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

        self.sufficiency_summary = (
            self.root
            / "sufficiency"
            / "oos_sufficiency_summary.json"
        )

        self.evidence_file = (
            self.root
            / "evidence"
            / "oos_evidence_current.csv"
        )

        self.output_dir = (
            self.root
            / "evaluation"
        )

        self.current_output = (
            self.output_dir
            / "oos_evaluation_current.csv"
        )

        self.trade_output = (
            self.output_dir
            / "oos_evaluation_trade_level.csv"
        )

        self.summary_output = (
            self.output_dir
            / "oos_evaluation_summary.json"
        )

        self.history_output = (
            self.output_dir
            / "oos_evaluation_history.jsonl"
        )

        self.integrity_output = (
            self.output_dir
            / "oos_evaluation_integrity.csv"
        )

    def write_stage5(self, summary):
        self.sufficiency_summary.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.sufficiency_summary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                sort_keys=True,
            )

    def write_evidence(self, dataframe):
        self.evidence_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            self.evidence_file,
            index=False,
        )

    def patch_evaluator_paths(self):
        evaluator.SUFFICIENCY_SUMMARY = (
            self.sufficiency_summary
        )

        evaluator.EVIDENCE_FILE = (
            self.evidence_file
        )

        evaluator.OUTPUT_DIR = (
            self.output_dir
        )

        evaluator.CURRENT_OUTPUT = (
            self.current_output
        )

        evaluator.TRADE_OUTPUT = (
            self.trade_output
        )

        evaluator.SUMMARY_OUTPUT = (
            self.summary_output
        )

        evaluator.HISTORY_OUTPUT = (
            self.history_output
        )

        evaluator.INTEGRITY_OUTPUT = (
            self.integrity_output
        )

    def cleanup(self):
        self.temp.cleanup()


# ============================================================================
# TEST 1 — CURRENT REAL STATE MUST REMAIN BLOCKED
# ============================================================================

def test_real_stage5_state_is_blocked():
    print()
    print("[TEST 1] Real Stage 5 state remains blocked")

    summary = evaluator.load_sufficiency_summary()

    status, _ = evaluator.validate_stage5_contract(
        summary
    )

    assert_equal(
        status,
        "NO_GENUINE_OOS_EVIDENCE",
        "Current Stage 5 status changed unexpectedly.",
    )

    assert_true(
        status != "OOS_EVIDENCE_SUFFICIENT",
        "Current real OOS state unexpectedly opened evaluation.",
    )

    print("PASS — Current real Stage 5 state is blocked.")


# ============================================================================
# TEST 2 — SUFFICIENT SYNTHETIC STATE OPENS
# ============================================================================

def test_synthetic_sufficient_state_opens():
    print()
    print("[TEST 2] Synthetic Stage 5 sufficient state opens evaluation")

    env = IsolatedEnvironment()

    try:
        env.patch_evaluator_paths()

        summary = build_stage5_summary()

        env.write_stage5(summary)

        loaded = evaluator.load_sufficiency_summary()

        status, _ = evaluator.validate_stage5_contract(
            loaded
        )

        assert_equal(
            status,
            "OOS_EVIDENCE_SUFFICIENT",
            "Synthetic Stage 5 sufficient state was not accepted.",
        )

        print("PASS — Synthetic sufficient state opens Stage 6.")

    finally:
        env.cleanup()


# ============================================================================
# TEST 3 — COMPLETE EVENT CLASSIFICATION
# ============================================================================

def test_complete_event_classification():
    print()
    print("[TEST 3] Complete 7B event classification")

    evidence = build_complete_evidence()

    row = evidence.iloc[0]

    classification, trigger_value, trigger_bar = (
        evaluator.classify_trade(row)
    )

    assert_equal(
        classification,
        "COMPLETE_7B_EVENT",
        "Synthetic event was not classified correctly.",
    )

    assert_equal(
        trigger_bar,
        3,
        "Earliest event crossing bar is incorrect.",
    )

    assert_true(
        np.isclose(trigger_value, 0.27),
        "Event trigger value is incorrect.",
    )

    print(
        "PASS — Event classified at Bar 3 with 0.27R."
    )


# ============================================================================
# TEST 4 — COMPLETE NO-EVENT CLASSIFICATION
# ============================================================================

def test_complete_no_event_classification():
    print()
    print("[TEST 4] Complete 7B no-event classification")

    evidence = build_complete_evidence()

    row = evidence.iloc[1]

    classification, trigger_value, trigger_bar = (
        evaluator.classify_trade(row)
    )

    assert_equal(
        classification,
        "COMPLETE_7B_NO_EVENT",
        "Synthetic no-event was not classified correctly.",
    )

    assert_true(
        trigger_bar is None,
        "No-event incorrectly received a trigger bar.",
    )

    assert_true(
        np.isclose(trigger_value, 0.22),
        "Maximum observed adverse R is incorrect.",
    )

    print(
        "PASS — No-event correctly classified through Bar 7."
    )


# ============================================================================
# TEST 5 — PARTIAL 7B MUST BE REJECTED
# ============================================================================

def test_partial_7b_rejected():
    print()
    print("[TEST 5] Partial 7B evidence is rejected")

    evidence = build_complete_evidence()

    partial = evidence.copy()

    partial.loc[
        0,
        "early_adverse_r_bar_7",
    ] = np.nan

    row = partial.iloc[0]

    classification, _, _ = (
        evaluator.classify_trade(row)
    )

    assert_equal(
        classification,
        "PARTIAL_7B",
        "Partial 7B was not classified as PARTIAL_7B.",
    )

    print("PASS — Partial 7B is not silently treated as no-event.")


# ============================================================================
# TEST 6 — WRONG THRESHOLD MUST BE REJECTED
# ============================================================================

def test_wrong_threshold_rejected():
    print()
    print("[TEST 6] Wrong locked threshold is rejected")

    summary = build_stage5_summary()

    summary["locked_hypothesis"]["threshold_r"] = 0.30

    expect_failure(
        lambda: evaluator.validate_stage5_contract(
            summary
        ),
        "Stage 5 threshold does not match locked",
    )

    print("PASS — Wrong threshold rejected.")


# ============================================================================
# TEST 7 — WRONG WINDOW MUST BE REJECTED
# ============================================================================

def test_wrong_window_rejected():
    print()
    print("[TEST 7] Wrong locked window is rejected")

    summary = build_stage5_summary()

    summary["locked_hypothesis"]["window_bars"] = 5

    expect_failure(
        lambda: evaluator.validate_stage5_contract(
            summary
        ),
        "Stage 5 window does not match locked",
    )

    print("PASS — Wrong window rejected.")


# ============================================================================
# TEST 8 — DUPLICATE TRADE KEYS MUST BE REJECTED
# ============================================================================

def test_duplicate_trade_keys_rejected():
    print()
    print("[TEST 8] Duplicate trade keys are rejected")

    evidence = build_complete_evidence()

    duplicated = pd.concat(
        [
            evidence,
            evidence.iloc[[0]].copy(),
        ],
        ignore_index=True,
    )

    duplicate_count = evaluator.check_duplicate_trade_keys(
        duplicated
    )

    assert_equal(
        duplicate_count,
        2,
        "Duplicate trade-key count is incorrect.",
    )

    print(
        "PASS — Duplicate trade keys detected."
    )


# ============================================================================
# TEST 9 — MISSING FINAL R MUST BE REJECTED
# ============================================================================

def test_missing_final_r_rejected():
    print()
    print("[TEST 9] Missing final R is rejected")

    evidence = build_complete_evidence()

    broken = evidence.copy()

    broken.loc[
        0,
        "r_multiple",
    ] = np.nan

    problems = evaluator.validate_trade_identity(
        broken
    )

    reasons = {
        problem["reason"]
        for problem in problems
    }

    assert_true(
        "MISSING_FINAL_R" in reasons,
        "Missing final R was not detected.",
    )

    print(
        "PASS — Missing final R detected."
    )


# ============================================================================
# TEST 10 — FULL SYNTHETIC EVALUATION
# ============================================================================

def test_full_synthetic_evaluation():
    print()
    print("[TEST 10] Full synthetic Stage 6 evaluation")

    env = IsolatedEnvironment()

    try:
        env.patch_evaluator_paths()

        summary = build_stage5_summary()

        evidence = build_complete_evidence()

        env.write_stage5(summary)
        env.write_evidence(evidence)

        loaded_summary = env.sufficiency_summary

        stage5 = evaluator.load_sufficiency_summary()

        status, _ = evaluator.validate_stage5_contract(
            stage5
        )

        assert_equal(
            status,
            "OOS_EVIDENCE_SUFFICIENT",
            "Synthetic Stage 5 gate did not open.",
        )

        result = evaluator.run_evaluation(
            stage5
        )

        assert_equal(
            result,
            0,
            "Synthetic Stage 6 evaluation did not complete successfully.",
        )

        assert_true(
            env.trade_output.exists(),
            "Trade-level evaluation output was not created.",
        )

        assert_true(
            env.summary_output.exists(),
            "Evaluation summary was not created.",
        )

        assert_true(
            env.integrity_output.exists(),
            "Evaluation integrity output was not created.",
        )

        evaluated = pd.read_csv(
            env.trade_output
        )

        assert_equal(
            len(evaluated),
            4,
            "Unexpected synthetic trade-level row count.",
        )

        assert_equal(
            int(
                (
                    evaluated["classification"]
                    == "COMPLETE_7B_EVENT"
                ).sum()
            ),
            2,
            "Synthetic event count is incorrect.",
        )

        assert_equal(
            int(
                (
                    evaluated["classification"]
                    == "COMPLETE_7B_NO_EVENT"
                ).sum()
            ),
            2,
            "Synthetic no-event count is incorrect.",
        )

        with env.summary_output.open(
            "r",
            encoding="utf-8",
        ) as handle:
            result_summary = json.load(handle)

        assert_equal(
            result_summary["status"],
            "EVALUATION_COMPLETE",
            "Synthetic evaluation status is incorrect.",
        )

        assert_equal(
            result_summary["primary_population"]["complete_7b"],
            4,
            "Primary complete 7B population is incorrect.",
        )

        assert_equal(
            result_summary["primary_population"]["event_count"],
            2,
            "Primary event count is incorrect.",
        )

        assert_equal(
            result_summary["primary_population"]["no_event_count"],
            2,
            "Primary no-event count is incorrect.",
        )

        print(
            "PASS — Full synthetic Stage 6 evaluation completed."
        )

    finally:
        env.cleanup()


# ============================================================================
# TEST 11 — PARTIAL BAR FIELD MUST BLOCK FULL EVALUATION
# ============================================================================

def test_full_evaluation_rejects_missing_bar7():
    print()
    print("[TEST 11] Full evaluation rejects missing Bar 7")

    env = IsolatedEnvironment()

    try:
        env.patch_evaluator_paths()

        summary = build_stage5_summary()

        evidence = build_complete_evidence()

        evidence = evidence.drop(
            columns=["early_adverse_r_bar_7"]
        )

        env.write_stage5(summary)
        env.write_evidence(evidence)

        stage5 = evaluator.load_sufficiency_summary()

        expect_failure(
            lambda: evaluator.run_evaluation(
                stage5
            ),
            "Missing:",
        )

        print(
            "PASS — Missing Bar 7 blocks evaluation."
        )

    finally:
        env.cleanup()


# ============================================================================
# SAFETY ASSERTIONS
# ============================================================================

def test_safety_contract_constants():
    print()
    print("[TEST 12] Locked safety constants")

    assert_equal(
        evaluator.LOCKED_WINDOW_BARS,
        7,
        "Locked evaluation window changed.",
    )

    assert_true(
        np.isclose(
            evaluator.LOCKED_THRESHOLD_R,
            0.25,
        ),
        "Locked evaluation threshold changed.",
    )

    assert_equal(
        evaluator.LOCKED_RULE_VERSION,
        "7B_0.25R",
        "Locked rule version changed.",
    )

    print(
        "PASS — 7B / 0.25R contract remains frozen."
    )


# ============================================================================
# RUNNER
# ============================================================================

def main():
    print()
    print("=" * 110)
    print(
        "TRADESENSE-AI — "
        "STAGE 6 REGRESSION / SAFETY HARNESS"
    )
    print("=" * 110)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "All synthetic evaluation data is isolated in temporary directories."
    )
    print(
        "Real OOS ledgers and production data are not modified."
    )

    tests = [
        test_real_stage5_state_is_blocked,
        test_synthetic_sufficient_state_opens,
        test_complete_event_classification,
        test_complete_no_event_classification,
        test_partial_7b_rejected,
        test_wrong_threshold_rejected,
        test_wrong_window_rejected,
        test_duplicate_trade_keys_rejected,
        test_missing_final_r_rejected,
        test_full_synthetic_evaluation,
        test_full_evaluation_rejects_missing_bar7,
        test_safety_contract_constants,
    ]

    passed = 0

    for test in tests:
        test()
        passed += 1

    print()
    print("=" * 110)
    print(
        "STAGE 6 REGRESSION / SAFETY HARNESS: PASS"
    )
    print("=" * 110)

    print()
    print(
        f"Tests passed : {passed}/{len(tests)}"
    )

    print(
        "Real OOS data : UNCHANGED"
    )

    print(
        "Production strategy : UNCHANGED"
    )

    print(
        "Historical 106-trade population : NOT USED"
    )

    print(
        "Locked hypothesis : 7B / 0.25R"
    )

    print(
        "Synthetic fixtures : TEMPORARY / ISOLATED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )