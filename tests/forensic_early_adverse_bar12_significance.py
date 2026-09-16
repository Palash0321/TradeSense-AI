"""
TradeSense-AI — Bar 1→2 Statistical Significance Forensics
============================================================

Purpose
-------
Statistically evaluate the four pre-specified Bar 1→2 state transitions
identified during the descriptive forensic analysis.

Frozen population
-----------------
106 trades
60 frozen early-adverse events
46 retained trades

Pre-specified states
--------------------
1. bar2_mfe_increased
2. bar2_favorable_failed_to_improve
3. bar2_net_deteriorated
4. adverse_up_net_down

This script is RESEARCH-ONLY.

Critical guardrails
-------------------
- Does NOT modify production code.
- Does NOT modify frozen trade data.
- Does NOT modify OOS/holdout data.
- Does NOT create or promote a trading rule.
- Does NOT optimize thresholds.
- Does NOT search for new states.
- Tests only states already selected by prior forensic analysis.
- Frozen event definition remains stored_frozen_event.
- Statistical significance is descriptive/inferential evidence only.
"""

from pathlib import Path
import math
import sys

import pandas as pd

try:
    from scipy.stats import fisher_exact
except ImportError:
    print("[FAIL] scipy is required for Fisher exact testing.")
    print("Install it with: pip install scipy")
    sys.exit(1)


# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_state_transitions.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_significance"
)

OUTPUT_FILE = OUTPUT_DIR / "early_adverse_bar12_significance.csv"
INTEGRITY_FILE = OUTPUT_DIR / "early_adverse_bar12_significance_integrity.csv"

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

ALPHA = 0.05
TEST_COUNT = 4

# Four states were selected BEFORE this significance test.
# No additional state discovery is performed here.
TEST_STATES = [
    "bar2_mfe_increased",
    "bar2_favorable_failed_to_improve",
    "bar2_net_deteriorated",
    "adverse_up_net_down",
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return float("nan")


def wilson_interval(successes, total, z=1.959963984540054):
    """
    Wilson score confidence interval for a binomial proportion.
    """
    if total <= 0:
        return float("nan"), float("nan")

    p = successes / total

    denominator = 1.0 + (z ** 2) / total

    center = (
        p
        + (z ** 2) / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / total
            )
            + (
                z ** 2 / (4.0 * total ** 2)
            )
        )
        / denominator
    )

    return center - margin, center + margin


def newcombe_risk_difference_ci(
    event_successes,
    event_total,
    retained_successes,
    retained_total,
):
    """
    Newcombe-style CI for the difference between two independent
    binomial proportions using Wilson intervals.

    Difference is:

        event_rate - retained_rate
    """

    event_low, event_high = wilson_interval(
        event_successes,
        event_total,
    )

    retained_low, retained_high = wilson_interval(
        retained_successes,
        retained_total,
    )

    event_rate = event_successes / event_total
    retained_rate = retained_successes / retained_total

    difference = event_rate - retained_rate

    # Newcombe CI:
    # lower = d - sqrt((p1-l1)^2 + (u2-p2)^2)
    # upper = d + sqrt((u1-p1)^2 + (p2-l2)^2)

    lower = difference - math.sqrt(
        (event_rate - event_low) ** 2
        + (retained_high - retained_rate) ** 2
    )

    upper = difference + math.sqrt(
        (event_high - event_rate) ** 2
        + (retained_rate - retained_low) ** 2
    )

    return difference, lower, upper


def odds_ratio_and_ci(
    event_positive,
    event_negative,
    retained_positive,
    retained_negative,
):
    """
    Odds ratio:

        (event_positive / event_negative)
        --------------------------------
        (retained_positive / retained_negative)

    Uses the standard log-OR Wald interval.

    If a cell is zero, applies a Haldane-Anscombe correction (+0.5
    to all four cells) for a finite estimate.
    """

    a = float(event_positive)
    b = float(event_negative)
    c = float(retained_positive)
    d = float(retained_negative)

    corrected = False

    if min(a, b, c, d) == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5
        corrected = True

    odds_ratio = (a * d) / (b * c)

    log_or = math.log(odds_ratio)

    se = math.sqrt(
        (1.0 / a)
        + (1.0 / b)
        + (1.0 / c)
        + (1.0 / d)
    )

    z = 1.959963984540054

    lower = math.exp(log_or - z * se)
    upper = math.exp(log_or + z * se)

    return odds_ratio, lower, upper, corrected


def bonferroni_adjust(p_value, number_of_tests):
    return min(1.0, p_value * number_of_tests)


def interpretation(
    feature,
    event_rate,
    retained_rate,
    risk_difference,
    ci_low,
    ci_high,
    p_value,
    adjusted_p,
):
    direction = "higher" if risk_difference > 0 else "lower"

    if ci_low > 0 or ci_high < 0:
        ci_support = "95% CI excludes 0"
    else:
        ci_support = "95% CI includes 0"

    if adjusted_p < ALPHA:
        significance = "statistically_significant_after_bonferroni"
    else:
        significance = "not_statistically_significant_after_bonferroni"

    return (
        f"{feature}: state rate is {direction} in events "
        f"({event_rate:.2%}) than retained ({retained_rate:.2%}); "
        f"risk difference={risk_difference:+.4f} "
        f"({risk_difference * 100:+.2f} pp), "
        f"95% CI=[{ci_low:+.4f}, {ci_high:+.4f}] "
        f"({ci_support}); "
        f"Fisher p={p_value:.6f}; "
        f"Bonferroni p={adjusted_p:.6f}; "
        f"{significance}."
    )


# ============================================================================
# LOAD INPUT
# ============================================================================

print("=" * 78)
print("TRADE SENSE-AI — BAR 1→2 STATISTICAL SIGNIFICANCE")
print("=" * 78)

print(f"[INFO] Input file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    print(f"[FAIL] Input file does not exist: {INPUT_FILE}")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)

print(f"[INFO] Input rows: {len(df)}")
print(f"[INFO] Input columns: {len(df.columns)}")


# ============================================================================
# INPUT VALIDATION
# ============================================================================

required_columns = {
    "state_transition",
    "event_count",
    "event_total",
    "event_rate",
    "retained_count",
    "retained_total",
    "retained_rate",
}

missing_columns = sorted(required_columns - set(df.columns))

if missing_columns:
    print("[FAIL] Missing required columns:")
    for column in missing_columns:
        print(f"  - {column}")
    sys.exit(1)

# Keep only the four pre-specified states.
tested = df[df["state_transition"].isin(TEST_STATES)].copy()

missing_states = [
    state
    for state in TEST_STATES
    if state not in set(tested["state_transition"])
]

if missing_states:
    print("[FAIL] Missing pre-specified states:")
    for state in missing_states:
        print(f"  - {state}")
    sys.exit(1)

# Make sure each state appears exactly once.
duplicate_states = (
    tested["state_transition"]
    .value_counts()
    .loc[lambda x: x > 1]
)

if len(duplicate_states) > 0:
    print("[FAIL] Duplicate state rows detected:")
    print(duplicate_states.to_string())
    sys.exit(1)


# ============================================================================
# FROZEN POPULATION VALIDATION
# ============================================================================

event_totals = set(
    pd.to_numeric(tested["event_total"], errors="coerce")
    .dropna()
    .astype(int)
)

retained_totals = set(
    pd.to_numeric(tested["retained_total"], errors="coerce")
    .dropna()
    .astype(int)
)

if event_totals != {EXPECTED_EVENTS}:
    print(
        "[FAIL] Event total mismatch: "
        f"expected {EXPECTED_EVENTS}, observed {event_totals}"
    )
    sys.exit(1)

if retained_totals != {EXPECTED_RETAINED}:
    print(
        "[FAIL] Retained total mismatch: "
        f"expected {EXPECTED_RETAINED}, observed {retained_totals}"
    )
    sys.exit(1)

if EXPECTED_EVENTS + EXPECTED_RETAINED != EXPECTED_TRADES:
    print("[FAIL] Frozen population arithmetic is inconsistent.")
    sys.exit(1)

print()
print("[PASS] Frozen population:")
print(f"       Trades   : {EXPECTED_TRADES}")
print(f"       Events   : {EXPECTED_EVENTS}")
print(f"       Retained : {EXPECTED_RETAINED}")
print(f"       States tested: {len(TEST_STATES)}")


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

results = []

print()
print("=" * 78)
print("STATISTICAL TESTS")
print("=" * 78)

for _, row in tested.iterrows():

    feature = str(row["state_transition"])

    event_positive = int(row["event_count"])
    event_total = int(row["event_total"])

    retained_positive = int(row["retained_count"])
    retained_total = int(row["retained_total"])

    event_negative = event_total - event_positive
    retained_negative = retained_total - retained_positive

    # ----------------------------------------------------------------------
    # Rates
    # ----------------------------------------------------------------------

    event_rate = event_positive / event_total
    retained_rate = retained_positive / retained_total

    # ----------------------------------------------------------------------
    # Risk difference + 95% CI
    # ----------------------------------------------------------------------

    risk_difference, rd_ci_low, rd_ci_high = (
        newcombe_risk_difference_ci(
            event_positive,
            event_total,
            retained_positive,
            retained_total,
        )
    )

    # ----------------------------------------------------------------------
    # Fisher exact test
    #
    # Table:
    #
    #                    State +    State -
    # Event                  a          b
    # Retained               c          d
    #
    # Odds ratio is event odds / retained odds.
    # ----------------------------------------------------------------------

    contingency_table = [
        [event_positive, event_negative],
        [retained_positive, retained_negative],
    ]

    odds_ratio_fisher, p_value = fisher_exact(
        contingency_table,
        alternative="two-sided",
    )

    # ----------------------------------------------------------------------
    # OR confidence interval
    # ----------------------------------------------------------------------

    odds_ratio, or_ci_low, or_ci_high, continuity_correction = (
        odds_ratio_and_ci(
            event_positive,
            event_negative,
            retained_positive,
            retained_negative,
        )
    )

    # ----------------------------------------------------------------------
    # Multiple testing correction
    # ----------------------------------------------------------------------

    bonferroni_p = bonferroni_adjust(
        p_value,
        TEST_COUNT,
    )

    significant_raw = p_value < ALPHA
    significant_bonferroni = bonferroni_p < ALPHA

    result = {
        "feature": feature,
        "event_positive": event_positive,
        "event_negative": event_negative,
        "event_total": event_total,
        "event_rate": event_rate,
        "event_rate_pct": event_rate * 100.0,
        "retained_positive": retained_positive,
        "retained_negative": retained_negative,
        "retained_total": retained_total,
        "retained_rate": retained_rate,
        "retained_rate_pct": retained_rate * 100.0,
        "risk_difference": risk_difference,
        "risk_difference_pp": risk_difference * 100.0,
        "risk_difference_ci_low": rd_ci_low,
        "risk_difference_ci_high": rd_ci_high,
        "risk_difference_ci_low_pp": rd_ci_low * 100.0,
        "risk_difference_ci_high_pp": rd_ci_high * 100.0,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_low": or_ci_low,
        "odds_ratio_ci_high": or_ci_high,
        "fisher_exact_p": p_value,
        "bonferroni_adjusted_p": bonferroni_p,
        "significant_raw_alpha_0_05": significant_raw,
        "significant_bonferroni_alpha_0_05": significant_bonferroni,
        "odds_ratio_continuity_correction": continuity_correction,
        "ci_excludes_zero": (
            rd_ci_low > 0.0 or rd_ci_high < 0.0
        ),
        "interpretation": interpretation(
            feature,
            event_rate,
            retained_rate,
            risk_difference,
            rd_ci_low,
            rd_ci_high,
            p_value,
            bonferroni_p,
        ),
    }

    results.append(result)

    print()
    print(f"[TEST] {feature}")
    print(
        f"       Event      : "
        f"{event_positive}/{event_total} "
        f"({event_rate * 100:.2f}%)"
    )
    print(
        f"       Retained   : "
        f"{retained_positive}/{retained_total} "
        f"({retained_rate * 100:.2f}%)"
    )
    print(
        f"       Risk diff  : "
        f"{risk_difference * 100:+.2f} pp"
    )
    print(
        f"       95% RD CI  : "
        f"[{rd_ci_low * 100:+.2f}, "
        f"{rd_ci_high * 100:+.2f}] pp"
    )
    print(
        f"       Odds ratio : "
        f"{odds_ratio:.4f}"
    )
    print(
        f"       95% OR CI  : "
        f"[{or_ci_low:.4f}, {or_ci_high:.4f}]"
    )
    print(
        f"       Fisher p   : "
        f"{p_value:.6f}"
    )
    print(
        f"       Bonferroni  : "
        f"{bonferroni_p:.6f}"
    )
    print(
        f"       Raw α=.05  : "
        f"{'YES' if significant_raw else 'NO'}"
    )
    print(
        f"       Adj α=.05  : "
        f"{'YES' if significant_bonferroni else 'NO'}"
    )


# ============================================================================
# SAVE RESULTS
# ============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print(f"[PASS] Saved: {OUTPUT_FILE}")


# ============================================================================
# GLOBAL INTEGRITY
# ============================================================================

integrity_rows = []

integrity_rows.append(
    {
        "check": "expected_trade_count",
        "expected": EXPECTED_TRADES,
        "observed": EXPECTED_EVENTS + EXPECTED_RETAINED,
        "status": (
            "PASS"
            if EXPECTED_EVENTS + EXPECTED_RETAINED == EXPECTED_TRADES
            else "FAIL"
        ),
    }
)

integrity_rows.append(
    {
        "check": "expected_event_count",
        "expected": EXPECTED_EVENTS,
        "observed": max(
            int(x)
            for x in tested["event_total"]
        ),
        "status": (
            "PASS"
            if event_totals == {EXPECTED_EVENTS}
            else "FAIL"
        ),
    }
)

integrity_rows.append(
    {
        "check": "expected_retained_count",
        "expected": EXPECTED_RETAINED,
        "observed": max(
            int(x)
            for x in tested["retained_total"]
        ),
        "status": (
            "PASS"
            if retained_totals == {EXPECTED_RETAINED}
            else "FAIL"
        ),
    }
)

integrity_rows.append(
    {
        "check": "pre_specified_state_count",
        "expected": len(TEST_STATES),
        "observed": len(results_df),
        "status": (
            "PASS"
            if len(results_df) == len(TEST_STATES)
            else "FAIL"
        ),
    }
)

integrity_rows.append(
    {
        "check": "duplicate_state_count",
        "expected": 0,
        "observed": int(
            tested["state_transition"].duplicated().sum()
        ),
        "status": (
            "PASS"
            if tested["state_transition"].duplicated().sum() == 0
            else "FAIL"
        ),
    }
)

integrity_rows.append(
    {
        "check": "bonferroni_test_count",
        "expected": TEST_COUNT,
        "observed": len(results_df),
        "status": (
            "PASS"
            if len(results_df) == TEST_COUNT
            else "FAIL"
        ),
    }
)

integrity_df = pd.DataFrame(integrity_rows)

integrity_df.to_csv(
    INTEGRITY_FILE,
    index=False,
)

print(f"[PASS] Saved: {INTEGRITY_FILE}")


# ============================================================================
# FINAL FORENSIC SUMMARY
# ============================================================================

print()
print("=" * 78)
print("FORENSIC SIGNIFICANCE SUMMARY")
print("=" * 78)

for _, result in results_df.iterrows():

    print(
        f"{result['feature']:<42} "
        f"Δ={result['risk_difference_pp']:+.2f} pp  "
        f"p={result['fisher_exact_p']:.6f}  "
        f"Bonf={result['bonferroni_adjusted_p']:.6f}"
    )

print()
print("Interpretation guardrails:")
print("1. These are pre-specified state tests.")
print("2. Fisher exact tests are two-sided.")
print("3. Bonferroni correction covers all four tests.")
print("4. Confidence intervals quantify uncertainty; they do not prove causality.")
print("5. The frozen event label remains unchanged.")
print("6. No trading rule is promoted.")
print("7. Production code remains untouched.")
print("8. Frozen/OOS data remain untouched.")

all_integrity_pass = bool(
    (integrity_df["status"] == "PASS").all()
)

if not all_integrity_pass:
    print()
    print("[FAIL] SIGNIFICANCE ANALYSIS INTEGRITY FAILURE")
    sys.exit(1)

print()
print("BAR 1→2 STATISTICAL SIGNIFICANCE — PASS")
print(f"Tests completed: {len(results_df)}")
print(f"Output: {OUTPUT_FILE}")
print(f"Integrity: {INTEGRITY_FILE}")
print("No strategy rule has been promoted.")
print("No production/OOS data has been modified.")
print("FINAL STATUS: PASS")