"""
Forensic robustness analysis for the fixed NIFTY 50
LONG_CONTINUATION exit counterfactual.

FIXED RULE — NO PARAMETER TUNING:

    First reach +1.5R
        ->
    activate +1R protection
        ->
    +2R target

IMPORTANT ACCOUNTING RULE:

    If a trade NEVER reaches +1.5R during its actual lifetime,
    the hypothetical exit architecture never activates.

    Therefore:

        counterfactual R = actual R

    We must NOT treat such a trade as 0R.

For AMBIGUOUS / UNRESOLVED trades:
    No hypothetical outcome is invented.

This script is research-only.

It does NOT modify:
    - production strategy
    - frozen historical trades
    - holdout trades
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = (
    "tests/output/index_backtests/"
    "nifty50_lc_exit_counterfactual.csv"
)

OUTPUT_DIR = "tests/output/index_backtests"

SLICE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_robustness_slices.csv"
)

LOO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_robustness_leave_one_out.csv"
)

REPORT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_robustness_report.txt"
)

EXPECTED_TRADES = 12
EXPECTED_BASELINE_R = 2.27
EXPECTED_TRIGGERED_CF_R = 10.00

R_TOLERANCE = 0.01

LATE_REVERSAL_DATES = {
    "2024-08-30",
    "2025-09-15",
}


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def classify_year(date_string: str) -> int:
    return int(str(date_string)[:4])


def classify_period(year: int) -> str:

    if 2021 <= year <= 2023:
        return "2021-2023"

    if 2024 <= year <= 2026:
        return "2024-2026"

    return "OTHER"


# ============================================================================
# LOAD
# ============================================================================

def load_results() -> pd.DataFrame:

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Counterfactual file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run the corrected counterfactual script first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "entry_date",
        "actual_r",
        "counterfactual_r",
        "status",
        "counterfactual_exit",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["entry_date"] = (
        df["entry_date"]
        .astype(str)
        .str[:10]
    )

    df["year"] = df["entry_date"].apply(
        classify_year
    )

    df["period"] = df["year"].apply(
        classify_period
    )

    df["actual_r"] = pd.to_numeric(
        df["actual_r"],
        errors="coerce"
    )

    df["counterfactual_r_raw"] = pd.to_numeric(
        df["counterfactual_r"],
        errors="coerce"
    )

    df["late_reversal"] = (
        df["entry_date"].isin(
            LATE_REVERSAL_DATES
        )
    )

    return df


# ============================================================================
# ACCOUNTING NORMALIZATION
# ============================================================================

def apply_full_counterfactual_accounting(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the trigger-only counterfactual output into a valid
    full-trade counterfactual.

    Rules:

        RESOLVED:
            use hypothetical counterfactual R

        NO_TRIGGER:
            retain actual R because the hypothetical architecture
            never activates

        AMBIGUOUS:
            remain unresolved / NaN

        UNRESOLVED:
            remain unresolved / NaN

    The important distinction is that NO_TRIGGER is NOT 0R.
    """

    result = df.copy()

    result["counterfactual_r"] = np.nan

    resolved_mask = (
        result["status"].astype(str)
        == "RESOLVED"
    )

    no_trigger_mask = (
        result["status"].astype(str)
        == "NO_TRIGGER"
    )

    result.loc[
        resolved_mask,
        "counterfactual_r"
    ] = result.loc[
        resolved_mask,
        "counterfactual_r_raw"
    ]

    result.loc[
        no_trigger_mask,
        "counterfactual_r"
    ] = result.loc[
        no_trigger_mask,
        "actual_r"
    ]

    # Label what happened to the counterfactual.
    result["counterfactual_accounting"] = "UNRESOLVED"

    result.loc[
        resolved_mask,
        "counterfactual_accounting"
    ] = "HYPOTHETICAL_EXIT"

    result.loc[
        no_trigger_mask,
        "counterfactual_accounting"
    ] = "ACTUAL_PATH_RETAINED"

    result.loc[
        result["status"].astype(str) == "AMBIGUOUS",
        "counterfactual_accounting"
    ] = "AMBIGUOUS"

    return result


# ============================================================================
# SLICE METRICS
# ============================================================================

def calculate_slice(
    df: pd.DataFrame,
    label: str,
) -> dict:

    total = len(df)

    baseline_r = float(
        df["actual_r"].sum()
    )

    valid_cf = df[
        df["counterfactual_r"].notna()
    ].copy()

    counterfactual_r = float(
        valid_cf["counterfactual_r"].sum()
    )

    delta_r = (
        counterfactual_r
        - baseline_r
    )

    triggered_count = int(
        (
            df["status"].astype(str)
            == "RESOLVED"
        ).sum()
    )

    no_trigger_count = int(
        (
            df["status"].astype(str)
            == "NO_TRIGGER"
        ).sum()
    )

    ambiguous_count = int(
        (
            df["status"].astype(str)
            == "AMBIGUOUS"
        ).sum()
    )

    unresolved_count = int(
        (
            df["status"].astype(str)
            == "UNRESOLVED"
        ).sum()
    )

    target_count = int(
        (
            df["counterfactual_exit"].astype(str)
            == "TARGET_2R"
        ).sum()
    )

    protection_count = int(
        (
            df["counterfactual_exit"].astype(str)
            == "PROTECTED_1R"
        ).sum()
    )

    improved = 0
    worsened = 0
    unchanged = 0

    for _, row in valid_cf.iterrows():

        actual = safe_float(
            row["actual_r"]
        )

        cf = safe_float(
            row["counterfactual_r"]
        )

        if not np.isfinite(actual):
            continue

        if not np.isfinite(cf):
            continue

        difference = cf - actual

        if difference > R_TOLERANCE:
            improved += 1

        elif difference < -R_TOLERANCE:
            worsened += 1

        else:
            unchanged += 1

    return {
        "slice": label,
        "trades": total,
        "triggered": triggered_count,
        "no_trigger": no_trigger_count,
        "ambiguous": ambiguous_count,
        "unresolved": unresolved_count,
        "target_2r": target_count,
        "protected_1r": protection_count,
        "baseline_r": baseline_r,
        "counterfactual_r": counterfactual_r,
        "delta_r": delta_r,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


# ============================================================================
# TIME SLICES
# ============================================================================

def build_time_slices(
    df: pd.DataFrame,
) -> list[dict]:

    rows = []

    rows.append(
        calculate_slice(
            df[
                df["period"]
                == "2021-2023"
            ],
            "2021-2023"
        )
    )

    rows.append(
        calculate_slice(
            df[
                df["period"]
                == "2024-2026"
            ],
            "2024-2026"
        )
    )

    for year in sorted(
        df["year"].unique()
    ):

        year_df = df[
            df["year"] == year
        ]

        rows.append(
            calculate_slice(
                year_df,
                str(year)
            )
        )

    return rows


# ============================================================================
# LATE REVERSAL ANALYSIS
# ============================================================================

def build_late_reversal_analysis(
    df: pd.DataFrame,
) -> list[dict]:

    late = df[
        df["late_reversal"]
    ].copy()

    without_late = df[
        ~df["late_reversal"]
    ].copy()

    return [
        calculate_slice(
            late,
            "ONLY_TWO_LATE_REVERSALS"
        ),
        calculate_slice(
            without_late,
            "ALL_OTHER_LC_TRADES"
        ),
    ]


# ============================================================================
# TRIGGER PARTICIPATION
# ============================================================================

def build_trigger_analysis(
    df: pd.DataFrame,
) -> list[dict]:

    triggered = df[
        df["status"].astype(str)
        == "RESOLVED"
    ].copy()

    non_triggered = df[
        df["status"].astype(str)
        == "NO_TRIGGER"
    ].copy()

    return [
        calculate_slice(
            triggered,
            "TRIGGERED_TRADES_ONLY"
        ),
        calculate_slice(
            non_triggered,
            "NO_TRIGGER_TRADES"
        ),
    ]


# ============================================================================
# LEAVE ONE TRADE OUT
# ============================================================================

def build_leave_one_out(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for index in df.index:

        removed_trade = df.loc[index]

        remaining = df.drop(
            index
        )

        remaining_metrics = calculate_slice(
            remaining,
            "LEAVE_ONE_OUT"
        )

        removed_actual = safe_float(
            removed_trade["actual_r"],
            0.0
        )

        removed_cf = safe_float(
            removed_trade["counterfactual_r"],
            np.nan
        )

        if np.isfinite(removed_cf):
            removed_delta = (
                removed_cf
                - removed_actual
            )
        else:
            removed_delta = 0.0

        rows.append(
            {
                "removed_entry_date":
                    removed_trade["entry_date"],

                "removed_status":
                    removed_trade["status"],

                "removed_actual_r":
                    removed_actual,

                "removed_counterfactual_r":
                    removed_cf,

                "removed_delta_r":
                    removed_delta,

                "remaining_trades":
                    remaining_metrics["trades"],

                "remaining_baseline_r":
                    remaining_metrics["baseline_r"],

                "remaining_counterfactual_r":
                    remaining_metrics[
                        "counterfactual_r"
                    ],

                "remaining_delta_r":
                    remaining_metrics["delta_r"],

                "remaining_improved":
                    remaining_metrics["improved"],

                "remaining_worsened":
                    remaining_metrics["worsened"],

                "removed_was_late_reversal":
                    bool(
                        removed_trade[
                            "late_reversal"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# PRINT TABLE
# ============================================================================

def print_slice_table(
    title: str,
    rows: list[dict],
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for row in rows:

        print(
            f"{row['slice']:<28} "
            f"trades={row['trades']:>2} | "
            f"trigger={row['triggered']:>2} | "
            f"base={row['baseline_r']:>7.2f}R | "
            f"CF={row['counterfactual_r']:>7.2f}R | "
            f"delta={row['delta_r']:>+7.2f}R | "
            f"imp={row['improved']:>2} | "
            f"worse={row['worsened']:>2}"
        )


# ============================================================================
# VERDICT
# ============================================================================

def determine_verdict(
    time_slices: list[dict],
    loo_df: pd.DataFrame,
    without_late: dict,
) -> str:

    positive_time_slices = sum(
        row["delta_r"] > R_TOLERANCE
        for row in time_slices
        if row["trades"] > 0
    )

    total_time_slices = sum(
        row["trades"] > 0
        for row in time_slices
    )

    loo_positive = int(
        (
            loo_df["remaining_delta_r"]
            > R_TOLERANCE
        ).sum()
    )

    loo_negative = int(
        (
            loo_df["remaining_delta_r"]
            < -R_TOLERANCE
        ).sum()
    )

    # ---------------------------------------------------------------------
    # IMPORTANT:
    # We deliberately do NOT require every year to be positive.
    #
    # This is a very small 12-trade sample.
    # The verdict requires:
    #
    #   1. positive effect outside the two late reversals
    #   2. positive leave-one-out robustness
    #   3. positive effect in multiple time slices
    # ---------------------------------------------------------------------

    if (
        without_late["delta_r"]
        > R_TOLERANCE
        and loo_positive > loo_negative
        and positive_time_slices >= 2
    ):
        return (
            "FIXED_EXIT_RULE_ROBUST_RESEARCH_SIGNAL"
        )

    if (
        without_late["delta_r"]
        > R_TOLERANCE
    ):
        return (
            "POSITIVE_BUT_LIMITED_ROBUSTNESS"
        )

    return (
        "ROBUSTNESS_NOT_ESTABLISHED"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 100)
    print("NIFTY 50 LC FIXED EXIT-RULE ROBUSTNESS")
    print("=" * 100)

    print()
    print(
        "FIXED RULE:"
    )
    print(
        "+1.5R trigger -> +1R protection -> +2R target"
    )

    print()
    print(
        "NO PARAMETER TUNING."
    )

    print()
    print(
        "FULL-TRADE ACCOUNTING:"
    )
    print(
        "NO_TRIGGER trades retain their actual R."
    )

    # ---------------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------------

    df = load_results()

    print()
    print(
        f"Loaded counterfactual rows: {len(df)}"
    )

    if len(df) != EXPECTED_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRADES} trades, "
            f"found {len(df)}."
        )

    # ---------------------------------------------------------------------
    # BASELINE RECONCILIATION
    # ---------------------------------------------------------------------

    baseline_r = float(
        df["actual_r"].sum()
    )

    if abs(
        baseline_r
        - EXPECTED_BASELINE_R
    ) > R_TOLERANCE:

        raise RuntimeError(
            "Baseline R mismatch.\n"
            f"Expected: {EXPECTED_BASELINE_R:.2f}R\n"
            f"Found:    {baseline_r:.2f}R"
        )

    print()
    print(
        f"Baseline reconciliation: "
        f"{baseline_r:.2f}R (PASS)"
    )

    # ---------------------------------------------------------------------
    # APPLY CORRECT ACCOUNTING
    # ---------------------------------------------------------------------

    df = apply_full_counterfactual_accounting(
        df
    )

    # ---------------------------------------------------------------------
    # TRIGGERED SUBSET RECONCILIATION
    # ---------------------------------------------------------------------

    triggered = df[
        df["status"].astype(str)
        == "RESOLVED"
    ]

    triggered_cf = float(
        triggered["counterfactual_r"].sum()
    )

    if abs(
        triggered_cf
        - EXPECTED_TRIGGERED_CF_R
    ) > R_TOLERANCE:

        raise RuntimeError(
            "Triggered counterfactual reconciliation failed.\n"
            f"Expected triggered CF: "
            f"{EXPECTED_TRIGGERED_CF_R:.2f}R\n"
            f"Found: {triggered_cf:.2f}R"
        )

    print(
        f"Triggered CF reconciliation: "
        f"{triggered_cf:.2f}R (PASS)"
    )

    # ---------------------------------------------------------------------
    # FULL COUNTERFACTUAL
    # ---------------------------------------------------------------------

    full_cf = float(
        df["counterfactual_r"].sum()
    )

    full_delta = (
        full_cf
        - baseline_r
    )

    print()
    print("=" * 100)
    print("CORRECT FULL-TRADE COUNTERFACTUAL")
    print("=" * 100)

    print(
        f"Baseline total R:             "
        f"{baseline_r:.2f}R"
    )

    print(
        f"Full counterfactual R:        "
        f"{full_cf:.2f}R"
    )

    print(
        f"Full counterfactual delta:    "
        f"{full_delta:+.2f}R"
    )

    print()
    print(
        "NO_TRIGGER trades retain their actual outcomes."
    )

    # ---------------------------------------------------------------------
    # TIME ROBUSTNESS
    # ---------------------------------------------------------------------

    time_slices = build_time_slices(
        df
    )

    print_slice_table(
        "TIME ROBUSTNESS",
        time_slices
    )

    # ---------------------------------------------------------------------
    # LATE REVERSAL
    # ---------------------------------------------------------------------

    late_analysis = (
        build_late_reversal_analysis(
            df
        )
    )

    print_slice_table(
        "LATE-REVERSAL DEPENDENCE",
        late_analysis
    )

    without_late = next(
        row
        for row in late_analysis
        if row["slice"]
        == "ALL_OTHER_LC_TRADES"
    )

    # ---------------------------------------------------------------------
    # TRIGGER PARTICIPATION
    # ---------------------------------------------------------------------

    trigger_analysis = (
        build_trigger_analysis(
            df
        )
    )

    print_slice_table(
        "TRIGGER PARTICIPATION",
        trigger_analysis
    )

    # ---------------------------------------------------------------------
    # LEAVE ONE OUT
    # ---------------------------------------------------------------------

    loo_df = build_leave_one_out(
        df
    )

    loo_positive = int(
        (
            loo_df["remaining_delta_r"]
            > R_TOLERANCE
        ).sum()
    )

    loo_negative = int(
        (
            loo_df["remaining_delta_r"]
            < -R_TOLERANCE
        ).sum()
    )

    loo_unchanged = (
        len(loo_df)
        - loo_positive
        - loo_negative
    )

    print()
    print("=" * 100)
    print("LEAVE-ONE-TRADE-OUT")
    print("=" * 100)

    print(
        f"Positive remaining delta:    "
        f"{loo_positive}"
    )

    print(
        f"Negative remaining delta:    "
        f"{loo_negative}"
    )

    print(
        f"Unchanged:                    "
        f"{loo_unchanged}"
    )

    print()

    for _, row in loo_df.iterrows():

        marker = ""

        if row[
            "removed_was_late_reversal"
        ]:
            marker = " | LATE_REVERSAL"

        print(
            f"Remove {row['removed_entry_date']} | "
            f"status={row['removed_status']} | "
            f"remaining delta="
            f"{row['remaining_delta_r']:+.2f}R | "
            f"removed delta="
            f"{row['removed_delta_r']:+.2f}R"
            f"{marker}"
        )

    # ---------------------------------------------------------------------
    # DELTA CONTRIBUTION
    # ---------------------------------------------------------------------

    late_df = df[
        df["late_reversal"]
    ].copy()

    late_delta = float(
        (
            late_df["counterfactual_r"]
            - late_df["actual_r"]
        ).sum()
    )

    other_delta = (
        full_delta
        - late_delta
    )

    print()
    print("=" * 100)
    print("DELTA CONTRIBUTION")
    print("=" * 100)

    print(
        f"Total counterfactual delta: "
        f"{full_delta:+.2f}R"
    )

    print(
        f"Two late reversals:          "
        f"{late_delta:+.2f}R"
    )

    print(
        f"All other LC trades:         "
        f"{other_delta:+.2f}R"
    )

    if abs(full_delta) > R_TOLERANCE:

        late_share = (
            late_delta
            / full_delta
            * 100.0
        )

        print(
            f"Late-reversal contribution:  "
            f"{late_share:.1f}%"
        )

    # ---------------------------------------------------------------------
    # ACCOUNTING INTEGRITY
    # ---------------------------------------------------------------------

    no_trigger = df[
        df["status"].astype(str)
        == "NO_TRIGGER"
    ]

    no_trigger_difference = float(
        (
            no_trigger["counterfactual_r"]
            - no_trigger["actual_r"]
        ).abs().sum()
    )

    accounting_pass = (
        no_trigger_difference
        <= R_TOLERANCE
    )

    print()
    print("=" * 100)
    print("ACCOUNTING INTEGRITY")
    print("=" * 100)

    print(
        f"NO_TRIGGER actual-path retention: "
        f"{'PASS' if accounting_pass else 'FAIL'}"
    )

    print(
        f"NO_TRIGGER accounting difference: "
        f"{no_trigger_difference:.4f}R"
    )

    # ---------------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------------

    verdict = determine_verdict(
        time_slices,
        loo_df,
        without_late,
    )

    print()
    print("=" * 100)
    print("RESEARCH VERDICT")
    print("=" * 100)

    print(verdict)

    print()
    print(
        "This remains research only."
    )

    print(
        "No production strategy change."
    )

    print(
        "No frozen historical dataset change."
    )

    print(
        "No holdout change."
    )

    # ---------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    slice_df = pd.DataFrame(
        time_slices
        + late_analysis
        + trigger_analysis
    )

    slice_df.to_csv(
        SLICE_OUTPUT,
        index=False
    )

    loo_df.to_csv(
        LOO_OUTPUT,
        index=False
    )

    # Save normalized trade-level accounting too.
    normalized_output = os.path.join(
        OUTPUT_DIR,
        "nifty50_lc_exit_robustness_trade_level.csv"
    )

    df.to_csv(
        normalized_output,
        index=False
    )

    report_lines = [
        "NIFTY 50 LC FIXED EXIT-RULE ROBUSTNESS",
        "",
        "FIXED RULE:",
        "+1.5R trigger -> +1R protection -> +2R target",
        "",
        "ACCOUNTING:",
        "NO_TRIGGER trades retain actual R.",
        "",
        f"Trades: {len(df)}",
        f"Baseline R: {baseline_r:.2f}",
        f"Triggered CF R: {triggered_cf:.2f}",
        f"Full CF R: {full_cf:.2f}",
        f"Full Delta R: {full_delta:+.2f}",
        "",
        "TIME SLICES:",
    ]

    for row in time_slices:

        report_lines.append(
            f"{row['slice']}: "
            f"{row['trades']} trades, "
            f"baseline {row['baseline_r']:+.2f}R, "
            f"CF {row['counterfactual_r']:+.2f}R, "
            f"delta {row['delta_r']:+.2f}R"
        )

    report_lines.extend(
        [
            "",
            "LATE REVERSALS:",
            f"Late reversal delta: "
            f"{late_delta:+.2f}R",
            f"All other LC delta: "
            f"{other_delta:+.2f}R",
            "",
            "LEAVE ONE OUT:",
            f"Positive: {loo_positive}",
            f"Negative: {loo_negative}",
            f"Unchanged: {loo_unchanged}",
            "",
            "ACCOUNTING:",
            f"NO_TRIGGER retention: "
            f"{'PASS' if accounting_pass else 'FAIL'}",
            "",
            f"Verdict: {verdict}",
            "",
            "No production/frozen/holdout modifications.",
        ]
    )

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)

    print(
        f"Slices:      {SLICE_OUTPUT}"
    )

    print(
        f"LOO:         {LOO_OUTPUT}"
    )

    print(
        f"Trade-level: {normalized_output}"
    )

    print(
        f"Report:      {REPORT_OUTPUT}"
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()