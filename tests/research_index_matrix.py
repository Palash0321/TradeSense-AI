from pathlib import Path
import csv
from datetime import datetime


# ============================================================
# TradeSense-AI
# Cross-Index Research Matrix
# ============================================================
#
# PURPOSE:
#   Consolidate already-validated independent index research
#   findings into one research matrix.
#
# IMPORTANT:
#   - READ-ONLY research artifact.
#   - Does NOT modify production strategy.
#   - Does NOT modify historical trade ledgers.
#   - Does NOT modify holdout ledgers.
#   - Does NOT combine strategies across indexes.
#   - Does NOT assume transferability between indexes.
#   - Does NOT make deployment decisions.
#
# Indexes covered:
#   1. NIFTY 50
#   2. SENSEX
#   3. BANK NIFTY
#   4. NIFTY Midcap Select
#
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "index_backtests"

CSV_OUTPUT = OUTPUT_DIR / "index_research_matrix.csv"
MD_OUTPUT = OUTPUT_DIR / "index_research_matrix.md"


# ============================================================
# VALIDATED INDEX RESEARCH RECORDS
# ============================================================

INDEXES = [
    {
        "index": "NIFTY 50",
        "source": "^NSEI",
        "source_status": "PASS",
        "research_start": "2021-01-01",
        "research_end": "2026-09-11",
        "rows": 1408,

        "baseline_trades": 22,
        "wins": 6,
        "losses": 16,
        "win_rate": "27.27%",
        "profit_factor": "0.674",
        "total_r": "-5.50R",

        "directional_finding":
            "LONG +2.01R / PF 1.214; SHORT -7.51R. "
            "Short-side weakness is visible but sample is small.",

        "setup_finding":
            "LC +2.27R, LR -0.26R, SC -2.17R, SR -5.34R. "
            "SR is the weakest setup in this sample.",

        "development_finding":
            "Early-adverse 5-bar / 0.25R filtering was diagnostic only. "
            "15 triggered trades -4.27R versus 7 non-triggered trades -1.23R.",

        "counterfactual_finding":
            "LC exit-path counterfactual improved baseline by +2.70R, "
            "but robustness was not established.",

        "robustness_status":
            "ROBUSTNESS_NOT_ESTABLISHED",

        "oos_status":
            "WAITING_FOR_GENUINE_OOS_TRADES",

        "candidate_hypothesis":
            "Early-adverse behavior and LC exit-path management remain "
            "research candidates only.",

        "deployment_blocker":
            "Insufficient index-level sample and no genuine forward validation.",

        "verdict":
            "RESEARCH ONLY",
    },

    {
        "index": "SENSEX",
        "source": "^BSESN",
        "source_status": "PASS",
        "research_start": "2021-01-01",
        "research_end": "2026-09-11",
        "rows": 1405,

        "baseline_trades": 21,
        "wins": 6,
        "losses": 15,
        "win_rate": "28.57%",
        "profit_factor": "0.811",
        "total_r": "-2.75R",

        "directional_finding":
            "LONG +3.63R / PF 1.586; SHORT -6.38R. "
            "Short-side weakness is substantial in-sample but sample is small.",

        "setup_finding":
            "LC +2.70R, LR +0.93R, SC +0.80R, SR -7.18R. "
            "SR is the dominant negative setup.",

        "development_finding":
            "Early-adverse 5-bar / 0.25R diagnostic produced only "
            "+0.49R improvement and is therefore weak.",

        "counterfactual_finding":
            "SR exit counterfactual improved by +2.02R, but depended on "
            "a very small trigger sample and failed leave-one-out robustness.",

        "robustness_status":
            "INSUFFICIENT_SAMPLE",

        "oos_status":
            "NO_DEPLOYMENT_VALIDATION",

        "candidate_hypothesis":
            "SR development/exit behavior may justify future research, "
            "but no implementation is supported.",

        "deployment_blocker":
            "Tiny setup sample and insufficient robustness/OOS evidence.",

        "verdict":
            "RESEARCH ONLY",
    },

    {
        "index": "BANK NIFTY",
        "source": "^NSEBANK",
        "source_status": "PASS",
        "research_start": "2021-01-01",
        "research_end": "2026-09-11",
        "rows": 1407,

        "baseline_trades": 24,
        "wins": 5,
        "losses": 19,
        "win_rate": "20.83%",
        "profit_factor": "0.518",
        "total_r": "-9.20R",

        "directional_finding":
            "LONG 11 trades versus SHORT 13 trades; overall short-side "
            "and continuation weakness are notable.",

        "setup_finding":
            "LC 9 trades, SC 8, SR 5, LR 2. "
            "SC produced 0W / -8.19R; SR produced 1W / 4L / -2.13R.",

        "development_finding":
            "All 8 SC trades triggered the 5-bar / 0.25R early-adverse "
            "condition and all were losses.",

        "counterfactual_finding":
            "SC early-adverse filtering showed improvement across all "
            "24 tested parameter cells, all 6 years, and all 8 leave-one-out "
            "tests; however, only 8 SC trades exist.",

        "robustness_status":
            "STRONG_IN_SAMPLE_FAILURE_SIGNAL_BUT_TINY_SAMPLE",

        "oos_status":
            "WAITING_FOR_GENUINE_OOS_TRADES",

        "candidate_hypothesis":
            "SC + early-adverse behavior is a strong candidate for "
            "frozen forward validation only.",

        "deployment_blocker":
            "Only 8 SC trades and zero genuine post-holdout trades so far.",

        "verdict":
            "RESEARCH ONLY",
    },

    {
        "index": "NIFTY MIDCAP SELECT",
        "source": "Nifty Indices official historical endpoint",
        "source_status": "PASS",
        "research_start": "2022-01-10",
        "research_end": "2026-09-11",
        "rows": 1160,

        "baseline_trades": 13,
        "wins": 4,
        "losses": 9,
        "win_rate": "30.77%",
        "profit_factor": "0.755",
        "total_r": "-2.25R",

        "directional_finding":
            "LONG 8 trades / -0.14R; SHORT 5 trades / -2.11R. "
            "Sample is insufficient for directional conclusions.",

        "setup_finding":
            "LC +0.89R, LR -1.03R, SC -2.05R, SR -0.06R. "
            "No setup has enough trades for deployment inference.",

        "development_finding":
            "Clear favorable-development gradient: trades reaching "
            "1.25R, 1.50R, and 1.75R had progressively stronger outcomes. "
            "Threshold arrival can be very late.",

        "counterfactual_finding":
            "Exit counterfactual improved baseline by +2.11R. "
            "Sensitivity showed a threshold gradient, but only 13 trades "
            "were available.",

        "robustness_status":
            "INSUFFICIENT_SAMPLE",

        "oos_status":
            "WAITING_FOR_GENUINE_OOS_TRADES",

        "candidate_hypothesis":
            "Favorable-development thresholds may be useful as a future "
            "development-state feature, not as an immediate exit rule.",

        "deployment_blocker":
            "Only 13 historical trades and no genuine forward trades.",

        "verdict":
            "RESEARCH ONLY",
    },
]


# ============================================================
# CROSS-INDEX CLASSIFICATION
# ============================================================

CROSS_INDEX_FINDINGS = [
    {
        "dimension": "Baseline strategy quality",
        "finding":
            "The current baseline is negative on all four independently "
            "tested indexes.",
        "interpretation":
            "The baseline strategy is not yet suitable for deployment "
            "as a universal index strategy.",
        "status": "BLOCKED",
    },
    {
        "dimension": "Data integrity",
        "finding":
            "NIFTY 50, SENSEX, and BANK NIFTY official/Yahoo index datasets "
            "passed integrity checks. MIDSELECT official Nifty Indices data "
            "was validated from 2022-01-10 onward.",
        "interpretation":
            "No current data-integrity issue blocks index research.",
        "status": "PASS",
    },
    {
        "dimension": "Directional behavior",
        "finding":
            "Long-side results are generally stronger than short-side results "
            "in these small index samples.",
        "interpretation":
            "Potential recurring behavior, but not enough evidence to create "
            "a universal long-only or short-filter rule.",
        "status": "RESEARCH ONLY",
    },
    {
        "dimension": "Setup behavior",
        "finding":
            "Different indexes exhibit different dominant weak setups.",
        "interpretation":
            "Setup-specific behavior appears index-dependent. "
            "Do not transfer exclusions between indexes.",
        "status": "INDEX-SPECIFIC",
    },
    {
        "dimension": "Early-adverse behavior",
        "finding":
            "Early-adverse behavior is strongly interesting in some "
            "index/setup combinations, especially BANK NIFTY SC and the "
            "previously validated stock research.",
        "interpretation":
            "Promising failure-development feature, but must be frozen "
            "and independently validated per index.",
        "status": "OOS VALIDATION REQUIRED",
    },
    {
        "dimension": "Favorable development",
        "finding":
            "MIDSELECT shows a strong relationship between favorable "
            "development and eventual trade outcome.",
        "interpretation":
            "Useful as a research feature; not evidence for changing exits.",
        "status": "RESEARCH ONLY",
    },
    {
        "dimension": "Exit counterfactuals",
        "finding":
            "Counterfactual exit improvements appear on multiple indexes, "
            "but robustness varies and samples are small.",
        "interpretation":
            "Counterfactual improvement alone is insufficient for implementation.",
        "status": "NO IMPLEMENTATION",
    },
    {
        "dimension": "OOS evidence",
        "finding":
            "Current forward ledgers contain no genuine post-start completed "
            "trades for the newly established BANK NIFTY and MIDSELECT holdouts.",
        "interpretation":
            "There is currently no legitimate unseen-trade evidence on which "
            "to accept or reject those frozen hypotheses.",
        "status": "WAITING",
    },
    {
        "dimension": "Transferability",
        "finding":
            "No hypothesis has yet demonstrated sufficient cross-index "
            "generalization to justify automatic transfer.",
        "interpretation":
            "Every hypothesis must remain index-specific until independently "
            "validated.",
        "status": "BLOCKED",
    },
    {
        "dimension": "Production strategy",
        "finding":
            "No research result currently meets the standard required to "
            "modify the production strategy.",
        "interpretation":
            "Production remains unchanged.",
        "status": "LOCKED",
    },
]


# ============================================================
# CSV WRITER
# ============================================================

def write_csv():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "index",
        "source",
        "source_status",
        "research_start",
        "research_end",
        "rows",
        "baseline_trades",
        "wins",
        "losses",
        "win_rate",
        "profit_factor",
        "total_r",
        "directional_finding",
        "setup_finding",
        "development_finding",
        "counterfactual_finding",
        "robustness_status",
        "oos_status",
        "candidate_hypothesis",
        "deployment_blocker",
        "verdict",
    ]

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in INDEXES:
            writer.writerow(row)


# ============================================================
# MARKDOWN REPORT
# ============================================================

def write_markdown():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    lines.append("# TradeSense-AI Cross-Index Research Matrix")
    lines.append("")
    lines.append(
        f"Generated: `{generated_at}`"
    )
    lines.append("")

    lines.append("## Research Scope")
    lines.append("")
    lines.append(
        "This document consolidates previously validated independent "
        "index research. It is a research artifact only."
    )
    lines.append("")
    lines.append(
        "**No strategy is combined across indexes. "
        "No transferability is assumed. "
        "No production strategy is modified.**"
    )
    lines.append("")

    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append(
        "| Index | Trades | Wins | Losses | Win Rate | PF | Total R | Status |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---|"
    )

    for row in INDEXES:
        lines.append(
            f"| {row['index']} | "
            f"{row['baseline_trades']} | "
            f"{row['wins']} | "
            f"{row['losses']} | "
            f"{row['win_rate']} | "
            f"{row['profit_factor']} | "
            f"{row['total_r']} | "
            f"{row['verdict']} |"
        )

    lines.append("")

    lines.append("## Independent Index Findings")
    lines.append("")

    for row in INDEXES:
        lines.append(f"### {row['index']}")
        lines.append("")
        lines.append(
            f"- **Data source:** {row['source']}"
        )
        lines.append(
            f"- **Data status:** {row['source_status']}"
        )
        lines.append(
            f"- **Research period:** {row['research_start']} → "
            f"{row['research_end']}"
        )
        lines.append(
            f"- **Rows:** {row['rows']}"
        )
        lines.append(
            f"- **Baseline:** {row['baseline_trades']} trades, "
            f"{row['win_rate']} win rate, "
            f"PF {row['profit_factor']}, "
            f"{row['total_r']}"
        )
        lines.append(
            f"- **Directional finding:** {row['directional_finding']}"
        )
        lines.append(
            f"- **Setup finding:** {row['setup_finding']}"
        )
        lines.append(
            f"- **Development finding:** {row['development_finding']}"
        )
        lines.append(
            f"- **Counterfactual finding:** {row['counterfactual_finding']}"
        )
        lines.append(
            f"- **Robustness:** {row['robustness_status']}"
        )
        lines.append(
            f"- **OOS status:** {row['oos_status']}"
        )
        lines.append(
            f"- **Candidate hypothesis:** {row['candidate_hypothesis']}"
        )
        lines.append(
            f"- **Deployment blocker:** {row['deployment_blocker']}"
        )
        lines.append(
            f"- **Verdict:** **{row['verdict']}**"
        )
        lines.append("")

    lines.append("## Cross-Index Findings")
    lines.append("")

    for finding in CROSS_INDEX_FINDINGS:
        lines.append(f"### {finding['dimension']}")
        lines.append("")
        lines.append(
            f"**Finding:** {finding['finding']}"
        )
        lines.append("")
        lines.append(
            f"**Interpretation:** {finding['interpretation']}"
        )
        lines.append("")
        lines.append(
            f"**Status:** `{finding['status']}`"
        )
        lines.append("")

    lines.append("## Hypothesis Classification")
    lines.append("")
    lines.append(
        "| Hypothesis / Observation | Classification | Action |"
    )
    lines.append(
        "|---|---|---|"
    )
    lines.append(
        "| NIFTY 50 early-adverse behavior | "
        "Promising research candidate | "
        "Require genuine OOS validation |"
    )
    lines.append(
        "| NIFTY 50 LC exit-path behavior | "
        "Interesting but robustness incomplete | "
        "Do not implement |"
    )
    lines.append(
        "| SENSEX SR weakness | "
        "Index-specific / insufficient sample | "
        "No implementation |"
    )
    lines.append(
        "| BANK NIFTY SC + early adverse | "
        "Strong in-sample failure signal | "
        "Frozen forward validation |"
    )
    lines.append(
        "| MIDSELECT favorable-development gradient | "
        "Interesting development feature | "
        "Do not use as exit rule yet |"
    )
    lines.append(
        "| Cross-index strategy transfer | "
        "Not established | "
        "Blocked |"
    )
    lines.append("")

    lines.append("## Current OOS Gate")
    lines.append("")
    lines.append(
        "The next meaningful evidence must come from genuine unseen "
        "post-holdout trades. Existing historical trades must not be "
        "reclassified as OOS trades."
    )
    lines.append("")
    lines.append(
        "Current forward collectors:"
    )
    lines.append("")
    lines.append(
        "- BANK NIFTY: waiting for genuine post-2026-09-14 completed trades."
    )
    lines.append(
        "- MIDSELECT: waiting for genuine post-2026-09-14 completed trades."
    )
    lines.append("")
    lines.append(
        "No tuning should be performed using those future observations."
    )
    lines.append("")

    lines.append("## Production Decision")
    lines.append("")
    lines.append(
        "**NO PRODUCTION STRATEGY CHANGE.**"
    )
    lines.append("")
    lines.append(
        "The current evidence supports continued research and frozen "
        "forward validation, not deployment."
    )
    lines.append("")
    lines.append(
        "Each index remains an independent validation problem."
    )
    lines.append("")

    lines.append("## Final Research Verdict")
    lines.append("")
    lines.append(
        "**TradeSense-AI is not yet ready to deploy an index-level "
        "strategy based on these findings.**"
    )
    lines.append("")
    lines.append(
        "The research has successfully identified several potentially "
        "valuable behaviors, but none currently satisfies the complete "
        "chain of evidence:"
    )
    lines.append("")
    lines.append(
        "baseline → hypothesis → frozen rule → robustness → "
        "genuine OOS validation → independent confirmation → deployment"
    )
    lines.append("")

    lines.append(
        "The project should now prioritize genuine forward evidence "
        "rather than further mining of the same historical samples."
    )
    lines.append("")

    with MD_OUTPUT.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# VALIDATION
# ============================================================

def validate_matrix():
    assert len(INDEXES) == 4

    expected_indexes = {
        "NIFTY 50",
        "SENSEX",
        "BANK NIFTY",
        "NIFTY MIDCAP SELECT",
    }

    actual_indexes = {row["index"] for row in INDEXES}

    assert actual_indexes == expected_indexes

    for row in INDEXES:
        assert row["baseline_trades"] == row["wins"] + row["losses"]
        assert row["source_status"] == "PASS"
        assert row["verdict"] == "RESEARCH ONLY"

    assert all(
        finding["status"] != "DEPLOY"
        for finding in CROSS_INDEX_FINDINGS
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("TradeSense-AI Cross-Index Research Matrix")
    print("=" * 70)

    validate_matrix()

    write_csv()
    write_markdown()

    print()
    print("VALIDATION: PASS")
    print()
    print("Indexes consolidated:")
    for row in INDEXES:
        print(
            f"  {row['index']:<22} "
            f"{row['baseline_trades']:>3} trades  "
            f"{row['total_r']:>7}"
        )

    print()
    print(f"CSV: {CSV_OUTPUT}")
    print(f"MD : {MD_OUTPUT}")
    print()
    print("Production strategy: UNCHANGED")
    print("Historical datasets: READ-ONLY")
    print("Holdout ledgers: READ-ONLY")
    print("Cross-index strategy transfer: NOT ASSUMED")
    print("Deployment recommendation: NONE")
    print()
    print("STATUS: CROSS_INDEX_RESEARCH_MATRIX_COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()