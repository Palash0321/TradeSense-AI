"""
TradeSense-AI — Bar 1→2 Conditional Incremental Information Forensic Analysis

Research-only forensic test.

Frozen hypothesis:
    5-bar / 0.25R early-adverse event

Frozen population:
    106 trades
    60 frozen early-adverse events
    46 retained

Question:
    Does the Bar-2 development state retain information about the frozen
    Bar-5 event after conditioning on a pre-specified Bar-1 state?

Pre-specified families:

    A. Bar-1 net-path state  -> Bar-2 net deterioration
    B. Bar-1 favorable state -> Bar-2 favorable failure
    C. Bar-1 adverse state   -> Bar-2 net deterioration

Bar-1 states are fixed global median splits.

No:
    - threshold optimization
    - ML
    - arbitrary cut-point search
    - production changes
    - OOS data
"""

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/"
    "early_adverse_bar12_conditional_incremental"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

ALPHA = 0.05
FAMILY_COUNT = 3
BONFERRONI_ALPHA = ALPHA / FAMILY_COUNT


# ============================================================
# PRE-SPECIFIED TEST FAMILIES
# ============================================================

FAMILY_SPECS = [
    {
        "family": "A_net_path",
        "bar1_feature": "bar1_net_excursion_r",
        "bar1_label": "Bar-1 net-path state",
        "bar2_state": "bar2_net_deteriorated",
        "bar2_label": "Bar-2 net deterioration",
    },
    {
        "family": "B_favorable",
        "bar1_feature": "bar1_cumulative_mfe_r",
        "bar1_label": "Bar-1 favorable-excursion state",
        "bar2_state": "bar2_favorable_failed_to_improve",
        "bar2_label": "Bar-2 favorable failure",
    },
    {
        "family": "C_adverse",
        "bar1_feature": "bar1_cumulative_mae_r",
        "bar1_label": "Bar-1 adverse-excursion state",
        "bar2_state": "bar2_net_deteriorated",
        "bar2_label": "Bar-2 net deterioration",
    },
]


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    raise RuntimeError(f"\nFAIL: {message}")


def check_required_columns(df):
    required = {
        "trade_id",
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",
    }

    for spec in FAMILY_SPECS:
        required.add(spec["bar1_feature"])
        required.add(spec["bar2_state"])

    missing = sorted(required - set(df.columns))

    if missing:
        fail(f"Missing required columns: {missing}")


def wilson_interval(x, n, z=1.959963984540054):

    if n == 0:
        return np.nan, np.nan

    p = x / n

    denominator = 1 + (z * z / n)

    centre = (
        p + (z * z / (2 * n))
    ) / denominator

    half = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + (z * z / (4 * n * n))
        )
        / denominator
    )

    return centre - half, centre + half


def risk_difference(a1, n1, a0, n0):

    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan

    p1 = a1 / n1
    p0 = a0 / n0

    rd = p1 - p0

    # Conservative normal approximation using Wilson-compatible
    # component proportions.
    se = math.sqrt(
        max(p1 * (1 - p1) / n1, 0)
        + max(p0 * (1 - p0) / n0, 0)
    )

    z = 1.959963984540054

    lo = rd - z * se
    hi = rd + z * se

    return rd, lo, hi


def build_2x2_table(frame, state_column):
    """
    Rows:
        frozen_event = 1
        frozen_event = 0

    Columns:
        Bar2 state = 1
        Bar2 state = 0
    """

    event_state1 = int(
        (
            (frame["frozen_event"] == 1)
            & (frame[state_column] == 1)
        ).sum()
    )

    event_state0 = int(
        (
            (frame["frozen_event"] == 1)
            & (frame[state_column] == 0)
        ).sum()
    )

    retained_state1 = int(
        (
            (frame["frozen_event"] == 0)
            & (frame[state_column] == 1)
        ).sum()
    )

    retained_state0 = int(
        (
            (frame["frozen_event"] == 0)
            & (frame[state_column] == 0)
        ).sum()
    )

    return np.array(
        [
            [event_state1, event_state0],
            [retained_state1, retained_state0],
        ],
        dtype=int,
    )


def analyze_stratum(frame, state_column, stratum_name):

    table = build_2x2_table(
        frame,
        state_column
    )

    event_total = int(table[0].sum())
    retained_total = int(table[1].sum())

    state1_total = int(table[:, 0].sum())
    state0_total = int(table[:, 1].sum())

    event_rate_state1 = (
        table[0, 0] / state1_total
        if state1_total > 0
        else np.nan
    )

    event_rate_state0 = (
        table[0, 1] / state0_total
        if state0_total > 0
        else np.nan
    )

    if (
        np.isfinite(event_rate_state1)
        and np.isfinite(event_rate_state0)
    ):
        rd = event_rate_state1 - event_rate_state0
    else:
        rd = np.nan

    if (
        table.sum() > 0
        and np.all(table.sum(axis=0) > 0)
    ):
        odds_ratio, fisher_p = fisher_exact(
            table,
            alternative="two-sided"
        )
    else:
        odds_ratio = np.nan
        fisher_p = np.nan

    return {
        "stratum": stratum_name,
        "n": int(len(frame)),
        "events": event_total,
        "retained": retained_total,
        "bar2_state1": state1_total,
        "bar2_state0": state0_total,
        "event_state1": int(table[0, 0]),
        "event_state0": int(table[0, 1]),
        "retained_state1": int(table[1, 0]),
        "retained_state0": int(table[1, 1]),
        "event_rate_state1": event_rate_state1,
        "event_rate_state0": event_rate_state0,
        "risk_difference": rd,
        "risk_difference_pp": (
            rd * 100
            if np.isfinite(rd)
            else np.nan
        ),
        "odds_ratio": odds_ratio,
        "fisher_p": fisher_p,
        "table": table.tolist(),
    }


def cmh_test(tables):
    """
    Cochran-Mantel-Haenszel test across Bar-1 strata.

    Computes the common CMH odds-ratio estimate and the
    large-sample CMH chi-square statistic directly.

    No statsmodels dependency required.

    Only strata containing both Bar-2 states are included.
    """

    valid_tables = []

    for table in tables:

        table = np.asarray(
            table,
            dtype=float
        )

        # Table:
        #
        #             Bar2=1   Bar2=0
        # Event          a         b
        # Retained       c         d
        #
        # Require both Bar2 states to be represented.

        if (
            table[:, 0].sum() > 0
            and table[:, 1].sum() > 0
        ):
            valid_tables.append(table)

    if not valid_tables:

        return {
            "cmh_odds_ratio": np.nan,
            "cmh_p": np.nan,
            "valid_strata": 0,
        }

    # --------------------------------------------------------
    # CMH pooled odds ratio
    #
    # OR_MH = sum(a_i*d_i/n_i) /
    #         sum(b_i*c_i/n_i)
    # --------------------------------------------------------

    numerator = 0.0
    denominator = 0.0

    for table in valid_tables:

        a = table[0, 0]
        b = table[0, 1]
        c = table[1, 0]
        d = table[1, 1]

        n = a + b + c + d

        if n <= 0:
            continue

        numerator += (
            a * d / n
        )

        denominator += (
            b * c / n
        )

    if denominator == 0:

        pooled_or = np.inf

    else:

        pooled_or = (
            numerator / denominator
        )

    # --------------------------------------------------------
    # CMH test of common OR = 1
    #
    # For each stratum:
    #
    # E[a_i] = ((a+b)*(a+c))/n
    #
    # Var(a_i) =
    # ((a+b)(c+d)(a+c)(b+d)) /
    # (n^2 (n-1))
    #
    # X²_MH =
    # (|sum(a-Ea)| - 0.5)^2 /
    # sum(Var(a))
    #
    # Continuity correction is used.
    # --------------------------------------------------------

    numerator_deviation = 0.0
    variance_sum = 0.0

    for table in valid_tables:

        a = table[0, 0]
        b = table[0, 1]
        c = table[1, 0]
        d = table[1, 1]

        n = a + b + c + d

        if n <= 1:
            continue

        expected_a = (
            (a + b)
            * (a + c)
            / n
        )

        variance_a = (
            (a + b)
            * (c + d)
            * (a + c)
            * (b + d)
            / (
                n * n * (n - 1)
            )
        )

        numerator_deviation += (
            a - expected_a
        )

        variance_sum += (
            variance_a
        )

    if variance_sum <= 0:

        cmh_chi2 = np.nan
        cmh_p = np.nan

    else:

        corrected_difference = (
            abs(numerator_deviation)
            - 0.5
        )

        # Do not allow continuity correction
        # to create a negative numerator.
        corrected_difference = max(
            corrected_difference,
            0.0
        )

        cmh_chi2 = (
            corrected_difference ** 2
            / variance_sum
        )

        # Chi-square with 1 degree of freedom.
        #
        # Survival function:
        # P(X² >= observed)
        #
        # scipy is already available because
        # fisher_exact is imported above.
        from scipy.stats import chi2

        cmh_p = float(
            chi2.sf(
                cmh_chi2,
                df=1
            )
        )

    return {
        "cmh_odds_ratio": float(
            pooled_or
        ),
        "cmh_p": cmh_p,
        "valid_strata": len(
            valid_tables
        ),
    }

# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 90)
    print(
        "TRADE SENSE-AI — BAR 1→2 "
        "CONDITIONAL INCREMENTAL INFORMATION"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        fail(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input rows: {len(df)}, "
        f"columns: {len(df.columns)}"
    )

    check_required_columns(df)

    # --------------------------------------------------------
    # FROZEN POPULATION VALIDATION
    # --------------------------------------------------------

    if len(df) != EXPECTED_TRADES:

        fail(
            f"Expected {EXPECTED_TRADES} rows, "
            f"found {len(df)}"
        )

    if df["trade_id"].nunique() != EXPECTED_TRADES:

        fail(
            "trade_id is not unique across "
            "the 106 trade-level rows."
        )

    df["frozen_event"] = pd.to_numeric(
        df["frozen_event"],
        errors="coerce"
    )

    if df["frozen_event"].isna().any():

        fail(
            "frozen_event contains NaN."
        )

    unique_labels = set(
        df["frozen_event"].unique()
    )

    if unique_labels != {0, 1}:

        fail(
            "frozen_event must contain exactly "
            "{0, 1}; found "
            f"{sorted(unique_labels)}"
        )

    event_count = int(
        (df["frozen_event"] == 1).sum()
    )

    retained_count = int(
        (df["frozen_event"] == 0).sum()
    )

    if event_count != EXPECTED_EVENTS:

        fail(
            f"Expected {EXPECTED_EVENTS} events, "
            f"found {event_count}"
        )

    if retained_count != EXPECTED_RETAINED:

        fail(
            f"Expected {EXPECTED_RETAINED} retained, "
            f"found {retained_count}"
        )

    print(
        f"Frozen population: "
        f"{len(df)} / "
        f"{event_count} / "
        f"{retained_count}"
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    primary_results = []
    stratum_results_all = []

    # --------------------------------------------------------
    # RUN EACH PRE-SPECIFIED FAMILY
    # --------------------------------------------------------

    for spec in FAMILY_SPECS:

        family = spec["family"]
        bar1_feature = spec["bar1_feature"]
        bar2_state = spec["bar2_state"]

        print()
        print("-" * 90)
        print(f"FAMILY: {family}")
        print("-" * 90)

        values = pd.to_numeric(
            df[bar1_feature],
            errors="coerce"
        )

        if values.isna().any():

            fail(
                f"{bar1_feature} contains NaN. "
                "Cannot construct fixed median split."
            )

        # ----------------------------------------------------
        # FIXED GLOBAL MEDIAN
        # ----------------------------------------------------

        median = float(
            values.median()
        )

        df["_bar1_state"] = (
            values >= median
        ).astype(int)

        print(
            f"Bar-1 feature : {bar1_feature}"
        )

        print(
            f"Bar-1 median  : {median:.6f}"
        )

        print(
            f"Bar-2 state   : {spec['bar2_label']}"
        )

        # ----------------------------------------------------
        # TWO BAR-1 STRATA
        # ----------------------------------------------------

        family_strata = []
        cmh_tables = []

        for state_value, state_name in [
            (0, "LOW"),
            (1, "HIGH"),
        ]:

            frame = df[
                df["_bar1_state"] == state_value
            ].copy()

            result = analyze_stratum(
                frame,
                bar2_state,
                state_name
            )

            result["family"] = family
            result["bar1_feature"] = bar1_feature
            result["bar1_median"] = median
            result["bar2_state"] = bar2_state

            family_strata.append(
                result
            )

            cmh_tables.append(
                result["table"]
            )

            export_result = dict(
                result
            )

            export_result.pop(
                "table",
                None
            )

            stratum_results_all.append(
                export_result
            )

            print(
                f"{state_name:>5} | "
                f"n={result['n']:>3} | "
                f"events={result['events']:>2} | "
                f"state1={result['bar2_state1']:>2} | "
                f"state0={result['bar2_state0']:>2} | "
                f"RD={result['risk_difference_pp']:+.2f} pp | "
                f"Fisher p={result['fisher_p']:.6f}"
            )

        # ----------------------------------------------------
        # CMH
        # ----------------------------------------------------

        cmh = cmh_test(
            cmh_tables
        )

        usable_rds = [
            r["risk_difference"]
            for r in family_strata
            if np.isfinite(
                r["risk_difference"]
            )
        ]

        nonzero_rds = [
            rd
            for rd in usable_rds
            if rd != 0
        ]

        # All pre-specified hypotheses expect positive
        # event-rate difference for the Bar-2 adverse state.
        direction_consistent = bool(
            nonzero_rds
            and all(
                rd > 0
                for rd in nonzero_rds
            )
        )

        conditional_significance = bool(
            np.isfinite(
                cmh["cmh_p"]
            )
            and cmh["cmh_p"]
            < BONFERRONI_ALPHA
        )

        family_pass = bool(
            conditional_significance
            and direction_consistent
            and cmh["valid_strata"] >= 1
        )

        primary = {
            "family": family,
            "bar1_feature": bar1_feature,
            "bar1_median": median,
            "bar2_state": bar2_state,
            "bar2_label": spec["bar2_label"],
            "cmh_odds_ratio": cmh["cmh_odds_ratio"],
            "cmh_p": cmh["cmh_p"],
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "valid_strata": cmh["valid_strata"],
            "direction_consistent": direction_consistent,
            "conditional_significance": conditional_significance,
            "family_pass": family_pass,
        }

        primary_results.append(
            primary
        )

        print()
        print(
            f"CMH pooled OR : "
            f"{cmh['cmh_odds_ratio']:.6f}"
        )

        print(
            f"CMH p-value   : "
            f"{cmh['cmh_p']:.6f}"
        )

        print(
            f"Bonferroni α  : "
            f"{BONFERRONI_ALPHA:.6f}"
        )

        print(
            f"Direction     : "
            f"{direction_consistent}"
        )

        print(
            f"Significant   : "
            f"{conditional_significance}"
        )

        print(
            f"FAMILY RESULT : "
            f"{'PASS' if family_pass else 'FAIL'}"
        )

    # ========================================================
    # OUTPUTS
    # ========================================================

    primary_df = pd.DataFrame(
        primary_results
    )

    strata_df = pd.DataFrame(
        stratum_results_all
    )

    primary_file = (
        OUTPUT_DIR
        / "bar12_conditional_incremental_primary.csv"
    )

    strata_file = (
        OUTPUT_DIR
        / "bar12_conditional_incremental_strata.csv"
    )

    integrity_rows = [
        {
            "check": "trade_count",
            "expected": EXPECTED_TRADES,
            "actual": len(df),
            "status": (
                "PASS"
                if len(df) == EXPECTED_TRADES
                else "FAIL"
            ),
        },
        {
            "check": "event_count",
            "expected": EXPECTED_EVENTS,
            "actual": event_count,
            "status": (
                "PASS"
                if event_count == EXPECTED_EVENTS
                else "FAIL"
            ),
        },
        {
            "check": "retained_count",
            "expected": EXPECTED_RETAINED,
            "actual": retained_count,
            "status": (
                "PASS"
                if retained_count == EXPECTED_RETAINED
                else "FAIL"
            ),
        },
        {
            "check": "trade_level_input_rows",
            "expected": EXPECTED_TRADES,
            "actual": len(df),
            "status": (
                "PASS"
                if len(df) == EXPECTED_TRADES
                else "FAIL"
            ),
        },
        {
            "check": "family_count",
            "expected": FAMILY_COUNT,
            "actual": len(primary_results),
            "status": (
                "PASS"
                if len(primary_results) == FAMILY_COUNT
                else "FAIL"
            ),
        },
        {
            "check": "stratum_rows",
            "expected": FAMILY_COUNT * 2,
            "actual": len(strata_df),
            "status": (
                "PASS"
                if len(strata_df) == FAMILY_COUNT * 2
                else "FAIL"
            ),
        },
    ]

    integrity_df = pd.DataFrame(
        integrity_rows
    )

    integrity_file = (
        OUTPUT_DIR
        / "bar12_conditional_incremental_integrity.csv"
    )

    primary_df.to_csv(
        primary_file,
        index=False
    )

    strata_df.to_csv(
        strata_file,
        index=False
    )

    integrity_df.to_csv(
        integrity_file,
        index=False
    )

    family_pass_count = int(
        primary_df["family_pass"].sum()
    )

    summary = {
        "input_file": str(INPUT_FILE),
        "frozen_population": {
            "trades": len(df),
            "events": event_count,
            "retained": retained_count,
        },
        "alpha": ALPHA,
        "family_count": FAMILY_COUNT,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "family_pass_count": family_pass_count,
        "conditional_evidence": (
            family_pass_count > 0
        ),
        "deployment_rule_promoted": False,
        "production_modified": False,
        "oos_modified": False,
    }

    summary_file = (
        OUTPUT_DIR
        / "bar12_conditional_incremental_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # FINAL INTEGRITY
    # ========================================================

    if integrity_df["status"].eq(
        "FAIL"
    ).any():

        fail(
            "One or more integrity checks failed."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 90)
    print("FINAL RESULT")
    print("=" * 90)

    print(
        f"Conditional family PASS count: "
        f"{family_pass_count} / {FAMILY_COUNT}"
    )

    if family_pass_count > 0:

        print(
            "CONDITIONAL EVIDENCE: PASS"
        )

        print(
            "At least one pre-specified Bar-2 state "
            "retains conditional association with "
            "the frozen event after accounting for "
            "the corresponding Bar-1 median state."
        )

    else:

        print(
            "CONDITIONAL EVIDENCE: FAIL"
        )

        print(
            "No pre-specified Bar-2 family retained "
            "sufficient conditional evidence after "
            "accounting for Bar-1."
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This remains association with the frozen "
        "5-bar / 0.25R event label."
    )

    print(
        "It does NOT establish causality, "
        "independent OOS predictive power, "
        "economic value, or a deployable trading rule."
    )

    print(
        "Production strategy logic and OOS data "
        "remain untouched."
    )

    print()
    print("Output files:")
    print(primary_file)
    print(strata_file)
    print(integrity_file)
    print(summary_file)


if __name__ == "__main__":
    main()