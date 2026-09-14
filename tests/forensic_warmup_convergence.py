"""
TradeSense-AI
FORENSIC WARM-UP CONVERGENCE TEST

Purpose:
    Determine how much historical warm-up is required before the
    structure/trend pipeline produces the same state as the stable
    research context.

Comparison:

    Stable reference:
        2021-01-01

    Test contexts:
        2021-01-01
        2021-03-01
        2021-05-01
        2021-07-01
        2021-08-01
        2021-09-01
        2021-09-13

IMPORTANT:
    Diagnostic only.
    Does NOT modify production code.
    Does NOT modify the frozen historical dataset.
"""

from pathlib import Path

import pandas as pd

from app.services.backtest_service import BacktestService


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

REFERENCE_START = "2021-01-01"

WARMUP_STARTS = [
    "2021-01-01",
    "2021-03-01",
    "2021-05-01",
    "2021-07-01",
    "2021-08-01",
    "2021-09-01",
    "2021-09-13",
]

OUTPUT_DIR = Path("tests/output/reconciliation")

OUTPUT_FILE = (
    OUTPUT_DIR / "warmup_convergence_forensic.csv"
)

FIELD_SUMMARY_FILE = (
    OUTPUT_DIR / "warmup_convergence_field_summary.csv"
)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_value(value):
    """
    Normalize values before comparison.
    """

    if pd.isna(value):
        return None

    if isinstance(value, float):
        return round(value, 8)

    return value


def compare_signals(reference, candidate, columns):
    """
    Compare aligned signal rows.

    Returns:
        total common rows
        total different rows
        field-level difference counts
    """

    common_dates = reference.index.intersection(
        candidate.index
    )

    field_counts = {
        column: 0
        for column in columns
    }

    differing_rows = 0

    for date in common_dates:

        reference_row = reference.loc[date]
        candidate_row = candidate.loc[date]

        row_different = False

        for column in columns:

            reference_value = normalize_value(
                reference_row.get(column)
            )

            candidate_value = normalize_value(
                candidate_row.get(column)
            )

            if reference_value != candidate_value:

                field_counts[column] += 1
                row_different = True

        if row_different:

            differing_rows += 1

    return (
        len(common_dates),
        differing_rows,
        field_counts,
    )


def get_comparison_columns(reference, candidate):
    """
    Fields specifically relevant to structural/trend convergence.
    """

    preferred = [
        "Structure_Bias",
        "Structure",
        "BOS",
        "CHOCH",
        "Break_Direction",
        "Break_Confirmed",
        "Break_Age",
        "Setup",
        "Setup_Direction",
        "Setup_Confidence",
        "Setup_Reason",
        "Trend",
        "Trend_Strength",
        "Trend_Slope",
    ]

    return [
        column
        for column in preferred
        if column in reference.columns
        and column in candidate.columns
    ]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []
    all_field_results = []

    print()
    print("=" * 110)
    print("TRADESENSE-AI — WARM-UP CONVERGENCE FORENSIC")
    print("=" * 110)
    print()

    print(
        f"Reference context: {REFERENCE_START}"
    )

    print(
        "Test contexts:"
    )

    for start in WARMUP_STARTS:

        print(
            f"  {start}"
        )

    print()

    # =================================================================
    # SYMBOL LOOP
    # =================================================================

    for symbol in SYMBOLS:

        print()
        print("=" * 110)
        print(f"SYMBOL: {symbol}")
        print("=" * 110)

        # -------------------------------------------------------------
        # Reference dataset
        # -------------------------------------------------------------

        reference_service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=100000,
        )

        reference_data = reference_service.load_data(
            start_date=REFERENCE_START
        )

        reference_signals = (
            reference_service.generate_tradesense_signals(
                df=reference_data
            )
        )

        reference_signals = reference_signals.copy()

        reference_signals.index = pd.to_datetime(
            reference_signals.index
        )

        print(
            f"Reference signal rows: "
            f"{len(reference_signals)}"
        )

        # -------------------------------------------------------------
        # Test each warm-up
        # -------------------------------------------------------------

        for warmup_start in WARMUP_STARTS:

            print()
            print(
                "-" * 110
            )

            print(
                f"Warm-up start: {warmup_start}"
            )

            # ---------------------------------------------------------
            # Load candidate context
            # ---------------------------------------------------------

            candidate_service = BacktestService(
                symbol=symbol,
                strategy="tradesense",
                initial_capital=100000,
            )

            candidate_data = candidate_service.load_data(
                start_date=warmup_start
            )

            candidate_signals = (
                candidate_service.generate_tradesense_signals(
                    df=candidate_data
                )
            )

            candidate_signals = candidate_signals.copy()

            candidate_signals.index = pd.to_datetime(
                candidate_signals.index
            )

            # ---------------------------------------------------------
            # Only compare dates that exist in BOTH datasets.
            # ---------------------------------------------------------

            common_dates = (
                reference_signals.index.intersection(
                    candidate_signals.index
                )
            )

            if len(common_dates) == 0:

                print(
                    "No common dates."
                )

                continue

            comparison_columns = (
                get_comparison_columns(
                    reference_signals,
                    candidate_signals,
                )
            )

            (
                common_count,
                differing_rows,
                field_counts,
            ) = compare_signals(
                reference_signals,
                candidate_signals,
                comparison_columns,
            )

            matching_rows = (
                common_count - differing_rows
            )

            if common_count > 0:

                convergence_pct = (
                    matching_rows
                    / common_count
                    * 100
                )

            else:

                convergence_pct = 0.0

            print(
                f"Common rows       : {common_count}"
            )

            print(
                f"Matching rows     : {matching_rows}"
            )

            print(
                f"Differing rows    : {differing_rows}"
            )

            print(
                f"Convergence       : "
                f"{convergence_pct:.2f}%"
            )

            # ---------------------------------------------------------
            # Record overall result
            # ---------------------------------------------------------

            all_results.append(
                {
                    "symbol": symbol,
                    "reference_start": REFERENCE_START,
                    "warmup_start": warmup_start,
                    "common_rows": common_count,
                    "matching_rows": matching_rows,
                    "differing_rows": differing_rows,
                    "convergence_pct": round(
                        convergence_pct,
                        4,
                    ),
                }
            )

            # ---------------------------------------------------------
            # Field-level results
            # ---------------------------------------------------------

            for field, count in field_counts.items():

                all_field_results.append(
                    {
                        "symbol": symbol,
                        "warmup_start": warmup_start,
                        "field": field,
                        "difference_count": count,
                    }
                )

        # -------------------------------------------------------------
        # Symbol summary
        # -------------------------------------------------------------

        print()
        print(
            f"Completed {symbol}"
        )

    # =================================================================
    # SAVE RESULTS
    # =================================================================

    results_df = pd.DataFrame(
        all_results
    )

    field_df = pd.DataFrame(
        all_field_results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    field_df.to_csv(
        FIELD_SUMMARY_FILE,
        index=False,
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print("=" * 110)
    print("WARM-UP CONVERGENCE SUMMARY")
    print("=" * 110)

    if not results_df.empty:

        summary = (
            results_df
            .groupby("warmup_start")
            .agg(
                common_rows=("common_rows", "sum"),
                matching_rows=("matching_rows", "sum"),
                differing_rows=("differing_rows", "sum"),
            )
            .reset_index()
        )

        summary["convergence_pct"] = (
            summary["matching_rows"]
            / summary["common_rows"]
            * 100
        )

        print()

        print(
            summary.to_string(
                index=False
            )
        )

        # -------------------------------------------------------------
        # Best convergence point
        # -------------------------------------------------------------

        best_row = summary.loc[
            summary["convergence_pct"].idxmax()
        ]

        print()
        print("=" * 110)
        print("BEST CONVERGENCE")
        print("=" * 110)

        print(
            f"Warm-up start : "
            f"{best_row['warmup_start']}"
        )

        print(
            f"Convergence   : "
            f"{best_row['convergence_pct']:.2f}%"
        )

        print(
            f"Differing rows: "
            f"{int(best_row['differing_rows'])}"
        )

    # =================================================================
    # FIELD-LEVEL GLOBAL SUMMARY
    # =================================================================

    if not field_df.empty:

        print()
        print("=" * 110)
        print("FIELD-LEVEL DIFFERENCES BY WARM-UP")
        print("=" * 110)

        field_pivot = (
            field_df
            .pivot_table(
                index="warmup_start",
                columns="field",
                values="difference_count",
                aggfunc="sum",
                fill_value=0,
            )
        )

        print()

        print(
            field_pivot.to_string()
        )

    # =================================================================
    # OUTPUT FILES
    # =================================================================

    print()
    print("=" * 110)
    print("OUTPUT FILES")
    print("=" * 110)

    print(
        f"Overall: {OUTPUT_FILE}"
    )

    print(
        f"Fields : {FIELD_SUMMARY_FILE}"
    )

    print()

    # =================================================================
    # INTERPRETATION
    # =================================================================

    if not results_df.empty:

        maximum_convergence = (
            results_df["convergence_pct"].max()
        )

        minimum_difference = (
            results_df["differing_rows"].min()
        )

        print("=" * 110)
        print("INTERPRETATION")
        print("=" * 110)

        print()

        if maximum_convergence >= 99.9:

            print(
                "STRONG CONVERGENCE:"
            )

            print(
                "At least one warm-up window reproduces "
                "the stable reference state almost completely."
            )

            print(
                "This strongly supports warm-up dependency "
                "as the explanation for the rolling-context divergence."
            )

        elif maximum_convergence >= 95:

            print(
                "HIGH CONVERGENCE:"
            )

            print(
                "A sufficiently long warm-up substantially "
                "reproduces the stable reference state."
            )

            print(
                "Further root-cause analysis is required "
                "before selecting a production warm-up."
            )

        else:

            print(
                "NO SUFFICIENT CONVERGENCE YET:"
            )

            print(
                "Even the tested warm-up windows retain "
                "material state differences."
            )

            print(
                "Do NOT modify production logic based on this test alone."
            )

        print()
        print(
            f"Best convergence observed: "
            f"{maximum_convergence:.2f}%"
        )

        print(
            f"Minimum differing rows observed: "
            f"{minimum_difference}"
        )


if __name__ == "__main__":
    main()