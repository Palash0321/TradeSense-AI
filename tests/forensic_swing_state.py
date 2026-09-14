"""
TradeSense-AI
FORENSIC SWING-STATE ROOT-CAUSE ANALYSIS

Purpose:
    Compare the stable research context (2021-01-01 onward)
    against the rolling 5-year context and determine whether
    structural divergence originates from:

        1. Swing detection
        2. Structure classification
        3. Break/BOS/CHOCH logic
        4. Trend / Trend Strength

IMPORTANT:
    Diagnostic only.
    No production strategy code is modified.
    No historical dataset is overwritten.
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

STABLE_START_DATE = "2021-01-01"

OUTPUT_DIR = Path("tests/output/reconciliation")
OUTPUT_FILE = OUTPUT_DIR / "swing_state_forensic.csv"

# Current rolling context is whatever the production load_data()
# returns when no explicit start/end date is supplied.


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def normalize_value(value):
    """
    Convert values into stable comparison-friendly representations.
    """
    if pd.isna(value):
        return None

    if isinstance(value, float):
        return round(value, 8)

    return value


def get_candidate_columns(df):
    """
    Dynamically identify columns related to swing / structure / break /
    trend state.

    This avoids assuming exact internal column names beyond the
    high-level fields already exposed by the signal pipeline.
    """

    keywords = [
        "swing",
        "structure",
        "bos",
        "choch",
        "break",
        "trend",
    ]

    selected = []

    for column in df.columns:
        lowered = str(column).lower()

        if any(keyword in lowered for keyword in keywords):
            selected.append(column)

    return selected


def compare_rows(stable_row, rolling_row, columns):
    """
    Return field-level differences between two aligned signal rows.
    """

    differences = []

    for column in columns:

        stable_value = normalize_value(stable_row.get(column))
        rolling_value = normalize_value(rolling_row.get(column))

        if stable_value != rolling_value:
            differences.append(column)

    return differences


def format_value(value):
    """
    Compact printable representation.
    """

    if pd.isna(value):
        return "None"

    return str(value)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    print()
    print("=" * 110)
    print("TRADESENSE-AI — SWING STATE ROOT-CAUSE FORENSIC")
    print("=" * 110)
    print()
    print(f"Stable context : {STABLE_START_DATE}")
    print("Rolling context: current production 5-year window")
    print()

    for symbol in SYMBOLS:

        print()
        print("=" * 110)
        print(f"SYMBOL: {symbol}")
        print("=" * 110)

        # -------------------------------------------------------------
        # Stable context
        # -------------------------------------------------------------

        stable_service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=100000,
        )

        stable_data = stable_service.load_data(
            start_date=STABLE_START_DATE
        )

        stable_signals = stable_service.generate_tradesense_signals(
            df=stable_data
        )

        # -------------------------------------------------------------
        # Rolling context
        # -------------------------------------------------------------

        rolling_service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=100000,
        )

        rolling_data = rolling_service.load_data()

        rolling_signals = rolling_service.generate_tradesense_signals(
            df=rolling_data
        )

        # -------------------------------------------------------------
        # Align signal dates
        # -------------------------------------------------------------

        stable_signals = stable_signals.copy()
        rolling_signals = rolling_signals.copy()

        stable_signals.index = pd.to_datetime(
            stable_signals.index
        )

        rolling_signals.index = pd.to_datetime(
            rolling_signals.index
        )

        common_dates = stable_signals.index.intersection(
            rolling_signals.index
        )

        if len(common_dates) == 0:
            print("No overlapping signal dates.")
            continue

        print(f"Common signal rows: {len(common_dates)}")

        # -------------------------------------------------------------
        # Identify relevant columns
        # -------------------------------------------------------------

        common_columns = [
            column
            for column in stable_signals.columns
            if column in rolling_signals.columns
        ]

        forensic_columns = get_candidate_columns(
            stable_signals[common_columns]
        )

        print()
        print("Forensic columns:")
        print(", ".join(forensic_columns))

        # -------------------------------------------------------------
        # Compare aligned rows
        # -------------------------------------------------------------

        first_divergence_found = False
        divergence_count = 0

        for date in common_dates:

            stable_row = stable_signals.loc[date]
            rolling_row = rolling_signals.loc[date]

            differences = compare_rows(
                stable_row,
                rolling_row,
                forensic_columns,
            )

            if not differences:
                continue

            divergence_count += 1

            # ---------------------------------------------------------
            # Record field-level result
            # ---------------------------------------------------------

            result = {
                "symbol": symbol,
                "date": date.strftime("%Y-%m-%d"),
                "difference_count": len(differences),
                "different_fields": "|".join(differences),
            }

            for field in forensic_columns:

                result[
                    f"stable_{field}"
                ] = normalize_value(
                    stable_row.get(field)
                )

                result[
                    f"rolling_{field}"
                ] = normalize_value(
                    rolling_row.get(field)
                )

            all_results.append(result)

            # ---------------------------------------------------------
            # Print FIRST divergence in detail
            # ---------------------------------------------------------

            if not first_divergence_found:

                first_divergence_found = True

                print()
                print("-" * 110)
                print("FIRST STRUCTURAL / TREND DIVERGENCE")
                print("-" * 110)
                print(f"Date: {date.date()}")
                print()

                for field in forensic_columns:

                    stable_value = normalize_value(
                        stable_row.get(field)
                    )

                    rolling_value = normalize_value(
                        rolling_row.get(field)
                    )

                    if stable_value != rolling_value:

                        print(
                            f"{field:<30} "
                            f"STABLE={format_value(stable_value):<30} "
                            f"ROLLING={format_value(rolling_value)}"
                        )

                # -----------------------------------------------------
                # Print OHLC around first divergence
                # -----------------------------------------------------

                print()
                print("RAW PRICE CONTEXT")
                print("-" * 110)

                try:

                    stable_price_row = stable_data.loc[date]

                    if isinstance(stable_price_row, pd.DataFrame):
                        stable_price_row = stable_price_row.iloc[0]

                    print(
                        "Stable:"
                    )

                    for field in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ]:

                        if field in stable_price_row.index:

                            print(
                                f"  {field:<10}: "
                                f"{format_value(stable_price_row[field])}"
                            )

                except Exception as exc:

                    print(
                        f"Stable price context unavailable: {exc}"
                    )

                try:

                    rolling_price_row = rolling_data.loc[date]

                    if isinstance(rolling_price_row, pd.DataFrame):
                        rolling_price_row = rolling_price_row.iloc[0]

                    print(
                        "Rolling:"
                    )

                    for field in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume",
                    ]:

                        if field in rolling_price_row.index:

                            print(
                                f"  {field:<10}: "
                                f"{format_value(rolling_price_row[field])}"
                            )

                except Exception as exc:

                    print(
                        f"Rolling price context unavailable: {exc}"
                    )

                print()

        print()
        print(f"Total divergent rows: {divergence_count}")

    # -----------------------------------------------------------------
    # SAVE FORENSIC RESULTS
    # -----------------------------------------------------------------

    if all_results:

        forensic_df = pd.DataFrame(all_results)

        forensic_df.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print()
        print("=" * 110)
        print("FORENSIC OUTPUT")
        print("=" * 110)
        print(
            f"Rows saved: {len(forensic_df)}"
        )
        print(
            f"File: {OUTPUT_FILE}"
        )

        # -------------------------------------------------------------
        # Global field divergence counts
        # -------------------------------------------------------------

        print()
        print("=" * 110)
        print("FIELD-LEVEL DIVERGENCE COUNTS")
        print("=" * 110)

        field_counts = {}

        for _, row in forensic_df.iterrows():

            fields = str(
                row["different_fields"]
            ).split("|")

            for field in fields:

                if not field:
                    continue

                field_counts[field] = (
                    field_counts.get(field, 0) + 1
                )

        field_counts = dict(
            sorted(
                field_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        for field, count in field_counts.items():

            print(
                f"{field:<35} {count}"
            )

        # -------------------------------------------------------------
        # Root-cause classification
        # -------------------------------------------------------------

        print()
        print("=" * 110)
        print("ROOT-CAUSE CLASSIFICATION")
        print("=" * 110)

        swing_fields = [
            field
            for field in field_counts
            if "swing" in field.lower()
        ]

        structure_fields = [
            field
            for field in field_counts
            if (
                "structure" in field.lower()
                or "bos" in field.lower()
                or "choch" in field.lower()
                or "break" in field.lower()
            )
        ]

        trend_fields = [
            field
            for field in field_counts
            if "trend" in field.lower()
        ]

        swing_difference_total = sum(
            field_counts[field]
            for field in swing_fields
        )

        structure_difference_total = sum(
            field_counts[field]
            for field in structure_fields
        )

        trend_difference_total = sum(
            field_counts[field]
            for field in trend_fields
        )

        print()
        print(
            f"Swing-related differences    : "
            f"{swing_difference_total}"
        )

        print(
            f"Structure/break differences   : "
            f"{structure_difference_total}"
        )

        print(
            f"Trend-related differences     : "
            f"{trend_difference_total}"
        )

        print()

        if swing_difference_total > 0:

            print(
                "RESULT: SWING DETECTION REQUIRES INVESTIGATION."
            )

            print(
                "At least one swing-related field diverges."
            )

        elif structure_difference_total > 0:

            print(
                "RESULT: SWINGS APPEAR STABLE; "
                "DIVERGENCE IS DOWNSTREAM IN STRUCTURE/BREAK LOGIC."
            )

        elif trend_difference_total > 0:

            print(
                "RESULT: STRUCTURE APPEARS STABLE; "
                "DIVERGENCE IS PRIMARILY IN TREND CALCULATION."
            )

        else:

            print(
                "RESULT: NO SWING / STRUCTURE / TREND DIVERGENCE FOUND."
            )

    else:

        print()
        print("=" * 110)
        print("NO DIVERGENCES FOUND")
        print("=" * 110)


if __name__ == "__main__":
    main()