from pathlib import Path

import pandas as pd


FEATURE_FILE = Path(
    "data/nifty50_model2/nifty50_model2_features.csv"
)

WEIGHTED_FEATURES = [
    "weighted_return_1d",
    "weighted_return_20d",
    "weighted_positive_pressure_1d",
    "weighted_positive_pressure_20d",
    "weighted_negative_pressure_1d",
    "weighted_negative_pressure_20d",
    "weighted_constituent_count",
]

TARGET_COLUMN = "target_t1_direction"

EXPECTED_WARMUP_DAYS = 20


def main():

    print("=" * 70)
    print("MODEL 2 FEATURE AUDIT")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Coverage: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Missing counts
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MISSING COUNTS")
    print("=" * 70)

    for column in WEIGHTED_FEATURES:

        missing_count = (
            df[column]
            .isna()
            .sum()
        )

        print(
            f"{column}: "
            f"{missing_count}"
        )

    # --------------------------------------------------------
    # Dates with missing weighted features
    # --------------------------------------------------------

    missing_mask = (
        df[WEIGHTED_FEATURES]
        .isna()
        .any(axis=1)
    )

    missing_dates = (
        df.loc[
            missing_mask,
            "date"
        ]
        .tolist()
    )

    print()
    print("=" * 70)
    print("MISSING WEIGHTED-FEATURE DATES")
    print("=" * 70)

    for date in missing_dates:
        print(date.date())

    # --------------------------------------------------------
    # Verify all weighted features share the same missingness
    # --------------------------------------------------------

    missing_patterns = (
        df[WEIGHTED_FEATURES]
        .isna()
        .drop_duplicates()
    )

    print()
    print("=" * 70)
    print("MISSINGNESS PATTERNS")
    print("=" * 70)

    print(
        missing_patterns.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Warm-up check
    # --------------------------------------------------------

    first_complete_mask = (
        df[WEIGHTED_FEATURES]
        .notna()
        .all(axis=1)
    )

    first_complete_position = (
        first_complete_mask.idxmax()
    )

    leading_missing_count = (
        first_complete_position
    )

    print()
    print("=" * 70)
    print("WARM-UP CHECK")
    print("=" * 70)

    print(
        f"Leading incomplete rows: "
        f"{leading_missing_count}"
    )

    print(
        f"Expected warm-up rows: "
        f"{EXPECTED_WARMUP_DAYS}"
    )

    warmup_pass = (
        leading_missing_count
        == EXPECTED_WARMUP_DAYS
    )

    print(
        "Warm-up status: "
        + ("PASS" if warmup_pass else "FAIL")
    )

    # --------------------------------------------------------
    # Check missing rows after warm-up
    # --------------------------------------------------------

    post_warmup = df.iloc[
        EXPECTED_WARMUP_DAYS:
    ].copy()

    post_warmup_missing_mask = (
        post_warmup[WEIGHTED_FEATURES]
        .isna()
        .any(axis=1)
    )

    post_warmup_missing = (
        post_warmup.loc[
            post_warmup_missing_mask
        ].copy()
    )

    print()
    print("=" * 70)
    print("POST-WARM-UP MISSINGNESS")
    print("=" * 70)

    print(
        f"Missing rows after warm-up: "
        f"{len(post_warmup_missing)}"
    )

    # --------------------------------------------------------
    # Terminal-date exception
    # --------------------------------------------------------

    terminal_date = df["date"].max()

    terminal_row = df[
        df["date"] == terminal_date
    ].iloc[0]

    terminal_weighted_missing = (
        terminal_row[
            WEIGHTED_FEATURES
        ]
        .isna()
        .any()
    )

    terminal_target_missing = pd.isna(
        terminal_row[TARGET_COLUMN]
    )

    terminal_exception = (
        terminal_weighted_missing
        and terminal_target_missing
        and terminal_date
        == df["date"].max()
    )

    print()
    print("=" * 70)
    print("TERMINAL DATE CHECK")
    print("=" * 70)

    print(
        f"Terminal date: "
        f"{terminal_date.date()}"
    )

    print(
        f"Terminal weighted features missing: "
        f"{terminal_weighted_missing}"
    )

    print(
        f"Terminal T+1 target missing: "
        f"{terminal_target_missing}"
    )

    if terminal_exception:

        print(
            "Terminal-date exception: PASS"
        )

        print(
            "The final dataset date has no usable "
            "weighted feature because it cannot "
            "participate in a T+1 target observation."
        )

    else:

        print(
            "Terminal-date exception: NOT APPLICABLE"
        )

    # --------------------------------------------------------
    # Identify unexpected missing dates
    # --------------------------------------------------------

    unexpected_missing_dates = []

    for _, row in post_warmup_missing.iterrows():

        row_date = row["date"]

        if row_date == terminal_date:

            if not terminal_exception:
                unexpected_missing_dates.append(
                    row_date
                )

        else:

            unexpected_missing_dates.append(
                row_date
            )

    print()
    print("=" * 70)
    print("UNEXPECTED POST-WARM-UP MISSINGNESS")
    print("=" * 70)

    if unexpected_missing_dates:

        for date in unexpected_missing_dates:
            print(date.date())

    else:

        print(
            "PASS: No unexpected missing weighted "
            "features after warm-up."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL AUDIT")
    print("=" * 70)

    if (
        warmup_pass
        and
        len(unexpected_missing_dates) == 0
    ):

        print(
            "STATUS: PASS"
        )

        print(
            "All historical weighted-feature "
            "missingness is either expected warm-up "
            "or the terminal non-target date."
        )

    else:

        print(
            "STATUS: FAIL"
        )

        print(
            "Unexpected missing weighted features "
            "remain in the usable research period."
        )


if __name__ == "__main__":
    main()