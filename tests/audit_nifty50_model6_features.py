from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_model6"
    / "nifty50_model6_features.csv"
)


SECTOR_FEATURES = [
    "sector_return_1d_mean",
    "sector_return_5d_mean",
    "sector_positive_breadth_mean",
    "sector_negative_breadth_mean",
    "sector_dispersion_mean",
    "sector_leadership",
    "sector_agreement",
]


EXPECTED_WARMUP_ROWS = 5


def main():

    print("=" * 70)
    print("NIFTY 50 MODEL 6 FEATURE AUDIT")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            FEATURE_FILE
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Coverage: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # MISSING COUNTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MISSING COUNTS")
    print("=" * 70)

    missing_counts = df[
        SECTOR_FEATURES
    ].isna().sum()

    for feature in SECTOR_FEATURES:

        print(
            f"{feature}: "
            f"{int(missing_counts[feature])}"
        )

    # --------------------------------------------------------
    # DATES WITH ANY MISSING SECTOR FEATURE
    # --------------------------------------------------------

    any_missing = df[
        SECTOR_FEATURES
    ].isna().any(
        axis=1
    )

    missing_dates = df.loc[
        any_missing,
        [
            "date"
        ]
        + SECTOR_FEATURES
    ].copy()

    print()
    print("=" * 70)
    print("DATES WITH ANY MISSING SECTOR FEATURE")
    print("=" * 70)

    if missing_dates.empty:

        print(
            "None"
        )

    else:

        print(
            missing_dates.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # LEADING WARM-UP CHECK
    # --------------------------------------------------------

    leading_incomplete = 0

    for _, row in df.iterrows():

        if row[
            SECTOR_FEATURES
        ].isna().any():

            leading_incomplete += 1

        else:

            break

    print()
    print("=" * 70)
    print("WARM-UP CHECK")
    print("=" * 70)

    print(
        f"Leading incomplete rows: "
        f"{leading_incomplete}"
    )

    print(
        f"Expected warm-up rows: "
        f"{EXPECTED_WARMUP_ROWS}"
    )

    if leading_incomplete == EXPECTED_WARMUP_ROWS:

        print(
            "Warm-up status: PASS"
        )

    else:

        print(
            "Warm-up status: FAIL"
        )

    # --------------------------------------------------------
    # POST-WARM-UP MISSINGNESS
    # --------------------------------------------------------

    post_warmup = df.iloc[
        EXPECTED_WARMUP_ROWS:
    ].copy()

    post_warmup_missing = post_warmup[
        SECTOR_FEATURES
    ].isna().any(
        axis=1
    )

    post_warmup_missing_rows = (
        post_warmup.loc[
            post_warmup_missing
        ]
    )

    print()
    print("=" * 70)
    print("POST-WARM-UP MISSINGNESS")
    print("=" * 70)

    print(
        f"Missing rows after warm-up: "
        f"{len(post_warmup_missing_rows)}"
    )

    if not post_warmup_missing_rows.empty:

        print()
        print(
            post_warmup_missing_rows[
                [
                    "date"
                ]
                + SECTOR_FEATURES
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # TERMINAL DATE CHECK
    # --------------------------------------------------------

    terminal_row = df.iloc[-1]

    terminal_date = terminal_row[
        "date"
    ]

    terminal_missing = terminal_row[
        SECTOR_FEATURES
    ].isna().any()

    target_missing = (
        "target_t1_direction" in df.columns
        and pd.isna(
            terminal_row[
                "target_t1_direction"
            ]
        )
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
        f"Terminal sector features missing: "
        f"{terminal_missing}"
    )

    print(
        f"Terminal T+1 target missing: "
        f"{target_missing}"
    )

    if terminal_missing and target_missing:

        print(
            "Terminal-date exception: PASS"
        )

    elif not terminal_missing:

        print(
            "Terminal-date exception: "
            "NOT NEEDED"
        )

    else:

        print(
            "Terminal-date exception: FAIL"
        )

    # --------------------------------------------------------
    # UNEXPECTED POST-WARM-UP MISSINGNESS
    # --------------------------------------------------------

    unexpected = post_warmup_missing_rows.copy()

    if not unexpected.empty:

        unexpected = unexpected[
            unexpected["date"]
            != terminal_date
        ]

    print()
    print("=" * 70)
    print("UNEXPECTED POST-WARM-UP MISSINGNESS")
    print("=" * 70)

    if unexpected.empty:

        print(
            "PASS: No unexpected missing "
            "sector features after warm-up."
        )

    else:

        print(
            "FAIL: Unexpected missing "
            "sector features found:"
        )

        print(
            unexpected[
                [
                    "date"
                ]
                + SECTOR_FEATURES
            ].to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SECTOR FEATURE RANGE CHECKS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RANGE CHECKS")
    print("=" * 70)

    range_failures = []

    breadth_features = [
        "sector_positive_breadth_mean",
        "sector_negative_breadth_mean",
        "sector_leadership",
        "sector_agreement",
    ]

    for feature in breadth_features:

        values = df[
            feature
        ].dropna()

        invalid = values[
            (values < 0)
            | (values > 1)
        ]

        if len(invalid) > 0:

            range_failures.append(
                f"{feature}: values outside [0,1]"
            )

    # --------------------------------------------------------
    # AGREEMENT LOGIC
    # --------------------------------------------------------

    agreement = df[
        "sector_agreement"
    ].dropna()

    if (
        (agreement < 0).any()
        or (agreement > 1).any()
    ):

        range_failures.append(
            "sector_agreement outside [0,1]"
        )

    if range_failures:

        print(
            "FAIL"
        )

        for failure in range_failures:

            print(
                failure
            )

    else:

        print(
            "PASS: Sector proportion features "
            "are within expected [0,1] ranges."
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    failures = []

    if leading_incomplete != EXPECTED_WARMUP_ROWS:

        failures.append(
            "INVALID_WARMUP"
        )

    if not unexpected.empty:

        failures.append(
            "UNEXPECTED_POST_WARMUP_MISSINGNESS"
        )

    if (
        terminal_missing
        and not target_missing
    ):

        failures.append(
            "INVALID_TERMINAL_MISSINGNESS"
        )

    if range_failures:

        failures.append(
            "RANGE_CHECK_FAILURE"
        )

    print()
    print("=" * 70)
    print("FINAL AUDIT")
    print("=" * 70)

    if failures:

        print(
            "STATUS: FAIL"
        )

        print(
            "Failure reasons:",
            ", ".join(
                failures
            )
        )

        raise RuntimeError(
            "Model 6 feature audit failed."
        )

    print(
        "STATUS: PASS"
    )

    print()
    print(
        "IMPORTANT RESEARCH LIMITATION:"
    )

    print(
        "Model 6 uses current official NSE "
        "sector classifications rather than "
        "point-in-time historical classifications."
    )

    print(
        "Therefore Model 6 remains exploratory "
        "even if this feature audit passes."
    )


if __name__ == "__main__":
    main()