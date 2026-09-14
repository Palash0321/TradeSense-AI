from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MEMBERSHIP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_constituent_membership.csv"
)

SECTOR_MAP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_sector_map.csv"
)

PRICE_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
)

BASE_FEATURE_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_features"
    / "nifty50_t1_features.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_model6"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_model6_features.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "nifty50_model6_features_audit.csv"
)


# ============================================================
# SPECIAL ENTITIES
# ============================================================

SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM",
}

SPECIAL_NO_PRICE_PERIODS = {
    "JIOFIN": [
        (
            pd.Timestamp("2023-07-20"),
            pd.Timestamp("2023-08-18"),
        )
    ],
    "DUMMYTATAM": [
        (
            pd.Timestamp("2025-10-14"),
            pd.Timestamp("2025-11-16"),
        )
    ],
}


# ============================================================
# MODEL 6 SECTOR FEATURES
# ============================================================

SECTOR_FEATURES = [
    "sector_return_1d_mean",
    "sector_return_5d_mean",
    "sector_positive_breadth_mean",
    "sector_negative_breadth_mean",
    "sector_dispersion_mean",
    "sector_leadership",
    "sector_agreement",
]


# ============================================================
# HELPERS
# ============================================================

def load_price_file(symbol):

    path = PRICE_DIR / f"{symbol}.csv"

    if not path.exists():
        return None

    data = pd.read_csv(
        path
    )

    if "date" not in data.columns:
        raise ValueError(
            f"Missing date column: {path}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data = (
        data
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    required = [
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{symbol}: missing columns {missing}"
        )

    data["return_1d"] = (
        data["close"]
        .pct_change(1)
    )

    data["return_5d"] = (
        data["close"]
        .pct_change(5)
    )

    return data[
        [
            "date",
            "close",
            "volume",
            "return_1d",
            "return_5d",
        ]
    ]


def remove_special_no_price_periods(
    symbol,
    data,
):

    periods = SPECIAL_NO_PRICE_PERIODS.get(
        symbol,
        [],
    )

    if not periods:
        return data

    result = data.copy()

    for start_date, end_date in periods:

        result = result[
            ~(
                (result["date"] >= start_date)
                & (result["date"] <= end_date)
            )
        ]

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NIFTY 50 MODEL 6 FEATURE BUILDER")
    print("NIFTY HISTORY + SECTOR PRESSURE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD BASE FEATURES
    # --------------------------------------------------------

    base_features = pd.read_csv(
        BASE_FEATURE_FILE
    )

    base_features["date"] = pd.to_datetime(
        base_features["date"],
        errors="raise",
    )

    base_features = (
        base_features
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # LOAD MEMBERSHIP
    # --------------------------------------------------------

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    membership["date"] = pd.to_datetime(
        membership["date"],
        errors="raise",
    )

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
    )

    membership = membership[
        membership["is_constituent"] == 1
    ].copy()

    membership = membership[
        ~membership["symbol"].isin(
            SPECIAL_INDEX_ENTITIES
        )
    ].copy()

    # --------------------------------------------------------
    # LOAD SECTOR MAP
    # --------------------------------------------------------

    sector_map = pd.read_csv(
        SECTOR_MAP_FILE
    )

    sector_map["symbol"] = (
        sector_map["symbol"]
        .astype(str)
        .str.strip()
    )

    sector_map["sector"] = (
        sector_map["sector"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # VALIDATE SECTOR MAP
    # --------------------------------------------------------

    historical_symbols = sorted(
        membership["symbol"]
        .unique()
    )

    sector_symbols = sorted(
        sector_map["symbol"]
        .unique()
    )

    missing_sector_symbols = sorted(
        set(historical_symbols)
        - set(sector_symbols)
    )

    extra_sector_symbols = sorted(
        set(sector_symbols)
        - set(historical_symbols)
    )

    duplicate_sector_symbols = (
        sector_map[
            sector_map["symbol"].duplicated(
                keep=False
            )
        ]["symbol"]
        .unique()
        .tolist()
    )

    if missing_sector_symbols:
        raise ValueError(
            "Missing sector mapping for:\n"
            + "\n".join(
                missing_sector_symbols
            )
        )

    if extra_sector_symbols:
        raise ValueError(
            "Extra sector mappings:\n"
            + "\n".join(
                extra_sector_symbols
            )
        )

    if duplicate_sector_symbols:
        raise ValueError(
            "Duplicate sector mappings:\n"
            + "\n".join(
                duplicate_sector_symbols
            )
        )

    # --------------------------------------------------------
    # JOIN MEMBERSHIP WITH SECTOR
    # --------------------------------------------------------

    membership = membership.merge(
        sector_map[
            [
                "symbol",
                "sector",
            ]
        ],
        on="symbol",
        how="left",
        validate="many_to_one",
    )

    if membership["sector"].isna().any():
        raise ValueError(
            "Some membership rows have no sector."
        )

    # --------------------------------------------------------
    # LOAD PRICE DATA
    # --------------------------------------------------------

    price_cache = {}

    for symbol in historical_symbols:

        data = load_price_file(
            symbol
        )

        if data is None:
            raise FileNotFoundError(
                f"Price file missing: {symbol}"
            )

        data = remove_special_no_price_periods(
            symbol,
            data,
        )

        price_cache[symbol] = data

    print(
        f"Historical ordinary constituents: "
        f"{len(historical_symbols)}"
    )

    print(
        f"Sector mappings: "
        f"{len(sector_symbols)}"
    )

    print(
        f"Base feature rows: "
        f"{len(base_features):,}"
    )

    # --------------------------------------------------------
    # BUILD DAILY SECTOR PRESSURE
    # --------------------------------------------------------

    all_dates = sorted(
        base_features["date"]
        .unique()
    )

    sector_rows = []

    for date in all_dates:

        day_members = membership[
            membership["date"] == date
        ][
            [
                "symbol",
                "sector",
            ]
        ]

        if day_members.empty:
            sector_rows.append(
                {
                    "date": date,
                }
            )

            continue

        stock_rows = []

        for _, member in day_members.iterrows():

            symbol = member["symbol"]
            sector = member["sector"]

            data = price_cache[symbol]

            row = data[
                data["date"] == date
            ]

            if row.empty:
                continue

            row = row.iloc[0]

            stock_rows.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "return_1d": row[
                        "return_1d"
                    ],
                    "return_5d": row[
                        "return_5d"
                    ],
                }
            )

        if not stock_rows:

            sector_rows.append(
                {
                    "date": date,
                }
            )

            continue

        stocks = pd.DataFrame(
            stock_rows
        )

        sector_daily = []

        for sector, group in stocks.groupby(
            "sector"
        ):

            returns_1d = (
                group["return_1d"]
                .dropna()
            )

            returns_5d = (
                group["return_5d"]
                .dropna()
            )

            if returns_1d.empty:
                continue

            positive_breadth = (
                returns_1d > 0
            ).mean()

            negative_breadth = (
                returns_1d < 0
            ).mean()

            return_1d_mean = (
                returns_1d.mean()
            )

            return_5d_mean = (
                returns_5d.mean()
                if not returns_5d.empty
                else np.nan
            )

            dispersion = (
                returns_1d.std(
                    ddof=0
                )
            )

            sector_daily.append(
                {
                    "sector": sector,
                    "return_1d_mean": return_1d_mean,
                    "return_5d_mean": return_5d_mean,
                    "positive_breadth": positive_breadth,
                    "negative_breadth": negative_breadth,
                    "dispersion": dispersion,
                }
            )

        if not sector_daily:

            sector_rows.append(
                {
                    "date": date,
                }
            )

            continue

        sector_df = pd.DataFrame(
            sector_daily
        )

        # ----------------------------------------------------
        # SECTOR-LEVEL AGGREGATES
        # ----------------------------------------------------

        sector_return_1d_mean = (
            sector_df[
                "return_1d_mean"
            ].mean()
        )

        sector_return_5d_mean = (
            sector_df[
                "return_5d_mean"
            ].mean()
        )

        sector_positive_breadth_mean = (
            sector_df[
                "positive_breadth"
            ].mean()
        )

        sector_negative_breadth_mean = (
            sector_df[
                "negative_breadth"
            ].mean()
        )

        sector_dispersion_mean = (
            sector_df[
                "dispersion"
            ].mean()
        )

        # ----------------------------------------------------
        # SECTOR LEADERSHIP
        #
        # Fraction of sectors with positive
        # 1D average return.
        # ----------------------------------------------------

        sector_leadership = (
            sector_df[
                "return_1d_mean"
            ] > 0
        ).mean()

        # ----------------------------------------------------
        # SECTOR AGREEMENT
        #
        # Measures how strongly sectors agree
        # with the broad direction.
        #
        # Range:
        #   0 = no agreement
        #   1 = complete agreement
        # ----------------------------------------------------

        positive_sectors = (
            sector_df[
                "return_1d_mean"
            ] > 0
        ).sum()

        negative_sectors = (
            sector_df[
                "return_1d_mean"
            ] < 0
        ).sum()

        total_directional_sectors = (
            positive_sectors
            + negative_sectors
        )

        if total_directional_sectors > 0:

            sector_agreement = (
                max(
                    positive_sectors,
                    negative_sectors,
                )
                / total_directional_sectors
            )

        else:

            sector_agreement = np.nan

        sector_rows.append(
            {
                "date": date,
                "sector_return_1d_mean":
                    sector_return_1d_mean,
                "sector_return_5d_mean":
                    sector_return_5d_mean,
                "sector_positive_breadth_mean":
                    sector_positive_breadth_mean,
                "sector_negative_breadth_mean":
                    sector_negative_breadth_mean,
                "sector_dispersion_mean":
                    sector_dispersion_mean,
                "sector_leadership":
                    sector_leadership,
                "sector_agreement":
                    sector_agreement,
            }
        )

    sector_features = pd.DataFrame(
        sector_rows
    )

    sector_features["date"] = pd.to_datetime(
        sector_features["date"]
    )

    # --------------------------------------------------------
    # MERGE WITH BASE FEATURES
    # --------------------------------------------------------

    result = base_features.merge(
        sector_features,
        on="date",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    audit_rows = []

    for feature in SECTOR_FEATURES:

        missing_count = int(
            result[feature]
            .isna()
            .sum()
        )

        audit_rows.append(
            {
                "feature": feature,
                "rows": len(result),
                "missing_count": missing_count,
                "missing_rate":
                    missing_count / len(result),
            }
        )

    audit_df = pd.DataFrame(
        audit_rows
    )

    audit_df.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL 6 FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"Rows: {len(result):,}"
    )

    print(
        f"Coverage: "
        f"{result['date'].min().date()} -> "
        f"{result['date'].max().date()}"
    )

    print()
    print(
        "SECTOR FEATURES:"
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("FILES WRITTEN")
    print("=" * 70)

    print(
        OUTPUT_FILE
    )

    print(
        AUDIT_FILE
    )

    print()
    print("=" * 70)
    print("MODEL 6 FEATURE BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        "RESEARCH STATUS:"
    )

    print(
        "EXPLORATORY ONLY — sector classifications "
        "are current official classifications, "
        "not point-in-time historical classifications."
    )


if __name__ == "__main__":
    main()