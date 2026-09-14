from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MEMBERSHIP_FILE = Path(
    "data/nifty50_constituent_membership.csv"
)

PRICE_DIR = Path(
    "data/nifty50_prices"
)

BASE_FEATURE_FILE = Path(
    "data/nifty50_features/nifty50_t1_features.csv"
)

OUTPUT_DIR = Path(
    "data/nifty50_model2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FEATURE_FILE = (
    OUTPUT_DIR /
    "nifty50_model2_features.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK_RETURN = 20
LOOKBACK_VOLATILITY = 20

MIN_VOLATILITY = 0.005

SPECIAL_INDEX_ENTITIES = {
    "DUMMYTATAM"
}

SPECIAL_NO_PRICE_PERIODS = {
    "JIOFIN": [
        (
            pd.Timestamp("2023-07-20"),
            pd.Timestamp("2023-08-18")
        )
    ]
}


# ============================================================
# HELPERS
# ============================================================

def load_membership():

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    membership["date"] = pd.to_datetime(
        membership["date"]
    )

    return membership


def load_price(symbol):

    path = PRICE_DIR / f"{symbol}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        "date"
    ).drop_duplicates(
        "date",
        keep="last"
    )

    return df


def calculate_stock_features(price):

    price = price.copy()

    price["return_1d"] = (
        price["close"].pct_change(1)
    )

    price["return_20d"] = (
        price["close"].pct_change(
            LOOKBACK_RETURN
        )
    )

    price["volatility_20d"] = (
        price["return_1d"]
        .rolling(
            LOOKBACK_VOLATILITY,
            min_periods=LOOKBACK_VOLATILITY
        )
        .std()
    )

    return price[
        [
            "date",
            "return_1d",
            "return_20d",
            "volatility_20d"
        ]
    ]


def calculate_weighted_pressure(
    daily_stock_data
):

    rows = []

    for date, group in daily_stock_data.groupby(
        "date"
    ):

        group = group.copy()

        # ----------------------------------------------------
        # 1-DAY PRESSURE
        # ----------------------------------------------------

        group_1d = group.dropna(
            subset=[
                "return_1d",
                "volatility_20d"
            ]
        ).copy()

        if len(group_1d) > 0:

            volatility_1d = (
                group_1d["volatility_20d"]
                .clip(
                    lower=MIN_VOLATILITY
                )
            )

            reliability_1d = (
                1.0 / volatility_1d
            )

            raw_weight_1d = (
                reliability_1d
                / reliability_1d.sum()
            )

            weighted_return_1d = (
                group_1d["return_1d"]
                * raw_weight_1d
            ).sum()

            weighted_positive_pressure_1d = (
                raw_weight_1d[
                    group_1d["return_1d"] > 0
                ].sum()
            )

            weighted_negative_pressure_1d = (
                raw_weight_1d[
                    group_1d["return_1d"] < 0
                ].sum()
            )

        else:

            weighted_return_1d = np.nan
            weighted_positive_pressure_1d = np.nan
            weighted_negative_pressure_1d = np.nan

        # ----------------------------------------------------
        # 20-DAY PRESSURE
        # ----------------------------------------------------

        group_20d = group.dropna(
            subset=[
                "return_20d",
                "volatility_20d"
            ]
        ).copy()

        if len(group_20d) > 0:

            volatility_20d = (
                group_20d["volatility_20d"]
                .clip(
                    lower=MIN_VOLATILITY
                )
            )

            reliability_20d = (
                1.0 / volatility_20d
            )

            raw_weight_20d = (
                reliability_20d
                / reliability_20d.sum()
            )

            weighted_return_20d = (
                group_20d["return_20d"]
                * raw_weight_20d
            ).sum()

            weighted_positive_pressure_20d = (
                raw_weight_20d[
                    group_20d["return_20d"] > 0
                ].sum()
            )

            weighted_negative_pressure_20d = (
                raw_weight_20d[
                    group_20d["return_20d"] < 0
                ].sum()
            )

        else:

            weighted_return_20d = np.nan
            weighted_positive_pressure_20d = np.nan
            weighted_negative_pressure_20d = np.nan

        # ----------------------------------------------------
        # Require at least one valid 1-day or 20-day
        # observation before creating a row.
        # ----------------------------------------------------

        if (
            pd.isna(weighted_return_1d)
            and
            pd.isna(weighted_return_20d)
        ):
            continue

        rows.append(
            {
                "date": date,

                "weighted_return_1d":
                    weighted_return_1d,

                "weighted_return_20d":
                    weighted_return_20d,

                "weighted_positive_pressure_1d":
                    weighted_positive_pressure_1d,

                "weighted_positive_pressure_20d":
                    weighted_positive_pressure_20d,

                "weighted_negative_pressure_1d":
                    weighted_negative_pressure_1d,

                "weighted_negative_pressure_20d":
                    weighted_negative_pressure_20d,

                "weighted_constituent_count":
                    len(group_1d),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL 2 FEATURE BUILDER")
    print("DYNAMIC CONSTITUENT WEIGHTED PRESSURE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load membership
    # --------------------------------------------------------

    membership = load_membership()

    print(
        f"Membership rows: "
        f"{len(membership)}"
    )

    # --------------------------------------------------------
    # Load base Model 0/1 feature dataset
    # --------------------------------------------------------

    base = pd.read_csv(
        BASE_FEATURE_FILE
    )

    base["date"] = pd.to_datetime(
        base["date"]
    )

    base = base.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"Base feature rows: "
        f"{len(base)}"
    )

    # --------------------------------------------------------
    # Prepare membership lookup
    # --------------------------------------------------------

    membership_lookup = (
        membership[
            [
                "date",
                "symbol"
            ]
        ]
        .drop_duplicates()
    )

    ordinary_symbols = sorted(
        set(
            membership_lookup["symbol"]
        )
        - SPECIAL_INDEX_ENTITIES
    )

    print(
        f"Ordinary historical constituents: "
        f"{len(ordinary_symbols)}"
    )

    # --------------------------------------------------------
    # Build constituent daily dataset
    # --------------------------------------------------------

    all_stock_data = []

    processed = 0

    for symbol in ordinary_symbols:

        price = load_price(symbol)

        if price is None:
            print(
                f"WARNING: missing price file: "
                f"{symbol}"
            )
            continue

        stock_features = calculate_stock_features(
            price
        )

        stock_features["symbol"] = symbol

        all_stock_data.append(
            stock_features
        )

        processed += 1

    if not all_stock_data:
        raise RuntimeError(
            "No constituent price data available."
        )

    stock_data = pd.concat(
        all_stock_data,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Apply point-in-time membership
    # --------------------------------------------------------

    stock_data = stock_data.merge(
        membership_lookup,
        on=[
            "date",
            "symbol"
        ],
        how="inner"
    )

    # --------------------------------------------------------
    # Remove special no-price periods
    # --------------------------------------------------------

    for symbol, periods in (
        SPECIAL_NO_PRICE_PERIODS.items()
    ):

        for start_date, end_date in periods:

            mask = (
                (stock_data["symbol"] == symbol)
                &
                (stock_data["date"] >= start_date)
                &
                (stock_data["date"] <= end_date)
            )

            stock_data = stock_data.loc[
                ~mask
            ]

    # --------------------------------------------------------
    # Calculate weighted pressure
    # --------------------------------------------------------

    weighted = calculate_weighted_pressure(
        stock_data
    )

    if weighted.empty:
        raise RuntimeError(
            "Weighted pressure calculation "
            "returned no rows."
        )

    # --------------------------------------------------------
    # Merge with existing NIFTY feature set
    # --------------------------------------------------------

    output = base.merge(
        weighted,
        on="date",
        how="left"
    )

    output = output.sort_values(
        "date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    required_weighted_columns = [
        "weighted_return_1d",
        "weighted_return_20d",
        "weighted_positive_pressure_1d",
        "weighted_positive_pressure_20d",
        "weighted_negative_pressure_1d",
        "weighted_negative_pressure_20d",
        "weighted_constituent_count",
    ]

    print()
    print("=" * 70)
    print("MODEL 2 FEATURE VALIDATION")
    print("=" * 70)

    for column in required_weighted_columns:

        missing = output[column].isna().sum()

        print(
            f"{column}: "
            f"missing={missing}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output.to_csv(
        OUTPUT_FEATURE_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("MODEL 2 FEATURE FILE CREATED")
    print("=" * 70)

    print(
        OUTPUT_FEATURE_FILE
    )

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Coverage: "
        f"{output['date'].min().date()} -> "
        f"{output['date'].max().date()}"
    )

    print(
        f"Processed constituents: "
        f"{processed}"
    )

    print()
    print("MODEL 2 FEATURE BUILD COMPLETE")


if __name__ == "__main__":
    main()