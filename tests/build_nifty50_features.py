from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MEMBERSHIP_FILE = (
    BASE_DIR
    / "data"
    / "nifty50_constituent_membership.csv"
)

PRICE_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_prices"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "nifty50_features"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "nifty50_t1_features.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "nifty50_t1_features_audit.csv"
)

NIFTY_SYMBOL = "^NSEI"

START_DATE = "2021-01-29"

# yfinance end date is exclusive.
END_DATE = (
    pd.Timestamp.today().normalize()
    + pd.Timedelta(days=1)
).strftime("%Y-%m-%d")


# =========================================================
# SPECIAL HISTORICAL CASES
# =========================================================

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
}


# =========================================================
# LOAD MEMBERSHIP
# =========================================================

def load_membership():

    membership = pd.read_csv(
        MEMBERSHIP_FILE
    )

    required_columns = {
        "date",
        "symbol",
        "is_constituent",
    }

    missing = (
        required_columns
        - set(membership.columns)
    )

    if missing:
        raise ValueError(
            "Membership file missing columns: "
            f"{sorted(missing)}"
        )

    membership["date"] = pd.to_datetime(
        membership["date"],
        errors="raise",
    ).dt.normalize()

    membership["symbol"] = (
        membership["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    membership["is_constituent"] = (
        pd.to_numeric(
            membership["is_constituent"],
            errors="raise",
        )
        .astype(int)
    )

    if membership.duplicated(
        subset=["date", "symbol"]
    ).any():

        raise ValueError(
            "Membership contains duplicate "
            "date/symbol rows."
        )

    return membership


# =========================================================
# LOAD NIFTY INDEX
# =========================================================

def load_nifty():

    print()
    print("=" * 70)
    print("DOWNLOADING NIFTY 50 INDEX DATA")
    print("=" * 70)

    nifty = yf.download(
        NIFTY_SYMBOL,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if nifty.empty:

        raise RuntimeError(
            "Unable to download NIFTY 50 index data."
        )

    if isinstance(
        nifty.columns,
        pd.MultiIndex,
    ):

        nifty.columns = (
            nifty.columns
            .get_level_values(0)
        )

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing = (
        required_columns
        - set(nifty.columns)
    )

    if missing:

        raise RuntimeError(
            "NIFTY data missing columns: "
            f"{sorted(missing)}"
        )

    nifty = nifty.reset_index()

    if "Date" not in nifty.columns:

        raise RuntimeError(
            "NIFTY data does not contain "
            "Date after reset_index()."
        )

    nifty["Date"] = pd.to_datetime(
        nifty["Date"],
        errors="raise",
    ).dt.normalize()

    nifty = nifty.rename(
        columns={
            "Date": "date",
            "Open": "nifty_open",
            "High": "nifty_high",
            "Low": "nifty_low",
            "Close": "nifty_close",
            "Volume": "nifty_volume",
        }
    )

    nifty = nifty[
        [
            "date",
            "nifty_open",
            "nifty_high",
            "nifty_low",
            "nifty_close",
            "nifty_volume",
        ]
    ].copy()

    nifty = nifty.dropna(
        subset=[
            "nifty_close"
        ]
    )

    nifty = (
        nifty
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )

    if nifty.empty:

        raise RuntimeError(
            "NIFTY dataset is empty."
        )

    print(
        f"NIFTY dates: {len(nifty)}"
    )

    print(
        f"Coverage: "
        f"{nifty['date'].min().date()} "
        f"-> "
        f"{nifty['date'].max().date()}"
    )

    return nifty


# =========================================================
# LOAD STOCK PRICE
# =========================================================

def load_price(symbol):

    price_file = (
        PRICE_DIR
        / f"{symbol}.csv"
    )

    if not price_file.exists():

        raise FileNotFoundError(
            f"Missing price file: "
            f"{price_file}"
        )

    data = pd.read_csv(
        price_file
    )

    required_columns = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:

        raise ValueError(
            f"{symbol}: missing price "
            f"columns: {sorted(missing)}"
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    ).dt.normalize()

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    if data["date"].duplicated().any():

        raise ValueError(
            f"{symbol}: duplicate dates found."
        )

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    return data


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    close,
    period=14,
):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    average_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    rs = (
        average_gain
        / average_loss
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return rsi


# =========================================================
# STOCK FEATURES
# =========================================================

def add_stock_features(
    data,
    nifty_returns,
):

    data = data.copy()

    close = data["close"]

    # -----------------------------------------------------
    # Returns
    # -----------------------------------------------------

    data["return_1d"] = (
        close.pct_change(1)
    )

    data["return_3d"] = (
        close.pct_change(3)
    )

    data["return_5d"] = (
        close.pct_change(5)
    )

    data["return_10d"] = (
        close.pct_change(10)
    )

    data["return_20d"] = (
        close.pct_change(20)
    )

    # -----------------------------------------------------
    # EMAs
    # -----------------------------------------------------

    data["ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    data["ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    data["ema200"] = (
        close
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    data["close_ema20_ratio"] = (
        close
        / data["ema20"]
    )

    data["close_ema50_ratio"] = (
        close
        / data["ema50"]
    )

    data["close_ema200_ratio"] = (
        close
        / data["ema200"]
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    data["rsi14"] = calculate_rsi(
        close,
        period=14,
    )

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    data["volume_20d_avg"] = (
        data["volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    data["volume_ratio_20d"] = (
        data["volume"]
        / data["volume_20d_avg"]
    )

    # -----------------------------------------------------
    # 20D Structure
    # -----------------------------------------------------

    rolling_high_20 = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    rolling_low_20 = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    data["distance_20d_high"] = (
        close
        / rolling_high_20
        - 1
    )

    data["distance_20d_low"] = (
        close
        / rolling_low_20
        - 1
    )

    # -----------------------------------------------------
    # Gap
    # -----------------------------------------------------

    previous_close = (
        close.shift(1)
    )

    data["gap_pct"] = (
        data["open"]
        / previous_close
        - 1
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    true_range_1 = (
        data["high"]
        - data["low"]
    )

    true_range_2 = (
        data["high"]
        - previous_close
    ).abs()

    true_range_3 = (
        data["low"]
        - previous_close
    ).abs()

    data["true_range"] = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3,
        ],
        axis=1,
    ).max(axis=1)

    data["atr14"] = (
        data["true_range"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    data["atr_pct"] = (
        data["atr14"]
        / close
    )

    # -----------------------------------------------------
    # Rolling volatility
    # -----------------------------------------------------

    data["rolling_volatility_20d"] = (
        data["return_1d"]
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    # -----------------------------------------------------
    # Relative strength vs NIFTY
    # -----------------------------------------------------

    data["nifty_return_20d"] = (
        nifty_returns
        .reindex(
            data["date"]
        )
        .to_numpy()
    )

    data["relative_strength_20d"] = (
        data["return_20d"]
        - data["nifty_return_20d"]
    )

    return data


# =========================================================
# NIFTY FEATURES
# =========================================================

def add_nifty_features(
    nifty
):

    nifty = nifty.copy()

    close = nifty["nifty_close"]

    # -----------------------------------------------------
    # Returns
    # -----------------------------------------------------

    nifty["nifty_return_1d"] = (
        close.pct_change(1)
    )

    nifty["nifty_return_3d"] = (
        close.pct_change(3)
    )

    nifty["nifty_return_5d"] = (
        close.pct_change(5)
    )

    nifty["nifty_return_10d"] = (
        close.pct_change(10)
    )

    nifty["nifty_return_20d"] = (
        close.pct_change(20)
    )

    # -----------------------------------------------------
    # EMAs
    # -----------------------------------------------------

    nifty["nifty_ema20"] = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    nifty["nifty_ema50"] = (
        close
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    nifty["nifty_ema200"] = (
        close
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )

    nifty["nifty_close_ema20_ratio"] = (
        close
        / nifty["nifty_ema20"]
    )

    nifty["nifty_close_ema50_ratio"] = (
        close
        / nifty["nifty_ema50"]
    )

    nifty["nifty_close_ema200_ratio"] = (
        close
        / nifty["nifty_ema200"]
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    nifty["nifty_rsi14"] = (
        calculate_rsi(
            close,
            period=14,
        )
    )

    # -----------------------------------------------------
    # 20D structure
    # -----------------------------------------------------

    rolling_high_20 = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    rolling_low_20 = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    nifty["nifty_distance_20d_high"] = (
        close
        / rolling_high_20
        - 1
    )

    nifty["nifty_distance_20d_low"] = (
        close
        / rolling_low_20
        - 1
    )

    # -----------------------------------------------------
    # Gap
    # -----------------------------------------------------

    previous_close = (
        close.shift(1)
    )

    nifty["nifty_gap_pct"] = (
        nifty["nifty_open"]
        / previous_close
        - 1
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    true_range_1 = (
        nifty["nifty_high"]
        - nifty["nifty_low"]
    )

    true_range_2 = (
        nifty["nifty_high"]
        - previous_close
    ).abs()

    true_range_3 = (
        nifty["nifty_low"]
        - previous_close
    ).abs()

    nifty["nifty_true_range"] = pd.concat(
        [
            true_range_1,
            true_range_2,
            true_range_3,
        ],
        axis=1,
    ).max(axis=1)

    nifty["nifty_atr14"] = (
        nifty["nifty_true_range"]
        .rolling(
            14,
            min_periods=14,
        )
        .mean()
    )

    nifty["nifty_atr_pct"] = (
        nifty["nifty_atr14"]
        / close
    )

    # -----------------------------------------------------
    # Volatility
    # -----------------------------------------------------

    nifty["nifty_rolling_volatility_20d"] = (
        nifty["nifty_return_1d"]
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    nifty["nifty_volume_20d_avg"] = (
        nifty["nifty_volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    nifty["nifty_volume_ratio_20d"] = (
        nifty["nifty_volume"]
        / nifty["nifty_volume_20d_avg"]
    )

    return nifty


# =========================================================
# APPLY POINT-IN-TIME MEMBERSHIP
# =========================================================

def apply_membership(
    stock_features,
    membership,
):

    active_membership = membership[
        membership["is_constituent"] == 1
    ][
        [
            "date",
            "symbol",
        ]
    ].copy()

    # DUMMYTATAM is an index adjustment entity,
    # not an ordinary tradable stock.
    active_membership = (
        active_membership[
            ~active_membership["symbol"].isin(
                SPECIAL_INDEX_ENTITIES
            )
        ]
    )

    result = stock_features.merge(
        active_membership,
        on=[
            "date",
            "symbol",
        ],
        how="inner",
        validate="many_to_one",
    )

    # JIOFIN was temporarily included in the index
    # before normal price observations began.
    for symbol, periods in (
        SPECIAL_NO_PRICE_PERIODS.items()
    ):

        for start_date, end_date in periods:

            mask = (
                (result["symbol"] == symbol)
                & (
                    result["date"]
                    >= start_date
                )
                & (
                    result["date"]
                    <= end_date
                )
            )

            result = result.loc[
                ~mask
            ].copy()

    return result


# =========================================================
# CONSTITUENT AGGREGATION
# =========================================================

def aggregate_constituents(
    stock_features
):

    data = stock_features.copy()

    # -----------------------------------------------------
    # Directional breadth
    # -----------------------------------------------------

    data["positive_1d"] = (
        data["return_1d"] > 0
    ).astype(float)

    data["positive_5d"] = (
        data["return_5d"] > 0
    ).astype(float)

    data["positive_20d"] = (
        data["return_20d"] > 0
    ).astype(float)

    # -----------------------------------------------------
    # Trend breadth
    # -----------------------------------------------------

    data["above_ema20"] = (
        data["close_ema20_ratio"] > 1
    ).astype(float)

    data["above_ema50"] = (
        data["close_ema50_ratio"] > 1
    ).astype(float)

    data["above_ema200"] = (
        data["close_ema200_ratio"] > 1
    ).astype(float)

    # -----------------------------------------------------
    # Relative strength breadth
    # -----------------------------------------------------

    data["positive_relative_strength"] = (
        data["relative_strength_20d"] > 0
    ).astype(float)

    # -----------------------------------------------------
    # Volume pressure
    # -----------------------------------------------------

    data["high_volume_up"] = (
        (data["return_1d"] > 0)
        & (
            data["volume_ratio_20d"] > 1
        )
    ).astype(float)

    data["high_volume_down"] = (
        (data["return_1d"] < 0)
        & (
            data["volume_ratio_20d"] > 1
        )
    ).astype(float)

    # -----------------------------------------------------
    # Aggregate by trading date
    # -----------------------------------------------------

    grouped = data.groupby(
        "date",
        sort=True,
    )

    aggregate = grouped.agg(

        constituent_count=(
            "symbol",
            "count",
        ),

        breadth_1d=(
            "positive_1d",
            "mean",
        ),

        breadth_5d=(
            "positive_5d",
            "mean",
        ),

        breadth_20d=(
            "positive_20d",
            "mean",
        ),

        above_ema20_breadth=(
            "above_ema20",
            "mean",
        ),

        above_ema50_breadth=(
            "above_ema50",
            "mean",
        ),

        above_ema200_breadth=(
            "above_ema200",
            "mean",
        ),

        relative_strength_breadth=(
            "positive_relative_strength",
            "mean",
        ),

        high_volume_up_breadth=(
            "high_volume_up",
            "mean",
        ),

        high_volume_down_breadth=(
            "high_volume_down",
            "mean",
        ),

        median_return_1d=(
            "return_1d",
            "median",
        ),

        median_return_5d=(
            "return_5d",
            "median",
        ),

        median_return_20d=(
            "return_20d",
            "median",
        ),

        median_rsi14=(
            "rsi14",
            "median",
        ),

        median_volume_ratio_20d=(
            "volume_ratio_20d",
            "median",
        ),

        median_atr_pct=(
            "atr_pct",
            "median",
        ),

        median_relative_strength_20d=(
            "relative_strength_20d",
            "median",
        ),

    ).reset_index()

    return aggregate


# =========================================================
# ADD T+1 TARGET
# =========================================================

def add_target(
    panel
):

    panel = panel.copy()

    panel = (
        panel
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # T+1 close
    # -----------------------------------------------------

    panel["nifty_t1_close"] = (
        panel["nifty_close"]
        .shift(-1)
    )

    # -----------------------------------------------------
    # T+1 close-to-close return
    # -----------------------------------------------------

    panel["nifty_t1_return"] = (
        panel["nifty_t1_close"]
        / panel["nifty_close"]
        - 1
    )

    # -----------------------------------------------------
    # Direction
    # -----------------------------------------------------

    panel["target_t1_direction"] = (
        panel["nifty_t1_return"] > 0
    ).astype("Int64")

    # Last row has no future observation.
    panel.loc[
        panel["nifty_t1_close"].isna(),
        "target_t1_direction",
    ] = pd.NA

    return panel


# =========================================================
# AUDIT
# =========================================================

def run_audit(
    panel,
    stock_features,
    membership,
):

    feature_columns = [
        "nifty_close",
        "nifty_return_1d",
        "nifty_return_3d",
        "nifty_return_5d",
        "nifty_return_10d",
        "nifty_return_20d",
        "nifty_close_ema20_ratio",
        "nifty_close_ema50_ratio",
        "nifty_close_ema200_ratio",
        "nifty_rsi14",
        "nifty_distance_20d_high",
        "nifty_distance_20d_low",
        "nifty_gap_pct",
        "nifty_atr_pct",
        "nifty_rolling_volatility_20d",
        "nifty_volume_ratio_20d",
        "constituent_count",
        "breadth_1d",
        "breadth_5d",
        "breadth_20d",
        "above_ema20_breadth",
        "above_ema50_breadth",
        "above_ema200_breadth",
        "relative_strength_breadth",
        "high_volume_up_breadth",
        "high_volume_down_breadth",
        "median_return_1d",
        "median_return_5d",
        "median_return_20d",
        "median_rsi14",
        "median_volume_ratio_20d",
        "median_atr_pct",
        "median_relative_strength_20d",
    ]

    failures = []

    # -----------------------------------------------------
    # Duplicate panel dates
    # -----------------------------------------------------

    duplicate_panel_dates = int(
        panel["date"]
        .duplicated()
        .sum()
    )

    if duplicate_panel_dates != 0:

        failures.append(
            "DUPLICATE_PANEL_DATES"
        )

    # -----------------------------------------------------
    # Duplicate stock observations
    # -----------------------------------------------------

    duplicate_stock_rows = int(
        stock_features[
            [
                "date",
                "symbol",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_stock_rows != 0:

        failures.append(
            "DUPLICATE_STOCK_DATE_SYMBOL"
        )

    # -----------------------------------------------------
    # Feature columns must not contain target columns
    # -----------------------------------------------------

    forbidden_columns = {
        "nifty_t1_close",
        "nifty_t1_return",
        "target_t1_direction",
    }

    accidental_target_columns = (
        set(feature_columns)
        & forbidden_columns
    )

    if accidental_target_columns:

        failures.append(
            "TARGET_COLUMNS_IN_FEATURE_SET"
        )

    # -----------------------------------------------------
    # Target validity
    # -----------------------------------------------------

    target_values = (
        panel["target_t1_direction"]
        .dropna()
        .astype(int)
        .unique()
    )

    invalid_target_values = sorted(
        set(target_values)
        - {0, 1}
    )

    if invalid_target_values:

        failures.append(
            "INVALID_TARGET_VALUES"
        )

    # -----------------------------------------------------
    # Target must be missing only at final row
    # -----------------------------------------------------

    target_missing_rows = int(
        panel[
            "target_t1_direction"
        ]
        .isna()
        .sum()
    )

    if target_missing_rows > 1:

        failures.append(
            "MULTIPLE_MISSING_TARGET_ROWS"
        )

    if target_missing_rows == 1:

        last_date = (
            panel["date"].iloc[-1]
        )

        missing_dates = (
            panel.loc[
                panel[
                    "target_t1_direction"
                ].isna(),
                "date",
            ]
        )

        if (
            len(missing_dates) != 1
            or missing_dates.iloc[0]
            != last_date
        ):

            failures.append(
                "TARGET_MISSING_NOT_ONLY_AT_FINAL_DATE"
            )

    # -----------------------------------------------------
    # Feature NaN count
    #
    # Warm-up NaNs are expected because indicators
    # require historical observations.
    # -----------------------------------------------------

    feature_missing_cells = int(
        panel[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    # -----------------------------------------------------
    # Constituent count
    # -----------------------------------------------------

    constituent_count_min = int(
        panel[
            "constituent_count"
        ]
        .min()
    )

    constituent_count_max = int(
        panel[
            "constituent_count"
        ]
        .max()
    )

    if constituent_count_min <= 0:

        failures.append(
            "INVALID_CONSTITUENT_COUNT"
        )

    # -----------------------------------------------------
    # Feature date coverage
    # -----------------------------------------------------

    panel_start = (
        panel["date"].min()
    )

    panel_end = (
        panel["date"].max()
    )

    # -----------------------------------------------------
    # Audit output
    # -----------------------------------------------------

    audit = pd.DataFrame(
        [
            {
                "panel_rows": len(panel),

                "panel_start": panel_start,

                "panel_end": panel_end,

                "stock_feature_rows": len(
                    stock_features
                ),

                "historical_constituents": (
                    stock_features[
                        "symbol"
                    ].nunique()
                ),

                "duplicate_panel_dates": (
                    duplicate_panel_dates
                ),

                "duplicate_stock_date_symbol": (
                    duplicate_stock_rows
                ),

                "target_missing_rows": (
                    target_missing_rows
                ),

                "feature_missing_cells": (
                    feature_missing_cells
                ),

                "constituent_count_min": (
                    constituent_count_min
                ),

                "constituent_count_max": (
                    constituent_count_max
                ),

                "invalid_target_values": (
                    ",".join(
                        map(
                            str,
                            invalid_target_values,
                        )
                    )
                ),

                "accidental_target_columns": (
                    ",".join(
                        sorted(
                            accidental_target_columns
                        )
                    )
                ),

                "status": (
                    "PASS"
                    if not failures
                    else "FAIL"
                ),

                "failure_reasons": (
                    ";".join(failures)
                ),
            }
        ]
    )

    return audit


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "NIFTY 50 POINT-IN-TIME FEATURE BUILDER"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load membership
    # -----------------------------------------------------

    membership = load_membership()

    print(
        f"Membership rows: "
        f"{len(membership)}"
    )

    # -----------------------------------------------------
    # Load NIFTY
    # -----------------------------------------------------

    nifty = load_nifty()

    nifty = add_nifty_features(
        nifty
    )

    # NIFTY 20D return indexed by date.
    nifty_returns = (
        nifty
        .set_index("date")[
            "nifty_return_20d"
        ]
        .sort_index()
    )

    # -----------------------------------------------------
    # Determine historical constituents
    # -----------------------------------------------------

    symbols = sorted(
        membership.loc[
            membership[
                "is_constituent"
            ] == 1,
            "symbol",
        ]
        .unique()
    )

    # DUMMYTATAM is not a normal price series.
    symbols = [
        symbol
        for symbol in symbols
        if symbol
        not in SPECIAL_INDEX_ENTITIES
    ]

    print()
    print(
        f"Ordinary historical constituents: "
        f"{len(symbols)}"
    )

    # -----------------------------------------------------
    # Build stock-level features
    # -----------------------------------------------------

    all_stock_features = []

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):

        print(
            f"[{index}/{len(symbols)}] "
            f"Processing {symbol}"
        )

        prices = load_price(
            symbol
        )

        prices = prices[
            (
                prices["date"]
                >= pd.Timestamp(
                    START_DATE
                )
            )
            & (
                prices["date"]
                <= nifty["date"].max()
            )
        ].copy()

        if prices.empty:

            raise RuntimeError(
                f"{symbol}: no usable "
                f"price data."
            )

        stock = add_stock_features(
            prices,
            nifty_returns,
        )

        stock["symbol"] = symbol

        all_stock_features.append(
            stock
        )

    # -----------------------------------------------------
    # Combine all stocks
    # -----------------------------------------------------

    stock_features = pd.concat(
        all_stock_features,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Apply point-in-time membership
    # -----------------------------------------------------

    stock_features = apply_membership(
        stock_features,
        membership,
    )

    stock_features = (
        stock_features
        .sort_values(
            [
                "date",
                "symbol",
            ]
        )
        .reset_index(drop=True)
    )

    if stock_features.empty:

        raise RuntimeError(
            "No point-in-time constituent "
            "observations were produced."
        )

    print()
    print("=" * 70)
    print(
        "AGGREGATING CONSTITUENT INTELLIGENCE"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Aggregate constituents
    # -----------------------------------------------------

    constituent_aggregate = (
        aggregate_constituents(
            stock_features
        )
    )

    # -----------------------------------------------------
    # Merge with NIFTY
    # -----------------------------------------------------

    panel = nifty.merge(
        constituent_aggregate,
        on="date",
        how="left",
        validate="one_to_one",
    )

    panel = (
        panel
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Add T+1 target
    # -----------------------------------------------------

    panel = add_target(
        panel
    )

    # =====================================================
    # FINAL OUTPUT COLUMNS
    # =====================================================

    output_columns = [

        "date",

        # -------------------------------------------------
        # NIFTY features
        # -------------------------------------------------

        "nifty_close",

        "nifty_return_1d",

        "nifty_return_3d",

        "nifty_return_5d",

        "nifty_return_10d",

        "nifty_return_20d",

        "nifty_close_ema20_ratio",

        "nifty_close_ema50_ratio",

        "nifty_close_ema200_ratio",

        "nifty_rsi14",

        "nifty_distance_20d_high",

        "nifty_distance_20d_low",

        "nifty_gap_pct",

        "nifty_atr_pct",

        "nifty_rolling_volatility_20d",

        "nifty_volume_ratio_20d",

        # -------------------------------------------------
        # Constituent intelligence
        # -------------------------------------------------

        "constituent_count",

        "breadth_1d",

        "breadth_5d",

        "breadth_20d",

        "above_ema20_breadth",

        "above_ema50_breadth",

        "above_ema200_breadth",

        "relative_strength_breadth",

        "high_volume_up_breadth",

        "high_volume_down_breadth",

        "median_return_1d",

        "median_return_5d",

        "median_return_20d",

        "median_rsi14",

        "median_volume_ratio_20d",

        "median_atr_pct",

        "median_relative_strength_20d",

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        "nifty_t1_close",

        "nifty_t1_return",

        "target_t1_direction",
    ]

    missing_output_columns = (
        set(output_columns)
        - set(panel.columns)
    )

    if missing_output_columns:

        raise RuntimeError(
            "Panel missing expected output "
            "columns: "
            f"{sorted(missing_output_columns)}"
        )

    panel = panel[
        output_columns
    ].copy()

    # =====================================================
    # AUDIT
    # =====================================================

    audit = run_audit(
        panel,
        stock_features,
        membership,
    )

    # =====================================================
    # SAVE
    # =====================================================

    panel.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("=" * 70)
    print(
        "FEATURE DATASET SUMMARY"
    )
    print("=" * 70)

    print(
        f"Rows: {len(panel)}"
    )

    print(
        f"Coverage: "
        f"{panel['date'].min().date()} "
        f"-> "
        f"{panel['date'].max().date()}"
    )

    target_rows = int(
        panel[
            "target_t1_direction"
        ]
        .notna()
        .sum()
    )

    positive_targets = int(
        (
            panel[
                "target_t1_direction"
            ] == 1
        )
        .sum()
    )

    negative_targets = int(
        (
            panel[
                "target_t1_direction"
            ] == 0
        )
        .sum()
    )

    print(
        f"Target rows: "
        f"{target_rows}"
    )

    print(
        f"Target positives: "
        f"{positive_targets}"
    )

    print(
        f"Target negatives: "
        f"{negative_targets}"
    )

    if target_rows > 0:

        positive_rate = (
            positive_targets
            / target_rows
        )

        print(
            f"Unconditional positive rate: "
            f"{positive_rate:.4f}"
        )

    print()
    print(
        f"Feature file: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Audit file: "
        f"{AUDIT_FILE}"
    )

    print()
    print(
        "AUDIT STATUS:",
        audit.loc[
            0,
            "status",
        ],
    )

    if (
        audit.loc[
            0,
            "status",
        ]
        != "PASS"
    ):

        raise RuntimeError(
            "Feature builder audit failed: "
            + str(
                audit.loc[
                    0,
                    "failure_reasons",
                ]
            )
        )

    print()
    print(
        "STATUS: PASS"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()