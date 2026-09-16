"""
TradeSense-AI
Early-Adverse Market Structure / Trend / Entry-Quality Forensic

Purpose
-------
Among the authoritative frozen 5B/.25R early-adverse events, determine
whether PRE-ENTRY information can distinguish eventual recovery from
genuinely poor trades.

This is forensic/research only.

It MUST NOT modify:
- production strategy
- frozen trade ledger
- OOS logic
- entry logic
- exit logic
- risk logic
- frozen hypothesis definition

No thresholds are tuned in this study.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FROZEN_TRADES = (
    ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

RETENTION_TRADES = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_7b025_retention_trade_level.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "early_adverse_market_structure"
)

EVENT_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_event_level.csv"
)

CATEGORY_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_categories.csv"
)

NUMERIC_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_numeric.csv"
)

SETUP_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_setup.csv"
)

SYMBOL_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_symbol_loo.csv"
)

YEAR_OUTPUT = (
    OUTPUT_DIR
    / "early_adverse_market_structure_year_loo.csv"
)

THRESHOLD_R = 0.25

RNG_SEED = 20260915
N_PERMUTATIONS = 20000
N_BOOTSTRAPS = 20000


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(value):
    if pd.isna(value):
        return value

    value = str(value).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def normalize_date(value):
    return pd.to_datetime(
        value,
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def require_columns(df, columns, name):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def mean_or_nan(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return np.nan

    return float(series.mean())


def median_or_nan(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return np.nan

    return float(series.median())


def win_rate(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return np.nan

    return float(
        (series > 0).mean() * 100
    )


def profit_factor(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()

    if losses == 0:
        if gains > 0:
            return np.inf
        return np.nan

    return float(gains / losses)


def describe_group(group):
    r = safe_numeric(group["R_Multiple"])

    return {
        "N": int(len(group)),
        "Wins": int((r > 0).sum()),
        "Win_Rate_Pct": win_rate(r),
        "Avg_R": mean_or_nan(r),
        "Median_R": median_or_nan(r),
        "Total_R": float(r.sum()),
        "Profit_Factor": profit_factor(r),
        "Avg_MAE_R": mean_or_nan(group["max_adverse_r"]),
        "Avg_MFE_R": mean_or_nan(group["max_favorable_r"]),
        "Reached_0_5R_Pct": (
            float(group["reached_0_5r"].mean() * 100)
            if "reached_0_5r" in group
            else np.nan
        ),
        "Reached_1R_Pct": (
            float(group["reached_1r"].mean() * 100)
            if "reached_1r" in group
            else np.nan
        ),
        "Reached_1_5R_Pct": (
            float(group["reached_1_5r"].mean() * 100)
            if "reached_1_5r" in group
            else np.nan
        ),
    }


# ============================================================
# LOAD FROZEN TRADES
# ============================================================

def load_frozen():
    print()
    print("Loading frozen trade ledger...")

    if not FROZEN_TRADES.exists():
        raise FileNotFoundError(
            f"Frozen ledger not found: {FROZEN_TRADES}"
        )

    df = pd.read_csv(FROZEN_TRADES)

    print(
        f"Frozen rows loaded : {len(df)}"
    )

    required = [
        "_symbol",
        "Signal_Date",
        "direction",
        "Setup",
        "r_multiple",
        "early_adverse_r_bar_5",
    ]

    require_columns(
        df,
        required,
        "Frozen trade ledger"
    )

    df["Research_Symbol"] = (
        df["_symbol"].map(normalize_symbol)
    )

    df["Signal_Date"] = pd.to_datetime(
        df["Signal_Date"],
        errors="coerce"
    )

    df["Research_Date"] = (
        df["Signal_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    df["Direction"] = (
        df["direction"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["Setup"] = (
        df["Setup"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["R_Multiple"] = safe_numeric(
        df["r_multiple"]
    )

    df["Bar5_Early_Adverse_R"] = safe_numeric(
        df["early_adverse_r_bar_5"]
    )

    df["Trade_Key"] = (
        df["Research_Symbol"]
        + "|"
        + df["Research_Date"]
        + "|"
        + df["Direction"]
    )

    duplicate_count = (
        df["Trade_Key"]
        .duplicated()
        .sum()
    )

    print(
        f"Frozen duplicate keys : {duplicate_count}"
    )

    if duplicate_count:
        raise RuntimeError(
            "Frozen trade keys are not unique."
        )

    return df


# ============================================================
# LOAD AUTHORITATIVE EARLY-ADVERSE CLASSIFICATION
# ============================================================

def load_authoritative_events():
    print()
    print(
        "Loading completed frozen 5B/.25R classification..."
    )

    if not RETENTION_TRADES.exists():
        raise FileNotFoundError(
            f"Retention trade output not found: "
            f"{RETENTION_TRADES}"
        )

    df = pd.read_csv(
        RETENTION_TRADES
    )

    require_columns(
        df,
        [
            "Trade_Key",
            "Early_Adverse",
        ],
        "MTF retention trade ledger"
    )

    df["Early_Adverse"] = (
        df["Early_Adverse"]
        .astype(bool)
    )

    events = df.loc[
        df["Early_Adverse"]
    ].copy()

    print(
        f"Retention rows loaded : {len(df)}"
    )

    print(
        f"Authoritative events  : {len(events)}"
    )

    if len(events) != 60:
        raise RuntimeError(
            "Authoritative 5B/.25R event count is not 60. "
            f"Observed: {len(events)}"
        )

    return events[
        [
            "Trade_Key",
            "Early_Adverse",
        ]
    ]


# ============================================================
# MERGE
# ============================================================

def build_event_dataset(
    frozen,
    authoritative_events,
):
    print()
    print(
        "Building early-adverse event dataset..."
    )

    df = frozen.merge(
        authoritative_events,
        on="Trade_Key",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matched event trades : {len(df)}"
    )

    if len(df) != 60:
        raise RuntimeError(
            "Expected exactly 60 authoritative "
            "early-adverse trades after merge."
        )

    if not df["Early_Adverse"].all():
        raise RuntimeError(
            "Merged dataset contains non-event trades."
        )

    return df


# ============================================================
# PRE-ENTRY FEATURE ENGINEERING
# ============================================================

def build_pre_entry_features(df):
    df = df.copy()

    # --------------------------------------------------------
    # Directional agreement features
    #
    # These are deterministic relationships between existing
    # pre-entry categorical fields.
    # --------------------------------------------------------

    df["Structure_Direction_Aligned"] = (
        (
            (df["Direction"] == "LONG")
            & (df["Structure_Bias"] == "BULLISH")
        )
        |
        (
            (df["Direction"] == "SHORT")
            & (df["Structure_Bias"] == "BEARISH")
        )
    )

    df["Trend_Direction_Aligned"] = (
        (
            (df["Direction"] == "LONG")
            & (df["entry_trend"] == "BULLISH")
        )
        |
        (
            (df["Direction"] == "SHORT")
            & (df["entry_trend"] == "BEARISH")
        )
    )

    df["Trend_Direction_Conflict"] = (
        (
            (df["Direction"] == "LONG")
            & (df["entry_trend"] == "BEARISH")
        )
        |
        (
            (df["Direction"] == "SHORT")
            & (df["entry_trend"] == "BULLISH")
        )
    )

    df["Structure_Trend_Agree"] = (
        (
            df["Structure_Bias"] == "BULLISH"
        )
        &
        (
            df["entry_trend"] == "BULLISH"
        )
    ) | (
        (
            df["Structure_Bias"] == "BEARISH"
        )
        &
        (
            df["entry_trend"] == "BEARISH"
        )
    )

    # --------------------------------------------------------
    # Entry quality / structural combinations
    # --------------------------------------------------------

    df["Structure_And_Trend_Aligned"] = (
        df["Structure_Direction_Aligned"]
        &
        df["Trend_Direction_Aligned"]
    )

    # --------------------------------------------------------
    # Directional normalized slope
    #
    # Positive means slope agrees with trade direction.
    # This does NOT introduce a threshold.
    # --------------------------------------------------------

    slope = safe_numeric(
        df["entry_trend_slope"]
    )

    df["Directional_Trend_Slope"] = np.where(
        df["Direction"] == "LONG",
        slope,
        -slope,
    )

    # --------------------------------------------------------
    # Keep only information available before entry.
    # --------------------------------------------------------

    return df


# ============================================================
# PRIMARY EVENT OUTCOME
# ============================================================

def print_event_overview(df):
    print()
    print("=" * 72)
    print("EARLY-ADVERSE EVENT OVERVIEW")
    print("=" * 72)

    r = safe_numeric(
        df["R_Multiple"]
    )

    print(
        f"Events                 : {len(df)}"
    )

    print(
        f"Winning events         : {(r > 0).sum()}"
    )

    print(
        f"Winning rate           : {win_rate(r):.2f}%"
    )

    print(
        f"Total R                : {r.sum():+.2f}R"
    )

    print(
        f"Average R              : {r.mean():+.4f}R"
    )

    print(
        f"Median R               : {r.median():+.4f}R"
    )

    print(
        f"Profit factor          : {profit_factor(r):.3f}"
    )

    print()

    for column in [
        "reached_0_5r",
        "reached_1r",
        "reached_1_5r",
    ]:
        if column in df:
            print(
                f"{column} : "
                f"{df[column].mean() * 100:.2f}%"
            )


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

CATEGORICAL_FEATURES = [
    "Direction",
    "Setup",
    "Structure_Bias",
    "entry_trend",
    "Liquidity_Sweep",
    "Setup_Confidence",
    "BOS",
    "Break_Direction",
    "CHOCH",
    "entry_price_position",
    "Fib_Premium_Discount",
    "Fib_Zone",
    "Fib_Extension_State",
    "Structure_Direction_Aligned",
    "Trend_Direction_Aligned",
    "Trend_Direction_Conflict",
    "Structure_Trend_Agree",
    "Structure_And_Trend_Aligned",
]


def categorical_analysis(df):
    print()
    print("=" * 72)
    print("PRE-ENTRY CATEGORICAL ANALYSIS")
    print("=" * 72)

    rows = []

    for feature in CATEGORICAL_FEATURES:

        if feature not in df.columns:
            continue

        working = df[
            [
                feature,
                "R_Multiple",
                "max_adverse_r",
                "max_favorable_r",
                "reached_0_5r",
                "reached_1r",
                "reached_1_5r",
            ]
        ].copy()

        working[feature] = (
            working[feature]
            .fillna("MISSING")
            .astype(str)
        )

        for level, group in working.groupby(
            feature,
            dropna=False,
        ):

            stats = describe_group(
                group
            )

            row = {
                "Feature": feature,
                "Level": level,
            }

            row.update(stats)

            rows.append(row)

            print()
            print(
                f"{feature} = {level}"
            )

            print(
                f"  N             : {stats['N']}"
            )

            print(
                f"  Win rate      : "
                f"{stats['Win_Rate_Pct']:.2f}%"
            )

            print(
                f"  Avg R         : "
                f"{stats['Avg_R']:+.4f}R"
            )

            print(
                f"  Total R       : "
                f"{stats['Total_R']:+.2f}R"
            )

            print(
                f"  Avg MFE       : "
                f"{stats['Avg_MFE_R']:.4f}R"
            )

    return pd.DataFrame(rows)


# ============================================================
# NUMERIC ANALYSIS
# ============================================================

NUMERIC_FEATURES = [
    "entry_trend_strength",
    "entry_trend_slope",
    "Directional_Trend_Slope",
    "Fib_Distance_382",
    "Fib_Distance_500",
    "Fib_Distance_618",
    "Fib_Retracement",
    "Fib_Extension_Percent",
]


def numeric_analysis(df):
    print()
    print("=" * 72)
    print("PRE-ENTRY NUMERIC ANALYSIS")
    print("=" * 72)

    rows = []

    target = safe_numeric(
        df["R_Multiple"]
    )

    for feature in NUMERIC_FEATURES:

        if feature not in df.columns:
            continue

        x = safe_numeric(
            df[feature]
        )

        valid = (
            x.notna()
            & target.notna()
        )

        x_valid = x.loc[valid]
        y_valid = target.loc[valid]

        if len(x_valid) < 5:
            continue

        pearson = x_valid.corr(
            y_valid,
            method="pearson"
        )

        spearman = x_valid.corr(
            y_valid,
            method="spearman"
        )

        rows.append(
            {
                "Feature": feature,
                "N": len(x_valid),
                "Pearson_R": pearson,
                "Spearman_Rho": spearman,
                "Feature_Mean": x_valid.mean(),
                "Feature_Median": x_valid.median(),
                "Outcome_Mean_R": y_valid.mean(),
            }
        )

        print()
        print(feature)

        print(
            f"  N              : {len(x_valid)}"
        )

        print(
            f"  Pearson        : {pearson:+.4f}"
        )

        print(
            f"  Spearman       : {spearman:+.4f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# SETUP-CONDITIONAL ANALYSIS
# ============================================================

def setup_conditional(df):
    print()
    print("=" * 72)
    print("SETUP-CONDITIONAL EARLY-ADVERSE ANALYSIS")
    print("=" * 72)

    rows = []

    features = [
        "Structure_Direction_Aligned",
        "Trend_Direction_Aligned",
        "Trend_Direction_Conflict",
        "Structure_And_Trend_Aligned",
        "Liquidity_Sweep",
        "entry_trend",
        "Structure_Bias",
        "entry_price_position",
        "Fib_Premium_Discount",
        "Fib_Zone",
    ]

    for setup, setup_group in df.groupby(
        "Setup",
        dropna=False,
    ):

        print()
        print("-" * 72)
        print(
            f"SETUP: {setup}"
        )

        print(
            f"Trades: {len(setup_group)}"
        )

        for feature in features:

            if feature not in setup_group:
                continue

            working = setup_group.copy()

            working[feature] = (
                working[feature]
                .fillna("MISSING")
                .astype(str)
            )

            for level, group in working.groupby(
                feature,
                dropna=False,
            ):

                if len(group) < 2:
                    continue

                r = safe_numeric(
                    group["R_Multiple"]
                )

                rows.append(
                    {
                        "Setup": setup,
                        "Feature": feature,
                        "Level": level,
                        "N": len(group),
                        "Wins": int(
                            (r > 0).sum()
                        ),
                        "Win_Rate_Pct": win_rate(r),
                        "Avg_R": mean_or_nan(r),
                        "Median_R": median_or_nan(r),
                        "Total_R": float(r.sum()),
                        "Avg_MFE_R": mean_or_nan(
                            group["max_favorable_r"]
                        ),
                    }
                )

                print(
                    f"  {feature}={level} "
                    f"N={len(group)} "
                    f"WR={win_rate(r):.1f}% "
                    f"AvgR={mean_or_nan(r):+.3f}"
                )

    return pd.DataFrame(rows)


# ============================================================
# PERMUTATION TEST
# ============================================================

def stratified_permutation_test(
    df,
    feature,
    iterations=N_PERMUTATIONS,
):
    """
    Test whether a binary pre-entry feature has independent
    outcome separation while preserving Setup composition.

    Statistic:
        mean R(feature=True) - mean R(feature=False)

    Permutation:
        shuffle feature labels WITHIN each Setup.
    """

    working = df[
        [
            "Setup",
            feature,
            "R_Multiple",
        ]
    ].copy()

    working[feature] = (
        working[feature]
        .astype(bool)
    )

    observed_parts = []

    for _, group in working.groupby(
        "Setup"
    ):

        yes = group.loc[
            group[feature],
            "R_Multiple"
        ]

        no = group.loc[
            ~group[feature],
            "R_Multiple"
        ]

        if len(yes) == 0 or len(no) == 0:
            continue

        observed_parts.append(
            (
                yes.mean()
                - no.mean()
            )
            * len(group)
        )

    if not observed_parts:
        return np.nan, np.nan

    observed = (
        sum(observed_parts)
        / len(working)
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    null = np.empty(
        iterations,
        dtype=float,
    )

    for i in range(iterations):

        weighted_deltas = []

        for _, group in working.groupby(
            "Setup"
        ):

            labels = (
                group[feature]
                .to_numpy()
                .copy()
            )

            rng.shuffle(labels)

            yes = group.loc[
                labels,
                "R_Multiple"
            ]

            no = group.loc[
                ~labels,
                "R_Multiple"
            ]

            if len(yes) == 0 or len(no) == 0:
                continue

            weighted_deltas.append(
                (
                    yes.mean()
                    - no.mean()
                )
                * len(group)
            )

        if weighted_deltas:
            null[i] = (
                sum(weighted_deltas)
                / len(working)
            )
        else:
            null[i] = np.nan

    null = null[
        np.isfinite(null)
    ]

    if len(null) == 0:
        return observed, np.nan

    p_value = (
        np.sum(
            np.abs(null)
            >= abs(observed)
        )
        + 1
    ) / (
        len(null)
        + 1
    )

    return observed, float(
        p_value
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_stratified_delta(
    df,
    feature,
    iterations=N_BOOTSTRAPS,
):
    """
    Bootstrap the feature effect while preserving Setup
    strata.
    """

    working = df[
        [
            "Setup",
            feature,
            "R_Multiple",
        ]
    ].copy()

    working[feature] = (
        working[feature]
        .astype(bool)
    )

    rng = np.random.default_rng(
        RNG_SEED + 1
    )

    observed_parts = []

    for _, group in working.groupby(
        "Setup"
    ):

        yes = group.loc[
            group[feature],
            "R_Multiple"
        ]

        no = group.loc[
            ~group[feature],
            "R_Multiple"
        ]

        if len(yes) == 0 or len(no) == 0:
            continue

        observed_parts.append(
            (
                yes.mean()
                - no.mean()
            )
            * len(group)
        )

    if not observed_parts:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    observed = (
        sum(observed_parts)
        / len(working)
    )

    samples = []

    for _ in range(iterations):

        weighted_deltas = []

        for _, group in working.groupby(
            "Setup"
        ):

            yes = group.loc[
                group[feature],
                "R_Multiple"
            ].to_numpy()

            no = group.loc[
                ~group[feature],
                "R_Multiple"
            ].to_numpy()

            if len(yes) == 0 or len(no) == 0:
                continue

            yes_boot = rng.choice(
                yes,
                size=len(yes),
                replace=True,
            )

            no_boot = rng.choice(
                no,
                size=len(no),
                replace=True,
            )

            weighted_deltas.append(
                (
                    yes_boot.mean()
                    - no_boot.mean()
                )
                * len(group)
            )

        if weighted_deltas:
            samples.append(
                sum(weighted_deltas)
                / len(working)
            )

    samples = np.asarray(
        samples,
        dtype=float,
    )

    if len(samples) == 0:
        return (
            observed,
            np.nan,
            np.nan,
        )

    low, high = np.percentile(
        samples,
        [2.5, 97.5],
    )

    return (
        observed,
        float(low),
        float(high),
    )


# ============================================================
# CORE INDEPENDENT TESTS
# ============================================================

def run_independent_tests(df):
    print()
    print("=" * 72)
    print("INDEPENDENT PRE-ENTRY INFORMATION TESTS")
    print("=" * 72)

    features = [
        "Structure_Direction_Aligned",
        "Trend_Direction_Aligned",
        "Trend_Direction_Conflict",
        "Structure_And_Trend_Aligned",
        "Liquidity_Sweep",
    ]

    rows = []

    for feature in features:

        if feature not in df:
            continue

        observed, p_value = (
            stratified_permutation_test(
                df,
                feature,
            )
        )

        boot_observed, ci_low, ci_high = (
            bootstrap_stratified_delta(
                df,
                feature,
            )
        )

        rows.append(
            {
                "Feature": feature,
                "Observed_Delta_R": observed,
                "Permutation_P": p_value,
                "Bootstrap_Observed_R": (
                    boot_observed
                ),
                "Bootstrap_CI_Low": ci_low,
                "Bootstrap_CI_High": ci_high,
            }
        )

        print()
        print(feature)

        print(
            f"  Stratified Δ AvgR : "
            f"{observed:+.4f}R"
        )

        print(
            f"  Permutation p     : "
            f"{p_value:.6f}"
        )

        print(
            f"  Bootstrap 95% CI  : "
            f"[{ci_low:+.4f}, "
            f"{ci_high:+.4f}]R"
        )

    return pd.DataFrame(rows)


# ============================================================
# SYMBOL LOO
# ============================================================

def symbol_loo(df):
    print()
    print("=" * 72)
    print("SYMBOL LEAVE-ONE-OUT")
    print("=" * 72)

    rows = []

    features = [
        "Structure_Direction_Aligned",
        "Trend_Direction_Aligned",
        "Trend_Direction_Conflict",
        "Structure_And_Trend_Aligned",
        "Liquidity_Sweep",
    ]

    for symbol in sorted(
        df["Research_Symbol"]
        .dropna()
        .unique()
    ):

        subset = df.loc[
            df["Research_Symbol"] != symbol
        ].copy()

        for feature in features:

            if feature not in subset:
                continue

            yes = subset.loc[
                subset[feature],
                "R_Multiple"
            ]

            no = subset.loc[
                ~subset[feature],
                "R_Multiple"
            ]

            if len(yes) == 0 or len(no) == 0:
                continue

            delta = (
                yes.mean()
                - no.mean()
            )

            rows.append(
                {
                    "Left_Out_Symbol": symbol,
                    "Feature": feature,
                    "N_Yes": len(yes),
                    "N_No": len(no),
                    "Yes_Avg_R": yes.mean(),
                    "No_Avg_R": no.mean(),
                    "Delta_Avg_R": delta,
                }
            )

            print(
                f"{symbol:12s} "
                f"{feature:36s} "
                f"Δ={delta:+.4f}R"
            )

    return pd.DataFrame(rows)


# ============================================================
# YEAR LOO
# ============================================================

def year_loo(df):
    print()
    print("=" * 72)
    print("YEAR LEAVE-ONE-OUT")
    print("=" * 72)

    temp = df.copy()

    temp["Year"] = (
        pd.to_datetime(
            temp["Research_Date"]
        ).dt.year
    )

    rows = []

    features = [
        "Structure_Direction_Aligned",
        "Trend_Direction_Aligned",
        "Trend_Direction_Conflict",
        "Structure_And_Trend_Aligned",
        "Liquidity_Sweep",
    ]

    for year in sorted(
        temp["Year"]
        .dropna()
        .unique()
    ):

        subset = temp.loc[
            temp["Year"] != year
        ].copy()

        for feature in features:

            yes = subset.loc[
                subset[feature],
                "R_Multiple"
            ]

            no = subset.loc[
                ~subset[feature],
                "R_Multiple"
            ]

            if len(yes) == 0 or len(no) == 0:
                continue

            delta = (
                yes.mean()
                - no.mean()
            )

            rows.append(
                {
                    "Left_Out_Year": int(year),
                    "Feature": feature,
                    "N_Yes": len(yes),
                    "N_No": len(no),
                    "Yes_Avg_R": yes.mean(),
                    "No_Avg_R": no.mean(),
                    "Delta_Avg_R": delta,
                }
            )

            print(
                f"{year} "
                f"{feature:36s} "
                f"Δ={delta:+.4f}R"
            )

    return pd.DataFrame(rows)


# ============================================================
# EVENT-LEVEL OUTPUT
# ============================================================

def build_event_output(df):
    columns = [
        "Trade_Key",
        "Research_Symbol",
        "Research_Date",
        "Direction",
        "Setup",
        "R_Multiple",
        "Bar5_Early_Adverse_R",

        # PRE-ENTRY STRUCTURE
        "Structure_Bias",
        "BOS",
        "Break_Age",
        "Break_Confirmed",
        "Break_Direction",
        "CHOCH",

        # PRE-ENTRY TREND
        "entry_trend",
        "entry_trend_slope",
        "entry_trend_strength",

        # PRE-ENTRY LIQUIDITY
        "Liquidity_Sweep",

        # PRE-ENTRY SETUP QUALITY
        "Setup_Confidence",
        "entry_price_position",

        # PRE-ENTRY FIBONACCI
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
        "Fib_Retracement",
        "Fib_Extension_Percent",
        "Fib_Extension_State",
        "Fib_Premium_Discount",
        "Fib_Zone",

        # DERIVED PRE-ENTRY RELATIONSHIPS
        "Structure_Direction_Aligned",
        "Trend_Direction_Aligned",
        "Trend_Direction_Conflict",
        "Structure_Trend_Agree",
        "Structure_And_Trend_Aligned",
        "Directional_Trend_Slope",

        # POST-ENTRY OUTCOMES
        "favorable_first",
        "immediate_failure",
        "max_adverse_r",
        "max_favorable_r",
        "reached_0_5r",
        "reached_1r",
        "reached_1_5r",
    ]

    columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[columns].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "TRADE SENSE-AI — EARLY-ADVERSE "
        "MARKET STRUCTURE FORENSIC"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    frozen = load_frozen()

    authoritative_events = (
        load_authoritative_events()
    )

    # --------------------------------------------------------
    # BUILD EVENT DATASET
    # --------------------------------------------------------

    df = build_event_dataset(
        frozen,
        authoritative_events,
    )

    # --------------------------------------------------------
    # PRE-ENTRY FEATURES
    # --------------------------------------------------------

    df = build_pre_entry_features(
        df
    )

    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    print_event_overview(
        df
    )

    # --------------------------------------------------------
    # CATEGORICAL
    # --------------------------------------------------------

    category_results = (
        categorical_analysis(df)
    )

    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    numeric_results = (
        numeric_analysis(df)
    )

    # --------------------------------------------------------
    # SETUP
    # --------------------------------------------------------

    setup_results = (
        setup_conditional(df)
    )

    # --------------------------------------------------------
    # INDEPENDENT TESTS
    # --------------------------------------------------------

    independent_results = (
        run_independent_tests(df)
    )

    # --------------------------------------------------------
    # SYMBOL LOO
    # --------------------------------------------------------

    symbol_results = (
        symbol_loo(df)
    )

    # --------------------------------------------------------
    # YEAR LOO
    # --------------------------------------------------------

    year_results = (
        year_loo(df)
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # EVENT LEVEL
    # --------------------------------------------------------

    event_output = build_event_output(
        df
    )

    event_output.to_csv(
        EVENT_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # CATEGORY OUTPUT
    # --------------------------------------------------------

    category_results.to_csv(
        CATEGORY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # NUMERIC OUTPUT
    # --------------------------------------------------------

    numeric_results.to_csv(
        NUMERIC_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # SETUP OUTPUT
    # --------------------------------------------------------

    setup_results.to_csv(
        SETUP_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # SYMBOL OUTPUT
    # --------------------------------------------------------

    symbol_results.to_csv(
        SYMBOL_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # YEAR OUTPUT
    # --------------------------------------------------------

    year_results.to_csv(
        YEAR_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESEARCH SUMMARY")
    print("=" * 72)

    print(
        "Question:"
    )

    print(
        "Can pre-entry market structure, trend, "
        "liquidity or entry-quality information "
        "distinguish recovery among the authoritative "
        "5B/.25R early-adverse events?"
    )

    print()

    if not independent_results.empty:

        for _, row in independent_results.iterrows():

            print(
                f"{row['Feature']}: "
                f"Δ={row['Observed_Delta_R']:+.4f}R, "
                f"p={row['Permutation_P']:.6f}, "
                f"CI=["
                f"{row['Bootstrap_CI_Low']:+.4f}, "
                f"{row['Bootstrap_CI_High']:+.4f}"
                "]"
            )

    print()
    print(
        "No production strategy changes were made."
    )

    print(
        "No frozen hypothesis changes were made."
    )

    print(
        "No OOS changes were made."
    )

    print()
    print(
        f"Event output : {EVENT_OUTPUT}"
    )

    print(
        f"Category     : {CATEGORY_OUTPUT}"
    )

    print(
        f"Numeric      : {NUMERIC_OUTPUT}"
    )

    print(
        f"Setup        : {SETUP_OUTPUT}"
    )

    print(
        f"Symbol LOO   : {SYMBOL_OUTPUT}"
    )

    print(
        f"Year LOO     : {YEAR_OUTPUT}"
    )

    print()
    print("=" * 72)
    print(
        "EARLY_ADVERSE_MARKET_STRUCTURE_COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()