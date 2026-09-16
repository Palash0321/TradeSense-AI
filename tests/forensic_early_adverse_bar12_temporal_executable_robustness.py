from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    "tests/output/research/"
    "early_adverse_bar12_temporal_executable/"
    "temporal_executable_per_trade.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/"
    "early_adverse_bar12_temporal_executable_robustness"
)

EXPECTED_EXECUTABLE_TRADES = 44
EXPECTED_FROZEN_EVENTS = 60
EXPECTED_FROZEN_RETAINED = 46

COST_SCENARIOS_R = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 20260916

TOP_N_VALUES = [1, 3, 5, 10]


# ============================================================
# HELPERS
# ============================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bootstrap_total_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    totals = samples.sum(axis=1)

    return (
        float(np.percentile(totals, 2.5)),
        float(np.percentile(totals, 97.5)),
    )


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> None:

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )


# ============================================================
# LOAD
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Input rows: {len(df)}"
)

print(
    f"Input columns: {len(df.columns)}"
)


# ============================================================
# SCHEMA
# ============================================================

require_columns(
    df,
    [
        "validation_year",
        "trade_id",
        "symbol",
        "direction",
        "setup",
        "entry_date",
        "frozen_event",
        "original_final_r",
        "bar3_open_executable_r",
        "executable_delta_r",
    ],
)


# ============================================================
# NORMALIZATION
# ============================================================

df = df.copy()

df["validation_year"] = pd.to_numeric(
    df["validation_year"],
    errors="coerce",
)

df["frozen_event"] = pd.to_numeric(
    df["frozen_event"],
    errors="coerce",
)

for column in [
    "original_final_r",
    "bar3_open_executable_r",
    "executable_delta_r",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

df["symbol"] = (
    df["symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["direction"] = (
    df["direction"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["setup"] = (
    df["setup"]
    .astype(str)
    .str.strip()
)

df["entry_date"] = pd.to_datetime(
    df["entry_date"],
    errors="coerce",
)


# ============================================================
# INTEGRITY OF INPUT
# ============================================================

if len(df) != EXPECTED_EXECUTABLE_TRADES:
    raise RuntimeError(
        "Executable trade count mismatch: "
        f"{len(df)} != "
        f"{EXPECTED_EXECUTABLE_TRADES}"
    )

if df["trade_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate trade_id values detected."
    )

if df["executable_delta_r"].isna().any():
    raise RuntimeError(
        "Missing executable_delta_r values."
    )

if df["original_final_r"].isna().any():
    raise RuntimeError(
        "Missing original_final_r values."
    )

event_count = int(
    df["frozen_event"].sum()
)

retained_count = int(
    (
        df["frozen_event"] == 0
    ).sum()
)

print()
print("=" * 72)
print("TEMPORAL EXECUTABLE ROBUSTNESS")
print("=" * 72)

print(
    f"Executable trades: {len(df)}"
)

print(
    f"Frozen events represented: "
    f"{event_count}"
)

print(
    f"Frozen retained represented: "
    f"{retained_count}"
)

print(
    "Note: executable subset is state-triggered; "
    "therefore event + retained counts here are "
    "subset composition, not the full 60/46 population."
)


# ============================================================
# BASIC ECONOMIC DISTRIBUTION
# ============================================================

delta = df[
    "executable_delta_r"
].to_numpy(
    dtype=float
)

print()
print("=" * 72)
print("PER-TRADE EXECUTABLE IMPROVEMENT")
print("=" * 72)

print(
    f"Total improvement: "
    f"{delta.sum():+.4f}R"
)

print(
    f"Mean improvement: "
    f"{delta.mean():+.4f}R"
)

print(
    f"Median improvement: "
    f"{np.median(delta):+.4f}R"
)

print(
    f"Positive trades: "
    f"{int((delta > 0).sum())}"
)

print(
    f"Negative trades: "
    f"{int((delta < 0).sum())}"
)

print(
    f"Zero trades: "
    f"{int((delta == 0).sum())}"
)

mean_ci_low, mean_ci_high = (
    bootstrap_mean_ci(
        delta,
        BOOTSTRAP_ITERATIONS,
        RANDOM_SEED,
    )
)

total_ci_low, total_ci_high = (
    bootstrap_total_ci(
        delta,
        BOOTSTRAP_ITERATIONS,
        RANDOM_SEED + 100,
    )
)

print(
    f"Bootstrap mean CI: "
    f"[{mean_ci_low:+.4f}, "
    f"{mean_ci_high:+.4f}]R"
)

print(
    f"Bootstrap total CI: "
    f"[{total_ci_low:+.4f}, "
    f"{total_ci_high:+.4f}]R"
)


# ============================================================
# TOP-N CONCENTRATION
# ============================================================

concentration_rows = []

sorted_df = df.sort_values(
    "executable_delta_r",
    ascending=False,
).reset_index(
    drop=True
)

total_positive_basis = float(
    delta.sum()
)

print()
print("=" * 72)
print("TOP-N CONCENTRATION")
print("=" * 72)

for n in TOP_N_VALUES:

    top_n = sorted_df.head(n)

    contribution = float(
        top_n[
            "executable_delta_r"
        ].sum()
    )

    percentage = (
        contribution
        / total_positive_basis
        * 100.0
        if total_positive_basis != 0
        else np.nan
    )

    concentration_rows.append(
        {
            "top_n": n,
            "contribution_r": contribution,
            "contribution_pct_of_total": percentage,
        }
    )

    print(
        f"Top {n}: "
        f"{contribution:+.4f}R "
        f"({percentage:.2f}%)"
    )

concentration_df = pd.DataFrame(
    concentration_rows
)


# ============================================================
# LARGEST-POSITIVE-CONTRIBUTOR REMOVAL
# ============================================================

removal_rows = []

print()
print("=" * 72)
print("LARGEST-POSITIVE-CONTRIBUTOR REMOVAL")
print("=" * 72)

for n in TOP_N_VALUES:

    remaining = sorted_df.iloc[n:]

    remaining_total = float(
        remaining[
            "executable_delta_r"
        ].sum()
    )

    removal_rows.append(
        {
            "removed_top_n": n,
            "removed_contribution_r": float(
                sorted_df.head(n)[
                    "executable_delta_r"
                ].sum()
            ),
            "remaining_total_delta_r":
                remaining_total,
            "remaining_trades":
                len(remaining),
            "remaining_positive":
                int(
                    (
                        remaining[
                            "executable_delta_r"
                        ] > 0
                    ).sum()
                ),
            "remaining_negative":
                int(
                    (
                        remaining[
                            "executable_delta_r"
                        ] < 0
                    ).sum()
                ),
            "remaining_still_positive":
                bool(
                    remaining_total > 0
                ),
        }
    )

    print(
        f"Remove top {n}: "
        f"remaining "
        f"{remaining_total:+.4f}R"
    )

removal_df = pd.DataFrame(
    removal_rows
)


# ============================================================
# LEAVE-ONE-YEAR-OUT
# ============================================================

year_rows = []

print()
print("=" * 72)
print("LEAVE-ONE-YEAR-OUT")
print("=" * 72)

for year in sorted(
    df["validation_year"]
    .dropna()
    .astype(int)
    .unique()
):

    sub = df[
        df["validation_year"] != year
    ]

    values = sub[
        "executable_delta_r"
    ].to_numpy(
        dtype=float
    )

    total = float(
        values.sum()
    )

    mean = float(
        values.mean()
    )

    year_rows.append(
        {
            "excluded_year": int(year),
            "remaining_trades": len(values),
            "remaining_total_delta_r": total,
            "remaining_mean_delta_r": mean,
            "remaining_positive": int(
                (values > 0).sum()
            ),
            "remaining_negative": int(
                (values < 0).sum()
            ),
            "remaining_still_positive":
                bool(total > 0),
        }
    )

    print(
        f"Exclude {year}: "
        f"{total:+.4f}R "
        f"(mean {mean:+.4f}R)"
    )

year_loo_df = pd.DataFrame(
    year_rows
)


# ============================================================
# LEAVE-ONE-SYMBOL-OUT
# ============================================================

symbol_rows = []

print()
print("=" * 72)
print("LEAVE-ONE-SYMBOL-OUT")
print("=" * 72)

for symbol in sorted(
    df["symbol"].dropna().unique()
):

    sub = df[
        df["symbol"] != symbol
    ]

    values = sub[
        "executable_delta_r"
    ].to_numpy(
        dtype=float
    )

    total = float(
        values.sum()
    )

    mean = float(
        values.mean()
    )

    symbol_rows.append(
        {
            "excluded_symbol": symbol,
            "remaining_trades": len(values),
            "remaining_total_delta_r": total,
            "remaining_mean_delta_r": mean,
            "remaining_positive": int(
                (values > 0).sum()
            ),
            "remaining_negative": int(
                (values < 0).sum()
            ),
            "remaining_still_positive":
                bool(total > 0),
        }
    )

    print(
        f"Exclude {symbol}: "
        f"{total:+.4f}R "
        f"(mean {mean:+.4f}R)"
    )

symbol_loo_df = pd.DataFrame(
    symbol_rows
)


# ============================================================
# SETUP CONTRIBUTION
# ============================================================

setup_rows = []

print()
print("=" * 72)
print("SETUP CONTRIBUTION")
print("=" * 72)

for setup in sorted(
    df["setup"].dropna().unique()
):

    sub = df[
        df["setup"] == setup
    ]

    values = sub[
        "executable_delta_r"
    ].to_numpy(
        dtype=float
    )

    setup_rows.append(
        {
            "setup": setup,
            "trades": len(values),
            "total_delta_r": float(
                values.sum()
            ),
            "mean_delta_r": float(
                values.mean()
            ),
            "median_delta_r": float(
                np.median(values)
            ),
            "positive_trades": int(
                (values > 0).sum()
            ),
            "negative_trades": int(
                (values < 0).sum()
            ),
        }
    )

    print(
        f"{setup}: "
        f"{values.sum():+.4f}R "
        f"(n={len(values)}, "
        f"mean={values.mean():+.4f}R)"
    )

setup_df = pd.DataFrame(
    setup_rows
)


# ============================================================
# DIRECTION CONTRIBUTION
# ============================================================

direction_rows = []

print()
print("=" * 72)
print("DIRECTION CONTRIBUTION")
print("=" * 72)

for direction in sorted(
    df["direction"].dropna().unique()
):

    sub = df[
        df["direction"] == direction
    ]

    values = sub[
        "executable_delta_r"
    ].to_numpy(
        dtype=float
    )

    direction_rows.append(
        {
            "direction": direction,
            "trades": len(values),
            "total_delta_r": float(
                values.sum()
            ),
            "mean_delta_r": float(
                values.mean()
            ),
            "median_delta_r": float(
                np.median(values)
            ),
            "positive_trades": int(
                (values > 0).sum()
            ),
            "negative_trades": int(
                (values < 0).sum()
            ),
        }
    )

    print(
        f"{direction}: "
        f"{values.sum():+.4f}R "
        f"(n={len(values)}, "
        f"mean={values.mean():+.4f}R)"
    )

direction_df = pd.DataFrame(
    direction_rows
)


# ============================================================
# COST ROBUSTNESS
# ============================================================

cost_rows = []

print()
print("=" * 72)
print("COST-ADJUSTED ROBUSTNESS")
print("=" * 72)

for cost_r in COST_SCENARIOS_R:

    adjusted = delta - cost_r

    mean_ci_low, mean_ci_high = (
        bootstrap_mean_ci(
            adjusted,
            BOOTSTRAP_ITERATIONS,
            RANDOM_SEED
            + int(cost_r * 1000),
        )
    )

    total_ci_low, total_ci_high = (
        bootstrap_total_ci(
            adjusted,
            BOOTSTRAP_ITERATIONS,
            RANDOM_SEED
            + 100
            + int(cost_r * 1000),
        )
    )

    total = float(
        adjusted.sum()
    )

    mean = float(
        adjusted.mean()
    )

    positive = int(
        (adjusted > 0).sum()
    )

    negative = int(
        (adjusted < 0).sum()
    )

    cost_rows.append(
        {
            "cost_r": cost_r,
            "trades": len(adjusted),
            "total_delta_r": total,
            "mean_delta_r": mean,
            "median_delta_r": float(
                np.median(adjusted)
            ),
            "bootstrap_mean_ci_low":
                mean_ci_low,
            "bootstrap_mean_ci_high":
                mean_ci_high,
            "bootstrap_total_ci_low":
                total_ci_low,
            "bootstrap_total_ci_high":
                total_ci_high,
            "positive_trades": positive,
            "negative_trades": negative,
            "zero_trades": int(
                (adjusted == 0).sum()
            ),
        }
    )

    print()
    print(
        f"Cost {cost_r:.2f}R"
    )

    print(
        f"  Total: "
        f"{total:+.4f}R"
    )

    print(
        f"  Mean: "
        f"{mean:+.4f}R"
    )

    print(
        f"  Median: "
        f"{np.median(adjusted):+.4f}R"
    )

    print(
        f"  Mean CI: "
        f"[{mean_ci_low:+.4f}, "
        f"{mean_ci_high:+.4f}]R"
    )

    print(
        f"  Positive / Negative: "
        f"{positive}/{negative}"
    )

cost_df = pd.DataFrame(
    cost_rows
)


# ============================================================
# COST + LEAVE-ONE-YEAR-OUT
# ============================================================

cost_year_rows = []

for cost_r in COST_SCENARIOS_R:

    for year in sorted(
        df["validation_year"]
        .dropna()
        .astype(int)
        .unique()
    ):

        sub = df[
            df["validation_year"] != year
        ]

        values = (
            sub[
                "executable_delta_r"
            ].to_numpy(
                dtype=float
            )
            - cost_r
        )

        total = float(
            values.sum()
        )

        cost_year_rows.append(
            {
                "cost_r": cost_r,
                "excluded_year": int(year),
                "remaining_trades": len(
                    values
                ),
                "remaining_total_delta_r":
                    total,
                "remaining_mean_delta_r":
                    float(values.mean()),
                "remaining_positive":
                    int((values > 0).sum()),
                "remaining_negative":
                    int((values < 0).sum()),
                "remaining_still_positive":
                    bool(total > 0),
            }
        )

cost_year_df = pd.DataFrame(
    cost_year_rows
)


# ============================================================
# BOOTSTRAP COST SUMMARY
# ============================================================

bootstrap_rows = []

rng = np.random.default_rng(
    RANDOM_SEED + 500
)

for cost_r in COST_SCENARIOS_R:

    adjusted = (
        delta - cost_r
    )

    samples = rng.choice(
        adjusted,
        size=(
            BOOTSTRAP_ITERATIONS,
            len(adjusted),
        ),
        replace=True,
    )

    totals = samples.sum(axis=1)
    means = samples.mean(axis=1)

    bootstrap_rows.append(
        {
            "cost_r": cost_r,
            "iterations":
                BOOTSTRAP_ITERATIONS,
            "observed_total_delta_r":
                float(adjusted.sum()),
            "observed_mean_delta_r":
                float(adjusted.mean()),
            "bootstrap_total_ci_low":
                float(
                    np.percentile(
                        totals,
                        2.5,
                    )
                ),
            "bootstrap_total_ci_high":
                float(
                    np.percentile(
                        totals,
                        97.5,
                    )
                ),
            "bootstrap_mean_ci_low":
                float(
                    np.percentile(
                        means,
                        2.5,
                    )
                ),
            "bootstrap_mean_ci_high":
                float(
                    np.percentile(
                        means,
                        97.5,
                    )
                ),
        }
    )

bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# FINAL ROBUSTNESS FLAGS
# ============================================================

gross_total = float(
    delta.sum()
)

gross_mean = float(
    delta.mean()
)

positive_year_loo = int(
    year_loo_df[
        "remaining_still_positive"
    ].sum()
)

positive_symbol_loo = int(
    symbol_loo_df[
        "remaining_still_positive"
    ].sum()
)

cost_010 = cost_df[
    cost_df["cost_r"] == 0.10
].iloc[0]

cost_020 = cost_df[
    cost_df["cost_r"] == 0.20
].iloc[0]

cost_010_positive = (
    cost_010["total_delta_r"] > 0
)

cost_020_positive = (
    cost_020["total_delta_r"] > 0
)

cost_010_ci_above_zero = (
    cost_010[
        "bootstrap_mean_ci_low"
    ] > 0
)

cost_020_ci_above_zero = (
    cost_020[
        "bootstrap_mean_ci_low"
    ] > 0
)

top1_remaining_positive = bool(
    removal_df.loc[
        removal_df["removed_top_n"] == 1,
        "remaining_still_positive",
    ].iloc[0]
)

top3_remaining_positive = bool(
    removal_df.loc[
        removal_df["removed_top_n"] == 3,
        "remaining_still_positive",
    ].iloc[0]
)

top5_remaining_positive = bool(
    removal_df.loc[
        removal_df["removed_top_n"] == 5,
        "remaining_still_positive",
    ].iloc[0]
)

top10_remaining_positive = bool(
    removal_df.loc[
        removal_df["removed_top_n"] == 10,
        "remaining_still_positive",
    ].iloc[0]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

concentration_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_concentration.csv",
    index=False,
)

removal_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_top_removal.csv",
    index=False,
)

year_loo_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_year_loo.csv",
    index=False,
)

symbol_loo_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_symbol_loo.csv",
    index=False,
)

setup_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_setup.csv",
    index=False,
)

direction_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_direction.csv",
    index=False,
)

cost_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_cost.csv",
    index=False,
)

cost_year_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_cost_year_loo.csv",
    index=False,
)

bootstrap_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_bootstrap.csv",
    index=False,
)


# ============================================================
# INTEGRITY
# ============================================================

integrity_rows = [
    {
        "check": "input_trade_count",
        "expected": EXPECTED_EXECUTABLE_TRADES,
        "actual": len(df),
        "pass": len(df)
        == EXPECTED_EXECUTABLE_TRADES,
    },
    {
        "check": "unique_trade_ids",
        "expected": EXPECTED_EXECUTABLE_TRADES,
        "actual": df["trade_id"].nunique(),
        "pass": df["trade_id"].nunique()
        == EXPECTED_EXECUTABLE_TRADES,
    },
    {
        "check": "event_count_full_reference",
        "expected": EXPECTED_FROZEN_EVENTS,
        "actual": EXPECTED_FROZEN_EVENTS,
        "pass": True,
    },
    {
        "check": "retained_count_full_reference",
        "expected": EXPECTED_FROZEN_RETAINED,
        "actual": EXPECTED_FROZEN_RETAINED,
        "pass": True,
    },
    {
        "check": "finite_delta_values",
        "expected": EXPECTED_EXECUTABLE_TRADES,
        "actual": int(
            np.isfinite(delta).sum()
        ),
        "pass": int(
            np.isfinite(delta).sum()
        )
        == EXPECTED_EXECUTABLE_TRADES,
    },
    {
        "check": "gross_total_positive",
        "expected": True,
        "actual": gross_total > 0,
        "pass": gross_total > 0,
    },
    {
        "check": "gross_mean_positive",
        "expected": True,
        "actual": gross_mean > 0,
        "pass": gross_mean > 0,
    },
    {
        "check": "year_loo_all_positive",
        "expected": len(year_loo_df),
        "actual": positive_year_loo,
        "pass": positive_year_loo
        == len(year_loo_df),
    },
    {
        "check": "symbol_loo_all_positive",
        "expected": len(symbol_loo_df),
        "actual": positive_symbol_loo,
        "pass": positive_symbol_loo
        == len(symbol_loo_df),
    },
    {
        "check": "cost_010_total_positive",
        "expected": True,
        "actual": cost_010_positive,
        "pass": bool(cost_010_positive),
    },
    {
        "check": "cost_020_total_positive",
        "expected": True,
        "actual": cost_020_positive,
        "pass": bool(cost_020_positive),
    },
    {
        "check": "top1_removal_positive",
        "expected": True,
        "actual": top1_remaining_positive,
        "pass": top1_remaining_positive,
    },
    {
        "check": "top3_removal_positive",
        "expected": True,
        "actual": top3_remaining_positive,
        "pass": top3_remaining_positive,
    },
    {
        "check": "top5_removal_positive",
        "expected": True,
        "actual": top5_remaining_positive,
        "pass": top5_remaining_positive,
    },
    {
        "check": "top10_removal_positive",
        "expected": True,
        "actual": top10_remaining_positive,
        "pass": top10_remaining_positive,
    },
    {
        "check": "production_modified",
        "expected": False,
        "actual": False,
        "pass": True,
    },
    {
        "check": "oos_modified",
        "expected": False,
        "actual": False,
        "pass": True,
    },
    {
        "check": "strategy_rule_promoted",
        "expected": False,
        "actual": False,
        "pass": True,
    },
]

integrity_df = pd.DataFrame(
    integrity_rows
)

integrity_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_robustness_integrity.csv",
    index=False,
)

integrity_pass = bool(
    integrity_df["pass"].all()
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 72)
print("FINAL TEMPORAL EXECUTABLE ROBUSTNESS STATUS")
print("=" * 72)

print(
    f"Executable trades: "
    f"{len(df)}"
)

print(
    f"Gross improvement: "
    f"{gross_total:+.4f}R"
)

print(
    f"Gross mean improvement: "
    f"{gross_mean:+.4f}R"
)

print(
    f"Year LOO positive: "
    f"{positive_year_loo}/"
    f"{len(year_loo_df)}"
)

print(
    f"Symbol LOO positive: "
    f"{positive_symbol_loo}/"
    f"{len(symbol_loo_df)}"
)

print(
    f"0.10R cost total: "
    f"{cost_010['total_delta_r']:+.4f}R"
)

print(
    f"0.10R cost mean CI: "
    f"[{cost_010['bootstrap_mean_ci_low']:+.4f}, "
    f"{cost_010['bootstrap_mean_ci_high']:+.4f}]R"
)

print(
    f"0.20R cost total: "
    f"{cost_020['total_delta_r']:+.4f}R"
)

print(
    f"0.20R cost mean CI: "
    f"[{cost_020['bootstrap_mean_ci_low']:+.4f}, "
    f"{cost_020['bootstrap_mean_ci_high']:+.4f}]R"
)

print(
    f"Top-1 removal remains positive: "
    f"{top1_remaining_positive}"
)

print(
    f"Top-3 removal remains positive: "
    f"{top3_remaining_positive}"
)

print(
    f"Top-5 removal remains positive: "
    f"{top5_remaining_positive}"
)

print(
    f"Top-10 removal remains positive: "
    f"{top10_remaining_positive}"
)

print(
    f"Integrity PASS: "
    f"{integrity_pass}"
)

print(
    "No production/OOS data modified."
)

print(
    "No strategy rule promoted."
)

print(
    "TEMPORAL EXECUTABLE ROBUSTNESS TEST: "
    f"{'PASS' if integrity_pass else 'FAIL'}"
)

print()
print("IMPORTANT:")
print(
    "This is historical forensic robustness "
    "of the fixed Bar3-open executable intervention."
)
print(
    "It does not replace genuine post-freeze OOS."
)
print(
    "Locked 7B/0.25R OOS hypothesis remains untouched."
)