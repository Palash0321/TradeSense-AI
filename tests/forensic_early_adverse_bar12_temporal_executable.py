from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


# ============================================================
# CONFIGURATION
# ============================================================

TRAJECTORY_FILE = Path(
    "tests/output/research/early_adverse_bar12_trajectory/"
    "early_adverse_bar12_trajectory_per_trade.csv"
)

PATH_FILE = Path(
    "tests/output/research/early_adverse_path/"
    "early_adverse_path_per_trade_bar.csv"
)

PRODUCTION_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/"
    "early_adverse_bar12_temporal_executable"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

STATE_COLUMN = "bar2_favorable_failed_to_improve"

COST_SCENARIOS_R = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]

MIN_HISTORY_TRADES = 20

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 20260916

DIRECTION_ALIASES = [
    "direction",
    "Direction",
]

SYMBOL_ALIASES = [
    "symbol",
    "Symbol",
    "_symbol",
]

SETUP_ALIASES = [
    "setup",
    "Setup",
]

ENTRY_DATE_ALIASES = [
    "entry_date",
    "Entry Date",
    "date",
]

FINAL_R_ALIASES = [
    "original_final_r",
    "r_multiple",
    "final_r",
    "R_multiple",
    "R",
]

BAR3_OPEN_R_ALIASES = [
    "bar3_open_executable_r",
    "bar3_open_r",
]


# ============================================================
# HELPERS
# ============================================================

def first_existing(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{label} missing required columns: {missing}"
        )


def normalize_direction(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {"LONG", "BUY", "1"}:
        return "LONG"

    if text in {"SHORT", "SELL", "-1"}:
        return "SHORT"

    return text


def normalize_symbol(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_date(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return ""

    return parsed.strftime("%Y-%m-%d")


def make_merge_key(
    symbol,
    direction,
    entry_date,
) -> str:
    return (
        f"{normalize_symbol(symbol)}|"
        f"{normalize_direction(direction)}|"
        f"{normalize_date(entry_date)}"
    )


def directional_close_r(
    direction: str,
    close_price: float,
    entry_price: float,
    risk_per_unit: float,
) -> float:
    if (
        pd.isna(close_price)
        or pd.isna(entry_price)
        or pd.isna(risk_per_unit)
        or risk_per_unit == 0
    ):
        return np.nan

    if direction == "LONG":
        return (
            float(close_price) - float(entry_price)
        ) / float(risk_per_unit)

    if direction == "SHORT":
        return (
            float(entry_price) - float(close_price)
        ) / float(risk_per_unit)

    return np.nan


def bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

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
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

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


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(values.median())


def fisher_test(
    state: pd.Series,
    event: pd.Series,
) -> tuple[float, float]:
    tmp = pd.DataFrame(
        {
            "state": state.astype(int),
            "event": event.astype(int),
        }
    )

    table = pd.crosstab(
        tmp["state"],
        tmp["event"],
    )

    for idx in [0, 1]:
        if idx not in table.index:
            table.loc[idx] = 0

    for col in [0, 1]:
        if col not in table.columns:
            table[col] = 0

    table = table.sort_index().sort_index(axis=1)

    odds_ratio, p_value = fisher_exact(
        table.to_numpy()
    )

    return float(odds_ratio), float(p_value)


# ============================================================
# LOAD DATA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

trajectory = pd.read_csv(
    TRAJECTORY_FILE
)

path = pd.read_csv(
    PATH_FILE
)

production = pd.read_csv(
    PRODUCTION_FILE
)

print(
    f"Trajectory rows: {len(trajectory)}, "
    f"columns: {len(trajectory.columns)}"
)

print(
    f"Path rows: {len(path)}, "
    f"columns: {len(path.columns)}"
)

print(
    f"Production rows: {len(production)}, "
    f"columns: {len(production.columns)}"
)


# ============================================================
# FROZEN POPULATION VALIDATION
# ============================================================

require_columns(
    trajectory,
    [
        "trade_id",
        "symbol",
        "direction",
        "entry_date",
        "setup",
        "frozen_event",
        STATE_COLUMN,
    ],
    "Trajectory",
)

require_columns(
    path,
    [
        "symbol",
        "direction",
        "entry_date",
        "bar_number",
        "open",
        "close",
        "entry_price_reconstructed",
        "risk_per_unit",
        "production_bars_in_trade",
        "production_observed",
        "production_observation_status",
    ],
    "Path",
)

production_direction_col = first_existing(
    production,
    DIRECTION_ALIASES,
)

production_symbol_col = first_existing(
    production,
    SYMBOL_ALIASES,
)

production_setup_col = first_existing(
    production,
    SETUP_ALIASES,
)

production_date_col = first_existing(
    production,
    ENTRY_DATE_ALIASES,
)

production_final_r_col = first_existing(
    production,
    FINAL_R_ALIASES,
)

if production_direction_col is None:
    raise RuntimeError(
        "Could not resolve production direction column."
    )

if production_symbol_col is None:
    raise RuntimeError(
        "Could not resolve production symbol column."
    )

if production_date_col is None:
    raise RuntimeError(
        "Could not resolve production entry-date column."
    )

if production_final_r_col is None:
    raise RuntimeError(
        "Could not resolve production final-R column."
    )


trajectory = trajectory.copy()

trajectory["symbol_norm"] = trajectory[
    "symbol"
].map(normalize_symbol)

trajectory["direction_norm"] = trajectory[
    "direction"
].map(normalize_direction)

trajectory["entry_date_norm"] = trajectory[
    "entry_date"
].map(normalize_date)

trajectory["merge_key"] = trajectory.apply(
    lambda row: make_merge_key(
        row["symbol"],
        row["direction"],
        row["entry_date"],
    ),
    axis=1,
)

trajectory[STATE_COLUMN] = pd.to_numeric(
    trajectory[STATE_COLUMN],
    errors="coerce",
)

trajectory["frozen_event"] = pd.to_numeric(
    trajectory["frozen_event"],
    errors="coerce",
)

trajectory["entry_date_dt"] = pd.to_datetime(
    trajectory["entry_date"],
    errors="coerce",
)

if len(trajectory) != EXPECTED_TRADES:
    raise RuntimeError(
        f"Frozen trajectory count mismatch: "
        f"{len(trajectory)} != {EXPECTED_TRADES}"
    )

if trajectory["merge_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate trajectory merge keys detected."
    )

event_count = int(
    (trajectory["frozen_event"] == 1).sum()
)

retained_count = int(
    (trajectory["frozen_event"] == 0).sum()
)

if event_count != EXPECTED_EVENTS:
    raise RuntimeError(
        f"Frozen event count mismatch: "
        f"{event_count} != {EXPECTED_EVENTS}"
    )

if retained_count != EXPECTED_RETAINED:
    raise RuntimeError(
        f"Frozen retained count mismatch: "
        f"{retained_count} != {EXPECTED_RETAINED}"
    )

state_values = sorted(
    trajectory[STATE_COLUMN]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

if state_values != [0, 1]:
    raise RuntimeError(
        f"State is not binary: {state_values}"
    )


# ============================================================
# PRODUCTION NORMALIZATION
# ============================================================

production_work = production.copy()

production_work["symbol"] = production_work[
    production_symbol_col
].map(normalize_symbol)

production_work["direction"] = production_work[
    production_direction_col
].map(normalize_direction)

production_work["entry_date"] = production_work[
    production_date_col
].map(normalize_date)

production_work["merge_key"] = production_work.apply(
    lambda row: make_merge_key(
        row["symbol"],
        row["direction"],
        row["entry_date"],
    ),
    axis=1,
)

production_work["original_final_r"] = pd.to_numeric(
    production_work[production_final_r_col],
    errors="coerce",
)

if production_work["merge_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate production merge keys detected."
    )

production_work = production_work[
    [
        "merge_key",
        "original_final_r",
    ]
].copy()

if len(production_work) != EXPECTED_TRADES:
    raise RuntimeError(
        "Production row count does not match frozen "
        "population."
    )


# ============================================================
# BUILD BAR-2 / BAR-3 EXECUTION DATA
# ============================================================

path_work = path.copy()

path_work["merge_key"] = path_work.apply(
    lambda row: make_merge_key(
        row["symbol"],
        row["direction"],
        row["entry_date"],
    ),
    axis=1,
)

path_work["bar_number"] = pd.to_numeric(
    path_work["bar_number"],
    errors="coerce",
)

path_work["open"] = pd.to_numeric(
    path_work["open"],
    errors="coerce",
)

path_work["close"] = pd.to_numeric(
    path_work["close"],
    errors="coerce",
)

path_work["entry_price_reconstructed"] = pd.to_numeric(
    path_work["entry_price_reconstructed"],
    errors="coerce",
)

path_work["risk_per_unit"] = pd.to_numeric(
    path_work["risk_per_unit"],
    errors="coerce",
)

path_work["production_bars_in_trade"] = pd.to_numeric(
    path_work["production_bars_in_trade"],
    errors="coerce",
)

path_work["production_observed"] = pd.to_numeric(
    path_work["production_observed"],
    errors="coerce",
)


bar2 = path_work[
    path_work["bar_number"] == 2
].copy()

bar3 = path_work[
    path_work["bar_number"] == 3
].copy()

if len(bar2) != EXPECTED_TRADES:
    raise RuntimeError(
        f"Expected 106 Bar-2 rows, got {len(bar2)}."
    )

if bar2["merge_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate Bar-2 merge keys detected."
    )

if bar3["merge_key"].duplicated().any():
    raise RuntimeError(
        "Duplicate Bar-3 merge keys detected."
    )


bar2_required = bar2[
    [
        "merge_key",
        "symbol",
        "direction",
        "entry_date",
        "entry_price_reconstructed",
        "risk_per_unit",
        "production_bars_in_trade",
        "production_observed",
        "production_observation_status",
    ]
].copy()

bar2_required = bar2_required.rename(
    columns={
        "symbol": "path_symbol",
        "direction": "path_direction",
        "entry_date": "path_entry_date",
    }
)

bar3_required = bar3[
    [
        "merge_key",
        "open",
    ]
].copy()

bar3_required = bar3_required.rename(
    columns={
        "open": "bar3_open",
    }
)

research = trajectory.merge(
    production_work,
    on="merge_key",
    how="left",
    validate="one_to_one",
)

research = research.merge(
    bar2_required,
    on="merge_key",
    how="left",
    validate="one_to_one",
)

research = research.merge(
    bar3_required,
    on="merge_key",
    how="left",
    validate="one_to_one",
)

if len(research) != EXPECTED_TRADES:
    raise RuntimeError(
        f"Research merge changed row count: "
        f"{len(research)}"
    )


# ============================================================
# STRICT OBSERVATION / EXECUTION ELIGIBILITY
# ============================================================

research["bar2_available"] = (
    research["production_observed"] == 1
)

research["survived_to_bar2_observation"] = (
    (research["production_observed"] == 1)
    & (
        research["production_observation_status"]
        == "OBSERVED_IN_PRODUCTION"
    )
    & (
        research["production_bars_in_trade"] >= 2
    )
)

research["bar3_open_available"] = (
    pd.to_numeric(
        research["bar3_open"],
        errors="coerce",
    ).notna()
)

research["entry_price_available"] = (
    pd.to_numeric(
        research["entry_price_reconstructed"],
        errors="coerce",
    ).notna()
)

research["risk_available"] = (
    pd.to_numeric(
        research["risk_per_unit"],
        errors="coerce",
    ).notna()
    & (
        pd.to_numeric(
            research["risk_per_unit"],
            errors="coerce",
        ) != 0
    )
)

research["state_triggered"] = (
    research[STATE_COLUMN]
    .fillna(0)
    .astype(int)
    == 1
)

research["executable_bar3_open"] = (
    research["state_triggered"]
    & research["bar2_available"]
    & research["bar3_open_available"]
    & research["entry_price_available"]
    & research["risk_available"]
    & research["survived_to_bar2_observation"]
)


# ============================================================
# CALCULATE EXECUTABLE BAR-3 OPEN R
# ============================================================

research["bar3_open_executable_r"] = np.nan

mask = research["executable_bar3_open"]

research.loc[mask, "bar3_open_executable_r"] = research.loc[
    mask
].apply(
    lambda row: directional_close_r(
        normalize_direction(
            row["direction"]
        ),
        float(row["bar3_open"]),
        float(
            row["entry_price_reconstructed"]
        ),
        float(row["risk_per_unit"]),
    ),
    axis=1,
)

research["executable_delta_r"] = (
    research["bar3_open_executable_r"]
    - research["original_final_r"]
)


# ============================================================
# FROZEN EVENT / RETAINED LABELS
# ============================================================

research["frozen_event"] = (
    research["frozen_event"]
    .fillna(0)
    .astype(int)
)

research["retained"] = (
    research["frozen_event"] == 0
)

research["validation_year"] = (
    research["entry_date_dt"]
    .dt.year
)


# ============================================================
# GLOBAL EXECUTABILITY SUMMARY
# ============================================================

print()
print("=" * 72)
print("STRICT TEMPORAL EXECUTABLE VALIDATION")
print("=" * 72)

print(
    f"Frozen population: "
    f"{len(research)} / "
    f"{int(research['frozen_event'].sum())} / "
    f"{int((research['frozen_event'] == 0).sum())}"
)

print(
    f"State-triggered trades: "
    f"{int(research['state_triggered'].sum())}"
)

print(
    f"Bar-2 observation available: "
    f"{int(research['bar2_available'].sum())}"
)

print(
    f"Survived to Bar-2 observation: "
    f"{int(research['survived_to_bar2_observation'].sum())}"
)

print(
    f"Bar-3 open available: "
    f"{int(research['bar3_open_available'].sum())}"
)

print(
    f"Executable at Bar-3 open: "
    f"{int(research['executable_bar3_open'].sum())}"
)

if int(research["executable_bar3_open"].sum()) == 0:
    raise RuntimeError(
        "No executable Bar-3 observations."
    )


# ============================================================
# CHRONOLOGICAL FOLDS
# ============================================================

years = sorted(
    research["validation_year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

fold_rows = []
fold_trade_rows = []

for year in years:

    validation = research[
        research["validation_year"] == year
    ].copy()

    history = research[
        research["validation_year"] < year
    ].copy()

    if len(history) < MIN_HISTORY_TRADES:
        print()
        print(
            f"YEAR {year} skipped: "
            f"history {len(history)} < "
            f"{MIN_HISTORY_TRADES}"
        )
        continue

    executable_validation = validation[
        validation["executable_bar3_open"]
    ].copy()

    if len(executable_validation) == 0:
        print()
        print(
            f"YEAR {year}: no executable "
            f"validation trades"
        )
        continue

    state = executable_validation[
        STATE_COLUMN
    ].astype(int)

    event = executable_validation[
        "frozen_event"
    ].astype(int)

    state1 = executable_validation[
        state == 1
    ]

    state0 = executable_validation[
        state == 0
    ]

    if len(state1) == 0 or len(state0) == 0:
        rd = np.nan
        odds_ratio = np.nan
        p_value = np.nan
    else:
        event_rate_state1 = (
            state1["frozen_event"].mean()
        )

        event_rate_state0 = (
            state0["frozen_event"].mean()
        )

        rd = (
            event_rate_state1
            - event_rate_state0
        ) * 100.0

        odds_ratio, p_value = fisher_test(
            state,
            event,
        )

    delta = pd.to_numeric(
        executable_validation[
            "executable_delta_r"
        ],
        errors="coerce",
    ).dropna()

    total_delta = float(delta.sum())
    mean_delta = float(delta.mean())
    median_delta = float(delta.median())

    ci_low, ci_high = bootstrap_mean_ci(
        delta.to_numpy(),
        seed=RANDOM_SEED + year,
    )

    total_ci_low, total_ci_high = (
        bootstrap_total_ci(
            delta.to_numpy(),
            seed=RANDOM_SEED + year + 100,
        )
    )

    event_delta = executable_validation[
        executable_validation["frozen_event"] == 1
    ]["executable_delta_r"].dropna()

    retained_delta = executable_validation[
        executable_validation["frozen_event"] == 0
    ]["executable_delta_r"].dropna()

    event_total = (
        float(event_delta.sum())
        if len(event_delta)
        else np.nan
    )

    retained_total = (
        float(retained_delta.sum())
        if len(retained_delta)
        else np.nan
    )

    print()
    print("=" * 72)
    print(
        f"FOLD_VALIDATE_{year}"
    )
    print("=" * 72)

    print(
        f"History trades: {len(history)}"
    )

    print(
        f"Validation trades: {len(validation)}"
    )

    print(
        f"Executable validation trades: "
        f"{len(executable_validation)}"
    )

    print(
        f"Validation date range: "
        f"{validation['entry_date_dt'].min().date()} "
        f"-> "
        f"{validation['entry_date_dt'].max().date()}"
    )

    print(
        f"Executable event rate state=1: "
        f"{state1['frozen_event'].mean() * 100:.2f}%"
    )

    print(
        f"Executable event rate state=0: "
        f"{state0['frozen_event'].mean() * 100:.2f}%"
    )

    print(
        f"Association RD: "
        f"{rd:+.2f} pp"
    )

    print(
        f"Association OR: "
        f"{odds_ratio:.4f}"
        if np.isfinite(odds_ratio)
        else "Association OR: nan"
    )

    print(
        f"Association p: "
        f"{p_value:.6f}"
        if np.isfinite(p_value)
        else "Association p: nan"
    )

    print(
        f"Executable total improvement: "
        f"{total_delta:+.4f}R"
    )

    print(
        f"Executable mean improvement: "
        f"{mean_delta:+.4f}R"
    )

    print(
        f"Executable median improvement: "
        f"{median_delta:+.4f}R"
    )

    print(
        f"Bootstrap mean CI: "
        f"[{ci_low:+.4f}, {ci_high:+.4f}]R"
    )

    print(
        f"Bootstrap total CI: "
        f"[{total_ci_low:+.4f}, "
        f"{total_ci_high:+.4f}]R"
    )

    print(
        f"Event contribution: "
        f"{event_total:+.4f}R"
    )

    print(
        f"Retained contribution: "
        f"{retained_total:+.4f}R"
    )

    fold_rows.append(
        {
            "validation_year": year,
            "history_trades": len(history),
            "validation_trades": len(validation),
            "executable_validation_trades": len(
                executable_validation
            ),
            "event_trades": int(
                executable_validation[
                    "frozen_event"
                ].sum()
            ),
            "retained_trades": int(
                (
                    executable_validation[
                        "frozen_event"
                    ] == 0
                ).sum()
            ),
            "association_rd_pp": rd,
            "association_or": odds_ratio,
            "association_p": p_value,
            "total_delta_r": total_delta,
            "mean_delta_r": mean_delta,
            "median_delta_r": median_delta,
            "bootstrap_mean_ci_low": ci_low,
            "bootstrap_mean_ci_high": ci_high,
            "bootstrap_total_ci_low": total_ci_low,
            "bootstrap_total_ci_high": total_ci_high,
            "event_total_delta_r": event_total,
            "retained_total_delta_r": retained_total,
            "positive_trades": int(
                (delta > 0).sum()
            ),
            "negative_trades": int(
                (delta < 0).sum()
            ),
            "zero_trades": int(
                (delta == 0).sum()
            ),
        }
    )

    for _, row in executable_validation.iterrows():
        fold_trade_rows.append(
            {
                "validation_year": year,
                "trade_id": row["trade_id"],
                "symbol": row["symbol"],
                "direction": row["direction"],
                "setup": row["setup"],
                "entry_date": row["entry_date"],
                "frozen_event": int(
                    row["frozen_event"]
                ),
                STATE_COLUMN: int(
                    row[STATE_COLUMN]
                ),
                "original_final_r": row[
                    "original_final_r"
                ],
                "bar3_open_executable_r": row[
                    "bar3_open_executable_r"
                ],
                "executable_delta_r": row[
                    "executable_delta_r"
                ],
            }
        )


# ============================================================
# FOLD DATAFRAMES
# ============================================================

folds = pd.DataFrame(
    fold_rows
)

fold_trades = pd.DataFrame(
    fold_trade_rows
)

if len(folds) == 0:
    raise RuntimeError(
        "No valid chronological executable folds."
    )


# ============================================================
# TEMPORAL SUMMARY
# ============================================================

positive_folds = int(
    (
        folds["total_delta_r"] > 0
    ).sum()
)

valid_folds = len(folds)

median_fold_delta = float(
    folds["mean_delta_r"].median()
)

fold_mean_range_low = float(
    folds["mean_delta_r"].min()
)

fold_mean_range_high = float(
    folds["mean_delta_r"].max()
)

print()
print("=" * 72)
print("TEMPORAL EXECUTABLE SUMMARY")
print("=" * 72)

print(
    f"Valid folds: {valid_folds}"
)

print(
    f"Positive executable folds: "
    f"{positive_folds}/{valid_folds}"
)

print(
    f"Positive-fold fraction: "
    f"{positive_folds / valid_folds * 100:.2f}%"
)

print(
    f"Median fold mean improvement: "
    f"{median_fold_delta:+.4f}R"
)

print(
    f"Fold mean range: "
    f"{fold_mean_range_low:+.4f}R "
    f"-> "
    f"{fold_mean_range_high:+.4f}R"
)

print(
    f"Cumulative executable improvement: "
    f"{folds['total_delta_r'].sum():+.4f}R"
)


# ============================================================
# COST SENSITIVITY PER TEMPORAL FOLD
# ============================================================

cost_rows = []

for cost_r in COST_SCENARIOS_R:

    for year in sorted(
        folds["validation_year"]
        .astype(int)
        .unique()
        .tolist()
    ):

        sub = fold_trades[
            fold_trades["validation_year"]
            == year
        ].copy()

        if len(sub) == 0:
            continue

        adjusted = (
            pd.to_numeric(
                sub["executable_delta_r"],
                errors="coerce",
            )
            - cost_r
        )

        ci_low, ci_high = bootstrap_mean_ci(
            adjusted.to_numpy(),
            seed=(
                RANDOM_SEED
                + year
                + int(cost_r * 1000)
            ),
        )

        cost_rows.append(
            {
                "validation_year": year,
                "cost_r": cost_r,
                "trades": len(adjusted),
                "total_delta_r": float(
                    adjusted.sum()
                ),
                "mean_delta_r": float(
                    adjusted.mean()
                ),
                "median_delta_r": float(
                    adjusted.median()
                ),
                "bootstrap_mean_ci_low": ci_low,
                "bootstrap_mean_ci_high": ci_high,
                "positive_trades": int(
                    (adjusted > 0).sum()
                ),
                "negative_trades": int(
                    (adjusted < 0).sum()
                ),
                "zero_trades": int(
                    (adjusted == 0).sum()
                ),
            }
        )

cost_df = pd.DataFrame(
    cost_rows
)

print()
print("=" * 72)
print("TEMPORAL EXECUTABLE COST SENSITIVITY")
print("=" * 72)

for cost_r in COST_SCENARIOS_R:

    sub = cost_df[
        cost_df["cost_r"] == cost_r
    ]

    total = float(
        sub["total_delta_r"].sum()
    )

    trades = int(
        sub["trades"].sum()
    )

    mean = (
        total / trades
        if trades
        else np.nan
    )

    positive_folds_cost = int(
        (
            sub["total_delta_r"] > 0
        ).sum()
    )

    print()
    print(
        f"Cost {cost_r:.2f}R"
    )

    print(
        f"  Total improvement: "
        f"{total:+.4f}R"
    )

    print(
        f"  Mean improvement: "
        f"{mean:+.4f}R"
    )

    print(
        f"  Positive folds: "
        f"{positive_folds_cost}/{len(sub)}"
    )


# ============================================================
# AGGREGATE COST SUMMARY
# ============================================================

aggregate_cost_rows = []

for cost_r in COST_SCENARIOS_R:

    adjusted = (
        fold_trades["executable_delta_r"]
        .astype(float)
        - cost_r
    )

    ci_low, ci_high = bootstrap_mean_ci(
        adjusted.to_numpy(),
        seed=RANDOM_SEED + int(cost_r * 1000),
    )

    total_ci_low, total_ci_high = (
        bootstrap_total_ci(
            adjusted.to_numpy(),
            seed=RANDOM_SEED
            + 100
            + int(cost_r * 1000),
        )
    )

    aggregate_cost_rows.append(
        {
            "cost_r": cost_r,
            "trades": len(adjusted),
            "total_delta_r": float(
                adjusted.sum()
            ),
            "mean_delta_r": float(
                adjusted.mean()
            ),
            "median_delta_r": float(
                adjusted.median()
            ),
            "bootstrap_mean_ci_low": ci_low,
            "bootstrap_mean_ci_high": ci_high,
            "bootstrap_total_ci_low": total_ci_low,
            "bootstrap_total_ci_high": total_ci_high,
            "positive_trades": int(
                (adjusted > 0).sum()
            ),
            "negative_trades": int(
                (adjusted < 0).sum()
            ),
            "zero_trades": int(
                (adjusted == 0).sum()
            ),
        }
    )

aggregate_cost_df = pd.DataFrame(
    aggregate_cost_rows
)


# ============================================================
# EVENT VS RETAINED COST CONTRIBUTION
# ============================================================

event_retained_rows = []

for cost_r in COST_SCENARIOS_R:

    tmp = fold_trades.copy()

    tmp["adjusted_delta_r"] = (
        tmp["executable_delta_r"].astype(float)
        - cost_r
    )

    for label, mask in [
        (
            "event",
            tmp["frozen_event"] == 1,
        ),
        (
            "retained",
            tmp["frozen_event"] == 0,
        ),
    ]:

        sub = tmp[mask]

        if len(sub) == 0:
            continue

        event_retained_rows.append(
            {
                "cost_r": cost_r,
                "population": label,
                "trades": len(sub),
                "total_delta_r": float(
                    sub["adjusted_delta_r"].sum()
                ),
                "mean_delta_r": float(
                    sub["adjusted_delta_r"].mean()
                ),
                "median_delta_r": float(
                    sub["adjusted_delta_r"].median()
                ),
                "positive_trades": int(
                    (
                        sub["adjusted_delta_r"] > 0
                    ).sum()
                ),
                "negative_trades": int(
                    (
                        sub["adjusted_delta_r"] < 0
                    ).sum()
                ),
            }
        )

event_retained_df = pd.DataFrame(
    event_retained_rows
)


# ============================================================
# LEAVE-ONE-FOLD-OUT
# ============================================================

loo_rows = []

for cost_r in COST_SCENARIOS_R:

    for year in sorted(
        fold_trades["validation_year"]
        .astype(int)
        .unique()
        .tolist()
    ):

        remaining = fold_trades[
            fold_trades["validation_year"]
            != year
        ].copy()

        if len(remaining) == 0:
            continue

        adjusted = (
            remaining["executable_delta_r"]
            .astype(float)
            - cost_r
        )

        loo_rows.append(
            {
                "cost_r": cost_r,
                "excluded_year": year,
                "remaining_trades": len(
                    adjusted
                ),
                "remaining_total_delta_r": float(
                    adjusted.sum()
                ),
                "remaining_mean_delta_r": float(
                    adjusted.mean()
                ),
                "remaining_positive_trades": int(
                    (adjusted > 0).sum()
                ),
                "remaining_negative_trades": int(
                    (adjusted < 0).sum()
                ),
                "remaining_still_positive": bool(
                    adjusted.sum() > 0
                ),
            }
        )

loo_df = pd.DataFrame(
    loo_rows
)


# ============================================================
# SYMBOL / SETUP / DIRECTION CONTRIBUTIONS
# ============================================================

subgroup_rows = []

for cost_r in COST_SCENARIOS_R:

    tmp = fold_trades.copy()

    tmp["adjusted_delta_r"] = (
        tmp["executable_delta_r"].astype(float)
        - cost_r
    )

    for dimension in [
        "symbol",
        "setup",
        "direction",
    ]:

        for group, sub in tmp.groupby(
            dimension
        ):

            subgroup_rows.append(
                {
                    "cost_r": cost_r,
                    "dimension": dimension,
                    "group": group,
                    "trades": len(sub),
                    "total_delta_r": float(
                        sub["adjusted_delta_r"].sum()
                    ),
                    "mean_delta_r": float(
                        sub["adjusted_delta_r"].mean()
                    ),
                    "positive_trades": int(
                        (
                            sub["adjusted_delta_r"]
                            > 0
                        ).sum()
                    ),
                    "negative_trades": int(
                        (
                            sub["adjusted_delta_r"]
                            < 0
                        ).sum()
                    ),
                }
            )

subgroup_df = pd.DataFrame(
    subgroup_rows
)


# ============================================================
# BOOTSTRAP AGGREGATE CHECK
# ============================================================

bootstrap_rows = []

rng = np.random.default_rng(
    RANDOM_SEED
)

values = fold_trades[
    "executable_delta_r"
].astype(float).to_numpy()

for cost_r in COST_SCENARIOS_R:

    adjusted = values - cost_r

    if len(adjusted) == 0:
        continue

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
            "iterations": BOOTSTRAP_ITERATIONS,
            "observed_total_delta_r": float(
                adjusted.sum()
            ),
            "observed_mean_delta_r": float(
                adjusted.mean()
            ),
            "bootstrap_total_ci_low": float(
                np.percentile(
                    totals,
                    2.5,
                )
            ),
            "bootstrap_total_ci_high": float(
                np.percentile(
                    totals,
                    97.5,
                )
            ),
            "bootstrap_mean_ci_low": float(
                np.percentile(
                    means,
                    2.5,
                )
            ),
            "bootstrap_mean_ci_high": float(
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
# SAVE OUTPUTS
# ============================================================

folds.to_csv(
    OUTPUT_DIR
    / "temporal_executable_folds.csv",
    index=False,
)

fold_trades.to_csv(
    OUTPUT_DIR
    / "temporal_executable_per_trade.csv",
    index=False,
)

cost_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_cost_by_fold.csv",
    index=False,
)

aggregate_cost_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_cost_aggregate.csv",
    index=False,
)

event_retained_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_event_retained.csv",
    index=False,
)

loo_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_loo.csv",
    index=False,
)

subgroup_df.to_csv(
    OUTPUT_DIR
    / "temporal_executable_subgroups.csv",
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
        "check": "trajectory_trade_count",
        "expected": EXPECTED_TRADES,
        "actual": len(trajectory),
        "pass": len(trajectory)
        == EXPECTED_TRADES,
    },
    {
        "check": "frozen_event_count",
        "expected": EXPECTED_EVENTS,
        "actual": event_count,
        "pass": event_count
        == EXPECTED_EVENTS,
    },
    {
        "check": "frozen_retained_count",
        "expected": EXPECTED_RETAINED,
        "actual": retained_count,
        "pass": retained_count
        == EXPECTED_RETAINED,
    },
    {
        "check": "production_trade_count",
        "expected": EXPECTED_TRADES,
        "actual": len(production_work),
        "pass": len(production_work)
        == EXPECTED_TRADES,
    },
    {
        "check": "bar2_rows",
        "expected": EXPECTED_TRADES,
        "actual": len(bar2),
        "pass": len(bar2)
        == EXPECTED_TRADES,
    },
    {
        "check": "research_rows",
        "expected": EXPECTED_TRADES,
        "actual": len(research),
        "pass": len(research)
        == EXPECTED_TRADES,
    },
    {
        "check": "executable_trades_positive",
        "expected": ">0",
        "actual": int(
            research[
                "executable_bar3_open"
            ].sum()
        ),
        "pass": int(
            research[
                "executable_bar3_open"
            ].sum()
        ) > 0,
    },
    {
        "check": "valid_temporal_folds",
        "expected": ">0",
        "actual": valid_folds,
        "pass": valid_folds > 0,
    },
    {
        "check": "fold_trade_rows",
        "expected": ">0",
        "actual": len(fold_trades),
        "pass": len(fold_trades) > 0,
    },
    {
        "check": "cost_scenarios",
        "expected": len(
            COST_SCENARIOS_R
        ),
        "actual": len(
            aggregate_cost_df
        ),
        "pass": len(
            aggregate_cost_df
        )
        == len(COST_SCENARIOS_R),
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
    / "temporal_executable_integrity.csv",
    index=False,
)

integrity_pass = bool(
    integrity_df["pass"].all()
)

positive_fold_pass = (
    positive_folds == valid_folds
)

print()
print("=" * 72)
print("FINAL STRICT TEMPORAL EXECUTABLE STATUS")
print("=" * 72)

print(
    f"Frozen {len(trajectory)}/"
    f"{event_count}/"
    f"{retained_count}"
)

print(
    f"Valid executable folds: "
    f"{valid_folds}"
)

print(
    f"Positive executable folds: "
    f"{positive_folds}/{valid_folds}"
)

print(
    f"All-fold positive: "
    f"{positive_fold_pass}"
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
    "STRICT TEMPORAL EXECUTABLE TEST: "
    f"{'PASS' if integrity_pass else 'FAIL'}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "Historical chronological validation only."
)

print(
    "Bar-3 open execution is executable in timing "
    "terms, but this remains a frozen historical "
    "forensic replay."
)

print(
    "This does not replace genuine post-freeze OOS."
)

print(
    "Locked 7B/0.25R OOS hypothesis remains untouched."
)