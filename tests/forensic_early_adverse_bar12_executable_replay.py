"""
TRADE SENSE-AI
BAR 1→2 EXECUTABLE REPLAY FORENSIC ANALYSIS

Purpose
-------
Test whether the Bar-2 early-adverse state can be converted into an
executable historical intervention without hindsight.

Frozen research population:
    106 total trades
     60 frozen early-adverse events
     46 retained trades

Frozen state:
    bar2_favorable_failed_to_improve

Decision timing:
    - Observe Bar 2 close.
    - State is therefore known only after Bar 2 completes.
    - Conservative executable implementation uses Bar 3 OPEN.
    - No future-bar information is used to form the state.

Comparisons:
    1. Original production outcome.
    2. Bar-2 close counterfactual (reference only).
    3. Bar-3 open executable counterfactual.
    4. Execution-cost sensitivity.

Research only:
    - Production strategy is NOT modified.
    - OOS data is NOT modified.
    - No deployment rule is promoted.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

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
    "tests/output/research/early_adverse_bar12_executable_replay"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

STATE_COLUMN = "bar2_favorable_failed_to_improve"
GROSS_CF_COLUMN = (
    "bar2_favorable_failed_to_improve_counterfactual_delta_r"
)

# Fixed cost stress levels.
COST_SCENARIOS_R = [0.00, 0.05, 0.10, 0.15, 0.20]

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42


# ============================================================================
# HELPERS
# ============================================================================

def first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_symbol(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    if not value.endswith(".NS"):
        value = value + ".NS"

    return value


def normalize_date(value):
    return pd.to_datetime(value, errors="coerce").dt.strftime("%Y-%m-%d")


def directional_r(direction, exit_price, entry_price, risk_per_unit):
    """
    R-multiple from reconstructed entry to an executable exit price.
    """

    if (
        pd.isna(exit_price)
        or pd.isna(entry_price)
        or pd.isna(risk_per_unit)
        or risk_per_unit == 0
    ):
        return np.nan

    direction = str(direction).upper()

    if direction == "LONG":
        pnl = exit_price - entry_price
    elif direction == "SHORT":
        pnl = entry_price - exit_price
    else:
        return np.nan

    return pnl / risk_per_unit


def bootstrap_mean_ci(values, iterations=10000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    boot_means = np.empty(iterations)

    for i in range(iterations):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )
        boot_means[i] = np.mean(sample)

    return (
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def safe_bool(value):
    if pd.isna(value):
        return False

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
        }

    return bool(value)


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================================
# LOAD INPUTS
# ============================================================================

print("=" * 72)
print("TRADE SENSE-AI — BAR 1→2 EXECUTABLE REPLAY FORENSIC ANALYSIS")
print("=" * 72)

trajectory = pd.read_csv(TRAJECTORY_FILE)
path = pd.read_csv(PATH_FILE)
production = pd.read_csv(PRODUCTION_FILE)

print()
print(f"Trajectory rows: {len(trajectory)}")
print(f"Trajectory columns: {len(trajectory.columns)}")
print(f"Path rows:       {len(path)}")
print(f"Production rows: {len(production)}")


# ============================================================================
# RESOLVE PRODUCTION COLUMNS
# ============================================================================

production_symbol_col = first_existing(
    production,
    [
        "_symbol",
        "symbol",
        "Symbol",
    ],
)

production_direction_col = first_existing(
    production,
    [
        "direction",
        "Direction",
    ],
)

production_date_col = first_existing(
    production,
    [
        "entry_date",
        "Entry Date",
        "date",
    ],
)

production_final_r_col = first_existing(
    production,
    [
        "r_multiple",
        "final_r",
        "final_R",
        "original_final_r",
    ],
)

if production_symbol_col is None:
    raise RuntimeError("Could not resolve production symbol column.")

if production_direction_col is None:
    raise RuntimeError("Could not resolve production direction column.")

if production_date_col is None:
    raise RuntimeError("Could not resolve production entry-date column.")

if production_final_r_col is None:
    raise RuntimeError("Could not resolve production final-R column.")


print_section("RESOLVED PRODUCTION COLUMNS")

print(f"Symbol      : {production_symbol_col}")
print(f"Direction   : {production_direction_col}")
print(f"Entry Date  : {production_date_col}")
print(f"Final R     : {production_final_r_col}")


# ============================================================================
# BUILD NORMALIZED PRODUCTION TABLE
# ============================================================================

production_work = production.copy()

production_work["symbol"] = production_work[
    production_symbol_col
].map(normalize_symbol)

production_work["direction"] = (
    production_work[production_direction_col]
    .astype(str)
    .str.upper()
    .str.strip()
)

production_work["entry_date"] = normalize_date(
    production_work[production_date_col]
)

production_work["original_final_r"] = pd.to_numeric(
    production_work[production_final_r_col],
    errors="coerce",
)

production_work["merge_key"] = (
    production_work["symbol"].astype(str)
    + "|"
    + production_work["direction"].astype(str)
    + "|"
    + production_work["entry_date"].astype(str)
)


# ============================================================================
# NORMALIZE TRAJECTORY
# ============================================================================

trajectory_work = trajectory.copy()

trajectory_work["symbol"] = trajectory_work[
    "symbol"
].map(normalize_symbol)

trajectory_work["direction"] = (
    trajectory_work["direction"]
    .astype(str)
    .str.upper()
    .str.strip()
)

trajectory_work["entry_date"] = normalize_date(
    trajectory_work["entry_date"]
)

trajectory_work["merge_key"] = (
    trajectory_work["symbol"].astype(str)
    + "|"
    + trajectory_work["direction"].astype(str)
    + "|"
    + trajectory_work["entry_date"].astype(str)
)


# ============================================================================
# FROZEN POPULATION CHECK
# ============================================================================

trajectory_work["frozen_event"] = trajectory_work[
    "frozen_event"
].astype(int)

trade_count = len(trajectory_work)
event_count = int(
    trajectory_work["frozen_event"].sum()
)
retained_count = trade_count - event_count

print_section("FROZEN POPULATION")

print(f"Trades:   {trade_count}")
print(f"Events:   {event_count}")
print(f"Retained: {retained_count}")

if trade_count != EXPECTED_TRADES:
    raise RuntimeError(
        f"Expected {EXPECTED_TRADES} trades, got {trade_count}"
    )

if event_count != EXPECTED_EVENTS:
    raise RuntimeError(
        f"Expected {EXPECTED_EVENTS} events, got {event_count}"
    )

if retained_count != EXPECTED_RETAINED:
    raise RuntimeError(
        f"Expected {EXPECTED_RETAINED} retained, got {retained_count}"
    )


# ============================================================================
# EXTRACT BAR-2 AND BAR-3 DATA
# ============================================================================

path_work = path.copy()

path_work["symbol"] = path_work[
    "symbol"
].map(normalize_symbol)

path_work["direction"] = (
    path_work["direction"]
    .astype(str)
    .str.upper()
    .str.strip()
)

path_work["entry_date"] = normalize_date(
    path_work["entry_date"]
)

path_work["bar_number"] = pd.to_numeric(
    path_work["bar_number"],
    errors="coerce",
)

path_work["merge_key"] = (
    path_work["symbol"].astype(str)
    + "|"
    + path_work["direction"].astype(str)
    + "|"
    + path_work["entry_date"].astype(str)
)


required_path_columns = [
    "merge_key",
    "bar_number",
    "open",
    "close",
    "entry_price_reconstructed",
    "risk_per_unit",
]

missing_path = [
    col
    for col in required_path_columns
    if col not in path_work.columns
]

if missing_path:
    raise RuntimeError(
        f"Missing required path columns: {missing_path}"
    )


bar2 = path_work[
    path_work["bar_number"] == 2
].copy()

bar3 = path_work[
    path_work["bar_number"] == 3
].copy()


# ============================================================================
# DUPLICATE CHECK
# ============================================================================

bar2_duplicates = int(
    bar2["merge_key"].duplicated().sum()
)

bar3_duplicates = int(
    bar3["merge_key"].duplicated().sum()
)

if bar2_duplicates != 0:
    raise RuntimeError(
        f"Duplicate Bar-2 rows detected: {bar2_duplicates}"
    )

if bar3_duplicates != 0:
    raise RuntimeError(
        f"Duplicate Bar-3 rows detected: {bar3_duplicates}"
    )


# ============================================================================
# SELECT REQUIRED BAR-2 DATA
# ============================================================================

bar2_required = bar2[
    [
        "merge_key",
        "bar_number",
        "open",
        "close",
        "entry_price_reconstructed",
        "risk_per_unit",
        "production_bars_in_trade",
        "production_observed",
        "production_observation_status",
        "bar5_reached_in_production",
    ]
].copy()

bar2_required = bar2_required.rename(
    columns={
        "open": "bar2_open",
        "close": "bar2_close",
        "entry_price_reconstructed":
            "entry_price_reconstructed_bar2",
        "risk_per_unit":
            "risk_per_unit_bar2",
    }
)


# ============================================================================
# SELECT REQUIRED BAR-3 DATA
# ============================================================================

bar3_required = bar3[
    [
        "merge_key",
        "bar_number",
        "open",
        "close",
    ]
].copy()

bar3_required = bar3_required.rename(
    columns={
        "open": "bar3_open",
        "close": "bar3_close",
    }
)


# ============================================================================
# MERGE RESEARCH STATE + PATH
# ============================================================================

research = trajectory_work.merge(
    production_work[
        [
            "merge_key",
            "original_final_r",
        ]
    ],
    on="merge_key",
    how="left",
    validate="one_to_one",
)

research = research.merge(
    bar2_required[
        [
            "merge_key",
            "bar2_open",
            "bar2_close",
            "entry_price_reconstructed_bar2",
            "risk_per_unit_bar2",
            "production_bars_in_trade",
            "production_observed",
            "production_observation_status",
            "bar5_reached_in_production",
        ]
    ],
    on="merge_key",
    how="left",
    validate="one_to_one",
)

research = research.merge(
    bar3_required[
        [
            "merge_key",
            "bar3_open",
            "bar3_close",
        ]
    ],
    on="merge_key",
    how="left",
    validate="one_to_one",
)


# ============================================================================
# STATE RESOLUTION
# ============================================================================

if STATE_COLUMN not in research.columns:
    raise RuntimeError(
        f"Missing frozen research state: {STATE_COLUMN}"
    )

research["state_triggered"] = research[
    STATE_COLUMN
].map(safe_bool)


# ============================================================================
# EXECUTION ELIGIBILITY
# ============================================================================

"""
Critical timing rule:

The state is evaluated from Bar-2 information.

Therefore:
    Bar 2 close = decision observation point
    Bar 3 open  = first conservative executable price

A trade is considered executable only when:
    - Bar 2 exists
    - Bar 3 exists
    - reconstructed entry/risk exist
    - the trade's production path reached at least Bar 2

We separately report whether the production trade actually survived
through the Bar-2 observation point using production_bars_in_trade.

This does NOT alter the frozen label.
"""

research["bar2_available"] = (
    research["bar2_close"].notna()
)

research["bar3_open_available"] = (
    research["bar3_open"].notna()
)

research["entry_price_available"] = (
    research["entry_price_reconstructed_bar2"].notna()
)

research["risk_available"] = (
    research["risk_per_unit_bar2"].notna()
)

required_production_observation_columns = [
    "production_bars_in_trade",
    "production_observed",
    "production_observation_status",
]

missing_production_observation = [
    col
    for col in required_production_observation_columns
    if col not in research.columns
]

if missing_production_observation:
    raise RuntimeError(
        "Missing production observation columns after merge: "
        f"{missing_production_observation}"
    )

research["production_bars_in_trade"] = pd.to_numeric(
    research["production_bars_in_trade"],
    errors="coerce",
)

research["production_observed"] = pd.to_numeric(
    research["production_observed"],
    errors="coerce",
)

research["production_observation_status"] = (
    research["production_observation_status"]
    .astype(str)
    .str.strip()
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

research["executable_bar3_open"] = (
    research["state_triggered"]
    & research["bar2_available"]
    & research["bar3_open_available"]
    & research["entry_price_available"]
    & research["risk_available"]
    & research["survived_to_bar2_observation"]
)


# ============================================================================
# RECONSTRUCT BAR-2 CLOSE COUNTERFACTUAL
# ============================================================================

research["bar2_close_r"] = research.apply(
    lambda row: directional_r(
        row["direction"],
        row["bar2_close"],
        row["entry_price_reconstructed_bar2"],
        row["risk_per_unit_bar2"],
    ),
    axis=1,
)


# ============================================================================
# RECONSTRUCT BAR-3 OPEN EXECUTABLE COUNTERFACTUAL
# ============================================================================

research["bar3_open_r"] = research.apply(
    lambda row: directional_r(
        row["direction"],
        row["bar3_open"],
        row["entry_price_reconstructed_bar2"],
        row["risk_per_unit_bar2"],
    ),
    axis=1,
)


# ============================================================================
# INCREMENTAL DELTAS
# ============================================================================

research["bar2_close_delta_r"] = np.where(
    research["state_triggered"],
    research["bar2_close_r"]
    - research["original_final_r"],
    np.nan,
)

research["bar3_open_delta_r"] = np.where(
    research["executable_bar3_open"],
    research["bar3_open_r"]
    - research["original_final_r"],
    np.nan,
)

research["execution_slippage_r"] = (
    research["bar3_open_r"]
    - research["bar2_close_r"]
)


# ============================================================================
# COST-ADJUSTED EXECUTABLE RESULTS
# ============================================================================

cost_rows = []

for cost_r in COST_SCENARIOS_R:

    temp = research.copy()

    temp["cost_adjusted_delta_r"] = np.where(
        temp["executable_bar3_open"],
        temp["bar3_open_delta_r"] - cost_r,
        np.nan,
    )

    triggered = temp[
        temp["executable_bar3_open"]
    ].copy()

    values = triggered[
        "cost_adjusted_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) > 0:
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=RANDOM_SEED,
        )

        total_delta = float(np.sum(values))
        mean_delta = float(np.mean(values))
        median_delta = float(np.median(values))
        positive = int(np.sum(values > 0))
        negative = int(np.sum(values < 0))
        zero = int(np.sum(values == 0))
    else:
        ci_low = np.nan
        ci_high = np.nan
        total_delta = np.nan
        mean_delta = np.nan
        median_delta = np.nan
        positive = 0
        negative = 0
        zero = 0

    cost_rows.append(
        {
            "cost_r": cost_r,
            "n": len(values),
            "total_delta_r": total_delta,
            "mean_delta_r": mean_delta,
            "median_delta_r": median_delta,
            "bootstrap_mean_ci_low": ci_low,
            "bootstrap_mean_ci_high": ci_high,
            "positive_trades": positive,
            "negative_trades": negative,
            "zero_trades": zero,
        }
    )

cost_results = pd.DataFrame(cost_rows)


# ============================================================================
# EXECUTION ELIGIBILITY SUMMARY
# ============================================================================

print_section("EXECUTION ELIGIBILITY")

state_count = int(
    research["state_triggered"].sum()
)

bar2_available_count = int(
    (
        research["state_triggered"]
        & research["bar2_available"]
    ).sum()
)

bar3_available_count = int(
    (
        research["state_triggered"]
        & research["bar3_open_available"]
    ).sum()
)

survived_bar2_count = int(
    (
        research["state_triggered"]
        & research["survived_to_bar2_observation"]
    ).sum()
)

executable_count = int(
    research["executable_bar3_open"].sum()
)

print(f"State-triggered trades:          {state_count}")
print(f"Bar-2 observation available:     {bar2_available_count}")
print(f"Bar-3 open available:             {bar3_available_count}")
print(f"Survived to Bar-2 observation:    {survived_bar2_count}")
print(f"Executable at Bar-3 open:         {executable_count}")


# ============================================================================
# MAIN EXECUTABLE RESULT
# ============================================================================

print_section("EXECUTABLE REPLAY RESULTS")

print()
print("Bar-2 close counterfactual:")
print(
    research.loc[
        research["state_triggered"],
        "bar2_close_delta_r",
    ]
    .describe()
)

print()
print("Bar-3 open executable counterfactual:")
print(
    research.loc[
        research["executable_bar3_open"],
        "bar3_open_delta_r",
    ]
    .describe()
)


# ============================================================================
# AGGREGATE EXECUTABLE RESULT
# ============================================================================

executable = research[
    research["executable_bar3_open"]
].copy()

if len(executable) > 0:

    executable_values = executable[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    executable_ci_low, executable_ci_high = (
        bootstrap_mean_ci(
            executable_values,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=RANDOM_SEED,
        )
    )

    executable_summary = pd.DataFrame(
        [
            {
                "n": len(executable_values),
                "total_delta_r":
                    float(np.sum(executable_values)),
                "mean_delta_r":
                    float(np.mean(executable_values)),
                "median_delta_r":
                    float(np.median(executable_values)),
                "bootstrap_mean_ci_low":
                    executable_ci_low,
                "bootstrap_mean_ci_high":
                    executable_ci_high,
                "positive_trades":
                    int(np.sum(executable_values > 0)),
                "negative_trades":
                    int(np.sum(executable_values < 0)),
                "zero_trades":
                    int(np.sum(executable_values == 0)),
            }
        ]
    )

else:

    executable_summary = pd.DataFrame(
        [
            {
                "n": 0,
                "total_delta_r": np.nan,
                "mean_delta_r": np.nan,
                "median_delta_r": np.nan,
                "bootstrap_mean_ci_low": np.nan,
                "bootstrap_mean_ci_high": np.nan,
                "positive_trades": 0,
                "negative_trades": 0,
                "zero_trades": 0,
            }
        ]
    )


print()
print(executable_summary.to_string(index=False))


# ============================================================================
# EXECUTION SLIPPAGE
# ============================================================================

print_section("BAR-2 CLOSE → BAR-3 OPEN EXECUTION SLIPPAGE")

slippage_values = executable[
    "execution_slippage_r"
].dropna()

if len(slippage_values) > 0:

    print(
        f"n      : {len(slippage_values)}"
    )
    print(
        f"median : {slippage_values.median():.6f}R"
    )
    print(
        f"mean   : {slippage_values.mean():.6f}R"
    )
    print(
        f"min    : {slippage_values.min():.6f}R"
    )
    print(
        f"max    : {slippage_values.max():.6f}R"
    )


# ============================================================================
# COST SENSITIVITY
# ============================================================================

print_section("EXECUTABLE COST SENSITIVITY")

print(
    cost_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================================
# EVENT VS RETAINED
# ============================================================================

print_section("EXECUTABLE EVENT VS RETAINED")

event_retained_rows = []

for population_name, mask in [
    (
        "event",
        research["frozen_event"] == 1,
    ),
    (
        "retained",
        research["frozen_event"] == 0,
    ),
]:

    subset = executable[mask.loc[executable.index]]

    values = subset[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    event_retained_rows.append(
        {
            "population": population_name,
            "n": len(values),
            "total_delta_r":
                float(np.sum(values)),
            "mean_delta_r":
                float(np.mean(values)),
            "median_delta_r":
                float(np.median(values)),
            "positive":
                int(np.sum(values > 0)),
            "negative":
                int(np.sum(values < 0)),
        }
    )

event_retained = pd.DataFrame(
    event_retained_rows
)

if len(event_retained) > 0:
    print(
        event_retained.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# YEAR ROBUSTNESS
# ============================================================================

print_section("EXECUTABLE YEAR CONTRIBUTION")

executable["year"] = pd.to_datetime(
    executable["entry_date"],
    errors="coerce",
).dt.year

year_rows = []

for year, group in executable.groupby(
    "year",
    dropna=True,
):

    values = group[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    year_rows.append(
        {
            "year": int(year),
            "n": len(values),
            "total_delta_r":
                float(np.sum(values)),
            "mean_delta_r":
                float(np.mean(values)),
            "positive":
                int(np.sum(values > 0)),
            "negative":
                int(np.sum(values < 0)),
        }
    )

year_results = pd.DataFrame(year_rows)

if len(year_results) > 0:
    print(
        year_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# SYMBOL ROBUSTNESS
# ============================================================================

print_section("EXECUTABLE SYMBOL CONTRIBUTION")

symbol_rows = []

for symbol, group in executable.groupby(
    "symbol",
    dropna=True,
):

    values = group[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    symbol_rows.append(
        {
            "symbol": symbol,
            "n": len(values),
            "total_delta_r":
                float(np.sum(values)),
            "mean_delta_r":
                float(np.mean(values)),
            "positive":
                int(np.sum(values > 0)),
            "negative":
                int(np.sum(values < 0)),
        }
    )

symbol_results = pd.DataFrame(symbol_rows)

if len(symbol_results) > 0:
    print(
        symbol_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# SETUP ROBUSTNESS
# ============================================================================

print_section("EXECUTABLE SETUP CONTRIBUTION")

setup_rows = []

for setup, group in executable.groupby(
    "setup",
    dropna=True,
):

    values = group[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    setup_rows.append(
        {
            "setup": setup,
            "n": len(values),
            "total_delta_r":
                float(np.sum(values)),
            "mean_delta_r":
                float(np.mean(values)),
            "positive":
                int(np.sum(values > 0)),
            "negative":
                int(np.sum(values < 0)),
        }
    )

setup_results = pd.DataFrame(setup_rows)

if len(setup_results) > 0:
    print(
        setup_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# DIRECTION ROBUSTNESS
# ============================================================================

print_section("EXECUTABLE DIRECTION CONTRIBUTION")

direction_rows = []

for direction, group in executable.groupby(
    "direction",
    dropna=True,
):

    values = group[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    direction_rows.append(
        {
            "direction": direction,
            "n": len(values),
            "total_delta_r":
                float(np.sum(values)),
            "mean_delta_r":
                float(np.mean(values)),
            "positive":
                int(np.sum(values > 0)),
            "negative":
                int(np.sum(values < 0)),
        }
    )

direction_results = pd.DataFrame(
    direction_rows
)

if len(direction_results) > 0:
    print(
        direction_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# LEAVE-ONE-YEAR-OUT
# ============================================================================

print_section("EXECUTABLE LEAVE-ONE-YEAR-OUT")

loo_year_rows = []

all_years = sorted(
    executable["year"].dropna().unique()
)

for year in all_years:

    remaining = executable[
        executable["year"] != year
    ]

    values = remaining[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    loo_year_rows.append(
        {
            "removed_year": int(year),
            "remaining_n": len(values),
            "remaining_total_delta_r":
                float(np.sum(values)),
            "remaining_mean_delta_r":
                float(np.mean(values)),
            "remaining_positive":
                int(np.sum(values > 0)),
            "remaining_negative":
                int(np.sum(values < 0)),
            "remaining_still_positive":
                bool(np.sum(values) > 0),
        }
    )

loo_year_results = pd.DataFrame(
    loo_year_rows
)

if len(loo_year_results) > 0:
    print(
        loo_year_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# LEAVE-ONE-SYMBOL-OUT
# ============================================================================

print_section("EXECUTABLE LEAVE-ONE-SYMBOL-OUT")

loo_symbol_rows = []

all_symbols = sorted(
    executable["symbol"].dropna().unique()
)

for symbol in all_symbols:

    remaining = executable[
        executable["symbol"] != symbol
    ]

    values = remaining[
        "bar3_open_delta_r"
    ].dropna().to_numpy(dtype=float)

    if len(values) == 0:
        continue

    loo_symbol_rows.append(
        {
            "removed_symbol": symbol,
            "remaining_n": len(values),
            "remaining_total_delta_r":
                float(np.sum(values)),
            "remaining_mean_delta_r":
                float(np.mean(values)),
            "remaining_positive":
                int(np.sum(values > 0)),
            "remaining_negative":
                int(np.sum(values < 0)),
            "remaining_still_positive":
                bool(np.sum(values) > 0),
        }
    )

loo_symbol_results = pd.DataFrame(
    loo_symbol_rows
)

if len(loo_symbol_results) > 0:
    print(
        loo_symbol_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================================
# PER-TRADE OUTPUT
# ============================================================================

print_section("PER-TRADE EXECUTION REPLAY")

per_trade_columns = [
    "merge_key",
    "symbol",
    "direction",
    "entry_date",
    "setup",
    "frozen_event",
    "state_triggered",
    "production_bars_in_trade",
    "survived_to_bar2_observation",
    "executable_bar3_open",
    "original_final_r",
    "bar2_close_r",
    "bar3_open_r",
    "bar2_close_delta_r",
    "bar3_open_delta_r",
    "execution_slippage_r",
]

per_trade_columns = [
    col
    for col in per_trade_columns
    if col in research.columns
]

per_trade = research[
    per_trade_columns
].copy()


# ============================================================================
# INTEGRITY
# ============================================================================

print_section("FINAL INTEGRITY")

integrity_rows = []

def add_check(name, expected, actual):
    integrity_rows.append(
        {
            "check": name,
            "expected": expected,
            "actual": actual,
            "pass": bool(expected == actual),
        }
    )


add_check(
    "trade_count",
    EXPECTED_TRADES,
    trade_count,
)

add_check(
    "event_count",
    EXPECTED_EVENTS,
    event_count,
)

add_check(
    "retained_count",
    EXPECTED_RETAINED,
    retained_count,
)

add_check(
    "trajectory_unique_keys",
    EXPECTED_TRADES,
    trajectory_work["merge_key"].nunique(),
)

add_check(
    "production_unique_keys",
    EXPECTED_TRADES,
    production_work["merge_key"].nunique(),
)

add_check(
    "bar2_duplicate_keys",
    0,
    bar2_duplicates,
)

add_check(
    "bar3_duplicate_keys",
    0,
    bar3_duplicates,
)

add_check(
    "state_trigger_count",
    65,
    state_count,
)

add_check(
    "executable_rows_nonzero",
    True,
    executable_count > 0,
)

add_check(
    "executable_delta_complete",
    True,
    bool(
        executable[
            "bar3_open_delta_r"
        ].notna().all()
    )
    if len(executable) > 0
    else False,
)

add_check(
    "cost_scenario_count",
    len(COST_SCENARIOS_R),
    len(cost_results),
)

add_check(
    "per_trade_rows",
    EXPECTED_TRADES,
    len(per_trade),
)

integrity = pd.DataFrame(
    integrity_rows
)

print(
    integrity.to_string(
        index=False
    )
)

all_pass = bool(
    integrity["pass"].all()
)

print()

if all_pass:
    print("=" * 72)
    print("PASS — EXECUTABLE REPLAY FORENSIC ANALYSIS COMPLETE")
    print("=" * 72)
else:
    print("=" * 72)
    print("FAIL — EXECUTABLE REPLAY INTEGRITY CHECK FAILED")
    print("=" * 72)


# ============================================================================
# WRITE OUTPUTS
# ============================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

per_trade.to_csv(
    OUTPUT_DIR
    / "executable_replay_per_trade.csv",
    index=False,
)

executable_summary.to_csv(
    OUTPUT_DIR
    / "executable_replay_summary.csv",
    index=False,
)

cost_results.to_csv(
    OUTPUT_DIR
    / "executable_replay_cost_sensitivity.csv",
    index=False,
)

if len(event_retained) > 0:
    event_retained.to_csv(
        OUTPUT_DIR
        / "executable_replay_event_retained.csv",
        index=False,
    )

if len(year_results) > 0:
    year_results.to_csv(
        OUTPUT_DIR
        / "executable_replay_year.csv",
        index=False,
    )

if len(symbol_results) > 0:
    symbol_results.to_csv(
        OUTPUT_DIR
        / "executable_replay_symbol.csv",
        index=False,
    )

if len(setup_results) > 0:
    setup_results.to_csv(
        OUTPUT_DIR
        / "executable_replay_setup.csv",
        index=False,
    )

if len(direction_results) > 0:
    direction_results.to_csv(
        OUTPUT_DIR
        / "executable_replay_direction.csv",
        index=False,
    )

loo_year_results.to_csv(
    OUTPUT_DIR
    / "executable_replay_loo_year.csv",
    index=False,
)

loo_symbol_results.to_csv(
    OUTPUT_DIR
    / "executable_replay_loo_symbol.csv",
    index=False,
)

integrity.to_csv(
    OUTPUT_DIR
    / "executable_replay_integrity.csv",
    index=False,
)

print()
print("Output directory:")
print(OUTPUT_DIR)