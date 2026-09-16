"""
TradeSense-AI
BAR 1->2 COUNTERFACTUAL MANAGEMENT / ECONOMIC-VALUE FORENSIC ANALYSIS

Research-only.

Frozen population
-----------------
106 total trades
60 frozen early-adverse events
46 retained

Purpose
-------
Evaluate fixed, pre-specified Bar-2 states using a hypothetical
Bar-2-close exit.

This is NOT a production strategy modification.

Important timing convention
---------------------------
The Bar-2 state is assumed to become known only after Bar 2 closes.

Therefore:
    state observed at Bar 2 close
    hypothetical exit = Bar 2 close

This is a counterfactual research simulation. It is not claimed to be
directly executable performance because real execution would require
slippage, spread, latency, and order-fill assumptions.

Primary questions
-----------------
1. What is the hypothetical R result if a trade exits at Bar 2 close
   whenever a fixed Bar-2 state is present?

2. How does that compare with the original production R outcome?

3. Does the intervention improve aggregate R?

4. Does it reduce downside / adverse excursion?

5. What opportunity cost occurs on trades that would have recovered?

No:
- threshold optimization
- ML
- production changes
- OOS changes
- new state definitions
"""


from pathlib import Path
import json
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

TRADE_FILE = Path(
    "tests/output/tradesense_trades.csv"
)

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_bar12_counterfactual"
)

EXPECTED_TRADES = 106
EXPECTED_EVENTS = 60
EXPECTED_RETAINED = 46

MAX_BAR = 5

STATE_COLUMNS = [
    "bar2_net_deteriorated",
    "bar2_favorable_failed_to_improve",
    "adverse_up_favorable_not_up_net_down",
]

STATE_LABELS = {
    "bar2_net_deteriorated":
        "Bar-2 net deterioration",

    "bar2_favorable_failed_to_improve":
        "Bar-2 favorable failure",

    "adverse_up_favorable_not_up_net_down":
        "Bar-2 adverse-up + favorable-not-up + net-down",
}


# ============================================================================
# HELPERS
# ============================================================================

def fail(message: str):
    raise RuntimeError(message)


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(
            f"{name} missing required columns: {missing}"
        )


def normalize_symbol(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_direction(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_date(series):
    return (
        pd.to_datetime(
            series,
            errors="coerce"
        )
        .dt.strftime("%Y-%m-%d")
    )


def make_key(df):
    symbol = normalize_symbol(df["symbol"])
    direction = normalize_direction(df["direction"])
    entry_date = normalize_date(df["entry_date"])

    invalid = (
        symbol.isin(["", "NAN", "NONE"])
        | direction.isin(["", "NAN", "NONE"])
        | entry_date.isna()
    )

    if int(invalid.sum()) > 0:
        fail(
            f"Invalid identity rows: {int(invalid.sum())}"
        )

    return (
        symbol
        + "|"
        + direction
        + "|"
        + entry_date
    )


def bootstrap_mean_difference(
    a,
    b,
    n_boot=5000,
    seed=42,
):
    a = pd.to_numeric(
        a,
        errors="coerce"
    ).dropna().to_numpy()

    b = pd.to_numeric(
        b,
        errors="coerce"
    ).dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan

    observed = float(
        np.mean(a) - np.mean(b)
    )

    rng = np.random.default_rng(seed)

    boot = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):
        sample_a = rng.choice(
            a,
            size=len(a),
            replace=True
        )

        sample_b = rng.choice(
            b,
            size=len(b),
            replace=True
        )

        boot[i] = (
            np.mean(sample_a)
            - np.mean(sample_b)
        )

    lo = float(
        np.percentile(boot, 2.5)
    )

    hi = float(
        np.percentile(boot, 97.5)
    )

    return observed, lo, hi


def bootstrap_median_difference(
    a,
    b,
    n_boot=5000,
    seed=42,
):
    a = pd.to_numeric(
        a,
        errors="coerce"
    ).dropna().to_numpy()

    b = pd.to_numeric(
        b,
        errors="coerce"
    ).dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, np.nan

    observed = float(
        np.median(a)
        - np.median(b)
    )

    rng = np.random.default_rng(seed)

    boot = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):
        sample_a = rng.choice(
            a,
            size=len(a),
            replace=True
        )

        sample_b = rng.choice(
            b,
            size=len(b),
            replace=True
        )

        boot[i] = (
            np.median(sample_a)
            - np.median(sample_b)
        )

    lo = float(
        np.percentile(boot, 2.5)
    )

    hi = float(
        np.percentile(boot, 97.5)
    )

    return observed, lo, hi


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def signed_r(x):
    if pd.isna(x):
        return "NA"
    return f"{x:+.4f}R"


# ============================================================================
# LOAD TRAJECTORY
# ============================================================================

def load_trajectory():

    if not TRAJECTORY_FILE.exists():
        fail(
            f"Trajectory file not found: "
            f"{TRAJECTORY_FILE}"
        )

    df = pd.read_csv(
        TRAJECTORY_FILE
    )

    print(
        f"Trajectory input: {TRAJECTORY_FILE}"
    )

    print(
        f"Trajectory rows: {len(df)}, "
        f"columns: {len(df.columns)}"
    )

    require_columns(
        df,
        [
            "symbol",
            "direction",
            "entry_date",
            "setup",
            "frozen_event",
            "bar2_net_deteriorated",
            "bar2_favorable_failed_to_improve",
            "adverse_up_favorable_not_up_net_down",
        ],
        "Trajectory"
    )

    if len(df) != EXPECTED_TRADES:
        fail(
            f"Trajectory rows {len(df)} "
            f"!= {EXPECTED_TRADES}"
        )

    df = df.copy()

    df["_merge_key"] = make_key(df)

    if df["_merge_key"].duplicated().any():
        fail(
            "Trajectory contains duplicate identity keys."
        )

    df["frozen_event"] = (
        df["frozen_event"]
        .astype(int)
        .astype(bool)
    )

    event_count = int(
        df["frozen_event"].sum()
    )

    retained_count = int(
        (~df["frozen_event"]).sum()
    )

    print(
        f"Frozen population: "
        f"{len(df)} / "
        f"{event_count} / "
        f"{retained_count}"
    )

    if event_count != EXPECTED_EVENTS:
        fail(
            f"Frozen events {event_count} "
            f"!= {EXPECTED_EVENTS}"
        )

    if retained_count != EXPECTED_RETAINED:
        fail(
            f"Frozen retained {retained_count} "
            f"!= {EXPECTED_RETAINED}"
        )

    return df


# ============================================================================
# LOAD BAR-2 OHLC
# ============================================================================

def load_bar2(path):

    if not PATH_FILE.exists():
        fail(
            f"Path file not found: {PATH_FILE}"
        )

    df = pd.read_csv(
        PATH_FILE
    )

    print(
        f"Path input: {PATH_FILE}"
    )

    print(
        f"Path rows: {len(df)}, "
        f"columns: {len(df.columns)}"
    )

    require_columns(
        df,
        [
            "symbol",
            "direction",
            "entry_date",
            "bar_number",
            "close",
            "entry_price_reconstructed",
            "risk_per_unit",
            "stored_frozen_event",
        ],
        "Path"
    )

    df = df.copy()

    df["_merge_key"] = make_key(df)

    df["bar_number"] = pd.to_numeric(
        df["bar_number"],
        errors="coerce"
    )

    bar2 = df[
        df["bar_number"] == 2
    ].copy()

    if len(bar2) != EXPECTED_TRADES:
        fail(
            f"Bar-2 rows {len(bar2)} "
            f"!= {EXPECTED_TRADES}"
        )

    if bar2["_merge_key"].duplicated().any():
        fail(
            "Duplicate Bar-2 identity keys."
        )

    return bar2


# ============================================================================
# LOAD PRODUCTION OUTCOMES
# ============================================================================

def load_production():

    if not TRADE_FILE.exists():
        fail(
            f"Production trade file not found: "
            f"{TRADE_FILE}"
        )

    df = pd.read_csv(
        TRADE_FILE
    )

    print(
        f"Production trade rows: {len(df)}, "
        f"columns: {len(df.columns)}"
    )

    require_columns(
        df,
        [
            "_symbol",
            "direction",
            "entry_date",
            "r_multiple",
            "max_favorable_r",
            "max_adverse_r",
        ],
        "Production"
    )

    if len(df) != EXPECTED_TRADES:
        fail(
            f"Production rows {len(df)} "
            f"!= {EXPECTED_TRADES}"
        )

    df = df.copy()

    df["symbol"] = normalize_symbol(
        df["_symbol"]
    )

    df["direction"] = normalize_direction(
        df["direction"]
    )

    df["entry_date"] = normalize_date(
        df["entry_date"]
    )

    df["_merge_key"] = make_key(df)

    if df["_merge_key"].duplicated().any():
        fail(
            "Production contains duplicate identity keys."
        )

    df["outcome_final_r"] = pd.to_numeric(
        df["r_multiple"],
        errors="coerce"
    )

    df["outcome_mfe_r"] = pd.to_numeric(
        df["max_favorable_r"],
        errors="coerce"
    )

    df["outcome_mae_r"] = pd.to_numeric(
        df["max_adverse_r"],
        errors="coerce"
    )

    if df["outcome_final_r"].isna().any():
        fail(
            "Production contains missing final R values."
        )

    return df[
        [
            "_merge_key",
            "outcome_final_r",
            "outcome_mfe_r",
            "outcome_mae_r",
        ]
    ]


# ============================================================================
# BUILD BAR-2 COUNTERFACTUAL R
# ============================================================================

def build_counterfactual(
    trajectory,
    bar2,
    production,
):

    merged = (
        trajectory
        .merge(
            bar2[
                [
                    "_merge_key",
                    "close",
                    "entry_price_reconstructed",
                    "risk_per_unit",
                ]
            ],
            on="_merge_key",
            how="left",
            validate="one_to_one",
        )
        .merge(
            production,
            on="_merge_key",
            how="left",
            validate="one_to_one",
        )
    )

    if len(merged) != EXPECTED_TRADES:
        fail(
            f"Counterfactual merge rows "
            f"{len(merged)} != {EXPECTED_TRADES}"
        )

    if merged["close"].isna().any():
        fail(
            "Missing Bar-2 close after merge."
        )

    if merged["entry_price_reconstructed"].isna().any():
        fail(
            "Missing reconstructed entry price."
        )

    if merged["risk_per_unit"].isna().any():
        fail(
            "Missing risk per unit."
        )

    # Direction-aware R calculation.
    merged["bar2_close_r"] = np.where(
        normalize_direction(
            merged["direction"]
        ) == "LONG",
        (
            merged["close"]
            - merged["entry_price_reconstructed"]
        ) / merged["risk_per_unit"],
        (
            merged["entry_price_reconstructed"]
            - merged["close"]
        ) / merged["risk_per_unit"],
    )

    merged["original_final_r"] = (
        merged["outcome_final_r"]
    )

    merged["counterfactual_difference_r"] = (
        merged["bar2_close_r"]
        - merged["original_final_r"]
    )

    return merged


# ============================================================================
# STATE ANALYSIS
# ============================================================================

def analyze_state(
    df,
    state,
):

    state_mask = (
        df[state]
        .astype(bool)
    )

    state1 = df[state_mask].copy()
    state0 = df[~state_mask].copy()

    if len(state1) == 0 or len(state0) == 0:
        fail(
            f"State {state} does not contain both groups."
        )

    # Aggregate original and counterfactual R.
    original_total_1 = float(
        state1["original_final_r"].sum()
    )

    counter_total_1 = float(
        state1["bar2_close_r"].sum()
    )

    original_mean_1 = float(
        state1["original_final_r"].mean()
    )

    counter_mean_1 = float(
        state1["bar2_close_r"].mean()
    )

    difference_mean_1 = (
        counter_mean_1
        - original_mean_1
    )

    original_median_1 = float(
        state1["original_final_r"].median()
    )

    counter_median_1 = float(
        state1["bar2_close_r"].median()
    )

    difference_median_1 = (
        counter_median_1
        - original_median_1
    )

    # How much original R was lost/gained by exiting at Bar2?
    lost_r = float(
        state1["original_final_r"]
        .sub(state1["bar2_close_r"])
        .sum()
    )

    # Opportunity-cost view:
    # trades with positive subsequent R after Bar2.
    state1["post_bar2_incremental_r"] = (
        state1["original_final_r"]
        - state1["bar2_close_r"]
    )

    recovered_profit_mask = (
        state1["post_bar2_incremental_r"] > 0
    )

    recovered_loss_mask = (
        state1["post_bar2_incremental_r"] < 0
    )

    recovered_profit_count = int(
        recovered_profit_mask.sum()
    )

    avoided_loss_count = int(
        recovered_loss_mask.sum()
    )

    # All-trade economic comparison.
    original_total_all = float(
        df["original_final_r"].sum()
    )

    counter_total_all = float(
        np.where(
            state_mask,
            df["bar2_close_r"],
            df["original_final_r"]
        ).sum()
    )

    original_mean_all = float(
        df["original_final_r"].mean()
    )

    counter_mean_all = float(
        np.where(
            state_mask,
            df["bar2_close_r"],
            df["original_final_r"]
        ).mean()
    )

    total_difference_all = (
        counter_total_all
        - original_total_all
    )

    mean_difference_all = (
        counter_mean_all
        - original_mean_all
    )

    # Event/retained breakdown.
    event_mask = df["frozen_event"]

    state1_event = state1[
        state1["frozen_event"]
    ]

    state1_retained = state1[
        ~state1["frozen_event"]
    ]

    event_n = len(state1_event)
    retained_n = len(state1_retained)

    event_mean_delta = (
        float(
            (
                state1_event["bar2_close_r"]
                - state1_event["original_final_r"]
            ).mean()
        )
        if event_n
        else np.nan
    )

    retained_mean_delta = (
        float(
            (
                state1_retained["bar2_close_r"]
                - state1_retained["original_final_r"]
            ).mean()
        )
        if retained_n
        else np.nan
    )

    # Bootstrap CI for counterfactual-original mean difference
    # within state-1 trades.
    mean_delta, mean_lo, mean_hi = (
        bootstrap_mean_difference(
            state1["bar2_close_r"],
            state1["original_final_r"],
            seed=42,
        )
    )

    median_delta, median_lo, median_hi = (
        bootstrap_median_difference(
            state1["bar2_close_r"],
            state1["original_final_r"],
            seed=42,
        )
    )

    result = {
        "state": state,
        "state_label": STATE_LABELS[state],

        "state1_n": len(state1),
        "state0_n": len(state0),

        "state1_event_n": event_n,
        "state1_retained_n": retained_n,

        "original_total_r_state1":
            original_total_1,

        "counterfactual_total_r_state1":
            counter_total_1,

        "counterfactual_minus_original_total_r_state1":
            counter_total_1 - original_total_1,

        "original_mean_r_state1":
            original_mean_1,

        "counterfactual_mean_r_state1":
            counter_mean_1,

        "counterfactual_minus_original_mean_r_state1":
            difference_mean_1,

        "original_median_r_state1":
            original_median_1,

        "counterfactual_median_r_state1":
            counter_median_1,

        "counterfactual_minus_original_median_r_state1":
            difference_median_1,

        "bootstrap_mean_difference_r":
            mean_delta,

        "bootstrap_mean_ci_low_r":
            mean_lo,

        "bootstrap_mean_ci_high_r":
            mean_hi,

        "bootstrap_median_difference_r":
            median_delta,

        "bootstrap_median_ci_low_r":
            median_lo,

        "bootstrap_median_ci_high_r":
            median_hi,

        "post_bar2_incremental_r_sum":
            float(
                state1["post_bar2_incremental_r"].sum()
            ),

        "positive_post_bar2_incremental_trades":
            recovered_profit_count,

        "negative_post_bar2_incremental_trades":
            avoided_loss_count,

        "original_total_r_all":
            original_total_all,

        "counterfactual_total_r_all":
            counter_total_all,

        "counterfactual_minus_original_total_r_all":
            total_difference_all,

        "original_mean_r_all":
            original_mean_all,

        "counterfactual_mean_r_all":
            counter_mean_all,

        "counterfactual_minus_original_mean_r_all":
            mean_difference_all,

        "event_mean_counterfactual_minus_original_r":
            event_mean_delta,

        "retained_mean_counterfactual_minus_original_r":
            retained_mean_delta,
    }

    return result, state1


# ============================================================================
# COMPARATIVE OUTCOME PROFILE
# ============================================================================

def analyze_outcome_profile(
    df,
    state,
):

    state_mask = df[state].astype(bool)

    state1 = df[state_mask].copy()

    if len(state1) == 0:
        return None

    rows = []

    for outcome in [
        "outcome_final_r",
        "outcome_mfe_r",
        "outcome_mae_r",
    ]:

        values = pd.to_numeric(
            state1[outcome],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        rows.append(
            {
                "state": state,
                "state_label": STATE_LABELS[state],
                "outcome": outcome,
                "n": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# RECOVERY / OPPORTUNITY COST
# ============================================================================

def analyze_opportunity_cost(
    df,
    state,
):

    state1 = df[
        df[state].astype(bool)
    ].copy()

    state1["post_bar2_incremental_r"] = (
        state1["original_final_r"]
        - state1["bar2_close_r"]
    )

    positive = state1[
        state1["post_bar2_incremental_r"] > 0
    ]

    negative = state1[
        state1["post_bar2_incremental_r"] < 0
    ]

    zero = state1[
        state1["post_bar2_incremental_r"] == 0
    ]

    return {
        "state": state,
        "state_label": STATE_LABELS[state],
        "state_n": len(state1),
        "positive_post_bar2_count": len(positive),
        "negative_post_bar2_count": len(negative),
        "zero_post_bar2_count": len(zero),
        "positive_post_bar2_rate":
            len(positive) / len(state1)
            if len(state1)
            else np.nan,
        "negative_post_bar2_rate":
            len(negative) / len(state1)
            if len(state1)
            else np.nan,
        "median_post_bar2_incremental_r":
            float(
                state1[
                    "post_bar2_incremental_r"
                ].median()
            )
            if len(state1)
            else np.nan,
        "mean_post_bar2_incremental_r":
            float(
                state1[
                    "post_bar2_incremental_r"
                ].mean()
            )
            if len(state1)
            else np.nan,
        "sum_post_bar2_incremental_r":
            float(
                state1[
                    "post_bar2_incremental_r"
                ].sum()
            )
            if len(state1)
            else np.nan,
    }


# ============================================================================
# INTEGRITY
# ============================================================================

def integrity_checks(
    trajectory,
    bar2,
    production,
    merged,
):

    checks = []

    checks.append(
        {
            "check": "trajectory_rows",
            "actual": len(trajectory),
            "expected": EXPECTED_TRADES,
            "pass": len(trajectory)
            == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "bar2_rows",
            "actual": len(bar2),
            "expected": EXPECTED_TRADES,
            "pass": len(bar2)
            == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "production_rows",
            "actual": len(production),
            "expected": EXPECTED_TRADES,
            "pass": len(production)
            == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "merged_rows",
            "actual": len(merged),
            "expected": EXPECTED_TRADES,
            "pass": len(merged)
            == EXPECTED_TRADES,
        }
    )

    checks.append(
        {
            "check": "frozen_events",
            "actual": int(
                trajectory["frozen_event"].sum()
            ),
            "expected": EXPECTED_EVENTS,
            "pass":
                int(
                    trajectory["frozen_event"].sum()
                )
                == EXPECTED_EVENTS,
        }
    )

    checks.append(
        {
            "check": "frozen_retained",
            "actual": int(
                (~trajectory["frozen_event"]).sum()
            ),
            "expected": EXPECTED_RETAINED,
            "pass":
                int(
                    (~trajectory["frozen_event"]).sum()
                )
                == EXPECTED_RETAINED,
        }
    )

    checks.append(
        {
            "check": "missing_original_final_r",
            "actual": int(
                merged["original_final_r"].isna().sum()
            ),
            "expected": 0,
            "pass":
                int(
                    merged["original_final_r"].isna().sum()
                )
                == 0,
        }
    )

    checks.append(
        {
            "check": "missing_bar2_close_r",
            "actual": int(
                merged["bar2_close_r"].isna().sum()
            ),
            "expected": 0,
            "pass":
                int(
                    merged["bar2_close_r"].isna().sum()
                )
                == 0,
        }
    )

    return pd.DataFrame(checks)


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "TRADE SENSE-AI — BAR 1→2 "
        "COUNTERFACTUAL MANAGEMENT / "
        "ECONOMIC-VALUE FORENSIC ANALYSIS"
    )
    print("=" * 72)

    trajectory = load_trajectory()
    bar2 = load_bar2(
        pd.read_csv(PATH_FILE)
    )
    production = load_production()

    merged = build_counterfactual(
        trajectory,
        bar2,
        production,
    )

    print()
    print(
        "Counterfactual merge: PASS"
    )

    # ------------------------------------------------------------------------
    # State analysis
    # ------------------------------------------------------------------------

    primary_rows = []
    opportunity_rows = []
    profile_frames = []

    print()
    print("=" * 72)
    print("COUNTERFACTUAL BAR-2 EXIT RESULTS")
    print("=" * 72)

    for state in STATE_COLUMNS:

        result, state1 = analyze_state(
            merged,
            state,
        )

        primary_rows.append(result)

        opportunity_rows.append(
            analyze_opportunity_cost(
                merged,
                state,
            )
        )

        profile = analyze_outcome_profile(
            merged,
            state,
        )

        if profile is not None:
            profile_frames.append(profile)

        print()
        print(STATE_LABELS[state])

        print(
            f"  State-1 trades: "
            f"{result['state1_n']}"
        )

        print(
            f"  State-1 events: "
            f"{result['state1_event_n']}"
        )

        print(
            f"  State-1 retained: "
            f"{result['state1_retained_n']}"
        )

        print(
            f"  Original total R: "
            f"{result['original_total_r_state1']:+.4f}R"
        )

        print(
            f"  Bar-2 counterfactual total R: "
            f"{result['counterfactual_total_r_state1']:+.4f}R"
        )

        print(
            f"  Counterfactual - original total: "
            f"{result['counterfactual_minus_original_total_r_state1']:+.4f}R"
        )

        print(
            f"  Original mean R: "
            f"{result['original_mean_r_state1']:+.4f}R"
        )

        print(
            f"  Bar-2 counterfactual mean R: "
            f"{result['counterfactual_mean_r_state1']:+.4f}R"
        )

        print(
            f"  Mean difference: "
            f"{result['counterfactual_minus_original_mean_r_state1']:+.4f}R"
        )

        print(
            f"  Bootstrap mean CI: "
            f"["
            f"{result['bootstrap_mean_ci_low_r']:+.4f}R, "
            f"{result['bootstrap_mean_ci_high_r']:+.4f}R"
            f"]"
        )

        print(
            f"  Original median R: "
            f"{result['original_median_r_state1']:+.4f}R"
        )

        print(
            f"  Bar-2 counterfactual median R: "
            f"{result['counterfactual_median_r_state1']:+.4f}R"
        )

        print(
            f"  Median difference: "
            f"{result['counterfactual_minus_original_median_r_state1']:+.4f}R"
        )

        print(
            f"  Positive post-Bar2 incremental trades: "
            f"{result['positive_post_bar2_incremental_trades']}"
        )

        print(
            f"  Negative post-Bar2 incremental trades: "
            f"{result['negative_post_bar2_incremental_trades']}"
        )

        print(
            f"  All-trade total-R difference: "
            f"{result['counterfactual_minus_original_total_r_all']:+.4f}R"
        )

        print(
            f"  All-trade mean-R difference: "
            f"{result['counterfactual_minus_original_mean_r_all']:+.4f}R"
        )

    # ------------------------------------------------------------------------
    # Save primary
    # ------------------------------------------------------------------------

    primary = pd.DataFrame(
        primary_rows
    )

    primary.to_csv(
        OUTPUT_DIR
        / "bar12_counterfactual_primary.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Opportunity cost
    # ------------------------------------------------------------------------

    opportunity = pd.DataFrame(
        opportunity_rows
    )

    opportunity.to_csv(
        OUTPUT_DIR
        / "bar12_counterfactual_opportunity_cost.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Outcome profile
    # ------------------------------------------------------------------------

    if profile_frames:
        profile_all = pd.concat(
            profile_frames,
            ignore_index=True,
        )
    else:
        profile_all = pd.DataFrame()

    profile_all.to_csv(
        OUTPUT_DIR
        / "bar12_counterfactual_outcome_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Per-trade output
    # ------------------------------------------------------------------------

    per_trade = merged.copy()

    for state in STATE_COLUMNS:

        state_mask = (
            per_trade[state]
            .astype(bool)
        )

        per_trade[
            f"{state}_counterfactual_r"
        ] = np.where(
            state_mask,
            per_trade["bar2_close_r"],
            per_trade["original_final_r"],
        )

        per_trade[
            f"{state}_counterfactual_delta_r"
        ] = (
            per_trade[
                f"{state}_counterfactual_r"
            ]
            - per_trade["original_final_r"]
        )

    per_trade.to_csv(
        OUTPUT_DIR
        / "bar12_counterfactual_per_trade.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------------

    integrity = integrity_checks(
        trajectory,
        bar2,
        production,
        merged,
    )

    integrity.to_csv(
        OUTPUT_DIR
        / "bar12_counterfactual_integrity.csv",
        index=False,
    )

    integrity_pass = bool(
        integrity["pass"].all()
    )

    # ------------------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------------------

    summary = {
        "analysis":
            "Bar1->Bar2 counterfactual management",
        "frozen_population": {
            "trades": EXPECTED_TRADES,
            "events": EXPECTED_EVENTS,
            "retained": EXPECTED_RETAINED,
        },
        "bar2_exit_timing":
            "after_bar2_close_state_observed",
        "states_tested": STATE_COLUMNS,
        "integrity_pass":
            integrity_pass,
        "production_modified":
            False,
        "oos_modified":
            False,
        "deployment_rule_promoted":
            False,
        "methodology_note":
            "Hypothetical Bar-2-close exit; "
            "not executable performance and excludes "
            "transaction-cost/slippage modeling.",
    }

    with open(
        OUTPUT_DIR
        / "bar12_counterfactual_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)

    print(
        f"States tested: "
        f"{len(STATE_COLUMNS)}"
    )

    print(
        f"Integrity PASS: "
        f"{integrity_pass}"
    )

    print()
    print("IMPORTANT:")
    print(
        "This is a counterfactual research simulation."
    )
    print(
        "It does NOT establish executable trading performance, "
        "causality, independent OOS predictive power, "
        "transaction-cost-adjusted economic value, "
        "or a deployable rule."
    )
    print(
        "Production strategy logic and OOS data remain untouched."
    )

    print()
    print("Outputs:")
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_primary.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_opportunity_cost.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_outcome_profile.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_per_trade.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_integrity.csv"
    )
    print(
        OUTPUT_DIR
        / "bar12_counterfactual_summary.json"
    )


if __name__ == "__main__":
    main()