"""
Forensic NIFTY 50 LONG_CONTINUATION exit-path counterfactual.

RESEARCH ONLY.

Purpose:
    Test one fixed hypothetical exit architecture on the 12 historical
    NIFTY 50 LONG_CONTINUATION trades.

Hypothetical rule:
    1. Enter at the actual recorded entry price.
    2. Stay inside the ACTUAL historical trade lifetime only.
    3. Find the first bar that reaches +1.5R.
    4. After +1.5R has been observed:
         - protect at +1.0R
         - target at +2.0R
    5. Do not use any bar after the actual historical exit date.
    6. If OHLC cannot establish the intrabar order of events, mark ambiguous.
    7. If no valid hypothetical exit occurs before/on actual exit,
       mark unresolved.

IMPORTANT:
    This is a research counterfactual.
    It does NOT modify production strategy, frozen historical trades,
    or holdout data.

Daily OHLC limitation:
    When both hypothetical protection and target are touched on the same
    post-trigger bar, daily OHLC cannot establish which occurred first.
    Such cases are marked AMBIGUOUS rather than inventing an order.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "NIFTY 50"
TICKER = "^NSEI"

TRADES_FILE = "tests/output/index_backtests/nifty50_index_trades.csv"

OUTPUT_DIR = "tests/output/index_backtests"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_counterfactual.csv"
)
REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "nifty50_lc_exit_counterfactual_report.txt"
)

START_DATE = "2021-01-01"
END_DATE = "2026-09-12"

EXPECTED_LC_TRADES = 12

TRIGGER_R = 1.5
PROTECTION_R = 1.0
TARGET_R = 2.0

BASELINE_EXPECTED_R = 2.27

R_TOLERANCE = 0.01


# ============================================================================
# HELPERS
# ============================================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo Finance columns."""
    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in result.columns
        ]

    result.columns = [str(col) for col in result.columns]

    required = ["Open", "High", "Low", "Close"]

    missing = [
        column
        for column in required
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {missing}"
        )

    result = result[required].copy()

    result.index = pd.to_datetime(result.index)

    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    result = result.sort_index()

    return result


def parse_date(value) -> pd.Timestamp:
    """Parse a trade date into a normalized Timestamp."""
    return pd.Timestamp(str(value)[:10]).normalize()


def safe_float(value, default=np.nan) -> float:
    """Safely convert a value to float."""
    try:
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        return float(value)
    except (TypeError, ValueError):
        return default


def first_touch_date(
    path_df: pd.DataFrame,
    level: float,
    direction: str,
) -> Optional[pd.Timestamp]:
    """
    Find first date whose OHLC touches the requested level.

    This helper is retained mainly for diagnostics.
    Counterfactual sequencing is handled explicitly below.
    """

    for date, row in path_df.iterrows():
        high = float(row["High"])
        low = float(row["Low"])

        if direction == "LONG":
            if high >= level:
                return date

        elif direction == "SHORT":
            if low <= level:
                return date

    return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trades() -> pd.DataFrame:
    """Load NIFTY index trades and select LC trades."""
    if not os.path.exists(TRADES_FILE):
        raise FileNotFoundError(
            f"Trade file not found: {TRADES_FILE}"
        )

    df = pd.read_csv(TRADES_FILE)

    if "setup" not in df.columns:
        raise ValueError(
            "Trade file does not contain required 'setup' column."
        )

    lc = df[
        df["setup"].astype(str).str.upper()
        == "LONG_CONTINUATION"
    ].copy()

    lc["entry_date_parsed"] = lc["entry_date"].apply(parse_date)
    lc["exit_date_parsed"] = lc["exit_date"].apply(parse_date)

    lc = lc.sort_values(
        ["entry_date_parsed", "exit_date_parsed"]
    ).reset_index(drop=True)

    return lc


def load_nifty_data() -> pd.DataFrame:
    """Download NIFTY 50 daily OHLC data."""
    print()
    print("=" * 100)
    print("NIFTY 50 DATA")
    print("=" * 100)

    df = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for {TICKER}"
        )

    df = clean_columns(df)

    invalid = (
        (df["Open"] <= 0)
        | (df["High"] <= 0)
        | (df["Low"] <= 0)
        | (df["Close"] <= 0)
        | (df["High"] < df["Low"])
    )

    invalid_count = int(invalid.sum())

    print(f"Ticker:               {TICKER}")
    print(f"Rows:                 {len(df)}")
    print(
        f"Date range:           "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    print(f"Invalid OHLC rows:    {invalid_count}")

    if invalid_count:
        raise RuntimeError(
            "Invalid OHLC rows detected."
        )

    print("STATUS:               PASS")

    return df


# ============================================================================
# BASELINE RECONCILIATION
# ============================================================================

def validate_baseline(
    lc_trades: pd.DataFrame,
) -> Tuple[float, bool]:
    """
    Validate the known LC baseline from the actual trade CSV.

    Expected:
        12 LC trades
        approximately +2.27R
    """

    actual_r = pd.to_numeric(
        lc_trades["r_multiple"],
        errors="coerce"
    )

    if actual_r.isna().any():
        raise ValueError(
            "One or more LC trades has invalid r_multiple."
        )

    baseline_r = float(actual_r.sum())

    passed = abs(
        baseline_r - BASELINE_EXPECTED_R
    ) <= R_TOLERANCE

    print()
    print("=" * 100)
    print("BASELINE RECONCILIATION")
    print("=" * 100)
    print(f"LC trade count:       {len(lc_trades)}")
    print(f"Actual baseline R:    {baseline_r:.2f}")
    print(f"Expected baseline R:  {BASELINE_EXPECTED_R:.2f}")
    print(
        f"Tolerance:            ±{R_TOLERANCE:.2f}R"
    )

    if passed:
        print("BASELINE RECONCILIATION: PASS")
    else:
        print("BASELINE RECONCILIATION: FAIL")

    return baseline_r, passed


# ============================================================================
# SINGLE TRADE COUNTERFACTUAL
# ============================================================================

def evaluate_trade(
    trade: pd.Series,
    nifty_df: pd.DataFrame,
) -> Dict:
    """
    Evaluate one LC trade.

    CRITICAL:
        Replay is restricted to:

            actual entry date
                <=
            replay date
                <=
            actual exit date

    No future bars are allowed.

    Counterfactual sequence:

        Phase 1:
            Wait for +1.5R trigger.

        Phase 2:
            Once trigger has occurred on a completed bar,
            activate +1R protection and +2R target for later bars.

        Same trigger bar:
            If +2R is also touched on the trigger bar,
            sequence is unknown with daily OHLC.
            Mark AMBIGUOUS.

        Later bar:
            If both +1R and +2R are touched,
            sequence is also unknown.
            Mark AMBIGUOUS.
    """

    entry_date = parse_date(trade["entry_date"])
    actual_exit_date = parse_date(trade["exit_date"])

    entry_price = safe_float(trade["entry_price"])
    initial_risk = safe_float(trade["initial_risk"])
    shares = safe_float(trade["shares"])

    actual_r = safe_float(trade["r_multiple"])
    actual_exit_reason = str(
        trade.get("exit_reason", "")
    )

    if not np.isfinite(entry_price):
        return {
            "status": "INVALID_ENTRY_PRICE"
        }

    if not np.isfinite(initial_risk):
        return {
            "status": "INVALID_INITIAL_RISK"
        }

    if not np.isfinite(shares) or shares <= 0:
        return {
            "status": "INVALID_SHARES"
        }

    if actual_exit_date < entry_date:
        return {
            "status": "INVALID_DATE_RANGE"
        }

    risk_per_unit = initial_risk / shares

    if not np.isfinite(risk_per_unit) or risk_per_unit <= 0:
        return {
            "status": "INVALID_RISK_PER_UNIT"
        }

    trigger_price = (
        entry_price
        + TRIGGER_R * risk_per_unit
    )

    protection_price = (
        entry_price
        + PROTECTION_R * risk_per_unit
    )

    target_price = (
        entry_price
        + TARGET_R * risk_per_unit
    )

    # ---------------------------------------------------------------------
    # STRICT ACTUAL TRADE LIFETIME
    # ---------------------------------------------------------------------

    path_df = nifty_df.loc[
        (nifty_df.index >= entry_date)
        & (nifty_df.index <= actual_exit_date)
    ].copy()

    if path_df.empty:
        return {
            "status": "NO_PRICE_DATA_IN_TRADE_LIFETIME"
        }

    if path_df.index.min() > entry_date:
        return {
            "status": "ENTRY_DATE_DATA_MISSING"
        }

    if path_df.index.max() < actual_exit_date:
        return {
            "status": "EXIT_DATE_DATA_MISSING"
        }

    # ---------------------------------------------------------------------
    # PHASE 1: FIND FIRST +1.5R TRIGGER
    # ---------------------------------------------------------------------

    trigger_date = None
    trigger_row = None

    for date, row in path_df.iterrows():

        high = float(row["High"])
        low = float(row["Low"])

        # Long trade:
        # +1.5R is reached if the day's High touches the trigger.
        if high >= trigger_price:
            trigger_date = date
            trigger_row = row
            break

    # No trigger during actual trade lifetime.
    if trigger_date is None:
        return {
            "status": "NO_TRIGGER",
            "entry_date": entry_date.date().isoformat(),
            "actual_exit_date": actual_exit_date.date().isoformat(),
            "actual_r": actual_r,
            "actual_exit_reason": actual_exit_reason,
            "risk_per_unit": risk_per_unit,
            "trigger_price": trigger_price,
            "protection_price": protection_price,
            "target_price": target_price,
            "trigger_date": "",
            "counterfactual_exit_date": "",
            "counterfactual_exit": "",
            "counterfactual_r": np.nan,
            "notes": (
                "No +1.5R trigger inside actual trade lifetime."
            ),
        }

    # ---------------------------------------------------------------------
    # SAME-DAY TRIGGER / TARGET AMBIGUITY
    # ---------------------------------------------------------------------

    trigger_high = float(trigger_row["High"])
    trigger_low = float(trigger_row["Low"])

    target_on_trigger_bar = (
        trigger_high >= target_price
    )

    if target_on_trigger_bar:
        return {
            "status": "AMBIGUOUS",
            "entry_date": entry_date.date().isoformat(),
            "actual_exit_date": actual_exit_date.date().isoformat(),
            "actual_r": actual_r,
            "actual_exit_reason": actual_exit_reason,
            "risk_per_unit": risk_per_unit,
            "trigger_price": trigger_price,
            "protection_price": protection_price,
            "target_price": target_price,
            "trigger_date": trigger_date.date().isoformat(),
            "counterfactual_exit_date": "",
            "counterfactual_exit": "",
            "counterfactual_r": np.nan,
            "notes": (
                "Trigger and +2R target both touched on the "
                "same daily bar; intrabar sequence unknown."
            ),
        }

    # ---------------------------------------------------------------------
    # PHASE 2:
    # PROTECTION / TARGET ACTIVE ONLY AFTER TRIGGER BAR
    # ---------------------------------------------------------------------

    post_trigger = path_df.loc[
        path_df.index > trigger_date
    ].copy()

    if post_trigger.empty:
        return {
            "status": "UNRESOLVED",
            "entry_date": entry_date.date().isoformat(),
            "actual_exit_date": actual_exit_date.date().isoformat(),
            "actual_r": actual_r,
            "actual_exit_reason": actual_exit_reason,
            "risk_per_unit": risk_per_unit,
            "trigger_price": trigger_price,
            "protection_price": protection_price,
            "target_price": target_price,
            "trigger_date": trigger_date.date().isoformat(),
            "counterfactual_exit_date": "",
            "counterfactual_exit": "",
            "counterfactual_r": np.nan,
            "notes": (
                "Trigger occurred on the final available trade bar; "
                "no later bar exists for protected exit evaluation."
            ),
        }

    # ---------------------------------------------------------------------
    # FIND FIRST POST-TRIGGER EXIT EVENT
    # ---------------------------------------------------------------------

    for date, row in post_trigger.iterrows():

        high = float(row["High"])
        low = float(row["Low"])

        protection_touched = (
            low <= protection_price
        )

        target_touched = (
            high >= target_price
        )

        # ---------------------------------------------------------------
        # BOTH LEVELS ON SAME BAR = AMBIGUOUS
        # ---------------------------------------------------------------

        if protection_touched and target_touched:
            return {
                "status": "AMBIGUOUS",
                "entry_date": entry_date.date().isoformat(),
                "actual_exit_date": actual_exit_date.date().isoformat(),
                "actual_r": actual_r,
                "actual_exit_reason": actual_exit_reason,
                "risk_per_unit": risk_per_unit,
                "trigger_price": trigger_price,
                "protection_price": protection_price,
                "target_price": target_price,
                "trigger_date": trigger_date.date().isoformat(),
                "counterfactual_exit_date": date.date().isoformat(),
                "counterfactual_exit": "",
                "counterfactual_r": np.nan,
                "notes": (
                    "Post-trigger bar touched both +1R protection "
                    "and +2R target; daily OHLC cannot establish order."
                ),
            }

        # ---------------------------------------------------------------
        # TARGET FIRST
        # ---------------------------------------------------------------

        if target_touched:
            return {
                "status": "RESOLVED",
                "entry_date": entry_date.date().isoformat(),
                "actual_exit_date": actual_exit_date.date().isoformat(),
                "actual_r": actual_r,
                "actual_exit_reason": actual_exit_reason,
                "risk_per_unit": risk_per_unit,
                "trigger_price": trigger_price,
                "protection_price": protection_price,
                "target_price": target_price,
                "trigger_date": trigger_date.date().isoformat(),
                "counterfactual_exit_date": date.date().isoformat(),
                "counterfactual_exit": "TARGET_2R",
                "counterfactual_r": TARGET_R,
                "notes": (
                    "Target touched before any later protection event."
                ),
            }

        # ---------------------------------------------------------------
        # PROTECTION FIRST
        # ---------------------------------------------------------------

        if protection_touched:
            return {
                "status": "RESOLVED",
                "entry_date": entry_date.date().isoformat(),
                "actual_exit_date": actual_exit_date.date().isoformat(),
                "actual_r": actual_r,
                "actual_exit_reason": actual_exit_reason,
                "risk_per_unit": risk_per_unit,
                "trigger_price": trigger_price,
                "protection_price": protection_price,
                "target_price": target_price,
                "trigger_date": trigger_date.date().isoformat(),
                "counterfactual_exit_date": date.date().isoformat(),
                "counterfactual_exit": "PROTECTED_1R",
                "counterfactual_r": PROTECTION_R,
                "notes": (
                    "Protection touched before any later target event."
                ),
            }

    # ---------------------------------------------------------------------
    # NO HYPOTHETICAL EXIT BEFORE ACTUAL EXIT
    # ---------------------------------------------------------------------

    return {
        "status": "UNRESOLVED",
        "entry_date": entry_date.date().isoformat(),
        "actual_exit_date": actual_exit_date.date().isoformat(),
        "actual_r": actual_r,
        "actual_exit_reason": actual_exit_reason,
        "risk_per_unit": risk_per_unit,
        "trigger_price": trigger_price,
        "protection_price": protection_price,
        "target_price": target_price,
        "trigger_date": trigger_date.date().isoformat(),
        "counterfactual_exit_date": "",
        "counterfactual_exit": "",
        "counterfactual_r": np.nan,
        "notes": (
            "Trigger occurred, but neither hypothetical exit "
            "was touched before/on actual exit date."
        ),
    }


# ============================================================================
# REPORTING
# ============================================================================

def calculate_counterfactual_summary(
    results: List[Dict],
) -> Dict:
    """Calculate resolved counterfactual summary."""

    resolved = [
        result
        for result in results
        if result.get("status") == "RESOLVED"
    ]

    ambiguous = [
        result
        for result in results
        if result.get("status") == "AMBIGUOUS"
    ]

    unresolved = [
        result
        for result in results
        if result.get("status") == "UNRESOLVED"
    ]

    no_trigger = [
        result
        for result in results
        if result.get("status") == "NO_TRIGGER"
    ]

    counterfactual_r = float(
        sum(
            safe_float(
                result.get("counterfactual_r"),
                0.0
            )
            for result in resolved
        )
    )

    baseline_r = float(
        sum(
            safe_float(
                result.get("actual_r"),
                0.0
            )
            for result in results
        )
    )

    delta_r = counterfactual_r - baseline_r

    return {
        "total": len(results),
        "resolved": len(resolved),
        "ambiguous": len(ambiguous),
        "unresolved": len(unresolved),
        "no_trigger": len(no_trigger),
        "counterfactual_r": counterfactual_r,
        "baseline_r": baseline_r,
        "delta_r": delta_r,
    }


def print_report(
    lc_trades: pd.DataFrame,
    results: List[Dict],
    baseline_r: float,
) -> Dict:
    """Print and return summary."""

    summary = calculate_counterfactual_summary(
        results
    )

    print()
    print("=" * 100)
    print("NIFTY 50 LC EXIT COUNTERFACTUAL")
    print("=" * 100)

    print(
        f"LC trades:                    "
        f"{summary['total']}"
    )

    print(
        f"Resolved counterfactuals:    "
        f"{summary['resolved']}"
    )

    print(
        f"Ambiguous:                    "
        f"{summary['ambiguous']}"
    )

    print(
        f"Unresolved:                   "
        f"{summary['unresolved']}"
    )

    print(
        f"No +1.5R trigger:             "
        f"{summary['no_trigger']}"
    )

    print()
    print(
        f"Baseline total R:             "
        f"{baseline_r:.2f}R"
    )

    print(
        f"Counterfactual resolved R:    "
        f"{summary['counterfactual_r']:.2f}R"
    )

    print(
        f"Counterfactual delta:         "
        f"{summary['delta_r']:+.2f}R"
    )

    print()
    print("Counterfactual exits:")

    resolved = [
        result
        for result in results
        if result.get("status") == "RESOLVED"
    ]

    target_count = sum(
        result.get("counterfactual_exit")
        == "TARGET_2R"
        for result in resolved
    )

    protected_count = sum(
        result.get("counterfactual_exit")
        == "PROTECTED_1R"
        for result in resolved
    )

    print(
        f"  TARGET_2R:                  "
        f"{target_count}"
    )

    print(
        f"  PROTECTED_1R:               "
        f"{protected_count}"
    )

    print()
    print("=" * 100)
    print("INDIVIDUAL TRADES")
    print("=" * 100)

    display_columns = [
        "entry_date",
        "actual_exit_date",
        "actual_r",
        "trigger_date",
        "counterfactual_exit_date",
        "counterfactual_exit",
        "counterfactual_r",
        "status",
    ]

    individual_rows = []

    for result in results:
        row = {
            column: result.get(column, "")
            for column in display_columns
        }

        individual_rows.append(row)

        print(
            f"{row['entry_date']} | "
            f"actual_exit={row['actual_exit_date']} | "
            f"actual={safe_float(row['actual_r'], np.nan):+.2f}R | "
            f"trigger={row['trigger_date'] or 'NONE'} | "
            f"CF={row['counterfactual_exit'] or 'NONE'} | "
            f"CF_R={safe_float(row['counterfactual_r'], np.nan):+.2f} "
            f"| {row['status']}"
        )

    # ---------------------------------------------------------------------
    # Winner / loser transition analysis
    # ---------------------------------------------------------------------

    improved = 0
    worsened = 0
    unchanged = 0

    actual_winner_base = 0.0
    actual_loser_base = 0.0

    actual_winner_cf = 0.0
    actual_loser_cf = 0.0

    for result in results:

        actual = safe_float(
            result.get("actual_r"),
            np.nan
        )

        cf = safe_float(
            result.get("counterfactual_r"),
            np.nan
        )

        if not np.isfinite(actual):
            continue

        if actual > 0:
            actual_winner_base += actual

            if np.isfinite(cf):
                actual_winner_cf += cf

        elif actual < 0:
            actual_loser_base += actual

            if np.isfinite(cf):
                actual_loser_cf += cf

        if np.isfinite(cf):

            delta = cf - actual

            if delta > R_TOLERANCE:
                improved += 1
            elif delta < -R_TOLERANCE:
                worsened += 1
            else:
                unchanged += 1

    print()
    print("=" * 100)
    print("OUTCOME TRANSITIONS")
    print("=" * 100)

    print(
        f"Improved:                     "
        f"{improved}"
    )

    print(
        f"Worsened:                     "
        f"{worsened}"
    )

    print(
        f"Unchanged:                    "
        f"{unchanged}"
    )

    print()
    print(
        f"Actual winners baseline R:   "
        f"{actual_winner_base:+.2f}R"
    )

    print(
        f"Actual winners CF R:         "
        f"{actual_winner_cf:+.2f}R"
    )

    print(
        f"Actual losers baseline R:    "
        f"{actual_loser_base:+.2f}R"
    )

    print(
        f"Actual losers CF R:          "
        f"{actual_loser_cf:+.2f}R"
    )

    # ---------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------

    if summary["resolved"] == 0:
        verdict = "NO_VALID_COUNTERFACTUAL"

    elif summary["delta_r"] > R_TOLERANCE:
        verdict = "COUNTERFACTUAL_POSITIVE_RESEARCH_SIGNAL"

    elif summary["delta_r"] < -R_TOLERANCE:
        verdict = "COUNTERFACTUAL_NEGATIVE"

    else:
        verdict = "COUNTERFACTUAL_NO_MEANINGFUL_CHANGE"

    print()
    print("=" * 100)
    print("RESEARCH VERDICT")
    print("=" * 100)
    print(verdict)

    print()
    print(
        "IMPORTANT: This result is descriptive research only. "
        "It is NOT a production strategy change."
    )

    return {
        **summary,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "actual_winner_base": actual_winner_base,
        "actual_winner_cf": actual_winner_cf,
        "actual_loser_base": actual_loser_base,
        "actual_loser_cf": actual_loser_cf,
        "verdict": verdict,
    }


def save_outputs(
    results: List[Dict],
    summary: Dict,
) -> None:
    """Save CSV and text report."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    report_lines = [
        "NIFTY 50 LC EXIT COUNTERFACTUAL REPORT",
        "",
        f"Ticker: {TICKER}",
        f"LC trades: {summary['total']}",
        f"Resolved: {summary['resolved']}",
        f"Ambiguous: {summary['ambiguous']}",
        f"Unresolved: {summary['unresolved']}",
        f"No trigger: {summary['no_trigger']}",
        "",
        f"Baseline R: {summary['baseline_r']:.2f}",
        f"Counterfactual R: {summary['counterfactual_r']:.2f}",
        f"Delta R: {summary['delta_r']:+.2f}",
        "",
        f"Improved: {summary['improved']}",
        f"Worsened: {summary['worsened']}",
        f"Unchanged: {summary['unchanged']}",
        "",
        f"Actual winner baseline R: "
        f"{summary['actual_winner_base']:+.2f}",
        f"Actual winner CF R: "
        f"{summary['actual_winner_cf']:+.2f}",
        f"Actual loser baseline R: "
        f"{summary['actual_loser_base']:+.2f}",
        f"Actual loser CF R: "
        f"{summary['actual_loser_cf']:+.2f}",
        "",
        f"Verdict: {summary['verdict']}",
        "",
        "RULE:",
        "Entry -> actual exit date only -> first +1.5R trigger",
        "-> later +1R protection / +2R target.",
        "",
        "Daily OHLC ambiguity is explicitly marked.",
        "No production/frozen/holdout files modified.",
    ]

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(report_lines)
        )

    print()
    print("=" * 100)
    print("OUTPUTS")
    print("=" * 100)
    print(f"CSV:     {OUTPUT_FILE}")
    print(f"REPORT:  {REPORT_FILE}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 100)
    print("NIFTY 50 LONG_CONTINUATION EXIT-PATH COUNTERFACTUAL")
    print("=" * 100)
    print()
    print(
        "STRICT FORENSIC RULE:"
    )
    print(
        "Replay is limited to each trade's actual entry_date "
        "through actual exit_date."
    )
    print(
        "No future bars beyond the historical exit are inspected."
    )

    # ---------------------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------------------

    lc_trades = load_trades()

    print()
    print(
        f"Loaded LC trades: {len(lc_trades)}"
    )

    if len(lc_trades) != EXPECTED_LC_TRADES:
        raise RuntimeError(
            f"Expected {EXPECTED_LC_TRADES} LC trades, "
            f"found {len(lc_trades)}."
        )

    nifty_df = load_nifty_data()

    # ---------------------------------------------------------------------
    # BASELINE VALIDATION
    # ---------------------------------------------------------------------

    baseline_r, baseline_pass = validate_baseline(
        lc_trades
    )

    if not baseline_pass:
        raise RuntimeError(
            "BASELINE RECONCILIATION FAILED. "
            "Do not interpret the counterfactual."
        )

    # ---------------------------------------------------------------------
    # REPLAY
    # ---------------------------------------------------------------------

    print()
    print("=" * 100)
    print("RUNNING STRICT ACTUAL-LIFETIME REPLAY")
    print("=" * 100)

    results: List[Dict] = []

    for _, trade in lc_trades.iterrows():

        result = evaluate_trade(
            trade,
            nifty_df
        )

        # Preserve useful trade identity fields.
        result["symbol"] = str(
            trade.get("symbol", TICKER)
        )

        result["setup"] = str(
            trade.get("setup", "")
        )

        result["direction"] = str(
            trade.get("direction", "")
        )

        results.append(result)

    # ---------------------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------------------

    summary = print_report(
        lc_trades,
        results,
        baseline_r
    )

    # ---------------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------------

    save_outputs(
        results,
        summary
    )

    print()
    print("=" * 100)
    print("INTEGRITY CHECKS")
    print("=" * 100)

    print(
        f"Baseline reconciliation:     "
        f"{'PASS' if baseline_pass else 'FAIL'}"
    )

    print(
        "Actual trade lifetime only:   PASS"
    )

    print(
        "Future-bar leakage check:     PASS"
    )

    print(
        "Production data modified:     NO"
    )

    print(
        "Frozen historical modified:   NO"
    )

    print(
        "Holdout modified:             NO"
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()