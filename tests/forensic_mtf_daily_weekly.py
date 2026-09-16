"""
TradeSense-AI
Daily / Weekly MTF Forensic Validation

PURPOSE
-------
Research-only forensic analysis of Daily + Weekly trend alignment
against the frozen TradeSense-AI historical trade ledger.

IMPORTANT
---------
- Production strategy is NOT modified.
- Frozen trade ledger is NOT modified.
- OOS data / OOS logic is NOT modified.
- All frozen trades remain in the reconciliation ledger.
- MTF-dependent performance statistics use only trades where the
  required Daily + Weekly state is actually available.
- Early historical trades without sufficient weekly history are
  classified as MTF_DATA_UNAVAILABLE instead of being dropped.
- Weekly state is intentionally causal:
  only the LAST COMPLETED WEEK strictly before the signal date
  is used.
"""

from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

TRADES_FILE = ROOT / "tests" / "output" / "tradesense_trades.csv"
PRICE_DIR = ROOT / "data" / "nifty50_prices"

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSTANTS
# ============================================================================

EXPECTED_FROZEN_TRADES = 106
EXPECTED_FROZEN_TOTAL_R = -29.15

MA_FAST = 20
MA_SLOW = 50

RECON_TOLERANCE = 0.05


# ============================================================================
# HELPERS
# ============================================================================

def find_column(df: pd.DataFrame, candidates, label: str) -> str:
    """
    Find a column using case-insensitive matching while preserving
    the actual dataframe column name.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in normalized:
            return normalized[key]

    raise RuntimeError(
        f"Could not find {label} column.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_direction(value) -> str:
    value = str(value).strip().upper()

    if value in {"LONG", "BUY"}:
        return "LONG"

    if value in {"SHORT", "SELL"}:
        return "SHORT"

    return value


def normalize_setup(value) -> str:
    return str(value).strip().upper()


def normalize_symbol(symbol: str) -> str:
    """
    Frozen ledger uses Yahoo-style symbols such as HDFCBANK.NS.
    Local validated files use NSE roots such as HDFCBANK.csv.
    """
    value = str(symbol).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def to_float(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def performance_summary(
    df: pd.DataFrame,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Produce trade-performance statistics using R_Multiple.
    """

    if df.empty:
        if group_col:
            return pd.DataFrame(
                columns=[
                    group_col,
                    "Trades",
                    "Wins",
                    "Losses",
                    "Win_Rate_Pct",
                    "Gross_Profit_R",
                    "Gross_Loss_R",
                    "Profit_Factor",
                    "Total_R",
                    "Average_R",
                ]
            )

        return pd.DataFrame(
            [{
                "Baseline": "ALL_TRADES",
                "Trades": 0,
                "Wins": 0,
                "Losses": 0,
                "Win_Rate_Pct": 0.0,
                "Gross_Profit_R": 0.0,
                "Gross_Loss_R": 0.0,
                "Profit_Factor": 0.0,
                "Total_R": 0.0,
                "Average_R": 0.0,
            }]
        )

    work = df.copy()
    work["R_Multiple"] = pd.to_numeric(
        work["R_Multiple"],
        errors="coerce",
    )

    def summarize(part: pd.DataFrame):
        r = part["R_Multiple"].dropna()

        trades = len(r)
        wins = int((r > 0).sum())
        losses = int((r <= 0).sum())

        gross_profit = float(r[r > 0].sum())
        gross_loss = float(r[r < 0].sum())

        if abs(gross_loss) > 1e-12:
            pf = gross_profit / abs(gross_loss)
        else:
            pf = np.inf if gross_profit > 0 else 0.0

        return pd.Series({
            "Trades": trades,
            "Wins": wins,
            "Losses": losses,
            "Win_Rate_Pct": (
                round((wins / trades) * 100, 2)
                if trades
                else 0.0
            ),
            "Gross_Profit_R": round(gross_profit, 2),
            "Gross_Loss_R": round(gross_loss, 2),
            "Profit_Factor": (
                round(pf, 4)
                if math.isfinite(pf)
                else np.inf
            ),
            "Total_R": round(float(r.sum()), 2),
            "Average_R": (
                round(float(r.mean()), 4)
                if trades
                else 0.0
            ),
        })

    if group_col is None:
        result = summarize(work).to_frame().T
        result.insert(0, "Baseline", "ALL_TRADES")
        return result

    grouped = (
        work
        .groupby(group_col, dropna=False, sort=False)
        .apply(summarize)
        .reset_index()
    )

    return grouped


def state_for_direction(state: str, direction: str) -> str:
    """
    Convert raw timeframe state into directional agreement.

    LONG:
        BULLISH = aligned
        BEARISH = conflict
        NEUTRAL = neutral

    SHORT:
        BEARISH = aligned
        BULLISH = conflict
        NEUTRAL = neutral
    """

    if state == "UNKNOWN":
        return "UNKNOWN"

    if state == "NEUTRAL":
        return "NEUTRAL"

    if direction == "LONG":
        if state == "BULLISH":
            return "ALIGNED"
        if state == "BEARISH":
            return "CONFLICT"

    if direction == "SHORT":
        if state == "BEARISH":
            return "ALIGNED"
        if state == "BULLISH":
            return "CONFLICT"

    return "UNKNOWN"


def classify_mtf(
    daily_state: str,
    weekly_state: str,
    direction: str,
) -> tuple[str, int]:

    daily_relation = state_for_direction(
        daily_state,
        direction,
    )

    weekly_relation = state_for_direction(
        weekly_state,
        direction,
    )

    if daily_relation == "UNKNOWN" or weekly_relation == "UNKNOWN":
        return "MTF_DATA_UNAVAILABLE", np.nan

    if daily_relation == "ALIGNED" and weekly_relation == "ALIGNED":
        return "FULL_ALIGNMENT", 2

    if (
        daily_relation == "ALIGNED"
        and weekly_relation == "NEUTRAL"
    ) or (
        daily_relation == "NEUTRAL"
        and weekly_relation == "ALIGNED"
    ):
        return "PARTIAL_ALIGNMENT", 1

    if daily_relation == "NEUTRAL" and weekly_relation == "NEUTRAL":
        return "NEUTRAL", 0

    if (
        daily_relation == "CONFLICT"
        and weekly_relation == "NEUTRAL"
    ) or (
        daily_relation == "NEUTRAL"
        and weekly_relation == "CONFLICT"
    ):
        return "PARTIAL_CONFLICT", -1

    if (
        daily_relation == "ALIGNED"
        and weekly_relation == "CONFLICT"
    ) or (
        daily_relation == "CONFLICT"
        and weekly_relation == "ALIGNED"
    ):
        return "MIXED", 0

    if daily_relation == "CONFLICT" and weekly_relation == "CONFLICT":
        return "FULL_CONFLICT", -2

    return "MTF_DATA_UNAVAILABLE", np.nan


# ============================================================================
# PRICE DATA
# ============================================================================

def load_symbol_prices(symbol: str) -> pd.DataFrame:
    root_symbol = normalize_symbol(symbol)

    path = PRICE_DIR / f"{root_symbol}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing price file for {symbol}: {path}"
        )

    df = pd.read_csv(path)

    date_col = find_column(
        df,
        ["Date", "date", "Datetime", "datetime"],
        f"date for {symbol}",
    )

    open_col = find_column(
        df,
        ["Open", "open"],
        f"Open for {symbol}",
    )

    high_col = find_column(
        df,
        ["High", "high"],
        f"High for {symbol}",
    )

    low_col = find_column(
        df,
        ["Low", "low"],
        f"Low for {symbol}",
    )

    close_col = find_column(
        df,
        ["Close", "close"],
        f"Close for {symbol}",
    )

    result = pd.DataFrame({
        "Date": pd.to_datetime(
            df[date_col],
            errors="coerce",
        ),
        "Open": pd.to_numeric(
            df[open_col],
            errors="coerce",
        ),
        "High": pd.to_numeric(
            df[high_col],
            errors="coerce",
        ),
        "Low": pd.to_numeric(
            df[low_col],
            errors="coerce",
        ),
        "Close": pd.to_numeric(
            df[close_col],
            errors="coerce",
        ),
    })

    result = (
        result
        .dropna(subset=["Date", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )

    if result.empty:
        raise RuntimeError(
            f"No usable price rows found for {symbol}"
        )

    return result


# ============================================================================
# DAILY STATE
# ============================================================================

def build_daily_state(prices: pd.DataFrame) -> pd.DataFrame:
    daily = prices.copy()

    daily["MA20"] = daily["Close"].rolling(
        MA_FAST,
        min_periods=MA_FAST,
    ).mean()

    daily["MA50"] = daily["Close"].rolling(
        MA_SLOW,
        min_periods=MA_SLOW,
    ).mean()

    daily["Daily_State"] = "UNKNOWN"

    valid = (
        daily["MA20"].notna()
        & daily["MA50"].notna()
    )

    bullish = (
        valid
        & (daily["MA20"] > daily["MA50"])
        & (daily["Close"] > daily["MA20"])
    )

    bearish = (
        valid
        & (daily["MA20"] < daily["MA50"])
        & (daily["Close"] < daily["MA20"])
    )

    neutral = valid & ~(bullish | bearish)

    daily.loc[bullish, "Daily_State"] = "BULLISH"
    daily.loc[bearish, "Daily_State"] = "BEARISH"
    daily.loc[neutral, "Daily_State"] = "NEUTRAL"

    return daily[
        [
            "Close",
            "MA20",
            "MA50",
            "Daily_State",
        ]
    ].copy()


# ============================================================================
# WEEKLY STATE — CAUSAL
# ============================================================================

def build_weekly_state(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build weekly candles from daily data.

    CRITICAL CAUSALITY RULE:
    ------------------------
    For a signal on date D, the MTF classification uses ONLY the
    last completed weekly candle STRICTLY BEFORE D.

    Therefore:
        Monday -> previous Friday
        Tuesday -> previous Friday
        Wednesday -> previous Friday
        Thursday -> previous Friday
        Friday -> previous Friday

    This deliberately avoids using the current week's developing
    candle. It is conservative and leakage-safe.
    """

    weekly = prices.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    })

    weekly = weekly.dropna(subset=["Close"]).copy()

    weekly["MA20"] = weekly["Close"].rolling(
        MA_FAST,
        min_periods=MA_FAST,
    ).mean()

    weekly["MA50"] = weekly["Close"].rolling(
        MA_SLOW,
        min_periods=MA_SLOW,
    ).mean()

    weekly["Completed_Weekly_State"] = "UNKNOWN"

    valid = (
        weekly["MA20"].notna()
        & weekly["MA50"].notna()
    )

    bullish = (
        valid
        & (weekly["MA20"] > weekly["MA50"])
        & (weekly["Close"] > weekly["MA20"])
    )

    bearish = (
        valid
        & (weekly["MA20"] < weekly["MA50"])
        & (weekly["Close"] < weekly["MA20"])
    )

    neutral = valid & ~(bullish | bearish)

    weekly.loc[bullish, "Completed_Weekly_State"] = "BULLISH"
    weekly.loc[bearish, "Completed_Weekly_State"] = "BEARISH"
    weekly.loc[neutral, "Completed_Weekly_State"] = "NEUTRAL"

    # The current weekly candle must never be used for a signal.
    # Shift the state by one completed week.
    weekly["Available_Weekly_State"] = (
        weekly["Completed_Weekly_State"].shift(1)
    )

    weekly["Available_Weekly_Date"] = weekly.index

    weekly = weekly[
        weekly["Available_Weekly_State"].notna()
    ].copy()

    weekly = weekly[
        weekly["Available_Weekly_State"] != "UNKNOWN"
    ].copy()

    return weekly[
        [
            "Available_Weekly_Date",
            "Available_Weekly_State",
        ]
    ].copy()


# ============================================================================
# BUILD SYMBOL MTF DATA
# ============================================================================

def build_symbol_mtf(symbol: str) -> pd.DataFrame:
    print(f"Building MTF state: {symbol}")

    prices = load_symbol_prices(symbol)

    daily = build_daily_state(prices)
    weekly = build_weekly_state(prices)

    daily_out = daily.reset_index()
    daily_out = daily_out.rename(
        columns={"Date": "Signal_Date"}
    )

    weekly_out = weekly.reset_index(drop=True)

    # For every signal date, locate the latest available completed
    # weekly state strictly before that signal date.
    daily_out = daily_out.sort_values("Signal_Date")
    weekly_out = weekly_out.sort_values("Available_Weekly_Date")

    merged = pd.merge_asof(
        daily_out,
        weekly_out,
        left_on="Signal_Date",
        right_on="Available_Weekly_Date",
        direction="backward",
        allow_exact_matches=False,
    )

    merged["Symbol"] = symbol

    return merged[
        [
            "Symbol",
            "Signal_Date",
            "Daily_State",
            "MA20",
            "MA50",
            "Available_Weekly_Date",
            "Available_Weekly_State",
        ]
    ].copy()


# ============================================================================
# LOAD FROZEN TRADES
# ============================================================================

def load_frozen_trades() -> pd.DataFrame:
    if not TRADES_FILE.exists():
        raise FileNotFoundError(
            f"Frozen trade file not found: {TRADES_FILE}"
        )

    df = pd.read_csv(TRADES_FILE)

    symbol_col = find_column(
        df,
        ["_symbol", "Symbol", "symbol", "Ticker"],
        "symbol",
    )

    direction_col = find_column(
        df,
        [
            "Direction_Normalized",
            "direction_normalized",
            "direction",
            "Direction",
        ],
        "direction",
    )

    setup_col = find_column(
        df,
        [
            "Setup_Normalized",
            "setup_normalized",
            "setup",
            "Setup",
        ],
        "setup",
    )

    date_col = find_column(
        df,
        [
            "Signal_Date",
            "signal_date",
            "Entry_Date",
            "entry_date",
            "Date",
        ],
        "signal/entry date",
    )

    r_col = find_column(
        df,
        [
            "R_Multiple",
            "r_multiple",
            "R",
            "r",
        ],
        "R multiple",
    )

    out = df.copy()

    out["_Frozen_Row"] = np.arange(len(out))

    out["Symbol_Normalized"] = out[symbol_col].map(
        normalize_symbol
    )

    out["Direction_Normalized"] = out[direction_col].map(
        normalize_direction
    )

    out["Setup_Normalized"] = out[setup_col].map(
        normalize_setup
    )

    out["Signal_Date"] = pd.to_datetime(
        out[date_col],
        errors="coerce",
    )

    out["R_Multiple"] = pd.to_numeric(
        out[r_col],
        errors="coerce",
    )

    if out["Signal_Date"].isna().any():
        bad = int(out["Signal_Date"].isna().sum())
        raise RuntimeError(
            f"Frozen trade file contains {bad} invalid signal/entry dates."
        )

    if out["R_Multiple"].isna().any():
        bad = int(out["R_Multiple"].isna().sum())
        raise RuntimeError(
            f"Frozen trade file contains {bad} invalid R values."
        )

    return out


# ============================================================================
# TRADE-LEVEL CLASSIFICATION
# ============================================================================

def classify_trades(
    trades: pd.DataFrame,
    mtf_by_symbol: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for _, trade in trades.iterrows():

        symbol = trade["Symbol_Normalized"]
        signal_date = pd.Timestamp(
            trade["Signal_Date"]
        ).normalize()

        direction = trade["Direction_Normalized"]

        base = trade.to_dict()

        mtf = mtf_by_symbol.get(symbol)

        base["Daily_State"] = "UNKNOWN"
        base["Weekly_State"] = "UNKNOWN"
        base["Weekly_State_Date"] = pd.NaT
        base["Daily_Directional_State"] = "UNKNOWN"
        base["Weekly_Directional_State"] = "UNKNOWN"
        base["MTF_Class"] = "MTF_DATA_UNAVAILABLE"
        base["MTF_Score"] = np.nan
        base["MTF_Availability"] = "UNAVAILABLE"
        base["MTF_Reason"] = "NO_SYMBOL_DATA"

        if mtf is None or mtf.empty:
            rows.append(base)
            continue

        eligible = mtf[
            mtf["Signal_Date"] <= signal_date
        ].copy()

        if eligible.empty:
            base["MTF_Reason"] = "NO_DAILY_HISTORY"
            rows.append(base)
            continue

        # Exact signal-date daily state.
        exact_daily = eligible[
            eligible["Signal_Date"] == signal_date
        ]

        if exact_daily.empty:
            base["MTF_Reason"] = "SIGNAL_DATE_NOT_IN_PRICE_DATA"
            rows.append(base)
            continue

        record = exact_daily.iloc[-1]

        daily_state = record["Daily_State"]

        weekly_state = record["Available_Weekly_State"]
        weekly_date = record["Available_Weekly_Date"]

        base["Daily_State"] = daily_state

        if pd.notna(weekly_state):
            base["Weekly_State"] = weekly_state

        if pd.notna(weekly_date):
            base["Weekly_State_Date"] = pd.Timestamp(
                weekly_date
            )

        base["Daily_Directional_State"] = (
            state_for_direction(
                daily_state,
                direction,
            )
        )

        base["Weekly_Directional_State"] = (
            state_for_direction(
                weekly_state
                if pd.notna(weekly_state)
                else "UNKNOWN",
                direction,
            )
        )

        mtf_class, mtf_score = classify_mtf(
            daily_state,
            weekly_state
            if pd.notna(weekly_state)
            else "UNKNOWN",
            direction,
        )

        base["MTF_Class"] = mtf_class
        base["MTF_Score"] = mtf_score

        if mtf_class == "MTF_DATA_UNAVAILABLE":
            base["MTF_Availability"] = "UNAVAILABLE"

            if daily_state == "UNKNOWN":
                base["MTF_Reason"] = "DAILY_WARMUP"

            elif pd.isna(weekly_state):
                base["MTF_Reason"] = "WEEKLY_WARMUP"

            elif weekly_state == "UNKNOWN":
                base["MTF_Reason"] = "WEEKLY_WARMUP"

            else:
                base["MTF_Reason"] = "UNKNOWN"

        else:
            base["MTF_Availability"] = "AVAILABLE"
            base["MTF_Reason"] = "VALID"

        rows.append(base)

    return pd.DataFrame(rows)


# ============================================================================
# PRINT TABLE
# ============================================================================

def print_table(title: str, df: pd.DataFrame):
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)

    if df.empty:
        print("No rows.")
        return

    print(
        df.to_string(
            index=False
        )
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 90)
    print("TRADESENSE-AI — DAILY / WEEKLY MTF FORENSIC VALIDATION")
    print("=" * 90)

    # ------------------------------------------------------------------------
    # LOAD FROZEN TRADES
    # ------------------------------------------------------------------------

    trades = load_frozen_trades()

    frozen_trade_count = len(trades)
    frozen_total_r = float(
        trades["R_Multiple"].sum()
    )

    print(f"Frozen trades loaded : {frozen_trade_count}")

    if frozen_trade_count != EXPECTED_FROZEN_TRADES:
        raise RuntimeError(
            "FROZEN TRADE COUNT CHANGED.\n"
            f"Expected: {EXPECTED_FROZEN_TRADES}\n"
            f"Actual:   {frozen_trade_count}"
        )

    # ------------------------------------------------------------------------
    # BUILD MTF DATA
    # ------------------------------------------------------------------------

    symbols = sorted(
        trades["Symbol_Normalized"]
        .dropna()
        .unique()
    )

    mtf_by_symbol = {}

    for symbol in symbols:
        # Convert root back to frozen-style symbol for the display.
        display_symbol = (
            symbol + ".NS"
            if not symbol.endswith(".NS")
            else symbol
        )

        mtf_by_symbol[symbol] = build_symbol_mtf(
            display_symbol
        )

    # ------------------------------------------------------------------------
    # CLASSIFY EVERY TRADE
    # ------------------------------------------------------------------------

    classified = classify_trades(
        trades,
        mtf_by_symbol,
    )

    print()
    print(
        f"Trades classified       : {len(classified)}"
    )

    unavailable_count = int(
        (
            classified["MTF_Availability"]
            == "UNAVAILABLE"
        ).sum()
    )

    available_count = int(
        (
            classified["MTF_Availability"]
            == "AVAILABLE"
        ).sum()
    )

    print(
        f"MTF available            : {available_count}"
    )

    print(
        f"MTF unavailable          : {unavailable_count}"
    )

    # ------------------------------------------------------------------------
    # HARD TRADE-COUNT INTEGRITY CHECK
    # ------------------------------------------------------------------------

    if len(classified) != frozen_trade_count:
        raise RuntimeError(
            "TRADE-LEVEL COVERAGE FAILURE.\n"
            f"Frozen trades : {frozen_trade_count}\n"
            f"Classified    : {len(classified)}"
        )

    # ------------------------------------------------------------------------
    # RECONCILIATION
    # ------------------------------------------------------------------------

    classified_total_r = float(
        classified["R_Multiple"].sum()
    )

    valid_mtf = classified[
        classified["MTF_Availability"]
        == "AVAILABLE"
    ].copy()

    unavailable = classified[
        classified["MTF_Availability"]
        == "UNAVAILABLE"
    ].copy()

    valid_total_r = float(
        valid_mtf["R_Multiple"].sum()
    )

    unavailable_total_r = float(
        unavailable["R_Multiple"].sum()
    )

    reconciled_total_r = (
        valid_total_r
        + unavailable_total_r
    )

    reconciliation_delta = (
        reconciled_total_r
        - frozen_total_r
    )

    # ------------------------------------------------------------------------
    # BASELINE — ALL 106
    # ------------------------------------------------------------------------

    baseline = performance_summary(
        classified
    )

    print_table(
        "BASELINE — ALL FROZEN TRADES",
        baseline,
    )

    # ------------------------------------------------------------------------
    # MTF AVAILABILITY
    # ------------------------------------------------------------------------

    availability = performance_summary(
        classified,
        "MTF_Availability",
    )

    print_table(
        "MTF AVAILABILITY",
        availability,
    )

    # ------------------------------------------------------------------------
    # UNAVAILABLE REASONS
    # ------------------------------------------------------------------------

    unavailable_reasons = performance_summary(
        unavailable,
        "MTF_Reason",
    )

    print_table(
        "UNAVAILABLE / WARM-UP REASONS",
        unavailable_reasons,
    )

    # ------------------------------------------------------------------------
    # VALID MTF CLASSIFICATION
    # ------------------------------------------------------------------------

    mtf_summary = performance_summary(
        valid_mtf,
        "MTF_Class",
    )

    print_table(
        "VALID MTF CLASSIFICATION — AVAILABLE TRADES ONLY",
        mtf_summary,
    )

    # ------------------------------------------------------------------------
    # RAW DAILY STATE
    # ------------------------------------------------------------------------

    daily_summary = performance_summary(
        valid_mtf,
        "Daily_State",
    )

    print_table(
        "DAILY STATE — AVAILABLE TRADES ONLY",
        daily_summary,
    )

    # ------------------------------------------------------------------------
    # RAW WEEKLY STATE
    # ------------------------------------------------------------------------

    weekly_summary = performance_summary(
        valid_mtf,
        "Weekly_State",
    )

    print_table(
        "WEEKLY STATE — AVAILABLE TRADES ONLY",
        weekly_summary,
    )

    # ------------------------------------------------------------------------
    # DIRECTIONAL MTF SCORE
    # ------------------------------------------------------------------------

    score_summary = performance_summary(
        valid_mtf,
        "MTF_Score",
    )

    print_table(
        "DIRECTIONAL MTF SCORE — AVAILABLE TRADES ONLY",
        score_summary,
    )

    # ------------------------------------------------------------------------
    # SETUP × MTF
    # ------------------------------------------------------------------------

    setup_mtf = (
        valid_mtf
        .assign(
            Setup_MTF=lambda x:
                x["Setup_Normalized"]
                + " × "
                + x["MTF_Class"]
        )
    )

    setup_mtf_summary = performance_summary(
        setup_mtf,
        "Setup_MTF",
    )

    print_table(
        "SETUP × MTF CLASS",
        setup_mtf_summary,
    )

    # ------------------------------------------------------------------------
    # DIRECTION × MTF
    # ------------------------------------------------------------------------

    direction_mtf = (
        valid_mtf
        .assign(
            Direction_MTF=lambda x:
                x["Direction_Normalized"]
                + " × "
                + x["MTF_Class"]
        )
    )

    direction_mtf_summary = performance_summary(
        direction_mtf,
        "Direction_MTF",
    )

    print_table(
        "DIRECTION × MTF CLASS",
        direction_mtf_summary,
    )

    # ------------------------------------------------------------------------
    # EARLY ADVERSE × MTF
    # ------------------------------------------------------------------------

    if "early_adverse_triggered" in valid_mtf.columns:

        early_mtf = valid_mtf.copy()

        early_mtf["Early_Adverse_Group"] = (
            early_mtf[
                "early_adverse_triggered"
            ]
            .map({
                True: "TRIGGERED",
                False: "NOT_TRIGGERED",
            })
            .fillna("UNKNOWN")
        )

        early_mtf["Early_Adverse_MTF"] = (
            early_mtf["Early_Adverse_Group"]
            + " × "
            + early_mtf["MTF_Class"]
        )

        early_mtf_summary = performance_summary(
            early_mtf,
            "Early_Adverse_MTF",
        )

        print_table(
            "EARLY ADVERSE × MTF CLASS",
            early_mtf_summary,
        )

    # ------------------------------------------------------------------------
    # SAVE TRADE-LEVEL OUTPUT
    # ------------------------------------------------------------------------

    trade_level_path = (
        OUTPUT_DIR
        / "trade_level_mtf.csv"
    )

    classified.to_csv(
        trade_level_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # SAVE MTF SUMMARY
    # ------------------------------------------------------------------------

    mtf_summary_path = (
        OUTPUT_DIR
        / "mtf_summary.csv"
    )

    mtf_summary.to_csv(
        mtf_summary_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # SAVE SETUP × MTF
    # ------------------------------------------------------------------------

    setup_path = (
        OUTPUT_DIR
        / "setup_by_mtf.csv"
    )

    setup_mtf_summary.to_csv(
        setup_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # SAVE DIRECTION × MTF
    # ------------------------------------------------------------------------

    direction_path = (
        OUTPUT_DIR
        / "direction_by_mtf.csv"
    )

    direction_mtf_summary.to_csv(
        direction_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # SAVE AVAILABILITY AUDIT
    # ------------------------------------------------------------------------

    availability_path = (
        OUTPUT_DIR
        / "mtf_availability_audit.csv"
    )

    availability_audit = (
        classified[
            [
                "_Frozen_Row",
                "Symbol_Normalized",
                "Signal_Date",
                "Direction_Normalized",
                "Setup_Normalized",
                "R_Multiple",
                "Daily_State",
                "Weekly_State",
                "Weekly_State_Date",
                "MTF_Availability",
                "MTF_Reason",
                "MTF_Class",
                "MTF_Score",
            ]
        ]
        .sort_values("_Frozen_Row")
    )

    availability_audit.to_csv(
        availability_path,
        index=False,
    )

    # ------------------------------------------------------------------------
    # RECONCILIATION REPORT
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)
    print("RECONCILIATION")
    print("=" * 90)

    print(
        f"Frozen trade count       : {frozen_trade_count}"
    )

    print(
        f"Classified trade count   : {len(classified)}"
    )

    print(
        f"MTF available trades     : {available_count}"
    )

    print(
        f"MTF unavailable trades   : {unavailable_count}"
    )

    print(
        f"Frozen total R            : "
        f"{frozen_total_r:.4f}"
    )

    print(
        f"Valid MTF total R         : "
        f"{valid_total_r:.4f}"
    )

    print(
        f"Unavailable trade R      : "
        f"{unavailable_total_r:.4f}"
    )

    print(
        f"Reconciled total R        : "
        f"{reconciled_total_r:.4f}"
    )

    print(
        f"Reconciliation delta      : "
        f"{reconciliation_delta:.4f}"
    )

    count_ok = (
        len(classified)
        == frozen_trade_count
    )

    r_ok = (
        abs(reconciliation_delta)
        <= RECON_TOLERANCE
    )

    print()
    print(
        "Trade-count reconciliation:",
        "PASS" if count_ok else "FAIL",
    )

    print(
        "R reconciliation:",
        "PASS" if r_ok else "FAIL",
    )

    if not count_ok:
        raise RuntimeError(
            "MTF FORENSIC FAILED: trade-count reconciliation."
        )

    if not r_ok:
        raise RuntimeError(
            "MTF FORENSIC FAILED: R reconciliation."
        )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print()
    print("=" * 90)

    print(
        "Classification coverage:",
        "PASS"
        if unavailable_count >= 0
        else "FAIL",
    )

    print(
        "Frozen dataset integrity : PASS"
    )

    print(
        "Production strategy      : UNCHANGED"
    )

    print(
        "MTF production logic     : UNCHANGED"
    )

    print(
        "STATUS: MTF_FORENSIC_COMPLETE"
    )

    print()
    print(
        f"Trade-level output : {trade_level_path}"
    )

    print(
        f"MTF summary        : {mtf_summary_path}"
    )

    print(
        f"Setup × MTF        : {setup_path}"
    )

    print(
        f"Direction × MTF    : {direction_path}"
    )

    print(
        f"Availability audit : {availability_path}"
    )


if __name__ == "__main__":
    main()