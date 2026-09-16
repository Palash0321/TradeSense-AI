"""
TradeSense-AI — Forensic Early-Adverse Path / First-5-Bar Microstructure
========================================================================

RESEARCH-ONLY FORENSIC SCRIPT

Frozen hypothesis:
    5-bar / 0.25R early-adverse event

Frozen population:
    106 total trades
     60 early-adverse events
     46 retained trades

Purpose:
    Reconstruct the exact first 1–5 post-entry OHLC bars and establish
    the foundational path dataset that later forensic analyses will use.

This script must NOT:
    - modify production strategy logic
    - modify the frozen trade dataset
    - modify OOS / holdout data
    - use final exit information to define the event
    - promote a deployment rule

The event label is reconstructed only from:
    cumulative adverse excursion through bar 5 >= 0.25R

The entry/risk reconstruction intentionally follows the established
early_adverse_ohlc_replay forensic boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

HISTORICAL_TRADES_FILE = "tests/output/tradesense_trades.csv"

OUTPUT_DIR = Path(
    "tests/output/research/early_adverse_path"
)

PATH_OUTPUT = OUTPUT_DIR / "early_adverse_path_per_trade_bar.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "early_adverse_path_validation.csv"
INTEGRITY_OUTPUT = OUTPUT_DIR / "early_adverse_path_integrity.csv"

EXPECTED_TRADE_COUNT = 106
EXPECTED_EVENT_COUNT = 60
EXPECTED_RETAINED_COUNT = 46

EARLY_ADVERSE_THRESHOLD_R = 0.25

MAX_BARS = 5

# Exact production-style entry reconstruction.
DEFAULT_SLIPPAGE_PCT = 0.001

# Exact production-style initial stop construction.
INITIAL_STOP_ATR = 3.5

# Same frozen historical boundary as the established OHLC replay.
START_DATE = "2021-09-07"
END_DATE = "2026-09-08"

VALIDATION_TOLERANCE = 0.005001


# ============================================================================
# BASIC HELPERS
# ============================================================================

def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def normalize_date(value) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    try:
        return pd.to_datetime(text).strftime("%Y-%m-%d")

    except Exception:
        return text[:10]


def normalize_symbol(value) -> str:
    if value is None:
        return ""

    return str(value).strip().upper()


def almost_equal(
    a: float,
    b: float,
    tolerance: float = VALIDATION_TOLERANCE,
) -> bool:

    if pd.isna(a) and pd.isna(b):
        return True

    if pd.isna(a) or pd.isna(b):
        return False

    return abs(float(a) - float(b)) <= tolerance


def get_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# ============================================================================
# OHLC COLUMN RESOLUTION
# ============================================================================

def get_ohlc_columns(
    df: pd.DataFrame,
) -> Tuple[str, str, str, str]:

    open_col = get_first_existing_column(
        df,
        ["open", "Open"],
    )

    high_col = get_first_existing_column(
        df,
        ["high", "High"],
    )

    low_col = get_first_existing_column(
        df,
        ["low", "Low"],
    )

    close_col = get_first_existing_column(
        df,
        ["close", "Close"],
    )

    if open_col is None:
        raise KeyError(
            "Could not locate OPEN column in OHLC data."
        )

    if high_col is None:
        raise KeyError(
            "Could not locate HIGH column in OHLC data."
        )

    if low_col is None:
        raise KeyError(
            "Could not locate LOW column in OHLC data."
        )

    if close_col is None:
        raise KeyError(
            "Could not locate CLOSE column in OHLC data."
        )

    return (
        open_col,
        high_col,
        low_col,
        close_col,
    )


# ============================================================================
# PRODUCTION ATR RECONSTRUCTION
# ============================================================================

def calculate_production_atr(
    ohlc: pd.DataFrame,
) -> pd.Series:

    """
    Exact ATR reconstruction used by the established OHLC replay:

        True Range =
            max(
                High - Low,
                abs(High - Previous Close),
                abs(Low - Previous Close)
            )

        ATR =
            14-period simple rolling mean of True Range
    """

    (
        open_col,
        high_col,
        low_col,
        close_col,
    ) = get_ohlc_columns(ohlc)

    high = pd.to_numeric(
        ohlc[high_col],
        errors="coerce",
    )

    low = pd.to_numeric(
        ohlc[low_col],
        errors="coerce",
    )

    close = pd.to_numeric(
        ohlc[close_col],
        errors="coerce",
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        14,
        min_periods=14,
    ).mean()


# ============================================================================
# EXCURSION CALCULATIONS
# ============================================================================

def adverse_r_from_bar(
    direction: str,
    entry_price: float,
    low: float,
    high: float,
    risk: float,
) -> float:

    if (
        pd.isna(entry_price)
        or pd.isna(low)
        or pd.isna(high)
        or pd.isna(risk)
        or risk <= 0
    ):
        return np.nan

    direction = str(direction).upper()

    if direction == "LONG":

        adverse_move = max(
            0.0,
            entry_price - low,
        )

    elif direction == "SHORT":

        adverse_move = max(
            0.0,
            high - entry_price,
        )

    else:
        return np.nan

    return adverse_move / risk


def favorable_r_from_bar(
    direction: str,
    entry_price: float,
    low: float,
    high: float,
    risk: float,
) -> float:

    if (
        pd.isna(entry_price)
        or pd.isna(low)
        or pd.isna(high)
        or pd.isna(risk)
        or risk <= 0
    ):
        return np.nan

    direction = str(direction).upper()

    if direction == "LONG":

        favorable_move = max(
            0.0,
            high - entry_price,
        )

    elif direction == "SHORT":

        favorable_move = max(
            0.0,
            entry_price - low,
        )

    else:
        return np.nan

    return favorable_move / risk


# ============================================================================
# SETUP COLUMN DETECTION
# ============================================================================

def get_setup_column(
    trades: pd.DataFrame,
) -> Optional[str]:

    return get_first_existing_column(
        trades,
        [
            "setup",
            "setup_name",
            "setup_type",
            "signal_setup",
            "trade_setup",
        ],
    )


# ============================================================================
# FROZEN TRADE DATASET
# ============================================================================

def load_frozen_trades() -> pd.DataFrame:

    if not os.path.exists(
        HISTORICAL_TRADES_FILE
    ):
        raise FileNotFoundError(
            "Frozen trade file not found:\n"
            f"    {HISTORICAL_TRADES_FILE}"
        )

    trades = pd.read_csv(
        HISTORICAL_TRADES_FILE
    )

    print(
        f"Loaded frozen trade dataset: {len(trades)} rows"
    )

    if len(trades) != EXPECTED_TRADE_COUNT:

        raise RuntimeError(
            "\n"
            "FROZEN POPULATION GUARD FAILED\n"
            f"Expected: {EXPECTED_TRADE_COUNT}\n"
            f"Observed: {len(trades)}\n"
            "\n"
            "STOP. Do not continue forensic analysis."
        )

    return trades


# ============================================================================
# OHLC DATA LOADING
# ============================================================================

def load_symbol_ohlc(
    symbol: str,
) -> pd.DataFrame:

    try:

        from app.services.backtest_service import (
            BacktestService
        )

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "Could not import BacktestService from:\n"
            "app.services.backtest_service\n"
            "\n"
            "Run this script from the TradeSense-AI project root."
        ) from exc

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )
    ohlc = service.load_data(
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if ohlc is None or len(ohlc) == 0:

        raise RuntimeError(
            f"No OHLC data returned for symbol: {symbol}"
        )

    ohlc = ohlc.copy()

    ohlc.index = pd.to_datetime(
        ohlc.index
    )

    ohlc = ohlc.sort_index()

    (
        open_col,
        high_col,
        low_col,
        close_col,
    ) = get_ohlc_columns(ohlc)

    ohlc["__path_open__"] = pd.to_numeric(
        ohlc[open_col],
        errors="coerce",
    )

    ohlc["__path_high__"] = pd.to_numeric(
        ohlc[high_col],
        errors="coerce",
    )

    ohlc["__path_low__"] = pd.to_numeric(
        ohlc[low_col],
        errors="coerce",
    )

    ohlc["__path_close__"] = pd.to_numeric(
        ohlc[close_col],
        errors="coerce",
    )

    ohlc["__path_atr__"] = (
        calculate_production_atr(ohlc)
    )

    return ohlc


# ============================================================================
# EXACT ENTRY RECONSTRUCTION
# ============================================================================

def reconstruct_entry(
    trade: pd.Series,
    ohlc: pd.DataFrame,
    symbol_column: str,
) -> Dict[str, object]:

    symbol = normalize_symbol(
        trade.get(symbol_column)
    )

    direction = str(
        trade.get("direction", "")
    ).strip().upper()

    entry_date = normalize_date(
        trade.get("entry_date")
    )

    matching = ohlc[
        ohlc.index.strftime("%Y-%m-%d")
        == entry_date
    ]

    if matching.empty:

        raise RuntimeError(
            f"Could not locate entry date "
            f"{entry_date} for {symbol}"
        )

    entry_timestamp = matching.index[0]

    entry_position = ohlc.index.get_loc(
        entry_timestamp
    )

    entry_row = ohlc.iloc[
        entry_position
    ]

    entry_open = safe_float(
        entry_row["__path_open__"]
    )

    signal_position = (
        entry_position - 1
    )

    if signal_position < 0:

        raise RuntimeError(
            f"No signal bar available before "
            f"entry for {symbol} on {entry_date}"
        )

    signal_timestamp = ohlc.index[
        signal_position
    ]

    entry_atr = safe_float(
        ohlc.iloc[
            signal_position
        ]["__path_atr__"]
    )

    if (
        pd.isna(entry_open)
        or pd.isna(entry_atr)
        or entry_atr <= 0
    ):

        raise RuntimeError(
            f"Invalid entry reconstruction for "
            f"{symbol} {entry_date}: "
            f"open={entry_open}, "
            f"ATR={entry_atr}"
        )

    if direction == "LONG":

        entry_price = (
            entry_open
            * (1.0 + DEFAULT_SLIPPAGE_PCT)
        )

    elif direction == "SHORT":

        entry_price = (
            entry_open
            * (1.0 - DEFAULT_SLIPPAGE_PCT)
        )

    else:

        raise RuntimeError(
            f"Unknown direction '{direction}' "
            f"for {symbol} {entry_date}"
        )

    risk_per_unit = (
        INITIAL_STOP_ATR
        * entry_atr
    )

    return {
        "symbol": symbol,
        "direction": direction,
        "entry_date": entry_date,
        "entry_timestamp": entry_timestamp,
        "signal_timestamp": signal_timestamp,
        "entry_position": entry_position,
        "entry_open": entry_open,
        "entry_price_reconstructed": entry_price,
        "entry_atr_exact": entry_atr,
        "initial_stop_atr": INITIAL_STOP_ATR,
        "risk_per_unit": risk_per_unit,
    }


# ============================================================================
# FIRST-5-BAR PATH RECONSTRUCTION
# ============================================================================

def reconstruct_first_five_bars(
    trade: pd.Series,
    ohlc: pd.DataFrame,
    setup_column: Optional[str],
    symbol_column: str,
):

    entry = reconstruct_entry(
        trade=trade,
        ohlc=ohlc,
        symbol_column=symbol_column,
    )

    symbol = entry["symbol"]
    direction = entry["direction"]
    entry_price = float(
        entry["entry_price_reconstructed"]
    )
    risk = float(
        entry["risk_per_unit"]
    )
    entry_atr = float(
        entry["entry_atr_exact"]
    )
    entry_position = int(
        entry["entry_position"]
    )

    if setup_column is not None:

        setup = trade.get(
            setup_column
        )

        if pd.isna(setup):

            setup = "UNKNOWN"

        else:

            setup = str(setup).strip()

    else:

        setup = "UNKNOWN"

    cumulative_mae_r = 0.0
    cumulative_mfe_r = 0.0

    rows = []

    for bar_number in range(
        1,
        MAX_BARS + 1,
    ):

        bar_position = (
            entry_position
            + bar_number
            - 1
        )

        if bar_position >= len(ohlc):

            raise RuntimeError(
                f"Insufficient OHLC bars for "
                f"{symbol} {entry['entry_date']}"
            )

        row = ohlc.iloc[
            bar_position
        ]

        timestamp = ohlc.index[
            bar_position
        ]

        open_price = safe_float(
            row["__path_open__"]
        )

        high = safe_float(
            row["__path_high__"]
        )

        low = safe_float(
            row["__path_low__"]
        )

        close = safe_float(
            row["__path_close__"]
        )

        if any(
            pd.isna(value)
            for value in [
                open_price,
                high,
                low,
                close,
            ]
        ):

            raise RuntimeError(
                f"Invalid OHLC values at "
                f"{symbol} {timestamp}"
            )

        # --------------------------------------------------------------
        # Single-bar excursions
        # --------------------------------------------------------------

        bar_adverse_r = adverse_r_from_bar(
            direction=direction,
            entry_price=entry_price,
            low=low,
            high=high,
            risk=risk,
        )

        bar_favorable_r = favorable_r_from_bar(
            direction=direction,
            entry_price=entry_price,
            low=low,
            high=high,
            risk=risk,
        )

        # --------------------------------------------------------------
        # Cumulative excursions
        # --------------------------------------------------------------

        previous_mae = cumulative_mae_r
        previous_mfe = cumulative_mfe_r

        cumulative_mae_r = max(
            cumulative_mae_r,
            bar_adverse_r,
        )

        cumulative_mfe_r = max(
            cumulative_mfe_r,
            bar_favorable_r,
        )

        incremental_mae_r = (
            cumulative_mae_r
            - previous_mae
        )

        incremental_mfe_r = (
            cumulative_mfe_r
            - previous_mfe
        )

        # --------------------------------------------------------------
        # Direction-normalized close movement
        # --------------------------------------------------------------

        if direction == "LONG":

            close_move_r = (
                close - open_price
            ) / risk

        else:

            close_move_r = (
                open_price - close
            ) / risk

        if close_move_r > 0:

            close_direction = "FAVORABLE"

        elif close_move_r < 0:

            close_direction = "ADVERSE"

        else:

            close_direction = "FLAT"

        # --------------------------------------------------------------
        # Candle geometry
        # --------------------------------------------------------------

        range_price = max(
            0.0,
            high - low,
        )

        body_price = abs(
            close - open_price
        )

        upper_wick_price = max(
            0.0,
            high - max(
                open_price,
                close,
            ),
        )

        lower_wick_price = max(
            0.0,
            min(
                open_price,
                close,
            ) - low,
        )

        if range_price > 0:

            body_pct_range = (
                body_price
                / range_price
            )

            upper_wick_pct_range = (
                upper_wick_price
                / range_price
            )

            lower_wick_pct_range = (
                lower_wick_price
                / range_price
            )

            close_location = (
                close - low
            ) / range_price

        else:

            body_pct_range = np.nan
            upper_wick_pct_range = np.nan
            lower_wick_pct_range = np.nan
            close_location = np.nan

        # --------------------------------------------------------------
        # Direction-normalized wick geometry
        # --------------------------------------------------------------

        if direction == "LONG":

            adverse_wick_price = (
                lower_wick_price
            )

            favorable_wick_price = (
                upper_wick_price
            )

        else:

            adverse_wick_price = (
                upper_wick_price
            )

            favorable_wick_price = (
                lower_wick_price
            )

        # --------------------------------------------------------------
        # ATR-normalized measurements
        # --------------------------------------------------------------

        bar_range_atr = (
            range_price
            / entry_atr
        )

        body_atr = (
            body_price
            / entry_atr
        )

        adverse_wick_atr = (
            adverse_wick_price
            / entry_atr
        )

        favorable_wick_atr = (
            favorable_wick_price
            / entry_atr
        )

        bar_adverse_atr = (
            (bar_adverse_r * risk)
            / entry_atr
        )

        bar_favorable_atr = (
            (bar_favorable_r * risk)
            / entry_atr
        )

        # --------------------------------------------------------------
        # Entry-relative displacement
        # --------------------------------------------------------------

        if direction == "LONG":

            open_displacement_r = (
                open_price - entry_price
            ) / risk

            close_from_entry_r = (
                close - entry_price
            ) / risk

        else:

            open_displacement_r = (
                entry_price - open_price
            ) / risk

            close_from_entry_r = (
                entry_price - close
            ) / risk

        # --------------------------------------------------------------
        # Net path state
        # --------------------------------------------------------------

        net_excursion_r = (
            cumulative_mfe_r
            - cumulative_mae_r
        )

        if cumulative_mae_r > 0:

            mfe_to_mae_ratio = (
                cumulative_mfe_r
                / cumulative_mae_r
            )

        else:

            mfe_to_mae_ratio = np.nan

        # --------------------------------------------------------------
        # Threshold state
        # --------------------------------------------------------------

        crossed_025r = int(
            cumulative_mae_r
            >= EARLY_ADVERSE_THRESHOLD_R
        )

        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "entry_date": entry["entry_date"],
                "entry_timestamp": entry["entry_timestamp"],
                "signal_timestamp": entry["signal_timestamp"],
                "setup": setup,

                "bar_number": bar_number,
                "bar_timestamp": timestamp,

                "entry_open": entry["entry_open"],
                "entry_price_reconstructed": entry_price,
                "entry_atr_exact": entry_atr,
                "initial_stop_atr": INITIAL_STOP_ATR,
                "risk_per_unit": risk,

                "open": open_price,
                "high": high,
                "low": low,
                "close": close,

                # Excursion path
                "bar_adverse_r": bar_adverse_r,
                "bar_favorable_r": bar_favorable_r,
                "cumulative_mae_r": cumulative_mae_r,
                "cumulative_mfe_r": cumulative_mfe_r,
                "incremental_mae_r": incremental_mae_r,
                "incremental_mfe_r": incremental_mfe_r,
                "net_excursion_r": net_excursion_r,
                "mfe_to_mae_ratio": mfe_to_mae_ratio,

                # Entry-relative movement
                "open_displacement_r": open_displacement_r,
                "close_from_entry_r": close_from_entry_r,
                "close_move_r": close_move_r,
                "close_direction": close_direction,

                # Candle structure
                "range_price": range_price,
                "body_price": body_price,
                "upper_wick_price": upper_wick_price,
                "lower_wick_price": lower_wick_price,
                "adverse_wick_price": adverse_wick_price,
                "favorable_wick_price": favorable_wick_price,
                "body_pct_range": body_pct_range,
                "upper_wick_pct_range": upper_wick_pct_range,
                "lower_wick_pct_range": lower_wick_pct_range,
                "close_location": close_location,

                # ATR normalized
                "bar_range_atr": bar_range_atr,
                "body_atr": body_atr,
                "adverse_wick_atr": adverse_wick_atr,
                "favorable_wick_atr": favorable_wick_atr,
                "bar_adverse_atr": bar_adverse_atr,
                "bar_favorable_atr": bar_favorable_atr,

                # Frozen threshold state
                "crossed_0_25r_by_this_bar": crossed_025r,
            }
        )

    return rows, entry


# ============================================================================
# STORED BAR-5 VALIDATION
# ============================================================================

def find_stored_bar5_column(
    trades: pd.DataFrame,
) -> Optional[str]:

    return get_first_existing_column(
        trades,
        [
            "early_adverse_r_bar_5",
            "early_adverse_r_5",
            "early_adverse_bar_5",
        ],
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    print("=" * 78)
    print(
        "TradeSense-AI — Early-Adverse Path / First-5-Bar Forensic"
    )
    print("=" * 78)

    print()
    print(
        "Frozen hypothesis: 5-bar / 0.25R early-adverse event"
    )

    print(
        "Frozen population: "
        f"{EXPECTED_TRADE_COUNT} trades | "
        f"{EXPECTED_EVENT_COUNT} events | "
        f"{EXPECTED_RETAINED_COUNT} retained"
    )

    print(
        "Research boundary: production/frozen/OOS data unchanged"
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load frozen trades
    # ------------------------------------------------------------------

    trades = load_frozen_trades()

    setup_column = get_setup_column(
        trades
    )

    if setup_column is None:

        print(
            "[WARN] Setup column not detected."
        )

        print(
            "[WARN] Core path reconstruction will continue "
            "with setup='UNKNOWN'."
        )

    else:

        print(
            f"[PASS] Setup column: {setup_column}"
        )

    stored_bar5_column = find_stored_bar5_column(
        trades
    )

    if stored_bar5_column is None:

        raise RuntimeError(
            "\n"
            "Could not find the frozen stored bar-5 "
            "early-adverse field.\n"
            "\n"
            "Expected one of:\n"
            "  early_adverse_r_bar_5\n"
            "  early_adverse_r_5\n"
            "  early_adverse_bar_5\n"
            "\n"
            "STOP."
        )

    print(
        f"[PASS] Stored bar-5 field: "
        f"{stored_bar5_column}"
    )

    # ------------------------------------------------------------------
    # Load symbol OHLC
    # ------------------------------------------------------------------

    symbol_column = get_first_existing_column(
        trades,
        ["_symbol", "symbol", "Symbol", "SYMBOL"],
    )

    if symbol_column is None:
        print("[FAIL] Symbol column not found in frozen trade dataset.")
        print("Available columns:")
        for column in trades.columns:
            print(f"  - {column}")
        return 1

    print(f"[PASS] Symbol column: {symbol_column}")

    symbols = sorted(
        {
            normalize_symbol(symbol)
            for symbol in trades[symbol_column].dropna()
            if normalize_symbol(symbol)
        }
    )

    print()
    print(
        f"Unique frozen symbols: {len(symbols)}"
    )

    symbol_data = {}

    for symbol in symbols:

        print(
            f"Loading OHLC: {symbol}"
        )

        symbol_data[symbol] = (
            load_symbol_ohlc(symbol)
        )

    # ------------------------------------------------------------------
    # Reconstruct all first-five-bar paths
    # ------------------------------------------------------------------

    all_path_rows = []
    validation_rows = []

    print()
    print(
        "Reconstructing first 5 post-entry bars..."
    )

    for trade_index, trade in trades.iterrows():

        symbol = normalize_symbol(
            trade.get(symbol_column)
        )

        ohlc = symbol_data[
            symbol
        ]

        path_rows, entry = (
            reconstruct_first_five_bars(
                trade=trade,
                ohlc=ohlc,
                setup_column=setup_column,
                symbol_column=symbol_column,
            )
        )

        all_path_rows.extend(
            path_rows
        )

        # ------------------------------------------------------------------
        # Production observation boundary
        #
        # Production records early-adverse path diagnostics only while the
        # actual production trade remains open.
        #
        # Therefore a missing stored bar-5 value is NOT a replay mismatch
        # when the production trade exited before bar 5.
        # ------------------------------------------------------------------

        bars_in_trade = safe_float(
            trade.get("bars_in_trade")
        )

        if pd.isna(bars_in_trade):
            raise RuntimeError(
                f"Missing bars_in_trade for "
                f"{symbol} {entry['entry_date']}"
            )

        bars_in_trade = int(
            bars_in_trade
        )

        if bars_in_trade < 1:
            raise RuntimeError(
                f"Invalid bars_in_trade={bars_in_trade} "
                f"for {symbol} {entry['entry_date']}"
            )

        # ------------------------------------------------------------------
        # Frozen production event
        #
        # IMPORTANT:
        # The frozen event population comes from the production event label,
        # NOT from stored bar-5 availability.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # FROZEN EVENT LABEL
        #
        # The authoritative frozen 60/46 population is defined by the
        # STORED production bar-5 diagnostic:
        #
        #     early_adverse_r_bar_5 >= 0.25R
        #
        # NaN means production did not reach bar 5 and therefore does not
        # belong to the frozen event population.
        #
        # IMPORTANT:
        # early_adverse_triggered is a separate production 3-bar diagnostic
        # and is NOT the frozen 60/46 event definition.
        # ------------------------------------------------------------------

        stored_bar5 = safe_float(
            trade.get(
                stored_bar5_column
            )
        )

        stored_frozen_event = int(
            (
                not pd.isna(stored_bar5)
            )
            and (
                stored_bar5
                >= EARLY_ADVERSE_THRESHOLD_R
            )
        )

        # ------------------------------------------------------------------
        # Production 3-bar trigger diagnostic.
        #
        # This is retained separately for forensic comparison.
        # ------------------------------------------------------------------

        production_event_raw = trade.get(
            "early_adverse_triggered"
        )

        if pd.isna(production_event_raw):
            raise RuntimeError(
                f"Missing early_adverse_triggered for "
                f"{symbol} {entry['entry_date']}"
            )

        production_triggered = int(
            bool(production_event_raw)
        )

        production_trigger_bar = safe_float(
            trade.get(
                "early_adverse_trigger_bar"
            )
        )

        if pd.isna(production_trigger_bar):
            production_trigger_bar = np.nan
        else:
            production_trigger_bar = int(
                production_trigger_bar
            )

        production_trigger_bar = safe_float(
            trade.get(
                "early_adverse_trigger_bar"
            )
        )

        if pd.isna(production_trigger_bar):
            production_trigger_bar = np.nan
        else:
            production_trigger_bar = int(
                production_trigger_bar
            )

        # ------------------------------------------------------------------
        # Replay the production event definition.
        #
        # Production trigger:
        #   cumulative adverse excursion >= 0.25R
        #   within first 3 bars
        #   while trade is still open.
        # ------------------------------------------------------------------

        production_reconstructed_event = 0
        production_reconstructed_trigger_bar = np.nan

        observed_event_bars = min(
            3,
            bars_in_trade,
            len(path_rows),
        )

        for observed_bar_index in range(
            observed_event_bars
        ):

            cumulative_mae = float(
                path_rows[
                    observed_bar_index
                ][
                    "cumulative_mae_r"
                ]
            )

            if (
                cumulative_mae
                >= EARLY_ADVERSE_THRESHOLD_R
            ):

                production_reconstructed_event = 1

                production_reconstructed_trigger_bar = (
                    observed_bar_index + 1
                )

                break

        production_trigger_replay_match = (
            production_triggered
            == production_reconstructed_event
        )

        # ------------------------------------------------------------------
        # Extended 5-bar research event.
        #
        # This is deliberately SEPARATE from the frozen production event.
        #
        # It asks whether cumulative adverse excursion reaches 0.25R
        # anywhere during the extended first-five-bar replay, even if
        # production had already exited.
        # ------------------------------------------------------------------

        reconstructed_bar5 = float(
            path_rows[-1][
                "cumulative_mae_r"
            ]
        )

        extended_5bar_event = int(
            reconstructed_bar5
            >= EARLY_ADVERSE_THRESHOLD_R
        )

        # ------------------------------------------------------------------
        # Stored production bar-5 value
        # ------------------------------------------------------------------

        stored_bar5 = safe_float(
            trade.get(
                stored_bar5_column
            )
        )

        # ------------------------------------------------------------------
        # Bar-5 validation semantics
        #
        # CASE 1:
        # Production reached bar 5:
        #     stored bar-5 MUST exist and match replay.
        #
        # CASE 2:
        # Production exited before bar 5:
        #     stored bar-5 is legitimately NaN.
        #     Extended replay is retained for research but is NOT a
        #     production validation mismatch.
        # ------------------------------------------------------------------

        if bars_in_trade < 5:

            if pd.isna(stored_bar5):

                bar5_validation_status = (
                    "NOT_REACHED_IN_PRODUCTION"
                )

                bar5_match = True

                absolute_difference = np.nan

            else:

                bar5_validation_status = (
                    "UNEXPECTED_PRODUCTION_BAR5_VALUE"
                )

                bar5_match = False

                absolute_difference = abs(
                    stored_bar5
                    - reconstructed_bar5
                )

        else:

            if pd.isna(stored_bar5):

                bar5_validation_status = (
                    "MISSING_PRODUCTION_VALUE"
                )

                bar5_match = False

                absolute_difference = np.nan

            elif almost_equal(
                stored_bar5,
                reconstructed_bar5,
            ):

                bar5_validation_status = (
                    "MATCH"
                )

                bar5_match = True

                absolute_difference = abs(
                    stored_bar5
                    - reconstructed_bar5
                )

            else:

                bar5_validation_status = (
                    "MISMATCH"
                )

                bar5_match = False

                absolute_difference = abs(
                    stored_bar5
                    - reconstructed_bar5
                )

        # ------------------------------------------------------------------
        # Validation record
        # ------------------------------------------------------------------

        validation_rows.append(
            {
                "trade_index": trade_index,

                "symbol": symbol,

                "entry_date": entry[
                    "entry_date"
                ],

                "direction": entry[
                    "direction"
                ],

                # Production observation boundary
                "production_bars_in_trade":
                    bars_in_trade,

                "bar5_reached_in_production":
                    int(
                        bars_in_trade >= 5
                    ),

                # Frozen production event
                "stored_frozen_event":
                    stored_frozen_event,

                "stored_production_triggered":
                    production_triggered,

                "stored_production_trigger_bar":
                    production_trigger_bar,

                # Replayed production event
                "reconstructed_production_event":
                    production_reconstructed_event,

                "reconstructed_production_trigger_bar":
                    production_reconstructed_trigger_bar,

                "production_trigger_replay_match":
                    production_trigger_replay_match,

                "frozen_event_matches_stored_bar5":
                    (
                        stored_frozen_event
                        ==
                        int(
                            (
                                not pd.isna(stored_bar5)
                            )
                            and (
                                stored_bar5
                                >= EARLY_ADVERSE_THRESHOLD_R
                            )
                        )
                    ),

                # Stored production bar-5 diagnostic
                "stored_bar5_early_adverse_r":
                    stored_bar5,

                # Extended research bar-5 replay
                "reconstructed_bar5_cumulative_mae_r":
                    reconstructed_bar5,

                "absolute_difference":
                    absolute_difference,

                "bar5_validation_status":
                    bar5_validation_status,

                "bar5_match":
                    bar5_match,

                # Separate extended 5-bar research label
                "extended_5bar_event":
                    extended_5bar_event,
            }
        )

        if (
            (trade_index + 1) % 10 == 0
            or trade_index == len(trades) - 1
        ):

            print(
                f"  Reconstructed "
                f"{trade_index + 1}/"
                f"{len(trades)}"
            )

    path_df = pd.DataFrame(
        all_path_rows
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    # ------------------------------------------------------------------
    # Frozen population validation
    # ------------------------------------------------------------------

    expected_path_rows = (
        EXPECTED_TRADE_COUNT
        * MAX_BARS
    )

    observed_path_rows = len(
        path_df
    )

    # ------------------------------------------------------------------
    # Genuine production-observed bar-5 mismatches only.
    #
    # Trades that exited before bar 5 are NOT mismatches.
    # ------------------------------------------------------------------

    mismatch_count = int(
        (
            ~validation_df[
                "bar5_match"
            ]
        ).sum()
    )

    # ------------------------------------------------------------------
    # Frozen production event replay.
    #
    # This must reproduce the authoritative 60 / 46 population.
    # ------------------------------------------------------------------

    production_event_mismatch_count = int(
        (
            ~validation_df[
                "production_trigger_replay_match"
            ]
        ).sum()
    )

    event_count = int(
        (
            validation_df[
                "stored_frozen_event"
            ]
            == 1
        ).sum()
    )

    retained_count = int(
        (
            validation_df[
                "stored_frozen_event"
            ]
            == 0
        ).sum()
    )

    # ------------------------------------------------------------------
    # Extended first-five-bar research population.
    #
    # Informational only.
    #
    # It is NOT the frozen production population.
    # ------------------------------------------------------------------

    extended_5bar_event_count = int(
        validation_df[
            "extended_5bar_event"
        ].sum()
    )

    extended_5bar_retained_count = int(
        len(validation_df)
        - extended_5bar_event_count
    )

    # ------------------------------------------------------------------
    # Production bar-5 reach statistics.
    # ------------------------------------------------------------------

    production_bar5_reached_count = int(
        (
            validation_df[
                "bar5_reached_in_production"
            ]
            == 1
        ).sum()
    )

    production_bar5_not_reached_count = int(
        (
            validation_df[
                "bar5_reached_in_production"
            ]
            == 0
        ).sum()
    )

    integrity_rows = [

        {
            "check":
                "frozen_trade_count",

            "expected":
                EXPECTED_TRADE_COUNT,

            "observed":
                len(trades),

            "status":
                (
                    "PASS"
                    if len(trades)
                    == EXPECTED_TRADE_COUNT
                    else "FAIL"
                ),
        },

        {
            "check":
                "path_row_count",

            "expected":
                expected_path_rows,

            "observed":
                observed_path_rows,

            "status":
                (
                    "PASS"
                    if observed_path_rows
                    == expected_path_rows
                    else "FAIL"
                ),
        },

        {
            "check":
                "production_observed_bar5_mismatches",

            "expected":
                0,

            "observed":
                mismatch_count,

            "status":
                (
                    "PASS"
                    if mismatch_count == 0
                    else "FAIL"
                ),
        },

        {
            "check":
                "production_event_replay_mismatches",

            "expected":
                0,

            "observed":
                production_event_mismatch_count,

            "status":
                (
                    "PASS"
                    if production_event_mismatch_count == 0
                    else "FAIL"
                ),
        },

        {
            "check":
                "frozen_event_count",

            "expected":
                EXPECTED_EVENT_COUNT,

            "observed":
                event_count,

            "status":
                (
                    "PASS"
                    if event_count
                    == EXPECTED_EVENT_COUNT
                    else "FAIL"
                ),
        },

        {
            "check":
                "frozen_retained_count",

            "expected":
                EXPECTED_RETAINED_COUNT,

            "observed":
                retained_count,

            "status":
                (
                    "PASS"
                    if retained_count
                    == EXPECTED_RETAINED_COUNT
                    else "FAIL"
                ),
        },

        {
            "check":
                "production_bar5_reached_count",

            "expected":
                "informational",

            "observed":
                production_bar5_reached_count,

            "status":
                "INFO",
        },

        {
            "check":
                "production_bar5_not_reached_count",

            "expected":
                "informational",

            "observed":
                production_bar5_not_reached_count,

            "status":
                "INFO",
        },

        {
            "check":
                "extended_5bar_event_count",

            "expected":
                "informational",

            "observed":
                extended_5bar_event_count,

            "status":
                "INFO",
        },

        {
            "check":
                "extended_5bar_retained_count",

            "expected":
                "informational",

            "observed":
                extended_5bar_retained_count,

            "status":
                "INFO",
        },
    ]

    # ------------------------------------------------------------------
    # Hard stop before interpreting data
    # ------------------------------------------------------------------

    print()
    print("-" * 78)
    print("INTEGRITY CHECK")
    print("-" * 78)

    for check in integrity_rows:

        print(
            f"[{check['status']}] "
            f"{check['check']}: "
            f"expected={check['expected']} "
            f"observed={check['observed']}"
        )

    hard_failures = sum(
        row["status"] == "FAIL"
        for row in integrity_rows
    )

    if hard_failures > 0:

        print()
        print(
            "FINAL STATUS: FAIL"
        )

        print(
            "STOP. Do not interpret the path results."
        )

        validation_df.to_csv(
            VALIDATION_OUTPUT,
            index=False,
        )

        pd.DataFrame(
            integrity_rows
        ).to_csv(
            INTEGRITY_OUTPUT,
            index=False,
        )

        return 1

    # ------------------------------------------------------------------
    # Add frozen production event label only after validation.
    #
    # IMPORTANT:
    # The frozen event label is the production event label.
    #
    # The extended 5-bar research event remains a separate field and
    # must never overwrite the frozen population.
    # ------------------------------------------------------------------

    validation_key = validation_df[
        [
            "symbol",
            "entry_date",
            "stored_frozen_event",
            "extended_5bar_event",
            "production_bars_in_trade",
            "bar5_reached_in_production",
        ]
    ].copy()

    validation_key = (
        validation_key
        .drop_duplicates(
            subset=[
                "symbol",
                "entry_date",
            ]
        )
    )

    validation_key[
        "event_label"
    ] = np.where(
        validation_key[
            "stored_frozen_event"
        ]
        == 1,
        "EARLY_ADVERSE_EVENT",
        "RETAINED",
    )

    path_df = path_df.merge(
        validation_key[
            [
                "symbol",
                "entry_date",
                "event_label",
                "stored_frozen_event",
                "extended_5bar_event",
                "production_bars_in_trade",
                "bar5_reached_in_production",
            ]
        ],
        on=[
            "symbol",
            "entry_date",
        ],
        how="left",
    )

    # ------------------------------------------------------------------
    # Explicit production-vs-extended observation state for every bar.
    # ------------------------------------------------------------------

    path_df[
        "production_observed"
    ] = (
        path_df[
            "bar_number"
        ]
        <= path_df[
            "production_bars_in_trade"
        ]
    ).astype(int)

    path_df[
        "production_observation_status"
    ] = np.where(
        path_df[
            "production_observed"
        ]
        == 1,
        "OBSERVED_IN_PRODUCTION",
        "EXTENDED_AFTER_PRODUCTION_EXIT",
    )

    # ------------------------------------------------------------------
    # Basic path integrity
    # ------------------------------------------------------------------

    bad_bar_sequences = 0

    for (
        symbol,
        entry_date,
    ), group in path_df.groupby(
        [
            "symbol",
            "entry_date",
        ]
    ):

        sequence = (
            group.sort_values(
                "bar_number"
            )["bar_number"]
            .tolist()
        )

        if sequence != [
            1,
            2,
            3,
            4,
            5,
        ]:

            bad_bar_sequences += 1

    integrity_rows.append(
        {
            "check": "bar_sequence_1_to_5",
            "expected": 0,
            "observed": bad_bar_sequences,
            "status": (
                "PASS"
                if bad_bar_sequences == 0
                else "FAIL"
            ),
        }
    )

    # ------------------------------------------------------------------
    # Save primary forensic dataset
    # ------------------------------------------------------------------

    path_df.to_csv(
        PATH_OUTPUT,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_OUTPUT,
        index=False,
    )

    integrity_rows.append(
        {
            "check": "primary_path_output_exists",
            "expected": True,
            "observed": PATH_OUTPUT.exists(),
            "status": (
                "PASS"
                if PATH_OUTPUT.exists()
                else "FAIL"
            ),
        }
    )

    integrity_rows.append(
        {
            "check": "validation_output_exists",
            "expected": True,
            "observed": VALIDATION_OUTPUT.exists(),
            "status": (
                "PASS"
                if VALIDATION_OUTPUT.exists()
                else "FAIL"
            ),
        }
    )

    final_integrity_df = pd.DataFrame(
        integrity_rows
    )

    final_integrity_df.to_csv(
        INTEGRITY_OUTPUT,
        index=False,
    )

    # ------------------------------------------------------------------
    # Console readout
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "FIRST-5-BAR PATH FOUNDATION — PASS"
    )
    print("=" * 78)

    print()
    print(
        f"Trades:              {len(trades)}"
    )

    print(
        f"Path rows:           {len(path_df)}"
    )

    print(
        f"Frozen events:       {event_count}"
    )

    print(
        f"Frozen retained:     {retained_count}"
    )

    print(
        f"Bar-5 mismatches:    {mismatch_count}"
    )

    print(
        f"Event replay mismatches: "
        f"{production_event_mismatch_count}"
    )

    print(
        f"Production bar-5 reached: "
        f"{production_bar5_reached_count}"
    )

    print(
        f"Production bar-5 not reached: "
        f"{production_bar5_not_reached_count}"
    )

    print(
        f"Extended 5-bar events: "
        f"{extended_5bar_event_count}"
    )

    print(
        f"Extended 5-bar retained: "
        f"{extended_5bar_retained_count}"
    )

    print(
        f"Bad bar sequences:   {bad_bar_sequences}"
    )

    print()
    print(
        "Primary output:"
    )

    print(
        f"  {PATH_OUTPUT}"
    )

    print()
    print(
        "Validation output:"
    )

    print(
        f"  {VALIDATION_OUTPUT}"
    )

    print()
    print(
        "Integrity output:"
    )

    print(
        f"  {INTEGRITY_OUTPUT}"
    )

    print()
    print(
        "No strategy rule has been promoted."
    )

    print(
        "No production/OOS data has been modified."
    )

    print("=" * 78)

    if bad_bar_sequences != 0:

        print(
            "FINAL STATUS: FAIL"
        )

        return 1

    print(
        "FINAL STATUS: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )