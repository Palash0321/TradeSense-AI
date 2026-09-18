"""
TradeSense-AI
FORENSIC VOLUME DATA LINEAGE AUDIT

Purpose
-------
Verify the production-equivalent Volume Ratio for every trade in the
frozen 106-trade population.

Production-equivalent volume definition
---------------------------------------
volume_ratio =
    signal-day Volume
    /
    20-period rolling mean of Volume

The rolling mean includes the signal day, matching:

    df["Volume"].rolling(20).mean()

Lineage
-------
Frozen trade
    -> Signal_Date
    -> historical OHLCV
    -> production BacktestService.load_data()
    -> 20-period Volume rolling mean
    -> signal-day Volume Ratio
    -> frozen trade identity

Safety
------
- Research only.
- Frozen 106-trade population is READ-ONLY.
- Production strategy is READ-ONLY.
- OOS ledgers are untouched.
- No profitability analysis.
- No threshold optimization.
- No strategy modification.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT ROOT / IMPORT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

FROZEN_TRADES_FILE = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "tradesense_trades.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "tests"
    / "output"
    / "research"
    / "volume_lineage"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "volume_lineage_per_trade.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "volume_lineage_summary.json"
)

INTEGRITY_FILE = (
    OUTPUT_DIR
    / "volume_lineage_integrity.csv"
)

EXPECTED_TRADES = 106

# Frozen population begins here.
FROZEN_START = pd.Timestamp(
    "2021-09-07"
)

# Latest frozen trade / historical boundary.
FROZEN_END = pd.Timestamp(
    "2026-09-08"
)

VOLUME_LOOKBACK = 20

# Need enough history BEFORE the earliest frozen signal date.
#
# BacktestService's warmup_days expands the downloaded history while
# preserving the requested analysis boundary.
#
# 60 calendar days is deliberately more than the required 20 trading
# observations and provides a safe buffer for weekends/holidays.
WARMUP_DAYS = 60

# Production-equivalent descriptive categories.
# These are NOT optimized in this audit.
VOLUME_CATEGORIES = (
    ("LOW", 0.0, 1.0),
    ("NORMAL", 1.0, 1.5),
    ("HIGH", 1.5, 2.0),
    ("VERY_HIGH", 2.0, math.inf),
)

SYMBOL_ALIASES = [
    "_symbol",
    "symbol",
    "Symbol",
    "SYMBOL",
]

ENTRY_DATE_ALIASES = [
    "entry_date",
    "Entry_Date",
    "Entry Date",
]

SIGNAL_DATE_ALIASES = [
    "Signal_Date",
    "signal_date",
    "Signal Date",
]

DIRECTION_ALIASES = [
    "direction",
    "Direction",
]


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:
        value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def normalize_symbol(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = (
        str(value)
        .strip()
        .upper()
    )

    return text


def normalize_date(
    value: Any,
) -> Optional[str]:

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return None

    return timestamp.strftime(
        "%Y-%m-%d"
    )


def find_column(
    dataframe: pd.DataFrame,
    aliases: List[str],
) -> Optional[str]:

    for alias in aliases:

        if alias in dataframe.columns:
            return alias

    normalized = {
        str(column)
        .strip()
        .lower(): column
        for column in dataframe.columns
    }

    for alias in aliases:

        key = (
            str(alias)
            .strip()
            .lower()
        )

        if key in normalized:
            return normalized[key]

    return None


def find_volume_column(
    dataframe: pd.DataFrame,
) -> Optional[str]:

    return find_column(
        dataframe,
        [
            "Volume",
            "volume",
            "VOLUME",
        ],
    )


def sha256_file(
    path: Path,
) -> Optional[str]:

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def classify_volume_ratio(
    ratio: Optional[float],
) -> str:

    if ratio is None:
        return "UNAVAILABLE"

    if ratio < 1.0:
        return "LOW"

    if ratio < 1.5:
        return "NORMAL"

    if ratio < 2.0:
        return "HIGH"

    return "VERY_HIGH"


# ============================================================================
# FROZEN DATASET
# ============================================================================

def load_frozen_trades() -> pd.DataFrame:

    print(
        "=" * 100
    )
    print(
        "TRADESENSE-AI — VOLUME DATA LINEAGE AUDIT"
    )
    print(
        "=" * 100
    )

    print()
    print(
        "FROZEN TRADE FILE:"
    )
    print(
        FROZEN_TRADES_FILE
    )

    if not FROZEN_TRADES_FILE.exists():

        raise RuntimeError(
            "Frozen trade file does not exist:\n"
            f"{FROZEN_TRADES_FILE}"
        )

    trades = pd.read_csv(
        FROZEN_TRADES_FILE,
        low_memory=False,
    )

    print(
        f"RAW FROZEN ROWS: {len(trades)}"
    )

    if len(trades) != EXPECTED_TRADES:

        raise RuntimeError(
            "FROZEN POPULATION GUARD FAILED\n"
            f"Expected: {EXPECTED_TRADES}\n"
            f"Observed: {len(trades)}\n"
            "STOP."
        )

    symbol_column = find_column(
        trades,
        SYMBOL_ALIASES,
    )

    entry_date_column = find_column(
        trades,
        ENTRY_DATE_ALIASES,
    )

    signal_date_column = find_column(
        trades,
        SIGNAL_DATE_ALIASES,
    )

    direction_column = find_column(
        trades,
        DIRECTION_ALIASES,
    )

    missing = []

    if symbol_column is None:
        missing.append("symbol")

    if entry_date_column is None:
        missing.append("entry_date")

    if signal_date_column is None:
        missing.append("Signal_Date")

    if direction_column is None:
        missing.append("direction")

    if missing:

        raise RuntimeError(
            "FROZEN TRADE SCHEMA FAILURE\n"
            "Missing columns: "
            + ", ".join(missing)
        )

    print()
    print(
        "FROZEN COLUMN MAPPING:"
    )

    print(
        f"  symbol          -> {symbol_column}"
    )

    print(
        f"  entry_date      -> {entry_date_column}"
    )

    print(
        f"  signal_date     -> {signal_date_column}"
    )

    print(
        f"  direction       -> {direction_column}"
    )

    result = trades.copy()

    result[
        "_lineage_symbol"
    ] = result[
        symbol_column
    ].map(
        normalize_symbol
    )

    result[
        "_lineage_entry_date"
    ] = result[
        entry_date_column
    ].map(
        normalize_date
    )

    result[
        "_lineage_signal_date"
    ] = result[
        signal_date_column
    ].map(
        normalize_date
    )

    result[
        "_lineage_direction"
    ] = (
        result[
            direction_column
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if (
        result[
            "_lineage_symbol"
        ].eq("").any()
    ):

        raise RuntimeError(
            "FROZEN DATA FAILURE: blank symbol."
        )

    if (
        result[
            "_lineage_entry_date"
        ].isna().any()
    ):

        raise RuntimeError(
            "FROZEN DATA FAILURE: invalid entry date."
        )

    if (
        result[
            "_lineage_signal_date"
        ].isna().any()
    ):

        raise RuntimeError(
            "FROZEN DATA FAILURE: invalid Signal_Date."
        )

    if (
        result[
            "_lineage_direction"
        ].eq("").any()
    ):

        raise RuntimeError(
            "FROZEN DATA FAILURE: blank direction."
        )

    return result


# ============================================================================
# PRODUCTION HISTORICAL DATA
# ============================================================================

def load_symbol_ohlcv(
    symbol: str,
) -> pd.DataFrame:

    from app.services.backtest_service import (
        BacktestService
    )

    service = BacktestService(
        symbol=symbol,
        strategy="tradesense",
        initial_capital=100000,
    )

    # IMPORTANT:
    #
    # Use the actual production BacktestService interface:
    # load_data(), not fetch_data().
    #
    # Warm-up is intentionally supplied because the volume calculation
    # requires a complete 20-period rolling history at the earliest
    # frozen Signal_Date.

    dataframe = service.load_data(
        start_date=FROZEN_START.strftime(
            "%Y-%m-%d"
        ),
        end_date=(
            FROZEN_END
            + pd.Timedelta(days=1)
        ).strftime(
            "%Y-%m-%d"
        ),
        warmup_days=WARMUP_DAYS,
    )

    if (
        dataframe is None
        or dataframe.empty
    ):

        raise RuntimeError(
            f"No historical OHLCV returned "
            f"for {symbol}."
        )

    dataframe = dataframe.copy()

    dataframe.index = pd.to_datetime(
        dataframe.index,
        errors="coerce",
    )

    dataframe = dataframe.loc[
        ~dataframe.index.isna()
    ]

    dataframe = dataframe.sort_index()

    volume_column = find_volume_column(
        dataframe
    )

    if volume_column is None:

        raise RuntimeError(
            f"Volume column not found for "
            f"{symbol}.\n"
            f"Columns: {list(dataframe.columns)}"
        )

    dataframe[
        "__lineage_volume__"
    ] = pd.to_numeric(
        dataframe[
            volume_column
        ],
        errors="coerce",
    )

    return dataframe


# ============================================================================
# PRODUCTION-EQUIVALENT VOLUME CALCULATION
# ============================================================================

def calculate_volume_ratio(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    # This intentionally mirrors the production operation:
    #
    # Volume.rolling(20).mean()
    #
    # The signal-day Volume is included in the rolling window.

    result[
        "__lineage_avg_volume_20__"
    ] = (
        result[
            "__lineage_volume__"
        ]
        .rolling(
            VOLUME_LOOKBACK
        )
        .mean()
    )

    result[
        "__lineage_volume_ratio__"
    ] = (
        result[
            "__lineage_volume__"
        ]
        /
        result[
            "__lineage_avg_volume_20__"
        ]
    )

    return result


# ============================================================================
# DATE MATCHING
# ============================================================================

def get_positions_for_date(
    dataframe: pd.DataFrame,
    date_value: str,
) -> np.ndarray:

    target = pd.Timestamp(
        date_value
    ).normalize()

    normalized = (
        dataframe.index
        .normalize()
    )

    return np.where(
        normalized == target
    )[0]


def validate_signal_entry_relationship(
    dataframe: pd.DataFrame,
    signal_date: str,
    entry_date: str,
) -> Dict[str, Any]:

    signal_positions = (
        get_positions_for_date(
            dataframe,
            signal_date,
        )
    )

    entry_positions = (
        get_positions_for_date(
            dataframe,
            entry_date,
        )
    )

    result = {
        "signal_row_found": (
            len(signal_positions) > 0
        ),
        "entry_row_found": (
            len(entry_positions) > 0
        ),
        "signal_before_entry": False,
        "adjacent_signal_entry": False,
        "signal_position": None,
        "entry_position": None,
    }

    if len(signal_positions) == 0:
        return result

    if len(entry_positions) == 0:
        return result

    signal_position = int(
        signal_positions[0]
    )

    entry_position = int(
        entry_positions[0]
    )

    result[
        "signal_position"
    ] = signal_position

    result[
        "entry_position"
    ] = entry_position

    result[
        "signal_before_entry"
    ] = (
        signal_position
        < entry_position
    )

    result[
        "adjacent_signal_entry"
    ] = (
        entry_position
        == signal_position + 1
    )

    return result


# ============================================================================
# SINGLE TRADE AUDIT
# ============================================================================

def audit_trade(
    trade: pd.Series,
    data_by_symbol: Dict[
        str,
        pd.DataFrame,
    ],
) -> Dict[str, Any]:

    symbol = str(
        trade[
            "_lineage_symbol"
        ]
    )

    direction = str(
        trade[
            "_lineage_direction"
        ]
    )

    signal_date = str(
        trade[
            "_lineage_signal_date"
        ]
    )

    entry_date = str(
        trade[
            "_lineage_entry_date"
        ]
    )

    trade_key = (
        f"{symbol}|"
        f"{direction}|"
        f"{entry_date}"
    )

    row = {
        "trade_key": trade_key,
        "symbol": symbol,
        "direction": direction,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "signal_volume": np.nan,
        "average_volume_20": np.nan,
        "volume_ratio": np.nan,
        "volume_category": "UNAVAILABLE",
        "signal_row_found": False,
        "entry_row_found": False,
        "signal_before_entry": False,
        "adjacent_signal_entry": False,
        "rolling_window_complete": False,
        "volume_available": False,
        "lineage_status": "FAIL",
        "error": "",
    }

    if symbol not in data_by_symbol:

        row["error"] = (
            "SYMBOL_OHLCV_NOT_AVAILABLE"
        )

        return row

    dataframe = data_by_symbol[
        symbol
    ]

    relationship = (
        validate_signal_entry_relationship(
            dataframe,
            signal_date,
            entry_date,
        )
    )

    row[
        "signal_row_found"
    ] = relationship[
        "signal_row_found"
    ]

    row[
        "entry_row_found"
    ] = relationship[
        "entry_row_found"
    ]

    row[
        "signal_before_entry"
    ] = relationship[
        "signal_before_entry"
    ]

    row[
        "adjacent_signal_entry"
    ] = relationship[
        "adjacent_signal_entry"
    ]

    if not row[
        "signal_row_found"
    ]:

        row["error"] = (
            "SIGNAL_DATE_NOT_FOUND"
        )

        return row

    if not row[
        "entry_row_found"
    ]:

        row["error"] = (
            "ENTRY_DATE_NOT_FOUND"
        )

        return row

    if not row[
        "signal_before_entry"
    ]:

        row["error"] = (
            "SIGNAL_NOT_BEFORE_ENTRY"
        )

        return row

    if not row[
        "adjacent_signal_entry"
    ]:

        row["error"] = (
            "SIGNAL_AND_ENTRY_NOT_ADJACENT"
        )

        return row

    signal_position = int(
        relationship[
            "signal_position"
        ]
    )

    signal_row = dataframe.iloc[
        signal_position
    ]

    volume = safe_float(
        signal_row[
            "__lineage_volume__"
        ]
    )

    average_volume = safe_float(
        signal_row[
            "__lineage_avg_volume_20__"
        ]
    )

    ratio = safe_float(
        signal_row[
            "__lineage_volume_ratio__"
        ]
    )

    row[
        "signal_volume"
    ] = (
        volume
        if volume is not None
        else np.nan
    )

    row[
        "average_volume_20"
    ] = (
        average_volume
        if average_volume is not None
        else np.nan
    )

    row[
        "volume_ratio"
    ] = (
        ratio
        if ratio is not None
        else np.nan
    )

    if volume is None:

        row["error"] = (
            "SIGNAL_VOLUME_MISSING"
        )

        return row

    if average_volume is None:

        row["error"] = (
            "20_PERIOD_AVERAGE_VOLUME_MISSING"
        )

        return row

    if average_volume <= 0:

        row["error"] = (
            "20_PERIOD_AVERAGE_VOLUME_NONPOSITIVE"
        )

        return row

    if ratio is None:

        row["error"] = (
            "VOLUME_RATIO_UNAVAILABLE"
        )

        return row

    # Explicitly verify that 20 valid Volume observations exist
    # ending at Signal_Date.

    window_start = max(
        0,
        signal_position
        - VOLUME_LOOKBACK
        + 1,
    )

    signal_window = dataframe.iloc[
        window_start:
        signal_position + 1
    ][
        "__lineage_volume__"
    ]

    row[
        "rolling_window_complete"
    ] = (
        len(signal_window)
        == VOLUME_LOOKBACK
        and signal_window.notna().all()
    )

    if not row[
        "rolling_window_complete"
    ]:

        row["error"] = (
            "20_PERIOD_ROLLING_WINDOW_INCOMPLETE"
        )

        return row

    row[
        "volume_available"
    ] = True

    row[
        "volume_category"
    ] = classify_volume_ratio(
        ratio
    )

    row[
        "lineage_status"
    ] = "PASS"

    return row


# ============================================================================
# INTEGRITY
# ============================================================================

def build_integrity(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    def check(
        name: str,
        actual: Any,
        expected: Any,
        passed: bool,
    ) -> None:

        rows.append(
            {
                "check": name,
                "actual": actual,
                "expected": expected,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    check(
        "FROZEN_TRADE_COUNT",
        len(detail),
        EXPECTED_TRADES,
        len(detail)
        == EXPECTED_TRADES,
    )

    pass_count = int(
        (
            detail[
                "lineage_status"
            ]
            == "PASS"
        ).sum()
    )

    check(
        "LINEAGE_PASS_COUNT",
        pass_count,
        EXPECTED_TRADES,
        pass_count
        == EXPECTED_TRADES,
    )

    fail_count = int(
        (
            detail[
                "lineage_status"
            ]
            != "PASS"
        ).sum()
    )

    check(
        "LINEAGE_FAILURE_COUNT",
        fail_count,
        0,
        fail_count == 0,
    )

    signal_found = int(
        detail[
            "signal_row_found"
        ].sum()
    )

    check(
        "SIGNAL_DATE_FOUND",
        signal_found,
        EXPECTED_TRADES,
        signal_found
        == EXPECTED_TRADES,
    )

    entry_found = int(
        detail[
            "entry_row_found"
        ].sum()
    )

    check(
        "ENTRY_DATE_FOUND",
        entry_found,
        EXPECTED_TRADES,
        entry_found
        == EXPECTED_TRADES,
    )

    signal_before = int(
        detail[
            "signal_before_entry"
        ].sum()
    )

    check(
        "SIGNAL_BEFORE_ENTRY",
        signal_before,
        EXPECTED_TRADES,
        signal_before
        == EXPECTED_TRADES,
    )

    adjacent = int(
        detail[
            "adjacent_signal_entry"
        ].sum()
    )

    check(
        "ADJACENT_SIGNAL_ENTRY",
        adjacent,
        EXPECTED_TRADES,
        adjacent
        == EXPECTED_TRADES,
    )

    rolling_complete = int(
        detail[
            "rolling_window_complete"
        ].sum()
    )

    check(
        "20_PERIOD_WINDOW_COMPLETE",
        rolling_complete,
        EXPECTED_TRADES,
        rolling_complete
        == EXPECTED_TRADES,
    )

    volume_available = int(
        detail[
            "volume_available"
        ].sum()
    )

    check(
        "VOLUME_AVAILABLE",
        volume_available,
        EXPECTED_TRADES,
        volume_available
        == EXPECTED_TRADES,
    )

    duplicate_keys = int(
        detail[
            "trade_key"
        ].duplicated()
        .sum()
    )

    check(
        "DUPLICATE_TRADE_KEYS",
        duplicate_keys,
        0,
        duplicate_keys == 0,
    )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# SUMMARY
# ============================================================================

def build_summary(
    detail: pd.DataFrame,
    frozen_sha256: Optional[str],
    data_by_symbol: Dict[
        str,
        pd.DataFrame,
    ],
) -> Dict[str, Any]:

    passed = detail.loc[
        detail[
            "lineage_status"
        ]
        == "PASS"
    ]

    failed = detail.loc[
        detail[
            "lineage_status"
        ]
        != "PASS"
    ]

    summary = {
        "audit": (
            "VOLUME_DATA_LINEAGE"
        ),
        "status": (
            "PASS"
            if len(passed)
            == EXPECTED_TRADES
            and len(failed)
            == 0
            else "FAIL"
        ),
        "expected_frozen_trades": (
            EXPECTED_TRADES
        ),
        "audited_trades": int(
            len(detail)
        ),
        "passed_trades": int(
            len(passed)
        ),
        "failed_trades": int(
            len(failed)
        ),
        "frozen_trade_sha256": (
            frozen_sha256
        ),
        "frozen_start": (
            FROZEN_START.strftime(
                "%Y-%m-%d"
            )
        ),
        "frozen_end": (
            FROZEN_END.strftime(
                "%Y-%m-%d"
            )
        ),
        "volume_lookback": (
            VOLUME_LOOKBACK
        ),
        "warmup_days": (
            WARMUP_DAYS
        ),
        "production_equivalent_definition": (
            "Signal-day Volume / "
            "20-period rolling mean of Volume "
            "through Signal_Date"
        ),
        "signal_entry_relationship": (
            "Signal_Date must immediately "
            "precede Entry_Date in the "
            "symbol's trading-day sequence"
        ),
        "threshold_optimization": "NONE",
        "profitability_analysis": (
            "NOT_PERFORMED"
        ),
        "early_adverse_analysis": (
            "NOT_PERFORMED"
        ),
        "strategy_modification": (
            "NONE"
        ),
        "oos_modification": (
            "NONE"
        ),
        "categories": {},
        "symbols_loaded": sorted(
            data_by_symbol.keys()
        ),
    }

    for category in [
        "LOW",
        "NORMAL",
        "HIGH",
        "VERY_HIGH",
    ]:

        summary[
            "categories"
        ][category] = int(
            (
                passed[
                    "volume_category"
                ]
                == category
            ).sum()
        )

    if len(failed) > 0:

        summary[
            "failure_reasons"
        ] = (
            failed[
                "error"
            ]
            .value_counts()
            .to_dict()
        )

    else:

        summary[
            "failure_reasons"
        ] = {}

    if len(passed) > 0:

        summary[
            "volume_ratio_descriptive"
        ] = {
            "min": float(
                passed[
                    "volume_ratio"
                ].min()
            ),
            "max": float(
                passed[
                    "volume_ratio"
                ].max()
            ),
            "mean": float(
                passed[
                    "volume_ratio"
                ].mean()
            ),
            "median": float(
                passed[
                    "volume_ratio"
                ].median()
            ),
        }

    return summary


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Load frozen population.
    # ------------------------------------------------------------------------

    trades = load_frozen_trades()

    frozen_sha256 = sha256_file(
        FROZEN_TRADES_FILE
    )

    symbols = sorted(
        trades[
            "_lineage_symbol"
        ]
        .unique()
        .tolist()
    )

    print()
    print(
        f"UNIQUE SYMBOLS: {len(symbols)}"
    )

    print(
        "SYMBOLS:",
        ", ".join(symbols),
    )

    # ------------------------------------------------------------------------
    # Load historical data.
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "LOADING PRODUCTION HISTORICAL DATA"
    )
    print(
        "=" * 100
    )

    data_by_symbol = {}

    for symbol in symbols:

        print()
        print(
            f"PROCESSING SYMBOL: {symbol}"
        )

        dataframe = load_symbol_ohlcv(
            symbol
        )

        dataframe = (
            calculate_volume_ratio(
                dataframe
            )
        )

        data_by_symbol[
            symbol
        ] = dataframe

        volume_column = (
            find_volume_column(
                dataframe
            )
        )

        print(
            f"Rows                 : "
            f"{len(dataframe)}"
        )

        print(
            f"Data start            : "
            f"{dataframe.index.min().date()}"
        )

        print(
            f"Data end              : "
            f"{dataframe.index.max().date()}"
        )

        print(
            f"Volume column         : "
            f"{volume_column}"
        )

        print(
            f"20-period rolling     : "
            f"CALCULATED"
        )

    # ------------------------------------------------------------------------
    # Audit trades.
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "AUDITING FROZEN 106 TRADES"
    )
    print(
        "=" * 100
    )

    audit_rows = []

    for position, (
        _,
        trade,
    ) in enumerate(
        trades.iterrows(),
        start=1,
    ):

        result = audit_trade(
            trade,
            data_by_symbol,
        )

        audit_rows.append(
            result
        )

        print(
            f"[{position:03d}/"
            f"{len(trades):03d}] "
            f"{result['trade_key']:<50} "
            f"{result['lineage_status']}"
        )

    detail = pd.DataFrame(
        audit_rows
    )

    # ------------------------------------------------------------------------
    # Integrity and summary.
    # ------------------------------------------------------------------------

    integrity = build_integrity(
        detail
    )

    summary = build_summary(
        detail,
        frozen_sha256,
        data_by_symbol,
    )

    # ------------------------------------------------------------------------
    # Write research outputs.
    # ------------------------------------------------------------------------

    detail.to_csv(
        DETAIL_FILE,
        index=False,
    )

    integrity.to_csv(
        INTEGRITY_FILE,
        index=False,
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            default=str,
        )

    # ------------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        "VOLUME DATA LINEAGE RESULT"
    )
    print(
        "=" * 100
    )

    print(
        f"Expected frozen trades : "
        f"{EXPECTED_TRADES}"
    )

    print(
        f"Audited trades         : "
        f"{summary['audited_trades']}"
    )

    print(
        f"Lineage PASS           : "
        f"{summary['passed_trades']}"
    )

    print(
        f"Lineage FAIL           : "
        f"{summary['failed_trades']}"
    )

    print()
    print(
        "PRODUCTION-EQUIVALENT VOLUME:"
    )

    print(
        "Signal-day Volume / "
        "20-period rolling mean Volume"
    )

    print()
    print(
        "SIGNAL -> ENTRY:"
    )

    print(
        "Signal_Date must be the "
        "immediately preceding trading day "
        "before Entry_Date."
    )

    print()
    print(
        "VOLUME CATEGORIES:"
    )

    for category in [
        "LOW",
        "NORMAL",
        "HIGH",
        "VERY_HIGH",
    ]:

        print(
            f"  {category:<10}: "
            f"{summary['categories'][category]}"
        )

    print()
    print(
        "THRESHOLD OPTIMIZATION : NONE"
    )

    print(
        "PROFITABILITY ANALYSIS : NOT PERFORMED"
    )

    print(
        "EARLY-ADVERSE ANALYSIS : NOT PERFORMED"
    )

    print(
        "STRATEGY MODIFICATION  : NONE"
    )

    print(
        "OOS MODIFICATION       : NONE"
    )

    print()
    print(
        "RESEARCH OUTPUTS:"
    )

    print(
        f"  Detail    : {DETAIL_FILE}"
    )

    print(
        f"  Summary   : {SUMMARY_FILE}"
    )

    print(
        f"  Integrity : {INTEGRITY_FILE}"
    )

    print()

    integrity_failures = integrity.loc[
        integrity["status"] == "FAIL"
    ]

    if (
        summary["status"] == "PASS"
        and len(integrity_failures) == 0
    ):

        print(
            "STATUS: VOLUME DATA LINEAGE PASS"
        )

        print()
        print(
            "Exact production-equivalent "
            "Volume Ratio is available for "
            "all 106 frozen trades."
        )

        print()
        print(
            "NEXT:"
        )

        print(
            "VOLUME × EARLY-ADVERSE "
            "INDEPENDENT-EDGE AUDIT"
        )

        return 0

    print(
        "STATUS: VOLUME DATA LINEAGE FAIL"
    )

    print()
    print(
        "DO NOT proceed to the Volume "
        "independent-edge test."
    )

    if summary[
        "failure_reasons"
    ]:

        print()
        print(
            "FAILURE REASONS:"
        )

        for reason, count in sorted(
            summary[
                "failure_reasons"
            ].items()
        ):

            print(
                f"  {reason}: {count}"
            )

    return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )