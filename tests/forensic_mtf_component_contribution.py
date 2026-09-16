"""
TradeSense-AI
MTF Component Contribution + Causal Timestamp Forensic

Purpose
-------
Determine whether individual reconstructed MTF components contribute
incremental predictive information before a frozen trade entry.

Historically defensible components:
    - Daily
    - Completed Weekly
    - Daily + Weekly

NOT reconstructed here:
    - 15m
    - 1h
    - 4h

Those require reliable historical intraday data and must not be fabricated.

Causal contract
---------------
For a trade with Signal_Date = D:

1. Daily state may use information available through the close of D.
   The trade enters on the next bar, so the completed signal-day close
   is available before entry.

2. Weekly state MUST be the latest COMPLETED weekly candle strictly
   before the signal-day's developing weekly candle.

3. Weekly candles are built using W-FRI.

4. No future weekly information may enter a trade's classification.

5. Trade R is always taken from the frozen authoritative trade ledger.
   This forensic does NOT recompute strategy P&L.

Production / frozen / OOS data are NEVER modified.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRICE_DIR = ROOT / "data" / "nifty50_prices"
FROZEN_TRADES = ROOT / "tests" / "output" / "tradesense_trades.csv"

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_OUTPUT = OUTPUT_DIR / "mtf_component_contribution_trade_level.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mtf_component_contribution_summary.csv"


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(symbol):
    """
    Convert frozen ledger symbols such as HDFCBANK.NS
    to local price-file root HDFCBANK.
    """
    if pd.isna(symbol):
        return None

    value = str(symbol).strip().upper()

    if value.endswith(".NS"):
        value = value[:-3]

    return value


def load_symbol_prices(symbol):
    """
    Load the validated local NIFTY50 price file.

    Local convention:
        data/nifty50_prices/HDFCBANK.csv

    Actual validated schema:
        date
        open
        high
        low
        close
        adj_close
        volume
        dividends
        stock_splits
    """

    root_symbol = normalize_symbol(symbol)

    path = PRICE_DIR / f"{root_symbol}.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]

    if not all(column in df.columns for column in required):
        return None

    df = df.copy()

    # Normalize the validated lowercase schema into the
    # column names used by the forensic calculations.
    df["Date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["Open"] = pd.to_numeric(
        df["open"],
        errors="coerce",
    )

    df["High"] = pd.to_numeric(
        df["high"],
        errors="coerce",
    )

    df["Low"] = pd.to_numeric(
        df["low"],
        errors="coerce",
    )

    df["Close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if "volume" in df.columns:
        df["Volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
        ]
    ).copy()

    df = df.sort_values("Date")

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last",
    )

    return df


def build_daily_state(df):
    """
    Daily state uses only information available through
    the daily candle close.

    State:
        BULLISH
        BEARISH
        NEUTRAL
    """

    daily = df.copy()

    daily["MA20"] = daily["Close"].rolling(20).mean()
    daily["MA50"] = daily["Close"].rolling(50).mean()

    daily["Daily_State"] = "UNAVAILABLE"

    valid = (
        daily["MA20"].notna()
        & daily["MA50"].notna()
    )

    daily.loc[
        valid & (daily["MA20"] > daily["MA50"]),
        "Daily_State"
    ] = "BULLISH"

    daily.loc[
        valid & (daily["MA20"] < daily["MA50"]),
        "Daily_State"
    ] = "BEARISH"

    daily.loc[
        valid & (daily["MA20"] == daily["MA50"]),
        "Daily_State"
    ] = "NEUTRAL"

    daily["Signal_Date"] = daily["Date"]

    return daily[
        ["Signal_Date", "Daily_State", "MA20", "MA50"]
    ].copy()


def build_weekly_state(df):
    """
    Build weekly candles using W-FRI.

    CRITICAL CAUSALITY RULE:
    The weekly state associated with a Signal_Date is the
    latest COMPLETED weekly candle BEFORE that signal date.

    Therefore a current developing weekly candle is NEVER used.
    """

    daily = df.copy()

    daily["Date"] = pd.to_datetime(daily["Date"])

    daily = daily.set_index("Date")

    weekly = (
        daily
        .resample("W-FRI")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )

    weekly["MA20"] = weekly["Close"].rolling(20).mean()
    weekly["MA50"] = weekly["Close"].rolling(50).mean()

    weekly["Weekly_State"] = "UNAVAILABLE"

    valid = (
        weekly["MA20"].notna()
        & weekly["MA50"].notna()
    )

    weekly.loc[
        valid & (weekly["MA20"] > weekly["MA50"]),
        "Weekly_State"
    ] = "BULLISH"

    weekly.loc[
        valid & (weekly["MA20"] < weekly["MA50"]),
        "Weekly_State"
    ] = "BEARISH"

    weekly.loc[
        valid & (weekly["MA20"] == weekly["MA50"]),
        "Weekly_State"
    ] = "NEUTRAL"

    weekly["Completed_Weekly_Date"] = weekly.index

    # The weekly candle ending on the same Friday as Signal_Date
    # is considered the developing/current weekly candle for that
    # signal date. Therefore merge_asof uses direction=backward
    # against a timestamp strictly BEFORE Signal_Date.
    #
    # We accomplish this by creating a lookup date one day before
    # each signal date and then selecting the latest completed week.
    weekly = weekly.reset_index(drop=True)

    return weekly[
        [
            "Completed_Weekly_Date",
            "Weekly_State",
            "MA20",
            "MA50",
        ]
    ].copy()


def build_symbol_component_state(symbol):
    """
    Construct causally valid Daily + completed Weekly state
    for one symbol.
    """

    df = load_symbol_prices(symbol)

    if df is None or df.empty:
        return None

    daily = build_daily_state(df)

    weekly = build_weekly_state(df)

    # Signal dates from the daily trading calendar.
    signal_dates = daily["Signal_Date"].copy()

    # For each signal date D, select the latest weekly candle
    # whose completed date is STRICTLY BEFORE D.
    #
    # This prevents Friday's current weekly candle from leaking
    # into Friday's signal.
    left = pd.DataFrame(
        {
            "Signal_Date": pd.to_datetime(signal_dates)
        }
    ).sort_values("Signal_Date")

    right = weekly.copy()

    right["Completed_Weekly_Date"] = pd.to_datetime(
        right["Completed_Weekly_Date"]
    )

    right = right.sort_values("Completed_Weekly_Date")

    left["Weekly_Lookup_Date"] = (
        left["Signal_Date"] - pd.Timedelta(days=1)
    )

    merged = pd.merge_asof(
        left.sort_values("Weekly_Lookup_Date"),
        right,
        left_on="Weekly_Lookup_Date",
        right_on="Completed_Weekly_Date",
        direction="backward",
    )

    result = daily.merge(
        merged[
            [
                "Signal_Date",
                "Completed_Weekly_Date",
                "Weekly_State",
            ]
        ],
        on="Signal_Date",
        how="left",
    )

    result["Symbol"] = normalize_symbol(symbol)

    return result[
        [
            "Symbol",
            "Signal_Date",
            "Daily_State",
            "Weekly_State",
            "Completed_Weekly_Date",
        ]
    ]


# ============================================================
# MTF CLASSIFICATION
# ============================================================

def classify_component_states(daily_state, weekly_state, direction):
    """
    Component-level classification.

    For LONG:
        BULLISH = aligned
        BEARISH = conflict
        NEUTRAL = neutral

    For SHORT:
        BEARISH = aligned
        BULLISH = conflict
        NEUTRAL = neutral
    """

    if pd.isna(daily_state) or daily_state == "UNAVAILABLE":
        daily_relation = "UNAVAILABLE"
    elif direction == "LONG":
        if daily_state == "BULLISH":
            daily_relation = "ALIGNED"
        elif daily_state == "BEARISH":
            daily_relation = "CONFLICT"
        else:
            daily_relation = "NEUTRAL"
    else:
        if daily_state == "BEARISH":
            daily_relation = "ALIGNED"
        elif daily_state == "BULLISH":
            daily_relation = "CONFLICT"
        else:
            daily_relation = "NEUTRAL"

    if pd.isna(weekly_state) or weekly_state == "UNAVAILABLE":
        weekly_relation = "UNAVAILABLE"
    elif direction == "LONG":
        if weekly_state == "BULLISH":
            weekly_relation = "ALIGNED"
        elif weekly_state == "BEARISH":
            weekly_relation = "CONFLICT"
        else:
            weekly_relation = "NEUTRAL"
    else:
        if weekly_state == "BEARISH":
            weekly_relation = "ALIGNED"
        elif weekly_state == "BULLISH":
            weekly_relation = "CONFLICT"
        else:
            weekly_relation = "NEUTRAL"

    return daily_relation, weekly_relation


def combined_class(daily_relation, weekly_relation):
    """
    Two-component agreement classification.
    """

    if (
        daily_relation == "UNAVAILABLE"
        or weekly_relation == "UNAVAILABLE"
    ):
        return "UNAVAILABLE"

    if (
        daily_relation == "ALIGNED"
        and weekly_relation == "ALIGNED"
    ):
        return "FULL_ALIGNMENT"

    if (
        daily_relation == "CONFLICT"
        and weekly_relation == "CONFLICT"
    ):
        return "FULL_CONFLICT"

    if daily_relation == weekly_relation:
        return "SAME_DIRECTION_NEUTRAL"

    return "MIXED"


# ============================================================
# FROZEN TRADE LOADING
# ============================================================

def load_frozen_trades():
    trades = pd.read_csv(FROZEN_TRADES)

    symbol_column = None

    for candidate in ["_symbol", "Symbol", "symbol"]:
        if candidate in trades.columns:
            symbol_column = candidate
            break

    if symbol_column is None:
        raise ValueError(
            "Frozen trade ledger does not contain a recognized symbol column."
        )

    date_column = None

    for candidate in ["Signal_Date", "signal_date"]:
        if candidate in trades.columns:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "Frozen trade ledger does not contain Signal_Date."
        )

    r_column = None

    for candidate in ["R_Multiple", "r_multiple", "R"]:
        if candidate in trades.columns:
            r_column = candidate
            break

    if r_column is None:
        raise ValueError(
            "Frozen trade ledger does not contain authoritative R_Multiple."
        )

    direction_column = None

    for candidate in ["Direction", "direction"]:
        if candidate in trades.columns:
            direction_column = candidate
            break

    if direction_column is None:
        raise ValueError(
            "Frozen trade ledger does not contain Direction."
        )

    trades = trades.copy()

    trades["_symbol_root"] = trades[symbol_column].map(normalize_symbol)

    trades["Signal_Date"] = pd.to_datetime(
        trades[date_column],
        errors="coerce",
    )

    trades["R_Multiple_Authoritative"] = pd.to_numeric(
        trades[r_column],
        errors="coerce",
    )

    trades["_direction"] = (
        trades[direction_column]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    trades["_trade_key"] = (
        trades["_symbol_root"].astype(str)
        + "|"
        + trades["Signal_Date"].dt.strftime("%Y-%m-%d")
        + "|"
        + trades["_direction"]
    )

    return trades


# ============================================================
# METRICS
# ============================================================

def profit_factor(values):
    values = pd.Series(values).dropna()

    gross_profit = values[values > 0].sum()
    gross_loss = abs(values[values < 0].sum())

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return np.nan

    return gross_profit / gross_loss


def summarize_group(df, group_column):
    rows = []

    for group_value, group in df.groupby(
        group_column,
        dropna=False,
    ):
        r = group["R_Multiple_Authoritative"].dropna()

        wins = int((r > 0).sum())
        losses = int((r < 0).sum())
        total = len(r)

        rows.append(
            {
                "Dimension": group_column,
                "Group": group_value,
                "Trades": total,
                "Wins": wins,
                "Losses": losses,
                "Win_Rate_%": (
                    round(wins / total * 100, 2)
                    if total
                    else np.nan
                ),
                "Profit_Factor": (
                    round(profit_factor(r), 4)
                    if len(r)
                    else np.nan
                ),
                "Total_R": round(r.sum(), 4),
                "Avg_R": round(r.mean(), 4)
                if len(r)
                else np.nan,
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MTF COMPONENT CONTRIBUTION + CAUSAL TIMESTAMP FORENSIC")
    print("=" * 70)

    trades = load_frozen_trades()

    print(f"Frozen trades loaded : {len(trades)}")

    duplicate_count = trades["_trade_key"].duplicated().sum()

    print(f"Frozen duplicate keys: {duplicate_count}")

    if duplicate_count:
        raise ValueError(
            "Duplicate frozen trade keys detected."
        )

    # --------------------------------------------------------
    # Build states for every frozen symbol
    # --------------------------------------------------------

    symbols = sorted(
        trades["_symbol_root"].dropna().unique()
    )

    state_frames = []

    for symbol in symbols:

        print(f"Building causal MTF state: {symbol}")

        state = build_symbol_component_state(symbol)

        if state is None:
            print(f"  WARNING: no valid price data for {symbol}")
            continue

        state_frames.append(state)

    if not state_frames:
        raise RuntimeError(
            "No symbol MTF state could be constructed."
        )

    all_states = pd.concat(
        state_frames,
        ignore_index=True,
    )

    all_states["Signal_Date"] = pd.to_datetime(
        all_states["Signal_Date"]
    )

    # --------------------------------------------------------
    # Merge frozen trades with component states
    # --------------------------------------------------------

    analysis = trades.merge(
        all_states,
        left_on=[
            "_symbol_root",
            "Signal_Date",
        ],
        right_on=[
            "Symbol",
            "Signal_Date",
        ],
        how="left",
        validate="one_to_one",
    )

    analysis["Daily_Relation"] = "UNAVAILABLE"
    analysis["Weekly_Relation"] = "UNAVAILABLE"

    for idx, row in analysis.iterrows():

        daily_relation, weekly_relation = classify_component_states(
            row["Daily_State"],
            row["Weekly_State"],
            row["_direction"],
        )

        analysis.at[idx, "Daily_Relation"] = daily_relation
        analysis.at[idx, "Weekly_Relation"] = weekly_relation

    analysis["Combined_MTF_Class"] = [
        combined_class(
            daily_relation,
            weekly_relation,
        )
        for daily_relation, weekly_relation
        in zip(
            analysis["Daily_Relation"],
            analysis["Weekly_Relation"],
        )
    ]

    # --------------------------------------------------------
    # Causal timestamp audit
    # --------------------------------------------------------

    analysis["Weekly_Causal_PASS"] = (
        pd.to_datetime(
            analysis["Completed_Weekly_Date"],
            errors="coerce",
        )
        < analysis["Signal_Date"]
    )

    # If weekly data is unavailable, causal PASS is not claimed.
    analysis.loc[
        analysis["Completed_Weekly_Date"].isna(),
        "Weekly_Causal_PASS"
    ] = False

    # Daily state is based on Signal_Date itself.
    analysis["Daily_Causal_PASS"] = (
        analysis["Signal_Date"].notna()
        & analysis["Daily_State"].notna()
        & (
            analysis["Daily_State"]
            != "UNAVAILABLE"
        )
    )

    # --------------------------------------------------------
    # Incremental contribution labels
    # --------------------------------------------------------

    analysis["Daily_Only_Usable"] = (
        analysis["Daily_Relation"]
        .isin(
            [
                "ALIGNED",
                "CONFLICT",
                "NEUTRAL",
            ]
        )
    )

    analysis["Weekly_Only_Usable"] = (
        analysis["Weekly_Relation"]
        .isin(
            [
                "ALIGNED",
                "CONFLICT",
                "NEUTRAL",
            ]
        )
    )

    analysis["Both_Usable"] = (
        analysis["Daily_Only_Usable"]
        & analysis["Weekly_Only_Usable"]
    )

    # --------------------------------------------------------
    # Reconciliation
    # --------------------------------------------------------

    frozen_total_r = (
        trades["R_Multiple_Authoritative"].sum()
    )

    analyzed_total_r = (
        analysis["R_Multiple_Authoritative"].sum()
    )

    r_delta = analyzed_total_r - frozen_total_r

    print()
    print("=" * 70)
    print("RECONCILIATION")
    print("=" * 70)

    print(f"Frozen total R       : {frozen_total_r:.2f}")
    print(f"Analyzed total R     : {analyzed_total_r:.2f}")
    print(f"R delta              : {r_delta:.10f}")

    if abs(r_delta) < 1e-9:
        print("R reconciliation     : PASS")
    else:
        print("R reconciliation     : FAIL")

    # --------------------------------------------------------
    # Causal audit
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CAUSAL TIMESTAMP AUDIT")
    print("=" * 70)

    print(
        "Daily causal PASS    : "
        f"{int(analysis['Daily_Causal_PASS'].sum())}"
        f"/{len(analysis)}"
    )

    print(
        "Weekly causal PASS   : "
        f"{int(analysis['Weekly_Causal_PASS'].sum())}"
        f"/{len(analysis)}"
    )

    causal_failures = analysis[
        analysis["Weekly_Causal_PASS"] == False
    ].copy()

    if len(causal_failures):
        print()
        print("Weekly causal failures:")

        print(
            causal_failures[
                [
                    "_symbol_root",
                    "Signal_Date",
                    "Completed_Weekly_Date",
                    "Weekly_State",
                ]
            ].to_string(index=False)
        )
    else:
        print("Weekly timestamp causality: PASS")

    # --------------------------------------------------------
    # Component-level summaries
    # --------------------------------------------------------

    summary_rows = []

    summary_rows.extend(
        summarize_group(
            analysis[
                analysis["Daily_Only_Usable"]
            ],
            "Daily_Relation",
        )
    )

    summary_rows.extend(
        summarize_group(
            analysis[
                analysis["Weekly_Only_Usable"]
            ],
            "Weekly_Relation",
        )
    )

    summary_rows.extend(
        summarize_group(
            analysis[
                analysis["Both_Usable"]
            ],
            "Combined_MTF_Class",
        )
    )

    # --------------------------------------------------------
    # Direction x component
    # --------------------------------------------------------

    for component_column in [
        "Daily_Relation",
        "Weekly_Relation",
    ]:

        subset = analysis[
            analysis[component_column].isin(
                [
                    "ALIGNED",
                    "CONFLICT",
                    "NEUTRAL",
                ]
            )
        ]

        for direction, group in subset.groupby(
            "_direction"
        ):

            for relation, relation_group in group.groupby(
                component_column
            ):

                r = relation_group[
                    "R_Multiple_Authoritative"
                ].dropna()

                summary_rows.append(
                    {
                        "Dimension": (
                            f"{component_column}_BY_DIRECTION"
                        ),
                        "Group": (
                            f"{direction}|{relation}"
                        ),
                        "Trades": len(r),
                        "Wins": int((r > 0).sum()),
                        "Losses": int((r < 0).sum()),
                        "Win_Rate_%": (
                            round(
                                (r > 0).mean() * 100,
                                2,
                            )
                            if len(r)
                            else np.nan
                        ),
                        "Profit_Factor": (
                            round(
                                profit_factor(r),
                                4,
                            )
                            if len(r)
                            else np.nan
                        ),
                        "Total_R": round(
                            r.sum(),
                            4,
                        ),
                        "Avg_R": round(
                            r.mean(),
                            4,
                        )
                        if len(r)
                        else np.nan,
                    }
                )

    # --------------------------------------------------------
    # Incremental information test
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPONENT CONTRIBUTION")
    print("=" * 70)

    for component in [
        "Daily_Relation",
        "Weekly_Relation",
    ]:

        usable = analysis[
            analysis[component].isin(
                [
                    "ALIGNED",
                    "CONFLICT",
                    "NEUTRAL",
                ]
            )
        ]

        print()
        print(component)

        for relation in [
            "ALIGNED",
            "CONFLICT",
            "NEUTRAL",
        ]:

            group = usable[
                usable[component] == relation
            ]

            if group.empty:
                continue

            r = group[
                "R_Multiple_Authoritative"
            ]

            print(
                f"  {relation:10s} "
                f"N={len(group):3d} "
                f"Win={((r > 0).mean() * 100):6.2f}% "
                f"PF={profit_factor(r):6.3f} "
                f"R={r.sum():8.2f} "
                f"Avg={r.mean():7.4f}"
            )

    # --------------------------------------------------------
    # Full combined matrix
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DAILY × WEEKLY MATRIX")
    print("=" * 70)

    matrix = analysis[
        analysis["Both_Usable"]
    ].copy()

    for (daily_relation, weekly_relation), group in matrix.groupby(
        [
            "Daily_Relation",
            "Weekly_Relation",
        ]
    ):

        r = group[
            "R_Multiple_Authoritative"
        ]

        print(
            f"  Daily={daily_relation:9s} "
            f"Weekly={weekly_relation:9s} "
            f"N={len(group):3d} "
            f"Win={((r > 0).mean() * 100):6.2f}% "
            f"PF={profit_factor(r):6.3f} "
            f"R={r.sum():8.2f} "
            f"Avg={r.mean():7.4f}"
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    analysis.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(
        f"Trade-level : {TRADE_OUTPUT}"
    )

    print(
        f"Summary     : {SUMMARY_OUTPUT}"
    )

    print()
    print("STATUS: MTF_COMPONENT_CONTRIBUTION_COMPLETE")


if __name__ == "__main__":
    main()