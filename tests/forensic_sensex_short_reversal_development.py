"""
TradeSense-AI
SENSEX SHORT_REVERSAL Trade Development Forensic

Purpose:
    Deep descriptive analysis of SENSEX SHORT_REVERSAL trades.

IMPORTANT:
    - Research only.
    - No production changes.
    - No frozen dataset changes.
    - No holdout changes.
    - No parameter optimization.
    - No strategy selection.
"""

import os
import pandas as pd


TRADES_FILE = (
    "tests/output/index_backtests/"
    "sensex_index_trades.csv"
)

OUTPUT_DIR = (
    "tests/output/index_backtests"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "sensex_short_reversal_development.csv"
)


def numeric(df, column):
    if column not in df.columns:
        return pd.Series(
            [float("nan")] * len(df),
            index=df.index
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def avg(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return float("nan")

    return float(series.mean())


def median(series):
    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return float("nan")

    return float(series.median())


def summarize_group(label, group):
    r = numeric(group, "r_multiple")
    mfe = numeric(group, "max_favorable_r")
    mae = numeric(group, "max_adverse_r")
    atr = numeric(group, "entry_atr")
    risk = numeric(group, "initial_risk")
    bars = numeric(group, "bars_in_trade")

    winners = int(
        (r > 0).sum()
    )

    losers = int(
        (r <= 0).sum()
    )

    print()
    print("-" * 100)
    print(label)
    print("-" * 100)

    print(f"Trades:              {len(group)}")
    print(f"Winners:             {winners}")
    print(f"Losers:              {losers}")

    if len(group):
        print(
            f"Win rate:            "
            f"{winners / len(group) * 100:.2f}%"
        )

    print(
        f"Total R:             "
        f"{r.sum():.2f}R"
    )

    print(
        f"Avg R:               "
        f"{avg(r):.3f}R"
    )

    print(
        f"Avg Entry ATR:       "
        f"{avg(atr):.2f}"
    )

    print(
        f"Median Entry ATR:    "
        f"{median(atr):.2f}"
    )

    print(
        f"Avg Initial Risk:    "
        f"{avg(risk):.2f}"
    )

    print(
        f"Avg MFE:             "
        f"{avg(mfe):.3f}R"
    )

    print(
        f"Median MFE:          "
        f"{median(mfe):.3f}R"
    )

    print(
        f"Avg MAE:             "
        f"{avg(mae):.3f}R"
    )

    print(
        f"Median MAE:          "
        f"{median(mae):.3f}R"
    )

    print(
        f"Avg Holding Bars:    "
        f"{avg(bars):.2f}"
    )

    print(
        f"Median Holding:      "
        f"{median(bars):.2f}"
    )


def milestone_analysis(group):
    print()
    print("=" * 100)
    print("MFE MILESTONE ANALYSIS")
    print("=" * 100)

    mfe = numeric(
        group,
        "max_favorable_r"
    )

    thresholds = [
        0.25,
        0.50,
        1.00,
        1.50,
        2.00,
    ]

    for threshold in thresholds:

        mask = (
            mfe >= threshold
        )

        subset = group.loc[
            mask
        ]

        r = numeric(
            subset,
            "r_multiple"
        )

        print(
            f"Reached {threshold:.2f}R: "
            f"{len(subset):>2}/{len(group)}  "
            f"TotalR={r.sum():>7.2f}R"
        )

        if len(subset):
            bars = numeric(
                subset,
                "bars_to_"
                + str(threshold).replace(
                    ".",
                    "_"
                )
                + "r"
            )

            if bars.notna().any():
                print(
                    f"    Avg bars: "
                    f"{bars.mean():.2f}"
                )


def early_adverse_analysis(group):
    print()
    print("=" * 100)
    print("EARLY-ADVERSE DEVELOPMENT")
    print("=" * 100)

    columns = [
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5",
    ]

    for column in columns:

        if column not in group.columns:
            continue

        values = numeric(
            group,
            column
        )

        print(
            f"{column}: "
            f"Avg={values.mean():.3f}R  "
            f"Median={values.median():.3f}R  "
            f"Max={values.max():.3f}R"
        )


def favorable_first_analysis(group):
    print()
    print("=" * 100)
    print("FAVORABLE-FIRST / IMMEDIATE-FAILURE")
    print("=" * 100)

    if "favorable_first" in group.columns:

        values = (
            group["favorable_first"]
            .astype(str)
            .str.upper()
        )

        print(
            "Favorable first counts:"
        )

        print(
            values.value_counts(
                dropna=False
            ).to_string()
        )

    if "immediate_failure" in group.columns:

        values = (
            group["immediate_failure"]
            .astype(str)
            .str.upper()
        )

        print()
        print(
            "Immediate failure counts:"
        )

        print(
            values.value_counts(
                dropna=False
            ).to_string()
        )


def yearly_analysis(group):
    print()
    print("=" * 100)
    print("YEAR-BY-YEAR SHORT_REVERSAL")
    print("=" * 100)

    if "entry_date" not in group.columns:
        return

    dates = pd.to_datetime(
        group["entry_date"],
        errors="coerce"
    )

    temp = group.copy()
    temp["_year"] = dates.dt.year

    for year, subset in temp.groupby(
        "_year",
        dropna=True
    ):

        r = numeric(
            subset,
            "r_multiple"
        )

        winners = int(
            (r > 0).sum()
        )

        print(
            f"{int(year)}: "
            f"N={len(subset):>2}  "
            f"W={winners:>2}  "
            f"Win%="
            f"{winners / len(subset) * 100:>6.2f}%  "
            f"TotalR="
            f"{r.sum():>7.2f}R  "
            f"AvgMFE="
            f"{avg(numeric(subset, 'max_favorable_r')):.2f}R  "
            f"AvgMAE="
            f"{avg(numeric(subset, 'max_adverse_r')):.2f}R"
        )


def compare_to_short_continuation(df):
    print()
    print("=" * 100)
    print("CONTROL GROUP — SHORT_REVERSAL vs SHORT_CONTINUATION")
    print("=" * 100)

    reversal = df[
        df["setup"].astype(str).str.upper()
        == "SHORT_REVERSAL"
    ]

    continuation = df[
        df["setup"].astype(str).str.upper()
        == "SHORT_CONTINUATION"
    ]

    rows = []

    for name, group in [
        ("SHORT_REVERSAL", reversal),
        ("SHORT_CONTINUATION", continuation),
    ]:

        r = numeric(
            group,
            "r_multiple"
        )

        rows.append(
            {
                "setup": name,
                "trades": len(group),
                "winners": int(
                    (r > 0).sum()
                ),
                "win_rate": (
                    (r > 0).mean() * 100
                    if len(group)
                    else 0
                ),
                "total_r": float(
                    r.sum()
                ),
                "avg_r": avg(r),
                "avg_mfe": avg(
                    numeric(
                        group,
                        "max_favorable_r"
                    )
                ),
                "avg_mae": avg(
                    numeric(
                        group,
                        "max_adverse_r"
                    )
                ),
                "avg_early5": avg(
                    numeric(
                        group,
                        "early_adverse_r_bar_5"
                    )
                ),
                "avg_bars": avg(
                    numeric(
                        group,
                        "bars_in_trade"
                    )
                ),
                "avg_atr": avg(
                    numeric(
                        group,
                        "entry_atr"
                    )
                ),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "win_rate": "{:.2f}%".format,
                "total_r": "{:.2f}R".format,
                "avg_r": "{:.3f}R".format,
                "avg_mfe": "{:.3f}R".format,
                "avg_mae": "{:.3f}R".format,
                "avg_early5": "{:.3f}R".format,
                "avg_bars": "{:.2f}".format,
                "avg_atr": "{:.2f}".format,
            }
        )
    )


def individual_trades(group):
    print()
    print("=" * 100)
    print("INDIVIDUAL SHORT_REVERSAL TRADES")
    print("=" * 100)

    columns = [
        "entry_date",
        "exit_date",
        "direction",
        "setup",
        "r_multiple",
        "bars_in_trade",
        "entry_atr",
        "initial_risk",
        "max_favorable_r",
        "max_adverse_r",
        "early_adverse_r_bar_1",
        "early_adverse_r_bar_2",
        "early_adverse_r_bar_3",
        "early_adverse_r_bar_4",
        "early_adverse_r_bar_5",
        "bars_to_0_5r",
        "bars_to_1r",
        "bars_to_1_5r",
        "exit_reason",
    ]

    available = [
        column
        for column in columns
        if column in group.columns
    ]

    table = group[
        available
    ].copy()

    if "entry_date" in table.columns:
        table = table.sort_values(
            "entry_date"
        )

    print(
        table.to_string(
            index=False
        )
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    table.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


def main():

    print()
    print("=" * 100)
    print(
        "TRADESENSE-AI — SENSEX "
        "SHORT_REVERSAL FORENSIC"
    )
    print("=" * 100)

    if not os.path.exists(
        TRADES_FILE
    ):
        raise FileNotFoundError(
            f"Missing: {TRADES_FILE}"
        )

    df = pd.read_csv(
        TRADES_FILE
    )

    if "setup" not in df.columns:
        raise RuntimeError(
            "Trade file has no setup column."
        )

    reversal = df[
        df["setup"].astype(str).str.upper()
        == "SHORT_REVERSAL"
    ].copy()

    if reversal.empty:
        raise RuntimeError(
            "No SHORT_REVERSAL trades found."
        )

    print(
        f"Total SENSEX trades: {len(df)}"
    )

    print(
        f"SHORT_REVERSAL trades: "
        f"{len(reversal)}"
    )

    summarize_group(
        "SENSEX SHORT_REVERSAL",
        reversal
    )

    milestone_analysis(
        reversal
    )

    early_adverse_analysis(
        reversal
    )

    favorable_first_analysis(
        reversal
    )

    yearly_analysis(
        reversal
    )

    compare_to_short_continuation(
        df
    )

    individual_trades(
        reversal
    )

    print()
    print("=" * 100)
    print("RESEARCH VERDICT")
    print("=" * 100)

    print(
        "This analysis is descriptive."
    )

    print(
        "SHORT_REVERSAL exclusion is NOT being implemented."
    )

    print(
        "No production code was modified."
    )

    print(
        "No frozen historical data was modified."
    )

    print(
        "No holdout data was modified."
    )

    print(
        "Any apparent edge requires controlled "
        "counterfactual and OOS validation."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()