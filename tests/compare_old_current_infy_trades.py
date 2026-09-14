"""
DIAGNOSTIC ONLY

Compare the OLD INFY reference trades against the CURRENT
explicit-boundary BacktestService execution.

Important:
run_directional_backtest() does not necessarily attach symbol
metadata to each trade, so current trades are treated as INFY
because this diagnostic explicitly runs BacktestService for INFY.NS.

NO strategy changes.
NO analyzer changes.
NO frozen-rule changes.
"""

import os
import pandas as pd

from app.services.backtest_service import BacktestService


SYMBOL = "INFY.NS"

OLD_FILE = "tests/output/tradesense_trades_OLD_REFERENCE.csv"

DATA_START = "2021-09-07"
DATA_END = "2022-02-01"

WINDOW_START = pd.Timestamp("2021-10-01")
WINDOW_END = pd.Timestamp("2022-02-01")


def get_value(row, candidates):

    for column in candidates:

        if column in row.index:

            value = row[column]

            if pd.notna(value):
                return value

    return None


def normalize_date(value):

    if value is None:
        return pd.NaT

    return pd.to_datetime(
        value,
        errors="coerce",
    )


def print_trade(label, trade):

    print(f"\n{label}")

    fields = [
        (
            "Entry Date",
            [
                "Entry_Date",
                "entry_date",
                "EntryDate",
                "date",
            ],
        ),
        (
            "Exit Date",
            [
                "Exit_Date",
                "exit_date",
                "ExitDate",
            ],
        ),
        (
            "Direction",
            [
                "Direction",
                "direction",
            ],
        ),
        (
            "Entry Price",
            [
                "Entry_Price",
                "entry_price",
                "EntryPrice",
            ],
        ),
        (
            "Initial Risk",
            [
                "Initial_Risk",
                "initial_risk",
                "Risk",
                "risk",
            ],
        ),
        (
            "Final R",
            [
                "R_Multiple",
                "r_multiple",
                "R",
                "final_r",
            ],
        ),
        (
            "Setup",
            [
                "Setup",
                "setup",
            ],
        ),
        (
            "Exit Reason",
            [
                "Exit_Reason",
                "exit_reason",
            ],
        ),
        (
            "Bars Held",
            [
                "Bars_Held",
                "bars_held",
                "bars_in_trade",
            ],
        ),
        (
            "Bar 5 Adverse R",
            [
                "early_adverse_r_bar_5",
                "Early_Adverse_R_Bar_5",
                "Bar5_Adverse_R",
            ],
        ),
    ]

    for display_name, candidates in fields:

        value = get_value(
            trade,
            candidates,
        )

        if value is not None:

            print(
                f"  {display_name:<20}: {value}"
            )


def main():

    # ================================================================
    # 1. OLD REFERENCE
    # ================================================================

    print("\n" + "=" * 100)
    print("1. OLD INFY REFERENCE TRADES")
    print("=" * 100)

    if not os.path.exists(OLD_FILE):

        print(
            f"ERROR: Old reference file not found:\n"
            f"{OLD_FILE}"
        )

        return

    old = pd.read_csv(
        OLD_FILE
    )

    print(
        f"Old reference rows: {len(old)}"
    )

    old["normalized_entry_date"] = pd.to_datetime(
        old["entry_date"],
        errors="coerce",
    )

    old_infy = old[
        (
            old["_symbol"]
            .astype(str)
            .str.strip()
            == SYMBOL
        )
        &
        (
            old["normalized_entry_date"]
            >= WINDOW_START
        )
        &
        (
            old["normalized_entry_date"]
            < WINDOW_END
        )
    ].copy()

    old_infy = old_infy.sort_values(
        "normalized_entry_date"
    )

    print(
        f"Old INFY trades in window: "
        f"{len(old_infy)}"
    )

    for index, (_, row) in enumerate(
        old_infy.iterrows(),
        start=1,
    ):

        print_trade(
            f"OLD TRADE #{index}",
            row,
        )

    # ================================================================
    # 2. CURRENT EXPLICIT-BOUNDARY DATA
    # ================================================================

    print("\n" + "=" * 100)
    print("2. CURRENT EXPLICIT-BOUNDARY BACKTEST")
    print("=" * 100)

    print(
        f"Symbol     : {SYMBOL}"
    )

    print(
        f"Data start : {DATA_START}"
    )

    print(
        f"Data end   : {DATA_END}"
    )

    service = BacktestService(
        symbol=SYMBOL,
        strategy="tradesense",
        initial_capital=100000,
    )

    data = service.load_data(
        start_date=DATA_START,
        end_date=DATA_END,
    )

    print(
        f"Loaded rows: {len(data)}"
    )

    print(
        f"First date : {data.index.min()}"
    )

    print(
        f"Last date  : {data.index.max()}"
    )

    # ================================================================
    # 3. CURRENT SIGNALS
    # ================================================================

    print("\n" + "=" * 100)
    print("3. CURRENT SIGNALS")
    print("=" * 100)

    signals = service.generate_tradesense_signals(
        df=data,
        swing_window=3,
    )

    print(
        f"Signal rows: {len(signals)}"
    )

    # Show all important signals around the discrepancy.

    signal_dates = [
        "2021-10-01",
        "2021-10-04",
        "2021-10-18",
        "2021-10-19",
        "2021-10-29",
        "2021-11-01",
        "2021-12-17",
        "2021-12-20",
        "2021-12-23",
        "2022-01-14",
        "2022-01-17",
        "2022-01-21",
        "2022-01-25",
    ]

    for date_string in signal_dates:

        target_date = pd.Timestamp(
            date_string
        )

        if target_date not in signals.index:

            continue

        row = signals.loc[target_date]

        print(
            f"\n{date_string}"
        )

        for column in [
            "Signal",
            "Setup",
            "Setup_Direction",
            "Setup_Confidence",
            "Structure_Bias",
            "Structure",
            "BOS",
            "CHOCH",
            "Break_Direction",
            "Break_Confirmed",
            "Break_Age",
        ]:

            if column in signals.columns:

                print(
                    f"  {column:<20}: "
                    f"{row[column]}"
                )

    # ================================================================
    # 4. CURRENT EXECUTED TRADES
    # ================================================================

    print("\n" + "=" * 100)
    print("4. CURRENT EXECUTED INFY TRADES")
    print("=" * 100)

    current_result = service.run_directional_backtest(
        signals
    )

    current_trades = current_result.get(
        "trades",
        [],
    )

    print(
        f"Current executed trades: "
        f"{len(current_trades)}"
    )

    # Because this BacktestService instance is explicitly INFY.NS,
    # every returned trade belongs to INFY.

    current_infy = []

    for trade in current_trades:

        entry_date = normalize_date(
            get_value(
                pd.Series(trade),
                [
                    "Entry_Date",
                    "entry_date",
                    "EntryDate",
                    "date",
                ],
            )
        )

        if (
            pd.notna(entry_date)
            and entry_date >= WINDOW_START
            and entry_date < WINDOW_END
        ):

            current_infy.append(
                trade
            )

    print(
        f"Current INFY trades in window: "
        f"{len(current_infy)}"
    )

    for index, trade in enumerate(
        current_infy,
        start=1,
    ):

        print_trade(
            f"CURRENT TRADE #{index}",
            pd.Series(trade),
        )

    # ================================================================
    # 5. TARGET DATE COMPARISON
    # ================================================================

    print("\n" + "=" * 100)
    print("5. OLD vs CURRENT TARGET TRADE COMPARISON")
    print("=" * 100)

    targets = [
        "2021-10-04",
        "2021-11-01",
        "2022-01-17",
    ]

    current_by_date = {}

    for trade in current_infy:

        entry_date = normalize_date(
            get_value(
                pd.Series(trade),
                [
                    "Entry_Date",
                    "entry_date",
                    "EntryDate",
                    "date",
                ],
            )
        )

        key = (
            entry_date.strftime("%Y-%m-%d")
            if pd.notna(entry_date)
            else None
        )

        if key is not None:

            current_by_date.setdefault(
                key,
                [],
            ).append(
                trade
            )

    for target in targets:

        print("\n" + "-" * 100)
        print(
            f"TARGET ENTRY DATE: {target}"
        )
        print("-" * 100)

        old_matches = old_infy[
            old_infy["normalized_entry_date"]
            == pd.Timestamp(target)
        ]

        current_matches = current_by_date.get(
            target,
            [],
        )

        print(
            f"Old matches    : "
            f"{len(old_matches)}"
        )

        print(
            f"Current matches: "
            f"{len(current_matches)}"
        )

        for index, (_, row) in enumerate(
            old_matches.iterrows(),
            start=1,
        ):

            print_trade(
                f"OLD MATCH #{index}",
                row,
            )

        for index, trade in enumerate(
            current_matches,
            start=1,
        ):

            print_trade(
                f"CURRENT MATCH #{index}",
                pd.Series(trade),
            )

    # ================================================================
    # 6. POSITION-OVERLAP ANALYSIS
    # ================================================================

    print("\n" + "=" * 100)
    print("6. POSITION-OVERLAP ANALYSIS")
    print("=" * 100)

    print(
        "\nCurrent execution sequence:"
    )

    for index, trade in enumerate(
        current_infy,
        start=1,
    ):

        entry = get_value(
            pd.Series(trade),
            [
                "Entry_Date",
                "entry_date",
                "EntryDate",
                "date",
            ],
        )

        exit_date = get_value(
            pd.Series(trade),
            [
                "Exit_Date",
                "exit_date",
                "ExitDate",
            ],
        )

        direction = get_value(
            pd.Series(trade),
            [
                "Direction",
                "direction",
            ],
        )

        print(
            f"  CURRENT #{index}: "
            f"{entry} -> {exit_date} "
            f"({direction})"
        )

    print(
        "\nTesting whether the old 2022-01-17 LONG "
        "would overlap an existing current position..."
    )

    current_dec20 = []

    for trade in current_infy:

        entry = normalize_date(
            get_value(
                pd.Series(trade),
                [
                    "Entry_Date",
                    "entry_date",
                    "EntryDate",
                    "date",
                ],
            )
        )

        if entry == pd.Timestamp(
            "2021-12-20"
        ):

            current_dec20.append(
                trade
            )

    old_jan17 = old_infy[
        old_infy["normalized_entry_date"]
        == pd.Timestamp("2022-01-17")
    ]

    if current_dec20 and not old_jan17.empty:

        print(
            "\n*** OVERLAP CANDIDATE FOUND ***"
        )

        for trade in current_dec20:

            print_trade(
                "CURRENT 2021-12-20 TRADE",
                pd.Series(trade),
            )

        for _, row in old_jan17.iterrows():

            print_trade(
                "OLD 2022-01-17 TRADE",
                row,
            )

        print(
            "\nIf the current 12/20 trade remains open "
            "through 01/17, then the old 01/17 trade "
            "cannot be reproduced by the current "
            "single-position state machine without "
            "different execution semantics."
        )

    else:

        print(
            "\nNo direct 12/20 + old 01/17 overlap "
            "was established."
        )

    # ================================================================
    # 7. FINAL
    # ================================================================

    print("\n" + "=" * 100)
    print("7. DIAGNOSTIC STATUS")
    print("=" * 100)

    print(
        "READ-ONLY diagnostic."
    )

    print(
        "No strategy code modified."
    )

    print(
        "No analyzer modified."
    )

    print(
        "Frozen 5B_0.25R rule unchanged."
    )


if __name__ == "__main__":
    main()