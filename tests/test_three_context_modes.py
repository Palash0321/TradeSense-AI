import csv
import os

from app.services.backtest_service import BacktestService


# ==============================================================
# CONFIGURATION
# ==============================================================

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

# IMPORTANT:
# yfinance END is exclusive.
# Therefore END_DATE = 2026-09-08 produces data through
# approximately 2026-09-07, matching the frozen dataset boundary.
END_DATE = "2026-09-08"

FROZEN_START_DATE = "2021-09-07"
BUFFERED_START_DATE = "2021-01-01"

FROZEN_FILE = "tests/output/tradesense_trades.csv"

OUTPUT_DIR = "tests/output/reconciliation"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "three_context_modes.csv"
)
MODE_A_TRADES_FILE = os.path.join(
    OUTPUT_DIR,
    "mode_a_frozen_trades.csv"
)

MODE_B_TRADES_FILE = os.path.join(
    OUTPUT_DIR,
    "mode_b_buffered_trades.csv"
)


# ==============================================================
# HELPERS
# ==============================================================

def trade_identity(trade):
    return (
        str(trade.get("_symbol", "")),
        str(trade.get("entry_date", ""))[:10],
        str(trade.get("direction", "")),
    )


def normalise_trade(trade):
    return {
        key: trade.get(key)
        for key in sorted(trade.keys())
    }


def load_frozen_trades():
    with open(
        FROZEN_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        trades = []

        for row in reader:
            trades.append(row)

    return trades


def get_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def run_mode(
    mode_name,
    calculation_start,
    evaluation_start
):
    print()
    print("=" * 100)
    print(f"MODE {mode_name}")
    print("=" * 100)

    print(
        f"Calculation context : "
        f"{calculation_start} -> {END_DATE}"
    )

    print(
        f"Evaluation period   : "
        f"{evaluation_start} -> {END_DATE}"
    )

    all_trades = []

    for symbol in SYMBOLS:

        print()
        print(f"[{symbol}]")

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        # ----------------------------------------------------------
        # Load the COMPLETE calculation context.
        # ----------------------------------------------------------

        historical_data = service.load_data(
            start_date=calculation_start,
            end_date=END_DATE
        )

        print(
            f"Calculation rows: {len(historical_data)}"
        )

        # ----------------------------------------------------------
        # Generate signals using the complete calculation context.
        # ----------------------------------------------------------

        signals = service.generate_tradesense_signals(
            df=historical_data
        )

        print(
            f"Signal rows: {len(signals)}"
        )

        # ----------------------------------------------------------
        # Evaluate ONLY from the requested evaluation boundary.
        # ----------------------------------------------------------

        evaluation_signals = signals.loc[
            signals.index >= evaluation_start
        ].copy()

        print(
            f"Evaluation rows: {len(evaluation_signals)}"
        )

        # ----------------------------------------------------------
        # Run the exact production directional backtest.
        # ----------------------------------------------------------

        result = service.run_directional_backtest(
            evaluation_signals,
            signal_column="Signal",
            initial_stop_atr=3.5,
            target_r=2.0,
            risk_per_trade=0.01,
        )

        if isinstance(result, tuple):

            trades = result[0]

        elif isinstance(result, dict):

            trades = result.get(
                "trades",
                []
            )

        else:

            trades = result

        # ----------------------------------------------------------
        # Make sure every trade carries the symbol.
        # ----------------------------------------------------------

        for trade in trades:

            trade = dict(trade)

            trade["_symbol"] = symbol

            all_trades.append(trade)

        print(
            f"Trades: {len(trades)}"
        )

    # --------------------------------------------------------------
    # Identity validation.
    # --------------------------------------------------------------

    identities = [
        trade_identity(trade)
        for trade in all_trades
    ]

    duplicate_identities = (
        len(identities)
        -
        len(set(identities))
    )

    total_r = sum(
        get_float(
            trade.get("r_multiple")
        )
        for trade in all_trades
    )

    print()
    print(
        f"FINAL {mode_name}:"
    )

    print(
        f"Trades: {len(all_trades)}"
    )

    print(
        f"Total R: {total_r:.6f}"
    )

    print(
        f"Duplicate identities: "
        f"{duplicate_identities}"
    )

    # --------------------------------------------------------------
    # SAVE MODE-SPECIFIC TRADES FOR FORENSIC COMPARISON
    # --------------------------------------------------------------

    mode_file = None

    if mode_name == "A_FROZEN_REFERENCE":

        mode_file = MODE_A_TRADES_FILE

    elif mode_name == "B_BUFFERED_CALCULATION":

        mode_file = MODE_B_TRADES_FILE

    if mode_file and all_trades:

        mode_columns = sorted(
            {
                key
                for trade in all_trades
                for key in trade.keys()
            }
        )

        with open(
            mode_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=mode_columns,
                extrasaction="ignore"
            )

            writer.writeheader()
            writer.writerows(all_trades)

        print(
            f"Saved forensic trades: {mode_file}"
        )

    return {
        "mode": mode_name,
        "calculation_start": calculation_start,
        "evaluation_start": evaluation_start,
        "trades": all_trades,
        "identities": set(identities),
        "total_r": total_r,
        "duplicate_identities": duplicate_identities,
    }


# ==============================================================
# RUN MODES
# ==============================================================

def main():

    print()
    print("#" * 100)
    print("TRADESENSE-AI THREE-CONTEXT MODE FORENSIC")
    print("#" * 100)

    print()
    print("Production code is NOT modified.")
    print("Frozen historical dataset is NOT modified.")
    print("Holdout dataset is NOT modified.")

    # ----------------------------------------------------------
    # MODE A
    #
    # Exact frozen historical calculation context.
    # ----------------------------------------------------------

    mode_a = run_mode(
        mode_name="A_FROZEN_REFERENCE",
        calculation_start=FROZEN_START_DATE,
        evaluation_start=FROZEN_START_DATE,
    )

    # ----------------------------------------------------------
    # MODE B
    #
    # Buffered calculation context,
    # identical evaluation period.
    # ----------------------------------------------------------

    mode_b = run_mode(
        mode_name="B_BUFFERED_CALCULATION",
        calculation_start=BUFFERED_START_DATE,
        evaluation_start=FROZEN_START_DATE,
    )

    # ----------------------------------------------------------
    # MODE C
    #
    # Rolling / boundary-coupled calculation.
    #
    # Same calculation and evaluation boundary as A.
    # This is intentionally included as a control.
    # ----------------------------------------------------------

    mode_c = run_mode(
        mode_name="C_ROLLING_BOUNDARY",
        calculation_start=FROZEN_START_DATE,
        evaluation_start=FROZEN_START_DATE,
    )

    # ==========================================================
    # MODE COMPARISON
    # ==========================================================

    print()
    print("#" * 100)
    print("MODE COMPARISON")
    print("#" * 100)

    modes = [
        mode_a,
        mode_b,
        mode_c,
    ]

    print()

    for mode in modes:

        print(
            f"{mode['mode']:<28} "
            f"Trades={len(mode['trades']):>4} "
            f"TotalR={mode['total_r']:>10.4f}"
        )

    # ----------------------------------------------------------
    # A vs B
    # ----------------------------------------------------------

    a_ids = mode_a["identities"]
    b_ids = mode_b["identities"]

    a_only = sorted(
        a_ids - b_ids
    )

    b_only = sorted(
        b_ids - a_ids
    )

    common_ab = (
        a_ids & b_ids
    )

    print()
    print("=" * 100)
    print("A vs B — FROZEN vs BUFFERED")
    print("=" * 100)

    print(
        f"Common trades : {len(common_ab)}"
    )

    print(
        f"A-only trades : {len(a_only)}"
    )

    print(
        f"B-only trades : {len(b_only)}"
    )

    print(
        f"Total-R delta : "
        f"{mode_b['total_r'] - mode_a['total_r']:.6f}"
    )

    if a_only:

        print()
        print("A-ONLY TRADE IDENTITIES:")

        for identity in a_only:

            print(
                f"  {identity}"
            )

    if b_only:

        print()
        print("B-ONLY TRADE IDENTITIES:")

        for identity in b_only:

            print(
                f"  {identity}"
            )

    # ----------------------------------------------------------
    # A vs C
    #
    # These should be identical because both use the same
    # calculation and evaluation boundaries.
    # ----------------------------------------------------------

    a_ids = mode_a["identities"]
    c_ids = mode_c["identities"]

    a_only_c = sorted(
        a_ids - c_ids
    )

    c_only_a = sorted(
        c_ids - a_ids
    )

    common_ac = (
        a_ids & c_ids
    )

    print()
    print("=" * 100)
    print("A vs C — CONTROL CHECK")
    print("=" * 100)

    print(
        f"Common trades : {len(common_ac)}"
    )

    print(
        f"A-only trades : {len(a_only_c)}"
    )

    print(
        f"C-only trades : {len(c_only_a)}"
    )

    print(
        f"Total-R delta : "
        f"{mode_c['total_r'] - mode_a['total_r']:.6f}"
    )

    if (
        a_ids == c_ids
        and
        abs(
            mode_a["total_r"]
            -
            mode_c["total_r"]
        ) < 1e-9
    ):

        print(
            "CONTROL RESULT: PASS"
        )

    else:

        print(
            "CONTROL RESULT: FAIL"
        )

    # ==========================================================
    # FROZEN DATASET COMPARISON
    # ==========================================================

    frozen_trades = load_frozen_trades()

    frozen_ids = {
        trade_identity(trade)
        for trade in frozen_trades
    }

    print()
    print("=" * 100)
    print("MODE A vs FROZEN DATASET")
    print("=" * 100)

    print(
        f"Frozen trades : {len(frozen_ids)}"
    )

    print(
        f"Mode A trades : {len(mode_a['identities'])}"
    )

    frozen_only = sorted(
        frozen_ids - mode_a["identities"]
    )

    mode_a_only = sorted(
        mode_a["identities"] - frozen_ids
    )

    print(
        f"Frozen-only   : {len(frozen_only)}"
    )

    print(
        f"Mode-A-only   : {len(mode_a_only)}"
    )

    if not frozen_only and not mode_a_only:

        print(
            "FROZEN IDENTITY REPRODUCTION: PASS"
        )

    else:

        print(
            "FROZEN IDENTITY REPRODUCTION: FAIL"
        )

    # ==========================================================
    # SAVE SUMMARY
    # ==========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "calculation_start",
                "evaluation_start",
                "trade_count",
                "total_r",
                "duplicate_identities",
            ]
        )

        writer.writeheader()

        for mode in modes:

            writer.writerow(
                {
                    "mode": mode["mode"],
                    "calculation_start": mode[
                        "calculation_start"
                    ],
                    "evaluation_start": mode[
                        "evaluation_start"
                    ],
                    "trade_count": len(
                        mode["trades"]
                    ),
                    "total_r": f"{mode['total_r']:.6f}",
                    "duplicate_identities": mode[
                        "duplicate_identities"
                    ],
                }
            )

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    print()
    print("#" * 100)
    print("FINAL RESULT")
    print("#" * 100)

    if (
        not frozen_only
        and
        not mode_a_only
        and
        a_ids == c_ids
        and
        abs(
            mode_a["total_r"]
            -
            mode_c["total_r"]
        ) < 1e-9
    ):

        print(
            "RESULT: DIAGNOSTIC PASS"
        )

    else:

        print(
            "RESULT: INVESTIGATION REQUIRED"
        )

    print()
    print(
        f"Summary saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()