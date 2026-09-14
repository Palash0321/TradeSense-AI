import csv
import os
import math


# ======================================================================
# CONFIGURATION
# ======================================================================

MODE_A_FILE = (
    "tests/output/reconciliation/mode_a_frozen_trades.csv"
)

MODE_B_FILE = (
    "tests/output/reconciliation/mode_b_buffered_trades.csv"
)

OUTPUT_DIR = (
    "tests/output/reconciliation"
)

COMMON_FILE = os.path.join(
    OUTPUT_DIR,
    "context_common_trade_outcomes.csv"
)

DIFFERENCE_FILE = os.path.join(
    OUTPUT_DIR,
    "context_identity_differences.csv"
)


# ======================================================================
# HELPERS
# ======================================================================

def read_csv(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        return list(
            csv.DictReader(file)
        )


def identity(trade):

    return (
        str(
            trade.get("_symbol", "")
        ),
        str(
            trade.get("entry_date", "")
        )[:10],
        str(
            trade.get("direction", "")
        ),
    )


def number(value):

    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:

        result = float(text)

        if math.isfinite(result):
            return result

    except (
        TypeError,
        ValueError
    ):
        pass

    return None


def first_value(trade, *fields):

    for field in fields:

        if field in trade:

            value = number(
                trade.get(field)
            )

            if value is not None:
                return value

    return None


def get_r(trade):

    return first_value(
        trade,
        "r_multiple",
        "R",
        "r"
    )


def get_entry_price(trade):

    return first_value(
        trade,
        "entry_price",
        "Entry_Price"
    )


def get_exit_price(trade):

    return first_value(
        trade,
        "exit_price",
        "Exit_Price"
    )


def get_bars(trade):

    return first_value(
        trade,
        "bars_held",
        "Bars_Held"
    )


def get_mfe(trade):

    return first_value(
        trade,
        "mfe_r",
        "MFE_R",
        "mfe"
    )


def get_mae(trade):

    return first_value(
        trade,
        "mae_r",
        "MAE_R",
        "mae"
    )


# ======================================================================
# LOAD DATA
# ======================================================================

print()
print("=" * 100)
print("TRADESENSE-AI — FROZEN VS BUFFERED TRADE OUTCOME FORENSIC")
print("=" * 100)

mode_a = read_csv(
    MODE_A_FILE
)

mode_b = read_csv(
    MODE_B_FILE
)

print()
print(
    f"Mode A trades: {len(mode_a)}"
)

print(
    f"Mode B trades: {len(mode_b)}"
)


# ======================================================================
# BUILD MAPS
# ======================================================================

a_map = {
    identity(trade): trade
    for trade in mode_a
}

b_map = {
    identity(trade): trade
    for trade in mode_b
}

a_ids = set(
    a_map.keys()
)

b_ids = set(
    b_map.keys()
)

common_ids = (
    a_ids & b_ids
)

a_only = (
    a_ids - b_ids
)

b_only = (
    b_ids - a_ids
)


# ======================================================================
# COMMON TRADE OUTCOMES
# ======================================================================

common_rows = []

common_r_a = 0.0
common_r_b = 0.0

common_r_delta = 0.0

r_changed_count = 0

entry_price_changed = 0
exit_price_changed = 0
bars_changed = 0
mfe_changed = 0
mae_changed = 0


for key in sorted(common_ids):

    a_trade = a_map[key]
    b_trade = b_map[key]

    a_r = get_r(a_trade)
    b_r = get_r(b_trade)

    if a_r is None:
        a_r = 0.0

    if b_r is None:
        b_r = 0.0

    delta_r = (
        b_r - a_r
    )

    common_r_a += a_r
    common_r_b += b_r
    common_r_delta += delta_r

    if abs(delta_r) > 1e-9:

        r_changed_count += 1

    a_entry = get_entry_price(a_trade)
    b_entry = get_entry_price(b_trade)

    a_exit = get_exit_price(a_trade)
    b_exit = get_exit_price(b_trade)

    a_bars = get_bars(a_trade)
    b_bars = get_bars(b_trade)

    a_mfe = get_mfe(a_trade)
    b_mfe = get_mfe(b_trade)

    a_mae = get_mae(a_trade)
    b_mae = get_mae(b_trade)

    if (
        a_entry is not None
        and b_entry is not None
        and abs(a_entry - b_entry) > 1e-9
    ):

        entry_price_changed += 1

    if (
        a_exit is not None
        and b_exit is not None
        and abs(a_exit - b_exit) > 1e-9
    ):

        exit_price_changed += 1

    if (
        a_bars is not None
        and b_bars is not None
        and abs(a_bars - b_bars) > 1e-9
    ):

        bars_changed += 1

    if (
        a_mfe is not None
        and b_mfe is not None
        and abs(a_mfe - b_mfe) > 1e-9
    ):

        mfe_changed += 1

    if (
        a_mae is not None
        and b_mae is not None
        and abs(a_mae - b_mae) > 1e-9
    ):

        mae_changed += 1

    common_rows.append(
        {
            "_symbol": key[0],
            "entry_date": key[1],
            "direction": key[2],
            "A_R": f"{a_r:.6f}",
            "B_R": f"{b_r:.6f}",
            "R_delta_B_minus_A": f"{delta_r:.6f}",
            "A_entry_price": a_entry,
            "B_entry_price": b_entry,
            "A_exit_price": a_exit,
            "B_exit_price": b_exit,
            "A_bars_held": a_bars,
            "B_bars_held": b_bars,
            "A_MFE_R": a_mfe,
            "B_MFE_R": b_mfe,
            "A_MAE_R": a_mae,
            "B_MAE_R": b_mae,
        }
    )


# ======================================================================
# IDENTITY DIFFERENCES
# ======================================================================

difference_rows = []


for key in sorted(a_only):

    trade = a_map[key]

    difference_rows.append(
        {
            "_symbol": key[0],
            "entry_date": key[1],
            "direction": key[2],
            "context": "A_ONLY",
            "R": get_r(trade),
            "exit_reason": trade.get(
                "exit_reason"
            ),
            "bars_held": get_bars(trade),
            "MFE_R": get_mfe(trade),
            "MAE_R": get_mae(trade),
            "Setup": trade.get("Setup"),
            "Setup_Direction": trade.get(
                "Setup_Direction"
            ),
            "Structure": trade.get(
                "Structure"
            ),
            "BOS": trade.get("BOS"),
            "CHOCH": trade.get("CHOCH"),
            "Break_Direction": trade.get(
                "Break_Direction"
            ),
            "Break_Confirmed": trade.get(
                "Break_Confirmed"
            ),
            "Break_Age": trade.get(
                "Break_Age"
            ),
            "Trend": trade.get("Trend"),
            "Trend_Strength": trade.get(
                "Trend_Strength"
            ),
            "Liquidity_Sweep": trade.get(
                "Liquidity_Sweep"
            ),
            "Signal_Date": trade.get(
                "Signal_Date"
            ),
        }
    )


for key in sorted(b_only):

    trade = b_map[key]

    difference_rows.append(
        {
            "_symbol": key[0],
            "entry_date": key[1],
            "direction": key[2],
            "context": "B_ONLY",
            "R": get_r(trade),
            "exit_reason": trade.get(
                "exit_reason"
            ),
            "bars_held": get_bars(trade),
            "MFE_R": get_mfe(trade),
            "MAE_R": get_mae(trade),
            "Setup": trade.get("Setup"),
            "Setup_Direction": trade.get(
                "Setup_Direction"
            ),
            "Structure": trade.get(
                "Structure"
            ),
            "BOS": trade.get("BOS"),
            "CHOCH": trade.get("CHOCH"),
            "Break_Direction": trade.get(
                "Break_Direction"
            ),
            "Break_Confirmed": trade.get(
                "Break_Confirmed"
            ),
            "Break_Age": trade.get(
                "Break_Age"
            ),
            "Trend": trade.get("Trend"),
            "Trend_Strength": trade.get(
                "Trend_Strength"
            ),
            "Liquidity_Sweep": trade.get(
                "Liquidity_Sweep"
            ),
            "Signal_Date": trade.get(
                "Signal_Date"
            ),
        }
    )


# ======================================================================
# R OF IDENTITY DIFFERENCES
# ======================================================================

a_only_r = sum(
    (
        get_r(a_map[key]) or 0.0
    )
    for key in a_only
)

b_only_r = sum(
    (
        get_r(b_map[key]) or 0.0
    )
    for key in b_only
)

identity_delta = (
    b_only_r - a_only_r
)

reconstructed_total_delta = (
    common_r_delta
    +
    identity_delta
)


# ======================================================================
# PRINT COMMON TRADE RESULTS
# ======================================================================

print()
print("=" * 100)
print("COMMON TRADE OUTCOME ANALYSIS")
print("=" * 100)

print(
    f"Common trades              : {len(common_ids)}"
)

print(
    f"Common A total R           : {common_r_a:.6f}"
)

print(
    f"Common B total R           : {common_r_b:.6f}"
)

print(
    f"Common R delta (B - A)     : {common_r_delta:.6f}"
)

print(
    f"Common trades with R change: {r_changed_count}"
)

print(
    f"Entry price changes        : {entry_price_changed}"
)

print(
    f"Exit price changes         : {exit_price_changed}"
)

print(
    f"Bars-held changes          : {bars_changed}"
)

print(
    f"MFE changes                : {mfe_changed}"
)

print(
    f"MAE changes                : {mae_changed}"
)


# ======================================================================
# PRINT IDENTITY CONTRIBUTION
# ======================================================================

print()
print("=" * 100)
print("IDENTITY-DIFFERENCE CONTRIBUTION")
print("=" * 100)

print(
    f"A-only trades : {len(a_only)}"
)

print(
    f"A-only total R: {a_only_r:.6f}"
)

print(
    f"B-only trades : {len(b_only)}"
)

print(
    f"B-only total R: {b_only_r:.6f}"
)

print(
    f"Identity delta (B-only - A-only): "
    f"{identity_delta:.6f}"
)

print()
print(
    f"Reconstructed B-A delta: "
    f"{reconstructed_total_delta:.6f}"
)


# ======================================================================
# PRINT ALL IDENTITY DIFFERENCES
# ======================================================================

print()
print("=" * 100)
print("IDENTITY DIFFERENCES")
print("=" * 100)

for row in difference_rows:

    print(
        f"{row['_symbol']:<18} "
        f"{row['entry_date']} "
        f"{row['direction']:<6} "
        f"{row['context']:<8} "
        f"R={row['R']}"
    )


# ======================================================================
# SAVE COMMON OUTCOMES
# ======================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

if common_rows:

    common_fields = list(
        common_rows[0].keys()
    )

    with open(
        COMMON_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=common_fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(common_rows)


# ======================================================================
# SAVE IDENTITY DIFFERENCES
# ======================================================================

if difference_rows:

    difference_fields = list(
        difference_rows[0].keys()
    )

    with open(
        DIFFERENCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=difference_fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(difference_rows)


# ======================================================================
# FINAL VALIDATION
# ======================================================================

a_total = sum(
    (
        get_r(a_map[key]) or 0.0
    )
    for key in a_ids
)

b_total = sum(
    (
        get_r(b_map[key]) or 0.0
    )
    for key in b_ids
)

print()
print("=" * 100)
print("TOTAL-R RECONCILIATION")
print("=" * 100)

print(
    f"Mode A total R             : {a_total:.6f}"
)

print(
    f"Mode B total R             : {b_total:.6f}"
)

print(
    f"Actual B-A delta           : "
    f"{b_total - a_total:.6f}"
)

print(
    f"Reconstructed B-A delta    : "
    f"{reconstructed_total_delta:.6f}"
)

reconciliation_ok = (
    abs(
        (
            b_total - a_total
        )
        -
        reconstructed_total_delta
    )
    < 1e-9
)

print(
    f"R reconciliation           : "
    f"{'PASS' if reconciliation_ok else 'FAIL'}"
)

print()
print(
    f"Common outcomes saved to   : {COMMON_FILE}"
)

print(
    f"Identity differences saved : {DIFFERENCE_FILE}"
)

print()
print("=" * 100)
print("RESULT")
print("=" * 100)

if reconciliation_ok:

    print(
        "RESULT: A vs B R-DIFFERENCE "
        "IS FULLY RECONCILED."
    )

else:

    print(
        "RESULT: R-DIFFERENCE REQUIRES "
        "FURTHER INVESTIGATION."
    )

print()
print(
    "IMPORTANT: NO PRODUCTION OR FROZEN "
    "DATA WAS MODIFIED."
)