from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(__file__).resolve().parents[1]

FROZEN_TRADES = ROOT / "tests" / "output" / "tradesense_trades.csv"

MTF_COMPONENTS = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
    / "mtf_component_contribution_trade_level.csv"
)

OUTPUT_DIR = (
    ROOT
    / "tests"
    / "output"
    / "reconciliation"
    / "mtf_daily_weekly"
)

TRADE_OUTPUT = OUTPUT_DIR / "mtf_setup_early_interaction_trade_level.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "mtf_setup_early_interaction_summary.csv"

SETUP_ORDER = [
    "LONG_CONTINUATION",
    "LONG_REVERSAL",
    "SHORT_CONTINUATION",
    "SHORT_REVERSAL",
]


# ======================================================================
# HELPERS
# ======================================================================

def normalize_symbol(value):
    value = str(value).strip().upper()
    if value.endswith(".NS"):
        value = value[:-3]
    return value


def find_column(df, candidates, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    if required:
        raise RuntimeError(
            f"Required column not found. Tried: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def normalize_setup(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    aliases = {
        "LC": "LONG_CONTINUATION",
        "LONG CONTINUATION": "LONG_CONTINUATION",
        "LONG_CONT": "LONG_CONTINUATION",

        "LR": "LONG_REVERSAL",
        "LONG REVERSAL": "LONG_REVERSAL",
        "LONG_REV": "LONG_REVERSAL",

        "SC": "SHORT_CONTINUATION",
        "SHORT CONTINUATION": "SHORT_CONTINUATION",
        "SHORT_CONT": "SHORT_CONTINUATION",

        "SR": "SHORT_REVERSAL",
        "SHORT REVERSAL": "SHORT_REVERSAL",
        "SHORT_REV": "SHORT_REVERSAL",
    }

    return aliases.get(text, text)


def safe_pf(r_values):
    r_values = pd.Series(r_values).dropna()

    gross_profit = r_values[r_values > 0].sum()
    gross_loss = -r_values[r_values < 0].sum()

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        return 0.0

    return gross_profit / gross_loss


def group_stats(df):
    if df.empty:
        return {
            "N": 0,
            "Win": np.nan,
            "PF": np.nan,
            "R": 0.0,
            "Avg": np.nan,
        }

    r = pd.to_numeric(df["R_Multiple"], errors="coerce").dropna()

    if r.empty:
        return {
            "N": 0,
            "Win": np.nan,
            "PF": np.nan,
            "R": 0.0,
            "Avg": np.nan,
        }

    return {
        "N": len(r),
        "Win": (r > 0).mean() * 100.0,
        "PF": safe_pf(r),
        "R": r.sum(),
        "Avg": r.mean(),
    }


def print_stats(label, df):
    s = group_stats(df)

    pf_text = "inf" if np.isinf(s["PF"]) else f"{s['PF']:.3f}"

    print(
        f"{label:<34}"
        f"N={s['N']:>3} "
        f"Win={s['Win']:>6.2f}% "
        f"PF={pf_text:>6} "
        f"R={s['R']:>8.2f} "
        f"Avg={s['Avg']:>8.4f}"
    )

    return s


def delta_stats(a, b):
    return {
        "win_delta_pp": a["Win"] - b["Win"],
        "r_delta": a["R"] - b["R"],
        "avg_delta": a["Avg"] - b["Avg"],
    }


# ======================================================================
# LOAD FROZEN TRADES
# ======================================================================

def load_frozen_trades():

    trades = pd.read_csv(FROZEN_TRADES)

    symbol_col = find_column(
        trades,
        [
            "_symbol",
            "Symbol",
            "symbol",
            "Ticker",
            "ticker",
        ],
    )

    date_col = find_column(
        trades,
        [
            "Signal_Date",
            "signal_date",
            "SignalDate",
        ],
    )

    r_col = find_column(
        trades,
        [
            "R_Multiple",
            "r_multiple",
            "R",
            "r",
        ],
    )

    setup_col = find_column(
        trades,
        [
            "Setup",
            "setup",
            "Setup_Type",
            "setup_type",
            "SetupType",
            "setup_name",
        ],
    )

    direction_col = find_column(
        trades,
        [
            "Direction",
            "direction",
        ],
    )

    trades["_symbol_root"] = trades[symbol_col].map(normalize_symbol)
    trades["Signal_Date"] = pd.to_datetime(
        trades[date_col],
        errors="coerce",
    )
    trades["R_Multiple"] = pd.to_numeric(
        trades[r_col],
        errors="coerce",
    )
    trades["Setup_Normalized"] = trades[setup_col].map(normalize_setup)
    trades["Direction_Normalized"] = (
        trades[direction_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    trades = trades.dropna(
        subset=[
            "_symbol_root",
            "Signal_Date",
            "R_Multiple",
            "Setup_Normalized",
        ]
    ).copy()

    return trades


# ======================================================================
# EARLY-ADVERSE EXTRACTION
# ======================================================================

def extract_early_adverse(trades):

    # --------------------------------------------------------------
    # Preferred explicit boolean/event columns.
    # --------------------------------------------------------------

    boolean_candidates = [
        "early_adverse_triggered",
        "Early_Adverse_Triggered",
        "early_adverse_event",
        "Early_Adverse_Event",
        "early_adverse",
        "Early_Adverse",
        "triggered_early_adverse",
        "Triggered_Early_Adverse",
    ]

    for candidate in boolean_candidates:
        col = find_column(trades, [candidate], required=False)

        if col is None:
            continue

        values = trades[col]

        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)

        normalized = (
            values.astype(str)
            .str.strip()
            .str.upper()
        )

        true_values = {
            "TRUE",
            "1",
            "YES",
            "Y",
            "TRIGGERED",
            "EVENT",
        }

        false_values = {
            "FALSE",
            "0",
            "NO",
            "N",
            "NOT_TRIGGERED",
            "NONE",
        }

        result = pd.Series(
            np.nan,
            index=trades.index,
            dtype="object",
        )

        result[normalized.isin(true_values)] = True
        result[normalized.isin(false_values)] = False

        if result.notna().all():
            return result.astype(bool)

    # --------------------------------------------------------------
    # If the frozen ledger contains early-adverse R columns,
    # reconstruct the event from them.
    # --------------------------------------------------------------

    bar_columns = {}

    for bar in range(1, 6):

        candidates = [
            f"early_adverse_r_bar_{bar}",
            f"Early_Adverse_R_Bar_{bar}",
            f"early_adverse_bar_{bar}_r",
            f"Early_Adverse_Bar_{bar}_R",
        ]

        col = find_column(
            trades,
            candidates,
            required=False,
        )

        if col is not None:
            bar_columns[bar] = col

    if bar_columns:

        event = pd.Series(
            False,
            index=trades.index,
            dtype=bool,
        )

        for col in bar_columns.values():
            values = pd.to_numeric(
                trades[col],
                errors="coerce",
            )

            event |= values <= -0.25

        return event

    raise RuntimeError(
        "Could not determine early-adverse event state from frozen "
        "trade ledger."
    )


# ======================================================================
# LOAD CAUSAL MTF COMPONENTS
# ======================================================================

def load_mtf_components():

    mtf = pd.read_csv(MTF_COMPONENTS)

    symbol_col = find_column(
        mtf,
        [
            "Symbol",
            "symbol",
            "_symbol_root",
            "_symbol",
        ],
    )

    date_col = find_column(
        mtf,
        [
            "Signal_Date",
            "signal_date",
            "SignalDate",
        ],
    )

    daily_col = find_column(
        mtf,
        ["Daily_Relation"],
    )

    weekly_col = find_column(
        mtf,
        ["Weekly_Relation"],
    )

    combined_col = find_column(
        mtf,
        ["Combined_MTF_Class"],
    )

    completed_weekly_col = find_column(
        mtf,
        ["Completed_Weekly_Date"],
        required=False,
    )

    mtf["_symbol_root"] = mtf[symbol_col].map(normalize_symbol)

    mtf["Signal_Date"] = pd.to_datetime(
        mtf[date_col],
        errors="coerce",
    )

    mtf["_mtf_key"] = (
        mtf["_symbol_root"].astype(str)
        + "|"
        + mtf["Signal_Date"].dt.strftime("%Y-%m-%d")
    )

    mtf["Daily_Relation_Normalized"] = (
        mtf[daily_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mtf["Weekly_Relation_Normalized"] = (
        mtf[weekly_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mtf["Combined_MTF_Class_Normalized"] = (
        mtf[combined_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if completed_weekly_col is not None:
        mtf["Completed_Weekly_Date_Normalized"] = pd.to_datetime(
            mtf[completed_weekly_col],
            errors="coerce",
        )
    else:
        mtf["Completed_Weekly_Date_Normalized"] = pd.NaT

    return mtf


# ======================================================================
# CAUSALITY CHECK
# ======================================================================

def validate_weekly_causality(analysis):

    valid = (
        analysis["Completed_Weekly_Date_Normalized"].notna()
        & analysis["Signal_Date"].notna()
    )

    passed = (
        analysis.loc[valid, "Completed_Weekly_Date_Normalized"]
        < analysis.loc[valid, "Signal_Date"]
    )

    return int(passed.sum()), int(valid.sum())


# ======================================================================
# BUILD INTERACTION VARIABLES
# ======================================================================

def build_interaction_variables(analysis):

    analysis["Daily_Aligned"] = (
        analysis["Daily_Relation_Normalized"] == "ALIGNED"
    )

    analysis["Weekly_Aligned"] = (
        analysis["Weekly_Relation_Normalized"] == "ALIGNED"
    )

    analysis["Full_MTF_Aligned"] = (
        analysis["Combined_MTF_Class_Normalized"]
        == "FULL_ALIGNMENT"
    )

    analysis["Full_MTF_Conflict"] = (
        analysis["Combined_MTF_Class_Normalized"]
        == "FULL_CONFLICT"
    )

    analysis["MTF_Not_Full_Aligned"] = ~analysis["Full_MTF_Aligned"]

    return analysis


# ======================================================================
# DID CALCULATION
# ======================================================================

def calculate_did(df, mtf_col):

    aligned = df[df[mtf_col]].copy()
    not_aligned = df[~df[mtf_col]].copy()

    aligned_event = aligned[aligned["Early_Adverse"]]
    aligned_no_event = aligned[~aligned["Early_Adverse"]]

    not_event = not_aligned[not_aligned["Early_Adverse"]]
    not_no_event = not_aligned[~not_aligned["Early_Adverse"]]

    a_event = group_stats(aligned_event)
    a_no_event = group_stats(aligned_no_event)
    n_event = group_stats(not_event)
    n_no_event = group_stats(not_no_event)

    # Effect of early adverse inside MTF-aligned group.
    aligned_effect = (
        a_event["Avg"] - a_no_event["Avg"]
        if a_event["N"] > 0 and a_no_event["N"] > 0
        else np.nan
    )

    # Effect of early adverse inside MTF-not-aligned group.
    not_aligned_effect = (
        n_event["Avg"] - n_no_event["Avg"]
        if n_event["N"] > 0 and n_no_event["N"] > 0
        else np.nan
    )

    did = (
        aligned_effect - not_aligned_effect
        if pd.notna(aligned_effect)
        and pd.notna(not_aligned_effect)
        else np.nan
    )

    return {
        "aligned_event": a_event,
        "aligned_no_event": a_no_event,
        "not_aligned_event": n_event,
        "not_aligned_no_event": n_no_event,
        "aligned_early_effect": aligned_effect,
        "not_aligned_early_effect": not_aligned_effect,
        "DID": did,
    }


# ======================================================================
# PERMUTATION TEST
# ======================================================================

def permutation_did(df, mtf_col, iterations=10000, seed=42):

    work = df[
        [
            mtf_col,
            "Early_Adverse",
            "R_Multiple",
        ]
    ].dropna().copy()

    if len(work) < 4:
        return np.nan, 0

    observed = calculate_did(
        df.loc[work.index],
        mtf_col,
    )["DID"]

    if pd.isna(observed):
        return np.nan, 0

    rng = np.random.default_rng(seed)

    values = work["R_Multiple"].to_numpy()
    mtf = work[mtf_col].to_numpy()
    event = work["Early_Adverse"].to_numpy()

    null_values = []

    for _ in range(iterations):

        shuffled_mtf = rng.permutation(mtf)

        temp = pd.DataFrame(
            {
                "MTF": shuffled_mtf,
                "Early_Adverse": event,
                "R_Multiple": values,
            }
        )

        result = calculate_did(
            temp,
            "MTF",
        )["DID"]

        if pd.notna(result):
            null_values.append(result)

    if not null_values:
        return observed, 0

    null_values = np.asarray(null_values)

    p_value = (
        np.sum(np.abs(null_values) >= abs(observed)) + 1
    ) / (len(null_values) + 1)

    return observed, float(p_value)


# ======================================================================
# BOOTSTRAP CI
# ======================================================================

def bootstrap_did(df, mtf_col, iterations=10000, seed=123):

    work = df[
        [
            mtf_col,
            "Early_Adverse",
            "R_Multiple",
        ]
    ].dropna().copy()

    observed = calculate_did(
        df.loc[work.index],
        mtf_col,
    )["DID"]

    if pd.isna(observed) or len(work) < 4:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = work.index.to_numpy()
    boot = []

    for _ in range(iterations):

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )

        sample = df.loc[sampled].copy()

        value = calculate_did(
            sample,
            mtf_col,
        )["DID"]

        if pd.notna(value):
            boot.append(value)

    if not boot:
        return observed, np.nan, np.nan

    low, high = np.percentile(
        np.asarray(boot),
        [2.5, 97.5],
    )

    return observed, float(low), float(high)


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("MTF × SETUP × EARLY-ADVERSE INTERACTION FORENSIC")
    print("=" * 70)

    trades = load_frozen_trades()
    mtf = load_mtf_components()

    print(f"Frozen trades loaded : {len(trades)}")
    print(f"MTF component rows   : {len(mtf)}")

    trades["_trade_key"] = (
        trades["_symbol_root"].astype(str)
        + "|"
        + trades["Signal_Date"].dt.strftime("%Y-%m-%d")
    )

    print(
        f"Frozen duplicate keys: "
        f"{trades['_trade_key'].duplicated().sum()}"
    )

    print(
        f"MTF duplicate keys   : "
        f"{mtf['_mtf_key'].duplicated().sum()}"
    )

    analysis = trades.merge(
        mtf[
            [
                "_mtf_key",
                "Daily_Relation_Normalized",
                "Weekly_Relation_Normalized",
                "Combined_MTF_Class_Normalized",
                "Completed_Weekly_Date_Normalized",
            ]
        ],
        left_on="_trade_key",
        right_on="_mtf_key",
        how="left",
        validate="one_to_one",
    )

    missing = analysis[
        analysis["Daily_Relation_Normalized"].isna()
    ]

    print(f"Missing MTF rows     : {len(missing)}")

    if len(missing) > 0:
        raise RuntimeError(
            "MTF merge incomplete. Aborting before analysis."
        )

    # --------------------------------------------------------------
    # Reconciliation
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("RECONCILIATION")
    print("=" * 70)

    frozen_total = trades["R_Multiple"].sum()
    analyzed_total = analysis["R_Multiple"].sum()
    r_delta = analyzed_total - frozen_total

    print(f"Frozen total R       : {frozen_total:.2f}")
    print(f"Analyzed total R     : {analyzed_total:.2f}")
    print(f"R delta              : {r_delta:.10f}")

    if abs(r_delta) < 1e-9:
        print("R reconciliation     : PASS")
    else:
        print("R reconciliation     : FAIL")
        raise RuntimeError("R reconciliation failed.")

    # --------------------------------------------------------------
    # Causality
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("CAUSAL TIMESTAMP")
    print("=" * 70)

    passed, valid = validate_weekly_causality(analysis)

    print(f"Weekly causal PASS   : {passed}/{valid}")

    if passed == valid:
        print("Weekly timestamp causality: PASS")
    else:
        print("Weekly timestamp causality: FAIL")
        raise RuntimeError("Weekly timestamp causality failed.")

    # --------------------------------------------------------------
    # Early adverse
    # --------------------------------------------------------------

    analysis["Early_Adverse"] = extract_early_adverse(
        analysis
    )

    print()
    print("=" * 70)
    print("EARLY-ADVERSE")
    print("=" * 70)

    event_count = int(analysis["Early_Adverse"].sum())
    event_rate = analysis["Early_Adverse"].mean() * 100.0

    print(f"Known event state    : {analysis['Early_Adverse'].notna().sum()}/{len(analysis)}")
    print(f"Event count          : {event_count}")
    print(f"Event rate           : {event_rate:.2f}%")

    build_interaction_variables(analysis)

    # --------------------------------------------------------------
    # Main DID analyses
    # --------------------------------------------------------------

    tests = [
        (
            "DAILY_ALIGNMENT",
            "Daily_Aligned",
        ),
        (
            "WEEKLY_ALIGNMENT",
            "Weekly_Aligned",
        ),
        (
            "FULL_MTF_ALIGNMENT",
            "Full_MTF_Aligned",
        ),
    ]

    summary_rows = []

    print()
    print("=" * 70)
    print("GLOBAL INTERACTION EFFECTS")
    print("=" * 70)

    for name, mtf_col in tests:

        print()
        print(f"--- {name} ---")

        result = calculate_did(
            analysis,
            mtf_col,
        )

        print_stats(
            "MTF aligned + EVENT",
            analysis[
                analysis[mtf_col]
                & analysis["Early_Adverse"]
            ],
        )

        print_stats(
            "MTF aligned + NO EVENT",
            analysis[
                analysis[mtf_col]
                & ~analysis["Early_Adverse"]
            ],
        )

        print_stats(
            "MTF not aligned + EVENT",
            analysis[
                ~analysis[mtf_col]
                & analysis["Early_Adverse"]
            ],
        )

        print_stats(
            "MTF not aligned + NO EVENT",
            analysis[
                ~analysis[mtf_col]
                & ~analysis["Early_Adverse"]
            ],
        )

        print(
            f"Aligned early-adverse effect : "
            f"{result['aligned_early_effect']:+.4f}R"
        )

        print(
            f"Not-aligned early effect      : "
            f"{result['not_aligned_early_effect']:+.4f}R"
        )

        print(
            f"DID interaction              : "
            f"{result['DID']:+.4f}R"
        )

        observed, p_value = permutation_did(
            analysis,
            mtf_col,
        )

        boot_obs, ci_low, ci_high = bootstrap_did(
            analysis,
            mtf_col,
        )

        print(
            f"Permutation p-value           : "
            f"{p_value:.6f}"
        )

        print(
            f"Bootstrap 95% CI              : "
            f"[{ci_low:+.4f}, {ci_high:+.4f}]R"
        )

        summary_rows.append(
            {
                "Scope": "GLOBAL",
                "Setup": "ALL",
                "MTF_Test": name,
                "N": len(analysis),
                "Aligned_Event_N": result["aligned_event"]["N"],
                "Aligned_NoEvent_N": result["aligned_no_event"]["N"],
                "NotAligned_Event_N": result["not_aligned_event"]["N"],
                "NotAligned_NoEvent_N": result["not_aligned_no_event"]["N"],
                "Aligned_Early_Effect_R": result["aligned_early_effect"],
                "NotAligned_Early_Effect_R": result["not_aligned_early_effect"],
                "DID_R": result["DID"],
                "Permutation_P": p_value,
                "Bootstrap_Low": ci_low,
                "Bootstrap_High": ci_high,
            }
        )

    # --------------------------------------------------------------
    # Setup conditional DID
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SETUP-CONDITIONAL INTERACTION")
    print("=" * 70)

    for setup in SETUP_ORDER:

        subset = analysis[
            analysis["Setup_Normalized"] == setup
        ].copy()

        print()
        print(f"SETUP: {setup} (N={len(subset)})")

        for name, mtf_col in tests:

            print()
            print(f"  {name}")

            result = calculate_did(
                subset,
                mtf_col,
            )

            print(
                f"    aligned + EVENT       N={result['aligned_event']['N']:>2} "
                f"R={result['aligned_event']['R']:+7.2f} "
                f"Avg={result['aligned_event']['Avg']:+.4f}"
            )

            print(
                f"    aligned + NO EVENT    N={result['aligned_no_event']['N']:>2} "
                f"R={result['aligned_no_event']['R']:+7.2f} "
                f"Avg={result['aligned_no_event']['Avg']:+.4f}"
            )

            print(
                f"    not aligned + EVENT   N={result['not_aligned_event']['N']:>2} "
                f"R={result['not_aligned_event']['R']:+7.2f} "
                f"Avg={result['not_aligned_event']['Avg']:+.4f}"
            )

            print(
                f"    not aligned + NO EVT  N={result['not_aligned_no_event']['N']:>2} "
                f"R={result['not_aligned_no_event']['R']:+7.2f} "
                f"Avg={result['not_aligned_no_event']['Avg']:+.4f}"
            )

            print(
                f"    aligned event effect  : "
                f"{result['aligned_early_effect']:+.4f}R"
            )

            print(
                f"    not-aligned effect    : "
                f"{result['not_aligned_early_effect']:+.4f}R"
            )

            print(
                f"    DID interaction       : "
                f"{result['DID']:+.4f}R"
            )

            observed, p_value = permutation_did(
                subset,
                mtf_col,
                iterations=10000,
            )

            boot_obs, ci_low, ci_high = bootstrap_did(
                subset,
                mtf_col,
                iterations=10000,
            )

            print(
                f"    permutation p-value   : "
                f"{p_value:.6f}"
            )

            print(
                f"    bootstrap 95% CI     : "
                f"[{ci_low:+.4f}, {ci_high:+.4f}]R"
            )

            summary_rows.append(
                {
                    "Scope": "SETUP",
                    "Setup": setup,
                    "MTF_Test": name,
                    "N": len(subset),
                    "Aligned_Event_N": result["aligned_event"]["N"],
                    "Aligned_NoEvent_N": result["aligned_no_event"]["N"],
                    "NotAligned_Event_N": result["not_aligned_event"]["N"],
                    "NotAligned_NoEvent_N": result["not_aligned_no_event"]["N"],
                    "Aligned_Early_Effect_R": result["aligned_early_effect"],
                    "NotAligned_Early_Effect_R": result["not_aligned_early_effect"],
                    "DID_R": result["DID"],
                    "Permutation_P": p_value,
                    "Bootstrap_Low": ci_low,
                    "Bootstrap_High": ci_high,
                }
            )

    # --------------------------------------------------------------
    # Direction × setup × interaction
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("DIRECTION × SETUP × MTF × EARLY-ADVERSE")
    print("=" * 70)

    for direction in ["LONG", "SHORT"]:

        for setup in SETUP_ORDER:

            subset = analysis[
                (analysis["Direction_Normalized"] == direction)
                & (analysis["Setup_Normalized"] == setup)
            ].copy()

            if subset.empty:
                continue

            print()
            print(
                f"{direction} / {setup} "
                f"(N={len(subset)})"
            )

            for name, mtf_col in tests:

                result = calculate_did(
                    subset,
                    mtf_col,
                )

                print(
                    f"  {name:<24} "
                    f"N={len(subset):>2} "
                    f"DID={result['DID']:+.4f}R "
                    f"AE={result['aligned_early_effect']:+.4f}R "
                    f"NAE={result['not_aligned_early_effect']:+.4f}R"
                )

    # --------------------------------------------------------------
    # Symbol LOO
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("SYMBOL LEAVE-ONE-OUT INTERACTION")
    print("=" * 70)

    symbols = sorted(
        analysis["_symbol_root"].dropna().unique()
    )

    for name, mtf_col in tests:

        print()
        print(f"--- {name} ---")

        loo_rows = []

        for symbol in symbols:

            subset = analysis[
                analysis["_symbol_root"] != symbol
            ].copy()

            result = calculate_did(
                subset,
                mtf_col,
            )

            print(
                f"  Excluding {symbol:<12} "
                f"N={len(subset):>3} "
                f"DID={result['DID']:+.4f}R"
            )

            loo_rows.append(
                {
                    "Scope": "SYMBOL_LOO",
                    "Setup": "ALL",
                    "MTF_Test": name,
                    "Excluded": symbol,
                    "N": len(subset),
                    "DID_R": result["DID"],
                }
            )

        summary_rows.extend(loo_rows)

    # --------------------------------------------------------------
    # Year LOO
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("YEAR LEAVE-ONE-OUT INTERACTION")
    print("=" * 70)

    analysis["_Year"] = analysis["Signal_Date"].dt.year

    years = sorted(
        analysis["_Year"].dropna().unique()
    )

    for name, mtf_col in tests:

        print()
        print(f"--- {name} ---")

        for year in years:

            subset = analysis[
                analysis["_Year"] != year
            ].copy()

            result = calculate_did(
                subset,
                mtf_col,
            )

            print(
                f"  Excluding {year} "
                f"N={len(subset):>3} "
                f"DID={result['DID']:+.4f}R"
            )

            summary_rows.append(
                {
                    "Scope": "YEAR_LOO",
                    "Setup": "ALL",
                    "MTF_Test": name,
                    "Excluded": year,
                    "N": len(subset),
                    "DID_R": result["DID"],
                }
            )

    # --------------------------------------------------------------
    # Compact verdict
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("VERDICT FRAMEWORK")
    print("=" * 70)

    global_results = []

    for name, mtf_col in tests:

        result = calculate_did(
            analysis,
            mtf_col,
        )

        _, p_value = permutation_did(
            analysis,
            mtf_col,
            iterations=10000,
        )

        _, ci_low, ci_high = bootstrap_did(
            analysis,
            mtf_col,
            iterations=10000,
        )

        global_results.append(
            (
                name,
                result["DID"],
                p_value,
                ci_low,
                ci_high,
            )
        )

    for name, did, p, low, high in global_results:

        if pd.isna(did):
            verdict = "INSUFFICIENT DATA"

        elif did > 0 and low > 0 and p < 0.05:
            verdict = "STATISTICALLY SUPPORTED POSITIVE INTERACTION"

        elif did > 0:
            verdict = "PROMISING BUT NOT CONFIRMED"

        elif did < 0 and high < 0 and p < 0.05:
            verdict = "STATISTICALLY SUPPORTED NEGATIVE INTERACTION"

        else:
            verdict = "NO RELIABLE INTERACTION EVIDENCE"

        print(
            f"{name:<24} "
            f"DID={did:+.4f}R "
            f"p={p:.6f} "
            f"CI=[{low:+.4f},{high:+.4f}] "
            f"-> {verdict}"
        )

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    analysis.to_csv(
        TRADE_OUTPUT,
        index=False,
    )

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(f"Trade-level : {TRADE_OUTPUT}")
    print(f"Summary     : {SUMMARY_OUTPUT}")

    print()
    print("STATUS: MTF_SETUP_EARLY_INTERACTION_COMPLETE")


if __name__ == "__main__":
    main()