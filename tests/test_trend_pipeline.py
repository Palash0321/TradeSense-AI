import pandas as pd

from app.services.backtest_service import BacktestService
from app.core.levels.market_structure import calculate_market_structure



SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "SBIN.NS",
]

INITIAL_CAPITAL = 100000

# ------------------------------------------------------------------
# FROZEN FUTURE HOLDOUT CONFIGURATION
# ------------------------------------------------------------------

HOLDOUT_START_DATE = "2026-09-09"

HISTORICAL_TRADES_FILE = (
    "tests/output/tradesense_trades.csv"
)

HOLDOUT_TRADES_FILE = (
    "tests/output/tradesense_holdout_trades.csv"
)


def performance_label(trades):
    if not trades:
        return "Trades=0"

    total_trades = len(trades)

    wins = [
        t for t in trades
        if float(t.get("profit", 0)) > 0
    ]

    gross_profit = sum(
        float(t.get("profit", 0))
        for t in trades
        if float(t.get("profit", 0)) > 0
    )

    gross_loss = abs(
        sum(
            float(t.get("profit", 0))
            for t in trades
            if float(t.get("profit", 0)) < 0
        )
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0
    )

    total_profit = sum(
        float(t.get("profit", 0))
        for t in trades
    )

    total_r = sum(
        float(t.get("r_multiple", 0))
        for t in trades
    )

    avg_r = (
        total_r / total_trades
        if total_trades > 0
        else 0
    )

    win_rate = (
        len(wins) / total_trades * 100
        if total_trades > 0
        else 0
    )

    return (
        f"Trades={total_trades:<3} "
        f"Win={win_rate:6.2f}% "
        f"PF={profit_factor:5.2f} "
        f"P&L={total_profit:9.2f} "
        f"R={total_r:7.2f} "
        f"AvgR={avg_r:6.3f}"
    )


def group_performance(trades, *columns):
    groups = {}

    for trade in trades:
        values = []

        for column in columns:
            value = trade.get(column)

            if value is None:
                value = "MISSING"

            values.append(str(value))

        key = " × ".join(values)

        groups.setdefault(key, []).append(trade)

    return groups


def print_group(title, trades, *columns):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    groups = group_performance(trades, *columns)

    if not groups:
        print("NO DATA")
        return

    for key in sorted(
        groups.keys(),
        key=lambda x: (-len(groups[x]), x)
    ):
        print(
            f"{key:<55} "
            f"{performance_label(groups[key])}"
        )


def confidence_bucket(confidence):
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "MISSING"

    if value < 50:
        return "0-49"

    if value < 60:
        return "50-59"

    if value < 70:
        return "60-69"

    if value < 80:
        return "70-79"

    if value < 90:
        return "80-89"

    return "90-100"


def break_age_bucket(age):
    try:
        value = int(age)
    except (TypeError, ValueError):
        return "MISSING"

    if value < 0:
        return "NEGATIVE"

    if value == 0:
        return "AGE_0"

    if value == 1:
        return "AGE_1"

    if value == 2:
        return "AGE_2"

    if value == 3:
        return "AGE_3"

    return "AGE_4+"


def add_diagnostic_fields(trade):
    trade["Diagnostic_Confidence_Bucket"] = confidence_bucket(
        trade.get("setup_confidence")
        if trade.get("setup_confidence") is not None
        else trade.get("Setup_Confidence")
    )

    trade["Diagnostic_Break_Age"] = break_age_bucket(
        trade.get("break_age")
        if trade.get("break_age") is not None
        else trade.get("Break_Age")
    )

    liquidity = trade.get("liquidity_sweep")

    if liquidity is None:
        liquidity = trade.get("Liquidity_Sweep")

    if isinstance(liquidity, str):
        liquidity = liquidity.strip().lower() in (
            "true",
            "1",
            "yes",
        )

    trade["Diagnostic_Liquidity"] = (
        "SWEEP"
        if liquidity
        else "NO_SWEEP"
    )

    setup = trade.get("setup")

    if setup is None:
        setup = trade.get("Setup", "MISSING")

    trade["Diagnostic_Setup"] = setup

    direction = trade.get("direction", "MISSING")

    trade["Diagnostic_Direction"] = direction

    trend = trade.get("entry_trend")

    if trend is None:
        trend = trade.get("Trend", "MISSING")

    trade["Diagnostic_Trend"] = trend

    structure_bias = trade.get("structure_bias")

    if structure_bias is None:
        structure_bias = trade.get("Structure_Bias", "MISSING")

    trade["Diagnostic_Structure_Bias"] = structure_bias

    bos = trade.get("bos")

    if bos is None:
        bos = trade.get("BOS", "MISSING")

    trade["Diagnostic_BOS"] = bos

    choch = trade.get("choch")

    if choch is None:
        choch = trade.get("CHOCH", "MISSING")

    trade["Diagnostic_CHOCH"] = choch

    return trade

def print_trade_level_diagnostic(trades):
    print()
    print("=" * 120)
    print("TRADE-LEVEL ENTRY QUALITY DIAGNOSTIC")
    print("=" * 120)

    if not trades:
        print("NO TRADES")
        return

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    winners = [
        trade
        for trade in trades
        if float(trade.get("profit", 0)) > 0
    ]

    losers = [
        trade
        for trade in trades
        if float(trade.get("profit", 0)) <= 0
    ]

    print()
    print(f"WINNING TRADES: {len(winners)}")
    print(f"LOSING TRADES:  {len(losers)}")

    # --------------------------------------------------------------
    # TRADE TABLE
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("INDIVIDUAL TRADE OUTCOMES")
    print("-" * 120)

    headers = [
        "SYMBOL",
        "ENTRY DATE",
        "DIR",
        "SETUP",
        "TREND",
        "STRENGTH",
        "SLOPE",
        "PRICE POS",
        "BIAS",
        "LIQ",
        "AGE",
        "FIB ZONE",
        "FIB R%",
        "PREM/DISC",
        "EXT",
        "R",
        "MAE%",
        "MFE%",
        "BARS",
        "MAX_FAV_R",
        "MAX_ADV_R",
        "0.5R",
        "1R",
        "1.5R",
        "FAV_FIRST",
        "FAIL",
        "EXIT",
    ]

    print(
        f"{headers[0]:<12}"
        f"{headers[1]:<13}"
        f"{headers[2]:<7}"
        f"{headers[3]:<23}"
        f"{headers[4]:<10}"
        f"{headers[5]:>9}"
        f"{headers[6]:>9}"
        f"{headers[7]:<11}"
        f"{headers[8]:<10}"
        f"{headers[9]:<10}"
        f"{headers[10]:<7}"
        f"{headers[11]:>8}"
        f"{headers[12]:>8}"
        f"{headers[13]:>8}"
        f"{headers[14]:<18}"
        f"{headers[15]:>8}"
        f"{headers[16]:>8}"
        f"{headers[17]:>8}"
        f"{headers[18]:>6}"
        f"{headers[19]:>10}"
        f"{headers[20]:>10}"
        f"{headers[21]:>7}"
        f"{headers[22]:>6}"
        f"{headers[23]:>8}"
        f"{headers[24]:>11}"
        f"{headers[25]:>7}"
        f"{headers[26]:<18}"
    )

    print("-" * 120)

    for trade in trades:
        symbol = trade.get("_symbol", "MISSING")
        entry_date = trade.get("entry_date", "MISSING")

        direction = trade.get("direction", "MISSING")
        setup = trade.get("setup", trade.get("Setup", "MISSING"))
        trend = trade.get("entry_trend", "MISSING")

        strength = trade.get("entry_trend_strength")
        slope = trade.get("entry_trend_slope")
        price_position = trade.get("entry_price_position")

        bias = trade.get(
            "structure_bias",
            trade.get("Structure_Bias", "MISSING")
        )

        liquidity = trade.get(
            "liquidity_sweep",
            trade.get("Liquidity_Sweep")
        )

        if isinstance(liquidity, str):
            liquidity = liquidity.strip().lower() in (
                "true",
                "1",
                "yes",
            )

        liquidity_label = "SWEEP" if liquidity else "NO_SWEEP"

        age = trade.get(
            "break_age",
            trade.get("Break_Age", "MISSING")
        )

        fib_zone = trade.get(
            "Fib_Zone",
            "UNKNOWN"
        )

        fib_retracement = trade.get(
            "Fib_Retracement"
        )

        fib_premium_discount = trade.get(
            "Fib_Premium_Discount",
            "UNKNOWN"
        )

        fib_extension_state = trade.get(
            "Fib_Extension_State",
            "UNKNOWN"
        )

        r_multiple = float(
            trade.get("r_multiple", 0)
        )

        mae = float(
            trade.get("mae_percent", 0)
        )

        mfe = float(
            trade.get("mfe_percent", 0)
        )

        exit_reason = trade.get(
            "exit_reason",
            "MISSING"
        )

        bars_in_trade = trade.get("bars_in_trade", 0)

        max_favorable_r = float(
            trade.get("max_favorable_r", 0)
        )

        max_adverse_r = float(
            trade.get("max_adverse_r", 0)
        )

        reached_0_5r = trade.get(
            "reached_0_5r",
            False
        )

        reached_1r = trade.get(
            "reached_1r",
            False
        )

        reached_1_5r = trade.get(
            "reached_1_5r",
            False
        )

        favorable_first = trade.get(
            "favorable_first"
        )

        immediate_failure = trade.get(
            "immediate_failure",
            False
        )

        try:
            entry_date_display = str(entry_date)[:10]
        except Exception:
            entry_date_display = "MISSING"

        strength_display = (
            f"{float(strength):.1f}"
            if strength is not None
            else "NA"
        )

        slope_display = (
            f"{float(slope):.3f}"
            if slope is not None
            else "NA"
        )

        fib_retracement_display = (
            f"{float(fib_retracement):.1f}"
            if fib_retracement is not None
            else "NA"
        )

        fav_first_display = (
            "YES"
            if favorable_first is True
            else "NO"
            if favorable_first is False
            else "NA"
        )

        print(
            f"{str(symbol):<12}"
            f"{entry_date_display:<13}"
            f"{str(direction):<7}"
            f"{str(setup):<23}"
            f"{str(trend):<10}"
            f"{strength_display:>9}"
            f"{slope_display:>9}"
            f"{str(price_position):<11}"
            f"{str(bias):<10}"
            f"{liquidity_label:<10}"
            f"{str(age):<7}"
            f"{str(fib_zone):<12}"
            f"{fib_retracement_display:>8}"
            f"{str(fib_premium_discount):<13}"
            f"{str(fib_extension_state):<18}"
            f"{r_multiple:>8.2f}"
            f"{mae:>8.2f}"
            f"{mfe:>8.2f}"
            f"{str(bars_in_trade):>6}"
            f"{max_favorable_r:>10.2f}"
            f"{max_adverse_r:>10.2f}"
            f"{'YES' if reached_0_5r else 'NO':>7}"
            f"{'YES' if reached_1r else 'NO':>6}"
            f"{'YES' if reached_1_5r else 'NO':>8}"
            f"{fav_first_display:>11}"
            f"{'YES' if immediate_failure else 'NO':>7}"
            f"{str(exit_reason):<18}"
        )

    # --------------------------------------------------------------
    # WINNERS VS LOSERS
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("WINNERS VS LOSERS")
    print("-" * 120)

    def numeric_average(group, field):
        values = []

        for trade in group:
            value = trade.get(field)

            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass

        if not values:
            return None

        return sum(values) / len(values)

    comparison_fields = [
        ("Trend Strength", "entry_trend_strength"),
        ("Trend Slope", "entry_trend_slope"),
        ("MAE %", "mae_percent"),
        ("MFE %", "mfe_percent"),
        ("Initial Risk", "initial_risk"),
        ("Entry ATR", "entry_atr"),
        ("R Multiple", "r_multiple"),
    ]

    print(
        f"{'METRIC':<25}"
        f"{'WINNERS':>15}"
        f"{'LOSERS':>15}"
        f"{'DIFFERENCE':>15}"
    )

    print("-" * 70)

    for label, field in comparison_fields:
        winner_avg = numeric_average(winners, field)
        loser_avg = numeric_average(losers, field)

        winner_display = (
            f"{winner_avg:.4f}"
            if winner_avg is not None
            else "NA"
        )

        loser_display = (
            f"{loser_avg:.4f}"
            if loser_avg is not None
            else "NA"
        )

        if winner_avg is not None and loser_avg is not None:
            difference = winner_avg - loser_avg
            difference_display = f"{difference:.4f}"
        else:
            difference_display = "NA"

        print(
            f"{label:<25}"
            f"{winner_display:>15}"
            f"{loser_display:>15}"
            f"{difference_display:>15}"
        )

    # --------------------------------------------------------------
    # WIN/LOSS BY ENTRY CONTEXT
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("ENTRY CONTEXT: WINNERS VS LOSERS")
    print("-" * 120)

    context_fields = [
        ("SETUP", "Diagnostic_Setup"),
        ("TREND", "Diagnostic_Trend"),
        ("STRUCTURE BIAS", "Diagnostic_Structure_Bias"),
        ("LIQUIDITY", "Diagnostic_Liquidity"),
        ("DIRECTION", "Diagnostic_Direction"),
        ("PRICE POSITION", "entry_price_position"),
        ("BREAK AGE", "Diagnostic_Break_Age"),
    ]

    for label, field in context_fields:
        print()
        print(f"{label}")

        winner_groups = group_performance(
            winners,
            field
        )

        loser_groups = group_performance(
            losers,
            field
        )

        all_keys = sorted(
            set(winner_groups.keys()) |
            set(loser_groups.keys())
        )

        print(
            f"{'CONTEXT':<30}"
            f"{'WIN TRADES':>12}"
            f"{'LOSS TRADES':>12}"
            f"{'WIN RATE':>12}"
        )

        print("-" * 70)

        for key in all_keys:
            win_count = len(winner_groups.get(key, []))
            loss_count = len(loser_groups.get(key, []))
            total = win_count + loss_count

            win_rate = (
                win_count / total * 100
                if total > 0
                else 0
            )

            print(
                f"{key:<30}"
                f"{win_count:>12}"
                f"{loss_count:>12}"
                f"{win_rate:>11.2f}%"
            )
    # --------------------------------------------------------------
    # BEST / WORST TRADES
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("BEST AND WORST TRADES")
    print("-" * 120)

    sorted_trades = sorted(
        trades,
        key=lambda trade: float(
            trade.get("r_multiple", 0)
        ),
        reverse=True
    )

    print()
    print("TOP 10 TRADES BY R:")

    for trade in sorted_trades[:10]:
        print(
            f"{trade.get('_symbol', 'MISSING'):<12} "
            f"{str(trade.get('entry_date', 'MISSING'))[:10]:<12} "
            f"{str(trade.get('setup', trade.get('Setup', 'MISSING'))):<23} "
            f"{str(trade.get('direction', 'MISSING')):<7} "
            f"R={float(trade.get('r_multiple', 0)):>6.2f} "
            f"MAE={float(trade.get('mae_percent', 0)):>7.2f}% "
            f"MFE={float(trade.get('mfe_percent', 0)):>7.2f}% "
            f"EXIT={trade.get('exit_reason', 'MISSING')}"
        )

    print()
    print("BOTTOM 10 TRADES BY R:")

    for trade in sorted_trades[-10:]:
        print(
            f"{trade.get('_symbol', 'MISSING'):<12} "
            f"{str(trade.get('entry_date', 'MISSING'))[:10]:<12} "
            f"{str(trade.get('setup', trade.get('Setup', 'MISSING'))):<23} "
            f"{str(trade.get('direction', 'MISSING')):<7} "
            f"R={float(trade.get('r_multiple', 0)):>6.2f} "
            f"MAE={float(trade.get('mae_percent', 0)):>7.2f}% "
            f"MFE={float(trade.get('mfe_percent', 0)):>7.2f}% "
            f"EXIT={trade.get('exit_reason', 'MISSING')}"
        )


    # --------------------------------------------------------------
    # MFE QUALITY OF LOSING TRADES
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("MFE QUALITY OF LOSING TRADES")
    print("=" * 120)

    losing_trades = [
        trade
        for trade in trades
        if float(trade.get("profit", 0)) < 0
    ]

    mfe_buckets = {
        "< 0R": [],
        "0R - 0.25R": [],
        "0.25R - 0.50R": [],
        "0.50R - 1.00R": [],
        "1.00R - 1.50R": [],
        ">= 1.50R": []
    }

    for trade in losing_trades:

        mfe = float(
            trade.get("max_favorable_r", 0)
            or 0
        )

        if mfe < 0:
            mfe_buckets["< 0R"].append(trade)

        elif mfe < 0.25:
            mfe_buckets["0R - 0.25R"].append(trade)

        elif mfe < 0.50:
            mfe_buckets["0.25R - 0.50R"].append(trade)

        elif mfe < 1.00:
            mfe_buckets["0.50R - 1.00R"].append(trade)

        elif mfe < 1.50:
            mfe_buckets["1.00R - 1.50R"].append(trade)

        else:
            mfe_buckets[">= 1.50R"].append(trade)

    print()

    for bucket, bucket_trades in mfe_buckets.items():

        print(
            f"{bucket:<18}"
            f"TRADES: {len(bucket_trades):>4}"
        )

    # --------------------------------------------------------------
    # LOSING TRADE MFE BY SETUP
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("LOSING TRADE MFE BY SETUP")
    print("=" * 120)

    setup_groups = {}

    for trade in losing_trades:

        setup = trade.get(
            "setup",
            trade.get("Setup", "UNKNOWN")
        )

        if setup not in setup_groups:
            setup_groups[setup] = []

        setup_groups[setup].append(trade)

    print()

    for setup, setup_trades in sorted(
        setup_groups.items()
    ):

        mfe_values = [
            float(
                trade.get(
                    "max_favorable_r",
                    0
                )
                or 0
            )
            for trade in setup_trades
        ]

        if not mfe_values:
            continue

        average_mfe = (
            sum(mfe_values)
            / len(mfe_values)
        )

        reached_05r = sum(
            1
            for mfe in mfe_values
            if mfe >= 0.50
        )

        reached_10r = sum(
            1
            for mfe in mfe_values
            if mfe >= 1.00
        )

        reached_15r = sum(
            1
            for mfe in mfe_values
            if mfe >= 1.50
        )

        print(
            f"{str(setup):<25}"
            f"LOSERS: {len(setup_trades):>4} "
            f"AVG_MFE: {average_mfe:>7.2f}R "
            f">=0.5R: {reached_05r:>4} "
            f">=1.0R: {reached_10r:>4} "
            f">=1.5R: {reached_15r:>4}"
        )

    # --------------------------------------------------------------
    # LOSING TRADE PROFIT GIVEBACK
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("LOSING TRADE PROFIT GIVEBACK")
    print("=" * 120)

    thresholds = [0.5, 1.0, 1.5]

    for threshold in thresholds:

        eligible = [
            trade
            for trade in losing_trades
            if float(
                trade.get("max_favorable_r", 0)
                or 0
            ) >= threshold
        ]

        print()
        print(
            f"LOSERS REACHING >= {threshold:.1f}R: "
            f"{len(eligible)}"
        )

        if not eligible:
            continue

        avg_mfe = sum(
            float(
                trade.get("max_favorable_r", 0)
                or 0
            )
            for trade in eligible
        ) / len(eligible)

        avg_realized = sum(
            float(
                trade.get("r_multiple", 0)
                or 0
            )
            for trade in eligible
        ) / len(eligible)

        avg_giveback = avg_mfe - avg_realized

        print(f"AVG MFE:        {avg_mfe:8.2f}R")
        print(f"AVG REALIZED:   {avg_realized:8.2f}R")
        print(f"AVG GIVEBACK:   {avg_giveback:8.2f}R")

    print()
    print("-" * 120)
    print("PROFIT GIVEBACK BY SETUP")
    print("-" * 120)

    setup_groups = {}

    for trade in losing_trades:

        setup = trade.get(
            "setup",
            trade.get("Setup", "UNKNOWN")
        )

        max_favorable_r = float(
            trade.get("max_favorable_r", 0)
            or 0
        )

        if max_favorable_r < 0.5:
            continue

        if setup not in setup_groups:
            setup_groups[setup] = []

        setup_groups[setup].append(trade)

    for setup, setup_trades in sorted(
        setup_groups.items()
    ):

        avg_mfe = sum(
            float(
                trade.get("max_favorable_r", 0)
                or 0
            )
            for trade in setup_trades
        ) / len(setup_trades)

        avg_realized = sum(
            float(
                trade.get("r_multiple", 0)
                or 0
            )
            for trade in setup_trades
        ) / len(setup_trades)

        avg_giveback = avg_mfe - avg_realized

        print(
            f"{str(setup):<25}"
            f"TRADES: {len(setup_trades):>4} "
            f"AVG_MFE: {avg_mfe:>7.2f}R "
            f"AVG_REALIZED: {avg_realized:>7.2f}R "
            f"AVG_GIVEBACK: {avg_giveback:>7.2f}R"
        )

    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # LOSING TRADE MFE / MAE TIMING
    # --------------------------------------------------------------

    print()
    print("=" * 120)
    print("LOSING TRADE MFE / MAE TIMING")
    print("=" * 120)

    timing_buckets = {
        "< 0.5R": [],
        "0.5R - 1.0R": [],
        "1.0R - 1.5R": [],
        ">= 1.5R": [],
    }

    for trade in losing_trades:

        max_favorable_r = float(
            trade.get("max_favorable_r", 0)
            or 0
        )

        if max_favorable_r < 0.5:
            timing_buckets["< 0.5R"].append(trade)

        elif max_favorable_r < 1.0:
            timing_buckets["0.5R - 1.0R"].append(trade)

        elif max_favorable_r < 1.5:
            timing_buckets["1.0R - 1.5R"].append(trade)

        else:
            timing_buckets[">= 1.5R"].append(trade)

    print()
    print(
        f"{'MFE BUCKET':<18}"
        f"{'TRADES':>8}"
        f"{'AVG_MFE':>12}"
        f"{'AVG_MAE':>12}"
        f"{'AVG_BARS':>12}"
        f"{'AVG_R':>12}"
    )

    print("-" * 75)

    for bucket, bucket_trades in timing_buckets.items():

        if not bucket_trades:

            print(
                f"{bucket:<18}"
                f"{0:>8}"
                f"{'NA':>12}"
                f"{'NA':>12}"
                f"{'NA':>12}"
                f"{'NA':>12}"
            )

            continue

        avg_mfe = sum(
            float(
                trade.get("max_favorable_r", 0)
                or 0
            )
            for trade in bucket_trades
        ) / len(bucket_trades)

        avg_mae = sum(
            float(
                trade.get("max_adverse_r", 0)
                or 0
            )
            for trade in bucket_trades
        ) / len(bucket_trades)

        avg_bars = sum(
            float(
                trade.get("bars_in_trade", 0)
                or 0
            )
            for trade in bucket_trades
        ) / len(bucket_trades)

        avg_realized_r = sum(
            float(
                trade.get("r_multiple", 0)
                or 0
            )
            for trade in bucket_trades
        ) / len(bucket_trades)

        print(
            f"{bucket:<18}"
            f"{len(bucket_trades):>8}"
            f"{avg_mfe:>12.2f}R"
            f"{avg_mae:>12.2f}R"
            f"{avg_bars:>12.1f}"
            f"{avg_realized_r:>12.2f}R"
        )

    # --------------------------------------------------------------
    # TIME TO FAVORABLE EXCURSION
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("TIME TO FAVORABLE EXCURSION")
    print("-" * 120)

    threshold_fields = [
        ("0.5R", "bars_to_0_5r", "reached_0_5r"),
        ("1.0R", "bars_to_1r", "reached_1r"),
        ("1.5R", "bars_to_1_5r", "reached_1_5r"),
    ]

    for threshold, bars_field, reached_field in threshold_fields:

        eligible = []

        for trade in losing_trades:

            reached = trade.get(reached_field, False)

            bars_to_threshold = trade.get(bars_field)

            if reached and bars_to_threshold is not None:

                try:
                    bars_to_threshold = float(
                        bars_to_threshold
                    )
                except (TypeError, ValueError):
                    continue

                eligible.append(bars_to_threshold)

        print()

        if not eligible:

            print(
                f"LOSERS REACHING {threshold}: 0"
            )
            continue

        average_bars = (
            sum(eligible)
            / len(eligible)
        )

        fastest = min(eligible)
        slowest = max(eligible)

        print(
            f"LOSERS REACHING {threshold}: "
            f"{len(eligible):>4}"
        )

        print(
            f"AVG BARS TO {threshold}: "
            f"{average_bars:>7.2f}"
        )

        print(
            f"FASTEST: {fastest:>7.0f} bars"
        )

        print(
            f"SLOWEST: {slowest:>7.0f} bars"
        )

    # --------------------------------------------------------------
    # PROFITABLE EXCURSION BEFORE LOSS
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("PROFITABLE EXCURSION BEFORE LOSS")
    print("-" * 120)

    giveback_thresholds = [
        (0.5, "reached_0_5r"),
        (1.0, "reached_1r"),
        (1.5, "reached_1_5r"),
    ]

    for threshold, reached_field in giveback_thresholds:

        eligible = [
            trade
            for trade in losing_trades
            if trade.get(reached_field, False)
        ]

        if not eligible:

            print()
            print(
                f"LOSERS REACHING {threshold:.1f}R: 0"
            )

            continue

        avg_mfe = sum(
            float(
                trade.get("max_favorable_r", 0)
                or 0
            )
            for trade in eligible
        ) / len(eligible)

        avg_realized = sum(
            float(
                trade.get("r_multiple", 0)
                or 0
            )
            for trade in eligible
        ) / len(eligible)

        avg_bars = sum(
            float(
                trade.get("bars_in_trade", 0)
                or 0
            )
            for trade in eligible
        ) / len(eligible)

        avg_giveback = (
            avg_mfe
            - avg_realized
        )

        print()
        print(
            f"LOSERS REACHING >= {threshold:.1f}R: "
            f"{len(eligible):>4}"
        )

        print(
            f"AVG MFE:        {avg_mfe:>8.2f}R"
        )

        print(
            f"AVG REALIZED:   {avg_realized:>8.2f}R"
        )

        print(
            f"AVG GIVEBACK:   {avg_giveback:>8.2f}R"
        )

        print(
            f"AVG BARS:       {avg_bars:>8.1f}"
        )

def print_infy_early_divergence_diagnostic(signals, trades):
    """
    INFY-only diagnostic for the early historical trade-sequence divergence.

    Compares:
    1. Signal candles generated by the current pipeline.
    2. Executed trades generated from those signals.

    This is diagnostic only. It does not modify signals, trades,
    the backtest engine, or the historical trade ledger.
    """

    if "INFY.NS" not in SYMBOLS:
        return

    print()
    print("=" * 120)
    print("INFY EARLY HISTORICAL DIVERGENCE DIAGNOSTIC")
    print("=" * 120)

    diagnostic_dates = pd.to_datetime([
        "2021-10-01",
        "2021-10-18",
        "2021-10-29",
        "2021-12-17",
        "2022-01-14",
        "2022-02-14",
        "2022-04-06",
        "2022-05-04",
    ])

    # --------------------------------------------------------------
    # CURRENT SIGNAL STREAM
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("CURRENT INFY SIGNALS AROUND DIVERGENCE WINDOW")
    print("-" * 120)

    if signals is None or signals.empty:
        print("NO SIGNAL DATA")
        return

    signal_index = pd.to_datetime(signals.index)

    mask = (
        (signal_index >= diagnostic_dates.min())
        & (signal_index <= diagnostic_dates.max())
    )

    selected_signals = signals.loc[mask]

    if selected_signals.empty:
        print("NO SIGNALS IN DIAGNOSTIC WINDOW")
    else:
        for signal_date, row in selected_signals.iterrows():
            signal_value = row.get("Signal", 0)

            try:
                signal_value = int(signal_value)
            except (TypeError, ValueError):
                signal_value = 0

            if signal_value == 1:
                direction = "LONG"
            elif signal_value == -1:
                direction = "SHORT"
            else:
                direction = "WAIT"

            next_date = None
            next_open = None

            try:
                position = signals.index.get_loc(signal_date)

                if position < len(signals.index) - 1:
                    next_date = signals.index[position + 1]
                    next_row = signals.iloc[position + 1]
                    next_open = next_row.get("Open")
            except (KeyError, TypeError, ValueError):
                pass

            print(
                f"SIGNAL={signal_date} | "
                f"VALUE={signal_value:+d} | "
                f"DIRECTION={direction:<5} | "
                f"NEXT_ENTRY_DATE={next_date} | "
                f"NEXT_OPEN={next_open}"
            )

    # --------------------------------------------------------------
    # CURRENT EXECUTED TRADES
    # --------------------------------------------------------------

    print()
    print("-" * 120)
    print("CURRENT INFY EXECUTED TRADES AROUND DIVERGENCE WINDOW")
    print("-" * 120)

    selected_trades = []

    for trade in trades:
        entry_date = trade.get("entry_date")

        if entry_date is None:
            continue

        try:
            entry_date = pd.Timestamp(entry_date)
        except (TypeError, ValueError):
            continue

        if (
            entry_date >= diagnostic_dates.min()
            and entry_date <= diagnostic_dates.max()
        ):
            selected_trades.append(trade)

    if not selected_trades:
        print("NO EXECUTED TRADES IN DIAGNOSTIC WINDOW")
    else:
        for trade in selected_trades:
            print(
                f"ENTRY={trade.get('entry_date')} | "
                f"DIRECTION={trade.get('direction')} | "
                f"SETUP={trade.get('Setup', trade.get('setup'))} | "
                f"SIGNAL_DATE={trade.get('Signal_Date')} | "
                f"ENTRY_PRICE={trade.get('entry_price')} | "
                f"R={trade.get('r_multiple')}"
            )

    print()
    print("-" * 120)
    print("TARGET HISTORICAL SIGNAL DATES")
    print("-" * 120)

    historical_signal_dates = [
        "2021-10-01",
        "2021-10-29",
        "2022-01-14",
        "2022-02-14",
        "2022-04-06",
    ]

    for date in historical_signal_dates:
        date_ts = pd.Timestamp(date)

        if date_ts in signal_index:
            row = signals.loc[date_ts]
            print(
                f"{date} -> CURRENT SIGNAL={row.get('Signal')}"
            )
        else:
            print(
                f"{date} -> CURRENT SIGNAL=NO ROW"
            )

def print_infy_signal_input_diagnostic(signals):
    """
    INFY historical signal divergence diagnostic.

    Purpose:
    Compare the exact current signal-engine inputs around the
    five historical signal dates where the protected trade file
    differs from the current engine.

    Research only.
    Does NOT modify signals or trading logic.
    """

    if signals is None or signals.empty:
        print()
        print("=" * 100)
        print("INFY SIGNAL INPUT DIAGNOSTIC")
        print("=" * 100)
        print("No signal dataframe available.")
        return

    if not isinstance(signals.index, pd.DatetimeIndex):
        signals = signals.copy()
        signals.index = pd.to_datetime(
            signals.index,
            errors="coerce"
        )

    historical_signal_dates = [
        "2021-10-01",
        "2021-10-29",
        "2022-01-14",
        "2022-02-14",
        "2022-04-06",
    ]

    # These are useful replacement/current signal dates observed
    # around the historical divergence windows.
    comparison_dates = [
        "2021-10-01",
        "2021-10-04",
        "2021-10-18",
        "2021-10-19",
        "2021-10-29",
        "2021-11-01",
        "2021-12-17",
        "2021-12-20",
        "2022-01-14",
        "2022-01-17",
        "2022-02-14",
        "2022-02-15",
        "2022-04-06",
        "2022-04-07",
    ]

    available_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "ATR",
        "Signal",
        "Setup",
        "Setup_Direction",
        "Setup_Confidence",
        "Setup_Reason",
        "Structure_Bias",
        "Structure",
        "BOS",
        "CHOCH",
        "Break_Direction",
        "Break_Confirmed",
        "Break_Age",
        "Liquidity_Sweep",
        "Trend",
        "Trend_Strength",
        "Fast_EMA",
        "Slow_EMA",
        "Trend_Slope",
        "Price_Position",
        "Price_Position_Value",
        "Fib_Retracement",
        "Fib_Zone",
        "Fib_Distance_382",
        "Fib_Distance_500",
        "Fib_Distance_618",
        "Fib_Premium_Discount",
        "Fib_Extension_State",
        "Fib_Extension_Percent",
    ]

    diagnostic_columns = [
        column
        for column in available_columns
        if column in signals.columns
    ]

    print()
    print("=" * 100)
    print("INFY SIGNAL INPUT DIAGNOSTIC")
    print("=" * 100)

    print()
    print("PURPOSE:")
    print(
        "Compare current engine structure/trend/setup inputs "
        "at historical INFY divergence dates."
    )

    print()
    print("AVAILABLE DIAGNOSTIC COLUMNS:")
    print(", ".join(diagnostic_columns))

    print()
    print("-" * 100)
    print("KEY HISTORICAL SIGNAL DATES")
    print("-" * 100)

    for date_text in historical_signal_dates:

        date = pd.Timestamp(date_text)

        print()
        print(f"DATE: {date.date()}")

        if date not in signals.index:
            print("STATUS: DATE NOT FOUND")
            continue

        row = signals.loc[date]

        for column in diagnostic_columns:
            value = row.get(column)

            if pd.isna(value):
                value = None

            print(
                f"{column:<28} = {value}"
            )

    print()
    print("-" * 100)
    print("NEARBY COMPARISON DATES")
    print("-" * 100)

    for date_text in comparison_dates:

        date = pd.Timestamp(date_text)

        if date not in signals.index:
            continue

        row = signals.loc[date]

        print()
        print(f"{date.date()}")

        compact_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "ATR",
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
            "Liquidity_Sweep",
            "Trend",
            "Trend_Strength",
            "Fast_EMA",
            "Slow_EMA",
            "Trend_Slope",
            "Price_Position",
            "Price_Position_Value",
        ]

        for column in compact_columns:

            if column not in signals.columns:
                continue

            value = row.get(column)

            if pd.isna(value):
                value = None

            print(
                f"  {column:<25} = {value}"
            )

    print()
    print("=" * 100)
    print("END INFY SIGNAL INPUT DIAGNOSTIC")
    print("=" * 100)

def print_infy_structure_state_diagnostic(signals):
    """
    INFY-only diagnostic for the first historical divergence.

    Research-only:
    - does not modify signals
    - does not modify trades
    - does not modify the backtest engine
    - does not modify the historical ledger

    Purpose:
    Inspect the actual market-structure state around
    INFY 2021-10-01 and compare the swing/structure inputs
    responsible for the current WAIT classification.
    """

    if signals is None or signals.empty:
        return

    if "INFY.NS" not in SYMBOLS:
        return

    target_dates = pd.to_datetime([
        "2021-10-01",
        "2021-10-04",
        "2021-10-18",
        "2021-10-19",
        "2021-10-29",
    ])

    print()
    print("=" * 120)
    print("INFY MARKET STRUCTURE STATE DIAGNOSTIC")
    print("=" * 120)

    print()
    print("TARGET DATES")
    print("-" * 120)

    for target_date in target_dates:

        if target_date not in signals.index:
            print(f"{target_date.date()} -> NO SIGNAL ROW")
            continue

        position = signals.index.get_loc(target_date)

        if isinstance(position, slice):
            position = position.start

        history = signals.iloc[:position + 1].copy()

        try:
            market_structure = calculate_market_structure(
                history,
                swing_window=3
            )
        except Exception as exc:
            print(
                f"{target_date.date()} -> "
                f"STRUCTURE ERROR: {exc}"
            )
            continue

        print()
        print(f"DATE: {target_date.date()}")
        print(f"History rows: {len(history)}")

        print(
            f"Current Signal: "
            f"{signals.loc[target_date].get('Signal')}"
        )

        print(
            f"Current Setup: "
            f"{signals.loc[target_date].get('Setup')}"
        )

        print(
            f"Bias: "
            f"{market_structure.get('bias')}"
        )

        print(
            f"Structure: "
            f"{market_structure.get('structure')}"
        )

        print(
            f"BOS: "
            f"{market_structure.get('bos')}"
        )

        print(
            f"CHOCH: "
            f"{market_structure.get('choch')}"
        )

        print(
            f"Break Direction: "
            f"{market_structure.get('break_direction')}"
        )

        print(
            f"Break Confirmed: "
            f"{market_structure.get('break_confirmed')}"
        )

        print(
            f"Break Age: "
            f"{market_structure.get('break_age')}"
        )

        print(
            f"Break Level: "
            f"{market_structure.get('break_level')}"
        )

        print(
            f"Break Date: "
            f"{market_structure.get('break_date')}"
        )

        print(
            f"Swing Highs: "
            f"{market_structure.get('swing_highs')}"
        )

        print(
            f"Swing Lows: "
            f"{market_structure.get('swing_lows')}"
        )

    print()
    print("=" * 120)
    print("END INFY MARKET STRUCTURE STATE DIAGNOSTIC")
    print("=" * 120)

def main():

    print("=" * 100)
    print("TRADESENSE STRUCTURAL FAILURE DIAGNOSTIC")
    print("=" * 100)

    all_trades = []

    for symbol in SYMBOLS:

        print()
        print("-" * 100)
        print(f"PROCESSING: {symbol}")
        print("-" * 100)

        service = BacktestService(
            symbol=symbol,
            strategy="tradesense",
            initial_capital=INITIAL_CAPITAL,
        )

        historical_data = service.load_data(
            start_date="2021-01-01"
        )

        signals = service.generate_tradesense_signals(
            df=historical_data
        )

        # ------------------------------------------------------------------
        # TEMPORAL COVERAGE DIAGNOSTIC
        # ------------------------------------------------------------------

        print()
        print("TEMPORAL COVERAGE DIAGNOSTIC")

        if signals is None or signals.empty:
            print("SIGNALS: EMPTY")
        else:
            data_max_date = signals.index.max()
            print(f"DATA / SIGNAL MAX DATE: {data_max_date}")

            post_cutoff = signals[
                signals.index >= pd.Timestamp("2026-07-29")
            ]

            post_long = post_cutoff[
                post_cutoff["Signal"] == 1
            ]

            post_short = post_cutoff[
                post_cutoff["Signal"] == -1
            ]

            print(
                f"POST-JUL-28 LONG SIGNALS: "
                f"{len(post_long)}"
            )

            print(
                f"POST-JUL-28 SHORT SIGNALS: "
                f"{len(post_short)}"
            )

            post_signal_dates = post_cutoff[
                post_cutoff["Signal"] != 0
            ].index

            print(
                "POST-JUL-28 SIGNAL DATES: "
                + (
                    ", ".join(
                        str(date)[:10]
                        for date in post_signal_dates
                    )
                    if len(post_signal_dates) > 0
                    else "NONE"
                )
            )

        backtest_result = service.run_directional_backtest(
            signals,
            signal_column="Signal",
            initial_stop_atr=3.5,
            target_r=2.0,
            risk_per_trade=0.01
        )

        trades = backtest_result.get("trades", [])

        # ------------------------------------------------------------------
        # POST-JUL-28 EXECUTED TRADE DIAGNOSTIC
        # ------------------------------------------------------------------

        post_jul28_trades = [
            trade
            for trade in trades
            if str(trade.get("entry_date", ""))[:10]
            >= "2026-07-29"
        ]

        print(
            "POST-JUL-28 EXECUTED TRADES: "
            f"{len(post_jul28_trades)}"
        )

        print(
            "POST-JUL-28 EXECUTED TRADE DATES: "
            + (
                ", ".join(
                    str(trade.get("entry_date", ""))[:10]
                    for trade in post_jul28_trades
                )
                if post_jul28_trades
                else "NONE"
            )
        )

        if not isinstance(trades, list):
            print(
                f"ERROR: Unexpected trades structure for {symbol}: "
                f"{type(trades)}"
            )
            continue

        print(f"EXECUTED TRADES: {len(trades)}")
        print(f"ROWS: {len(signals)}")

        if "Signal" in signals.columns:

            print(
                f"LONG SIGNALS: "
                f"{(signals['Signal'] == 1).sum()}"
            )

            print(
                f"SHORT SIGNALS: "
                f"{(signals['Signal'] == -1).sum()}"
            )

            print(
                f"WAIT: "
                f"{(signals['Signal'] == 0).sum()}"
            )

        print(f"TRADES: {len(trades)}")


        # ------------------------------------------------------------------
        # MAP EACH EXECUTED TRADE BACK TO ITS SIGNAL CANDLE
        # ------------------------------------------------------------------

        signal_metadata = {}

        for i in range(len(signals) - 1):

            signal_date = signals.index[i]
            entry_date = signals.index[i + 1]

            row = signals.iloc[i]

            signal_metadata[entry_date] = {
                "Setup": row.get("Setup"),
                "Setup_Confidence": row.get("Setup_Confidence"),
                "Structure_Bias": row.get("Structure_Bias"),
                "BOS": row.get("BOS"),
                "CHOCH": row.get("CHOCH"),
                "Break_Direction": row.get("Break_Direction"),
                "Break_Confirmed": row.get("Break_Confirmed"),
                "Break_Age": row.get("Break_Age"),
                "Liquidity_Sweep": row.get("Liquidity_Sweep"),
                "Price_Position_Value": row.get("Price_Position_Value"),

                # Fibonacci context
                "Fib_Retracement": row.get("Fib_Retracement"),
                "Fib_Zone": row.get("Fib_Zone"),
                "Fib_Distance_382": row.get("Fib_Distance_382"),
                "Fib_Distance_500": row.get("Fib_Distance_500"),
                "Fib_Distance_618": row.get("Fib_Distance_618"),
                "Fib_Premium_Discount": row.get("Fib_Premium_Discount"),
                "Fib_Extension_State": row.get("Fib_Extension_State"),
                "Fib_Extension_Percent": row.get("Fib_Extension_Percent"),

                "Signal_Date": signal_date,
            }


        # ------------------------------------------------------------------
        # ATTACH SIGNAL METADATA TO EXECUTED TRADES
        # ------------------------------------------------------------------

        for trade in trades:

            trade["_symbol"] = symbol

            entry_date = trade.get("entry_date")

            metadata = signal_metadata.get(entry_date)

            if metadata is not None:

                trade.update(metadata)

            else:

                print(
                    f"WARNING: No signal metadata found for "
                    f"{symbol} entry date {entry_date}"
                )

            add_diagnostic_fields(trade)

        if symbol == "INFY.NS":
            print_infy_early_divergence_diagnostic(
                signals,
                trades
            )

            print_infy_signal_input_diagnostic(
                signals
            )

            print_infy_structure_state_diagnostic(
                signals
            )

        all_trades.extend(trades)

    print()
    print("=" * 100)
    print("OVERALL CURRENT RESULT")
    print("=" * 100)

    print(
        f"ALL TRADESENSE TRADES              "
        f"{performance_label(all_trades)}"
    )

    # ------------------------------------------------------------------
    # STOCK
    # ------------------------------------------------------------------

    print_group(
        "STOCK × PERFORMANCE",
        all_trades,
        "_symbol",
    )

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    print_group(
        "SETUP × PERFORMANCE",
        all_trades,
        "Diagnostic_Setup",
    )

    # ------------------------------------------------------------------
    # SETUP × DIRECTION
    # ------------------------------------------------------------------

    print_group(
        "SETUP × DIRECTION",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Direction",
    )

    # ------------------------------------------------------------------
    # SETUP × TREND
    # ------------------------------------------------------------------

    print_group(
        "SETUP × TREND",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Trend",
    )

    # ------------------------------------------------------------------
    # SETUP × LIQUIDITY
    # ------------------------------------------------------------------

    print_group(
        "SETUP × LIQUIDITY SWEEP",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Liquidity",
    )

    # ------------------------------------------------------------------
    # SETUP × BOS
    # ------------------------------------------------------------------

    print_group(
        "SETUP × BOS",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_BOS",
    )

    # ------------------------------------------------------------------
    # SETUP × CHOCH
    # ------------------------------------------------------------------

    print_group(
        "SETUP × CHOCH",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_CHOCH",
    )

    # ------------------------------------------------------------------
    # SETUP × BREAK AGE
    # ------------------------------------------------------------------

    print_group(
        "SETUP × BREAK AGE",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Break_Age",
    )

    # ------------------------------------------------------------------
    # SETUP × STRUCTURE BIAS
    # ------------------------------------------------------------------

    print_group(
        "SETUP × STRUCTURE BIAS",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Structure_Bias",
    )

    # ------------------------------------------------------------------
    # SETUP × CONFIDENCE
    # ------------------------------------------------------------------

    print_group(
        "SETUP × CONFIDENCE BUCKET",
        all_trades,
        "Diagnostic_Setup",
        "Diagnostic_Confidence_Bucket",
    )

    # ------------------------------------------------------------------
    # SETUP × TREND STRENGTH
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("SETUP × TREND STRENGTH")
    print("=" * 100)

    strength_buckets = {
        "< 20": [],
        "20 - 40": [],
        "40 - 60": [],
        "60 - 80": [],
        ">= 80": [],
    }

    for trade in all_trades:

        strength = trade.get(
            "entry_trend_strength",
            0
        )

        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 0.0

        if strength < 20:
            bucket = "< 20"
        elif strength < 40:
            bucket = "20 - 40"
        elif strength < 60:
            bucket = "40 - 60"
        elif strength < 80:
            bucket = "60 - 80"
        else:
            bucket = ">= 80"

        strength_buckets[bucket].append(trade)


    for bucket, bucket_trades in strength_buckets.items():

        print()
        print(
            f"{bucket:<12}"
            f"TRADES: {len(bucket_trades):>4}"
        )

        setup_groups = {}

        for trade in bucket_trades:

            setup = trade.get(
                "Diagnostic_Setup",
                trade.get("setup", "UNKNOWN")
            )

            if setup not in setup_groups:
                setup_groups[setup] = []

            setup_groups[setup].append(trade)

        for setup, setup_trades in sorted(
            setup_groups.items()
        ):

            profits = [
                float(
                    trade.get("profit", 0)
                    or 0
                )
                for trade in setup_trades
            ]

            r_values = [
                float(
                    trade.get("r_multiple", 0)
                    or 0
                )
                for trade in setup_trades
            ]

            wins = [
                value
                for value in r_values
                if value > 0
            ]

            losses = [
                value
                for value in r_values
                if value <= 0
            ]

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))

            if gross_loss > 0:
                profit_factor = (
                    gross_profit / gross_loss
                )
            else:
                profit_factor = 0.0

            win_rate = (
                len(wins) / len(r_values) * 100
                if r_values
                else 0.0
            )

            total_r = sum(r_values)

            print(
                f"  {str(setup):<25}"
                f"Trades={len(r_values):>3} "
                f"Win={win_rate:6.2f}% "
                f"PF={profit_factor:5.2f} "
                f"R={total_r:7.2f}"
            )

    # ------------------------------------------------------------------
    # SETUP × FIBONACCI ZONE
    # ------------------------------------------------------------------

    print_group(
        "SETUP × FIB ZONE",
        all_trades,
        "Diagnostic_Setup",
        "Fib_Zone",
    )

    # ------------------------------------------------------------------
    # SETUP × TREND STRENGTH × FIBONACCI ZONE
    # ------------------------------------------------------------------

    print()
    print("=" * 120)
    print("SETUP × TREND STRENGTH × FIB ZONE")
    print("=" * 120)

    three_way_groups = {}

    for trade in all_trades:

        setup = trade.get(
            "Diagnostic_Setup",
            trade.get("setup", "UNKNOWN")
        )

        strength = trade.get(
            "entry_trend_strength",
            0
        )

        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 0.0

        if strength < 20:
            strength_bucket = "<20"
        elif strength < 40:
            strength_bucket = "20-40"
        elif strength < 60:
            strength_bucket = "40-60"
        elif strength < 80:
            strength_bucket = "60-80"
        else:
            strength_bucket = ">=80"

        fib_zone = trade.get(
            "Fib_Zone",
            "UNKNOWN"
        )

        key = (
            setup,
            strength_bucket,
            fib_zone
        )

        if key not in three_way_groups:
            three_way_groups[key] = []

        three_way_groups[key].append(trade)


    for (
        setup,
        strength_bucket,
        fib_zone
    ), group_trades in sorted(
        three_way_groups.items()
    ):

        r_values = [
            float(
                trade.get("r_multiple", 0)
                or 0
            )
            for trade in group_trades
        ]

        if not r_values:
            continue

        wins = [
            r
            for r in r_values
            if r > 0
        ]

        losses = [
            r
            for r in r_values
            if r <= 0
        ]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        if gross_loss > 0:
            profit_factor = (
                gross_profit / gross_loss
            )
        else:
            profit_factor = 0.0

        win_rate = (
            len(wins)
            / len(r_values)
            * 100
        )

        total_r = sum(r_values)

        print(
            f"{setup:<25}"
            f"{strength_bucket:<8}"
            f"{str(fib_zone):<15}"
            f"Trades={len(r_values):>3} "
            f"Win={win_rate:6.2f}% "
            f"PF={profit_factor:5.2f} "
            f"R={total_r:7.2f}"
        )

    # ------------------------------------------------------------------
    # FIBONACCI ZONE
    # ------------------------------------------------------------------

    print_group(
        "FIBONACCI ZONE × PERFORMANCE",
        all_trades,
        "Fib_Zone",
    )

    # ------------------------------------------------------------------
    # FIBONACCI PREMIUM / DISCOUNT
    # ------------------------------------------------------------------

    print_group(
        "FIBONACCI PREMIUM / DISCOUNT × PERFORMANCE",
        all_trades,
        "Fib_Premium_Discount",
    )

    # ------------------------------------------------------------------
    # FIBONACCI EXTENSION
    # ------------------------------------------------------------------

    print_group(
        "FIBONACCI EXTENSION × PERFORMANCE",
        all_trades,
        "Fib_Extension_State",
    )

    # ------------------------------------------------------------------
    # LIQUIDITY ONLY
    # ------------------------------------------------------------------

    print_group(
        "LIQUIDITY × TREND",
        all_trades,
        "Diagnostic_Liquidity",
        "Diagnostic_Trend",
    )

    # ------------------------------------------------------------------
    # STRUCTURE ONLY
    # ------------------------------------------------------------------

    print_group(
        "STRUCTURE BIAS × TREND",
        all_trades,
        "Diagnostic_Structure_Bias",
        "Diagnostic_Trend",
    )

    # ------------------------------------------------------------------
    # DIRECTION × LIQUIDITY
    # ------------------------------------------------------------------

    print_group(
        "DIRECTION × LIQUIDITY",
        all_trades,
        "Diagnostic_Direction",
        "Diagnostic_Liquidity",
    )

    # ------------------------------------------------------------------
    # DIRECTION × TREND
    # ------------------------------------------------------------------

    print_group(
        "DIRECTION × TREND",
        all_trades,
        "Diagnostic_Direction",
        "Diagnostic_Trend",
    )

    # ------------------------------------------------------------------
    # EXIT REASON
    # ------------------------------------------------------------------

    print_group(
        "EXIT REASON",
        all_trades,
        "exit_reason",
    )

    # ------------------------------------------------------------------
    # SETUP × EXIT
    # ------------------------------------------------------------------

    print_group(
        "SETUP × EXIT REASON",
        all_trades,
        "Diagnostic_Setup",
        "exit_reason",
    )

    # ------------------------------------------------------------------
    # SAVE FUTURE HOLDOUT TRADES
    # ------------------------------------------------------------------

    import csv
    import os

    output_dir = "tests/output"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # The historical research dataset is FROZEN.
    # This generator must NEVER overwrite it.
    # --------------------------------------------------------------

    historical_file = HISTORICAL_TRADES_FILE
    holdout_file = HOLDOUT_TRADES_FILE

    # --------------------------------------------------------------
    # SAVE FRESH FULL-HISTORY RESEARCH CANDIDATE
    # --------------------------------------------------------------
    # IMPORTANT:
    # This is NOT the frozen historical dataset.
    # It exists only for reconciliation against the frozen dataset.
    # --------------------------------------------------------------

    research_candidate_file = (
        "tests/output/tradesense_trades_research_candidate.csv"
    )

    if all_trades:

        research_fieldnames = sorted(
            {
                key
                for trade in all_trades
                for key in trade.keys()
            }
        )

        with open(
            research_candidate_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=research_fieldnames,
                extrasaction="ignore"
            )

            writer.writeheader()
            writer.writerows(all_trades)

        print()
        print("=" * 100)
        print("RESEARCH CANDIDATE DATASET")
        print("=" * 100)
        print(
            f"Fresh candidate trades: {len(all_trades)}"
        )
        print(
            f"File: {research_candidate_file}"
        )

    # --------------------------------------------------------------
    # SELECT ONLY GENUINELY FUTURE + COMPLETED TRADES
    # --------------------------------------------------------------
    #
    # IMPORTANT:
    # FINAL_LIQUIDATION means the position was still open when the
    # available data ended. It must NOT be frozen into the holdout.
    #
    # A later generator run may obtain additional bars and therefore
    # produce the real exit for that trade.
    # --------------------------------------------------------------

    holdout_trades = []

    for trade in all_trades:

        entry_date = trade.get("entry_date")

        if entry_date is None:
            continue

        entry_date = str(entry_date)[:10]

        if entry_date < HOLDOUT_START_DATE:
            continue

        exit_reason = str(
            trade.get("exit_reason", "")
        ).strip().upper()

        if exit_reason == "FINAL_LIQUIDATION":
            continue

        holdout_trades.append(trade)

    # --------------------------------------------------------------
    # LOAD EXISTING HOLDOUT DATA
    # --------------------------------------------------------------

    existing_holdout = []

    if os.path.exists(holdout_file):

        with open(
            holdout_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            existing_holdout = list(reader)

    # --------------------------------------------------------------
    # DEDUPLICATE USING:
    # SYMBOL + ENTRY DATE + DIRECTION
    #
    # Existing observations are preserved.
    # This makes repeated generator runs idempotent.
    # --------------------------------------------------------------

    existing_keys = set()

    for trade in existing_holdout:

        key = (
            str(trade.get("_symbol", "")),
            str(trade.get("entry_date", ""))[:10],
            str(trade.get("direction", ""))
        )

        existing_keys.add(key)

    new_holdout_trades = []

    for trade in holdout_trades:

        key = (
            str(trade.get("_symbol", "")),
            str(trade.get("entry_date", ""))[:10],
            str(trade.get("direction", ""))
        )

        if key in existing_keys:
            continue

        new_holdout_trades.append(trade)
        existing_keys.add(key)

    # --------------------------------------------------------------
    # MERGE EXISTING + NEW HOLDOUT DATA
    # --------------------------------------------------------------

    combined_holdout_trades = (
        existing_holdout
        + new_holdout_trades
    )

    # --------------------------------------------------------------
    # SAVE HOLDOUT DATASET
    # --------------------------------------------------------------

    if combined_holdout_trades:

        fieldnames = sorted(
            {
                key
                for trade in combined_holdout_trades
                for key in trade.keys()
            }
        )

        with open(
            holdout_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore"
            )

            writer.writeheader()

            writer.writerows(
                combined_holdout_trades
            )

    # --------------------------------------------------------------
    # VERIFY HISTORICAL DATASET WAS NOT TOUCHED
    # --------------------------------------------------------------

    if os.path.exists(historical_file):

        with open(
            historical_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            historical_reader = csv.DictReader(file)
            historical_rows = list(historical_reader)

        print()
        print("=" * 100)
        print("HISTORICAL DATASET PROTECTION")
        print("=" * 100)

        print(
            f"Historical file: {historical_file}"
        )

        print(
            f"Historical trades: {len(historical_rows)}"
        )

        print(
            "Historical dataset was NOT overwritten."
        )

    # --------------------------------------------------------------
    # HOLDOUT SAVE SUMMARY
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("FUTURE HOLDOUT DATASET")
    print("=" * 100)

    print(
        f"Holdout start date: {HOLDOUT_START_DATE}"
    )

    print(
        f"New holdout trades: {len(new_holdout_trades)}"
    )

    print(
        f"Existing holdout trades: {len(existing_holdout)}"
    )

    print(
        f"Total holdout trades: "
        f"{len(combined_holdout_trades)}"
    )

    print(
        f"File: {holdout_file}"
    )


    # ------------------------------------------------------------------
    # TRADE DATA INTEGRITY
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("TRADE DATA INTEGRITY")
    print("=" * 100)

    integrity_fields = [
        "entry_trend",
        "entry_trend_strength",
        "entry_trend_slope",
        "entry_price_position",
        "Price_Position_Value",
        "Diagnostic_Setup",
        "Diagnostic_Direction",
        "Diagnostic_Liquidity",
        "Diagnostic_Break_Age",
        "Diagnostic_Structure_Bias",
    ]

    for field in integrity_fields:

        missing = sum(
            1
            for trade in all_trades
            if trade.get(field) is None
        )

        print(
            f"{field:<35} "
            f"{missing}/{len(all_trades)} missing"
        )

    # ------------------------------------------------------------------
    # TRADE-LEVEL ENTRY QUALITY
    # ------------------------------------------------------------------

    print_trade_level_diagnostic(all_trades)

    # ------------------------------------------------------------------
    # OUT-OF-SAMPLE TEST:
    # SHORT_REVERSAL EXCLUSION
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("OUT-OF-SAMPLE TEST — SHORT_REVERSAL EXCLUSION")
    print("=" * 100)

    OOS_START = "2025-01-01"

    oos_trades = []

    for trade in all_trades:

        entry_date = trade.get(
            "entry_date"
        )

        if entry_date is None:
            continue

        entry_date = str(entry_date)[:10]

        if entry_date >= OOS_START:
            oos_trades.append(trade)

    baseline_trades = oos_trades

    filtered_trades = [
        trade
        for trade in oos_trades
        if trade.get(
            "Diagnostic_Setup",
            trade.get("setup", "")
        ) != "SHORT_REVERSAL"
    ]

    print()
    print("OOS PERIOD:")
    print(
    f"{OOS_START} -> latest available date"
)

    print()
    print("BASELINE — ALL SETUPS")
    print(
        performance_label(
            baseline_trades
        )
    )

    print()
    print("FILTERED — EXCLUDE SHORT_REVERSAL")
    print(
        performance_label(
            filtered_trades
        )
    )

    print()
    print(
        f"SHORT_REVERSAL TRADES REMOVED: "
        f"{len(baseline_trades) - len(filtered_trades)}"
    )

    print()
    print("=" * 100)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 100)

    print(f"Stocks processed: {len(SYMBOLS)}")
    print(f"Total executed trades: {len(all_trades)}")

    total_profit = sum(
        float(t.get("profit", 0))
        for t in all_trades
    )

    total_r = sum(
        float(t.get("r_multiple", 0))
        for t in all_trades
    )

    print(f"Total P&L: {total_profit:.2f}")
    print(f"Total R: {total_r:.2f}")

    if all_trades:
        wins = sum(
            1
            for t in all_trades
            if float(t.get("profit", 0)) > 0
        )

        print(
            f"Overall win rate: "
            f"{wins / len(all_trades) * 100:.2f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()