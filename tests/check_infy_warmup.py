from app.services.backtest_service import BacktestService


REFERENCE_START = "2021-08-01"
TEST_START = "2021-09-08"

DATES = [
    "2021-09-23",
    "2021-10-01",
]


def generate_signals(start):
    service = BacktestService(
        symbol="INFY.NS",
        strategy="tradesense",
        initial_capital=100000,
    )

    df = service.load_data(
        start_date=start,
        end_date="2026-07-29"
    )

    return service.generate_tradesense_signals(df)


reference = generate_signals(REFERENCE_START)
test = generate_signals(TEST_START)

reference.index = reference.index.astype(str).str[:10]
test.index = test.index.astype(str).str[:10]


FIELDS = [
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
    "Fib_Retracement",
    "Fib_Zone",
    "Fib_Distance_382",
    "Fib_Distance_500",
    "Fib_Distance_618",
    "Fib_Premium_Discount",
    "Fib_Extension_State",
    "Fib_Extension_Percent",
    "Trend",
    "Trend_Strength",
    "Fast_EMA",
    "Slow_EMA",
    "Trend_Slope",
    "Price_Position",
    "Price_Position_Value",
]


print()
print("INFY.NS BOUNDARY-SENSITIVE SIGNAL FIELD COMPARISON")
print("=" * 110)


for date in DATES:

    print()
    print(f"DATE: {date}")
    print(f"REFERENCE START: {REFERENCE_START}")
    print(f"TEST START:      {TEST_START}")
    print("-" * 110)

    print(f"{'FIELD':<28} {'REFERENCE':<35} {'TEST':<35}")
    print("-" * 110)

    for field in FIELDS:

        ref_value = reference.loc[date, field]
        test_value = test.loc[date, field]

        if ref_value != test_value:

            print(
                f"{field:<28} "
                f"{str(ref_value):<35} "
                f"{str(test_value):<35}"
            )


print()
print("=" * 110)
print("Comparison complete.")