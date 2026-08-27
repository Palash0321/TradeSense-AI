from app.services.walk_forward_engine import WalkForwardEngine


symbols = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
    "AXISBANK.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "M&M.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "SUNPHARMA.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "TITAN.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
]


windows = [
    {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    },
    {
        "train_start": "2021-01-01",
        "train_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-12-31",
    },
]


print("\n=== SELECTED CONFIG VALIDATION ===")
print("ADX = 15")
print("EMA Gap = 0.0")
print("Trailing ATR = 2.5")
print("Activation ATR = 2.5")
print("Initial Stop ATR = 3.0")
print()


for symbol in symbols:

    engine = WalkForwardEngine(
        symbol,
        brokerage=20,
        slippage=0.10
    )

    result = engine.run(
        windows=windows,
        adx_values=[15],
        ema_gap_values=[0.0],
        trailing_atr_values=[2.5],
        trailing_activation_atr=2.5,
        momentum_min=None,
        use_market_regime=False,
        initial_stop_atr=3.0
    )

    print(f"\n{symbol}")

    for window in result["windows"]:

        testing = window["testing"]

        if testing:

            print(
                f"  {window['test_start']} -> "
                f"Profit: {testing['net_profit']:.2f} | "
                f"Trades: {testing['total_trades']} | "
                f"Return: {testing['total_return']:.2f}% | "
                f"PF: {testing['profit_factor']:.2f} | "
                f"DD: {testing['max_drawdown']:.2f}% | "
                f"Sharpe: {testing['sharpe_ratio']:.2f}"
            )