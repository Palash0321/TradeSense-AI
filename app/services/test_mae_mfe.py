from app.services.backtest_service import BacktestService


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


all_trades = []


for symbol in symbols:

    service = BacktestService(
        symbol=symbol,
        strategy="ema_atr",
        brokerage=20,
        slippage=0.10,
        initial_capital=100000,
    )

    metrics = service.performance_metrics(
        start_date="2025-01-01",
        end_date="2025-12-31",
        adx_min=15,
        ema_gap_min=0.0,
        trailing_atr=2.5,
        trailing_activation_atr=2.5,
        momentum_min=None,
        use_market_regime=False,
        initial_stop_atr=3.0,
    )

    for trade in metrics["trades"]:

        trade_copy = dict(trade)

        trade_copy["symbol"] = symbol

        all_trades.append(trade_copy)


print()
print("=" * 90)
print("MAE / MFE DIAGNOSTIC — 2025 HOLDOUT")
print("=" * 90)

print()
print(f"Total trades : {len(all_trades)}")


# ==========================================================
# ALL TRADES
# ==========================================================

def average(values):

    if not values:
        return 0

    return sum(values) / len(values)


all_mae = [
    float(t["mae_percent"])
    for t in all_trades
    if t.get("mae_percent") is not None
]

all_mfe = [
    float(t["mfe_percent"])
    for t in all_trades
    if t.get("mfe_percent") is not None
]


print()
print("=" * 90)
print("ALL TRADES")
print("=" * 90)

print(f"Average MAE : {average(all_mae):.2f}%")
print(f"Average MFE : {average(all_mfe):.2f}%")


# ==========================================================
# STOP LOSS TRADES
# ==========================================================

stop_trades = [
    t
    for t in all_trades
    if t.get("exit_reason") == "STOP_LOSS"
]


stop_mae = [
    float(t["mae_percent"])
    for t in stop_trades
    if t.get("mae_percent") is not None
]

stop_mfe = [
    float(t["mfe_percent"])
    for t in stop_trades
    if t.get("mfe_percent") is not None
]


print()
print("=" * 90)
print("STOP LOSS TRADES")
print("=" * 90)

print(f"Trades      : {len(stop_trades)}")
print(f"Average MAE : {average(stop_mae):.2f}%")
print(f"Average MFE : {average(stop_mfe):.2f}%")


# ==========================================================
# STOP LOSS ENTRY CONDITIONS
# ==========================================================

stop_adx = [
    float(t["adx_at_entry"])
    for t in stop_trades
    if t.get("adx_at_entry") is not None
]

stop_ema = [
    float(t["ema_gap_at_entry"])
    for t in stop_trades
    if t.get("ema_gap_at_entry") is not None
]


print()
print("STOP LOSS ENTRY CONDITIONS")
print("-" * 90)

print(f"Average ADX     : {average(stop_adx):.2f}")
print(f"Average EMA Gap : {average(stop_ema):.4f}")


# ==========================================================
# WINNING TRAILING STOP TRADES
# ==========================================================

winning_trailing = [
    t
    for t in all_trades
    if (
        t.get("exit_reason") == "TRAILING_STOP"
        and float(t.get("profit", 0)) > 0
    )
]


trail_mae = [
    float(t["mae_percent"])
    for t in winning_trailing
    if t.get("mae_percent") is not None
]

trail_mfe = [
    float(t["mfe_percent"])
    for t in winning_trailing
    if t.get("mfe_percent") is not None
]


trail_adx = [
    float(t["adx_at_entry"])
    for t in winning_trailing
    if t.get("adx_at_entry") is not None
]

trail_ema = [
    float(t["ema_gap_at_entry"])
    for t in winning_trailing
    if t.get("ema_gap_at_entry") is not None
]


print()
print("=" * 90)
print("WINNING TRAILING-STOP TRADES")
print("=" * 90)

print(f"Trades          : {len(winning_trailing)}")
print(f"Average MAE     : {average(trail_mae):.2f}%")
print(f"Average MFE     : {average(trail_mfe):.2f}%")
print(f"Average ADX     : {average(trail_adx):.2f}")
print(f"Average EMA Gap : {average(trail_ema):.4f}")


# ==========================================================
# STOP LOSS BY STOCK
# ==========================================================

print()
print("=" * 90)
print("STOP LOSS COUNT BY STOCK")
print("=" * 90)


stop_by_stock = {}


for trade in stop_trades:

    symbol = trade["symbol"]

    stop_by_stock[symbol] = (
        stop_by_stock.get(symbol, 0) + 1
    )


for symbol, count in sorted(
    stop_by_stock.items(),
    key=lambda x: x[1],
    reverse=True,
):

    print(
        f"{symbol:16} : {count:>2} stop losses"
    )


# ==========================================================
# STOP LOSS DETAILS
# ==========================================================

print()
print("=" * 90)
print("STOP LOSS TRADE DETAILS")
print("=" * 90)


for trade in sorted(
    stop_trades,
    key=lambda t: float(t.get("mfe_percent", 0)),
    reverse=True,
):

    print(
        f"{trade['symbol']:16} | "
        f"ADX={float(trade.get('adx_at_entry', 0)):>6.2f} | "
        f"EMA={float(trade.get('ema_gap_at_entry', 0)):>7.4f} | "
        f"MAE={float(trade.get('mae_percent', 0)):>7.2f}% | "
        f"MFE={float(trade.get('mfe_percent', 0)):>7.2f}% | "
        f"R={float(trade.get('r_multiple', 0)):>6.2f}"
    )


print()
print("=" * 90)
print("MAE / MFE DIAGNOSTIC COMPLETE")
print("=" * 90)