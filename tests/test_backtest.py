from app.services.backtest_service import BacktestService

service = BacktestService("RELIANCE.NS")

metrics = service.performance_metrics()

print()

print("===== PERFORMANCE METRICS =====")

for key, value in metrics.items():

    if key != "trades":

        print(f"{key}: {value}")

print()

print("===== FIRST 5 TRADES =====")

for trade in metrics["trades"][:5]:

    print(trade)