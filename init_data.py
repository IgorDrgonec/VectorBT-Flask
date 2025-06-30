from vectorbtpro import vbt
from datetime import datetime, timedelta

symbol = "BTCUSDC"
csv_file = "ema_macd_data.csv"

print("[INFO] Pre-fetching historical data...")

data = vbt.BinanceData.pull(
    symbol,
    start=datetime.now() - timedelta(days=60),
    timeframe="15m",
    klines_type=2
)

data.to_csv(csv_file)
print("[INFO] Data pulled and saved to CSV.")