import os
import pandas as pd
import time
import json
from vectorbtpro import vbt
from datetime import datetime, timedelta
from binance.client import Client

symbol = "BTCUSDC"
csv_file = "ema_macd_data.csv"

is_test=False
# Binance API Keys
BINANCE_API_KEY = "SyWHwZv9BTOiFN3NxJvbTlNjXdRvW9HEQdGJrZp0PFTK4aMekC2tt8d9qRNwUEej"
BINANCE_SECRET_KEY = "XkryIgFQgZhIg4l77sFfcU6LQjYlklCRqf1Eedo6XJvNJT3rjESgad0gswX8BpZY"
#Testnet Futures Binance
API_KEY_TEST = "c0bf32af094d1b6f97e53d79e2d585003754d12fbe53a65f383d71e769d5b943"
API_SECRET_TEST = "cf01902200ac97101266ec6247c80a6bcb2d005286e34b6684ea30cf6d88e20a"

api_key = API_KEY_TEST if is_test else  BINANCE_API_KEY
api_secret = API_SECRET_TEST if is_test else BINANCE_SECRET_KEY

client = Client(api_key, api_secret)
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi" if is_test else "https://fapi.binance.com/fapi"

# Fetch balance only once before app starts
def get_account_balance():
    try:
        account_info = client.futures_account_balance()
        for balance in account_info:
            if balance["asset"] == "USDC":
                initial_balance = float(balance["balance"])
                break
        else:
            initial_balance = None

        if initial_balance is not None:
            with open("initial_balance.json", "w") as f:
                json.dump({"USDC": initial_balance}, f)
            print(f"[BALANCE] Saved initial balance: {initial_balance} USDC")
        else:
            print("[WARN] USDC balance not found.")
    except Exception as e:
        print(f"[ERROR] Could not fetch initial balance: {e}")

# Pull historical data once
def pull_historical_data():
    if os.path.exists(csv_file):
        print("[INFO] CSV already exists, skipping API pull.")
    else:
        print("[INFO] Pre-fetching historical data...")
        data = vbt.BinanceData.pull(
            symbol,
            start=datetime.now() - timedelta(days=60),
            timeframe='15m',
            klines_type=2,
        )
        data.to_csv(csv_file)
        print("[INFO] Data pulled and saved to CSV.")

if __name__ == "__main__":
    vbt.BinanceData.set_custom_settings(
        client_config=dict(
            api_key=BINANCE_API_KEY,
            api_secret=BINANCE_SECRET_KEY
        )
    )
    pull_historical_data()
    get_account_balance()