# init_data.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from vectorbtpro import vbt
from strategy_config import IS_TEST, BINANCE_KEYS, SYMBOL, CSV_FILE, LOOKBACK_DAYS

def get_api_client():
    keys = BINANCE_KEYS["test" if IS_TEST else "live"]
    client = Client(keys["api_key"], keys["api_secret"])
    if IS_TEST:
        client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return client

def get_account_balance(client, asset="USDC"):
    try:
        account_info = client.futures_account_balance()
        for balance in account_info:
            if balance["asset"] == asset:
                initial_balance = float(balance["balance"])
                break
        else:
            initial_balance = None

        if initial_balance is not None:
            with open("initial_balance.json", "w") as f:
                json.dump({asset: initial_balance}, f)
            print(f"[BALANCE] Saved initial balance: {initial_balance} {asset}")
        else:
            print(f"[WARN] {asset} balance not found.")
    except Exception as e:
        print(f"[ERROR] Could not fetch initial balance: {e}")

def pull_historical_data():
    if os.path.exists(CSV_FILE):
        print("[INFO] CSV already exists, skipping API pull.")
    else:
        print("[INFO] Pre-fetching historical data...")
        data = vbt.BinanceData.pull(
            SYMBOL,
            start=datetime.now() - timedelta(days=LOOKBACK_DAYS),
            timeframe='15m',
            klines_type=2,
        )
        data.to_csv(CSV_FILE)
        print("[INFO] Data pulled and saved to CSV.")

if __name__ == "__main__":
    keys = BINANCE_KEYS["test" if IS_TEST else "live"]
    vbt.BinanceData.set_custom_settings(
        client_config=dict(api_key=keys["api_key"], api_secret=keys["api_secret"])
    )
    client = get_api_client()
    pull_historical_data()
    get_account_balance(client)
