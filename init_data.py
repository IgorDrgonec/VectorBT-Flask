# init_data.py
# Env vars: STRATEGY_NAME, SYMBOL, DATA_DIR, BALANCE_DIR, CSV_FILENAME.
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from vectorbtpro import vbt
from strategy_config import IS_TEST, get_binance_keys, SYMBOL as DEFAULT_SYMBOL, LOOKBACK_DAYS, TIMEFRAME

STRATEGY_NAME = os.getenv("STRATEGY_NAME", "ema_macd")
SYMBOL = os.getenv("SYMBOL", DEFAULT_SYMBOL)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
BALANCE_DIR = os.getenv("BALANCE_DIR", DATA_DIR)
CSV_FILENAME = os.getenv("CSV_FILENAME", f"{STRATEGY_NAME}_{SYMBOL}.csv")
CSV_PATH = os.path.join(DATA_DIR, CSV_FILENAME)
BALANCE_PATH = os.path.join(BALANCE_DIR, "initial_balance.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BALANCE_DIR, exist_ok=True)


def get_api_client():
    api_key, api_secret = get_binance_keys(IS_TEST)
    client = Client(api_key, api_secret)
    if IS_TEST:
        client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    return client

def get_account_balance(client, asset="BNFCR"):
    try:
        account_info = client.futures_account_balance()
        for balance in account_info:
            if balance["asset"] == asset:
                initial_balance = float(balance["availableBalance"])
                break
        else:
            initial_balance = None

        if initial_balance is not None:
            with open(BALANCE_PATH, "w") as f:
                json.dump({asset: initial_balance}, f)
            print(f"[BALANCE] Saved initial balance: {initial_balance} {asset}")
        else:
            print(f"[WARN] {asset} balance not found.")
    except Exception as e:
        print(f"[ERROR] Could not fetch initial balance: {e}")

def pull_historical_data():
    if os.path.exists(CSV_PATH):
        print("[INFO] CSV already exists, skipping API pull.")
    else:
        print("[INFO] Pre-fetching historical data...")
        data = vbt.BinanceData.pull(
            SYMBOL,
            start=datetime.now() - timedelta(days=LOOKBACK_DAYS),
            timeframe=TIMEFRAME,
            klines_type=2,
        )
        data.to_csv(CSV_PATH)
        print("[INFO] Data pulled and saved to CSV.")

if __name__ == "__main__":
    api_key, api_secret = get_binance_keys(IS_TEST)
    vbt.BinanceData.set_custom_settings(
        client_config=dict(api_key=api_key, api_secret=api_secret)
    )
    client = get_api_client()
    pull_historical_data()
    get_account_balance(client)
