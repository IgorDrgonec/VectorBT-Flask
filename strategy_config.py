# strategy_config.py
# Env vars (used by app.py/init_data.py): STRATEGY_NAME, SYMBOL, DATA_DIR, LOG_DIR, BALANCE_DIR, CSV_FILENAME.
import os

# Set True to use testnet keys and endpoints
IS_TEST = False
test_mode = False # Use test mode for order execution

# Trading symbol default (override via env in app.py/init_data.py)
SYMBOL = "BTCUSDC"
TIMEFRAME = "1h"
LOOKBACK_DAYS = 160

# Binance API credentials are loaded from environment variables.
# Do not commit secrets into this repository.
def get_binance_keys(is_test: bool):
    if is_test:
        api_key = os.getenv("BINANCE_API_KEY_TEST")
        api_secret = os.getenv("BINANCE_API_SECRET_TEST")
    else:
        api_key = os.getenv("BINANCE_API_KEY_LIVE")
        api_secret = os.getenv("BINANCE_API_SECRET_LIVE")

    if not api_key or not api_secret:
        mode = "TEST" if is_test else "LIVE"
        raise RuntimeError(
            f"Missing Binance API credentials for {mode}. "
            f"Set env vars BINANCE_API_KEY_{mode} and BINANCE_API_SECRET_{mode}."
        )
    return api_key, api_secret

# Strategy-specific parameters
ATR_MULTIPLIER = 2
RR= 1.5
ATR_PERIOD = 14
EMA_WINDOW = 200  
risk_percent = 0.01  # Risk per trade as a percentage of account balance
leverage = 25  # Leverage factor
