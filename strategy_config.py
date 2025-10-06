# strategy_config.py

# Set True to use testnet keys and endpoints
IS_TEST = False
test_mode = False # Use test mode for order execution

# Trading symbol and data location
SYMBOL = "BTCUSDC"
CSV_FILE = "ema_macd_data.csv"
TIMEFRAME = "1h"
LOOKBACK_DAYS = 160

# Binance API keys
BINANCE_KEYS = {
    "live": {
        "api_key": "SyWHwZv9BTOiFN3NxJvbTlNjXdRvW9HEQdGJrZp0PFTK4aMekC2tt8d9qRNwUEej",
        "api_secret": "XkryIgFQgZhIg4l77sFfcU6LQjYlklCRqf1Eedo6XJvNJT3rjESgad0gswX8BpZY"
    },
    "test": {
        "api_key": "c0bf32af094d1b6f97e53d79e2d585003754d12fbe53a65f383d71e769d5b943",
        "api_secret": "cf01902200ac97101266ec6247c80a6bcb2d005286e34b6684ea30cf6d88e20a"
    }
}

# Strategy-specific parameters
ATR_MULTIPLIER = 2
RR= 1.5
ATR_PERIOD = 14
EMA_WINDOW = 200  
risk_percent = 0.01  # Risk per trade as a percentage of account balance
leverage = 25  # Leverage factor
