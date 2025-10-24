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
        "api_key": "Y5YKyeloqeBzQjmjGQRseXtHnwInIuFe3ksDm0cPVOS4zlObzS1dFYnriYJmtZWm",
        "api_secret": "vm5j2ZZqZUBS5oFc1FQyPPsZ1y5CPlpXYzHGedaLRpaK5FMdqULHiLazXGgjF24f"
    }
}

# Strategy-specific parameters
ATR_MULTIPLIER = 2
RR= 1.5
ATR_PERIOD = 14
EMA_WINDOW = 200  
risk_percent = 0.01  # Risk per trade as a percentage of account balance
leverage = 25  # Leverage factor
