from binance.client import Client

is_test=False
# Binance API Keys
BINANCE_API_KEY = "SyWHwZv9BTOiFN3NxJvbTlNjXdRvW9HEQdGJrZp0PFTK4aMekC2tt8d9qRNwUEej"
BINANCE_SECRET_KEY = "XkryIgFQgZhIg4l77sFfcU6LQjYlklCRqf1Eedo6XJvNJT3rjESgad0gswX8BpZY"
#Testnet Futures Binance
API_KEY_TEST = "c0bf32af094d1b6f97e53d79e2d585003754d12fbe53a65f383d71e769d5b943"
API_SECRET_TEST = "cf01902200ac97101266ec6247c80a6bcb2d005286e34b6684ea30cf6d88e20a"

api_key =  API_KEY_TEST if is_test else  BINANCE_API_KEY
api_secret = API_SECRET_TEST if is_test else BINANCE_SECRET_KEY

client = Client(api_key, api_secret)

# ⚠️ Point to the Binance Futures TESTNET
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi" if is_test else "https://fapi.binance.com/fapi"


try:
    balance = client.futures_account_balance()
    print("✅ Connected to testnet! Balance:", balance)
except Exception as e:
    print("❌ Failed to connect:", e)