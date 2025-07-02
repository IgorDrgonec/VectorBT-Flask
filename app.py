import numpy as np
import pandas as pd
import time
import talib
import os
import threading
import asyncio
import json
import requests
from binance.enums import *
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from vectorbtpro import vbt
from binance import AsyncClient, BinanceSocketManager
from datetime import datetime, timedelta
from vectorbtpro import *
from binance.client import Client
import websockets
import nest_asyncio
from flask import send_from_directory

nest_asyncio.apply()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket for real-time updates

is_test=True
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
bsm = BinanceSocketManager(client)

def check_api_weight(api_key):
    """Check and log current API usage headers for futures account."""
    url = "https://fapi.binance.com/fapi/v2/account"
    headers = {
        'X-MBX-APIKEY': api_key
    }
    try:
        response = requests.get(url, headers=headers)
        print("[WEIGHT] API Rate Limits:")
        print("  X-MBX-USED-WEIGHT-1M:", response.headers.get("X-MBX-USED-WEIGHT-1M"))
        print("  X-MBX-ORDER-COUNT-10S:", response.headers.get("X-MBX-ORDER-COUNT-10S"))
        print("  X-MBX-ORDER-COUNT-1M:", response.headers.get("X-MBX-ORDER-COUNT-1M"))
    except Exception as e:
        print(f"[ERROR] Failed to fetch API weights: {e}")

# Configure Vectorbt with Binance API
vbt.BinanceData.set_custom_settings(
    client_config=dict(
        api_key=api_key,
        api_secret=api_secret
    )
)

# Parameters for historical data retrieval
symbol = 'BTCUSDC'
kwargs = dict(
    start=datetime.now() - timedelta(days=60),
    timeframe='15m',
    klines_type=2,
)

csv_file = "ema_macd_data.csv"
data = None

if os.path.exists(csv_file):
    try:
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print("[INFO] Loaded data from CSV")
    except Exception as e:
        print(f"[ERROR] Failed to load cached CSV: {e}")
        data = pd.DataFrame()
else:
    print("[CRITICAL] CSV file not found. Please run init_data.py before starting app.")
    data = pd.DataFrame()

# Strategy Parameters
ATR_MULTIPLIER = 2
RR = 1.5
ATR_PERIOD = 14
EMA_WINDOW = 200

# Order Parameters
risk_percent = 0.01
# qty_precision = 2
leverage = 25

_balance_cache = {}
_balance_timestamp = {}


def get_account_balance(asset="USDC", cache_seconds=60):
    """Return futures account balance while respecting API rate limits."""
    now = time.time()
    # Prefer value provided by the account WebSocket if available
    if asset in _balance_cache:
        if now - _balance_timestamp.get(asset, 0) < cache_seconds:
            return _balance_cache[asset]
        # even if stale return to avoid REST call unless absolutely needed
        cached = _balance_cache.get(asset)
        if cached is not None:
            return cached

    try:
        account_info = client.futures_account()
    except Exception as e:
        print(f"[WARN] Failed to fetch futures account info: {e}")
        return _balance_cache.get(asset)

    for balance in account_info.get("assets", []):
        if balance["asset"] == asset:
            try:
                available = round(float(balance["availableBalance"]), 3)
                _balance_cache[asset] = available
                _balance_timestamp[asset] = now
                return available
            except Exception as e:
                print(f"[WARN] Could not parse balance for {asset}: {e}")
                return _balance_cache.get(asset)
    return None

# Function to update HDF with WebSocket data
def update_hdf_with_websocket(kline):
    global data

    # Extract the open time and symbol for the index
    open_time = pd.to_datetime(kline['t'], unit='ms', utc=True)

    # Build a DataFrame with the same MultiIndex as your HDF file
    new_row = pd.DataFrame({
        "Open": [float(kline['o'])],
        "High": [float(kline['h'])],
        "Low": [float(kline['l'])],
        "Close": [float(kline['c'])],
        "Volume": [float(kline['v'])],
        "Quote volume": [float(kline.get('q', 0))],
        "Trade count": [int(kline.get('n', 0))],
        "Taker base volume": [float(kline.get('V', 0))],
        "Taker quote volume": [float(kline.get('Q', 0))]
    }, index=pd.DatetimeIndex([open_time], name='Open time'))

    # Append to in-memory DataFrame and CSV
    data = pd.concat([data, new_row])
    data = data[~data.index.duplicated(keep='last')]
    data.to_csv(csv_file)
    #print("[INFO] Appended new row and updated CSV file.")

def handle_account_update(msg):
    event_type = msg.get("e")

    if event_type == "ACCOUNT_UPDATE":
        for bal in msg.get("a", {}).get("B", []):
            asset = bal.get("a")
            wallet_balance = bal.get("wb")
            if asset and wallet_balance is not None:
                try:
                    balance = float(wallet_balance)
                    _balance_cache[asset] = balance
                    _balance_timestamp[asset] = time.time()
                    print(f"[BALANCE] Updated {asset} (ACCOUNT_UPDATE): {balance}")
                except (TypeError, ValueError):
                    continue

    elif event_type == "ORDER_TRADE_UPDATE":
        order = msg.get("o", {})
        if order.get("X") == "FILLED":
            symbol = order.get("s")
            side = order.get("S")
            qty = order.get("z")
            price = order.get("ap")
            realized_pnl = order.get("rp")

            print(f"[ORDER] Trade filled for {symbol}, side: {side}, qty: {qty}, price: {price}")
            if realized_pnl is not None:
                print(f"[PNL] Realized PnL: {realized_pnl} {order.get('N', 'USDC')}")

            # Trigger a manual refresh from REST as backup
            refreshed = get_account_balance("USDC", cache_seconds=0)
            print(f"[BALANCE] Manually refreshed (after fill): {refreshed}")

def execute_trade(side, order_price,stopPrice,targetPrice,risk_percent,leverage):
    usdt_balance = get_account_balance("USDC")
    print(usdt_balance)
    if usdt_balance is None:
        print("[WARN] Unable to fetch account balance")
        return 
    symbol_info = client.futures_exchange_info()
    for entry in symbol_info["symbols"]:
        if entry["symbol"] == symbol:
            qty_precision = 1  # int(entry["quantityPrecision"])
            cancel_quantity = int(entry["filters"][2]["maxQty"]) * 0.95
            break
    order_price = order_price
    position_size = order_price
    side = side
    stopPrice = round(stopPrice,qty_precision)
    targetPrice = round(targetPrice,qty_precision)
    quantity = round((usdt_balance*risk_percent)/(abs(order_price-stopPrice)),3)
    print(f"{quantity} = ({usdt_balance} * {risk_percent}) / ({order_price} - {stopPrice})")
    print(f"Variables: side={side}, order_price={order_price}, stopPrice={stopPrice}, targetPrice={targetPrice}, risk_percent={risk_percent}, leverage={leverage}, quantity={quantity}, position_size={position_size}, cancel_quantity={cancel_quantity}")
    order(side,quantity,symbol,stopPrice,targetPrice,position_size,cancel_quantity,leverage)

# Function to execute strategy on new candle close
def execute_strategy(data):
    print("[INFO] Fetching latest data and executing strategy...")

    # Extract OHLCV values
    high = data["High"].values
    low = data["Low"].values
    close = data["Close"].values

    # Calculate indicators
    atr = talib.ATR(high, low, close, ATR_PERIOD)
    macd, macd_signal, _ = talib.MACD(close)
    ema = talib.EMA(close, EMA_WINDOW)

    # Calculate entry conditions
    long_entry = (vbt.nb.crossed_above_1d_nb(macd, macd_signal)) & (close > ema) & (macd < 0)
    short_entry = (vbt.nb.crossed_below_1d_nb(macd, macd_signal)) & (close < ema) & (macd > 0)
    #long_entry = close > ema
    #short_entry = close < ema

    # Determine trade entry
    latest_candle_idx = -1  # Check latest closed candle
    if long_entry[latest_candle_idx]:
        print("[ENTRY] 🔥 Long Entry Signal Detected!")
        socketio.emit('trade_signal', {"side": "long", "price": close[latest_candle_idx]})
        side = "long"
        stopPrice = close[latest_candle_idx] - (atr[latest_candle_idx] * ATR_MULTIPLIER)
        targetPrice = close[latest_candle_idx] + (atr[latest_candle_idx] * ATR_MULTIPLIER * RR)

        execute_trade(side, close[latest_candle_idx], stopPrice, targetPrice,risk_percent,leverage)
    elif short_entry[latest_candle_idx]:
        print("[ENTRY] ❄️ Short Entry Signal Detected!")
        socketio.emit('trade_signal', {"side": "short", "price": close[latest_candle_idx]})
        side = "short"
        stopPrice = close[latest_candle_idx] + (atr[latest_candle_idx] * ATR_MULTIPLIER)
        targetPrice = close[latest_candle_idx] - (atr[latest_candle_idx] * ATR_MULTIPLIER * RR)

        execute_trade(side, close[latest_candle_idx], stopPrice, targetPrice, risk_percent,leverage)
    else:
        print("[INFO] No trade entry on this candle.")

    return long_entry, short_entry


# WebSocket event listener function
def handle_socket_message(msg):
    """Handles real-time kline messages from Binance WebSocket."""
    if msg['e'] == 'kline':  # Kline (candlestick) event
        kline = msg['k']
        is_closed = kline['x']

        # Execute strategy when a new candle closes
        if is_closed:
            print(f"[INFO] Candle closed at {datetime.fromtimestamp(kline['t']/1000)}")
            update_hdf_with_websocket(kline)
            execute_strategy(data)  # Pass updated data to strategy

# ✅ New API: Manually Open & Close Trades for Testing Binance API
# Store active trade state globally
active_trade = {"side": None, "entry_price": None, "quantity": 0}

#Order from manual JSON
def order(side, quantity, symbol, stopPrice, targetPrice, position_size, cancel_quantity, leverage, order_type=ORDER_TYPE_MARKET, Isolated=True):
    try:
        if side == "long" and position_size != 0:
            result = client.futures_change_leverage(symbol = symbol, leverage = leverage)
            #print(result)
            #print(f"sending order {order_type} - {side} {quantity} {symbol}")
            cancel = client.futures_cancel_all_open_orders(symbol = symbol)
            sl_order = client.futures_create_order(symbol=symbol, side='SELL', type=FUTURE_ORDER_TYPE_STOP_MARKET, quantity=quantity,stopPrice=stopPrice, timeInForce=TIME_IN_FORCE_GTC, closePosition = True)
            #print(sl_order)
            tp_order = client.futures_create_order(symbol=symbol, side='SELL', type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, quantity=quantity,stopPrice=targetPrice, timeInForce=TIME_IN_FORCE_GTC, closePosition = True)
            #print(tp_order)
            order = client.futures_create_order(symbol=symbol, side='BUY', type=order_type, quantity=quantity, Isolated=Isolated)
            #print(order)
        if side == "short" and position_size != 0:
            result = client.futures_change_leverage(symbol = symbol, leverage = leverage)
            #print(result)
            #print(f"sending order {order_type} - {side} {quantity} {symbol}") 
            cancel = client.futures_cancel_all_open_orders(symbol = symbol)         
            sl_order = client.futures_create_order(symbol=symbol, side='BUY', type=FUTURE_ORDER_TYPE_STOP_MARKET, quantity=quantity,stopPrice=stopPrice, timeInForce=TIME_IN_FORCE_GTC, closePosition = True)
            #print(sl_order)
            tp_order = client.futures_create_order(symbol=symbol, side='BUY', type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET, quantity=quantity,stopPrice=targetPrice, timeInForce=TIME_IN_FORCE_GTC, closePosition = True)
            #print(tp_order)      
            order = client.futures_create_order(symbol=symbol, side='SELL', type=ORDER_TYPE_MARKET, quantity=quantity, Isolated=True)
            #print(order)     
        if side == "flat" and position_size == 0:
            if stopPrice < targetPrice or stopPrice == "long": #long
                cancel_buy = client.futures_create_order(symbol = symbol, side = "SELL", type = ORDER_TYPE_MARKET, quantity = cancel_quantity, reduceOnly = True)
            if stopPrice > targetPrice or stopPrice == "short": #short
                cancel_sell = client.futures_create_order(symbol = symbol, side = "BUY", type = ORDER_TYPE_MARKET, quantity = cancel_quantity, reduceOnly = True)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False
    else:   
        if side == "long" and position_size != 0 and sl_order and tp_order:
            check_api_weight(api_key)
            return True
        if side == "short" and position_size != 0 and sl_order and tp_order:
            check_api_weight(api_key)
            return True
        if side == "flat" and position_size == 0:
            check_api_weight(api_key)
            return True

pf1 = None

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Bot</title>
        </head>
        <body style="display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
            <h1 style="font-size: 5rem; color: #333;">Tu sa zarábajú milióny!</h1>
        </body>
    </html>
    """

@app.route("/chart")
def serve_chart():
    return send_from_directory("static", "backtest_chart.html")

@app.route("/trade", methods=['POST'])
def manual_trade():
    #print(request.data)
    data = json.loads(request.data)
     
    side = data['strategy']['market_position']
    symbol= str(data["ticker"]).rstrip(".P")
    leverage_dict = {
    "AXSUSDT": 25,
    "RENUSDT": 25,
    "XLMUSDT": 25,
    "ZRXUSDT": 25,
    "GMTUSDT": 25,
    "TOMOUSDT": 25,
    "STXUSDT": 20,
    "AUDIOUSDT": 20
    }
    leverage = leverage_dict.get(symbol, 30)

    symbol_info = client.futures_exchange_info()
    for entry in symbol_info["symbols"]:
        if entry["symbol"] == symbol:
            qty_precision = int(entry["quantityPrecision"])
            for filter in entry["filters"]:
                cancel_quantity = (int(entry["filters"][2]["maxQty"]))*0.95

    usdt_balance = get_account_balance("USDC")
    if usdt_balance is None:
        print("[WARN] Unable to fetch account balance")
        return {"code": "error", "message": "balance fetch failed"}

    if side != "flat": 
        if data['strategy']['order_contracts'] == "DaviddTech":
            risk_percent = data["risk_percent"]    
            order_price = data["strategy"]["order_price"]
            position_size = order_price
            stoploss_price = data["stop_loss"]
            quantity = round((usdt_balance*risk_percent)/(abs(order_price-stoploss_price)),qty_precision)
            print(f"{quantity} = ({usdt_balance} * {risk_percent}) / ({order_price} - {stoploss_price})")
        else:
            if symbol == "AAVEUSDT":
                quantity = round(data['strategy']['order_contracts'],1)
            if symbol == "DOGEUSDT" or symbol == "ONEUSDT" or symbol == "MATICUSDT":
                quantity = round(data['strategy']['order_contracts'],0)
            else:
                quantity = round(data['strategy']['order_contracts'],3)
            strategy_equity = data['strategy']['Strategy_equity']    
            quantity = round((quantity/strategy_equity)*usdt_balance,3) #If strategy has equity and position size
            position_size = data["strategy"]["position_size"]
        if symbol == "LTCUSDT":
            stopPrice = round(data["stop_loss"],2)
            targetPrice = round(data["take_profit"],2)
        else:
            stopPrice = data["stop_loss"]
            targetPrice = data["take_profit"]    
    else: #If side == "flat"
        quantity = cancel_quantity
        stopPrice = data["stop_loss"]
        targetPrice = data["take_profit"] 
        position_size = 0

    order_response = order(side,quantity,symbol,stopPrice,targetPrice,position_size,cancel_quantity,leverage)

    if order_response or side =="flat":
        return {
            "code": "success",
            "message": "order executed"
        }
    else:
        print("order failed")

        return {
            "code": "error",
            "message": "order failed"
        }

# Function to start Binance WebSocket (candlestick stream)
async def launch_all_sockets():
    print("[INFO] Launching all WebSockets...")

    async def binance_socket():
        while True:
            try:
                local_client = await AsyncClient.create(api_key, api_secret)
                if is_test:
                    local_client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
                manager = BinanceSocketManager(local_client)

                async with manager.kline_socket(symbol='btcusdc', interval=AsyncClient.KLINE_INTERVAL_15MINUTE) as stream:
                    print("[SOCKET] Kline socket connected.")
                    while True:
                        msg = await stream.recv()
                        handle_socket_message(msg)
            except Exception as e:
                print(f"[ERROR] Kline WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                await local_client.close_connection()

    async def account_socket():
        while True:
            try:
                local_client = await AsyncClient.create(api_key, api_secret)
                if is_test:
                    local_client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
                manager = BinanceSocketManager(local_client)

                # Fetch the listenKey just for keepalive (not to use directly in socket)
                listen_key = await local_client.futures_stream_get_listen_key()
                print(f"[SOCKET] Account socket connected. ListenKey: {listen_key}")

                # Task to keep listen key alive
                async def keepalive():
                    while True:
                        await asyncio.sleep(30 * 60)  # Every 30 minutes
                        try:
                            await local_client.futures_stream_keepalive(listen_key)
                            print("[KEEPALIVE] Sent ping to keep user stream alive.")
                        except Exception as e:
                            print(f"[KEEPALIVE ERROR] Failed to ping listenKey: {e}")

                asyncio.create_task(keepalive())

                async with manager.futures_user_socket() as stream:
                    while True:
                        msg = await stream.recv()
                        print(f"[WS MESSAGE] {msg}")
                        handle_account_update(msg)

            except Exception as e:
                print(f"[ERROR] Account WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                await local_client.close_connection()

    await asyncio.gather(
        binance_socket(),
        account_socket()
    )


# Launch sockets only when running with Gunicorn
if __name__ != "__main__":
    def start_async_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(launch_all_sockets())

    threading.Thread(target=start_async_loop, daemon=True).start()

    port = int(os.environ.get("PORT", 8080))
    # socketio.run(app, host="0.0.0.0", port=port, debug=True, use_reloader=False)