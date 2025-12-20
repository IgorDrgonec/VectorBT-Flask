import numpy as np
import pandas as pd
import time
import talib
import os
import threading
import asyncio
import json
import requests
import schedule
import hmac
import hashlib
import time
from urllib.parse import urlencode
from EMA_MACD import refresh_strategy_html
from binance.enums import *
from flask import Flask, request, jsonify, send_file
from flask_socketio import SocketIO
from vectorbtpro import vbt
from binance import AsyncClient, BinanceSocketManager
from datetime import datetime, timedelta
from vectorbtpro import *
from binance.client import Client
import websockets
import nest_asyncio
from strategy_config import IS_TEST, BINANCE_KEYS, SYMBOL, CSV_FILE, LOOKBACK_DAYS, TIMEFRAME, risk_percent, leverage, ATR_MULTIPLIER, RR, ATR_PERIOD, EMA_WINDOW, test_mode

nest_asyncio.apply()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket for real-time updates

# Use imported variables from strategy_config instead of manual inputs
is_test = IS_TEST
api_key = BINANCE_KEYS["test"]["api_key"] if is_test else BINANCE_KEYS["live"]["api_key"]
api_secret = BINANCE_KEYS["test"]["api_secret"] if is_test else BINANCE_KEYS["live"]["api_secret"]

# Initialize Binance client safely
client = None
try:
    client = Client(api_key, api_secret)
    # Only set FUTURES_URL if the client was created successfully
    if client is not None:
        client.FUTURES_URL = (
            "https://testnet.binancefuture.com/fapi"
            if is_test
            else "https://fapi.binance.com/fapi"
        )
        print("[INFO] Binance client initialized successfully.")
except Exception as e:
    print(f"[WARN] Binance client init failed: {e}")
    client = None

bsm = BinanceSocketManager(client)

try:
    exchange_info = client.futures_exchange_info()
    print("[INFO] Cached futures exchange info.")
except Exception as e:
    print(f"[WARN] Could not cache futures exchange info: {e}")
    exchange_info = None

def _sign(params: dict, api_secret: str) -> str:
    query = urlencode(params, doseq=True)
    return hmac.new(api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()

def futures_algo_order_request(http_method: str, path: str, params: dict):
    """Signed request to USD-M futures algo endpoints (e.g., /fapi/v1/algoOrder)."""
    base_url = "https://testnet.binancefuture.com" if is_test else "https://fapi.binance.com"
    url = base_url + path

    params = dict(params)
    params["timestamp"] = int(time.time() * 1000)
    params["signature"] = _sign(params, api_secret)

    headers = {"X-MBX-APIKEY": api_key}

    if http_method.upper() == "POST":
        return requests.post(url, headers=headers, params=params, timeout=10).json()
    if http_method.upper() == "DELETE":
        return requests.delete(url, headers=headers, params=params, timeout=10).json()
    if http_method.upper() == "GET":
        return requests.get(url, headers=headers, params=params, timeout=10).json()

    raise ValueError(f"Unsupported method: {http_method}")


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
symbol = SYMBOL
csv_file = CSV_FILE
kwargs = dict(
    start=datetime.now() - timedelta(days=LOOKBACK_DAYS),
    timeframe=TIMEFRAME,
    klines_type=2,
)
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

def get_account_balance(asset="BNFCR"):
    """Return cached balance from local JSON file (no REST)."""
    try:
        with open("initial_balance.json", "r") as f:
            data = json.load(f)
        return round(float(data.get(asset, 0.0)), 3)
    except Exception as e:
        print(f"[WARN] Failed to read balance from cache: {e}")
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
    if open_time in data.index:
        data.to_csv(csv_file)
        print(f"[DATA] Updated existing candle: {open_time}")
    else:
        new_row.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file))
        print(f"[DATA] Saved new candle: {open_time}")

def update_balance_from_api(asset="BNFCR"):
    """Fetch fresh balance from Binance Futures REST API and update local JSON cache."""
    try:
        account_info = client.futures_account_balance()
        for balance in account_info:
            if balance["asset"] == asset:
                available = round(float(balance["availableBalance"]), 8)

                # Update JSON with same flat structure
                data = {asset: available}
                with open("initial_balance.json", "w") as f:
                    json.dump(data, f, indent=4)

                print(f"[BALANCE] Updated {asset} = {available} (from REST API)")
                return available

        print(f"[WARN] Asset {asset} not found in account balance response.")
        return None

    except Exception as e:
        print(f"[ERROR] Failed to update balance from REST API: {e}")
        return None


def handle_account_update(msg):
    event_type = msg.get("e")

    if event_type == "ORDER_TRADE_UPDATE":
        order = msg.get("o", {})
        if order.get("X") == "FILLED":
            symbol = order.get("s")
            side = order.get("S")
            qty = order.get("z")
            price = order.get("ap")
            realized_pnl = order.get("rp")

            print(f"[ORDER] Trade filled for {symbol}, side: {side}, qty: {qty}, price: {price}")
            # Only update if a PnL was realized (position actually closed)
            if realized_pnl is not None and float(realized_pnl) != 0:
                print(f"[PNL] Realized PnL: {realized_pnl} {order.get('N', 'BNFCR')}")
                update_balance_from_api("BNFCR")
            else:
                print("[INFO] No realized PnL, skipping balance update.")

def execute_trade(side, order_price,stopPrice,targetPrice,risk_percent,leverage):
    usdt_balance = get_account_balance("BNFCR")
    print(usdt_balance)
    if usdt_balance is None:
        print("[WARN] Unable to fetch account balance")
        return 
    for entry in exchange_info["symbols"]:
        if entry["symbol"] == symbol:
            qty_precision = entry["quantityPrecision"]
            price_precision = entry["pricePrecision"]
            cancel_quantity = int(entry["filters"][2]["maxQty"]) * 0.95
            break
    order_price = order_price
    position_size = order_price
    side = side
    stopPrice = round(stopPrice,price_precision)
    targetPrice = round(targetPrice,price_precision)
    quantity = round((usdt_balance*risk_percent)/(abs(order_price-stopPrice)),qty_precision)
    print(f"{quantity} = ({usdt_balance} * {risk_percent}) / ({order_price} - {stopPrice})")
    print(f"Variables: side={side}, order_price={order_price}, stopPrice={stopPrice}, targetPrice={targetPrice}, risk_percent={risk_percent}, leverage={leverage}, quantity={quantity}, position_size={position_size}, cancel_quantity={cancel_quantity}")
    order(price_precision,qty_precision,side,quantity,symbol,stopPrice,targetPrice,position_size,cancel_quantity,leverage,test_mode=test_mode)

def get_position_amt(symbol):
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos["symbol"] == symbol:
                return float(pos["positionAmt"])
    except Exception as e:
        print(f"[ERROR] Failed to fetch position for {symbol}: {e}")
    return 0.0

# Function to execute strategy on new candle close
def execute_strategy(data):
    print("[INFO] Fetching latest data and executing strategy...")
    data = data[~data.index.duplicated(keep='last')].sort_index()


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

    if long_entry[-1] or short_entry[-1]:
        last_ts = data.index[-1]
        print(f"[DEBUG] Candle: {last_ts}, MACD={macd[-2]:.5f}->{macd[-1]:.5f}, "
            f"Signal={macd_signal[-2]:.5f}->{macd_signal[-1]:.5f}, "
            f"EMA={ema[-1]:.2f}, Close={close[-1]:.2f}")

    # Determine trade entry
    latest_candle_idx = -1  # Check latest closed candle
    current_position = get_position_amt(symbol)
    if current_position != 0:
        print(f"[INFO] Skipping signal: Active position exists ({current_position})")
        return  # Skip signal if a trade is open
    
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
    if msg['e'] == 'continuous_kline':  # Kline (candlestick) event
        kline = msg['k']
        is_closed = kline['x']

        # Execute strategy when a new candle closes
        if is_closed:
            print(f"[INFO] Candle closed at {datetime.fromtimestamp(kline['t']/1000)}")
            update_hdf_with_websocket(kline)
            time.sleep(0.2)
            execute_strategy(data)  # Pass updated data to strategy

# ✅ New API: Manually Open & Close Trades for Testing Binance API
# Store active trade state globally
active_trade = {"side": None, "entry_price": None, "quantity": 0}

def ok(resp: dict) -> bool:
    return isinstance(resp, dict) and resp.get("code") is None and resp.get("algoId") is not None

#Order from manual JSON
def order(price_precision, qty_precision, side, quantity, symbol,
          stopPrice, targetPrice, position_size, cancel_quantity, leverage,
          order_type=ORDER_TYPE_MARKET, Isolated=True, test_mode=False):

    # ---- normalize / round first ----
    quantity = round(quantity, qty_precision)
    stopPrice = round(stopPrice, price_precision)
    targetPrice = round(targetPrice, price_precision)

    # 0.2% buffer around stop for emergency market stop
    # NOTE: use ABS buffer so it works for both sides
    sl_offset = abs(stopPrice) * 0.002

    # Optional: “price slippage” for stop-limit to increase fill probability
    # (if you want pure maker attempts, set it to 0, but fill probability drops)
    limit_slippage = abs(stopPrice) * 0.0002  # 0.02% tweak; adjust or set 0

    try:
        create_order_func = client.futures_create_order
        if test_mode:
            create_order_func = client.futures_create_test_order

        # ---- always clear existing orders on that symbol ----
        client.futures_cancel_all_open_orders(symbol=symbol)
        futures_algo_order_request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol})
        # (You already added algoOpenOrders cancel in long branch; do it for short too.)

        # ---- leverage ----
        client.futures_change_leverage(symbol=symbol, leverage=leverage)

        # ---- place ENTRY first (so reduceOnly protectors make sense) ----
        if side == "long" and position_size != 0:
            entry = create_order_func(
                symbol=symbol, side="BUY", type=order_type, quantity=quantity, Isolated=Isolated
            )

            # Protective orders must be Algo (CONDITIONAL)
            # 1) SL STOP-LIMIT (primary) - tries to fill as maker/limit
            # For a long, SL is SELL. Trigger at stopPrice, place limit slightly below stopPrice.
            sl_limit_price = round(stopPrice - limit_slippage, price_precision)

            sl_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP",
                "triggerPrice": stopPrice,         # <-- NEW name (was stopPrice on old endpoint)
                "price": sl_limit_price,
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "quantity": quantity
            })

            # 2) Emergency SL STOP_MARKET (backup)
            emergency_trigger = round(stopPrice - sl_offset, price_precision)
            sl_market_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_MARKET",
                "triggerPrice": emergency_trigger,
                "reduceOnly": "true",
                "quantity": quantity
            })

            # 3) TP TAKE_PROFIT (limit TP like you had)
            tp_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "SELL",
                "type": "TAKE_PROFIT",
                "triggerPrice": targetPrice,
                "price": targetPrice,
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "quantity": quantity
            })
            print("[ALGO] SL_LIMIT algoId:", sl_order.get("algoId"))
            print("[ALGO] SL_MKT  algoId:", sl_market_order.get("algoId"))
            print("[ALGO] TP      algoId:", tp_order.get("algoId"))

            return True if ok(sl_order) and ok(sl_market_order) and ok(tp_order) and entry else False

        if side == "short" and position_size != 0:
            entry = create_order_func(
                symbol=symbol, side="SELL", type=order_type, quantity=quantity, Isolated=Isolated
            )

            # For a short, SL is BUY. Trigger at stopPrice, place limit slightly above stopPrice.
            sl_limit_price = round(stopPrice + limit_slippage, price_precision)

            sl_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "BUY",
                "type": "STOP",
                "triggerPrice": stopPrice,
                "price": sl_limit_price,
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "quantity": quantity
            })

            emergency_trigger = round(stopPrice + sl_offset, price_precision)
            sl_market_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "BUY",
                "type": "STOP_MARKET",
                "triggerPrice": emergency_trigger,
                "reduceOnly": "true",
                "quantity": quantity
            })

            tp_order = futures_algo_order_request("POST", "/fapi/v1/algoOrder", {
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "BUY",
                "type": "TAKE_PROFIT",
                "triggerPrice": targetPrice,
                "price": targetPrice,
                "timeInForce": "GTC",
                "reduceOnly": "true",
                "quantity": quantity
            })
            print("[ALGO] SL_LIMIT algoId:", sl_order.get("algoId"))
            print("[ALGO] SL_MKT  algoId:", sl_market_order.get("algoId"))
            print("[ALGO] TP      algoId:", tp_order.get("algoId"))

            return True if ok(sl_order) and ok(sl_market_order) and ok(tp_order) and entry else False

        # ---- FLAT: close position with market reduceOnly (unchanged) ----
        if side == "flat" and position_size == 0:
            # (your logic here is a bit odd: stopPrice/targetPrice comparisons to detect direction)
            # but I'll keep it to avoid changing behavior.
            if stopPrice < targetPrice or stopPrice == "long":  # long close
                create_order_func(symbol=symbol, side="SELL", type="MARKET",
                                  quantity=cancel_quantity, reduceOnly=True)
            if stopPrice > targetPrice or stopPrice == "short":  # short close
                create_order_func(symbol=symbol, side="BUY", type="MARKET",
                                  quantity=cancel_quantity, reduceOnly=True)
            return True

        return False

    except Exception as e:
        print(f"an exception occured - {e}")
        return False


""" def run_scheduler():
    html_file = "backtest_chart.html"
    
    # Generate once if missing
    if not os.path.exists(html_file):
        print("[SCHEDULER] Chart missing. Generating initially...")
        try:
            refresh_strategy_html()
        except Exception as e:
            print(f"[ERROR] Failed to generate chart: {e}")
    
    # Then refresh hourly
    schedule.every().hour.do(refresh_strategy_html)

    while True:
        schedule.run_pending()
        time.sleep(60) """

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

""" @app.route("/chart")
def chart():
    if not os.path.exists("backtest_chart.html"):
        return "Chart not generated yet", 500
    return send_file("backtest_chart.html") """

@app.route("/data/preview")
def preview_csv():
    if not os.path.exists(csv_file):
        return "CSV file not found", 404
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    return df.tail(20).to_html()

@app.route("/data")
def download_csv():
    """Serve the latest CSV file for inspection."""
    if not os.path.exists(csv_file):
        return "CSV file not found", 404
    return send_file(csv_file, as_attachment=True)

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

    for entry in exchange_info["symbols"]:
        if entry["symbol"] == symbol:
            qty_precision = int(entry["quantityPrecision"])
            for filter in entry["filters"]:
                cancel_quantity = (int(entry["filters"][2]["maxQty"]))*0.95

    usdt_balance = get_account_balance("BNFCR")
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

listen_key = None
keepalive_task_ref = None

async def start_keepalive_task(local_client):
    """Run a single background keepalive loop for the global listenKey."""
    global listen_key

    while True:
        await asyncio.sleep(30 * 60)  # every 30 minutes
        try:
            if listen_key:
                await local_client.futures_stream_keepalive(listen_key)
                print("[KEEPALIVE] Sent ping to keep user stream alive.")
            else:
                print("[KEEPALIVE] Skipped: listenKey not initialized yet.")
        except Exception as e:
            print(f"[KEEPALIVE ERROR] {e}")


async def account_socket():
    """Maintain a single account WebSocket with persistent listenKey."""
    global listen_key, keepalive_task_ref

    while True:
        local_client = None
        try:
            local_client = await AsyncClient.create(api_key, api_secret)
            if is_test:
                local_client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            manager = BinanceSocketManager(local_client)

            # Create or reuse existing listenKey
            if not listen_key:
                listen_key = await local_client.futures_stream_get_listen_key()
                print(f"[INIT] Created listenKey: {listen_key}")
            else:
                print(f"[REUSE] Using existing listenKey: {listen_key}")

            # Start only one keepalive task (if not already running)
            if not keepalive_task_ref or keepalive_task_ref.done():
                keepalive_task_ref = asyncio.create_task(start_keepalive_task(local_client))
                print("[TASK] Started keepalive background task")

            # Start the user data socket using the persistent listenKey
            async with manager.futures_user_socket() as stream:
                print("[SOCKET] Account socket connected.")
                while True:
                    try:
                        msg = await stream.recv()
                        handle_account_update(msg)
                    except Exception as e:
                        # Detect listenKey expiration error
                        if "listenKey does not exist" in str(e) or "code 4009" in str(e):
                            print("[WARN] ListenKey expired, requesting a new one...")
                            listen_key = await local_client.futures_stream_get_listen_key()
                            print(f"[NEW] Refreshed listenKey: {listen_key}")
                            break  # restart socket loop with new key
                        else:
                            print(f"[ERROR] Account stream message error: {e}")
                            break  # reconnect

        except Exception as e:
            print(f"[ERROR] Account WebSocket error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

        finally:
            if local_client:
                await local_client.close_connection()

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

                async with manager.kline_futures_socket(symbol=SYMBOL.lower(), interval=TIMEFRAME) as stream:
                    print("[SOCKET] Kline socket connected.")
                    while True:
                        msg = await stream.recv()
                        handle_socket_message(msg)
            except Exception as e:
                print(f"[ERROR] Kline WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                await local_client.close_connection()

    await asyncio.gather(
        binance_socket(),
        account_socket()
    )

def start_async_loop():
    """Run Binance WebSocket streams asynchronously in background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(launch_all_sockets())


print("[INIT] Starting Binance WebSocket streams...")
threading.Thread(target=start_async_loop, daemon=True).start()

initial_balance_cached = None

try:
    with open("initial_balance.json", "r") as f:
        balance_data = json.load(f)
        initial_balance_cached = round(float(balance_data["BNFCR"]), 3)
        print(f"[STARTUP] Preloaded BNFCR balance: {initial_balance_cached}")
except FileNotFoundError:
    print("[WARN] initial_balance.json not found. Did you forget to run init_data.py?")
except Exception as e:
    print(f"[ERROR] Failed to preload balance: {e}")