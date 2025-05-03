# app.py
from flask import Flask
from flask_socketio import SocketIO
from vectorbtpro import vbt
from binance.client import Client
from binance.streams import BinanceSocketManager
from datetime import datetime, timedelta
import os
import threading
import asyncio

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")  # WebSocket for real-time updates

# Binance API Keys
BINANCE_API_KEY = "SyWHwZv9BTOiFN3NxJvbTlNjXdRvW9HEQdGJrZp0PFTK4aMekC2tt8d9qRNwUEej"
BINANCE_SECRET_KEY = "XkryIgFQgZhIg4l77sFfcU6LQjYlklCRqf1Eedo6XJvNJT3rjESgad0gswX8BpZY"
client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
bsm = BinanceSocketManager(client)

# Configure Vectorbt with Binance API
vbt.BinanceData.set_custom_settings(
    client_config=dict(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_SECRET_KEY
    )
)

# Parameters for historical data retrieval
kwargs = dict(
    start=datetime.now() - timedelta(days=1),
    timeframe='1m',
    klines_type=2,
)

# Function to generate and return the latest chart HTML
def generate_chart():
    data = vbt.HDFData.pull('chart_data.h5')
    df = data.get()
    fig = df.vbt.ohlcv.plot()
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

# WebSocket event listener function
def handle_socket_message(msg):
    """Handles real-time kline messages from Binance WebSocket."""
    if msg['e'] == 'kline':  # Kline (candlestick) event
        kline = msg['k']
        candlestick_data = {
            "time": kline['t'],
            "open": float(kline['o']),
            "high": float(kline['h']),
            "low": float(kline['l']),
            "close": float(kline['c']),
            "is_closed": kline['x']
        }
        
        # Save latest data to HDF
        data = vbt.BinanceData.pull('BTCUSDT', **kwargs)
        data.to_hdf('chart_data.h5')

        # Generate the updated chart
        updated_chart = generate_chart()

        # Emit the latest candlestick data and chart to WebSocket clients
        socketio.emit('new_candle', {
            "candlestick": candlestick_data,
            "chart": updated_chart  # Send the updated chart HTML
        })

# Function to start Binance WebSocket in a separate thread
def start_binance_socket():
    print("[INFO] Starting Binance WebSocket...")

    async def handle_socket():
        async with bsm.kline_socket(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE) as stream:
            while True:
                msg = await stream.recv()  # Receive messages asynchronously
                handle_socket_message(msg)  # Process each message

    # Run the WebSocket in an asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(handle_socket())

# Flask route to return the latest chart as HTML
@app.route("/")
def refresh_chart():
    return generate_chart()

if __name__ == "__main__":
    # Manually start WebSocket before running Flask
    threading.Thread(target=start_binance_socket, daemon=True).start()

    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
