import socketio

sio = socketio.Client()

@sio.on('trade_signal')
def handle_trade_signal(data):
    print("\n🔔 Trade Signal Received:")
    print(f"Status: {data['status']}")
    print(f"Side: {data['side'].upper()}")
    print(f"Price: {data['price']}")

sio.connect('http://127.0.0.1:8080')
sio.wait()
