# app.py
from flask import Flask
from vectorbtpro import *
from binance.client import Client
from datetime import datetime, timedelta
import os
import time

app = Flask(__name__)

vbt.BinanceData.set_custom_settings(
    client_config=dict(
        api_key="SyWHwZv9BTOiFN3NxJvbTlNjXdRvW9HEQdGJrZp0PFTK4aMekC2tt8d9qRNwUEej",
        api_secret="XkryIgFQgZhIg4l77sFfcU6LQjYlklCRqf1Eedo6XJvNJT3rjESgad0gswX8BpZY"
    )
)

kwargs = dict(
    start=datetime.now() - timedelta(days=1), 
    timeframe='1m',
    klines_type = 2,
    #delay = 2
 )

@app.route("/")
def refresh_price():
    data = vbt.BinanceData.pull(
        'BTCUSDT',
        **kwargs
    )

    data.to_hdf('chart_data.h5')
    data = vbt.HDFData.pull('chart_data.h5')
    
    # 4. Convert to Pandas DataFrame
    df = data.get()

     # 5. Plot as a candlestick chart using vectorbt's built-in ohlcv plotting
    fig = df.vbt.ohlcv.plot()

    # 6. Convert the Plotly figure to HTML so Flask can return it
    html_chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

    return html_chart

if __name__ == "__main__":
    # On Render, the environment variable PORT is typically set (e.g. 10000).
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)