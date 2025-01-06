# app.py
from flask import Flask
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from VectorBTPro + Flask on Render!"

if __name__ == "__main__":
    # On Render, the environment variable PORT is typically set (e.g. 10000).
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)