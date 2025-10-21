# ─────────────── Base from VectorBT Pro author ───────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive

# --- System dependencies + TA-Lib build ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git wget sudo vim && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# --- VectorBT Pro core ---
COPY ./vectorbtpro ./vectorbtpro
COPY pyproject.toml LICENSE README.md ./

# Keep NumPy <2 for TA-Lib compatibility
RUN echo "numpy>1.24.3,<2.0" > override.txt
RUN uv pip install --no-cache-dir ".[all-stable]" --override override.txt

# Unstable deps from VectorBT Pro author
RUN uv pip install --no-cache-dir --no-deps pandas-ta

# --- Your app dependencies ---
RUN uv pip install --no-cache-dir \
    flask Flask-SocketIO gunicorn gevent nest-asyncio \
    python-binance==1.0.19 "websockets>=12,<13" \
    schedule requests tqdm ccxt>=1.89.14 \
    SQLAlchemy>=2.0.0 duckdb duckdb-engine pyarrow tables>=3.8.0 \
    ta technical numexpr>=2.8.4 hyperopt optuna pathos mpire dask \
    ray>=2.10.0 "plotly<=5.22.0" plotly-resampler kaleido \
    PyPortfolioOpt>=1.5.1 Riskfolio-Lib>=3.3.0 \
    python-telegram-bot>=13.4 dill lz4 blosc2 tabulate pandas-datareader \
    polygon-api-client>=1.0.0 beautifulsoup4 nasdaq-data-link alpha_vantage databento

# --- Copy your trading bot ---
WORKDIR /usr/src/app
COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .
COPY strategy_config.py .

RUN uv pip install --no-cache-dir --upgrade "websockets>=12,<13"

EXPOSE 8080
CMD python init_data.py && gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:8080 app:app
