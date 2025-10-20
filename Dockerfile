########## ────────────────  STAGE 1 – BUILD & COMPILE  ──────────────── ##########
FROM quay.io/jupyter/scipy-notebook:python-3.12 AS builder

USER root
WORKDIR /tmp

# --- Build dependencies & TA-Lib 0.6.4 ---
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake git wget nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

USER ${NB_UID}
RUN pip install uv

# --- VectorBT Pro base ---
COPY ./vectorbtpro ./vectorbtpro
COPY pyproject.toml LICENSE README.md ./

RUN echo "numpy>1.24.3" > override.txt
RUN uv pip install --system --no-cache-dir ".[all-stable]" --override override.txt

# --- Optional unstable deps ---
RUN uv pip install --system --no-cache-dir --no-deps pandas-ta
RUN uv pip install --system --no-cache-dir --no-deps git+https://github.com/Marigold/universal-portfolios.git

# --- TA-Lib Python binding compatible with NumPy 2 / Py 3.12 ---
RUN uv pip install --system --no-cache-dir --no-deps \
    git+https://github.com/mrjbq7/ta-lib@numpy2-fix


########## ────────────────  STAGE 2 – RUNTIME (DEPLOYMENT)  ──────────────── ##########
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive

# --- Copy compiled TA-Lib native library from builder ---
COPY --from=builder /usr/lib/libta_lib.so* /usr/lib/

# --- Minimal system deps for runtime ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

# --- Copy installed Python environment from builder (faster than reinstall) ---
COPY --from=builder /usr/local /usr/local

# --- Runtime-specific dependencies ---
RUN uv pip install --no-cache-dir \
    flask Flask-SocketIO gunicorn gevent nest-asyncio \
    python-binance==1.0.19 "websockets>=12,<13" \
    schedule requests tqdm yfinance>=0.2.20 ccxt>=1.89.14 \
    SQLAlchemy>=2.0.0 duckdb duckdb-engine pyarrow tables>=3.8.0 \
    ta technical numexpr>=2.8.4 hyperopt optuna pathos mpire dask \
    ray>=2.10.0 plotly plotly-resampler kaleido quantstats>=0.0.37 \
    PyPortfolioOpt>=1.5.1 Riskfolio-Lib>=3.3.0 \
    python-telegram-bot>=13.4 dill lz4 blosc2 tabulate pandas-datareader \
    polygon-api-client>=1.0.0 beautifulsoup4 nasdaq-data-link alpha_vantage databento

# --- Copy your bot ---
WORKDIR /usr/src/app
COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .
COPY strategy_config.py .

EXPOSE 8080
CMD python init_data.py && gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:8080 app:app
