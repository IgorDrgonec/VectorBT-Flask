FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_SYSTEM_PYTHON=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git wget && \
    rm -rf /var/lib/apt/lists/*

# TA-Lib 0.6.4
RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

# Copy VBT Pro package
WORKDIR /tmp
COPY ./vectorbtpro ./vectorbtpro
COPY pyproject.toml LICENSE README.md ./

# Prefer newer NumPy when resolving extras
RUN echo "numpy>1.24.3" > override.txt
# Install VBT Pro with "all-stable" extras
RUN uv pip install --no-cache-dir ".[all-stable]" --override override.txt

# Unstable/strict deps (pin to things that play well with Py 3.12)
# - pandas-ta: OK on 3.12 from PyPI
RUN uv pip install --no-cache-dir --no-deps pandas-ta
# - universal-portfolios: use patched fork instead of PyPI to avoid pip.req error
RUN uv pip install --no-cache-dir --no-deps git+https://github.com/Marigold/universal-portfolios.git

# === Your bot runtime deps ===
# Keep websockets compatible with Py 3.12 (drop <11 pin); 12.x works well
RUN uv pip install --no-cache-dir \
    flask \
    Flask-SocketIO \
    gunicorn \
    gevent \
    nest-asyncio \
    python-binance==1.0.19 \
    websockets>=12,<13 \
    schedule requests tqdm \
    yfinance>=0.2.20 \
    ccxt>=1.89.14 \
    SQLAlchemy>=2.0.0 \
    duckdb duckdb-engine pyarrow \
    tables>=3.8.0 \
    TA-Lib==0.4.28 \
    ta technical \
    numexpr>=2.8.4 \
    hyperopt optuna pathos mpire dask \
    ray>=2.10.0 \
    plotly plotly-resampler kaleido \
    quantstats>=0.0.37 \
    PyPortfolioOpt>=1.5.1 \
    Riskfolio-Lib>=3.3.0 \
    python-telegram-bot>=13.4 \
    dill lz4 blosc2 tabulate \
    pandas-datareader \
    polygon-api-client>=1.0.0 \
    beautifulsoup4 nasdaq-data-link alpha_vantage databento

# Copy your app
WORKDIR /usr/src/app
COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .
COPY strategy_config.py .

EXPOSE 8080

CMD python init_data.py && gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:8080 app:app
