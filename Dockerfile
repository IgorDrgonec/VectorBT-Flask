# --- Base image: official vectorbtpro dev setup ---
FROM quay.io/jupyter/scipy-notebook:python-3.12

USER root
WORKDIR /tmp

# === System deps + TA-Lib 0.6.4 (native C lib) ===
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake nodejs git build-essential wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz

USER ${NB_UID}
RUN pip install uv

# === VectorBT Pro core ===
COPY ./vectorbtpro ./vectorbtpro
COPY pyproject.toml LICENSE README.md ./

RUN echo "numpy>1.24.3" > override.txt
RUN uv pip install --system --no-cache-dir ".[all-stable]" --override override.txt

# === Optional unstable deps ===
RUN uv pip install --system --no-cache-dir --no-deps pandas-ta
RUN uv pip install --system --no-cache-dir --no-deps git+https://github.com/Marigold/universal-portfolios.git

# === Install TA-Lib Python binding compatible with NumPy 2 / Py 3.12 ===
RUN uv pip install --system --no-cache-dir --no-deps \
    git+https://github.com/mrjbq7/ta-lib@numpy2-fix

# === Bot runtime dependencies ===
RUN uv pip install --system --no-cache-dir \
    flask Flask-SocketIO gunicorn gevent nest-asyncio \
    python-binance==1.0.19 "websockets>=12,<13" \
    schedule requests tqdm yfinance>=0.2.20 ccxt>=1.89.14 \
    SQLAlchemy>=2.0.0 duckdb duckdb-engine pyarrow tables>=3.8.0 \
    ta technical numexpr>=2.8.4 hyperopt optuna pathos mpire dask \
    ray>=2.10.0 plotly plotly-resampler kaleido quantstats>=0.0.37 \
    PyPortfolioOpt>=1.5.1 Riskfolio-Lib>=3.3.0 \
    python-telegram-bot>=13.4 dill lz4 blosc2 tabulate pandas-datareader \
    polygon-api-client>=1.0.0 beautifulsoup4 nasdaq-data-link alpha_vantage databento

# === Copy your trading bot ===
WORKDIR /usr/src/app
COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .
COPY strategy_config.py .

EXPOSE 8080
CMD python init_data.py && gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:8080 app:app
