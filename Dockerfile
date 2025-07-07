# ---------- Stage 1: Build TA-Lib ----------
FROM debian:bullseye-slim AS ta-lib-builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential wget cmake && \
    wget https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install

# ---------- Stage 2: Main Runtime Image ----------
FROM jupyter/scipy-notebook:python-3.9.12

# Switch to root to install system dependencies
USER root

# Copy TA-Lib from builder stage
COPY --from=ta-lib-builder /usr/lib/libta* /usr/lib/
COPY --from=ta-lib-builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Install Python dependencies (keep this early for layer caching)
RUN pip install --quiet --no-cache-dir \
    'pybind11' \
    'llvmlite' \
    'jupyter-dash' \
    'plotly>=5.0.0' \
    'kaleido' \
    'numpy==1.23.5' \
    'pandas>=1.5.0' \
    'numba==0.56.4' \
    'scipy' \
    'scikit-learn' \
    'schedule' \
    'requests' \
    'tqdm' \
    'python-dateutil' \
    'dateparser' \
    'imageio' \
    'mypy_extensions' \
    'humanize' \
    'attrs>=21.1.0' \
    'websocket-client' \
    'websockets>=10.4,<11' \
    'yfinance>=0.2.20' \
    'python-binance>=1.0.16' \
    'alpaca-py' \
    'ccxt>=1.89.14' \
    'tables>=3.8.0' \
    'SQLAlchemy>=2.0.0' \
    'duckdb' \
    'duckdb-engine' \
    'pyarrow' \
    'polygon-api-client>=1.0.0' \
    'beautifulsoup4' \
    'nasdaq-data-link' \
    'alpha_vantage' \
    'databento' \
    'TA-Lib==0.4.21' \
    'ta' \
    'pandas_ta' \
    'technical' \
    'numexpr>=2.8.4' \
    'hyperopt' \
    'optuna' \
    'pathos' \
    'mpire' \
    'dask' \
    'ray>=1.4.1' \
    'plotly-resampler' \
    'quantstats>=0.0.37' \
    'PyPortfolioOpt>=1.5.1' \
    'Riskfolio-Lib>=3.3.0' \
    'python-telegram-bot>=13.4' \
    'dill' \
    'lz4' \
    'blosc2' \
    'tabulate' \
    'universal-portfolios' \
    'pandas_datareader' \
    'bottleneck' \
    'flask' \
    'Flask-SocketIO' \
    'gunicorn' \
    'gevent' \
    'nest-asyncio'

# Optional: install cvxopt using conda
RUN conda install --quiet --yes -c conda-forge cvxopt

# Add vectorbtpro package and install
COPY ./vectorbtpro /tmp/vectorbtpro
COPY pyproject.toml LICENSE README.md /tmp/
RUN pip install --quiet --no-cache-dir --no-deps /tmp

# Set working dir for app
WORKDIR /usr/src/app

# Copy app source
COPY app.py init_data.py EMA_MACD.py ./

# Expose port for web
EXPOSE 8080

# Startup command
CMD python init_data.py && gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
