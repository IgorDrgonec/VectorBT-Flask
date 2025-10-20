# Base image (keeps conda + Jupyter tooling + Python 3.9)
FROM jupyter/scipy-notebook:python-3.9.12

# Switch to root for system installs
USER root
WORKDIR /tmp

# --- System dependencies ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        wget \
        git \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Build TA-Lib from source ---
RUN wget https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr --build=unknown-unknown-linux && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Switch back to default notebook user
USER ${NB_UID}

# --- Core Jupyter utilities ---
RUN pip install --quiet --no-cache-dir \
    'jupyter-dash' \
    'plotly>=5.0.0' \
    'kaleido' && \
    jupyter lab build --minimize=False

# --- Core numerical stack (via conda for stability) ---
RUN conda install --quiet --yes -c conda-forge \
    'llvmlite==0.39.1' \
    'numba==0.56.4' \
    'cvxopt' \
    'bottleneck' && \
    conda clean -afy

# --- Python dependencies (via pip) ---
RUN pip install --quiet --no-cache-dir \
    'pybind11' \
    'numpy==1.23.5' \
    'pandas>=1.5.0' \
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
    'pandas-datareader' \
    'universal-portfolios'

# --- Install pandas_ta from ZIP (no GitHub auth required) ---
RUN pip install --quiet --no-cache-dir \
    https://github.com/twopirllc/pandas-ta/archive/refs/heads/main.zip

# --- Add your custom vectorbtpro package ---
ADD ./vectorbtpro ./vectorbtpro
ADD pyproject.toml LICENSE README.md ./
RUN pip install --quiet --no-cache-dir --no-deps .

# --- Web app dependencies ---
WORKDIR /usr/src/app
RUN pip install --quiet --no-cache-dir \
    'flask' \
    'Flask-SocketIO' \
    'gunicorn' \
    'gevent' \
    'nest-asyncio' \
    'python-binance==1.0.19' \
    --upgrade pip

# --- Copy bot files ---
COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .
COPY strategy_config.py .

EXPOSE 8080

# --- Start bot ---
CMD python init_data.py && gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:8080 app:app

#CMD python init_data.py && python EMA_MACD.py && gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
#CMD ["gunicorn", "--worker-class", "gevent", "--bind", "0.0.0.0:8080", "app:app"]
