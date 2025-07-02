FROM jupyter/scipy-notebook:python-3.9.12

USER root
WORKDIR /tmp

RUN apt-get update && \
 apt-get install -yq --no-install-recommends cmake --fix-missing && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*

RUN wget https://netcologne.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz && \
  tar -xvzf ta-lib-0.4.0-src.tar.gz && \
  cd ta-lib/ && \
  ./configure --prefix=/usr --build=unknown-unknown-linux && \
  make && \
  make install

RUN rm -R ta-lib ta-lib-0.4.0-src.tar.gz

USER ${NB_UID}

RUN pip install --quiet --no-cache-dir \
    'jupyter-dash' \
    'plotly>=5.0.0' \
    'kaleido' && \
    jupyter lab build --minimize=False

RUN pip install --quiet --no-cache-dir 'pybind11'
RUN pip install --quiet --no-cache-dir --ignore-installed 'llvmlite'

RUN pip install --quiet --no-cache-dir \
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
    'tabulate'


RUN pip install --quiet --no-cache-dir --no-deps 'universal-portfolios'
RUN pip install --quiet --no-cache-dir 'pandas_datareader'
RUN conda install --quiet --yes -c conda-forge cvxopt
RUN pip install --quiet --no-cache-dir --upgrade bottleneck

ADD ./vectorbtpro ./vectorbtpro
ADD pyproject.toml ./
ADD LICENSE ./
ADD README.md ./
RUN pip install --quiet --no-cache-dir --no-deps .

WORKDIR /usr/src/app

RUN pip install --quiet --no-cache-dir \
    'flask' \
    'Flask-SocketIO' \
    'gunicorn' \
    'gevent' \
    'python-binance==1.0.19' \
    'nest-asyncio' \
    --upgrade pip



COPY app.py .
COPY init_data.py .
COPY EMA_MACD.py .

EXPOSE 8080

CMD python init_data.py && gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
#CMD python init_data.py && python EMA_MACD.py && gunicorn --worker-class gevent --bind 0.0.0.0:8080 app:app
#CMD ["gunicorn", "--worker-class", "gevent", "--bind", "0.0.0.0:8080", "app:app"]
