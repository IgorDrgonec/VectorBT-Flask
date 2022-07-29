FROM jupyter/scipy-notebook:python-3.9.12

USER root
WORKDIR /tmp

RUN apt-get update && \
 apt-get install -yq --no-install-recommends cmake && \
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
    'plotly>=5.0.0' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    jupyter lab build --minimize=False

RUN pip install --quiet --no-cache-dir \
    'numpy==1.21' \
    'numba==0.55.1' \
    'schedule' \
    'requests' \
    'tqdm' \
    'python-dateutil' \
    'dateparser' \
    'imageio' \
    'mypy_extensions' \
    'humanize' \
    'attrs>=19.2.0' \
    'yfinance>=0.1.63' \
    'python-binance>=1.0.16' \
    'alpaca_trade_api' \
    'ccxt>=1.89.14' \
    'polygon-api-client>=1.0.0' \
    'nasdaq-data-link' \
    'tvdatafeed' \
    'ta' \
    'pandas_ta' \
    'TA-Lib' \
    'plotly-resampler' \
    'quantstats>=0.0.37' \
    'PyPortfolioOpt>=1.5.1' \
    'python-telegram-bot>=13.4' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"


ADD ./vectorbtpro ./vectorbtpro
ADD setup.py ./
ADD extra-requirements.txt ./
ADD LICENSE.md ./
ADD README.md ./
RUN pip install --quiet --no-cache-dir --no-deps .

WORKDIR "$HOME/work"
