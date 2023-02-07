# Copyright (c) 2023 Oleg Polakow. All rights reserved.

"""Optional dependencies."""

from vectorbtpro.utils.config import HybridConfig

__all__ = []

__pdoc__ = {}

opt_dep_config = HybridConfig(
    dict(
        yfinance=dict(name="Yahoo! Finance", link="https://github.com/ranaroussi/yfinance"),
        binance=dict(name="Python Binance", link="https://github.com/sammchardy/python-binance"),
        ccxt=dict(name="CCXT", link="https://github.com/ccxt/ccxt"),
        ta=dict(name="Technical Analysis Library", link="https://github.com/bukosabino/ta"),
        pandas_ta=dict(name="Pandas TA", link="https://github.com/twopirllc/pandas-ta"),
        talib=dict(name="TA-Lib", link="https://github.com/mrjbq7/ta-lib"),
        bottleneck=dict(name="Bottleneck", link="https://github.com/pydata/bottleneck"),
        numexpr=dict(name="NumExpr", link="https://github.com/pydata/numexpr"),
        ray=dict(name="Ray", link="https://github.com/ray-project/ray"),
        dask=dict(name="Dask", link="https://github.com/dask/dask"),
        matplotlib=dict(name="Matplotlib", link="https://github.com/matplotlib/matplotlib"),
        plotly=dict(name="Plotly", link="https://github.com/plotly/plotly.py"),
        ipywidgets=dict(name="ipywidgets", link="https://github.com/jupyter-widgets/ipywidgets"),
        kaleido=dict(name="Kaleido", link="https://github.com/plotly/Kaleido"),
        telegram=dict(name="Python Telegram Bot", link="https://github.com/python-telegram-bot/python-telegram-bot"),
        quantstats=dict(name="QuantStats", link="https://github.com/ranaroussi/quantstats"),
        dill=dict(name="Dill", link="https://github.com/uqfoundation/dill"),
        alpaca=dict(name="Python SDK for Alpaca API", link="https://github.com/alpacahq/alpaca-py"),
        polygon=dict(name="Polygon API Client", link="https://github.com/polygon-io/client-python"),
        bs4=dict(name="Beautiful Soup", link="https://www.crummy.com/software/BeautifulSoup/bs4/doc/"),
        nasdaqdatalink=dict(name="Nasdaq Data Link", link="https://github.com/Nasdaq/data-link-python"),
        pypfopt=dict(name="PyPortfolioOpt", link="https://github.com/robertmartin8/PyPortfolioOpt"),
        universal=dict(name="Universal Portfolios", link="https://github.com/Marigold/universal-portfolios"),
        plotly_resampler=dict(name="Plotly Resampler", link="https://github.com/predict-idlab/plotly-resampler"),
        technical=dict(name="Technical", link="https://github.com/freqtrade/technical"),
        riskfolio=dict(name="Riskfolio-Lib", link="https://github.com/dcajasn/Riskfolio-Lib"),
        pathos=dict(name="Pathos", link="https://github.com/uqfoundation/pathos"),
        lz4=dict(name="LZ4", link="https://github.com/python-lz4/python-lz4"),
        blosc=dict(name="Blosc", link="https://github.com/Blosc/python-blosc"),
    )
)
"""_"""

__pdoc__[
    "opt_dep_config"
] = f"""Config for optional packages.

```python
{opt_dep_config.prettify()}
```
"""
