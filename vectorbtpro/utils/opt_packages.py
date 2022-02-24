# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for managing optional dependencies."""

import importlib
import warnings

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import ReadonlyConfig

__pdoc__ = {}

opt_package_config = ReadonlyConfig(
    dict(
        yfinance=dict(name="Yahoo! Finance", link="https://github.com/ranaroussi/yfinance"),
        binance=dict(name="python-binance", link="https://github.com/sammchardy/python-binance"),
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
        dill=dict(name="dill", link="https://github.com/uqfoundation/dill"),
        alpaca_trade_api=dict(name="Alpaca Trade API", link="https://github.com/alpacahq/alpaca-trade-api-python"),
        polygon=dict(name="Polygon API Client", link="https://github.com/polygon-io/client-python"),
        bs4=dict(name="Beautiful Soup", link="https://www.crummy.com/software/BeautifulSoup/bs4/doc/"),
    )
)
"""_"""

__pdoc__[
    "opt_package_config"
] = f"""Config for optional packages.

```python
{opt_package_config.prettify()}
```
"""


def check_installed(pkg_name: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(pkg_name) is not None


def get_installed_overview() -> tp.Dict[str, bool]:
    """Get an overview of installed packages in `opt_package_config`."""
    return {pkg_name: check_installed(pkg_name) for pkg_name in opt_package_config.keys()}


def assert_can_import(pkg_name: str) -> None:
    """Assert that the package can be imported. Must be listed in `opt_package_config`."""
    if pkg_name not in opt_package_config:
        raise KeyError(f"Package '{pkg_name}' not found in opt_package_config")
    if not check_installed(pkg_name):
        raise ImportError(
            f"Please install {opt_package_config[pkg_name]['name']}: {opt_package_config[pkg_name]['link']}"
        )


def warn_cannot_import(pkg_name: str) -> None:
    """Warn if the package is cannot be imported. Must be listed in `opt_package_config`."""
    if pkg_name not in opt_package_config:
        raise KeyError(f"Package '{pkg_name}' not found in opt_package_config")
    if not check_installed(pkg_name):
        warnings.warn(
            f"Consider installing {opt_package_config[pkg_name]['name']}: {opt_package_config[pkg_name]['link']}",
            stacklevel=2,
        )
