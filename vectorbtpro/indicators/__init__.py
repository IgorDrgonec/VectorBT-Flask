# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for building and running indicators.

Technical indicators are used to see past trends and anticipate future moves.
See [Using Technical Indicators to Develop Trading Strategies](https://www.investopedia.com/articles/trading/11/indicators-and-strategies-explained.asp)."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.custom import (
    MA,
    MSTD,
    BBANDS,
    RSI,
    STOCH,
    MACD,
    ATR,
    OBV
)
from vectorbtpro.indicators.factory import IndicatorFactory, IndicatorBase
from vectorbtpro.utils.module_ import create__all__


def talib(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_talib`."""
    return IndicatorFactory.from_talib(*args, **kwargs)


def pandas_ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_pandas_ta`."""
    return IndicatorFactory.from_pandas_ta(*args, **kwargs)


def ta(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_ta`."""
    return IndicatorFactory.from_ta(*args, **kwargs)


def wqa101(*args, **kwargs) -> tp.Type[IndicatorBase]:
    """Shortcut for `vectorbtpro.indicators.factory.IndicatorFactory.from_wqa101`."""
    return IndicatorFactory.from_wqa101(*args, **kwargs)


__whitelist__ = [
    'talib',
    'pandas_ta',
    'ta',
    'wqa101'
]
__all__ = create__all__(__name__)
__pdoc__ = {k: k in __whitelist__ for k in __all__}


