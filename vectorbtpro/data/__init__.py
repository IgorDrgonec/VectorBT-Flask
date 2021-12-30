# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with data sources."""

from vectorbtpro.data.base import symbol_dict, Data
from vectorbtpro.data.custom import (
    CSVData,
    HDFData,
    SyntheticData,
    RandomData,
    GBMData,
    YFData,
    BinanceData,
    CCXTData,
    AlpacaData
)
from vectorbtpro.data.updater import DataUpdater

__all__ = [
    'symbol_dict',
    'Data',
    'DataUpdater',
    'CSVData',
    'HDFData',
    'SyntheticData',
    'RandomData',
    'GBMData',
    'YFData',
    'BinanceData',
    'CCXTData',
    'AlpacaData'
]

__pdoc__ = {k: False for k in __all__}
