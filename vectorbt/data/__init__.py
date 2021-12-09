# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with data sources."""

from vectorbt.data.base import symbol_dict, Data
from vectorbt.data.custom import (
    CSVData,
    HDFData,
    SyntheticData,
    RandomData,
    GBMData,
    YFData,
    BinanceData,
    CCXTData
)
from vectorbt.data.updater import DataUpdater

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
    'CCXTData'
]

__pdoc__ = {k: False for k in __all__}
