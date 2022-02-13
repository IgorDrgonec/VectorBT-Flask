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
    AlpacaData,
)
from vectorbtpro.data.updater import DataUpdater
from vectorbtpro.utils.module_ import create__all__

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
