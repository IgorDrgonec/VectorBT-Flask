# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Modules with custom data sources."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.data.custom.alpaca import *
    from vectorbtpro.data.custom.av import *
    from vectorbtpro.data.custom.binance import *
    from vectorbtpro.data.custom.ccxt import *
    from vectorbtpro.data.custom.csv import *
    from vectorbtpro.data.custom.custom import *
    from vectorbtpro.data.custom.file import *
    from vectorbtpro.data.custom.gbm import *
    from vectorbtpro.data.custom.gbm import *
    from vectorbtpro.data.custom.gbm_ohlc import *
    from vectorbtpro.data.custom.hdf import *
    from vectorbtpro.data.custom.local import *
    from vectorbtpro.data.custom.ndl import *
    from vectorbtpro.data.custom.polygon import *
    from vectorbtpro.data.custom.random import *
    from vectorbtpro.data.custom.random_ohlc import *
    from vectorbtpro.data.custom.remote import *
    from vectorbtpro.data.custom.synthetic import *
    from vectorbtpro.data.custom.tv import *
    from vectorbtpro.data.custom.yf import *
