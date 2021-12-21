# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with signals."""

from vectorbtpro.signals.enums import *
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators import (
    RAND,
    RANDX,
    RANDNX,
    RPROB,
    RPROBX,
    RPROBCX,
    RPROBNX,
    STX,
    STCX,
    OHLCSTX,
    OHLCSTCX
)

__all__ = [
    'SignalFactory',
    'RAND',
    'RANDX',
    'RANDNX',
    'RPROB',
    'RPROBX',
    'RPROBCX',
    'RPROBNX',
    'STX',
    'STCX',
    'OHLCSTX',
    'OHLCSTCX'
]

__pdoc__ = {k: False for k in __all__}
