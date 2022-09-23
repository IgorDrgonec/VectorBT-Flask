# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with returns.

Offers common financial risk and performance metrics as found in [empyrical](https://github.com/quantopian/empyrical),
an adapter for quantstats, and other features based on returns."""

from vectorbtpro.returns.accessors import ReturnsAccessor, ReturnsSRAccessor, ReturnsDFAccessor
from vectorbtpro.returns.enums import *
from vectorbtpro.utils.module_ import create__all__

__blacklist__ = []

try:
    import quantstats
except ImportError:
    __blacklist__.append("qs_adapter")

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
