# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, focuses on the data itself."""

from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.generic.splitters import RangeSplitter, RollingSplitter, ExpandingSplitter

__all__ = [
    'Ranges',
    'Drawdowns',
    'RangeSplitter',
    'RollingSplitter',
    'ExpandingSplitter'
]

__blacklist__ = []

try:
    import plotly
except ImportError:
    __blacklist__.append('plotting')
else:
    from vectorbtpro.generic.plotting import Gauge, Bar, Scatter, Histogram, Box, Heatmap, Volume

    __all__.append('Gauge')
    __all__.append('Bar')
    __all__.append('Scatter')
    __all__.append('Histogram')
    __all__.append('Box')
    __all__.append('Heatmap')
    __all__.append('Volume')

__pdoc__ = {k: False for k in __all__}
