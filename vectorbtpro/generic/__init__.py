# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, focuses on the data itself."""

from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.ranges import Ranges
from vectorbtpro.generic.splitters import RangeSplitter, RollingSplitter, ExpandingSplitter
from vectorbtpro.utils.module_ import create__all__

__blacklist__ = []

try:
    import plotly
except ImportError:
    __blacklist__.append('plotting')
else:
    from vectorbtpro.generic.plotting import Gauge, Bar, Scatter, Histogram, Box, Heatmap, Volume

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
