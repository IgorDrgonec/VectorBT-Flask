# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, focuses on the data itself."""

from vectorbtpro.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.ranges import Ranges, PatternRanges, PSC
from vectorbtpro.generic.splitters import RangeSplitter, RollingSplitter, ExpandingSplitter
from vectorbtpro.generic.splitting import *
from vectorbtpro.utils.module_ import create__all__
from vectorbtpro.utils.opt_packages import check_installed
from vectorbtpro._settings import settings

__blacklist__ = []

if not check_installed("plotly") or not settings["importing"]["plotly"]:
    __blacklist__.append("plotting")
else:
    from vectorbtpro.generic.plotting import *

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
