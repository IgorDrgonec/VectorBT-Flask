# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for plotting with Plotly Express."""

from vectorbtpro.utils.module_ import create__all__
from vectorbtpro.utils.module_ import check_installed
from vectorbtpro._settings import settings

__blacklist__ = []

if not check_installed("plotly") or not settings["importing"]["plotly"]:
    __blacklist__.append("accessors")
else:
    from vectorbtpro.px.accessors import PXAccessor, PXSRAccessor, PXDFAccessor

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
