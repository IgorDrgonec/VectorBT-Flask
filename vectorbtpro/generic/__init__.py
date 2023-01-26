# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, focuses on the data itself."""

__dont_climb_from__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["plotting"] = "plotly"
