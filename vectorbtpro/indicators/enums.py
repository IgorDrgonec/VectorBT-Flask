# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for indicators."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__all__ = ["Pivot", "TrendMode"]

__pdoc__ = {}


# ############# Enums ############# #


class PivotT(tp.NamedTuple):
    Valley: int = -1
    Peak: int = 1


Pivot = PivotT()
"""_"""

__pdoc__[
    "Pivot"
] = f"""Pivot.

```python
{prettify(Pivot)}
```
"""


class TrendModeT(tp.NamedTuple):
    Downward: int = -1
    Upward: int = 1


TrendMode = TrendModeT()
"""_"""

__pdoc__[
    "TrendMode"
] = f"""Trend mode.

```python
{prettify(TrendMode)}
```
"""
