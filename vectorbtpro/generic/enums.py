# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for generic data.

Defines enums and other schemas for `vectorbtpro.generic`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__all__ = [
    'RangeStatus',
    'DrawdownStatus',
    'drawdown_dt',
    'range_dt'
]

__pdoc__ = {}


# ############# Enums ############# #

class RangeStatusT(tp.NamedTuple):
    Open: int
    Closed: int


RangeStatus = RangeStatusT(*range(2))
"""_"""

__pdoc__['RangeStatus'] = f"""Range status.

```python
{prettify(RangeStatus)}
```
"""


class DrawdownStatusT(tp.NamedTuple):
    Active: int
    Recovered: int


DrawdownStatus = DrawdownStatusT(*range(2))
"""_"""

__pdoc__['DrawdownStatus'] = f"""Drawdown status.

```python
{prettify(DrawdownStatus)}
```
"""

# ############# Records ############# #

range_dt = np.dtype([
    ('id', np.int_),
    ('col', np.int_),
    ('start_idx', np.int_),
    ('end_idx', np.int_),
    ('status', np.int_)
], align=True)
"""_"""

__pdoc__['range_dt'] = f"""`np.dtype` of range records.

```python
{prettify(range_dt)}
```
"""

drawdown_dt = np.dtype([
    ('id', np.int_),
    ('col', np.int_),
    ('peak_idx', np.int_),
    ('start_idx', np.int_),
    ('valley_idx', np.int_),
    ('end_idx', np.int_),
    ('peak_val', np.float_),
    ('valley_val', np.float_),
    ('end_val', np.float_),
    ('status', np.int_),
], align=True)
"""_"""

__pdoc__['drawdown_dt'] = f"""`np.dtype` of drawdown records.

```python
{prettify(drawdown_dt)}
```
"""
