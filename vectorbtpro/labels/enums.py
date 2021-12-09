# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types.

Defines enums and other schemas for `vectorbtpro.labels`."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.docs import stringify

__all__ = [
    'TrendMode'
]

__pdoc__ = {}


class TrendModeT(tp.NamedTuple):
    Binary: int = 0
    BinaryCont: int = 1
    BinaryContSat: int = 2
    PctChange: int = 3
    PctChangeNorm: int = 4


TrendMode = TrendModeT()
"""_"""

__pdoc__['TrendMode'] = f"""Trend mode.

```json
{stringify(TrendMode)}
```

Attributes:
    Binary: See `vectorbtpro.labels.nb.bn_trend_labels_nb`.
    BinaryCont: See `vectorbtpro.labels.nb.bn_cont_trend_labels_nb`.
    BinaryContSat: See `vectorbtpro.labels.nb.bn_cont_sat_trend_labels_nb`.
    PctChange: See `vectorbtpro.labels.nb.pct_trend_labels_nb`.
    PctChangeNorm: See `vectorbtpro.labels.nb.pct_trend_labels_nb` with `normalize` set to True.
"""
