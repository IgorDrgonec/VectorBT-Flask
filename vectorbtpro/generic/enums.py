# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for generic data.

Defines enums and other schemas for `vectorbtpro.generic`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__all__ = [
    "RangeStatus",
    "DrawdownStatus",
    "drawdown_dt",
    "range_dt",
    "RollSumAIS",
    "RollSumAOS",
    "RollProdAIS",
    "RollProdAOS",
    "RollMeanAIS",
    "RollMeanAOS",
    "RollStdAIS",
    "RollStdAOS",
    "WMMeanAIS",
    "WMMeanAOS",
    "EWMMeanAIS",
    "EWMMeanAOS",
    "EWMStdAIS",
    "EWMStdAOS",
    "RollCovAIS",
    "RollCovAOS",
    "RollCorrAIS",
    "RollCorrAOS",
]

__pdoc__ = {}


# ############# Enums ############# #


class RangeStatusT(tp.NamedTuple):
    Open: int
    Closed: int


RangeStatus = RangeStatusT(*range(2))
"""_"""

__pdoc__[
    "RangeStatus"
] = f"""Range status.

```python
{prettify(RangeStatus)}
```
"""


class DrawdownStatusT(tp.NamedTuple):
    Active: int
    Recovered: int


DrawdownStatus = DrawdownStatusT(*range(2))
"""_"""

__pdoc__[
    "DrawdownStatus"
] = f"""Drawdown status.

```python
{prettify(DrawdownStatus)}
```
"""

# ############# Records ############# #

range_dt = np.dtype(
    [("id", np.int_), ("col", np.int_), ("start_idx", np.int_), ("end_idx", np.int_), ("status", np.int_)],
    align=True,
)
"""_"""

__pdoc__[
    "range_dt"
] = f"""`np.dtype` of range records.

```python
{prettify(range_dt)}
```
"""

drawdown_dt = np.dtype(
    [
        ("id", np.int_),
        ("col", np.int_),
        ("peak_idx", np.int_),
        ("start_idx", np.int_),
        ("valley_idx", np.int_),
        ("end_idx", np.int_),
        ("peak_val", np.float_),
        ("valley_val", np.float_),
        ("end_val", np.float_),
        ("status", np.int_),
    ],
    align=True,
)
"""_"""

__pdoc__[
    "drawdown_dt"
] = f"""`np.dtype` of drawdown records.

```python
{prettify(drawdown_dt)}
```
"""


# ############# States ############# #


class RollSumAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollSumAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""


class RollSumAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollSumAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_sum_acc_nb`."""


class RollProdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumprod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollProdAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""


class RollProdAOS(tp.NamedTuple):
    cumprod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollProdAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_prod_acc_nb`."""


class RollMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollMeanAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""


class RollMeanAOS(tp.NamedTuple):
    cumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollMeanAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_mean_acc_nb`."""


class RollStdAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__[
    "RollStdAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""


class RollStdAOS(tp.NamedTuple):
    cumsum: float
    cumsum_sq: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollStdAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_std_acc_nb`."""


class WMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    pre_window_value: float
    cumsum: float
    wcumsum: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "WMMeanAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""


class WMMeanAOS(tp.NamedTuple):
    cumsum: float
    wcumsum: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "WMMeanAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.wm_mean_acc_nb`."""


class EWMMeanAIS(tp.NamedTuple):
    i: int
    value: float
    old_wt: float
    weighted_avg: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMMeanAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`.

To get `alpha`, use one of the following:

* `vectorbtpro.generic.nb.rolling.alpha_from_com_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_span_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_halflife_nb`
* `vectorbtpro.generic.nb.rolling.alpha_from_wilder_nb`"""


class EWMMeanAOS(tp.NamedTuple):
    old_wt: float
    weighted_avg: float
    nobs: int
    value: float


__pdoc__[
    "EWMMeanAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.ewm_mean_acc_nb`."""


class EWMStdAIS(tp.NamedTuple):
    i: int
    value: float
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    alpha: float
    minp: tp.Optional[int]
    adjust: bool


__pdoc__[
    "EWMStdAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`.

For tips on `alpha`, see `EWMMeanAIS`."""


class EWMStdAOS(tp.NamedTuple):
    mean_x: float
    mean_y: float
    cov: float
    sum_wt: float
    sum_wt2: float
    old_wt: float
    nobs: int
    value: float


__pdoc__[
    "EWMStdAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.ewm_std_acc_nb`."""


class RollCovAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]
    ddof: int


__pdoc__[
    "RollCovAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""


class RollCovAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollCovAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_cov_acc_nb`."""


class RollCorrAIS(tp.NamedTuple):
    i: int
    value1: float
    value2: float
    pre_window_value1: float
    pre_window_value2: float
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window: int
    minp: tp.Optional[int]


__pdoc__[
    "RollCorrAIS"
] = """A named tuple representing an input state of 
`vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""


class RollCorrAOS(tp.NamedTuple):
    cumsum1: float
    cumsum2: float
    cumsum_sq1: float
    cumsum_sq2: float
    cumsum_prod: float
    nancnt: int
    window_len: int
    value: float


__pdoc__[
    "RollCorrAOS"
] = """A named tuple representing an output state of 
`vectorbtpro.generic.nb.rolling.rolling_corr_acc_nb`."""
