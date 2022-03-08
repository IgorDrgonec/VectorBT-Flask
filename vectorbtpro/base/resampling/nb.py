# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for resampling."""

import numpy as np
from numba import prange
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

ns_dt = np.timedelta64(1, "ns")
"""Nanosecond."""

us_dt = ns_dt * 1000
"""Microsecond."""

ms_dt = us_dt * 1000
"""Millisecond."""

s_dt = ms_dt * 1000
"""Second."""

m_dt = s_dt * 60
"""Minute."""

h_dt = m_dt * 60
"""Hour."""

d_dt = h_dt * 24
"""Day."""


@register_jitted(cache=True)
def date_range_nb(
    start: np.datetime64,
    end: np.datetime64,
    freq: np.timedelta64 = d_dt,
    incl_left: bool = True,
    incl_right: bool = True,
) -> tp.Array1d:
    """Generate a datetime index with nanosecond precision from a date range.

    Inspired by [pandas.date_range](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html)."""
    values_len = int(np.floor((end - start) / freq)) + 1
    values = np.empty(values_len, dtype="datetime64[ns]")
    for i in range(values_len):
        values[i] = start + i * freq
    if start == end:
        if not incl_left and not incl_right:
            values = values[1:-1]
    else:
        if not incl_left or not incl_right:
            if not incl_left and len(values) and values[0] == start:
                values = values[1:]
            if not incl_right and len(values) and values[-1] == end:
                values = values[:-1]
    return values


@register_jitted(cache=True)
def map_to_index_nb(
    from_index: tp.Array1d,
    to_index: tp.Array1d,
    before: bool = False,
    raise_missing: bool = True,
) -> tp.Array1d:
    """Get index of each in `from_index` in `to_index`.

    If `before` is True, applied on elements that come before and including that index.
    Otherwise, applied on elements that come after and including that index.

    If `raise_missing` is True, will throw an error if an index cannot be mapped.
    Otherwise, the element for that index becomes -1."""
    out = np.empty(len(from_index), dtype=np.int_)
    from_j = 0
    for i in range(len(from_index)):
        if i > 0 and from_index[i] <= from_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(to_index)):
            if j > 0 and to_index[j] <= to_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if before and from_index[i] <= to_index[j]:
                if j == 0 or to_index[j - 1] < from_index[i]:
                    out[i] = from_j = j
                    found = True
                    break
            if not before and to_index[j] <= from_index[i]:
                if j == len(to_index) - 1 or from_index[i] < to_index[j + 1]:
                    out[i] = from_j = j
                    found = True
                    break
        if not found:
            if raise_missing:
                raise ValueError("Resampling failed: cannot map some indices")
            out[i] = -1
    return out


@register_jitted(cache=True)
def index_difference_nb(
    from_index: tp.Array1d,
    to_index: tp.Array1d,
) -> tp.Array1d:
    """Get elements in `from_index` not present in `to_index`."""
    out = np.empty(len(from_index), dtype=np.int_)
    from_j = 0
    k = 0
    for i in range(len(from_index)):
        if i > 0 and from_index[i] <= from_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(to_index)):
            if j > 0 and to_index[j] <= to_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if from_index[i] < to_index[j]:
                break
            if from_index[i] == to_index[j]:
                from_j = j
                found = True
                break
            from_j = j
        if not found:
            out[k] = i
            k += 1
    return out[:k]
