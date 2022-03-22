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
def map_to_target_index_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
    raise_missing: bool = True,
) -> tp.Array1d:
    """Get index of each in `source_index` in `target_index`.

    If `before` is True, applied on elements that come before and including that index.
    Otherwise, applied on elements that come after and including that index.

    If `raise_missing` is True, will throw an error if an index cannot be mapped.
    Otherwise, the element for that index becomes -1."""
    out = np.empty(len(source_index), dtype=np.int_)
    from_j = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] <= source_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if target_freq is None:
                if before and source_index[i] <= target_index[j]:
                    if j == 0 or target_index[j - 1] < source_index[i]:
                        out[i] = from_j = j
                        found = True
                        break
                if not before and target_index[j] <= source_index[i]:
                    if j == len(target_index) - 1 or source_index[i] < target_index[j + 1]:
                        out[i] = from_j = j
                        found = True
                        break
            else:
                if before and target_index[j] - target_freq < source_index[i] <= target_index[j]:
                    out[i] = from_j = j
                    found = True
                    break
                if not before and target_index[j] <= source_index[i] < target_index[j] + target_freq:
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
    source_index: tp.Array1d,
    target_index: tp.Array1d,
) -> tp.Array1d:
    """Get elements in `source_index` not present in `target_index`."""
    out = np.empty(len(source_index), dtype=np.int_)
    from_j = 0
    k = 0
    for i in range(len(source_index)):
        if i > 0 and source_index[i] <= source_index[i - 1]:
            raise ValueError("Array index must be strictly increasing")
        found = False
        for j in range(from_j, len(target_index)):
            if j > 0 and target_index[j] <= target_index[j - 1]:
                raise ValueError("Target index must be strictly increasing")
            if source_index[i] < target_index[j]:
                break
            if source_index[i] == target_index[j]:
                from_j = j
                found = True
                break
            from_j = j
        if not found:
            out[k] = i
            k += 1
    return out[:k]


@register_jitted(cache=True)
def map_index_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    target_freq: tp.Optional[tp.Scalar] = None,
    before: bool = False,
) -> tp.Array2d:
    """Get source bounds that correspond to each target index.

    If `target_freq` is not None, the right bound is limited by the frequency in `target_freq`.
    Otherwise, the right bound is the next index in `target_index`.

    Returns a 2-dim array where the first column is the absolute start index (including) nad
    the second column is the absolute end index (excluding).

    If an element cannot be mapped, the start and end of the range becomes -1.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed."""
    out = np.empty((len(target_index), 2), dtype=np.int_)

    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")

        from_j = -1
        for j in range(to_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if target_freq is None:
                if before:
                    if i == 0 and source_index[j] <= target_index[i]:
                        found = True
                    elif i > 0 and target_index[i - 1] < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if i == len(target_index) - 1 and target_index[i] <= source_index[j]:
                        found = True
                    elif i < len(target_index) - 1 and target_index[i] <= source_index[j] < target_index[i + 1]:
                        found = True
                    elif i < len(target_index) - 1 and source_index[j] >= target_index[i + 1]:
                        break
            else:
                if before:
                    if target_index[i] - target_freq < source_index[j] <= target_index[i]:
                        found = True
                    elif source_index[j] > target_index[i]:
                        break
                else:
                    if target_index[i] <= source_index[j] < target_index[i] + target_freq:
                        found = True
                    elif source_index[j] >= target_index[i] + target_freq:
                        break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            out[i, 0] = -1
            out[i, 1] = -1
        else:
            out[i, 0] = from_j
            out[i, 1] = to_j

    return out


@register_jitted(cache=True)
def map_bounds_to_source_ranges_nb(
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool = True,
    closed_rbound: bool = False,
    skipna: bool = False,
) -> tp.Array2d:
    """Get source bounds that correspond to target bounds.

    Returns a 2-dim array where the first column is the absolute start index (including) nad
    the second column is the absolute end index (excluding).

    If an element cannot be mapped, the start and end of the range becomes -1.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed."""
    out = np.empty((len(target_lbound_index), 2), dtype=np.int_)
    k = 0

    to_j = 0
    for i in range(len(target_lbound_index)):
        if i > 0 and target_lbound_index[i] < target_lbound_index[i - 1]:
            raise ValueError("Target left-bound index must be increasing")
        if i > 0 and target_rbound_index[i] < target_rbound_index[i - 1]:
            raise ValueError("Target right-bound index must be increasing")

        from_j = -1
        for j in range(len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
            if closed_lbound and closed_rbound:
                if target_lbound_index[i] <= source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            elif closed_lbound:
                if target_lbound_index[i] <= source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            elif closed_rbound:
                if target_lbound_index[i] < source_index[j] <= target_rbound_index[i]:
                    found = True
                elif source_index[j] > target_rbound_index[i]:
                    break
            else:
                if target_lbound_index[i] < source_index[j] < target_rbound_index[i]:
                    found = True
                elif source_index[j] >= target_rbound_index[i]:
                    break
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            if skipna:
                continue
            out[i, 0] = -1
            out[i, 1] = -1
        else:
            out[i, 0] = from_j
            out[i, 1] = to_j
            k += 1

    if skipna:
        return out[:k]
    return out


@register_jitted(cache=True, is_generated_jit=True)
def latest_at_index_1d_nb(
    arr: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = None,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array1d:
    """Get the latest in `arr` at each index in `target_index` based on `source_index`.

    If `source_rbound` is True, then each element in `source_index` is effectively located at
    the right bound, which is the frequency or the next element (excluding) if the frequency is None.
    The same for `target_rbound` and `target_index`.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed.

        If `arr` contains bar data, both indexes must represent the opening time."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _latest_at_index_1d_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty(target_index.shape[0], dtype=dtype)
        curr_j = -1
        last_j = curr_j
        last_valid = np.nan
        for i in range(len(target_index)):
            if i > 0 and target_index[i] < target_index[i - 1]:
                raise ValueError("Target index must be increasing")
            target_bound_inf = target_rbound and i == len(target_index) - 1 and target_freq is None

            last_valid_at_i = np.nan
            for j in range(curr_j + 1, source_index.shape[0]):
                if j > 0 and source_index[j] < source_index[j - 1]:
                    raise ValueError("Array index must be increasing")
                source_bound_inf = source_rbound and j == len(source_index) - 1 and source_freq is None

                if source_bound_inf and target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    break
                if source_bound_inf:
                    break
                if target_bound_inf:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    continue

                if source_rbound and target_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_val > target_val:
                        break
                elif source_rbound:
                    if source_freq is None:
                        source_val = source_index[j + 1]
                    else:
                        source_val = source_index[j] + source_freq
                    if source_val > target_index[i]:
                        break
                elif target_rbound:
                    if target_freq is None:
                        target_val = target_index[i + 1]
                    else:
                        target_val = target_index[i] + target_freq
                    if source_index[j] >= target_val:
                        break
                else:
                    if source_index[j] > target_index[i]:
                        break
                curr_j = j
                if not np.isnan(arr[curr_j]):
                    last_valid_at_i = arr[curr_j]

            if ffill and not np.isnan(last_valid_at_i):
                last_valid = last_valid_at_i
            if curr_j == -1 or (not ffill and curr_j == last_j):
                out[i] = nan_value
            else:
                if ffill:
                    if np.isnan(last_valid):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid
                else:
                    if np.isnan(last_valid_at_i):
                        out[i] = nan_value
                    else:
                        out[i] = last_valid_at_i
                last_j = curr_j

        return out

    if not nb_enabled:
        return _latest_at_index_1d_nb(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _latest_at_index_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        source_index=None,
        target_index=None,
        source_freq=None,
        target_freq=None,
        source_rbound=None,
        target_rbound=None,
        nan_value=None,
        ffill=None,
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"}, is_generated_jit=True)
def latest_at_index_nb(
    arr: tp.Array2d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    source_freq: tp.Optional[tp.Scalar] = None,
    target_freq: tp.Optional[tp.Scalar] = None,
    source_rbound: bool = False,
    target_rbound: bool = False,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array2d:
    """2-dim version of `latest_at_index_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _latest_at_index_nb(
        arr,
        source_index,
        target_index,
        source_freq,
        target_freq,
        source_rbound,
        target_rbound,
        nan_value,
        ffill,
    ):
        out = np.empty((target_index.shape[0], arr.shape[1]), dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = latest_at_index_1d_nb(
                arr[:, col],
                source_index,
                target_index,
                source_freq=source_freq,
                target_freq=target_freq,
                source_rbound=source_rbound,
                target_rbound=target_rbound,
                nan_value=nan_value,
                ffill=ffill,
            )
        return out

    if not nb_enabled:
        return _latest_at_index_nb(
            arr,
            source_index,
            target_index,
            source_freq,
            target_freq,
            source_rbound,
            target_rbound,
            nan_value,
            ffill,
        )

    return _latest_at_index_nb
