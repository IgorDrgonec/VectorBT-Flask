# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for generic data.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of a backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    Rolling functions with `minp=None` have `min_periods` set to the window size.
    
    All functions passed as argument must be Numba-compiled.

!!! warning
    Make sure to use `parallel=True` only if your columns are independent.
"""

import numpy as np
from numba import prange
from numba.core.types import Omitted
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic.enums import RangeStatus, DrawdownStatus, range_dt, drawdown_dt
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.template import Rep


@register_jitted(cache=True)
def shuffle_1d_nb(arr: tp.Array1d, seed: tp.Optional[int] = None) -> tp.Array1d:
    """Shuffle each column in the array.

    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(arr)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        seed=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True)
def shuffle_nb(arr: tp.Array2d, seed: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `shuffle_1d_nb`."""
    if seed is not None:
        np.random.seed(seed)
    out = np.empty_like(arr, dtype=arr.dtype)

    for col in range(arr.shape[1]):
        out[:, col] = np.random.permutation(arr[:, col])
    return out


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_1d_nb(arr: tp.Array1d, mask: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Set each element to a value by boolean mask."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_1d_nb(arr, mask, value):
        out = arr.astype(dtype)
        out[mask] = value
        return out

    if not nb_enabled:
        return _set_by_mask_1d_nb(arr, mask, value)

    return _set_by_mask_1d_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_nb(arr: tp.Array2d, mask: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """2-dim version of `set_by_mask_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_nb(arr, mask, value):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = value
        return out

    if not nb_enabled:
        return _set_by_mask_nb(arr, mask, value)

    return _set_by_mask_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_mult_1d_nb(arr: tp.Array1d, mask: tp.Array1d, values: tp.Array1d) -> tp.Array1d:
    """Set each element in one array to the corresponding element in another by boolean mask.

    `values` must be of the same shape as in the array."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_mult_1d_nb(arr, mask, values):
        out = arr.astype(dtype)
        out[mask] = values[mask]
        return out

    if not nb_enabled:
        return _set_by_mask_mult_1d_nb(arr, mask, values)

    return _set_by_mask_mult_1d_nb


@register_jitted(cache=True, is_generated_jit=True)
def set_by_mask_mult_nb(arr: tp.Array2d, mask: tp.Array2d, values: tp.Array2d) -> tp.Array2d:
    """2-dim version of `set_by_mask_mult_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(values.dtype)
    else:
        a_dtype = arr.dtype
        value_dtype = values.dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _set_by_mask_mult_nb(arr, mask, values):
        out = arr.astype(dtype)
        for col in range(arr.shape[1]):
            out[mask[:, col], col] = values[mask[:, col], col]
        return out

    if not nb_enabled:
        return _set_by_mask_mult_nb(arr, mask, values)

    return _set_by_mask_mult_nb


@register_jitted(cache=True)
def fillna_1d_nb(arr: tp.Array1d, value: tp.Scalar) -> tp.Array1d:
    """Replace NaNs with value.

    Numba equivalent to `pd.Series(arr).fillna(value)`."""
    return set_by_mask_1d_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        value=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True)
def fillna_nb(arr: tp.Array2d, value: tp.Scalar) -> tp.Array2d:
    """2-dim version of `fillna_1d_nb`."""
    return set_by_mask_nb(arr, np.isnan(arr), value)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def fbfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """Forward and backward fill NaN values.

    !!! note
        If there are no NaN values, will return `arr`."""
    need_fbfill = False
    for col in range(arr.shape[1]):
        for i in range(arr.shape[0]):
            if np.isnan(arr[i, col]):
                need_fbfill = True
                break
        if need_fbfill:
            break
    if not need_fbfill:
        return arr

    out = np.empty_like(arr)
    for col in prange(arr.shape[1]):
        last_valid = np.nan
        need_bfill = False
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i, col]):
                last_valid = arr[i, col]
            else:
                need_bfill = np.isnan(last_valid)
            out[i, col] = last_valid
        if need_bfill:
            last_valid = np.nan
            for i in range(arr.shape[0] - 1, -1, -1):
                if not np.isnan(arr[i, col]):
                    last_valid = arr[i, col]
                if np.isnan(out[i, col]):
                    out[i, col] = last_valid
    return out


@register_jitted(cache=True, is_generated_jit=True)
def bshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Shift backward by `n` positions.

    Numba equivalent to `pd.Series(arr).shift(n)`.

    !!! warning
        This operation looks ahead."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _bshift_1d_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        out[-n:] = fill_value
        out[:-n] = arr[n:]
        return out

    if not nb_enabled:
        return _bshift_1d_nb(arr, n, fill_value)

    return _bshift_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        n=None,
        fill_value=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, is_generated_jit=True)
def bshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """2-dim version of `bshift_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _bshift_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = bshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _bshift_nb(arr, n, fill_value)

    return _bshift_nb


@register_jitted(cache=True, is_generated_jit=True)
def fshift_1d_nb(arr: tp.Array1d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array1d:
    """Shift forward by `n` positions.

    Numba equivalent to `pd.Series(arr).shift(n)`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _fshift_1d_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        out[:n] = fill_value
        out[n:] = arr[:-n]
        return out

    if not nb_enabled:
        return _fshift_1d_nb(arr, n, fill_value)

    return _fshift_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        n=None,
        fill_value=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, is_generated_jit=True)
def fshift_nb(arr: tp.Array2d, n: int = 1, fill_value: tp.Scalar = np.nan) -> tp.Array2d:
    """2-dim version of `fshift_1d_nb`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        if isinstance(fill_value, Omitted):
            fill_value_dtype = np.asarray(fill_value.value).dtype
        else:
            fill_value_dtype = as_dtype(fill_value)
    else:
        a_dtype = arr.dtype
        fill_value_dtype = np.array(fill_value).dtype
    dtype = np.promote_types(a_dtype, fill_value_dtype)

    def _fshift_nb(arr, n, fill_value):
        out = np.empty_like(arr, dtype=dtype)
        for col in range(arr.shape[1]):
            out[:, col] = fshift_1d_nb(arr[:, col], n=n, fill_value=fill_value)
        return out

    if not nb_enabled:
        return _fshift_nb(arr, n, fill_value)

    return _fshift_nb


@register_jitted(cache=True)
def diff_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the 1-th discrete difference.

    Numba equivalent to `pd.Series(arr).diff()`."""
    out = np.empty_like(arr, dtype=np.float_)
    out[:n] = np.nan
    out[n:] = arr[n:] - arr[:-n]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        n=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def diff_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """2-dim version of `diff_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = diff_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def pct_change_1d_nb(arr: tp.Array1d, n: int = 1) -> tp.Array1d:
    """Compute the percentage change.

    Numba equivalent to `pd.Series(arr).pct_change()`."""
    out = np.empty_like(arr, dtype=np.float_)
    out[:n] = np.nan
    out[n:] = arr[n:] / arr[:-n] - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        n=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def pct_change_nb(arr: tp.Array2d, n: int = 1) -> tp.Array2d:
    """2-dim version of `pct_change_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = pct_change_1d_nb(arr[:, col], n=n)
    return out


@register_jitted(cache=True)
def bfill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaNs by propagating first valid observation backward.

    Numba equivalent to `pd.Series(arr).fillna(method='bfill')`.

    !!! warning
        This operation looks ahead."""
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[-1]
    for i in range(arr.shape[0] - 1, -1, -1):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def bfill_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `bfill_1d_nb`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = bfill_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def ffill_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Fill NaNs by propagating last valid observation forward.

    Numba equivalent to `pd.Series(arr).fillna(method='ffill')`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    lastval = arr[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            out[i] = lastval
        else:
            lastval = out[i] = arr[i]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def ffill_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `ffill_1d_nb`."""
    out = np.empty_like(arr, dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[:, col] = ffill_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, is_generated_jit=True, tags={'can_parallel'})
def nanprod_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanprod` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nanprod_nb(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nanprod(arr[:, col])
        return out

    if not nb_enabled:
        return _nanprod_nb(arr)

    return _nanprod_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, is_generated_jit=True, tags={'can_parallel'})
def nancumsum_nb(arr: tp.Array2d) -> tp.Array2d:
    """Numba equivalent of `np.nancumsum` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nancumsum_nb(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumsum(arr[:, col])
        return out

    if not nb_enabled:
        return _nancumsum_nb(arr)

    return _nancumsum_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, is_generated_jit=True, tags={'can_parallel'})
def nancumprod_nb(arr: tp.Array2d) -> tp.Array2d:
    """Numba equivalent of `np.nancumprod` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nancumprod_nb(arr):
        out = np.empty(arr.shape, dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = np.nancumprod(arr[:, col])
        return out

    if not nb_enabled:
        return _nancumprod_nb(arr)

    return _nancumprod_nb


@register_jitted(cache=True)
def nancnt_1d_nb(arr: tp.Array1d) -> int:
    """Compute count while ignoring NaNs and not allocating any arrays."""
    cnt = 0
    for i in range(arr.shape[0]):
        if not np.isnan(arr[i]):
            cnt += 1
    return cnt


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nancnt_nb(arr: tp.Array2d) -> tp.Array1d:
    """2-dim version of `nancnt_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[col] = nancnt_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, is_generated_jit=True, tags={'can_parallel'})
def nansum_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nansum` along axis 0."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
    else:
        a_dtype = arr.dtype
    dtype = np.promote_types(a_dtype, int)

    def _nansum_nb(arr):
        out = np.empty(arr.shape[1], dtype=dtype)
        for col in prange(arr.shape[1]):
            out[col] = np.nansum(arr[:, col])
        return out

    if not nb_enabled:
        return _nansum_nb(arr)

    return _nansum_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nanmin_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmin` along axis 0."""
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmin(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nanmax_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmax` along axis 0."""
    out = np.empty(arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmax(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nanmean_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmean` along axis 0."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmean(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nanmedian_nb(arr: tp.Array2d) -> tp.Array1d:
    """Numba equivalent of `np.nanmedian` along axis 0."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = np.nanmedian(arr[:, col])
    return out


@register_jitted(cache=True)
def nanpercentile_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Numba equivalent of `np.nanpercentile` that does not allocate any arrays.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.nanpercentile`,
        but still faster if used in rolling calculations, especially for `q` near 0 and 100."""
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    do_min = q < 50
    if not do_min:
        q = 100 - q
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth_float = q / 100 * (cnt - 1)
    if nth_float % 1 == 0:
        nth1 = nth2 = int(nth_float)
    else:
        nth1 = int(nth_float)
        nth2 = nth1 + 1
    found1 = np.nan
    found2 = np.nan
    k = 0
    if do_min:
        prev_val = -np.inf
    else:
        prev_val = np.inf
    while True:
        n_same = 0
        if do_min:
            curr_val = np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] > prev_val:
                        if arr[i] < curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        else:
            curr_val = -np.inf
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i]):
                    if arr[i] < prev_val:
                        if arr[i] > curr_val:
                            curr_val = arr[i]
                            n_same = 0
                        if arr[i] == curr_val:
                            n_same += 1
        prev_val = curr_val
        k += n_same
        if np.isnan(found1) and k >= nth1 + 1:
            found1 = curr_val
        if np.isnan(found2) and k >= nth2 + 1:
            found2 = curr_val
            break
    if found1 == found2:
        return found1
    factor = (nth_float - nth1) / (nth2 - nth1)
    return factor * (found2 - found1) + found1


@register_jitted(cache=True)
def nanpartition_mean_noarr_1d_nb(arr: tp.Array1d, q: float) -> float:
    """Average of `np.partition` that ignores NaN values and does not allocate any arrays.

    !!! note
        Has worst case time complexity of O(N^2), which makes it much slower than `np.partition`,
        but still faster if used in rolling calculations, especially for `q` near 0."""
    if q < 0:
        q = 0
    elif q > 100:
        q = 100
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    if cnt == 0:
        return np.nan
    nth = int(q / 100 * (cnt - 1))
    prev_val = -np.inf
    partition_sum = 0.
    partition_cnt = 0
    k = 0
    while True:
        n_same = 0
        curr_val = np.inf
        for i in range(arr.shape[0]):
            if not np.isnan(arr[i]):
                if arr[i] > prev_val:
                    if arr[i] < curr_val:
                        curr_val = arr[i]
                        n_same = 0
                    if arr[i] == curr_val:
                        n_same += 1
        if k + n_same >= nth + 1:
            partition_sum += (nth + 1 - k) * curr_val
            partition_cnt += nth + 1 - k
            break
        else:
            partition_sum += n_same * curr_val
            partition_cnt += n_same
        prev_val = curr_val
        k += n_same
    return partition_sum / partition_cnt


@register_jitted(cache=True)
def nanvar_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.nanvar` that does not allocate any arrays."""
    cnt = arr.shape[0]
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            cnt -= 1
    rcount = max(cnt - ddof, 0)
    if rcount == 0:
        return np.nan
    out = 0.
    a_mean = np.nanmean(arr)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            out += abs(arr[i] - a_mean) ** 2
    return out / rcount


@register_jitted(cache=True)
def nanstd_1d_nb(arr: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.nanstd`."""
    return np.sqrt(nanvar_1d_nb(arr, ddof=ddof))


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nanstd_nb(arr: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `nanstd_1d_nb`."""
    out = np.empty(arr.shape[1], dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[col] = nanstd_1d_nb(arr[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancov_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, ddof: int = 0) -> float:
    """Numba equivalent of `np.cov` that ignores NaN values."""
    arr1_sum = 0.
    arr2_sum = 0.
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            k += 1
    if k - ddof <= 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    num = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            num += (arr1[i] - arr1_mean) * (arr2[i] - arr2_mean)
    return num / (k - ddof)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr1', axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nancov_nb(arr1: tp.Array2d, arr2: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `nancov_1d_nb`."""
    out = np.empty(arr1.shape[1], dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancov_1d_nb(arr1[:, col], arr2[:, col], ddof=ddof)
    return out


@register_jitted(cache=True)
def nancorr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d) -> float:
    """Numba equivalent of `np.corrcoef` that ignores NaN values.

    Numerically stable."""
    arr1_sum = 0.
    arr2_sum = 0.
    k = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            arr1_sum += arr1[i]
            arr2_sum += arr2[i]
            k += 1
    if k == 0:
        return np.nan
    arr1_mean = arr1_sum / k
    arr2_mean = arr2_sum / k
    num = 0
    denom1 = 0
    denom2 = 0
    for i in range(arr1.shape[0]):
        if not np.isnan(arr1[i]) and not np.isnan(arr2[i]):
            num += (arr1[i] - arr1_mean) * (arr2[i] - arr2_mean)
            denom1 += (arr1[i] - arr1_mean) ** 2
            denom2 += (arr2[i] - arr2_mean) ** 2
    return num / np.sqrt(denom1 * denom2)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr1', axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nancorr_nb(arr1: tp.Array2d, arr2: tp.Array2d) -> tp.Array1d:
    """2-dim version of `nancorr_1d_nb`."""
    out = np.empty(arr1.shape[1], dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[col] = nancorr_1d_nb(arr1[:, col], arr2[:, col])
    return out


@register_jitted(cache=True)
def rank_1d_nb(arr: tp.Array1d, argsorted: tp.Optional[tp.Array1d] = None, pct: bool = False) -> tp.Array1d:
    """Compute numerical data ranks.

    Numba equivalent to `pd.Series(arr).rank(pct=pct)`."""
    if argsorted is None:
        argsorted = np.argsort(arr)
    out = np.empty_like(arr, dtype=np.float_)
    rank_sum = 0
    rank_cnt = 0
    nan_cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nan_cnt += 1
    if nan_cnt == arr.shape[0]:
        out[:] = np.nan
        return out
    valid_cnt = out.shape[0] - nan_cnt
    for i in range(argsorted.shape[0]):
        rank = i + 1
        if np.isnan(arr[argsorted[i]]):
            out[argsorted[i]] = np.nan
        elif i < out.shape[0] - 1 and arr[argsorted[i]] == arr[argsorted[i + 1]]:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
        elif rank_sum > 0:
            rank_sum += rank
            rank_cnt += 1
            if pct:
                v = rank_sum / rank_cnt / valid_cnt
            else:
                v = rank_sum / rank_cnt
            out[argsorted[i - rank_cnt + 1:i + 1]] = v
            rank_sum = 0
            rank_cnt = 0
        else:
            if pct:
                v = rank / valid_cnt
            else:
                v = rank
            out[argsorted[i]] = v
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        argsorted=ch.ArraySlicer(axis=1),
        pct=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rank_nb(arr: tp.Array2d, argsorted: tp.Optional[tp.Array2d] = None, pct: bool = False) -> tp.Array2d:
    """2-dim version of `rank_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        if argsorted is None:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=None, pct=pct)
        else:
            out[:, col] = rank_1d_nb(arr[:, col], argsorted=argsorted[:, col], pct=pct)
    return out


# ############# Rolling functions ############# #


@register_jitted(cache=True)
def rolling_sum_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling sum.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).sum()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + arr[i]
        if i < window:
            window_len = i + 1 - nancnt
            window_cumsum = cumsum
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum = cumsum - arr[i - window]
            window_len = window - nancnt
            window_cumsum = cumsum
        if window_len < minp:
            out[i] = np.nan
        else:
            out[i] = window_cumsum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_sum_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_sum_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_sum_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_prod_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling product.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).apply(np.prod)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumprod = 1
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        else:
            cumprod = cumprod * arr[i]
        if i < window:
            window_len = i + 1 - nancnt
            window_cumprod = cumprod
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            else:
                cumprod = cumprod / arr[i - window]
            window_len = window - nancnt
            window_cumprod = cumprod
        if window_len < minp:
            out[i] = np.nan
        else:
            out[i] = window_cumprod
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_prod_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_prod_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_prod_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_min_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling min.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_min_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_min_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_max_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling max.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_max_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_max_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_argmin_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None,
                         local: bool = False) -> tp.Array1d:
    """Compute rolling min index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        if local:
            mini = 0
        else:
            mini = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
                if local:
                    mini = k
                else:
                    mini = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = mini
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        local=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_argmin_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None,
                      local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmin_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmin_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_argmax_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None,
                         local: bool = False) -> tp.Array1d:
    """Compute rolling max index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        if local:
            maxi = 0
        else:
            maxi = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
                if local:
                    maxi = k
                else:
                    maxi = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = maxi
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        local=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_argmax_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None,
                      local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmax_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmax_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling mean.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + arr[i]
        if i < window:
            window_len = i + 1 - nancnt
            window_cumsum = cumsum
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum = cumsum - arr[i - window]
            window_len = window - nancnt
            window_cumsum = cumsum
        if window_len < minp:
            out[i] = np.nan
        else:
            out[i] = window_cumsum / window_len
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_std_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array1d:
    """Compute rolling standard deviation.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).std(ddof=ddof)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0
    cumsum_sq = 0
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + arr[i]
            cumsum_sq = cumsum_sq + arr[i] ** 2
        if i < window:
            window_len = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum = cumsum - arr[i - window]
                cumsum_sq = cumsum_sq - arr[i - window] ** 2
            window_len = window - nancnt
        if window_len < minp or window_len == ddof:
            out[i] = np.nan
        else:
            mean = cumsum / window_len
            out[i] = np.sqrt(np.abs(cumsum_sq - 2 * cumsum * mean + window_len * mean ** 2) / (window_len - ddof))
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        ddof=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_std_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_std_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def wm_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute weighted moving average."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0
    wcumsum = 0
    nancnt = 0
    for i in range(arr.shape[0]):
        if i >= window and not np.isnan(arr[i - window]):
            wcumsum = wcumsum - cumsum
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + arr[i]
        if i < window:
            window_len = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum = cumsum - arr[i - window]
            window_len = window - nancnt
        if not np.isnan(arr[i]):
            wcumsum = wcumsum + arr[i] * window_len
        if window_len < minp:
            out[i] = np.nan
        else:
            out[i] = wcumsum * 2 / (window_len + 1) / window_len
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def wm_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `wm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wm_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def ewm_mean_1d_nb(arr: tp.Array1d, span: int, minp: int = 0, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving average.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp, adjust=adjust).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(arr)
    out = np.empty(N, dtype=np.float_)
    if N == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha
    weighted_avg = arr[0]
    is_observation = (weighted_avg == weighted_avg)
    nobs = int(is_observation)
    out[0] = weighted_avg if (nobs >= minp) else np.nan
    old_wt = 1.

    for i in range(1, N):
        cur = arr[i]
        is_observation = (cur == cur)
        nobs += is_observation
        if weighted_avg == weighted_avg:
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt)
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.
        elif is_observation:
            weighted_avg = cur
        out[i] = weighted_avg if (nobs >= minp) else np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        span=None,
        minp=None,
        adjust=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def ewm_mean_nb(arr: tp.Array2d, span: int, minp: int = 0, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_mean_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def ewm_std_1d_nb(arr: tp.Array1d, span: int, minp: int = 0, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving standard deviation.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp).std()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(arr)
    out = np.empty(N, dtype=np.float_)
    if N == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha
    mean_x = arr[0]
    mean_y = arr[0]
    is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
    nobs = int(is_observation)
    if not is_observation:
        mean_x = np.nan
        mean_y = np.nan
    out[0] = np.nan
    cov = 0.
    sum_wt = 1.
    sum_wt2 = 1.
    old_wt = 1.

    for i in range(1, N):
        cur_x = arr[i]
        cur_y = arr[i]
        is_observation = ((cur_x == cur_x) and (cur_y == cur_y))
        nobs += is_observation
        if mean_x == mean_x:
            sum_wt *= old_wt_factor
            sum_wt2 *= (old_wt_factor * old_wt_factor)
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) +
                              (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) +
                              (new_wt * cur_y)) / (old_wt + new_wt)
                cov = ((old_wt * (cov + ((old_mean_x - mean_x) *
                                         (old_mean_y - mean_y)))) +
                       (new_wt * ((cur_x - mean_x) *
                                  (cur_y - mean_y)))) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += (new_wt * new_wt)
                old_wt += new_wt
                if not adjust:
                    sum_wt /= old_wt
                    sum_wt2 /= (old_wt * old_wt)
                    old_wt = 1.
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y

        if nobs >= minp:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if denominator > 0.:
                out[i] = ((numerator / denominator) * cov)
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan
    return np.sqrt(out)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        span=None,
        minp=None,
        adjust=None,
        ddof=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def ewm_std_nb(arr: tp.Array2d, span: int, minp: int = 0, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_std_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def rolling_cov_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d,
                      window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array1d:
    """Compute rolling covariance.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).cov(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0
    cumsum2 = 0
    cumsum_prod = 0
    nancnt = 0
    for i in range(arr1.shape[0]):
        if np.isnan(arr1[i]) or np.isnan(arr2[i]):
            nancnt = nancnt + 1
        else:
            cumsum1 = cumsum1 + arr1[i]
            cumsum2 = cumsum2 + arr2[i]
            cumsum_prod = cumsum_prod + arr1[i] * arr2[i]
        if i < window:
            window_len = i + 1 - nancnt
        else:
            if np.isnan(arr1[i - window]) or np.isnan(arr2[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum1 = cumsum1 - arr1[i - window]
                cumsum2 = cumsum2 - arr2[i - window]
                cumsum_prod = cumsum_prod - arr1[i - window] * arr2[i - window]
            window_len = window - nancnt
        if window_len < minp or window_len == ddof:
            out[i] = np.nan
        else:
            window_prod_mean = cumsum_prod / (window_len - ddof)
            window_mean1 = cumsum1 / window_len
            window_mean2 = cumsum2 / window_len
            window_mean_prod = window_mean1 * window_mean2 * window_len / (window_len - ddof)
            out[i] = window_prod_mean - window_mean_prod
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr1', axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        ddof=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_cov_nb(arr1: tp.Array2d, arr2: tp.Array2d, window: int, minp: int = 0, ddof: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_cov_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_cov_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_corr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d,
                       window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling correlation coefficient.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).corr(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0
    cumsum2 = 0
    cumsum_sq1 = 0
    cumsum_sq2 = 0
    cumsum_prod = 0
    nancnt = 0
    for i in range(arr1.shape[0]):
        if np.isnan(arr1[i]) or np.isnan(arr2[i]):
            nancnt = nancnt + 1
        else:
            cumsum1 = cumsum1 + arr1[i]
            cumsum2 = cumsum2 + arr2[i]
            cumsum_sq1 = cumsum_sq1 + arr1[i] ** 2
            cumsum_sq2 = cumsum_sq2 + arr2[i] ** 2
            cumsum_prod = cumsum_prod + arr1[i] * arr2[i]
        if i < window:
            window_len = i + 1 - nancnt
        else:
            if np.isnan(arr1[i - window]) or np.isnan(arr2[i - window]):
                nancnt = nancnt - 1
            else:
                cumsum1 = cumsum1 - arr1[i - window]
                cumsum2 = cumsum2 - arr2[i - window]
                cumsum_sq1 = cumsum_sq1 - arr1[i - window] ** 2
                cumsum_sq2 = cumsum_sq2 - arr2[i - window] ** 2
                cumsum_prod = cumsum_prod - arr1[i - window] * arr2[i - window]
            window_len = window - nancnt
        if window_len < minp:
            out[i] = np.nan
        else:
            nom = window_len * cumsum_prod - cumsum1 * cumsum2
            denom1 = np.sqrt(window_len * cumsum_sq1 - cumsum1 ** 2)
            denom2 = np.sqrt(window_len * cumsum_sq2 - cumsum2 ** 2)
            denom = denom1 * denom2
            if denom == 0:
                out[i] = np.nan
            else:
                out[i] = nom / denom
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr1', axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        window=None,
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_corr_nb(arr1: tp.Array2d, arr2: tp.Array2d, window: int, minp: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_corr_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_corr_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_rank_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int], pct: bool = False) -> tp.Array1d:
    """Rolling version of `rank_1d_nb`."""
    if minp is None:
        minp = window
    out = np.empty_like(arr, dtype=np.float_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            valid_cnt = window - nancnt
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            window_a = arr[from_i:to_i]
            out[i] = rank_1d_nb(window_a, pct=pct)[-1]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        pct=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def rolling_rank_nb(arr: tp.Array2d, window: int, minp: int = 0, pct: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_rank_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_rank_1d_nb(arr[:, col], window, minp=minp, pct=pct)
    return out


# ############# Expanding functions ############# #


@register_jitted(cache=True)
def expanding_min_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding min.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).min()`."""
    out = np.empty_like(arr, dtype=np.float_)
    minv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(minv) or arr[i] < minv:
            minv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def expanding_min_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_min_1d_nb(arr[:, col], minp=minp)
    return out


@register_jitted(cache=True)
def expanding_max_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding max.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).max()`."""
    out = np.empty_like(arr, dtype=np.float_)
    maxv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(maxv) or arr[i] > maxv:
            maxv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        minp=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def expanding_max_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_max_1d_nb(arr[:, col], minp=minp)
    return out


# ############# Map, apply, and reduce ############# #


@register_jitted
def map_1d_nb(arr: tp.Array1d, map_func_nb: tp.MapFunc, *args) -> tp.Array1d:
    """Map elements element-wise using `map_func_nb`.

    `map_func_nb` must accept the element and `*args`. Must return a single value."""
    i_0_out = map_func_nb(arr[0], *args)
    out = np.empty_like(arr, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, arr.shape[0]):
        out[i] = map_func_nb(arr[i], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        map_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def map_nb(arr: tp.Array2d, map_func_nb: tp.MapFunc, *args) -> tp.Array2d:
    """2-dim version of `map_1d_nb`."""
    col_0_out = map_1d_nb(arr[:, 0], map_func_nb, *args)
    out = np.empty_like(arr, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_nb(arr[:, col], map_func_nb, *args)
    return out


@register_jitted
def map_1d_meta_nb(n: int, col: int, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array1d:
    """Meta version of `map_1d_nb`.

    `map_func_nb` must accept the row index, the column index, and `*args`.
    Must return a single value."""
    i_0_out = map_func_nb(0, col, *args)
    out = np.empty(n, dtype=np.asarray(i_0_out).dtype)
    out[0] = i_0_out
    for i in range(1, n):
        out[i] = map_func_nb(i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        map_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def map_meta_nb(target_shape: tp.Shape, map_func_nb: tp.MapMetaFunc, *args) -> tp.Array2d:
    """2-dim version of `map_1d_meta_nb`."""
    col_0_out = map_1d_meta_nb(target_shape[0], 0, map_func_nb, *args)
    out = np.empty(target_shape, dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, out.shape[1]):
        out[:, col] = map_1d_meta_nb(target_shape[0], col, map_func_nb, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """Apply function on each column of an object.

    `apply_func_nb` must accept the array and `*args`.
    Must return a single value or an array of shape `a.shape[1]`."""
    col_0_out = apply_func_nb(arr[:, 0], *args)
    out = np.empty_like(arr, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = apply_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Meta version of `apply_nb` that prepends the column index to the arguments of `apply_func_nb`."""
    col_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(col_0_out).dtype)
    out[:, 0] = col_0_out
    for col in prange(1, target_shape[1]):
        out[:, col] = apply_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=0),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=0),
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.row_stack
)
@register_jitted(tags={'can_parallel'})
def row_apply_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """`apply_nb` but applied on rows rather than columns."""
    row_0_out = apply_func_nb(arr[0, :], *args)
    out = np.empty_like(arr, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, arr.shape[0]):
        out[i, :] = apply_func_nb(arr[i, :], *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=0),
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.row_stack
)
@register_jitted(tags={'can_parallel'})
def row_apply_meta_nb(target_shape: tp.Shape, apply_func_nb: tp.ApplyMetaFunc, *args) -> tp.Array2d:
    """Meta version of `row_apply_nb` that prepends the row index to the arguments of `apply_func_nb`."""
    row_0_out = apply_func_nb(0, *args)
    out = np.empty(target_shape, dtype=np.asarray(row_0_out).dtype)
    out[0, :] = row_0_out
    for i in prange(1, target_shape[0]):
        out[i, :] = apply_func_nb(i, *args)
    return out


@register_jitted
def rolling_apply_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int],
                        apply_func_nb: tp.ApplyFunc, *args) -> tp.Array1d:
    """Provide rolling window calculations.

    `apply_func_nb` must accept the array and `*args`. Must return a single value."""
    if minp is None:
        minp = window
    out = np.empty_like(arr, dtype=np.float_)
    nancnt_arr = np.empty(arr.shape[0], dtype=np.int_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            valid_cnt = window - (nancnt - nancnt_arr[i - window])
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            window_a = arr[from_i:to_i]
            out[i] = apply_func_nb(window_a, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def rolling_apply_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int],
                     apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """2-dim version of `rolling_apply_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_apply_1d_nb(arr[:, col], window, minp, apply_func_nb, *args)
    return out


@register_jitted
def rolling_apply_1d_meta_nb(n: int, col: int, window: int, minp: tp.Optional[int],
                             apply_func_nb: tp.RollApplyMetaFunc, *args) -> tp.Array1d:
    """Meta version of `rolling_apply_1d_nb`.

    `apply_func_nb` must accept the start row index, the end row index, the column, and `*args`.
    Must return a single value."""
    if minp is None:
        minp = window
    out = np.empty(n, dtype=np.float_)
    for i in range(n):
        valid_cnt = min(i + 1, window)
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            out[i] = apply_func_nb(from_i, to_i, col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        window=None,
        minp=None,
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def rolling_apply_meta_nb(target_shape: tp.Shape, window: int, minp: tp.Optional[int],
                          apply_func_nb: tp.RollApplyMetaFunc, *args) -> tp.Array2d:
    """2-dim version of `rolling_apply_1d_meta_nb`."""
    out = np.empty(target_shape, dtype=np.float_)
    for col in prange(target_shape[1]):
        out[:, col] = rolling_apply_1d_meta_nb(target_shape[0], col, window, minp, apply_func_nb, *args)
    return out


@register_jitted
def groupby_apply_1d_nb(arr: tp.Array1d, group_map: tp.GroupMap,
                        apply_func_nb: tp.ApplyFunc, *args) -> tp.Array1d:
    """Provide group-by calculations.

    `apply_func_nb` must accept the array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = apply_func_nb(arr[group_0_idxs], *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx:start_idx + group_len]
        out[group] = apply_func_nb(arr[idxs], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        group_map=None,
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def groupby_apply_nb(arr: tp.Array2d, group_map: tp.GroupMap,
                     apply_func_nb: tp.ApplyFunc, *args) -> tp.Array2d:
    """2-dim version of `groupby_apply_1d_nb`."""
    col_0_out = groupby_apply_1d_nb(arr[:, 0], group_map, apply_func_nb, *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = groupby_apply_1d_nb(arr[:, col], group_map, apply_func_nb, *args)
    return out


@register_jitted
def groupby_apply_1d_meta_nb(col: int, group_map: tp.GroupMap,
                             apply_func_nb: tp.GroupByApplyMetaFunc, *args) -> tp.Array1d:
    """Meta version of `groupby_apply_1d_nb`.

    `apply_func_nb` must accept the array of indices in the group, the group index, the column index,
    and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = apply_func_nb(group_0_idxs, 0, col, *args)
    out = np.empty(group_lens.shape[0], dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in range(1, group_lens.shape[0]):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        idxs = group_idxs[start_idx:start_idx + group_len]
        out[group] = apply_func_nb(idxs, group, col, *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query='n_cols'),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        group_map=None,
        apply_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def groupby_apply_meta_nb(n_cols: int, group_map: tp.GroupMap,
                          apply_func_nb: tp.GroupByApplyMetaFunc, *args) -> tp.Array2d:
    """2-dim version of `groupby_apply_1d_meta_nb`."""
    col_0_out = groupby_apply_1d_meta_nb(0, group_map, apply_func_nb, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = groupby_apply_1d_meta_nb(col, group_map, apply_func_nb, *args)
    return out


@register_jitted
def apply_and_reduce_1d_nb(arr: tp.Array1d, apply_func_nb: tp.ApplyFunc, apply_args: tuple,
                           reduce_func_nb: tp.ReduceFunc, reduce_args: tuple) -> tp.Scalar:
    """Apply `apply_func_nb` and reduce into a single value using `reduce_func_nb`.

    `apply_func_nb` must accept the array and `*apply_args`.
    Must return an array.

    `reduce_func_nb` must accept the array of results from `apply_func_nb` and `*reduce_args`.
    Must return a single value."""
    temp = apply_func_nb(arr, *apply_args)
    return reduce_func_nb(temp, *reduce_args)


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def apply_and_reduce_nb(arr: tp.Array2d, apply_func_nb: tp.ApplyFunc, apply_args: tuple,
                        reduce_func_nb: tp.ReduceFunc, reduce_args: tuple) -> tp.Array1d:
    """2-dim version of `apply_and_reduce_1d_nb`."""
    col_0_out = apply_and_reduce_1d_nb(arr[:, 0], apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = apply_and_reduce_1d_nb(arr[:, col], apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    return out


@register_jitted
def apply_and_reduce_1d_meta_nb(col: int, apply_func_nb: tp.ApplyMetaFunc, apply_args: tuple,
                                reduce_func_nb: tp.ReduceMetaFunc, reduce_args: tuple) -> tp.Scalar:
    """Meta version of `apply_and_reduce_1d_nb`.

    `apply_func_nb` must accept the column index, the array, and `*apply_args`.
    Must return an array.

    `reduce_func_nb` must accept the column index, the array of results from `apply_func_nb`, and `*reduce_args`.
    Must return a single value."""
    temp = apply_func_nb(col, *apply_args)
    return reduce_func_nb(col, temp, *reduce_args)


@register_chunkable(
    size=ch.ArgSizer(arg_query='n_cols'),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        apply_func_nb=None,
        apply_args=ch.ArgsTaker(),
        reduce_func_nb=None,
        reduce_args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def apply_and_reduce_meta_nb(n_cols: int, apply_func_nb: tp.ApplyMetaFunc, apply_args: tuple,
                             reduce_func_nb: tp.ReduceMetaFunc, reduce_args: tuple) -> tp.Array1d:
    """2-dim version of `apply_and_reduce_1d_meta_nb`."""
    col_0_out = apply_and_reduce_1d_meta_nb(0, apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = apply_and_reduce_1d_meta_nb(col, apply_func_nb, apply_args, reduce_func_nb, reduce_args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def reduce_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceFunc, *args) -> tp.Array1d:
    """Reduce each column into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty(arr.shape[1], dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query='n_cols'),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def reduce_meta_nb(n_cols: int, reduce_func_nb: tp.ReduceMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_nb`.

    `reduce_func_nb` must accept the column index and `*args`. Must return a single value."""
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty(n_cols, dtype=np.asarray(col_0_out).dtype)
    out[0] = col_0_out
    for col in prange(1, n_cols):
        out[col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def reduce_to_array_nb(arr: tp.Array2d, reduce_func_nb: tp.ReduceToArrayFunc, *args) -> tp.Array2d:
    """Same as `reduce_nb` but `reduce_func_nb` must return an array."""
    col_0_out = reduce_func_nb(arr[:, 0], *args)
    out = np.empty((col_0_out.shape[0], arr.shape[1]), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, arr.shape[1]):
        out[:, col] = reduce_func_nb(arr[:, col], *args)
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query='n_cols'),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def reduce_to_array_meta_nb(n_cols: int, reduce_func_nb: tp.ReduceToArrayMetaFunc, *args) -> tp.Array2d:
    """Same as `reduce_meta_nb` but `reduce_func_nb` must return an array."""
    col_0_out = reduce_func_nb(0, *args)
    out = np.empty((col_0_out.shape[0], n_cols), dtype=col_0_out.dtype)
    out[:, 0] = col_0_out
    for col in prange(1, n_cols):
        out[:, col] = reduce_func_nb(col, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def reduce_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap,
                      reduce_func_nb: tp.ReduceGroupedFunc, *args) -> tp.Array1d:
    """Reduce each group of columns into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept the 2-dim array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        out[group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def reduce_grouped_meta_nb(group_map: tp.GroupMap, reduce_func_nb: tp.ReduceGroupedMetaFunc, *args) -> tp.Array1d:
    """Meta version of `reduce_grouped_nb`.

    `reduce_func_nb` must accept the column indices of the group, the group index, and `*args`.
    Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        out[group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def flatten_forder_nb(arr: tp.Array2d) -> tp.Array1d:
    """Flatten the array in F order."""
    out = np.empty(arr.shape[0] * arr.shape[1], dtype=arr.dtype)
    for col in prange(arr.shape[1]):
        out[col * arr.shape[0]:(col + 1) * arr.shape[0]] = arr[:, col]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.concat
)
@register_jitted(tags={'can_parallel'})
def reduce_flat_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool,
                           reduce_func_nb: tp.ReduceToArrayFunc, *args) -> tp.Array1d:
    """Same as `reduce_grouped_nb` but passes flattened array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty(len(group_lens), dtype=np.asarray(group_0_out).dtype)
    out[0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        if in_c_order:
            out[group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def reduce_grouped_to_array_nb(arr: tp.Array2d, group_map: tp.GroupMap,
                               reduce_func_nb: tp.ReduceGroupedToArrayFunc, *args) -> tp.Array2d:
    """Same as `reduce_grouped_nb` but `reduce_func_nb` must return an array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(arr[:, group_0_idxs], *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        out[:, group] = reduce_func_nb(arr[:, col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        group_map=base_ch.GroupMapSlicer(),
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def reduce_grouped_to_array_meta_nb(group_map: tp.GroupMap,
                                    reduce_func_nb: tp.ReduceGroupedToArrayMetaFunc, *args) -> tp.Array2d:
    """Same as `reduce_grouped_meta_nb` but `reduce_func_nb` must return an array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_0_out = reduce_func_nb(group_0_idxs, 0, *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        out[:, group] = reduce_func_nb(col_idxs, group, *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        in_c_order=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def reduce_flat_grouped_to_array_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool,
                                    reduce_func_nb: tp.ReduceToArrayFunc, *args) -> tp.Array2d:
    """Same as `reduce_grouped_to_array_nb` but passes flattened array."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    if in_c_order:
        group_0_out = reduce_func_nb(arr[:, group_0_idxs].flatten(), *args)
    else:
        group_0_out = reduce_func_nb(flatten_forder_nb(arr[:, group_0_idxs]), *args)
    out = np.empty((group_0_out.shape[0], len(group_lens)), dtype=group_0_out.dtype)
    out[:, 0] = group_0_out

    for group in prange(1, len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        if in_c_order:
            out[:, group] = reduce_func_nb(arr[:, col_idxs].flatten(), *args)
        else:
            out[:, group] = reduce_func_nb(flatten_forder_nb(arr[:, col_idxs]), *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer(),
        squeeze_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def squeeze_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap,
                       squeeze_func_nb: tp.ReduceFunc, *args) -> tp.Array2d:
    """Squeeze each group of columns into a single column using `squeeze_func_nb`.

    `squeeze_func_nb` must accept index the array and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(arr[0][group_0_idxs], *args)
    out = np.empty((arr.shape[0], len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for i in range(arr.shape[0]):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(arr[i][col_idxs], *args)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        n_rows=None,
        group_map=base_ch.GroupMapSlicer(),
        squeeze_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def squeeze_grouped_meta_nb(n_rows: int, group_map: tp.GroupMap,
                            squeeze_func_nb: tp.GroupSqueezeMetaFunc, *args) -> tp.Array2d:
    """Meta version of `squeeze_grouped_nb`.

    `squeeze_func_nb` must accept the row index, the column indices of the group,
    the group index, and `*args`. Must return a single value."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    group_0_idxs = group_idxs[group_start_idxs[0]:group_start_idxs[0] + group_lens[0]]
    group_i_0_out = squeeze_func_nb(0, group_0_idxs, 0, *args)
    out = np.empty((n_rows, len(group_lens)), dtype=np.asarray(group_i_0_out).dtype)
    out[0, 0] = group_i_0_out

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for i in range(n_rows):
            if group == 0 and i == 0:
                continue
            out[i, group] = squeeze_func_nb(i, col_idxs, group, *args)
    return out


# ############# Flattening ############# #

@register_jitted(cache=True)
def flatten_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool) -> tp.Array2d:
    """Flatten each group of columns."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((arr.shape[0] * np.max(group_lens), len(group_lens)), np.nan, dtype=np.float_)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0]:(k + 1) * arr.shape[0], group] = arr[:, col]
    return out


@register_jitted(cache=True)
def flatten_uniform_grouped_nb(arr: tp.Array2d, group_map: tp.GroupMap, in_c_order: bool) -> tp.Array2d:
    """Flatten each group of columns of the same length."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty((arr.shape[0] * np.max(group_lens), len(group_lens)), dtype=arr.dtype)
    max_len = np.max(group_lens)

    for group in range(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            if in_c_order:
                out[k::max_len, group] = arr[:, col]
            else:
                out[k * arr.shape[0]:(k + 1) * arr.shape[0], group] = arr[:, col]
    return out


# ############# Reducers ############# #


@register_jitted(cache=True)
def nth_reduce_nb(arr: tp.Array1d, n: int) -> float:
    """Compute n-th element."""
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    return arr[n]


@register_jitted(cache=True)
def nth_index_reduce_nb(arr: tp.Array1d, n: int) -> int:
    """Compute index of n-th element."""
    if (n < 0 and abs(n) > arr.shape[0]) or n >= arr.shape[0]:
        raise ValueError("index is out of bounds")
    if n >= 0:
        return n
    return arr.shape[0] + n


@register_jitted(cache=True)
def any_reduce_nb(arr: tp.Array1d) -> bool:
    """Compute whether any of the elements are True."""
    return np.any(arr)


@register_jitted(cache=True)
def all_reduce_nb(arr: tp.Array1d) -> bool:
    """Compute whether all of the elements are True."""
    return np.all(arr)


@register_jitted(cache=True)
def min_reduce_nb(arr: tp.Array1d) -> float:
    """Compute min (ignores NaNs)."""
    return np.nanmin(arr)


@register_jitted(cache=True)
def max_reduce_nb(arr: tp.Array1d) -> float:
    """Compute max (ignores NaNs)."""
    return np.nanmax(arr)


@register_jitted(cache=True)
def mean_reduce_nb(arr: tp.Array1d) -> float:
    """Compute mean (ignores NaNs)."""
    return np.nanmean(arr)


@register_jitted(cache=True)
def median_reduce_nb(arr: tp.Array1d) -> float:
    """Compute median (ignores NaNs)."""
    return np.nanmedian(arr)


@register_jitted(cache=True)
def std_reduce_nb(arr: tp.Array1d, ddof) -> float:
    """Compute std (ignores NaNs)."""
    return nanstd_1d_nb(arr, ddof=ddof)


@register_jitted(cache=True)
def sum_reduce_nb(arr: tp.Array1d) -> float:
    """Compute sum (ignores NaNs)."""
    return np.nansum(arr)


@register_jitted(cache=True)
def count_reduce_nb(arr: tp.Array1d) -> int:
    """Compute count (ignores NaNs)."""
    return np.sum(~np.isnan(arr))


@register_jitted(cache=True)
def argmin_reduce_nb(arr: tp.Array1d) -> int:
    """Compute position of min."""
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = np.inf
    return np.argmin(arr)


@register_jitted(cache=True)
def argmax_reduce_nb(arr: tp.Array1d) -> int:
    """Compute position of max."""
    arr = np.copy(arr)
    mask = np.isnan(arr)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    arr[mask] = -np.inf
    return np.argmax(arr)


@register_jitted(cache=True)
def describe_reduce_nb(arr: tp.Array1d, perc: tp.Array1d, ddof: int) -> tp.Array1d:
    """Compute descriptive statistics (ignores NaNs).

    Numba equivalent to `pd.Series(arr).describe(perc)`."""
    arr = arr[~np.isnan(arr)]
    out = np.empty(5 + len(perc), dtype=np.float_)
    out[0] = len(arr)
    if len(arr) > 0:
        out[1] = np.mean(arr)
        out[2] = nanstd_1d_nb(arr, ddof=ddof)
        out[3] = np.min(arr)
        out[4:-1] = np.percentile(arr, perc * 100)
        out[4 + len(perc)] = np.max(arr)
    else:
        out[1:] = np.nan
    return out


@register_jitted(cache=True)
def cov_reduce_grouped_meta_nb(group_idxs: tp.GroupIdxs, group: int, arr1: tp.Array2d,
                               arr2: tp.Array2d, ddof: int) -> float:
    """Compute correlation coefficient (ignores NaNs)."""
    return nancov_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten(), ddof=ddof)


@register_jitted(cache=True)
def corr_reduce_grouped_meta_nb(group_idxs: tp.GroupIdxs, group: int, arr1: tp.Array2d,
                                arr2: tp.Array2d) -> float:
    """Compute correlation coefficient (ignores NaNs)."""
    return nancorr_1d_nb(arr1[:, group_idxs].flatten(), arr2[:, group_idxs].flatten())


# ############# Value counts ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query='codes', axis=1),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        n_uniques=None,
        group_map=base_ch.GroupMapSlicer()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def value_counts_nb(codes: tp.Array2d, n_uniques: int, group_map: tp.GroupMap) -> tp.Array2d:
    """Compute value counts per column/group."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.full((n_uniques, group_lens.shape[0]), 0, dtype=np.int_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for k in range(group_len):
            col = col_idxs[k]
            for i in range(codes.shape[0]):
                out[codes[i, col], group] += 1
    return out


@register_jitted(cache=True)
def value_counts_1d_nb(codes: tp.Array1d, n_uniques: int) -> tp.Array1d:
    """Compute value counts."""
    out = np.full(n_uniques, 0, dtype=np.int_)

    for i in range(codes.shape[0]):
        out[codes[i]] += 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='codes', axis=0),
    arg_take_spec=dict(
        codes=ch.ArraySlicer(axis=0),
        n_uniques=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def value_counts_per_row_nb(codes: tp.Array2d, n_uniques: int) -> tp.Array2d:
    """Compute value counts per row."""
    out = np.empty((n_uniques, codes.shape[0]), dtype=np.int_)

    for i in prange(codes.shape[0]):
        out[:, i] = value_counts_1d_nb(codes[i, :], n_uniques)
    return out


# ############# Repartitioning ############# #


@register_jitted(cache=True)
def repartition_nb(arr: tp.Array2d, counts: tp.Array1d) -> tp.Array1d:
    """Repartition a 2-dimensional array into a 1-dimensional by removing empty elements."""
    out = np.empty(np.sum(counts), dtype=arr.dtype)
    j = 0
    for col in range(counts.shape[0]):
        out[j:j + counts[col]] = arr[:counts[col], col]
        j += counts[col]
    return out


# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        gap_value=None
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def get_ranges_nb(arr: tp.Array2d, gap_value: tp.Scalar) -> tp.RecordArray:
    """Fill range records between gaps.

    Usage:
        * Find ranges in time series:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_ranges_nb

        >>> a = np.asarray([
        ...     [np.nan, np.nan, np.nan, np.nan],
        ...     [     2, np.nan, np.nan, np.nan],
        ...     [     3,      3, np.nan, np.nan],
        ...     [np.nan,      4,      4, np.nan],
        ...     [     5, np.nan,      5,      5],
        ...     [     6,      6, np.nan,      6]
        ... ])
        >>> records = get_ranges_nb(a, np.nan)

        >>> pd.DataFrame.from_records(records)
           id  col  start_idx  end_idx  status
        0   0    0          1        3       1
        1   1    0          4        5       0
        2   0    1          2        4       1
        3   1    1          5        5       0
        4   0    2          3        5       1
        5   0    3          4        5       0
        ```
    """
    new_records = np.empty(arr.shape, dtype=range_dt)
    counts = np.full(arr.shape[1], 0, dtype=np.int_)

    for col in prange(arr.shape[1]):
        range_started = False
        start_idx = -1
        end_idx = -1
        store_record = False
        status = -1

        for i in range(arr.shape[0]):
            cur_val = arr[i, col]

            if cur_val == gap_value or np.isnan(cur_val) and np.isnan(gap_value):
                if range_started:
                    # If stopped, save the current range
                    end_idx = i
                    range_started = False
                    store_record = True
                    status = RangeStatus.Closed
            else:
                if not range_started:
                    # If started, register a new range
                    start_idx = i
                    range_started = True

            if i == arr.shape[0] - 1 and range_started:
                # If still running, mark for save
                end_idx = arr.shape[0] - 1
                range_started = False
                store_record = True
                status = RangeStatus.Open

            if store_record:
                # Save range to the records
                r = counts[col]
                new_records['id'][r, col] = r
                new_records['col'][r, col] = col
                new_records['start_idx'][r, col] = start_idx
                new_records['end_idx'][r, col] = end_idx
                new_records['status'][r, col] = status
                counts[col] += 1

                # Reset running vars for a new range
                store_record = False

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query='start_idx_arr', axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
        status_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def range_duration_nb(start_idx_arr: tp.Array1d,
                      end_idx_arr: tp.Array1d,
                      status_arr: tp.Array2d) -> tp.Array1d:
    """Get duration of each duration record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.int_)
    for r in prange(start_idx_arr.shape[0]):
        if status_arr[r] == RangeStatus.Open:
            out[r] = end_idx_arr[r] - start_idx_arr[r] + 1
        else:
            out[r] = end_idx_arr[r] - start_idx_arr[r]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='col_map'),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_lens=ch.ArraySlicer(axis=0),
        overlapping=None,
        normalize=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def range_coverage_nb(start_idx_arr: tp.Array1d,
                      end_idx_arr: tp.Array1d,
                      status_arr: tp.Array2d,
                      col_map: tp.GroupMap,
                      index_lens: tp.Array1d,
                      overlapping: bool = False,
                      normalize: bool = False) -> tp.Array1d:
    """Get coverage of range records.

    Set `overlapping` to True to get the number of overlapping steps.
    Set `normalize` to True to get the number of steps in relation either to the total number of steps
    (when `overlapping=False`) or to the number of covered steps (when `overlapping=True`).
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], np.nan, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx:col_start_idx + col_len]
        temp = np.full(index_lens[col], 0, dtype=np.int_)
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                temp[start_idx_arr[r]:end_idx_arr[r] + 1] += 1
            else:
                temp[start_idx_arr[r]:end_idx_arr[r]] += 1
        if overlapping:
            if normalize:
                out[col] = np.sum(temp > 1) / np.sum(temp > 0)
            else:
                out[col] = np.sum(temp > 1)
        else:
            if normalize:
                out[col] = np.sum(temp > 0) / index_lens[col]
            else:
                out[col] = np.sum(temp > 0)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='col_map'),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_len=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def ranges_to_mask_nb(start_idx_arr: tp.Array1d,
                      end_idx_arr: tp.Array1d,
                      status_arr: tp.Array2d,
                      col_map: tp.GroupMap,
                      index_len: int) -> tp.Array2d:
    """Convert ranges to 2-dim mask."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((index_len, col_lens.shape[0]), False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx:col_start_idx + col_len]
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                out[start_idx_arr[r]:end_idx_arr[r] + 1, col] = True
            else:
                out[start_idx_arr[r]:end_idx_arr[r], col] = True

    return out


# ############# Drawdowns ############# #

@register_jitted(cache=True)
def drawdown_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Compute drawdown."""
    out = np.empty_like(arr, dtype=np.float_)
    max_val = np.nan
    for i in range(arr.shape[0]):
        if np.isnan(max_val) or arr[i] > max_val:
            max_val = arr[i]
        out[i] = arr[i] / max_val - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def drawdown_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `drawdown_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = drawdown_1d_nb(arr[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def get_drawdowns_nb(arr: tp.Array2d) -> tp.RecordArray:
    """Fill drawdown records by analyzing a time series.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_drawdowns_nb

        >>> a = np.asarray([
        ...     [1, 5, 1, 3],
        ...     [2, 4, 2, 2],
        ...     [3, 3, 3, 1],
        ...     [4, 2, 2, 2],
        ...     [5, 1, 1, 3]
        ... ])
        >>> records = get_drawdowns_nb(a)

        >>> pd.DataFrame.from_records(records)
           id  col  peak_idx  start_idx  valley_idx  end_idx  peak_val  valley_val  \\
        0   0    1         0          1           4        4       5.0         1.0
        1   0    2         2          3           4        4       3.0         1.0
        2   0    3         0          1           2        4       3.0         1.0

           end_val  status
        0      1.0       0
        1      1.0       0
        2      3.0       1
        ```
    """
    new_records = np.empty(arr.shape, dtype=drawdown_dt)
    counts = np.full(arr.shape[1], 0, dtype=np.int_)

    for col in prange(arr.shape[1]):
        drawdown_started = False
        peak_idx = -1
        valley_idx = -1
        peak_val = arr[0, col]
        valley_val = arr[0, col]
        store_record = False
        status = -1

        for i in range(arr.shape[0]):
            cur_val = arr[i, col]

            if not np.isnan(cur_val):
                if np.isnan(peak_val) or cur_val >= peak_val:
                    # Value increased
                    if not drawdown_started:
                        # If not running, register new peak
                        peak_val = cur_val
                        peak_idx = i
                    else:
                        # If running, potential recovery
                        if cur_val >= peak_val:
                            drawdown_started = False
                            store_record = True
                            status = DrawdownStatus.Recovered
                else:
                    # Value decreased
                    if not drawdown_started:
                        # If not running, start new drawdown
                        drawdown_started = True
                        valley_val = cur_val
                        valley_idx = i
                    else:
                        # If running, potential valley
                        if cur_val < valley_val:
                            valley_val = cur_val
                            valley_idx = i

                if i == arr.shape[0] - 1 and drawdown_started:
                    # If still running, mark for save
                    drawdown_started = False
                    store_record = True
                    status = DrawdownStatus.Active

                if store_record:
                    # Save drawdown to the records
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['peak_idx'][r, col] = peak_idx
                    new_records['start_idx'][r, col] = peak_idx + 1
                    new_records['valley_idx'][r, col] = valley_idx
                    new_records['end_idx'][r, col] = i
                    new_records['peak_val'][r, col] = peak_val
                    new_records['valley_val'][r, col] = valley_val
                    new_records['end_val'][r, col] = cur_val
                    new_records['status'][r, col] = status
                    counts[col] += 1

                    # Reset running vars for a new drawdown
                    peak_idx = i
                    valley_idx = i
                    peak_val = cur_val
                    valley_val = cur_val
                    store_record = False
                    status = -1

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query='peak_val_arr', axis=0),
    arg_take_spec=dict(
        peak_val_arr=ch.ArraySlicer(axis=0),
        valley_val_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def dd_drawdown_nb(peak_val_arr: tp.Array1d, valley_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the drawdown of each drawdown record."""
    out = np.empty(valley_val_arr.shape[0], dtype=np.float_)
    for r in prange(valley_val_arr.shape[0]):
        out[r] = (valley_val_arr[r] - peak_val_arr[r]) / peak_val_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='start_idx_arr', axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        valley_idx_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def dd_decline_duration_nb(start_idx_arr: tp.Array1d, valley_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the peak-to-valley phase of each drawdown record."""
    out = np.empty(valley_idx_arr.shape[0], dtype=np.float_)
    for r in prange(valley_idx_arr.shape[0]):
        out[r] = valley_idx_arr[r] - start_idx_arr[r] + 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='valley_idx_arr', axis=0),
    arg_take_spec=dict(
        valley_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def dd_recovery_duration_nb(valley_idx_arr: tp.Array1d, end_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the valley-to-recovery phase of each drawdown record."""
    out = np.empty(end_idx_arr.shape[0], dtype=np.float_)
    for r in prange(end_idx_arr.shape[0]):
        out[r] = end_idx_arr[r] - valley_idx_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='start_idx_arr', axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        valley_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def dd_recovery_duration_ratio_nb(start_idx_arr: tp.Array1d,
                                  valley_idx_arr: tp.Array1d,
                                  end_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the ratio of the recovery duration to the decline duration of each drawdown record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.float_)
    for r in prange(start_idx_arr.shape[0]):
        out[r] = (end_idx_arr[r] - valley_idx_arr[r]) / (valley_idx_arr[r] - start_idx_arr[r] + 1)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='valley_val_arr', axis=0),
    arg_take_spec=dict(
        valley_val_arr=ch.ArraySlicer(axis=0),
        end_val_arr=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def dd_recovery_return_nb(valley_val_arr: tp.Array1d, end_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the recovery return of each drawdown record."""
    out = np.empty(end_val_arr.shape[0], dtype=np.float_)
    for r in prange(end_val_arr.shape[0]):
        out[r] = (end_val_arr[r] - valley_val_arr[r]) / valley_val_arr[r]
    return out


# ############# Crossover ############# #

@register_jitted(cache=True)
def crossed_above_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, wait: int = 0) -> tp.Array1d:
    """Get the crossover of the first array going above the second array."""
    out = np.empty(arr1.shape, dtype=np.bool_)
    was_below = False
    crossed_ago = -1

    for i in range(arr1.shape[0]):
        if was_below:
            if arr1[i] > arr2[i]:
                crossed_ago += 1
                out[i] = crossed_ago == wait
            elif crossed_ago != -1 or (np.isnan(arr1[i]) or np.isnan(arr2[i])):
                crossed_ago = -1
                was_below = False
                out[i] = False
            else:
                out[i] = False
        else:
            if arr1[i] < arr2[i]:
                was_below = True
            out[i] = False
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr1', axis=1),
    arg_take_spec=dict(
        arr1=ch.ArraySlicer(axis=1),
        arr2=ch.ArraySlicer(axis=1),
        wait=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def crossed_above_nb(arr1: tp.Array2d, arr2: tp.Array2d, wait: int = 0) -> tp.Array2d:
    """2-dim version of `crossed_above_1d_nb`."""
    out = np.empty(arr1.shape, dtype=np.bool_)
    for col in prange(arr1.shape[1]):
        out[:, col] = crossed_above_1d_nb(arr1[:, col], arr2[:, col], wait=wait)
    return out


# ############# Transformation ############# #

@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query='group_map'),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1, mapper=base_ch.group_idxs_mapper),
        group_map=base_ch.GroupMapSlicer()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def demean_nb(arr: tp.Array2d, group_map: tp.GroupMap) -> tp.Array2d:
    """Demean each value within its group."""
    group_idxs, group_lens = group_map
    group_start_idxs = np.cumsum(group_lens) - group_lens
    out = np.empty_like(arr, dtype=np.float_)

    for group in prange(len(group_lens)):
        group_len = group_lens[group]
        start_idx = group_start_idxs[group]
        col_idxs = group_idxs[start_idx:start_idx + group_len]
        for i in range(arr.shape[0]):
            group_sum = 0
            group_cnt = 0
            for k in range(group_len):
                col = col_idxs[k]
                if not np.isnan(arr[i, col]):
                    group_sum += arr[i, col]
                    group_cnt += 1
            for k in range(group_len):
                col = col_idxs[k]
                if np.isnan(arr[i, col]) or group_cnt == 0:
                    out[i, col] = np.nan
                else:
                    out[i, col] = arr[i, col] - group_sum / group_cnt
    return out