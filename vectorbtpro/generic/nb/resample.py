# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for resampling."""

import numpy as np
from numba import prange
from numba.np.numpy_support import as_dtype

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


@register_jitted(cache=True, is_generated_jit=True)
def latest_at_index_1d_nb(
    arr: tp.Array1d,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
    nan_value: tp.Scalar = np.nan,
    ffill: bool = True,
) -> tp.Array1d:
    """Get the latest in `arr` at each index in `target_index` based on `arr_index`."""
    nb_enabled = not isinstance(arr, np.ndarray)
    if nb_enabled:
        a_dtype = as_dtype(arr.dtype)
        value_dtype = as_dtype(nan_value)
    else:
        a_dtype = arr.dtype
        value_dtype = np.array(nan_value).dtype
    dtype = np.promote_types(a_dtype, value_dtype)

    def _latest_at_index_1d_nb(arr, arr_index, target_index, nan_value, ffill):
        out = np.empty(target_index.shape[0], dtype=dtype)
        curr_j = -1
        last_j = curr_j
        last_valid = np.nan
        for i in range(len(target_index)):
            last_valid_at_i = np.nan
            if i > 0 and target_index[i] < target_index[i - 1]:
                raise ValueError("Target index must be strictly increasing")
            for j in range(curr_j + 1, arr_index.shape[0]):
                if j > 0 and arr_index[j] < arr_index[j - 1]:
                    raise ValueError("Array index must be strictly increasing")
                if arr_index[j] == target_index[i]:
                    curr_j = j
                    if not np.isnan(arr[curr_j]):
                        last_valid_at_i = arr[curr_j]
                    break
                if arr_index[j] > target_index[i]:
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
        return _latest_at_index_1d_nb(arr, arr_index, target_index, nan_value, ffill)

    return _latest_at_index_1d_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        arr_index=None,
        target_index=None,
        nan_value=None,
        ffill=None,
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"}, is_generated_jit=True)
def latest_at_index_nb(
    arr: tp.Array2d,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
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

    def _latest_at_index_nb(arr, arr_index, target_index, nan_value, ffill):
        out = np.empty((target_index.shape[0], arr.shape[1]), dtype=dtype)
        for col in prange(arr.shape[1]):
            out[:, col] = latest_at_index_1d_nb(
                arr[:, col],
                arr_index,
                target_index,
                nan_value=nan_value,
                ffill=ffill,
            )
        return out

    if not nb_enabled:
        return _latest_at_index_nb(arr, arr_index, target_index, nan_value, ffill)

    return _latest_at_index_nb


@register_jitted
def resample_to_index_1d_nb(
    arr: tp.Array1d,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_one: bool,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce `arr` after/before each index in `target_index` based on `arr_index`.

    If `before` is True, applied on elements that come before and including that index.
    Otherwise, applied on elements that come after and including that index.

    If `reduce_one` is True, applies also on arrays with one element. Otherwise, directly sets
    that element to the output index.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value."""
    out = np.empty(len(target_index), dtype=np.float_)
    from_j = 0
    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be strictly increasing")
        if (
            to_j == len(arr_index)
            or (before and arr_index[to_j] > target_index[i])
            or (not before and arr_index[to_j] < target_index[i])
        ):
            out[i] = np.nan
            continue
        found = False
        for j in range(to_j, len(arr_index)):
            if j > 0 and arr_index[j] < arr_index[j - 1]:
                raise ValueError("Array index must be strictly increasing")
            if (before and arr_index[j] >= target_index[i]) or (
                not before and i < len(target_index) - 1 and arr_index[j] >= target_index[i + 1]
            ):
                found = True
                if before:
                    to_j = j + 1
                else:
                    to_j = j
                break
        if not found:
            to_j = len(arr_index)
        if to_j - from_j == 0:
            out[i] = np.nan
        elif to_j - from_j == 1 and not reduce_one:
            out[i] = arr[from_j]
        else:
            out[i] = reduce_func_nb(arr[from_j:to_j], *args)
        from_j = to_j
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        arr_index=None,
        target_index=None,
        before=None,
        reduce_one=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(tags={"can_parallel"})
def resample_to_index_nb(
    arr: tp.Array2d,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_one: bool,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `resample_to_index_1d_nb`."""
    out = np.empty((target_index.shape[0], arr.shape[1]), dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = resample_to_index_1d_nb(
            arr[:, col],
            arr_index,
            target_index,
            before,
            reduce_one,
            reduce_func_nb,
            *args,
        )
    return out


@register_jitted
def resample_to_index_1d_meta_nb(
    col: int,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `resample_to_index_1d_nb`.

    `reduce_func_nb` must accept the (absolute) start row index, the end row index, the column index,
    and `*args`. Must return a single value."""
    out = np.empty(len(target_index), dtype=np.float_)
    from_j = 0
    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be strictly increasing")
        if (
            to_j == len(arr_index)
            or (before and arr_index[to_j] > target_index[i])
            or (not before and arr_index[to_j] < target_index[i])
        ):
            out[i] = np.nan
            continue
        found = False
        for j in range(to_j, len(arr_index)):
            if j > 0 and arr_index[j] < arr_index[j - 1]:
                raise ValueError("Array index must be strictly increasing")
            if (before and arr_index[j] >= target_index[i]) or (
                not before and i < len(target_index) - 1 and arr_index[j] >= target_index[i + 1]
            ):
                found = True
                if before:
                    to_j = j + 1
                else:
                    to_j = j
                break
        if not found:
            to_j = len(arr_index)
        if to_j - from_j == 0:
            out[i] = np.nan
        else:
            out[i] = reduce_func_nb(from_j, to_j, col, *args)
        from_j = to_j
    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        arr_index=None,
        target_index=None,
        before=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(tags={"can_parallel"})
def resample_to_index_meta_nb(
    n_cols: int,
    arr_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `resample_to_index_1d_meta_nb`."""
    out = np.empty((target_index.shape[0], n_cols), dtype=np.float_)
    for col in prange(n_cols):
        out[:, col] = resample_to_index_1d_meta_nb(
            col,
            arr_index,
            target_index,
            before,
            reduce_func_nb,
            *args,
        )
    return out
