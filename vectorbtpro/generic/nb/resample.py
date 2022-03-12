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


@register_jitted
def resample_to_index_1d_nb(
    arr: tp.Array1d,
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_one: bool,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce `arr` after/before each index in `target_index` based on `source_index`.

    If `before` is True, applied on elements that come before and including that index.
    Otherwise, applied on elements that come after and including that index.

    If `reduce_one` is True, applies also on arrays with one element. Otherwise, directly sets
    that element to the output index.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value.

    !!! note
        Both index arrays must be increasing. Repeating values are allowed."""
    out = np.empty(len(target_index), dtype=np.float_)

    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")

        from_j = -1
        for j in range(to_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
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
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            out[i] = np.nan
        elif to_j - from_j == 1 and not reduce_one:
            out[i] = arr[from_j]
        else:
            out[i] = reduce_func_nb(arr[from_j:to_j], *args)

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        source_index=None,
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
    source_index: tp.Array1d,
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
            source_index,
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
    source_index: tp.Array1d,
    target_index: tp.Array1d,
    before: bool,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `resample_to_index_1d_nb`.

    `reduce_func_nb` must accept the (absolute) start row index, the end row index, the column index,
    and `*args`. Must return a single value."""
    out = np.empty(len(target_index), dtype=np.float_)

    to_j = 0
    for i in range(len(target_index)):
        if i > 0 and target_index[i] < target_index[i - 1]:
            raise ValueError("Target index must be increasing")

        from_j = -1
        for j in range(to_j, len(source_index)):
            if j > 0 and source_index[j] < source_index[j - 1]:
                raise ValueError("Array index must be increasing")
            found = False
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
            if found:
                if from_j == -1:
                    from_j = j
                to_j = j + 1

        if from_j == -1:
            out[i] = np.nan
        else:
            out[i] = reduce_func_nb(from_j, to_j, col, *args)

    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        source_index=None,
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
    source_index: tp.Array1d,
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
            source_index,
            target_index,
            before,
            reduce_func_nb,
            *args,
        )
    return out


@register_jitted
def resample_between_bounds_1d_nb(
    arr: tp.Array1d,
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool,
    closed_rbound: bool,
    reduce_one: bool,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array1d:
    """Reduce `arr` between each pair of indexes in `target_lbound_index` and `target_rbound_index`
    based on `source_index`.

    Both index arrays are acting as the left and right bounds explicitly. Set `closed_lbound` to True
    to include the left index, and `closed_rbound` to True to include the right one.

    If `reduce_one` is True, applies also on arrays with one element. Otherwise, directly sets
    that element to the output index.

    `reduce_func_nb` must accept the array and `*args`. Must return a single value.

    !!! note
        All index arrays must be increasing. Repeating values are allowed."""
    out = np.empty(len(target_lbound_index), dtype=np.float_)

    from_j = 0
    to_j = 0
    for i in range(len(target_lbound_index)):
        if i > 0 and target_lbound_index[i] < target_lbound_index[i - 1]:
            raise ValueError("Target left-bound index must be increasing")
        if i > 0 and target_rbound_index[i] < target_rbound_index[i - 1]:
            raise ValueError("Target right-bound index must be increasing")

        found_any = False
        start_from_j = from_j
        if closed_lbound and closed_rbound:
            if i > 0 and target_lbound_index[i] > target_rbound_index[i - 1]:
                start_from_j = to_j
        else:
            if i > 0 and target_lbound_index[i] >= target_rbound_index[i - 1]:
                start_from_j = to_j
        for j in range(start_from_j, len(source_index)):
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
                if not found_any:
                    from_j = j
                to_j = j + 1
                found_any = True

        if not found_any:
            out[i] = np.nan
        elif to_j - from_j == 1 and not reduce_one:
            out[i] = arr[from_j]
        else:
            out[i] = reduce_func_nb(arr[from_j:to_j], *args)

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        source_index=None,
        target_lbound_index=None,
        target_rbound_index=None,
        closed_lbound=None,
        closed_rbound=None,
        reduce_one=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(tags={"can_parallel"})
def resample_between_bounds_nb(
    arr: tp.Array2d,
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool,
    closed_rbound: bool,
    reduce_one: bool,
    reduce_func_nb: tp.ReduceFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `resample_between_bounds_1d_nb`."""
    out = np.empty((target_lbound_index.shape[0], arr.shape[1]), dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = resample_between_bounds_1d_nb(
            arr[:, col],
            source_index,
            target_lbound_index,
            target_rbound_index,
            closed_lbound,
            closed_rbound,
            reduce_one,
            reduce_func_nb,
            *args,
        )
    return out


@register_jitted
def resample_between_bounds_1d_meta_nb(
    col: int,
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool,
    closed_rbound: bool,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array1d:
    """Meta version of `resample_between_bounds_1d_nb`.

    `reduce_func_nb` must accept the (absolute) start row index, the end row index, the column index,
    and `*args`. Must return a single value."""
    out = np.empty(len(target_lbound_index), dtype=np.float_)

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
            out[i] = np.nan
        else:
            out[i] = reduce_func_nb(from_j, to_j, col, *args)

    return out


@register_chunkable(
    size=ch.ArgSizer(arg_query="n_cols"),
    arg_take_spec=dict(
        n_cols=ch.CountAdapter(),
        source_index=None,
        target_lbound_index=None,
        target_rbound_index=None,
        closed_lbound=None,
        closed_rbound=None,
        reduce_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(tags={"can_parallel"})
def resample_between_bounds_meta_nb(
    n_cols: int,
    source_index: tp.Array1d,
    target_lbound_index: tp.Array1d,
    target_rbound_index: tp.Array1d,
    closed_lbound: bool,
    closed_rbound: bool,
    reduce_func_nb: tp.RangeReduceMetaFunc,
    *args,
) -> tp.Array2d:
    """2-dim version of `resample_between_bounds_1d_meta_nb`."""
    out = np.empty((target_lbound_index.shape[0], n_cols), dtype=np.float_)
    for col in prange(n_cols):
        out[:, col] = resample_between_bounds_1d_meta_nb(
            col,
            source_index,
            target_lbound_index,
            target_rbound_index,
            closed_lbound,
            closed_rbound,
            reduce_func_nb,
            *args,
        )
    return out
