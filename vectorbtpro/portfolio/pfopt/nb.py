# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio optimization."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


@register_jitted(cache=True)
def get_alloc_points_nb(
    filled_allocations: tp.Array2d,
    nonzero_only: bool = True,
    unique_only: bool = True,
) -> tp.Array1d:
    """Get allocation points from filled allocations.

    Set `nonzero_only` to False to not register a new allocation when all points are 0 or NaN.
    Set `unique_only` to False to not register a new allocation when it's the same as the last one."""
    out = np.empty(len(filled_allocations), dtype=np.int_)
    k = 0
    for i in range(filled_allocations.shape[0]):
        all_zeros = True
        all_unique = True
        for col in range(filled_allocations.shape[1]):
            if abs(filled_allocations[i, col]) > 0:
                all_zeros = False
            if k == 0 or (k > 0 and filled_allocations[i, col] != filled_allocations[out[k - 1], col]):
                all_unique = False
        if nonzero_only and all_zeros:
            continue
        if unique_only and all_unique:
            continue
        out[k] = i
        k += 1
    return out[:k]


@register_chunkable(
    size=ch.ArraySizer(arg_query="index_ranges", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        index_ranges=ch.ArraySlicer(axis=0),
        optimize_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.row_stack,
)
@register_jitted(tags={"can_parallel"})
def optimize_meta_nb(
    n_cols: int,
    index_ranges: tp.Array2d,
    reduce_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Optimize by reducing each index range.

    `reduce_func_nb` must take the range index, the range start, the range end, and `*args`.
    Must return a 1-dim array with the same size as `n_cols`."""
    out = np.empty((index_ranges.shape[0], n_cols), dtype=np.float_)
    for i in prange(len(index_ranges)):
        out[i] = reduce_func_nb(i, index_ranges[i][0], index_ranges[i][1], *args)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="index_points", axis=0),
    arg_take_spec=dict(
        n_cols=None,
        index_points=ch.ArraySlicer(axis=0),
        map_func_nb=None,
        args=ch.ArgsTaker(),
    ),
    merge_func=base_ch.row_stack,
)
@register_jitted(tags={"can_parallel"})
def allocate_meta_nb(
    n_cols: int,
    index_points: tp.Array1d,
    map_func_nb: tp.Callable,
    *args,
) -> tp.Array2d:
    """Allocate by mapping each index point.

    `map_func_nb` must take the point index, the index point, and `*args`.
    Must return a 1-dim array with the same size as `n_cols`."""
    out = np.empty((index_points.shape[0], n_cols), dtype=np.float_)
    for i in prange(len(index_points)):
        out[i] = map_func_nb(i, index_points[i], *args)
    return out


@register_jitted(cache=True)
def pick_idx_allocate_func_nb(i, index_point, allocations):
    """Pick the allocation at an absolute position in an array."""
    return allocations[i]


@register_jitted(cache=True)
def pick_point_allocate_func_nb(i, index_point, allocations):
    """Pick the allocation at an index point in an array."""
    return allocations[index_point]


@register_jitted(cache=True)
def random_allocate_func_nb(i, index_point, n_cols):
    """Generate a random allocation."""
    weights = np.random.uniform(0, 1, n_cols)
    return weights / weights.sum()
