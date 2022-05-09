# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for OHLCV.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0)."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch


@register_jitted(cache=True)
def ohlc_every_1d_nb(price: tp.Array1d, n: int) -> tp.Array2d:
    """Aggregate every `n` price points into an OHLC point."""
    out = np.empty((price.shape[0], 4), dtype=np.float_)
    vmin = np.inf
    vmax = -np.inf
    j = -1
    for i in range(price.shape[0]):
        if price[i] < vmin:
            vmin = price[i]
        if price[i] > vmax:
            vmax = price[i]
        if i % n == 0:
            j += 1
            out[j, 0] = price[i]
        if i % n == n - 1 or i == price.shape[0] - 1:
            out[j, 1] = vmax
            out[j, 2] = vmin
            out[j, 3] = price[i]
            vmin = np.inf
            vmax = -np.inf
    return out[: j + 1]


@register_jitted(cache=True)
def vwap_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    volume: tp.Array1d,
    group_lens: tp.GroupLens,
) -> tp.Array1d:
    """Compute the volume-weighted average price (VWAP)."""
    out = np.empty_like(volume, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        nom_cumsum = 0
        denum_cumsum = 0
        for i in range(from_i, to_i):
            nom_cumsum += volume[i] * (high[i] + low[i] + close[i]) / 3
            denum_cumsum += volume[i]
            out[i] = nom_cumsum / denum_cumsum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        volume=ch.ArraySlicer(axis=1),
        group_lens=None,
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def vwap_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    volume: tp.Array2d,
    group_lens: tp.GroupLens,
) -> tp.Array2d:
    """2-dim version of `vwap_1d_nb`."""
    out = np.empty_like(high, dtype=np.float_)
    for col in prange(high.shape[1]):
        out[:, col] = vwap_1d_nb(
            high[:, col],
            low[:, col],
            close[:, col],
            volume[:, col],
            group_lens,
        )
    return out
