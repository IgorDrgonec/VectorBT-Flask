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
def vwap_1d_nb(high: tp.Array1d, low: tp.Array1d, volume: tp.Array1d) -> tp.Array1d:
    """Compute the volume-weighted average price (VWAP)."""
    out = np.empty_like(volume, dtype=np.float_)
    nom_cumsum = 0
    denum_cumsum = 0
    for i in range(volume.shape[0]):
        nom_cumsum += volume[i] * (high[i] + low[i]) / 2
        denum_cumsum += volume[i]
        out[i] = nom_cumsum / denum_cumsum
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(high=ch.ArraySlicer(axis=1), low=ch.ArraySlicer(axis=1), volume=ch.ArraySlicer(axis=1)),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def vwap_nb(high: tp.Array2d, low: tp.Array2d, volume: tp.Array2d) -> tp.Array2d:
    """2-dim version of `vwap_1d_nb`."""
    out = np.empty_like(high, dtype=np.float_)
    for col in prange(high.shape[1]):
        out[:, col] = vwap_1d_nb(high[:, col], low[:, col], volume[:, col])
    return out
