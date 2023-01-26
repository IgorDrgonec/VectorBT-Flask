# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for OHLCV.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0)."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted


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
