# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for generating data.

Provides an arsenal of Numba-compiled functions that are used to generate data.
These only accept NumPy arrays and other Numba-compatible types."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted(cache=True, tags={'can_parallel'})
def generate_random_data_nb(shape: tp.Shape, start_value: float, mean: float, std: float) -> tp.Array2d:
    """Generate data using cumulative product of returns drawn from normal (Gaussian) distribution."""
    out = np.empty(shape, dtype=np.float_)
    for col in prange(shape[1]):
        for i in range(shape[0]):
            if i == 0:
                prev_value = start_value
            else:
                prev_value = out[i - 1, col]
            out[i, col] = prev_value * (1 + np.random.normal(mean, std))
    return out


@register_jitted(cache=True, tags={'can_parallel'})
def generate_gbm_data_nb(shape: tp.Shape, start_value: float, mean: float, std: float, dt: float) -> tp.Array2d:
    """Generate data using Geometric Brownian Motion (GBM)."""
    out = np.empty(shape, dtype=np.float_)
    for col in prange(shape[1]):
        for i in range(shape[0]):
            if i == 0:
                prev_value = start_value
            else:
                prev_value = out[i - 1, col]
            rand = np.random.standard_normal()
            out[i, col] = prev_value * np.exp((mean - 0.5 * std ** 2) * dt + std * np.sqrt(dt) * rand)
    return out
