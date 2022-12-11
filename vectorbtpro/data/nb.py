# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for generating data.

Provides an arsenal of Numba-compiled functions that are used to generate data.
These only accept NumPy arrays and other Numba-compatible types."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb


@register_jitted(cache=True, tags={"can_parallel"})
def generate_random_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    symmetric: tp.FlexArray1dLike = False,
) -> tp.Array2d:
    """Generate data using cumulative product of returns drawn from normal (Gaussian) distribution.

    Turn on `symmetric` to diminish negative returns and make them symmetric to positive ones.
    Otherwise, the majority of generated paths will go downward.

    Each argument can be provided per column thanks to flexible indexing."""
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    symmetric_ = to_1d_array_nb(np.asarray(symmetric))

    out = np.empty(shape, dtype=np.float_)

    for col in prange(shape[1]):
        _start_value = flex_select_1d_pc_nb(start_value_, col)
        _mean = flex_select_1d_pc_nb(mean_, col)
        _std = flex_select_1d_pc_nb(std_, col)
        _symmetric = flex_select_1d_pc_nb(symmetric_, col)

        for i in range(shape[0]):
            if i == 0:
                prev_value = _start_value
            else:
                prev_value = out[i - 1, col]
            return_ = np.random.normal(_mean, _std)
            if _symmetric and return_ < 0:
                return_ = -abs(return_) / (1 + abs(return_))
            out[i, col] = prev_value * (1 + return_)

    return out


@register_jitted(cache=True, tags={"can_parallel"})
def generate_gbm_data_nb(
    shape: tp.Shape,
    start_value: tp.FlexArray1dLike = 100.0,
    mean: tp.FlexArray1dLike = 0.0,
    std: tp.FlexArray1dLike = 0.01,
    dt: tp.FlexArray1dLike = 1.0,
) -> tp.Array2d:
    """Generate data using Geometric Brownian Motion (GBM)."""
    start_value_ = to_1d_array_nb(np.asarray(start_value))
    mean_ = to_1d_array_nb(np.asarray(mean))
    std_ = to_1d_array_nb(np.asarray(std))
    dt_ = to_1d_array_nb(np.asarray(dt))

    out = np.empty(shape, dtype=np.float_)

    for col in prange(shape[1]):
        _start_value = flex_select_1d_pc_nb(start_value_, col)
        _mean = flex_select_1d_pc_nb(mean_, col)
        _std = flex_select_1d_pc_nb(std_, col)
        _dt = flex_select_1d_pc_nb(dt_, col)

        for i in range(shape[0]):
            if i == 0:
                prev_value = _start_value
            else:
                prev_value = out[i - 1, col]
            rand = np.random.standard_normal()
            out[i, col] = prev_value * np.exp((_mean - 0.5 * _std**2) * _dt + _std * np.sqrt(_dt) * rand)

    return out
