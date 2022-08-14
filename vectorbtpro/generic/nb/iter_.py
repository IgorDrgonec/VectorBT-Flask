# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for iterative use."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.base.indexing import flex_select_auto_nb


@register_jitted(cache=True)
def iter_above_nb(arr1: tp.FlexArray, arr2: tp.FlexArray, i: int, col: int, flex_2d: bool = False) -> bool:
    """Check whether `arr1` is above `arr2` at specific row and column."""
    if i < 0:
        return False
    arr1_now = flex_select_auto_nb(arr1, i, col, flex_2d)
    arr2_now = flex_select_auto_nb(arr2, i, col, flex_2d)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now > arr2_now


@register_jitted(cache=True)
def iter_below_nb(arr1: tp.FlexArray, arr2: tp.FlexArray, i: int, col: int, flex_2d: bool = False) -> bool:
    """Check whether `arr1` is below `arr2` at specific row and column."""
    if i < 0:
        return False
    arr1_now = flex_select_auto_nb(arr1, i, col, flex_2d)
    arr2_now = flex_select_auto_nb(arr2, i, col, flex_2d)
    if np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_now < arr2_now


@register_jitted(cache=True)
def iter_crossed_above_nb(arr1: tp.FlexArray, arr2: tp.FlexArray, i: int, col: int, flex_2d: bool = False) -> bool:
    """Check whether `arr1` crossed above `arr2` at specific row and column."""
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_auto_nb(arr1, i - 1, col, flex_2d)
    arr2_prev = flex_select_auto_nb(arr2, i - 1, col, flex_2d)
    arr1_now = flex_select_auto_nb(arr1, i, col, flex_2d)
    arr2_now = flex_select_auto_nb(arr2, i, col, flex_2d)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev < arr2_prev and arr1_now > arr2_now


@register_jitted(cache=True)
def iter_crossed_below_nb(arr1: tp.FlexArray, arr2: tp.FlexArray, i: int, col: int, flex_2d: bool = False) -> bool:
    """Check whether `arr1` crossed below `arr2` at specific row and column."""
    if i < 0 or i - 1 < 0:
        return False
    arr1_prev = flex_select_auto_nb(arr1, i - 1, col, flex_2d)
    arr2_prev = flex_select_auto_nb(arr2, i - 1, col, flex_2d)
    arr1_now = flex_select_auto_nb(arr1, i, col, flex_2d)
    arr2_now = flex_select_auto_nb(arr2, i, col, flex_2d)
    if np.isnan(arr1_prev) or np.isnan(arr2_prev) or np.isnan(arr1_now) or np.isnan(arr2_now):
        return False
    return arr1_prev > arr2_prev and arr1_now < arr2_now
