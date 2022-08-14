# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for iterative portfolio modeling."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import flex_select_auto_nb
from vectorbtpro.generic.nb.iter_ import (
    iter_above_nb as _iter_above_nb,
    iter_below_nb as _iter_below_nb,
    iter_crossed_above_nb as _iter_crossed_above_nb,
    iter_crossed_below_nb as _iter_crossed_below_nb,
)
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted
def select_nb(ctx: tp.NamedTuple, arr: tp.FlexArray) -> tp.Scalar:
    """Get the current element using flexible indexing given the context's `i` and `col`."""
    return flex_select_auto_nb(arr, ctx.i, ctx.col, ctx.flex_2d)


@register_jitted
def group_select_nb(ctx: tp.NamedTuple, arr: tp.FlexArray) -> tp.Scalar:
    """Get the current element using flexible indexing given the context's `i` and `group`."""
    return flex_select_auto_nb(arr, ctx.i, ctx.group, ctx.flex_2d)


@register_jitted
def select_from_col_nb(ctx: tp.NamedTuple, col_or_group: int, arr: tp.FlexArray) -> tp.Scalar:
    """Get the current element using flexible indexing given a column/group and the context's `i`."""
    return flex_select_auto_nb(arr, ctx.i, col_or_group, ctx.flex_2d)


@register_jitted
def iter_above_nb(c: tp.NamedTuple, arr1: tp.FlexArray, arr2: tp.FlexArray) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_above_nb` on the context."""
    return _iter_above_nb(arr1, arr2, c.i, c.col, c.flex_2d)


@register_jitted
def iter_below_nb(c: tp.NamedTuple, arr1: tp.FlexArray, arr2: tp.FlexArray) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_below_nb` on the context."""
    return _iter_below_nb(arr1, arr2, c.i, c.col, c.flex_2d)


@register_jitted
def iter_crossed_above_nb(c: tp.NamedTuple, arr1: tp.FlexArray, arr2: tp.FlexArray) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_above_nb` on the context."""
    return _iter_crossed_above_nb(arr1, arr2, c.i, c.col, c.flex_2d)


@register_jitted
def iter_crossed_below_nb(c: tp.NamedTuple, arr1: tp.FlexArray, arr2: tp.FlexArray) -> bool:
    """Call `vectorbtpro.generic.nb.iter_.iter_crossed_below_nb` on the context."""
    return _iter_crossed_below_nb(arr1, arr2, c.i, c.col, c.flex_2d)
