# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled helper functions for portfolio simulation."""

from vectorbtpro.portfolio.nb.core import *


@register_jitted
def in_position_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> bool:
    """Check whether the current column is in a position."""
    return c.last_position[c.col] != 0


@register_jitted
def in_long_position_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> bool:
    """Check whether the current column is in a long position."""
    return c.last_position[c.col] > 0


@register_jitted
def in_short_position_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> bool:
    """Check whether the current column is in a short position."""
    return c.last_position[c.col] < 0


@register_jitted
def get_n_active_positions_nb(
    c: tp.Union[
        GroupContext,
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
    ]
) -> int:
    """Get the number of active positions in the current group (regardless of cash sharing)."""
    n_active_positions = 0
    for col in range(c.from_col, c.to_col):
        if c.last_position[col] != 0:
            n_active_positions += 1
    return n_active_positions


@register_jitted
def get_cash_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> float:
    """Get cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return c.last_cash[c.group]
    return c.last_cash[c.col]


@register_jitted
def get_free_cash_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> float:
    """Get free cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return c.last_free_cash[c.group]
    return c.last_free_cash[c.col]


@register_jitted
def has_free_cash_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> bool:
    """Check whether the current column or group with cash sharing has free cash."""
    return get_free_cash_nb(c) > 0


@register_jitted
def any_order_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> bool:
    """Check whether there is any order in the current column."""
    return c.order_counts[c.col] > 0


@register_jitted
def get_last_order_nb(c: tp.Union[OrderContext, PostOrderContext, SignalContext]) -> tp.Record:
    """Get the last order in the current column."""
    if not any_order_nb(c):
        raise ValueError("There are no orders. Use any_order_nb to check for any orders first.")
    return c.order_records[c.order_counts[c.col] - 1, c.col]
