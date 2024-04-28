# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled helper functions for portfolio simulation."""

from vectorbtpro.base.flex_indexing import flex_select_col_nb
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb import records as pf_records_nb
from vectorbtpro.records import nb as records_nb


# ############# Position ############# #


@register_jitted
def get_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Get position of the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_position[_col]


@register_jitted
def in_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Check whether the current column is in a position."""
    position = get_position_nb(c, col=col)
    return position != 0


@register_jitted
def in_long_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Check whether the current column is in a long position."""
    position = get_position_nb(c, col=col)
    return position > 0


@register_jitted
def in_short_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Check whether the current column is in a short position."""
    position = get_position_nb(c, col=col)
    return position < 0


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
        PostSignalContext,
    ],
    all_groups: bool = False,
) -> int:
    """Get the number of active positions in the current group (regardless of cash sharing).

    To calculate across all groups, set `all_groups` to True."""
    n_active_positions = 0
    if all_groups:
        for col in range(c.target_shape[1]):
            if c.last_position[col] != 0:
                n_active_positions += 1
    else:
        for col in range(c.from_col, c.to_col):
            if c.last_position[col] != 0:
                n_active_positions += 1
    return n_active_positions


# ############# Cash ############# #


@register_jitted
def get_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col_or_group: tp.Optional[int] = None,
) -> float:
    """Get cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        if col_or_group is None:
            group = c.group
        else:
            group = col_or_group
        return c.last_cash[group]

    if col_or_group is None:
        col = c.col
    else:
        col = col_or_group
    return c.last_cash[col]


# ############# Debt ############# #


@register_jitted
def get_debt_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Get debt of the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_debt[_col]


# ############# Locked cash ############# #


@register_jitted
def get_locked_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Get locked cash of the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_locked_cash[_col]


# ############# Free cash ############# #


@register_jitted
def get_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col_or_group: tp.Optional[int] = None,
) -> float:
    """Get free cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        if col_or_group is None:
            group = c.group
        else:
            group = col_or_group
        return c.last_free_cash[group]

    if col_or_group is None:
        col = c.col
    else:
        col = col_or_group
    return c.last_free_cash[col]


@register_jitted
def has_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col_or_group: tp.Optional[int] = None,
) -> bool:
    """Check whether the current column or group with cash sharing has free cash."""
    return get_free_cash_nb(c, col_or_group=col_or_group) > 0


# ############# Valuation price ############# #


@register_jitted
def get_val_price_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Get valuation price of the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_val_price[_col]


# ############# Value ############# #


@register_jitted
def get_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col_or_group: tp.Optional[int] = None,
) -> float:
    """Get value of the current column or group with cash sharing."""
    if c.cash_sharing:
        if col_or_group is None:
            group = c.group
        else:
            group = col_or_group
        return c.last_value[group]

    if col_or_group is None:
        col = c.col
    else:
        col = col_or_group
    return c.last_value[col]


# ############# Leverage ############# #


@register_jitted
def get_leverage_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get leverage of the current column."""
    position = get_position_nb(c, col=col)
    debt = get_debt_nb(c, col=col)
    locked_cash = get_locked_cash_nb(c, col=col)
    if locked_cash == 0:
        return np.nan
    leverage = debt / locked_cash
    if position > 0:
        leverage += 1
    return leverage


# ############# Allocation ############# #


@register_jitted
def get_position_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get position value of the current column."""
    position = get_position_nb(c, col=col)
    val_price = get_val_price_nb(c, col=col)
    if position:
        return 0.0
    return position * val_price


@register_jitted
def get_allocation_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
    col_or_group: tp.Optional[int] = None,
) -> float:
    """Get asset value of the current column."""
    position_value = get_position_value_nb(c, col=col)
    value = get_value_nb(c, col_or_group=col_or_group)
    if position_value == 0:
        return 0.0
    if value <= 0:
        return np.nan
    return position_value / value


# ############# Orders ############# #


@register_jitted
def get_order_count_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> int:
    """Get number of order records for the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.order_counts[_col]


@register_jitted
def get_order_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.RecordArray:
    """Get order records for the current column."""
    if col is None:
        _col = c.col
    else:
        _col = col
    order_count = get_order_count_nb(c, col=col)
    return c.order_records[:order_count, _col]


@register_jitted
def any_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> bool:
    """Check whether there is any order in the current column."""
    return get_order_count_nb(c, col=col) > 0


@register_jitted
def get_last_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.Record:
    """Get the last order in the current column."""
    if not any_order_nb(c, col=col):
        raise ValueError("There are no orders. Use any_order_nb to check for any orders first.")
    return get_order_records_nb(c, col=col)[-1]


# ############# Order result ############# #


@register_jitted
def order_filled_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order was filled."""
    return c.order_result.status == OrderStatus.Filled


@register_jitted
def order_opened_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has opened a new position."""
    position_now = get_position_nb(c)
    return order_reversed_position_nb(c) or (c.position_before == 0 and position_now != 0)


@register_jitted
def order_increased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has opened or increased an existing position."""
    position_now = get_position_nb(c)
    return order_opened_position_nb(c) or (
        np.sign(position_now) == np.sign(c.position_before) and abs(position_now) > abs(c.position_before)
    )


@register_jitted
def order_decreased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has decreased or closed an existing position."""
    position_now = get_position_nb(c)
    return (
        order_closed_position_nb(c)
        or order_reversed_position_nb(c)
        or (np.sign(position_now) == np.sign(c.position_before) and abs(position_now) < abs(c.position_before))
    )


@register_jitted
def order_closed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has closed out an existing position."""
    position_now = get_position_nb(c)
    return c.position_before != 0 and position_now == 0


@register_jitted
def order_reversed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has reversed an existing position."""
    position_now = get_position_nb(c)
    return c.position_before != 0 and position_now != 0 and np.sign(c.position_before) != np.sign(position_now)


# ############# Limit orders ############# #


@register_jitted
def get_limit_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.Record:
    """Get limit information."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_limit_info[_col]


@register_jitted
def get_limit_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get limit target price."""
    if not in_position_nb(c, col=col):
        return np.nan
    limit_info = get_limit_info_nb(c, col=col)
    return get_limit_info_target_price_nb(limit_info)


# ############# Stop orders ############# #


@register_jitted
def get_sl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.Record:
    """Get SL information."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_sl_info[_col]


@register_jitted
def get_sl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get SL target price."""
    if not in_position_nb(c, col=col):
        return np.nan
    position = get_position_nb(c, col=col)
    sl_info = get_sl_info_nb(c, col=col)
    return get_sl_info_target_price_nb(sl_info, position)


@register_jitted
def get_tsl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.Record:
    """Get TSL/TTP information."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_tsl_info[_col]


@register_jitted
def get_tsl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get SL/TTP target price."""
    if not in_position_nb(c, col=col):
        return np.nan
    position = get_position_nb(c, col=col)
    tsl_info = get_tsl_info_nb(c, col=col)
    return get_tsl_info_target_price_nb(tsl_info, position)


@register_jitted
def get_tp_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> tp.Record:
    """Get TP information."""
    if col is None:
        _col = c.col
    else:
        _col = col
    return c.last_tp_info[_col]


@register_jitted
def get_tp_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
    col: tp.Optional[int] = None,
) -> float:
    """Get TP target price."""
    if not in_position_nb(c, col=col):
        return np.nan
    position = get_position_nb(c, col=col)
    tp_info = get_tp_info_nb(c, col=col)
    return get_tp_info_target_price_nb(tp_info, position)


# ############# Trades ############# #


@register_jitted
def get_entry_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    col: tp.Optional[int] = None,
) -> tp.Array1d:
    """Get entry trade records up to this point."""
    if col is None:
        _col = c.col
    else:
        _col = col
    order_records = get_order_records_nb(c, col=col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, _col)
    entry_trades = pf_records_nb.get_entry_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return entry_trades


@register_jitted
def get_exit_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    col: tp.Optional[int] = None,
) -> tp.Array1d:
    """Get exit trade records up to this point."""
    if col is None:
        _col = c.col
    else:
        _col = col
    order_records = get_order_records_nb(c, col=col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, _col)
    exit_trades = pf_records_nb.get_exit_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return exit_trades


@register_jitted
def get_position_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    col: tp.Optional[int] = None,
) -> tp.Array1d:
    """Get position records up to this point."""
    exit_trade_records = get_exit_trade_records_nb(
        c,
        init_position=init_position,
        init_price=init_price,
        col=col,
    )
    col_map = records_nb.col_map_nb(exit_trade_records["col"], c.target_shape[1])
    position_records = pf_records_nb.get_positions_nb(exit_trade_records, col_map)
    return position_records


# ############# Simulation ############# #


@register_jitted
def stop_group_sim_nb(
    c: tp.Union[
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
        PostSignalContext,
    ],
    group: tp.Optional[int] = None,
) -> None:
    """Stop the simulation of the current group."""
    if group is None:
        _group = c.group
    else:
        _group = group
    c.sim_end[_group] = c.i + 1
