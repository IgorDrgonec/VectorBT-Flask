# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Numba-compiled helper functions for portfolio simulation."""

from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb import records as pf_records_nb
from vectorbtpro.records import nb as records_nb


@register_jitted
def in_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the current column is in a position."""
    return c.last_position[c.col] != 0


@register_jitted
def in_long_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the current column is in a long position."""
    return c.last_position[c.col] > 0


@register_jitted
def in_short_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> bool:
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
        PostSignalContext,
    ]
) -> int:
    """Get the number of active positions in the current group (regardless of cash sharing)."""
    n_active_positions = 0
    for col in range(c.from_col, c.to_col):
        if c.last_position[col] != 0:
            n_active_positions += 1
    return n_active_positions


@register_jitted
def get_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return c.last_cash[c.group]
    return c.last_cash[c.col]


@register_jitted
def get_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get free cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return c.last_free_cash[c.group]
    return c.last_free_cash[c.col]


@register_jitted
def has_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the current column or group with cash sharing has free cash."""
    return get_free_cash_nb(c) > 0


@register_jitted
def any_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether there is any order in the current column."""
    return c.order_counts[c.col] > 0


@register_jitted
def get_last_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> tp.Record:
    """Get the last order in the current column."""
    if not any_order_nb(c):
        raise ValueError("There are no orders. Use any_order_nb to check for any orders first.")
    return c.order_records[c.order_counts[c.col] - 1, c.col]


@register_jitted
def get_asset_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get asset value of the current column."""
    if c.last_position[c.col] == 0:
        return 0.0
    return c.last_position[c.col] * c.last_val_price[c.col]


@register_jitted
def get_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get value of the current column or group with cash sharing."""
    if c.cash_sharing:
        return c.last_value[c.group]
    return c.last_value[c.col]


@register_jitted
def get_allocation_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get asset value of the current column."""
    if c.last_position[c.col] == 0:
        return 0.0
    return get_asset_value_nb(c) / get_value_nb(c)


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
    return order_reversed_position_nb(c) or (c.position_before == 0 and c.last_position[c.col] != 0)


@register_jitted
def order_increased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has increased an existing position."""
    return np.sign(c.last_position[c.col]) == np.sign(c.position_before) and abs(c.last_position[c.col]) > abs(
        c.position_before
    )


@register_jitted
def order_decreased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has decreased an existing position."""
    return (
        order_closed_position_nb(c)
        or order_reversed_position_nb(c)
        or (
            np.sign(c.last_position[c.col]) == np.sign(c.position_before)
            and abs(c.last_position[c.col]) < abs(c.position_before)
        )
    )


@register_jitted
def order_closed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has closed out an existing position."""
    return c.position_before != 0 and c.last_position[c.col] == 0


@register_jitted
def order_reversed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has reversed an existing position."""
    return (
        c.position_before != 0
        and c.last_position[c.col] != 0
        and np.sign(c.position_before) != np.sign(c.last_position[c.col])
    )


@register_jitted
def get_limit_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get limit target price."""
    if not in_position_nb(c):
        return np.nan
    limit_info = c.last_limit_info[c.col]
    return get_limit_info_target_price_nb(limit_info)


@register_jitted
def get_sl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get SL target price."""
    if not in_position_nb(c):
        return np.nan
    sl_info = c.last_sl_info[c.col]
    return get_sl_info_target_price_nb(sl_info, c.last_position[c.col])


@register_jitted
def get_tsl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get TSL/TTP target price."""
    if not in_position_nb(c):
        return np.nan
    tsl_info = c.last_tsl_info[c.col]
    return get_tsl_info_target_price_nb(tsl_info, c.last_position[c.col])


@register_jitted
def get_tp_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ]
) -> float:
    """Get TP target price."""
    if not in_position_nb(c):
        return np.nan
    tp_info = c.last_tp_info[c.col]
    return get_tp_info_target_price_nb(tp_info, c.last_position[c.col])


@register_jitted
def get_order_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ]
) -> tp.Array1d:
    """Get order records up to this point."""
    return c.order_records[: c.order_counts[c.col], c.col]


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
) -> tp.Array1d:
    """Get entry trade records up to this point."""
    order_records = get_order_records_nb(c)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    entry_trades = pf_records_nb.get_entry_trades_nb(
        order_records,
        c.close[: c.i + 1],
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
) -> tp.Array1d:
    """Get exit trade records up to this point."""
    order_records = get_order_records_nb(c)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    exit_trades = pf_records_nb.get_exit_trades_nb(
        order_records,
        c.close[: c.i + 1],
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
) -> tp.Array1d:
    """Get position records up to this point."""
    exit_trade_records = get_exit_trade_records_nb(c, init_position=init_position, init_price=init_price)
    col_map = records_nb.col_map_nb(exit_trade_records["col"], c.target_shape[1])
    position_records = pf_records_nb.get_positions_nb(exit_trade_records, col_map)
    return position_records
