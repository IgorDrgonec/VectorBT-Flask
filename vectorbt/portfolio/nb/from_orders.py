# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Numba-compiled functions for `vectorbt.portfolio.base.Portfolio.from_orders`."""

from numba import prange

from vectorbt.base import chunking as base_ch
from vectorbt.ch_registry import register_chunkable
from vectorbt.portfolio import chunking as portfolio_ch
from vectorbt.portfolio.nb.core import *
from vectorbt.utils import chunking as ch
from vectorbt.utils.array_ import insert_argsort_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query='group_lens', axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        call_seq=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        init_cash=ch.ArraySlicer(axis=0),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_earnings=base_ch.FlexArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        cash_dividends=base_ch.FlexArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        size=portfolio_ch.flex_array_gl_slicer,
        price=portfolio_ch.flex_array_gl_slicer,
        size_type=portfolio_ch.flex_array_gl_slicer,
        direction=portfolio_ch.flex_array_gl_slicer,
        fees=portfolio_ch.flex_array_gl_slicer,
        fixed_fees=portfolio_ch.flex_array_gl_slicer,
        slippage=portfolio_ch.flex_array_gl_slicer,
        min_size=portfolio_ch.flex_array_gl_slicer,
        max_size=portfolio_ch.flex_array_gl_slicer,
        size_granularity=portfolio_ch.flex_array_gl_slicer,
        reject_prob=portfolio_ch.flex_array_gl_slicer,
        price_area_vio_mode=portfolio_ch.flex_array_gl_slicer,
        lock_cash=portfolio_ch.flex_array_gl_slicer,
        allow_partial=portfolio_ch.flex_array_gl_slicer,
        raise_reject=portfolio_ch.flex_array_gl_slicer,
        log=portfolio_ch.flex_array_gl_slicer,
        val_price=portfolio_ch.flex_array_gl_slicer,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(cache=True, tags={'can_parallel'})
def simulate_from_orders_nb(target_shape: tp.Shape,
                            group_lens: tp.Array1d,
                            call_seq: tp.Array2d,
                            init_cash: tp.FlexArray = np.asarray(100.),
                            init_position: tp.FlexArray = np.asarray(0.),
                            cash_deposits: tp.FlexArray = np.asarray(0.),
                            cash_earnings: tp.FlexArray = np.asarray(0.),
                            cash_dividends: tp.FlexArray = np.asarray(0.),
                            size: tp.FlexArray = np.asarray(np.inf),
                            price: tp.FlexArray = np.asarray(np.inf),
                            size_type: tp.FlexArray = np.asarray(SizeType.Amount),
                            direction: tp.FlexArray = np.asarray(Direction.Both),
                            fees: tp.FlexArray = np.asarray(0.),
                            fixed_fees: tp.FlexArray = np.asarray(0.),
                            slippage: tp.FlexArray = np.asarray(0.),
                            min_size: tp.FlexArray = np.asarray(0.),
                            max_size: tp.FlexArray = np.asarray(np.inf),
                            size_granularity: tp.FlexArray = np.asarray(np.nan),
                            reject_prob: tp.FlexArray = np.asarray(0.),
                            price_area_vio_mode: tp.FlexArray = np.asarray(PriceAreaVioMode.Ignore),
                            lock_cash: tp.FlexArray = np.asarray(False),
                            allow_partial: tp.FlexArray = np.asarray(True),
                            raise_reject: tp.FlexArray = np.asarray(False),
                            log: tp.FlexArray = np.asarray(False),
                            val_price: tp.FlexArray = np.asarray(np.inf),
                            open: tp.FlexArray = np.asarray(np.nan),
                            high: tp.FlexArray = np.asarray(np.nan),
                            low: tp.FlexArray = np.asarray(np.nan),
                            close: tp.FlexArray = np.asarray(np.nan),
                            auto_call_seq: bool = False,
                            ffill_val_price: bool = True,
                            update_value: bool = False,
                            max_orders: tp.Optional[int] = None,
                            max_logs: tp.Optional[int] = 0,
                            flex_2d: bool = True) -> SimulationOutput:
    """Creates on order out of each element.

    Iterates in the column-major order.
    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).

    ## Example

    Buy and hold using all cash and closing price (default):

    ```python-repl
    >>> import numpy as np
    >>> from vectorbt.records.nb import col_map_nb
    >>> from vectorbt.portfolio.nb import simulate_from_orders_nb, asset_flow_nb

    >>> close = np.array([1, 2, 3, 4, 5])[:, None]
    >>> sim_out = simulate_from_orders_nb(
    ...     target_shape=close.shape,
    ...     group_lens=np.array([1]),
    ...     call_seq=np.full(close.shape, 0),
    ...     close=close
    ... )
    >>> col_map = col_map_nb(sim_out.order_records['col'], close.shape[1])
    >>> asset_flow = asset_flow_nb(close.shape, sim_out.order_records, col_map)
    >>> asset_flow
    array([[100.],
           [  0.],
           [  0.],
           [  0.],
           [  0.]])
    ```"""
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    order_records, log_records = prepare_records_nb(target_shape, max_orders, max_logs)
    last_cash = prepare_last_cash_nb(target_shape, group_lens, cash_sharing, init_cash)
    last_position = prepare_last_position_nb(target_shape, init_position)

    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0., dtype=np.float_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)
    last_oidx = np.full(target_shape[1], -1, dtype=np.int_)
    last_lidx = np.full(target_shape[1], -1, dtype=np.int_)
    track_cash_earnings = np.any(cash_earnings) or np.any(cash_dividends)
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0., dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0., dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col
        cash_now = last_cash[group]
        free_cash_now = cash_now

        for i in range(target_shape[0]):
            # Add cash
            _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
            cash_now += _cash_deposits
            free_cash_now += _cash_deposits

            for k in range(group_len):
                col = from_col + k

                # Update valuation price using current open
                _open = flex_select_auto_nb(open, i, col, flex_2d)
                if not np.isnan(_open) or not ffill_val_price:
                    last_val_price[col] = _open

                # Resolve valuation price
                _val_price = flex_select_auto_nb(val_price, i, col, flex_2d)
                if np.isinf(_val_price):
                    if _val_price > 0:
                        _price = flex_select_auto_nb(price, i, col, flex_2d)
                        if np.isinf(_price):
                            if _price > 0:
                                _price = flex_select_auto_nb(close, i, col, flex_2d)
                            else:
                                _price = _open
                        _val_price = _price
                    else:
                        _val_price = last_val_price[col]
                if not np.isnan(_val_price) or not ffill_val_price:
                    last_val_price[col] = _val_price

            # Calculate group value and rearrange if cash sharing is enabled
            if cash_sharing:
                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k

                    if last_position[col] != 0:
                        value_now += last_position[col] * last_val_price[col]

                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for k in range(group_len):
                        col = from_col + k
                        temp_order_value[k] = approx_order_value_nb(
                            flex_select_auto_nb(size, i, col, flex_2d),
                            flex_select_auto_nb(size_type, i, col, flex_2d),
                            flex_select_auto_nb(direction, i, col, flex_2d),
                            cash_now,
                            last_position[col],
                            free_cash_now,
                            last_val_price[col],
                            value_now
                        )

                    # Sort by order value
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

            for k in range(group_len):
                col = from_col + k
                if cash_sharing:
                    col_i = call_seq[i, col]
                    if col_i >= group_len:
                        raise ValueError("Call index out of bounds of the group")
                    col = from_col + col_i

                # Get current values per column
                position_now = last_position[col]
                debt_now = last_debt[col]
                val_price_now = last_val_price[col]
                if not cash_sharing:
                    value_now = cash_now
                    if position_now != 0:
                        value_now += position_now * val_price_now

                # Generate the next order
                order = order_nb(
                    size=flex_select_auto_nb(size, i, col, flex_2d),
                    price=flex_select_auto_nb(price, i, col, flex_2d),
                    size_type=flex_select_auto_nb(size_type, i, col, flex_2d),
                    direction=flex_select_auto_nb(direction, i, col, flex_2d),
                    fees=flex_select_auto_nb(fees, i, col, flex_2d),
                    fixed_fees=flex_select_auto_nb(fixed_fees, i, col, flex_2d),
                    slippage=flex_select_auto_nb(slippage, i, col, flex_2d),
                    min_size=flex_select_auto_nb(min_size, i, col, flex_2d),
                    max_size=flex_select_auto_nb(max_size, i, col, flex_2d),
                    size_granularity=flex_select_auto_nb(size_granularity, i, col, flex_2d),
                    reject_prob=flex_select_auto_nb(reject_prob, i, col, flex_2d),
                    price_area_vio_mode=flex_select_auto_nb(price_area_vio_mode, i, col, flex_2d),
                    lock_cash=flex_select_auto_nb(lock_cash, i, col, flex_2d),
                    allow_partial=flex_select_auto_nb(allow_partial, i, col, flex_2d),
                    raise_reject=flex_select_auto_nb(raise_reject, i, col, flex_2d),
                    log=flex_select_auto_nb(log, i, col, flex_2d)
                )

                # Process the order
                price_area = PriceArea(
                    open=flex_select_auto_nb(open, i, col, flex_2d),
                    high=flex_select_auto_nb(high, i, col, flex_2d),
                    low=flex_select_auto_nb(low, i, col, flex_2d),
                    close=flex_select_auto_nb(close, i, col, flex_2d)
                )
                state = ProcessOrderState(
                    cash=cash_now,
                    position=position_now,
                    debt=debt_now,
                    free_cash=free_cash_now,
                    val_price=val_price_now,
                    value=value_now
                )
                order_result, new_state = process_order_nb(
                    group=group,
                    col=col,
                    i=i,
                    price_area=price_area,
                    state=state,
                    update_value=update_value,
                    order=order,
                    order_records=order_records,
                    last_oidx=last_oidx,
                    log_records=log_records,
                    last_lidx=last_lidx
                )

                # Update state
                cash_now = new_state.cash
                position_now = new_state.position
                debt_now = new_state.debt
                free_cash_now = new_state.free_cash
                val_price_now = new_state.val_price
                value_now = new_state.value

                # Now becomes last
                last_position[col] = position_now
                last_debt[col] = debt_now
                if not np.isnan(val_price_now) or not ffill_val_price:
                    last_val_price[col] = val_price_now

            # Update valuation price using current close
            for col in range(from_col, to_col):
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                _cash_dividends = flex_select_auto_nb(cash_dividends, i, col, flex_2d)
                _cash_earnings += _cash_dividends * last_position[col]
                cash_now += _cash_earnings
                free_cash_now += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings

    return prepare_simout_nb(
        order_records=order_records,
        last_oidx=last_oidx,
        log_records=log_records,
        last_lidx=last_lidx,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=None
    )
