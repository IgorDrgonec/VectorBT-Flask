# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio modeling based on orders."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns.nb import get_return_nb
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        call_seq=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        init_cash=ch.ArraySlicer(axis=0),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
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
        fill_returns=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(cache=True, tags={"can_parallel"})
def simulate_from_orders_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    call_seq: tp.Optional[tp.Array2d] = None,
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(np.nan),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    cash_dividends: tp.FlexArray = np.asarray(0.0),
    size: tp.FlexArray = np.asarray(np.inf),
    price: tp.FlexArray = np.asarray(np.inf),
    size_type: tp.FlexArray = np.asarray(SizeType.Amount),
    direction: tp.FlexArray = np.asarray(Direction.Both),
    fees: tp.FlexArray = np.asarray(0.0),
    fixed_fees: tp.FlexArray = np.asarray(0.0),
    slippage: tp.FlexArray = np.asarray(0.0),
    min_size: tp.FlexArray = np.asarray(0.0),
    max_size: tp.FlexArray = np.asarray(np.inf),
    size_granularity: tp.FlexArray = np.asarray(np.nan),
    reject_prob: tp.FlexArray = np.asarray(0.0),
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
    fill_returns: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = True,
) -> SimulationOutput:
    """Creates on order out of each element.

    Iterates in the column-major order.
    Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).

    Usage:
        * Buy and hold using all cash and closing price (default):

        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.records.nb import col_map_nb
        >>> from vectorbtpro.portfolio.nb import simulate_from_orders_nb, asset_flow_nb

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
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    order_records, log_records = prepare_records_nb(target_shape, max_orders, max_logs)
    last_cash = prepare_last_cash_nb(target_shape, group_lens, cash_sharing, init_cash)
    last_position = prepare_last_position_nb(target_shape, init_position)
    last_value = prepare_last_value_nb(
        target_shape,
        group_lens,
        cash_sharing,
        init_cash,
        init_position=init_position,
        init_price=init_price,
    )

    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)
    track_cash_earnings = np.any(cash_earnings) or np.any(cash_dividends)
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0.0, dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=np.float_)

    if fill_returns:
        returns = np.empty((target_shape[0], len(group_lens)), dtype=np.float_)
    else:
        returns = np.empty((0, 0), dtype=np.float_)
    in_outputs = FSInOutputs(returns=returns)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

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

            for c in range(group_len):
                col = from_col + c

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
                for c in range(group_len):
                    col = from_col + c

                    if last_position[col] != 0:
                        value_now += last_position[col] * last_val_price[col]

                # Dynamically sort by order value -> selling comes first to release funds early
                if call_seq is None:
                    for c in range(group_len):
                        temp_call_seq[c] = c
                    call_seq_now = temp_call_seq[:group_len]
                else:
                    call_seq_now = call_seq[i, from_col:to_col]
                if auto_call_seq:
                    # Same as sort_by_order_value_ctx_nb but with flexible indexing
                    for c in range(group_len):
                        col = from_col + c
                        exec_state = ExecState(
                            cash=cash_now,
                            position=last_position[col],
                            debt=last_debt[col],
                            free_cash=free_cash_now,
                            val_price=last_val_price[col],
                            value=value_now,
                        )
                        temp_order_value[c] = approx_order_value_nb(
                            exec_state,
                            flex_select_auto_nb(size, i, col, flex_2d),
                            flex_select_auto_nb(size_type, i, col, flex_2d),
                            flex_select_auto_nb(direction, i, col, flex_2d),
                        )
                        if call_seq_now[c] != c:
                            raise ValueError("Call sequence must follow CallSeqType.Default")

                    # Sort by order value
                    insert_argsort_nb(temp_order_value[:group_len], call_seq_now)

            for k in range(group_len):
                if cash_sharing:
                    c = call_seq_now[k]
                    if c >= group_len:
                        raise ValueError("Call index out of bounds of the group")
                else:
                    c = k
                col = from_col + c

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
                    log=flex_select_auto_nb(log, i, col, flex_2d),
                )

                # Process the order
                price_area = PriceArea(
                    open=flex_select_auto_nb(open, i, col, flex_2d),
                    high=flex_select_auto_nb(high, i, col, flex_2d),
                    low=flex_select_auto_nb(low, i, col, flex_2d),
                    close=flex_select_auto_nb(close, i, col, flex_2d),
                )
                exec_state = ExecState(
                    cash=cash_now,
                    position=position_now,
                    debt=debt_now,
                    free_cash=free_cash_now,
                    val_price=val_price_now,
                    value=value_now,
                )
                new_exec_state, order_result = process_order_nb(
                    group=group,
                    col=col,
                    i=i,
                    exec_state=exec_state,
                    order=order,
                    price_area=price_area,
                    update_value=update_value,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                )

                # Update execution state
                cash_now = new_exec_state.cash
                position_now = new_exec_state.position
                debt_now = new_exec_state.debt
                free_cash_now = new_exec_state.free_cash
                val_price_now = new_exec_state.val_price
                value_now = new_exec_state.value

                # Now becomes last
                last_position[col] = position_now
                last_debt[col] = debt_now
                if not np.isnan(val_price_now) or not ffill_val_price:
                    last_val_price[col] = val_price_now

            group_value = cash_now
            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                _cash_dividends = flex_select_auto_nb(cash_dividends, i, col, flex_2d)
                _cash_earnings += _cash_dividends * last_position[col]
                cash_now += _cash_earnings
                free_cash_now += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings

                # Update previous value, current value, and return
                if fill_returns:
                    if cash_sharing:
                        if last_position[col] != 0:
                            group_value += last_position[col] * last_val_price[col]
                    else:
                        if last_position[col] == 0:
                            last_value[col] = cash_now
                        else:
                            last_value[col] = cash_now + last_position[col] * last_val_price[col]
                        last_return[col] = get_return_nb(
                            prev_close_value[col],
                            last_value[col] - _cash_deposits,
                        )
                        prev_close_value[col] = last_value[col]
                        in_outputs.returns[i, group] = last_return[col]

            if fill_returns and cash_sharing:
                last_value[group] = group_value
                last_return[group] = get_return_nb(
                    prev_close_value[group],
                    last_value[group] - _cash_deposits,
                )
                prev_close_value[group] = last_value[group]
                in_outputs.returns[i, group] = last_return[group]

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )
