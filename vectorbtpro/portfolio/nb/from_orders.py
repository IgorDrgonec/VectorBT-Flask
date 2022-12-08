# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio modeling based on orders."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns.nb import get_return_nb
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        open=base_ch.flex_array_gl_slicer,
        high=base_ch.flex_array_gl_slicer,
        low=base_ch.flex_array_gl_slicer,
        close=base_ch.flex_array_gl_slicer,
        init_cash=base_ch.FlexArraySlicer(),
        init_position=base_ch.flex_1d_array_gl_slicer,
        init_price=base_ch.flex_1d_array_gl_slicer,
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_earnings=base_ch.flex_array_gl_slicer,
        cash_dividends=base_ch.flex_array_gl_slicer,
        size=base_ch.flex_array_gl_slicer,
        price=base_ch.flex_array_gl_slicer,
        size_type=base_ch.flex_array_gl_slicer,
        direction=base_ch.flex_array_gl_slicer,
        fees=base_ch.flex_array_gl_slicer,
        fixed_fees=base_ch.flex_array_gl_slicer,
        slippage=base_ch.flex_array_gl_slicer,
        min_size=base_ch.flex_array_gl_slicer,
        max_size=base_ch.flex_array_gl_slicer,
        size_granularity=base_ch.flex_array_gl_slicer,
        reject_prob=base_ch.flex_array_gl_slicer,
        price_area_vio_mode=base_ch.flex_array_gl_slicer,
        lock_cash=base_ch.flex_array_gl_slicer,
        allow_partial=base_ch.flex_array_gl_slicer,
        raise_reject=base_ch.flex_array_gl_slicer,
        log=base_ch.flex_array_gl_slicer,
        val_price=base_ch.flex_array_gl_slicer,
        from_ago=base_ch.flex_array_gl_slicer,
        call_seq=base_ch.array_gl_slicer,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        fill_returns=None,
        max_orders=None,
        max_logs=None,
        skipna=None,
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(cache=True, tags={"can_parallel"})
def simulate_from_orders_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    open: tp.FlexArray2d = np.array([[np.nan]]),
    high: tp.FlexArray2d = np.array([[np.nan]]),
    low: tp.FlexArray2d = np.array([[np.nan]]),
    close: tp.FlexArray2d = np.array([[np.nan]]),
    init_cash: tp.FlexArray1d = np.array([100.0]),
    init_position: tp.FlexArray1d = np.array([0.0]),
    init_price: tp.FlexArray1d = np.array([np.nan]),
    cash_deposits: tp.FlexArray2d = np.array([[0.0]]),
    cash_earnings: tp.FlexArray2d = np.array([[0.0]]),
    cash_dividends: tp.FlexArray2d = np.array([[0.0]]),
    size: tp.FlexArray2d = np.array([[np.inf]]),
    price: tp.FlexArray2d = np.array([[np.inf]]),
    size_type: tp.FlexArray2d = np.array([[SizeType.Amount]]),
    direction: tp.FlexArray2d = np.array([[Direction.Both]]),
    fees: tp.FlexArray2d = np.array([[0.0]]),
    fixed_fees: tp.FlexArray2d = np.array([[0.0]]),
    slippage: tp.FlexArray2d = np.array([[0.0]]),
    min_size: tp.FlexArray2d = np.array([[np.nan]]),
    max_size: tp.FlexArray2d = np.array([[np.nan]]),
    size_granularity: tp.FlexArray2d = np.array([[np.nan]]),
    reject_prob: tp.FlexArray2d = np.array([[0.0]]),
    price_area_vio_mode: tp.FlexArray2d = np.array([[PriceAreaVioMode.Ignore]]),
    lock_cash: tp.FlexArray2d = np.array([[False]]),
    allow_partial: tp.FlexArray2d = np.array([[True]]),
    raise_reject: tp.FlexArray2d = np.array([[False]]),
    log: tp.FlexArray2d = np.array([[False]]),
    val_price: tp.FlexArray2d = np.array([[np.inf]]),
    from_ago: tp.FlexArray2d = np.array([[0]]),
    call_seq: tp.Optional[tp.Array2d] = None,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_returns: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    skipna: bool = False,
) -> SimulationOutput:
    """Creates on order out of each element.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

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
    if ffill_val_price and skipna:
        raise ValueError("Cannot skip NaN and forward-fill valuation price simultaneously")
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)
    track_cash_deposits = np.any(cash_deposits)
    if track_cash_deposits:
        if skipna:
            raise ValueError("Cannot skip NaN and track cash deposits simultaneously")
        cash_deposits_out = np.full((target_shape[0], len(group_lens)), 0.0, dtype=np.float_)
    else:
        cash_deposits_out = np.full((1, 1), 0.0, dtype=np.float_)
    track_cash_earnings = np.any(cash_earnings) or np.any(cash_dividends)
    if track_cash_earnings:
        if skipna:
            raise ValueError("Cannot skip NaN and track cash earnings simultaneously")
        cash_earnings_out = np.full(target_shape, 0.0, dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=np.float_)

    if fill_returns:
        if skipna:
            raise ValueError("Cannot skip NaN and fill returns simultaneously")
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
            if skipna:
                skip = True
                for c in range(group_len):
                    col = from_col + c
                    _i = i - abs(flex_select_nb(from_ago, i, col))
                    if _i < 0:
                        continue
                    if not np.isnan(flex_select_nb(size, _i, col)):
                        skip = False
                        break
                if skip:
                    continue

            # Add cash
            _cash_deposits = flex_select_nb(cash_deposits, i, group)
            if _cash_deposits < 0:
                _cash_deposits = max(_cash_deposits, -cash_now)
            cash_now += _cash_deposits
            free_cash_now += _cash_deposits
            if track_cash_deposits:
                cash_deposits_out[i, group] += _cash_deposits

            for c in range(group_len):
                col = from_col + c

                # Update valuation price using current open
                _open = flex_select_nb(open, i, col)
                if not np.isnan(_open) or not ffill_val_price:
                    last_val_price[col] = _open

                # Resolve valuation price
                _val_price = flex_select_nb(val_price, i, col)
                if np.isinf(_val_price):
                    if _val_price > 0:
                        _i = i - abs(flex_select_nb(from_ago, i, col))
                        if _i < 0:
                            _price = np.nan
                        else:
                            _price = flex_select_nb(price, _i, col)
                        if np.isinf(_price):
                            if _price > 0:
                                _price = flex_select_nb(close, i, col)
                            else:
                                _price = _open
                        _val_price = _price
                    else:
                        _val_price = last_val_price[col]
                if not np.isnan(_val_price) or not ffill_val_price:
                    last_val_price[col] = _val_price

            # Calculate group value and rearrange if cash sharing is enabled
            if cash_sharing:
                # Same as get_ctx_group_value_nb but with flexible indexing
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
                        _i = i - abs(flex_select_nb(from_ago, i, col))
                        if _i < 0:
                            temp_order_value[c] = 0.0
                        else:
                            temp_order_value[c] = approx_order_value_nb(
                                exec_state,
                                flex_select_nb(size, _i, col),
                                flex_select_nb(size_type, _i, col),
                                flex_select_nb(direction, _i, col),
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
                _i = i - abs(flex_select_nb(from_ago, i, col))
                if _i < 0:
                    continue
                order = order_nb(
                    size=flex_select_nb(size, _i, col),
                    price=flex_select_nb(price, _i, col),
                    size_type=flex_select_nb(size_type, _i, col),
                    direction=flex_select_nb(direction, _i, col),
                    fees=flex_select_nb(fees, _i, col),
                    fixed_fees=flex_select_nb(fixed_fees, _i, col),
                    slippage=flex_select_nb(slippage, _i, col),
                    min_size=flex_select_nb(min_size, _i, col),
                    max_size=flex_select_nb(max_size, _i, col),
                    size_granularity=flex_select_nb(size_granularity, _i, col),
                    reject_prob=flex_select_nb(reject_prob, _i, col),
                    price_area_vio_mode=flex_select_nb(price_area_vio_mode, _i, col),
                    lock_cash=flex_select_nb(lock_cash, _i, col),
                    allow_partial=flex_select_nb(allow_partial, _i, col),
                    raise_reject=flex_select_nb(raise_reject, _i, col),
                    log=flex_select_nb(log, _i, col),
                )

                # Process the order
                price_area = PriceArea(
                    open=flex_select_nb(open, i, col),
                    high=flex_select_nb(high, i, col),
                    low=flex_select_nb(low, i, col),
                    close=flex_select_nb(close, i, col),
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
                _close = flex_select_nb(close, i, col)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_nb(cash_earnings, i, col)
                _cash_dividends = flex_select_nb(cash_dividends, i, col)
                _cash_earnings += _cash_dividends * last_position[col]
                if _cash_earnings < 0:
                    _cash_earnings = max(_cash_earnings, -cash_now)
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
        cash_deposits=cash_deposits_out,
        cash_earnings=cash_earnings_out,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )
