# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for `vectorbtpro.portfolio.base.Portfolio.from_signals`."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.ch_registry import register_chunkable
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb
from vectorbtpro.utils.math_ import is_less_nb


@register_jitted(cache=True)
def generate_stop_signal_nb(position_now: float,
                            upon_stop_exit: int,
                            accumulate: int) -> tp.Tuple[bool, bool, bool, bool, int]:
    """Generate stop signal and change accumulation if needed."""
    is_long_entry = False
    is_long_exit = False
    is_short_entry = False
    is_short_exit = False
    if position_now > 0:
        if upon_stop_exit == StopExitMode.Close:
            is_long_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_stop_exit == StopExitMode.CloseReduce:
            is_long_exit = True
        elif upon_stop_exit == StopExitMode.Reverse:
            is_short_entry = True
            accumulate = AccumulationMode.Disabled
        else:
            is_short_entry = True
    elif position_now < 0:
        if upon_stop_exit == StopExitMode.Close:
            is_short_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_stop_exit == StopExitMode.CloseReduce:
            is_short_exit = True
        elif upon_stop_exit == StopExitMode.Reverse:
            is_long_entry = True
            accumulate = AccumulationMode.Disabled
        else:
            is_long_entry = True
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@register_jitted(cache=True)
def resolve_stop_price_and_slippage_nb(stop_price: float,
                                       price: float,
                                       close: float,
                                       slippage: float,
                                       stop_exit_price: int) -> tp.Tuple[float, float]:
    """Resolve price and slippage of a stop order."""
    if stop_exit_price == StopExitPrice.StopMarket:
        return stop_price, slippage
    elif stop_exit_price == StopExitPrice.StopLimit:
        return stop_price, 0.
    elif stop_exit_price == StopExitPrice.Close:
        return close, slippage
    return price, slippage


@register_jitted(cache=True)
def resolve_signal_conflict_nb(position_now: float,
                               is_entry: bool,
                               is_exit: bool,
                               direction: int,
                               conflict_mode: int) -> tp.Tuple[bool, bool]:
    """Resolve any conflict between an entry and an exit."""
    if is_entry and is_exit:
        # Conflict
        if conflict_mode == ConflictMode.Entry:
            # Ignore exit signal
            is_exit = False
        elif conflict_mode == ConflictMode.Exit:
            # Ignore entry signal
            is_entry = False
        elif conflict_mode == ConflictMode.Adjacent:
            # Take the signal adjacent to the position we are in
            if position_now == 0:
                # Cannot decide -> ignore
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.Both:
                    if position_now > 0:
                        is_exit = False
                    elif position_now < 0:
                        is_entry = False
                else:
                    is_exit = False
        elif conflict_mode == ConflictMode.Opposite:
            # Take the signal opposite to the position we are in
            if position_now == 0:
                # Cannot decide -> ignore
                is_entry = False
                is_exit = False
            else:
                if direction == Direction.Both:
                    if position_now > 0:
                        is_entry = False
                    elif position_now < 0:
                        is_exit = False
                else:
                    is_entry = False
        else:
            is_entry = False
            is_exit = False
    return is_entry, is_exit


@register_jitted(cache=True)
def resolve_dir_conflict_nb(position_now: float,
                            is_long_entry: bool,
                            is_short_entry: bool,
                            upon_dir_conflict: int) -> tp.Tuple[bool, bool]:
    """Resolve any direction conflict between a long entry and a short entry."""
    if is_long_entry and is_short_entry:
        if upon_dir_conflict == DirectionConflictMode.Long:
            is_short_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Short:
            is_long_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Adjacent:
            if position_now > 0:
                is_short_entry = False
            elif position_now < 0:
                is_long_entry = False
            else:
                is_long_entry = False
                is_short_entry = False
        elif upon_dir_conflict == DirectionConflictMode.Opposite:
            if position_now > 0:
                is_long_entry = False
            elif position_now < 0:
                is_short_entry = False
            else:
                is_long_entry = False
                is_short_entry = False
        else:
            is_long_entry = False
            is_short_entry = False
    return is_long_entry, is_short_entry


@register_jitted(cache=True)
def resolve_opposite_entry_nb(position_now: float,
                              is_long_entry: bool,
                              is_long_exit: bool,
                              is_short_entry: bool,
                              is_short_exit: bool,
                              upon_opposite_entry: int,
                              accumulate: int) -> tp.Tuple[bool, bool, bool, bool, int]:
    """Resolve opposite entry."""
    if position_now > 0 and is_short_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            is_short_entry = False
        elif upon_opposite_entry == OppositeEntryMode.Close:
            is_short_entry = False
            is_long_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            is_short_entry = False
            is_long_exit = True
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            accumulate = AccumulationMode.Disabled
    if position_now < 0 and is_long_entry:
        if upon_opposite_entry == OppositeEntryMode.Ignore:
            is_long_entry = False
        elif upon_opposite_entry == OppositeEntryMode.Close:
            is_long_entry = False
            is_short_exit = True
            accumulate = AccumulationMode.Disabled
        elif upon_opposite_entry == OppositeEntryMode.CloseReduce:
            is_long_entry = False
            is_short_exit = True
        elif upon_opposite_entry == OppositeEntryMode.Reverse:
            accumulate = AccumulationMode.Disabled
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate


@register_jitted(cache=True)
def signals_to_size_nb(position_now: float,
                       is_long_entry: bool,
                       is_long_exit: bool,
                       is_short_entry: bool,
                       is_short_exit: bool,
                       size: float,
                       size_type: int,
                       accumulate: int,
                       val_price_now: float) -> tp.Tuple[float, int, int]:
    """Translate direction-aware signals into size, size type, and direction."""
    if size_type != SizeType.Amount and size_type != SizeType.Value and size_type != SizeType.Percent:
        raise ValueError("Only SizeType.Amount, SizeType.Value, and SizeType.Percent are supported")
    order_size = np.nan
    direction = Direction.Both
    abs_position_now = abs(position_now)
    if is_less_nb(size, 0):
        raise ValueError("Negative size is not allowed. Please express direction using signals.")

    if position_now > 0:
        # We're in a long position
        if is_short_entry:
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = -size
            else:
                # Reverse the position
                order_size = -abs_position_now
                if not np.isnan(size):
                    if size_type == SizeType.Percent:
                        raise ValueError(
                            "SizeType.Percent does not support position reversal using signals")
                    if size_type == SizeType.Value:
                        order_size -= size / val_price_now
                    else:
                        order_size -= size
        elif is_long_exit:
            direction = Direction.LongOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = -size
            else:
                # Close the position
                order_size = -abs_position_now
                size_type = SizeType.Amount
        elif is_long_entry:
            direction = Direction.LongOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # Increase the position
                order_size = size
    elif position_now < 0:
        # We're in a short position
        if is_long_entry:
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = size
            else:
                # Reverse the position
                order_size = abs_position_now
                if not np.isnan(size):
                    if size_type == SizeType.Percent:
                        raise ValueError("SizeType.Percent does not support position reversal using signals")
                    if size_type == SizeType.Value:
                        order_size += size / val_price_now
                    else:
                        order_size += size
        elif is_short_exit:
            direction = Direction.ShortOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.RemoveOnly:
                # Decrease the position
                order_size = size
            else:
                # Close the position
                order_size = abs_position_now
                size_type = SizeType.Amount
        elif is_short_entry:
            direction = Direction.ShortOnly
            if accumulate == AccumulationMode.Both or accumulate == AccumulationMode.AddOnly:
                # Increase the position
                order_size = -size
    else:
        if is_long_entry:
            # Open long position
            order_size = size
        elif is_short_entry:
            # Open short position
            order_size = -size

    return order_size, size_type, direction


@register_jitted(cache=True)
def should_update_stop_nb(stop: float, upon_stop_update: int) -> bool:
    """Whether to update stop."""
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if not np.isnan(stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
    return False


@register_jitted(cache=True)
def get_stop_price_nb(position_now: float,
                      stop_price: float,
                      stop: float,
                      open: float,
                      high: float,
                      low: float,
                      hit_below: bool) -> float:
    """Get stop price.

    If hit before open, returns open."""
    if stop < 0:
        raise ValueError("Stop value must be nan, 0, or greater than 0")
    if (position_now > 0 and hit_below) or (position_now < 0 and not hit_below):
        stop_price = stop_price * (1 - stop)
        if open <= stop_price:
            return open
        if low <= stop_price:
            return stop_price
        return np.nan
    if (position_now < 0 and hit_below) or (position_now > 0 and not hit_below):
        stop_price = stop_price * (1 + stop)
        if open >= stop_price:
            return open
        if high >= stop_price:
            return stop_price
        return np.nan
    return np.nan


@register_jitted
def no_signal_func_nb(c: SignalContext, *args) -> tp.Tuple[bool, bool, bool, bool]:
    """Placeholder signal function that returns no signal."""
    return False, False, False, False


@register_jitted
def no_adjust_sl_func_nb(c: AdjustSLContext, *args) -> tp.Tuple[float, bool]:
    """Placeholder function that returns the initial stop-loss value and trailing flag."""
    return c.curr_stop, c.curr_trail


@register_jitted
def no_adjust_tp_func_nb(c: AdjustTPContext, *args) -> float:
    """Placeholder function that returns the initial take-profit value."""
    return c.curr_stop


SignalFuncT = tp.Callable[[SignalContext, tp.VarArg()], tp.Tuple[bool, bool, bool, bool]]
AdjustSLFuncT = tp.Callable[[AdjustSLContext, tp.VarArg()], tp.Tuple[float, bool]]
AdjustTPFuncT = tp.Callable[[AdjustTPContext, tp.VarArg()], float]


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
        signal_func_nb=None,
        signal_args=ch.ArgsTaker(),
        size=portfolio_ch.flex_array_gl_slicer,
        price=portfolio_ch.flex_array_gl_slicer,
        size_type=portfolio_ch.flex_array_gl_slicer,
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
        accumulate=portfolio_ch.flex_array_gl_slicer,
        upon_long_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_short_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_dir_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opposite_entry=portfolio_ch.flex_array_gl_slicer,
        val_price=portfolio_ch.flex_array_gl_slicer,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        sl_stop=portfolio_ch.flex_array_gl_slicer,
        sl_trail=portfolio_ch.flex_array_gl_slicer,
        tp_stop=portfolio_ch.flex_array_gl_slicer,
        stop_entry_price=portfolio_ch.flex_array_gl_slicer,
        stop_exit_price=portfolio_ch.flex_array_gl_slicer,
        upon_stop_exit=portfolio_ch.flex_array_gl_slicer,
        upon_stop_update=portfolio_ch.flex_array_gl_slicer,
        signal_priority=portfolio_ch.flex_array_gl_slicer,
        adjust_sl_func_nb=None,
        adjust_sl_args=ch.ArgsTaker(),
        adjust_tp_func_nb=None,
        adjust_tp_args=ch.ArgsTaker(),
        use_stops=None,
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(tags={'can_parallel'})
def simulate_from_signal_func_nb(target_shape: tp.Shape,
                                 group_lens: tp.Array1d,
                                 call_seq: tp.Array2d,
                                 init_cash: tp.FlexArray = np.asarray(100.),
                                 init_position: tp.FlexArray = np.asarray(0.),
                                 cash_deposits: tp.FlexArray = np.asarray(0.),
                                 cash_earnings: tp.FlexArray = np.asarray(0.),
                                 cash_dividends: tp.FlexArray = np.asarray(0.),
                                 signal_func_nb: SignalFuncT = no_signal_func_nb,
                                 signal_args: tp.ArgsLike = (),
                                 size: tp.FlexArray = np.asarray(np.inf),
                                 price: tp.FlexArray = np.asarray(np.inf),
                                 size_type: tp.FlexArray = np.asarray(SizeType.Amount),
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
                                 accumulate: tp.FlexArray = np.asarray(AccumulationMode.Disabled),
                                 upon_long_conflict: tp.FlexArray = np.asarray(ConflictMode.Ignore),
                                 upon_short_conflict: tp.FlexArray = np.asarray(ConflictMode.Ignore),
                                 upon_dir_conflict: tp.FlexArray = np.asarray(DirectionConflictMode.Ignore),
                                 upon_opposite_entry: tp.FlexArray = np.asarray(OppositeEntryMode.ReverseReduce),
                                 val_price: tp.FlexArray = np.asarray(np.inf),
                                 open: tp.FlexArray = np.asarray(np.nan),
                                 high: tp.FlexArray = np.asarray(np.nan),
                                 low: tp.FlexArray = np.asarray(np.nan),
                                 close: tp.FlexArray = np.asarray(np.nan),
                                 sl_stop: tp.FlexArray = np.asarray(np.nan),
                                 sl_trail: tp.FlexArray = np.asarray(False),
                                 tp_stop: tp.FlexArray = np.asarray(np.nan),
                                 stop_entry_price: tp.FlexArray = np.asarray(StopEntryPrice.Close),
                                 stop_exit_price: tp.FlexArray = np.asarray(StopExitPrice.StopLimit),
                                 upon_stop_exit: tp.FlexArray = np.asarray(StopExitMode.Close),
                                 upon_stop_update: tp.FlexArray = np.asarray(StopUpdateMode.Override),
                                 signal_priority: tp.FlexArray = np.asarray(SignalPriority.Stop),
                                 adjust_sl_func_nb: AdjustSLFuncT = no_adjust_sl_func_nb,
                                 adjust_sl_args: tp.Args = (),
                                 adjust_tp_func_nb: AdjustTPFuncT = no_adjust_tp_func_nb,
                                 adjust_tp_args: tp.Args = (),
                                 use_stops: bool = True,
                                 auto_call_seq: bool = False,
                                 ffill_val_price: bool = True,
                                 update_value: bool = False,
                                 max_orders: tp.Optional[int] = None,
                                 max_logs: tp.Optional[int] = 0,
                                 flex_2d: bool = True) -> SimulationOutput:
    """Creates an order out of each element by resolving entry and exit signals returned by `signal_func_nb`.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    Signals are processed using the following pipeline:

    1. If there is a stop signal, convert it to direction-aware signals and proceed to the step 7
    2. Get direction-aware signals using `signal_func_nb`
    3. Resolve any entry and exit conflict of each direction using `resolve_signal_conflict_nb`
    4. Resolve any direction conflict using `resolve_dir_conflict_nb`
    5. Resolve an opposite entry signal scenario using `resolve_opposite_entry_nb`
    7. Convert the final signals into size, size type, and direction using `signals_to_size_nb`

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).

    ## Example

    Buy and hold using all cash and closing price (default):

    ```python-repl
    >>> import numpy as np
    >>> from vectorbtpro.records.nb import col_map_nb
    >>> from vectorbtpro.portfolio import nb
    >>> from vectorbtpro.portfolio.enums import Direction

    >>> close = np.array([1, 2, 3, 4, 5])[:, None]
    >>> sim_out = nb.simulate_from_signal_func_nb(
    ...     target_shape=close.shape,
    ...     group_lens=np.array([1]),
    ...     call_seq=np.full(close.shape, 0),
    ...     signal_func_nb=nb.dir_enex_signal_func_nb,
    ...     signal_args=(
    ...         np.asarray(True),
    ...         np.asarray(False),
    ...         np.asarray(Direction.LongOnly)
    ...     ),
    ...     close=close
    ... )
    >>> col_map = col_map_nb(sim_out.order_records['col'], close.shape[1])
    >>> asset_flow = nb.asset_flow_nb(close.shape, sim_out.order_records, col_map)
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

    if use_stops:
        sl_init_i = np.full(target_shape[1], -1, dtype=np.int_)
        sl_init_price = np.full(target_shape[1], np.nan, dtype=np.float_)
        sl_curr_i = np.full(target_shape[1], -1, dtype=np.int_)
        sl_curr_price = np.full(target_shape[1], np.nan, dtype=np.float_)
        sl_curr_stop = np.full(target_shape[1], np.nan, dtype=np.float_)
        sl_curr_trail = np.full(target_shape[1], False, dtype=np.bool_)
        tp_init_i = np.full(target_shape[1], -1, dtype=np.int_)
        tp_init_price = np.full(target_shape[1], np.nan, dtype=np.float_)
        tp_curr_stop = np.full(target_shape[1], np.nan, dtype=np.float_)
    else:
        sl_init_i = np.empty(0, dtype=np.int_)
        sl_init_price = np.empty(0, dtype=np.float_)
        sl_curr_i = np.empty(0, dtype=np.int_)
        sl_curr_price = np.empty(0, dtype=np.float_)
        sl_curr_stop = np.empty(0, dtype=np.float_)
        sl_curr_trail = np.empty(0, dtype=np.bool_)
        tp_init_i = np.empty(0, dtype=np.int_)
        tp_init_price = np.empty(0, dtype=np.float_)
        tp_curr_stop = np.empty(0, dtype=np.float_)
    price_arr = np.full(target_shape[1], np.nan, dtype=np.float_)
    size_arr = np.empty(target_shape[1], dtype=np.float_)
    size_type_arr = np.empty(target_shape[1], dtype=np.float_)
    slippage_arr = np.empty(target_shape[1], dtype=np.float_)
    direction_arr = np.empty(target_shape[1], dtype=np.int_)

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

            # Get size and value of each order
            for k in range(group_len):
                col = from_col + k  # order doesn't matter

                position_now = last_position[col]
                stop_price = np.nan
                if use_stops:
                    # Adjust stops
                    adjust_sl_ctx = AdjustSLContext(
                        i=i,
                        col=col,
                        position_now=last_position[col],
                        val_price_now=last_val_price[col],
                        init_i=sl_init_i[col],
                        init_price=sl_init_price[col],
                        curr_i=sl_curr_i[col],
                        curr_price=sl_curr_price[col],
                        curr_stop=sl_curr_stop[col],
                        curr_trail=sl_curr_trail[col]
                    )
                    sl_curr_stop[col], sl_curr_trail[col] = adjust_sl_func_nb(adjust_sl_ctx, *adjust_sl_args)
                    adjust_tp_ctx = AdjustTPContext(
                        i=i,
                        col=col,
                        position_now=last_position[col],
                        val_price_now=last_val_price[col],
                        init_i=tp_init_i[col],
                        init_price=tp_init_price[col],
                        curr_stop=tp_curr_stop[col]
                    )
                    tp_curr_stop[col] = adjust_tp_func_nb(adjust_tp_ctx, *adjust_tp_args)

                    if not np.isnan(sl_curr_stop[col]) or not np.isnan(tp_curr_stop[col]):
                        # Resolve current bar
                        _open = flex_select_auto_nb(open, i, col, flex_2d)
                        _high = flex_select_auto_nb(high, i, col, flex_2d)
                        _low = flex_select_auto_nb(low, i, col, flex_2d)
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        if np.isnan(_high):
                            if np.isnan(_open):
                                _high = _close
                            elif np.isnan(_close):
                                _high = _open
                            else:
                                _high = max(_open, _close)
                        if np.isnan(_low):
                            if np.isnan(_open):
                                _low = _close
                            elif np.isnan(_close):
                                _low = _open
                            else:
                                _low = min(_open, _close)

                        # Get stop price
                        if not np.isnan(sl_curr_stop[col]):
                            stop_price = get_stop_price_nb(
                                position_now,
                                sl_curr_price[col],
                                sl_curr_stop[col],
                                _open, _high, _low,
                                True
                            )
                        if np.isnan(stop_price) and not np.isnan(tp_curr_stop[col]):
                            stop_price = get_stop_price_nb(
                                position_now,
                                tp_init_price[col],
                                tp_curr_stop[col],
                                _open, _high, _low,
                                False
                            )

                        if not np.isnan(sl_curr_stop[col]) and sl_curr_trail[col]:
                            # Update trailing stop
                            if position_now > 0:
                                if _high > sl_curr_price[col]:
                                    sl_curr_i[col] = i
                                    sl_curr_price[col] = _high
                            elif position_now < 0:
                                if _low < sl_curr_price[col]:
                                    sl_curr_i[col] = i
                                    sl_curr_price[col] = _low

                _accumulate = flex_select_auto_nb(accumulate, i, col, flex_2d)
                stop_signal_set = False
                if use_stops and not np.isnan(stop_price):
                    # Get stop signal
                    _upon_stop_exit = flex_select_auto_nb(upon_stop_exit, i, col, flex_2d)
                    is_long_entry, is_long_exit, is_short_entry, is_short_exit, _accumulate = \
                        generate_stop_signal_nb(position_now, _upon_stop_exit, _accumulate)

                    # Resolve price and slippage
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    _stop_exit_price = flex_select_auto_nb(stop_exit_price, i, col, flex_2d)
                    _price = flex_select_auto_nb(price, i, col, flex_2d)
                    _slippage = flex_select_auto_nb(slippage, i, col, flex_2d)
                    _stex_price, _stex_slippage = resolve_stop_price_and_slippage_nb(
                        stop_price,
                        _price,
                        _close,
                        _slippage,
                        _stop_exit_price
                    )

                    # Convert both signals to size (direction-aware), size type, and direction
                    _stex_size, _stex_size_type, _stex_direction = signals_to_size_nb(
                        last_position[col],
                        is_long_entry,
                        is_long_exit,
                        is_short_entry,
                        is_short_exit,
                        flex_select_auto_nb(size, i, col, flex_2d),
                        flex_select_auto_nb(size_type, i, col, flex_2d),
                        _accumulate,
                        last_val_price[col]
                    )
                    stop_signal_set = not np.isnan(_stex_size)

                # Get user-defined signal
                signal_ctx = SignalContext(
                    i=i,
                    col=col,
                    position_now=position_now,
                    val_price_now=last_val_price[col],
                    flex_2d=flex_2d
                )
                is_long_entry, is_long_exit, is_short_entry, is_short_exit = \
                    signal_func_nb(signal_ctx, *signal_args)

                # Resolve signal conflicts
                if is_long_entry or is_short_entry:
                    _upon_long_conflict = flex_select_auto_nb(upon_long_conflict, i, col, flex_2d)
                    is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                        position_now,
                        is_long_entry,
                        is_long_exit,
                        Direction.LongOnly,
                        _upon_long_conflict
                    )
                    _upon_short_conflict = flex_select_auto_nb(upon_short_conflict, i, col, flex_2d)
                    is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                        position_now,
                        is_short_entry,
                        is_short_exit,
                        Direction.ShortOnly,
                        _upon_short_conflict
                    )

                    # Resolve direction conflicts
                    _upon_dir_conflict = flex_select_auto_nb(upon_dir_conflict, i, col, flex_2d)
                    is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                        position_now,
                        is_long_entry,
                        is_short_entry,
                        _upon_dir_conflict
                    )

                    # Resolve opposite entry
                    _upon_opposite_entry = flex_select_auto_nb(upon_opposite_entry, i, col, flex_2d)
                    is_long_entry, is_long_exit, is_short_entry, is_short_exit, _accumulate = \
                        resolve_opposite_entry_nb(
                            position_now,
                            is_long_entry,
                            is_long_exit,
                            is_short_entry,
                            is_short_exit,
                            _upon_opposite_entry,
                            _accumulate
                        )

                # Resolve price and slippage
                _price = flex_select_auto_nb(price, i, col, flex_2d)
                _slippage = flex_select_auto_nb(slippage, i, col, flex_2d)

                # Convert both signals to size (direction-aware), size type, and direction
                _size, _size_type, _direction = signals_to_size_nb(
                    last_position[col],
                    is_long_entry,
                    is_long_exit,
                    is_short_entry,
                    is_short_exit,
                    flex_select_auto_nb(size, i, col, flex_2d),
                    flex_select_auto_nb(size_type, i, col, flex_2d),
                    _accumulate,
                    last_val_price[col]
                )
                user_signal_set = not np.isnan(_size)

                # Decide on which signal should be executed: stop or user-defined?
                if stop_signal_set and user_signal_set:
                    if signal_priority == SignalPriority.Stop:
                        _price = _stex_price
                        _slippage = _stex_slippage
                        _size = _stex_size
                        _size_type = _stex_size_type
                        _direction = _stex_direction
                elif stop_signal_set:
                    _price = _stex_price
                    _slippage = _stex_slippage
                    _size = _stex_size
                    _size_type = _stex_size_type
                    _direction = _stex_direction

                # Save all info
                price_arr[col] = _price
                slippage_arr[col] = _slippage
                size_arr[col] = _size
                size_type_arr[col] = _size_type
                direction_arr[col] = _direction

                if cash_sharing:
                    if np.isnan(_size):
                        temp_order_value[k] = 0.
                    else:
                        # Approximate order value
                        if _size_type == SizeType.Amount:
                            temp_order_value[k] = _size * last_val_price[col]
                        elif _size_type == SizeType.Value:
                            temp_order_value[k] = _size
                        else:  # SizeType.Percent
                            if _size >= 0:
                                temp_order_value[k] = _size * cash_now
                            else:
                                asset_value_now = last_position[col] * last_val_price[col]
                                if _direction == Direction.LongOnly:
                                    temp_order_value[k] = _size * asset_value_now
                                else:
                                    max_exposure = (2 * max(asset_value_now, 0) + max(free_cash_now, 0))
                                    temp_order_value[k] = _size * max_exposure

            if cash_sharing:
                # Dynamically sort by order value -> selling comes first to release funds early
                if auto_call_seq:
                    insert_argsort_nb(temp_order_value[:group_len], call_seq[i, from_col:to_col])

                # Same as get_group_value_ctx_nb but with flexible indexing
                value_now = cash_now
                for k in range(group_len):
                    col = from_col + k
                    if last_position[col] != 0:
                        value_now += last_position[col] * last_val_price[col]

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
                _price = price_arr[col]
                _size = size_arr[col]  # already takes into account direction
                _size_type = size_type_arr[col]
                _direction = direction_arr[col]
                _slippage = slippage_arr[col]
                if not np.isnan(_size):
                    if _size > 0:  # long order
                        if _direction == Direction.ShortOnly:
                            _size *= -1  # must reverse for process_order_nb
                    elif _size < 0:  # short order
                        if _direction == Direction.ShortOnly:
                            _size *= -1
                    order = order_nb(
                        size=_size,
                        price=_price,
                        size_type=_size_type,
                        direction=_direction,
                        fees=flex_select_auto_nb(fees, i, col, flex_2d),
                        fixed_fees=flex_select_auto_nb(fixed_fees, i, col, flex_2d),
                        slippage=_slippage,
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

                    if use_stops:
                        # Update stop price
                        if order_result.status == OrderStatus.Filled:
                            if position_now == 0:
                                # Position closed -> clear stops
                                sl_curr_i[col] = sl_init_i[col] = -1
                                sl_curr_price[col] = sl_init_price[col] = np.nan
                                sl_curr_stop[col] = np.nan
                                sl_curr_trail[col] = False
                                tp_init_i[col] = -1
                                tp_init_price[col] = np.nan
                                tp_curr_stop[col] = np.nan
                            else:
                                _stop_entry_price = flex_select_auto_nb(stop_entry_price, i, col, flex_2d)
                                if _stop_entry_price == StopEntryPrice.ValPrice:
                                    new_init_price = val_price_now
                                elif _stop_entry_price == StopEntryPrice.Price:
                                    new_init_price = order.price
                                elif _stop_entry_price == StopEntryPrice.FillPrice:
                                    new_init_price = order_result.price
                                else:
                                    new_init_price = flex_select_auto_nb(close, i, col, flex_2d)
                                _upon_stop_update = flex_select_auto_nb(upon_stop_update, i, col, flex_2d)
                                _sl_stop = flex_select_auto_nb(sl_stop, i, col, flex_2d)
                                _sl_trail = flex_select_auto_nb(sl_trail, i, col, flex_2d)
                                _tp_stop = flex_select_auto_nb(tp_stop, i, col, flex_2d)

                                if state.position == 0 or np.sign(position_now) != np.sign(state.position):
                                    # Position opened/reversed -> set stops
                                    sl_curr_i[col] = sl_init_i[col] = i
                                    sl_curr_price[col] = sl_init_price[col] = new_init_price
                                    sl_curr_stop[col] = _sl_stop
                                    sl_curr_trail[col] = _sl_trail
                                    tp_init_i[col] = i
                                    tp_init_price[col] = new_init_price
                                    tp_curr_stop[col] = _tp_stop
                                elif abs(position_now) > abs(state.position):
                                    # Position increased -> keep/override stops
                                    if should_update_stop_nb(_sl_stop, _upon_stop_update):
                                        sl_curr_i[col] = sl_init_i[col] = i
                                        sl_curr_price[col] = sl_init_price[col] = new_init_price
                                        sl_curr_stop[col] = _sl_stop
                                        sl_curr_trail[col] = _sl_trail
                                    if should_update_stop_nb(_tp_stop, _upon_stop_update):
                                        tp_init_i[col] = i
                                        tp_init_price[col] = new_init_price
                                        tp_curr_stop[col] = _tp_stop

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


@register_jitted
def dir_enex_signal_func_nb(c: SignalContext,
                            entries: tp.FlexArray,
                            exits: tp.FlexArray,
                            direction: tp.FlexArray) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals out of entries, exits, and direction."""
    is_entry = flex_select_auto_nb(entries, c.i, c.col, c.flex_2d)
    is_exit = flex_select_auto_nb(exits, c.i, c.col, c.flex_2d)
    _direction = flex_select_auto_nb(direction, c.i, c.col, c.flex_2d)
    if _direction == Direction.LongOnly:
        return is_entry, is_exit, False, False
    if _direction == Direction.ShortOnly:
        return False, False, is_entry, is_exit
    return is_entry, False, is_exit, False


@register_jitted
def ls_enex_signal_func_nb(c: SignalContext,
                           long_entries: tp.FlexArray,
                           long_exits: tp.FlexArray,
                           short_entries: tp.FlexArray,
                           short_exits: tp.FlexArray) -> tp.Tuple[bool, bool, bool, bool]:
    """Get an element of direction-aware signals."""
    is_long_entry = flex_select_auto_nb(long_entries, c.i, c.col, c.flex_2d)
    is_long_exit = flex_select_auto_nb(long_exits, c.i, c.col, c.flex_2d)
    is_short_entry = flex_select_auto_nb(short_entries, c.i, c.col, c.flex_2d)
    is_short_exit = flex_select_auto_nb(short_exits, c.i, c.col, c.flex_2d)
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit
