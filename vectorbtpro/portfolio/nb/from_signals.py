# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio modeling based on signals."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb
from vectorbtpro.utils.math_ import is_less_nb


@register_jitted(cache=True)
def resolve_pending_conflict_nb(
    is_pending_long: bool,
    is_user_long: bool,
    upon_adj_conflict: int,
    upon_opp_conflict: int,
) -> tp.Tuple[bool, bool]:
    """Resolve any conflict between a pending signal and a user-defined signal."""
    if (is_pending_long and is_user_long) or (not is_pending_long and not is_user_long):
        if upon_adj_conflict == PendingConflictMode.KeepIgnore:
            return True, False
        if upon_adj_conflict == PendingConflictMode.KeepExecute:
            return True, True
        if upon_adj_conflict == PendingConflictMode.CancelIgnore:
            return False, False
        return False, True
    else:
        if upon_opp_conflict == PendingConflictMode.KeepIgnore:
            return True, False
        if upon_opp_conflict == PendingConflictMode.KeepExecute:
            return True, True
        if upon_opp_conflict == PendingConflictMode.CancelIgnore:
            return False, False
        return False, True


@register_jitted(cache=True)
def generate_stop_signal_nb(
    position_now: float,
    upon_stop_exit: int,
    accumulate: int,
) -> tp.Tuple[bool, bool, bool, bool, int]:
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
def resolve_stop_price_and_slippage_nb(
    stop_price: float,
    close: float,
    slippage: float,
    stop_exit_price: int,
) -> tp.Tuple[float, float]:
    """Resolve price and slippage of a stop order."""
    if stop_exit_price == StopExitPrice.StopMarket:
        return stop_price, slippage
    if stop_exit_price == StopExitPrice.StopLimit:
        return stop_price, 0.0
    return close, slippage


@register_jitted(cache=True)
def resolve_signal_conflict_nb(
    position_now: float,
    is_entry: bool,
    is_exit: bool,
    direction: int,
    conflict_mode: int,
) -> tp.Tuple[bool, bool]:
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
def resolve_dir_conflict_nb(
    position_now: float,
    is_long_entry: bool,
    is_short_entry: bool,
    upon_dir_conflict: int,
) -> tp.Tuple[bool, bool]:
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
def resolve_opposite_entry_nb(
    position_now: float,
    is_long_entry: bool,
    is_long_exit: bool,
    is_short_entry: bool,
    is_short_exit: bool,
    upon_opposite_entry: int,
    accumulate: int,
) -> tp.Tuple[bool, bool, bool, bool, int]:
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
def signals_to_size_nb(
    position_now: float,
    is_long_entry: bool,
    is_long_exit: bool,
    is_short_entry: bool,
    is_short_exit: bool,
    size: float,
    size_type: int,
    accumulate: int,
    val_price_now: float,
) -> tp.Tuple[float, int, int]:
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
                        raise ValueError("SizeType.Percent does not support position reversal using signals")
                    if size_type == SizeType.Value:
                        order_size -= size / val_price_now
                    else:
                        order_size -= size
                size_type = SizeType.Amount
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
                size_type = SizeType.Amount
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

    if direction == Direction.ShortOnly:
        order_size = -order_size
    return order_size, size_type, direction


@register_jitted(cache=True)
def should_update_stop_nb(stop: float, upon_stop_update: int) -> bool:
    """Whether to update stop."""
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if not np.isnan(stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
    return False


@register_jitted(cache=True)
def resolve_hl_nb(curr_open, curr_high, curr_low, curr_close):
    """Resolve the current high and low."""
    if np.isnan(curr_high):
        if np.isnan(curr_open):
            curr_high = curr_close
        elif np.isnan(curr_close):
            curr_high = curr_open
        else:
            curr_high = max(curr_open, curr_close)
    if np.isnan(curr_low):
        if np.isnan(curr_open):
            curr_low = curr_close
        elif np.isnan(curr_close):
            curr_low = curr_open
        else:
            curr_low = min(curr_open, curr_close)
    return curr_high, curr_low


@register_jitted(cache=True)
def resolve_stop_price_nb(
    position_now: float,
    stop_price: float,
    stop: float,
    stop_format: int,
    curr_open: float,
    curr_high: float,
    curr_low: float,
    hit_below: bool,
) -> float:
    """Resolve the current stop price.

    If hit before open, returns open."""
    if (position_now > 0 and hit_below) or (position_now < 0 and not hit_below):
        if stop_format == StopFormat.Relative:
            stop_price = stop_price * (1 - abs(stop))
        else:
            stop_price = stop_price - abs(stop)
        if curr_open <= stop_price:
            return curr_open
        if curr_low <= stop_price:
            return stop_price
        return np.nan
    if (position_now < 0 and hit_below) or (position_now > 0 and not hit_below):
        if stop_format == StopFormat.Relative:
            stop_price = stop_price * (1 + abs(stop))
        else:
            stop_price = stop_price + abs(stop)
        if curr_open >= stop_price:
            return curr_open
        if curr_high >= stop_price:
            return stop_price
        return np.nan
    return np.nan


@register_jitted(cache=True)
def tsl_delta_hit_nb(
    position_now: float,
    init_price: float,
    curr_price: float,
    threshold: float,
    stop_format: int,
) -> bool:
    """Return whether TSL delta has been hit."""
    if position_now > 0:
        if stop_format == StopFormat.Relative:
            if curr_price / init_price - 1 >= abs(threshold):
                return True
        else:
            if curr_price - init_price >= abs(threshold):
                return True
    if position_now < 0:
        if stop_format == StopFormat.Relative:
            if curr_price / init_price - 1 <= -abs(threshold):
                return True
        else:
            if curr_price - init_price <= -abs(threshold):
                return True
    return False


@register_jitted
def no_signal_func_nb(c: SignalContext, *args) -> tp.Tuple[bool, bool, bool, bool]:
    """Placeholder signal function that returns no signal."""
    return False, False, False, False


SignalFuncT = tp.Callable[[SignalContext, tp.VarArg()], tp.Tuple[bool, bool, bool, bool]]


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        init_cash=base_ch.FlexArraySlicer(axis=1, flex_2d=True),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
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
        val_price=portfolio_ch.flex_array_gl_slicer,
        accumulate=portfolio_ch.flex_array_gl_slicer,
        upon_long_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_short_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_dir_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opposite_entry=portfolio_ch.flex_array_gl_slicer,
        signal_type=portfolio_ch.flex_array_gl_slicer,
        upon_adj_limit_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opp_limit_conflict=portfolio_ch.flex_array_gl_slicer,
        use_stops=None,
        sl_stop=portfolio_ch.flex_array_gl_slicer,
        tsl_delta=portfolio_ch.flex_array_gl_slicer,
        tsl_stop=portfolio_ch.flex_array_gl_slicer,
        tp_stop=portfolio_ch.flex_array_gl_slicer,
        stop_format=portfolio_ch.flex_array_gl_slicer,
        stop_entry_price=portfolio_ch.flex_array_gl_slicer,
        stop_exit_price=portfolio_ch.flex_array_gl_slicer,
        upon_stop_exit=portfolio_ch.flex_array_gl_slicer,
        upon_stop_update=portfolio_ch.flex_array_gl_slicer,
        upon_adj_stop_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opp_stop_conflict=portfolio_ch.flex_array_gl_slicer,
        call_seq=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        auto_call_seq=None,
        ffill_val_price=None,
        update_value=None,
        fill_returns=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
    ),
    **portfolio_ch.merge_sim_outs_config,
)
@register_jitted(tags={"can_parallel"})
def simulate_from_signal_func_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    open: tp.FlexArray = np.asarray(np.nan),
    high: tp.FlexArray = np.asarray(np.nan),
    low: tp.FlexArray = np.asarray(np.nan),
    close: tp.FlexArray = np.asarray(np.nan),
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(np.nan),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    cash_dividends: tp.FlexArray = np.asarray(0.0),
    signal_func_nb: SignalFuncT = no_signal_func_nb,
    signal_args: tp.ArgsLike = (),
    size: tp.FlexArray = np.asarray(np.inf),
    price: tp.FlexArray = np.asarray(np.inf),
    size_type: tp.FlexArray = np.asarray(SizeType.Amount),
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
    accumulate: tp.FlexArray = np.asarray(AccumulationMode.Disabled),
    upon_long_conflict: tp.FlexArray = np.asarray(ConflictMode.Ignore),
    upon_short_conflict: tp.FlexArray = np.asarray(ConflictMode.Ignore),
    upon_dir_conflict: tp.FlexArray = np.asarray(DirectionConflictMode.Ignore),
    upon_opposite_entry: tp.FlexArray = np.asarray(OppositeEntryMode.ReverseReduce),
    signal_type: tp.FlexArray = np.asarray(SignalType.Market),
    upon_adj_limit_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepIgnore),
    upon_opp_limit_conflict: tp.FlexArray = np.asarray(PendingConflictMode.CancelExecute),
    use_stops: bool = True,
    sl_stop: tp.FlexArray = np.asarray(np.nan),
    tsl_delta: tp.FlexArray = np.asarray(np.nan),
    tsl_stop: tp.FlexArray = np.asarray(np.nan),
    tp_stop: tp.FlexArray = np.asarray(np.nan),
    stop_format: tp.FlexArray = np.asarray(StopFormat.Relative),
    stop_entry_price: tp.FlexArray = np.asarray(StopEntryPrice.Close),
    stop_exit_price: tp.FlexArray = np.asarray(StopExitPrice.StopLimit),
    upon_stop_exit: tp.FlexArray = np.asarray(StopExitMode.Close),
    upon_stop_update: tp.FlexArray = np.asarray(StopUpdateMode.Keep),
    upon_adj_stop_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepExecute),
    upon_opp_stop_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepExecute),
    call_seq: tp.Optional[tp.Array2d] = None,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_returns: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = False,
) -> SimulationOutput:
    """Simulate given a signal function `signal_func_nb`.

    Iterates in the column-major order. Utilizes flexible broadcasting.

    !!! note
        Should be only grouped if cash sharing is enabled.

        If `auto_call_seq` is True, make sure that `call_seq` follows `CallSeqType.Default`.

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).
    """
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    order_records, log_records = prepare_records_nb(target_shape, max_orders, max_logs)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

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
    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full(target_shape[1], 0.0, dtype=np.float_)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    track_cash_deposits = np.any(cash_deposits)
    if track_cash_deposits:
        cash_deposits_out = np.full((target_shape[0], len(group_lens)), 0.0, dtype=np.float_)
    else:
        cash_deposits_out = np.full((1, 1), 0.0, dtype=np.float_)
    track_cash_earnings = np.any(cash_earnings) or np.any(cash_dividends)
    if track_cash_earnings:
        cash_earnings_out = np.full(target_shape, 0.0, dtype=np.float_)
    else:
        cash_earnings_out = np.full((1, 1), 0.0, dtype=np.float_)
    if fill_returns:
        returns_out = np.empty((target_shape[0], len(group_lens)), dtype=np.float_)
    else:
        returns_out = np.empty((0, 0), dtype=np.float_)
    in_outputs = FSInOutputs(returns=returns_out)

    last_limit_info = np.empty(target_shape[1], dtype=limit_info_dt)
    last_limit_info["init_i"][:] = -1
    last_limit_info["init_price"][:] = np.nan
    last_limit_info["init_size"][:] = np.nan
    last_limit_info["init_size_type"][:] = -1
    last_limit_info["init_direction"][:] = -1

    if use_stops:
        last_sl_info = np.empty(target_shape[1], dtype=sl_info_dt)
        last_sl_info["init_i"][:] = -1
        last_sl_info["init_price"][:] = np.nan
        last_sl_info["curr_stop"][:] = np.nan

        last_tsl_info = np.empty(target_shape[1], dtype=tsl_info_dt)
        last_tsl_info["init_i"][:] = -1
        last_tsl_info["init_price"][:] = np.nan
        last_tsl_info["curr_i"][:] = -1
        last_tsl_info["curr_price"][:] = np.nan
        last_tsl_info["curr_delta"][:] = np.nan
        last_tsl_info["curr_stop"][:] = np.nan

        last_tp_info = np.empty(target_shape[1], dtype=tp_info_dt)
        last_tp_info["init_i"][:] = -1
        last_tp_info["init_price"][:] = np.nan
        last_tp_info["curr_stop"][:] = np.nan
    else:
        last_sl_info = np.empty(0, dtype=sl_info_dt)
        last_tsl_info = np.empty(0, dtype=tsl_info_dt)
        last_tp_info = np.empty(0, dtype=tp_info_dt)

    limit_signal_set = np.full(target_shape[1], False, dtype=np.bool_)
    stop_signal_set = np.full(target_shape[1], False, dtype=np.bool_)
    user_signal_set = np.full(target_shape[1], False, dtype=np.bool_)

    trigger_i_arr = np.full(target_shape[1], -1, dtype=np.int_)
    price_arr = np.empty(target_shape[1], dtype=np.float_)
    size_arr = np.empty(target_shape[1], dtype=np.float_)
    size_type_arr = np.empty(target_shape[1], dtype=np.float_)
    slippage_arr = np.empty(target_shape[1], dtype=np.float_)
    direction_arr = np.empty(target_shape[1], dtype=np.int_)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)
    temp_order_value = np.empty(target_shape[1], dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        for i in range(target_shape[0]):
            # Add cash
            _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
            if _cash_deposits < 0:
                _cash_deposits = max(_cash_deposits, -last_cash[group])
            last_cash[group] += _cash_deposits
            last_free_cash[group] += _cash_deposits
            last_cash_deposits[group] = _cash_deposits
            if track_cash_deposits:
                cash_deposits_out[i, group] += _cash_deposits

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

            # Update value and return
            group_value = last_cash[group]
            for col in range(from_col, to_col):
                if last_position[col] != 0:
                    group_value += last_position[col] * last_val_price[col]
            last_value[group] = group_value
            last_return[group] = returns_nb_.get_return_nb(
                prev_close_value[group],
                last_value[group] - _cash_deposits,
            )

            # Get size and value of each order
            for c in range(group_len):
                col = from_col + c  # order doesn't matter

                # Set defaults
                limit_signal_set[col] = False
                user_signal_set[col] = False
                stop_signal_set[col] = False

                trigger_i_arr[col] = i
                price_arr[col] = np.nan
                slippage_arr[col] = np.nan
                size_arr[col] = np.nan
                size_type_arr[col] = -1
                direction_arr[col] = -1
                temp_order_value[col] = 0.0

                # Get signals
                signal_ctx = SignalContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    order_counts=order_counts,
                    log_records=log_records,
                    log_counts=log_counts,
                    track_cash_deposits=track_cash_deposits,
                    cash_deposits_out=cash_deposits_out,
                    track_cash_earnings=track_cash_earnings,
                    cash_earnings_out=cash_earnings_out,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    last_limit_info=last_limit_info,
                    last_sl_info=last_sl_info,
                    last_tsl_info=last_tsl_info,
                    last_tp_info=last_tp_info,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    col=col,
                )
                is_long_entry, is_long_exit, is_short_entry, is_short_exit = signal_func_nb(signal_ctx, *signal_args)

                # Shortcut
                any_limit_signal = last_limit_info["init_i"][col] != -1
                any_stop_signal = use_stops and (
                    not np.isnan(last_sl_info["curr_stop"][col])
                    or not np.isnan(last_tsl_info["curr_stop"][col])
                    or not np.isnan(last_tp_info["curr_stop"][col])
                )
                any_user_signal = is_long_entry or is_long_exit or is_short_entry or is_short_exit
                if not any_limit_signal and not any_stop_signal and not any_user_signal:  # shortcut
                    continue

                # Process limit signal
                if any_limit_signal:
                    # Resolve current bar
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    _high = flex_select_auto_nb(high, i, col, flex_2d)
                    _low = flex_select_auto_nb(low, i, col, flex_2d)
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    _high, _low = resolve_hl_nb(_open, _high, _low, _close)

                    # Check whether the limit price has been hit
                    _init_i = last_limit_info["init_i"][col]
                    _price = last_limit_info["init_price"][col]
                    _size = last_limit_info["init_size"][col]
                    _size_type = last_limit_info["init_size_type"][col]
                    _direction = last_limit_info["init_direction"][col]

                    if _direction == Direction.ShortOnly:
                        __size = _size * -1
                    else:
                        __size = _size
                    if __size == 0:
                        raise ValueError("Limit order size cannot be zero")
                    if __size > 0:  # buy order
                        if _open <= _price:
                            _price = _open
                            can_execute = True
                        else:
                            can_execute = _low <= _price
                    else:  # sell order
                        if _open >= _price:
                            _price = _open
                            can_execute = True
                        else:
                            can_execute = _high >= _price
                    if can_execute:
                        # Executable limit signal
                        limit_signal_set[col] = True
                        trigger_i_arr[col] = _init_i
                        price_arr[col] = _price
                        slippage_arr[col] = 0.0
                        size_arr[col] = _size
                        size_type_arr[col] = _size_type
                        direction_arr[col] = _direction

                # Process stop signal
                if any_stop_signal:
                    # Resolve current bar
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    _high = flex_select_auto_nb(high, i, col, flex_2d)
                    _low = flex_select_auto_nb(low, i, col, flex_2d)
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    _high, _low = resolve_hl_nb(_open, _high, _low, _close)

                    # Get stop price
                    stop_price = np.nan
                    if not np.isnan(last_sl_info["curr_stop"][col]):
                        _stop_format = flex_select_auto_nb(stop_format, last_sl_info["init_i"][col], col, flex_2d)
                        # Check against high and low
                        stop_price = resolve_stop_price_nb(
                            last_position[col],
                            last_sl_info["init_price"][col],
                            last_sl_info["curr_stop"][col],
                            _stop_format,
                            _open,
                            _high,
                            _low,
                            True,
                        )
                    if np.isnan(stop_price) and not np.isnan(last_tsl_info["curr_stop"][col]):
                        _stop_format = flex_select_auto_nb(stop_format, last_tsl_info["init_i"][col], col, flex_2d)
                        # Check against high and low
                        if np.isnan(last_tsl_info["curr_delta"][col]):
                            delta_hit = True
                        else:
                            delta_hit = tsl_delta_hit_nb(
                                last_position[col],
                                last_tsl_info["init_price"][col],
                                last_tsl_info["curr_price"][col],
                                last_tsl_info["curr_delta"][col],
                                _stop_format,
                            )
                        if delta_hit:
                            stop_price = resolve_stop_price_nb(
                                last_position[col],
                                last_tsl_info["curr_price"][col],
                                last_tsl_info["curr_stop"][col],
                                _stop_format,
                                _open,
                                _high,
                                _low,
                                True,
                            )
                        if last_position[col] > 0:
                            if _high > last_tsl_info["curr_price"][col]:
                                last_tsl_info["curr_i"][col] = i
                                last_tsl_info["curr_price"][col] = _high
                        elif last_position[col] < 0:
                            if _low < last_tsl_info["curr_price"][col]:
                                last_tsl_info["curr_i"][col] = i
                                last_tsl_info["curr_price"][col] = _low
                        if np.isnan(stop_price):
                            # After update, check one more time against close
                            if np.isnan(last_tsl_info["curr_delta"][col]):
                                delta_hit = True
                            else:
                                delta_hit = tsl_delta_hit_nb(
                                    last_position[col],
                                    last_tsl_info["init_price"][col],
                                    last_tsl_info["curr_price"][col],
                                    last_tsl_info["curr_delta"][col],
                                    _stop_format,
                                )
                            if delta_hit:
                                stop_price = resolve_stop_price_nb(
                                    last_position[col],
                                    last_tsl_info["curr_price"][col],
                                    last_tsl_info["curr_stop"][col],
                                    _stop_format,
                                    _open,
                                    _close,
                                    _close,
                                    True,
                                )
                    if np.isnan(stop_price) and not np.isnan(last_tp_info["curr_stop"][col]):
                        _stop_format = flex_select_auto_nb(stop_format, last_tp_info["init_i"][col], col, flex_2d)
                        # Check against high and low
                        stop_price = resolve_stop_price_nb(
                            last_position[col],
                            last_tp_info["init_price"][col],
                            last_tp_info["curr_stop"][col],
                            _stop_format,
                            _open,
                            _high,
                            _low,
                            False,
                        )

                    if not np.isnan(stop_price):
                        # Get stop signal
                        _accumulate = flex_select_auto_nb(accumulate, i, col, flex_2d)
                        _upon_stop_exit = flex_select_auto_nb(upon_stop_exit, i, col, flex_2d)
                        (
                            stop_is_long_entry,
                            stop_is_long_exit,
                            stop_is_short_entry,
                            stop_is_short_exit,
                            _accumulate,
                        ) = generate_stop_signal_nb(
                            last_position[col],
                            _upon_stop_exit,
                            _accumulate,
                        )

                        # Resolve price and slippage
                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                        _stop_exit_price = flex_select_auto_nb(stop_exit_price, i, col, flex_2d)
                        _slippage = flex_select_auto_nb(slippage, i, col, flex_2d)
                        _price, _slippage = resolve_stop_price_and_slippage_nb(
                            stop_price,
                            _close,
                            _slippage,
                            _stop_exit_price,
                        )

                        # Convert both signals to size (direction-aware), size type, and direction
                        _size, _size_type, _direction = signals_to_size_nb(
                            last_position[col],
                            stop_is_long_entry,
                            stop_is_long_exit,
                            stop_is_short_entry,
                            stop_is_short_exit,
                            flex_select_auto_nb(size, i, col, flex_2d),
                            flex_select_auto_nb(size_type, i, col, flex_2d),
                            _accumulate,
                            last_val_price[col],
                        )

                        if not np.isnan(_size):
                            if not limit_signal_set[col]:
                                # When stop and limit are hit within the same bar, we pessimistically
                                # assume limit was hit before the stop (i.e. "before it counts")
                                stop_signal_set[col] = True
                                trigger_i_arr[col] = i
                                price_arr[col] = _price
                                slippage_arr[col] = _slippage
                                size_arr[col] = _size
                                size_type_arr[col] = _size_type
                                direction_arr[col] = _direction

                # Process user signal
                if any_user_signal:
                    user_signal_first = True
                    if limit_signal_set[col] or stop_signal_set[col]:
                        _price = flex_select_auto_nb(close, i, col, flex_2d)
                        user_signal_first = np.isinf(_price) and _price < 0
                    if user_signal_first:
                        # Either use signal is alone, or before any pending signal is executed
                        _accumulate = flex_select_auto_nb(accumulate, i, col, flex_2d)
                        if is_long_entry or is_short_entry:
                            # Resolve conflicts
                            _upon_long_conflict = flex_select_auto_nb(upon_long_conflict, i, col, flex_2d)
                            is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                                last_position[col],
                                is_long_entry,
                                is_long_exit,
                                Direction.LongOnly,
                                _upon_long_conflict,
                            )
                            _upon_short_conflict = flex_select_auto_nb(upon_short_conflict, i, col, flex_2d)
                            is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                                last_position[col],
                                is_short_entry,
                                is_short_exit,
                                Direction.ShortOnly,
                                _upon_short_conflict,
                            )

                            # Resolve direction conflicts
                            _upon_dir_conflict = flex_select_auto_nb(upon_dir_conflict, i, col, flex_2d)
                            is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                                last_position[col],
                                is_long_entry,
                                is_short_entry,
                                _upon_dir_conflict,
                            )

                            # Resolve opposite entry
                            _upon_opposite_entry = flex_select_auto_nb(upon_opposite_entry, i, col, flex_2d)
                            (
                                is_long_entry,
                                is_long_exit,
                                is_short_entry,
                                is_short_exit,
                                _accumulate,
                            ) = resolve_opposite_entry_nb(
                                last_position[col],
                                is_long_entry,
                                is_long_exit,
                                is_short_entry,
                                is_short_exit,
                                _upon_opposite_entry,
                                _accumulate,
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
                            last_val_price[col],
                        )

                        if not np.isnan(_size):
                            execute_user = True
                            if any_limit_signal:
                                # Pending limit signal
                                if _direction == Direction.ShortOnly:
                                    __size = _size * -1
                                else:
                                    __size = _size
                                if last_limit_info["init_direction"][col] == Direction.ShortOnly:
                                    stop_size = -1 * last_limit_info["init_size"][col]
                                else:
                                    stop_size = last_limit_info["init_size"][col]
                                keep_limit, execute_user = resolve_pending_conflict_nb(
                                    stop_size >= 0,
                                    __size >= 0,
                                    flex_select_auto_nb(upon_adj_limit_conflict, i, col, flex_2d),
                                    flex_select_auto_nb(upon_opp_limit_conflict, i, col, flex_2d),
                                )
                                if not keep_limit:
                                    any_limit_signal = False
                                    limit_signal_set[col] = False
                                    last_limit_info["init_i"][col] = -1
                                    last_limit_info["init_price"][col] = np.nan
                                    last_limit_info["init_size"][col] = np.nan
                                    last_limit_info["init_size_type"][col] = -1
                                    last_limit_info["init_direction"][col] = -1

                            if any_stop_signal:
                                # Pending stop signal
                                if _direction == Direction.ShortOnly:
                                    __size = _size * -1
                                else:
                                    __size = _size
                                keep_stop, execute_user = resolve_pending_conflict_nb(
                                    last_position[col] < 0,
                                    __size >= 0,
                                    flex_select_auto_nb(upon_adj_stop_conflict, i, col, flex_2d),
                                    flex_select_auto_nb(upon_opp_stop_conflict, i, col, flex_2d),
                                )
                                if not keep_stop:
                                    any_stop_signal = False
                                    stop_signal_set[col] = False
                                    last_sl_info["init_i"][col] = -1
                                    last_sl_info["init_price"][col] = np.nan
                                    last_sl_info["curr_stop"][col] = np.nan

                                    last_tsl_info["init_i"][col] = -1
                                    last_tsl_info["init_price"][col] = np.nan
                                    last_tsl_info["curr_i"][col] = -1
                                    last_tsl_info["curr_price"][col] = np.nan
                                    last_tsl_info["curr_delta"][col] = np.nan
                                    last_tsl_info["curr_stop"][col] = np.nan

                                    last_tp_info["init_i"][col] = -1
                                    last_tp_info["init_price"][col] = np.nan
                                    last_tp_info["curr_stop"][col] = np.nan

                            if execute_user:
                                # Executable user signal
                                can_execute = True
                                _signal_type = flex_select_auto_nb(signal_type, i, col, flex_2d)
                                if _signal_type == SignalType.Limit:
                                    if any_limit_signal:
                                        raise ValueError("Only one active limit signal is allowed at a time")
                                    if np.isinf(_price) and _price > 0:
                                        # Cannot place a limit order at the close price and execute right away
                                        can_execute = False
                                        _price = flex_select_auto_nb(close, i, col, flex_2d)
                                    else:
                                        if _direction == Direction.ShortOnly:
                                            __size = _size * -1
                                        else:
                                            __size = _size
                                        if __size == 0:
                                            raise ValueError("Limit order size cannot be zero")
                                        can_use_ohlc = np.isinf(_price) and _price < 0
                                        if can_use_ohlc:
                                            _open = flex_select_auto_nb(open, i, col, flex_2d)
                                            _high = flex_select_auto_nb(high, i, col, flex_2d)
                                            _low = flex_select_auto_nb(low, i, col, flex_2d)
                                            _close = flex_select_auto_nb(close, i, col, flex_2d)
                                            _high, _low = resolve_hl_nb(_open, _high, _low, _close)
                                            if __size > 0:  # buy order
                                                can_execute = _low <= _price
                                            else:  # sell order
                                                can_execute = _high >= _price
                                        else:
                                            _close = flex_select_auto_nb(close, i, col, flex_2d)
                                            if __size > 0:  # buy order
                                                can_execute = _close <= _price
                                            else:  # sell order
                                                can_execute = _close >= _price
                                    _slippage = 0.0
                                # Save info
                                if can_execute:
                                    user_signal_set[col] = True
                                    trigger_i_arr[col] = i
                                    price_arr[col] = _price
                                    slippage_arr[col] = _slippage
                                    size_arr[col] = _size
                                    size_type_arr[col] = _size_type
                                    direction_arr[col] = _direction
                                else:
                                    last_limit_info["init_i"][col] = i
                                    last_limit_info["init_price"][col] = _price
                                    last_limit_info["init_size"][col] = _size
                                    last_limit_info["init_size_type"][col] = _size_type
                                    last_limit_info["init_direction"][col] = _direction
                                    continue

                if limit_signal_set[col] or stop_signal_set[col] or user_signal_set[col]:
                    if cash_sharing and auto_call_seq:
                        # Approximate order value
                        if np.isnan(size_arr[col]):
                            temp_order_value[c] = 0.0
                        else:
                            exec_state = ExecState(
                                cash=last_cash[group],
                                position=last_position[col],
                                debt=last_debt[col],
                                free_cash=last_free_cash[group],
                                val_price=last_val_price[col],
                                value=last_value[group],
                            )
                            temp_order_value[c] = approx_order_value_nb(
                                exec_state,
                                size_arr[col],
                                size_type_arr[col],
                                direction_arr[col],
                            )

            any_signal_set = False
            for col in range(from_col, to_col):
                if limit_signal_set[col] or stop_signal_set[col] or user_signal_set[col]:
                    any_signal_set = True
                    break
            if any_signal_set:
                if cash_sharing:
                    # Dynamically sort by order value -> selling comes first to release funds early
                    if call_seq is None:
                        for c in range(group_len):
                            temp_call_seq[c] = c
                        call_seq_now = temp_call_seq[:group_len]
                    else:
                        call_seq_now = call_seq[i, from_col:to_col]
                    if auto_call_seq:
                        for c in range(group_len):
                            if call_seq_now[c] != c:
                                raise ValueError("Call sequence must follow CallSeqType.Default")
                        insert_argsort_nb(temp_order_value[:group_len], call_seq_now)

                for k in range(group_len):
                    if cash_sharing:
                        c = call_seq_now[k]
                        if c >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        c = k
                    col = from_col + c
                    if not limit_signal_set[col] and not stop_signal_set[col] and not user_signal_set[col]:  # shortcut
                        continue

                    # Get current values per column
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    cash_now = last_cash[group]
                    free_cash_now = last_free_cash[group]
                    value_now = last_value[group]
                    return_now = last_return[group]

                    # Generate the next order
                    trigger_i = trigger_i_arr[col]
                    order = order_nb(
                        size=size_arr[col],
                        price=price_arr[col],
                        size_type=size_type_arr[col],
                        direction=direction_arr[col],
                        fees=flex_select_auto_nb(fees, trigger_i, col, flex_2d),
                        fixed_fees=flex_select_auto_nb(fixed_fees, trigger_i, col, flex_2d),
                        slippage=slippage_arr[col],
                        min_size=flex_select_auto_nb(min_size, trigger_i, col, flex_2d),
                        max_size=flex_select_auto_nb(max_size, trigger_i, col, flex_2d),
                        size_granularity=flex_select_auto_nb(size_granularity, trigger_i, col, flex_2d),
                        reject_prob=flex_select_auto_nb(reject_prob, trigger_i, col, flex_2d),
                        price_area_vio_mode=flex_select_auto_nb(price_area_vio_mode, trigger_i, col, flex_2d),
                        lock_cash=flex_select_auto_nb(lock_cash, trigger_i, col, flex_2d),
                        allow_partial=flex_select_auto_nb(allow_partial, trigger_i, col, flex_2d),
                        raise_reject=flex_select_auto_nb(raise_reject, trigger_i, col, flex_2d),
                        log=flex_select_auto_nb(log, trigger_i, col, flex_2d),
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

                    if limit_signal_set[col]:
                        # Clear limit signal
                        last_limit_info["init_i"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1

                    if use_stops:
                        # Update stop price
                        if stop_signal_set[col] or position_now == 0:
                            # Stop signal executed or not in position -> clear stops (irrespective of order success)
                            last_sl_info["init_i"][col] = -1
                            last_sl_info["init_price"][col] = np.nan
                            last_sl_info["curr_stop"][col] = np.nan

                            last_tsl_info["init_i"][col] = -1
                            last_tsl_info["init_price"][col] = np.nan
                            last_tsl_info["curr_i"][col] = -1
                            last_tsl_info["curr_price"][col] = np.nan
                            last_tsl_info["curr_delta"][col] = np.nan
                            last_tsl_info["curr_stop"][col] = np.nan

                            last_tp_info["init_i"][col] = -1
                            last_tp_info["init_price"][col] = np.nan
                            last_tp_info["curr_stop"][col] = np.nan

                        if order_result.status == OrderStatus.Filled and position_now != 0:
                            # Order filled and in position -> possibly set stops
                            _price = price_arr[col]
                            _stop_entry_price = flex_select_auto_nb(stop_entry_price, i, col, flex_2d)
                            if _stop_entry_price < 0:
                                if _stop_entry_price == StopEntryPrice.ValPrice:
                                    new_init_price = val_price_now
                                    can_use_ohlc = False
                                elif _stop_entry_price == StopEntryPrice.Price:
                                    new_init_price = order.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                    if np.isinf(new_init_price):
                                        if new_init_price > 0:
                                            new_init_price = flex_select_auto_nb(close, i, col, flex_2d)
                                        else:
                                            new_init_price = flex_select_auto_nb(open, i, col, flex_2d)
                                elif _stop_entry_price == StopEntryPrice.FillPrice:
                                    new_init_price = order_result.price
                                    can_use_ohlc = np.isinf(_price) and _price < 0
                                elif _stop_entry_price == StopEntryPrice.Open:
                                    new_init_price = flex_select_auto_nb(open, i, col, flex_2d)
                                    can_use_ohlc = True
                                else:
                                    new_init_price = flex_select_auto_nb(close, i, col, flex_2d)
                                    can_use_ohlc = False
                            else:
                                new_init_price = _stop_entry_price
                                can_use_ohlc = False

                            _sl_stop = abs(flex_select_auto_nb(sl_stop, i, col, flex_2d))
                            if _sl_stop == 0:
                                _sl_stop = np.nan
                            _tsl_delta = abs(flex_select_auto_nb(tsl_delta, i, col, flex_2d))
                            if _tsl_delta == 0:
                                _tsl_delta = np.nan
                            _tsl_stop = abs(flex_select_auto_nb(tsl_stop, i, col, flex_2d))
                            if _tsl_stop == 0:
                                _tsl_stop = np.nan
                            _tp_stop = abs(flex_select_auto_nb(tp_stop, i, col, flex_2d))
                            if _tp_stop == 0:
                                _tp_stop = np.nan

                            sl_updated = tsl_updated = tp_updated = False
                            if exec_state.position == 0 or np.sign(position_now) != np.sign(exec_state.position):
                                # Position opened/reversed -> set stops
                                sl_updated = True
                                last_sl_info["init_i"][col] = i
                                last_sl_info["init_price"][col] = new_init_price
                                last_sl_info["curr_stop"][col] = _sl_stop

                                tsl_updated = True
                                last_tsl_info["init_i"][col] = i
                                last_tsl_info["init_price"][col] = new_init_price
                                last_tsl_info["curr_i"][col] = i
                                last_tsl_info["curr_price"][col] = new_init_price
                                last_tsl_info["curr_delta"][col] = _tsl_delta
                                last_tsl_info["curr_stop"][col] = _tsl_stop

                                tp_updated = True
                                last_tp_info["init_i"][col] = i
                                last_tp_info["init_price"][col] = new_init_price
                                last_tp_info["curr_stop"][col] = _tp_stop

                            elif abs(position_now) > abs(exec_state.position):
                                # Position increased -> keep/override stops
                                _upon_stop_update = flex_select_auto_nb(upon_stop_update, i, col, flex_2d)
                                if should_update_stop_nb(_sl_stop, _upon_stop_update):
                                    sl_updated = True
                                    last_sl_info["init_i"][col] = i
                                    last_sl_info["init_price"][col] = new_init_price
                                    last_sl_info["curr_stop"][col] = _sl_stop
                                if should_update_stop_nb(_tsl_stop, _upon_stop_update):
                                    tsl_updated = True
                                    last_tsl_info["init_i"][col] = i
                                    last_tsl_info["init_price"][col] = new_init_price
                                    last_tsl_info["curr_i"][col] = i
                                    last_tsl_info["curr_price"][col] = new_init_price
                                    last_tsl_info["curr_delta"][col] = _tsl_delta
                                    last_tsl_info["curr_stop"][col] = _tsl_stop
                                if should_update_stop_nb(_tp_stop, _upon_stop_update):
                                    tp_updated = True
                                    last_tp_info["init_i"][col] = i
                                    last_tp_info["init_price"][col] = new_init_price
                                    last_tp_info["curr_stop"][col] = _tp_stop

                            if sl_updated or tsl_updated or tp_updated:
                                if tsl_updated:
                                    # Update highest/lowest price
                                    if can_use_ohlc:
                                        _open = flex_select_auto_nb(open, i, col, flex_2d)
                                        _high = flex_select_auto_nb(high, i, col, flex_2d)
                                        _low = flex_select_auto_nb(low, i, col, flex_2d)
                                        _close = flex_select_auto_nb(close, i, col, flex_2d)
                                        _high, _low = resolve_hl_nb(_open, _high, _low, _close)
                                    else:
                                        _open = np.nan
                                        _high = _low = _close = flex_select_auto_nb(close, i, col, flex_2d)
                                    if tsl_updated:
                                        if position_now > 0:
                                            if _high > last_tsl_info["curr_price"][col]:
                                                last_tsl_info["curr_i"][col] = i
                                                last_tsl_info["curr_price"][col] = _high
                                        elif position_now < 0:
                                            if _low < last_tsl_info["curr_price"][col]:
                                                last_tsl_info["curr_i"][col] = i
                                                last_tsl_info["curr_price"][col] = _low

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    last_cash[group] = cash_now
                    last_free_cash[group] = free_cash_now
                    last_value[group] = value_now
                    last_return[group] = return_now

            for col in range(from_col, to_col):
                # Update valuation price using current close
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                if not np.isnan(_close) or not ffill_val_price:
                    last_val_price[col] = _close

                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                _cash_dividends = flex_select_auto_nb(cash_dividends, i, col, flex_2d)
                _cash_earnings += _cash_dividends * last_position[col]
                if _cash_earnings < 0:
                    _cash_earnings = max(_cash_earnings, -last_cash[group])
                last_cash[group] += _cash_earnings
                last_free_cash[group] += _cash_earnings
                if track_cash_earnings:
                    cash_earnings_out[i, col] += _cash_earnings

            # Update value and return
            group_value = last_cash[group]
            for col in range(from_col, to_col):
                if last_position[col] != 0:
                    group_value += last_position[col] * last_val_price[col]
            last_value[group] = group_value
            last_return[group] = returns_nb_.get_return_nb(
                prev_close_value[group],
                last_value[group] - _cash_deposits,
            )
            prev_close_value[group] = last_value[group]
            if fill_returns:
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


@register_jitted
def holding_enex_signal_func_nb(
    c: SignalContext,
    direction: int,
    close_at_end: bool,
) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals for holding."""
    if c.last_position[c.col] == 0:
        if direction == Direction.LongOnly:
            return True, False, False, False
        if direction == Direction.ShortOnly:
            return False, False, True, False
        raise ValueError("Either long-only or short-only direction is allowed for holding")
    if close_at_end and c.i == c.target_shape[0] - 1:
        if direction == Direction.LongOnly:
            return False, True, False, False
        if direction == Direction.ShortOnly:
            return False, False, False, True
        raise ValueError("Either long-only or short-only direction is allowed for holding")
    return False, False, False, False


AdjustFuncT = tp.Callable[[SignalContext, tp.VarArg()], None]


@register_jitted
def no_adjust_func_nb(c: SignalContext, *args) -> None:
    """Placeholder adjustment function."""
    return None


@register_jitted
def dir_enex_signal_func_nb(
    c: SignalContext,
    entries: tp.FlexArray,
    exits: tp.FlexArray,
    direction: tp.FlexArray,
    adjust_func_nb: AdjustFuncT = no_adjust_func_nb,
    adjust_args: tp.Args = (),
) -> tp.Tuple[bool, bool, bool, bool]:
    """Resolve direction-aware signals out of entries, exits, and direction.

    The direction of each pair of signals is taken from `direction` argument:

    * True, True, `Direction.LongOnly` -> True, True, False, False
    * True, True, `Direction.ShortOnly` -> False, False, True, True
    * True, True, `Direction.Both` -> True, False, True, False

    Best to use when the direction doesn't change throughout time.

    Prior to returning the signals, calls user-defined `adjust_func_nb`, which can be used to adjust
    stop values in the context. Must accept `vectorbtpro.portfolio.enums.SignalContext` and `*adjust_args`,
    and return nothing."""
    adjust_func_nb(c, *adjust_args)

    is_entry = flex_select_auto_nb(entries, c.i, c.col, c.flex_2d)
    is_exit = flex_select_auto_nb(exits, c.i, c.col, c.flex_2d)
    _direction = flex_select_auto_nb(direction, c.i, c.col, c.flex_2d)
    if _direction == Direction.LongOnly:
        return is_entry, is_exit, False, False
    if _direction == Direction.ShortOnly:
        return False, False, is_entry, is_exit
    return is_entry, False, is_exit, False


@register_jitted
def ls_enex_signal_func_nb(
    c: SignalContext,
    long_entries: tp.FlexArray,
    long_exits: tp.FlexArray,
    short_entries: tp.FlexArray,
    short_exits: tp.FlexArray,
    adjust_func_nb: AdjustFuncT = no_adjust_func_nb,
    adjust_args: tp.Args = (),
) -> tp.Tuple[bool, bool, bool, bool]:
    """Get an element of direction-aware signals.

    The direction is already built into the arrays. Best to use when the direction changes frequently
    (for example, if you have one indicator providing long signals and one providing short signals).

    Prior to returning the signals, calls user-defined `adjust_func_nb`, which can be used to adjust
    stop values in the context. Must accept `vectorbtpro.portfolio.enums.SignalContext` and `*adjust_args`,
    and return nothing."""
    adjust_func_nb(c, *adjust_args)

    is_long_entry = flex_select_auto_nb(long_entries, c.i, c.col, c.flex_2d)
    is_long_exit = flex_select_auto_nb(long_exits, c.i, c.col, c.flex_2d)
    is_short_entry = flex_select_auto_nb(short_entries, c.i, c.col, c.flex_2d)
    is_short_exit = flex_select_auto_nb(short_exits, c.i, c.col, c.flex_2d)
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit
