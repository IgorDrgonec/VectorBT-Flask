# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio modeling based on signals."""

import numpy as np
from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.signals.enums import StopType
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
def resolve_stop_exit_price_nb(
    stop_price: float,
    curr_close: float,
    stop_exit_price: int,
) -> float:
    """Resolve the exit price of a stop order."""
    if stop_exit_price == StopExitPrice.Close:
        return curr_close
    return stop_price


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
    value_now: float,
) -> tp.Tuple[float, int, int]:
    """Translate direction-aware signals into size, size type, and direction."""
    if (
        size_type != SizeType.Amount
        and size_type != SizeType.Value
        and size_type != SizeType.Percent
        and size_type != SizeType.Percent100
        and size_type != SizeType.ValuePercent
        and size_type != SizeType.ValuePercent100
    ):
        raise ValueError("Only Amount, Value, Percent(100), and ValuePercent(100) are supported as size type")
    if is_less_nb(size, 0):
        raise ValueError("Negative size is not allowed. Please express direction using signals.")
    if size_type == SizeType.Percent100:
        size /= 100
        size_type = SizeType.Percent
    if size_type == SizeType.ValuePercent100:
        size /= 100
        size_type = SizeType.ValuePercent
    if size_type == SizeType.ValuePercent:
        size *= value_now
        size_type = SizeType.Value
    order_size = np.nan
    direction = Direction.Both
    abs_position_now = abs(position_now)

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
def check_limit_hit_nb(
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
    price: float,
    size: float,
    direction: int,
    limit_delta: float,
    delta_format: int,
    can_use_ohlc: bool = True,
    check_open: bool = True,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the limit price and check whether it has been hit.

    Returns the limit price, whether it has been hit before open, and whether it has been hit during this bar.

    If `check_open` is True and hit before open, returns open."""
    if size == 0:
        raise ValueError("Limit order size cannot be zero")
    _size = get_diraware_size_nb(size, direction)
    if delta_format == DeltaFormat.Percent100:
        limit_delta /= 100
    if not np.isnan(limit_delta):
        if _size > 0:
            if np.isinf(limit_delta):
                if limit_delta > 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = price - limit_delta
                else:
                    limit_price = price * (1 - limit_delta)
        else:
            if np.isinf(limit_delta):
                if limit_delta < 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = price + limit_delta
                else:
                    limit_price = price * (1 + limit_delta)
    else:
        limit_price = price
    hit_on_open = False
    if can_use_ohlc:
        curr_high, curr_low = resolve_hl_nb(curr_open, curr_high, curr_low, curr_close)
        if _size > 0:  # buy order
            if check_open and curr_open <= limit_price:
                hit_on_open = True
                hit = True
                limit_price = curr_open
            else:
                hit = curr_low <= limit_price
                if hit and np.isinf(limit_price):
                    limit_price = curr_low
        else:  # sell order
            if check_open and curr_open >= limit_price:
                hit_on_open = True
                hit = True
                limit_price = curr_open
            else:
                hit = curr_high >= limit_price
                if hit and np.isinf(limit_price):
                    limit_price = curr_high
    else:
        if _size > 0:  # buy order
            hit = curr_close <= limit_price
        else:  # sell order
            hit = curr_close >= limit_price
        if hit and np.isinf(limit_price):
            limit_price = curr_close
    return limit_price, hit_on_open, hit


@register_jitted(cache=True)
def check_limit_expired_nb(
    index: tp.Optional[tp.Array1d],
    freq: tp.Optional[int],
    creation_i: int,
    i: int,
    tif: int,
    expiry: int,
    time_delta_format: int,
) -> tp.Tuple[bool, bool]:
    """Check whether limit is expired by comparing the current index with the creation index.

    Returns whether the limit expires already on open, and whether the limit expires during this bar."""
    if tif == -1 and expiry == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if creation_i + tif <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < creation_i + tif < i + 1:
                is_expired = True
        if expiry != -1:
            if expiry <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < expiry < i + 1:
                is_expired = True
        return is_expired_on_open, is_expired
    if time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Index must be provided for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Frequency must be provided for TimeDeltaFormat.Index")
        if index is not None and freq is not None:
            is_expired_on_open = False
            is_expired = False
            if tif != -1:
                if index[creation_i] + tif <= index[i]:
                    is_expired_on_open = True
                    is_expired = True
                elif index[i] < index[creation_i] + tif < index[i] + freq:
                    is_expired = True
            if expiry != -1:
                if expiry <= index[i]:
                    is_expired_on_open = True
                    is_expired = True
                elif index[i] < expiry < index[i] + freq:
                    is_expired = True
            return is_expired_on_open, is_expired
    return False, False


@register_jitted(cache=True)
def check_stop_hit_nb(
    is_position_long: bool,
    stop_price: float,
    stop: float,
    delta_format: int,
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
    hit_below: bool,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the stop price and check whether it has been hit.

    If hit before open, returns open.

    Returns the stop price, whether it has been hit before open, and whether it has been hit during this bar."""
    curr_high, curr_low = resolve_hl_nb(curr_open, curr_high, curr_low, curr_close)
    if delta_format == DeltaFormat.Percent100:
        stop /= 100
    if (is_position_long and hit_below) or (not is_position_long and not hit_below):
        if delta_format == DeltaFormat.Absolute:
            stop_price = stop_price - abs(stop)
        else:
            stop_price = stop_price * (1 - abs(stop))
        if curr_open <= stop_price:
            return curr_open, True, True
        if curr_low <= stop_price:
            return stop_price, False, True
        return stop_price, False, False
    if delta_format == DeltaFormat.Absolute:
        stop_price = stop_price + abs(stop)
    else:
        stop_price = stop_price * (1 + abs(stop))
    if curr_open >= stop_price:
        return curr_open, True, True
    if curr_high >= stop_price:
        return stop_price, False, True
    return stop_price, False, False


@register_jitted(cache=True)
def check_tsl_th_hit_nb(
    is_position_long: bool,
    init_price: float,
    curr_price: float,
    threshold: float,
    delta_format: int,
) -> bool:
    """Return whether TSL delta has been hit."""
    if delta_format == DeltaFormat.Percent100:
        threshold /= 100
    if is_position_long:
        if delta_format == DeltaFormat.Absolute:
            if curr_price - init_price >= abs(threshold):
                return True
        else:
            if curr_price / init_price - 1 >= abs(threshold):
                return True
    else:
        if delta_format == DeltaFormat.Absolute:
            if curr_price - init_price <= -abs(threshold):
                return True
        else:
            if curr_price / init_price - 1 <= -abs(threshold):
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
        index=None,
        freq=None,
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
        order_type=portfolio_ch.flex_array_gl_slicer,
        limit_delta=portfolio_ch.flex_array_gl_slicer,
        limit_tif=portfolio_ch.flex_array_gl_slicer,
        limit_expiry=portfolio_ch.flex_array_gl_slicer,
        upon_adj_limit_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opp_limit_conflict=portfolio_ch.flex_array_gl_slicer,
        use_stops=None,
        sl_stop=portfolio_ch.flex_array_gl_slicer,
        tsl_th=portfolio_ch.flex_array_gl_slicer,
        tsl_stop=portfolio_ch.flex_array_gl_slicer,
        tp_stop=portfolio_ch.flex_array_gl_slicer,
        stop_entry_price=portfolio_ch.flex_array_gl_slicer,
        stop_exit_price=portfolio_ch.flex_array_gl_slicer,
        stop_order_type=portfolio_ch.flex_array_gl_slicer,
        stop_limit_delta=portfolio_ch.flex_array_gl_slicer,
        upon_stop_exit=portfolio_ch.flex_array_gl_slicer,
        upon_stop_update=portfolio_ch.flex_array_gl_slicer,
        upon_adj_stop_conflict=portfolio_ch.flex_array_gl_slicer,
        upon_opp_stop_conflict=portfolio_ch.flex_array_gl_slicer,
        delta_format=portfolio_ch.flex_array_gl_slicer,
        time_delta_format=portfolio_ch.flex_array_gl_slicer,
        from_ago=portfolio_ch.flex_array_gl_slicer,
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
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
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
    order_type: tp.FlexArray = np.asarray(OrderType.Market),
    limit_delta: tp.FlexArray = np.asarray(np.nan),
    limit_tif: tp.FlexArray = np.asarray(-1),
    limit_expiry: tp.FlexArray = np.asarray(-1),
    upon_adj_limit_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepIgnore),
    upon_opp_limit_conflict: tp.FlexArray = np.asarray(PendingConflictMode.CancelExecute),
    use_stops: bool = True,
    sl_stop: tp.FlexArray = np.asarray(np.nan),
    tsl_th: tp.FlexArray = np.asarray(np.nan),
    tsl_stop: tp.FlexArray = np.asarray(np.nan),
    tp_stop: tp.FlexArray = np.asarray(np.nan),
    stop_entry_price: tp.FlexArray = np.asarray(StopEntryPrice.Close),
    stop_exit_price: tp.FlexArray = np.asarray(StopExitPrice.Stop),
    stop_order_type: tp.FlexArray = np.asarray(OrderType.Market),
    stop_limit_delta: tp.FlexArray = np.asarray(np.nan),
    upon_stop_exit: tp.FlexArray = np.asarray(StopExitMode.Close),
    upon_stop_update: tp.FlexArray = np.asarray(StopUpdateMode.Keep),
    upon_adj_stop_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepExecute),
    upon_opp_stop_conflict: tp.FlexArray = np.asarray(PendingConflictMode.KeepExecute),
    delta_format: tp.FlexArray = np.asarray(DeltaFormat.Percent),
    time_delta_format: tp.FlexArray = np.asarray(TimeDeltaFormat.Index),
    from_ago: tp.FlexArray = np.asarray(0),
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

        Single value must be passed as a 0-dim array (for example, by using `np.asarray(value)`).
    """
    check_group_lens_nb(group_lens, target_shape[1])
    cash_sharing = is_grouped_nb(group_lens)

    if max_orders is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=fs_order_dt)
    else:
        order_records = np.empty((max_orders, target_shape[1]), dtype=fs_order_dt)
    if max_logs is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_logs, target_shape[1]), dtype=log_dt)
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
    last_limit_info["signal_i"][:] = -1
    last_limit_info["creation_i"][:] = -1
    last_limit_info["init_i"][:] = -1
    last_limit_info["init_price"][:] = np.nan
    last_limit_info["init_size"][:] = np.nan
    last_limit_info["init_size_type"][:] = -1
    last_limit_info["init_direction"][:] = -1
    last_limit_info["init_stop_type"][:] = -1
    last_limit_info["delta"][:] = np.nan
    last_limit_info["delta_format"][:] = -1
    last_limit_info["tif"][:] = -1
    last_limit_info["expiry"][:] = -1
    last_limit_info["time_delta_format"][:] = -1

    if use_stops:
        last_sl_info = np.empty(target_shape[1], dtype=sl_info_dt)
        last_sl_info["init_i"][:] = -1
        last_sl_info["init_price"][:] = np.nan
        last_sl_info["stop"][:] = np.nan
        last_sl_info["limit_delta"][:] = np.nan
        last_sl_info["delta_format"][:] = -1

        last_tsl_info = np.empty(target_shape[1], dtype=tsl_info_dt)
        last_tsl_info["init_i"][:] = -1
        last_tsl_info["init_price"][:] = np.nan
        last_tsl_info["peak_i"][:] = -1
        last_tsl_info["peak_price"][:] = np.nan
        last_tsl_info["th"][:] = np.nan
        last_tsl_info["stop"][:] = np.nan
        last_tsl_info["limit_delta"][:] = np.nan
        last_tsl_info["delta_format"][:] = -1

        last_tp_info = np.empty(target_shape[1], dtype=tp_info_dt)
        last_tp_info["init_i"][:] = -1
        last_tp_info["init_price"][:] = np.nan
        last_tp_info["stop"][:] = np.nan
        last_tp_info["limit_delta"][:] = np.nan
        last_tp_info["delta_format"][:] = -1
    else:
        last_sl_info = np.empty(0, dtype=sl_info_dt)
        last_tsl_info = np.empty(0, dtype=tsl_info_dt)
        last_tp_info = np.empty(0, dtype=tp_info_dt)

    main_info = np.empty(target_shape[1], dtype=main_info_dt)

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
                        _i = i - abs(flex_select_auto_nb(from_ago, i, col, flex_2d))
                        if _i < 0:
                            _price = np.nan
                        else:
                            _price = flex_select_auto_nb(price, _i, col, flex_2d)
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
                main_info["signal_i"][col] = -1
                main_info["creation_i"][col] = -1
                main_info["i"][col] = i
                main_info["price"][col] = np.nan
                main_info["size"][col] = np.nan
                main_info["size_type"][col] = -1
                main_info["direction"][col] = -1
                main_info["type"][col] = -1
                main_info["stop_type"][col] = -1
                temp_order_value[col] = 0.0

                # Get signals
                signal_ctx = SignalContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    index=index,
                    freq=freq,
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
                    not np.isnan(last_sl_info["stop"][col])
                    or not np.isnan(last_tsl_info["stop"][col])
                    or not np.isnan(last_tp_info["stop"][col])
                )
                any_user_signal = is_long_entry or is_long_exit or is_short_entry or is_short_exit
                if not any_limit_signal and not any_stop_signal and not any_user_signal:  # shortcut
                    continue

                # Set initial info
                exec_limit_set = False
                exec_limit_set_on_open = False
                exec_limit_signal_i = -1
                exec_limit_creation_i = -1
                exec_limit_init_i = -1
                exec_limit_price = np.nan
                exec_limit_size = np.nan
                exec_limit_size_type = -1
                exec_limit_direction = -1
                exec_limit_stop_type = -1

                exec_stop_set = False
                exec_stop_set_on_open = False
                exec_stop_set_on_close = False
                exec_stop_init_i = -1
                exec_stop_price = np.nan
                exec_stop_size = np.nan
                exec_stop_size_type = -1
                exec_stop_direction = -1
                exec_stop_type = -1
                exec_stop_stop_type = -1
                exec_stop_delta = np.nan
                exec_stop_delta_format = -1
                exec_stop_make_limit = False

                user_on_open = False
                user_on_close = False
                exec_user_set = False
                exec_user_price = np.nan
                exec_user_size = np.nan
                exec_user_size_type = -1
                exec_user_direction = -1
                exec_user_type = -1
                exec_user_stop_type = -1
                exec_user_make_limit = False

                # Resolve the current bar
                _open = flex_select_auto_nb(open, i, col, flex_2d)
                _high = flex_select_auto_nb(high, i, col, flex_2d)
                _low = flex_select_auto_nb(low, i, col, flex_2d)
                _close = flex_select_auto_nb(close, i, col, flex_2d)
                _high, _low = resolve_hl_nb(_open, _high, _low, _close)

                # Process the limit signal
                if any_limit_signal:
                    # Check whether the limit price has been hit
                    _signal_i = last_limit_info["signal_i"][col]
                    _creation_i = last_limit_info["creation_i"][col]
                    _init_i = last_limit_info["init_i"][col]
                    _price = last_limit_info["init_price"][col]
                    _size = last_limit_info["init_size"][col]
                    _size_type = last_limit_info["init_size_type"][col]
                    _direction = last_limit_info["init_direction"][col]
                    _stop_type = last_limit_info["init_stop_type"][col]
                    _delta = last_limit_info["delta"][col]
                    _delta_format = last_limit_info["delta_format"][col]
                    _tif = last_limit_info["tif"][col]
                    _expiry = last_limit_info["expiry"][col]
                    _time_delta_format = last_limit_info["time_delta_format"][col]

                    limit_expired_on_open, limit_expired = check_limit_expired_nb(
                        index,
                        freq,
                        _creation_i,
                        i,
                        _tif,
                        _expiry,
                        _time_delta_format,
                    )
                    limit_price, limit_hit_on_open, limit_hit = check_limit_hit_nb(
                        _open,
                        _high,
                        _low,
                        _close,
                        _price,
                        _size,
                        _direction,
                        _delta,
                        _delta_format,
                        can_use_ohlc=True,
                        check_open=True,
                    )
                    if limit_expired_on_open or (not limit_hit_on_open and limit_expired):
                        # Expired limit signal
                        any_limit_signal = False

                        last_limit_info["signal_i"][col] = -1
                        last_limit_info["creation_i"][col] = -1
                        last_limit_info["init_i"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1
                    else:
                        # Save info
                        if limit_hit:
                            # Executable limit signal
                            exec_limit_set = True
                            exec_limit_set_on_open = limit_hit_on_open
                            exec_limit_signal_i = _signal_i
                            exec_limit_creation_i = _creation_i
                            exec_limit_init_i = _init_i
                            exec_limit_price = limit_price
                            exec_limit_size = _size
                            exec_limit_size_type = _size_type
                            exec_limit_direction = _direction
                            exec_limit_stop_type = _stop_type

                # Process the stop signal
                if any_stop_signal:
                    # Resolve the stop price
                    stop_price = np.nan
                    stop_hit_on_open = False
                    stop_hit = False

                    # Check SL
                    if not np.isnan(last_sl_info["stop"][col]):
                        # Check against high and low
                        stop_price, stop_hit_on_open, stop_hit = check_stop_hit_nb(
                            last_position[col] > 0,
                            last_sl_info["init_price"][col],
                            last_sl_info["stop"][col],
                            last_sl_info["delta_format"][col],
                            _open,
                            _high,
                            _low,
                            _close,
                            True,
                        )
                        if stop_hit:
                            _stop_type = StopType.SL
                            _init_i = last_sl_info["init_i"][col]
                            _limit_delta = last_sl_info["limit_delta"][col]
                            _delta_format = last_sl_info["delta_format"][col]

                    # Check TSL and TTP
                    if not stop_hit and not np.isnan(last_tsl_info["stop"][col]):
                        # Check against high and low
                        if np.isnan(last_tsl_info["th"][col]):
                            delta_hit = True
                        else:
                            delta_hit = check_tsl_th_hit_nb(
                                last_position[col] > 0,
                                last_tsl_info["init_price"][col],
                                last_tsl_info["peak_price"][col],
                                last_tsl_info["th"][col],
                                last_tsl_info["delta_format"][col],
                            )
                        if delta_hit:
                            stop_price, stop_hit_on_open, stop_hit = check_stop_hit_nb(
                                last_position[col] > 0,
                                last_tsl_info["peak_price"][col],
                                last_tsl_info["stop"][col],
                                last_tsl_info["delta_format"][col],
                                _open,
                                _high,
                                _low,
                                _close,
                                True,
                            )
                        if last_position[col] > 0:
                            if _high > last_tsl_info["peak_price"][col]:
                                last_tsl_info["peak_i"][col] = i
                                last_tsl_info["peak_price"][col] = _high
                        elif last_position[col] < 0:
                            if _low < last_tsl_info["peak_price"][col]:
                                last_tsl_info["peak_i"][col] = i
                                last_tsl_info["peak_price"][col] = _low
                        if not stop_hit:
                            # After update, check one more time against close
                            if np.isnan(last_tsl_info["th"][col]):
                                delta_hit = True
                            else:
                                delta_hit = check_tsl_th_hit_nb(
                                    last_position[col] > 0,
                                    last_tsl_info["init_price"][col],
                                    last_tsl_info["peak_price"][col],
                                    last_tsl_info["th"][col],
                                    last_tsl_info["delta_format"][col],
                                )
                            if delta_hit:
                                stop_price, stop_hit_on_open, stop_hit = check_stop_hit_nb(
                                    last_position[col] > 0,
                                    last_tsl_info["peak_price"][col],
                                    last_tsl_info["stop"][col],
                                    last_tsl_info["delta_format"][col],
                                    _open,
                                    _close,
                                    _close,
                                    _close,
                                    True,
                                )
                        if stop_hit:
                            if np.isnan(last_tsl_info["th"][col]):
                                _stop_type = StopType.TSL
                            else:
                                _stop_type = StopType.TTP
                            _init_i = last_tsl_info["init_i"][col]
                            _limit_delta = last_tsl_info["limit_delta"][col]
                            _delta_format = last_tsl_info["delta_format"][col]

                    # Check TP
                    if not stop_hit and not np.isnan(last_tp_info["stop"][col]):
                        # Check against high and low
                        stop_price, stop_hit_on_open, stop_hit = check_stop_hit_nb(
                            last_position[col] > 0,
                            last_tp_info["init_price"][col],
                            last_tp_info["stop"][col],
                            last_tp_info["delta_format"][col],
                            _open,
                            _high,
                            _low,
                            _close,
                            False,
                        )
                        if stop_hit:
                            _stop_type = StopType.TP
                            _init_i = last_tp_info["init_i"][col]
                            _limit_delta = last_tp_info["limit_delta"][col]
                            _delta_format = last_tp_info["delta_format"][col]

                    if stop_hit:
                        # Stop price has been hit
                        # Resolve the final stop signal
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

                        # Resolve the price
                        _stop_exit_price = flex_select_auto_nb(stop_exit_price, i, col, flex_2d)
                        _price = resolve_stop_exit_price_nb(stop_price, _close, _stop_exit_price)

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
                            last_value[group],
                        )

                        if not np.isnan(_size):
                            # Executable stop signal
                            can_execute = True
                            _stop_order_type = flex_select_auto_nb(stop_order_type, i, col, flex_2d)
                            if _stop_order_type == OrderType.Limit:
                                # Use close to check whether the limit price has been hit
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    _open,
                                    _high,
                                    _low,
                                    _close,
                                    _price,
                                    _size,
                                    _direction,
                                    _limit_delta,
                                    _delta_format,
                                    can_use_ohlc=False,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                            # Save info
                            exec_stop_set = True
                            exec_stop_set_on_open = stop_hit_on_open
                            exec_stop_set_on_close = _stop_exit_price == StopExitPrice.Close
                            exec_stop_init_i = _init_i
                            exec_stop_price = _price
                            exec_stop_size = _size
                            exec_stop_size_type = _size_type
                            exec_stop_direction = _direction
                            exec_stop_type = _stop_order_type
                            exec_stop_stop_type = _stop_type
                            exec_stop_delta = _limit_delta
                            exec_stop_delta_format = _delta_format
                            exec_stop_make_limit = not can_execute

                # Process user signal
                if any_user_signal:
                    _i = i - abs(flex_select_auto_nb(from_ago, i, col, flex_2d))
                    if _i < 0:
                        _price = np.nan
                        _size = np.nan
                        _size_type = -1
                        _direction = -1
                    else:
                        _accumulate = flex_select_auto_nb(accumulate, _i, col, flex_2d)
                        if is_long_entry or is_short_entry:
                            # Resolve any conflicts
                            _upon_long_conflict = flex_select_auto_nb(upon_long_conflict, _i, col, flex_2d)
                            is_long_entry, is_long_exit = resolve_signal_conflict_nb(
                                last_position[col],
                                is_long_entry,
                                is_long_exit,
                                Direction.LongOnly,
                                _upon_long_conflict,
                            )
                            _upon_short_conflict = flex_select_auto_nb(upon_short_conflict, _i, col, flex_2d)
                            is_short_entry, is_short_exit = resolve_signal_conflict_nb(
                                last_position[col],
                                is_short_entry,
                                is_short_exit,
                                Direction.ShortOnly,
                                _upon_short_conflict,
                            )

                            # Resolve anu direction conflicts
                            _upon_dir_conflict = flex_select_auto_nb(upon_dir_conflict, _i, col, flex_2d)
                            is_long_entry, is_short_entry = resolve_dir_conflict_nb(
                                last_position[col],
                                is_long_entry,
                                is_short_entry,
                                _upon_dir_conflict,
                            )

                            # Resolve opposite entry
                            _upon_opposite_entry = flex_select_auto_nb(upon_opposite_entry, _i, col, flex_2d)
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

                        # Resolve the price
                        _price = flex_select_auto_nb(price, _i, col, flex_2d)

                        # Convert both signals to size (direction-aware), size type, and direction
                        _size, _size_type, _direction = signals_to_size_nb(
                            last_position[col],
                            is_long_entry,
                            is_long_exit,
                            is_short_entry,
                            is_short_exit,
                            flex_select_auto_nb(size, _i, col, flex_2d),
                            flex_select_auto_nb(size_type, _i, col, flex_2d),
                            _accumulate,
                            last_val_price[col],
                            last_value[group],
                        )

                    if np.isinf(_price):
                        if _price > 0:
                            user_on_close = True
                        else:
                            user_on_open = True
                    if not np.isnan(_size):
                        # Executable user signal
                        can_execute = True
                        _order_type = flex_select_auto_nb(order_type, _i, col, flex_2d)
                        if _order_type == OrderType.Limit:
                            # Use close to check whether the limit price has been hit
                            can_use_ohlc = False
                            if np.isinf(_price):
                                if _price > 0:
                                    # Cannot place a limit order at the close price and execute right away
                                    _price = _close
                                    can_execute = False
                                else:
                                    can_use_ohlc = True
                                    _price = _open
                            if can_execute:
                                _limit_delta = flex_select_auto_nb(limit_delta, _i, col, flex_2d)
                                _delta_format = flex_select_auto_nb(delta_format, _i, col, flex_2d)
                                limit_price, _, can_execute = check_limit_hit_nb(
                                    _open,
                                    _high,
                                    _low,
                                    _close,
                                    _price,
                                    _size,
                                    _direction,
                                    _limit_delta,
                                    _delta_format,
                                    can_use_ohlc=can_use_ohlc,
                                    check_open=False,
                                )
                                if can_execute:
                                    _price = limit_price

                        # Save info
                        exec_user_set = True
                        exec_user_price = _price
                        exec_user_size = _size
                        exec_user_size_type = _size_type
                        exec_user_direction = _direction
                        exec_user_type = _order_type
                        exec_user_stop_type = -1
                        exec_user_make_limit = not can_execute

                if (
                    exec_limit_set
                    or exec_stop_set
                    or exec_user_set
                    or ((any_limit_signal or any_stop_signal) and any_user_signal)
                ):
                    # Choose the main executable signal
                    # Priority: limit -> stop -> user

                    # Check whether the main signal comes on open
                    keep_limit = True
                    keep_stop = True
                    execute_limit = False
                    execute_stop = False
                    execute_user = False
                    if exec_limit_set_on_open:
                        keep_limit = False
                        keep_stop = False
                        execute_limit = True
                    elif exec_stop_set_on_open:
                        keep_limit = False
                        keep_stop = False
                        execute_stop = True
                    elif any_user_signal and user_on_open:
                        execute_user = True
                        if any_limit_signal and (execute_user or not exec_user_set):
                            stop_size = get_diraware_size_nb(
                                last_limit_info["init_size"][col],
                                last_limit_info["init_direction"][col],
                            )
                            keep_limit, execute_user = resolve_pending_conflict_nb(
                                stop_size >= 0,
                                is_long_entry or is_short_exit,
                                flex_select_auto_nb(upon_adj_limit_conflict, i, col, flex_2d),
                                flex_select_auto_nb(upon_opp_limit_conflict, i, col, flex_2d),
                            )
                        if any_stop_signal and (execute_user or not exec_user_set):
                            keep_stop, execute_user = resolve_pending_conflict_nb(
                                last_position[col] < 0,
                                is_long_entry or is_short_exit,
                                flex_select_auto_nb(upon_adj_stop_conflict, i, col, flex_2d),
                                flex_select_auto_nb(upon_opp_stop_conflict, i, col, flex_2d),
                            )
                        if not exec_user_set:
                            execute_user = False
                    if not execute_limit and not execute_stop and not execute_user:
                        # Check whether the main signal comes in the middle of the bar
                        if exec_limit_set and not exec_limit_set_on_open and keep_limit:
                            keep_limit = False
                            keep_stop = False
                            execute_limit = True
                        elif exec_stop_set and not exec_stop_set_on_open and not exec_stop_set_on_close and keep_stop:
                            keep_limit = False
                            keep_stop = False
                            execute_stop = True
                        elif any_user_signal and not user_on_open and not user_on_close:
                            execute_user = True
                            if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                stop_size = get_diraware_size_nb(
                                    last_limit_info["init_size"][col],
                                    last_limit_info["init_direction"][col],
                                )
                                keep_limit, execute_user = resolve_pending_conflict_nb(
                                    stop_size >= 0,
                                    is_long_entry or is_short_exit,
                                    flex_select_auto_nb(upon_adj_limit_conflict, i, col, flex_2d),
                                    flex_select_auto_nb(upon_opp_limit_conflict, i, col, flex_2d),
                                )
                            if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                keep_stop, execute_user = resolve_pending_conflict_nb(
                                    last_position[col] < 0,
                                    is_long_entry or is_short_exit,
                                    flex_select_auto_nb(upon_adj_stop_conflict, i, col, flex_2d),
                                    flex_select_auto_nb(upon_opp_stop_conflict, i, col, flex_2d),
                                )
                            if not exec_user_set:
                                execute_user = False
                        if not execute_limit and not execute_stop and not execute_user:
                            # Check whether the main signal comes on close
                            if exec_stop_set_on_close and keep_stop:
                                keep_limit = False
                                keep_stop = False
                                execute_stop = True
                            elif any_user_signal and user_on_close:
                                execute_user = True
                                if any_limit_signal and keep_limit and (execute_user or not exec_user_set):
                                    stop_size = get_diraware_size_nb(
                                        last_limit_info["init_size"][col],
                                        last_limit_info["init_direction"][col],
                                    )
                                    keep_limit, execute_user = resolve_pending_conflict_nb(
                                        stop_size >= 0,
                                        is_long_entry or is_short_exit,
                                        flex_select_auto_nb(upon_adj_limit_conflict, i, col, flex_2d),
                                        flex_select_auto_nb(upon_opp_limit_conflict, i, col, flex_2d),
                                    )
                                if any_stop_signal and keep_stop and (execute_user or not exec_user_set):
                                    keep_stop, execute_user = resolve_pending_conflict_nb(
                                        last_position[col] < 0,
                                        is_long_entry or is_short_exit,
                                        flex_select_auto_nb(upon_adj_stop_conflict, i, col, flex_2d),
                                        flex_select_auto_nb(upon_opp_stop_conflict, i, col, flex_2d),
                                    )
                                if not exec_user_set:
                                    execute_user = False

                    # Process the limit signal
                    if execute_limit:
                        # Execute the signal
                        main_info["signal_i"][col] = exec_limit_signal_i
                        main_info["creation_i"][col] = exec_limit_creation_i
                        main_info["i"][col] = exec_limit_init_i
                        main_info["price"][col] = exec_limit_price
                        main_info["size"][col] = exec_limit_size
                        main_info["size_type"][col] = exec_limit_size_type
                        main_info["direction"][col] = exec_limit_direction
                        main_info["type"][col] = OrderType.Limit
                        main_info["stop_type"][col] = exec_limit_stop_type
                    if execute_limit or (any_limit_signal and not keep_limit):
                        # Clear the pending info
                        any_limit_signal = False

                        last_limit_info["signal_i"][col] = -1
                        last_limit_info["creation_i"][col] = -1
                        last_limit_info["init_i"][col] = -1
                        last_limit_info["init_price"][col] = np.nan
                        last_limit_info["init_size"][col] = np.nan
                        last_limit_info["init_size_type"][col] = -1
                        last_limit_info["init_direction"][col] = -1
                        last_limit_info["init_stop_type"][col] = -1
                        last_limit_info["delta"][col] = np.nan
                        last_limit_info["delta_format"][col] = -1
                        last_limit_info["tif"][col] = -1
                        last_limit_info["expiry"][col] = -1
                        last_limit_info["time_delta_format"][col] = -1

                    # Process the stop signal
                    if execute_stop:
                        # Execute the signal
                        if exec_stop_make_limit:
                            if any_limit_signal:
                                raise ValueError("Only one active limit signal is allowed at a time")

                            _limit_tif = flex_select_auto_nb(limit_tif, i, col, flex_2d)
                            _limit_expiry = flex_select_auto_nb(limit_expiry, i, col, flex_2d)
                            _time_delta_format = flex_select_auto_nb(time_delta_format, i, col, flex_2d)
                            last_limit_info["signal_i"][col] = exec_stop_init_i
                            last_limit_info["creation_i"][col] = i
                            last_limit_info["init_i"][col] = i
                            last_limit_info["init_price"][col] = exec_stop_price
                            last_limit_info["init_size"][col] = exec_stop_size
                            last_limit_info["init_size_type"][col] = exec_stop_size_type
                            last_limit_info["init_direction"][col] = exec_stop_direction
                            last_limit_info["init_stop_type"][col] = exec_stop_stop_type
                            last_limit_info["delta"][col] = exec_stop_delta
                            last_limit_info["delta_format"][col] = exec_stop_delta_format
                            last_limit_info["tif"][col] = _limit_tif
                            last_limit_info["expiry"][col] = _limit_expiry
                            last_limit_info["time_delta_format"][col] = _time_delta_format
                        else:
                            main_info["signal_i"][col] = exec_stop_init_i
                            main_info["creation_i"][col] = i
                            main_info["i"][col] = i
                            main_info["price"][col] = exec_stop_price
                            main_info["size"][col] = exec_stop_size
                            main_info["size_type"][col] = exec_stop_size_type
                            main_info["direction"][col] = exec_stop_direction
                            main_info["type"][col] = exec_stop_type
                            main_info["stop_type"][col] = exec_stop_stop_type
                    if execute_stop or (any_stop_signal and not keep_stop):
                        # Clear the pending info
                        any_stop_signal = False

                        last_sl_info["init_i"][col] = -1
                        last_sl_info["init_price"][col] = np.nan
                        last_sl_info["stop"][col] = np.nan
                        last_sl_info["limit_delta"][col] = np.nan
                        last_sl_info["delta_format"][col] = -1

                        last_tsl_info["init_i"][col] = -1
                        last_tsl_info["init_price"][col] = np.nan
                        last_tsl_info["peak_i"][col] = -1
                        last_tsl_info["peak_price"][col] = np.nan
                        last_tsl_info["th"][col] = np.nan
                        last_tsl_info["stop"][col] = np.nan
                        last_tsl_info["limit_delta"][col] = np.nan
                        last_tsl_info["delta_format"][col] = -1

                        last_tp_info["init_i"][col] = -1
                        last_tp_info["init_price"][col] = np.nan
                        last_tp_info["stop"][col] = np.nan
                        last_tp_info["limit_delta"][col] = np.nan
                        last_tp_info["delta_format"][col] = -1

                    # Process the user signal
                    if execute_user:
                        # Execute the signal
                        _i = i - abs(flex_select_auto_nb(from_ago, i, col, flex_2d))
                        if _i >= 0:
                            if exec_user_make_limit:
                                if any_limit_signal:
                                    raise ValueError("Only one active limit signal is allowed at a time")

                                _limit_delta = flex_select_auto_nb(limit_delta, _i, col, flex_2d)
                                _delta_format = flex_select_auto_nb(delta_format, _i, col, flex_2d)
                                _limit_tif = flex_select_auto_nb(limit_tif, _i, col, flex_2d)
                                _limit_expiry = flex_select_auto_nb(limit_expiry, _i, col, flex_2d)
                                _time_delta_format = flex_select_auto_nb(time_delta_format, _i, col, flex_2d)
                                last_limit_info["signal_i"][col] = _i
                                last_limit_info["creation_i"][col] = i
                                last_limit_info["init_i"][col] = _i
                                last_limit_info["init_price"][col] = exec_user_price
                                last_limit_info["init_size"][col] = exec_user_size
                                last_limit_info["init_size_type"][col] = exec_user_size_type
                                last_limit_info["init_direction"][col] = exec_user_direction
                                last_limit_info["init_stop_type"][col] = -1
                                last_limit_info["delta"][col] = _limit_delta
                                last_limit_info["delta_format"][col] = _delta_format
                                last_limit_info["tif"][col] = _limit_tif
                                last_limit_info["expiry"][col] = _limit_expiry
                                last_limit_info["time_delta_format"][col] = _time_delta_format
                            else:
                                main_info["signal_i"][col] = _i
                                main_info["creation_i"][col] = i
                                main_info["i"][col] = _i
                                main_info["price"][col] = exec_user_price
                                main_info["size"][col] = exec_user_size
                                main_info["size_type"][col] = exec_user_size_type
                                main_info["direction"][col] = exec_user_direction
                                main_info["type"][col] = exec_user_type
                                main_info["stop_type"][col] = exec_user_stop_type

                    if execute_limit or execute_stop or execute_user:
                        if cash_sharing and auto_call_seq:
                            # Approximate order value
                            if np.isnan(main_info["size"][col]):
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
                                    main_info["size"][col],
                                    main_info["size_type"][col],
                                    main_info["direction"][col],
                                )

            any_signal_set = False
            for col in range(from_col, to_col):
                if not np.isnan(main_info["size"][col]):
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
                    if np.isnan(main_info["size"][col]):  # shortcut
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
                    _i = main_info["i"][col]
                    if main_info["type"][col] == OrderType.Limit:
                        _slippage = 0.0
                    else:
                        _slippage = flex_select_auto_nb(slippage, _i, col, flex_2d)
                    order = order_nb(
                        size=main_info["size"][col],
                        price=main_info["price"][col],
                        size_type=main_info["size_type"][col],
                        direction=main_info["direction"][col],
                        fees=flex_select_auto_nb(fees, _i, col, flex_2d),
                        fixed_fees=flex_select_auto_nb(fixed_fees, _i, col, flex_2d),
                        slippage=_slippage,
                        min_size=flex_select_auto_nb(min_size, _i, col, flex_2d),
                        max_size=flex_select_auto_nb(max_size, _i, col, flex_2d),
                        size_granularity=flex_select_auto_nb(size_granularity, _i, col, flex_2d),
                        reject_prob=flex_select_auto_nb(reject_prob, _i, col, flex_2d),
                        price_area_vio_mode=flex_select_auto_nb(price_area_vio_mode, _i, col, flex_2d),
                        lock_cash=flex_select_auto_nb(lock_cash, _i, col, flex_2d),
                        allow_partial=flex_select_auto_nb(allow_partial, _i, col, flex_2d),
                        raise_reject=flex_select_auto_nb(raise_reject, _i, col, flex_2d),
                        log=flex_select_auto_nb(log, _i, col, flex_2d),
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

                    # Append more order information
                    if order_result.status == OrderStatus.Filled:
                        if order_counts[col] >= 1:
                            order_records["signal_idx"][order_counts[col] - 1, col] = main_info["signal_i"][col]
                            order_records["creation_idx"][order_counts[col] - 1, col] = main_info["creation_i"][col]
                            order_records["type"][order_counts[col] - 1, col] = main_info["type"][col]
                            order_records["stop_type"][order_counts[col] - 1, col] = main_info["stop_type"][col]

                    # Update execution state
                    cash_now = new_exec_state.cash
                    position_now = new_exec_state.position
                    debt_now = new_exec_state.debt
                    free_cash_now = new_exec_state.free_cash
                    val_price_now = new_exec_state.val_price
                    value_now = new_exec_state.value

                    if use_stops:
                        # Update stop price
                        if position_now == 0:
                            # Not in position anymore -> clear stops (irrespective of order success)
                            last_sl_info["init_i"][col] = -1
                            last_sl_info["init_price"][col] = np.nan
                            last_sl_info["stop"][col] = np.nan
                            last_sl_info["limit_delta"][col] = np.nan
                            last_sl_info["delta_format"][col] = -1

                            last_tsl_info["init_i"][col] = -1
                            last_tsl_info["init_price"][col] = np.nan
                            last_tsl_info["peak_i"][col] = -1
                            last_tsl_info["peak_price"][col] = np.nan
                            last_tsl_info["th"][col] = np.nan
                            last_tsl_info["stop"][col] = np.nan
                            last_tsl_info["limit_delta"][col] = np.nan
                            last_tsl_info["delta_format"][col] = -1

                            last_tp_info["init_i"][col] = -1
                            last_tp_info["init_price"][col] = np.nan
                            last_tp_info["stop"][col] = np.nan
                            last_tp_info["limit_delta"][col] = np.nan
                            last_tp_info["delta_format"][col] = -1

                        if order_result.status == OrderStatus.Filled and position_now != 0:
                            # Order filled and in position -> possibly set stops
                            _price = main_info["price"][col]
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
                            _tsl_th = abs(flex_select_auto_nb(tsl_th, i, col, flex_2d))
                            _tsl_stop = abs(flex_select_auto_nb(tsl_stop, i, col, flex_2d))
                            _tp_stop = abs(flex_select_auto_nb(tp_stop, i, col, flex_2d))
                            _stop_limit_delta = abs(flex_select_auto_nb(stop_limit_delta, i, col, flex_2d))
                            _delta_format = abs(flex_select_auto_nb(delta_format, i, col, flex_2d))

                            sl_updated = tsl_updated = tp_updated = False
                            if exec_state.position == 0 or np.sign(position_now) != np.sign(exec_state.position):
                                # Position opened/reversed -> set stops
                                sl_updated = True
                                last_sl_info["init_i"][col] = i
                                last_sl_info["init_price"][col] = new_init_price
                                last_sl_info["stop"][col] = _sl_stop
                                last_sl_info["limit_delta"][col] = _stop_limit_delta
                                last_sl_info["delta_format"][col] = _delta_format

                                tsl_updated = True
                                last_tsl_info["init_i"][col] = i
                                last_tsl_info["init_price"][col] = new_init_price
                                last_tsl_info["peak_i"][col] = i
                                last_tsl_info["peak_price"][col] = new_init_price
                                last_tsl_info["th"][col] = _tsl_th
                                last_tsl_info["stop"][col] = _tsl_stop
                                last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                last_tsl_info["delta_format"][col] = _delta_format

                                tp_updated = True
                                last_tp_info["init_i"][col] = i
                                last_tp_info["init_price"][col] = new_init_price
                                last_tp_info["stop"][col] = _tp_stop
                                last_tp_info["limit_delta"][col] = _stop_limit_delta
                                last_tp_info["delta_format"][col] = _delta_format

                            elif abs(position_now) > abs(exec_state.position):
                                # Position increased -> keep/override stops
                                _upon_stop_update = flex_select_auto_nb(upon_stop_update, i, col, flex_2d)
                                if should_update_stop_nb(_sl_stop, _upon_stop_update):
                                    sl_updated = True
                                    last_sl_info["init_i"][col] = i
                                    last_sl_info["init_price"][col] = new_init_price
                                    last_sl_info["stop"][col] = _sl_stop
                                    last_sl_info["limit_delta"][col] = _stop_limit_delta
                                    last_sl_info["delta_format"][col] = _delta_format
                                if should_update_stop_nb(_tsl_stop, _upon_stop_update):
                                    tsl_updated = True
                                    last_tsl_info["init_i"][col] = i
                                    last_tsl_info["init_price"][col] = new_init_price
                                    last_tsl_info["peak_i"][col] = i
                                    last_tsl_info["peak_price"][col] = new_init_price
                                    last_tsl_info["th"][col] = _tsl_th
                                    last_tsl_info["stop"][col] = _tsl_stop
                                    last_tsl_info["limit_delta"][col] = _stop_limit_delta
                                    last_tsl_info["delta_format"][col] = _delta_format
                                if should_update_stop_nb(_tp_stop, _upon_stop_update):
                                    tp_updated = True
                                    last_tp_info["init_i"][col] = i
                                    last_tp_info["init_price"][col] = new_init_price
                                    last_tp_info["stop"][col] = _tp_stop
                                    last_tp_info["limit_delta"][col] = _stop_limit_delta
                                    last_tp_info["delta_format"][col] = _delta_format

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
                                            if _high > last_tsl_info["peak_price"][col]:
                                                last_tsl_info["peak_i"][col] = i
                                                last_tsl_info["peak_price"][col] = _high
                                        elif position_now < 0:
                                            if _low < last_tsl_info["peak_price"][col]:
                                                last_tsl_info["peak_i"][col] = i
                                                last_tsl_info["peak_price"][col] = _low

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
    from_ago: tp.FlexArray = np.asarray(0),
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

    _i = c.i - abs(flex_select_auto_nb(from_ago, c.i, c.col, c.flex_2d))
    if _i < 0:
        return False, False, False, False
    is_entry = flex_select_auto_nb(entries, _i, c.col, c.flex_2d)
    is_exit = flex_select_auto_nb(exits, _i, c.col, c.flex_2d)
    _direction = flex_select_auto_nb(direction, _i, c.col, c.flex_2d)
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
    from_ago: tp.FlexArray = np.asarray(0),
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

    _i = c.i - abs(flex_select_auto_nb(from_ago, c.i, c.col, c.flex_2d))
    if _i < 0:
        return False, False, False, False
    is_long_entry = flex_select_auto_nb(long_entries, _i, c.col, c.flex_2d)
    is_long_exit = flex_select_auto_nb(long_exits, _i, c.col, c.flex_2d)
    is_short_entry = flex_select_auto_nb(short_entries, _i, c.col, c.flex_2d)
    is_short_exit = flex_select_auto_nb(short_exits, _i, c.col, c.flex_2d)
    return is_long_entry, is_long_exit, is_short_entry, is_short_exit
