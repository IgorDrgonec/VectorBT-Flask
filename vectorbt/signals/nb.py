# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> import numpy as np
>>> import vectorbt as vbt

>>> # vectorbt.signals.nb.pos_rank_nb
>>> vbt.signals.nb.pos_rank_nb(np.array([False, True, True, True, False])[:, None])[:, 0]
[-1  0  1  2 -1]
```

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbt import _typing as tp
from vectorbt.base import chunking as base_ch
from vectorbt.base.indexing import flex_select_auto_nb
from vectorbt.ch_registry import register_chunkable
from vectorbt.generic import nb as generic_nb
from vectorbt.generic.enums import range_dt, RangeStatus
from vectorbt.jit_registry import register_jitted
from vectorbt.records import chunking as records_ch
from vectorbt.signals.enums import StopType
from vectorbt.utils import chunking as ch
from vectorbt.utils.array_ import uniform_summing_to_one_nb, rescale_float_to_int_nb, renormalize_nb
from vectorbt.utils.template import Rep


# ############# Generation ############# #


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        place_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def generate_nb(target_shape: tp.Shape, place_func_nb: tp.PlaceFunc, *args) -> tp.Array2d:
    """Create a boolean matrix of `target_shape` and pick signals using `place_func_nb`.

    Args:
        target_shape (array): Target shape.
        place_func_nb (callable): Signal placement function.

            `place_func_nb` must accept the boolean array for writing in place,
            index of the start of the range `from_i`, index of the end of the range `to_i`,
            index of the column `col`, and `*args`. Must return nothing.

            !!! note
                The first argument is always a 1-dimensional boolean array that contains only those
                elements where signals can be placed. The range and column indices only describe which
                range this array maps to.
        *args: Arguments passed to `place_func_nb`.
    """
    out = np.full(target_shape, False, dtype=np.bool_)

    for col in prange(target_shape[1]):
        place_func_nb(out[:, col], 0, target_shape[0], col, *args)
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        entry_wait=None,
        exit_wait=None,
        max_one_entry=None,
        max_one_exit=None,
        entry_place_func_nb=None,
        entry_args=ch.ArgsTaker(),
        exit_place_func_nb=None,
        exit_args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def generate_enex_nb(target_shape: tp.Shape,
                     entry_wait: int,
                     exit_wait: int,
                     max_one_entry: bool,
                     max_one_exit: bool,
                     entry_place_func_nb: tp.PlaceFunc,
                     entry_args: tp.Args,
                     exit_place_func_nb: tp.PlaceFunc,
                     exit_args: tp.Args) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick entry signals using `entry_place_func_nb` and exit signals using
    `exit_place_func_nb` one after another.

    Args:
        target_shape (array): Target shape.
        entry_wait (int): Number of ticks to wait before placing entries.

            !!! note
                Setting `entry_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and exit can be processed before entry.
        exit_wait (int): Number of ticks to wait before placing exits.

            !!! note
                Setting `exit_wait` to 0 or False assumes that both entry and exit can be processed
                within the same bar, and entry can be processed before exit.
        max_one_entry (bool): Whether `entry_place_func_nb` returns only once signal at most.

            Makes the execution a lot faster.
        max_one_exit (bool): Whether `exit_place_func_nb` returns only once signal at most.

            Makes the execution a lot faster.
        entry_place_func_nb (callable): Entry place function.

            See `place_func_nb` in `generate_nb`.
        entry_args (tuple): Arguments unpacked and passed to `entry_place_func_nb`.
        exit_place_func_nb (callable): Exit place function.

            See `place_func_nb` in `generate_nb`.
        exit_args (tuple): Arguments unpacked and passed to `exit_place_func_nb`.
    """
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")

    def _place_signals(out, from_i, col, only_one, place_func_nb, args):
        to_i = target_shape[0]
        if to_i > from_i:
            place_func_nb(out[from_i:to_i, col], from_i, to_i, col, *args)
            last_i = -1
            for j in range(from_i, to_i):
                if out[j, col]:
                    if only_one:
                        return j
                    last_i = j
            return last_i
        return -1

    for col in prange(target_shape[1]):
        from_i = 0
        entries_turn = True
        first_signal = True
        while from_i != -1:
            if entries_turn:
                if not first_signal:
                    from_i += entry_wait
                from_i = _place_signals(
                    entries,
                    from_i,
                    col,
                    max_one_entry,
                    entry_place_func_nb,
                    entry_args
                )
                entries_turn = False
            else:
                from_i += exit_wait
                from_i = _place_signals(
                    exits,
                    from_i,
                    col,
                    max_one_exit,
                    exit_place_func_nb,
                    exit_args
                )
                entries_turn = True
            first_signal = False

    return entries, exits


@register_chunkable(
    size=ch.ArraySizer(arg_query='entries', axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        wait=None,
        until_next=None,
        skip_until_exit=None,
        exit_place_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def generate_ex_nb(entries: tp.Array2d,
                   wait: int,
                   until_next: bool,
                   skip_until_exit: bool,
                   exit_place_func_nb: tp.PlaceFunc, *args) -> tp.Array2d:
    """Pick exit signals using `exit_place_func_nb` after each signal in `entries`.

    Args:
        entries (array): Boolean array with entry signals.
        wait (int): Number of ticks to wait before placing exits.

            !!! note
                Setting `wait` to 0 or False may result in two signals at one bar.
        until_next (int): Whether to place signals up to the next entry signal.

            !!! note
                Setting it to False makes it difficult to tell which exit belongs to which entry.
        skip_until_exit (bool): Whether to skip processing entry signals until the next exit.

            Has only effect when `until_next` is disabled.

            !!! note
                Setting it to True makes it impossible to tell which exit belongs to which entry.
        exit_place_func_nb (callable): Exit place function.

            See `place_func_nb` in `generate_nb`.
        *args (callable): Arguments passed to `exit_place_func_nb`.
    """
    out = np.full_like(entries, False)

    def _place_exits(from_i, to_i, col, last_exit_i):
        if from_i > -1:
            if skip_until_exit and from_i <= last_exit_i:
                return last_exit_i
            from_i += wait
            if not until_next:
                to_i = entries.shape[0]
            if to_i > from_i:
                exit_place_func_nb(out[from_i:to_i, col], from_i, to_i, col, *args)
                if skip_until_exit:
                    for j in range(from_i, to_i):
                        if out[j, col]:
                            last_exit_i = j
        return last_exit_i

    for col in prange(entries.shape[1]):
        from_i = -1
        last_exit_i = -1
        for i in range(entries.shape[0]):
            if entries[i, col]:
                last_exit_i = _place_exits(from_i, i, col, last_exit_i)
                from_i = i
        last_exit_i = _place_exits(from_i, entries.shape[0], col, last_exit_i)
    return out


# ############# Filtering ############# #


@register_jitted(cache=True)
def clean_enex_1d_nb(entries: tp.Array1d,
                     exits: tp.Array1d,
                     entry_first: bool) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Clean entry and exit arrays by picking the first signal out of each.

    Entry signal must be picked first. If both signals are present, selects none."""
    entries_out = np.full(entries.shape, False, dtype=np.bool_)
    exits_out = np.full(exits.shape, False, dtype=np.bool_)

    phase = -1
    for i in range(entries.shape[0]):
        if entries[i] and exits[i]:
            continue
        if entries[i]:
            if phase == -1 or phase == 0:
                phase = 1
                entries_out[i] = True
        if exits[i]:
            if (not entry_first and phase == -1) or phase == 1:
                phase = 0
                exits_out[i] = True

    return entries_out, exits_out


@register_chunkable(
    size=ch.ArraySizer(arg_query='entries', axis=1),
    arg_take_spec=dict(
        entries=ch.ArraySlicer(axis=1),
        exits=ch.ArraySlicer(axis=1),
        entry_first=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def clean_enex_nb(entries: tp.Array2d,
                  exits: tp.Array2d,
                  entry_first: bool) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """2-dim version of `clean_enex_1d_nb`."""
    entries_out = np.empty(entries.shape, dtype=np.bool_)
    exits_out = np.empty(exits.shape, dtype=np.bool_)

    for col in prange(entries.shape[1]):
        entries_out[:, col], exits_out[:, col] = clean_enex_1d_nb(entries[:, col], exits[:, col], entry_first)
    return entries_out, exits_out


# ############# Random signals ############# #


@register_jitted(cache=True)
def rand_place_nb(out: tp.Array1d, from_i: int, to_i: int, col: int, n: tp.FlexArray) -> None:
    """`place_func_nb` to randomly pick `n` values.

    `n` uses flexible indexing."""
    size = min(to_i - from_i, flex_select_auto_nb(n, 0, col, True))
    k = 0
    while k < size:
        i = np.random.choice(len(out))
        if not out[i]:
            out[i] = True
            k += 1


@register_jitted(cache=True)
def rand_by_prob_place_nb(out: tp.Array1d,
                          from_i: int,
                          to_i: int,
                          col: int,
                          prob: tp.FlexArray,
                          pick_first: bool,
                          flex_2d: bool) -> None:
    """`place_func_nb` to randomly place signals with probability `prob`.

    `prob` uses flexible indexing."""
    for i in range(from_i, to_i):
        if np.random.uniform(0, 1) < flex_select_auto_nb(prob, i, col, flex_2d):
            out[i - from_i] = True
            if pick_first:
                break


@register_chunkable(
    size=ch.ShapeSizer(arg_query='target_shape', axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        n=base_ch.FlexArraySlicer(axis=1, flex_2d=True),
        entry_wait=None,
        exit_wait=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def generate_rand_enex_nb(target_shape: tp.Shape,
                          n: tp.FlexArray,
                          entry_wait: int,
                          exit_wait: int) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Pick a number of entries and the same number of exits one after another.

    Respects `entry_wait` and `exit_wait` constraints through a number of tricks.
    Tries to mimic a uniform distribution as much as possible.

    The idea is the following: with constraints, there is some fixed amount of total
    space required between first entry and last exit. Upscale this space in a way that
    distribution of entries and exit is similar to a uniform distribution. This means
    randomizing the position of first entry, last exit, and all signals between them.

    `n` uses flexible indexing and thus must be at least a 0-dim array."""
    entries = np.full(target_shape, False)
    exits = np.full(target_shape, False)
    if entry_wait == 0 and exit_wait == 0:
        raise ValueError("entry_wait and exit_wait cannot be both 0")

    if entry_wait == 1 and exit_wait == 1:
        # Basic case
        both = generate_nb(target_shape, rand_place_nb, n * 2)
        for col in prange(both.shape[1]):
            both_idxs = np.flatnonzero(both[:, col])
            entries[both_idxs[0::2], col] = True
            exits[both_idxs[1::2], col] = True
    else:
        for col in prange(target_shape[1]):
            _n = flex_select_auto_nb(n, 0, col, True)
            if _n == 1:
                entry_idx = np.random.randint(0, target_shape[0] - exit_wait)
                entries[entry_idx, col] = True
            else:
                # Minimum range between two entries
                min_range = entry_wait + exit_wait

                # Minimum total range between first and last entry
                min_total_range = min_range * (_n - 1)
                if target_shape[0] < min_total_range + exit_wait + 1:
                    raise ValueError("Cannot take a larger sample than population")

                # We should decide how much space should be allocate before first and after last entry
                # Maximum space outside of min_total_range
                max_free_space = target_shape[0] - min_total_range - 1

                # If min_total_range is tiny compared to max_free_space, limit it
                # otherwise we would have huge space before first and after last entry
                # Limit it such as distribution of entries mimics uniform
                free_space = min(max_free_space, 3 * target_shape[0] // (_n + 1))

                # What about last exit? it requires exit_wait space
                free_space -= exit_wait

                # Now we need to distribute free space among three ranges:
                # 1) before first, 2) between first and last added to min_total_range, 3) after last
                # We do 2) such that min_total_range can freely expand to maximum
                # We allocate twice as much for 3) as for 1) because an exit is missing
                rand_floats = uniform_summing_to_one_nb(6)
                chosen_spaces = rescale_float_to_int_nb(rand_floats, (0, free_space), free_space)
                first_idx = chosen_spaces[0]
                last_idx = target_shape[0] - np.sum(chosen_spaces[-2:]) - exit_wait - 1

                # Selected range between first and last entry
                total_range = last_idx - first_idx

                # Maximum range between two entries within total_range
                max_range = total_range - (_n - 2) * min_range

                # Select random ranges within total_range
                rand_floats = uniform_summing_to_one_nb(_n - 1)
                chosen_ranges = rescale_float_to_int_nb(rand_floats, (min_range, max_range), total_range)

                # Translate them into entries
                entry_idxs = np.empty(_n, dtype=np.int_)
                entry_idxs[0] = first_idx
                entry_idxs[1:] = chosen_ranges
                entry_idxs = np.cumsum(entry_idxs)
                entries[entry_idxs, col] = True

        # Generate exits
        for col in range(target_shape[1]):
            entry_idxs = np.flatnonzero(entries[:, col])
            for j in range(len(entry_idxs)):
                entry_i = entry_idxs[j] + exit_wait
                if j < len(entry_idxs) - 1:
                    exit_i = entry_idxs[j + 1] - entry_wait
                else:
                    exit_i = entries.shape[0] - 1
                i = np.random.randint(exit_i - entry_i + 1)
                exits[entry_i + i, col] = True
    return entries, exits


def rand_enex_apply_nb(target_shape: tp.Shape,
                       n: tp.FlexArray,
                       entry_wait: int,
                       exit_wait: int) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """`apply_func_nb` that calls `generate_rand_enex_nb`."""
    return generate_rand_enex_nb(target_shape, n, entry_wait, exit_wait)


# ############# Stop signals ############# #


@register_jitted(cache=True)
def first_place_nb(out: tp.Array1d, from_i: int, to_i: int, col: int, mask: tp.Array2d) -> None:
    """`place_func_nb` that returns the index of the first signal in `mask`."""
    for i in range(from_i, to_i):
        if mask[i, col]:
            out[i - from_i] = True
            break


@register_jitted(cache=True)
def stop_place_nb(out: tp.Array1d,
                  from_i: int,
                  to_i: int,
                  col: int,
                  ts: tp.FlexArray,
                  stop: tp.FlexArray,
                  trailing: tp.FlexArray,
                  wait: int,
                  pick_first: bool,
                  flex_2d: bool) -> None:
    """`place_func_nb` that returns the indices of the stop being hit.

    Args:
        out (array): Boolean array to write.
        from_i (int): Index to start generation from (inclusive).
        to_i (int): Index to run generation to (exclusive).
        col (int): Current column.
        ts (array of float): 2-dim time series array such as price.
        stop (array of float): Stop value for stop loss.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
            Set an element to `np.nan` to disable.
        trailing (array of bool): Whether to use trailing stop.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
            Set an element to False to disable.
        wait (int): Number of ticks to wait before placing exits.

            Setting False or 0 may result in two signals at one bar.

            !!! note
                If `wait` is greater than 0, trailing stop won't update at bars that come before `from_i`.
        pick_first (bool): Whether to stop as soon as the first exit signal is found.
        flex_2d (bool): See `vectorbt.base.indexing.flex_select_auto_nb`."""
    init_i = from_i - wait
    init_ts = flex_select_auto_nb(ts, init_i, col, flex_2d)
    init_stop = flex_select_auto_nb(stop, init_i, col, flex_2d)
    init_trailing = flex_select_auto_nb(trailing, init_i, col, flex_2d)
    max_high = min_low = init_ts

    for i in range(from_i, to_i):
        if not np.isnan(init_stop):
            if init_trailing:
                if init_stop >= 0:
                    # Trailing stop buy
                    curr_stop_price = min_low * (1 + abs(init_stop))
                else:
                    # Trailing stop sell
                    curr_stop_price = max_high * (1 - abs(init_stop))
            else:
                curr_stop_price = init_ts * (1 + init_stop)

        # Check if stop price is within bar
        curr_ts = flex_select_auto_nb(ts, i, col, flex_2d)
        if not np.isnan(init_stop):
            if init_stop >= 0:
                exit_signal = curr_ts >= curr_stop_price
            else:
                exit_signal = curr_ts <= curr_stop_price
            if exit_signal:
                out[i - from_i] = True
                if pick_first:
                    break

        # Keep track of lowest low and highest high if trailing
        if init_trailing:
            if curr_ts < min_low:
                min_low = curr_ts
            elif curr_ts > max_high:
                max_high = curr_ts


@register_jitted(cache=True)
def ohlc_stop_place_nb(out: tp.Array1d,
                       from_i: int,
                       to_i: int,
                       col: int,
                       open: tp.FlexArray,
                       high: tp.FlexArray,
                       low: tp.FlexArray,
                       close: tp.FlexArray,
                       stop_price_out: tp.Array2d,
                       stop_type_out: tp.Array2d,
                       sl_stop: tp.FlexArray,
                       sl_trail: tp.FlexArray,
                       tp_stop: tp.FlexArray,
                       reverse: tp.FlexArray,
                       is_open_safe: bool,
                       wait: int,
                       pick_first: bool,
                       flex_2d: bool) -> None:
    """`place_func_nb` that returns the indices of the stop price being hit within OHLC.

    Compared to `stop_place_nb`, takes into account the whole bar, can check for both
    (trailing) stop loss and take profit simultaneously, and tracks hit price and stop type.

    !!! note
        We don't have intra-candle data. If there was a huge price fluctuation in both directions,
        we can't determine whether SL was triggered before TP and vice versa. So some assumptions
        need to be made:

        1) if stop has been hit before open, stop price becomes current open,
        2) trailing stop can only be based on previous close/high, and
        3) we pessimistically assume that SL comes before TP.
    
    Args:
        out (array): Boolean array to write.
        col (int): Current column.
        from_i (int): Index to start generation from (inclusive).
        to_i (int): Index to run generation to (exclusive).
        open (array of float): Entry price such as open or previous close.
        high (array of float): High price.
        low (array of float): Low price.
        close (array of float): Close price.
        stop_price_out (array of float): Array where hit price of each exit will be stored.
        stop_type_out (array of int): Array where stop type of each exit will be stored.

            0 for stop loss, 1 for take profit.
        sl_stop (array of float): Percentage value for stop loss.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
            Set an element to `np.nan` to disable.
        sl_trail (array of bool): Whether `sl_stop` is trailing.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
            Set an element to False to disable.
        tp_stop (array of float): Percentage value for take profit.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
            Set an element to `np.nan` to disable.
        reverse (array of float): Whether to do the opposite, i.e.: prices are followed downwards.

            Can be per frame, column, row, or element-wise. Must be at least a 0-dim array.
        is_open_safe (bool): Whether entry price comes right at or before open.

            If True and wait is 0, can use high/low at entry bar. Otherwise uses only close.
        wait (int): Number of ticks to wait before placing exits.

            Setting False or 0 may result in entry and exit signal at one bar.

            !!! note
                If `wait` is greater than 0, even with `is_open_safe` set to True,
                trailing stop won't update at bars that come before `from_i`.
        pick_first (bool): Whether to stop as soon as the first exit signal is found.
        flex_2d (bool): See `vectorbt.base.indexing.flex_select_auto_nb`.
    """
    init_i = from_i - wait
    init_open = flex_select_auto_nb(open, init_i, col, flex_2d)
    init_sl_stop = flex_select_auto_nb(sl_stop, init_i, col, flex_2d)
    if init_sl_stop < 0:
        raise ValueError("Stop value must be nan, 0, or greater than 0")
    init_sl_trail = flex_select_auto_nb(sl_trail, init_i, col, flex_2d)
    init_tp_stop = flex_select_auto_nb(tp_stop, init_i, col, flex_2d)
    if init_tp_stop < 0:
        raise ValueError("Stop value must be nan, 0, or greater than 0")
    init_reverse = flex_select_auto_nb(reverse, init_i, col, flex_2d)
    max_p = min_p = init_open

    for i in range(from_i, to_i):
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

        # Calculate stop price
        if not np.isnan(init_sl_stop):
            if init_sl_trail:
                if init_reverse:
                    curr_sl_stop_price = min_p * (1 + init_sl_stop)
                else:
                    curr_sl_stop_price = max_p * (1 - init_sl_stop)
            else:
                if init_reverse:
                    curr_sl_stop_price = init_open * (1 + init_sl_stop)
                else:
                    curr_sl_stop_price = init_open * (1 - init_sl_stop)
        if not np.isnan(init_tp_stop):
            if init_reverse:
                curr_tp_stop_price = init_open * (1 - init_tp_stop)
            else:
                curr_tp_stop_price = init_open * (1 + init_tp_stop)

        # Check if stop price is within bar
        if i > init_i or is_open_safe:
            # is_open_safe means open is either open or any other price before it
            # so it's safe to use high/low at entry bar
            curr_high = _high
            curr_low = _low
        else:
            # Otherwise, we can only use close price at entry bar
            curr_high = curr_low = _close

        exit_signal = False
        if not np.isnan(init_sl_stop):
            # SL/TSL hit?
            stop_price = np.nan
            if not init_reverse:
                if _open <= curr_sl_stop_price:
                    stop_price = _open
                if curr_low <= curr_sl_stop_price:
                    stop_price = curr_sl_stop_price
            else:
                if _open >= curr_sl_stop_price:
                    stop_price = _open
                if curr_high >= curr_sl_stop_price:
                    stop_price = curr_sl_stop_price
            if not np.isnan(stop_price):
                stop_price_out[i, col] = stop_price
                if init_sl_trail:
                    stop_type_out[i, col] = StopType.TrailStop
                else:
                    stop_type_out[i, col] = StopType.StopLoss
                exit_signal = True

        if not exit_signal and not np.isnan(init_tp_stop):
            # TP hit?
            stop_price = np.nan
            if not init_reverse:
                if _open >= curr_tp_stop_price:
                    stop_price = _open
                if curr_high >= curr_tp_stop_price:
                    stop_price = curr_tp_stop_price
            else:
                if _open <= curr_tp_stop_price:
                    stop_price = _open
                if curr_low <= curr_tp_stop_price:
                    stop_price = curr_tp_stop_price
            if not np.isnan(stop_price):
                stop_price_out[i, col] = stop_price
                stop_type_out[i, col] = StopType.TakeProfit
                exit_signal = True

        if exit_signal:
            out[i - from_i] = True
            if pick_first:
                break

        # Keep track of highest high if trailing
        if init_sl_trail:
            if curr_low < min_p:
                min_p = curr_low
            if curr_high > max_p:
                max_p = curr_high


# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def between_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record of type `vectorbt.generic.enums.range_dt` for each range between two signals in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if from_i > -1:
                    to_i = i
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['start_idx'][r, col] = from_i
                    new_records['end_idx'][r, col] = to_i
                    new_records['status'][r, col] = RangeStatus.Closed
                    counts[col] += 1
                from_i = i

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1),
        other_mask=ch.ArraySlicer(axis=1),
        from_other=None
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def between_two_ranges_nb(mask: tp.Array2d, other_mask: tp.Array2d, from_other: bool = False) -> tp.RecordArray:
    """Create a record of type `vectorbt.generic.enums.range_dt` for each range between two
    signals in `mask` and `other_mask`.

    If `from_other` is False, returns ranges from each in `mask` to the succeeding in `other_mask`.
    Otherwise, returns ranges from each in `other_mask` to the preceding in `mask`.

    When `mask` and `other_mask` overlap (two signals at the same time), the distance between overlapping
    signals is still considered and `from_i` would match `to_i`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        if from_other:
            to_i = -1
            for i in range(mask.shape[0] - 1, -1, -1):
                if other_mask[i, col]:
                    to_i = i
                if mask[i, col]:
                    from_i = i
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['start_idx'][r, col] = from_i
                    new_records['end_idx'][r, col] = to_i
                    new_records['status'][r, col] = RangeStatus.Closed
                    counts[col] += 1
        else:
            from_i = -1
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    from_i = i
                if other_mask[i, col]:
                    to_i = i
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['start_idx'][r, col] = from_i
                    new_records['end_idx'][r, col] = to_i
                    new_records['status'][r, col] = RangeStatus.Closed
                    counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record of type `vectorbt.generic.enums.range_dt` for each partition of signals in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition:
                    from_i = i
                is_partition = True
            elif is_partition:
                to_i = i
                r = counts[col]
                new_records['id'][r, col] = r
                new_records['col'][r, col] = col
                new_records['start_idx'][r, col] = from_i
                new_records['end_idx'][r, col] = to_i
                new_records['status'][r, col] = RangeStatus.Closed
                counts[col] += 1
                is_partition = False
            if i == mask.shape[0] - 1:
                if is_partition:
                    to_i = mask.shape[0] - 1
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['start_idx'][r, col] = from_i
                    new_records['end_idx'][r, col] = to_i
                    new_records['status'][r, col] = RangeStatus.Open
                    counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def between_partition_ranges_nb(mask: tp.Array2d) -> tp.RecordArray:
    """Create a record of type `vectorbt.generic.enums.range_dt` for each range between two partitions in `mask`."""
    new_records = np.empty(mask.shape, dtype=range_dt)
    counts = np.full(mask.shape[1], 0, dtype=np.int_)

    for col in prange(mask.shape[1]):
        is_partition = False
        from_i = -1
        for i in range(mask.shape[0]):
            if mask[i, col]:
                if not is_partition and from_i != -1:
                    to_i = i
                    r = counts[col]
                    new_records['id'][r, col] = r
                    new_records['col'][r, col] = col
                    new_records['start_idx'][r, col] = from_i
                    new_records['end_idx'][r, col] = to_i
                    new_records['status'][r, col] = RangeStatus.Closed
                    counts[col] += 1
                is_partition = True
                from_i = i
            else:
                is_partition = False

    return generic_nb.repartition_nb(new_records, counts)


# ############# Ranking ############# #

@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1),
        reset_by_mask=None,
        after_false=None,
        rank_func_nb=None,
        args=ch.ArgsTaker()
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(tags={'can_parallel'})
def rank_nb(mask: tp.Array2d,
            reset_by_mask: tp.Optional[tp.Array2d],
            after_false: bool,
            rank_func_nb: tp.RankFunc, *args) -> tp.Array2d:
    """Rank each signal using `rank_func_nb`.

    Applies `rank_func_nb` on each True value. Must accept the row index, the column index,
    index of the last reset signal, index of the end of the previous partition,
    index of the start of the current partition, and `*args`.
    Must return -1 for no rank, otherwise 0 or greater.

    Setting `after_false` to True will disregard the first partition of True values
    if there is no False value before them."""
    out = np.full(mask.shape, -1, dtype=np.int_)

    for col in prange(mask.shape[1]):
        reset_i = 0
        prev_part_end_i = -1
        part_start_i = -1
        in_partition = False
        false_seen = not after_false
        for i in range(mask.shape[0]):
            if reset_by_mask is not None:
                if reset_by_mask[i, col]:
                    reset_i = i
            if mask[i, col] and not (after_false and not false_seen):
                if not in_partition:
                    part_start_i = i
                in_partition = True
                out[i, col] = rank_func_nb(i, col, reset_i, prev_part_end_i, part_start_i, *args)
            elif not mask[i, col]:
                if in_partition:
                    prev_part_end_i = i - 1
                in_partition = False
                false_seen = True
    return out


@register_jitted(cache=True)
def sig_pos_rank_nb(i: int, col: int, reset_i: int, prev_part_end_i: int, part_start_i: int,
                    sig_pos_temp: tp.Array1d, allow_gaps: bool) -> int:
    """`rank_func_nb` that returns the rank of each signal by its position in the partition."""
    if reset_i > prev_part_end_i and max(reset_i, part_start_i) == i:
        sig_pos_temp[col] = -1
    elif not allow_gaps and part_start_i == i:
        sig_pos_temp[col] = -1
    sig_pos_temp[col] += 1
    return sig_pos_temp[col]


@register_jitted(cache=True)
def part_pos_rank_nb(i: int, col: int, reset_i: int, prev_part_end_i: int, part_start_i: int,
                     part_pos_temp: tp.Array1d) -> int:
    """`rank_func_nb` that returns the rank of each partition by its position in the series."""
    if reset_i > prev_part_end_i and max(reset_i, part_start_i) == i:
        part_pos_temp[col] = 0
    elif part_start_i == i:
        part_pos_temp[col] += 1
    return part_pos_temp[col]


# ############# Index ############# #


@register_jitted(cache=True)
def nth_index_1d_nb(mask: tp.Array1d, n: int) -> int:
    """Get the index of the n-th True value.

    !!! note
        `n` starts with 0 and can be negative."""
    if n >= 0:
        found = -1
        for i in range(mask.shape[0]):
            if mask[i]:
                found += 1
                if found == n:
                    return i
    else:
        found = 0
        for i in range(mask.shape[0] - 1, -1, -1):
            if mask[i]:
                found -= 1
                if found == n:
                    return i
    return -1


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1),
        n=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def nth_index_nb(mask: tp.Array2d, n: int) -> tp.Array1d:
    """2-dim version of `nth_index_1d_nb`."""
    out = np.empty(mask.shape[1], dtype=np.int_)
    for col in prange(mask.shape[1]):
        out[col] = nth_index_1d_nb(mask[:, col], n)
    return out


@register_jitted(cache=True)
def norm_avg_index_1d_nb(mask: tp.Array1d) -> float:
    """Get mean index normalized to (-1, 1)."""
    mean_index = np.mean(np.flatnonzero(mask))
    return renormalize_nb(mean_index, (0, len(mask) - 1), (-1, 1))


@register_chunkable(
    size=ch.ArraySizer(arg_query='mask', axis=1),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def norm_avg_index_nb(mask: tp.Array2d) -> tp.Array1d:
    """2-dim version of `norm_avg_index_1d_nb`."""
    out = np.empty(mask.shape[1], dtype=np.float_)
    for col in prange(mask.shape[1]):
        out[col] = norm_avg_index_1d_nb(mask[:, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='group_lens', axis=0),
    arg_take_spec=dict(
        mask=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def norm_avg_index_grouped_nb(mask, group_lens):
    """Grouped version of `norm_avg_index_nb`."""
    out = np.empty(len(group_lens), dtype=np.float_)
    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        temp_sum = 0
        temp_cnt = 0
        for col in range(from_col, to_col):
            for i in range(mask.shape[0]):
                if mask[i, col]:
                    temp_sum += i
                    temp_cnt += 1
        out[group] = renormalize_nb(temp_sum / temp_cnt, (0, mask.shape[0] - 1), (-1, 1))
    return out
