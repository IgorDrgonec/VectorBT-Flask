# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio records."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import (
    is_close_nb,
    is_close_or_less_nb,
    is_less_nb,
    add_nb
)
from vectorbtpro.utils.template import Rep

invalid_size_msg = "Encountered an order with size 0 or less"
invalid_price_msg = "Encountered an order with price less than 0"


@register_jitted(cache=True)
def fill_trade_record_nb(new_records: tp.Record,
                         r: int,
                         col: int,
                         size: float,
                         entry_idx: int,
                         entry_price: float,
                         entry_fees: float,
                         exit_idx: int,
                         exit_price: float,
                         exit_fees: float,
                         direction: int,
                         status: int,
                         parent_id: int) -> None:
    """Fill a trade record."""
    # Calculate PnL and return
    pnl, ret = get_trade_stats_nb(
        size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction
    )

    # Save trade
    new_records['id'][r] = r
    new_records['col'][r] = col
    new_records['size'][r] = size
    new_records['entry_idx'][r] = entry_idx
    new_records['entry_price'][r] = entry_price
    new_records['entry_fees'][r] = entry_fees
    new_records['exit_idx'][r] = exit_idx
    new_records['exit_price'][r] = exit_price
    new_records['exit_fees'][r] = exit_fees
    new_records['pnl'][r] = pnl
    new_records['return'][r] = ret
    new_records['direction'][r] = direction
    new_records['status'][r] = status
    new_records['parent_id'][r] = parent_id


@register_jitted(cache=True)
def fill_entry_trades_in_position_nb(order_records: tp.RecordArray,
                                     col_map: tp.ColMap,
                                     col: int,
                                     first_c: int,
                                     last_c: int,
                                     init_price: float,
                                     first_entry_size: float,
                                     first_entry_fees: float,
                                     exit_idx: int,
                                     exit_size_sum: float,
                                     exit_gross_sum: float,
                                     exit_fees_sum: float,
                                     direction: int,
                                     status: int,
                                     parent_id: int,
                                     new_records: tp.RecordArray,
                                     r: int) -> int:
    """Fill entry trades located within a single position.

    Returns the next trade id."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens

    # Iterate over orders located within a single position
    for c in range(first_c, last_c + 1):
        if c == -1:
            entry_idx = -1
            entry_price = init_price
            entry_size = first_entry_size
            entry_fees = first_entry_fees
        else:
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            order_side = order_record['side']
            entry_idx = order_record['idx']
            entry_price = order_record['price']

            # Ignore exit orders
            if (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                continue

            if c == first_c:
                entry_size = first_entry_size
                entry_fees = first_entry_fees
            else:
                entry_size = order_record['size']
                entry_fees = order_record['fees']

        # Take a size-weighted average of exit price
        exit_price = exit_gross_sum / exit_size_sum

        # Take a fraction of exit fees
        size_fraction = entry_size / exit_size_sum
        exit_fees = size_fraction * exit_fees_sum

        # Fill the record
        fill_trade_record_nb(
            new_records,
            r,
            col,
            entry_size,
            entry_idx,
            entry_price,
            entry_fees,
            exit_idx,
            exit_price,
            exit_fees,
            direction,
            status,
            parent_id
        )
        r += 1

    return r


@register_chunkable(
    size=records_ch.ColLensSizer(arg_query='col_map'),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=ch.ArraySlicer(axis=1),
        col_map=records_ch.ColMapSlicer(),
        init_position=base_ch.FlexArraySlicer(axis=1, flex_2d=True)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def get_entry_trades_nb(order_records: tp.RecordArray,
                        close: tp.Array2d,
                        col_map: tp.ColMap,
                        init_position: tp.FlexArray = np.asarray(0.)) -> tp.RecordArray:
    """Fill entry trade records by aggregating order records.

    Entry trade records are buy orders in a long position and sell orders in a short position.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.records.nb import col_map_nb
        >>> from vectorbtpro.portfolio.nb import simulate_from_orders_nb, get_entry_trades_nb

        >>> close = order_price = np.array([
        ...     [1, 6],
        ...     [2, 5],
        ...     [3, 4],
        ...     [4, 3],
        ...     [5, 2],
        ...     [6, 1]
        ... ])
        >>> size = np.asarray([
        ...     [1, -1],
        ...     [0.1, -0.1],
        ...     [-1, 1],
        ...     [-0.1, 0.1],
        ...     [1, -1],
        ...     [-2, 2]
        ... ])
        >>> target_shape = close.shape
        >>> group_lens = np.full(target_shape[1], 1)
        >>> init_cash = np.full(target_shape[1], 100)
        >>> call_seq = np.full(target_shape, 0)

        >>> sim_out = simulate_from_orders_nb(
        ...     target_shape,
        ...     group_lens,
        ...     init_cash,
        ...     call_seq,
        ...     size=size,
        ...     price=close,
        ...     fees=np.asarray(0.01),
        ...     slippage=np.asarray(0.01)
        ... )

        >>> col_map = col_map_nb(sim_out.order_records['col'], target_shape[1])
        >>> entry_trade_records = get_entry_trades_nb(sim_out.order_records, close, col_map)
        >>> pd.DataFrame.from_records(entry_trade_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.0          0         1.01     0.01010         3    3.060000
        1   1    0   0.1          1         2.02     0.00202         3    3.060000
        2   2    0   1.0          4         5.05     0.05050         5    5.940000
        3   3    0   1.0          5         5.94     0.05940         5    6.000000
        4   0    1   1.0          0         5.94     0.05940         3    3.948182
        5   1    1   0.1          1         4.95     0.00495         3    3.948182
        6   2    1   1.0          4         1.98     0.01980         5    1.010000
        7   3    1   1.0          5         1.01     0.01010         5    1.000000

           exit_fees       pnl    return  direction  status  parent_id
        0   0.030600  2.009300  1.989406          0       1          0
        1   0.003060  0.098920  0.489703          0       1          0
        2   0.059400  0.780100  0.154475          0       1          1
        3   0.000000 -0.119400 -0.020101          1       0          2
        4   0.039482  1.892936  0.318676          1       1          0
        5   0.003948  0.091284  0.184411          1       1          0
        6   0.010100  0.940100  0.474798          1       1          1
        7   0.000000 -0.020100 -0.019901          0       0          2
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    max_records = np.max(col_lens) + int((init_position != 0).any())
    new_records = np.empty((max_records, len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=np.int_)

    for col in prange(col_lens.shape[0]):

        _init_position = flex_select_auto_nb(init_position, 0, col, True)
        if _init_position != 0:
            # Prepare initial position
            first_c = -1
            in_position = True
            parent_id = 0
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            init_price = np.nan
            for i in range(close.shape[0]):
                if not np.isnan(close[i, col]):
                    init_price = close[i, col]
                    break
            entry_gross_sum = abs(_init_position) * init_price
            entry_fees_sum = 0.
            exit_size_sum = 0.
            exit_gross_sum = 0.
            exit_fees_sum = 0.
            first_entry_size = _init_position
            first_entry_fees = 0.
        else:
            in_position = False
            parent_id = -1

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]

            if order_record['id'] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record['id']

            order_idx = order_record['idx']
            order_size = order_record['size']
            order_price = order_record['price']
            order_fees = order_record['fees']
            order_side = order_record['side']

            if order_size <= 0.:
                raise ValueError(invalid_size_msg)
            if order_price < 0.:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # New position opened
                first_c = c
                in_position = True
                parent_id += 1
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                entry_size_sum = 0.
                entry_gross_sum = 0.
                entry_fees_sum = 0.
                exit_size_sum = 0.
                exit_gross_sum = 0.
                exit_fees_sum = 0.
                first_entry_size = order_size
                first_entry_fees = order_fees

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Sell):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                if is_close_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position closed
                    last_c = c
                    in_position = False
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees

                    # Fill trade records
                    counts[col] = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        first_c,
                        last_c,
                        close[0, col],
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        new_records[:, col],
                        counts[col]
                    )
                elif is_less_nb(exit_size_sum + order_size, entry_size_sum):
                    # Position decreased
                    exit_size_sum += order_size
                    exit_gross_sum += order_size * order_price
                    exit_fees_sum += order_fees
                else:
                    # Position closed
                    last_c = c
                    remaining_size = add_nb(entry_size_sum, -exit_size_sum)
                    exit_size_sum = entry_size_sum
                    exit_gross_sum += remaining_size * order_price
                    exit_fees_sum += remaining_size / order_size * order_fees

                    # Fill trade records
                    counts[col] = fill_entry_trades_in_position_nb(
                        order_records,
                        col_map,
                        col,
                        first_c,
                        last_c,
                        close[0, col],
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TradeStatus.Closed,
                        parent_id,
                        new_records[:, col],
                        counts[col]
                    )

                    # New position opened
                    first_c = c
                    parent_id += 1
                    if order_side == OrderSide.Buy:
                        direction = TradeDirection.Long
                    else:
                        direction = TradeDirection.Short
                    entry_size_sum = add_nb(order_size, -remaining_size)
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = entry_size_sum / order_size * order_fees
                    first_entry_size = entry_size_sum
                    first_entry_fees = entry_fees_sum
                    exit_size_sum = 0.
                    exit_gross_sum = 0.
                    exit_fees_sum = 0.

        if in_position and is_less_nb(exit_size_sum, entry_size_sum):
            # Position hasn't been closed
            last_c = col_len - 1
            remaining_size = add_nb(entry_size_sum, -exit_size_sum)
            exit_size_sum = entry_size_sum
            exit_gross_sum += remaining_size * close[close.shape[0] - 1, col]

            # Fill trade records
            counts[col] = fill_entry_trades_in_position_nb(
                order_records,
                col_map,
                col,
                first_c,
                last_c,
                close[0, col],
                first_entry_size,
                first_entry_fees,
                close.shape[0] - 1,
                exit_size_sum,
                exit_gross_sum,
                exit_fees_sum,
                direction,
                TradeStatus.Open,
                parent_id,
                new_records[:, col],
                counts[col]
            )

    return generic_nb.repartition_nb(new_records, counts)


@register_chunkable(
    size=records_ch.ColLensSizer(arg_query='col_map'),
    arg_take_spec=dict(
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        close=ch.ArraySlicer(axis=1),
        col_map=records_ch.ColMapSlicer(),
        init_position=base_ch.FlexArraySlicer(axis=1, flex_2d=True)
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep('chunk_meta'))
)
@register_jitted(cache=True, tags={'can_parallel'})
def get_exit_trades_nb(order_records: tp.RecordArray,
                       close: tp.Array2d,
                       col_map: tp.ColMap,
                       init_position: tp.FlexArray = np.asarray(0.)) -> tp.RecordArray:
    """Fill exit trade records by aggregating order records.

    Exit trade records are sell orders in a long position and buy orders in a short position.

    Usage:
        * Building upon the example in `get_exit_trades_nb`:

        ```pycon
        >>> from vectorbtpro.portfolio.nb import get_exit_trades_nb

        >>> exit_trade_records = get_exit_trades_nb(sim_out.order_records, close, col_map)
        >>> pd.DataFrame.from_records(exit_trade_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.0          0     1.101818    0.011018         2        2.97
        1   1    0   0.1          0     1.101818    0.001102         3        3.96
        2   2    0   1.0          4     5.050000    0.050500         5        5.94
        3   3    0   1.0          5     5.940000    0.059400         5        6.00
        4   0    1   1.0          0     5.850000    0.058500         2        4.04
        5   1    1   0.1          0     5.850000    0.005850         3        3.03
        6   2    1   1.0          4     1.980000    0.019800         5        1.01
        7   3    1   1.0          5     1.010000    0.010100         5        1.00

           exit_fees       pnl    return  direction  status  parent_id
        0    0.02970  1.827464  1.658589          0       1          0
        1    0.00396  0.280756  2.548119          0       1          0
        2    0.05940  0.780100  0.154475          0       1          1
        3    0.00000 -0.119400 -0.020101          1       0          2
        4    0.04040  1.711100  0.292496          1       1          0
        5    0.00303  0.273120  0.466872          1       1          0
        6    0.01010  0.940100  0.474798          1       1          1
        7    0.00000 -0.020100 -0.019901          0       0          2
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    max_records = np.max(col_lens) + int((init_position != 0).any())
    new_records = np.empty((max_records, len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=np.int_)

    for col in prange(col_lens.shape[0]):

        _init_position = flex_select_auto_nb(init_position, 0, col, True)
        if _init_position != 0:
            # Prepare initial position
            in_position = True
            parent_id = 0
            entry_idx = -1
            if _init_position >= 0:
                direction = TradeDirection.Long
            else:
                direction = TradeDirection.Short
            entry_size_sum = abs(_init_position)
            init_price = np.nan
            for i in range(close.shape[0]):
                if not np.isnan(close[i, col]):
                    init_price = close[i, col]
                    break
            entry_gross_sum = abs(_init_position) * init_price
            entry_fees_sum = 0.
        else:
            in_position = False
            parent_id = -1

        col_len = col_lens[col]
        if col_len == 0 and not in_position:
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]

            if order_record['id'] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record['id']

            i = order_record['idx']
            order_size = order_record['size']
            order_price = order_record['price']
            order_fees = order_record['fees']
            order_side = order_record['side']

            if order_size <= 0.:
                raise ValueError(invalid_size_msg)
            if order_price < 0.:
                raise ValueError(invalid_price_msg)

            if not in_position:
                # Trade opened
                in_position = True
                entry_idx = i
                if order_side == OrderSide.Buy:
                    direction = TradeDirection.Long
                else:
                    direction = TradeDirection.Short
                parent_id += 1
                entry_size_sum = 0.
                entry_gross_sum = 0.
                entry_fees_sum = 0.

            if (direction == TradeDirection.Long and order_side == OrderSide.Buy) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Sell):
                # Position increased
                entry_size_sum += order_size
                entry_gross_sum += order_size * order_price
                entry_fees_sum += order_fees

            elif (direction == TradeDirection.Long and order_side == OrderSide.Sell) \
                    or (direction == TradeDirection.Short and order_side == OrderSide.Buy):
                if is_close_or_less_nb(order_size, entry_size_sum):
                    # Trade closed
                    if is_close_nb(order_size, entry_size_sum):
                        exit_size = entry_size_sum
                    else:
                        exit_size = order_size
                    exit_price = order_price
                    exit_fees = order_fees
                    exit_idx = i

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        new_records[:, col],
                        counts[col],
                        col,
                        exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        exit_idx,
                        exit_price,
                        exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id
                    )
                    counts[col] += 1

                    if is_close_nb(order_size, entry_size_sum):
                        # Position closed
                        entry_idx = -1
                        direction = -1
                        in_position = False
                    else:
                        # Position decreased, previous orders have now less impact
                        size_fraction = (entry_size_sum - order_size) / entry_size_sum
                        entry_size_sum *= size_fraction
                        entry_gross_sum *= size_fraction
                        entry_fees_sum *= size_fraction
                else:
                    # Trade reversed
                    # Close current trade
                    cl_exit_size = entry_size_sum
                    cl_exit_price = order_price
                    cl_exit_fees = cl_exit_size / order_size * order_fees
                    cl_exit_idx = i

                    # Take a size-weighted average of entry price
                    entry_price = entry_gross_sum / entry_size_sum

                    # Take a fraction of entry fees
                    size_fraction = cl_exit_size / entry_size_sum
                    entry_fees = size_fraction * entry_fees_sum

                    fill_trade_record_nb(
                        new_records[:, col],
                        counts[col],
                        col,
                        cl_exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        cl_exit_idx,
                        cl_exit_price,
                        cl_exit_fees,
                        direction,
                        TradeStatus.Closed,
                        parent_id
                    )
                    counts[col] += 1

                    # Open a new trade
                    entry_size_sum = order_size - cl_exit_size
                    entry_gross_sum = entry_size_sum * order_price
                    entry_fees_sum = order_fees - cl_exit_fees
                    entry_idx = i
                    if direction == TradeDirection.Long:
                        direction = TradeDirection.Short
                    else:
                        direction = TradeDirection.Long
                    parent_id += 1

        if in_position and is_less_nb(-entry_size_sum, 0):
            # Trade hasn't been closed
            exit_size = entry_size_sum
            exit_price = close[close.shape[0] - 1, col]
            exit_fees = 0.
            exit_idx = close.shape[0] - 1

            # Take a size-weighted average of entry price
            entry_price = entry_gross_sum / entry_size_sum

            # Take a fraction of entry fees
            size_fraction = exit_size / entry_size_sum
            entry_fees = size_fraction * entry_fees_sum

            fill_trade_record_nb(
                new_records[:, col],
                counts[col],
                col,
                exit_size,
                entry_idx,
                entry_price,
                entry_fees,
                exit_idx,
                exit_price,
                exit_fees,
                direction,
                TradeStatus.Open,
                parent_id
            )
            counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_jitted(cache=True)
def fill_position_record_nb(new_records: tp.RecordArray, r: int, trade_records: tp.RecordArray) -> None:
    """Fill a position record by aggregating trade records."""
    # Aggregate trades
    col = trade_records['col'][0]
    size = np.sum(trade_records['size'])
    entry_idx = trade_records['entry_idx'][0]
    entry_price = np.sum(trade_records['size'] * trade_records['entry_price']) / size
    entry_fees = np.sum(trade_records['entry_fees'])
    exit_idx = trade_records['exit_idx'][-1]
    exit_price = np.sum(trade_records['size'] * trade_records['exit_price']) / size
    exit_fees = np.sum(trade_records['exit_fees'])
    direction = trade_records['direction'][-1]
    status = trade_records['status'][-1]
    pnl, ret = get_trade_stats_nb(
        size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction
    )

    # Save position
    new_records['id'][r] = r
    new_records['col'][r] = col
    new_records['size'][r] = size
    new_records['entry_idx'][r] = entry_idx
    new_records['entry_price'][r] = entry_price
    new_records['entry_fees'][r] = entry_fees
    new_records['exit_idx'][r] = exit_idx
    new_records['exit_price'][r] = exit_price
    new_records['exit_fees'][r] = exit_fees
    new_records['pnl'][r] = pnl
    new_records['return'][r] = ret
    new_records['direction'][r] = direction
    new_records['status'][r] = status
    new_records['parent_id'][r] = r


@register_jitted(cache=True)
def copy_trade_record_nb(new_records: tp.RecordArray, r: int, trade_record: tp.Record) -> None:
    """Copy a trade record."""
    new_records['id'][r] = r
    new_records['col'][r] = trade_record['col']
    new_records['size'][r] = trade_record['size']
    new_records['entry_idx'][r] = trade_record['entry_idx']
    new_records['entry_price'][r] = trade_record['entry_price']
    new_records['entry_fees'][r] = trade_record['entry_fees']
    new_records['exit_idx'][r] = trade_record['exit_idx']
    new_records['exit_price'][r] = trade_record['exit_price']
    new_records['exit_fees'][r] = trade_record['exit_fees']
    new_records['pnl'][r] = trade_record['pnl']
    new_records['return'][r] = trade_record['return']
    new_records['direction'][r] = trade_record['direction']
    new_records['status'][r] = trade_record['status']
    new_records['parent_id'][r] = r


@register_chunkable(
    size=records_ch.ColLensSizer(arg_query='col_map'),
    arg_take_spec=dict(
        trade_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=records_ch.ColMapSlicer()
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def get_positions_nb(trade_records: tp.RecordArray, col_map: tp.ColMap) -> tp.RecordArray:
    """Fill position records by aggregating trade records.

    Trades can be entry trades, exit trades, and even positions themselves - all will produce the same results.

    Usage:
        * Building upon the example in `get_exit_trades_nb`:

        ```pycon
        >>> from vectorbtpro.portfolio.nb import get_positions_nb

        >>> col_map = col_map_nb(exit_trade_records['col'], target_shape[1])
        >>> position_records = get_positions_nb(exit_trade_records, col_map)
        >>> pd.DataFrame.from_records(position_records)
           id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0   0    0   1.1          0     1.101818     0.01212         3    3.060000
        1   1    0   1.0          4     5.050000     0.05050         5    5.940000
        2   2    0   1.0          5     5.940000     0.05940         5    6.000000
        3   0    1   1.1          0     5.850000     0.06435         3    3.948182
        4   1    1   1.0          4     1.980000     0.01980         5    1.010000
        5   2    1   1.0          5     1.010000     0.01010         5    1.000000

           exit_fees      pnl    return  direction  status  parent_id
        0    0.03366  2.10822  1.739455          0       1          0
        1    0.05940  0.78010  0.154475          0       1          1
        2    0.00000 -0.11940 -0.020101          1       0          2
        3    0.04343  1.98422  0.308348          1       1          0
        4    0.01010  0.94010  0.474798          1       1          1
        5    0.00000 -0.02010 -0.019901          0       0          2
        ```
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    new_records = np.empty((np.max(col_lens), len(col_lens)), dtype=trade_dt)
    counts = np.full(len(col_lens), 0, dtype=np.int_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        last_position_id = -1
        from_trade_r = -1

        for c in range(col_len):
            trade_r = col_idxs[col_start_idxs[col] + c]
            record = trade_records[trade_r]

            if record['id'] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = record['id']

            parent_id = record['parent_id']

            if parent_id != last_position_id:
                if last_position_id != -1:
                    if trade_r - from_trade_r > 1:
                        fill_position_record_nb(new_records[:, col], counts[col], trade_records[from_trade_r:trade_r])
                    else:
                        # Speed up
                        copy_trade_record_nb(new_records[:, col], counts[col], trade_records[from_trade_r])
                    counts[col] += 1
                from_trade_r = trade_r
                last_position_id = parent_id

        if trade_r - from_trade_r > 0:
            fill_position_record_nb(new_records[:, col], counts[col], trade_records[from_trade_r:trade_r + 1])
        else:
            # Speed up
            copy_trade_record_nb(new_records[:, col], counts[col], trade_records[from_trade_r])
        counts[col] += 1

    return generic_nb.repartition_nb(new_records, counts)


@register_jitted(cache=True)
def trade_winning_streak_nb(records: tp.RecordArray) -> tp.Array1d:
    """Return the current winning streak of each trade."""
    out = np.full(len(records), 0, dtype=np.int_)
    curr_rank = 0
    for i in range(len(records)):
        if records[i]['pnl'] > 0:
            curr_rank += 1
        else:
            curr_rank = 0
        out[i] = curr_rank
    return out


@register_jitted(cache=True)
def trade_losing_streak_nb(records: tp.RecordArray) -> tp.Array1d:
    """Return the current losing streak of each trade."""
    out = np.full(len(records), 0, dtype=np.int_)
    curr_rank = 0
    for i in range(len(records)):
        if records[i]['pnl'] < 0:
            curr_rank += 1
        else:
            curr_rank = 0
        out[i] = curr_rank
    return out


@register_jitted(cache=True)
def win_rate_1d_nb(pnl_arr: tp.Array1d) -> float:
    """Win rate of a PnL array."""
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_count = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_count += 1
    return win_count / pnl_arr.shape[0]


@register_jitted(cache=True)
def profit_factor_1d_nb(pnl_arr: tp.Array1d) -> float:
    """Profit factor of a PnL array."""
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_sum = 0
    loss_sum = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_sum += pnl_arr[i]
            elif pnl_arr[i] < 0:
                loss_sum += abs(pnl_arr[i])
    if loss_sum == 0:
        return np.inf
    return win_sum / loss_sum


@register_jitted(cache=True)
def expectancy_1d_nb(pnl_arr: tp.Array1d) -> float:
    """Expectancy of a PnL array."""
    if pnl_arr.shape[0] == 0:
        return np.nan
    win_count = 0
    win_sum = 0
    loss_count = 0
    loss_sum = 0
    count = 0
    for i in range(len(pnl_arr)):
        if not np.isnan(pnl_arr[i]):
            count += 1
            if pnl_arr[i] > 0:
                win_count += 1
                win_sum += pnl_arr[i]
            elif pnl_arr[i] < 0:
                loss_count += 1
                loss_sum += abs(pnl_arr[i])
    win_rate = win_count / pnl_arr.shape[0]
    if win_count == 0:
        win_mean = 0.
    else:
        win_mean = win_sum / win_count
    loss_rate = loss_count / pnl_arr.shape[0]
    if loss_count == 0:
        loss_mean = 0.
    else:
        loss_mean = loss_sum / loss_count
    return win_rate * win_mean - loss_rate * loss_mean


@register_jitted(cache=True)
def sqn_1d_nb(pnl_arr: tp.Array1d, ddof: int = 0) -> float:
    """SQN of a PnL array."""
    count = generic_nb.nancnt_1d_nb(pnl_arr)
    mean = np.nanmean(pnl_arr)
    std = generic_nb.nanstd_1d_nb(pnl_arr, ddof=ddof)
    return np.sqrt(count) * mean / std
