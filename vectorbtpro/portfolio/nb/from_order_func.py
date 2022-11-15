# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio modeling based on an order function."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.portfolio.nb.iter_ import *
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.array_ import insert_argsort_nb
from vectorbtpro.utils.template import RepFunc


@register_jitted
def get_group_value_ctx_nb(seg_ctx: SegmentContext) -> float:
    """Get group value from context.

    Accepts `vectorbtpro.portfolio.enums.SegmentContext`.

    Best called once from `pre_segment_func_nb`.
    To set the valuation price, change `last_val_price` of the context in-place.

    !!! note
        Cash sharing must be enabled."""
    if not seg_ctx.cash_sharing:
        raise ValueError("Cash sharing must be enabled")
    return get_group_value_nb(
        seg_ctx.from_col,
        seg_ctx.to_col,
        seg_ctx.last_cash[seg_ctx.group],
        seg_ctx.last_position,
        seg_ctx.last_val_price,
    )


@register_jitted
def sort_call_seq_out_nb(
    ctx: SegmentContext,
    size: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    order_value_out: tp.Array1d,
    call_seq_out: tp.Array1d,
    ctx_select: bool = True,
) -> None:
    """Sort call sequence `call_seq_out` based on the value of each potential order.

    Accepts `vectorbtpro.portfolio.enums.SegmentContext` and other arguments, sorts `call_seq_out` in place,
    and returns nothing.

    Arrays `size`, `size_type`, and `direction` utilize flexible indexing and must have at least 0 dimensions.
    If `ctx_select` is True, selects the elements of each `size`, `size_type`, and `direction`
    using `select_from_col_nb` assuming that each array can broadcast to `target_shape`.
    Otherwise, selects using `vectorbtpro.base.indexing.flex_select_auto_nb` assuming that each array
    can broadcast to `group_len`.

    The lengths of `order_value_out` and `call_seq_out` must match the number of columns in the group.
    Array `order_value_out` must be empty and will contain sorted order values after execution.
    Array `call_seq_out` must be filled with integers ranging from 0 to the number of columns in the group
    (in this exact order).

    Best called once from `pre_segment_func_nb`.

    !!! note
        Cash sharing must be enabled and `call_seq_out` must follow `CallSeqType.Default`.

        Should be used in flexible simulation functions."""
    if not ctx.cash_sharing:
        raise ValueError("Cash sharing must be enabled")

    group_value_now = get_group_value_ctx_nb(ctx)
    group_len = ctx.to_col - ctx.from_col
    for c in range(group_len):
        if call_seq_out[c] != c:
            raise ValueError("call_seq_out must follow CallSeqType.Default")
        col = ctx.from_col + c
        if ctx_select:
            _size = select_from_col_nb(ctx, col, size)
            _size_type = select_from_col_nb(ctx, col, size_type)
            _direction = select_from_col_nb(ctx, col, direction)
        else:
            _size = flex_select_auto_nb(size, c, 0, False)
            _size_type = flex_select_auto_nb(size_type, c, 0, False)
            _direction = flex_select_auto_nb(direction, c, 0, False)
        if ctx.cash_sharing:
            cash_now = ctx.last_cash[ctx.group]
            free_cash_now = ctx.last_free_cash[ctx.group]
        else:
            cash_now = ctx.last_cash[col]
            free_cash_now = ctx.last_free_cash[col]
        exec_state = ExecState(
            cash=cash_now,
            position=ctx.last_position[col],
            debt=ctx.last_debt[col],
            free_cash=free_cash_now,
            val_price=ctx.last_val_price[col],
            value=group_value_now,
        )
        order_value_out[c] = approx_order_value_nb(
            exec_state,
            _size,
            _size_type,
            _direction,
        )
    # Sort by order value
    insert_argsort_nb(order_value_out, call_seq_out)


@register_jitted
def sort_call_seq_nb(
    ctx: SegmentContext,
    size: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    order_value_out: tp.Array1d,
    ctx_select: bool = True,
) -> None:
    """Sort call sequence attached to `vectorbtpro.portfolio.enums.SegmentContext`.

    See `sort_call_seq_out_nb`.

    !!! note
        Can only be used in non-flexible simulation functions."""
    if ctx.call_seq_now is None:
        raise ValueError("Call sequence array is None. Use sort_call_seq_out_nb to sort a custom array.")
    sort_call_seq_out_nb(ctx, size, size_type, direction, order_value_out, ctx.call_seq_now, ctx_select=ctx_select)


@register_jitted
def try_order_nb(ctx: OrderContext, order: Order) -> tp.Tuple[AccountState, OrderResult]:
    """Execute an order without persistence."""
    exec_state = ExecState(
        cash=ctx.cash_now,
        position=ctx.position_now,
        debt=ctx.debt_now,
        free_cash=ctx.free_cash_now,
        val_price=ctx.val_price_now,
        value=ctx.value_now,
    )
    price_area = PriceArea(
        open=flex_select_auto_nb(ctx.open, ctx.i, ctx.col, ctx.flex_2d),
        high=flex_select_auto_nb(ctx.high, ctx.i, ctx.col, ctx.flex_2d),
        low=flex_select_auto_nb(ctx.low, ctx.i, ctx.col, ctx.flex_2d),
        close=flex_select_auto_nb(ctx.close, ctx.i, ctx.col, ctx.flex_2d),
    )
    return execute_order_nb(exec_state=exec_state, order=order, price_area=price_area)


@register_jitted
def no_pre_func_nb(c: tp.NamedTuple, *args) -> tp.Args:
    """Placeholder preprocessing function that forwards received arguments down the stack."""
    return args


@register_jitted
def no_order_func_nb(c: OrderContext, *args) -> Order:
    """Placeholder order function that returns no order."""
    return NoOrder


@register_jitted
def no_post_func_nb(c: tp.NamedTuple, *args) -> None:
    """Placeholder postprocessing function that returns nothing."""
    return None


@register_jitted
def set_val_price_nb(c: SegmentContext, val_price: tp.FlexArray, price: tp.FlexArray) -> None:
    """Override valuation price in a context.

    Allows specifying a valuation price of positive infinity (takes the current price)
    and negative infinity (takes the latest valuation price)."""
    for col in range(c.from_col, c.to_col):
        _val_price = select_from_col_nb(c, col, val_price)
        if np.isinf(_val_price):
            if _val_price > 0:
                _price = select_from_col_nb(c, col, price)
                if np.isinf(_price):
                    if _price > 0:
                        _price = select_from_col_nb(c, col, c.close)
                    else:
                        _price = select_from_col_nb(c, col, c.open)
                _val_price = _price
            else:
                _val_price = c.last_val_price[col]
        if not np.isnan(_val_price) or not c.ffill_val_price:
            c.last_val_price[col] = _val_price


@register_jitted
def def_pre_segment_func_nb(
    c: SegmentContext,
    val_price: tp.FlexArray,
    price: tp.FlexArray,
    size: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    auto_call_seq: bool,
) -> tp.Args:
    """Pre-segment function that overrides the valuation price and optionally sorts the call sequence."""
    set_val_price_nb(c, val_price, price)
    if auto_call_seq:
        order_value_out = np.empty(c.group_len, dtype=np.float_)
        sort_call_seq_nb(c, size, size_type, direction, order_value_out)
    return ()


@register_jitted
def def_order_func_nb(
    c: OrderContext,
    size: tp.FlexArray,
    price: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    fees: tp.FlexArray,
    fixed_fees: tp.FlexArray,
    slippage: tp.FlexArray,
    min_size: tp.FlexArray,
    max_size: tp.FlexArray,
    size_granularity: tp.FlexArray,
    reject_prob: tp.FlexArray,
    price_area_vio_mode: tp.FlexArray,
    lock_cash: tp.FlexArray,
    allow_partial: tp.FlexArray,
    raise_reject: tp.FlexArray,
    log: tp.FlexArray,
) -> tp.Tuple[int, Order]:
    """Order function that creates an order based on default information."""
    return order_nb(
        size=select_nb(c, size),
        price=select_nb(c, price),
        size_type=select_nb(c, size_type),
        direction=select_nb(c, direction),
        fees=select_nb(c, fees),
        fixed_fees=select_nb(c, fixed_fees),
        slippage=select_nb(c, slippage),
        min_size=select_nb(c, min_size),
        max_size=select_nb(c, max_size),
        size_granularity=select_nb(c, size_granularity),
        reject_prob=select_nb(c, reject_prob),
        price_area_vio_mode=select_nb(c, price_area_vio_mode),
        lock_cash=select_nb(c, lock_cash),
        allow_partial=select_nb(c, allow_partial),
        raise_reject=select_nb(c, raise_reject),
        log=select_nb(c, log),
    )


PreSimFuncT = tp.Callable[[SimulationContext, tp.VarArg()], tp.Args]
PostSimFuncT = tp.Callable[[SimulationContext, tp.VarArg()], None]
PreGroupFuncT = tp.Callable[[GroupContext, tp.VarArg()], tp.Args]
PostGroupFuncT = tp.Callable[[GroupContext, tp.VarArg()], None]
PreRowFuncT = tp.Callable[[RowContext, tp.VarArg()], tp.Args]
PostRowFuncT = tp.Callable[[RowContext, tp.VarArg()], None]
PreSegmentFuncT = tp.Callable[[SegmentContext, tp.VarArg()], tp.Args]
PostSegmentFuncT = tp.Callable[[SegmentContext, tp.VarArg()], None]
OrderFuncT = tp.Callable[[OrderContext, tp.VarArg()], Order]
PostOrderFuncT = tp.Callable[[PostOrderContext, OrderResult, tp.VarArg()], None]


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        call_seq=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=portfolio_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,
        post_sim_args=ch.ArgsTaker(),
        pre_group_func_nb=None,
        pre_group_args=ch.ArgsTaker(),
        post_group_func_nb=None,
        post_group_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,
        post_segment_args=ch.ArgsTaker(),
        order_func_nb=None,
        order_args=ch.ArgsTaker(),
        post_order_func_nb=None,
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        bm_close=portfolio_ch.flex_array_gl_slicer,
        ffill_val_price=None,
        update_value=None,
        fill_pos_record=None,
        track_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(tags={"can_parallel"})
def simulate_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    call_seq: tp.Optional[tp.Array2d] = None,
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(np.nan),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    segment_mask: tp.FlexArray = np.asarray(True),
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: PostSimFuncT = no_post_func_nb,
    post_sim_args: tp.Args = (),
    pre_group_func_nb: PreGroupFuncT = no_pre_func_nb,
    pre_group_args: tp.Args = (),
    post_group_func_nb: PostGroupFuncT = no_post_func_nb,
    post_group_args: tp.Args = (),
    pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
    post_segment_args: tp.Args = (),
    order_func_nb: OrderFuncT = no_order_func_nb,
    order_args: tp.Args = (),
    post_order_func_nb: PostOrderFuncT = no_post_func_nb,
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray = np.asarray(np.nan),
    high: tp.FlexArray = np.asarray(np.nan),
    low: tp.FlexArray = np.asarray(np.nan),
    close: tp.FlexArray = np.asarray(np.nan),
    bm_close: tp.FlexArray = np.asarray(np.nan),
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_record: bool = True,
    track_value: bool = True,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = False,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Fill order and log records by iterating over a shape and calling a range of user-defined functions.

    Starting with initial cash `init_cash`, iterates over each group and column in `target_shape`,
    and for each data point, generates an order using `order_func_nb`. Tries then to fulfill that
    order. Upon success, updates the current state including the cash balance and the position.
    Returns `vectorbtpro.portfolio.enums.SimulationOutput`.

    As opposed to `simulate_row_wise_nb`, order processing happens in column-major order.
    Column-major order means processing the entire column/group with all rows before moving to the next one.
    See [Row- and column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).

    Args:
        target_shape (tuple): See `vectorbtpro.portfolio.enums.SimulationContext.target_shape`.
        group_lens (array_like of int): See `vectorbtpro.portfolio.enums.SimulationContext.group_lens`.
        cash_sharing (bool): See `vectorbtpro.portfolio.enums.SimulationContext.cash_sharing`.
        call_seq (array_like of int): See `vectorbtpro.portfolio.enums.SimulationContext.call_seq`.
        init_cash (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.init_cash`.
        init_position (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.init_position`.
        init_price (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.init_price`.
        cash_deposits (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.cash_deposits`.
        cash_earnings (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.cash_earnings`.
        segment_mask (array_like of bool): See `vectorbtpro.portfolio.enums.SimulationContext.segment_mask`.
        call_pre_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_pre_segment`.
        call_post_segment (bool): See `vectorbtpro.portfolio.enums.SimulationContext.call_post_segment`.
        pre_sim_func_nb (callable): Function called before simulation.

            Can be used for creation of global arrays and setting the seed.

            Must accept `vectorbtpro.portfolio.enums.SimulationContext` and `*pre_sim_args`.
            Must return a tuple of any content, which is then passed to `pre_group_func_nb` and
            `post_group_func_nb`.
        pre_sim_args (tuple): Packed arguments passed to `pre_sim_func_nb`.
        post_sim_func_nb (callable): Function called after simulation.

            Must accept `vectorbtpro.portfolio.enums.SimulationContext` and `*post_sim_args`.
            Must return nothing.
        post_sim_args (tuple): Packed arguments passed to `post_sim_func_nb`.
        pre_group_func_nb (callable): Function called before each group.

            Must accept `vectorbtpro.portfolio.enums.GroupContext`, unpacked tuple from `pre_sim_func_nb`,
            and `*pre_group_args`. Must return a tuple of any content, which is then passed to
            `pre_segment_func_nb` and `post_segment_func_nb`.
        pre_group_args (tuple): Packed arguments passed to `pre_group_func_nb`.
        post_group_func_nb (callable): Function called after each group.

            Must accept `vectorbtpro.portfolio.enums.GroupContext`, unpacked tuple from `pre_sim_func_nb`,
            and `*post_group_args`. Must return nothing.
        post_group_args (tuple): Packed arguments passed to `post_group_func_nb`.
        pre_segment_func_nb (callable): Function called before each segment.

            Called if `segment_mask` or `call_pre_segment` is True.

            Must accept `vectorbtpro.portfolio.enums.SegmentContext`, unpacked tuple from `pre_group_func_nb`,
            and `*pre_segment_args`. Must return a tuple of any content, which is then passed to
            `order_func_nb` and `post_order_func_nb`.

            This is the right place to change call sequence and set the valuation price.
            Group re-valuation and update of the open position stats happens right after this function,
            regardless of whether it has been called.

            !!! note
                To change the call sequence of a segment, access
                `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` and change it in-place.
                Make sure to not generate any new arrays as it may negatively impact performance.
                Assigning `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now` as any other context
                (named tuple) value is not supported. See `vectorbtpro.portfolio.enums.SegmentContext.call_seq_now`.

            !!! note
                You can override elements of `last_val_price` to manipulate group valuation.
                See `vectorbtpro.portfolio.enums.SimulationContext.last_val_price`.
        pre_segment_args (tuple): Packed arguments passed to `pre_segment_func_nb`.
        post_segment_func_nb (callable): Function called after each segment.

            Called if `segment_mask` or `call_post_segment` is True.

            Addition of cash_earnings, the final group re-valuation, and the final update of the open
            position stats happens right before this function, regardless of whether it has been called.

            The passed context represents the final state of each segment, thus makes sure
            to do any changes before this function is called.

            Must accept `vectorbtpro.portfolio.enums.SegmentContext`, unpacked tuple from `pre_group_func_nb`,
            and `*post_segment_args`. Must return nothing.
        post_segment_args (tuple): Packed arguments passed to `post_segment_func_nb`.
        order_func_nb (callable): Order generation function.

            Used for either generating an order or skipping.

            Must accept `vectorbtpro.portfolio.enums.OrderContext`, unpacked tuple from `pre_segment_func_nb`,
            and `*order_args`. Must return `vectorbtpro.portfolio.enums.Order`.

            !!! note
                If the returned order has been rejected, there is no way of issuing a new order.
                You should make sure that the order passes, for example, by using `try_order_nb`.

                To have a greater freedom in order management, use `flex_simulate_nb`.
        order_args (tuple): Arguments passed to `order_func_nb`.
        post_order_func_nb (callable): Callback that is called after the order has been processed.

            Used for checking the order status and doing some post-processing.

            Must accept `vectorbtpro.portfolio.enums.PostOrderContext`, unpacked tuple from
            `pre_segment_func_nb`, and `*post_order_args`. Must return nothing.
        post_order_args (tuple): Arguments passed to `post_order_func_nb`.
        index (array): See `vectorbtpro.portfolio.enums.SimulationContext.index`.
        freq (int): See `vectorbtpro.portfolio.enums.SimulationContext.freq`.
        open (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.open`.
        high (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.high`.
        low (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.low`.
        close (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.close`.
        bm_close (array_like of float): See `vectorbtpro.portfolio.enums.SimulationContext.bm_close`.
        ffill_val_price (bool): See `vectorbtpro.portfolio.enums.SimulationContext.ffill_val_price`.
        update_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.update_value`.
        fill_pos_record (bool): See `vectorbtpro.portfolio.enums.SimulationContext.fill_pos_record`.
        track_value (bool): See `vectorbtpro.portfolio.enums.SimulationContext.track_value`.
        max_orders (int): The max number of order records expected to be filled at each column.
        max_logs (int): The max number of log records expected to be filled at each column.
        flex_2d (bool): See `vectorbtpro.portfolio.enums.SimulationContext.flex_2d`.
        in_outputs (bool): See `vectorbtpro.portfolio.enums.SimulationContext.in_outputs`.

    !!! note
        Remember that indexing of 2-dim arrays in vectorbt follows that of pandas: `a[i, col]`.

    !!! warning
        You can only safely access data of columns that are to the left of the current group and
        rows that are to the top of the current row within the same group. Other data points have
        not been processed yet and thus empty. Accessing them will not trigger any errors or warnings,
        but provide you with arbitrary data (see [np.empty](https://numpy.org/doc/stable/reference/generated/numpy.empty.html)).

    Call hierarchy:
        Like most things in the vectorbt universe, simulation is also done by iterating over a (imaginary) frame.
        This frame consists of two dimensions: time (rows) and assets/features (columns).
        Each element of this frame is a potential order, which gets generated by calling an order function.

        The question is: how do we move across this frame to simulate trading? There are two movement patterns:
        column-major (as done by `simulate_nb`) and row-major order (as done by `simulate_row_wise_nb`).
        In each of these patterns, we are always moving from top to bottom (time axis) and from left to right
        (asset/feature axis); the only difference between them is across which axis we are moving faster:
        do we want to process each column first (thus assuming that columns are independent) or each row?
        Choosing between them is mostly a matter of preference, but it also makes different data being
        available when generating an order.

        The frame is further divided into "blocks": columns, groups, rows, segments, and elements.
        For example, columns can be grouped into groups that may or may not share the same capital.
        Regardless of capital sharing, each collection of elements within a group and a time step is called
        a segment, which simply defines a single context (such as shared capital) for one or multiple orders.
        Each segment can also define a custom sequence (a so-called call sequence) in which orders are executed.

        You can imagine each of these blocks as a rectangle drawn over different parts of the frame,
        and having its own context and pre/post-processing function. The pre-processing function is a
        simple callback that is called before entering the block, and can be provided by the user to, for example,
        prepare arrays or do some custom calculations. It must return a tuple (can be empty) that is then unpacked and
        passed as arguments to the pre- and postprocessing function coming next in the call hierarchy.
        The postprocessing function can be used, for example, to write user-defined arrays such as returns.

        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_group_out = pre_group_func_nb(GroupContext, *pre_sim_out, *pre_group_args)
                3. if call_pre_segment or segment_mask: pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_group_out, *pre_segment_args)
                    4. if segment_mask: order = order_func_nb(OrderContext, *pre_segment_out, *order_args)
                    5. if order: post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                    ...
                6. if call_post_segment or segment_mask: post_segment_func_nb(SegmentContext, *pre_group_out, *post_segment_args)
                ...
            7. post_group_func_nb(GroupContext, *pre_sim_out, *post_group_args)
            ...
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's demonstrate a frame with one group of two columns and one group of one column, and the
        following call sequence:

        ```plaintext
        array([[0, 1, 0],
               [1, 0, 0]])
        ```

        ![](/assets/images/api/simulate_nb.svg)

        And here is the context information available at each step:

        ![](/assets/images/api/context_info.svg)

    Usage:
        * Create a group of three assets together sharing 100$ and simulate an equal-weighted portfolio
        that rebalances every second tick, all without leaving Numba:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from collections import namedtuple
        >>> from numba import njit
        >>> from vectorbtpro.generic.plotting import Scatter
        >>> from vectorbtpro.records.nb import col_map_nb
        >>> from vectorbtpro.portfolio.enums import SizeType, Direction
        >>> from vectorbtpro.portfolio.call_seq import build_call_seq
        >>> from vectorbtpro.portfolio.nb import (
        ...     select_from_col_nb,
        ...     select_nb,
        ...     order_nb,
        ...     simulate_nb,
        ...     simulate_row_wise_nb,
        ...     sort_call_seq_nb,
        ...     asset_flow_nb,
        ...     assets_nb,
        ...     asset_value_nb
        ... )

        >>> @njit
        ... def pre_sim_func_nb(c):
        ...     print('before simulation')
        ...     # Create a temporary array and pass it down the stack
        ...     order_value_out = np.empty(c.target_shape[1], dtype=np.float_)
        ...     return (order_value_out,)

        >>> @njit
        ... def pre_group_func_nb(c, order_value_out):
        ...     print('\\tbefore group', c.group)
        ...     # Forward down the stack (you can omit pre_group_func_nb entirely)
        ...     return (order_value_out,)

        >>> @njit
        ... def pre_segment_func_nb(c, order_value_out, size, price, size_type, direction):
        ...     print('\\t\\tbefore segment', c.i)
        ...     for col in range(c.from_col, c.to_col):
        ...         # Here we use order price for group valuation
        ...         c.last_val_price[col] = select_from_col_nb(c, col, price)
        ...
        ...     # Reorder call sequence of this segment such that selling orders come first and buying last
        ...     # Rearranges c.call_seq_now based on order value (size, size_type, direction, and val_price)
        ...     # Utilizes flexible indexing using select_from_col_nb (as we did above)
        ...     sort_call_seq_nb(c, size, size_type, direction, order_value_out[c.from_col:c.to_col])
        ...     # Forward nothing
        ...     return ()

        >>> @njit
        ... def order_func_nb(c, size, price, size_type, direction, fees, fixed_fees, slippage):
        ...     print('\\t\\t\\tcreating order', c.call_idx, 'at column', c.col)
        ...     # Create and return an order
        ...     return order_nb(
        ...         size=select_nb(c, size),
        ...         price=select_nb(c, price),
        ...         size_type=select_nb(c, size_type),
        ...         direction=select_nb(c, direction),
        ...         fees=select_nb(c, fees),
        ...         fixed_fees=select_nb(c, fixed_fees),
        ...         slippage=select_nb(c, slippage)
        ...     )

        >>> @njit
        ... def post_order_func_nb(c):
        ...     print('\\t\\t\\t\\torder status:', c.order_result.status)
        ...     return None

        >>> @njit
        ... def post_segment_func_nb(c, order_value_out):
        ...     print('\\t\\tafter segment', c.i)
        ...     return None

        >>> @njit
        ... def post_group_func_nb(c, order_value_out):
        ...     print('\\tafter group', c.group)
        ...     return None

        >>> @njit
        ... def post_sim_func_nb(c):
        ...     print('after simulation')
        ...     return None

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_lens)  # will be overridden
        >>> segment_mask = np.array([True, False, True, False, True])[:, None]
        >>> segment_mask = np.copy(np.broadcast_to(segment_mask, target_shape))
        >>> size = np.asarray(1 / target_shape[1])  # scalars must become 0-dim arrays
        >>> price = close = np.random.uniform(1, 10, size=target_shape)
        >>> size_type = np.asarray(SizeType.TargetPercent)
        >>> direction = np.asarray(Direction.LongOnly)
        >>> fees = np.asarray(0.001)
        >>> fixed_fees = np.asarray(1.)
        >>> slippage = np.asarray(0.001)

        >>> sim_out = simulate_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     call_seq,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_group_func_nb=pre_group_func_nb,
        ...     post_group_func_nb=post_group_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     order_func_nb=order_func_nb,
        ...     order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before group 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                after segment 0
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                after segment 2
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                after segment 4
            after group 0
        after simulation

        >>> pd.DataFrame.from_records(sim_out.order_records)
           id  col  idx       size     price      fees  side
        0   0    0    0   7.626262  4.375232  1.033367     0
        1   1    1    0   3.488053  9.565985  1.033367     0
        2   2    2    0   3.972040  7.595533  1.030170     0
        3   3    1    2   0.920352  8.786790  1.008087     1
        4   4    2    2   0.448747  6.403625  1.002874     1
        5   5    0    2   5.210115  1.524275  1.007942     0
        6   6    0    4   7.899568  8.483492  1.067016     1
        7   7    2    4  12.378281  2.639061  1.032667     0
        8   8    1    4  10.713236  2.913963  1.031218     0

        >>> call_seq
        array([[0, 1, 2],
               [0, 1, 2],
               [1, 2, 0],
               [0, 1, 2],
               [0, 2, 1]])

        >>> col_map = col_map_nb(sim_out.order_records['col'], target_shape[1])
        >>> asset_flow = asset_flow_nb(target_shape, sim_out.order_records, col_map)
        >>> assets = assets_nb(asset_flow)
        >>> asset_value = asset_value_nb(close, assets)
        >>> Scatter(data=asset_value).fig.show()
        ```

        ![](/assets/images/api/simulate_nb_example.svg)

        Note that the last order in a group with cash sharing is always disadvantaged
        as it has a bit less funds than the previous orders due to costs, which are not
        included when valuating the group.
    """
    check_group_lens_nb(group_lens, target_shape[1])

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
    last_pos_record = prepare_last_pos_record_nb(
        target_shape,
        init_position=init_position,
        init_price=init_price,
        fill_pos_record=fill_pos_record,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        for i in range(target_shape[0]):
            if call_seq is None:
                for c in range(group_len):
                    temp_call_seq[c] = c
                call_seq_now = temp_call_seq[:group_len]
            else:
                call_seq_now = call_seq[i, from_col:to_col]

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(prev_close_value[group], last_value[group])
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(prev_close_value[col], last_value[col])

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_group_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_auto_nb(cash_deposits, i, col, flex_2d)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if is_segment_active:
                for k in range(group_len):
                    if cash_sharing:
                        c = call_seq_now[k]
                        if c >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        c = k
                    col = from_col + c

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)

                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError("Cannot use size type that depends on not tracked value")

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(prev_close_value[col], value_now - cash_deposits_now)

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_record:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_record_nb(
                                pos_record_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function before the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        call_seq=ch.ArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=portfolio_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,
        post_sim_args=ch.ArgsTaker(),
        pre_row_func_nb=None,
        pre_row_args=ch.ArgsTaker(),
        post_row_func_nb=None,
        post_row_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,
        post_segment_args=ch.ArgsTaker(),
        order_func_nb=None,
        order_args=ch.ArgsTaker(),
        post_order_func_nb=None,
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        bm_close=portfolio_ch.flex_array_gl_slicer,
        ffill_val_price=None,
        update_value=None,
        fill_pos_record=None,
        track_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted
def simulate_row_wise_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    call_seq: tp.Array2d,
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(np.nan),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    segment_mask: tp.FlexArray = np.asarray(True),
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: PostSimFuncT = no_post_func_nb,
    post_sim_args: tp.Args = (),
    pre_row_func_nb: PreRowFuncT = no_pre_func_nb,
    pre_row_args: tp.Args = (),
    post_row_func_nb: PostRowFuncT = no_post_func_nb,
    post_row_args: tp.Args = (),
    pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
    post_segment_args: tp.Args = (),
    order_func_nb: OrderFuncT = no_order_func_nb,
    order_args: tp.Args = (),
    post_order_func_nb: PostOrderFuncT = no_post_func_nb,
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray = np.asarray(np.nan),
    high: tp.FlexArray = np.asarray(np.nan),
    low: tp.FlexArray = np.asarray(np.nan),
    close: tp.FlexArray = np.asarray(np.nan),
    bm_close: tp.FlexArray = np.asarray(np.nan),
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_record: bool = True,
    track_value: bool = True,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = False,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `simulate_nb`, but iterates in row-major order.

    Row-major order means processing the entire row with all groups/columns before moving to the next one.

    The main difference is that instead of `pre_group_func_nb` it now exposes `pre_row_func_nb`,
    which is executed per entire row. It must accept `vectorbtpro.portfolio.enums.RowContext`.

    !!! note
        Function `pre_row_func_nb` is only called if there is at least on active segment in
        the row. Functions `pre_segment_func_nb` and `order_func_nb` are only called if their
        segment is active. If the main task of `pre_row_func_nb` is to activate/deactivate segments,
        all segments must be activated by default to allow `pre_row_func_nb` to be called.

    !!! warning
        You can only safely access data points that are to the left of the current group and
        rows that are to the top of the current row.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_row_out = pre_row_func_nb(RowContext, *pre_sim_out, *pre_row_args)
                3. if call_pre_segment or segment_mask: pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_row_out, *pre_segment_args)
                    4. if segment_mask: order = order_func_nb(OrderContext, *pre_segment_out, *order_args)
                    5. if order: post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                    ...
                6. if call_post_segment or segment_mask: post_segment_func_nb(SegmentContext, *pre_row_out, *post_segment_args)
                ...
            7. post_row_func_nb(RowContext, *pre_sim_out, *post_row_args)
            ...
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate the same example as in `simulate_nb` but adapted for this function:

        ![](/assets/images/api/simulate_row_wise_nb.svg)

    Usage:
        * Running the same example as in `simulate_nb` but adapted for this function:

        ```pycon
        >>> @njit
        ... def pre_row_func_nb(c, order_value_out):
        ...     print('\\tbefore row', c.i)
        ...     # Forward down the stack
        ...     return (order_value_out,)

        >>> @njit
        ... def post_row_func_nb(c, order_value_out):
        ...     print('\\tafter row', c.i)
        ...     return None

        >>> call_seq = build_call_seq(target_shape, group_lens)
        >>> sim_out = simulate_row_wise_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     call_seq,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_row_func_nb=pre_row_func_nb,
        ...     post_row_func_nb=post_row_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     order_func_nb=order_func_nb,
        ...     order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before row 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                after segment 0
            after row 0
            before row 1
            after row 1
            before row 2
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                after segment 2
            after row 2
            before row 3
            after row 3
            before row 4
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                after segment 4
            after row 4
        after simulation
        ```
    """
    check_group_lens_nb(group_lens, target_shape[1])

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
    last_pos_record = prepare_last_pos_record_nb(
        target_shape,
        init_position=init_position,
        init_price=init_price,
        fill_pos_record=fill_pos_record,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    temp_call_seq = np.empty(target_shape[1], dtype=np.int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for i in range(target_shape[0]):

        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            i=i,
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        for group in range(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            group_len = to_col - from_col

            if call_seq is None:
                for c in range(group_len):
                    temp_call_seq[c] = c
                call_seq_now = temp_call_seq[:group_len]
            else:
                call_seq_now = call_seq[i, from_col:to_col]

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(prev_close_value[group], last_value[group])
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(prev_close_value[col], last_value[col])

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_auto_nb(cash_deposits, i, col, flex_2d)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if is_segment_active:
                for k in range(group_len):
                    if cash_sharing:
                        c = call_seq_now[k]
                        if c >= group_len:
                            raise ValueError("Call index out of bounds of the group")
                    else:
                        c = k
                    col = from_col + c

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

                    # Generate the next order
                    order_ctx = OrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    order = order_func_nb(order_ctx, *pre_segment_out, *order_args)

                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError("Cannot use size type that depends on not tracked value")

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(prev_close_value[col], value_now - cash_deposits_now)

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_record:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_record_nb(
                                pos_record_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=call_seq,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=call_seq_now,
                        col=col,
                        call_idx=k,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=call_seq,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=call_seq_now,
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            i=i,
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=call_seq,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


@register_jitted
def no_flex_order_func_nb(c: FlexOrderContext, *args) -> tp.Tuple[int, Order]:
    """Placeholder flexible order function that returns "break" column and no order."""
    return -1, NoOrder


@register_jitted
def def_flex_pre_segment_func_nb(
    c: SegmentContext,
    val_price: tp.FlexArray,
    price: tp.FlexArray,
    size: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    auto_call_seq: bool,
) -> tp.Args:
    """Flexible pre-segment function that overrides the valuation price and optionally sorts the call sequence."""
    set_val_price_nb(c, val_price, price)
    call_seq_out = np.arange(c.group_len)
    if auto_call_seq:
        order_value_out = np.empty(c.group_len, dtype=np.float_)
        sort_call_seq_out_nb(c, size, size_type, direction, order_value_out, call_seq_out)
    return (call_seq_out,)


@register_jitted
def def_flex_order_func_nb(
    c: FlexOrderContext,
    call_seq_now: tp.Array1d,
    size: tp.FlexArray,
    price: tp.FlexArray,
    size_type: tp.FlexArray,
    direction: tp.FlexArray,
    fees: tp.FlexArray,
    fixed_fees: tp.FlexArray,
    slippage: tp.FlexArray,
    min_size: tp.FlexArray,
    max_size: tp.FlexArray,
    size_granularity: tp.FlexArray,
    reject_prob: tp.FlexArray,
    price_area_vio_mode: tp.FlexArray,
    lock_cash: tp.FlexArray,
    allow_partial: tp.FlexArray,
    raise_reject: tp.FlexArray,
    log: tp.FlexArray,
) -> tp.Tuple[int, Order]:
    """Flexible order function that creates an order based on default information."""
    if c.call_idx < c.group_len:
        col = c.from_col + call_seq_now[c.call_idx]
        order = order_nb(
            size=select_from_col_nb(c, col, size),
            price=select_from_col_nb(c, col, price),
            size_type=select_from_col_nb(c, col, size_type),
            direction=select_from_col_nb(c, col, direction),
            fees=select_from_col_nb(c, col, fees),
            fixed_fees=select_from_col_nb(c, col, fixed_fees),
            slippage=select_from_col_nb(c, col, slippage),
            min_size=select_from_col_nb(c, col, min_size),
            max_size=select_from_col_nb(c, col, max_size),
            size_granularity=select_from_col_nb(c, col, size_granularity),
            reject_prob=select_from_col_nb(c, col, reject_prob),
            price_area_vio_mode=select_from_col_nb(c, col, price_area_vio_mode),
            lock_cash=select_from_col_nb(c, col, lock_cash),
            allow_partial=select_from_col_nb(c, col, allow_partial),
            raise_reject=select_from_col_nb(c, col, raise_reject),
            log=select_from_col_nb(c, col, log),
        )
        return col, order
    return -1, order_nothing_nb()


FlexOrderFuncT = tp.Callable[[FlexOrderContext, tp.VarArg()], tp.Tuple[int, Order]]


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=portfolio_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,
        post_sim_args=ch.ArgsTaker(),
        pre_group_func_nb=None,
        pre_group_args=ch.ArgsTaker(),
        post_group_func_nb=None,
        post_group_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,
        post_segment_args=ch.ArgsTaker(),
        flex_order_func_nb=None,
        flex_order_args=ch.ArgsTaker(),
        post_order_func_nb=None,
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        bm_close=portfolio_ch.flex_array_gl_slicer,
        ffill_val_price=None,
        update_value=None,
        fill_pos_record=None,
        track_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted(tags={"can_parallel"})
def flex_simulate_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(0.0),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    segment_mask: tp.FlexArray = np.asarray(True),
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: PostSimFuncT = no_post_func_nb,
    post_sim_args: tp.Args = (),
    pre_group_func_nb: PreGroupFuncT = no_pre_func_nb,
    pre_group_args: tp.Args = (),
    post_group_func_nb: PostGroupFuncT = no_post_func_nb,
    post_group_args: tp.Args = (),
    pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
    post_segment_args: tp.Args = (),
    flex_order_func_nb: FlexOrderFuncT = no_flex_order_func_nb,
    flex_order_args: tp.Args = (),
    post_order_func_nb: PostOrderFuncT = no_post_func_nb,
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray = np.asarray(np.nan),
    high: tp.FlexArray = np.asarray(np.nan),
    low: tp.FlexArray = np.asarray(np.nan),
    close: tp.FlexArray = np.asarray(np.nan),
    bm_close: tp.FlexArray = np.asarray(np.nan),
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_record: bool = True,
    track_value: bool = True,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = False,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `simulate_nb`, but with no predefined call sequence.

    In contrast to `order_func_nb` in`simulate_nb`, `post_order_func_nb` is a segment-level order function
    that returns a column along with the order, and gets repeatedly called until some condition is met.
    This allows multiple orders to be issued within a single element and in an arbitrary order.

    The order function must accept `vectorbtpro.portfolio.enums.FlexOrderContext`, unpacked tuple from
    `pre_segment_func_nb`, and `*flex_order_args`. Must return column and `vectorbtpro.portfolio.enums.Order`.
    To break out of the loop, return column of -1.

    !!! note
        Since one element can now accommodate multiple orders, you may run into "order_records index out of range"
        exception. In this case, you must increase `max_orders`. This cannot be done automatically and
        dynamically to avoid performance degradation.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_group_out = pre_group_func_nb(GroupContext, *pre_sim_out, *pre_group_args)
                3. if call_pre_segment or segment_mask: pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_group_out, *pre_segment_args)
                    while col != -1:
                        4. if segment_mask: col, order = flex_order_func_nb(FlexOrderContext, *pre_segment_out, *flex_order_args)
                        5. if order: post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                        ...
                6. if call_post_segment or segment_mask: post_segment_func_nb(SegmentContext, *pre_group_out, *post_segment_args)
                ...
            7. post_group_func_nb(GroupContext, *pre_sim_out, *post_group_args)
            ...
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate the same example as in `simulate_nb` but adapted for this function:

        ![](/assets/images/api/flex_simulate_nb.svg)

    Usage:
        * The same example as in `simulate_nb`:

        ```pycon
        >>> import numpy as np
        >>> from numba import njit
        >>> from vectorbtpro.portfolio.enums import SizeType, Direction
        >>> from vectorbtpro.portfolio.nb import (
        ...     select_from_col_nb,
        ...     order_nb,
        ...     order_nothing_nb,
        ...     flex_simulate_nb,
        ...     flex_simulate_row_wise_nb,
        ...     sort_call_seq_out_nb
        ... )

        >>> @njit
        ... def pre_sim_func_nb(c):
        ...     print('before simulation')
        ...     return ()

        >>> @njit
        ... def pre_group_func_nb(c):
        ...     print('\\tbefore group', c.group)
        ...     # Create temporary arrays and pass them down the stack
        ...     order_value_out = np.empty(c.group_len, dtype=np.float_)
        ...     call_seq_out = np.empty(c.group_len, dtype=np.int_)
        ...     # Forward down the stack
        ...     return (order_value_out, call_seq_out)

        >>> @njit
        ... def pre_segment_func_nb(c, order_value_out, call_seq_out, size, price, size_type, direction):
        ...     print('\\t\\tbefore segment', c.i)
        ...     for col in range(c.from_col, c.to_col):
        ...         # Here we use order price for group valuation
        ...         c.last_val_price[col] = select_from_col_nb(c, col, price)
        ...
        ...     # Same as for simulate_nb, but since we don't have a predefined c.call_seq_now anymore,
        ...     # we need to store our new call sequence somewhere else
        ...     call_seq_out[:] = np.arange(c.group_len)
        ...     sort_call_seq_out_nb(c, size, size_type, direction, order_value_out, call_seq_out)
        ...
        ...     # Forward the sorted call sequence
        ...     return (call_seq_out,)

        >>> @njit
        ... def flex_order_func_nb(c, call_seq_out, size, price, size_type, direction, fees, fixed_fees, slippage):
        ...     if c.call_idx < c.group_len:
        ...         col = c.from_col + call_seq_out[c.call_idx]
        ...         print('\\t\\t\\tcreating order', c.call_idx, 'at column', col)
        ...         # # Create and return an order
        ...         return col, order_nb(
        ...             size=select_from_col_nb(c, col, size),
        ...             price=select_from_col_nb(c, col, price),
        ...             size_type=select_from_col_nb(c, col, size_type),
        ...             direction=select_from_col_nb(c, col, direction),
        ...             fees=select_from_col_nb(c, col, fees),
        ...             fixed_fees=select_from_col_nb(c, col, fixed_fees),
        ...             slippage=select_from_col_nb(c, col, slippage)
        ...         )
        ...     # All columns already processed -> break the loop
        ...     print('\\t\\t\\tbreaking out of the loop')
        ...     return -1, order_nothing_nb()

        >>> @njit
        ... def post_order_func_nb(c, call_seq_out):
        ...     print('\\t\\t\\t\\torder status:', c.order_result.status)
        ...     return None

        >>> @njit
        ... def post_segment_func_nb(c, order_value_out, call_seq_out):
        ...     print('\\t\\tafter segment', c.i)
        ...     return None

        >>> @njit
        ... def post_group_func_nb(c):
        ...     print('\\tafter group', c.group)
        ...     return None

        >>> @njit
        ... def post_sim_func_nb(c):
        ...     print('after simulation')
        ...     return None

        >>> target_shape = (5, 3)
        >>> np.random.seed(42)
        >>> group_lens = np.array([3])  # one group of three columns
        >>> cash_sharing = True
        >>> call_seq = build_call_seq(target_shape, group_lens)  # will be overridden
        >>> segment_mask = np.array([True, False, True, False, True])[:, None]
        >>> segment_mask = np.copy(np.broadcast_to(segment_mask, target_shape))
        >>> size = np.asarray(1 / target_shape[1])  # scalars must become 0-dim arrays
        >>> price = close = np.random.uniform(1, 10, size=target_shape)
        >>> size_type = np.asarray(SizeType.TargetPercent)
        >>> direction = np.asarray(Direction.LongOnly)
        >>> fees = np.asarray(0.001)
        >>> fixed_fees = np.asarray(1.)
        >>> slippage = np.asarray(0.001)

        >>> sim_out = flex_simulate_nb(
        ...     target_shape,
        ...     group_lens,
        ...     cash_sharing,
        ...     segment_mask=segment_mask,
        ...     pre_sim_func_nb=pre_sim_func_nb,
        ...     post_sim_func_nb=post_sim_func_nb,
        ...     pre_group_func_nb=pre_group_func_nb,
        ...     post_group_func_nb=post_group_func_nb,
        ...     pre_segment_func_nb=pre_segment_func_nb,
        ...     pre_segment_args=(size, price, size_type, direction),
        ...     post_segment_func_nb=post_segment_func_nb,
        ...     flex_order_func_nb=flex_order_func_nb,
        ...     flex_order_args=(size, price, size_type, direction, fees, fixed_fees, slippage),
        ...     post_order_func_nb=post_order_func_nb
        ... )
        before simulation
            before group 0
                before segment 0
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 1
                        order status: 0
                    creating order 2 at column 2
                        order status: 0
                    breaking out of the loop
                after segment 0
                before segment 2
                    creating order 0 at column 1
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 0
                        order status: 0
                    breaking out of the loop
                after segment 2
                before segment 4
                    creating order 0 at column 0
                        order status: 0
                    creating order 1 at column 2
                        order status: 0
                    creating order 2 at column 1
                        order status: 0
                    breaking out of the loop
                after segment 4
            after group 0
        after simulation
        ```
    """

    check_group_lens_nb(group_lens, target_shape[1])

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
    last_pos_record = prepare_last_pos_record_nb(
        target_shape,
        init_position=init_position,
        init_price=init_price,
        fill_pos_record=fill_pos_record,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        group_len = to_col - from_col

        # Call function before the group
        pre_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        pre_group_out = pre_group_func_nb(pre_group_ctx, *pre_sim_out, *pre_group_args)

        for i in range(target_shape[0]):

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(prev_close_value[group], last_value[group])
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(prev_close_value[col], last_value[col])

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_group_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_auto_nb(cash_deposits, i, col, flex_2d)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if is_segment_active:

                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx,
                    )
                    col, order = flex_order_func_nb(flex_order_ctx, *pre_segment_out, *flex_order_args)

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column out of bounds of the group")
                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError("Cannot use size type that depends on not tracked value")

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(prev_close_value[col], value_now - cash_deposits_now)

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_record:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_record_nb(
                                pos_record_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function before the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                post_segment_func_nb(post_seg_ctx, *pre_group_out, *post_segment_args)

        # Call function after the group
        post_group_ctx = GroupContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            group=group,
            group_len=group_len,
            from_col=from_col,
            to_col=to_col,
        )
        post_group_func_nb(post_group_ctx, *pre_sim_out, *post_group_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=None,
        in_outputs=in_outputs,
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1, mapper=base_ch.group_lens_mapper),
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        init_cash=RepFunc(portfolio_ch.get_init_cash_slicer),
        init_position=portfolio_ch.flex_1d_array_gl_slicer,
        init_price=portfolio_ch.flex_1d_array_gl_slicer,
        cash_deposits=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        cash_earnings=portfolio_ch.flex_array_gl_slicer,
        segment_mask=base_ch.FlexArraySlicer(axis=1),
        call_pre_segment=None,
        call_post_segment=None,
        pre_sim_func_nb=None,
        pre_sim_args=ch.ArgsTaker(),
        post_sim_func_nb=None,
        post_sim_args=ch.ArgsTaker(),
        pre_row_func_nb=None,
        pre_row_args=ch.ArgsTaker(),
        post_row_func_nb=None,
        post_row_args=ch.ArgsTaker(),
        pre_segment_func_nb=None,
        pre_segment_args=ch.ArgsTaker(),
        post_segment_func_nb=None,
        post_segment_args=ch.ArgsTaker(),
        flex_order_func_nb=None,
        flex_order_args=ch.ArgsTaker(),
        post_order_func_nb=None,
        post_order_args=ch.ArgsTaker(),
        index=None,
        freq=None,
        open=portfolio_ch.flex_array_gl_slicer,
        high=portfolio_ch.flex_array_gl_slicer,
        low=portfolio_ch.flex_array_gl_slicer,
        close=portfolio_ch.flex_array_gl_slicer,
        bm_close=portfolio_ch.flex_array_gl_slicer,
        ffill_val_price=None,
        update_value=None,
        fill_pos_record=None,
        track_value=None,
        max_orders=None,
        max_logs=None,
        flex_2d=None,
        in_outputs=ch.ArgsTaker(),
    ),
    **portfolio_ch.merge_sim_outs_config
)
@register_jitted
def flex_simulate_row_wise_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray = np.asarray(100.0),
    init_position: tp.FlexArray = np.asarray(0.0),
    init_price: tp.FlexArray = np.asarray(np.nan),
    cash_deposits: tp.FlexArray = np.asarray(0.0),
    cash_earnings: tp.FlexArray = np.asarray(0.0),
    segment_mask: tp.FlexArray = np.asarray(True),
    call_pre_segment: bool = False,
    call_post_segment: bool = False,
    pre_sim_func_nb: PreSimFuncT = no_pre_func_nb,
    pre_sim_args: tp.Args = (),
    post_sim_func_nb: PostSimFuncT = no_post_func_nb,
    post_sim_args: tp.Args = (),
    pre_row_func_nb: PreRowFuncT = no_pre_func_nb,
    pre_row_args: tp.Args = (),
    post_row_func_nb: PostRowFuncT = no_post_func_nb,
    post_row_args: tp.Args = (),
    pre_segment_func_nb: PreSegmentFuncT = no_pre_func_nb,
    pre_segment_args: tp.Args = (),
    post_segment_func_nb: PostSegmentFuncT = no_post_func_nb,
    post_segment_args: tp.Args = (),
    flex_order_func_nb: FlexOrderFuncT = no_flex_order_func_nb,
    flex_order_args: tp.Args = (),
    post_order_func_nb: PostOrderFuncT = no_post_func_nb,
    post_order_args: tp.Args = (),
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
    open: tp.FlexArray = np.asarray(np.nan),
    high: tp.FlexArray = np.asarray(np.nan),
    low: tp.FlexArray = np.asarray(np.nan),
    close: tp.FlexArray = np.asarray(np.nan),
    bm_close: tp.FlexArray = np.asarray(np.nan),
    ffill_val_price: bool = True,
    update_value: bool = False,
    fill_pos_record: bool = True,
    track_value: bool = True,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
    flex_2d: bool = False,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Same as `flex_simulate_nb`, but iterates using row-major order, with the rows
    changing fastest, and the columns/groups changing slowest.

    Call hierarchy:
        ```plaintext
        1. pre_sim_out = pre_sim_func_nb(SimulationContext, *pre_sim_args)
            2. pre_row_out = pre_row_func_nb(RowContext, *pre_sim_out, *pre_row_args)
                3. if call_pre_segment or segment_mask: pre_segment_out = pre_segment_func_nb(SegmentContext, *pre_row_out, *pre_segment_args)
                    while col != -1:
                        4. if segment_mask: col, order = flex_order_func_nb(FlexOrderContext, *pre_segment_out, *flex_order_args)
                        5. if order: post_order_func_nb(PostOrderContext, *pre_segment_out, *post_order_args)
                        ...
                6. if call_post_segment or segment_mask: post_segment_func_nb(SegmentContext, *pre_row_out, *post_segment_args)
                ...
            7. post_row_func_nb(RowContext, *pre_sim_out, *post_row_args)
            ...
        8. post_sim_func_nb(SimulationContext, *post_sim_args)
        ```

        Let's illustrate the same example as in `simulate_nb` but adapted for this function:

        ![](/assets/images/api/flex_simulate_row_wise_nb.svg)
    """

    check_group_lens_nb(group_lens, target_shape[1])

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
    last_pos_record = prepare_last_pos_record_nb(
        target_shape,
        init_position=init_position,
        init_price=init_price,
        fill_pos_record=fill_pos_record,
    )

    last_cash_deposits = np.full_like(last_cash, 0.0)
    last_val_price = np.full_like(last_position, np.nan)
    last_debt = np.full_like(last_position, 0.0)
    last_free_cash = last_cash.copy()
    prev_close_value = last_value.copy()
    last_return = np.full_like(last_cash, np.nan)
    order_counts = np.full(target_shape[1], 0, dtype=np.int_)
    log_counts = np.full(target_shape[1], 0, dtype=np.int_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens

    # Call function before the simulation
    pre_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    pre_sim_out = pre_sim_func_nb(pre_sim_ctx, *pre_sim_args)

    for i in range(target_shape[0]):

        # Call function before the row
        pre_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            i=i,
        )
        pre_row_out = pre_row_func_nb(pre_row_ctx, *pre_sim_out, *pre_row_args)

        for group in range(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]
            group_len = to_col - from_col

            if track_value:
                # Update valuation price using current open
                for col in range(from_col, to_col):
                    _open = flex_select_auto_nb(open, i, col, flex_2d)
                    if not np.isnan(_open) or not ffill_val_price:
                        last_val_price[col] = _open

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(prev_close_value[group], last_value[group])
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(prev_close_value[col], last_value[col])

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if call_pre_segment or is_segment_active:
                # Call function before the segment
                pre_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                pre_segment_out = pre_segment_func_nb(pre_seg_ctx, *pre_row_out, *pre_segment_args)

            # Add cash
            if cash_sharing:
                _cash_deposits = flex_select_auto_nb(cash_deposits, i, group, flex_2d)
                last_cash[group] += _cash_deposits
                last_free_cash[group] += _cash_deposits
                last_cash_deposits[group] = _cash_deposits
            else:
                for col in range(from_col, to_col):
                    _cash_deposits = flex_select_auto_nb(cash_deposits, i, col, flex_2d)
                    last_cash[col] += _cash_deposits
                    last_free_cash[col] += _cash_deposits
                    last_cash_deposits[col] = _cash_deposits

            if track_value:
                # Update value and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            is_segment_active = flex_select_auto_nb(segment_mask, i, group, flex_2d)
            if is_segment_active:

                call_idx = -1
                while True:
                    call_idx += 1

                    # Generate the next order
                    flex_order_ctx = FlexOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        call_idx=call_idx,
                    )
                    col, order = flex_order_func_nb(flex_order_ctx, *pre_segment_out, *flex_order_args)

                    if col == -1:
                        break
                    if col < from_col or col >= to_col:
                        raise ValueError("Column out of bounds of the group")
                    if not track_value:
                        if (
                            order.size_type == SizeType.Value
                            or order.size_type == SizeType.TargetValue
                            or order.size_type == SizeType.TargetPercent
                        ):
                            raise ValueError("Cannot use size type that depends on not tracked value")

                    # Get current values
                    position_now = last_position[col]
                    debt_now = last_debt[col]
                    val_price_now = last_val_price[col]
                    pos_record_now = last_pos_record[col]
                    if cash_sharing:
                        cash_now = last_cash[group]
                        free_cash_now = last_free_cash[group]
                        value_now = last_value[group]
                        return_now = last_return[group]
                        cash_deposits_now = last_cash_deposits[group]
                    else:
                        cash_now = last_cash[col]
                        free_cash_now = last_free_cash[col]
                        value_now = last_value[col]
                        return_now = last_return[col]
                        cash_deposits_now = last_cash_deposits[col]

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

                    if track_value:
                        val_price_now = new_exec_state.val_price
                        value_now = new_exec_state.value
                        if cash_sharing:
                            return_now = returns_nb_.get_return_nb(
                                prev_close_value[group],
                                value_now - cash_deposits_now,
                            )
                        else:
                            return_now = returns_nb_.get_return_nb(prev_close_value[col], value_now - cash_deposits_now)

                    # Now becomes last
                    last_position[col] = position_now
                    last_debt[col] = debt_now
                    if not np.isnan(val_price_now) or not ffill_val_price:
                        last_val_price[col] = val_price_now
                    if cash_sharing:
                        last_cash[group] = cash_now
                        last_free_cash[group] = free_cash_now
                        last_value[group] = value_now
                        last_return[group] = return_now
                    else:
                        last_cash[col] = cash_now
                        last_free_cash[col] = free_cash_now
                        last_value[col] = value_now
                        last_return[col] = return_now

                    if track_value:
                        if not np.isnan(val_price_now) or not ffill_val_price:
                            last_val_price[col] = val_price_now
                        if cash_sharing:
                            last_value[group] = value_now
                            last_return[group] = return_now
                        else:
                            last_value[col] = value_now
                            last_return[col] = return_now

                    # Update position record
                    if fill_pos_record:
                        if order_result.status == OrderStatus.Filled:
                            if order_counts[col] > 0:
                                order_id = order_records["id"][order_counts[col] - 1, col]
                            else:
                                order_id = -1
                            update_pos_record_nb(
                                pos_record_now,
                                i,
                                col,
                                exec_state.position,
                                position_now,
                                order_result,
                                order_id,
                            )

                    # Post-order callback
                    post_order_ctx = PostOrderContext(
                        target_shape=target_shape,
                        group_lens=group_lens,
                        cash_sharing=cash_sharing,
                        call_seq=None,
                        init_cash=init_cash,
                        init_position=init_position,
                        init_price=init_price,
                        cash_deposits=cash_deposits,
                        cash_earnings=cash_earnings,
                        segment_mask=segment_mask,
                        call_pre_segment=call_pre_segment,
                        call_post_segment=call_post_segment,
                        index=index,
                        freq=freq,
                        open=open,
                        high=high,
                        low=low,
                        close=close,
                        bm_close=bm_close,
                        ffill_val_price=ffill_val_price,
                        update_value=update_value,
                        fill_pos_record=fill_pos_record,
                        track_value=track_value,
                        flex_2d=flex_2d,
                        order_records=order_records,
                        log_records=log_records,
                        in_outputs=in_outputs,
                        last_cash=last_cash,
                        last_position=last_position,
                        last_debt=last_debt,
                        last_free_cash=last_free_cash,
                        last_val_price=last_val_price,
                        last_value=last_value,
                        last_return=last_return,
                        order_counts=order_counts,
                        log_counts=log_counts,
                        last_pos_record=last_pos_record,
                        group=group,
                        group_len=group_len,
                        from_col=from_col,
                        to_col=to_col,
                        i=i,
                        call_seq_now=None,
                        col=col,
                        call_idx=call_idx,
                        cash_before=exec_state.cash,
                        position_before=exec_state.position,
                        debt_before=exec_state.debt,
                        free_cash_before=exec_state.free_cash,
                        val_price_before=exec_state.val_price,
                        value_before=exec_state.value,
                        order_result=order_result,
                        cash_now=cash_now,
                        position_now=position_now,
                        debt_now=debt_now,
                        free_cash_now=free_cash_now,
                        val_price_now=val_price_now,
                        value_now=value_now,
                        return_now=return_now,
                        pos_record_now=pos_record_now,
                    )
                    post_order_func_nb(post_order_ctx, *pre_segment_out, *post_order_args)

            # NOTE: Regardless of segment_mask, we still need to update stats to be accessed by future rows
            # Add earnings in cash
            for col in range(from_col, to_col):
                _cash_earnings = flex_select_auto_nb(cash_earnings, i, col, flex_2d)
                if cash_sharing:
                    last_cash[group] += _cash_earnings
                    last_free_cash[group] += _cash_earnings
                else:
                    last_cash[col] += _cash_earnings
                    last_free_cash[col] += _cash_earnings

            if track_value:
                # Update valuation price using current close
                for col in range(from_col, to_col):
                    _close = flex_select_auto_nb(close, i, col, flex_2d)
                    if not np.isnan(_close) or not ffill_val_price:
                        last_val_price[col] = _close

                # Update previous value, current value, and return
                if cash_sharing:
                    last_value[group] = get_group_value_nb(
                        from_col,
                        to_col,
                        last_cash[group],
                        last_position,
                        last_val_price,
                    )
                    last_return[group] = returns_nb_.get_return_nb(
                        prev_close_value[group],
                        last_value[group] - last_cash_deposits[group],
                    )
                    prev_close_value[group] = last_value[group]
                else:
                    for col in range(from_col, to_col):
                        if last_position[col] == 0:
                            last_value[col] = last_cash[col]
                        else:
                            last_value[col] = last_cash[col] + last_position[col] * last_val_price[col]
                        last_return[col] = returns_nb_.get_return_nb(
                            prev_close_value[col],
                            last_value[col] - last_cash_deposits[col],
                        )
                        prev_close_value[col] = last_value[col]

                # Update open position stats
                if fill_pos_record:
                    for col in range(from_col, to_col):
                        update_open_pos_stats_nb(last_pos_record[col], last_position[col], last_val_price[col])

            # Is this segment active?
            if call_post_segment or is_segment_active:
                # Call function after the segment
                post_seg_ctx = SegmentContext(
                    target_shape=target_shape,
                    group_lens=group_lens,
                    cash_sharing=cash_sharing,
                    call_seq=None,
                    init_cash=init_cash,
                    init_position=init_position,
                    init_price=init_price,
                    cash_deposits=cash_deposits,
                    cash_earnings=cash_earnings,
                    segment_mask=segment_mask,
                    call_pre_segment=call_pre_segment,
                    call_post_segment=call_post_segment,
                    index=index,
                    freq=freq,
                    open=open,
                    high=high,
                    low=low,
                    close=close,
                    bm_close=bm_close,
                    ffill_val_price=ffill_val_price,
                    update_value=update_value,
                    fill_pos_record=fill_pos_record,
                    track_value=track_value,
                    flex_2d=flex_2d,
                    order_records=order_records,
                    log_records=log_records,
                    in_outputs=in_outputs,
                    last_cash=last_cash,
                    last_position=last_position,
                    last_debt=last_debt,
                    last_free_cash=last_free_cash,
                    last_val_price=last_val_price,
                    last_value=last_value,
                    last_return=last_return,
                    order_counts=order_counts,
                    log_counts=log_counts,
                    last_pos_record=last_pos_record,
                    group=group,
                    group_len=group_len,
                    from_col=from_col,
                    to_col=to_col,
                    i=i,
                    call_seq_now=None,
                )
                post_segment_func_nb(post_seg_ctx, *pre_row_out, *post_segment_args)

        # Call function after the row
        post_row_ctx = RowContext(
            target_shape=target_shape,
            group_lens=group_lens,
            cash_sharing=cash_sharing,
            call_seq=None,
            init_cash=init_cash,
            init_position=init_position,
            init_price=init_price,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            segment_mask=segment_mask,
            call_pre_segment=call_pre_segment,
            call_post_segment=call_post_segment,
            index=index,
            freq=freq,
            open=open,
            high=high,
            low=low,
            close=close,
            bm_close=bm_close,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            fill_pos_record=fill_pos_record,
            track_value=track_value,
            flex_2d=flex_2d,
            order_records=order_records,
            log_records=log_records,
            in_outputs=in_outputs,
            last_cash=last_cash,
            last_position=last_position,
            last_debt=last_debt,
            last_free_cash=last_free_cash,
            last_val_price=last_val_price,
            last_value=last_value,
            last_return=last_return,
            order_counts=order_counts,
            log_counts=log_counts,
            last_pos_record=last_pos_record,
            i=i,
        )
        post_row_func_nb(post_row_ctx, *pre_sim_out, *post_row_args)

    # Call function after the simulation
    post_sim_ctx = SimulationContext(
        target_shape=target_shape,
        group_lens=group_lens,
        cash_sharing=cash_sharing,
        call_seq=None,
        init_cash=init_cash,
        init_position=init_position,
        init_price=init_price,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        segment_mask=segment_mask,
        call_pre_segment=call_pre_segment,
        call_post_segment=call_post_segment,
        index=index,
        freq=freq,
        open=open,
        high=high,
        low=low,
        close=close,
        bm_close=bm_close,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        fill_pos_record=fill_pos_record,
        track_value=track_value,
        flex_2d=flex_2d,
        order_records=order_records,
        log_records=log_records,
        in_outputs=in_outputs,
        last_cash=last_cash,
        last_position=last_position,
        last_debt=last_debt,
        last_free_cash=last_free_cash,
        last_val_price=last_val_price,
        last_value=last_value,
        last_return=last_return,
        order_counts=order_counts,
        log_counts=log_counts,
        last_pos_record=last_pos_record,
    )
    post_sim_func_nb(post_sim_ctx, *post_sim_args)

    return prepare_simout_nb(
        order_records=order_records,
        order_counts=order_counts,
        log_records=log_records,
        log_counts=log_counts,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=None,
        in_outputs=in_outputs,
    )
