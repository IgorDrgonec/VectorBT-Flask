# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Core Numba-compiled functions for portfolio modeling."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_1d_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.math_ import is_close_nb, is_close_or_less_nb, is_less_nb, add_nb


@register_jitted(cache=True)
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """Return `OrderResult` for order that hasn't been filled."""
    return OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=status, status_info=status_info)


@register_jitted(cache=True)
def check_adj_price_nb(
    adj_price: float,
    price_area: PriceArea,
    is_closing_price: bool,
    price_area_vio_mode: int,
) -> float:
    """Check whether adjusted price is within price boundaries."""
    if price_area_vio_mode == PriceAreaVioMode.Ignore:
        return adj_price
    if adj_price > price_area.high:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is above the highest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.high
    if adj_price < price_area.low:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is below the lowest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.low
    if is_closing_price and adj_price != price_area.close:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is beyond the closing price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.close
    return adj_price


@register_jitted(cache=True)
def buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    lock_cash: bool = False,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[AccountState, OrderResult]:
    """Buy or/and cover."""

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Set cash limit
    if lock_cash:
        if account_state.position >= 0:
            # cash == free_cash in a long position, unless other column(s) locked some of the cash
            cash_limit = account_state.free_cash
        else:
            # How much free cash remains after closing out the short position?
            cover_req_cash = abs(account_state.position) * adj_price * (1 + fees) + fixed_fees
            cover_free_cash = add_nb(account_state.free_cash + 2 * account_state.debt, -cover_req_cash)
            if cover_free_cash > 0:
                # Enough cash to close out the short position and open a long one
                cash_limit = account_state.free_cash + 2 * account_state.debt
            elif cover_free_cash < 0:
                # Not enough cash to close out the short position
                avg_entry_price = account_state.debt / abs(account_state.position)
                max_short_size = (account_state.free_cash - fixed_fees) / (adj_price * (1 + fees) - 2 * avg_entry_price)
                cash_limit = max_short_size * adj_price * (1 + fees) + fixed_fees
            else:
                # Exact amount of cash to close out the short position
                cash_limit = account_state.cash
    else:
        cash_limit = account_state.cash
    cash_limit = min(cash_limit, account_state.cash)
    if not np.isnan(percent):
        # Apply percentage
        cash_limit = min(cash_limit, percent * cash_limit)

    if direction == Direction.ShortOnly:
        if account_state.position == 0:
            return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition)
    if cash_limit == 0:
        return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCashLong)

    # Get optimal order size
    if direction == Direction.ShortOnly:
        adj_size = min(-account_state.position, size)
    else:
        adj_size = size
    if np.isinf(adj_size) and np.isinf(cash_limit):
        raise ValueError("Attempt to go in long direction infinitely")

    if not np.isnan(max_size) and adj_size > max_size:
        if not allow_partial:
            return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded)

        adj_size = max_size

    # Adjust granularity
    if not np.isnan(size_granularity):
        adj_size = adj_size // size_granularity * size_granularity

    # Get cash required to complete this order
    if np.isinf(adj_size):
        total_req_cash = np.inf
        req_fees = np.inf
    else:
        req_cash = adj_size * adj_price
        req_fees = req_cash * fees + fixed_fees
        total_req_cash = req_cash + req_fees

    if is_close_or_less_nb(total_req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = adj_size
        fees_paid = req_fees
        final_req_cash = total_req_cash
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

        max_acq_size = max_req_cash / adj_price

        if not np.isnan(size_granularity):
            # Adjust granularity
            final_size = max_acq_size // size_granularity * size_granularity
            new_req_cash = final_size * adj_price
            fees_paid = new_req_cash * fees + fixed_fees
            final_req_cash = new_req_cash + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            final_req_cash = cash_limit

    # Check size of zero
    if is_close_nb(final_size, 0):
        return account_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return account_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached)

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill)

    # Update current cash balance and position
    new_cash = add_nb(account_state.cash, -final_req_cash)
    new_position = add_nb(account_state.position, final_size)

    # Update current debt and free cash
    if account_state.position < 0:
        if new_position < 0:
            short_size = final_size
        else:
            short_size = abs(account_state.position)
        avg_entry_price = account_state.debt / abs(account_state.position)
        debt_diff = short_size * avg_entry_price
        new_debt = add_nb(account_state.debt, -debt_diff)
        new_free_cash = add_nb(account_state.free_cash + 2 * debt_diff, -final_req_cash)
    else:
        new_debt = account_state.debt
        new_free_cash = add_nb(account_state.free_cash, -final_req_cash)

    # Return filled order
    order_result = OrderResult(final_size, adj_price, fees_paid, OrderSide.Buy, OrderStatus.Filled, -1)
    new_account_state = AccountState(cash=new_cash, position=new_position, debt=new_debt, free_cash=new_free_cash)
    return new_account_state, order_result


@register_jitted(cache=True)
def sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    lock_cash: bool = False,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[AccountState, OrderResult]:
    """Sell or/and short sell."""

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get optimal order size
    if direction == Direction.LongOnly:
        size_limit = min(account_state.position, size)
    else:
        if lock_cash or (np.isinf(size) and not np.isnan(percent)):
            # Get the maximum size that can be (short) sold
            long_size = max(account_state.position, 0)
            long_cash = long_size * adj_price * (1 - fees)
            total_free_cash = add_nb(account_state.free_cash, long_cash)

            if total_free_cash <= 0:
                if account_state.position <= 0:
                    # There is nothing to sell, and no free cash to short sell
                    return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCashShort)

                # There is position to close, but no free cash to short sell
                max_size_limit = long_size
            else:
                # There is position to close and/or free cash to short sell
                max_short_size = add_nb(total_free_cash, -fixed_fees) / (adj_price * (1 + fees))
                max_size_limit = add_nb(long_size, max_short_size)
                if max_size_limit <= 0:
                    return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

            if lock_cash:
                # Size has upper limit
                if np.isinf(size) and not np.isnan(percent):
                    size_limit = min(percent * max_size_limit, max_size_limit)
                    percent = np.nan
                elif not np.isnan(percent):
                    size_limit = min(percent * size, max_size_limit)
                    percent = np.nan
                else:
                    size_limit = min(size, max_size_limit)
            else:  # np.isinf(size) and not np.isnan(percent)
                # Size has no upper limit
                size_limit = max_size_limit
        else:
            size_limit = size

    if not np.isnan(percent):
        # Apply percentage
        size_limit = percent * size_limit

    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded)

        size_limit = max_size

    if direction == Direction.LongOnly:
        if account_state.position == 0:
            return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition)
    if np.isinf(size_limit):
        raise ValueError("Attempt to go in short direction infinitely")

    # Adjust granularity
    if not np.isnan(size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Check size of zero
    if is_close_nb(size_limit, 0):
        return account_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero)

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return account_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached)

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill)

    # Get acquired cash
    acq_cash = size_limit * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(acq_cash, -fees_paid)
    if final_acq_cash < 0 and is_less_nb(account_state.cash, -final_acq_cash):
        return account_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees)

    # Update current cash balance and position
    new_cash = account_state.cash + final_acq_cash
    new_position = add_nb(account_state.position, -size_limit)

    # Update current debt and free cash
    if new_position < 0:
        if account_state.position < 0:
            short_size = size_limit
        else:
            short_size = abs(new_position)
        short_value = short_size * adj_price
        new_debt = account_state.debt + short_value
        free_cash_diff = add_nb(final_acq_cash, -2 * short_value)
        new_free_cash = add_nb(account_state.free_cash, free_cash_diff)
    else:
        new_debt = account_state.debt
        new_free_cash = account_state.free_cash + final_acq_cash

    # Return filled order
    order_result = OrderResult(size_limit, adj_price, fees_paid, OrderSide.Sell, OrderStatus.Filled, -1)
    new_account_state = AccountState(cash=new_cash, position=new_position, debt=new_debt, free_cash=new_free_cash)
    return new_account_state, order_result


@register_jitted(cache=True)
def update_value_nb(
    cash_before: float,
    cash_now: float,
    position_before: float,
    position_now: float,
    val_price_before: float,
    price: float,
    value_before: float,
) -> tp.Tuple[float, float]:
    """Update valuation price and value."""
    val_price_now = price
    cash_flow = cash_now - cash_before
    if position_before != 0:
        asset_value_before = position_before * val_price_before
    else:
        asset_value_before = 0.0
    if position_now != 0:
        asset_value_now = position_now * val_price_now
    else:
        asset_value_now = 0.0
    asset_value_diff = asset_value_now - asset_value_before
    value_now = value_before + cash_flow + asset_value_diff
    return val_price_now, value_now


@register_jitted(cache=True)
def get_diraware_size_nb(size: float, direction: int) -> float:
    """Get direction-aware size."""
    if direction == Direction.ShortOnly:
        return size * -1
    return size


@register_jitted(cache=True)
def resolve_size_nb(
    size: float,
    size_type: int,
    direction: int,
    position: float,
    val_price: float,
    value: float,
    as_requirement: bool = False,
) -> tp.Tuple[float, float]:
    """Resolve size into an absolute amount of assets and percentage of resources.

    Percentage is only set if the option `SizeType.Percent(100)` is used."""
    if size_type == SizeType.ValuePercent100:
        size /= 100
        size_type = SizeType.ValuePercent
    if size_type == SizeType.TargetPercent100:
        size /= 100
        size_type = SizeType.TargetPercent
    if size_type == SizeType.ValuePercent or size_type == SizeType.TargetPercent:
        # Percentage or target percentage of the current value
        size *= value
        if size_type == SizeType.ValuePercent:
            size_type = SizeType.Value
        else:
            size_type = SizeType.TargetValue

    if size_type == SizeType.Value or size_type == SizeType.TargetValue:
        # Value or target value
        size /= val_price
        if size_type == SizeType.Value:
            size_type = SizeType.Amount
        else:
            size_type = SizeType.TargetAmount

    if size_type == SizeType.TargetAmount:
        # Target amount
        if not as_requirement:
            size -= position
        size_type = SizeType.Amount

    if size_type == SizeType.Amount:
        if direction == Direction.ShortOnly or direction == Direction.Both:
            if size < 0 and np.isinf(size):
                # Infinite negative size has a special meaning: 100% to short
                size = -1.0
                size_type = SizeType.Percent

    percent = np.nan
    if size_type == SizeType.Percent100:
        size /= 100
        size_type = SizeType.Percent
    if size_type == SizeType.Percent:
        # Percentage of resources
        percent = abs(size)
        size = np.sign(size) * np.inf

    if as_requirement:
        size = abs(size)
    return size, percent


@register_jitted(cache=True)
def execute_order_nb(
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
) -> tp.Tuple[ExecState, OrderResult]:
    """Execute an order given the current state.

    Args:
        exec_state (ExecState): See `vectorbtpro.portfolio.enums.ExecState`.
        order (Order): See `vectorbtpro.portfolio.enums.Order`.
        price_area (OrderPriceArea): See `vectorbtpro.portfolio.enums.PriceArea`.
        update_value (bool): Whether to update the value.

    Error is thrown if an input has value that is not expected.
    Order is ignored if its execution has no effect on the current balance.
    Order is rejected if an input goes over a limit or against a restriction.
    """
    # numerical stability
    cash = exec_state.cash
    if is_close_nb(cash, 0):
        cash = 0.0
    position = exec_state.position
    if is_close_nb(position, 0):
        position = 0.0
    debt = exec_state.debt
    if is_close_nb(debt, 0):
        debt = 0.0
    free_cash = exec_state.free_cash
    if is_close_nb(free_cash, 0):
        free_cash = 0.0
    val_price = exec_state.val_price
    if is_close_nb(val_price, 0):
        val_price = 0.0
    value = exec_state.value
    if is_close_nb(value, 0):
        value = 0.0

    # Pre-fill account state
    account_state = AccountState(cash=cash, position=position, debt=debt, free_cash=free_cash)

    # Check price area
    if np.isinf(price_area.open) or price_area.open < 0:
        raise ValueError("price_area.open must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.high) or price_area.high < 0:
        raise ValueError("price_area.high must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.low) or price_area.low < 0:
        raise ValueError("price_area.low must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.close) or price_area.close < 0:
        raise ValueError("price_area.close must be either NaN, or finite and 0 or greater")

    # Resolve price
    order_price = order.price
    is_closing_price = False
    if np.isinf(order_price):
        if order_price > 0:
            order_price = price_area.close
            is_closing_price = True
        else:
            order_price = price_area.open

    # Ignore order if size or price is nan
    if np.isnan(order.size):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeNaN)
    if np.isnan(order_price):
        return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.PriceNaN)

    # Check account state
    if np.isnan(cash) or cash < 0:
        raise ValueError("exec_state.cash cannot be NaN and must be greater than 0")
    if not np.isfinite(position):
        raise ValueError("exec_state.position must be finite")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("exec_state.debt must be finite and 0 or greater")
    if np.isnan(free_cash):
        raise ValueError("exec_state.free_cash cannot be NaN")

    # Check order
    if not np.isfinite(order_price) or order_price < 0:
        raise ValueError("order.price must be finite and 0 or greater")
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("order.size_type is invalid")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("order.direction is invalid")
    if not np.isfinite(order.fees):
        raise ValueError("order.fees must be finite")
    if not np.isfinite(order.fixed_fees):
        raise ValueError("order.fixed_fees must be finite")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("order.slippage must be finite and 0 or greater")
    if np.isinf(order.min_size) or order.min_size < 0:
        raise ValueError("order.min_size must be either NaN, 0, or greater")
    if order.max_size <= 0:
        raise ValueError("order.max_size must be either NaN or greater than 0")
    if np.isinf(order.size_granularity) or order.size_granularity <= 0:
        raise ValueError("order.size_granularity must be either NaN, or finite and greater than 0")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("order.reject_prob must be between 0 and 1")

    # Positive/negative size in short direction should be treated as negative/positive
    order_size = get_diraware_size_nb(order.size, order.direction)
    min_order_size = order.min_size
    max_order_size = order.max_size
    order_size_type = order.size_type

    if (
        order_size_type == SizeType.ValuePercent100
        or order_size_type == SizeType.ValuePercent
        or order_size_type == SizeType.TargetPercent100
        or order_size_type == SizeType.TargetPercent
        or order_size_type == SizeType.Value
        or order_size_type == SizeType.TargetValue
    ):
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price):
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValPriceNaN)
        if (
            order_size_type == SizeType.ValuePercent100
            or order_size_type == SizeType.ValuePercent
            or order_size_type == SizeType.TargetPercent100
            or order_size_type == SizeType.TargetPercent
        ):
            if np.isnan(value):
                return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValueNaN)
            if value <= 0:
                return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.ValueZeroNeg)

    order_size, percent = resolve_size_nb(
        size=order_size,
        size_type=order_size_type,
        direction=order.direction,
        position=position,
        val_price=val_price,
        value=value,
    )
    if not np.isnan(min_order_size):
        min_order_size, min_percent = resolve_size_nb(
            size=min_order_size,
            size_type=order_size_type,
            direction=order.direction,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if is_less_nb(percent, min_percent):
            return exec_state, order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached)
    if not np.isnan(max_order_size):
        max_order_size, max_percent = resolve_size_nb(
            size=max_order_size,
            size_type=order_size_type,
            direction=order.direction,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if is_less_nb(max_percent, percent):
            percent = max_percent

    if order_size >= 0:
        new_account_state, order_result = buy_nb(
            account_state=account_state,
            size=order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            price_area_vio_mode=order.price_area_vio_mode,
            lock_cash=order.lock_cash,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    else:
        new_account_state, order_result = sell_nb(
            account_state=account_state,
            size=-order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            price_area_vio_mode=order.price_area_vio_mode,
            lock_cash=order.lock_cash,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            return exec_state, order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.RandomEvent)

    if order_result.status == OrderStatus.Rejected and order.raise_reject:
        raise_rejected_order_nb(order_result)

    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        new_val_price, new_value = update_value_nb(
            cash,
            new_account_state.cash,
            position,
            new_account_state.position,
            val_price,
            order_result.price,
            value,
        )
    else:
        new_val_price = val_price
        new_value = value

    new_exec_state = ExecState(
        cash=new_account_state.cash,
        position=new_account_state.position,
        debt=new_account_state.debt,
        free_cash=new_account_state.free_cash,
        val_price=new_val_price,
        value=new_value,
    )

    return new_exec_state, order_result


@register_jitted(cache=True)
def fill_log_record_nb(
    records: tp.RecordArray2d,
    r: int,
    group: int,
    col: int,
    i: int,
    price_area: PriceArea,
    exec_state: ExecState,
    order: Order,
    new_exec_state: ExecState,
    order_result: OrderResult,
    order_id: int,
) -> None:
    """Fill a log record."""

    records["id"][r, col] = r
    records["group"][r, col] = group
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["price_area_open"][r, col] = price_area.open
    records["price_area_high"][r, col] = price_area.high
    records["price_area_low"][r, col] = price_area.low
    records["price_area_close"][r, col] = price_area.close
    records["exec_state_cash"][r, col] = exec_state.cash
    records["exec_state_position"][r, col] = exec_state.position
    records["exec_state_debt"][r, col] = exec_state.debt
    records["exec_state_free_cash"][r, col] = exec_state.free_cash
    records["exec_state_val_price"][r, col] = exec_state.val_price
    records["exec_state_value"][r, col] = exec_state.value
    records["req_size"][r, col] = order.size
    records["req_price"][r, col] = order.price
    records["req_size_type"][r, col] = order.size_type
    records["req_direction"][r, col] = order.direction
    records["req_fees"][r, col] = order.fees
    records["req_fixed_fees"][r, col] = order.fixed_fees
    records["req_slippage"][r, col] = order.slippage
    records["req_min_size"][r, col] = order.min_size
    records["req_max_size"][r, col] = order.max_size
    records["req_size_granularity"][r, col] = order.size_granularity
    records["req_reject_prob"][r, col] = order.reject_prob
    records["req_price_area_vio_mode"][r, col] = order.price_area_vio_mode
    records["req_lock_cash"][r, col] = order.lock_cash
    records["req_allow_partial"][r, col] = order.allow_partial
    records["req_raise_reject"][r, col] = order.raise_reject
    records["req_log"][r, col] = order.log
    records["new_exec_state_cash"][r, col] = new_exec_state.cash
    records["new_exec_state_position"][r, col] = new_exec_state.position
    records["new_exec_state_debt"][r, col] = new_exec_state.debt
    records["new_exec_state_free_cash"][r, col] = new_exec_state.free_cash
    records["new_exec_state_val_price"][r, col] = new_exec_state.val_price
    records["new_exec_state_value"][r, col] = new_exec_state.value
    records["res_size"][r, col] = order_result.size
    records["res_price"][r, col] = order_result.price
    records["res_fees"][r, col] = order_result.fees
    records["res_side"][r, col] = order_result.side
    records["res_status"][r, col] = order_result.status
    records["res_status_info"][r, col] = order_result.status_info
    records["order_id"][r, col] = order_id


@register_jitted(cache=True)
def fill_order_record_nb(records: tp.RecordArray2d, r: int, col: int, i: int, order_result: OrderResult) -> None:
    """Fill an order record."""

    records["id"][r, col] = r
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["size"][r, col] = order_result.size
    records["price"][r, col] = order_result.price
    records["fees"][r, col] = order_result.fees
    records["side"][r, col] = order_result.side


@register_jitted(cache=True)
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """Raise an `vectorbtpro.portfolio.enums.RejectedOrderError`."""

    if order_result.status_info == OrderStatusInfo.SizeNaN:
        raise RejectedOrderError("Size is NaN")
    if order_result.status_info == OrderStatusInfo.PriceNaN:
        raise RejectedOrderError("Price is NaN")
    if order_result.status_info == OrderStatusInfo.ValPriceNaN:
        raise RejectedOrderError("Asset valuation price is NaN")
    if order_result.status_info == OrderStatusInfo.ValueNaN:
        raise RejectedOrderError("Asset/group value is NaN")
    if order_result.status_info == OrderStatusInfo.ValueZeroNeg:
        raise RejectedOrderError("Asset/group value is zero or negative")
    if order_result.status_info == OrderStatusInfo.SizeZero:
        raise RejectedOrderError("Size is zero")
    if order_result.status_info == OrderStatusInfo.NoCashShort:
        raise RejectedOrderError("Not enough cash to short")
    if order_result.status_info == OrderStatusInfo.NoCashLong:
        raise RejectedOrderError("Not enough cash to long")
    if order_result.status_info == OrderStatusInfo.NoOpenPosition:
        raise RejectedOrderError("No open position to reduce/close")
    if order_result.status_info == OrderStatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("Size is greater than maximum allowed")
    if order_result.status_info == OrderStatusInfo.RandomEvent:
        raise RejectedOrderError("Random event happened")
    if order_result.status_info == OrderStatusInfo.CantCoverFees:
        raise RejectedOrderError("Not enough cash to cover fees")
    if order_result.status_info == OrderStatusInfo.MinSizeNotReached:
        raise RejectedOrderError("Final size is less than minimum allowed")
    if order_result.status_info == OrderStatusInfo.PartialFill:
        raise RejectedOrderError("Final size is less than requested")
    raise RejectedOrderError


@register_jitted(cache=True)
def process_order_nb(
    group: int,
    col: int,
    i: int,
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
    order_records: tp.Optional[tp.RecordArray2d] = None,
    order_counts: tp.Optional[tp.Array1d] = None,
    log_records: tp.Optional[tp.RecordArray2d] = None,
    log_counts: tp.Optional[tp.Array1d] = None,
) -> tp.Tuple[ExecState, OrderResult]:
    """Process an order by executing it, saving relevant information to the logs, and returning a new state."""

    # Execute the order
    new_exec_state, order_result = execute_order_nb(
        exec_state=exec_state,
        order=order,
        price_area=price_area,
        update_value=update_value,
    )

    is_filled = order_result.status == OrderStatus.Filled
    if order_records is not None and order_counts is not None:
        if is_filled and order_records.shape[0] > 0:
            # Fill order record
            if order_counts[col] >= order_records.shape[0]:
                raise IndexError("order_records index out of range. Set a higher max_orders.")
            fill_order_record_nb(order_records, order_counts[col], col, i, order_result)
            order_counts[col] += 1

    if log_records is not None and log_counts is not None:
        if order.log and log_records.shape[0] > 0:
            # Fill log record
            if log_counts[col] >= log_records.shape[0]:
                raise IndexError("log_records index out of range. Set a higher max_logs.")
            fill_log_record_nb(
                log_records,
                log_counts[col],
                group,
                col,
                i,
                price_area,
                exec_state,
                order,
                new_exec_state,
                order_result,
                order_counts[col] - 1 if order_counts is not None and is_filled else -1,
            )
            log_counts[col] += 1

    return new_exec_state, order_result


@register_jitted(cache=True)
def order_nb(
    size: float = np.inf,
    price: float = np.inf,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    lock_cash: bool = False,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Create an order.

    See `vectorbtpro.portfolio.enums.Order` for details on arguments."""

    return Order(
        size=float(size),
        price=float(price),
        size_type=int(size_type),
        direction=int(direction),
        fees=float(fees),
        fixed_fees=float(fixed_fees),
        slippage=float(slippage),
        min_size=float(min_size),
        max_size=float(max_size),
        size_granularity=float(size_granularity),
        reject_prob=float(reject_prob),
        price_area_vio_mode=int(price_area_vio_mode),
        lock_cash=bool(lock_cash),
        allow_partial=bool(allow_partial),
        raise_reject=bool(raise_reject),
        log=bool(log),
    )


@register_jitted(cache=True)
def close_position_nb(
    price: float = np.inf,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    lock_cash: bool = False,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Close the current position."""

    return order_nb(
        size=0.0,
        price=price,
        size_type=SizeType.TargetAmount,
        direction=Direction.Both,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size,
        max_size=max_size,
        size_granularity=size_granularity,
        reject_prob=reject_prob,
        price_area_vio_mode=price_area_vio_mode,
        lock_cash=lock_cash,
        allow_partial=allow_partial,
        raise_reject=raise_reject,
        log=log,
    )


@register_jitted(cache=True)
def order_nothing_nb() -> Order:
    """Convenience function to order nothing."""
    return NoOrder


@register_jitted(cache=True)
def check_group_lens_nb(group_lens: tp.Array1d, n_cols: int) -> None:
    """Check `group_lens`."""
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@register_jitted(cache=True)
def is_grouped_nb(group_lens: tp.Array1d) -> bool:
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_lens > 1)


@register_jitted(cache=True)
def get_group_value_nb(
    from_col: int,
    to_col: int,
    cash_now: float,
    last_position: tp.Array1d,
    last_val_price: tp.Array1d,
) -> float:
    """Get group value."""
    group_value = cash_now
    group_len = to_col - from_col
    for k in range(group_len):
        col = from_col + k
        if last_position[col] != 0:
            group_value += last_position[col] * last_val_price[col]
    return group_value


@register_jitted(cache=True)
def approx_order_value_nb(
    exec_state: ExecState,
    size: float,
    size_type: int,
    direction: int,
) -> float:
    """Approximate the value of an order."""
    if direction == Direction.ShortOnly:
        size *= -1
    asset_value_now = exec_state.position * exec_state.val_price
    if size_type == SizeType.Amount:
        return size * exec_state.val_price
    if size_type == SizeType.Value:
        return size
    if size_type == SizeType.Percent100:
        size /= 100
        size_type = SizeType.Percent
    if size_type == SizeType.Percent:
        if size >= 0:
            return size * exec_state.cash
        else:
            if direction == Direction.LongOnly:
                return size * asset_value_now
            return size * (2 * max(asset_value_now, 0) + max(exec_state.free_cash, 0))
    if size_type == SizeType.ValuePercent100:
        size /= 100
        size_type = SizeType.ValuePercent
    if size_type == SizeType.ValuePercent:
        return size * exec_state.value
    if size_type == SizeType.TargetAmount:
        return size * exec_state.val_price - asset_value_now
    if size_type == SizeType.TargetValue:
        return size - asset_value_now
    if size_type == SizeType.TargetPercent100:
        size /= 100
        size_type = SizeType.TargetPercent
    if size_type == SizeType.TargetPercent:
        return size * exec_state.value - asset_value_now
    return np.nan


@register_jitted(cache=True)
def prepare_records_nb(
    target_shape: tp.Shape,
    max_orders: tp.Optional[int] = None,
    max_logs: tp.Optional[int] = 0,
) -> tp.Tuple[tp.RecordArray2d, tp.RecordArray2d]:
    """Prepare records."""
    if max_orders is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=order_dt)
    else:
        order_records = np.empty((max_orders, target_shape[1]), dtype=order_dt)
    if max_logs is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_logs, target_shape[1]), dtype=log_dt)
    return order_records, log_records


@register_jitted(cache=True)
def prepare_last_cash_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare `last_cash`."""
    if cash_sharing:
        last_cash = np.empty(len(group_lens), dtype=np.float_)
        for group in range(len(group_lens)):
            last_cash[group] = float(flex_select_1d_nb(init_cash, group))
    else:
        last_cash = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            last_cash[col] = float(flex_select_1d_nb(init_cash, col))
    return last_cash


@register_jitted(cache=True)
def prepare_last_position_nb(target_shape: tp.Shape, init_position: tp.FlexArray1d) -> tp.Array1d:
    """Prepare `last_position`."""
    last_position = np.empty(target_shape[1], dtype=np.float_)
    for col in range(target_shape[1]):
        last_position[col] = float(flex_select_1d_nb(init_position, col))
    return last_position


@register_jitted(cache=True)
def prepare_last_value_nb(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
    init_position: tp.FlexArray1d = np.array([0.0]),
    init_price: tp.FlexArray1d = np.array([np.nan]),
) -> tp.Array1d:
    """Prepare `last_value`."""
    if cash_sharing:
        last_value = np.empty(len(group_lens), dtype=np.float_)
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = float(flex_select_1d_nb(init_cash, group))
            last_value[group] = _init_cash
            for col in range(from_col, to_col):
                _init_position = float(flex_select_1d_nb(init_position, col))
                _init_price = float(flex_select_1d_nb(init_price, col))
                if _init_position != 0:
                    last_value[group] += _init_position * _init_price
            from_col = to_col
    else:
        last_value = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            _init_cash = float(flex_select_1d_nb(init_cash, col))
            _init_position = float(flex_select_1d_nb(init_position, col))
            _init_price = float(flex_select_1d_nb(init_price, col))
            if _init_position == 0:
                last_value[col] = _init_cash
            else:
                last_value[col] = _init_cash + _init_position * _init_price
    return last_value


@register_jitted(cache=True)
def prepare_last_pos_record_nb(
    target_shape: tp.Shape,
    init_position: tp.FlexArray1d = np.array([0.0]),
    init_price: tp.FlexArray1d = np.array([np.nan]),
    fill_pos_record: bool = True,
) -> tp.RecordArray:
    """Prepare `last_pos_record`."""
    last_pos_record = np.empty(target_shape[1], dtype=trade_dt)
    last_pos_record["id"][:] = -1
    if fill_pos_record:
        for col in range(target_shape[1]):
            _init_position = float(flex_select_1d_nb(init_position, col))
            _init_price = float(flex_select_1d_nb(init_price, col))
            if _init_position != 0:
                fill_init_pos_record_nb(last_pos_record[col], col, _init_position, _init_price)
    return last_pos_record


@register_jitted
def prepare_simout_nb(
    order_records: tp.RecordArray2d,
    order_counts: tp.Array1d,
    log_records: tp.RecordArray2d,
    log_counts: tp.Array1d,
    cash_deposits: tp.Array2d,
    cash_earnings: tp.Array2d,
    call_seq: tp.Optional[tp.Array2d] = None,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
) -> SimulationOutput:
    """Prepare simulation output."""
    order_records_flat = generic_nb.repartition_nb(order_records, order_counts)
    log_records_flat = generic_nb.repartition_nb(log_records, log_counts)
    return SimulationOutput(
        order_records=order_records_flat,
        log_records=log_records_flat,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
    )


@register_jitted(cache=True)
def get_trade_stats_nb(
    size: float,
    entry_price: float,
    entry_fees: float,
    exit_price: float,
    exit_fees: float,
    direction: int,
) -> tp.Tuple[float, float]:
    """Get trade statistics."""
    entry_val = size * entry_price
    exit_val = size * exit_price
    val_diff = add_nb(exit_val, -entry_val)
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    pnl = val_diff - entry_fees - exit_fees
    if is_close_nb(entry_val, 0):
        ret = np.nan
    else:
        ret = pnl / entry_val
    return pnl, ret


@register_jitted(cache=True)
def update_open_pos_stats_nb(record: tp.Record, position_now: float, price: float) -> None:
    """Update statistics of an open position record using custom price."""
    if record["id"] >= 0 and record["status"] == TradeStatus.Open:
        if np.isnan(record["exit_price"]):
            exit_price = price
        else:
            exit_size_sum = record["size"] - abs(position_now)
            exit_gross_sum = exit_size_sum * record["exit_price"]
            exit_gross_sum += abs(position_now) * price
            exit_price = exit_gross_sum / record["size"]
        pnl, ret = get_trade_stats_nb(
            record["size"],
            record["entry_price"],
            record["entry_fees"],
            exit_price,
            record["exit_fees"],
            record["direction"],
        )
        record["pnl"] = pnl
        record["return"] = ret


@register_jitted(cache=True)
def fill_init_pos_record_nb(record: tp.Record, col: int, position_now: float, price: float) -> None:
    """Fill position record for an initial position."""
    record["id"] = 0
    record["col"] = col
    record["size"] = abs(position_now)
    record["entry_order_id"] = -1
    record["entry_idx"] = -1
    record["entry_price"] = price
    record["entry_fees"] = 0.0
    record["exit_order_id"] = -1
    record["exit_idx"] = -1
    record["exit_price"] = np.nan
    record["exit_fees"] = 0.0
    if position_now >= 0:
        record["direction"] = TradeDirection.Long
    else:
        record["direction"] = TradeDirection.Short
    record["status"] = TradeStatus.Open
    record["parent_id"] = record["id"]

    # Update open position stats
    update_open_pos_stats_nb(record, position_now, np.nan)


@register_jitted(cache=True)
def update_pos_record_nb(
    record: tp.Record,
    i: int,
    col: int,
    position_before: float,
    position_now: float,
    order_result: OrderResult,
    order_id: int,
) -> None:
    """Update position record after filling an order."""
    if order_result.status == OrderStatus.Filled:
        if position_before == 0 and position_now != 0:
            # New position opened
            record["id"] += 1
            record["col"] = col
            record["size"] = order_result.size
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            record["entry_fees"] = order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        elif position_before != 0 and position_now == 0:
            # Position closed
            record["exit_order_id"] = order_id
            record["exit_idx"] = i
            if np.isnan(record["exit_price"]):
                exit_price = order_result.price
            else:
                exit_size_sum = record["size"] - abs(position_before)
                exit_gross_sum = exit_size_sum * record["exit_price"]
                exit_gross_sum += abs(position_before) * order_result.price
                exit_price = exit_gross_sum / record["size"]
            record["exit_price"] = exit_price
            record["exit_fees"] += order_result.fees
            pnl, ret = get_trade_stats_nb(
                record["size"],
                record["entry_price"],
                record["entry_fees"],
                record["exit_price"],
                record["exit_fees"],
                record["direction"],
            )
            record["pnl"] = pnl
            record["return"] = ret
            record["status"] = TradeStatus.Closed
        elif np.sign(position_before) != np.sign(position_now):
            # Position reversed
            record["id"] += 1
            record["size"] = abs(position_now)
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            new_pos_fraction = abs(position_now) / abs(position_now - position_before)
            record["entry_fees"] = new_pos_fraction * order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        else:
            # Position changed
            if abs(position_before) <= abs(position_now):
                # Position increased
                entry_gross_sum = record["size"] * record["entry_price"]
                entry_gross_sum += order_result.size * order_result.price
                entry_price = entry_gross_sum / (record["size"] + order_result.size)
                record["entry_price"] = entry_price
                record["entry_fees"] += order_result.fees
                record["size"] += order_result.size
            else:
                # Position decreased
                record["exit_order_id"] = order_id
                if np.isnan(record["exit_price"]):
                    exit_price = order_result.price
                else:
                    exit_size_sum = record["size"] - abs(position_before)
                    exit_gross_sum = exit_size_sum * record["exit_price"]
                    exit_gross_sum += order_result.size * order_result.price
                    exit_price = exit_gross_sum / (exit_size_sum + order_result.size)
                record["exit_price"] = exit_price
                record["exit_fees"] += order_result.fees

        # Update open position stats
        update_open_pos_stats_nb(record, position_now, order_result.price)
