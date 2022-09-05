# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for custom indicators.

Provides an arsenal of Numba-compiled functions that are used by indicator
classes. These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import flex_select_auto_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.indicators.enums import Pivot


@register_jitted(cache=True)
def ma_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MA`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.ma_nb(close, windows[i], wtypes[i], adjust=adjust, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def ma_apply_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MA`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return cache_dict[h]
    return generic_nb.ma_nb(close, window, wtype, adjust=adjust, minp=minp)


@register_jitted(cache=True)
def msd_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MSD`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.msd_nb(close, windows[i], wtypes[i], adjust=adjust, ddof=ddof, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def msd_apply_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MSD`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return cache_dict[h]
    return generic_nb.msd_nb(close, window, wtype, adjust=adjust, ddof=ddof, minp=minp)


@register_jitted(cache=True)
def bb_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    alphas: tp.List[float],
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Dict[int, tp.Array2d]], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.BBANDS`."""
    if per_column:
        return None, None

    ma_cache_dict = ma_cache_nb(close, windows, wtypes, adjust=adjust, minp=minp)
    msd_cache_dict = msd_cache_nb(close, windows, wtypes, adjust=adjust, ddof=ddof, minp=minp)
    return ma_cache_dict, msd_cache_dict


@register_jitted(cache=True)
def bb_apply_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    alpha: float,
    adjust: bool = False,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    ma_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
    msd_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.BBANDS`."""
    if ma_cache_dict is not None and msd_cache_dict is not None:
        h = hash((window, wtype))
        ma = np.copy(ma_cache_dict[h])
        msd = np.copy(msd_cache_dict[h])
    else:
        ma = generic_nb.ma_nb(close, window, wtype, adjust=adjust, minp=minp)
        msd = generic_nb.msd_nb(close, window, wtype, adjust=adjust, ddof=ddof, minp=minp)
    return ma, ma + alpha * msd, ma - alpha * msd


@register_jitted(cache=True, tags={"can_parallel"})
def rsi_up_down_nb(close: tp.Array2d) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Calculate the `up` and `down` arrays for RSI."""
    up = np.empty_like(close, dtype=np.float_)
    down = np.empty_like(close, dtype=np.float_)
    for col in prange(close.shape[1]):
        for i in range(close.shape[0]):
            if i == 0:
                up[i, col] = np.nan
                down[i, col] = np.nan
            else:
                delta = close[i, col] - close[i - 1, col]
                if delta < 0:
                    up[i, col] = 0.0
                    down[i, col] = abs(delta)
                else:
                    up[i, col] = delta
                    down[i, col] = 0.0
    return up, down


@register_jitted(cache=True)
def rsi_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.RSI`."""
    up, down = rsi_up_down_nb(close)
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            roll_up = generic_nb.ma_nb(up, windows[i], wtypes[i], adjust=adjust, minp=minp)
            roll_down = generic_nb.ma_nb(down, windows[i], wtypes[i], adjust=adjust, minp=minp)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@register_jitted(cache=True)
def rsi_apply_nb(
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.RSI`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        roll_up, roll_down = cache_dict[h]
    else:
        up, down = rsi_up_down_nb(close)
        roll_up = generic_nb.ma_nb(up, window, wtype, adjust=adjust, minp=minp)
        roll_down = generic_nb.ma_nb(down, window, wtype, adjust=adjust, minp=minp)
    return 100 * roll_up / (roll_up + roll_down)


@register_jitted(cache=True)
def stoch_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    fast_k_windows: tp.List[int],
    slow_k_windows: tp.List[int],
    slow_d_windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.STOCH`."""
    if per_column:
        return None

    cache_dict = dict()
    for i in range(len(fast_k_windows)):
        h = hash(fast_k_windows[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low, fast_k_windows[i], minp=minp)
            roll_max = generic_nb.rolling_max_nb(high, fast_k_windows[i], minp=minp)
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@register_jitted(cache=True)
def stoch_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    fast_k_window: int,
    slow_k_window: int,
    slow_d_window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.STOCH`."""
    if cache_dict is not None:
        h = hash(fast_k_window)
        roll_min, roll_max = cache_dict[h]
    else:
        roll_min = generic_nb.rolling_min_nb(low, fast_k_window, minp=minp)
        roll_max = generic_nb.rolling_max_nb(high, fast_k_window, minp=minp)
    fast_k = 100 * (close - roll_min) / (roll_max - roll_min)
    slow_k = generic_nb.ma_nb(fast_k, slow_k_window, wtype, adjust=adjust, minp=minp)
    slow_d = generic_nb.ma_nb(slow_k, slow_d_window, wtype, adjust=adjust, minp=minp)
    return fast_k, slow_k, slow_d


@register_jitted(cache=True)
def macd_cache_nb(
    close: tp.Array2d,
    fast_windows: tp.List[int],
    slow_windows: tp.List[int],
    signal_windows: tp.List[int],
    macd_wtypes: tp.List[int],
    signal_wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MACD`."""
    if per_column:
        return None

    windows = fast_windows.copy()
    windows.extend(slow_windows)
    wtypes = macd_wtypes.copy()
    wtypes.extend(macd_wtypes)
    return ma_cache_nb(close, windows, wtypes, adjust=adjust, minp=minp)


@register_jitted(cache=True)
def macd_apply_nb(
    close: tp.Array2d,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    macd_wtype: int,
    signal_wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.MACD`."""
    if cache_dict is not None:
        fast_h = hash((fast_window, macd_wtype))
        slow_h = hash((slow_window, macd_wtype))
        fast_ma = cache_dict[fast_h]
        slow_ma = cache_dict[slow_h]
    else:
        fast_ma = generic_nb.ma_nb(close, fast_window, macd_wtype, adjust=adjust, minp=minp)
        slow_ma = generic_nb.ma_nb(close, slow_window, macd_wtype, adjust=adjust, minp=minp)
    macd_ts = fast_ma - slow_ma
    signal_ts = generic_nb.ma_nb(macd_ts, signal_window, signal_wtype, adjust=adjust, minp=minp)
    return macd_ts, signal_ts


@register_jitted(cache=True)
def true_range_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Calculate true range."""
    tr = np.empty(high.shape, dtype=np.float_)
    for col in range(high.shape[1]):
        for i in range(high.shape[0]):
            prev_close = close[i - 1, col] if i > 0 else np.nan
            tr1 = high[i, col] - low[i, col]
            tr2 = abs(high[i, col] - prev_close)
            tr3 = abs(low[i, col] - prev_close)
            tr[i, col] = max(tr1, tr2, tr3)
    return tr


@register_jitted(cache=True)
def atr_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    windows: tp.List[int],
    wtypes: tp.List[int],
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Array2d], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.ATR`."""
    tr = true_range_nb(high, low, close)
    if per_column:
        return None, None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], wtypes[i]))
        if h not in cache_dict:
            cache_dict[h] = generic_nb.ma_nb(tr, windows[i], wtypes[i], adjust=adjust, minp=minp)
    return tr, cache_dict


@register_jitted(cache=True)
def atr_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: int,
    wtype: int,
    adjust: bool = False,
    minp: tp.Optional[int] = None,
    tr: tp.Optional[tp.Array2d] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.ATR`."""
    if cache_dict is not None:
        h = hash((window, wtype))
        return tr, cache_dict[h]
    if tr is None:
        tr = true_range_nb(high, low, close)
    return tr, generic_nb.ma_nb(tr, window, wtype, adjust=adjust, minp=minp)


@register_jitted(cache=True)
def obv_custom_nb(close: tp.Array2d, volume: tp.Array2d) -> tp.Array2d:
    """Custom calculation function for `vectorbtpro.indicators.custom.OBV`."""
    obv = np.empty_like(close)
    cumsum = 0.0
    for col in range(close.shape[1]):
        for i in range(close.shape[0]):
            prev_close = close[i - 1, col] if i > 0 else np.nan
            if close[i, col] < prev_close:
                value = -volume[i, col]
            else:
                value = volume[i, col]
            if not np.isnan(value):
                cumsum += value
            obv[i, col] = cumsum
    return obv


@register_jitted(cache=True)
def ols_cache_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    windows: tp.List[int],
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.OLS`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash(windows[i])
        if h not in cache_dict:
            cache_dict[h] = generic_nb.rolling_ols_nb(x, y, windows[i], minp=minp)
    return cache_dict


@register_jitted(cache=True)
def ols_apply_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.OLS`."""
    if cache_dict is not None:
        h = hash(window)
        return cache_dict[h]
    return generic_nb.rolling_ols_nb(x, y, window, minp=minp)


@register_jitted(cache=True)
def ols_spread_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Calculate the OLS spread and z-score."""
    slope, intercept = generic_nb.rolling_ols_nb(x, y, window, minp=minp)
    pred = intercept + slope * x
    spread = y - pred
    spread_mean = generic_nb.rolling_mean_nb(spread, window, minp=minp)
    spread_std = generic_nb.rolling_std_nb(spread, window, ddof=ddof, minp=minp)
    spread_zscore = (spread - spread_mean) / spread_std
    return spread, spread_zscore


@register_jitted(cache=True)
def ols_spread_cache_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    windows: tp.List[int],
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.OLSS`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash(windows[i])
        if h not in cache_dict:
            cache_dict[h] = ols_spread_nb(x, y, windows[i], ddof=ddof, minp=minp)
    return cache_dict


@register_jitted(cache=True)
def ols_spread_apply_nb(
    x: tp.Array2d,
    y: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.OLSS`."""
    if cache_dict is not None:
        h = hash(window)
        return cache_dict[h]
    return ols_spread_nb(x, y, window, ddof=ddof, minp=minp)


@register_jitted(cache=True)
def vwap_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    close: tp.Array1d,
    volume: tp.Array1d,
    group_lens: tp.GroupLens,
) -> tp.Array1d:
    """Compute the volume-weighted average price (VWAP)."""
    out = np.empty_like(volume, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        nom_cumsum = 0
        denum_cumsum = 0
        for i in range(from_i, to_i):
            nom_cumsum += volume[i] * (high[i] + low[i] + close[i]) / 3
            denum_cumsum += volume[i]
            out[i] = nom_cumsum / denum_cumsum
    return out


@register_jitted(cache=True)
def vwap_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    volume: tp.Array2d,
    group_lens: tp.GroupLens,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.VWAP`."""
    out = np.empty_like(volume, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in range(len(group_lens)):
        from_i = group_start_idxs[group]
        to_i = group_end_idxs[group]
        for col in range(volume.shape[1]):
            nom_cumsum = 0
            denum_cumsum = 0
            for i in range(from_i, to_i):
                typical_price = (high[i, col] + low[i, col] + close[i, col]) / 3
                nom_cumsum += volume[i, col] * typical_price
                denum_cumsum += volume[i, col]
                out[i, col] = nom_cumsum / denum_cumsum
    return out


@register_jitted(cache=True)
def initial_pivots_nb(
    arr: tp.Array2d,
    up_th: tp.FlexArray,
    down_th: tp.FlexArray,
    flex_2d: bool = False,
) -> tp.Array1d:
    """Find the initial pivot in each column."""
    initial_pivots = np.empty(arr.shape[1], dtype=np.int_)
    for col in range(arr.shape[1]):
        minv = arr[0, col]
        maxv = arr[0, col]
        min_i = 0
        max_i = 0
        found_pivot = Pivot.Valley if arr[0, col] < arr[-1, col] else Pivot.Peak

        for i in range(1, arr.shape[0]):
            _up_th = 1 + abs(flex_select_auto_nb(up_th, i, col, flex_2d))
            _down_th = 1 + abs(flex_select_auto_nb(down_th, i, col, flex_2d))
            if arr[i, col] / minv >= _up_th:
                found_pivot = Pivot.Valley if min_i == 0 else Pivot.Peak
                break
            if arr[i, col] / maxv <= _down_th:
                found_pivot = Pivot.Peak if max_i == 0 else Pivot.Valley
                break
            if arr[i, col] > maxv:
                maxv = arr[i, col]
                max_i = i
            if arr[i, col] < minv:
                minv = arr[i, col]
                min_i = i

        initial_pivots[col] = found_pivot

    return initial_pivots


@register_jitted(cache=True)
def zigzag_apply_nb(
    arr: tp.Array2d,
    up_th: tp.FlexArray,
    down_th: tp.FlexArray,
    finalized_only: bool = True,
    eager_switching: bool = False,
    flex_2d: bool = False,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.ZIGZAG`.

    Based on https://github.com/jbn/ZigZag

    Specify `up_th` and `down_th` to set the minimum and maximum relative change
    necessary to define a peak and a valley respectively.

    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias
    analysis.
    """
    initial_pivots = initial_pivots_nb(arr, up_th, down_th)
    pivots = np.zeros(arr.shape, dtype=np.int_)

    for col in range(arr.shape[1]):
        trend = -initial_pivots[col]
        last_pivot_i = 0
        last_pivot = arr[0, col]
        pivots[0, col] = initial_pivots[col]

        for i in range(1, arr.shape[0]):
            _up_th = 1 + abs(flex_select_auto_nb(up_th, i, col, flex_2d))
            _down_th = 1 + abs(flex_select_auto_nb(down_th, i, col, flex_2d))
            r = arr[i, col] / last_pivot

            if trend == -1:
                if r >= _up_th:
                    pivots[last_pivot_i, col] = trend
                    trend = Pivot.Peak
                    last_pivot = arr[i, col]
                    last_pivot_i = i
                elif arr[i, col] < last_pivot:
                    last_pivot = arr[i, col]
                    last_pivot_i = i
            else:
                if r <= _down_th:
                    pivots[last_pivot_i, col] = trend
                    trend = Pivot.Valley
                    last_pivot = arr[i, col]
                    last_pivot_i = i
                elif arr[i, col] > last_pivot:
                    last_pivot = arr[i, col]
                    last_pivot_i = i

        if finalized_only:
            if eager_switching:
                if 0 < last_pivot_i < arr.shape[0] - 1:
                    pivots[last_pivot_i, col] = trend
                    pivots[-1, col] = -trend
                else:
                    pivots[-1, col] = trend
            else:
                if last_pivot_i == arr.shape[0] - 1:
                    pivots[last_pivot_i, col] = trend
                elif pivots[-1, col] == 0:
                    pivots[-1, col] = -trend

    return pivots


@register_jitted(cache=True)
def pivots_to_modes_nb(pivots: tp.Array2d) -> tp.Array2d:
    """Translate pivots into trend modes."""
    modes = np.zeros(pivots.shape, dtype=np.int_)

    for col in range(pivots.shape[1]):
        if pivots[0, col] != 0:
            mode = -pivots[0, col]
        else:
            mode = 0
        modes[0, col] = pivots[0, col]
        for i in range(1, pivots.shape[0]):
            if pivots[i, col] != 0:
                modes[i, col] = mode
                mode = -pivots[i, col]
            else:
                modes[i, col] = mode

    return modes


@register_jitted(cache=True)
def pivot_info_apply_nb(
    arr: tp.Array2d,
    up_th: tp.FlexArray,
    down_th: tp.FlexArray,
    flex_2d: bool = False,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.PIVOTINFO`."""
    conf_pivot = np.empty(arr.shape, dtype=np.int_)
    conf_idx = np.empty(arr.shape, dtype=np.int_)
    last_pivot = np.empty(arr.shape, dtype=np.int_)
    last_idx = np.empty(arr.shape, dtype=np.int_)

    for col in range(arr.shape[1]):
        _conf_pivot = 0
        _conf_idx = -1
        _conf_value = np.nan
        _last_pivot = 0
        _last_idx = -1
        _last_value = np.nan

        for i in range(arr.shape[0]):
            _up_th = 1 + abs(flex_select_auto_nb(up_th, i, col, flex_2d))
            _down_th = 1 - abs(flex_select_auto_nb(down_th, i, col, flex_2d))
            if _last_pivot == Pivot.Valley:
                if arr[i, col] >= _last_value * _up_th:
                    _conf_pivot = _last_pivot
                    _conf_idx = _last_idx
                    _conf_value = _last_value
                    _last_pivot = Pivot.Peak
                    _last_idx = i
                    _last_value = arr[i, col]
                elif arr[i, col] < _last_value:
                    _last_idx = i
                    _last_value = arr[i, col]
            elif _last_pivot == Pivot.Peak:
                _last_value = arr[_last_idx, col]
                if arr[i, col] <= _last_value * _down_th:
                    _conf_pivot = _last_pivot
                    _conf_idx = _last_idx
                    _conf_value = _last_value
                    _last_pivot = Pivot.Valley
                    _last_idx = i
                    _last_value = arr[i, col]
                elif arr[i, col] > _last_value:
                    _last_idx = i
                    _last_value = arr[i, col]
            else:
                if arr[i, col] >= arr[0, col] * _up_th:
                    _last_pivot = Pivot.Peak
                    _last_idx = i
                    _last_value = arr[i, col]
                if arr[i, col] <= arr[0, col] * _down_th:
                    _last_pivot = Pivot.Valley
                    _last_idx = i
                    _last_value = arr[i, col]

            conf_pivot[i, col] = _conf_pivot
            conf_idx[i, col] = _conf_idx
            last_pivot[i, col] = _last_pivot
            last_idx[i, col] = _last_idx

    return conf_pivot, conf_idx, last_pivot, last_idx
