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

from vectorbtpro import _typing as tp
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.registries.jit_registry import register_jitted


@register_jitted(cache=True)
def ma_nb(a: tp.Array2d, window: int, ewm: bool, adjust: bool = False) -> tp.Array2d:
    """Compute simple or exponential moving average (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_mean_nb(a, window, minp=window, adjust=adjust)
    return generic_nb.rolling_mean_nb(a, window, minp=window)


@register_jitted(cache=True)
def mstd_nb(a: tp.Array2d, window: int, ewm: int, adjust: bool = False, ddof: int = 0) -> tp.Array2d:
    """Compute simple or exponential moving STD (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_std_nb(a, window, minp=window, adjust=adjust)
    return generic_nb.rolling_std_nb(a, window, minp=window, ddof=ddof)


@register_jitted(cache=True)
def ma_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool = False,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MA`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(close, windows[i], ewms[i], adjust=adjust)
    return cache_dict


@register_jitted(cache=True)
def ma_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MA`."""
    if cache_dict is not None:
        h = hash((window, ewm))
        return cache_dict[h]
    return ma_nb(close, window, ewm, adjust=adjust)


@register_jitted(cache=True)
def mstd_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool = False,
    ddof: int = 0,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MSTD`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(close, windows[i], ewms[i], adjust=adjust, ddof=ddof)
    return cache_dict


@register_jitted(cache=True)
def mstd_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    ddof: int = 0,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.MSTD`."""
    if cache_dict is not None:
        h = hash((window, ewm))
        return cache_dict[h]
    return mstd_nb(close, window, ewm, adjust=adjust, ddof=ddof)


@register_jitted(cache=True)
def bb_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    alphas: tp.List[float],
    adjust: bool = False,
    ddof: int = 0,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Dict[int, tp.Array2d]], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.BBANDS`."""
    if per_column:
        return None, None
    ma_cache_dict = ma_cache_nb(close, windows, ewms, adjust)
    mstd_cache_dict = mstd_cache_nb(close, windows, ewms, adjust, ddof)
    return ma_cache_dict, mstd_cache_dict


@register_jitted(cache=True)
def bb_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    alpha: float,
    adjust: bool = False,
    ddof: int = 0,
    ma_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
    mstd_cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.BBANDS`."""
    # Calculate lower, middle and upper bands
    if ma_cache_dict is not None and mstd_cache_dict is not None:
        h = hash((window, ewm))
        ma = np.copy(ma_cache_dict[h])
        mstd = np.copy(mstd_cache_dict[h])
    else:
        ma = ma_nb(close, window, ewm, adjust=adjust)
        mstd = mstd_nb(close, window, ewm, adjust=adjust, ddof=ddof)
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@register_jitted(cache=True)
def rsi_up_down_nb(close: tp.Array2d) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Calculate the `up` and `down` arrays for RSI."""
    delta = generic_nb.diff_nb(close)
    up, down = delta.copy(), delta.copy()
    up = generic_nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(generic_nb.set_by_mask_nb(down, down > 0, 0))
    return up, down


@register_jitted(cache=True)
def rsi_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool = False,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.RSI`."""
    up, down = rsi_up_down_nb(close)
    if per_column:
        return None

    # Cache
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            roll_up = ma_nb(up, windows[i], ewms[i], adjust=adjust)
            roll_down = ma_nb(down, windows[i], ewms[i], adjust=adjust)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@register_jitted(cache=True)
def rsi_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Array2d:
    """Apply function for `vectorbtpro.indicators.custom.RSI`."""
    if cache_dict is not None:
        h = hash((window, ewm))
        roll_up, roll_down = cache_dict[h]
    else:
        up, down = rsi_up_down_nb(close)
        roll_up = ma_nb(up, window, ewm, adjust=adjust)
        roll_down = ma_nb(down, window, ewm, adjust=adjust)
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


@register_jitted(cache=True)
def stoch_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_windows: tp.List[int],
    d_windows: tp.List[int],
    d_ewms: tp.List[bool],
    adjust: bool = False,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.STOCH`."""
    if per_column:
        return None
    cache_dict = dict()
    for i in range(len(k_windows)):
        h = hash(k_windows[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low, k_windows[i])
            roll_max = generic_nb.rolling_max_nb(high, k_windows[i])
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@register_jitted(cache=True)
def stoch_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_window: int,
    d_window: int,
    d_ewm: bool,
    adjust: bool = False,
    cache_dict: tp.Optional[tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.STOCH`."""
    if cache_dict is not None:
        h = hash(k_window)
        roll_min, roll_max = cache_dict[h]
    else:
        roll_min = generic_nb.rolling_min_nb(low, k_window)
        roll_max = generic_nb.rolling_max_nb(high, k_window)
    percent_k = 100 * (close - roll_min) / (roll_max - roll_min)
    percent_d = ma_nb(percent_k, d_window, d_ewm, adjust=adjust)
    return percent_k, percent_d


@register_jitted(cache=True)
def macd_cache_nb(
    close: tp.Array2d,
    fast_windows: tp.List[int],
    slow_windows: tp.List[int],
    signal_windows: tp.List[int],
    macd_ewms: tp.List[bool],
    signal_ewms: tp.List[bool],
    adjust: bool = False,
    per_column: bool = False,
) -> tp.Optional[tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbtpro.indicators.custom.MACD`."""
    if per_column:
        return None
    windows = fast_windows.copy()
    windows.extend(slow_windows)
    ewms = macd_ewms.copy()
    ewms.extend(macd_ewms)
    return ma_cache_nb(close, windows, ewms, adjust)


@register_jitted(cache=True)
def macd_apply_nb(
    close: tp.Array2d,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    macd_ewm: bool,
    signal_ewm: bool,
    adjust: bool = False,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.MACD`."""
    if cache_dict is not None:
        fast_h = hash((fast_window, macd_ewm))
        slow_h = hash((slow_window, macd_ewm))
        fast_ma = cache_dict[fast_h]
        slow_ma = cache_dict[slow_h]
    else:
        fast_ma = ma_nb(close, fast_window, macd_ewm, adjust=adjust)
        slow_ma = ma_nb(close, slow_window, macd_ewm, adjust=adjust)
    macd_ts = fast_ma - slow_ma
    signal_ts = ma_nb(macd_ts, signal_window, signal_ewm, adjust=adjust)
    return macd_ts, signal_ts


@register_jitted(cache=True)
def true_range_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Calculate true range."""
    prev_close = generic_nb.fshift_nb(close, 1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.empty(prev_close.shape, dtype=np.float_)
    for col in range(tr.shape[1]):
        for i in range(tr.shape[0]):
            tr[i, col] = max(tr1[i, col], tr2[i, col], tr3[i, col])
    return tr


@register_jitted(cache=True)
def atr_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool = False,
    per_column: bool = False,
) -> tp.Tuple[tp.Optional[tp.Array2d], tp.Optional[tp.Dict[int, tp.Array2d]]]:
    """Caching function for `vectorbtpro.indicators.custom.ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_nb
    tr = true_range_nb(high, low, close)
    if per_column:
        return None, None

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(tr, windows[i], ewms[i], adjust=adjust)
    return tr, cache_dict


@register_jitted(cache=True)
def atr_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    tr: tp.Optional[tp.Array2d] = None,
    cache_dict: tp.Optional[tp.Dict[int, tp.Array2d]] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbtpro.indicators.custom.ATR`."""
    if cache_dict is not None:
        h = hash((window, ewm))
        return tr, cache_dict[h]
    tr = true_range_nb(high, low, close)
    return tr, ma_nb(tr, window, ewm, adjust=adjust)


@register_jitted(cache=True)
def obv_custom_nb(close: tp.Array2d, volume_ts: tp.Array2d) -> tp.Array2d:
    """Custom calculation function for `vectorbtpro.indicators.custom.OBV`."""
    obv = generic_nb.set_by_mask_mult_nb(volume_ts, close < generic_nb.fshift_nb(close, 1), -volume_ts)
    obv = generic_nb.nancumsum_nb(obv)
    return obv
