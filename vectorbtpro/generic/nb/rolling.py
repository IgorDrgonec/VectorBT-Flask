"""Generic Numba-compiled functions for rolling and expanding windows."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import rank_1d_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

# ############# Rolling functions ############# #


@register_jitted(cache=True)
def rolling_sum_acc_nb(in_state: RollSumAIS) -> RollSumAOS:
    """Accumulator of `rolling_sum_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollSumAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollSumAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum

    return RollSumAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_sum_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling sum.

    Uses `rolling_sum_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).sum()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollSumAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_sum_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sum_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_sum_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_sum_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_prod_acc_nb(in_state: RollProdAIS) -> RollProdAOS:
    """Accumulator of `rolling_prod_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollProdAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollProdAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumprod = in_state.cumprod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumprod = cumprod * value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumprod = cumprod
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumprod = cumprod / pre_window_value
        window_len = window - nancnt
        window_cumprod = cumprod
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumprod

    return RollProdAOS(cumprod=cumprod, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_prod_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling product.

    Uses `rolling_prod_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).apply(np.prod)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumprod = 1.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollProdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumprod=cumprod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_prod_acc_nb(in_state)
        cumprod = out_state.cumprod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_prod_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_prod_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_prod_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_mean_acc_nb(in_state: RollMeanAIS) -> RollMeanAOS:
    """Accumulator of `rolling_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollMeanAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
        window_cumsum = cumsum
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
        window_cumsum = cumsum
    if window_len < minp:
        value = np.nan
    else:
        value = window_cumsum / window_len

    return RollMeanAOS(cumsum=cumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling mean.

    Uses `rolling_mean_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_std_acc_nb(in_state: RollStdAIS) -> RollStdAOS:
    """Accumulator of `rolling_std_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollStdAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollStdAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    cumsum_sq = in_state.cumsum_sq
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
        cumsum_sq = cumsum_sq + value**2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
            cumsum_sq = cumsum_sq - pre_window_value**2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        mean = cumsum / window_len
        value = np.sqrt(np.abs(cumsum_sq - 2 * cumsum * mean + window_len * mean**2) / (window_len - ddof))

    return RollStdAOS(cumsum=cumsum, cumsum_sq=cumsum_sq, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def rolling_std_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array1d:
    """Compute rolling standard deviation.

    Uses `rolling_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).std(ddof=ddof)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    cumsum_sq = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = RollStdAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            cumsum_sq=cumsum_sq,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_std_acc_nb(in_state)
        cumsum = out_state.cumsum
        cumsum_sq = out_state.cumsum_sq
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_std_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, ddof: int = 0) -> tp.Array2d:
    """2-dim version of `rolling_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_std_1d_nb(arr[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def wm_mean_acc_nb(in_state: WMMeanAIS) -> WMMeanAOS:
    """Accumulator of `wm_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.WMMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.WMMeanAOS`."""
    i = in_state.i
    value = in_state.value
    pre_window_value = in_state.pre_window_value
    cumsum = in_state.cumsum
    wcumsum = in_state.wcumsum
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if i >= window and not np.isnan(pre_window_value):
        wcumsum = wcumsum - cumsum
    if np.isnan(value):
        nancnt = nancnt + 1
    else:
        cumsum = cumsum + value
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value):
            nancnt = nancnt - 1
        else:
            cumsum = cumsum - pre_window_value
        window_len = window - nancnt
    if not np.isnan(value):
        wcumsum = wcumsum + value * window_len
    if window_len < minp:
        value = np.nan
    else:
        value = wcumsum * 2 / (window_len + 1) / window_len

    return WMMeanAOS(cumsum=cumsum, wcumsum=wcumsum, nancnt=nancnt, window_len=window_len, value=value)


@register_jitted(cache=True)
def wm_mean_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute weighted moving average.

    Uses `wm_mean_acc_nb` at each iteration."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    cumsum = 0.0
    wcumsum = 0.0
    nancnt = 0

    for i in range(arr.shape[0]):
        in_state = WMMeanAIS(
            i=i,
            value=arr[i],
            pre_window_value=arr[i - window] if i - window >= 0 else np.nan,
            cumsum=cumsum,
            wcumsum=wcumsum,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = wm_mean_acc_nb(in_state)
        cumsum = out_state.cumsum
        wcumsum = out_state.wcumsum
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def wm_mean_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `wm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wm_mean_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def alpha_from_com_nb(com: float) -> float:
    """Get the smoothing factor `alpha` from a center of mass."""
    return 1.0 / (1.0 + com)


@register_jitted(cache=True)
def alpha_from_span_nb(span: float) -> float:
    """Get the smoothing factor `alpha` from a span."""
    com = (span - 1) / 2.0
    return alpha_from_com_nb(com)


@register_jitted(cache=True)
def alpha_from_halflife_nb(halflife: float) -> float:
    """Get the smoothing factor `alpha` from a half-life."""
    return 1 - np.exp(-np.log(2) / halflife)


@register_jitted(cache=True)
def alpha_from_wilder_nb(period: int) -> float:
    """Get the smoothing factor `alpha` from a Wilder's period."""
    return 1 / period


@register_jitted(cache=True)
def ewm_mean_acc_nb(in_state: EWMMeanAIS) -> EWMMeanAOS:
    """Accumulator of `ewm_mean_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.EWMMeanAIS` and returns
    a state of type `vectorbtpro.generic.enums.EWMMeanAOS`."""
    i = in_state.i
    value = in_state.value
    old_wt = in_state.old_wt
    weighted_avg = in_state.weighted_avg
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    if i > 0:
        is_observation = not np.isnan(value)
        nobs += is_observation
        if not np.isnan(weighted_avg):
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != value:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * value)) / (old_wt + new_wt)
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.0
        elif is_observation:
            weighted_avg = value
    else:
        is_observation = not np.isnan(weighted_avg)
        nobs += int(is_observation)
    value = weighted_avg if (nobs >= minp) else np.nan

    return EWMMeanAOS(old_wt=old_wt, weighted_avg=weighted_avg, nobs=nobs, value=value)


@register_jitted(cache=True)
def ewm_mean_1d_nb(arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving average.

    Uses `ewm_mean_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp, adjust=adjust).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=np.float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    weighted_avg = float(arr[0])
    nobs = 0
    old_wt = 1.0

    for i in range(len(arr)):
        in_state = EWMMeanAIS(
            i=i,
            value=arr[i],
            old_wt=old_wt,
            weighted_avg=weighted_avg,
            nobs=nobs,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_mean_acc_nb(in_state)
        old_wt = out_state.old_wt
        weighted_avg = out_state.weighted_avg
        nobs = out_state.nobs
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_mean_nb(arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_mean_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def ewm_std_acc_nb(in_state: EWMStdAIS) -> EWMStdAOS:
    """Accumulator of `ewm_std_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.EWMStdAIS` and returns
    a state of type `vectorbtpro.generic.enums.EWMStdAOS`."""
    i = in_state.i
    value = in_state.value
    mean_x = in_state.mean_x
    mean_y = in_state.mean_y
    cov = in_state.cov
    sum_wt = in_state.sum_wt
    sum_wt2 = in_state.sum_wt2
    old_wt = in_state.old_wt
    nobs = in_state.nobs
    alpha = in_state.alpha
    minp = in_state.minp
    adjust = in_state.adjust

    old_wt_factor = 1.0 - alpha
    new_wt = 1.0 if adjust else alpha

    cur_x = value
    cur_y = value
    is_observation = not np.isnan(cur_x) and not np.isnan(cur_y)
    nobs += is_observation
    if i > 0:
        if not np.isnan(mean_x):
            sum_wt *= old_wt_factor
            sum_wt2 *= old_wt_factor * old_wt_factor
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt)
                cov = (
                    (old_wt * (cov + ((old_mean_x - mean_x) * (old_mean_y - mean_y))))
                    + (new_wt * ((cur_x - mean_x) * (cur_y - mean_y)))
                ) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += new_wt * new_wt
                old_wt += new_wt
                if not adjust:
                    sum_wt /= old_wt
                    sum_wt2 /= old_wt * old_wt
                    old_wt = 1.0
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y
    else:
        if not is_observation:
            mean_x = np.nan
            mean_y = np.nan

    if nobs >= minp:
        numerator = sum_wt * sum_wt
        denominator = numerator - sum_wt2
        if denominator > 0.0:
            value = (numerator / denominator) * cov
        else:
            value = np.nan
    else:
        value = np.nan

    return EWMStdAOS(
        mean_x=mean_x,
        mean_y=mean_y,
        cov=cov,
        sum_wt=sum_wt,
        sum_wt2=sum_wt2,
        old_wt=old_wt,
        nobs=nobs,
        value=value,
    )


@register_jitted(cache=True)
def ewm_std_1d_nb(arr: tp.Array1d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array1d:
    """Compute exponential weighted moving standard deviation.

    Uses `ewm_std_acc_nb` at each iteration.

    Numba equivalent to `pd.Series(arr).ewm(span=span, min_periods=minp).std()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    out = np.empty(len(arr), dtype=np.float_)
    if len(arr) == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1.0 / (1.0 + com)
    mean_x = float(arr[0])
    mean_y = float(arr[0])
    nobs = 0
    cov = 0.0
    sum_wt = 1.0
    sum_wt2 = 1.0
    old_wt = 1.0

    for i in range(len(arr)):
        in_state = EWMStdAIS(
            i=i,
            value=arr[i],
            mean_x=mean_x,
            mean_y=mean_y,
            cov=cov,
            sum_wt=sum_wt,
            sum_wt2=sum_wt2,
            old_wt=old_wt,
            nobs=nobs,
            alpha=alpha,
            minp=minp,
            adjust=adjust,
        )
        out_state = ewm_std_acc_nb(in_state)
        mean_x = out_state.mean_x
        mean_y = out_state.mean_y
        cov = out_state.cov
        sum_wt = out_state.sum_wt
        sum_wt2 = out_state.sum_wt2
        old_wt = out_state.old_wt
        nobs = out_state.nobs
        out[i] = out_state.value

    return np.sqrt(out)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), span=None, minp=None, adjust=None, ddof=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def ewm_std_nb(arr: tp.Array2d, span: int, minp: tp.Optional[int] = None, adjust: bool = False) -> tp.Array2d:
    """2-dim version of `ewm_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = ewm_std_1d_nb(arr[:, col], span, minp=minp, adjust=adjust)
    return out


@register_jitted(cache=True)
def wwm_mean_1d_nb(arr: tp.Array1d, period: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving average."""
    if minp is None:
        minp = period
    return ewm_mean_1d_nb(arr, 2 * period - 1, minp=minp, adjust=False)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_mean_nb(arr: tp.Array2d, period: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `wwm_mean_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_mean_1d_nb(arr[:, col], period, minp=minp)
    return out


@register_jitted(cache=True)
def wwm_std_1d_nb(arr: tp.Array1d, period: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute Wilder's exponential weighted moving standard deviation."""
    if minp is None:
        minp = period
    return ewm_std_1d_nb(arr, 2 * period - 1, minp=minp, adjust=False)


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), period=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def wwm_std_nb(arr: tp.Array2d, period: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `wwm_std_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = wwm_std_1d_nb(arr[:, col], period, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_cov_acc_nb(in_state: RollCovAIS) -> RollCovAOS:
    """Accumulator of `rolling_cov_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollCovAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollCovAOS`."""
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp
    ddof = in_state.ddof

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp or window_len == ddof:
        value = np.nan
    else:
        window_prod_mean = cumsum_prod / (window_len - ddof)
        window_mean1 = cumsum1 / window_len
        window_mean2 = cumsum2 / window_len
        window_mean_prod = window_mean1 * window_mean2 * window_len / (window_len - ddof)
        value = window_prod_mean - window_mean_prod

    return RollCovAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_cov_1d_nb(
    arr1: tp.Array1d,
    arr2: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array1d:
    """Compute rolling covariance.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).cov(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCovAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
            ddof=ddof,
        )
        out_state = rolling_cov_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None, ddof=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_cov_nb(
    arr1: tp.Array2d,
    arr2: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
) -> tp.Array2d:
    """2-dim version of `rolling_cov_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_cov_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp, ddof=ddof)
    return out


@register_jitted(cache=True)
def rolling_corr_acc_nb(in_state: RollCorrAIS) -> RollCorrAOS:
    """Accumulator of `rolling_corr_1d_nb`.

    Takes a state of type `vectorbtpro.generic.enums.RollCorrAIS` and returns
    a state of type `vectorbtpro.generic.enums.RollCorrAOS`."""
    i = in_state.i
    value1 = in_state.value1
    value2 = in_state.value2
    pre_window_value1 = in_state.pre_window_value1
    pre_window_value2 = in_state.pre_window_value2
    cumsum1 = in_state.cumsum1
    cumsum2 = in_state.cumsum2
    cumsum_sq1 = in_state.cumsum_sq1
    cumsum_sq2 = in_state.cumsum_sq2
    cumsum_prod = in_state.cumsum_prod
    nancnt = in_state.nancnt
    window = in_state.window
    minp = in_state.minp

    if np.isnan(value1) or np.isnan(value2):
        nancnt = nancnt + 1
    else:
        cumsum1 = cumsum1 + value1
        cumsum2 = cumsum2 + value2
        cumsum_sq1 = cumsum_sq1 + value1**2
        cumsum_sq2 = cumsum_sq2 + value2**2
        cumsum_prod = cumsum_prod + value1 * value2
    if i < window:
        window_len = i + 1 - nancnt
    else:
        if np.isnan(pre_window_value1) or np.isnan(pre_window_value2):
            nancnt = nancnt - 1
        else:
            cumsum1 = cumsum1 - pre_window_value1
            cumsum2 = cumsum2 - pre_window_value2
            cumsum_sq1 = cumsum_sq1 - pre_window_value1**2
            cumsum_sq2 = cumsum_sq2 - pre_window_value2**2
            cumsum_prod = cumsum_prod - pre_window_value1 * pre_window_value2
        window_len = window - nancnt
    if window_len < minp:
        value = np.nan
    else:
        nom = window_len * cumsum_prod - cumsum1 * cumsum2
        denom1 = np.sqrt(window_len * cumsum_sq1 - cumsum1**2)
        denom2 = np.sqrt(window_len * cumsum_sq2 - cumsum2**2)
        denom = denom1 * denom2
        if denom == 0:
            value = np.nan
        else:
            value = nom / denom

    return RollCorrAOS(
        cumsum1=cumsum1,
        cumsum2=cumsum2,
        cumsum_sq1=cumsum_sq1,
        cumsum_sq2=cumsum_sq2,
        cumsum_prod=cumsum_prod,
        nancnt=nancnt,
        window_len=window_len,
        value=value,
    )


@register_jitted(cache=True)
def rolling_corr_1d_nb(arr1: tp.Array1d, arr2: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling correlation coefficient.

    Numba equivalent to `pd.Series(arr1).rolling(window, min_periods=minp).corr(arr2)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr1, dtype=np.float_)
    cumsum1 = 0.0
    cumsum2 = 0.0
    cumsum_sq1 = 0.0
    cumsum_sq2 = 0.0
    cumsum_prod = 0.0
    nancnt = 0

    for i in range(arr1.shape[0]):
        in_state = RollCorrAIS(
            i=i,
            value1=arr1[i],
            value2=arr2[i],
            pre_window_value1=arr1[i - window] if i - window >= 0 else np.nan,
            pre_window_value2=arr2[i - window] if i - window >= 0 else np.nan,
            cumsum1=cumsum1,
            cumsum2=cumsum2,
            cumsum_sq1=cumsum_sq1,
            cumsum_sq2=cumsum_sq2,
            cumsum_prod=cumsum_prod,
            nancnt=nancnt,
            window=window,
            minp=minp,
        )
        out_state = rolling_corr_acc_nb(in_state)
        cumsum1 = out_state.cumsum1
        cumsum2 = out_state.cumsum2
        cumsum_sq1 = out_state.cumsum_sq1
        cumsum_sq2 = out_state.cumsum_sq2
        cumsum_prod = out_state.cumsum_prod
        nancnt = out_state.nancnt
        out[i] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr1", axis=1),
    arg_take_spec=dict(arr1=ch.ArraySlicer(axis=1), arr2=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_corr_nb(arr1: tp.Array2d, arr2: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_corr_1d_nb`."""
    out = np.empty_like(arr1, dtype=np.float_)
    for col in prange(arr1.shape[1]):
        out[:, col] = rolling_corr_1d_nb(arr1[:, col], arr2[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_rank_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int], pct: bool = False) -> tp.Array1d:
    """Rolling version of `rank_1d_nb`."""
    if minp is None:
        minp = window
    out = np.empty_like(arr, dtype=np.float_)
    nancnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            nancnt = nancnt + 1
        if i < window:
            valid_cnt = i + 1 - nancnt
        else:
            if np.isnan(arr[i - window]):
                nancnt = nancnt - 1
            valid_cnt = window - nancnt
        if valid_cnt < minp:
            out[i] = np.nan
        else:
            from_i = max(0, i + 1 - window)
            to_i = i + 1
            window_a = arr[from_i:to_i]
            out[i] = rank_1d_nb(window_a, pct=pct)[-1]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, pct=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_rank_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, pct: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_rank_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_rank_1d_nb(arr[:, col], window, minp=minp, pct=pct)
    return out


@register_jitted(cache=True)
def rolling_min_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling min.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_min_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_min_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_max_1d_nb(arr: tp.Array1d, window: int, minp: tp.Optional[int] = None) -> tp.Array1d:
    """Compute rolling max.

    Numba equivalent to `pd.Series(arr).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.float_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        cnt = 0
        for j in range(from_i, to_i):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_max_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None) -> tp.Array2d:
    """2-dim version of `rolling_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_max_1d_nb(arr[:, col], window, minp=minp)
    return out


@register_jitted(cache=True)
def rolling_argmin_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Compute rolling min index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        minv = arr[from_i]
        if local:
            mini = 0
        else:
            mini = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(minv) or arr[j] < minv:
                minv = arr[j]
                if local:
                    mini = k
                else:
                    mini = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = mini
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmin_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmin_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmin_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


@register_jitted(cache=True)
def rolling_argmax_1d_nb(
    arr: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    local: bool = False,
) -> tp.Array1d:
    """Compute rolling max index."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(arr, dtype=np.int_)
    for i in range(arr.shape[0]):
        from_i = max(i - window + 1, 0)
        to_i = i + 1
        maxv = arr[from_i]
        if local:
            maxi = 0
        else:
            maxi = from_i
        cnt = 0
        for k, j in enumerate(range(from_i, to_i)):
            if np.isnan(arr[j]):
                continue
            if np.isnan(maxv) or arr[j] > maxv:
                maxv = arr[j]
                if local:
                    maxi = k
                else:
                    maxi = j
            cnt += 1
        if cnt < minp:
            out[i] = -1
        else:
            out[i] = maxi
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), window=None, minp=None, local=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_argmax_nb(arr: tp.Array2d, window: int, minp: tp.Optional[int] = None, local: bool = False) -> tp.Array2d:
    """2-dim version of `rolling_argmax_1d_nb`."""
    out = np.empty_like(arr, dtype=np.int_)
    for col in prange(arr.shape[1]):
        out[:, col] = rolling_argmax_1d_nb(arr[:, col], window, minp=minp, local=local)
    return out


# ############# Expanding functions ############# #


@register_jitted(cache=True)
def expanding_min_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding min.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).min()`."""
    out = np.empty_like(arr, dtype=np.float_)
    minv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(minv) or arr[i] < minv:
            minv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_min_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_min_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_min_1d_nb(arr[:, col], minp=minp)
    return out


@register_jitted(cache=True)
def expanding_max_1d_nb(arr: tp.Array1d, minp: int = 1) -> tp.Array1d:
    """Compute expanding max.

    Numba equivalent to `pd.Series(arr).expanding(min_periods=minp).max()`."""
    out = np.empty_like(arr, dtype=np.float_)
    maxv = arr[0]
    cnt = 0
    for i in range(arr.shape[0]):
        if np.isnan(maxv) or arr[i] > maxv:
            maxv = arr[i]
        if not np.isnan(arr[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), minp=None),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def expanding_max_nb(arr: tp.Array2d, minp: int = 1) -> tp.Array2d:
    """2-dim version of `expanding_max_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = expanding_max_1d_nb(arr[:, col], minp=minp)
    return out
