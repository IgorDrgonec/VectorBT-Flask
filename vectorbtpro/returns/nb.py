# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for returns.

Provides an arsenal of Numba-compiled functions that are used by accessors and for measuring
portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

```pycon
>>> import numpy as np
>>> import vectorbtpro as vbt

>>> price = np.array([1.1, 1.2, 1.3, 1.2, 1.1])
>>> returns = vbt.generic.nb.pct_change_1d_nb(price)

>>> # vectorbtpro.returns.nb.cum_returns_1d_nb
>>> vbt.returns.nb.cum_returns_1d_nb(returns, 0)
array([0., 0.09090909, 0.18181818, 0.09090909, 0.])
```

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb


@register_jitted(cache=True)
def get_return_nb(input_value: float, output_value: float) -> float:
    """Calculate return from input and output value."""
    if input_value == 0:
        if output_value == 0:
            return 0.
        return np.inf * np.sign(output_value)
    return_value = add_nb(output_value, -input_value) / input_value
    if input_value < 0:
        return_value *= -1
    return return_value


@register_jitted(cache=True)
def returns_1d_nb(arr: tp.Array1d, init_value: float) -> tp.Array1d:
    """Calculate returns."""
    out = np.empty(arr.shape, dtype=np.float_)
    input_value = init_value
    for i in range(out.shape[0]):
        output_value = arr[i]
        out[i] = get_return_nb(input_value, output_value)
        input_value = output_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='arr', axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        init_value=ch.ArraySlicer(axis=0)
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def returns_nb(arr: tp.Array2d, init_value: tp.Array1d) -> tp.Array2d:
    """2-dim version of `returns_1d_nb`."""
    out = np.empty(arr.shape, dtype=np.float_)
    for col in prange(out.shape[1]):
        out[:, col] = returns_1d_nb(arr[:, col], init_value[col])
    return out


@register_jitted(cache=True)
def cum_returns_1d_nb(rets: tp.Array1d, start_value: float) -> tp.Array1d:
    """Cumulative returns."""
    out = np.empty_like(rets, dtype=np.float_)
    cumprod = 1
    for i in range(rets.shape[0]):
        if not np.isnan(rets[i]):
            cumprod *= rets[i] + 1
        out[i] = cumprod
    if start_value == 0:
        return out - 1.
    return out * start_value


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        start_value=None
    ),
    merge_func=base_ch.column_stack
)
@register_jitted(cache=True, tags={'can_parallel'})
def cum_returns_nb(rets: tp.Array2d, start_value: float) -> tp.Array2d:
    """2-dim version of `cum_returns_1d_nb`."""
    out = np.empty_like(rets, dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[:, col] = cum_returns_1d_nb(rets[:, col], start_value)
    return out


@register_jitted(cache=True)
def cum_returns_final_1d_nb(rets: tp.Array1d, start_value: float = 0.) -> float:
    """Total return."""
    out = np.nan
    for i in range(rets.shape[0]):
        if not np.isnan(rets[i]):
            if np.isnan(out):
                out = 1.
            out *= rets[i] + 1.
    if start_value == 0:
        return out - 1.
    return out * start_value


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        start_value=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def cum_returns_final_nb(rets: tp.Array2d, start_value: float = 0.) -> tp.Array1d:
    """2-dim version of `cum_returns_final_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = cum_returns_final_1d_nb(rets[:, col], start_value)
    return out


@register_jitted(cache=True)
def annualized_return_1d_nb(rets: tp.Array1d, ann_factor: float, period: tp.Optional[float] = None) -> float:
    """Annualized total return.

    This is equivalent to the compound annual growth rate."""
    if period is None:
        period = rets.shape[0]
    cum_return = cum_returns_final_1d_nb(rets, 1.)
    return cum_return ** (ann_factor / period) - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def annualized_return_nb(rets: tp.Array2d, ann_factor: float, period: tp.Optional[float] = None) -> tp.Array1d:
    """2-dim version of `annualized_return_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = annualized_return_1d_nb(rets[:, col], ann_factor, period=period)
    return out


@register_jitted(cache=True)
def annualized_volatility_1d_nb(rets: tp.Array1d,
                                ann_factor: float,
                                levy_alpha: float = 2.0,
                                ddof: int = 0) -> float:
    """Annualized volatility of a strategy."""
    return generic_nb.nanstd_1d_nb(rets, ddof) * ann_factor ** (1.0 / levy_alpha)


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        levy_alpha=None,
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def annualized_volatility_nb(rets: tp.Array2d,
                             ann_factor: float,
                             levy_alpha: float = 2.0,
                             ddof: int = 0) -> tp.Array1d:
    """2-dim version of `annualized_volatility_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = annualized_volatility_1d_nb(rets[:, col], ann_factor, levy_alpha, ddof)
    return out


@register_jitted(cache=True)
def max_drawdown_1d_nb(rets: tp.Array1d) -> float:
    """Total maximum drawdown (MDD)."""
    cum_ret = np.nan
    value_max = 1.
    out = 0.
    for i in range(rets.shape[0]):
        if not np.isnan(rets[i]):
            if np.isnan(cum_ret):
                cum_ret = 1.
            cum_ret *= rets[i] + 1.
        if cum_ret > value_max:
            value_max = cum_ret
        elif cum_ret < value_max:
            dd = cum_ret / value_max - 1
            if dd < out:
                out = dd
    if np.isnan(cum_ret):
        return np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def max_drawdown_nb(rets: tp.Array2d) -> tp.Array1d:
    """2-dim version of `max_drawdown_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = max_drawdown_1d_nb(rets[:, col])
    return out


@register_jitted(cache=True)
def calmar_ratio_1d_nb(rets: tp.Array1d, ann_factor: float, period: tp.Optional[float] = None) -> float:
    """Calmar ratio, or drawdown ratio, of a strategy."""
    max_drawdown = max_drawdown_1d_nb(rets)
    if max_drawdown == 0:
        return np.nan
    annualized_return = annualized_return_1d_nb(rets, ann_factor, period=period)
    if max_drawdown == 0:
        return np.inf
    return annualized_return / np.abs(max_drawdown)


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def calmar_ratio_nb(rets: tp.Array2d, ann_factor: float, period: tp.Optional[float] = None) -> tp.Array1d:
    """2-dim version of `calmar_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = calmar_ratio_1d_nb(rets[:, col], ann_factor, period=period)
    return out


@register_jitted(cache=True)
def deannualized_return_nb(ret: float, ann_factor: float) -> float:
    """Deannualized return."""
    if ann_factor == 1:
        return ret
    if ann_factor <= -1:
        return np.nan
    return (1 + ret) ** (1. / ann_factor) - 1


@register_jitted(cache=True)
def omega_ratio_1d_nb(adj_rets: tp.Array1d) -> float:
    """Omega ratio of a strategy."""
    numer = 0.
    denom = 0.
    for i in range(adj_rets.shape[0]):
        ret = adj_rets[i]
        if ret > 0:
            numer += ret
        elif ret < 0:
            denom -= ret
    if denom == 0:
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def omega_ratio_nb(adj_rets: tp.Array2d) -> tp.Array1d:
    """2-dim version of `omega_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = omega_ratio_1d_nb(adj_rets[:, col])
    return out


@register_jitted(cache=True)
def sharpe_ratio_1d_nb(adj_rets: tp.Array1d,
                       ann_factor: float,
                       ddof: int = 0) -> float:
    """Sharpe ratio of a strategy."""
    mean = np.nanmean(adj_rets)
    std = generic_nb.nanstd_1d_nb(adj_rets, ddof)
    if std == 0:
        return np.inf
    return mean / std * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def sharpe_ratio_nb(adj_rets: tp.Array2d,
                    ann_factor: float,
                    ddof: int = 0) -> tp.Array1d:
    """2-dim version of `sharpe_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = sharpe_ratio_1d_nb(adj_rets[:, col], ann_factor, ddof)
    return out


@register_jitted(cache=True)
def downside_risk_1d_nb(adj_rets: tp.Array1d, ann_factor: float) -> float:
    """Downside deviation below a threshold."""
    cnt = 0
    adj_ret_sqrd_sum = np.nan
    for i in range(adj_rets.shape[0]):
        if not np.isnan(adj_rets[i]):
            cnt += 1
            if np.isnan(adj_ret_sqrd_sum):
                adj_ret_sqrd_sum = 0.
            if adj_rets[i] <= 0:
                adj_ret_sqrd_sum += adj_rets[i] ** 2
    adj_ret_sqrd_mean = adj_ret_sqrd_sum / cnt
    return np.sqrt(adj_ret_sqrd_mean) * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1),
        ann_factor=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def downside_risk_nb(adj_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `downside_risk_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = downside_risk_1d_nb(adj_rets[:, col], ann_factor)
    return out


@register_jitted(cache=True)
def sortino_ratio_1d_nb(adj_rets: tp.Array1d, ann_factor: float) -> float:
    """Sortino ratio of a strategy."""
    avg_annualized_return = np.nanmean(adj_rets) * ann_factor
    downside_risk = downside_risk_1d_nb(adj_rets, ann_factor)
    if downside_risk == 0:
        return np.inf
    return avg_annualized_return / downside_risk


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1),
        ann_factor=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def sortino_ratio_nb(adj_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `sortino_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = sortino_ratio_1d_nb(adj_rets[:, col], ann_factor)
    return out


@register_jitted(cache=True)
def information_ratio_1d_nb(adj_rets: tp.Array1d, ddof: int = 0) -> float:
    """Information ratio of a strategy."""
    mean = np.nanmean(adj_rets)
    std = generic_nb.nanstd_1d_nb(adj_rets, ddof)
    if std == 0:
        return np.inf
    return mean / std


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1),
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def information_ratio_nb(adj_rets: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `information_ratio_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = information_ratio_1d_nb(adj_rets[:, col], ddof)
    return out


@register_jitted(cache=True)
def beta_1d_nb(rets: tp.Array1d, benchmark_rets: tp.Array1d, ddof: int = 0) -> float:
    """Beta."""
    cov = generic_nb.nancov_1d_nb(rets, benchmark_rets, ddof=ddof)
    var = generic_nb.nanvar_1d_nb(benchmark_rets, ddof=ddof)
    if var == 0:
        return np.inf
    return cov / var


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        benchmark_rets=ch.ArraySlicer(axis=1),
        ddof=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def beta_nb(rets: tp.Array2d, benchmark_rets: tp.Array2d, ddof: int = 0) -> tp.Array1d:
    """2-dim version of `beta_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = beta_1d_nb(rets[:, col], benchmark_rets[:, col], ddof=ddof)
    return out


@register_jitted
def beta_rollmeta_nb(from_i: int, to_i: int, col: int, rets: tp.Array2d,
                     benchmark_rets: tp.Array1d, ddof: int = 0) -> float:
    """Rolling apply meta function based on `beta_1d_nb`."""
    return beta_1d_nb(rets[from_i:to_i, col], benchmark_rets[from_i:to_i, col], ddof)


@register_jitted(cache=True)
def alpha_1d_nb(adj_rets: tp.Array1d,
                adj_benchmark_rets: tp.Array1d,
                ann_factor: float) -> float:
    """Annualized alpha."""
    beta = beta_1d_nb(adj_rets, adj_benchmark_rets)
    return (np.nanmean(adj_rets) - beta * np.nanmean(adj_benchmark_rets) + 1) ** ann_factor - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query='adj_rets', axis=1),
    arg_take_spec=dict(
        adj_rets=ch.ArraySlicer(axis=1),
        adj_benchmark_rets=ch.ArraySlicer(axis=1),
        ann_factor=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def alpha_nb(adj_rets: tp.Array2d, adj_benchmark_rets: tp.Array2d, ann_factor: float) -> tp.Array1d:
    """2-dim version of `alpha_1d_nb`."""
    out = np.empty(adj_rets.shape[1], dtype=np.float_)
    for col in prange(adj_rets.shape[1]):
        out[col] = alpha_1d_nb(adj_rets[:, col], adj_benchmark_rets[:, col], ann_factor)
    return out


@register_jitted
def alpha_rollmeta_nb(from_i: int, to_i: int, col: int, adj_rets: tp.Array2d,
                      adj_benchmark_rets: tp.Array1d, ann_factor: float) -> float:
    """Rolling apply meta function based on `alpha_1d_nb`."""
    return alpha_1d_nb(adj_rets[from_i:to_i, col], adj_benchmark_rets[from_i:to_i, col], ann_factor)


@register_jitted(cache=True)
def tail_ratio_1d_nb(rets: tp.Array1d) -> float:
    """Ratio between the right (95%) and left tail (5%)."""
    perc_95 = np.abs(np.nanpercentile(rets, 95))
    perc_5 = np.abs(np.nanpercentile(rets, 5))
    if perc_5 == 0:
        return np.inf
    return perc_95 / perc_5


@register_jitted(cache=True)
def tail_ratio_noarr_1d_nb(rets: tp.Array1d) -> float:
    """`tail_ratio_1d_nb` that does not allocate any arrays."""
    perc_95 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(rets, 95))
    perc_5 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(rets, 5))
    if perc_5 == 0:
        return np.inf
    return perc_95 / perc_5


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1)
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def tail_ratio_nb(rets: tp.Array2d) -> tp.Array1d:
    """2-dim version of `tail_ratio_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = tail_ratio_1d_nb(rets[:, col])
    return out


@register_jitted(cache=True)
def value_at_risk_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """Value at risk (VaR) of a returns stream."""
    return np.nanpercentile(rets, 100 * cutoff)


@register_jitted(cache=True)
def value_at_risk_noarr_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """`value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpercentile_noarr_1d_nb(rets, 100 * cutoff)


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        cutoff=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def value_at_risk_nb(rets: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """2-dim version of `value_at_risk_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = value_at_risk_1d_nb(rets[:, col], cutoff)
    return out


@register_jitted(cache=True)
def cond_value_at_risk_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """Conditional value at risk (CVaR) of a returns stream."""
    cutoff_index = int((len(rets) - 1) * cutoff)
    return np.mean(np.partition(rets, cutoff_index)[:cutoff_index + 1])


@register_jitted(cache=True)
def cond_value_at_risk_noarr_1d_nb(rets: tp.Array1d, cutoff: float = 0.05) -> float:
    """`cond_value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpartition_mean_noarr_1d_nb(rets, cutoff * 100)


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        cutoff=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def cond_value_at_risk_nb(rets: tp.Array2d, cutoff: float = 0.05) -> tp.Array1d:
    """2-dim version of `cond_value_at_risk_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = cond_value_at_risk_1d_nb(rets[:, col], cutoff)
    return out


@register_jitted(cache=True)
def capture_1d_nb(rets: tp.Array1d, benchmark_rets: tp.Array1d,
                  ann_factor: float, period: tp.Optional[float] = None) -> float:
    """Capture ratio."""
    annualized_return1 = annualized_return_1d_nb(rets, ann_factor, period=period)
    annualized_return2 = annualized_return_1d_nb(benchmark_rets, ann_factor, period=period)
    if annualized_return2 == 0:
        return np.inf
    return annualized_return1 / annualized_return2


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        benchmark_rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def capture_nb(rets: tp.Array2d, benchmark_rets: tp.Array2d,
               ann_factor: float, period: tp.Optional[float] = None) -> tp.Array1d:
    """2-dim version of `capture_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = capture_1d_nb(rets[:, col], benchmark_rets[:, col], ann_factor, period=period)
    return out


@register_jitted
def capture_rollmeta_nb(from_i: int, to_i: int, col: int, rets: tp.Array2d,
                        benchmark_rets: tp.Array1d, ann_factor: float,
                        period: tp.Optional[float] = None) -> float:
    """Rolling apply meta function based on `capture_1d_nb`."""
    return capture_1d_nb(rets[from_i:to_i, col], benchmark_rets[from_i:to_i, col], ann_factor, period=period)


@register_jitted(cache=True)
def up_capture_1d_nb(rets: tp.Array1d, benchmark_rets: tp.Array1d,
                     ann_factor: float, period: tp.Optional[float] = None) -> float:
    """Capture ratio for periods when the benchmark return is positive."""
    if period is None:
        period = rets.shape[0]

    def _annualized_pos_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if np.isnan(ann_ret):
                    ann_ret = 1.
                if a[i] > 0:
                    ann_ret *= a[i] + 1.
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_pos_return(rets)
    annualized_benchmark_return = _annualized_pos_return(benchmark_rets)
    if annualized_benchmark_return == 0:
        return np.inf
    return annualized_return / annualized_benchmark_return


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        benchmark_rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def up_capture_nb(rets: tp.Array2d, benchmark_rets: tp.Array2d,
                  ann_factor: float, period: tp.Optional[float] = None) -> tp.Array1d:
    """2-dim version of `up_capture_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = up_capture_1d_nb(rets[:, col], benchmark_rets[:, col], ann_factor, period=period)
    return out


@register_jitted
def up_capture_rollmeta_nb(from_i: int, to_i: int, col: int, rets: tp.Array2d,
                           benchmark_rets: tp.Array1d, ann_factor: float,
                           period: tp.Optional[float] = None) -> float:
    """Rolling apply meta function based on `up_capture_1d_nb`."""
    return up_capture_1d_nb(rets[from_i:to_i, col], benchmark_rets[from_i:to_i, col], ann_factor, period=period)


@register_jitted(cache=True)
def down_capture_1d_nb(rets: tp.Array1d, benchmark_rets: tp.Array1d,
                       ann_factor: float, period: tp.Optional[float] = None) -> float:
    """Capture ratio for periods when the benchmark return is negative."""
    if period is None:
        period = rets.shape[0]

    def _annualized_neg_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if np.isnan(ann_ret):
                    ann_ret = 1.
                if a[i] < 0:
                    ann_ret *= a[i] + 1.
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_neg_return(rets)
    annualized_benchmark_return = _annualized_neg_return(benchmark_rets)
    if annualized_benchmark_return == 0:
        return np.inf
    return annualized_return / annualized_benchmark_return


@register_chunkable(
    size=ch.ArraySizer(arg_query='rets', axis=1),
    arg_take_spec=dict(
        rets=ch.ArraySlicer(axis=1),
        benchmark_rets=ch.ArraySlicer(axis=1),
        ann_factor=None,
        period=None
    ),
    merge_func=base_ch.concat
)
@register_jitted(cache=True, tags={'can_parallel'})
def down_capture_nb(rets: tp.Array2d, benchmark_rets: tp.Array2d,
                    ann_factor: float, period: tp.Optional[float] = None) -> tp.Array1d:
    """2-dim version of `down_capture_1d_nb`."""
    out = np.empty(rets.shape[1], dtype=np.float_)
    for col in prange(rets.shape[1]):
        out[col] = down_capture_1d_nb(rets[:, col], benchmark_rets[:, col], ann_factor, period=period)
    return out


@register_jitted
def down_capture_rollmeta_nb(from_i: int, to_i: int, col: int, rets: tp.Array2d,
                             benchmark_rets: tp.Array1d, ann_factor: float,
                             period: tp.Optional[float] = None) -> float:
    """Rolling apply meta function based on `down_capture_1d_nb`."""
    return down_capture_1d_nb(rets[from_i:to_i, col], benchmark_rets[from_i:to_i, col], ann_factor, period=period)
