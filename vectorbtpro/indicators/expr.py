# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Functions and config for evaluating indicator expressions."""

import math

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.grouping import Grouper
from vectorbtpro.generic.nb import (
    fshift_nb,
    diff_nb,
    rank_nb,
    rolling_sum_nb,
    rolling_mean_nb,
    rolling_std_nb,
    wm_mean_nb,
    rolling_rank_nb,
    rolling_prod_nb,
    rolling_min_nb,
    rolling_max_nb,
    rolling_argmin_nb,
    rolling_argmax_nb,
    rolling_cov_nb,
    rolling_corr_nb,
    demean_nb
)
from vectorbtpro.ohlcv.nb import vwap_nb
from vectorbtpro.returns.nb import returns_nb
from vectorbtpro.utils.config import HybridConfig


# ############# Delay ############# #


def delay(x: tp.Array2d, d) -> tp.Array2d:
    """Value of `x` `d` days ago."""
    return fshift_nb(x, math.floor(d))


def delta(x: tp.Array2d, d) -> tp.Array2d:
    """Today’s value of `x` minus the value of `x` `d` days ago."""
    return diff_nb(x, math.floor(d))


# ############# Rescale ############# #


def rescale(x: tp.Array2d) -> tp.Array2d:
    """Rescale `x` such that `sum(abs(x)) = 1`."""
    return np.divide(x, np.sum(np.abs(x), axis=1))


# ############# Cross-section ############# #


def cs_rank(x: tp.Array2d) -> tp.Array2d:
    """Rank cross-sectionally."""
    return rank_nb(x.T, pct=True).T


def cs_demean(x: tp.Array2d, g, mapping: tp.KwargsLike = None) -> tp.Array2d:
    """Demean `x` against groups `g` cross-sectionally."""
    group_map = Grouper(mapping['wrapper'].columns, g).get_group_map()
    return demean_nb(x, group_map)


# ############# Rolling ############# #


def ts_min(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling min."""
    return rolling_min_nb(x, math.floor(d))


def ts_max(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling max."""
    return rolling_max_nb(x, math.floor(d))


def ts_argmin(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling argmin."""
    return np.add(rolling_argmin_nb(x, math.floor(d), local=True), 1)


def ts_argmax(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling argmax."""
    return np.add(rolling_argmax_nb(x, math.floor(d), local=True), 1)


def ts_rank(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling rank."""
    return rolling_rank_nb(x, math.floor(d), pct=True)


def ts_sum(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling sum."""
    return rolling_sum_nb(x, math.floor(d))


def ts_product(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling product."""
    return rolling_prod_nb(x, math.floor(d))


def ts_mean(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling mean."""
    return rolling_mean_nb(x, math.floor(d))


def ts_weighted_mean(x: tp.Array2d, d) -> tp.Array2d:
    """Weighted moving average over the past `d` days with linearly decaying weight."""
    return wm_mean_nb(x, math.floor(d))


def ts_std(x: tp.Array2d, d) -> tp.Array2d:
    """Return the rolling standard deviation."""
    return rolling_std_nb(x, math.floor(d))


def ts_corr(x: tp.Array2d, y, d) -> tp.Array2d:
    """Time-serial correlation of `x` and `y` for the past `d` days."""
    return rolling_corr_nb(x, y, math.floor(d))


def ts_cov(arr, d) -> tp.Array2d:
    """Time-serial covariance of `x` and `y` for the past `d` days."""
    return rolling_cov_nb(arr, math.floor(d))


def adv(d, mapping: tp.KwargsLike = None) -> tp.Array2d:
    """Average daily dollar volume for the past `d` days."""
    return ts_mean(mapping['volume'], math.floor(d))


# ############# Substitutions ############# #


def returns(mapping: tp.KwargsLike = None) -> tp.Array2d:
    """Daily close-to-close returns."""
    return returns_nb(mapping['close'])


def vwap(mapping: tp.KwargsLike = None) -> tp.Array2d:
    """VWAP."""
    return vwap_nb(mapping['high'], mapping['low'], mapping['volume'])


def cap(mapping: tp.KwargsLike = None) -> tp.Array2d:
    """Market capitalization."""
    return mapping['close'] * mapping['volume']


# ############# Configs ############# #

__pdoc__ = {}

expr_func_config = HybridConfig(
    dict(
        delay=dict(func=delay),
        delta=dict(func=delta),
        rescale=dict(func=rescale),
        cs_rank=dict(func=cs_rank),
        cs_demean=dict(func=cs_demean),
        ts_min=dict(func=ts_min),
        ts_max=dict(func=ts_max),
        ts_argmin=dict(func=ts_argmin),
        ts_argmax=dict(func=ts_argmax),
        ts_rank=dict(func=ts_rank),
        ts_sum=dict(func=ts_sum),
        ts_product=dict(func=ts_product),
        ts_mean=dict(func=ts_mean),
        ts_weighted_mean=dict(func=ts_weighted_mean),
        ts_std=dict(func=ts_std),
        ts_corr=dict(func=ts_corr),
        ts_cov=dict(func=ts_cov),
        adv=dict(func=adv, magnet_input_names=['volume'])
    )
)
"""_"""

__pdoc__['expr_func_config'] = f"""Config for functions used in indicator expressions.

Can be modified.

```python
{expr_func_config.prettify()}
```
"""

expr_res_func_config = HybridConfig(
    dict(
        returns=dict(func=returns, magnet_input_names=['close']),
        vwap=dict(func=vwap, magnet_input_names=['high', 'low', 'volume']),
        cap=dict(func=cap, magnet_input_names=['close', 'volume'])
    )
)
"""_"""

__pdoc__['expr_res_func_config'] = f"""Config for resolvable functions used in indicator expressions.

Can be modified.

```python
{expr_res_func_config.prettify()}
```
"""
