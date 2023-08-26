# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `GBMOHLCData`."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.data import nb
from vectorbtpro.ohlcv import nb as ohlcv_nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.data.custom.synthetic import SyntheticData

__all__ = [
    "GBMOHLCData",
]

__pdoc__ = {}


class GBMOHLCData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_gbm_data_1d_nb`
    and then resampled using `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb`."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.gbm_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        dt: tp.Optional[float] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SymbolData:
        """Generate a symbol.

        Args:
            symbol (hashable): Symbol.
            index (pd.Index): Pandas index.
            n_ticks (int or array_like): Number of ticks per bar.

                Flexible argument. Can be a template with a context containing `symbol` and `index`.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            dt (float): Time change (one period of time).
            seed (int): Set seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (dict): Template context.

        For defaults, see `custom.gbm` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        gbm_cfg = cls.get_settings(key_id="custom")

        if n_ticks is None:
            n_ticks = gbm_cfg["n_ticks"]
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, sub_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        if start_value is None:
            start_value = gbm_cfg["start_value"]
        if mean is None:
            mean = gbm_cfg["mean"]
        if std is None:
            std = gbm_cfg["std"]
        if dt is None:
            dt = gbm_cfg["dt"]
        if seed is None:
            seed = gbm_cfg["seed"]
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_gbm_data_1d_nb, jitted)
        ticks = func(
            np.sum(n_ticks),
            start_value=start_value,
            mean=mean,
            std=std,
            dt=dt,
        )
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)
