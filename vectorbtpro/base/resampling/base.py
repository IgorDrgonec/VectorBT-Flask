# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Base classes and functions for resampling."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.datetime_ import freq_to_timedelta
from vectorbtpro.base.resampling import nb
from vectorbtpro.registries.jit_registry import jit_reg


ResamplerT = tp.TypeVar("ResamplerT", bound="Resampler")


class Resampler(Configured):
    """Class that exposes methods to resample index."""

    def __init__(
        self,
        from_index: tp.IndexLike,
        to_index: tp.IndexLike,
        from_freq: tp.Optional[tp.FrequencyLike] = None,
        to_freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> None:
        if not isinstance(from_index, pd.DatetimeIndex):
            from_index = pd.DatetimeIndex(from_index)
        if not isinstance(to_index, pd.DatetimeIndex):
            to_index = pd.DatetimeIndex(to_index)
        if from_freq is not None and not isinstance(from_freq, (pd.Timedelta, pd.DateOffset)):
            from_freq = freq_to_timedelta(from_freq)
        if to_freq is not None and not isinstance(to_freq, (pd.Timedelta, pd.DateOffset)):
            to_freq = freq_to_timedelta(to_freq)

        self._from_index = from_index
        self._to_index = to_index
        self._from_freq = from_freq
        self._to_freq = to_freq

        Configured.__init__(
            self,
            from_index=from_index,
            to_index=to_index,
            from_freq=from_freq,
            to_freq=to_freq,
        )

    @classmethod
    def from_pd_resampler(
        cls: tp.Type[ResamplerT],
        from_index: tp.IndexLike,
        pd_resampler: tp.PandasResampler,
        from_freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.core.resample.Resampler](https://pandas.pydata.org/docs/reference/resampling.html).
        """
        to_index = pd_resampler.count().index
        return cls(
            from_index=from_index,
            to_index=to_index,
            from_freq=from_freq,
            to_freq=None,
        )

    @classmethod
    def from_pd_resample(
        cls: tp.Type[ResamplerT],
        from_index: tp.IndexLike,
        *args,
        from_freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.DataFrame.resample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html).
        """
        if not isinstance(from_index, pd.DatetimeIndex):
            from_index = pd.DatetimeIndex(from_index)
        pd_resampler = pd.Series(index=from_index, dtype=object).resample(*args, **kwargs)
        return cls.from_pd_resampler(from_index, pd_resampler, from_freq=from_freq)

    @classmethod
    def from_pd_date_range(
        cls: tp.Type[ResamplerT],
        from_index: tp.IndexLike,
        *args,
        from_freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> ResamplerT:
        """Build `Resampler` from
        [pandas.date_range](https://pandas.pydata.org/docs/reference/api/pandas.date_range.html).
        """
        to_index = pd.date_range(*args, **kwargs)
        return cls(
            from_index=from_index,
            to_index=to_index,
            from_freq=from_freq,
            to_freq=None,
        )

    @property
    def from_index(self) -> tp.PandasDatetimeIndex:
        """Source index."""
        return self._from_index

    @property
    def to_index(self) -> tp.PandasDatetimeIndex:
        """Target index."""
        return self._to_index

    @property
    def from_freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Source frequency or date offset."""
        if self._from_freq is None:
            if self.from_index.freq is not None:
                return freq_to_timedelta(self.from_index.freq)
            if self.from_index.inferred_freq is not None:
                return freq_to_timedelta(self.from_index.inferred_freq)
        return self._from_freq

    @property
    def to_freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Target frequency or date offset."""
        if self._to_freq is None:
            if self.to_index.freq is not None:
                return freq_to_timedelta(self.to_index.freq)
            if self.to_index.inferred_freq is not None:
                return freq_to_timedelta(self.to_index.inferred_freq)
        return self._to_freq

    def map_to_index(
        self,
        before: bool = False,
        raise_missing: bool = True,
        return_index: bool = False,
        jitted: tp.JittedOption = None,
    ) -> tp.Union[tp.Array1d, tp.Index]:
        """See `vectorbtpro.base.resampling.nb.map_to_index_nb`."""
        func = jit_reg.resolve_option(nb.map_to_index_nb, jitted)
        mapped_arr = func(
            self.from_index.values,
            self.to_index.values,
            before=before,
            raise_missing=raise_missing,
        )
        if return_index:
            nan_mask = mapped_arr == -1
            if nan_mask.any():
                mapped_index = self.from_index.to_series().copy()
                mapped_index[nan_mask] = np.nan
                mapped_index[~nan_mask] = self.to_index[mapped_arr]
                mapped_index = pd.Index(mapped_index)
            else:
                mapped_index = self.to_index[mapped_arr]
            return mapped_index
        return mapped_arr
