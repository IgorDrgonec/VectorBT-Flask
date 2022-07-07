# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with range records.

Range records capture information on ranges. They are useful for analyzing duration of processes,
such as drawdowns, trades, and positions. They also come in handy when analyzing distance between events,
such as entry and exit signals.

Each range has a starting point and an ending point. For example, the points for `range(20)`
are 0 and 20 (not 19!) respectively.

!!! note
    Be aware that if a range hasn't ended in a column, its `end_idx` will point at the latest index.
    Make sure to account for this when computing custom metrics involving duration.

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbtpro as vbt

>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> price = vbt.YFData.fetch('BTC-USD', start=start, end=end).get('Close')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> fast_ma = vbt.MA.run(price, 10)
>>> slow_ma = vbt.MA.run(price, 50)
>>> fast_below_slow = fast_ma.ma_above(slow_ma)

>>> ranges = vbt.Ranges.from_pd(fast_below_slow, wrapper_kwargs=dict(freq='d'))

>>> ranges.records_readable
   Range Id  Column           Start Timestamp             End Timestamp  \\
0         0       0 2019-02-19 00:00:00+00:00 2019-07-25 00:00:00+00:00
1         1       0 2019-08-08 00:00:00+00:00 2019-08-19 00:00:00+00:00
2         2       0 2019-11-01 00:00:00+00:00 2019-11-20 00:00:00+00:00

   Status
0  Closed
1  Closed
2  Closed

>>> ranges.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('156 days 00:00:00')
```

## From accessors

Moreover, all generic accessors have a property `ranges` and a method `get_ranges`:

```pycon
>>> # vectorbtpro.generic.accessors.GenericAccessor.ranges.coverage
>>> fast_below_slow.vbt.ranges.coverage
0.5081967213114754
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Ranges.metrics`.

```pycon
>>> df = pd.DataFrame({
...     'a': [1, 2, np.nan, np.nan, 5, 6],
...     'b': [np.nan, 2, np.nan, 4, np.nan, 6]
... })
>>> ranges = df.vbt(freq='d').ranges

>>> ranges['a'].stats()
Start                             0
End                               5
Period              6 days 00:00:00
Total Records                     2
Coverage                   0.666667
Overlap Coverage                0.0
Duration: Min       2 days 00:00:00
Duration: Median    2 days 00:00:00
Duration: Max       2 days 00:00:00
Duration: Mean      2 days 00:00:00
Duration: Std       0 days 00:00:00
Name: a, dtype: object
```

`Ranges.stats` also supports (re-)grouping:

```pycon
>>> ranges.stats(group_by=True)
Start                                       0
End                                         5
Period                        6 days 00:00:00
Total Records                               5
Coverage                             0.416667
Overlap Coverage                          0.4
Duration: Min                 1 days 00:00:00
Duration: Median              1 days 00:00:00
Duration: Max                 2 days 00:00:00
Duration: Mean                1 days 09:36:00
Duration: Std       0 days 13:08:43.228968446
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Ranges.subplots`.

`Ranges` class has a single subplot based on `Ranges.plot`:

```pycon
>>> ranges['a'].plots()
```

![](/assets/images/ranges_plots.svg)
"""

import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_pd_array, to_2d_array, broadcast_to
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.generic import nb
from vectorbtpro.generic.enums import RangeStatus, range_dt
from vectorbtpro.generic.price_records import PriceRecords
from vectorbtpro.records.base import Records
from vectorbtpro.records.decorators import override_field_config, attach_fields, attach_shortcut_properties
from vectorbtpro.records.mapped_array import MappedArray
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.colors import adjust_lightness, adjust_opacity
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, ReadonlyConfig, HybridConfig
from vectorbtpro.utils.datetime_ import freq_to_timedelta, freq_to_timedelta64

__pdoc__ = {}

ranges_field_config = ReadonlyConfig(
    dict(
        dtype=range_dt,
        settings=dict(
            id=dict(title="Range Id"),
            idx=dict(name="end_idx"),  # remap field of Records
            start_idx=dict(title="Start Timestamp", mapping="index"),
            end_idx=dict(title="End Timestamp", mapping="index"),
            status=dict(title="Status", mapping=RangeStatus),
        ),
    )
)
"""_"""

__pdoc__[
    "ranges_field_config"
] = f"""Field config for `Ranges`.

```python
{ranges_field_config.prettify()}
```
"""

ranges_attach_field_config = ReadonlyConfig(dict(status=dict(attach_filters=True)))
"""_"""

__pdoc__[
    "ranges_attach_field_config"
] = f"""Config of fields to be attached to `Ranges`.

```python
{ranges_attach_field_config.prettify()}
```
"""

ranges_shortcut_config = ReadonlyConfig(
    dict(
        mask=dict(obj_type="array"),
        duration=dict(obj_type="mapped_array"),
        real_duration=dict(obj_type="mapped_array"),
        avg_duration=dict(obj_type="red_array"),
        max_duration=dict(obj_type="red_array"),
        coverage=dict(obj_type="red_array"),
        overlap_coverage=dict(method_name="get_coverage", obj_type="red_array", method_kwargs=dict(overlapping=True)),
        projections=dict(obj_type="array"),
    )
)
"""_"""

__pdoc__[
    "ranges_shortcut_config"
] = f"""Config of shortcut properties to be attached to `Ranges`.

```python
{ranges_shortcut_config.prettify()}
```
"""

RangesT = tp.TypeVar("RangesT", bound="Ranges")


@attach_shortcut_properties(ranges_shortcut_config)
@attach_fields(ranges_attach_field_config)
@override_field_config(ranges_field_config)
class Ranges(PriceRecords):
    """Extends `vectorbtpro.generic.price_records.PriceRecords` for working with range records.

    Requires `records_arr` to have all fields defined in `vectorbtpro.generic.enums.range_dt`."""

    @property
    def field_config(self) -> Config:
        return self._field_config

    @classmethod
    def from_pd(
        cls: tp.Type[RangesT],
        generic: tp.ArrayLike,
        gap_value: tp.Optional[tp.Scalar] = None,
        attach_as_close: bool = True,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> RangesT:
        """Build `Ranges` from Series/DataFrame.

        Searches for sequences of

        * True values in boolean data (False acts as a gap),
        * positive values in integer data (-1 acts as a gap), and
        * non-NaN values in any other data (NaN acts as a gap).

        If `attach_as_close` is True, will attach `generic` as `close`.

        `**kwargs` will be passed to `Ranges.__init__`."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        generic_arr = to_2d_array(generic)
        if gap_value is None:
            if np.issubdtype(generic_arr.dtype, np.bool_):
                gap_value = False
            elif np.issubdtype(generic_arr.dtype, np.integer):
                gap_value = -1
            else:
                gap_value = np.nan
        func = jit_reg.resolve_option(nb.get_ranges_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        records_arr = func(generic_arr, gap_value)
        wrapper = ArrayWrapper.from_obj(generic, **wrapper_kwargs)
        if attach_as_close:
            return cls(wrapper, records_arr, close=generic_arr, **kwargs)
        return cls(wrapper, records_arr, **kwargs)

    @classmethod
    def from_delta(
        self,
        records_or_mapped: tp.Union[Records, MappedArray],
        delta: tp.Union[str, int, tp.FrequencyLike],
        idx_field_or_arr: tp.Union[None, str, tp.Array1d] = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> "Ranges":
        """Build `Ranges` from a record/mapped array with a timedelta applied on its index field.

        See `vectorbtpro.generic.nb.records.get_ranges_from_delta_nb`.

        Set `delta` to an integer to wait a certain amount of rows. Set it to anything else to
        wait a timedelta. The conversion is done using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
        The second option requires the index to be datetime-like, or at least the frequency to be set."""
        from vectorbtpro.generic.ranges import Ranges

        if idx_field_or_arr is None:
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr("idx")
            else:
                idx_field_or_arr = records_or_mapped.idx_arr
        if isinstance(idx_field_or_arr, str):
            if isinstance(records_or_mapped, Records):
                idx_field_or_arr = records_or_mapped.get_field_arr(idx_field_or_arr)
            else:
                raise ValueError("Providing an index field is allowed for records only")
        if isinstance(records_or_mapped, Records):
            id_arr = records_or_mapped.get_field_arr("id")
        else:
            id_arr = records_or_mapped.id_arr
        if isinstance(delta, int):
            delta_use_index = False
            index = None
        else:
            delta = freq_to_timedelta64(delta).astype(np.int_)
            if isinstance(records_or_mapped.wrapper.index, pd.DatetimeIndex):
                index = records_or_mapped.wrapper.index.values.astype(np.int_)
            else:
                freq = freq_to_timedelta64(records_or_mapped.wrapper.freq).astype(np.int_)
                index = np.arange(records_or_mapped.wrapper.shape[0]) * freq
            delta_use_index = True
        col_map = records_or_mapped.col_mapper.get_col_map(group_by=False)
        func = jit_reg.resolve_option(nb.get_ranges_from_delta_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_records_arr = func(
            records_or_mapped.wrapper.shape[0],
            idx_field_or_arr,
            id_arr,
            col_map,
            index=index,
            delta=delta,
            delta_use_index=delta_use_index,
        )
        if isinstance(records_or_mapped, PriceRecords):
            kwargs = merge_dicts(
                dict(
                    open=records_or_mapped._open,
                    high=records_or_mapped._high,
                    low=records_or_mapped._low,
                    close=records_or_mapped._close,
                ),
                kwargs,
            )
        return Ranges.from_records(records_or_mapped.wrapper, new_records_arr, **kwargs)

    # ############# Stats ############# #

    def get_mask(
        self,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """Get mask from ranges.

        See `vectorbtpro.generic.nb.records.ranges_to_mask_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        func = jit_reg.resolve_option(nb.ranges_to_mask_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        mask = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            len(self.wrapper.index),
        )
        return self.wrapper.wrap(mask, group_by=group_by, **resolve_dict(wrap_kwargs))

    def get_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get the effective duration of each range in integer format."""
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            freq=1,
        )
        return self.map_array(duration, **kwargs)

    def get_real_duration(
        self,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        **kwargs,
    ) -> MappedArray:
        """Get the real duration of each range in timedelta format."""
        func = jit_reg.resolve_option(nb.range_duration_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        duration = func(
            self.get_map_field_to_index("start_idx").values.astype(np.int_),
            self.get_map_field_to_index("end_idx").values.astype(np.int_),
            self.get_field_arr("status"),
            freq=freq_to_timedelta64(self.wrapper.freq).astype(np.int_),
        ).astype("timedelta64[ns]")
        return self.map_array(duration, **kwargs)

    def get_avg_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get average range duration (as timedelta)."""
        if real:
            duration = self.real_duration
            wrap_kwargs = merge_dicts(dict(name_or_index="avg_real_duration", dtype="timedelta64[ns]"), wrap_kwargs)
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index="avg_duration"), wrap_kwargs)
        return duration.mean(group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs)

    def get_max_duration(
        self,
        real: bool = False,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeSeries:
        """Get maximum range duration (as timedelta)."""
        if real:
            duration = self.real_duration
            wrap_kwargs = merge_dicts(dict(name_or_index="max_real_duration", dtype="timedelta64[ns]"), wrap_kwargs)
        else:
            duration = self.duration
            wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index="max_duration"), wrap_kwargs)
        return duration.max(group_by=group_by, jitted=jitted, chunked=chunked, wrap_kwargs=wrap_kwargs, **kwargs)

    def filter_min_duration(
        self: RangesT,
        min_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges that last less than a minimum duration."""
        if isinstance(min_duration, int):
            return self.apply_mask(self.duration.values >= min_duration, **kwargs)
        min_duration = freq_to_timedelta64(min_duration)
        if real:
            return self.apply_mask(self.real_duration.values >= min_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq >= min_duration, **kwargs)

    def filter_max_duration(
        self: RangesT,
        max_duration: tp.Union[str, int, tp.FrequencyLike],
        real: bool = False,
        **kwargs,
    ) -> RangesT:
        """Filter out ranges that last more than a maximum duration."""
        if isinstance(max_duration, int):
            return self.apply_mask(self.duration.values <= max_duration, **kwargs)
        max_duration = freq_to_timedelta64(max_duration)
        if real:
            return self.apply_mask(self.real_duration.values <= max_duration, **kwargs)
        return self.apply_mask(self.duration.values * self.wrapper.freq <= max_duration, **kwargs)

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        group_by: tp.GroupByLike = None,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.MaybeSeries:
        """Get coverage, that is, the number of steps that are covered by all ranges.

        See `vectorbtpro.generic.nb.records.range_coverage_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        index_lens = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        func = jit_reg.resolve_option(nb.range_coverage_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        coverage = func(
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            col_map,
            index_lens,
            overlapping=overlapping,
            normalize=normalize,
        )
        wrap_kwargs = merge_dicts(dict(name_or_index="coverage"), wrap_kwargs)
        return self.wrapper.wrap_reduced(coverage, group_by=group_by, **wrap_kwargs)

    def get_projections(
        self,
        close: tp.Optional[tp.ArrayLike] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = None,
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = None,
        stretch: bool = False,
        normalize: bool = True,
        start_value: float = 1.0,
        ffill: bool = False,
        remove_empty: bool = True,
        return_raw: bool = False,
        jitted: tp.JittedOption = None,
        wrap_kwargs: tp.KwargsLike = None,
        stack_indexes_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[tp.Tuple[tp.Array1d, tp.Array2d], tp.Frame]:
        """Generate a projection for each range record.

        See `vectorbtpro.generic.nb.records.map_ranges_to_projections_nb`.

        Set `proj_start` to an integer to generate a projection after a certain row
        after the start row. Set it to anything else to wait a timedelta.
        The conversion is done using `vectorbtpro.utils.datetime_.freq_to_timedelta64`.
        The second option requires the index to be datetime-like, or at least the frequency to be set.

        Set `proj_period` the same way as `proj_start` to generate a projection of a certain length.
        Unless `stretch` is True, it still respects the duration of the range.

        Set `stretch` to True to stretch the projection even after the end of the range.
        The stretching period is taken from the longest range duration if `proj_period` is None,
        and from the longest `proj_period` if it's not None.

        Set `normalize` to True to make each projection start with 1, otherwise, each projection
        will consist of original `close` values during the projected period. Use `start_value`
        to replace 1 with another start value.

        Set `ffill` to True to forward fill NaN values, even if they are NaN in `close` itself.

        Set `remove_empty` to True to remove projections that are either NaN or with only one element.
        The index of each projection is still being tracked and will appear in the multi-index of the
        returned DataFrame.

        !!! note
            As opposed to the Numba-compiled function, the returned DataFrame will have
            projections stacked along columns rather than rows. Set `return_raw` to True
            to return them in the original format.
        """
        if close is None:
            close = self.close
        else:
            close = self.wrapper.wrap(close, group_by=False)
        if proj_start is None:
            proj_start = 0
        if isinstance(proj_start, int):
            proj_start_use_index = False
            index = None
        else:
            proj_start = freq_to_timedelta64(proj_start).astype(np.int_)
            if isinstance(self.wrapper.index, pd.DatetimeIndex):
                index = self.wrapper.index.values.astype(np.int_)
            else:
                freq = freq_to_timedelta64(self.wrapper.freq).astype(np.int_)
                index = np.arange(self.wrapper.shape[0]) * freq
            proj_start_use_index = True
        if proj_period is not None:
            if isinstance(proj_period, int):
                proj_period_use_index = False
            else:
                proj_period = freq_to_timedelta64(proj_period).astype(np.int_)
                if index is None:
                    if isinstance(self.wrapper.index, pd.DatetimeIndex):
                        index = self.wrapper.index.values.astype(np.int_)
                    else:
                        freq = freq_to_timedelta64(self.wrapper.freq).astype(np.int_)
                        index = np.arange(self.wrapper.shape[0]) * freq
                proj_period_use_index = True
        else:
            proj_period_use_index = False

        func = jit_reg.resolve_option(nb.map_ranges_to_projections_nb, jitted)
        ridxs, projections = func(
            to_2d_array(close),
            self.get_field_arr("col"),
            self.get_field_arr("start_idx"),
            self.get_field_arr("end_idx"),
            self.get_field_arr("status"),
            index=index,
            proj_start=proj_start,
            proj_start_use_index=proj_start_use_index,
            proj_period=proj_period,
            proj_period_use_index=proj_period_use_index,
            stretch=stretch,
            normalize=normalize,
            start_value=start_value,
            ffill=ffill,
            remove_empty=remove_empty,
        )
        if return_raw:
            return ridxs, projections
        projections = projections.T
        freq = self.wrapper.get_freq(allow_numeric=False)
        wrapper = ArrayWrapper.from_obj(projections, freq=freq)
        wrap_kwargs = merge_dicts(
            dict(
                index=pd.date_range(
                    start=close.index[-1],
                    periods=projections.shape[0],
                    freq=freq,
                ),
                columns=stack_indexes(
                    self.wrapper.columns[self.col_arr[ridxs]],
                    pd.Index(self.id_arr[ridxs], name="range_id"),
                    **resolve_dict(stack_indexes_kwargs),
                ),
            ),
            wrap_kwargs,
        )
        return wrapper.wrap(projections, **wrap_kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Ranges.stats`.

        Merges `vectorbtpro.records.base.Records.stats_defaults` and
        `stats` from `vectorbtpro._settings.ranges`."""
        from vectorbtpro._settings import settings

        ranges_stats_cfg = settings["ranges"]["stats"]

        return merge_dicts(Records.stats_defaults.__get__(self), ranges_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(title="Start", calc_func=lambda self: self.wrapper.index[0], agg_func=None, tags="wrapper"),
            end=dict(title="End", calc_func=lambda self: self.wrapper.index[-1], agg_func=None, tags="wrapper"),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_records=dict(title="Total Records", calc_func="count", tags="records"),
            coverage=dict(
                title="Coverage",
                calc_func="coverage",
                overlapping=False,
                tags=["ranges", "coverage"],
            ),
            overlap_coverage=dict(
                title="Overlap Coverage",
                calc_func="coverage",
                overlapping=True,
                tags=["ranges", "coverage"],
            ),
            duration=dict(
                title="Duration",
                calc_func="duration.describe",
                post_calc_func=lambda self, out, settings: {
                    "Min": out.loc["min"],
                    "Median": out.loc["50%"],
                    "Max": out.loc["max"],
                    "Mean": out.loc["mean"],
                    "Std": out.loc["std"],
                },
                apply_to_timedelta=True,
                tags=["ranges", "duration"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_projections(
        self,
        column: tp.Optional[tp.Label] = None,
        min_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        max_duration: tp.Union[str, int, tp.FrequencyLike] = None,
        last_n: tp.Optional[int] = None,
        top_n: tp.Optional[int] = None,
        random_n: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        proj_start: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_0",
        proj_period: tp.Union[None, str, int, tp.FrequencyLike] = "max",
        stretch: bool = False,
        ffill: bool = False,
        plot_past_period: tp.Union[None, str, int, tp.FrequencyLike] = "current_or_proj_period",
        plot_ohlc: tp.Union[bool, tp.Frame] = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_projections: bool = True,
        plot_lower: tp.Union[bool, str, tp.Callable] = True,
        plot_middle: tp.Union[bool, str, tp.Callable] = True,
        plot_upper: tp.Union[bool, str, tp.Callable] = True,
        plot_fill: bool = True,
        colorize: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        projection_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:  # pragma: no cover
        """Plot projections.

        Combines generation of projections using `Ranges.get_projections` and
        their plotting using `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.

        Args:
            column (str): Name of the column to plot.
            min_duration (str, int, or frequency_like): Filter range records by minimum duration.
            max_duration (str, int, or frequency_like): Filter range records by maximum duration.
            last_n (int): Select last N range records.
            top_n (int): Select top N range records by maximum duration.
            random_n (int): Select N range records randomly.
            seed (int): Set seed to make output deterministic.
            proj_start (str, int, or frequency_like): See `Ranges.get_projections`.

                Allows an additional option "current_or_{value}", which sets `proj_start` to
                the duration of the current open range, and to the specified value if there is no open range.
            proj_period (str, int, or frequency_like): See `Ranges.get_projections`.

                Allows additional options "current_or_{option}", "mean", "min", "max", "median", or
                a percentage such as "50%" representing a quantile. All of those options are based
                on the duration of all the closed ranges filtered by the arguments above.
            stretch (bool): See `Ranges.get_projections`.
            ffill (bool): See `Ranges.get_projections`.
            plot_past_period (str, int, or frequency_like): Past period to plot.

                Allows the same options as `proj_period` plus "proj_period" and "current_or_proj_period".
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            plot_projections (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_lower (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_middle (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_upper (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            plot_fill (bool): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            colorize (bool, str, or callable): See `vectorbtpro.generic.accessors.GenericDFAccessor.plot_projections`.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Ranges.close`.
            projection_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for projections.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.plotly.graph_objects.Scatter` for lower band.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for middle band.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for upper band.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> price = pd.Series(
            ...     [11, 12, 13, 14, 11, 12, 13, 12, 11, 12],
            ...     index=pd.date_range("2020", periods=10),
            ...     name='Price'
            ... )
            >>> vbt.Ranges.from_pd(
            ...     price >= 12,
            ...     attach_as_close=False,
            ...     close=price,
            ... ).plot_projections(
            ...     proj_start=0,
            ...     proj_period=4,
            ...     stretch=True,
            ...     plot_past_period=None
            ... )
            ```

            ![](/assets/images/ranges_plot_projections.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        self_col_open = self_col.status_open
        self_col = self_col.status_closed
        if proj_start is not None:
            if isinstance(proj_start, str) and proj_start.startswith("current_or_"):
                proj_start = proj_start.replace("current_or_", "")
                if proj_start.isnumeric():
                    proj_start = int(proj_start)
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    proj_start = int(self_col_open.duration.values[0])
            if proj_start != 0:
                self_col = self_col.filter_min_duration(proj_start, real=True)
        if min_duration is not None:
            self_col = self_col.filter_min_duration(min_duration, real=True)
        if max_duration is not None:
            self_col = self_col.filter_max_duration(max_duration, real=True)
        if last_n is not None:
            self_col = self_col.last_n(last_n)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))
        if random_n is not None:
            self_col = self_col.random_n(random_n, seed=seed)

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"])), close_trace_kwargs
        )
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True
        if close is None:
            raise ValueError("Close cannot be None")

        # Resolve windows
        def _resolve_period(period):
            if self_col.count() == 0:
                period = None
            if period is not None:
                if isinstance(period, str):
                    period = period.lower().strip()
                    if period == "median":
                        period = "50%"
                    if "%" in period:
                        period = int(
                            np.quantile(
                                self_col.duration.values,
                                float(period.replace("%", "")) / 100,
                            )
                        )
                    elif period.startswith("current_or_"):
                        if self_col_open.count() > 0:
                            if self_col_open.count() > 1:
                                raise ValueError("Only one open range is allowed")
                            period = int(self_col_open.duration.values[0])
                        else:
                            period = period.replace("current_or_", "")
                            return _resolve_period(period)
                    elif period == "mean":
                        period = int(np.mean(self_col.duration.values))
                    elif period == "min":
                        period = int(np.min(self_col.duration.values))
                    elif period == "max":
                        period = int(np.max(self_col.duration.values))
            return period

        proj_period = _resolve_period(proj_period)
        if isinstance(proj_period, int) and proj_period == 0:
            warnings.warn("Projection period is zero. Setting to maximum.")
            proj_period = int(np.max(self_col.duration.values))
        if plot_past_period is not None and isinstance(plot_past_period, str):
            plot_past_period = plot_past_period.lower().strip()
            if plot_past_period == "proj_period":
                plot_past_period = proj_period
            elif plot_past_period == "current_or_proj_period":
                if self_col_open.count() > 0:
                    if self_col_open.count() > 1:
                        raise ValueError("Only one open range is allowed")
                    plot_past_period = int(self_col_open.duration.values[0])
                else:
                    plot_past_period = proj_period
        plot_past_period = _resolve_period(plot_past_period)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot OHLC/close
        if plot_ohlc and ohlc is not None:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _ohlc = ohlc.iloc[-plot_past_period:]
                else:
                    plot_past_period = freq_to_timedelta(plot_past_period)
                    _ohlc = ohlc[ohlc.index > ohlc.index[-1] - plot_past_period]
            else:
                _ohlc = ohlc
            if _ohlc.size > 0:
                if "opacity" not in ohlc_trace_kwargs:
                    ohlc_trace_kwargs["opacity"] = 0.5
                fig = _ohlc.vbt.ohlcv.plot(
                    ohlc_type=ohlc_type,
                    plot_volume=False,
                    ohlc_trace_kwargs=ohlc_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
        elif plot_close:
            if plot_past_period is not None:
                if isinstance(plot_past_period, int):
                    _close = close.iloc[-plot_past_period:]
                else:
                    plot_past_period = freq_to_timedelta(plot_past_period)
                    _close = close[close.index > close.index[-1] - plot_past_period]
            else:
                _close = close
            if _close.size > 0:
                fig = _close.vbt.plot(
                    trace_kwargs=close_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        if self_col.count() > 0:
            # Get projections
            projections = self_col.get_projections(
                close=close,
                proj_start=proj_start,
                proj_period=proj_period,
                stretch=stretch,
                normalize=True,
                start_value=close.iloc[-1],
                ffill=ffill,
                remove_empty=True,
                return_raw=False,
            )

            if len(projections.columns) > 0:
                # Plot projections
                rename_levels = dict(range_id=self_col.get_field_title("id"))
                fig = projections.vbt.plot_projections(
                    plot_projections=plot_projections,
                    plot_lower=plot_lower,
                    plot_middle=plot_middle,
                    plot_upper=plot_upper,
                    plot_fill=plot_fill,
                    colorize=colorize,
                    rename_levels=rename_levels,
                    projection_trace_kwargs=projection_trace_kwargs,
                    upper_trace_kwargs=upper_trace_kwargs,
                    middle_trace_kwargs=middle_trace_kwargs,
                    lower_trace_kwargs=lower_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )

        return fig

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        top_n: tp.Optional[int] = None,
        plot_ohlc: bool = True,
        plot_close: tp.Union[bool, tp.Series] = True,
        plot_markers: bool = True,
        plot_zones: bool = True,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        close_trace_kwargs: tp.KwargsLike = None,
        start_trace_kwargs: tp.KwargsLike = None,
        end_trace_kwargs: tp.KwargsLike = None,
        open_shape_kwargs: tp.KwargsLike = None,
        closed_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        xref: str = "x",
        yref: str = "y",
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:  # pragma: no cover
        """Plot ranges.

        Args:
            column (str): Name of the column to plot.
            top_n (int): Filter top N range records by maximum duration.
            plot_ohlc (bool): Whether to plot OHLC.
            plot_close (bool): Whether to plot close.
            plot_markers (bool): Whether to plot markers.
            plot_zones (bool): Whether to plot zones.
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Ranges.close`.
            start_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for start values.
            end_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for end values.
            open_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for open zones.
            closed_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for closed zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> price = pd.Series(
            ...     [1, 2, 1, 2, 3, 2, 1, 2],
            ...     index=pd.date_range("2020", periods=8),
            ...     name='Price'
            ... )
            >>> vbt.Ranges.from_pd(price >= 2).plot()
            ```

            ![](/assets/images/ranges_plot.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure, get_domain
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column, group_by=False)
        if top_n is not None:
            self_col = self_col.apply_mask(self_col.duration.top_n_mask(top_n))

        if ohlc_trace_kwargs is None:
            ohlc_trace_kwargs = {}
        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(line=dict(color=plotting_cfg["color_schema"]["blue"])), close_trace_kwargs
        )
        if start_trace_kwargs is None:
            start_trace_kwargs = {}
        if end_trace_kwargs is None:
            end_trace_kwargs = {}
        if open_shape_kwargs is None:
            open_shape_kwargs = {}
        if closed_shape_kwargs is None:
            closed_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        if isinstance(plot_ohlc, bool):
            if (
                self_col._open is not None
                and self_col._high is not None
                and self_col._low is not None
                and self_col._close is not None
            ):
                ohlc = pd.DataFrame(
                    {
                        "open": self_col.open,
                        "high": self_col.high,
                        "low": self_col.low,
                        "close": self_col.close,
                    }
                )
            else:
                ohlc = None
        else:
            ohlc = plot_ohlc
            plot_ohlc = True
        if isinstance(plot_close, bool):
            if ohlc is not None:
                close = ohlc.vbt.ohlcv.close
            else:
                close = self_col.close
        else:
            close = plot_close
            plot_close = True

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        y_domain = get_domain(yref, fig)

        # Plot OHLC/close
        plotting_ohlc = False
        if plot_ohlc and ohlc is not None:
            ohlc_df = pd.DataFrame(
                {
                    "open": self_col.open,
                    "high": self_col.high,
                    "low": self_col.low,
                    "close": self_col.close,
                }
            )
            if "opacity" not in ohlc_trace_kwargs:
                ohlc_trace_kwargs["opacity"] = 0.5
            fig = ohlc_df.vbt.ohlcv.plot(
                ohlc_type=ohlc_type,
                plot_volume=False,
                ohlc_trace_kwargs=ohlc_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
            plotting_ohlc = True
        elif plot_close and close is not None:
            fig = close.vbt.plot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        if self_col.count() > 0:
            # Extract information
            id_ = self_col.get_field_arr("id")
            id_title = self_col.get_field_title("id")

            start_idx = self_col.get_map_field_to_index("start_idx")
            start_idx_title = self_col.get_field_title("start_idx")
            if plotting_ohlc and self_col.open is not None:
                start_val = self_col.open.loc[start_idx]
            elif close is not None:
                start_val = close.loc[start_idx]
            else:
                start_val = np.full(len(start_idx), 0)

            end_idx = self_col.get_map_field_to_index("end_idx")
            end_idx_title = self_col.get_field_title("end_idx")
            if close is not None:
                end_val = close.loc[end_idx]
            else:
                end_val = np.full(len(end_idx), 0)

            duration = np.vectorize(str)(
                self_col.wrapper.to_timedelta(self_col.duration.values, to_pd=True, silence_warnings=True)
            )

            status = self_col.get_field_arr("status")

            if plot_markers:
                # Plot start markers
                start_customdata = id_[:, None]
                start_scatter = go.Scatter(
                    x=start_idx,
                    y=start_val,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        color=plotting_cfg["contrast_color_schema"]["blue"],
                        size=7,
                        line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["blue"])),
                    ),
                    name="Start",
                    customdata=start_customdata,
                    hovertemplate=f"{id_title}: %{{customdata[0]}}<br>{start_idx_title}: %{{x}}",
                )
                start_scatter.update(**start_trace_kwargs)
                fig.add_trace(start_scatter, **add_trace_kwargs)

            closed_mask = status == RangeStatus.Closed
            if closed_mask.any():
                if plot_markers:
                    # Plot end markers
                    closed_end_customdata = np.stack((id_[closed_mask], duration[closed_mask]), axis=1)
                    closed_end_scatter = go.Scatter(
                        x=end_idx[closed_mask],
                        y=end_val[closed_mask],
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            color=plotting_cfg["contrast_color_schema"]["green"],
                            size=7,
                            line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["green"])),
                        ),
                        name="Closed",
                        customdata=closed_end_customdata,
                        hovertemplate=(
                            f"{id_title}: %{{customdata[0]}}<br>{end_idx_title}: %{{x}}<br>Duration: %{{customdata[1]}}"
                        ),
                    )
                    closed_end_scatter.update(**end_trace_kwargs)
                    fig.add_trace(closed_end_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot closed range zones
                    for i in range(len(id_[closed_mask])):
                        fig.add_shape(
                            **merge_dicts(
                                dict(
                                    type="rect",
                                    xref=xref,
                                    yref="paper",
                                    x0=start_idx[closed_mask][i],
                                    y0=y_domain[0],
                                    x1=end_idx[closed_mask][i],
                                    y1=y_domain[1],
                                    fillcolor="royalblue",
                                    opacity=0.2,
                                    layer="below",
                                    line_width=0,
                                ),
                                closed_shape_kwargs,
                            )
                        )

            open_mask = status == RangeStatus.Open
            if open_mask.any():
                if plot_markers:
                    # Plot end markers
                    open_end_customdata = np.stack((id_[open_mask], duration[open_mask]), axis=1)
                    open_end_scatter = go.Scatter(
                        x=end_idx[open_mask],
                        y=end_val[open_mask],
                        mode="markers",
                        marker=dict(
                            symbol="diamond",
                            color=plotting_cfg["contrast_color_schema"]["orange"],
                            size=7,
                            line=dict(width=1, color=adjust_lightness(plotting_cfg["contrast_color_schema"]["orange"])),
                        ),
                        name="Open",
                        customdata=open_end_customdata,
                        hovertemplate=(
                            f"{id_title}: %{{customdata[0]}}<br>{end_idx_title}: %{{x}}<br>Duration: %{{customdata[1]}}"
                        ),
                    )
                    open_end_scatter.update(**end_trace_kwargs)
                    fig.add_trace(open_end_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot open range zones
                    for i in range(len(id_[open_mask])):
                        fig.add_shape(
                            **merge_dicts(
                                dict(
                                    type="rect",
                                    xref=xref,
                                    yref="paper",
                                    x0=start_idx[open_mask][i],
                                    y0=y_domain[0],
                                    x1=end_idx[open_mask][i],
                                    y1=y_domain[1],
                                    fillcolor="orange",
                                    opacity=0.2,
                                    layer="below",
                                    line_width=0,
                                ),
                                open_shape_kwargs,
                            )
                        )

        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Ranges.plots`.

        Merges `vectorbtpro.records.base.Records.plots_defaults` and
        `plots` from `vectorbtpro._settings.ranges`."""
        from vectorbtpro._settings import settings

        ranges_plots_cfg = settings["ranges"]["plots"]

        return merge_dicts(Records.plots_defaults.__get__(self), ranges_plots_cfg)

    _subplots: tp.ClassVar[Config] = Config(
        dict(plot=dict(title="Ranges", check_is_not_grouped=True, plot_func="plot", tags="ranges")),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Ranges.override_field_config_doc(__pdoc__)
Ranges.override_metrics_doc(__pdoc__)
Ranges.override_subplots_doc(__pdoc__)
