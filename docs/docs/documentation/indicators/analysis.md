---
title: Analysis
description: Documentation on indicator analysis
icon: material/chart-timeline-variant-shimmer
---

# :material-chart-timeline-variant-shimmer: Analysis

To analyze an indicator, we can use the indicator instance returned by the `run` method.

## Helper methods

Whenever we create an instance of [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory),
it builds and sets up an indicator class. During this setup, the factory attaches many interesting
attributes to the class. For instance, for each in `input_names`, `in_output_names`, `output_names`, 
and `lazy_outputs`, it will create and attach a bunch of comparison and combination methods. 
What characteristics any of these attributes should have can be regulated in the `attr_settings` dictionary.

Let's modify the class `CrossSig` created earlier by replacing the entries and exits with a single signal
array and also returning an enumerated array that contains the signal type. We will also specify 
the data type of each array in the `attr_settings` dictionary:

```pycon
>>> import vectorbtpro as vbt
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> from collections import namedtuple

>>> SignalType = namedtuple('SigType', ['Entry', 'Exit'])(0, 1)  # (1)!

>>> def apply_func(ts, fastw, sloww, minp=None):
...     fast_ma = vbt.nb.rolling_mean_nb(ts, fastw, minp=minp)
...     slow_ma = vbt.nb.rolling_mean_nb(ts, sloww, minp=minp)
...     entries = vbt.nb.crossed_above_nb(fast_ma, slow_ma)
...     exits = vbt.nb.crossed_above_nb(slow_ma, fast_ma)
...     signals = entries | exits
...     signal_type = np.full(ts.shape, -1, dtype=np.int_)  # (2)!
...     signal_type[entries] = SignalType.Entry
...     signal_type[exits] = SignalType.Exit
...     return (fast_ma, slow_ma, signals, signal_type)

>>> CrossSig = vbt.IF(
...     class_name="CrossSig",
...     input_names=['ts'],
...     param_names=['fastw', 'sloww'],
...     output_names=['fast_ma', 'slow_ma', 'signals', 'signal_type'],
...     attr_settings=dict(
...         fast_ma=dict(dtype=np.float_),
...         slow_ma=dict(dtype=np.float_),
...         signals=dict(dtype=np.bool_),
...         signal_type=dict(dtype=SignalType),
...     )
... ).with_apply_func(apply_func)

>>> def generate_index(n):
...     return pd.date_range("2020-01-01", periods=n)

>>> ts = pd.DataFrame({
...     'a': [1, 2, 3, 2, 1, 2, 3],
...     'b': [3, 2, 1, 2, 3, 2, 1]
... }, index=generate_index(7))
>>> cross_sig = CrossSig.run(ts, 2, 3)
```

1. Values of an enum in vectorbt usually start with 0 and increase incrementally
2. A missing value in vectorbt is usually represented with `-1`

We can explore the helper methods that were attached using the Python's `dir` command:

```pycon
>>> dir(cross_sig)
...
'fast_ma',
'fast_ma_above',
'fast_ma_below',
'fast_ma_crossed_above',
'fast_ma_crossed_below',
'fast_ma_equal',
'fast_ma_stats',
...
'signal_type',
'signal_type_readable',
'signal_type_stats',
...
'signals',
'signals_and',
'signals_or',
'signals_stats',
'signals_xor',
...
'slow_ma',
'slow_ma_above',
'slow_ma_below',
'slow_ma_crossed_above',
'slow_ma_crossed_below',
'slow_ma_equal',
'slow_ma_stats',
...
'ts',
'ts_above',
'ts_below',
'ts_crossed_above',
'ts_crossed_below',
'ts_equal',
'ts_stats',
```

One helper method that appears for each array is `stats`, which calls 
[StatsBuilderMixin.stats](/api/generic/stats_builder/#vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats)
on the accessor that corresponds to the data type of the array:

```pycon
>>> cross_sig.fast_ma_stats(column=(2, 3, 'a'))  # (1)!
Start        2020-01-01 00:00:00
End          2020-01-07 00:00:00
Period           7 days 00:00:00
Count                          6
Mean                         2.0
Std                     0.547723
Min                          1.5
Median                       2.0
Max                          2.5
Min Index    2020-01-02 00:00:00
Max Index    2020-01-03 00:00:00
Name: (2, 3, a), dtype: object
```

1. Using [GenericAccessor.stats](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.stats)

The same can be done manually:

```pycon
>>> cross_sig.fast_ma.vbt.stats(column=(2, 3, 'a'))
```

### Numeric

The factory generated the comparison methods `above`, `below`, and `equal` for the numeric arrays. 
Each of those methods is based on [combine_objs](/api/indicators/factory/#vectorbtpro.indicators.factory.combine_objs), 
which in turn is based on [BaseAccessor.combine](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.combine).
All operations are done strictly using NumPy. Another advantage is utilization of vectorbt's own 
broadcasting, such that we can combine the arrays with an arbitrary array-like object, 
given their shapes can broadcast together. We can also do comparison with multiple objects 
at once by passing them as a tuple or list. 

Let's return True when the fast moving average is above a range of thresholds:

```pycon
>>> cross_sig.fast_ma_above([2, 3])
crosssig_fast_ma_above             2             3
crosssig_fastw                     2             2
crosssig_sloww                     3             3
                            a      b      a      b
2020-01-01              False  False  False  False
2020-01-02              False   True  False  False
2020-01-03               True  False  False  False
2020-01-04               True  False  False  False
2020-01-05              False   True  False  False
2020-01-06              False   True  False  False
2020-01-07               True  False  False  False
```

Or, manually:

```pycon
>>> cross_sig.fast_ma.vbt > pd.Index([2, 3], name='crosssig_fast_ma_above')
```

Additionally, the factory attached the methods `crossed_above` and `crossed_below`,
which are based on [GenericAccessor.crossed_above](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_above) 
and [GenericAccessor.crossed_below](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.crossed_below) 
respectively.

```pycon
>>> cross_sig.fast_ma_crossed_above(cross_sig.slow_ma)
crosssig_fastw             2
crosssig_sloww             3
                    a      b
2020-01-01      False  False
2020-01-02      False  False
2020-01-03      False  False
2020-01-04      False  False
2020-01-05      False   True
2020-01-06      False  False
2020-01-07       True  False
```

Or, manually:

```pycon
>>> cross_sig.fast_ma.vbt.crossed_above(cross_sig.slow_ma)
```

### Boolean

The factory generated the comparison methods `and`, `or`, and `xor` for the boolean arrays. 
Similarly to the methods generated for the numeric arrays, they are also based on 
[combine_objs](/api/indicators/factory/#vectorbtpro.indicators.factory.combine_objs):

```pycon
>>> other_signals = pd.Series([False, False, False, False, True, False, False])
>>> cross_sig.signals_and(other_signals)
crosssig_fastw             2
crosssig_sloww             3
                    a      b
2020-01-01      False  False
2020-01-02      False  False
2020-01-03      False  False
2020-01-04      False  False
2020-01-05       True   True
2020-01-06      False  False
2020-01-07      False  False
```

Or, manually:

```pycon
>>> cross_sig.signals.vbt & other_signals
```

### Enumerated

Enumerated (or categorical) arrays, such as our `signal_type`, contain integer data that can be mapped
to a certain category using a named tuple or any other enum. In contrast to the numeric and boolean arrays, 
comparing them with other arrays would make no sense. Thus, there is only one attached method - `readable` -
that allows us to print the array in a human-readable format:

```pycon
>>> cross_sig.signal_type_readable
crosssig_fastw             2
crosssig_sloww             3
                    a      b
2020-01-01       None   None
2020-01-02       None   None
2020-01-03       None   None
2020-01-04       None   None
2020-01-05       Exit  Entry
2020-01-06       None   None
2020-01-07      Entry   Exit
```

!!! hint
    In vectorbt, if `-1` is not listed in the enum, it automatically means a missing value and 
    gets replaced by `None`.

Or, manually:

```pycon
>>> cross_sig.signal_type.vbt(mapping=SignalType).apply_mapping()
```

## Indexing

Each indicator class subclasses [Analyzable](../../building-blocks/#analyzing), so we can perform
Pandas indexing on the indicator instance to select rows and columns in all Pandas objects.
Supported operations are `iloc`, `loc`, `xs`, and `__getitem__`. 

```pycon
>>> cross_sig = CrossSig.run(ts, [2, 3], [3, 4], param_product=True)

>>> cross_sig.loc["2020-01-03":, (2, 3, 'a')]  # (1)!
<vectorbtpro.indicators.factory.CrossSig at 0x7fcaf8e2f5c0>

>>> cross_sig.loc["2020-01-03":, (2, 3, 'a')].signals
2020-01-03    False
2020-01-04    False
2020-01-05     True
2020-01-06    False
2020-01-07     True
Name: (2, 3, a), dtype: bool
```

1. Using one Pandas indexing opreation

Additionally, [IndicatorFactory](/api/indicators/factory/#vectorbtpro.indicators.factory.IndicatorFactory) 
uses the class factory function [build_param_indexer](/api/base/indexing/#vectorbtpro.base.indexing.build_param_indexer)
to create an indexing class that enables Pandas indexing on each defined parameter.
Since the indicator class subclasses this indexing class, we can use `*param_name*_loc` to select
one or more values of any parameter.

```pycon
>>> cross_sig.fastw_loc[2].sloww_loc[3]['a']  # (1)!
<vectorbtpro.indicators.factory.CrossSig at 0x7fcac80742b0>

>>> cross_sig.fastw_loc[2].sloww_loc[3]['a'].signals
2020-01-01    False
2020-01-02    False
2020-01-03    False
2020-01-04    False
2020-01-05     True
2020-01-06    False
2020-01-07     True
Name: a, dtype: bool
```

1. Using two parameter and one Pandas indexing operation. Each of those operations returns 
a separate instance of `CrossSig`.

This all makes possible accessing rows and columns by labels, integer positions, and parameters - 
full flexibility :man_cartwheeling:

## Stats and plots

As with every [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable) instance, 
we can compute and plot various properties of the (input and output) data stored in the instance.

Metrics can be defined in two ways: by passing them via the `metrics` argument, or by 
subclassing the indicator class. The same applies to the `stats_defaults` argument, which 
can be provided either as a dictionary or a function, and which defines 
the default settings for [StatsBuilderMixin.stats](/api/generic/stats_builder/#vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats).
Subplots can be defined in a similar way to metrics, except that they are set up
by `subplots` and `plots_defaults` arguments, and invoked by 
[PlotsBuilderMixin.plots](/api/generic/plots_builder/#vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots).

Let's define some metrics and subplots for `CrossSig`:

```pycon
>>> metrics = dict(
...     start=dict(  # (1)!
...         title='Start',
...         calc_func=lambda self: self.wrapper.index[0],
...         agg_func=None
...     ),
...     end=dict(  # (2)!
...         title='End',
...         calc_func=lambda self: self.wrapper.index[-1],
...         agg_func=None
...     ),
...     period=dict(  # (3)!
...         title='Period',
...         calc_func=lambda self: len(self.wrapper.index),
...         apply_to_timedelta=True,
...         agg_func=None
...     ),
...     fast_stats=dict(  # (4)!
...         title="Fast Stats",
...         calc_func=lambda self: 
...         self.fast_ma.describe()\
...         .loc[['count', 'mean', 'std', 'min', 'max']]\
...         .vbt.to_dict(orient='index_series')
...     ),
...     slow_stats=dict(
...         title="Slow Stats",
...         calc_func=lambda self: 
...         self.slow_ma.describe()\
...         .loc[['count', 'mean', 'std', 'min', 'max']]\
...         .vbt.to_dict(orient='index_series')
...     ),
...     num_entries=dict(  # (5)!
...         title="Entries",
...         calc_func=lambda self: 
...         np.sum(self.signal_type == SignalType.Entry)
...     ),
...     num_exits=dict(
...         title="Exits",
...         calc_func=lambda self: 
...         np.sum(self.signal_type == SignalType.Exit)
...     )
... )

>>> def plot_mas(self, column=None, add_trace_kwargs=None, fig=None):  # (6)!
...     ts = self.select_col_from_obj(self.ts, column).rename('TS')  # (7)!
...     fast_ma = self.select_col_from_obj(self.fast_ma, column).rename('Fast MA')
...     slow_ma = self.select_col_from_obj(self.slow_ma, column).rename('Slow MA')
...     ts.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...     fast_ma.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)  # (8)!
...     slow_ma.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)

>>> def plot_signals(self, column=None, add_trace_kwargs=None, fig=None):
...     signal_type = self.select_col_from_obj(self.signal_type, column)
...     entries = (signal_type == SignalType.Entry).rename('Entries')
...     exits = (signal_type == SignalType.Exit).rename('Exits')
...     entries.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...     exits.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)

>>> subplots = dict(
...     mas=dict(
...         title="Moving averages",
...         plot_func=plot_mas
...     ),
...     signals=dict(
...         title="Signals",
...         plot_func=plot_signals
...     )
... )

>>> CrossSig = vbt.IF(
...     class_name="CrossSig",
...     input_names=['ts'],
...     param_names=['fastw', 'sloww'],
...     output_names=['fast_ma', 'slow_ma', 'signals', 'signal_type'],
...     attr_settings=dict(
...         fast_ma=dict(dtype=np.float_),
...         slow_ma=dict(dtype=np.float_),
...         signals=dict(dtype=np.bool_),
...         signal_type=dict(dtype=SignalType),
...     ),
...     metrics=metrics,  # (9)!
...     subplots=subplots
... ).with_apply_func(apply_func)

>>> cross_sig = CrossSig.run(ts, [2, 3], 4)
```

1. Index of the first data point
2. Index of the last data point
3. The number of data points multiplied by the index frequency
4. Takes `fast_ma`, generates the descriptive statistics using Pandas, and converts them to a
dictionary with Series (of columns) per statistic using 
[BaseAccessor.to_dict](/api/base/accessors/#vectorbtpro.base.accessors.BaseAccessor.to_dict).
The `dict` type instructs vectorbt to make multiple metrics out of one. 
5. The number of entry signals
6. [PlotsBuilderMixin.plots](/api/generic/plots_builder/#vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots)
can automatically recognize the arguments the plotting function takes, and pass the 
necessary information. For example, by seeing `self`, it will pass the indicator instance.
7. Since we are already plotting multiple time series, we cannot plot multiple columns at once,
so we must select the provided column from each time series using 
[Wrapping.select_col_from_obj](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping.select_col_from_obj)
8. `fig` contains the figure and `add_trace_kwargs` contains the specification for the subplot 
this time series should be plotted over. Note how the function returns nothing: the figure is being
modified in-place.
9. Metric and subplot dictionaries are passed to the constructor

Calculate the metrics:

```pycon
>>> cross_sig.stats(column=(2, 4, 'a'))
Start                2020-01-01 00:00:00
End                  2020-01-07 00:00:00
Period                   7 days 00:00:00
Fast Stats: count                    6.0
Fast Stats: mean                     2.0
Fast Stats: std                 0.547723
Fast Stats: min                      1.5
Fast Stats: max                      2.5
Slow Stats: count                    4.0
Slow Stats: mean                     2.0
Slow Stats: std                      0.0
Slow Stats: min                      2.0
Slow Stats: max                      2.0
Entries                                1
Exits                                  1
Name: (2, 4, a), dtype: object

>>> cross_sig.plots(column=(2, 4, 'a'))
```

Plot the subplots:

```pycon
>>> cross_sig.plots(column=(2, 4, 'a'))
```

![](/assets/images/indicators_plots.svg)

We have created a smart indicator, yay! :partying_face:

## Extending

Indicator classes can be extended and modified just as regular Python classes - by subclassing. 
Let's make the newly created functions `plot_mas` and `plot_signals` methods of the indicator 
class, such that we can plot both graphs separately. We will also redefine the `subplots`
config to account for this change:

```pycon
>>> class SmartCrossSig(CrossSig):
...     def plot_mas(self, column=None, add_trace_kwargs=None, fig=None):
...         ts = self.select_col_from_obj(self.ts, column).rename('TS')
...         fast_ma = self.select_col_from_obj(self.fast_ma, column).rename('Fast MA')
...         slow_ma = self.select_col_from_obj(self.slow_ma, column).rename('Slow MA')
...         fig = ts.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...         fast_ma.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...         slow_ma.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...         return fig  # (1)!
...
...     def plot_signals(self, column=None, add_trace_kwargs=None, fig=None):
...         signal_type = self.select_col_from_obj(self.signal_type, column)
...         entries = (signal_type == SignalType.Entry).rename('Entries')
...         exits = (signal_type == SignalType.Exit).rename('Exits')
...         fig = entries.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...         exits.vbt.plot(add_trace_kwargs=add_trace_kwargs, fig=fig)
...         return fig
...
...     subplots = vbt.HybridConfig(  # (2)!
...         mas=dict(
...             title="Moving averages",
...             plot_func='plot_mas'  # (3)!
...         ),
...         signals=dict(
...             title="Signals",
...             plot_func='plot_signals'
...         )
...     )

>>> smart_cross_sig = SmartCrossSig.run(ts, [2, 3], 4)
>>> smart_cross_sig.plot_signals(column=(2, 4, 'a'))
```

1. Since both plotting functions can now be invoked by the user, we need to return the figure
in case it wasn't provided
2. We need to use [Config](/api/utils/config/#vectorbtpro.utils.config.Config) instead of `dict` when
setting `metrics` and `subplots` manually
3. Since both functions have become attributes of the class, we can specify them as strings
to make use of vectorbt's [attribute resolution](../../building-blocks/#attribute-resolution)

![](/assets/images/indicators_plot_signals.svg)
