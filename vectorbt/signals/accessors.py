# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Custom pandas accessors for signals data.

Methods can be accessed as follows:

* `SignalsSRAccessor` -> `pd.Series.vbt.signals.*`
* `SignalsDFAccessor` -> `pd.DataFrame.vbt.signals.*`

```python-repl
>>> import vectorbt as vbt
>>> from vectorbt.signals.enums import StopType
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> # vectorbt.signals.accessors.SignalsAccessor.pos_rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.pos_rank()
0    0
1    1
2    2
3    3
4    0
dtype: int64
```

The accessors extend `vectorbt.generic.accessors`.

!!! note
    The underlying Series/DataFrame must already be a signal series and have boolean data type.

    Grouping is only supported by the methods that accept the `group_by` argument.

    Accessors do not utilize caching.

Run for the examples below:
    
```python-repl
>>> mask = pd.DataFrame({
...     'a': [True, False, False, False, False],
...     'b': [True, False, True, False, True],
...     'c': [True, True, True, False, False]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> mask
                a      b      c
2020-01-01   True   True   True
2020-01-02  False  False   True
2020-01-03  False   True   True
2020-01-04  False  False  False
2020-01-05  False   True  False
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `SignalsAccessor.metrics`.

```python-repl
>>> mask.vbt.signals.stats(column='a')
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         1
Rate [%]                                     20
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance: Min                               NaT
Distance: Max                               NaT
Distance: Mean                              NaT
Distance: Std                               NaT
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std                       NaT
Partition Distance: Min                     NaT
Partition Distance: Max                     NaT
Partition Distance: Mean                    NaT
Partition Distance: Std                     NaT
Name: a, dtype: object
```

We can pass another signal array to compare this array with:

```python-repl
>>> mask.vbt.signals.stats(column='a', settings=dict(other=mask['b']))
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         1
Rate [%]                                     20
Total Overlapping                             1
Overlapping Rate [%]                    33.3333
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance -> Other: Min          0 days 00:00:00
Distance -> Other: Max          0 days 00:00:00
Distance -> Other: Mean         0 days 00:00:00
Distance -> Other: Std                      NaT
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std                       NaT
Partition Distance: Min                     NaT
Partition Distance: Max                     NaT
Partition Distance: Mean                    NaT
Partition Distance: Std                     NaT
Name: a, dtype: object
```

We can also return duration as a floating number rather than a timedelta:

```python-repl
>>> mask.vbt.signals.stats(column='a', settings=dict(to_timedelta=False))
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                                        5
Total                                         1
Rate [%]                                     20
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-01 00:00:00
Norm Avg Index [-1, 1]                       -1
Distance: Min                               NaN
Distance: Max                               NaN
Distance: Mean                              NaN
Distance: Std                               NaN
Total Partitions                              1
Partition Rate [%]                          100
Partition Length: Min                         1
Partition Length: Max                         1
Partition Length: Mean                        1
Partition Length: Std                       NaN
Partition Distance: Min                     NaN
Partition Distance: Max                     NaN
Partition Distance: Mean                    NaN
Partition Distance: Std                     NaN
Name: a, dtype: object
```

`SignalsAccessor.stats` also supports (re-)grouping:

```python-repl
>>> mask.vbt.signals.stats(column=0, group_by=[0, 0, 1])
Start                       2020-01-01 00:00:00
End                         2020-01-05 00:00:00
Period                          5 days 00:00:00
Total                                         4
Rate [%]                                     40
First Index                 2020-01-01 00:00:00
Last Index                  2020-01-05 00:00:00
Norm Avg Index [-1, 1]                    -0.25
Distance: Min                   2 days 00:00:00
Distance: Max                   2 days 00:00:00
Distance: Mean                  2 days 00:00:00
Distance: Std                   0 days 00:00:00
Total Partitions                              4
Partition Rate [%]                          100
Partition Length: Min           1 days 00:00:00
Partition Length: Max           1 days 00:00:00
Partition Length: Mean          1 days 00:00:00
Partition Length: Std           0 days 00:00:00
Partition Distance: Min         2 days 00:00:00
Partition Distance: Max         2 days 00:00:00
Partition Distance: Mean        2 days 00:00:00
Partition Distance: Std         0 days 00:00:00
Name: 0, dtype: object
```

## Plots

!!! hint
    See `vectorbt.generic.plots_builder.PlotsBuilderMixin.plots` and `SignalsAccessor.subplots`.

This class inherits subplots from `vectorbt.generic.accessors.GenericAccessor`.
"""

import warnings

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.base import chunking as base_ch
from vectorbt.base import reshaping
from vectorbt.base.wrapping import ArrayWrapper
from vectorbt.ch_registry import ch_registry
from vectorbt.generic import nb as generic_nb
from vectorbt.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor
from vectorbt.generic.ranges import Ranges
from vectorbt.jit_registry import jit_registry
from vectorbt.records.mapped_array import MappedArray
from vectorbt.root_accessors import register_vbt_accessor, register_df_vbt_accessor, register_sr_vbt_accessor
from vectorbt.signals import nb
from vectorbt.utils import checks
from vectorbt.utils import chunking as ch
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbt.utils.decorators import class_or_instancemethod
from vectorbt.utils.random_ import set_seed_nb
from vectorbt.utils.template import RepEval, deep_substitute

__pdoc__ = {}


@register_vbt_accessor('signals')
class SignalsAccessor(GenericAccessor):
    """Accessor on top of signal series. For both, Series and DataFrames.

    Accessible via `pd.Series.vbt.signals` and `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj: tp.SeriesFrame, **kwargs) -> None:
        checks.assert_dtype(obj, np.bool_)

        GenericAccessor.__init__(self, obj, **kwargs)

    @property
    def sr_accessor_cls(self) -> tp.Type["SignalsSRAccessor"]:
        """Accessor class for `pd.Series`."""
        return SignalsSRAccessor

    @property
    def df_accessor_cls(self) -> tp.Type["SignalsDFAccessor"]:
        """Accessor class for `pd.DataFrame`."""
        return SignalsDFAccessor

    # ############# Overriding ############# #

    def bshift(self, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.generic.accessors.GenericAccessor.bshift` with `fill_value=False`."""
        return GenericAccessor.bshift(self, *args, fill_value=fill_value, **kwargs)

    def fshift(self, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.generic.accessors.GenericAccessor.fshift` with `fill_value=False`."""
        return GenericAccessor.fshift(self, *args, fill_value=fill_value, **kwargs)

    @classmethod
    def empty(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.base.accessors.BaseAccessor.empty` with `fill_value=False`."""
        return GenericAccessor.empty(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    @classmethod
    def empty_like(cls, *args, fill_value: bool = False, **kwargs) -> tp.SeriesFrame:
        """`vectorbt.base.accessors.BaseAccessor.empty_like` with `fill_value=False`."""
        return GenericAccessor.empty_like(*args, fill_value=fill_value, dtype=np.bool_, **kwargs)

    # ############# Generation ############# #

    @classmethod
    def generate(cls,
                 shape: tp.ShapeLike,
                 place_func_nb: tp.PlaceFunc, *args,
                 broadcast_named_args: tp.KwargsLike = None,
                 broadcast_kwargs: tp.KwargsLike = None,
                 template_mapping: tp.Optional[tp.Mapping] = None,
                 jitted: tp.JittedOption = None,
                 chunked: tp.ChunkedOption = None,
                 wrapper: tp.Optional[ArrayWrapper] = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """See `vectorbt.signals.nb.generate_nb`.

        ## Example

        Generate random signals manually:

        ```python-repl
        >>> @njit
        ... def place_func_nb(out, from_i, to_i, col):
        ...     out[np.random.choice(len(out))] = True

        >>> vbt.pd_acc.signals.generate(
        ...     (5, 3),
        ...     place_func_nb,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
                        a      b      c
        2020-01-01   True  False  False
        2020-01-02  False   True  False
        2020-01-03  False  False   True
        2020-01-04  False  False  False
        2020-01-05  False  False  False
        ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_mapping is None:
            template_mapping = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(broadcast_named_args, to_shape=shape_2d, **broadcast_kwargs)
        template_mapping = merge_dicts(
            broadcast_named_args,
            dict(shape=shape, shape_2d=shape_2d),
            template_mapping
        )
        args = deep_substitute(args, template_mapping, sub_id='args')
        func = jit_registry.resolve_option(nb.generate_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        result = func(shape_2d, place_func_nb, *args)

        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result, **wrap_kwargs)

    @classmethod
    def generate_both(cls,
                      shape: tp.ShapeLike,
                      entry_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
                      entry_args: tp.ArgsLike = None,
                      exit_place_func_nb: tp.Optional[tp.PlaceFunc] = None,
                      exit_args: tp.ArgsLike = None,
                      entry_wait: int = 1,
                      exit_wait: int = 1,
                      max_one_entry: bool = True,
                      max_one_exit: bool = True,
                      broadcast_named_args: tp.KwargsLike = None,
                      broadcast_kwargs: tp.KwargsLike = None,
                      template_mapping: tp.Optional[tp.Mapping] = None,
                      jitted: tp.JittedOption = None,
                      chunked: tp.ChunkedOption = None,
                      wrapper: tp.Optional[ArrayWrapper] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """See `vectorbt.signals.nb.generate_enex_nb`.

        !!! note
            Make sure that both functions return one signal at most. Otherwise, set `max_one_entry` and/or
            `max_one_exit` to False. In this case, the generator will search for the last signal and proceed
            with placing opposite signals right after it. This makes generation slower.

        ## Example

        Generate entry and exit signals one after another:

        ```python-repl
        >>> @njit
        ... def place_func_nb(out, from_i, to_i, col):
        ...     out[0] = True

        >>> en, ex = vbt.pd_acc.signals.generate_both(
        ...     (5, 3),
        ...     entry_place_func_nb=place_func_nb,
        ...     exit_place_func_nb=place_func_nb,
        ...     max_one_entry=True,
        ...     max_one_exit=True,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
        >>> en
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False  False  False
        2020-01-03   True   True   True
        2020-01-04  False  False  False
        2020-01-05   True   True   True
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True   True
        2020-01-03  False  False  False
        2020-01-04   True   True   True
        2020-01-05  False  False  False
        ```

        Generate three entries and one exit one after another:

        ```python-repl
        >>> @njit
        ... def entry_place_func_nb(out, from_i, to_i, col, n):
        ...     out[:n] = True

        >>> @njit
        ... def exit_place_func_nb(out, from_i, to_i, col, n):
        ...     out[:n] = True

        >>> en, ex = vbt.pd_acc.signals.generate_both(
        ...     (5, 3),
        ...     entry_place_func_nb=entry_place_func_nb,
        ...     entry_args=(3,),
        ...     exit_place_func_nb=exit_place_func_nb,
        ...     exit_args=(1,),
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
        >>> en
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02   True   True   True
        2020-01-03   True   True   True
        2020-01-04  False  False  False
        2020-01-05   True   True   True
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04   True   True   True
        2020-01-05  False  False  False
        ```
        """
        if entry_args is None:
            entry_args = ()
        if exit_args is None:
            exit_args = ()
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_mapping is None:
            template_mapping = {}

        shape_2d = cls.resolve_shape(shape)
        if len(broadcast_named_args) > 0:
            broadcast_named_args = reshaping.broadcast(broadcast_named_args, to_shape=shape_2d, **broadcast_kwargs)
        template_mapping = merge_dicts(
            broadcast_named_args,
            dict(
                shape=shape, shape_2d=shape_2d,
                entry_wait=entry_wait, exit_wait=exit_wait,
                max_one_entry=max_one_entry, max_one_exit=max_one_exit
            ),
            template_mapping
        )
        entry_args = deep_substitute(entry_args, template_mapping, sub_id='entry_args')
        exit_args = deep_substitute(exit_args, template_mapping, sub_id='exit_args')
        func = jit_registry.resolve_option(nb.generate_enex_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        result1, result2 = func(
            shape_2d,
            entry_wait,
            exit_wait,
            max_one_entry,
            max_one_exit,
            entry_place_func_nb, entry_args,
            exit_place_func_nb, exit_args
        )
        if wrapper is None:
            wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
        if wrap_kwargs is None:
            wrap_kwargs = resolve_dict(wrap_kwargs)
        return wrapper.wrap(result1, **wrap_kwargs), wrapper.wrap(result2, **wrap_kwargs)

    def generate_exits(self,
                       exit_place_func_nb: tp.PlaceFunc, *args,
                       wait: int = 1,
                       until_next: bool = True,
                       skip_until_exit: bool = False,
                       broadcast_named_args: tp.KwargsLike = None,
                       broadcast_kwargs: tp.KwargsLike = None,
                       template_mapping: tp.Optional[tp.Mapping] = None,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """See `vectorbt.signals.nb.generate_ex_nb`.

        ## Example

        Generate an exit just before the next entry:

        ```python-repl
        >>> @njit
        ... def exit_place_func_nb(out, from_i, to_i, col):
        ...     out[-1] = True

        >>> mask.vbt.signals.generate_exits(exit_place_func_nb)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False   True  False
        2020-01-03  False  False  False
        2020-01-04  False   True  False
        2020-01-05   True  False   True
        ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_mapping is None:
            template_mapping = {}

        obj = self.obj
        if len(broadcast_named_args) > 0:
            broadcast_named_args = {'obj': obj, **broadcast_named_args}
            broadcast_kwargs = merge_dicts(dict(to_pd=False, post_func=reshaping.to_2d_array), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args, return_wrapper=True, **broadcast_kwargs)
            obj = broadcast_named_args['obj']
        else:
            wrapper = self.wrapper
            obj = reshaping.to_2d_array(obj)
        template_mapping = merge_dicts(
            broadcast_named_args,
            dict(wait=wait, until_next=until_next, skip_until_exit=skip_until_exit),
            template_mapping
        )
        args = deep_substitute(args, template_mapping, sub_id='args')
        func = jit_registry.resolve_option(nb.generate_ex_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        exits = func(
            obj,
            wait,
            until_next,
            skip_until_exit,
            exit_place_func_nb,
            *args
        )
        return wrapper.wrap(exits, group_by=False, **resolve_dict(wrap_kwargs))

    # ############# Filtering ############# #

    @class_or_instancemethod
    def clean(cls_or_self,
              *args,
              entry_first: bool = True,
              broadcast_kwargs: tp.KwargsLike = None,
              jitted: tp.JittedOption = None,
              chunked: tp.ChunkedOption = None,
              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Clean signals.

        If one array passed, see `SignalsAccessor.first`.
        If two arrays passed, entries and exits, see `vectorbt.signals.nb.clean_enex_nb`."""
        if wrap_kwargs is None:
            wrap_kwargs = {}
        if not isinstance(cls_or_self, type):
            args = (cls_or_self.obj, *args)
        if len(args) == 1:
            obj = args[0]
            if not isinstance(obj, (pd.Series, pd.DataFrame)):
                obj = ArrayWrapper.from_obj(obj).wrap(obj)
            return obj.vbt.signals.first(wrap_kwargs=wrap_kwargs, jitted=jitted, chunked=chunked)
        if len(args) == 2:
            if broadcast_kwargs is None:
                broadcast_kwargs = {}
            broadcasted_args = reshaping.broadcast(dict(entries=args[0], exits=args[1]), **broadcast_kwargs)
            entries = broadcasted_args['entries']
            exits = broadcasted_args['exits']
            func = jit_registry.resolve_option(nb.clean_enex_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            entries_out, exits_out = func(
                reshaping.to_2d_array(entries),
                reshaping.to_2d_array(exits),
                entry_first
            )
            return (
                ArrayWrapper.from_obj(entries).wrap(entries_out, group_by=False, **wrap_kwargs),
                ArrayWrapper.from_obj(exits).wrap(exits_out, group_by=False, **wrap_kwargs)
            )
        raise ValueError("Either one or two arrays must be passed")

    # ############# Random signals ############# #

    @classmethod
    def generate_random(cls,
                        shape: tp.ShapeLike,
                        n: tp.Optional[tp.ArrayLike] = None,
                        prob: tp.Optional[tp.ArrayLike] = None,
                        pick_first: bool = False,
                        seed: tp.Optional[int] = None,
                        jitted: tp.JittedOption = None,
                        chunked: tp.ChunkedOption = None,
                        **kwargs) -> tp.SeriesFrame:
        """Generate signals randomly.

        If `n` is set, uses `vectorbt.signals.nb.rand_place_nb`.
        If `prob` is set, uses `vectorbt.signals.nb.rand_by_prob_place_nb`.

        For arguments, see `SignalsAccessor.generate_random`.

        `n` must be either a scalar or an array that will broadcast to the number of columns.
        `prob` must be either a single number or an array that will broadcast to match `shape`.

        Specify `seed` to make output deterministic.

        ## Example

        For each column, generate a variable number of signals:

        ```python-repl
        >>> vbt.pd_acc.signals.generate_random(
        ...     (5, 3),
        ...     n=[0, 1, 2],
        ...     seed=42,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False   True
        2020-01-04  False   True  False
        2020-01-05  False  False   True
        ```

        For each column and time step, pick a signal with 50% probability:

        ```python-repl
        >>> vbt.pd_acc.signals.generate_random(
        ...     (5, 3),
        ...     prob=0.5,
        ...     seed=42,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False   True  False
        2020-01-03  False  False  False
        2020-01-04  False  False   True
        2020-01-05   True  False   True
        ```
        """
        shape_2d = cls.resolve_shape(shape)
        if n is not None and prob is not None:
            raise ValueError("Either n or prob must be provided, not both")

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = np.broadcast_to(n, (shape_2d[1],))
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(
                    base_ch.FlexArraySlicer(axis=1, flex_2d=True)
                ))
            )
            return cls.generate(
                shape,
                jit_registry.resolve_option(nb.rand_place_nb, jitted),
                n,
                jitted=jitted,
                chunked=chunked,
                **kwargs
            )
        if prob is not None:
            prob = np.broadcast_to(prob, shape)
            flex_2d = isinstance(shape, tuple) and len(shape) > 1
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(
                    base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                    None,
                    None
                ))
            )
            return cls.generate(
                shape,
                jit_registry.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob, pick_first, flex_2d,
                jitted=jitted,
                chunked=chunked,
                **kwargs
            )
        raise ValueError("At least n or prob must be provided")

    @classmethod
    def generate_random_both(cls,
                             shape: tp.ShapeLike,
                             n: tp.Optional[tp.ArrayLike] = None,
                             entry_prob: tp.Optional[tp.ArrayLike] = None,
                             exit_prob: tp.Optional[tp.ArrayLike] = None,
                             seed: tp.Optional[int] = None,
                             entry_wait: int = 1,
                             exit_wait: int = 1,
                             entry_pick_first: bool = True,
                             exit_pick_first: bool = True,
                             jitted: tp.JittedOption = None,
                             chunked: tp.ChunkedOption = None,
                             wrapper: tp.Optional[ArrayWrapper] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.Tuple[tp.SeriesFrame, tp.SeriesFrame]:
        """Generate chain of entry and exit signals randomly.

        If `n` is set, uses `vectorbt.signals.nb.generate_rand_enex_nb`.
        If `entry_prob` and `exit_prob` are set, uses `SignalsAccessor.generate_both` with
        `vectorbt.signals.nb.rand_by_prob_place_nb`.

        ## Example

        For each column, generate two entries and exits randomly:

        ```python-repl
        >>> en, ex = vbt.pd_acc.signals.generate_random_both(
        ...     (5, 3),
        ...     n=2,
        ...     seed=42,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
        >>> en
                        a      b      c
        2020-01-01  False  False   True
        2020-01-02   True   True  False
        2020-01-03  False  False  False
        2020-01-04   True   True   True
        2020-01-05  False  False  False
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False   True
        2020-01-03   True   True  False
        2020-01-04  False  False  False
        2020-01-05   True   True   True
        ```

        For each column and time step, pick entry with 50% probability and exit right after:

        ```python-repl
        >>> en, ex = vbt.pd_acc.signals.generate_random_both(
        ...     (5, 3),
        ...     entry_prob=0.5,
        ...     exit_prob=1.,
        ...     seed=42,
        ...     wrap_kwargs=dict(
        ...         index=mask.index,
        ...         columns=mask.columns
        ...     )
        ... )
        >>> en
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04  False  False   True
        2020-01-05   True  False  False
        >>> ex
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True   True
        2020-01-03  False  False  False
        2020-01-04  False  False  False
        2020-01-05  False  False   True
        ```
        """
        shape_2d = cls.resolve_shape(shape)
        if n is not None and (entry_prob is not None or exit_prob is not None):
            raise ValueError("Either n or any of the entry_prob and exit_prob must be provided, not both")

        if seed is not None:
            set_seed_nb(seed)
        if n is not None:
            n = np.broadcast_to(n, (shape_2d[1],))
            func = jit_registry.resolve_option(nb.generate_rand_enex_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            entries, exits = func(shape_2d, n, entry_wait, exit_wait)
            if wrapper is None:
                wrapper = ArrayWrapper.from_shape(shape_2d, ndim=cls.ndim)
            if wrap_kwargs is None:
                wrap_kwargs = resolve_dict(wrap_kwargs)
            return wrapper.wrap(entries, **wrap_kwargs), wrapper.wrap(exits, **wrap_kwargs)
        elif entry_prob is not None and exit_prob is not None:
            entry_prob = np.broadcast_to(entry_prob, shape)
            exit_prob = np.broadcast_to(exit_prob, shape)
            flex_2d = isinstance(shape, tuple) and len(shape) > 1
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None
                    ),
                    exit_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None
                    )
                )
            )
            return cls.generate_both(
                shape,
                entry_place_func_nb=jit_registry.resolve_option(nb.rand_by_prob_place_nb, jitted),
                entry_args=(
                    entry_prob,
                    entry_pick_first,
                    flex_2d
                ),
                exit_place_func_nb=jit_registry.resolve_option(nb.rand_by_prob_place_nb, jitted),
                exit_args=(
                    exit_prob,
                    exit_pick_first,
                    flex_2d
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                max_one_entry=entry_pick_first,
                max_one_exit=exit_pick_first,
                jitted=jitted,
                chunked=chunked,
                wrapper=wrapper,
                wrap_kwargs=wrap_kwargs
            )
        raise ValueError("At least n, or entry_prob and exit_prob must be provided")

    def generate_random_exits(self,
                              prob: tp.Optional[tp.ArrayLike] = None,
                              seed: tp.Optional[int] = None,
                              wait: int = 1,
                              until_next: bool = True,
                              skip_until_exit: bool = False,
                              broadcast_kwargs: tp.KwargsLike = None,
                              jitted: tp.JittedOption = None,
                              chunked: tp.ChunkedOption = None,
                              wrap_kwargs: tp.KwargsLike = None,
                              **kwargs) -> tp.SeriesFrame:
        """Generate exit signals randomly.

        If `prob` is None, uses `vectorbt.signals.nb.rand_place_nb`.
        Otherwise, uses `vectorbt.signals.nb.rand_by_prob_place_nb`.

        Uses `SignalsAccessor.generate_exits`.

        Specify `seed` to make output deterministic.

        ## Example

        After each entry in `mask`, generate exactly one exit:

        ```python-repl
        >>> mask.vbt.signals.generate_random_exits(seed=42)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False   True  False
        2020-01-03  False  False  False
        2020-01-04   True   True  False
        2020-01-05  False  False   True
        ```

        After each entry in `mask` and at each time step, generate exit with 50% probability:

        ```python-repl
        >>> mask.vbt.signals.generate_random_exits(prob=0.5, seed=42)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True  False  False
        2020-01-03  False  False  False
        2020-01-04  False  False  False
        2020-01-05  False  False   True
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if seed is not None:
            set_seed_nb(seed)
        if prob is not None:
            broadcast_kwargs = merge_dicts(dict(keep_flex=dict(obj=False, prob=True)), broadcast_kwargs)
            broadcasted_args = reshaping.broadcast(dict(obj=self.obj, prob=prob), **broadcast_kwargs)
            obj = broadcasted_args['obj']
            prob = broadcasted_args['prob']
            flex_2d = obj.ndim == 2
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(args=ch.ArgsTaker(
                    base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                    None,
                    None
                ))
            )
            return obj.vbt.signals.generate_exits(
                jit_registry.resolve_option(nb.rand_by_prob_place_nb, jitted),
                prob, True, flex_2d,
                wait=wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )
        n = np.broadcast_to(1, (self.wrapper.shape_2d[1],))
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(args=ch.ArgsTaker(
                base_ch.FlexArraySlicer(axis=1, flex_2d=True)
            ))
        )
        return self.generate_exits(
            jit_registry.resolve_option(nb.rand_place_nb, jitted),
            n,
            wait=wait,
            until_next=until_next,
            skip_until_exit=skip_until_exit,
            jitted=jitted,
            chunked=chunked,
            wrap_kwargs=wrap_kwargs,
            **kwargs
        )

    # ############# Stop signals ############# #

    def generate_stop_exits(self,
                            ts: tp.ArrayLike,
                            stop: tp.ArrayLike,
                            trailing: tp.ArrayLike = False,
                            entry_wait: int = 1,
                            exit_wait: int = 1,
                            until_next: bool = True,
                            skip_until_exit: bool = False,
                            pick_first: bool = True,
                            chain: bool = False,
                            broadcast_kwargs: tp.KwargsLike = None,
                            jitted: tp.JittedOption = None,
                            chunked: tp.ChunkedOption = None,
                            wrap_kwargs: tp.KwargsLike = None,
                            **kwargs) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when `ts` hits the stop.

        For arguments, see `vectorbt.signals.nb.stop_place_nb`.
        If `chain` is True, uses `SignalsAccessor.generate_both`.
        Otherwise, uses `SignalsAccessor.generate_exits`.

        Arguments `entries`, `ts` and `stop` will broadcast using `vectorbt.base.reshaping.broadcast`
        and `broadcast_kwargs`.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that come between an entry and its exit,
            set `until_next` to False and `skip_until_exit` to True.

            To remove all entries that come between an entry and its exit,
            set `chain` to True. This will return two arrays: new entries and exits.

        ## Example

        * Regular stop loss:

        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1])

        >>> mask.vbt.signals.generate_stop_exits(ts, -0.1)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04  False   True   True
        2020-01-05  False  False  False
        ```

        * Trailing stop loss:

        ```python-repl
        >>> mask.vbt.signals.generate_stop_exits(ts, -0.1, trailing=True)
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04   True   True   True
        2020-01-05  False  False  False
        ```

        * Testing multiple take profit stops:

        ```python-repl
        >>> mask.vbt.signals.generate_stop_exits(ts, pd.Index([1.0, 1.5]))
        stop                        1.0                  1.5
                        a      b      c      a      b      c
        2020-01-01  False  False  False  False  False  False
        2020-01-02   True   True  False  False  False  False
        2020-01-03  False  False  False   True  False  False
        2020-01-04  False  False  False  False  False  False
        2020-01-05  False  False  False  False  False  False
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self.obj

        broadcastable_args = dict(
            entries=entries,
            ts=ts,
            stop=stop,
            trailing=trailing
        )
        broadcast_kwargs = merge_dicts(dict(
            keep_flex=dict(
                entries=False,
                _default=True
            ),
            require_kwargs=dict(
                requirements='W'
            )
        ), broadcast_kwargs)
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args['entries']
        flex_2d = entries.ndim == 2

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(
                        ch.ArraySlicer(axis=1)
                    ),
                    exit_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None,
                        None
                    )
                )
            )
            return cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_registry.resolve_option(nb.first_place_nb, jitted),
                entry_args=(entries_arr,),
                exit_place_func_nb=jit_registry.resolve_option(nb.stop_place_nb, jitted),
                exit_args=(
                    broadcasted_args['ts'],
                    broadcasted_args['stop'],
                    broadcasted_args['trailing'],
                    exit_wait,
                    pick_first,
                    flex_2d
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                max_one_entry=True,
                max_one_exit=pick_first,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )
        else:
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None,
                        None
                    )
                )
            )
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)
            return entries.vbt.signals.generate_exits(
                jit_registry.resolve_option(nb.stop_place_nb, jitted),
                broadcasted_args['ts'],
                broadcasted_args['stop'],
                broadcasted_args['trailing'],
                exit_wait,
                pick_first,
                flex_2d,
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )

    def generate_ohlc_stop_exits(self,
                                 open: tp.ArrayLike,
                                 high: tp.ArrayLike = np.nan,
                                 low: tp.ArrayLike = np.nan,
                                 close: tp.ArrayLike = np.nan,
                                 is_open_safe: bool = True,
                                 out_dict: tp.Optional[tp.Dict[str, tp.ArrayLike]] = None,
                                 sl_stop: tp.ArrayLike = np.nan,
                                 sl_trail: tp.ArrayLike = False,
                                 tp_stop: tp.ArrayLike = np.nan,
                                 reverse: tp.ArrayLike = False,
                                 entry_wait: int = 1,
                                 exit_wait: int = 1,
                                 until_next: bool = True,
                                 skip_until_exit: bool = False,
                                 pick_first: bool = True,
                                 chain: bool = False,
                                 broadcast_kwargs: tp.KwargsLike = None,
                                 jitted: tp.JittedOption = None,
                                 chunked: tp.ChunkedOption = None,
                                 wrap_kwargs: tp.KwargsLike = None,
                                 **kwargs) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Generate exits based on when the price hits (trailing) stop loss or take profit.

        !!! hint
            This function is meant for signal analysis. For backtesting, consider using
            the stop logic integrated into `vectorbt.portfolio.base.Portfolio.from_signals`.

        Use `out_dict` as a dict to pass `stop_price` and `stop_type` arrays. You can also
        set `out_dict` to {} to produce these arrays automatically and still have access to them.

        For arguments, see `vectorbt.signals.nb.ohlc_stop_place_nb`.
        If `chain` is True, uses `SignalsAccessor.generate_both`.
        Otherwise, uses `SignalsAccessor.generate_exits`.

        All array-like arguments including stops and `out_dict` will broadcast using
        `vectorbt.base.reshaping.broadcast` and `broadcast_kwargs`.

        For arguments, see `vectorbt.signals.nb.ohlc_stop_place_nb`.

        !!! note
            `open` isn't necessarily open price, but can be any entry price (even previous close).
            Stop price is calculated based solely on the entry price.

        !!! hint
            Default arguments will generate an exit signal strictly between two entry signals.
            If both entry signals are too close to each other, no exit will be generated.

            To ignore all entries that come between an entry and its exit,
            set `until_next` to False and `skip_until_exit` to True.

            To remove all entries that come between an entry and its exit,
            set `chain` to True. This will return two arrays: new entries and exits.

        ## Example

        ```python-repl
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10],
        ...     'high': [11, 12, 13, 12, 11],
        ...     'low': [9, 10, 11, 10, 9],
        ...     'close': [10, 11, 12, 11, 10]
        ... })
        >>> out_dict = {}
        >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=0.1,
        ...     sl_trail=True,
        ...     tp_stop=0.1,
        ...     out_dict=out_dict
        ... )
        >>> exits
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True  False
        2020-01-03  False  False  False
        2020-01-04  False   True   True
        2020-01-05  False  False  False

        >>> out_dict['stop_price']
                       a     b     c
        2020-01-01   NaN   NaN   NaN
        2020-01-02  11.0  11.0   NaN
        2020-01-03   NaN   NaN   NaN
        2020-01-04   NaN  10.8  10.8
        2020-01-05   NaN   NaN   NaN

        >>> out_dict['stop_type'].vbt(mapping=StopType).apply_mapping()
                             a           b          c
        2020-01-01        None        None       None
        2020-01-02  TakeProfit  TakeProfit       None
        2020-01-03        None        None       None
        2020-01-04        None   TrailStop  TrailStop
        2020-01-05        None        None       None
        ```

        Notice how the first two entry signals in the third column have no exit signal - there is
        no room between them for an exit signal. To find an exit for the first entry and ignore all
        entries that are in-between them, we can pass `until_next=False` and `skip_until_exit=True`:

        ```python-repl
        >>> out_dict = {}
        >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=0.1,
        ...     sl_trail=True,
        ...     tp_stop=0.1,
        ...     out_dict=out_dict,
        ...     until_next=False,
        ...     skip_until_exit=True
        ... )
        >>> exits
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True   True
        2020-01-03  False  False  False
        2020-01-04  False   True   True
        2020-01-05  False  False  False

        >>> out_dict['stop_price']
                       a     b     c
        2020-01-01   NaN   NaN   NaN
        2020-01-02  11.0  11.0  11.0
        2020-01-03   NaN   NaN   NaN
        2020-01-04   NaN  10.8  10.8
        2020-01-05   NaN   NaN   NaN

        >>> out_dict['stop_type'].vbt(mapping=StopType).apply_mapping()
                             a           b           c
        2020-01-01        None        None        None
        2020-01-02  TakeProfit  TakeProfit  TakeProfit
        2020-01-03        None        None        None
        2020-01-04        None   TrailStop   TrailStop
        2020-01-05        None        None        None
        ```

        Now, the first signal in the third column gets executed regardless of the entries that come next,
        which is very similar to the logic that is implemented in `vectorbt.portfolio.base.Portfolio.from_signals`.

        To automatically remove all ignored entry signals, pass `chain=True`.
        This will return a new entries array:

        ```python-repl
        >>> out_dict = {}
        >>> new_entries, exits = mask.vbt.signals.generate_ohlc_stop_exits(
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=0.1,
        ...     sl_trail=True,
        ...     tp_stop=0.1,
        ...     out_dict=out_dict,
        ...     chain=True
        ... )
        >>> new_entries
                        a      b      c
        2020-01-01   True   True   True
        2020-01-02  False  False  False  << removed entry in the third column
        2020-01-03  False   True   True
        2020-01-04  False  False  False
        2020-01-05  False   True  False
        >>> exits
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02   True   True   True
        2020-01-03  False  False  False
        2020-01-04  False   True   True
        2020-01-05  False  False  False
        ```

        !!! warning
            The last two examples above make entries dependent upon exits - this makes only sense
            if you have no other exit arrays to combine this stop exit array with.

        Test multiple parameter combinations:

        ```python-repl
        >>> exits = mask.vbt.signals.generate_ohlc_stop_exits(
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=pd.Index([0.1, 0.2]),
        ...     sl_trail=pd.Index([False, True])
        ... )
        >>> exits
        sl_stop                                          0.1                       \\
        sl_trail                  False                 True                False
                        a      b      c      a      b      c      a      b      c
        2020-01-01  False  False  False  False  False  False  False  False  False
        2020-01-02  False  False  False  False  False  False  False  False  False
        2020-01-03  False  False  False  False  False  False  False  False  False
        2020-01-04  False   True   True   True   True   True  False  False  False
        2020-01-05   True  False  False  False  False  False  False  False   True

        sl_stop                     0.2
        sl_trail                   True
                        a      b      c
        2020-01-01  False  False  False
        2020-01-02  False  False  False
        2020-01-03  False  False  False
        2020-01-04   True  False  False
        2020-01-05  False  False   True
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        entries = self.obj
        if out_dict is None:
            out_dict_passed = False
            out_dict = {}
        else:
            out_dict_passed = True
        stop_price = out_dict.get('stop_price', np.nan if out_dict_passed else None)
        stop_type = out_dict.get('stop_type', -1 if out_dict_passed else None)
        if wrap_kwargs is None:
            wrap_kwargs = {}

        broadcastable_args = dict(
            entries=entries,
            open=open,
            high=high,
            low=low,
            close=close,
            sl_stop=sl_stop,
            sl_trail=sl_trail,
            tp_stop=tp_stop,
            reverse=reverse,
            stop_price=stop_price,
            stop_type=stop_type
        )
        broadcast_kwargs = merge_dicts(dict(
            keep_flex=dict(
                entries=False,
                stop_price=False,
                stop_type=False,
                _default=True
            ),
            require_kwargs=dict(
                requirements='W'
            )
        ), broadcast_kwargs)
        broadcasted_args = reshaping.broadcast(broadcastable_args, **broadcast_kwargs)
        entries = broadcasted_args['entries']
        flex_2d = entries.ndim == 2
        stop_price = broadcasted_args['stop_price']
        if stop_price is None:
            stop_price = np.empty_like(entries, dtype=np.float_)
        stop_price = reshaping.to_2d_array(stop_price)
        stop_type = broadcasted_args['stop_type']
        if stop_type is None:
            stop_type = np.empty_like(entries, dtype=np.int_)
        stop_type = reshaping.to_2d_array(stop_type)

        entries_arr = reshaping.to_2d_array(entries)
        wrapper = ArrayWrapper.from_obj(entries)
        if chain:
            if checks.is_series(entries):
                cls = self.sr_accessor_cls
            else:
                cls = self.df_accessor_cls
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    entry_args=ch.ArgsTaker(
                        ch.ArraySlicer(axis=1)
                    ),
                    exit_args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None,
                        None,
                        None
                    )
                )
            )
            new_entries, exits = cls.generate_both(
                entries.shape,
                entry_place_func_nb=jit_registry.resolve_option(nb.first_place_nb, jitted),
                entry_args=(entries_arr,),
                exit_place_func_nb=jit_registry.resolve_option(nb.ohlc_stop_place_nb, jitted),
                exit_args=(
                    broadcasted_args['open'],
                    broadcasted_args['high'],
                    broadcasted_args['low'],
                    broadcasted_args['close'],
                    stop_price,
                    stop_type,
                    broadcasted_args['sl_stop'],
                    broadcasted_args['sl_trail'],
                    broadcasted_args['tp_stop'],
                    broadcasted_args['reverse'],
                    is_open_safe,
                    exit_wait,
                    pick_first,
                    flex_2d
                ),
                entry_wait=entry_wait,
                exit_wait=exit_wait,
                max_one_entry=True,
                max_one_exit=pick_first,
                wrapper=wrapper,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )
            out_dict['stop_price'] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict['stop_type'] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return new_entries, exits
        else:
            if skip_until_exit and until_next:
                warnings.warn("skip_until_exit=True has only effect when until_next=False", stacklevel=2)
            chunked = ch.specialize_chunked_option(
                chunked,
                arg_take_spec=dict(
                    args=ch.ArgsTaker(
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        ch.ArraySlicer(axis=1),
                        ch.ArraySlicer(axis=1),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        base_ch.FlexArraySlicer(axis=1, flex_2d=flex_2d),
                        None,
                        None,
                        None,
                        None
                    )
                )
            )
            exits = entries.vbt.signals.generate_exits(
                jit_registry.resolve_option(nb.ohlc_stop_place_nb, jitted),
                broadcasted_args['open'],
                broadcasted_args['high'],
                broadcasted_args['low'],
                broadcasted_args['close'],
                stop_price,
                stop_type,
                broadcasted_args['sl_stop'],
                broadcasted_args['sl_trail'],
                broadcasted_args['tp_stop'],
                broadcasted_args['reverse'],
                is_open_safe,
                exit_wait,
                pick_first,
                flex_2d,
                wait=exit_wait,
                until_next=until_next,
                skip_until_exit=skip_until_exit,
                jitted=jitted,
                chunked=chunked,
                wrap_kwargs=wrap_kwargs,
                **kwargs
            )
            out_dict['stop_price'] = wrapper.wrap(stop_price, group_by=False, **wrap_kwargs)
            out_dict['stop_type'] = wrapper.wrap(stop_type, group_by=False, **wrap_kwargs)
            return exits

    # ############# Ranges ############# #

    def between_ranges(self,
                       other: tp.Optional[tp.ArrayLike] = None,
                       from_other: bool = False,
                       broadcast_kwargs: tp.KwargsLike = None,
                       group_by: tp.GroupByLike = None,
                       attach_ts: bool = True,
                       attach_other: bool = False,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       **kwargs) -> Ranges:
        """Wrap the result of `vectorbt.signals.nb.between_ranges_nb`
        with `vectorbt.generic.ranges.Ranges`.

        If `other` specified, see `vectorbt.signals.nb.between_two_ranges_nb`.
        Both will broadcast using `vectorbt.base.reshaping.broadcast` and `broadcast_kwargs`.

        ## Example

        One array:

        ```python-repl
        >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
        >>> ranges = mask_sr.vbt.signals.between_ranges()
        >>> ranges
        <vectorbt.generic.ranges.Ranges at 0x7ff29ea7c7b8>

        >>> ranges.records_readable
           Range Id  Column  Start Timestamp  End Timestamp  Status
        0         0       0                0              3  Closed
        1         1       0                3              5  Closed
        2         2       0                5              6  Closed

        >>> ranges.duration.values
        array([3, 2, 1])
        ```

        Two arrays, traversing the signals of the first array:

        ```python-repl
        >>> mask_sr = pd.Series([True, True, True, False, False])
        >>> mask_sr2 = pd.Series([False, False, True, False, True])
        >>> ranges = mask_sr.vbt.signals.between_ranges(other=mask_sr2)
        >>> ranges
        <vectorbt.generic.ranges.Ranges at 0x7ff29e3b80f0>

        >>> ranges.records_readable
           Range Id  Column  Start Timestamp  End Timestamp  Status
        0         0       0                0              2  Closed
        1         1       0                1              2  Closed
        2         2       0                2              2  Closed

        >>> ranges.duration.values
        array([2, 1, 0])
        ```

        Two arrays, traversing the signals of the second array:

        ```python-repl
        >>> ranges = mask_sr.vbt.signals.between_ranges(other=mask_sr2, from_other=True)
        >>> ranges
        <vectorbt.generic.ranges.Ranges at 0x7ff29eccbd68>

        >>> ranges.records_readable
           Range Id  Column  Start Timestamp  End Timestamp  Status
        0         0       0                2              2  Closed
        1         1       0                2              4  Closed

        >>> ranges.duration.values
        array([0, 2])
        ```
        """
        if broadcast_kwargs is None:
            broadcast_kwargs = {}

        if other is None:
            # One input array
            func = jit_registry.resolve_option(nb.between_ranges_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            range_records = func(self.to_2d_array())
            wrapper = self.wrapper
            to_attach = self.obj
        else:
            # Two input arrays
            broadcasted_args = reshaping.broadcast(dict(obj=self.obj, other=other), **broadcast_kwargs)
            obj = broadcasted_args['obj']
            other = broadcasted_args['other']
            func = jit_registry.resolve_option(nb.between_two_ranges_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            range_records = func(
                reshaping.to_2d_array(obj),
                reshaping.to_2d_array(other),
                from_other=from_other
            )
            wrapper = ArrayWrapper.from_obj(obj)
            to_attach = other if attach_other else obj
        return Ranges(
            wrapper,
            range_records,
            ts=to_attach if attach_ts else None,
            **kwargs
        ).regroup(group_by)

    def partition_ranges(self,
                         group_by: tp.GroupByLike = None,
                         attach_ts: bool = True,
                         jitted: tp.JittedOption = None,
                         chunked: tp.ChunkedOption = None,
                         **kwargs) -> Ranges:
        """Wrap the result of `vectorbt.signals.nb.partition_ranges_nb`
        with `vectorbt.generic.ranges.Ranges`.

        If `use_end_idxs` is True, uses the index of the last signal in each partition as `idx_arr`.
        Otherwise, uses the index of the first signal.

        ## Example

        ```python-repl
        >>> mask_sr = pd.Series([True, True, True, False, True, True])
        >>> mask_sr.vbt.signals.partition_ranges().records_readable
           Range Id  Column  Start Timestamp  End Timestamp  Status
        0         0       0                0              3  Closed
        1         1       0                4              5    Open
        ```"""
        func = jit_registry.resolve_option(nb.partition_ranges_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        return Ranges(
            self.wrapper,
            range_records,
            ts=self.obj if attach_ts else None,
            **kwargs
        ).regroup(group_by)

    def between_partition_ranges(self,
                                 group_by: tp.GroupByLike = None,
                                 attach_ts: bool = True,
                                 jitted: tp.JittedOption = None,
                                 chunked: tp.ChunkedOption = None,
                                 **kwargs) -> Ranges:
        """Wrap the result of `vectorbt.signals.nb.between_partition_ranges_nb`
        with `vectorbt.generic.ranges.Ranges`.

        ## Example

        ```python-repl
        >>> mask_sr = pd.Series([True, False, False, True, False, True, True])
        >>> mask_sr.vbt.signals.between_partition_ranges().records_readable
           Range Id  Column  Start Timestamp  End Timestamp  Status
        0         0       0                0              3  Closed
        1         1       0                3              5  Closed
         ```"""
        func = jit_registry.resolve_option(nb.between_partition_ranges_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        range_records = func(self.to_2d_array())
        return Ranges(
            self.wrapper,
            range_records,
            ts=self.obj if attach_ts else None,
            **kwargs
        ).regroup(group_by)

    # ############# Ranking ############# #

    def rank(self,
             rank_func_nb: tp.RankFunc, *args,
             prepare_func: tp.Optional[tp.Callable] = None,
             reset_by: tp.Optional[tp.ArrayLike] = None,
             after_false: bool = False,
             as_mapped: bool = False,
             broadcast_named_args: tp.KwargsLike = None,
             broadcast_kwargs: tp.KwargsLike = None,
             template_mapping: tp.Optional[tp.Mapping] = None,
             jitted: tp.JittedOption = None,
             chunked: tp.ChunkedOption = None,
             wrap_kwargs: tp.KwargsLike = None,
             **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """See `vectorbt.signals.nb.rank_nb`.

        Will broadcast with `reset_by` using `vectorbt.base.reshaping.broadcast` and `broadcast_kwargs`.

        Use `prepare_func` to prepare further arguments to be passed before `*args`, such as temporary arrays.
        It must take both broadcasted arrays (`reset_by` can be None) and return a tuple.

        Set `as_mapped` to True to return an instance of `vectorbt.records.mapped_array.MappedArray`."""
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_mapping is None:
            template_mapping = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        if reset_by is not None:
            broadcast_named_args = {'obj': self.obj, 'reset_by': reset_by, **broadcast_named_args}
        else:
            broadcast_named_args = {'obj': self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_kwargs = merge_dicts(dict(to_pd=False, post_func=reshaping.to_2d_array), broadcast_kwargs)
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args, return_wrapper=True, **broadcast_kwargs)
        else:
            wrapper = self.wrapper
        obj = reshaping.to_2d_array(broadcast_named_args['obj'])
        if reset_by is not None:
            reset_by = reshaping.to_2d_array(broadcast_named_args['reset_by'])
        if prepare_func is not None:
            temp_arrs = prepare_func(obj, reset_by)
        else:
            temp_arrs = ()
        template_mapping = merge_dicts(
            broadcast_named_args,
            dict(after_false=after_false, temp_arrs=temp_arrs),
            template_mapping
        )
        args = deep_substitute(args, template_mapping, sub_id='args')
        func = jit_registry.resolve_option(nb.rank_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        rank = func(
            obj,
            reset_by,
            after_false,
            rank_func_nb,
            *temp_arrs,
            *args
        )
        rank_wrapped = wrapper.wrap(rank, group_by=False, **wrap_kwargs)
        if as_mapped:
            rank_wrapped = rank_wrapped.replace(-1, np.nan)
            return rank_wrapped.vbt.to_mapped(
                dropna=True,
                dtype=np.int_,
                **kwargs
            )
        return rank_wrapped

    def pos_rank(self,
                 jitted: tp.JittedOption = None,
                 chunked: tp.ChunkedOption = None,
                 allow_gaps: bool = False,
                 **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get signal position ranks.

        Uses `SignalsAccessor.rank` with `vectorbt.signals.nb.sig_pos_rank_nb`.

        ## Example

        Rank each True value in each partition in `mask`:

        ```python-repl
        >>> mask.vbt.signals.pos_rank()
                    a  b  c
        2020-01-01  0  0  0
        2020-01-02 -1 -1  1
        2020-01-03 -1  0  2
        2020-01-04 -1 -1 -1
        2020-01-05 -1  0 -1

        >>> mask.vbt.signals.pos_rank(after_false=True)
                    a  b  c
        2020-01-01 -1 -1 -1
        2020-01-02 -1 -1 -1
        2020-01-03 -1  0 -1
        2020-01-04 -1 -1 -1
        2020-01-05 -1  0 -1

        >>> mask.vbt.signals.pos_rank(allow_gaps=True)
                    a  b  c
        2020-01-01  0  0  0
        2020-01-02 -1 -1  1
        2020-01-03 -1  1  2
        2020-01-04 -1 -1 -1
        2020-01-05 -1  2 -1

        >>> mask.vbt.signals.pos_rank(reset_by=~mask, allow_gaps=True)
                    a  b  c
        2020-01-01  0  0  0
        2020-01-02 -1 -1  1
        2020-01-03 -1  0  2
        2020-01-04 -1 -1 -1
        2020-01-05 -1  0 -1
        ```
        """
        prepare_func = lambda obj, reset_by: (np.full(obj.shape[1], -1, dtype=np.int_),)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(args=ch.ArgsTaker(ch.ArraySlicer(axis=0), None))
        )
        return self.rank(
            jit_registry.resolve_option(nb.sig_pos_rank_nb, jitted),
            allow_gaps,
            prepare_func=prepare_func,
            jitted=jitted,
            chunked=chunked,
            **kwargs
        )

    def partition_pos_rank(self,
                           jitted: tp.JittedOption = None,
                           chunked: tp.ChunkedOption = None,
                           **kwargs) -> tp.Union[tp.SeriesFrame, MappedArray]:
        """Get partition position ranks.

        Uses `SignalsAccessor.rank` with `vectorbt.signals.nb.part_pos_rank_nb`.

        ## Example

        Rank each partition of True values in `mask`:

        ```python-repl
        >>> mask.vbt.signals.partition_pos_rank()
                    a  b  c
        2020-01-01  0  0  0
        2020-01-02 -1 -1  0
        2020-01-03 -1  1  0
        2020-01-04 -1 -1 -1
        2020-01-05 -1  2 -1

        >>> mask.vbt.signals.partition_pos_rank(after_false=True)
                    a  b  c
        2020-01-01 -1 -1 -1
        2020-01-02 -1 -1 -1
        2020-01-03 -1  0 -1
        2020-01-04 -1 -1 -1
        2020-01-05 -1  1 -1

        >>> mask.vbt.signals.partition_pos_rank(reset_by=mask)
                    a  b  c
        2020-01-01  0  0  0
        2020-01-02 -1 -1  0
        2020-01-03 -1  0  0
        2020-01-04 -1 -1 -1
        2020-01-05 -1  0 -1
        ```
        """
        prepare_func = lambda obj, reset_by: (np.full(obj.shape[1], -1, dtype=np.int_),)
        chunked = ch.specialize_chunked_option(
            chunked,
            arg_take_spec=dict(args=ch.ArgsTaker(ch.ArraySlicer(axis=0)))
        )
        return self.rank(
            jit_registry.resolve_option(nb.part_pos_rank_nb, jitted),
            prepare_func=prepare_func,
            jitted=jitted,
            chunked=chunked,
            **kwargs
        )

    def first(self, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == 0`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == 0, group_by=False, **resolve_dict(wrap_kwargs))

    def nth(self, n: int, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank == n`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank == n, group_by=False, **resolve_dict(wrap_kwargs))

    def from_nth(self, n: int, wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.SeriesFrame:
        """Select signals that satisfy the condition `pos_rank >= n`."""
        pos_rank = self.pos_rank(**kwargs).values
        return self.wrapper.wrap(pos_rank >= n, group_by=False, **resolve_dict(wrap_kwargs))

    def pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of signal position ranks.

        See `SignalsAccessor.pos_rank`."""
        return self.pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    def partition_pos_rank_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of partition position ranks.

        See `SignalsAccessor.partition_pos_rank`."""
        return self.partition_pos_rank(as_mapped=True, group_by=group_by, **kwargs)

    # ############# Index ############# #

    def nth_index(self,
                  n: int,
                  group_by: tp.GroupByLike = None,
                  jitted: tp.JittedOption = None,
                  chunked: tp.ChunkedOption = None,
                  wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """See `vectorbt.signals.nb.nth_index_nb`.

        ## Example

        ```python-repl
        >>> mask.vbt.signals.nth_index(0)
        a   2020-01-01
        b   2020-01-01
        c   2020-01-01
        Name: nth_index, dtype: datetime64[ns]

        >>> mask.vbt.signals.nth_index(2)
        a          NaT
        b   2020-01-05
        c   2020-01-03
        Name: nth_index, dtype: datetime64[ns]

        >>> mask.vbt.signals.nth_index(-1)
        a   2020-01-01
        b   2020-01-05
        c   2020-01-03
        Name: nth_index, dtype: datetime64[ns]

        >>> mask.vbt.signals.nth_index(-1, group_by=True)
        Timestamp('2020-01-05 00:00:00')
        ```"""
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            squeezed = self.squeeze_grouped(
                jit_registry.resolve_option(generic_nb.any_reduce_nb, jitted),
                group_by=group_by,
                jitted=jitted,
                chunked=chunked
            )
            arr = reshaping.to_2d_array(squeezed)
        else:
            arr = self.to_2d_array()
        func = jit_registry.resolve_option(nb.nth_index_nb, jitted)
        func = ch_registry.resolve_option(func, chunked)
        nth_index = func(arr, n)
        wrap_kwargs = merge_dicts(dict(name_or_index='nth_index', to_index=True), wrap_kwargs)
        return self.wrapper.wrap_reduced(nth_index, group_by=group_by, **wrap_kwargs)

    def norm_avg_index(self,
                       group_by: tp.GroupByLike = None,
                       jitted: tp.JittedOption = None,
                       chunked: tp.ChunkedOption = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """See `vectorbt.signals.nb.norm_avg_index_nb`.

        Normalized average index measures the average signal location relative to the middle of the column.
        This way, we can quickly see where the majority of signals are located.

        Common values are:

        * -1.0: only the first signal is set
        * 1.0: only the last signal is set
        * 0.0: symmetric distribution around the middle
        * [-1.0, 0.0): average signal is on the left
        * (0.0, 1.0]: average signal is on the right

        ## Example

        ```python-repl
        >>> pd.Series([True, False, False, False]).vbt.signals.norm_avg_index()
        -1.0

        >>> pd.Series([False, False, False, True]).vbt.signals.norm_avg_index()
        1.0

        >>> pd.Series([True, False, False, True]).vbt.signals.norm_avg_index()
        0.0
        ```"""
        if self.is_frame() and self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            func = jit_registry.resolve_option(nb.norm_avg_index_grouped_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array(), group_lens)
        else:
            func = jit_registry.resolve_option(nb.norm_avg_index_nb, jitted)
            func = ch_registry.resolve_option(func, chunked)
            norm_avg_index = func(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index='norm_avg_index'), wrap_kwargs)
        return self.wrapper.wrap_reduced(norm_avg_index, group_by=group_by, **wrap_kwargs)

    def index_mapped(self, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Get a mapped array of indices.

        See `vectorbt.generic.accessors.GenericAccessor.to_mapped`.

        Only True values will be considered."""
        indices = np.arange(len(self.wrapper.index), dtype=np.float_)[:, None]
        indices = np.tile(indices, (1, len(self.wrapper.columns)))
        indices = reshaping.soft_to_ndim(indices, self.wrapper.ndim)
        indices[~self.obj.values] = np.nan
        return self.wrapper.wrap(indices).vbt.to_mapped(
            dropna=True,
            dtype=np.int_,
            group_by=group_by,
            **kwargs
        )

    def total(self, wrap_kwargs: tp.KwargsLike = None,
              group_by: tp.GroupByLike = None) -> tp.MaybeSeries:
        """Total number of True values in each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='total'), wrap_kwargs)
        return self.sum(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def rate(self, wrap_kwargs: tp.KwargsLike = None,
             group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """`SignalsAccessor.total` divided by the total index length in each column/group."""
        total = reshaping.to_1d_array(self.total(group_by=group_by, **kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index='rate'), wrap_kwargs)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total / total_steps, group_by=group_by, **wrap_kwargs)

    def total_partitions(self, wrap_kwargs: tp.KwargsLike = None,
                         group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """Total number of partitions of True values in each column/group."""
        wrap_kwargs = merge_dicts(dict(name_or_index='total_partitions'), wrap_kwargs)
        return self.partition_ranges(**kwargs).count(group_by=group_by, wrap_kwargs=wrap_kwargs)

    def partition_rate(self, wrap_kwargs: tp.KwargsLike = None,
                       group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """`SignalsAccessor.total_partitions` divided by `SignalsAccessor.total` in each column/group."""
        total_partitions = reshaping.to_1d_array(self.total_partitions(group_by=group_by, *kwargs))
        total = reshaping.to_1d_array(self.total(group_by=group_by, *kwargs))
        wrap_kwargs = merge_dicts(dict(name_or_index='partition_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_partitions / total, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `SignalsAccessor.stats`.

        Merges `vectorbt.generic.accessors.GenericAccessor.stats_defaults` and
        `signals.stats` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        signals_stats_cfg = settings['signals']['stats']

        return merge_dicts(
            GenericAccessor.stats_defaults.__get__(self),
            signals_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags='wrapper'
            ),
            total=dict(
                title='Total',
                calc_func='total',
                tags='signals'
            ),
            rate=dict(
                title='Rate [%]',
                calc_func='rate',
                post_calc_func=lambda self, out, settings: out * 100,
                tags='signals'
            ),
            total_overlapping=dict(
                title='Total Overlapping',
                calc_func=lambda self, other, group_by:
                (self & other).vbt.signals.total(group_by=group_by),
                check_silent_has_other=True,
                tags=['signals', 'other']
            ),
            overlapping_rate=dict(
                title='Overlapping Rate [%]',
                calc_func=lambda self, other, group_by:
                (self & other).vbt.signals.total(group_by=group_by) /
                (self | other).vbt.signals.total(group_by=group_by),
                post_calc_func=lambda self, out, settings: out * 100,
                check_silent_has_other=True,
                tags=['signals', 'other']
            ),
            first_index=dict(
                title='First Index',
                calc_func='nth_index',
                n=0,
                wrap_kwargs=dict(to_index=True),
                tags=['signals', 'index']
            ),
            last_index=dict(
                title='Last Index',
                calc_func='nth_index',
                n=-1,
                wrap_kwargs=dict(to_index=True),
                tags=['signals', 'index']
            ),
            norm_avg_index=dict(
                title='Norm Avg Index [-1, 1]',
                calc_func='norm_avg_index',
                tags=['signals', 'index']
            ),
            distance=dict(
                title=RepEval("f'Distance {\"<-\" if from_other else \"->\"} {other_name}' "
                              "if other is not None else 'Distance'"),
                calc_func='between_ranges.duration',
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),
                    'Max': out.max(),
                    'Mean': out.mean(),
                    'Std': out.std(ddof=settings.get('ddof', 1))
                },
                apply_to_timedelta=True,
                tags=RepEval("['signals', 'distance', 'other'] if other is not None else ['signals', 'distance']")
            ),
            total_partitions=dict(
                title='Total Partitions',
                calc_func='total_partitions',
                tags=['signals', 'partitions']
            ),
            partition_rate=dict(
                title='Partition Rate [%]',
                calc_func='partition_rate',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=['signals', 'partitions']
            ),
            partition_len=dict(
                title='Partition Length',
                calc_func='partition_ranges.duration',
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),
                    'Max': out.max(),
                    'Mean': out.mean(),
                    'Std': out.std(ddof=settings.get('ddof', 1))
                },
                apply_to_timedelta=True,
                tags=['signals', 'partitions', 'distance']
            ),
            partition_distance=dict(
                title='Partition Distance',
                calc_func='between_partition_ranges.duration',
                post_calc_func=lambda self, out, settings: {
                    'Min': out.min(),
                    'Max': out.max(),
                    'Mean': out.mean(),
                    'Std': out.std(ddof=settings.get('ddof', 1))
                },
                apply_to_timedelta=True,
                tags=['signals', 'partitions', 'distance']
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(self, yref: str = 'y', **kwargs) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot signals.

        Args:
            yref (str): Y coordinate axis.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.lineplot`.

        ## Example

        ```python-repl
        >>> mask[['a', 'c']].vbt.signals.plot()
        ```

        ![](/docs/img/signals_df_plot.svg)
        """
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['false', 'true']
        )
        return self.obj.vbt.lineplot(**merge_dicts(default_layout, kwargs))

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `SignalsAccessor.plots`.

        Merges `vectorbt.generic.accessors.GenericAccessor.plots_defaults` and
        `signals.plots` from `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        signals_plots_cfg = settings['signals']['plots']

        return merge_dicts(
            GenericAccessor.plots_defaults.__get__(self),
            signals_plots_cfg
        )

    @property
    def subplots(self) -> Config:
        return self._subplots


SignalsAccessor.override_metrics_doc(__pdoc__)
SignalsAccessor.override_subplots_doc(__pdoc__)


@register_sr_vbt_accessor('signals')
class SignalsSRAccessor(SignalsAccessor, GenericSRAccessor):
    """Accessor on top of signal series. For Series only.

    Accessible via `pd.Series.vbt.signals`."""

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        GenericSRAccessor.__init__(self, obj, **kwargs)
        SignalsAccessor.__init__(self, obj, **kwargs)

    def plot_as_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                        **kwargs) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot Series as markers.

        Args:
            y (array_like): Y-axis values to plot markers on.
            **kwargs: Keyword arguments passed to `vectorbt.generic.accessors.GenericAccessor.scatterplot`.

        ## Example

        ```python-repl
        >>> ts = pd.Series([1, 2, 3, 2, 1], index=mask.index)
        >>> fig = ts.vbt.lineplot()
        >>> mask['b'].vbt.signals.plot_as_entry_markers(y=ts, fig=fig)
        >>> (~mask['b']).vbt.signals.plot_as_exit_markers(y=ts, fig=fig)
        ```

        ![](/docs/img/signals_plot_as_markers.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if y is None:
            y = pd.Series.vbt.empty_like(self.obj, 1)
        else:
            y = reshaping.to_pd_array(y)

        return y[self.obj].vbt.scatterplot(**merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='circle',
                    color=plotting_cfg['contrast_color_schema']['blue'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])
                    )
                )
            )
        ), kwargs))

    def plot_as_entry_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                              **kwargs) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot signals as entry markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-up',
                    color=plotting_cfg['contrast_color_schema']['green'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
                    )
                ),
                name='Entry'
            )
        ), kwargs))

    def plot_as_exit_markers(self, y: tp.Optional[tp.ArrayLike] = None,
                             **kwargs) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot signals as exit markers.

        See `SignalsSRAccessor.plot_as_markers`."""
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        return self.plot_as_markers(y=y, **merge_dicts(dict(
            trace_kwargs=dict(
                marker=dict(
                    symbol='triangle-down',
                    color=plotting_cfg['contrast_color_schema']['red'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])
                    )
                ),
                name='Exit'
            )
        ), kwargs))


@register_df_vbt_accessor('signals')
class SignalsDFAccessor(SignalsAccessor, GenericDFAccessor):
    """Accessor on top of signal series. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.signals`."""

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        GenericDFAccessor.__init__(self, obj, **kwargs)
        SignalsAccessor.__init__(self, obj, **kwargs)
