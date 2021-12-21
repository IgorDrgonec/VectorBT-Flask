# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with data sources.

Class `Data` allows storing, downloading, updating, and managing data. It stores data
as a dictionary of Series/DataFrames keyed by symbol, and makes sure that
all pandas objects have the same index and columns by aligning them.

## Downloading

Data can be fetched by overriding the `Data.fetch_symbol` class method. What `Data` does
under the hood is iterating over all symbols and calling this method.

Let's create a simple data class `RandomSNData` that generates price based on
standard-normally distributed returns:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbtpro as vbt

>>> class RandomSNData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, start_value=100, start_dt='2021-01-01', end_dt='2021-01-10'):
...         index = pd.date_range(start_dt, end_dt)
...         rand_returns = np.random.standard_normal(size=len(index))
...         rand_price = start_value + np.cumprod(rand_returns + 1)
...         return pd.Series(rand_price, index=index)

>>> rand_data = RandomSNData.fetch(['RANDNX1', 'RANDNX2'])
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  101.042956  100.920462
2021-01-02  100.987327  100.956455
2021-01-03  101.022333  100.955128
2021-01-04  101.084243  100.791793
2021-01-05  101.158619  100.781000
2021-01-06  101.172688  100.786198
2021-01-07  101.311609  100.848192
2021-01-08  101.331841  100.861500
2021-01-09  101.440530  100.944935
2021-01-10  101.585689  100.993223
```

To provide different keyword arguments for different symbols, we can use `symbol_dict`:

```pycon
>>> start_value = vbt.symbol_dict({'RANDNX2': 200})
>>> rand_data = RandomSNData.fetch(['RANDNX1', 'RANDNX2'], start_value=start_value)
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  101.083324  200.886078
2021-01-02  101.113405  200.791934
2021-01-03  101.169194  200.852877
2021-01-04  101.164033  200.820111
2021-01-05  101.326248  201.060448
2021-01-06  101.394482  200.876984
2021-01-07  101.494227  200.845519
2021-01-08  101.422012  200.963474
2021-01-09  101.493162  200.790369
2021-01-10  101.606052  200.752296
```

In case two symbols have different index or columns, they are automatically aligned based on
`missing_index` and `missing_columns` respectively (see `vectorbtpro._settings.data`):

```pycon
>>> start_dt = vbt.symbol_dict({'RANDNX2': '2021-01-03'})
>>> end_dt = vbt.symbol_dict({'RANDNX2': '2021-01-07'})
>>> rand_data = RandomSNData.fetch(
...     ['RANDNX1', 'RANDNX2'], start_value=start_value,
...     start_dt=start_dt, end_dt=end_dt)
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data.get()
UserWarning: Symbols have mismatching index. Setting missing data points to NaN.

symbol         RANDNX1     RANDNX2
2021-01-01  101.028054         NaN
2021-01-02  101.032090         NaN
2021-01-03  101.038531  200.936283
2021-01-04  101.068265  200.926764
2021-01-05  100.878492  200.898898
2021-01-06  100.857444  200.922368
2021-01-07  100.933123  200.987094
2021-01-08  100.938034         NaN
2021-01-09  101.044736         NaN
2021-01-10  101.098133         NaN
```

## Updating

Updating can be used to either update existing data points or to add new ones. It requires a
starting timestamp or row, and assumes that all data points before this timestamp or row remain unchanged.

Updating can be implemented by overriding the `Data.update_symbol` instance method, which mostly
just prepares the arguments and calls `Data.fetch_symbol`. In contrast to the fetch method, the update
method is an instance method and can access the data fetched earlier. It can also access the
keyword arguments initially passed to the fetch method, accessible under `Data.fetch_kwargs`.
Those arguments can be used as default arguments and overriden by arguments passed directly
to the update method, using `vectorbtpro.utils.config.merge_dicts`. Any instance of `Data` also
has the property `Data.last_index`, which contains the last fetched index for each symbol.
We can use this index to as the starting point for the next update.

Let's define an update method that updates the latest data point and adds a couple more.

!!! note
    Updating data always returns a new `Data` instance.

```pycon
>>> from datetime import timedelta
>>> from vectorbtpro.utils.config import merge_dicts

>>> class RandomSNData(vbt.Data):
...     @classmethod
...     def fetch_symbol(cls, symbol, start_value=100, start_dt='2021-01-01', end_dt='2021-01-10'):
...         index = pd.date_range(start_dt, end_dt)
...         rand_returns = np.random.standard_normal(size=len(index))
...         rand_price = start_value + np.cumprod(rand_returns + 1)
...         return pd.Series(rand_price, index=index)
...
...     def update_symbol(self, symbol, days_more=2, **kwargs):
...         fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
...         fetch_kwargs['start_dt'] = self.last_index[symbol]
...         fetch_kwargs['end_dt'] = fetch_kwargs['start_dt'] + timedelta(days=days_more)
...         kwargs = merge_dicts(fetch_kwargs, kwargs)
...         return self.fetch_symbol(symbol, **kwargs)

>>> rand_data = RandomSNData.fetch(['RANDNX1', 'RANDNX2'], end_dt='2021-01-05')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  100.956601  100.970865
2021-01-02  100.919011  100.987026
2021-01-03  101.062733  100.835376
2021-01-04  100.960535  100.820817
2021-01-05  100.834387  100.866549

>>> rand_data = rand_data.update()
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  100.956601  100.970865
2021-01-02  100.919011  100.987026
2021-01-03  101.062733  100.835376
2021-01-04  100.960535  100.820817
2021-01-05  101.011255  100.887049  << updated last
2021-01-06  101.004149  100.808410  << added new
2021-01-07  101.023673  100.714583  << added new

>>> rand_data = rand_data.update()
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  100.956601  100.970865
2021-01-02  100.919011  100.987026
2021-01-03  101.062733  100.835376
2021-01-04  100.960535  100.820817
2021-01-05  101.011255  100.887049
2021-01-06  101.004149  100.808410
2021-01-07  100.883400  100.874922  << updated last
2021-01-08  101.011738  100.780188  << added new
2021-01-09  100.912639  100.934014  << added new
```

## Handling exceptions

`Data.fetch` won't catch exceptions coming from `Data.fetch_symbol` - it's the task
of `Data.fetch_symbol` to handle them. The best approach is to show a user warning
whenever an exception has been thrown and return the data fetched up to this point in time
(`vectorbtpro.data.custom.BinanceData` and `vectorbtpro.data.custom.CCXTData` do this).
In such case, vectorbt will replace all missing data with NaN and keep track of the last valid index.
You can then wait until your connection is stable and re-fetch the missing data using `Data.update`.

## Merging

You can merge symbols from different `Data` instances either by subclassing `Data` and
defining custom fetch and update methods, or by manually merging their data dicts
into one data dict and passing it to the `Data.from_data` class method.

```pycon
>>> rand_data1 = RandomSNData.fetch('RANDNX1')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data2 = RandomSNData.fetch('RANDNX2', start_value=200, start_dt='2021-01-05')
```

[=100% "100%"]{: .candystripe}

```pycon
>>> merged_data = vbt.Data.from_data(vbt.merge_dicts(rand_data1.data, rand_data2.data))
>>> merged_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  101.160718         NaN
2021-01-02  101.421020         NaN
2021-01-03  101.959176         NaN
2021-01-04  102.076670         NaN
2021-01-05  102.447234  200.916198
2021-01-06  103.195187  201.033907
2021-01-07  103.595915  200.908229
2021-01-08  104.332550  201.000497
2021-01-09  105.159708  201.019157
2021-01-10  106.729495  200.910210
```

## Indexing

Like any other class subclassing `vectorbtpro.base.wrapping.Wrapping`, we can do pandas indexing
on a `Data` instance, which forwards indexing operation to each Series/DataFrame:

```pycon
>>> rand_data.loc['2021-01-07':'2021-01-09']
<__main__.RandomSNData at 0x7fdba4e36198>

>>> rand_data.loc['2021-01-07':'2021-01-09'].get()
symbol         RANDNX1     RANDNX2
2021-01-07  100.883400  100.874922
2021-01-08  101.011738  100.780188
2021-01-09  100.912639  100.934014
```

## Saving and loading

Like any other class subclassing `vectorbtpro.utils.pickling.Pickleable`, we can save a `Data`
instance to the disk with `Data.save` and load it with `Data.load`:

```pycon
>>> rand_data.save('rand_data')
>>> rand_data = RandomSNData.load('rand_data')
>>> rand_data.get()
symbol         RANDNX1     RANDNX2
2021-01-01  100.956601  100.970865
2021-01-02  100.919011  100.987026
2021-01-03  101.062733  100.835376
2021-01-04  100.960535  100.820817
2021-01-05  101.011255  100.887049
2021-01-06  101.004149  100.808410
2021-01-07  100.883400  100.874922
2021-01-08  101.011738  100.780188
2021-01-09  100.912639  100.934014
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `Data.metrics`.

```pycon
>>> rand_data = RandomSNData.fetch(['RANDNX1', 'RANDNX2'])
```

[=100% "100%"]{: .candystripe}

```pycon
>>> rand_data.stats(column='a')
Start                   2021-01-01 00:00:00+00:00
End                     2021-01-10 00:00:00+00:00
Period                           10 days 00:00:00
Total Symbols                                   2
Null Counts: RANDNX1                            0
Null Counts: RANDNX2                            0
dtype: object
```

`Data.stats` also supports (re-)grouping:

```pycon
>>> rand_data.stats(group_by=True)
Start                   2021-01-01 00:00:00+00:00
End                     2021-01-10 00:00:00+00:00
Period                           10 days 00:00:00
Total Symbols                                   2
Null Counts: RANDNX1                            0
Null Counts: RANDNX2                            0
Name: group, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `Data.subplots`.

`Data` class has a single subplot based on `Data.plot`:

```pycon
>>> rand_data.plots(settings=dict(base=100)).show()
```

![](/assets/images/data_plots.svg)
"""

import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_pd_array
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic.plots_builder import PlotsBuilderMixin
from vectorbtpro.generic.stats_builder import StatsBuilderMixin
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.datetime_ import is_tz_aware, to_timezone
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.pbar import get_pbar

__pdoc__ = {}


class symbol_dict(dict):
    """Dict that contains symbols as keys."""
    pass


class MetaData(type(StatsBuilderMixin), type(PlotsBuilderMixin)):
    pass


DataT = tp.TypeVar("DataT", bound="Data")


class Data(Wrapping, StatsBuilderMixin, PlotsBuilderMixin, metaclass=MetaData):
    """Class that downloads, updates, and manages data coming from a data source."""

    def __init__(self,
                 wrapper: ArrayWrapper,
                 data: tp.DataDict,
                 single_symbol: bool,
                 tz_localize: tp.Optional[tp.TimezoneLike],
                 tz_convert: tp.Optional[tp.TimezoneLike],
                 missing_index: str,
                 missing_columns: str,
                 fetch_kwargs: tp.Kwargs,
                 last_index: tp.Dict[tp.Symbol, int],
                 returned_kwargs: tp.Dict[tp.Symbol, tp.Any],
                 **kwargs) -> None:

        checks.assert_instance_of(data, dict)
        for symbol, obj in data.items():
            checks.assert_meta_equal(obj, data[list(data.keys())[0]])

        Wrapping.__init__(
            self,
            wrapper,
            data=data,
            single_symbol=single_symbol,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            fetch_kwargs=fetch_kwargs,
            last_index=last_index,
            returned_kwargs=returned_kwargs,
            **kwargs
        )
        StatsBuilderMixin.__init__(self)
        PlotsBuilderMixin.__init__(self)

        self._data = data
        self._single_symbol = single_symbol
        self._tz_localize = tz_localize
        self._tz_convert = tz_convert
        self._missing_index = missing_index
        self._missing_columns = missing_columns
        self._fetch_kwargs = fetch_kwargs
        self._last_index = last_index
        self._returned_kwargs = returned_kwargs

    def indexing_func(self: DataT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> DataT:
        """Perform indexing on `Data`."""
        new_wrapper = pd_indexing_func(self.wrapper)
        new_data = {symbol: pd_indexing_func(obj) for symbol, obj in self.data.items()}
        return self.replace(
            wrapper=new_wrapper,
            data=new_data
        )

    @property
    def data(self) -> tp.DataDict:
        """Data dictionary keyed by symbol."""
        return self._data

    @property
    def single_symbol(self) -> bool:
        """Whether there is only one symbol in `Data.data`."""
        return self._single_symbol

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        """List of symbols."""
        return list(self.data.keys())

    @property
    def tz_localize(self) -> tp.Optional[tp.TimezoneLike]:
        """`tz_localize` initially passed to `Data.fetch_symbol`."""
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Optional[tp.TimezoneLike]:
        """`tz_convert` initially passed to `Data.fetch_symbol`."""
        return self._tz_convert

    @property
    def missing_index(self) -> str:
        """`missing_index` initially passed to `Data.fetch_symbol`."""
        return self._missing_index

    @property
    def missing_columns(self) -> str:
        """`missing_columns` initially passed to `Data.fetch_symbol`."""
        return self._missing_columns

    @property
    def fetch_kwargs(self) -> tp.Kwargs:
        """Keyword arguments initially passed to `Data.fetch_symbol`."""
        return self._fetch_kwargs

    @property
    def last_index(self) -> tp.Dict[tp.Symbol, int]:
        """Last fetched index per symbol."""
        return self._last_index

    @property
    def returned_kwargs(self) -> tp.Dict[tp.Symbol, tp.Any]:
        """Keyword arguments returned by `Data.fetch_symbol` along with the data."""
        return self._returned_kwargs

    @classmethod
    def prepare_tzaware_index(cls,
                              obj: tp.SeriesFrame,
                              tz_localize: tp.Optional[tp.TimezoneLike] = None,
                              tz_convert: tp.Optional[tp.TimezoneLike] = None) -> tp.SeriesFrame:
        """Prepare a timezone-aware index of a pandas object.

        If the index is tz-naive, convert to a timezone using `tz_localize`.
        Convert the index from one timezone to another using `tz_convert`.
        See `vectorbtpro.utils.datetime_.to_timezone`.

        For defaults, see `vectorbtpro._settings.data`."""
        from vectorbtpro._settings import settings
        data_cfg = settings['data']

        if tz_localize is None:
            tz_localize = data_cfg['tz_localize']
        if tz_convert is None:
            tz_convert = data_cfg['tz_convert']

        if isinstance(obj.index, pd.DatetimeIndex):
            if tz_localize is not None:
                if not is_tz_aware(obj.index):
                    obj = obj.tz_localize(to_timezone(tz_localize))
            if tz_convert is not None:
                obj = obj.tz_convert(to_timezone(tz_convert))
        return obj

    @classmethod
    def align_index(cls, data: tp.DataDict, missing: tp.Optional[str] = None) -> tp.DataDict:
        """Align data to have the same index.

        The argument `missing` accepts the following values:

        * 'nan': set missing data points to NaN
        * 'drop': remove missing data points
        * 'raise': raise an error

        For defaults, see `vectorbtpro._settings.data`."""
        if len(data) == 1:
            return data

        from vectorbtpro._settings import settings
        data_cfg = settings['data']

        if missing is None:
            missing = data_cfg['missing_index']

        index = None
        for symbol, obj in data.items():
            if index is None:
                index = obj.index
            else:
                if len(index.intersection(obj.index)) != len(index.union(obj.index)):
                    if missing == 'nan':
                        warnings.warn("Symbols have mismatching index. "
                                      "Setting missing data points to NaN.", stacklevel=2)
                        index = index.union(obj.index)
                    elif missing == 'drop':
                        warnings.warn("Symbols have mismatching index. "
                                      "Dropping missing data points.", stacklevel=2)
                        index = index.intersection(obj.index)
                    elif missing == 'raise':
                        raise ValueError("Symbols have mismatching index")
                    else:
                        raise ValueError(f"missing='{missing}' is not recognized")

        # reindex
        new_data = {symbol: obj.reindex(index=index) for symbol, obj in data.items()}
        return new_data

    @classmethod
    def align_columns(cls, data: tp.DataDict, missing: tp.Optional[str] = None) -> tp.DataDict:
        """Align data to have the same columns.

        See `Data.align_index` for `missing`."""
        if len(data) == 1:
            return data

        from vectorbtpro._settings import settings
        data_cfg = settings['data']

        if missing is None:
            missing = data_cfg['missing_columns']

        columns = None
        multiple_columns = False
        name_is_none = False
        for symbol, obj in data.items():
            if isinstance(obj, pd.Series):
                if obj.name is None:
                    name_is_none = True
                obj = obj.to_frame()
            else:
                multiple_columns = True
            if columns is None:
                columns = obj.columns
            else:
                if len(columns.intersection(obj.columns)) != len(columns.union(obj.columns)):
                    if missing == 'nan':
                        warnings.warn("Symbols have mismatching columns. "
                                      "Setting missing data points to NaN.", stacklevel=2)
                        columns = columns.union(obj.columns)
                    elif missing == 'drop':
                        warnings.warn("Symbols have mismatching columns. "
                                      "Dropping missing data points.", stacklevel=2)
                        columns = columns.intersection(obj.columns)
                    elif missing == 'raise':
                        raise ValueError("Symbols have mismatching columns")
                    else:
                        raise ValueError(f"missing='{missing}' is not recognized")

        # reindex
        new_data = {}
        for symbol, obj in data.items():
            if isinstance(obj, pd.Series):
                obj = obj.to_frame(name=obj.name)
            obj = obj.reindex(columns=columns)
            if not multiple_columns:
                obj = obj[columns[0]]
                if name_is_none:
                    obj = obj.rename(None)
            new_data[symbol] = obj
        return new_data

    @classmethod
    def select_symbol_kwargs(cls, symbol: tp.Symbol, kwargs: dict) -> dict:
        """Select keyword arguments belonging to `symbol`."""
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, symbol_dict):
                if symbol in v:
                    _kwargs[k] = v[symbol]
            else:
                _kwargs[k] = v
        return _kwargs

    @classmethod
    def from_data(cls: tp.Type[DataT],
                  data: tp.DataDict,
                  single_symbol: bool = False,
                  tz_localize: tp.Optional[tp.TimezoneLike] = None,
                  tz_convert: tp.Optional[tp.TimezoneLike] = None,
                  missing_index: tp.Optional[str] = None,
                  missing_columns: tp.Optional[str] = None,
                  wrapper_kwargs: tp.KwargsLike = None,
                  fetch_kwargs: tp.KwargsLike = None,
                  last_index: tp.Optional[tp.Dict[tp.Symbol, int]] = None,
                  returned_kwargs: tp.Optional[tp.Dict[tp.Symbol, tp.Any]] = None,
                  **kwargs) -> DataT:
        """Create a new `Data` instance from data.

        Args:
            data (dict): Dictionary of array-like objects keyed by symbol.
            single_symbol (bool): Whether there is only one symbol in `data`.
            tz_localize (timezone_like): See `Data.prepare_tzaware_index`.
            tz_convert (timezone_like): See `Data.prepare_tzaware_index`.
            missing_index (str): See `Data.align_index`.
            missing_columns (str): See `Data.align_columns`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            fetch_kwargs (dict): Keyword arguments initially passed to `Data.fetch_symbol`.
            last_index (dict): Last fetched index per symbol.
            returned_kwargs (dict): Keyword arguments returned by `Data.fetch_symbol` along with the data.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbtpro._settings.data`."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if fetch_kwargs is None:
            fetch_kwargs = {}
        if last_index is None:
            last_index = {}
        if returned_kwargs is None:
            returned_kwargs = {}

        data = data.copy()
        for symbol, obj in data.items():
            obj = to_pd_array(obj)
            obj = cls.prepare_tzaware_index(obj, tz_localize=tz_localize, tz_convert=tz_convert)
            data[symbol] = obj
            if symbol not in last_index:
                last_index[symbol] = obj.index[-1]

        data = cls.align_index(data, missing=missing_index)
        data = cls.align_columns(data, missing=missing_columns)

        for symbol, obj in data.items():
            if isinstance(obj.index, pd.DatetimeIndex):
                obj.index.freq = obj.index.inferred_freq

        symbols = list(data.keys())
        wrapper = ArrayWrapper.from_obj(data[symbols[0]], **wrapper_kwargs)
        return cls(
            wrapper,
            data,
            single_symbol,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            fetch_kwargs=fetch_kwargs,
            last_index=last_index,
            returned_kwargs=returned_kwargs,
            **kwargs
        )

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> \
            tp.Union[tp.SeriesFrame, tp.Tuple[tp.SeriesFrame, tp.Kwargs]]:
        """Fetch a symbol.

        May also return a dictionary that will be accessible as `Data.returned_kwargs`."""
        raise NotImplementedError

    @classmethod
    def fetch(cls: tp.Type[DataT],
              symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
              tz_localize: tp.Optional[tp.TimezoneLike] = None,
              tz_convert: tp.Optional[tp.TimezoneLike] = None,
              missing_index: tp.Optional[str] = None,
              missing_columns: tp.Optional[str] = None,
              wrapper_kwargs: tp.KwargsLike = None,
              show_progress: tp.Optional[bool] = None,
              pbar_kwargs: tp.KwargsLike = None,
              **kwargs) -> DataT:
        """Fetch data using `Data.fetch_symbol` and pass to `Data.from_data`.

        Args:
            symbols (hashable or sequence of hashable): One or multiple symbols.

                !!! note
                    Tuple is considered as a single symbol (tuple is a hashable).
            tz_localize (any): See `Data.from_data`.
            tz_convert (any): See `Data.from_data`.
            missing_index (str): See `Data.from_data`.
            missing_columns (str): See `Data.from_data`.
            wrapper_kwargs (dict): See `Data.from_data`.
            show_progress (bool): Whether to show the progress bar.
                Defaults to True if the global flag for data is True and there is more than one symbol.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            **kwargs: Passed to `Data.fetch_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.
        """
        from vectorbtpro._settings import settings
        data_cfg = settings['data']

        if checks.is_hashable(symbols):
            single_symbol = True
            symbols = [symbols]
        elif not checks.is_sequence(symbols):
            raise TypeError("Symbols must be either a hashable or a sequence of hashable")
        else:
            single_symbol = False
        show_symbol_progress = show_progress
        if show_symbol_progress is None:
            show_symbol_progress = data_cfg['show_progress']
        if show_progress is None:
            show_progress = data_cfg['show_progress'] and not single_symbol
        pbar_kwargs = merge_dicts(data_cfg['pbar_kwargs'], pbar_kwargs)

        data = {}
        returned_kwargs = {}
        with get_pbar(total=len(symbols), show_progress=show_progress, **pbar_kwargs) as pbar:
            for symbol in symbols:
                if symbol is not None:
                    pbar.set_description(str(symbol))

                _kwargs = cls.select_symbol_kwargs(symbol, kwargs)
                func_arg_names = get_func_arg_names(cls.fetch_symbol)
                if 'show_progress' in func_arg_names:
                    _kwargs['show_progress'] = show_symbol_progress
                if 'pbar_kwargs' in func_arg_names:
                    _kwargs['pbar_kwargs'] = pbar_kwargs
                out = cls.fetch_symbol(symbol, **_kwargs)
                if isinstance(out, tuple):
                    data[symbol] = out[0]
                    returned_kwargs[symbol] = out[1]
                else:
                    data[symbol] = out
                    returned_kwargs[symbol] = {}

                pbar.update(1)

        # Create new instance from data
        return cls.from_data(
            data,
            single_symbol=single_symbol,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            wrapper_kwargs=wrapper_kwargs,
            fetch_kwargs=kwargs,
            returned_kwargs=returned_kwargs
        )

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SeriesFrame:
        """Update a symbol."""
        raise NotImplementedError

    def update(self: DataT, show_progress: bool = False, pbar_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Fetch additional data using `Data.update_symbol` and append it to the existing data.

        Args:
            show_progress (bool): Whether to show the progress bar.
                Defaults to False.

                Will also forward this argument to `Data.update_symbol` if in `Data.fetch_kwargs`.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

                Will also forward this argument to `Data.update_symbol` if in `Data.fetch_kwargs`.
            **kwargs: Passed to `Data.update_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.

        !!! note
            Returns a new `Data` instance instead of changing the data in place."""
        from vectorbtpro._settings import settings
        data_cfg = settings['data']

        pbar_kwargs = merge_dicts(data_cfg['pbar_kwargs'], pbar_kwargs)

        new_data = {}
        new_last_index = {}
        new_returned_kwargs = {}
        with get_pbar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for symbol, obj in self.data.items():
                if symbol is not None:
                    pbar.set_description(str(symbol))

                _kwargs = self.select_symbol_kwargs(symbol, kwargs)
                func_arg_names = get_func_arg_names(self.fetch_symbol)
                if 'show_progress' in func_arg_names:
                    _kwargs['show_progress'] = show_progress
                if 'pbar_kwargs' in func_arg_names:
                    _kwargs['pbar_kwargs'] = pbar_kwargs
                out = self.update_symbol(symbol, **_kwargs)
                if isinstance(out, tuple):
                    new_obj = out[0]
                    new_returned_kwargs[symbol] = out[1]
                else:
                    new_obj = out
                    new_returned_kwargs[symbol] = {}

                if not isinstance(new_obj, (pd.Series, pd.DataFrame)):
                    new_obj = to_pd_array(new_obj)
                    new_obj.index = pd.RangeIndex(
                        start=obj.index[-1],
                        stop=obj.index[-1] + new_obj.shape[0],
                        step=1
                    )
                new_obj = self.prepare_tzaware_index(
                    new_obj,
                    tz_localize=self.tz_localize,
                    tz_convert=self.tz_convert
                )
                new_data[symbol] = new_obj
                if len(new_obj.index) > 0:
                    new_last_index[symbol] = new_obj.index[-1]
                else:
                    new_last_index[symbol] = self.last_index[symbol]

                pbar.update(1)

        # Prepend existing data starting from lowest updated index (including) to new data
        from_index = None
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                index = new_obj.index[0]
            else:
                continue
            if from_index is None or index < from_index:
                from_index = index
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                to_index = new_obj.index[0]
            else:
                to_index = None
            obj = self.data[symbol]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                shared_columns = obj.columns.intersection(new_obj.columns)
                obj = obj[shared_columns]
                new_obj = new_obj[shared_columns]
            elif isinstance(new_obj, pd.DataFrame):
                new_obj = new_obj[obj.name]
            elif isinstance(obj, pd.DataFrame):
                obj = obj[new_obj.name]
            obj = obj.loc[from_index:to_index]
            new_obj = pd.concat((obj, new_obj), axis=0)
            new_obj = new_obj[~new_obj.index.duplicated(keep='last')]
            new_data[symbol] = new_obj

        new_data = self.align_index(new_data, missing=self.missing_index)
        new_data = self.align_columns(new_data, missing=self.missing_columns)

        # Append new data to existing data ending at lowest updated index (excluding)
        for symbol, new_obj in new_data.items():
            obj = self.data[symbol]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                new_obj = new_obj[obj.columns]
            elif isinstance(new_obj, pd.DataFrame):
                new_obj = new_obj[obj.name]
            obj = obj.loc[:from_index]
            if obj.index[-1] == from_index:
                obj = obj.iloc[:-1]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if isinstance(new_obj.index, pd.DatetimeIndex):
                new_obj.index.freq = new_obj.index.inferred_freq
            new_data[symbol] = new_obj

        new_index = new_data[self.symbols[0]].index
        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),
            data=new_data,
            last_index=new_last_index,
            returned_kwargs=new_returned_kwargs
        )

    def concat(self, level_name: str = 'symbol') -> tp.DataDict:
        """Return a dict of Series/DataFrames with symbols as columns, keyed by column name."""
        first_data = self.data[self.symbols[0]]
        index = first_data.index
        if isinstance(first_data, pd.Series):
            columns = pd.Index([first_data.name])
        else:
            columns = first_data.columns
        if self.single_symbol:
            new_data = {c: pd.Series(
                index=index,
                name=self.symbols[0],
                dtype=self.data[self.symbols[0]].dtype
                if isinstance(self.data[self.symbols[0]], pd.Series)
                else self.data[self.symbols[0]][c].dtype
            ) for c in columns}
        else:
            new_data = {c: pd.DataFrame(
                index=index,
                columns=pd.Index(self.symbols, name=level_name),
                dtype=self.data[self.symbols[0]].dtype
                if isinstance(self.data[self.symbols[0]], pd.Series)
                else self.data[self.symbols[0]][c].dtype
            ) for c in columns}
        for c in columns:
            for s in self.symbols:
                if isinstance(self.data[s], pd.Series):
                    col_data = self.data[s]
                else:
                    col_data = self.data[s][c]
                if self.single_symbol:
                    new_data[c].loc[:] = col_data
                else:
                    new_data[c].loc[:, s] = col_data

        return new_data

    def get(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Get column data.

        If one symbol, returns data for that symbol. If multiple symbols, performs concatenation
        first and returns a DataFrame if one column and a tuple of DataFrames if a list of columns passed."""
        if self.single_symbol:
            if column is None:
                return self.data[self.symbols[0]]
            return self.data[self.symbols[0]][column]

        concat_data = self.concat(**kwargs)
        if len(concat_data) == 1:
            return tuple(concat_data.values())[0]
        if column is not None:
            if isinstance(column, list):
                return tuple([concat_data[c] for c in column])
            return concat_data[column]
        return tuple(concat_data.values())

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.data`."""
        from vectorbtpro._settings import settings
        data_stats_cfg = settings['data']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            data_stats_cfg
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
            total_symbols=dict(
                title='Total Symbols',
                calc_func=lambda self: len(self.symbols),
                agg_func=None,
                tags='data'
            ),
            null_counts=dict(
                title='Null Counts',
                calc_func=lambda self, group_by:
                {
                    symbol: obj.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by)
                    for symbol, obj in self.data.items()
                },
                tags='data'
            )
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             base: tp.Optional[float] = None,
             **kwargs) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            base (float): Rebase all series of a column to a given intial base.

                !!! note
                    The column must contain prices.
            kwargs (dict): Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`.

        Usage:
            ```pycon
            >>> import vectorbtpro as vbt

            >>> start = '2021-01-01 UTC'  # crypto is in UTC
            >>> end = '2021-06-01 UTC'
            >>> data = vbt.YFData.fetch(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
            ```

            [=100% "100%"]{: .candystripe}

            ```pycon
            >>> data.plot(column='Close', base=1)
            ```

            ![](/assets/images/data_plot.svg)
        """
        self_col = self.select_one(column=column, group_by=False)
        data = self_col.get()
        if base is not None:
            data = data.vbt.rebase(base)
        return data.vbt.plot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.data`."""
        from vectorbtpro._settings import settings
        data_plots_cfg = settings['data']['plots']

        return merge_dicts(
            PlotsBuilderMixin.plots_defaults.__get__(self),
            data_plots_cfg
        )

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=dict(
                check_is_not_grouped=True,
                plot_func='plot',
                pass_add_trace_kwargs=True,
                tags='data'
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Data.override_metrics_doc(__pdoc__)
Data.override_subplots_doc(__pdoc__)
