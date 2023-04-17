# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Classes and functions for indexing."""

import attr
from datetime import time

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.template import CustomTemplate
from vectorbtpro.utils.datetime_ import (
    try_to_datetime_index,
    try_align_to_dt_index,
    try_align_dt_to_index,
    time_to_timedelta,
    infer_index_freq,
    prepare_freq,
)
from vectorbtpro.utils.config import hdict, merge_dicts
from vectorbtpro.utils.pickling import pdict

__all__ = [
    "PandasIndexer",
    "hslice",
    "get_index_points",
    "get_index_ranges",
    "index_dict",
    "pointidx",
    "rangeidx",
    "rowidx",
    "colidx",
    "idx",
]

__pdoc__ = {}


class IndexingError(Exception):
    """Exception raised when an indexing error has occurred."""


IndexingBaseT = tp.TypeVar("IndexingBaseT", bound="IndexingBase")


class IndexingBase:
    """Class that supports indexing through `IndexingBase.indexing_func`."""

    def indexing_func(self: IndexingBaseT, pd_indexing_func: tp.Callable, **kwargs) -> IndexingBaseT:
        """Apply `pd_indexing_func` on all pandas objects in question and return a new instance of the class.

        Should be overridden."""
        raise NotImplementedError


class LocBase:
    """Class that implements location-based indexing."""

    def __init__(self, indexing_func: tp.Callable, **kwargs) -> None:
        self._indexing_func = indexing_func
        self._indexing_kwargs = kwargs

    @property
    def indexing_func(self) -> tp.Callable:
        """Indexing function."""
        return self._indexing_func

    @property
    def indexing_kwargs(self) -> dict:
        """Keyword arguments passed to `LocBase.indexing_func`."""
        return self._indexing_kwargs

    def __getitem__(self, key: tp.Any) -> tp.Any:
        raise NotImplementedError


class iLoc(LocBase):
    """Forwards `pd.Series.iloc`/`pd.DataFrame.iloc` operation to each
    Series/DataFrame and returns a new class instance."""

    def __getitem__(self, key: tp.Any) -> tp.Any:
        return self.indexing_func(lambda x: x.iloc.__getitem__(key), **self.indexing_kwargs)


class Loc(LocBase):
    """Forwards `pd.Series.loc`/`pd.DataFrame.loc` operation to each
    Series/DataFrame and returns a new class instance."""

    def __getitem__(self, key: tp.Any) -> tp.Any:
        return self.indexing_func(lambda x: x.loc.__getitem__(key), **self.indexing_kwargs)


PandasIndexerT = tp.TypeVar("PandasIndexerT", bound="PandasIndexer")


class PandasIndexer(IndexingBase):
    """Implements indexing using `iloc`, `loc`, `xs` and `__getitem__`.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.indexing import PandasIndexer

        >>> class C(PandasIndexer):
        ...     def __init__(self, df1, df2):
        ...         self.df1 = df1
        ...         self.df2 = df2
        ...         super().__init__()
        ...
        ...     def indexing_func(self, pd_indexing_func):
        ...         return type(self)(
        ...             pd_indexing_func(self.df1),
        ...             pd_indexing_func(self.df2)
        ...         )

        >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        >>> c = C(df1, df2)

        >>> c.iloc[:, 0]
        <__main__.C object at 0x1a1cacbbe0>

        >>> c.iloc[:, 0].df1
        0    1
        1    2
        Name: a, dtype: int64

        >>> c.iloc[:, 0].df2
        0    5
        1    6
        Name: a, dtype: int64
        ```
    """

    def __init__(self, **kwargs) -> None:
        self._iloc = iLoc(self.indexing_func, **kwargs)
        self._loc = Loc(self.indexing_func, **kwargs)
        self._indexing_kwargs = kwargs

    @property
    def indexing_kwargs(self) -> dict:
        """Indexing keyword arguments."""
        return self._indexing_kwargs

    @property
    def iloc(self) -> iLoc:
        """Purely integer-location based indexing for selection by position."""
        return self._iloc

    iloc.__doc__ = iLoc.__doc__

    @property
    def loc(self) -> Loc:
        """Purely label-location based indexer for selection by label."""
        return self._loc

    loc.__doc__ = Loc.__doc__

    def xs(self: PandasIndexerT, *args, **kwargs) -> PandasIndexerT:
        """Forwards `pd.Series.xs`/`pd.DataFrame.xs`
        operation to each Series/DataFrame and returns a new class instance."""
        return self.indexing_func(lambda x: x.xs(*args, **kwargs), **self.indexing_kwargs)

    def __getitem__(self: PandasIndexerT, key: tp.Any) -> PandasIndexerT:
        return self.indexing_func(lambda x: x.__getitem__(key), **self.indexing_kwargs)


class ParamLoc(LocBase):
    """Access a group of columns by parameter using `pd.Series.loc`.

    Uses `mapper` to establish link between columns and parameter values."""

    def __init__(self, mapper: tp.Series, indexing_func: tp.Callable, level_name: tp.Level = None, **kwargs) -> None:
        checks.assert_instance_of(mapper, pd.Series)

        if mapper.dtype == "O":
            # If params are objects, we must cast them to string first
            # The original mapper isn't touched
            mapper = mapper.astype(str)
        self._mapper = mapper
        self._level_name = level_name

        LocBase.__init__(self, indexing_func, **kwargs)

    @property
    def mapper(self) -> tp.Series:
        """Mapper."""
        return self._mapper

    @property
    def level_name(self) -> tp.Level:
        """Level name."""
        return self._level_name

    def get_indices(self, key: tp.Any) -> tp.Array1d:
        """Get array of indices affected by this key."""
        if self.mapper.dtype == "O":
            # We must also cast the key to string
            if isinstance(key, (slice, hslice)):
                start = str(key.start) if key.start is not None else None
                stop = str(key.stop) if key.stop is not None else None
                key = slice(start, stop, key.step)
            elif isinstance(key, (list, np.ndarray)):
                key = list(map(str, key))
            else:
                # Tuples, objects, etc.
                key = str(key)
        # Use pandas to perform indexing
        mapper = pd.Series(np.arange(len(self.mapper.index)), index=self.mapper.values)
        indices = mapper.loc.__getitem__(key)
        if isinstance(indices, pd.Series):
            indices = indices.values
        return indices

    def __getitem__(self, key: tp.Any) -> tp.Any:
        indices = self.get_indices(key)
        is_multiple = isinstance(key, (slice, hslice, list, np.ndarray))

        def pd_indexing_func(obj: tp.SeriesFrame) -> tp.MaybeSeriesFrame:
            from vectorbtpro.base.indexes import drop_levels

            new_obj = obj.iloc[:, indices]
            if not is_multiple:
                # If we selected only one param, then remove its columns levels to keep it clean
                if self.level_name is not None:
                    if checks.is_frame(new_obj):
                        if isinstance(new_obj.columns, pd.MultiIndex):
                            new_obj.columns = drop_levels(new_obj.columns, self.level_name)
            return new_obj

        return self.indexing_func(pd_indexing_func, **self.indexing_kwargs)


def indexing_on_mapper(
    mapper: tp.Series,
    ref_obj: tp.SeriesFrame,
    pd_indexing_func: tp.Callable,
) -> tp.Optional[tp.Series]:
    """Broadcast `mapper` Series to `ref_obj` and perform pandas indexing using `pd_indexing_func`."""
    from vectorbtpro.base.reshaping import broadcast_to

    checks.assert_instance_of(mapper, pd.Series)
    checks.assert_instance_of(ref_obj, (pd.Series, pd.DataFrame))

    if isinstance(ref_obj, pd.Series):
        range_mapper = broadcast_to(0, ref_obj)
    else:
        range_mapper = broadcast_to(np.arange(len(mapper.index))[None], ref_obj)
    loced_range_mapper = pd_indexing_func(range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)
    return None


def build_param_indexer(
    param_names: tp.Sequence[str],
    class_name: str = "ParamIndexer",
    module_name: tp.Optional[str] = None,
) -> tp.Type[IndexingBase]:
    """A factory to create a class with parameter indexing.

    Parameter indexer enables accessing a group of rows and columns by a parameter array (similar to `loc`).
    This way, one can query index/columns by another Series called a parameter mapper, which is just a
    `pd.Series` that maps columns (its index) to params (its values).

    Parameter indexing is important, since querying by column/index labels alone is not always the best option.
    For example, `pandas` doesn't let you query by list at a specific index/column level.

    Args:
        param_names (list of str): Names of the parameters.
        class_name (str): Name of the generated class.
        module_name (str): Name of the module to which the class should be bound.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.indexing import build_param_indexer, indexing_on_mapper

        >>> MyParamIndexer = build_param_indexer(['my_param'])
        >>> class C(MyParamIndexer):
        ...     def __init__(self, df, param_mapper):
        ...         self.df = df
        ...         self._my_param_mapper = param_mapper
        ...         super().__init__([param_mapper])
        ...
        ...     def indexing_func(self, pd_indexing_func):
        ...         return type(self)(
        ...             pd_indexing_func(self.df),
        ...             indexing_on_mapper(self._my_param_mapper, self.df, pd_indexing_func)
        ...         )

        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> param_mapper = pd.Series(['First', 'Second'], index=['a', 'b'])
        >>> c = C(df, param_mapper)

        >>> c.my_param_loc['First'].df
        0    1
        1    2
        Name: a, dtype: int64

        >>> c.my_param_loc['Second'].df
        0    3
        1    4
        Name: b, dtype: int64

        >>> c.my_param_loc[['First', 'First', 'Second', 'Second']].df
              a     b
        0  1  1  3  3
        1  2  2  4  4
        ```
    """

    class ParamIndexer(IndexingBase):
        def __init__(
            self,
            param_mappers: tp.Sequence[tp.Series],
            level_names: tp.Optional[tp.LevelSequence] = None,
            **kwargs,
        ) -> None:
            checks.assert_len_equal(param_names, param_mappers)

            for i, param_name in enumerate(param_names):
                level_name = level_names[i] if level_names is not None else None
                _param_loc = ParamLoc(param_mappers[i], self.indexing_func, level_name=level_name, **kwargs)
                setattr(self, f"_{param_name}_loc", _param_loc)

    for i, param_name in enumerate(param_names):

        def param_loc(self, _param_name=param_name) -> ParamLoc:
            return getattr(self, f"_{_param_name}_loc")

        param_loc.__doc__ = f"""Access a group of columns by parameter `{param_name}` using `pd.Series.loc`.
        
        Forwards this operation to each Series/DataFrame and returns a new class instance.
        """

        setattr(ParamIndexer, param_name + "_loc", property(param_loc))

    ParamIndexer.__name__ = class_name
    ParamIndexer.__qualname__ = class_name
    if module_name is not None:
        ParamIndexer.__module__ = module_name

    return ParamIndexer


hsliceT = tp.TypeVar("hsliceT", bound="hslice")


_DEF = object()
"""Default value for internal purposes."""


@attr.s(frozen=True, init=False)
class hslice:
    """Hashable slice."""

    start: object = attr.ib()
    """Start."""

    stop: object = attr.ib()
    """Stop."""

    step: object = attr.ib()
    """Step."""

    def __init__(self, start: object = _DEF, stop: object = _DEF, step: object = _DEF) -> None:
        if start is not _DEF and stop is _DEF and step is _DEF:
            stop = start
            start, step = None, None
        else:
            if start is _DEF:
                start = None
            if stop is _DEF:
                stop = None
            if step is _DEF:
                step = None
        self.__attrs_init__(start=start, stop=stop, step=step)

    @classmethod
    def from_slice(cls: tp.Type[hsliceT], slice_: slice) -> hsliceT:
        """Construct from a slice."""
        return cls(slice_.start, slice_.stop, slice_.step)

    def to_slice(self) -> slice:
        """Convert to a slice."""
        return slice(self.start, self.stop, self.step)


class IdxrBase:
    """Abstract class for resolving indices."""

    def get(self, *args, **kwargs) -> tp.Any:
        """Get indices."""
        raise NotImplementedError

    def slice_indexer(self, index: tp.Index, slice_: tp.Slice, closed_end: bool = False) -> slice:
        """Compute the slice indexer for input labels and step."""
        start = slice_.start
        end = slice_.stop
        if closed_end:
            return index.slice_indexer(start, end, slice_.step)
        if start is not None:
            start = index.get_slice_bound(start, side="left")
        if end is not None:
            new_end = index.get_slice_bound(end, side="right")
            if new_end != index.get_slice_bound(end, side="left"):
                new_end = new_end - 1
            end = new_end
        return slice(start, end, slice_.step)

    def check_indices(self, indices: tp.MaybeIndexArray) -> None:
        """Check indices after resolving them."""
        if isinstance(indices, slice):
            if indices.start is not None and not checks.is_int(indices.start):
                raise TypeError("Start of a returned index slice must be an integer or None")
            if indices.stop is not None and not checks.is_int(indices.stop):
                raise TypeError("Stop of a returned index slice must be an integer or None")
            if indices.step is not None and not checks.is_int(indices.step):
                raise TypeError("Step of a returned index slice must be an integer or None")
            if indices.start == -1:
                raise ValueError("Range start index couldn't be matched")
            elif indices.stop == -1:
                raise ValueError("Range end index couldn't be matched")
        elif checks.is_int(indices):
            if indices == -1:
                raise ValueError("Index couldn't be matched")
        elif checks.is_sequence(indices) and not np.isscalar(indices):
            if not isinstance(indices, np.ndarray):
                raise ValueError(f"Indices must be a NumPy array, not {type(indices)}")
            if not np.issubdtype(indices.dtype, np.integer) or np.issubdtype(indices.dtype, np.bool_):
                raise ValueError(f"Indices must be of integer data type, not {indices.dtype}")
            if -1 in indices:
                raise ValueError("Some indices couldn't be matched")
            if indices.ndim not in (1, 2):
                raise ValueError("Indices array must have either 1 or 2 dimensions")
            if indices.ndim == 2 and indices.shape[1] != 2:
                raise ValueError("Indices array provided as ranges must have exactly two columns")
        else:
            raise TypeError(f"Indices must be an integer, a slice, a NumPy array, or a tuple "
                            f"of two NumPy arrays, not {type(indices)}")


class UniIdxr(IdxrBase):
    """Abstract class for resolving indices."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        raise NotImplementedError


@attr.s(frozen=True)
class PosIdxr(UniIdxr):
    """Class for resolving indices provided as integer positions."""

    value: tp.Union[None, tp.MaybeSequence[tp.MaybeSequence[int]], tp.Slice] = attr.ib()
    """One or more integer positions."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        indices = self.value
        if checks.is_sequence(indices) and not np.isscalar(indices):
            indices = np.asarray(indices)
        if isinstance(indices, hslice):
            indices = indices.to_slice()
        self.check_indices(indices)
        return indices


@attr.s(frozen=True)
class MaskIdxr(UniIdxr):
    """Class for resolving indices provided as a mask."""

    value: tp.Union[None, tp.Sequence[bool]] = attr.ib()
    """Mask."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        indices = np.flatnonzero(self.value)
        self.check_indices(indices)
        return indices


@attr.s(frozen=True)
class LabelIdxr(UniIdxr):
    """Class for resolving indices provided as labels."""

    value: tp.Union[None, tp.MaybeSequence[tp.Label], tp.Slice] = attr.ib()
    """One or more labels."""

    closed_end: bool = attr.ib(default=True)
    """Whether `end` should be inclusive."""

    level: tp.MaybeLevelSequence = attr.ib(default=None)
    """One or more levels."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        if self.level is not None:
            from vectorbtpro.base.indexes import select_levels

            index = select_levels(index, self.level)

        if isinstance(self.value, (slice, hslice)):
            indices = self.slice_indexer(index, self.value, closed_end=self.closed_end)
        elif (checks.is_sequence(self.value) and not np.isscalar(self.value)) and (
            not isinstance(index, pd.MultiIndex)
            or (isinstance(index, pd.MultiIndex) and isinstance(self.value[0], tuple))
        ):
            indices = index.get_indexer_for(self.value)
        else:
            indices = index.get_loc(self.value)
            if isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.bool_):
                indices = np.flatnonzero(indices)
        self.check_indices(indices)
        return indices


@attr.s(frozen=True)
class DatetimeIdxr(UniIdxr):
    """Class for resolving indices provided as datetime-like objects."""

    value: tp.Union[None, tp.MaybeSequence[tp.DatetimeLike], tp.Slice] = attr.ib()
    """One or more datetime-like objects."""

    closed_end: bool = attr.ib(default=False)
    """Whether `end` should be inclusive."""

    indexer_method: tp.Optional[str] = attr.ib(default="bfill")
    """Method for `pd.Index.get_indexer`."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        index = try_to_datetime_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if not index.is_unique:
            raise ValueError("Datetime index must be unique")
        if not index.is_monotonic_increasing:
            raise ValueError("Datetime index must be monotonically increasing")

        if isinstance(self.value, (slice, hslice)):
            start = try_align_dt_to_index(self.value.start, index)
            stop = try_align_dt_to_index(self.value.stop, index)
            new_value = slice(start, stop, self.value.step)
            indices = self.slice_indexer(index, new_value, closed_end=self.closed_end)
        elif checks.is_sequence(self.value) and not np.isscalar(self.value):
            new_value = try_align_to_dt_index(self.value, index)
            indices = index.get_indexer(new_value, method=self.indexer_method)
        else:
            new_value = try_align_dt_to_index(self.value, index)
            if self.indexer_method is None or new_value in index:
                indices = index.get_loc(new_value)
                if isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.bool_):
                    indices = np.flatnonzero(indices)
            else:
                indices = index.get_indexer([new_value], method=self.indexer_method)[0]
        self.check_indices(indices)
        return indices


@attr.s(frozen=True)
class AutoIdxr(UniIdxr):
    """Class for resolving indices, labels, or datetime-like objects for one axis."""

    value: tp.Union[
        None,
        tp.MaybeSequence[tp.MaybeSequence[int]],
        tp.MaybeSequence[tp.Label],
        tp.MaybeSequence[tp.DatetimeLike],
        tp.Slice,
    ] = attr.ib()
    """One or more integer indices, datetime-like objects, or labels."""

    closed_end: bool = attr.ib(default=_DEF)
    """Whether `end` should be inclusive."""

    indexer_method: tp.Optional[str] = attr.ib(default=_DEF)
    """Method for `pd.Index.get_indexer`."""

    level: tp.MaybeLevelSequence = attr.ib(default=None)
    """One or more levels.
    
    If `level` is not None and `kind` is None, `kind` becomes "labels"."""

    kind: tp.Optional[str] = attr.ib(default=None)
    """Kind of value.

    Allowed are
    
    * "positions" for `PosIdxr`, 
    * "mask" for `MaskIdxr`, 
    * "labels" for `LabelIdxr`, and 
    * "datetime" for `DatetimeIdxr`.
    
    If None, will (try to) determine automatically based on the type of indices."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        kind = self.kind
        if self.level is not None:
            from vectorbtpro.base.indexes import select_levels

            index = select_levels(index, self.level)
            if kind is None:
                kind = "labels"

        if kind is None:
            if isinstance(self.value, (slice, hslice)):
                if checks.is_int(self.value.start) or checks.is_int(self.value.stop):
                    kind = "positions"
                elif self.value.start is None and self.value.stop is None:
                    kind = "positions"
                elif isinstance(index, pd.DatetimeIndex):
                    kind = "datetime"
                else:
                    kind = "labels"
            elif (checks.is_sequence(self.value) and not np.isscalar(self.value)) and (
                not isinstance(index, pd.MultiIndex)
                or (isinstance(index, pd.MultiIndex) and isinstance(self.value[0], tuple))
            ):
                if isinstance(index, pd.MultiIndex) and isinstance(self.value[0], tuple):
                    kind = "labels"
                elif checks.is_bool(self.value[0]):
                    kind = "mask"
                elif checks.is_int(self.value[0]):
                    kind = "positions"
                elif isinstance(self.value[0], (slice, hslice)):
                    kind = "positions"
                elif checks.is_sequence(self.value[0]) and not np.isscalar(self.value[0][0]):
                    kind = "positions"
                elif isinstance(index, pd.DatetimeIndex):
                    kind = "datetime"
                else:
                    kind = "labels"
            else:
                if checks.is_bool(self.value):
                    kind = "mask"
                elif checks.is_int(self.value):
                    kind = "positions"
                elif isinstance(index, pd.DatetimeIndex):
                    kind = "datetime"
                else:
                    kind = "labels"

        if kind.lower() == "positions":
            idx = PosIdxr(self.value)
        elif kind.lower() == "mask":
            idx = MaskIdxr(self.value)
        elif kind.lower() == "labels":
            idxr_kwargs = dict()
            if self.closed_end is not _DEF:
                idxr_kwargs["closed_end"] = self.closed_end
            idx = LabelIdxr(self.value, **idxr_kwargs)
        elif kind.lower() == "datetime":
            idxr_kwargs = dict()
            if self.closed_end is not _DEF:
                idxr_kwargs["closed_end"] = self.closed_end
            if self.indexer_method is not _DEF:
                idxr_kwargs["indexer_method"] = self.indexer_method
            idx = DatetimeIdxr(self.value, **idxr_kwargs)
        else:
            raise ValueError(f"Invalid option kind='{kind}'")
        return idx.get(index, freq=freq)


@attr.s(frozen=True)
class PointIdxr(UniIdxr):
    """Class for resolving index points."""

    every: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Frequency either as an integer or timedelta.
    
    Gets translated into `on` array by creating a range. If integer, an index sequence from `start` to `end` 
    (exclusive) is created and 'indices' as `kind` is used. If timedelta-like, a date sequence from 
    `start` to `end` (inclusive) is created and 'labels' as `kind` is used.
    
    If `at_time` is not None and `every` and `on` are None, `every` defaults to one day."""

    normalize_every: bool = attr.ib(default=False)
    """Normalize start/end dates to midnight before generating date range."""

    at_time: tp.Optional[tp.TimeLike] = attr.ib(default=None)
    """Time of the day either as a (human-readable) string or `datetime.time`. 
    
    Every datetime in `on` gets floored to the daily frequency, while `at_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_delta`. 
    Index must be datetime-like."""

    start: tp.Optional[tp.Union[int, tp.DatetimeLike]] = attr.ib(default=None)
    """Start index/date.
    
    If (human-readable) string, gets converted into a datetime.
    
    If `every` is None, gets used to filter the final index array."""

    end: tp.Optional[tp.Union[int, tp.DatetimeLike]] = attr.ib(default=None)
    """End index/date.
    
    If (human-readable) string, gets converted into a datetime.
    
    If `every` is None, gets used to filter the final index array."""

    exact_start: bool = attr.ib(default=False)
    """Whether the first index should be exactly `start`.
    
    Depending on `every`, the first index picked by `pd.date_range` may happen after `start`.
    In such a case, `start` gets injected before the first index generated by `pd.date_range`."""

    on: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = attr.ib(default=None)
    """Index/label or a sequence of such.
    
    Gets converted into datetime format whenever possible."""

    add_delta: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Offset to be added to each in `on`.
    
    If string, gets converted into an offset using 
    [to_offset](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html)."""

    kind: tp.Optional[str] = attr.ib(default=None)
    """Kind of data in `on`: indices or labels.
    
    If None, gets assigned to `indices` if `on` contains integer data, otherwise to `labels`.
    
    If `kind` is 'labels', `on` gets converted into indices using `pd.Index.get_indexer`. 
    Prior to this, gets its timezone aligned to the timezone of the index. If `kind` is 'indices', 
    `on` gets wrapped with NumPy."""

    indexer_method: str = attr.ib(default="bfill")
    """Method for `pd.Index.get_indexer`."""

    indexer_tolerance: tp.Optional[tp.Union[int, tp.TimedeltaLike, tp.IndexLike]] = attr.ib(default=None)
    """Tolerance for `pd.Index.get_indexer`.
    
    If `at_time` is set and `indexer_method` is neither exact nor nearest, `indexer_tolerance` 
    becomes such that the next element must be within the current day."""

    skip_minus_one: bool = attr.ib(default=True)
    """Whether to remove indices that are -1 (not found)."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        indices = get_index_points(index, **attr.asdict(self))
        self.check_indices(indices)
        return indices


point_idxr_defaults = {a.name: a.default for a in PointIdxr.__attrs_attrs__}


def get_index_points(
    index: tp.Index,
    every: tp.Optional[tp.FrequencyLike] = point_idxr_defaults["every"],
    normalize_every: bool = point_idxr_defaults["normalize_every"],
    at_time: tp.Optional[tp.TimeLike] = point_idxr_defaults["at_time"],
    start: tp.Optional[tp.Union[int, tp.DatetimeLike]] = point_idxr_defaults["start"],
    end: tp.Optional[tp.Union[int, tp.DatetimeLike]] = point_idxr_defaults["end"],
    exact_start: bool = point_idxr_defaults["exact_start"],
    on: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = point_idxr_defaults["on"],
    add_delta: tp.Optional[tp.FrequencyLike] = point_idxr_defaults["add_delta"],
    kind: tp.Optional[str] = point_idxr_defaults["kind"],
    indexer_method: str = point_idxr_defaults["indexer_method"],
    indexer_tolerance: str = point_idxr_defaults["indexer_tolerance"],
    skip_minus_one: bool = point_idxr_defaults["skip_minus_one"],
) -> tp.Array1d:
    """Translate indices or labels into index points.

    See `PointIdxr` for arguments.

    Usage:
        * Provide nothing to generate at the beginning:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd

        >>> index = pd.date_range("2020-01", "2020-02", freq="1d")

        >>> vbt.get_index_points(index)
        array([0])
        ```

        * Provide `every` as an integer frequency to generate index points using NumPy:

        ```pycon
        >>> # Generate a point every five rows
        >>> vbt.get_index_points(index, every=5)
        array([ 0,  5, 10, 15, 20, 25, 30])

        >>> # Generate a point every five rows starting at 6th row
        >>> vbt.get_index_points(index, every=5, start=5)
        array([ 5, 10, 15, 20, 25, 30])

        >>> # Generate a point every five rows from 6th to 16th row
        >>> vbt.get_index_points(index, every=5, start=5, end=15)
        array([ 5, 10])
        ```

        * Provide `every` as a time delta frequency to generate index points using Pandas:

        ```pycon
        >>> # Generate a point every week
        >>> vbt.get_index_points(index, every="W")
        array([ 4, 11, 18, 25])

        >>> # Generate a point every second day of the week
        >>> vbt.get_index_points(index, every="W", add_delta="2d")
        array([ 6, 13, 20, 27])

        >>> # Generate a point every week, starting at 11th row
        >>> vbt.get_index_points(index, every="W", start=10)
        array([11, 18, 25])

        >>> # Generate a point every week, starting exactly at 11th row
        >>> vbt.get_index_points(index, every="W", start=10, exact_start=True)
        array([10, 11, 18, 25])

        >>> # Generate a point every week, starting at 2020-01-10
        >>> vbt.get_index_points(index, every="W", start="2020-01-10")
        array([11, 18, 25])
        ```

        * Instead of using `every`, provide indices explicitly:

        ```pycon
        >>> # Generate one point
        >>> vbt.get_index_points(index, on="2020-01-07")
        array([6])

        >>> # Generate multiple points
        >>> vbt.get_index_points(index, on=["2020-01-07", "2020-01-14"])
        array([ 6, 13])
        ```
    """
    import dateparser

    if on is not None and isinstance(on, str):
        try:
            on = pd.Timestamp(on, tz=index.tzinfo)
        except Exception as e:
            on = dateparser.parse(on)
            if index.tzinfo is not None:
                on = on.replace(tzinfo=index.tzinfo)
    if start is not None and isinstance(start, str):
        try:
            start = pd.Timestamp(start, tz=index.tzinfo)
        except Exception as e:
            start = dateparser.parse(start)
            if index.tzinfo is not None:
                start = start.replace(tzinfo=index.tzinfo)
    if end is not None and isinstance(end, str):
        try:
            end = pd.Timestamp(end, tz=index.tzinfo)
        except Exception as e:
            end = dateparser.parse(end)
            if index.tzinfo is not None:
                end = end.replace(tzinfo=index.tzinfo)

    start_used = False
    end_used = False
    if at_time is not None and every is None and on is None:
        every = "D"
    if every is not None:
        start_used = True
        end_used = True
        if checks.is_int(every):
            if start is None:
                start = 0
            if end is None:
                end = len(index)
            on = np.arange(start, end, every)
            kind = "indices"
        else:
            if start is None:
                start = 0
            if checks.is_int(start):
                start_date = index[start]
            else:
                start_date = start
            if end is None:
                end = len(index) - 1
            if checks.is_int(end):
                end_date = index[end]
            else:
                end_date = end
            on = pd.date_range(
                start_date,
                end_date,
                freq=every,
                tz=index.tzinfo,
                normalize=normalize_every,
            )
            if exact_start and on[0] > start_date:
                on = on.insert(0, start_date)
            kind = "labels"

    if kind is None:
        if on is None:
            if start is not None:
                if checks.is_int(start):
                    kind = "indices"
                else:
                    kind = "labels"
            else:
                kind = "indices"
        else:
            on = try_to_datetime_index(on)
            if on.is_integer():
                kind = "indices"
            else:
                kind = "labels"
    checks.assert_in(kind, ("indices", "labels"))
    if on is None:
        if start is not None:
            on = start
            start_used = True
        else:
            if kind.lower() in ("labels",):
                on = index[0]
            else:
                on = 0
    on = try_to_datetime_index(on)

    if at_time is not None:
        checks.assert_instance_of(on, pd.DatetimeIndex)
        on = on.floor("D")
        add_time_delta = time_to_timedelta(at_time)
        if indexer_tolerance is None:
            if indexer_method in ("pad", "ffill"):
                indexer_tolerance = add_time_delta
            elif indexer_method in ("backfill", "bfill"):
                indexer_tolerance = pd.Timedelta(days=1) - pd.Timedelta(1, "ns") - add_time_delta
        if add_delta is None:
            add_delta = add_time_delta
        else:
            add_delta += add_time_delta

    if add_delta is not None:
        if isinstance(add_delta, str):
            add_delta = prepare_freq(add_delta)
            try:
                add_delta = to_offset(add_delta)
            except Exception as e:
                add_delta = to_offset(pd.Timedelta(add_delta))
        on += add_delta

    if kind.lower() == "labels":
        on = try_align_to_dt_index(on, index)
        index_points = index.get_indexer(on, method=indexer_method, tolerance=indexer_tolerance)
    else:
        index_points = np.asarray(on)

    if start is not None and not start_used:
        if not checks.is_int(start):
            start = index.get_indexer([start], method="bfill").item(0)
        index_points = index_points[index_points >= start]
    if end is not None and not end_used:
        if not checks.is_int(end):
            end = index.get_indexer([end], method="ffill").item(0)
            index_points = index_points[index_points <= end]
        else:
            index_points = index_points[index_points < end]

    if skip_minus_one:
        index_points = index_points[index_points != -1]

    return index_points


@attr.s(frozen=True)
class RangeIdxr(UniIdxr):
    """Class for resolving index ranges."""

    every: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Frequency either as an integer or timedelta.

    Gets translated into `start` and `end` arrays by creating a range. If integer, an index sequence from `start` 
    to `end` (exclusive) is created and 'indices' as `kind` is used. If timedelta-like, a date sequence 
    from `start` to `end` (inclusive) is created and 'bounds' as `kind` is used. 

    If `start_time` and `end_time` are not None and `every`, `start`, and `end` are None, 
    `every` defaults to one day."""

    normalize_every: bool = attr.ib(default=False)
    """Normalize start/end dates to midnight before generating date range."""

    split_every: bool = attr.ib(default=True)
    """Whether to split the sequence generated using `every` into `start` and `end` arrays.

    After creation, and if `split_every` is True, an index range is created from each pair of elements in 
    the generated sequence. Otherwise, the entire sequence is assigned to `start` and `end`, and only time 
    and delta instructions can be used to further differentiate between them.

    Forced to False if `every`, `start_time`, and `end_time` are not None and `fixed_start` is False."""

    start_time: tp.Optional[tp.TimeLike] = attr.ib(default=None)
    """Start time of the day either as a (human-readable) string or `datetime.time`. 

    Every datetime in `start` gets floored to the daily frequency, while `start_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_start_delta`. 
    Index must be datetime-like."""

    end_time: tp.Optional[tp.TimeLike] = attr.ib(default=None)
    """End time of the day either as a (human-readable) string or `datetime.time`. 

    Every datetime in `end` gets floored to the daily frequency, while `end_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_end_delta`. 
    Index must be datetime-like."""

    lookback_period: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Lookback period either as an integer or offset.

    If `lookback_period` is set, `start` becomes `end-lookback_period`. If `every` is not None, 
    the sequence is generated from `start+lookback_period` to `end` and then assigned to `end`.

    If string, gets converted into an offset using 
    [to_offset](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html).
    If integer, gets multiplied by the frequency of the index if the index is not integer."""

    start: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = attr.ib(default=None)
    """Start index/label or a sequence of such.

    Gets converted into datetime format whenever possible.

    Gets broadcasted together with `end`."""

    end: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = attr.ib(default=None)
    """End index/label or a sequence of such.

    Gets converted into datetime format whenever possible.

    Gets broadcasted together with `start`."""

    exact_start: bool = attr.ib(default=False)
    """Whether the first index in the `start` array should be exactly `start`.

    Depending on `every`, the first index picked by `pd.date_range` may happen after `start`.
    In such a case, `start` gets injected before the first index generated by `pd.date_range`.

    Cannot be used together with `lookback_period`."""

    fixed_start: bool = attr.ib(default=False)
    """Whether all indices in the `start` array should be exactly `start`.

    Works only together with `every`.

    Cannot be used together with `lookback_period`."""

    closed_start: bool = attr.ib(default=True)
    """Whether `start` should be inclusive."""

    closed_end: bool = attr.ib(default=False)
    """Whether `end` should be inclusive.

    !!! note
        Indices will still be exclusive."""

    add_start_delta: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Offset to be added to each in `start`.

    If string, gets converted into an offset using 
    [to_offset](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html)."""

    add_end_delta: tp.Optional[tp.FrequencyLike] = attr.ib(default=None)
    """Offset to be added to each in `end`.

    If string, gets converted into an offset using 
    [to_offset](https://pandas.pydata.org/docs/reference/api/pandas.tseries.frequencies.to_offset.html)."""

    kind: tp.Optional[str] = attr.ib(default=None)
    """Kind of data in `on`: indices, labels or bounds.

    If None, gets assigned to `indices` if `start` and `end` contain integer data, to `bounds`
    if `start`, `end`, and index are datetime-like, otherwise to `labels`.

    If `kind` is 'labels', `start` and `end` get converted into indices using `pd.Index.get_indexer`. 
    Prior to this, get their timezone aligned to the timezone of the index. If `kind` is 'indices', 
    `start` and `end` get wrapped with NumPy. If kind` is 'bounds', 
    `vectorbtpro.base.resampling.base.Resampler.map_bounds_to_source_ranges` is used."""

    skip_minus_one: bool = attr.ib(default=True)
    """Whether to remove indices that are -1 (not found)."""

    jitted: tp.JittedOption = attr.ib(default=None)
    """Jitting option passed to `vectorbtpro.base.resampling.base.Resampler.map_bounds_to_source_ranges`."""

    def get(self, index: tp.Index, freq: tp.Optional[tp.FrequencyLike] = None) -> tp.MaybeIndexArray:
        start_indices, end_indices = get_index_ranges(index, index_freq=freq, **attr.asdict(self))
        indices = np.column_stack((start_indices, end_indices))
        return indices


range_idxr_defaults = {a.name: a.default for a in RangeIdxr.__attrs_attrs__}


def get_index_ranges(
    index: tp.Index,
    index_freq: tp.Optional[tp.FrequencyLike] = None,
    every: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["every"],
    normalize_every: bool = range_idxr_defaults["normalize_every"],
    split_every: bool = range_idxr_defaults["split_every"],
    start_time: tp.Optional[tp.TimeLike] = range_idxr_defaults["start_time"],
    end_time: tp.Optional[tp.TimeLike] = range_idxr_defaults["end_time"],
    lookback_period: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["lookback_period"],
    start: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = range_idxr_defaults["start"],
    end: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = range_idxr_defaults["end"],
    exact_start: bool = range_idxr_defaults["exact_start"],
    fixed_start: bool = range_idxr_defaults["fixed_start"],
    closed_start: bool = range_idxr_defaults["closed_start"],
    closed_end: bool = range_idxr_defaults["closed_end"],
    add_start_delta: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["add_start_delta"],
    add_end_delta: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["add_end_delta"],
    kind: tp.Optional[str] = range_idxr_defaults["kind"],
    skip_minus_one: bool = range_idxr_defaults["skip_minus_one"],
    jitted: tp.JittedOption = range_idxr_defaults["jitted"],
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Translate indices, labels, or bounds into index ranges.

    See `RangeIdxr` for arguments.

    Usage:
        * Provide nothing to generate one largest index range:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd
        >>> import numpy as np

        >>> index = pd.date_range("2020-01", "2020-02", freq="1d")

        >>> np.column_stack(vbt.get_index_ranges(index))
        array([[ 0, 32]])
        ```

        * Provide `every` as an integer frequency to generate index ranges using NumPy:

        ```pycon
        >>> # Generate a range every five rows
        >>> np.column_stack(vbt.get_index_ranges(index, every=5))
        array([[ 0,  5],
               [ 5, 10],
               [10, 15],
               [15, 20],
               [20, 25],
               [25, 30]])

        >>> # Generate a range every five rows, starting at 6th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every=5,
        ...     start=5
        ... ))
        array([[ 5, 10],
               [10, 15],
               [15, 20],
               [20, 25],
               [25, 30]])

        >>> # Generate a range every five rows from 6th to 16th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every=5,
        ...     start=5,
        ...     end=15
        ... ))
        array([[ 5, 10],
               [10, 15]])
        ```

        * Provide `every` as a time delta frequency to generate index ranges using Pandas:

        ```pycon
        >>> # Generate a range every week
        >>> np.column_stack(vbt.get_index_ranges(index, every="W"))
        array([[ 4, 11],
               [11, 18],
               [18, 25]])

        >>> # Generate a range every second day of the week
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     add_start_delta="2d"
        ... ))
        array([[ 6, 11],
               [13, 18],
               [20, 25]])

        >>> # Generate a range every week, starting at 11th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start=10
        ... ))
        array([[11, 18],
               [18, 25]])

        >>> # Generate a range every week, starting exactly at 11th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start=10,
        ...     exact_start=True
        ... ))
        array([[10, 11],
               [11, 18],
               [18, 25]])

        >>> # Generate a range every week, starting at 2020-01-10
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start="2020-01-10"
        ... ))
        array([[11, 18],
               [18, 25]])

        >>> # Generate a range every week, each starting at 2020-01-10
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start="2020-01-10",
        ...     fixed_start=True
        ... ))
        array([[11, 18],
               [11, 25]])
        ```

        * Use a look-back period (instead of an end index):

        ```pycon
        >>> # Generate a range every week, looking 5 days back
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     lookback_period=5
        ... ))
        array([[ 6, 11],
               [13, 18],
               [20, 25]])

        >>> # Generate a range every week, looking 2 weeks back
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     lookback_period="2W"
        ... ))
        array([[ 0, 11],
               [ 4, 18],
               [11, 25]])
        ```

        * Instead of using `every`, provide start and end indices explicitly:

        ```pycon
        >>> # Generate one range
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start="2020-01-01",
        ...     end="2020-01-07"
        ... ))
        array([[0, 6]])

        >>> # Generate ranges between multiple dates
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start=["2020-01-01", "2020-01-07"],
        ...     end=["2020-01-07", "2020-01-14"]
        ... ))
        array([[ 0,  6],
               [ 6, 13]])

        >>> # Generate ranges with a fixed start
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start="2020-01-01",
        ...     end=["2020-01-07", "2020-01-14"]
        ... ))
        array([[ 0,  6],
               [ 0, 13]])
        ```

        * Use `closed_start` and `closed_end` to exclude any of the bounds:

        ```pycon
        >>> # Generate ranges between multiple dates
        >>> # by excluding the start date and including the end date
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start=["2020-01-01", "2020-01-07"],
        ...     end=["2020-01-07", "2020-01-14"],
        ...     closed_start=False,
        ...     closed_end=True
        ... ))
        array([[ 1,  7],
               [ 7, 14]])
        ```
    """
    from vectorbtpro.base.indexes import repeat_index
    from vectorbtpro.base.resampling.base import Resampler

    index = try_to_datetime_index(index)
    if isinstance(index, pd.DatetimeIndex):
        if start is not None:
            start = try_align_to_dt_index(start, index)
            if isinstance(start, pd.DatetimeIndex):
                start = start.tz_localize(None)
        if end is not None:
            end = try_align_to_dt_index(end, index)
            if isinstance(end, pd.DatetimeIndex):
                end = end.tz_localize(None)
        naive_index = index.tz_localize(None)
    else:
        if start is not None:
            if not isinstance(start, pd.Index):
                try:
                    start = pd.Index(start)
                except Exception as e:
                    start = pd.Index([start])
        if end is not None:
            if not isinstance(end, pd.Index):
                try:
                    end = pd.Index(end)
                except Exception as e:
                    end = pd.Index([end])
        naive_index = index
    if lookback_period is not None and not checks.is_int(lookback_period):
        try:
            lookback_period = to_offset(lookback_period)
        except Exception as e:
            lookback_period = to_offset(pd.Timedelta(lookback_period))
    if fixed_start and lookback_period is not None:
        raise ValueError("Cannot use fixed_start and lookback_period together")
    if exact_start and lookback_period is not None:
        raise ValueError("Cannot use exact_start and lookback_period together")

    if start_time is not None or end_time is not None:
        if every is None and start is None and end is None:
            every = "D"
    if every is not None:
        if not fixed_start:
            if start_time is None and end_time is not None:
                start_time = time(0, 0, 0, 0)
                closed_start = True
            if start_time is not None and end_time is None:
                end_time = time(0, 0, 0, 0)
                closed_end = False
        if start_time is not None and end_time is not None and not fixed_start:
            split_every = False

        if checks.is_int(every):
            if start is None:
                start = 0
            else:
                start = start[0]
            if end is None:
                end = len(naive_index)
            else:
                end = end[-1]
            if closed_end:
                end -= 1
            if lookback_period is None:
                new_index = np.arange(start, end + 1, every)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = np.full(len(new_index) - 1, new_index[0])
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]
            else:
                end = np.arange(start + lookback_period, end + 1, every)
                start = end - lookback_period
            kind = "indices"
            lookback_period = None
        else:
            if start is None:
                start = 0
            else:
                start = start[0]
            if checks.is_int(start):
                start_date = naive_index[start]
            else:
                start_date = start
            if end is None:
                end = len(naive_index) - 1
            else:
                end = end[-1]
            if checks.is_int(end):
                end_date = naive_index[end]
            else:
                end_date = end
            if lookback_period is None:
                new_index = pd.date_range(
                    start_date,
                    end_date,
                    freq=every,
                    normalize=normalize_every,
                )
                if exact_start and new_index[0] > start_date:
                    new_index = new_index.insert(0, start_date)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = repeat_index(new_index[[0]], len(new_index) - 1)
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]
            else:
                if checks.is_int(lookback_period):
                    lookback_period *= infer_index_freq(naive_index, freq=index_freq)
                end = pd.date_range(
                    start_date + lookback_period,
                    end_date,
                    freq=every,
                    normalize=normalize_every,
                )
                start = end - lookback_period
            kind = "bounds"
            lookback_period = None

    if kind is None:
        if start is None and end is None:
            kind = "indices"
        else:
            if start is not None:
                ref_index = start
            if end is not None:
                ref_index = end
            if ref_index.is_integer():
                kind = "indices"
            elif isinstance(ref_index, pd.DatetimeIndex) and isinstance(naive_index, pd.DatetimeIndex):
                kind = "bounds"
            else:
                kind = "labels"
    checks.assert_in(kind, ("indices", "labels", "bounds"))
    if end is None:
        if kind.lower() in ("labels", "bounds"):
            end = pd.Index([naive_index[-1]])
        else:
            end = pd.Index([len(naive_index)])
    if start is not None and lookback_period is not None:
        raise ValueError("Cannot use start and lookback_period together")
    if start is None:
        if lookback_period is None:
            if kind.lower() in ("labels", "bounds"):
                start = pd.Index([naive_index[0]])
            else:
                start = pd.Index([0])
        else:
            if checks.is_int(lookback_period) and not end.is_integer():
                lookback_period *= infer_index_freq(naive_index, freq=index_freq)
            start = end - lookback_period
    if len(start) == 1 and len(end) > 1:
        start = repeat_index(start, len(end))
    elif len(start) > 1 and len(end) == 1:
        end = repeat_index(end, len(start))
    checks.assert_len_equal(start, end)

    if start_time is not None:
        checks.assert_instance_of(start, pd.DatetimeIndex)
        start = start.floor("D")
        add_start_time_delta = time_to_timedelta(start_time)
        if add_start_delta is None:
            add_start_delta = add_start_time_delta
        else:
            add_start_delta += add_start_time_delta
    else:
        add_start_time_delta = None
    if end_time is not None:
        checks.assert_instance_of(end, pd.DatetimeIndex)
        end = end.floor("D")
        add_end_time_delta = time_to_timedelta(end_time)
        if add_start_time_delta is not None:
            if add_end_time_delta < add_start_delta:
                add_end_time_delta += pd.Timedelta(days=1)
        if add_end_delta is None:
            add_end_delta = add_end_time_delta
        else:
            add_end_delta += add_end_time_delta

    if add_start_delta is not None:
        if isinstance(add_start_delta, str):
            add_start_delta = prepare_freq(add_start_delta)
            try:
                add_start_delta = to_offset(add_start_delta)
            except Exception as e:
                add_start_delta = to_offset(pd.Timedelta(add_start_delta))
        start += add_start_delta
    if add_end_delta is not None:
        if isinstance(add_end_delta, str):
            add_end_delta = prepare_freq(add_end_delta)
            try:
                add_end_delta = to_offset(add_end_delta)
            except Exception as e:
                add_end_delta = to_offset(pd.Timedelta(add_end_delta))
        end += add_end_delta

    if kind.lower() == "bounds":
        range_starts, range_ends = Resampler.map_bounds_to_source_ranges(
            source_index=naive_index.values,
            target_lbound_index=start.values,
            target_rbound_index=end.values,
            closed_lbound=closed_start,
            closed_rbound=closed_end,
            skip_minus_one=skip_minus_one,
            jitted=jitted,
        )
    elif kind.lower() == "labels":
        range_starts = np.empty(len(start), dtype=np.int_)
        range_ends = np.empty(len(end), dtype=np.int_)
        range_index = pd.Series(np.arange(len(naive_index)), index=naive_index)
        for i in range(len(range_starts)):
            selected_range = range_index[start[i]:end[i]]
            if len(selected_range) > 0 and not closed_start and selected_range.index[0] == start[i]:
                selected_range = selected_range.iloc[1:]
            if len(selected_range) > 0 and not closed_end and selected_range.index[-1] == end[i]:
                selected_range = selected_range.iloc[:-1]
            if len(selected_range) > 0:
                range_starts[i] = selected_range.iloc[0]
                range_ends[i] = selected_range.iloc[-1]
            else:
                range_starts[i] = -1
                range_ends[i] = -1
        if skip_minus_one:
            valid_mask = (range_starts != -1) & (range_ends != -1)
            range_starts = range_starts[valid_mask]
            range_ends = range_ends[valid_mask]
    else:
        if not closed_start:
            start = start + 1
        if closed_end:
            end = end + 1
        range_starts = np.asarray(start)
        range_ends = np.asarray(end)
        if skip_minus_one:
            valid_mask = (range_starts != -1) & (range_ends != -1)
            range_starts = range_starts[valid_mask]
            range_ends = range_ends[valid_mask]

    if np.any(range_starts >= range_ends):
        raise ValueError("Some start indices are equal to or higher than end indices")

    return range_starts, range_ends


@attr.s(frozen=True, init=False)
class RowIdxr(IdxrBase):
    """Class for resolving row indices."""

    idxr: object = attr.ib()
    """Indexer.
    
    Can be an instance of `UniIdxr`, a custom template, or a value to be wrapped with `AutoIdxr`."""

    idxr_kwargs: tp.KwargsLike = attr.ib()
    """Keyword arguments passed to `AutoIdxr`."""

    def __init__(self, idxr: object, **idxr_kwargs) -> None:
        self.__attrs_init__(idxr=idxr, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        index: tp.Index,
        freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.MaybeIndexArray:
        idxr = self.idxr
        if isinstance(idxr, CustomTemplate):
            _template_context = merge_dicts(dict(index=index, freq=freq), template_context)
            idxr = idxr.substitute(_template_context, sub_id="idxr")
        if not isinstance(idxr, UniIdxr):
            if isinstance(idxr, IdxrBase):
                raise TypeError(f"Indexer of {type(self)} must be an instance of UniIdxr")
            idxr = AutoIdxr(idxr, **self.idxr_kwargs)
        return idxr.get(index, freq=freq)


@attr.s(frozen=True, init=False)
class ColIdxr(IdxrBase):
    """Class for resolving column indices."""

    idxr: object = attr.ib()
    """Indexer.
        
    Can be an instance of `UniIdxr`, a custom template, or a value to be wrapped with `AutoIdxr`."""

    idxr_kwargs: tp.KwargsLike = attr.ib()
    """Keyword arguments passed to `AutoIdxr`."""

    def __init__(self, idxr: object, **idxr_kwargs) -> None:
        self.__attrs_init__(idxr=idxr, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        columns: tp.Index,
        template_context: tp.KwargsLike = None,
    ) -> tp.MaybeIndexArray:
        idxr = self.idxr
        if isinstance(idxr, CustomTemplate):
            _template_context = merge_dicts(dict(columns=columns), template_context)
            idxr = idxr.substitute(_template_context, sub_id="idxr")
        if not isinstance(idxr, UniIdxr):
            if isinstance(idxr, IdxrBase):
                raise TypeError(f"Indexer of {type(self)} must be an instance of UniIdxr")
            idxr = AutoIdxr(idxr, **self.idxr_kwargs)
        return idxr.get(columns)


@attr.s(frozen=True, init=False)
class Idxr(IdxrBase):
    """Class for resolving indices."""

    idxrs: tp.Tuple[object, ...] = attr.ib()
    """A tuple of one or more indexers.
    
    If one indexer is provided, can be an instance of `RowIdxr` or `ColIdxr`, 
    a custom template, or a value to wrapped with `RowIdxr`.
    
    If two indexers are provided, can be an instance of `RowIdxr` and `ColIdxr` respectively,
    or a value to wrapped with `RowIdxr` and `ColIdxr` respectively."""

    idxr_kwargs: tp.KwargsLike = attr.ib()
    """Keyword arguments passed to `RowIdxr` and `ColIdxr`."""

    def __init__(self, *idxrs: object, **idxr_kwargs) -> None:
        self.__attrs_init__(idxrs=idxrs, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        index: tp.Index,
        columns: tp.Index,
        freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.MaybeIndexArray, tp.MaybeIndexArray]:
        if len(self.idxrs) == 0:
            raise ValueError("At least one indexer must be provided")
        elif len(self.idxrs) == 1:
            idxr = self.idxrs[0]
            if isinstance(idxr, CustomTemplate):
                _template_context = merge_dicts(dict(index=index, columns=columns, freq=freq), template_context)
                idxr = idxr.substitute(_template_context, sub_id="idxr")
                if isinstance(idxr, tuple):
                    return type(self)(*idxr).get(index, columns, freq=freq, template_context=template_context)
                return type(self)(idxr).get(index, columns, freq=freq, template_context=template_context)
            if isinstance(idxr, ColIdxr):
                row_idxr = None
                col_idxr = idxr
            else:
                row_idxr = idxr
                col_idxr = None
        elif len(self.idxrs) == 2:
            row_idxr = self.idxrs[0]
            col_idxr = self.idxrs[1]
        else:
            raise ValueError("At most two indexers must be provided")
        if not isinstance(row_idxr, RowIdxr):
            if isinstance(row_idxr, (ColIdxr, Idxr)):
                raise TypeError(f"Indexer {type(row_idxr)} not supported as a row indexer")
            row_idxr = RowIdxr(row_idxr, **self.idxr_kwargs)
        row_indices = row_idxr.get(index, freq=freq, template_context=template_context)
        if not isinstance(col_idxr, ColIdxr):
            if isinstance(col_idxr, (RowIdxr, Idxr)):
                raise TypeError(f"Indexer {type(col_idxr)} not supported as a column indexer")
            col_idxr = ColIdxr(col_idxr, **self.idxr_kwargs)
        col_indices = col_idxr.get(columns, template_context=template_context)
        return row_indices, col_indices


class index_dict(pdict):
    """Dict that contains indexer objects as keys.

    Each indexer object must be hashable. To make a slice hashable, use `hslice`."""

    pass


def get_indices(
    index: tp.Index,
    columns: tp.Index,
    idxr: object,
    freq: tp.Optional[tp.FrequencyLike] = None,
    template_context: tp.KwargsLike = None,
    **kwargs,
) -> tp.Tuple[tp.MaybeIndexArray, tp.MaybeIndexArray]:
    """Translate indexer to row and column indices.

    If `idxr` is not an indexer class, wraps it with `Idxr`.

    Keyword arguments are passed when constructing a new `Idxr`."""
    if not isinstance(idxr, Idxr):
        idxr = Idxr(idxr, **kwargs)
    return idxr.get(index, columns, freq=freq, template_context=template_context)


pointidx = PointIdxr
"""Shortcut for `PointIdxr`."""

__pdoc__["pointidx"] = False

rangeidx = RangeIdxr
"""Shortcut for `RangeIdxr`."""

__pdoc__["rangeidx"] = False

rowidx = RowIdxr
"""Shortcut for `RowIdxr`."""

__pdoc__["rowidx"] = False

colidx = ColIdxr
"""Shortcut for `ColIdxr`."""

__pdoc__["colidx"] = False

idx = Idxr
"""Shortcut for `Idxr`."""

__pdoc__["idx"] = False


def set_rows(a: tp.Array, x: tp.MaybeIndexArray, v: tp.Any) -> None:
    """Set row indices in an array."""
    from vectorbtpro.base.reshaping import broadcast_array_to

    if not isinstance(v, np.ndarray):
        v = np.asarray(v)
    single_v = v.size == 1 or (v.ndim == 2 and v.shape[0] == 1)
    if a.ndim == 2:
        single_x = not isinstance(x, slice) and (np.isscalar(x) or x.size == 1)
        if not single_x:
            if v.ndim == 1 and v.size > 1:
                v = v[:, None]

    if isinstance(x, np.ndarray) and x.ndim == 2:
        if not single_v:
            if a.ndim == 2:
                v = broadcast_array_to(v, (len(x), a.shape[1]))
            else:
                v = broadcast_array_to(v, (len(x),))
        for i in range(len(x)):
            x_slice = slice(x[i, 0], x[i, 1])
            if not single_v:
                set_rows(a, x_slice, v[[i]])
            else:
                set_rows(a, x_slice, v)
    else:
        a[x] = v


def set_cols(a: tp.Array, y: tp.MaybeIndexArray, v: tp.Any) -> None:
    """Set column indices in an array."""
    from vectorbtpro.base.reshaping import broadcast_array_to

    if not isinstance(v, np.ndarray):
        v = np.asarray(v)
    single_v = v.size == 1 or (v.ndim == 2 and v.shape[1] == 1)

    if isinstance(y, np.ndarray) and y.ndim == 2:
        if not single_v:
            v = broadcast_array_to(v, (a.shape[0], len(y)))
        for j in range(len(y)):
            y_slice = slice(y[j, 0], y[j, 1])
            if not single_v:
                set_cols(a, y_slice, v[:, [j]])
            else:
                set_cols(a, y_slice, v)
    else:
        a[:, y] = v


def set_rows_and_cols(a: tp.Array, x: tp.MaybeIndexArray, y: tp.MaybeIndexArray, v: tp.Any) -> None:
    """Set row and column indices in an array."""
    from vectorbtpro.base.reshaping import broadcast_array_to

    if not isinstance(v, np.ndarray):
        v = np.asarray(v)
    single_v = v.size == 1
    if isinstance(x, np.ndarray) and x.ndim == 2 and isinstance(y, np.ndarray) and y.ndim == 2:
        if not single_v:
            v = broadcast_array_to(v, (len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                x_slice = slice(x[i, 0], x[i, 1])
                y_slice = slice(y[j, 0], y[j, 1])
                if not single_v:
                    set_rows_and_cols(a, x_slice, y_slice, v[i, j])
                else:
                    set_rows_and_cols(a, x_slice, y_slice, v)
    elif isinstance(x, np.ndarray) and x.ndim == 2:
        if not single_v:
            if isinstance(y, slice):
                y = np.arange(a.shape[1])[y]
            v = broadcast_array_to(v, (len(x), len(y)))
        for i in range(len(x)):
            x_slice = slice(x[i, 0], x[i, 1])
            if not single_v:
                set_rows_and_cols(a, x_slice, y, v[[i]])
            else:
                set_rows_and_cols(a, x_slice, y, v)
    elif isinstance(y, np.ndarray) and y.ndim == 2:
        if not single_v:
            if isinstance(x, slice):
                x = np.arange(a.shape[0])[x]
            v = broadcast_array_to(v, (len(x), len(y)))
        for j in range(len(y)):
            y_slice = slice(y[j, 0], y[j, 1])
            if not single_v:
                set_rows_and_cols(a, x, y_slice, v[:, [j]])
            else:
                set_rows_and_cols(a, x, y_slice, v)
    else:
        if np.isscalar(x) or np.isscalar(y):
            a[x, y] = v
        elif np.isscalar(v) and (isinstance(x, slice) or isinstance(y, slice)):
            a[x, y] = v
        elif np.isscalar(v):
            a[np.ix_(x, y)] = v
        else:
            x = np.arange(a.shape[0])[x]
            y = np.arange(a.shape[1])[y]
            v = broadcast_array_to(v, (len(x), len(y)))
            a[np.ix_(x, y)] = v
