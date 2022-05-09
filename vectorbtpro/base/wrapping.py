# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Classes for wrapping NumPy arrays into Series/DataFrames."""

import dateparser
import warnings
from datetime import time

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.indexing import IndexingError, PandasIndexer
from vectorbtpro.base.indexes import repeat_index
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import AttrResolverMixin, AttrResolverMixinT
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.datetime_ import infer_index_freq, try_to_datetime_index, time_to_timedelta
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.decorators import class_or_instancemethod

ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")
IndexingMetaT = tp.Tuple[ArrayWrapperT, tp.MaybeArray, tp.MaybeArray, tp.Array1d]


class ArrayWrapper(Configured, PandasIndexer):
    """Class that stores index, columns, and shape metadata for wrapping NumPy arrays.
    Tightly integrated with `vectorbtpro.base.grouping.base.Grouper` for grouping columns.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    `**kwargs` are passed to `vectorbtpro.base.grouping.base.Grouper`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `ArrayWrapper.replace`.

        Use methods that begin with `get_` to get group-aware results."""

    def __init__(
        self,
        index: tp.IndexLike,
        columns: tp.IndexLike,
        ndim: int,
        freq: tp.Optional[tp.FrequencyLike] = None,
        column_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        grouped_ndim: tp.Optional[int] = None,
        grouper: tp.Optional[Grouper] = None,
        **kwargs,
    ) -> None:

        checks.assert_not_none(index)
        checks.assert_not_none(columns)
        checks.assert_not_none(ndim)
        index = try_to_datetime_index(index)
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        grouper_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in grouper_arg_names:
                grouper_kwargs[k] = kwargs.pop(k)
        if grouper is None:
            grouper = Grouper(columns, **grouper_kwargs)
        elif not checks.is_index_equal(columns, grouper.index) or len(grouper_kwargs) > 0:
            grouper = grouper.replace(index=columns, **grouper_kwargs)

        PandasIndexer.__init__(self)
        Configured.__init__(
            self,
            index=index,
            columns=columns,
            ndim=ndim,
            freq=freq,
            column_only_select=column_only_select,
            group_select=group_select,
            grouped_ndim=grouped_ndim,
            grouper=grouper,
            **kwargs,
        )

        self._index = index
        self._columns = columns
        self._ndim = ndim
        self._freq = freq
        self._column_only_select = column_only_select
        self._group_select = group_select
        self._grouper = grouper
        self._grouped_ndim = grouped_ndim

    def indexing_func_meta(
        self: ArrayWrapperT,
        pd_indexing_func: tp.PandasIndexingFunc,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        column_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        group_by: tp.GroupByLike = None,
    ) -> IndexingMetaT:
        """Perform indexing on `ArrayWrapper` and also return metadata.

        Takes into account column grouping.

        Set `column_only_select` to True to index the array wrapper as a Series of columns.
        This way, selection of index (axis 0) can be avoided. Set `group_select` to True
        to select groups rather than columns. Takes effect only if grouping is enabled.

        !!! note
            If `column_only_select` is True, make sure to index the array wrapper
            as a Series of columns rather than a DataFrame. For example, the operation
            `.iloc[:, :2]` should become `.iloc[:2]`. Operations are not allowed if the
            object is already a Series and thus has only one column/group."""
        if column_only_select is None:
            column_only_select = self.column_only_select
        if group_select is None:
            group_select = self.group_select
        _self = self.regroup(group_by)
        group_select = group_select and _self.grouper.is_grouped()
        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            if group_select:
                columns = _self.grouper.get_index()
            else:
                columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if group_select:
            # Groups as columns
            i_wrapper = ArrayWrapper(index, columns, _self.get_ndim())
        else:
            # Columns as columns
            i_wrapper = ArrayWrapper(index, columns, _self.ndim)
        n_rows = len(index)
        n_cols = len(columns)

        if column_only_select:
            if i_wrapper.ndim == 1:
                raise IndexingError("Columns only: This object already contains one column of data")
            try:
                col_mapper = pd_indexing_func(i_wrapper.wrap_reduced(np.arange(n_cols), columns=columns))
            except pd.core.indexing.IndexingError as e:
                warnings.warn(
                    "Columns only: Make sure to treat this object as a Series of columns rather than a DataFrame",
                    stacklevel=2,
                )
                raise e
            if checks.is_series(col_mapper):
                new_columns = col_mapper.index
                col_idxs = col_mapper.values
                new_ndim = 2
            else:
                new_columns = columns[[col_mapper]]
                col_idxs = col_mapper
                new_ndim = 1
            new_index = index
            idx_idxs = np.arange(len(index))
        else:
            idx_mapper = pd_indexing_func(
                i_wrapper.wrap(
                    np.broadcast_to(np.arange(n_rows)[:, None], (n_rows, n_cols)),
                    index=index,
                    columns=columns,
                )
            )
            if i_wrapper.ndim == 1:
                if not checks.is_series(idx_mapper):
                    raise IndexingError("Selection of a scalar is not allowed")
                idx_idxs = idx_mapper.values
                col_idxs = 0
            else:
                col_mapper = pd_indexing_func(
                    i_wrapper.wrap(np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)), index=index, columns=columns),
                )
                if checks.is_frame(idx_mapper):
                    idx_idxs = idx_mapper.values[:, 0]
                    col_idxs = col_mapper.values[0]
                elif checks.is_series(idx_mapper):
                    one_col = np.all(col_mapper.values == col_mapper.values.item(0))
                    one_idx = np.all(idx_mapper.values == idx_mapper.values.item(0))
                    if one_col and one_idx:
                        # One index and one column selected, multiple times
                        raise IndexingError("Must select at least two unique indices in one of both axes")
                    elif one_col:
                        # One column selected
                        idx_idxs = idx_mapper.values
                        col_idxs = col_mapper.values[0]
                    elif one_idx:
                        # One index selected
                        idx_idxs = idx_mapper.values[0]
                        col_idxs = col_mapper.values
                    else:
                        raise IndexingError
                else:
                    raise IndexingError("Selection of a scalar is not allowed")
            new_index = indexes.get_index(idx_mapper, 0)
            if not isinstance(idx_idxs, np.ndarray):
                # One index selected
                new_columns = index[[idx_idxs]]
            elif not isinstance(col_idxs, np.ndarray):
                # One column selected
                new_columns = columns[[col_idxs]]
            else:
                new_columns = indexes.get_index(idx_mapper, 1)
            new_ndim = idx_mapper.ndim

        if _self.grouper.is_grouped():
            # Grouping enabled
            if np.asarray(idx_idxs).ndim == 0:
                raise IndexingError("Flipping index and columns is not allowed")

            if group_select:
                # Selection based on groups
                # Get indices of columns corresponding to selected groups
                group_idxs = col_idxs
                new_group_idxs, new_groups = _self.grouper.select_groups(group_idxs)
                ungrouped_columns = _self.columns[new_group_idxs]
                if new_ndim == 1 and len(ungrouped_columns) == 1:
                    ungrouped_ndim = 1
                    new_group_idxs = new_group_idxs[0]
                else:
                    ungrouped_ndim = 2

                return (
                    _self.replace(
                        index=new_index,
                        columns=ungrouped_columns,
                        ndim=ungrouped_ndim,
                        grouped_ndim=new_ndim,
                        group_by=new_columns[new_groups],
                    ),
                    idx_idxs,
                    group_idxs,
                    new_group_idxs,
                )

            # Selection based on columns
            col_idxs_arr = reshaping.to_1d_array(col_idxs)
            return (
                _self.replace(
                    index=new_index,
                    columns=new_columns,
                    ndim=new_ndim,
                    grouped_ndim=None,
                    group_by=_self.grouper.group_by[col_idxs_arr],
                ),
                idx_idxs,
                col_idxs,
                col_idxs,
            )

        # Grouping disabled
        return (
            _self.replace(index=new_index, columns=new_columns, ndim=new_ndim, grouped_ndim=None, group_by=None),
            idx_idxs,
            col_idxs,
            col_idxs,
        )

    def indexing_func(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform indexing on `ArrayWrapper`"""
        return self.indexing_func_meta(*args, **kwargs)[0]

    @classmethod
    def from_obj(cls: tp.Type[ArrayWrapperT], obj: tp.ArrayLike, *args, **kwargs) -> ArrayWrapperT:
        """Derive metadata from an object."""
        from vectorbtpro.base.reshaping import to_pd_array

        pd_obj = to_pd_array(obj)
        index = indexes.get_index(pd_obj, 0)
        columns = indexes.get_index(pd_obj, 1)
        ndim = pd_obj.ndim
        kwargs.pop("index", None)
        kwargs.pop("columns", None)
        kwargs.pop("ndim", None)
        return cls(index, columns, ndim, *args, **kwargs)

    @classmethod
    def from_shape(
        cls: tp.Type[ArrayWrapperT],
        shape: tp.ShapeLike,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        ndim: tp.Optional[int] = None,
        *args,
        **kwargs,
    ) -> ArrayWrapperT:
        """Derive metadata from shape."""
        shape = reshaping.shape_to_tuple(shape)
        if index is None:
            index = pd.RangeIndex(start=0, step=1, stop=shape[0])
        if columns is None:
            columns = pd.RangeIndex(start=0, step=1, stop=shape[1] if len(shape) > 1 else 1)
        if ndim is None:
            ndim = len(shape)
        return cls(index, columns, ndim, *args, **kwargs)

    @property
    def index(self) -> tp.Index:
        """Index."""
        return self._index

    @property
    def columns(self) -> tp.Index:
        """Columns."""
        return self._columns

    def get_columns(self, group_by: tp.GroupByLike = None) -> tp.Index:
        """Get group-aware `ArrayWrapper.columns`."""
        return self.resolve(group_by=group_by).columns

    @property
    def name(self) -> tp.Any:
        """Name."""
        if self.ndim == 1:
            if self.columns[0] == 0:
                return None
            return self.columns[0]
        return None

    def get_name(self, group_by: tp.GroupByLike = None) -> tp.Any:
        """Get group-aware `ArrayWrapper.name`."""
        return self.resolve(group_by=group_by).name

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._ndim

    def get_ndim(self, group_by: tp.GroupByLike = None) -> int:
        """Get group-aware `ArrayWrapper.ndim`."""
        return self.resolve(group_by=group_by).ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape."""
        if self.ndim == 1:
            return (len(self.index),)
        return len(self.index), len(self.columns)

    def get_shape(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape`."""
        return self.resolve(group_by=group_by).shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object was two-dimensional."""
        if self.ndim == 1:
            return self.shape[0], 1
        return self.shape

    def get_shape_2d(self, group_by: tp.GroupByLike = None) -> tp.Shape:
        """Get group-aware `ArrayWrapper.shape_2d`."""
        return self.resolve(group_by=group_by).shape_2d

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """Index frequency as `pd.Timedelta` or None if it cannot be converted.

        Date offsets and integer frequencies are not allowed."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        freq = self._freq
        if freq is None:
            freq = wrapping_cfg["freq"]
        try:
            return infer_index_freq(self.index, freq=freq, allow_date_offset=False, allow_numeric=False)
        except Exception as e:
            return None

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """Index frequency of any type."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        freq = self._freq
        if freq is None:
            freq = wrapping_cfg["freq"]
        return infer_index_freq(self.index, freq=freq)

    @property
    def period(self) -> int:
        """Get the period of the index, without taking into account its datetime-like properties."""
        return len(self.index)

    @property
    def dt_period(self) -> float:
        """Get the period of the index, taking into account its datetime-like properties."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if isinstance(self.index, pd.DatetimeIndex):
            if self.freq is not None:
                return (self.index[-1] - self.index[0]) / self.freq + 1
            if not wrapping_cfg["silence_warnings"]:
                warnings.warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`.",
                    stacklevel=2,
                )
        if isinstance(self.index[0], int) and isinstance(self.index[-1], int):
            return self.index[-1] - self.index[0] + 1
        if not wrapping_cfg["silence_warnings"]:
            warnings.warn("Index is neither datetime-like nor integer", stacklevel=2)
        return self.period

    def to_timedelta(
        self,
        a: tp.MaybeArray[float],
        to_pd: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.Union[pd.Timedelta, np.timedelta64, tp.Array]:
        """Convert array to duration using `ArrayWrapper.freq`."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        if self.freq is None:
            if not silence_warnings:
                warnings.warn(
                    "Couldn't parse the frequency of index. Pass it as `freq` or "
                    "define it globally under `settings.wrapping`.",
                    stacklevel=2,
                )
            return a
        if to_pd:
            return pd.to_timedelta(a * self.freq)
        return a * self.freq

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on columns only."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        column_only_select = self._column_only_select
        if column_only_select is None:
            column_only_select = wrapping_cfg["column_only_select"]
        return column_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on groups."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        group_select = self._group_select
        if group_select is None:
            group_select = wrapping_cfg["group_select"]
        return group_select

    @property
    def grouper(self) -> Grouper:
        """Column grouper."""
        return self._grouper

    @property
    def grouped_ndim(self) -> int:
        """Number of dimensions under column grouping."""
        if self._grouped_ndim is None:
            if self.grouper.is_grouped():
                return 2 if self.grouper.get_group_count() > 1 else 1
            return self.ndim
        return self._grouped_ndim

    def regroup(self: ArrayWrapperT, group_by: tp.GroupByLike, **kwargs) -> ArrayWrapperT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself."""
        if self.grouper.is_grouping_changed(group_by=group_by):
            self.grouper.check_group_by(group_by=group_by)
            grouped_ndim = None
            if self.grouper.is_grouped(group_by=group_by):
                if not self.grouper.is_group_count_changed(group_by=group_by):
                    grouped_ndim = self.grouped_ndim
            return self.replace(grouped_ndim=grouped_ndim, group_by=group_by, **kwargs)
        return self  # important for keeping cache

    def resolve(self: ArrayWrapperT, group_by: tp.GroupByLike = None, **kwargs) -> ArrayWrapperT:
        """Resolve this object.

        Replaces columns and other metadata with groups."""
        _self = self.regroup(group_by=group_by, **kwargs)
        if _self.grouper.is_grouped():
            return _self.replace(
                columns=_self.grouper.get_index(),
                ndim=_self.grouped_ndim,
                grouped_ndim=None,
                group_by=None,
            )
        return _self  # important for keeping cache

    def wrap(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SeriesFrame:
        """Wrap a NumPy array using the stored metadata.

        Runs the following pipeline:

        1) Converts to NumPy array
        2) Fills NaN (optional)
        3) Wraps using index, columns, and dtype (optional)
        4) Converts to index (optional)
        5) Converts to timedelta using `ArrayWrapper.to_timedelta` (optional)"""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        _self = self.resolve(group_by=group_by)

        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if len(columns) == 1:
            name = columns[0]
            if name == 0:  # was a Series before
                name = None
        else:
            name = None

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap(arr):
            arr = np.asarray(arr)
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            shape_2d = (arr.shape[0] if arr.ndim > 0 else 1, arr.shape[1] if arr.ndim > 1 else 1)
            target_shape_2d = (len(index), len(columns))
            if shape_2d != target_shape_2d:
                arr = np.broadcast_to(arr, target_shape_2d)
            arr = reshaping.soft_to_ndim(arr, self.ndim)
            if arr.ndim == 1:
                return _apply_dtype(pd.Series(arr, index=index, name=name))
            if arr.ndim == 2:
                if arr.shape[1] == 1 and _self.ndim == 1:
                    return _apply_dtype(pd.Series(arr[:, 0], index=index, name=name))
                return _apply_dtype(pd.DataFrame(arr, index=index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
        if to_timedelta:
            # Convert to timedelta
            out = self.to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def wrap_reduced(
        self,
        arr: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        name_or_index: tp.NameIndex = None,
        columns: tp.Optional[tp.IndexLike] = None,
        fillna: tp.Optional[tp.Scalar] = None,
        dtype: tp.Optional[tp.PandasDTypeLike] = None,
        to_timedelta: bool = False,
        to_index: bool = False,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.MaybeSeriesFrame:
        """Wrap result of reduction.

        `name_or_index` can be the name of the resulting series if reducing to a scalar per column,
        or the index of the resulting series/dataframe if reducing to an array per column.
        `columns` can be set to override object's default columns.

        See `ArrayWrapper.wrap` for the pipeline."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        checks.assert_not_none(self.ndim)
        _self = self.resolve(group_by=group_by)

        if columns is None:
            columns = _self.columns
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)

        if to_index:
            if dtype is None:
                dtype = np.int_
            if fillna is None:
                fillna = -1

        def _apply_dtype(obj):
            if dtype is None:
                return obj
            return obj.astype(dtype, errors="ignore")

        def _wrap_reduced(arr):
            nonlocal name_or_index

            arr = np.asarray(arr)
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            if arr.ndim == 0:
                # Scalar per Series/DataFrame
                return _apply_dtype(pd.Series(arr))[0]
            if arr.ndim == 1:
                if _self.ndim == 1:
                    if arr.shape[0] == 1:
                        # Scalar per Series/DataFrame with one column
                        return _apply_dtype(pd.Series(arr))[0]
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:  # was arr Series before
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Scalar per column in arr DataFrame
                return _apply_dtype(pd.Series(arr, index=columns, name=name_or_index))
            if arr.ndim == 2:
                if arr.shape[1] == 1 and _self.ndim == 1:
                    arr = reshaping.soft_to_ndim(arr, 1)
                    # Array per Series
                    sr_name = columns[0]
                    if sr_name == 0:  # was arr Series before
                        sr_name = None
                    if isinstance(name_or_index, str):
                        name_or_index = None
                    return _apply_dtype(pd.Series(arr, index=name_or_index, name=sr_name))
                # Array per column in DataFrame
                if isinstance(name_or_index, str):
                    name_or_index = None
                return _apply_dtype(pd.DataFrame(arr, index=name_or_index, columns=columns))
            raise ValueError(f"{arr.ndim}-d input is not supported")

        out = _wrap_reduced(arr)
        if to_index:
            # Convert to index
            if checks.is_series(out):
                out = out.map(lambda x: self.index[x] if x != -1 else np.nan)
            elif checks.is_frame(out):
                out = out.applymap(lambda x: self.index[x] if x != -1 else np.nan)
            else:
                out = self.index[out] if out != -1 else np.nan
        if to_timedelta:
            # Convert to timedelta
            out = self.to_timedelta(out, silence_warnings=silence_warnings)
        return out

    def dummy(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Create a dummy Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.empty(_self.shape), **kwargs)

    def fill(self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Fill a Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap(np.full(_self.shape_2d, fill_value), **kwargs)

    def fill_reduced(self, fill_value: tp.Scalar = np.nan, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Fill a reduced Series/DataFrame."""
        _self = self.resolve(group_by=group_by)
        return _self.wrap_reduced(np.full(_self.shape_2d[1], fill_value), **kwargs)

    def create_index_grouper(self, by: tp.Union[Grouper, tp.PandasGroupByLike], **kwargs) -> Grouper:
        """Create an index grouper of type `vectorbtpro.base.grouping.base.Grouper`."""
        if isinstance(by, Grouper):
            return by
        if isinstance(by, (PandasGroupBy, PandasResampler)):
            return Grouper.from_pd_group_by(by)
        try:
            return Grouper(index=self.index, group_by=by)
        except Exception as e:
            pass
        if isinstance(self.index, pd.DatetimeIndex):
            try:
                return Grouper(index=self.index, group_by=self.index.to_period(by))
            except Exception as e:
                pass
            try:
                pd_group_by = pd.Series(index=self.index, dtype=object).resample(by, **kwargs)
                return Grouper.from_pd_group_by(pd_group_by)
            except Exception as e:
                pass
        pd_group_by = pd.Series(index=self.index, dtype=object).groupby(by, axis=0, **kwargs)
        return Grouper.from_pd_group_by(pd_group_by)

    def create_resampler(
        self,
        rule: tp.Union[Resampler, tp.PandasResampler, tp.PandasFrequencyLike],
        resample_kwargs: tp.KwargsLike = None,
        return_pd_resampler: bool = False,
    ) -> tp.Union[Resampler, tp.PandasResampler]:
        """Create a resampler of type `vectorbtpro.base.resampling.base.Resampler`."""
        if not isinstance(rule, Resampler):
            if not isinstance(rule, PandasResampler):
                resample_kwargs = merge_dicts(
                    dict(closed="left", label="left"),
                    resample_kwargs,
                )
                rule = pd.Series(index=self.index, dtype=object).resample(rule, **resolve_dict(resample_kwargs))
            if return_pd_resampler:
                return rule
            rule = Resampler.from_pd_resampler(rule)
        if return_pd_resampler:
            raise TypeError("Cannot convert Resampler to Pandas Resampler")
        return rule

    def resample_meta(
        self: ArrayWrapperT,
        *args,
        **kwargs,
    ) -> tp.Tuple[tp.Union[Resampler, tp.PandasResampler], ArrayWrapperT]:
        """Perform resampling on `ArrayWrapper` and also return metadata.

        `*args` and `**kwargs` are passed to `ArrayWrapper.create_resampler`."""
        resampler = self.create_resampler(*args, **kwargs)
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        new_index = _resampler.target_index
        new_freq = infer_index_freq(new_index)
        new_wrapper = self.replace(index=new_index, freq=new_freq)
        return resampler, new_wrapper

    def resample(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform resampling on `ArrayWrapper`.

        Uses `ArrayWrapper.resample_meta`."""
        return self.resample_meta(*args, **kwargs)[1]

    def get_index_points(
        self,
        every: tp.Optional[tp.FrequencyLike] = None,
        normalize_every: bool = False,
        at_time: tp.Optional[tp.TimeLike] = None,
        start: tp.Optional[tp.Union[int, tp.DatetimeLike]] = None,
        end: tp.Optional[tp.Union[int, tp.DatetimeLike]] = None,
        exact_start: bool = False,
        on: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = None,
        add_delta: tp.Optional[tp.FrequencyLike] = None,
        kind: tp.Optional[str] = None,
        indexer_method: str = "bfill",
        skip_minus_one: bool = True,
    ) -> tp.Array1d:
        """Translate indices or labels into index points.

        !!! note
            `start` and `end` are both inclusive if provided as dates or used in `pd.date_range`.
            Otherwise, only `start` is inclusive.

            If `at_time` is not None, the resulting `on` index will be floored to a daily index
            and `at_time` will be converted to a timedelta and added to `add_delta`.
            Also, if `every` and `on` are both None, `every` will be set to "D".

        Usage:
            * Provide nothing to generate at the beginning:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> data = vbt.YFData.fetch("BTC-USD", start="2020-01-01", end="2020-02-01")

            >>> data.wrapper.get_index_points()
            array([0])
            ```

            * Provide `every` as an integer frequency to generate index points using NumPy:

            ```pycon
            >>> # Generate a point every five rows
            >>> data.wrapper.get_index_points(every=5)
            array([ 0,  5, 10, 15, 20, 25, 30])

            >>> # Generate a point every five rows starting at 6th row
            >>> data.wrapper.get_index_points(every=5, start=5)
            array([ 5, 10, 15, 20, 25, 30])

            >>> # Generate a point every five rows from 6th to 16th row
            >>> data.wrapper.get_index_points(every=5, start=5, end=15)
            array([ 5, 10])
            ```

            * Provide `every` as a time delta frequency to generate index points using Pandas:

            ```pycon
            >>> # Generate a point every week
            >>> data.wrapper.get_index_points(every="W")
            array([ 5, 12, 19, 26])

            >>> # Generate a point every second day of the week
            >>> data.wrapper.get_index_points(every="W", add_delta="2d")
            array([ 7, 14, 21, 28])

            >>> # Generate a point every week, starting at 11th row
            >>> data.wrapper.get_index_points(every="W", start=10)
            array([12, 19, 26])

            >>> # Generate a point every week, starting exactly at 11th row
            >>> data.wrapper.get_index_points(every="W", start=10, exact_start=True)
            array([10, 12, 19, 26])

            >>> # Generate a point every week, starting at 2020-01-10
            >>> data.wrapper.get_index_points(every="W", start="2020-01-10")
            array([12, 19, 26])
            ```

            * Instead of using `every`, provide indices explicitly:

            ```pycon
            >>> # Generate one point
            >>> data.wrapper.get_index_points(on="2020-01-07")
            array([7])

            >>> # Generate multiple points
            >>> data.wrapper.get_index_points(on=["2020-01-07", "2020-01-14"])
            array([ 7, 14])
            ```
        """
        if on is not None and isinstance(on, str):
            try:
                on = pd.Timestamp(on, tz=self.index.tzinfo)
            except Exception as e:
                on = dateparser.parse(on)
                if self.index.tzinfo is not None:
                    on = on.replace(tzinfo=self.index.tzinfo)
        if start is not None and isinstance(start, str):
            try:
                start = pd.Timestamp(start, tz=self.index.tzinfo)
            except Exception as e:
                start = dateparser.parse(start)
                if self.index.tzinfo is not None:
                    start = start.replace(tzinfo=self.index.tzinfo)
        if end is not None and isinstance(end, str):
            try:
                end = pd.Timestamp(end, tz=self.index.tzinfo)
            except Exception as e:
                end = dateparser.parse(end)
                if self.index.tzinfo is not None:
                    end = end.replace(tzinfo=self.index.tzinfo)

        start_used = False
        end_used = False
        if at_time is not None and every is None and on is None:
            every = "D"
        if every is not None:
            start_used = True
            end_used = True
            if isinstance(every, int):
                if start is None:
                    start = 0
                if end is None:
                    end = len(self.index)
                on = np.arange(start, end, every)
                kind = "indices"
            else:
                if start is None:
                    start = 0
                if isinstance(start, int):
                    start_date = self.index[start]
                else:
                    start_date = start
                if end is None:
                    end = len(self.index) - 1
                if isinstance(end, int):
                    end_date = self.index[end]
                else:
                    end_date = end
                on = pd.date_range(
                    start_date,
                    end_date,
                    freq=every,
                    tz=self.index.tzinfo,
                    normalize=normalize_every,
                )
                if exact_start and on[0] > start_date:
                    on = on.insert(0, start_date)
                kind = "labels"

        if kind is None:
            if on is None:
                if start is not None:
                    if isinstance(start, int):
                        kind = "indices"
                    else:
                        kind = "labels"
                else:
                    kind = "indices"
            else:
                on = try_to_datetime_index(on)
                if on.is_integer() and not self.index.is_integer():
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
                    on = self.index[0]
                else:
                    on = 0
        on = try_to_datetime_index(on)

        if at_time is not None:
            checks.assert_instance_of(on, pd.DatetimeIndex)
            on = on.floor("D")
            add_time_delta = time_to_timedelta(at_time)
            if add_delta is None:
                add_delta = add_time_delta
            else:
                add_delta += add_time_delta

        if add_delta is not None:
            if isinstance(add_delta, str):
                try:
                    add_delta = to_offset(add_delta)
                except Exception as e:
                    add_delta = to_offset(pd.Timedelta(add_delta))
            on += add_delta

        if kind.lower() == "labels":
            if isinstance(on, pd.DatetimeIndex):
                if on.tzinfo is None and self.index.tzinfo is not None:
                    on = on.tz_localize(self.index.tzinfo)
                elif on.tzinfo is not None and self.index.tzinfo is not None:
                    on = on.tz_convert(self.index.tzinfo)
            index_points = self.index.get_indexer(on, method=indexer_method)
        else:
            index_points = np.asarray(on)

        if start is not None and not start_used:
            if not isinstance(start, int):
                start = self.index.get_indexer([start], method=indexer_method)[0]
            index_points = index_points[index_points >= start]
        if end is not None and not end_used:
            if not isinstance(end, int):
                end = self.index.get_indexer([end], method=indexer_method)[0]
                index_points = index_points[index_points <= end]
            else:
                index_points = index_points[index_points < end]

        if skip_minus_one:
            index_points = index_points[index_points != -1]

        return index_points

    def get_index_ranges(
        self,
        every: tp.Optional[tp.FrequencyLike] = None,
        normalize_every: bool = False,
        split_every: bool = True,
        start_time: tp.Optional[tp.TimeLike] = None,
        end_time: tp.Optional[tp.TimeLike] = None,
        lookback_period: tp.Optional[tp.FrequencyLike] = None,
        start: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = None,
        end: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = None,
        exact_start: bool = False,
        fixed_start: bool = False,
        closed_start: bool = True,
        closed_end: bool = False,
        add_start_delta: tp.Optional[tp.FrequencyLike] = None,
        add_end_delta: tp.Optional[tp.FrequencyLike] = None,
        kind: tp.Optional[str] = None,
        indexer_method: str = "bfill",
        skip_minus_one: bool = True,
        jitted: tp.JittedOption = None,
    ) -> tp.Array2d:
        """Translate indices, labels, or bounds into index ranges.

        Usage:
            * Provide nothing to generate one largest index range:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> data = vbt.YFData.fetch("BTC-USD", start="2020-01-01", end="2020-02-01")

            >>> data.wrapper.get_index_ranges()
            array([[ 0, 32]])
            ```

            * Provide `every` as an integer frequency to generate index ranges using NumPy:

            ```pycon
            >>> # Generate a range every five rows
            >>> data.wrapper.get_index_ranges(every=5)
            array([[ 0,  5],
                   [ 5, 10],
                   [10, 15],
                   [15, 20],
                   [20, 25],
                   [25, 30]])

            >>> # Generate a range every five rows, starting at 6th row
            >>> data.wrapper.get_index_ranges(every=5, start=5)
            array([[ 5, 10],
                   [10, 15],
                   [15, 20],
                   [20, 25],
                   [25, 30]])

            >>> # Generate a range every five rows from 6th to 16th row
            >>> data.wrapper.get_index_ranges(every=5, start=5, end=15)
            array([[ 5, 10],
                   [10, 15]])
            ```

            * Provide `every` as a time delta frequency to generate index ranges using Pandas:

            ```pycon
            >>> # Generate a range every week
            >>> data.wrapper.get_index_ranges(every="W")
            array([[ 5, 12],
                   [12, 19],
                   [19, 26]])

            >>> # Generate a range every second day of the week
            >>> data.wrapper.get_index_ranges(every="W", add_start_delta="2d")
            array([[ 7, 12],
                   [14, 19],
                   [21, 26]])

            >>> # Generate a range every week, starting at 11th row
            >>> data.wrapper.get_index_ranges(every="W", start=10)
            array([[12, 19],
                   [19, 26]])

            >>> # Generate a range every week, starting exactly at 11th row
            >>> data.wrapper.get_index_ranges(every="W", start=10, exact_start=True)
            array([[10, 12],
                   [12, 19],
                   [19, 26]])

            >>> # Generate a range every week, starting at 2020-01-10
            >>> data.wrapper.get_index_ranges(every="W", start="2020-01-10")
            array([[12, 19],
                   [19, 26]])

            >>> # Generate a range every week, each starting at 2020-01-10
            >>> data.wrapper.get_index_ranges(every="W", start="2020-01-10", fixed_start=True)
            array([[12, 19],
                   [12, 26]])
            ```

            * Use a look-back period (instead of an end index):

            ```pycon
            >>> # Generate a range every week, looking 5 days back
            >>> data.wrapper.get_index_ranges(every="W", lookback_period=5)
            array([[ 0,  5],
                   [ 7, 12],
                   [14, 19],
                   [21, 26]])

            >>> # Generate a range every week, looking 2 weeks back
            >>> data.wrapper.get_index_ranges(every="W", lookback_period="2W")
            array([[ 0, 12],
                   [ 5, 19],
                   [12, 26]])
            ```

            * Instead of using `every`, provide start and end indices explicitly:

            ```pycon
            >>> # Generate one range
            >>> data.wrapper.get_index_ranges(
            ...     start="2020-01-01",
            ...     end="2020-01-07"
            ... )
            array([[1, 7]])

            >>> # Generate ranges between multiple dates
            >>> data.wrapper.get_index_ranges(
            ...     start=["2020-01-01", "2020-01-07"],
            ...     end=["2020-01-07", "2020-01-14"]
            ... )
            array([[ 1,  7],
                   [ 7, 14]])

            >>> # Generate ranges with a fixed start
            >>> data.wrapper.get_index_ranges(
            ...     start="2020-01-01",
            ...     end=["2020-01-07", "2020-01-14"]
            ... )
            array([[ 1,  7],
                   [ 1, 14]])
            ```

            * Use `closed_start` and `closed_end` to exclude any of the bounds:

            ```pycon
            >>> # Generate ranges between multiple dates
            >>> # by excluding the start date and including the end date
            >>> data.wrapper.get_index_ranges(
            ...     start=["2020-01-01", "2020-01-07"],
            ...     end=["2020-01-07", "2020-01-14"],
            ...     closed_start=False,
            ...     closed_end=True
            ... )
            array([[ 2,  8],
                   [ 8, 15]])
            ```
        """
        if start is not None and isinstance(start, str):
            try:
                start = pd.Timestamp(start, tz=self.index.tzinfo)
            except Exception as e:
                start = dateparser.parse(start)
                if self.index.tzinfo is not None:
                    start = start.replace(tzinfo=self.index.tzinfo)
        if end is not None and isinstance(end, str):
            try:
                end = pd.Timestamp(end, tz=self.index.tzinfo)
            except Exception as e:
                end = dateparser.parse(end)
                if self.index.tzinfo is not None:
                    end = end.replace(tzinfo=self.index.tzinfo)
        if lookback_period is not None and not isinstance(lookback_period, int):
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
                    if add_end_delta is None:
                        add_end_delta = pd.Timedelta(days=1)
                    else:
                        add_end_delta += pd.Timedelta(days=1)
                    closed_end = False
            if start_time is not None and end_time is not None and not fixed_start:
                split_every = False

            if isinstance(every, int):
                if start is None:
                    start = 0
                if end is None:
                    end = len(self.index)
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
                if isinstance(start, int):
                    start_date = self.index[start]
                else:
                    start_date = start
                if end is None:
                    end = len(self.index) - 1
                if isinstance(end, int):
                    end_date = self.index[end]
                else:
                    end_date = end
                if lookback_period is None:
                    new_index = pd.date_range(
                        start_date,
                        end_date,
                        freq=every,
                        tz=self.index.tzinfo,
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
                    if isinstance(lookback_period, int):
                        lookback_period *= self.any_freq
                    end = pd.date_range(
                        start_date + lookback_period,
                        end_date,
                        freq=every,
                        tz=self.index.tzinfo,
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
                    start = try_to_datetime_index(start)
                    ref_index = start
                if end is not None:
                    end = try_to_datetime_index(end)
                    ref_index = end
                if ref_index.is_integer() and not self.index.is_integer():
                    kind = "indices"
                elif isinstance(ref_index, pd.DatetimeIndex) and isinstance(self.index, pd.DatetimeIndex):
                    kind = "bounds"
                else:
                    kind = "labels"
        checks.assert_in(kind, ("indices", "labels", "bounds"))
        if end is None:
            if kind.lower() in ("labels", "bounds"):
                end = self.index[-1]
            else:
                end = len(self.index)
        end = try_to_datetime_index(end)
        if start is not None and lookback_period is not None:
            raise ValueError("Cannot use start and lookback_period together")
        if start is None:
            if lookback_period is None:
                if kind.lower() in ("labels", "bounds"):
                    start = self.index[0]
                else:
                    start = 0
            else:
                if isinstance(lookback_period, int) and not end.is_integer():
                    lookback_period *= self.any_freq
                start = end - lookback_period
        start = try_to_datetime_index(start)
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
        if end_time is not None:
            checks.assert_instance_of(end, pd.DatetimeIndex)
            end = end.floor("D")
            add_end_time_delta = time_to_timedelta(end_time)
            if add_end_delta is None:
                add_end_delta = add_end_time_delta
            else:
                add_end_delta += add_end_time_delta

        if add_start_delta is not None:
            if isinstance(add_start_delta, str):
                try:
                    add_start_delta = to_offset(add_start_delta)
                except Exception as e:
                    add_start_delta = to_offset(pd.Timedelta(add_start_delta))
            start += add_start_delta
        if add_end_delta is not None:
            if isinstance(add_end_delta, str):
                try:
                    add_end_delta = to_offset(add_end_delta)
                except Exception as e:
                    add_end_delta = to_offset(pd.Timedelta(add_end_delta))
            end += add_end_delta

        if kind.lower() == "bounds":
            index_ranges = Resampler.map_bounds_to_source_ranges(
                source_index=self.index.values,
                target_lbound_index=start.values,
                target_rbound_index=end.values,
                closed_lbound=closed_start,
                closed_rbound=closed_end,
                skip_minus_one=skip_minus_one,
                jitted=jitted,
            )
        elif kind.lower() == "labels":
            if isinstance(start, pd.DatetimeIndex):
                if start.tzinfo is None and self.index.tzinfo is not None:
                    start = start.tz_localize(self.index.tzinfo)
                elif start.tzinfo is not None and self.index.tzinfo is not None:
                    start = start.tz_convert(self.index.tzinfo)
            if isinstance(end, pd.DatetimeIndex):
                if end.tzinfo is None and self.index.tzinfo is not None:
                    end = end.tz_localize(self.index.tzinfo)
                elif end.tzinfo is not None and self.index.tzinfo is not None:
                    end = end.tz_convert(self.index.tzinfo)
            if closed_start:
                new_start = self.index.get_indexer(start, method=indexer_method)
            else:
                new_start = np.empty(len(start), dtype=np.int_)
                for i in range(len(start)):
                    if start[i] in self.index:
                        new_start[i] = self.index.get_loc(start[i]) + 1
                    else:
                        new_start[i] = self.index.get_indexer([start[i]], method=indexer_method)[0]
            if closed_end:
                new_end = np.empty(len(end), dtype=np.int_)
                for i in range(len(end)):
                    if end[i] in self.index:
                        new_end[i] = self.index.get_loc(end[i]) + 1
                    else:
                        new_end[i] = self.index.get_indexer([end[i]], method=indexer_method)[0]
            else:
                new_end = self.index.get_indexer(end, method=indexer_method)
            index_ranges = np.column_stack((new_start, new_end))
            if skip_minus_one:
                index_ranges = index_ranges[(index_ranges != -1).all(axis=1)]
        else:
            if not closed_start:
                start += 1
            if closed_end:
                end += 1
            index_ranges = np.column_stack((start, end))
            if skip_minus_one:
                index_ranges = index_ranges[(index_ranges != -1).all(axis=1)]

        if np.any(index_ranges[:, 0] >= index_ranges[:, 1]):
            raise ValueError("Some start indices are higher than end indices")

        return index_ranges


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, PandasIndexer, AttrResolverMixin):
    """Class that uses `ArrayWrapper` globally."""

    def __init__(self, wrapper: ArrayWrapper, **kwargs) -> None:
        checks.assert_instance_of(wrapper, ArrayWrapper)
        self._wrapper = wrapper

        Configured.__init__(self, wrapper=wrapper, **kwargs)
        PandasIndexer.__init__(self)
        AttrResolverMixin.__init__(self)

    def indexing_func(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform indexing on `Wrapping`."""
        new_wrapper = self.wrapper.indexing_func(
            *args,
            column_only_select=self.column_only_select,
            group_select=self.group_select,
            **kwargs,
        )
        return self.replace(wrapper=new_wrapper)

    def resample(self: WrappingT, *args, **kwargs) -> WrappingT:
        """Perform resampling on `Wrapping`.

        When overriding, make sure to create a resampler by passing `*args` and `**kwargs`
        to `ArrayWrapper.create_resampler`."""
        raise NotImplementedError

    @property
    def wrapper(self) -> ArrayWrapper:
        """Array wrapper."""
        return self._wrapper

    @property
    def column_only_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.column_only_select`."""
        column_only_select = getattr(self, "_column_only_select", None)
        if column_only_select is None:
            return self.wrapper.column_only_select
        return column_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.group_select`."""
        group_select = getattr(self, "_group_select", None)
        if group_select is None:
            return self.wrapper.group_select
        return group_select

    def regroup(self: WrappingT, group_by: tp.GroupByLike, **kwargs) -> WrappingT:
        """Regroup this object.

        Only creates a new instance if grouping has changed, otherwise returns itself.

        `**kwargs` will be passed to `ArrayWrapper.regroup`."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.replace(wrapper=self.wrapper.regroup(group_by, **kwargs))
        return self  # important for keeping cache

    def resolve_self(
        self: AttrResolverMixinT,
        cond_kwargs: tp.KwargsLike = None,
        custom_arg_names: tp.ClassVar[tp.Optional[tp.Set[str]]] = None,
        impacts_caching: bool = True,
        silence_warnings: tp.Optional[bool] = None,
    ) -> AttrResolverMixinT:
        """Resolve self.

        Creates a copy of this instance if a different `freq` can be found in `cond_kwargs`."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if cond_kwargs is None:
            cond_kwargs = {}
        if custom_arg_names is None:
            custom_arg_names = set()
        if silence_warnings is None:
            silence_warnings = wrapping_cfg["silence_warnings"]

        if "freq" in cond_kwargs:
            wrapper_copy = self.wrapper.replace(freq=cond_kwargs["freq"])

            if wrapper_copy.freq != self.wrapper.freq:
                if not silence_warnings:
                    warnings.warn(
                        f"Changing the frequency will create a copy of this object. "
                        f"Consider setting it upon object creation to re-use existing cache.",
                        stacklevel=2,
                    )
                self_copy = self.replace(wrapper=wrapper_copy)
                for alias in self.self_aliases:
                    if alias not in custom_arg_names:
                        cond_kwargs[alias] = self_copy
                cond_kwargs["freq"] = self_copy.wrapper.freq
                if impacts_caching:
                    cond_kwargs["use_caching"] = False
                return self_copy
        return self

    def select_col(self: WrappingT, column: tp.Any = None, group_by: tp.GroupByLike = None, **kwargs) -> WrappingT:
        """Select one column/group.

        `column` can be a label-based position as well as an integer position (if label fails)."""
        _self = self.regroup(group_by, **kwargs)

        def _check_out_dim(out: WrappingT) -> WrappingT:
            if _self.wrapper.grouper.is_grouped():
                if out.wrapper.grouped_ndim != 1:
                    raise TypeError("Could not select one group: multiple groups returned")
            else:
                if out.wrapper.ndim != 1:
                    raise TypeError("Could not select one column: multiple columns returned")
            return out

        if column is None:
            if _self.wrapper.get_ndim() == 2 and _self.wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if _self.wrapper.grouper.is_grouped() and _self.group_select:
                if _self.wrapper.grouped_ndim == 1:
                    raise TypeError("This object already contains one group of data")
                if column not in _self.wrapper.get_columns():
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Group '{column}' not found")
            else:
                if _self.wrapper.ndim == 1:
                    raise TypeError("This object already contains one column of data")
                if column not in _self.wrapper.columns:
                    if isinstance(column, int):
                        if _self.column_only_select:
                            return _check_out_dim(_self.iloc[column])
                        return _check_out_dim(_self.iloc[:, column])
                    raise KeyError(f"Column '{column}' not found")
            return _check_out_dim(_self[column])
        if not _self.wrapper.grouper.is_grouped() and _self.group_select:
            if _self.wrapper.ndim == 1:
                return _self
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if _self.wrapper.grouped_ndim == 1:
            return _self
        raise TypeError("Only one group is allowed. Use indexing or column argument.")

    @class_or_instancemethod
    def select_col_from_obj(
        cls_or_self,
        obj: tp.Optional[tp.SeriesFrame],
        column: tp.Any = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> tp.MaybeSeries:
        """Select one column/group from a pandas object.

        `column` can be a label-based position as well as an integer position (if label fails)."""
        if not isinstance(cls_or_self, type) and wrapper is None:
            wrapper = cls_or_self.wrapper
        if obj is None:
            return None
        if column is None:
            if wrapper.get_ndim() == 2 and wrapper.get_shape_2d()[1] == 1:
                column = 0
        if column is not None:
            if wrapper.ndim == 1:
                raise TypeError("This object already contains one column of data")
            if wrapper.grouper.is_grouped():
                if column not in wrapper.get_columns():
                    if isinstance(column, int):
                        if isinstance(obj, pd.DataFrame):
                            return obj.iloc[:, column]
                        return obj.iloc[column]
                    raise KeyError(f"Group '{column}' not found")
            else:
                if column not in wrapper.columns:
                    if isinstance(column, int):
                        if isinstance(obj, pd.DataFrame):
                            return obj.iloc[:, column]
                        return obj.iloc[column]
                    raise KeyError(f"Column '{column}' not found")
            return obj[column]
        if not wrapper.grouper.is_grouped():
            if wrapper.ndim == 1:
                return obj
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        if wrapper.grouped_ndim == 1:
            return obj
        raise TypeError("Only one group is allowed. Use indexing or column argument.")
