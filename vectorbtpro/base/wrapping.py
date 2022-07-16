# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Classes for wrapping NumPy arrays into Series/DataFrames."""

import warnings

import numpy as np
import pandas as pd
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler
from pandas.tseries.frequencies import to_offset

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, reshaping
from vectorbtpro.base.grouping.base import Grouper
from vectorbtpro.base.resampling.base import Resampler
from vectorbtpro.base.indexing import (
    IndexingError,
    PandasIndexer,
    get_index_points,
    get_index_ranges,
    index_dict,
    hslice,
    get_indices,
)
from vectorbtpro.base.indexes import repeat_index, stack_indexes
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import AttrResolverMixin, AttrResolverMixinT
from vectorbtpro.utils.config import Configured, merge_dicts, resolve_dict
from vectorbtpro.utils.datetime_ import infer_index_freq, try_to_datetime_index
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.decorators import class_or_instancemethod, cached_method
from vectorbtpro.utils.array_ import is_range
from vectorbtpro.utils.template import CustomTemplate

ArrayWrapperT = tp.TypeVar("ArrayWrapperT", bound="ArrayWrapper")


class ArrayWrapper(Configured, PandasIndexer):
    """Class that stores index, columns, and shape metadata for wrapping NumPy arrays.
    Tightly integrated with `vectorbtpro.base.grouping.base.Grouper` for grouping columns.

    If the underlying object is a Series, pass `[sr.name]` as `columns`.

    `**kwargs` are passed to `vectorbtpro.base.grouping.base.Grouper`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `ArrayWrapper.replace`.

        Use methods that begin with `get_` to get group-aware results."""

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
            index = pd.RangeIndex(stop=shape[0])
        if columns is None:
            columns = pd.RangeIndex(stop=shape[1] if len(shape) > 1 else 1)
        if ndim is None:
            ndim = len(shape)
        return cls(index, columns, ndim, *args, **kwargs)

    @staticmethod
    def extract_init_kwargs(**kwargs) -> tp.Tuple[tp.Kwargs, tp.Kwargs]:
        """Extract keyword arguments that can be passed to `ArrayWrapper` or `Grouper`."""
        wrapper_arg_names = get_func_arg_names(ArrayWrapper.__init__)
        grouper_arg_names = get_func_arg_names(Grouper.__init__)
        init_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in wrapper_arg_names or k in grouper_arg_names:
                init_kwargs[k] = kwargs.pop(k)
        return init_kwargs, kwargs

    @classmethod
    def resolve_stack_kwargs(cls, *wrappers: tp.MaybeTuple[ArrayWrapperT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `ArrayWrapper` after stacking."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)

        common_keys = set()
        for wrapper in wrappers:
            common_keys = common_keys.union(set(wrapper.config.keys()))
            if "grouper" not in kwargs:
                common_keys = common_keys.union(set(wrapper.grouper.config.keys()))
        common_keys.remove("grouper")
        init_wrapper = wrappers[0]
        for i in range(1, len(wrappers)):
            wrapper = wrappers[i]
            for k in common_keys:
                if k not in kwargs:
                    same_k = True
                    try:
                        if k in wrapper.config:
                            if not checks.is_deep_equal(init_wrapper.config[k], wrapper.config[k]):
                                same_k = False
                        elif "grouper" not in kwargs and k in wrapper.grouper.config:
                            if not checks.is_deep_equal(init_wrapper.grouper.config[k], wrapper.grouper.config[k]):
                                same_k = False
                        else:
                            same_k = False
                    except KeyError as e:
                        same_k = False
                    if not same_k:
                        raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        for k in common_keys:
            if k not in kwargs:
                if k in init_wrapper.config:
                    kwargs[k] = init_wrapper.config[k]
                elif "grouper" not in kwargs and k in init_wrapper.grouper.config:
                    kwargs[k] = init_wrapper.grouper.config[k]
                else:
                    raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[ArrayWrapperT],
        *wrappers: tp.MaybeTuple[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        stack_columns: bool = True,
        stack_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along rows.

        Indexes will be concatenated in the order instances appear in `wrappers`.
        The merged index must have no duplicates or mixed data, and must be monotonically increasing.
        A custom index can be provided via `index`.

        Frequency must be the same across all indexes. A custom frequency can be provided via `freq`.

        If column levels in some instances differ, they will be stacked upon each other.
        Custom columns can be provided via `columns`.

        If `group_by` is None, all instances must be either grouped or not, and they must
        contain the same group values and labels.

        All instances must contain the same keys and values in their configs and configs of their
        grouper instances, apart from those arguments provided explicitly via `kwargs`."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")

        if index is None:
            new_index = None
            all_ranges = True
            name = None
            for wrapper in wrappers:
                if not checks.is_default_index(wrapper.index, check_names=False) or (
                    name is not None and wrapper.index != name
                ):
                    all_ranges = False
                    break
                if name is None:
                    name = wrapper.index.name
            if all_ranges:
                new_index = pd.RangeIndex(stop=sum([len(wrapper.index) for wrapper in wrappers]), name=name)
            else:
                for wrapper in wrappers:
                    if new_index is None:
                        new_index = wrapper.index
                    else:
                        if new_index.dtype != wrapper.index.dtype:
                            raise ValueError("Indexes to be merged must have the same data type")
                        new_index = new_index.append(wrapper.index)
            if new_index.has_duplicates:
                raise ValueError("Merged index contains duplicates")
            if not new_index.is_monotonic_increasing:
                raise ValueError("Merged index must be monotonically increasing")
            index = new_index
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            freq = infer_index_freq(index)
            if freq is None:
                new_freq = None
                for wrapper in wrappers:
                    if new_freq is None:
                        new_freq = wrapper.freq
                    else:
                        if new_freq is not None and wrapper.freq is not None and new_freq != wrapper.freq:
                            raise ValueError("Objects to be merged must have the same frequency")
                freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            new_columns = None
            for wrapper in wrappers:
                if new_columns is None:
                    new_columns = wrapper.columns
                else:
                    if not checks.is_index_equal(new_columns, wrapper.columns):
                        if not stack_columns:
                            raise ValueError("Objects to be merged must have the same columns")
                        new_columns = stack_indexes((new_columns, wrapper.columns), **resolve_dict(stack_kwargs))
            columns = new_columns
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                grouped = None
                for wrapper in wrappers:
                    wrapper_grouped = wrapper.grouper.is_grouped()
                    if grouped is None:
                        grouped = wrapper_grouped
                    else:
                        if grouped is not wrapper_grouped:
                            raise ValueError("Objects to be merged must be either grouped or not")
                if grouped:
                    new_group_by = None
                    for wrapper in wrappers:
                        wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
                        wrapper_group_by = wrapper_grouped_index[wrapper_groups]
                        if new_group_by is None:
                            new_group_by = wrapper_group_by
                        else:
                            if not checks.is_index_equal(new_group_by, wrapper_group_by):
                                raise ValueError("Objects to be merged must have the same groups")
                    group_by = new_group_by
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            ndim = None
            for wrapper in wrappers:
                if ndim is None or wrapper.ndim > 1:
                    ndim = wrapper.ndim
            kwargs["ndim"] = ndim

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    @classmethod
    def column_stack(
        cls: tp.Type[ArrayWrapperT],
        *wrappers: tp.MaybeTuple[ArrayWrapperT],
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        group_by: tp.GroupByLike = None,
        union_index: bool = True,
        normalize_columns: tp.Optional[bool] = None,
        normalize_groups: tp.Optional[bool] = None,
        normalize_locally: bool = True,
        keys: tp.Optional[tp.IndexLike] = None,
        stack_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> ArrayWrapperT:
        """Stack multiple `ArrayWrapper` instances along columns.

        If indexes are the same in each wrapper index, will use that index. If indexes differ and
        `union_index` is True, they will be merged into a single one by the set union operation.
        Otherwise, an error will be raised. The merged index must have no duplicates or mixed data,
        and must be monotonically increasing. A custom index can be provided via `index`.

        Frequency must be the same across all indexes. A custom frequency can be provided via `freq`.

        If columns level names are the same in each instance, will concatenate all columns.
        If column level names differ and `normalize_columns` is True, will create a new column index
        with the name `col_idx`. Otherwise, an error will be raised. If `normalize_locally` is True,
        will index unique columns under their respective objects, otherwise globally.

        If group level names differ and `normalize_groups` is True, applies the same approach as for columns.
        Otherwise, an error will be raised (also when some instances are grouped and some are not).

        If any of the instances has `column_only_select` being enabled, the final wrapper will also enable it.
        If any of the instances has `group_select` or other grouping-related flags being disabled, the final
        wrapper will also disable them.

        All instances must contain the same keys and values in their configs and configs of their
        grouper instances, apart from those arguments provided explicitly via `kwargs`."""
        if len(wrappers) == 1:
            wrappers = wrappers[0]
        wrappers = list(wrappers)
        for wrapper in wrappers:
            if not checks.is_instance_of(wrapper, ArrayWrapper):
                raise TypeError("Each object to be merged must be an instance of ArrayWrapper")
        if keys is not None and not isinstance(keys, pd.Index):
            keys = pd.Index(keys)

        for wrapper in wrappers:
            if wrapper.index.has_duplicates:
                raise ValueError("Index of some objects to be merged contains duplicates")
        if index is None:
            new_index = None
            for wrapper in wrappers:
                if new_index is None:
                    new_index = wrapper.index
                else:
                    if not checks.is_index_equal(new_index, wrapper.index):
                        if not union_index:
                            raise ValueError(
                                "Objects to be merged must have the same index. "
                                "Use index='union' to merge index as well."
                            )
                        else:
                            if new_index.dtype != wrapper.index.dtype:
                                raise ValueError("Indexes to be merged must have the same data type")
                            new_index = new_index.union(wrapper.index)
            if not new_index.is_monotonic_increasing:
                raise ValueError("Merged index must be monotonically increasing")
            index = new_index
        elif not isinstance(index, pd.Index):
            index = pd.Index(index)
        kwargs["index"] = index

        if freq is None:
            freq = infer_index_freq(index)
            if freq is None:
                new_freq = None
                for wrapper in wrappers:
                    if new_freq is None:
                        new_freq = wrapper.freq
                    else:
                        if new_freq is not None and wrapper.freq is not None and new_freq != wrapper.freq:
                            raise ValueError("Objects to be merged must have the same frequency")
                freq = new_freq
        kwargs["freq"] = freq

        if columns is None:
            normalize = False
            if normalize_columns is None or not normalize_columns:
                new_columns = None
                for wrapper in wrappers:
                    wrapper_columns = wrapper.columns
                    if new_columns is None:
                        new_columns = wrapper_columns
                    else:
                        if new_columns.names != wrapper_columns.names:
                            if normalize_columns is not None and not normalize_columns:
                                raise ValueError(
                                    "Objects to be merged must have the same column level names. "
                                    "Use columns='normalize' to normalize columns."
                                )
                            normalize = True
                            break
                        new_columns = new_columns.append(wrapper_columns)
            if normalize:
                new_columns = None
                if keys is not None and normalize_locally:
                    for wrapper in wrappers:
                        wrapper_columns = pd.RangeIndex(stop=len(wrapper.columns), name="col_idx")
                        if new_columns is None:
                            new_columns = wrapper_columns
                        else:
                            new_columns = new_columns.append(wrapper_columns)
                else:
                    total_column_count = sum([len(wrapper.columns) for wrapper in wrappers])
                    new_columns = pd.RangeIndex(stop=total_column_count, name="col_idx")
            elif keys is None:
                if (new_columns == 0).all():
                    new_columns = pd.RangeIndex(stop=len(new_columns))
            if keys is not None:
                top_columns = None
                for i, wrapper in enumerate(wrappers):
                    top_wrapper_columns = repeat_index(keys[[i]], len(wrapper.columns))
                    if top_columns is None:
                        top_columns = top_wrapper_columns
                    else:
                        top_columns = top_columns.append(top_wrapper_columns)
                new_columns = stack_indexes((top_columns, new_columns), **resolve_dict(stack_kwargs))
            columns = new_columns
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        kwargs["columns"] = columns

        if "grouper" in kwargs:
            if not checks.is_index_equal(columns, kwargs["grouper"].index):
                raise ValueError("Columns and grouper index must match")
            if group_by is not None:
                kwargs["group_by"] = group_by
        else:
            if group_by is None:
                any_grouped = False
                for wrapper in wrappers:
                    if wrapper.grouper.is_grouped():
                        any_grouped = True
                        break
                if any_grouped:
                    normalize = False
                    if normalize_groups is None or not normalize_groups:
                        new_group_by = None
                        for wrapper in wrappers:
                            wrapper_groups, wrapper_grouped_index = wrapper.grouper.get_groups_and_index()
                            wrapper_group_by = wrapper_grouped_index[wrapper_groups]
                            if new_group_by is None:
                                new_group_by = wrapper_group_by
                            else:
                                if len(new_group_by.intersection(wrapper_group_by)) > 0:
                                    if normalize_groups is not None and not normalize_groups:
                                        raise ValueError("Objects to be merged must must have compatible group labels")
                                    normalize = True
                                    break
                                if new_group_by.names != wrapper_group_by.names:
                                    if normalize_groups is not None and not normalize_groups:
                                        raise ValueError("Objects to be merged must have the same group level names")
                                    normalize = True
                                    break
                                new_group_by = new_group_by.append(wrapper_group_by)
                    if normalize:
                        new_group_by = None
                        if keys is not None and normalize_locally:
                            for i, wrapper in enumerate(wrappers):
                                wrapper_group_by = pd.Index(wrapper.grouper.get_groups(), name="group_idx")
                                if new_group_by is None:
                                    new_group_by = wrapper_group_by
                                else:
                                    new_group_by = new_group_by.append(wrapper_group_by)
                        else:
                            cnt_sum = 0
                            indices = []
                            for i, wrapper in enumerate(wrappers):
                                indices.append(wrapper.grouper.get_groups() + cnt_sum)
                                cnt_sum += wrapper.grouper.get_group_count()
                            new_group_by = pd.Index(np.concatenate(indices), name="group_idx")
                    if keys is not None:
                        top_group_by = None
                        for i, wrapper in enumerate(wrappers):
                            top_wrapper_group_by = repeat_index(keys[[i]], len(wrapper.columns))
                            if top_group_by is None:
                                top_group_by = top_wrapper_group_by
                            else:
                                top_group_by = top_group_by.append(top_wrapper_group_by)
                        new_group_by = stack_indexes((top_group_by, new_group_by), **resolve_dict(stack_kwargs))
                    group_by = new_group_by
                else:
                    group_by = False
            kwargs["group_by"] = group_by

        if "ndim" not in kwargs:
            kwargs["ndim"] = 2
        if "grouped_ndim" not in kwargs:
            kwargs["grouped_ndim"] = None
        if "column_only_select" not in kwargs:
            column_only_select = None
            for wrapper in wrappers:
                if column_only_select is None or wrapper.column_only_select:
                    column_only_select = wrapper.column_only_select
            kwargs["column_only_select"] = column_only_select
        if "range_only_select" not in kwargs:
            range_only_select = None
            for wrapper in wrappers:
                if range_only_select is None or wrapper.range_only_select:
                    range_only_select = wrapper.range_only_select
            kwargs["range_only_select"] = range_only_select
        if "group_select" not in kwargs:
            group_select = None
            for wrapper in wrappers:
                if group_select is None or not wrapper.group_select:
                    group_select = wrapper.group_select
            kwargs["group_select"] = group_select
        if "grouper" not in kwargs:
            if "allow_enable" not in kwargs:
                allow_enable = None
                for wrapper in wrappers:
                    if allow_enable is None or not wrapper.grouper.allow_enable:
                        allow_enable = wrapper.grouper.allow_enable
                kwargs["allow_enable"] = allow_enable
            if "allow_disable" not in kwargs:
                allow_disable = None
                for wrapper in wrappers:
                    if allow_disable is None or not wrapper.grouper.allow_disable:
                        allow_disable = wrapper.grouper.allow_disable
                kwargs["allow_disable"] = allow_disable
            if "allow_modify" not in kwargs:
                allow_modify = None
                for wrapper in wrappers:
                    if allow_modify is None or not wrapper.grouper.allow_modify:
                        allow_modify = wrapper.grouper.allow_modify
                kwargs["allow_modify"] = allow_modify

        return cls(**ArrayWrapper.resolve_stack_kwargs(*wrappers, **kwargs))

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "index",
        "columns",
        "ndim",
        "freq",
        "column_only_select",
        "range_only_select",
        "group_select",
        "grouped_ndim",
        "grouper",
    }

    def __init__(
        self,
        index: tp.IndexLike,
        columns: tp.IndexLike,
        ndim: tp.Optional[int] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        grouped_ndim: tp.Optional[int] = None,
        grouper: tp.Optional[Grouper] = None,
        **kwargs,
    ) -> None:

        checks.assert_not_none(index)
        checks.assert_not_none(columns)
        index = try_to_datetime_index(index)
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        if ndim is None:
            if len(columns) == 1 and not isinstance(columns, pd.MultiIndex):
                ndim = 1
            else:
                ndim = 2

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
            range_only_select=range_only_select,
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
        self._range_only_select = range_only_select
        self._group_select = group_select
        self._grouper = grouper
        self._grouped_ndim = grouped_ndim

    def indexing_func_meta(
        self: ArrayWrapperT,
        pd_indexing_func: tp.PandasIndexingFunc,
        index: tp.Optional[tp.IndexLike] = None,
        columns: tp.Optional[tp.IndexLike] = None,
        column_only_select: tp.Optional[bool] = None,
        range_only_select: tp.Optional[bool] = None,
        group_select: tp.Optional[bool] = None,
        return_slices: bool = True,
        return_none_slices: bool = True,
        group_by: tp.GroupByLike = None,
        wrapper_kwargs: tp.KwargsLike = None,
    ) -> dict:
        """Perform indexing on `ArrayWrapper` and also return metadata.

        Takes into account column grouping.

        Set `column_only_select` to True to index the array wrapper as a Series of columns/groups.
        This way, selection of index (axis 0) can be avoided. Set `range_only_select` to True to
        allow selection of rows only using slices. Set `group_select` to True to allow selection of groups.
        Otherwise, indexing is performed on columns, even if grouping is enabled. Takes effect only if
        grouping is enabled.

        Returns the new array wrapper, row indices, column indices, and group indices.
        If `return_slices` is True (default), indices will be returned as a slice if they were
        identified as a range. If `return_none_slices` is True (default), indices will be returned as a slice
        `(None, None, None)` if the axis hasn't been changed.

        !!! note
            If `column_only_select` is True, make sure to index the array wrapper
            as a Series of columns rather than a DataFrame. For example, the operation
            `.iloc[:, :2]` should become `.iloc[:2]`. Operations are not allowed if the
            object is already a Series and thus has only one column/group."""
        if column_only_select is None:
            column_only_select = self.column_only_select
        if range_only_select is None:
            range_only_select = self.range_only_select
        if group_select is None:
            group_select = self.group_select
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        _self = self.regroup(group_by)
        group_select = group_select and _self.grouper.is_grouped()
        if index is None:
            index = _self.index
        if not isinstance(index, pd.Index):
            index = pd.Index(index)
        if columns is None:
            if group_select:
                columns = _self.get_columns()
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

        def _resolve_arr(arr: tp.MaybeArray, n: int) -> tp.Union[tp.MaybeArray, slice]:
            if return_slices and checks.is_np_array(arr) and is_range(arr):
                if return_none_slices and arr[0] == 0 and arr[-1] == n - 1:
                    return slice(None, None, None), False
                return slice(arr[0], arr[-1] + 1, None), True
            if isinstance(arr, np.integer):
                return arr.item(), True
            return arr, True

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
            row_idxs = np.arange(len(index))
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
                row_idxs = idx_mapper.values
                col_idxs = 0
            else:
                col_mapper_values = np.broadcast_to(np.arange(n_cols), (n_rows, n_cols))
                col_mapper = pd_indexing_func(i_wrapper.wrap(col_mapper_values, index=index, columns=columns))
                if checks.is_frame(idx_mapper):
                    row_idxs = idx_mapper.values[:, 0]
                    col_idxs = col_mapper.values[0]
                elif checks.is_series(idx_mapper):
                    one_col = np.all(col_mapper.values == col_mapper.values.item(0))
                    one_idx = np.all(idx_mapper.values == idx_mapper.values.item(0))
                    if one_col and one_idx:
                        # One index and one column selected, multiple times
                        raise IndexingError("Must select at least two unique indices in one of both axes")
                    elif one_col:
                        # One column selected
                        row_idxs = idx_mapper.values
                        col_idxs = col_mapper.values[0]
                    elif one_idx:
                        # One index selected
                        row_idxs = idx_mapper.values[0]
                        col_idxs = col_mapper.values
                    else:
                        raise IndexingError
                else:
                    raise IndexingError("Selection of a scalar is not allowed")
            new_index = indexes.get_index(idx_mapper, 0)
            if not isinstance(row_idxs, np.ndarray):
                # One index selected
                new_columns = index[[row_idxs]]
            elif not isinstance(col_idxs, np.ndarray):
                # One column selected
                new_columns = columns[[col_idxs]]
            else:
                new_columns = indexes.get_index(idx_mapper, 1)
            new_ndim = idx_mapper.ndim

        if _self.grouper.is_grouped():
            # Grouping enabled
            if np.asarray(row_idxs).ndim == 0:
                raise IndexingError("Flipping index and columns is not allowed")

            if group_select:
                # Selection based on groups
                # Get indices of columns corresponding to selected groups
                group_idxs = col_idxs
                col_idxs, new_groups = _self.grouper.select_groups(group_idxs)
                ungrouped_columns = _self.columns[col_idxs]
                if new_ndim == 1 and len(ungrouped_columns) == 1:
                    ungrouped_ndim = 1
                    col_idxs = col_idxs[0]
                else:
                    ungrouped_ndim = 2

                row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
                if range_only_select and rows_changed:
                    if not isinstance(row_idxs, slice):
                        raise ValueError("Rows can be selected only by slicing")
                    if row_idxs.step not in (1, None):
                        raise ValueError("Slice for selecting rows must have a step of 1 or None")
                col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
                group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
                return dict(
                    new_wrapper=_self.replace(
                        **merge_dicts(
                            dict(
                                index=new_index,
                                columns=ungrouped_columns,
                                ndim=ungrouped_ndim,
                                grouped_ndim=new_ndim,
                                group_by=new_columns[new_groups],
                            ),
                            wrapper_kwargs,
                        )
                    ),
                    row_idxs=row_idxs,
                    rows_changed=rows_changed,
                    col_idxs=col_idxs,
                    columns_changed=columns_changed,
                    group_idxs=group_idxs,
                    groups_changed=groups_changed,
                )

            # Selection based on columns
            group_idxs = _self.grouper.get_groups()[col_idxs]
            new_group_by = _self.grouper.group_by[reshaping.to_1d_array(col_idxs)]
            row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
            if range_only_select and rows_changed:
                if not isinstance(row_idxs, slice):
                    raise ValueError("Rows can be selected only by slicing")
                if row_idxs.step not in (1, None):
                    raise ValueError("Slice for selecting rows must have a step of 1 or None")
            col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
            group_idxs, groups_changed = _resolve_arr(group_idxs, _self.get_shape_2d()[1])
            return dict(
                new_wrapper=_self.replace(
                    **merge_dicts(
                        dict(
                            index=new_index,
                            columns=new_columns,
                            ndim=new_ndim,
                            grouped_ndim=None,
                            group_by=new_group_by,
                        ),
                        wrapper_kwargs,
                    )
                ),
                row_idxs=row_idxs,
                rows_changed=rows_changed,
                col_idxs=col_idxs,
                columns_changed=columns_changed,
                group_idxs=group_idxs,
                groups_changed=groups_changed,
            )

        # Grouping disabled
        row_idxs, rows_changed = _resolve_arr(row_idxs, _self.shape[0])
        if range_only_select and rows_changed:
            if not isinstance(row_idxs, slice):
                raise ValueError("Rows can be selected only by slicing")
            if row_idxs.step not in (1, None):
                raise ValueError("Slice for selecting rows must have a step of 1 or None")
        col_idxs, columns_changed = _resolve_arr(col_idxs, _self.shape_2d[1])
        return dict(
            new_wrapper=_self.replace(
                **merge_dicts(
                    dict(
                        index=new_index,
                        columns=new_columns,
                        ndim=new_ndim,
                        grouped_ndim=None,
                        group_by=None,
                    ),
                    wrapper_kwargs,
                )
            ),
            row_idxs=row_idxs,
            rows_changed=rows_changed,
            col_idxs=col_idxs,
            columns_changed=columns_changed,
            group_idxs=col_idxs,
            groups_changed=columns_changed,
        )

    def indexing_func(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform indexing on `ArrayWrapper`"""
        return self.indexing_func_meta(*args, **kwargs)["new_wrapper"]

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

    def resample_meta(self: ArrayWrapperT, *args, wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> dict:
        """Perform resampling on `ArrayWrapper` and also return metadata.

        `*args` and `**kwargs` are passed to `ArrayWrapper.create_resampler`."""
        resampler = self.create_resampler(*args, **kwargs)
        if isinstance(resampler, Resampler):
            _resampler = resampler
        else:
            _resampler = Resampler.from_pd_resampler(resampler)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if "index" not in wrapper_kwargs:
            wrapper_kwargs["index"] = _resampler.target_index
        if "freq" not in wrapper_kwargs:
            wrapper_kwargs["freq"] = infer_index_freq(wrapper_kwargs["index"])
        new_wrapper = self.replace(**wrapper_kwargs)
        return dict(resampler=resampler, new_wrapper=new_wrapper)

    def resample(self: ArrayWrapperT, *args, **kwargs) -> ArrayWrapperT:
        """Perform resampling on `ArrayWrapper`.

        Uses `ArrayWrapper.resample_meta`."""
        return self.resample_meta(*args, **kwargs)["new_wrapper"]

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

    def get_freq(
        self,
        freq: tp.Optional[tp.FrequencyLike] = None,
        **kwargs,
    ) -> tp.Union[None, float, tp.PandasFrequency]:
        """Index frequency as `pd.Timedelta` or None if it cannot be converted."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        if freq is None:
            freq = self._freq
        if freq is None:
            freq = wrapping_cfg["freq"]
        try:
            return infer_index_freq(self.index, freq=freq, **kwargs)
        except Exception as e:
            return None

    @property
    def freq(self) -> tp.Optional[pd.Timedelta]:
        """`ArrayWrapper.get_freq` with date offsets and integer frequencies not allowed."""
        return self.get_freq(allow_date_offset=False, allow_numeric=False)

    @property
    def any_freq(self) -> tp.Union[None, float, tp.PandasFrequency]:
        """Index frequency of any type."""
        return self.get_freq()

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
    def range_only_select(self) -> tp.Optional[bool]:
        """Whether to perform indexing on rows using slices only."""
        from vectorbtpro._settings import settings

        wrapping_cfg = settings["wrapping"]

        range_only_select = self._range_only_select
        if range_only_select is None:
            range_only_select = wrapping_cfg["range_only_select"]
        return range_only_select

    @property
    def group_select(self) -> tp.Optional[bool]:
        """Whether to allow indexing on groups."""
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

    @cached_method(whitelist=True)
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
        if len(kwargs) > 0:
            return self.replace(**kwargs)
        return self  # important for keeping cache

    @cached_method(whitelist=True)
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
                by = to_offset(by)
                if by.n == 1:
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
            orig_arr = arr
            arr = np.asarray(arr)
            if fillna is not None:
                arr[pd.isnull(arr)] = fillna
            shape_2d = (arr.shape[0] if arr.ndim > 0 else 1, arr.shape[1] if arr.ndim > 1 else 1)
            target_shape_2d = (len(index), len(columns))
            if shape_2d != target_shape_2d:
                if isinstance(orig_arr, (pd.Series, pd.DataFrame)):
                    arr = reshaping.align_pd_arrays(orig_arr, to_index=index, to_columns=columns).values
                if isinstance(orig_arr, pd.Series) and arr.ndim == 1:
                    arr = arr[:, None]
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

    def row_stack_and_wrap(self, *objs: tp.ArrayLike, group_by: tp.GroupByLike = None, **kwargs) -> tp.SeriesFrame:
        """Stack objects along rows and wrap the final object."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        new_objs = []
        for obj in objs:
            obj = reshaping.to_2d_array(obj)
            if obj.shape[1] != _self.shape_2d[1]:
                if obj.shape[1] != 1:
                    raise ValueError(f"Cannot broadcast {obj.shape[1]} to {_self.shape_2d[1]} columns")
                obj = np.repeat(obj, _self.shape_2d[1], axis=1)
            new_objs.append(obj)
        stacked_obj = np.row_stack(new_objs)
        return _self.wrap(stacked_obj, **kwargs)

    def column_stack_and_wrap(
        self,
        *objs: tp.ArrayLike,
        reindex_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Stack objects along columns and wrap the final object.

        `reindex_kwargs` will be passed to
        [pandas.DataFrame.reindex](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html)."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        new_objs = []
        for obj in objs:
            if not checks.is_index_equal(obj.index, _self.index, check_names=False):
                was_bool = (isinstance(obj, pd.Series) and obj.dtype == "bool") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "bool").all()
                )
                obj = obj.reindex(_self.index, **resolve_dict(reindex_kwargs))
                is_object = (isinstance(obj, pd.Series) and obj.dtype == "object") or (
                    isinstance(obj, pd.DataFrame) and (obj.dtypes == "object").all()
                )
                if was_bool and is_object:
                    obj = obj.astype(None)
            new_objs.append(reshaping.to_2d_array(obj))
        stacked_obj = np.column_stack(new_objs)
        return _self.wrap(stacked_obj, **kwargs)

    def column_stack_and_wrap_reduced(
        self,
        *objs: tp.ArrayLike,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Stack reduced objects along columns and wrap the final object."""
        _self = self.resolve(group_by=group_by)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        new_objs = []
        for obj in objs:
            new_objs.append(reshaping.to_1d_array(obj))
        stacked_obj = np.concatenate(new_objs)
        return _self.wrap_reduced(stacked_obj, **kwargs)

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

    def get_index_points(self, *args, **kwargs) -> tp.Array1d:
        """See `vectorbtpro.base.indexing.get_index_points`.

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
        return get_index_points(self.index, *args, **kwargs)

    def get_index_ranges(self, *args, **kwargs) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """See `vectorbtpro.base.indexing.get_index_ranges`.

        Usage:
            * Provide nothing to generate one largest index range:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import numpy as np

            >>> data = vbt.YFData.fetch("BTC-USD", start="2020-01-01", end="2020-02-01")

            >>> np.column_stack(data.wrapper.get_index_ranges())
            array([[ 0, 32]])
            ```

            * Provide `every` as an integer frequency to generate index ranges using NumPy:

            ```pycon
            >>> # Generate a range every five rows
            >>> np.column_stack(data.wrapper.get_index_ranges(every=5))
            array([[ 0,  5],
                   [ 5, 10],
                   [10, 15],
                   [15, 20],
                   [20, 25],
                   [25, 30]])

            >>> # Generate a range every five rows, starting at 6th row
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every=5,
            ...     start=5
            ... ))
            array([[ 5, 10],
                   [10, 15],
                   [15, 20],
                   [20, 25],
                   [25, 30]])

            >>> # Generate a range every five rows from 6th to 16th row
            >>> np.column_stack(data.wrapper.get_index_ranges(
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
            >>> np.column_stack(data.wrapper.get_index_ranges(every="W"))
            array([[ 5, 12],
                   [12, 19],
                   [19, 26]])

            >>> # Generate a range every second day of the week
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     add_start_delta="2d"
            ... ))
            array([[ 7, 12],
                   [14, 19],
                   [21, 26]])

            >>> # Generate a range every week, starting at 11th row
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     start=10
            ... ))
            array([[12, 19],
                   [19, 26]])

            >>> # Generate a range every week, starting exactly at 11th row
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     start=10,
            ...     exact_start=True
            ... ))
            array([[10, 12],
                   [12, 19],
                   [19, 26]])

            >>> # Generate a range every week, starting at 2020-01-10
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     start="2020-01-10"
            ... ))
            array([[12, 19],
                   [19, 26]])

            >>> # Generate a range every week, each starting at 2020-01-10
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     start="2020-01-10",
            ...     fixed_start=True
            ... ))
            array([[12, 19],
                   [12, 26]])
            ```

            * Use a look-back period (instead of an end index):

            ```pycon
            >>> # Generate a range every week, looking 5 days back
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     lookback_period=5
            ... ))
            array([[ 0,  5],
                   [ 7, 12],
                   [14, 19],
                   [21, 26]])

            >>> # Generate a range every week, looking 2 weeks back
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     every="W",
            ...     lookback_period="2W"
            ... ))
            array([[ 0, 12],
                   [ 5, 19],
                   [12, 26]])
            ```

            * Instead of using `every`, provide start and end indices explicitly:

            ```pycon
            >>> # Generate one range
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     start="2020-01-01",
            ...     end="2020-01-07"
            ... ))
            array([[1, 7]])

            >>> # Generate ranges between multiple dates
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     start=["2020-01-01", "2020-01-07"],
            ...     end=["2020-01-07", "2020-01-14"]
            ... ))
            array([[ 1,  7],
                   [ 7, 14]])

            >>> # Generate ranges with a fixed start
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     start="2020-01-01",
            ...     end=["2020-01-07", "2020-01-14"]
            ... ))
            array([[ 1,  7],
                   [ 1, 14]])
            ```

            * Use `closed_start` and `closed_end` to exclude any of the bounds:

            ```pycon
            >>> # Generate ranges between multiple dates
            >>> # by excluding the start date and including the end date
            >>> np.column_stack(data.wrapper.get_index_ranges(
            ...     start=["2020-01-01", "2020-01-07"],
            ...     end=["2020-01-07", "2020-01-14"],
            ...     closed_start=False,
            ...     closed_end=True
            ... ))
            array([[ 2,  8],
                   [ 8, 15]])
            ```
        """
        return get_index_ranges(self.index, self.any_freq, *args, **kwargs)

    def fill_using_index_dict(
        self,
        index_dct: index_dict,
        fill_value: tp.Scalar = np.nan,
        keep_flex: bool = False,
    ) -> tp.AnyArray:
        """Fill a new array using an index dictionary.

        Goes through each key acting as an indexer and puts its value at that index.
        Resolves an indexer using `vectorbtpro.base.indexing.get_indices`. Values can
        be scalars, arrays, and templates. Setting takes place on NumPy arrays, wrapping
        is done on the final object.

        If `to_pd` is True, will return a Pandas object, otherwise a NumPy array.
        If `keep_flex` is True, will return the most memory-efficient array representation
        capable of flexible indexing.

        Usage:
            * Set a single row:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd
            >>> import numpy as np

            >>> index = pd.date_range("2020", periods=5)
            >>> columns = pd.Index(["a", "b", "c"])
            >>> wrapper = vbt.ArrayWrapper(index, columns)

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     "2020-01-02": 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     1: 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     1: [1, 2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  NaN  NaN  NaN
            2020-01-05  NaN  NaN  NaN
            ```

            * Set multiple rows:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [2, 3]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  3.0  3.0  3.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     ("2020-01-02", "2020-01-04"): [[1, 2, 3]]
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  1.0  2.0  3.0
            2020-01-05  NaN  NaN  NaN
            ```

            * Set rows using slices:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.hslice("2020-01-02", "2020-01-04"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  2.0  2.0  2.0
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1], [2]]
            ... }))
                          a    b    c
            2020-01-01  1.0  1.0  1.0
            2020-01-02  1.0  1.0  1.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  2.0  2.0  2.0

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     ((0, 2), (3, 5)): [[1, 2, 3], [4, 5, 6]]
            ... }))
                          a    b    c
            2020-01-01  1.0  2.0  3.0
            2020-01-02  1.0  2.0  3.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  4.0  5.0  6.0
            2020-01-05  4.0  5.0  6.0
            ```

            All the above indexers can be wrapped with `vectorbtpro.base.indexing.RowIdx`.
            If the index is integer-like when querying an integer position, set `is_labels` to False.

            * Set rows using index points:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.RowPoints(every="2D"): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  NaN  NaN  NaN
            2020-01-05  2.0  2.0  2.0
            ```

            * Set rows using index ranges:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.RowRanges(
            ...         start=("2020-01-01", "2020-01-03"),
            ...         end=("2020-01-02", "2020-01-05")
            ...     ): 2
            ... }))
                          a    b    c
            2020-01-01  2.0  2.0  2.0
            2020-01-02  NaN  NaN  NaN
            2020-01-03  2.0  2.0  2.0
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```

            * Set column indices:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ColIdx("a"): 2
            ... }))
                          a   b   c
            2020-01-01  2.0 NaN NaN
            2020-01-02  2.0 NaN NaN
            2020-01-03  2.0 NaN NaN
            2020-01-04  2.0 NaN NaN
            2020-01-05  2.0 NaN NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ColIdx(("a", "b")): [1, 2]
            ... }))
                          a    b   c
            2020-01-01  1.0  2.0 NaN
            2020-01-02  1.0  2.0 NaN
            2020-01-03  1.0  2.0 NaN
            2020-01-04  1.0  2.0 NaN
            2020-01-05  1.0  2.0 NaN

            >>> multi_columns = pd.MultiIndex.from_arrays(
            ...     [["a", "a", "b", "b"], [1, 2, 1, 2]],
            ...     names=["c1", "c2"]
            ... )
            >>> multi_wrapper = vbt.ArrayWrapper(index, multi_columns)

            >>> multi_wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ColIdx(("a", 2)): 2
            ... }))
            c1           a        b
            c2           1    2   1   2
            2020-01-01 NaN  2.0 NaN NaN
            2020-01-02 NaN  2.0 NaN NaN
            2020-01-03 NaN  2.0 NaN NaN
            2020-01-04 NaN  2.0 NaN NaN
            2020-01-05 NaN  2.0 NaN NaN

            >>> multi_wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ColIdx("b", level="c1"): [3, 4]
            ... }))
            c1           a        b
            c2           1   2    1    2
            2020-01-01 NaN NaN  3.0  4.0
            2020-01-02 NaN NaN  3.0  4.0
            2020-01-03 NaN NaN  3.0  4.0
            2020-01-04 NaN NaN  3.0  4.0
            2020-01-05 NaN NaN  3.0  4.0
            ```

            * Set element indices:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ElemIdx(2, 2): 2
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  NaN
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ElemIdx(("2020-01-01", "2020-01-03"), 2): [1, 2]
            ... }))
                         a   b    c
            2020-01-01 NaN NaN  1.0
            2020-01-02 NaN NaN  NaN
            2020-01-03 NaN NaN  2.0
            2020-01-04 NaN NaN  NaN
            2020-01-05 NaN NaN  NaN

            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ElemIdx(("2020-01-01", "2020-01-03"), (0, 2)): [[1, 2], [3, 4]]
            ... }))
                          a   b    c
            2020-01-01  1.0 NaN  2.0
            2020-01-02  NaN NaN  NaN
            2020-01-03  3.0 NaN  4.0
            2020-01-04  NaN NaN  NaN
            2020-01-05  NaN NaN  NaN

            >>> multi_wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ElemIdx(
            ...         vbt.RowPoints(every="2d"),
            ...         vbt.ColIdx(1, level="c2")
            ...     ): [1, 2]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  2.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  1.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  1.0 NaN  2.0 NaN

            >>> multi_wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.ElemIdx(
            ...         vbt.RowPoints(every="2d"),
            ...         vbt.ColIdx(1, level="c2")
            ...     ): [[1], [2], [3]]
            ... }))
            c1            a        b
            c2            1   2    1   2
            2020-01-01  1.0 NaN  1.0 NaN
            2020-01-02  NaN NaN  NaN NaN
            2020-01-03  2.0 NaN  2.0 NaN
            2020-01-04  NaN NaN  NaN NaN
            2020-01-05  3.0 NaN  3.0 NaN
            ```

            * Set rows using a template:

            ```pycon
            >>> wrapper.fill_using_index_dict(vbt.index_dict({
            ...     vbt.RepEval("index.day % 2 == 0"): 2
            ... }))
                          a    b    c
            2020-01-01  NaN  NaN  NaN
            2020-01-02  2.0  2.0  2.0
            2020-01-03  NaN  NaN  NaN
            2020-01-04  2.0  2.0  2.0
            2020-01-05  NaN  NaN  NaN
            ```
        """
        changed_rows = False
        changed_cols = False
        set_ops = []
        if "_default" in index_dct:
            fill_value = index_dct["_default"]
        for indexer, set_v in index_dct.items():
            if indexer == "_default":
                continue
            row_indices, col_indices = get_indices(
                self.index,
                self.columns,
                indexer,
                check_indices=True,
            )
            if isinstance(set_v, CustomTemplate):
                context = dict(
                    wrapper=self,
                    keep_flex=keep_flex,
                    fill_value=fill_value,
                    indexer=indexer,
                    row_indices=row_indices,
                    col_indices=col_indices,
                )
                set_v = set_v.substitute(context, sub_id="fill_using_index_dict")

            def _check_use_indices(indices, _indexer=indexer):
                use_indices = True
                if isinstance(indices, (slice, hslice)):
                    if indices.start is None and indices.stop is None:
                        use_indices = False
                if isinstance(indices, tuple):
                    if all(_indices.size == 0 for _indices in indices):
                        use_indices = False
                if isinstance(indices, np.ndarray):
                    if indices.size == 0:
                        use_indices = False
                return use_indices

            use_row_indices = _check_use_indices(row_indices)
            use_col_indices = _check_use_indices(col_indices)
            set_v = np.asarray(set_v)

            if use_row_indices and use_col_indices:

                def _set_op(x, y=row_indices, z=col_indices, v=set_v):
                    if isinstance(y, tuple):
                        if np.isscalar(z):
                            v = np.broadcast_to(v, (len(y[0]),))
                        else:
                            if isinstance(z, slice):
                                z = np.arange(x.shape[1])[z]
                            v = np.broadcast_to(v, (len(y[0]), len(z)))
                        for j in range(len(y[0])):
                            if len(y) == 2:
                                x[y[0][j] : y[1][j], z] = v[j]
                            else:
                                x[y[0][j] : y[1][j] : y[2][j], z] = v[j]
                    else:
                        if np.isscalar(y) or np.isscalar(z):
                            x[y, z] = v
                        else:
                            if isinstance(y, slice):
                                y = np.arange(x.shape[1])[y]
                            if isinstance(z, slice):
                                z = np.arange(x.shape[1])[z]
                            _y = np.repeat(y, len(z))
                            _z = np.tile(z, len(y))
                            if np.isscalar(v):
                                x[_y, _z] = v
                            else:
                                v = np.broadcast_to(v, (len(y), len(z)))
                                x[_y, _z] = v.flatten()

                set_ops.append(_set_op)
                changed_rows = True
                changed_cols = True
            elif use_col_indices:

                def _set_op(x, z=col_indices, v=set_v):
                    x[:, z] = v

                set_ops.append(_set_op)
                if isinstance(col_indices, (int, np.integer)):
                    if set_v.size > 1:
                        changed_rows = True
                else:
                    if set_v.ndim == 2:
                        if set_v.shape[0] > 1:
                            changed_rows = True
                changed_cols = True
            else:

                def _set_op(x, y=row_indices, v=set_v):
                    if isinstance(y, tuple):
                        if x.ndim == 2:
                            v = np.broadcast_to(v, (len(y[0]), x.shape[1]))
                        else:
                            v = np.broadcast_to(v, (len(y[0]),))
                        for j in range(len(y[0])):
                            if len(y) == 2:
                                x[y[0][j] : y[1][j]] = v[j]
                            else:
                                x[y[0][j] : y[1][j] : y[2][j]] = v[j]
                    else:
                        if x.ndim == 2:
                            if not np.isscalar(y):
                                if v.ndim == 1 and v.size > 1:
                                    v = v[:, None]
                        x[y] = v

                set_ops.append(_set_op)
                if use_row_indices:
                    changed_rows = True
                if self.ndim == 2:
                    if isinstance(row_indices, (int, np.integer)):
                        if set_v.size > 1:
                            changed_cols = True
                    else:
                        if set_v.ndim == 2:
                            if set_v.shape[1] > 1:
                                changed_cols = True

        if keep_flex and not changed_cols and not changed_rows:
            new_obj = np.full((1,) if len(self.shape) == 1 else (1, 1), fill_value)
        elif keep_flex and not changed_cols:
            new_obj = np.full(self.shape if len(self.shape) == 1 else (self.shape[0], 1), fill_value)
        elif keep_flex and not changed_rows:
            new_obj = np.full((1, self.shape[1]), fill_value)
        else:
            new_obj = np.full(self.shape, fill_value)
        for set_op in set_ops:
            set_op(new_obj)

        if not keep_flex:
            new_obj = self.wrap(new_obj, group_by=False)
        return new_obj


WrappingT = tp.TypeVar("WrappingT", bound="Wrapping")


class Wrapping(Configured, PandasIndexer, AttrResolverMixin):
    """Class that uses `ArrayWrapper` globally."""

    @classmethod
    def resolve_row_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along rows."""
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking along columns."""
        return kwargs

    @classmethod
    def resolve_stack_kwargs(cls, *wrappings: tp.MaybeTuple[WrappingT], **kwargs) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Wrapping` after stacking.

        Should be called after `Wrapping.resolve_row_stack_kwargs` or `Wrapping.resolve_column_stack_kwargs`."""
        if len(wrappings) == 1:
            wrappings = wrappings[0]
        wrappings = list(wrappings)

        common_keys = set()
        for wrapping in wrappings:
            common_keys = common_keys.union(set(wrapping.config.keys()))
        init_wrapping = wrappings[0]
        for i in range(1, len(wrappings)):
            wrapping = wrappings[i]
            for k in common_keys:
                if k not in kwargs:
                    same_k = True
                    try:
                        if k in wrapping.config:
                            if not checks.is_deep_equal(init_wrapping.config[k], wrapping.config[k]):
                                same_k = False
                        else:
                            same_k = False
                    except KeyError as e:
                        same_k = False
                    if not same_k:
                        raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        for k in common_keys:
            if k not in kwargs:
                if k in init_wrapping.config:
                    kwargs[k] = init_wrapping.config[k]
                else:
                    raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
        return kwargs

    @classmethod
    def row_stack(cls: tp.Type[WrappingT], *args: tp.MaybeTuple[WrappingT], **kwargs) -> WrappingT:
        """Stack multiple `Wrapping` instances along rows.

        Should use `ArrayWrapper.row_stack`."""
        raise NotImplementedError

    @classmethod
    def column_stack(cls: tp.Type[WrappingT], *args: tp.MaybeTuple[WrappingT], **kwargs) -> WrappingT:
        """Stack multiple `Wrapping` instances along columns.

        Should use `ArrayWrapper.column_stack`."""
        raise NotImplementedError

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "wrapper",
    }

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
            range_only_select=self.range_only_select,
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
    def range_only_select(self) -> tp.Optional[bool]:
        """Overrides `ArrayWrapper.range_only_select`."""
        range_only_select = getattr(self, "_range_only_select", None)
        if range_only_select is None:
            return self.wrapper.range_only_select
        return range_only_select

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
