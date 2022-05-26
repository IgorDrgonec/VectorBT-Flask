# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Custom pandas accessors for base operations with pandas objects."""

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from vectorbtpro import _typing as tp
from vectorbtpro.base import combining, reshaping, indexes
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import class_or_instanceproperty, class_or_instancemethod
from vectorbtpro.utils.magic_decorators import attach_binary_magic_methods, attach_unary_magic_methods
from vectorbtpro.utils.parsing import get_func_arg_names, get_expr_var_names, get_context_vars
from vectorbtpro.utils.template import deep_substitute
from vectorbtpro.utils.datetime_ import freq_to_timedelta

BaseAccessorT = tp.TypeVar("BaseAccessorT", bound="BaseAccessor")


@attach_binary_magic_methods(lambda self, other, np_func: self.combine(other, combine_func=np_func))
@attach_unary_magic_methods(lambda self, np_func: self.apply(apply_func=np_func))
class BaseAccessor(Wrapping):
    """Accessor on top of Series and DataFrames.

    Accessible via `pd.Series.vbt` and `pd.DataFrame.vbt`, and all child accessors.

    Series is just a DataFrame with one column, hence to avoid defining methods exclusively for 1-dim data,
    we will convert any Series to a DataFrame and perform matrix computation on it. Afterwards,
    by using `BaseAccessor.wrapper`, we will convert the 2-dim output back to a Series.

    `**kwargs` will be passed to `vectorbtpro.base.wrapping.ArrayWrapper`.

    !!! note
        When using magic methods, ensure that `.vbt` is called on the operand on the left
        if the other operand is an array.

        Accessors do not utilize caching.

        Grouping is only supported by the methods that accept the `group_by` argument.

    Usage:
        * Build a symmetric matrix:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> import vectorbtpro as vbt

        >>> # vectorbtpro.base.accessors.BaseAccessor.make_symmetric
        >>> pd.Series([1, 2, 3]).vbt.make_symmetric()
             0    1    2
        0  1.0  2.0  3.0
        1  2.0  NaN  NaN
        2  3.0  NaN  NaN
        ```

        * Broadcast pandas objects:

        ```pycon
        >>> sr = pd.Series([1])
        >>> df = pd.DataFrame([1, 2, 3])

        >>> vbt.base.reshaping.broadcast_to(sr, df)
           0
        0  1
        1  1
        2  1

        >>> sr.vbt.broadcast_to(df)
           0
        0  1
        1  1
        2  1
        ```

        * Many methods such as `BaseAccessor.broadcast` are both class and instance methods:

        ```pycon
        >>> from vectorbtpro.base.accessors import BaseAccessor

        >>> # Same as sr.vbt.broadcast(df)
        >>> new_sr, new_df = BaseAccessor.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        * Instead of explicitly importing `BaseAccessor` or any other accessor, we can use `pd_acc` instead:

        ```pycon
        >>> vbt.pd_acc.broadcast(sr, df)
        >>> new_sr
           0
        0  1
        1  1
        2  1
        >>> new_df
           0
        0  1
        1  2
        2  3
        ```

        * `BaseAccessor` implements arithmetic (such as `+`), comparison (such as `>`) and
        logical operators (such as `&`) by forwarding the operation to `BaseAccessor.combine`:

        ```pycon
        >>> sr.vbt + df
           0
        0  2
        1  3
        2  4
        ```

        Many interesting use cases can be implemented this way.

        * For example, let's compare an array with 3 different thresholds
        (index is treated as multiple values by `BaseAccessor.combine`):

        ```pycon
        >>> df.vbt > pd.Index(np.arange(3), name='threshold')
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```

        * The same using the broadcasting mechanism:

        ```pycon
        >>> df.vbt > vbt.BCO(np.arange(3), name='threshold', product=True)
        threshold     0                  1                  2
                     a2    b2    c2     a2    b2    c2     a2     b2    c2
        x2         True  True  True  False  True  True  False  False  True
        y2         True  True  True   True  True  True   True   True  True
        z2         True  True  True   True  True  True   True   True  True
        ```
    """

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `BaseAccessor` after stacking along rows."""
        wrapper_kwargs, kwargs = ArrayWrapper.extract_init_kwargs(**kwargs)
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper
        if "obj" not in kwargs:
            kwargs["obj"] = wrapper.row_stack_and_wrap(*[obj.obj for obj in objs], group_by=False)
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `BaseAccessor` after stacking along columns."""
        wrapper_kwargs, kwargs = ArrayWrapper.extract_init_kwargs(**kwargs)
        if "wrapper" in kwargs and kwargs["wrapper"] is not None:
            wrapper = kwargs["wrapper"]
            if len(wrapper_kwargs) > 0:
                wrapper = wrapper.replace(**wrapper_kwargs)
        else:
            wrapper = ArrayWrapper.column_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)
        kwargs["wrapper"] = wrapper
        if "obj" not in kwargs:
            kwargs["obj"] = wrapper.column_stack_and_wrap(
                *[obj.obj for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
            )
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        if checks.is_series(kwargs["obj"]):
            return cls.sr_accessor_cls(**kwargs)
        return cls.df_accessor_cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[BaseAccessorT],
        *objs: tp.MaybeTuple[BaseAccessorT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> BaseAccessorT:
        """Stack multiple `BaseAccessor` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, BaseAccessor):
                raise TypeError("Each object to be merged must be an instance of BaseAccessor")
        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls.df_accessor_cls(**kwargs)

    def indexing_func(self: BaseAccessorT, *args, **kwargs) -> BaseAccessorT:
        """Perform indexing on `BaseAccessor`."""
        wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_obj = wrapper_meta["new_wrapper"].wrap(
            self.to_2d_array()[wrapper_meta["row_idxs"], :][:, wrapper_meta["col_idxs"]],
            group_by=False,
        )
        if checks.is_series(new_obj):
            return self.replace(cls_=self.sr_accessor_cls, obj=new_obj, wrapper=wrapper_meta["new_wrapper"])
        return self.replace(cls_=self.df_accessor_cls, obj=new_obj, wrapper=wrapper_meta["new_wrapper"])

    def __init__(self, obj: tp.SeriesFrame, wrapper: tp.Optional[ArrayWrapper] = None, **kwargs) -> None:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame))

        wrapper_kwargs, kwargs = ArrayWrapper.extract_init_kwargs(**kwargs)
        if wrapper is None:
            wrapper = ArrayWrapper.from_obj(obj, **wrapper_kwargs)
        elif len(wrapper_kwargs) > 0:
            wrapper = wrapper.replace(**wrapper_kwargs)

        Wrapping.__init__(self, wrapper, obj=obj, **kwargs)

        self._obj = obj

    def __call__(self: BaseAccessorT, **kwargs) -> BaseAccessorT:
        """Allows passing arguments to the initializer."""

        return self.replace(**kwargs)

    @class_or_instanceproperty
    def sr_accessor_cls(cls_or_self) -> tp.Type["BaseSRAccessor"]:
        """Accessor class for `pd.Series`."""
        return BaseSRAccessor

    @class_or_instanceproperty
    def df_accessor_cls(cls_or_self) -> tp.Type["BaseDFAccessor"]:
        """Accessor class for `pd.DataFrame`."""
        return BaseDFAccessor

    @property
    def obj(self):
        """Pandas object."""
        return self._obj

    @class_or_instanceproperty
    def ndim(cls_or_self) -> tp.Optional[int]:
        if isinstance(cls_or_self, type):
            return None
        return cls_or_self.obj.ndim

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.Series)

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        if isinstance(cls_or_self, type):
            raise NotImplementedError
        return isinstance(cls_or_self.obj, pd.DataFrame)

    @classmethod
    def resolve_shape(cls, shape: tp.ShapeLike) -> tp.Shape:
        """Resolve shape."""
        shape_2d = reshaping.shape_to_2d(shape)
        try:
            if cls.is_series() and shape_2d[1] > 1:
                raise ValueError("Use DataFrame accessor")
        except NotImplementedError:
            pass
        return shape_2d

    # ############# Creation ############# #

    @classmethod
    def empty(cls, shape: tp.Shape, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """Generate an empty Series/DataFrame of shape `shape` and fill with `fill_value`."""
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            return pd.Series(np.full(shape, fill_value), **kwargs)
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(cls, other: tp.SeriesFrame, fill_value: tp.Scalar = np.nan, **kwargs) -> tp.SeriesFrame:
        """Generate an empty Series/DataFrame like `other` and fill with `fill_value`."""
        if checks.is_series(other):
            return cls.empty(other.shape, fill_value=fill_value, index=other.index, name=other.name, **kwargs)
        return cls.empty(other.shape, fill_value=fill_value, index=other.index, columns=other.columns, **kwargs)

    # ############# Indexes ############# #

    def apply_on_index(
        self,
        apply_func: tp.Callable,
        *args,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Apply function `apply_func` on index of the pandas object.

        Set `axis` to 1 for columns, 0 for index, and None to determine automatically.
        Set `copy_data` to True to make a deep copy of the data."""
        if axis is None:
            axis = 0 if self.is_series() else 1
        if self.is_series() and axis == 1:
            raise TypeError("Axis 1 is not supported in Series")
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            obj_index = self.wrapper.columns
        else:
            obj_index = self.wrapper.index
        obj_index = apply_func(obj_index, *args, **kwargs)
        obj = self.obj.values
        if copy_data:
            obj = obj.copy()
        if axis == 1:
            return self.wrapper.wrap(obj, group_by=False, columns=obj_index)
        return self.wrapper.wrap(obj, group_by=False, index=obj_index)

    def stack_index(
        self,
        index: tp.Index,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        on_top: bool = True,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.stack_indexes`.

        Set `on_top` to False to stack at bottom.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            if on_top:
                return indexes.stack_indexes([index, obj_index], **kwargs)
            return indexes.stack_indexes([obj_index, index], **kwargs)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_levels(
        self,
        levels: tp.MaybeLevelSequence,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        strict: bool = True,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_levels(obj_index, levels, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def rename_levels(
        self,
        name_dict: tp.Dict[str, tp.Any],
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
        strict: bool = True,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.rename_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.rename_levels(obj_index, name_dict, strict=strict)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def select_levels(
        self,
        level_names: tp.MaybeLevelSequence,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.select_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.select_levels(obj_index, level_names)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_redundant_levels(
        self,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_redundant_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_redundant_levels(obj_index)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def drop_duplicate_levels(
        self,
        keep: tp.Optional[str] = None,
        axis: tp.Optional[int] = None,
        copy_data: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """See `vectorbtpro.base.indexes.drop_duplicate_levels`.

        See `BaseAccessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index: tp.Index) -> tp.Index:
            return indexes.drop_duplicate_levels(obj_index, keep=keep)

        return self.apply_on_index(apply_func, axis=axis, copy_data=copy_data)

    def sort_index(self, axis: tp.Optional[int] = None, **kwargs) -> tp.SeriesFrame:
        """Sort index/column by their values."""
        if axis is None:
            axis = 0 if self.is_series() else 1
        if self.is_series() and axis == 1:
            raise TypeError("Axis 1 is not supported in Series")
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            obj_index = self.wrapper.columns
        else:
            obj_index = self.wrapper.index
        _, indexer = obj_index.sort_values(return_indexer=True, **kwargs)
        if axis == 1:
            return self.obj.iloc[:, indexer]
        return self.obj.iloc[indexer]

    # ############# Getting ############# #

    def ago(self, delta: tp.FrequencyLike, **kwargs) -> tp.SeriesFrame:
        """Get the value some periods behind each value.

        Keyword arguments are passed to
        [pandas.Index.get_indexer](https://pandas.pydata.org/docs/reference/api/pandas.Index.get_indexer.html)."""
        if isinstance(delta, str):
            delta = freq_to_timedelta(delta)
        indices = self.wrapper.index.get_indexer(self.wrapper.index - delta, **kwargs)
        new_obj = self.wrapper.fill()
        found_mask = indices != -1
        new_obj.iloc[np.flatnonzero(found_mask)] = self.obj.iloc[indices[found_mask]]
        return new_obj

    # ############# Setting ############# #

    def set(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set value at each index point using `vectorbtpro.base.wrapping.ArrayWrapper.get_index_points`.

        If `value_or_func` is a function, selects all keyword arguments that were not passed
        to the `get_index_points` method, substitutes any templates, and passes everything to the function.
        As context uses `kwargs`, `template_context`, and various variables such as `i` (iteration index),
        `index_point` (absolute position in the index), `wrapper`, and `obj`."""
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_points = self.wrapper.get_index_points(**kwargs)

        if callable(value_or_func):
            arg_names = get_func_arg_names(self.wrapper.get_index_points)
            func_kwargs = {k: v for k, v in kwargs.items() if k not in arg_names}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        if callable(value_or_func):
            for i in range(len(index_points)):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_point=index_points[i],
                        index_points=index_points,
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = deep_substitute(args, _template_context, sub_id="func_args")
                _func_kwargs = deep_substitute(func_kwargs, _template_context, sub_id="func_kwargs")
                v = value_or_func(*_func_args, **_func_kwargs)
                if self.is_series() or columns is None:
                    obj.iloc[index_points[i]] = v
                elif is_scalar(columns):
                    obj.iloc[index_points[i], obj.columns.get_indexer([columns])[0]] = v
                else:
                    obj.iloc[index_points[i], obj.columns.get_indexer(columns)] = v
        elif checks.is_sequence(value_or_func) and not is_scalar(value_or_func):
            if self.is_series():
                obj.iloc[index_points] = reshaping.to_1d_array(value_or_func)
            elif columns is None:
                obj.iloc[index_points] = reshaping.to_2d_array(value_or_func)
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = reshaping.to_1d_array(value_or_func)
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = reshaping.to_2d_array(value_or_func)
        else:
            if self.is_series() or columns is None:
                obj.iloc[index_points] = value_or_func
            elif is_scalar(columns):
                obj.iloc[index_points, obj.columns.get_indexer([columns])[0]] = value_or_func
            else:
                obj.iloc[index_points, obj.columns.get_indexer(columns)] = value_or_func
        if inplace:
            return None
        return obj

    def set_between(
        self,
        value_or_func: tp.Union[tp.MaybeArray, tp.Callable],
        *args,
        inplace: bool = False,
        columns: tp.Optional[tp.MaybeSequence[tp.Hashable]] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Set value at each index range using `vectorbtpro.base.wrapping.ArrayWrapper.get_index_ranges`.

        If `value_or_func` is a function, selects all keyword arguments that were not passed
        to the `get_index_points` method, substitutes any templates, and passes everything to the function.
        As context uses `kwargs`, `template_context`, and various variables such as `i` (iteration index),
        `index_slice` (absolute slice of the index), `wrapper`, and `obj`."""
        if inplace:
            obj = self.obj
        else:
            obj = self.obj.copy()
        index_ranges = self.wrapper.get_index_ranges(**kwargs)

        if callable(value_or_func):
            arg_names = get_func_arg_names(self.wrapper.get_index_points)
            func_kwargs = {k: v for k, v in kwargs.items() if k not in arg_names}
            template_context = merge_dicts(kwargs, template_context)
        else:
            func_kwargs = None
        for i in range(len(index_ranges)):
            if callable(value_or_func):
                _template_context = merge_dicts(
                    dict(
                        i=i,
                        index_slice=slice(index_ranges[i, 0], index_ranges[i, 1]),
                        index_ranges=index_ranges,
                        wrapper=self.wrapper,
                        obj=self.obj,
                        columns=columns,
                        args=args,
                        kwargs=kwargs,
                    ),
                    template_context,
                )
                _func_args = deep_substitute(args, _template_context, sub_id="func_args")
                _func_kwargs = deep_substitute(func_kwargs, _template_context, sub_id="func_kwargs")
                v = value_or_func(*_func_args, **_func_kwargs)
            elif checks.is_sequence(value_or_func) and not isinstance(value_or_func, str):
                v = value_or_func[i]
            else:
                v = value_or_func
            if self.is_series() or columns is None:
                obj.iloc[index_ranges[i, 0] : index_ranges[i, 1]] = v
            elif is_scalar(columns):
                obj.iloc[index_ranges[i, 0] : index_ranges[i, 1], obj.columns.get_indexer([columns])[0]] = v
            else:
                obj.iloc[index_ranges[i, 0] : index_ranges[i, 1], obj.columns.get_indexer(columns)] = v
        if inplace:
            return None
        return obj

    # ############# Reshaping ############# #

    def to_1d_array(self) -> tp.Array1d:
        """See `vectorbtpro.base.reshaping.to_1d` with `raw` set to True."""
        return reshaping.to_1d_array(self.obj)

    def to_2d_array(self) -> tp.Array2d:
        """See `vectorbtpro.base.reshaping.to_2d` with `raw` set to True."""
        return reshaping.to_2d_array(self.obj)

    def tile(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.base.reshaping.tile`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        tiled = reshaping.tile(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([keys, self.wrapper.columns])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([keys, self.wrapper.index])
                return ArrayWrapper.from_obj(tiled).wrap(
                    tiled.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return tiled

    def repeat(
        self,
        n: int,
        keys: tp.Optional[tp.IndexLike] = None,
        axis: int = 1,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.SeriesFrame:
        """See `vectorbtpro.base.reshaping.repeat`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        repeated = reshaping.repeat(self.obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = indexes.combine_indexes([self.wrapper.columns, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(columns=new_columns), wrap_kwargs),
                )
            else:
                new_index = indexes.combine_indexes([self.wrapper.index, keys])
                return ArrayWrapper.from_obj(repeated).wrap(
                    repeated.values,
                    **merge_dicts(dict(index=new_index), wrap_kwargs),
                )
        return repeated

    def align_to(self, other: tp.SeriesFrame, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Align to `other` on their axes.

        Usage:
            ```pycon
            >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['x', 'y'], columns=['a', 'b'])
            >>> df1
               a  b
            x  1  2
            y  3  4

            >>> df2 = pd.DataFrame([[5, 6, 7, 8], [9, 10, 11, 12]], index=['x', 'y'],
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']]))
            >>> df2
                   1       2
               a   b   a   b
            x  5   6   7   8
            y  9  10  11  12

            >>> df1.vbt.align_to(df2)
                  1     2
               a  b  a  b
            x  1  2  1  2
            y  3  4  3  4
            ```
        """
        checks.assert_instance_of(other, (pd.Series, pd.DataFrame))
        obj = reshaping.to_2d(self.obj)
        other = reshaping.to_2d(other)

        aligned_index = indexes.align_index_to(obj.index, other.index)
        aligned_columns = indexes.align_index_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrapper.wrap(
            obj.values,
            group_by=False,
            **merge_dicts(dict(index=other.index, columns=other.columns), wrap_kwargs),
        )

    @class_or_instancemethod
    def broadcast(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return reshaping.broadcast(*others, **kwargs)
        return reshaping.broadcast(cls_or_self.obj, *others, **kwargs)

    def broadcast_to(self, other: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast_to`."""
        if isinstance(other, BaseAccessor):
            other = other.obj
        return reshaping.broadcast_to(self.obj, other, **kwargs)

    @class_or_instancemethod
    def broadcast_combs(cls_or_self, *others: tp.Union[tp.ArrayLike, "BaseAccessor"], **kwargs) -> tp.Any:
        """See `vectorbtpro.base.reshaping.broadcast_combs`."""
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            return reshaping.broadcast_combs(*others, **kwargs)
        return reshaping.broadcast_combs(cls_or_self.obj, *others, **kwargs)

    def make_symmetric(self) -> tp.Frame:  # pragma: no cover
        """See `vectorbtpro.base.reshaping.make_symmetric`."""
        return reshaping.make_symmetric(self.obj)

    def unstack_to_array(self, **kwargs) -> tp.Array:  # pragma: no cover
        """See `vectorbtpro.base.reshaping.unstack_to_array`."""
        return reshaping.unstack_to_array(self.obj, **kwargs)

    def unstack_to_df(self, **kwargs) -> tp.Frame:  # pragma: no cover
        """See `vectorbtpro.base.reshaping.unstack_to_df`."""
        return reshaping.unstack_to_df(self.obj, **kwargs)

    def to_dict(self, **kwargs) -> tp.Mapping:
        """See `vectorbtpro.base.reshaping.to_dict`."""
        return reshaping.to_dict(self.obj, **kwargs)

    # ############# Combining ############# #

    def apply(
        self,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Apply a function `apply_func`.

        Set `keep_pd` to True to keep inputs as pandas objects, otherwise convert to NumPy arrays.

        Set `to_2d` to True to reshape inputs to 2-dim arrays, otherwise keep as-is.

        `*args` and `**kwargs` are passed to `apply_func`.

        !!! note
            The resulted array must have the same shape as the original array.

        Usage:
            * Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> sr.vbt.apply(lambda x: x ** 2)
            x    1
            y    4
            dtype: int64
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr.vbt.apply(
            ...     lambda x, y: x + y,
            ...     vbt.Rep('y'),
            ...     broadcast_named_args=dict(
            ...         y=pd.DataFrame([[3, 4]], columns=['a', 'b'])
            ...     )
            ... )
               a  b
            x  4  5
            y  5  6
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = deep_substitute(args, template_context, sub_id="args")
        kwargs = deep_substitute(kwargs, template_context, sub_id="kwargs")
        out = apply_func(broadcast_named_args["obj"], *args, **kwargs)
        return wrapper.wrap(out, group_by=False, **resolve_dict(wrap_kwargs))

    @class_or_instancemethod
    def concat(
        cls_or_self,
        *others: tp.ArrayLike,
        broadcast_kwargs: tp.KwargsLike = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Frame:
        """Concatenate with `others` along columns.

        Usage:
            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> sr.vbt.concat(df, keys=['c', 'd'])
                  c     d
               a  b  a  b
            x  1  1  3  4
            y  2  2  5  6
            ```
        """
        others = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, others))
        if isinstance(cls_or_self, type):
            objs = others
        else:
            objs = (cls_or_self.obj,) + others
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        broadcasted = reshaping.broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(reshaping.to_2d, broadcasted))
        out = pd.concat(broadcasted, axis=1, keys=keys)
        if not isinstance(out.columns, pd.MultiIndex) and np.all(out.columns == 0):
            out.columns = pd.RangeIndex(start=0, stop=len(out.columns), step=1)
        return out

    def apply_and_concat(
        self,
        ntimes: int,
        apply_func: tp.Callable,
        *args,
        keep_pd: bool = False,
        to_2d: bool = False,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.Frame]:
        """Apply `apply_func` `ntimes` times and concatenate the results along columns.

        See `vectorbtpro.base.combining.apply_and_concat`.

        `ntimes` is the number of times to call `apply_func`, while `n_outputs` is the number of outputs to expect.

        `*args` and `**kwargs` are passed to `vectorbtpro.base.combining.apply_and_concat`.

        !!! note
            The resulted arrays to be concatenated must have the same shape as broadcast input arrays.

        Usage:
            * Using instance method:

            ```pycon
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
            >>> df.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b[i],
            ...     [1, 2, 3],
            ...     keys=['c', 'd', 'e']
            ... )
                  c       d       e
               a  b   a   b   a   b
            x  3  4   6   8   9  12
            y  5  6  10  12  15  18
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.apply_and_concat(
            ...     3,
            ...     lambda i, a, b: a * b + i,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            apply_idx        0         1         2
                       a  b  c  a  b   c  a  b   c
            x          1  2  3  2  3   4  3  4   5
            y          2  4  6  3  5   7  4  6   8
            z          3  6  9  4  7  10  5  8  11
            ```

            * To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(i, a):
            ...     time.sleep(1)
            ...     return a

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func)
            3.02 s ± 3.76 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.apply_and_concat(3, apply_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 927 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        checks.assert_not_none(apply_func)
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        broadcast_named_args = {"obj": self.obj, **broadcast_named_args}
        if len(broadcast_named_args) > 1:
            broadcast_named_args, wrapper = reshaping.broadcast(
                broadcast_named_args,
                return_wrapper=True,
                **broadcast_kwargs,
            )
        else:
            wrapper = self.wrapper
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, dict(ntimes=ntimes), template_context)
        args = deep_substitute(args, template_context, sub_id="args")
        kwargs = deep_substitute(kwargs, template_context, sub_id="kwargs")
        out = combining.apply_and_concat(ntimes, apply_func, broadcast_named_args["obj"], *args, **kwargs)
        if keys is not None:
            new_columns = indexes.combine_indexes([keys, wrapper.columns])
        else:
            top_columns = pd.Index(np.arange(ntimes), name="apply_idx")
            new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
        if out is None:
            return None
        wrap_kwargs = merge_dicts(dict(columns=new_columns), wrap_kwargs)
        if isinstance(out, list):
            return tuple(map(lambda x: wrapper.wrap(x, group_by=False, **wrap_kwargs), out))
        return wrapper.wrap(out, group_by=False, **wrap_kwargs)

    @class_or_instancemethod
    def combine(
        cls_or_self,
        obj: tp.MaybeTupleList[tp.Union[tp.ArrayLike, "BaseAccessor"]],
        combine_func: tp.Callable,
        *args,
        allow_multiple: bool = True,
        keep_pd: bool = False,
        to_2d: bool = False,
        concat: tp.Optional[bool] = None,
        keys: tp.Optional[tp.IndexLike] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        wrap_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Combine with `other` using `combine_func`.

        Args:
            obj (array_like): Object(s) to combine this array with.
            combine_func (callable): Function to combine two arrays.

                Can be Numba-compiled.
            *args: Variable arguments passed to `combine_func`.
            allow_multiple (bool): Whether a tuple/list/Index will be considered as multiple objects in `other`.

                Takes effect only when using the instance method.
            keep_pd (bool): Whether to keep inputs as pandas objects, otherwise convert to NumPy arrays.
            to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
            concat (bool): Whether to concatenate the results along the column axis.
                Otherwise, pairwise combine into a Series/DataFrame of the same shape.

                If True, see `vectorbtpro.base.combining.combine_and_concat`.
                If False, see `vectorbtpro.base.combining.combine_multiple`.
                If None, becomes True if there are more than two arguments to combine.

                Can only concatenate using the instance method.
            keys (index_like): Outermost column level.
            broadcast_named_args (dict): Dictionary with arguments to broadcast against each other.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            template_context (dict): Mapping used to substitute templates in `args` and `kwargs`.
            wrap_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            **kwargs: Keyword arguments passed to `combine_func`.

        !!! note
            If `combine_func` is Numba-compiled, will broadcast using `WRITEABLE` and `C_CONTIGUOUS`
            flags, which can lead to an expensive computation overhead if passed objects are large and
            have different shape/memory order. You also must ensure that all objects have the same data type.

            Also remember to bring each in `*args` to a Numba-compatible format.

        Usage:
            * Using instance method:

            ```pycon
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> # using instance method
            >>> sr.vbt.combine(df, np.add)
               a  b
            x  4  5
            y  7  8

            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20

            >>> sr.vbt.combine([df, df * 2], np.add)
            combine_idx     0       1
                         a  b   a   b
            x            4  5   7   9
            y            7  8  12  14

            >>> sr.vbt.combine([df, df * 2], np.add, keys=['c', 'd'])
                  c       d
               a  b   a   b
            x  4  5   7   9
            y  7  8  12  14

            >>> sr.vbt.combine(pd.Index([1, 2], name='param'), np.add)
            param  1  2
            x      2  3
            y      3  4

            >>> # using class method
            >>> sr.vbt.combine([df, df * 2], np.add, concat=False)
                a   b
            x  10  13
            y  17  20
            ```

            * Using class method, templates, and broadcasting:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> sr.vbt.combine(
            ...     [1, 2, 3],
            ...     lambda x, y, z: x + y + z,
            ...     vbt.Rep('df'),
            ...     broadcast_named_args=dict(
            ...         df=pd.DataFrame([[1, 2, 3]], columns=['a', 'b', 'c'])
            ...     )
            ... )
            combine_idx        0        1        2
                         a  b  c  a  b  c  a  b  c
            x            3  4  5  4  5  6  5  6  7
            y            4  5  6  5  6  7  6  7  8
            z            5  6  7  6  7  8  7  8  9
            ```

            * To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def combine_func(a, b):
            ...     time.sleep(1)
            ...     return a + b

            >>> sr = pd.Series([1, 2, 3])

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func)
            3.01 s ± 2.98 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit sr.vbt.combine([1, 1, 1], combine_func, execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.18 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        if broadcast_named_args is None:
            broadcast_named_args = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        if isinstance(cls_or_self, type):
            objs = obj
        else:
            if allow_multiple and isinstance(obj, (tuple, list)):
                objs = obj
            else:
                objs = (obj,)
        objs = tuple(map(lambda x: x.obj if isinstance(x, BaseAccessor) else x, objs))
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self.obj,) + objs
        if checks.is_numba_func(combine_func):
            # Numba requires writeable arrays and in the same order
            broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements=["W", "C"])), broadcast_kwargs)

        # Broadcast and substitute templates
        broadcast_named_args = {**{"obj_" + str(i): obj for i, obj in enumerate(objs)}, **broadcast_named_args}
        broadcast_named_args, wrapper = reshaping.broadcast(
            broadcast_named_args,
            return_wrapper=True,
            **broadcast_kwargs,
        )
        if to_2d:
            broadcast_named_args = {k: reshaping.to_2d(v, raw=not keep_pd) for k, v in broadcast_named_args.items()}
        elif not keep_pd:
            broadcast_named_args = {k: np.asarray(v) for k, v in broadcast_named_args.items()}
        template_context = merge_dicts(broadcast_named_args, template_context)
        args = deep_substitute(args, template_context, sub_id="args")
        kwargs = deep_substitute(kwargs, template_context, sub_id="kwargs")
        inputs = [broadcast_named_args["obj_" + str(i)] for i in range(len(objs))]

        if concat is None:
            concat = len(inputs) > 2
        if concat:
            # Concat the results horizontally
            if isinstance(cls_or_self, type):
                raise TypeError("Use instance method to concatenate")
            out = combining.combine_and_concat(inputs[0], inputs[1:], combine_func, *args, **kwargs)
            if keys is not None:
                new_columns = indexes.combine_indexes([keys, wrapper.columns])
            else:
                top_columns = pd.Index(np.arange(len(objs) - 1), name="combine_idx")
                new_columns = indexes.combine_indexes([top_columns, wrapper.columns])
            return wrapper.wrap(out, **merge_dicts(dict(columns=new_columns), wrap_kwargs))
        else:
            # Combine arguments pairwise into one object
            out = combining.combine_multiple(inputs, combine_func, *args, **kwargs)
            return wrapper.wrap(out, **resolve_dict(wrap_kwargs))

    @classmethod
    def eval(
        cls,
        expression: str,
        use_numexpr: tp.Optional[bool] = None,
        numexpr_kwargs: tp.KwargsLike = None,
        local_dict: tp.Optional[tp.Mapping] = None,
        global_dict: tp.Optional[tp.Mapping] = None,
        broadcast_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ):
        """Evaluate a simple array expression element-wise using NumExpr or NumPy.

        The only advantage of this method over `pd.eval` is using vectorbt's own broadcasting.

        !!! note
            All variables will broadcast against each other prior to the evaluation.

        Usage:
            * A bit slower than `pd.eval`:

            ```pycon
            >>> df = pd.DataFrame(np.full((1000, 1000), 0))
            >>> sr = pd.Series(np.full(1000, 1))
            >>> a = np.full(1000, 2)

            >>> %timeit pd.eval('df + sr + a')
            1.12 ms ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

            >>> %timeit vbt.pd_acc.eval('df + sr + a')
            1.3 ms ± 5.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
            ```

            * But broadcasts nicely:

            ```pycon
            >>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
            >>> df = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a', 'b', 'c'])
            >>> pd.eval('sr + df')
                a   b   c   x   y   z
            x NaN NaN NaN NaN NaN NaN
            y NaN NaN NaN NaN NaN NaN
            z NaN NaN NaN NaN NaN NaN

            >>> vbt.pd_acc.eval('sr + df')
               a  b  c
            x  5  6  7
            y  6  7  8
            z  7  8  9
            ```
        """
        if numexpr_kwargs is None:
            numexpr_kwargs = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        var_names = get_expr_var_names(expression)
        objs = get_context_vars(var_names, frames_back=1, local_dict=local_dict, global_dict=global_dict)
        objs = reshaping.broadcast(*objs, **broadcast_kwargs)
        vars_by_name = {}
        for i, obj in enumerate(objs):
            vars_by_name[var_names[i]] = np.asarray(obj)
        if use_numexpr is None:
            if objs[0].size >= 100000:
                from vectorbtpro.utils.opt_packages import warn_cannot_import

                warn_cannot_import("numexpr")
                try:
                    import numexpr

                    use_numexpr = True
                except ImportError:
                    use_numexpr = False
            else:
                use_numexpr = False
        if use_numexpr:
            from vectorbtpro.utils.opt_packages import assert_can_import

            assert_can_import("numexpr")
            import numexpr

            out = numexpr.evaluate(expression, local_dict=vars_by_name, **numexpr_kwargs)
        else:
            out = eval(expression, {}, vars_by_name)
        return ArrayWrapper.from_obj(objs[0]).wrap(out, **wrap_kwargs)


class BaseSRAccessor(BaseAccessor):
    """Accessor on top of Series.

    Accessible via `pd.Series.vbt` and all child accessors."""

    def __init__(self, obj: tp.Series, **kwargs) -> None:
        checks.assert_instance_of(obj, pd.Series)

        BaseAccessor.__init__(self, obj, **kwargs)

    @class_or_instanceproperty
    def ndim(cls_or_self) -> int:
        return 1

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        return True

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        return False


class BaseDFAccessor(BaseAccessor):
    """Accessor on top of DataFrames.

    Accessible via `pd.DataFrame.vbt` and all child accessors."""

    def __init__(self, obj: tp.Frame, **kwargs) -> None:
        checks.assert_instance_of(obj, pd.DataFrame)

        BaseAccessor.__init__(self, obj, **kwargs)

    @class_or_instanceproperty
    def ndim(cls_or_self) -> int:
        return 2

    @class_or_instancemethod
    def is_series(cls_or_self) -> bool:
        return False

    @class_or_instancemethod
    def is_frame(cls_or_self) -> bool:
        return True
