# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Functions for reshaping arrays.

Reshape functions transform a Pandas object/NumPy array in some way."""

import functools
import itertools
from collections import OrderedDict, defaultdict

import attr
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import _broadcast_shape

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, wrapping
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import resolve_dict, merge_dicts
from vectorbtpro.utils.params import generate_param_combs
from vectorbtpro.utils.parsing import get_func_arg_names


def shape_to_tuple(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a tuple."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def shape_to_2d(shape: tp.ShapeLike) -> tp.Shape:
    """Convert a shape-like object to a 2-dim shape."""
    shape = shape_to_tuple(shape)
    if len(shape) == 1:
        return shape[0], 1
    return shape


def index_to_series(arg: tp.Index) -> tp.Series:
    """Convert Index to Series."""
    return arg.to_series()


def mapping_to_series(arg: tp.MappingLike) -> tp.Series:
    """Convert a mapping-like object to Series."""
    if checks.is_namedtuple(arg):
        arg = arg._asdict()
    return pd.Series(arg)


def to_any_array(arg: tp.ArrayLike, raw: bool = False, convert_index: bool = True) -> tp.AnyArray:
    """Convert any array-like object to an array.

    Pandas objects are kept as-is unless `raw` is True."""
    if not raw:
        if checks.is_any_array(arg):
            if convert_index and checks.is_index(arg):
                return index_to_series(arg)
            return arg
        if checks.is_mapping_like(arg):
            return mapping_to_series(arg)
    return np.asarray(arg)


def to_pd_array(arg: tp.ArrayLike, convert_index: bool = True) -> tp.PandasArray:
    """Convert any array-like object to a Pandas object."""
    if checks.is_pandas(arg):
        if convert_index and checks.is_index(arg):
            return index_to_series(arg)
        return arg
    if checks.is_mapping_like(arg):
        return mapping_to_series(arg)

    arg = np.asarray(arg)
    if arg.ndim == 0:
        arg = arg[None]
    if arg.ndim == 1:
        return pd.Series(arg)
    if arg.ndim == 2:
        return pd.DataFrame(arg)
    raise ValueError("Wrong number of dimensions: cannot convert to Series or DataFrame")


def soft_to_ndim(arg: tp.ArrayLike, ndim: int, raw: bool = False) -> tp.AnyArray:
    """Try to softly bring `arg` to the specified number of dimensions `ndim` (max 2)."""
    arg = to_any_array(arg, raw=raw)
    if ndim == 1:
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if checks.is_frame(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]  # downgrade
    if ndim == 2:
        if arg.ndim == 1:
            if checks.is_series(arg):
                return arg.to_frame()
            return arg[:, None]  # upgrade
    return arg  # do nothing


def to_1d(arg: tp.ArrayLike, raw: bool = False) -> tp.AnyArray1d:
    """Reshape argument to one dimension.

    If `raw` is True, returns NumPy array.
    If 2-dim, will collapse along axis 1 (i.e., DataFrame with one column to Series)."""
    arg = to_any_array(arg, raw=raw)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if checks.is_frame(arg):
                return arg.iloc[:, 0]
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


to_1d_array = functools.partial(to_1d, raw=True)


def to_2d(arg: tp.ArrayLike, raw: bool = False, expand_axis: int = 1) -> tp.AnyArray2d:
    """Reshape argument to two dimensions.

    If `raw` is True, returns NumPy array.
    If 1-dim, will expand along axis 1 (i.e., Series to DataFrame with one column)."""
    arg = to_any_array(arg, raw=raw)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if checks.is_series(arg):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


to_2d_array = functools.partial(to_2d, raw=True)


def to_dict(arg: tp.ArrayLike, orient: str = "dict") -> dict:
    """Convert object to dict."""
    arg = to_pd_array(arg)
    if orient == "index_series":
        return {arg.index[i]: arg.iloc[i] for i in range(len(arg.index))}
    return arg.to_dict(orient)


def repeat(
    arg: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Repeat each element in `arg` `n` times along the specified axis."""
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        if checks.is_pandas(arg):
            new_index = indexes.repeat_index(arg.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.repeat(arg.values, n, axis=0), index=new_index)
        return np.repeat(arg, n, axis=0)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            new_columns = indexes.repeat_index(arg.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.repeat(arg.values, n, axis=1), columns=new_columns)
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def tile(
    arg: tp.ArrayLike,
    n: int,
    axis: int = 1,
    raw: bool = False,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """Repeat the whole `arg` `n` times along the specified axis."""
    arg = to_any_array(arg, raw=raw)
    if axis == 0:
        if arg.ndim == 2:
            if checks.is_pandas(arg):
                new_index = indexes.tile_index(arg.index, n, ignore_ranges=ignore_ranges)
                return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, (n, 1)), index=new_index)
            return np.tile(arg, (n, 1))
        if checks.is_pandas(arg):
            new_index = indexes.tile_index(arg.index, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, n), index=new_index)
        return np.tile(arg, n)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            new_columns = indexes.tile_index(arg.columns, n, ignore_ranges=ignore_ranges)
            return wrapping.ArrayWrapper.from_obj(arg).wrap(np.tile(arg.values, (1, n)), columns=new_columns)
        return np.tile(arg, (1, n))
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def column_stack(arrs: tp.Sequence[tp.AnyArray]) -> tp.Array2d:
    """Stack arrays along columns."""
    first_arr = arrs[0]
    if first_arr.ndim == 1 or (first_arr.ndim == 2 and first_arr.shape[1] == 1):
        return np.concatenate(arrs).reshape((len(arrs), len(first_arr))).T
    return np.column_stack(arrs)


IndexFromLike = tp.Union[None, str, int, tp.Any]
"""Any object that can be coerced into a `index_from` argument."""


def broadcast_index(
    args: tp.Sequence[tp.AnyArray],
    to_shape: tp.Shape,
    index_from: IndexFromLike = None,
    axis: int = 0,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    **stack_kwargs,
) -> tp.Optional[tp.Index]:
    """Produce a broadcast index/columns.

    Args:
        args (iterable of array_like): Array-like objects.
        to_shape (tuple of int): Target shape.
        index_from (any): Broadcasting rule for this index/these columns.

            Accepts the following values:

            * 'keep' or None - keep the original index/columns of the objects in `args`
            * 'stack' - stack different indexes/columns using `vectorbtpro.base.indexes.stack_indexes`
            * 'strict' - ensure that all Pandas objects have the same index/columns
            * 'reset' - reset any index/columns (they become a simple range)
            * integer - use the index/columns of the i-th object in `args`
            * everything else will be converted to `pd.Index`
        axis (int): Set to 0 for index and 1 for columns.
        ignore_sr_names (bool): Whether to ignore Series names if they are in conflict.

            Conflicting Series names are those that are different but not None.
        ignore_ranges (bool): Whether to ignore indexes of type `pd.RangeIndex`.
        check_index_names (bool): See `vectorbtpro.utils.checks.is_index_equal`.
        **stack_kwargs: Keyword arguments passed to `vectorbtpro.base.indexes.stack_indexes`.

    For defaults, see `vectorbtpro._settings.broadcasting`.

    !!! note
        Series names are treated as columns with a single element but without a name.
        If a column level without a name loses its meaning, better to convert Series to DataFrames
        with one column prior to broadcasting. If the name of a Series is not that important,
        better to drop it altogether by setting it to None.
    """
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if ignore_sr_names is None:
        ignore_sr_names = broadcasting_cfg["ignore_sr_names"]
    if check_index_names is None:
        check_index_names = broadcasting_cfg["check_index_names"]

    index_str = "columns" if axis == 1 else "index"
    to_shape_2d = (to_shape[0], 1) if len(to_shape) == 1 else to_shape
    # maxlen stores the length of the longest index
    maxlen = to_shape_2d[1] if axis == 1 else to_shape_2d[0]
    new_index = None
    args = list(args)

    if index_from is None or (isinstance(index_from, str) and index_from.lower() == "keep"):
        return None
    if isinstance(index_from, int):
        # Take index/columns of the object indexed by index_from
        if not checks.is_pandas(args[index_from]):
            raise TypeError(f"Argument under index {index_from} must be a pandas object")
        new_index = indexes.get_index(args[index_from], axis)
    elif isinstance(index_from, str):
        if index_from.lower() == "reset":
            # Ignore index/columns
            new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
        elif index_from.lower() in ("stack", "strict"):
            # Check whether all indexes/columns are equal
            last_index = None  # of type pd.Index
            index_conflict = False
            for arg in args:
                if checks.is_pandas(arg):
                    index = indexes.get_index(arg, axis)
                    if last_index is not None:
                        if not checks.is_index_equal(index, last_index, check_names=check_index_names):
                            index_conflict = True
                    last_index = index
                    continue
            if not index_conflict:
                new_index = last_index
            else:
                # If pandas objects have different index/columns, stack them together
                for arg in args:
                    if checks.is_pandas(arg):
                        index = indexes.get_index(arg, axis)
                        if axis == 1 and checks.is_series(arg) and ignore_sr_names:
                            # ignore Series name
                            continue
                        if checks.is_default_index(index):
                            # ignore simple ranges without name
                            continue
                        if new_index is None:
                            new_index = index
                        else:
                            if checks.is_index_equal(index, new_index, check_names=check_index_names):
                                continue
                            if index_from.lower() == "strict":
                                # If pandas objects have different index/columns, raise an exception
                                raise ValueError(
                                    f"Arrays have different index. Broadcasting {index_str} "
                                    f"is not allowed when {index_str}_from=strict"
                                )

                            # Broadcasting index must follow the rules of a regular broadcasting operation
                            # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
                            # 1. rule: if indexes are of the same length, they are simply stacked
                            # 2. rule: if index has one element, it gets repeated and then stacked
                            if len(index) != len(new_index):
                                if len(index) > 1 and len(new_index) > 1:
                                    raise ValueError("Indexes could not be broadcast together")
                                if len(index) > len(new_index):
                                    new_index = indexes.repeat_index(new_index, len(index), ignore_ranges=ignore_ranges)
                                elif len(index) < len(new_index):
                                    index = indexes.repeat_index(index, len(new_index), ignore_ranges=ignore_ranges)
                            new_index = indexes.stack_indexes([new_index, index], **stack_kwargs)
        else:
            raise ValueError(f"Invalid value '{index_from}' for {'columns' if axis == 1 else 'index'}_from")
    else:
        if not isinstance(index_from, pd.Index):
            index_from = pd.Index(index_from)
        new_index = index_from
    if new_index is not None:
        if maxlen > len(new_index):
            if isinstance(index_from, str) and index_from.lower() == "strict":
                raise ValueError(f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
            # This happens only when some numpy object is longer than the new pandas index
            # In this case, new pandas index (one element) must be repeated to match this length.
            if maxlen > 1 and len(new_index) > 1:
                raise ValueError("Indexes could not be broadcast together")
            new_index = indexes.repeat_index(new_index, maxlen, ignore_ranges=ignore_ranges)
    else:
        # new_index=None can mean two things: 1) take original metadata or 2) reset index/columns
        # In case when index_from is not None, we choose 2)
        new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
    return new_index


def wrap_broadcasted(
    old_arg: tp.AnyArray,
    new_arg: tp.Array,
    is_pd: bool = False,
    new_index: tp.Optional[tp.Index] = None,
    new_columns: tp.Optional[tp.Index] = None,
    ignore_ranges: tp.Optional[bool] = None,
) -> tp.AnyArray:
    """If the newly brodcasted array was originally a Pandas object, make it Pandas object again
    and assign it the newly broadcast index/columns."""
    if is_pd:
        if checks.is_pandas(old_arg):
            if new_index is None:
                # Take index from original pandas object
                old_index = indexes.get_index(old_arg, 0)
                if old_arg.shape[0] == new_arg.shape[0]:
                    new_index = old_index
                else:
                    new_index = indexes.repeat_index(old_index, new_arg.shape[0], ignore_ranges=ignore_ranges)
            if new_columns is None:
                # Take columns from original pandas object
                old_columns = indexes.get_index(old_arg, 1)
                new_ncols = new_arg.shape[1] if new_arg.ndim == 2 else 1
                if len(old_columns) == new_ncols:
                    new_columns = old_columns
                else:
                    new_columns = indexes.repeat_index(old_columns, new_ncols, ignore_ranges=ignore_ranges)
        if new_arg.ndim == 2:
            return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
        if new_columns is not None and len(new_columns) == 1:
            name = new_columns[0]
            if name == 0:
                name = None
        else:
            name = None
        return pd.Series(new_arg, index=new_index, name=name)
    return new_arg


def align_pd_arrays(
    args: tp.Iterable[tp.ArrayLike],
    align_index: bool = True,
    align_columns: bool = True,
) -> tp.List[tp.ArrayLike]:
    """Align Pandas arrays against common index and/or column levels using
    `vectorbtpro.base.indexes.align_indexes`."""
    args = list(args)
    if align_index:
        index_to_align = []
        for i in range(len(args)):
            if checks.is_pandas(args[i]) and len(args[i].index) > 1:
                index_to_align.append(i)
        if len(index_to_align) > 1:
            indexes_ = [args[i].index for i in index_to_align]
            if len(set(map(len, indexes_))) > 1:
                index_indices = indexes.align_indexes(indexes_)
                for i in index_to_align:
                    args[i] = args[i].iloc[index_indices[index_to_align.index(i)]]
    if align_columns:
        cols_to_align = []
        for i in range(len(args)):
            if checks.is_frame(args[i]) and len(args[i].columns) > 1:
                cols_to_align.append(i)
        if len(cols_to_align) > 1:
            indexes_ = [args[i].columns for i in cols_to_align]
            if len(set(map(len, indexes_))) > 1:
                col_indices = indexes.align_indexes(indexes_)
                for i in cols_to_align:
                    args[i] = args[i].iloc[:, col_indices[cols_to_align.index(i)]]
    return args


@attr.s(frozen=True)
class BCO:
    """Class that represents an object passed to `broadcast`.

    If any value is None, mostly defaults to the global value passed to `broadcast`."""

    value: tp.Any = attr.ib()
    """Value of the object."""

    to_pd: tp.Optional[bool] = attr.ib(default=None)
    """Whether to convert the output array to a Pandas object."""

    keep_flex: tp.Optional[bool] = attr.ib(default=None)
    """Whether to keep the raw version of the output for flexible indexing.
    
    Only makes sure that the array can broadcast to the target shape."""

    min_one_dim: tp.Optional[bool] = attr.ib(default=None)
    """Whether to convert a constant into a 1-dim array."""

    post_func: tp.Optional[tp.Callable] = attr.ib(default=None)
    """Function to post-process the output array."""

    require_kwargs: tp.Optional[tp.Kwargs] = attr.ib(default=None)
    """Keyword arguments passed to `np.require`."""

    index_to_product: tp.Optional[bool] = attr.ib(default=None)
    """Whether to set `BCO.product` to True if `BCO.value` is an index."""

    product: tp.Optional[bool] = attr.ib(default=None)
    """Build a Cartesian product of parameter combinations in `BCO.value` and other objects.
    
    Treats `BCO.value` as a parameter holding a sequence of scalar values, one per entire shape.
    
    If None, becomes True if `BCO.value` is an index and `BCO.index_to_product` is True, 
    otherwise False."""

    product_idx: tp.Optional[int] = attr.ib(default=None)
    """Index of the product the parameter take part in.
    
    Parameters in the same product broadcast but are not combined together, 
    and appear in the column hierarchy next to each other.
    
    Product index can be used to order column levels: the higher the product index, 
    the lower the column level. Column levels with the same product index appear in the same 
    order as the parameters were passed to `broadcast`."""

    keys_from_sr_index: tp.Optional[bool] = attr.ib(default=None)
    """Whether to set `BCO.keys` to the index of `BCO.value` if `BCO.value` is a Series."""

    keys: tp.Optional[tp.IndexLike] = attr.ib(default=None)
    """Keys acting as a column level if `BCO.product` is True.
    
    If None, becomes the index of `BCO.value` if `BCO.value` is a Series and 
    `keys_from_sr_index` is True, otherwise the values of `BCO.value`."""

    repeat_product: tp.Optional[bool] = attr.ib(default=None)
    """Whether to repeat every parameter value to match the number of columns in regular arrays."""


@attr.s(frozen=True)
class Default:
    """Class for wrapping default values."""

    value: tp.Any = attr.ib()
    """Default value."""


@attr.s(frozen=True)
class Ref:
    """Class for wrapping references to other values."""

    key: tp.Hashable = attr.ib()
    """Reference to another key."""


def resolve_ref(dct: dict, k: tp.Hashable, inside_bco: bool = False, keep_wrap_default: bool = False) -> tp.Any:
    """Resolve a potential reference."""
    v = dct[k]
    is_default = False
    if isinstance(v, Default):
        v = v.value
        is_default = True
    if isinstance(v, Ref):
        new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
        if keep_wrap_default and is_default:
            return Default(new_v)
        return new_v
    if isinstance(v, BCO) and inside_bco:
        v = v.value
        is_default = False
        if isinstance(v, Default):
            v = v.value
            is_default = True
        if isinstance(v, Ref):
            new_v = resolve_ref(dct, v.key, inside_bco=inside_bco)
            if keep_wrap_default and is_default:
                return Default(new_v)
            return new_v
    return v


def broadcast(
    *args,
    to_shape: tp.Optional[tp.ShapeLike] = None,
    align_index: tp.Optional[bool] = None,
    align_columns: tp.Optional[bool] = None,
    index_from: tp.Optional[IndexFromLike] = None,
    columns_from: tp.Optional[IndexFromLike] = None,
    to_frame: tp.Optional[bool] = None,
    to_pd: tp.Optional[tp.MaybeMappingSequence[bool]] = None,
    keep_flex: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    min_one_dim: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    post_func: tp.MaybeMappingSequence[tp.Optional[tp.Callable]] = None,
    require_kwargs: tp.MaybeMappingSequence[tp.Optional[tp.Kwargs]] = None,
    index_to_product: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    product: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    product_idx: tp.MaybeMappingSequence[tp.Optional[int]] = None,
    repeat_product: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    keys_from_sr_index: tp.MaybeMappingSequence[tp.Optional[bool]] = None,
    keys: tp.MaybeMappingSequence[tp.Optional[tp.IndexLike]] = None,
    random_subset: tp.Optional[int] = None,
    keep_wrap_default: tp.Optional[bool] = None,
    return_wrapper: bool = False,
    wrapper_kwargs: tp.KwargsLike = None,
    ignore_sr_names: tp.Optional[bool] = None,
    ignore_ranges: tp.Optional[bool] = None,
    check_index_names: tp.Optional[bool] = None,
    **stack_kwargs,
) -> tp.Any:
    """Bring any array-like object in `args` to the same shape by using NumPy broadcasting.

    See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

    Can broadcast Pandas objects by broadcasting their index/columns with `broadcast_index`.

    Args:
        *args: Objects to broadcast.

            If the first and only argument is a mapping, will return a dict.

            Allows using `BCO` and `vectorbtpro.base.reshaping.Ref`.
        to_shape (tuple of int): Target shape. If set, will broadcast every object in `args` to `to_shape`.
        align_index (bool): Whether to align index of Pandas objects using multi-index.

            Pass None to use the default.
        align_columns (bool): Whether to align columns of Pandas objects using multi-index.

            Pass None to use the default.
        index_from (any): Broadcasting rule for index.

            Pass None to use the default.
        columns_from (any): Broadcasting rule for columns.

            Pass None to use the default.
        to_frame (bool): Whether to convert all Series to DataFrames.
        to_pd (bool, sequence or mapping): See `BCO.to_pd`.

            If None, converts only if there is at least one Pandas object among them.
        keep_flex (bool, sequence or mapping): See `BCO.keep_flex`.
        min_one_dim (bool, sequence or mapping): See `BCO.min_one_dim`.
        post_func (callable, sequence or mapping): See `BCO.post_func`.

            Applied only when `keep_flex` is False.
        require_kwargs (dict, sequence or mapping): See `BCO.require_kwargs`.

            This key will be merged with any argument-specific dict. If the mapping contains all keys in
            `np.require`, it will be applied on all objects.
        index_to_product (bool, sequence or mapping): See `BCO.index_to_product`.
        product (bool, sequence or mapping): See `BCO.product`.
        product_idx (int, sequence or mapping): See `BCO.product_idx`.
        repeat_product (bool, sequence or mapping): See `BCO.repeat_product`.
        keys_from_sr_index (bool, sequence or mapping): See `BCO.keys_from_sr_index`.
        keys (index_like, sequence or mapping): See `BCO.keys`.
        random_subset (int): Select a random subset of product parameter values.

            Seed can be set using NumPy before calling this function.
        keep_wrap_default (bool): Whether to keep wrapping with `vectorbtpro.base.reshaping.Default`.
        return_wrapper (bool): Whether to also return the wrapper associated with the operation.
        wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
        ignore_sr_names (bool): See `broadcast_index`.
        ignore_ranges (bool): See `broadcast_index`.
        check_index_names (bool): See `broadcast_index`.
        **stack_kwargs: Keyword arguments passed to `vectorbtpro.base.indexes.stack_indexes`.

    For defaults, see `vectorbtpro._settings.broadcasting`.

    Any keyword argument that can be associated with an object can be passed as

    * a const that is applied on all objects,
    * a sequence with value per object, and
    * a mapping with value per object name and the special key `_default` denoting the default value.

    Additionally, any object can be passed wrapped with `BCO`, which attributes will override
    any of the above arguments if not None.

    Usage:
        * Without broadcasting index and columns:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> import vectorbtpro as vbt

        >>> v = 0
        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([1, 2, 3], index=pd.Index(['x', 'y', 'z']), name='a')
        >>> df = pd.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ...     index=pd.Index(['x2', 'y2', 'z2']),
        ...     columns=pd.Index(['a2', 'b2', 'c2']))

        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='keep',
        ...     columns_from='keep',
        ... ): print(i)
           0  1  2
        0  0  0  0
        1  0  0  0
        2  0  0  0
           0  1  2
        0  1  2  3
        1  1  2  3
        2  1  2  3
           a  a  a
        x  1  1  1
        y  2  2  2
        z  3  3  3
            a2  b2  c2
        x2   1   2   3
        y2   4   5   6
        z2   7   8   9
        ```

        * Take index and columns from the argument at specific position:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=2,
        ...     columns_from=3
        ... ): print(i)
           a2  b2  c2
        x   0   0   0
        y   0   0   0
        z   0   0   0
           a2  b2  c2
        x   1   2   3
        y   1   2   3
        z   1   2   3
           a2  b2  c2
        x   1   1   1
        y   2   2   2
        z   3   3   3
           a2  b2  c2
        x   1   2   3
        y   4   5   6
        z   7   8   9
        ```

        * Broadcast index and columns through stacking:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from='stack',
        ...     columns_from='stack'
        ... ): print(i)
              a2  b2  c2
        x x2   0   0   0
        y y2   0   0   0
        z z2   0   0   0
              a2  b2  c2
        x x2   1   2   3
        y y2   1   2   3
        z z2   1   2   3
              a2  b2  c2
        x x2   1   1   1
        y y2   2   2   2
        z z2   3   3   3
              a2  b2  c2
        x x2   1   2   3
        y y2   4   5   6
        z z2   7   8   9
        ```

        * Set index and columns manually:

        ```pycon
        >>> for i in vbt.broadcast(
        ...     v, a, sr, df,
        ...     index_from=['a', 'b', 'c'],
        ...     columns_from=['d', 'e', 'f']
        ... ): print(i)
           d  e  f
        a  0  0  0
        b  0  0  0
        c  0  0  0
           d  e  f
        a  1  2  3
        b  1  2  3
        c  1  2  3
           d  e  f
        a  1  1  1
        b  2  2  2
        c  3  3  3
           d  e  f
        a  1  2  3
        b  4  5  6
        c  7  8  9
        ```

        * Pass arguments as a mapping returns a mapping:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack'
        ... )
        {'v':       a2  b2  c2
              x x2   0   0   0
              y y2   0   0   0
              z z2   0   0   0,
         'a':       a2  b2  c2
              x x2   1   2   3
              y y2   1   2   3
              z z2   1   2   3,
         'sr':       a2  b2  c2
               x x2   1   1   1
               y y2   2   2   2
               z z2   3   3   3,
         'df':       a2  b2  c2
               x x2   1   2   3
               y y2   4   5   6
               z z2   7   8   9}
        ```

        * Keep all results in a format suitable for flexible indexing apart from one:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df),
        ...     index_from='stack',
        ...     keep_flex=dict(_default=True, df=False),
        ...     require_kwargs=dict(df=dict(dtype=float))
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        * Specify arguments per object using `BCO`:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco),
        ...     index_from='stack',
        ...     keep_flex=True
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3]),
         'sr': array([[1],
                      [2],
                      [3]]),
         'df':        a2   b2   c2
               x x2  1.0  2.0  3.0
               y y2  4.0  5.0  6.0
               z z2  7.0  8.0  9.0}
        ```

        * Introduce a parameter that should build a Cartesian product of its values and other objects:

        ```pycon
        >>> df_bco = vbt.BCO(df, keep_flex=False, require_kwargs=dict(dtype=float))
        >>> p_bco = vbt.BCO(pd.Series([1, 2, 3], name='my_p'), product=True)
        >>> vbt.broadcast(
        ...     dict(v=v, a=a, sr=sr, df=df_bco, p=p_bco),
        ...     index_from='stack',
        ...     keep_flex=True
        ... )
        {'v': array([0]),
         'a': array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
         'sr': array([[1],
                [2],
                [3]]),
         'df': my_p        1              2              3
                a2   b2   c2   a2   b2   c2   a2   b2   c2
         x x2  1.0  2.0  3.0  1.0  2.0  3.0  1.0  2.0  3.0
         y y2  4.0  5.0  6.0  4.0  5.0  6.0  4.0  5.0  6.0
         z z2  7.0  8.0  9.0  7.0  8.0  9.0  7.0  8.0  9.0,
         'p': array([[1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3]])}
        ```

        * Build a Cartesian product of all parameters:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.BCO([1, 2, 3], product=True),
        ...         b=vbt.BCO(['x', 'y'], product=True),
        ...         c=vbt.BCO([False, True], product=True)
        ...     )
        ... )
        {'a': array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]),
         'b': array([['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'y']], dtype='<U1'),
         'c': array([[False, True, False, True, False, True, False, True, False, True, False, True]])}
        ```

        * Or the same using `pd.Index`:

        ```pycon
        >>> # or the same globally
        >>> vbt.broadcast(
        ...     dict(
        ...         a=pd.Index([1, 2, 3]),
        ...         b=pd.Index(['x', 'y']),
        ...         c=pd.Index([False, True])
        ...     )
        ... )
        {'a': array([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]]),
         'b': array([['x', 'x', 'y', 'y', 'x', 'x', 'y', 'y', 'x', 'x', 'y', 'y']], dtype='<U1'),
         'c': array([[False, True, False, True, False, True, False, True, False, True, False, True]])}
        ```

        * Build a Cartesian product of two groups of parameters - (a, d) and (b, c):

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=vbt.BCO(pd.Index([1, 2, 3]), product_idx=0),
        ...         b=vbt.BCO(pd.Index(['x', 'y']), product_idx=1),
        ...         d=vbt.BCO(pd.Index([100., 200., 300.]), product_idx=0),
        ...         c=vbt.BCO(pd.Index([False, True]), product_idx=1)
        ...     )
        ... )
        {'a': array([[1, 1, 2, 2, 3, 3]]),
         'b': array([['x', 'y', 'x', 'y', 'x', 'y']], dtype='<U1'),
         'd': array([[100., 100., 200., 200., 300., 300.]]),
         'c': array([[False,  True, False,  True, False,  True]])}
        ```

        * Select a random subset of parameter combinations:

        ```pycon
        >>> vbt.broadcast(
        ...     dict(
        ...         a=pd.Index([1, 2, 3]),
        ...         b=pd.Index(['x', 'y']),
        ...         c=pd.Index([False, True])
        ...     ),
        ...     random_subset=5
        ... )
        {'a': array([[1, 1, 2, 2, 3]]),
         'b': array([['x', 'y', 'x', 'x', 'x']], dtype='<U1'),
         'c': array([[False, False, False,  True,  True]])}
        ```
    """
    # Get defaults
    from vectorbtpro._settings import settings

    broadcasting_cfg = settings["broadcasting"]

    if align_index is None:
        align_index = broadcasting_cfg["align_index"]
    if align_columns is None:
        align_columns = broadcasting_cfg["align_columns"]
    if index_from is None:
        index_from = broadcasting_cfg["index_from"]
    if columns_from is None:
        columns_from = broadcasting_cfg["columns_from"]
    if keep_wrap_default is None:
        keep_wrap_default = broadcasting_cfg["keep_wrap_default"]
    require_kwargs_per_obj = True
    if require_kwargs is not None and checks.is_mapping(require_kwargs):
        require_arg_names = get_func_arg_names(np.require)
        if set(require_kwargs) <= set(require_arg_names):
            require_kwargs_per_obj = False
    if checks.is_mapping(args[0]):
        if len(args) > 1:
            raise ValueError("Only one argument is allowed when passing a mapping")
        all_keys = list(dict(args[0]).keys())
        objs = list(args[0].values())
        return_dict = True
    else:
        objs = list(args)
        all_keys = list(range(len(args)))
        return_dict = False

    def _resolve_arg(obj: tp.Any, arg_name: str, global_value: tp.Any, default_value: tp.Any) -> tp.Any:
        if isinstance(obj, BCO) and getattr(obj, arg_name) is not None:
            return getattr(obj, arg_name)
        if checks.is_mapping(global_value):
            return global_value.get(k, global_value.get("_default", default_value))
        if checks.is_sequence(global_value):
            return global_value[i]
        return global_value

    # Build BCO instances
    none_keys = set()
    default_keys = set()
    product_keys = set()
    bco_instances = {}
    pool = dict(zip(all_keys, objs))
    for i, k in enumerate(all_keys):
        obj = objs[i]

        if isinstance(obj, Default):
            obj = obj.value
            default_keys.add(k)
        if isinstance(obj, Ref):
            obj = resolve_ref(pool, k)
        if isinstance(obj, BCO):
            value = obj.value
        else:
            value = obj
        if isinstance(value, Default):
            value = value.value
            default_keys.add(k)
        if isinstance(value, Ref):
            value = resolve_ref(pool, k, inside_bco=True)
        if value is None:
            none_keys.add(k)
            continue
        value_is_index = checks.is_index(value)
        value = to_any_array(value)

        _to_pd = _resolve_arg(obj, "to_pd", to_pd, None)

        _keep_flex = _resolve_arg(obj, "keep_flex", keep_flex, None)
        if _keep_flex is None:
            _keep_flex = broadcasting_cfg["keep_flex"]

        _min_one_dim = _resolve_arg(obj, "min_one_dim", min_one_dim, None)
        if _min_one_dim is None:
            _min_one_dim = broadcasting_cfg["min_one_dim"]

        _post_func = _resolve_arg(obj, "post_func", post_func, None)

        if isinstance(obj, BCO) and obj.require_kwargs is not None:
            _require_kwargs = obj.require_kwargs
        else:
            _require_kwargs = None
        if checks.is_mapping(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(
                require_kwargs.get("_default", None),
                require_kwargs.get(k, None),
                _require_kwargs,
            )
        elif checks.is_sequence(require_kwargs) and require_kwargs_per_obj:
            _require_kwargs = merge_dicts(require_kwargs[i], _require_kwargs)
        else:
            _require_kwargs = merge_dicts(require_kwargs, _require_kwargs)

        _index_to_product = _resolve_arg(obj, "index_to_product", index_to_product, None)
        if _index_to_product is None:
            _index_to_product = broadcasting_cfg["index_to_product"]

        _product = _resolve_arg(obj, "product", product, None)
        if _product is None:
            _product = _index_to_product and value_is_index
        if _product:
            product_keys.add(k)

        _product_idx = _resolve_arg(obj, "product_idx", product_idx, None)

        _repeat_product = _resolve_arg(obj, "repeat_product", repeat_product, None)
        if _repeat_product is None:
            _repeat_product = broadcasting_cfg["repeat_product"]

        _keys_from_sr_index = _resolve_arg(obj, "keys_from_sr_index", keys_from_sr_index, None)
        if _keys_from_sr_index is None:
            _keys_from_sr_index = broadcasting_cfg["keys_from_sr_index"]

        _keys = _resolve_arg(obj, "keys", keys, None)
        if _product:
            if _keys is None:
                if _keys_from_sr_index and checks.is_series(value) and not checks.is_default_index(value.index):
                    _keys = value.index
                else:
                    _keys = value
            if _keys is not None:
                _keys = indexes.to_any_index(_keys)
                if not checks.is_multi_index(_keys):
                    if _keys.name is None and hasattr(value, "name"):
                        _keys = _keys.rename(value.name)
                    if _keys.name is None:
                        _keys = _keys.rename(k)

        bco_instances[k] = BCO(
            value,
            to_pd=_to_pd,
            keep_flex=_keep_flex,
            min_one_dim=_min_one_dim,
            post_func=_post_func,
            require_kwargs=_require_kwargs,
            index_to_product=_index_to_product,
            product=_product,
            product_idx=_product_idx,
            repeat_product=_repeat_product,
            keys_from_sr_index=_keys_from_sr_index,
            keys=_keys,
        )

    # Check whether we should broadcast Pandas metadata and work on 2-dim data
    is_pd = False
    is_2d = False

    old_objs = []
    for k, bco_obj in bco_instances.items():
        if k in none_keys or k in product_keys:
            continue

        obj = bco_obj.value
        if obj.ndim > 1:
            is_2d = True
        if checks.is_pandas(obj):
            is_pd = True
        if bco_obj.to_pd is not None and bco_obj.to_pd:
            is_pd = True
        old_objs.append(obj)

    if to_shape is not None:
        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        if len(to_shape) > 1:
            is_2d = True

    if to_frame is not None:
        is_2d = to_frame

    if to_pd is not None:
        is_pd = to_pd or (return_wrapper and is_pd)

    # Align pandas arrays
    old_objs = align_pd_arrays(old_objs, align_index=align_index, align_columns=align_columns)

    # Convert all pd.Series objects to pd.DataFrame if we work on 2-dim data
    ready_objs = []
    for obj in old_objs:
        if is_2d and checks.is_series(obj):
            ready_objs.append(obj.values[:, None])
        else:
            ready_objs.append(np.asarray(obj))

    # Get final shape
    if to_shape is None:
        try:
            to_shape = _broadcast_shape(*ready_objs)
        except ValueError:
            arr_shapes = {}
            for i, k in enumerate(bco_instances):
                if k in none_keys or k in product_keys:
                    continue

                if len(ready_objs[i].shape) > 0:
                    arr_shapes[k] = ready_objs[i].shape
            raise ValueError("Could not broadcast shapes: %s" % str(arr_shapes))
    if not isinstance(to_shape, tuple):
        to_shape = (to_shape,)
    if len(to_shape) == 0:
        to_shape = (1,)
    shape_2d = to_shape if len(to_shape) > 1 else (*to_shape, 1)

    if is_pd:
        # Decide on index and columns
        # NOTE: Important to pass old_objs, not ready_objs, to preserve original shape info
        new_index = broadcast_index(
            old_objs,
            to_shape,
            index_from=index_from,
            axis=0,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **stack_kwargs,
        )
        new_columns = broadcast_index(
            old_objs,
            to_shape,
            index_from=columns_from,
            axis=1,
            ignore_sr_names=ignore_sr_names,
            ignore_ranges=ignore_ranges,
            check_index_names=check_index_names,
            **stack_kwargs,
        )
    else:
        new_index, new_columns = None, None

    # Build a product
    param_product = None
    param_columns = None
    n_params = 0
    if len(product_keys) > 0:
        # Prepare and group parameters
        product_idx_values = defaultdict(OrderedDict)
        product_indexes = OrderedDict()
        product_idx_seen = False
        curr_idx = 0
        max_idx = 0
        for k, bco_obj in bco_instances.items():
            if k not in product_keys:
                continue
            if bco_obj.product_idx is None:
                if product_idx_seen:
                    raise ValueError("Please provide product index for all product parameters")
                product_idx = curr_idx
            else:
                if curr_idx > 0 and not product_idx_seen:
                    raise ValueError("Please provide product index for all product parameters")
                product_idx_seen = True
                product_idx = bco_obj.product_idx
            if product_idx > max_idx:
                max_idx = product_idx

            value = np.asarray(bco_obj.value)
            if value.ndim == 0:
                value = value[None]
            elif value.ndim > 1:
                raise ValueError(f"Product parameter '{k}' cannot be multi-dimensional")
            product_idx_values[product_idx][k] = value
            product_indexes[k] = bco_obj.keys
            curr_idx += 1

        # Build an operation tree and parameter columns
        op_tree_operands = []
        param_keys = []
        for product_idx in range(max_idx + 1):
            if product_idx not in product_idx_values:
                raise ValueError("Group index must come in a strict order starting with 0 and without gaps")
            for k in product_idx_values[product_idx].keys():
                param_keys.append(k)

            # Broadcast parameter arrays
            param_arrays = tuple(product_idx_values[product_idx].values())
            if len(param_arrays) > 1:
                param_arrays = np.broadcast_arrays(*param_arrays)
                op_tree_operands.append((zip, *param_arrays))
            else:
                op_tree_operands.append(param_arrays[0])

            # Stack or combine parameter columns together
            if new_columns is not None:
                levels = []
                for k in product_idx_values[product_idx].keys():
                    levels.append(product_indexes[k])
                if len(levels) > 1:
                    _param_columns = indexes.stack_indexes(levels, **stack_kwargs)
                else:
                    _param_columns = levels[0]
                if param_columns is None:
                    param_columns = _param_columns
                else:
                    param_columns = indexes.combine_indexes(
                        [param_columns, _param_columns],
                        ignore_ranges=ignore_ranges,
                        **stack_kwargs,
                    )

        # Generate parameter combinations using the operation tree
        if len(op_tree_operands) > 1:
            param_product = dict(zip(param_keys, generate_param_combs((itertools.product, *op_tree_operands))))
        elif isinstance(op_tree_operands[0], tuple):
            param_product = dict(zip(param_keys, generate_param_combs(op_tree_operands[0])))
        else:
            param_product = dict(zip(param_keys, op_tree_operands))
        n_params = len(param_product[param_keys[0]])

        # Select random subset
        if random_subset is not None:
            random_indices = np.sort(np.random.permutation(np.arange(n_params))[:random_subset])
            n_params = len(random_indices)
            param_product = {k: np.asarray(v)[random_indices] for k, v in param_product.items()}
            if param_columns is not None:
                param_columns = param_columns[random_indices]

        # Combine parameter columns with new columns
        if param_columns is not None and new_columns is not None:
            new_columns = indexes.combine_indexes(
                [param_columns, new_columns],
                ignore_ranges=ignore_ranges,
                **stack_kwargs,
            )

    # Perform broadcasting
    new_objs = []
    for i, k in enumerate(all_keys):
        if k in none_keys:
            continue
        _keep_flex = bco_instances[k].keep_flex
        _repeat_product = bco_instances[k].repeat_product

        if k in product_keys:
            # Broadcast parameters
            obj = param_product[k]
            if _repeat_product:
                obj = np.repeat(obj, shape_2d[1])
            if not _keep_flex:
                if _repeat_product:
                    obj = np.broadcast_to(obj, (to_shape[0], len(obj)))
                else:
                    obj = np.broadcast_to(obj, (to_shape[0], n_params))
            old_obj = obj
            new_obj = obj
        else:
            # Broadcast regular objects
            old_obj = old_objs.pop(0)
            new_obj = ready_objs.pop(0)
            _min_one_dim = bco_instances[k].min_one_dim
            if _min_one_dim and new_obj.ndim == 0:
                new_obj = new_obj[None]
            if _keep_flex:
                if n_params > 0:
                    if len(to_shape) == 1:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = new_obj[:, None]  # product changes is_2d behavior
                    else:
                        if new_obj.ndim == 1 and new_obj.shape[0] > 1:
                            new_obj = np.tile(new_obj, n_params)
                        elif new_obj.ndim == 2 and new_obj.shape[1] > 1:
                            new_obj = np.tile(new_obj, (1, n_params))
            else:
                new_obj = np.broadcast_to(new_obj, to_shape)
                if n_params > 0:
                    if new_obj.ndim == 1:
                        new_obj = new_obj[:, None]  # product changes is_2d behavior
                    new_obj = np.tile(new_obj, (1, n_params))

        # Force to match requirements
        _require_kwargs = bco_instances[k].require_kwargs
        new_obj = np.require(new_obj, **resolve_dict(_require_kwargs))

        # Bring arrays to their old types (e.g. array -> pandas)
        if _keep_flex:
            new_objs.append(new_obj)
        else:
            _is_pd = bco_instances[k].to_pd
            if _is_pd is None:
                _is_pd = is_pd
            if k in product_keys and not _repeat_product:
                wrapped_arr = wrap_broadcasted(
                    old_obj,
                    new_obj,
                    is_pd=_is_pd,
                    new_index=new_index,
                    new_columns=param_columns,
                    ignore_ranges=ignore_ranges,
                )
            else:
                wrapped_arr = wrap_broadcasted(
                    old_obj,
                    new_obj,
                    is_pd=_is_pd,
                    new_index=new_index,
                    new_columns=new_columns,
                    ignore_ranges=ignore_ranges,
                )
            _post_func = bco_instances[k].post_func
            if _post_func is not None:
                wrapped_arr = _post_func(wrapped_arr)
            new_objs.append(wrapped_arr)

    # Prepare outputs
    return_objs = []
    for k in all_keys:
        if k not in none_keys:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(new_objs.pop(0)))
            else:
                return_objs.append(new_objs.pop(0))
        else:
            if k in default_keys and keep_wrap_default:
                return_objs.append(Default(None))
            else:
                return_objs.append(None)
    if return_dict:
        return_objs = dict(zip(all_keys, return_objs))
    else:
        return_objs = tuple(return_objs)
    if return_wrapper:
        wrapper = wrapping.ArrayWrapper.from_shape(
            to_shape,
            index=new_index,
            columns=new_columns,
            **resolve_dict(wrapper_kwargs),
        )
    if len(return_objs) > 1 or return_dict:
        if return_wrapper:
            return return_objs, wrapper
        return return_objs
    if return_wrapper:
        return return_objs[0], wrapper
    return return_objs[0]


def broadcast_to(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    to_pd: tp.Optional[bool] = None,
    index_from: tp.Optional[IndexFromLike] = None,
    columns_from: tp.Optional[IndexFromLike] = None,
    **kwargs,
) -> tp.Any:
    """Broadcast `arg1` to `arg2`.

    Pass None to `index_from`/`columns_from` to use index/columns of the second argument.

    Keyword arguments `**kwargs` are passed to `broadcast`.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import broadcast_to

        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')

        >>> broadcast_to(a, sr)
        x    1
        y    2
        z    3
        Name: a, dtype: int64

        >>> broadcast_to(sr, a)
        array([4, 5, 6])
        ```
    """
    arg1 = to_any_array(arg1)
    arg2 = to_any_array(arg2)
    if to_pd is None:
        to_pd = checks.is_pandas(arg2)
    if to_pd:
        # Take index and columns from arg2
        if index_from is None:
            index_from = indexes.get_index(arg2, 0)
        if columns_from is None:
            columns_from = indexes.get_index(arg2, 1)
    return broadcast(arg1, to_shape=arg2.shape, to_pd=to_pd, index_from=index_from, columns_from=columns_from, **kwargs)


def broadcast_to_array_of(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> tp.Array:
    """Broadcast `arg1` to the shape `(1, *arg2.shape)`.

    `arg1` must be either a scalar, a 1-dim array, or have 1 dimension more than `arg2`.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.base.reshaping import broadcast_to_array_of

        >>> broadcast_to_array_of([0.1, 0.2], np.empty((2, 2)))
        [[[0.1 0.1]
          [0.1 0.1]]

         [[0.2 0.2]
          [0.2 0.2]]]
        ```
    """
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
    # From here on arg1 can be only a 1-dim array
    if arg1.ndim == 0:
        arg1 = to_1d(arg1)
    checks.assert_ndim(arg1, 1)

    if arg2.ndim == 0:
        return arg1
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
    return np.tile(arg1, (1, *arg2.shape))


def broadcast_to_axis_of(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    axis: int,
    require_kwargs: tp.KwargsLike = None,
) -> tp.Array:
    """Broadcast `arg1` to an axis of `arg2`.

    If `arg2` has less dimensions than requested, will broadcast `arg1` to a single number.

    For other keyword arguments, see `broadcast`."""
    if require_kwargs is None:
        require_kwargs = {}
    arg2 = to_any_array(arg2)
    if arg2.ndim < axis + 1:
        return np.broadcast_to(arg1, (1,))[0]  # to a single number
    arg1 = np.broadcast_to(arg1, (arg2.shape[axis],))
    arg1 = np.require(arg1, **require_kwargs)
    return arg1


def broadcast_combs(
    *args: tp.ArrayLike,
    axis: int = 1,
    comb_func: tp.Callable = itertools.product,
    broadcast_kwargs: tp.KwargsLike = None,
) -> tp.Any:
    """Align an axis of each array using a combinatoric function and broadcast their indexes.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> from vectorbtpro.base.reshaping import broadcast_combs

        >>> df = pd.DataFrame([[1, 2, 3], [3, 4, 5]], columns=pd.Index(['a', 'b', 'c'], name='df_param'))
        >>> df2 = pd.DataFrame([[6, 7], [8, 9]], columns=pd.Index(['d', 'e'], name='df2_param'))
        >>> sr = pd.Series([10, 11], name='f')

        >>> new_df, new_df2, new_sr = broadcast_combs((df, df2, sr))

        >>> new_df
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          1  1  2  2  3  3
        1          3  3  4  4  5  5

        >>> new_df2
        df_param   a     b     c
        df2_param  d  e  d  e  d  e
        0          6  7  6  7  6  7
        1          8  9  8  9  8  9

        >>> new_sr
        df_param    a       b       c
        df2_param   d   e   d   e   d   e
        0          10  10  10  10  10  10
        1          11  11  11  11  11  11
        ```
    """
    if broadcast_kwargs is None:
        broadcast_kwargs = {}

    args = list(args)
    if len(args) < 2:
        raise ValueError("At least two arguments are required")
    for i in range(len(args)):
        arg = to_any_array(args[i])
        if axis == 1:
            arg = to_2d(arg)
        args[i] = arg
    indices = []
    for arg in args:
        indices.append(np.arange(len(indexes.get_index(to_pd_array(arg), axis))))
    new_indices = list(map(list, zip(*list(comb_func(*indices)))))
    results = []
    for i, arg in enumerate(args):
        if axis == 1:
            if checks.is_pandas(arg):
                results.append(arg.iloc[:, new_indices[i]])
            else:
                results.append(arg[:, new_indices[i]])
        else:
            if checks.is_pandas(arg):
                results.append(arg.iloc[new_indices[i]])
            else:
                results.append(arg[new_indices[i]])
    if axis == 1:
        broadcast_kwargs = merge_dicts(dict(columns_from="stack"), broadcast_kwargs)
    else:
        broadcast_kwargs = merge_dicts(dict(index_from="stack"), broadcast_kwargs)
    return broadcast(*results, **broadcast_kwargs)


def get_multiindex_series(arg: tp.SeriesFrame) -> tp.Series:
    """Get Series with a multi-index.

    If DataFrame has been passed, must at maximum have one row or column."""
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    if checks.is_frame(arg):
        if arg.shape[0] == 1:
            arg = arg.iloc[0, :]
        elif arg.shape[1] == 1:
            arg = arg.iloc[:, 0]
        else:
            raise ValueError("Supported are either Series or DataFrame with one column/row")
    checks.assert_instance_of(arg.index, pd.MultiIndex)
    return arg


def unstack_to_array(arg: tp.SeriesFrame, levels: tp.Optional[tp.MaybeLevelSequence] = None) -> tp.Array:
    """Reshape `arg` based on its multi-index into a multi-dimensional array.

    Use `levels` to specify what index levels to unstack and in which order.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import unstack_to_array

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_array(sr).shape
        (2, 2, 4)

        >>> unstack_to_array(sr)
        [[[ 1. nan nan nan]
         [nan  2. nan nan]]

         [[nan nan  3. nan]
        [nan nan nan  4.]]]

        >>> unstack_to_array(sr, levels=(2, 0))
        [[ 1. nan]
         [ 2. nan]
         [nan  3.]
         [nan  4.]]
        ```
    """
    # Extract series
    sr: tp.Series = to_1d(get_multiindex_series(arg))
    if sr.index.duplicated().any():
        raise ValueError("Index contains duplicate entries, cannot reshape")

    unique_idx_list = []
    vals_idx_list = []
    if levels is None:
        levels = range(sr.index.nlevels)
    if isinstance(levels, (int, str)):
        levels = (levels,)
    for level in levels:
        vals = indexes.select_levels(sr.index, level).to_numpy()
        unique_vals = np.unique(vals)
        unique_idx_list.append(unique_vals)
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        vals_idx = list(map(lambda x: idx_map[x], vals))
        vals_idx_list.append(vals_idx)

    a = np.full(list(map(len, unique_idx_list)), np.nan)
    a[tuple(zip(vals_idx_list))] = sr.values
    return a


def make_symmetric(arg: tp.SeriesFrame, sort: bool = True) -> tp.Frame:
    """Make `arg` symmetric.

    The index and columns of the resulting DataFrame will be identical.

    Requires the index and columns to have the same number of levels.

    Pass `sort=False` if index and columns should not be sorted, but concatenated
    and get duplicates removed.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import make_symmetric

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])

        >>> make_symmetric(df)
             a    b    c    d
        a  NaN  NaN  1.0  2.0
        b  NaN  NaN  3.0  4.0
        c  1.0  3.0  NaN  NaN
        d  2.0  4.0  NaN  NaN
        ```
    """
    checks.assert_instance_of(arg, (pd.Series, pd.DataFrame))
    df: tp.Frame = to_2d(arg)
    if isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        checks.assert_instance_of(df.index, pd.MultiIndex)
        checks.assert_instance_of(df.columns, pd.MultiIndex)
        checks.assert_array_equal(df.index.nlevels, df.columns.nlevels)
        names1, names2 = tuple(df.index.names), tuple(df.columns.names)
    else:
        names1, names2 = df.index.name, df.columns.name

    if names1 == names2:
        new_name = names1
    else:
        if isinstance(df.index, pd.MultiIndex):
            new_name = tuple(zip(*[names1, names2]))
        else:
            new_name = (names1, names2)
    if sort:
        idx_vals = np.unique(np.concatenate((df.index, df.columns))).tolist()
    else:
        idx_vals = list(dict.fromkeys(np.concatenate((df.index, df.columns))))
    df_index = df.index.copy()
    df_columns = df.columns.copy()
    if isinstance(df.index, pd.MultiIndex):
        unique_index = pd.MultiIndex.from_tuples(idx_vals, names=new_name)
        df_index.names = new_name
        df_columns.names = new_name
    else:
        unique_index = pd.Index(idx_vals, name=new_name)
        df_index.name = new_name
        df_columns.name = new_name
    df = df.copy(deep=False)
    df.index = df_index
    df.columns = df_columns
    df_out_dtype = np.promote_types(df.values.dtype, np.min_scalar_type(np.nan))
    df_out = pd.DataFrame(index=unique_index, columns=unique_index, dtype=df_out_dtype)
    df_out.loc[:, :] = df
    df_out[df_out.isnull()] = df.transpose()
    return df_out


def unstack_to_df(
    arg: tp.SeriesFrame,
    index_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    column_levels: tp.Optional[tp.MaybeLevelSequence] = None,
    symmetric: bool = False,
    sort: bool = True,
) -> tp.Frame:
    """Reshape `arg` based on its multi-index into a DataFrame.

    Use `index_levels` to specify what index levels will form new index, and `column_levels`
    for new columns. Set `symmetric` to True to make DataFrame symmetric.

    Usage:
        ```pycon
        >>> import pandas as pd
        >>> from vectorbtpro.base.reshaping import unstack_to_df

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']],
        ...     names=['x', 'y', 'z'])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> unstack_to_df(sr, index_levels=(0, 1), column_levels=2)
        z      a    b    c    d
        x y
        1 3  1.0  NaN  NaN  NaN
        1 4  NaN  2.0  NaN  NaN
        2 3  NaN  NaN  3.0  NaN
        2 4  NaN  NaN  NaN  4.0
        ```
    """
    # Extract series
    sr: tp.Series = to_1d(get_multiindex_series(arg))

    if len(sr.index.levels) > 2:
        if index_levels is None:
            raise ValueError("index_levels must be specified")
        if column_levels is None:
            raise ValueError("column_levels must be specified")
    else:
        if index_levels is None:
            index_levels = 0
        if column_levels is None:
            column_levels = 1

    # Build new index and column hierarchies
    new_index = indexes.select_levels(arg.index, index_levels).unique()
    new_columns = indexes.select_levels(arg.index, column_levels).unique()

    # Unstack and post-process
    unstacked = unstack_to_array(sr, levels=(index_levels, column_levels))
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    if symmetric:
        return make_symmetric(df, sort=sort)
    return df
