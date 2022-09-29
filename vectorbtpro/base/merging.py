# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Functions for merging arrays."""

from functools import partial

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import resolve_dict, merge_dicts
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.reshaping import column_stack


def concat_merge(
    *objs,
    wrap: tp.Optional[bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    keys: tp.Optional[tp.Index] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like objects through concatenation.

    Supports a sequence of tuples.

    If `wrap` is None, it will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None.
    If `wrap` is True, each array will be wrapped with Pandas Series and merged using `pd.concat`.
    Otherwise, arrays will be kept as-is and merged using `np.concatenate`.
    `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`.

    Keyword arguments `**kwargs` are passed to `pd.concat` only.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                concat_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: concat_merge(
                    x,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )
    if isinstance(objs[0], Wrapping):
        raise TypeError("Concatenating Wrapping instances is not supported")

    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if not checks.is_iterable(objs[0]) or isinstance(objs[0], str):
        if wrap:
            wrap_kwargs = merge_dicts(dict(index=keys), wrap_kwargs)
            return pd.Series(objs, **wrap_kwargs)
        return np.asarray(objs)
    if not isinstance(objs[0], pd.Series):
        if isinstance(objs[0], pd.DataFrame):
            raise ValueError("Use row stacking for concatenating DataFrames")
        if wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap_reduced(obj, **_wrap_kwargs))
                else:
                    new_objs.append(pd.Series(obj, **_wrap_kwargs))
            objs = new_objs
        else:
            return np.concatenate(objs)
    return pd.concat(objs, axis=0, keys=keys, **kwargs)


def row_stack_merge(
    *objs,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    keys: tp.Optional[tp.Index] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects through row stacking.

    Supports a sequence of tuples.

    Argument `wrap` supports the following options:

    * None: will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None
    * True: each array will be wrapped with Pandas Series/DataFrame (depending on dimensions)
    * 'sr', 'series': each array will be wrapped with Pandas Series
    * 'df', 'frame', 'dataframe': each array will be wrapped with Pandas DataFrame

    Argument `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

    Keyword arguments `**kwargs` are passed to `pd.concat` and
    `vectorbtpro.base.wrapping.Wrapping.row_stack` only.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                row_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: row_stack_merge(
                    x,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )

    if isinstance(objs[0], Wrapping):
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).row_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option '{wrap}'")
            objs = new_objs
        else:
            return np.row_stack(objs)
    return pd.concat(objs, axis=0, keys=keys, **kwargs)


def column_stack_merge(
    *objs,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    keys: tp.Optional[tp.Index] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge multiple array-like or `vectorbtpro.base.wrapping.Wrapping` objects through column stacking.

    Supports a sequence of tuples.

    Argument `wrap` supports the following options:

    * None: will become True if `wrapper`, `keys`, or `wrap_kwargs` are not None
    * True: each array will be wrapped with Pandas Series/DataFrame (depending on dimensions)
    * 'sr', 'series': each array will be wrapped with Pandas Series
    * 'df', 'frame', 'dataframe': each array will be wrapped with Pandas DataFrame

    Argument `wrap_kwargs` can be a dictionary or a list of dictionaries.

    If `wrapper` is provided, will use `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.

    Keyword arguments `**kwargs` are passed to `pd.concat` and
    `vectorbtpro.base.wrapping.Wrapping.column_stack` only.

    !!! note
        All arrays are assumed to have the same type and dimensionality."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)

    if isinstance(objs[0], tuple):
        if len(objs[0]) == 1:
            return (
                column_stack_merge(
                    list(map(lambda x: x[0], objs)),
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
            )
        return tuple(
            map(
                lambda x: column_stack_merge(
                    x,
                    keys=keys,
                    wrap=wrap,
                    wrapper=wrapper,
                    wrap_kwargs=wrap_kwargs,
                    **kwargs,
                ),
                zip(*objs),
            )
        )

    if isinstance(objs[0], Wrapping):
        kwargs = merge_dicts(dict(wrapper_kwargs=dict(keys=keys)), kwargs)
        return type(objs[0]).column_stack(objs, **kwargs)
    if wrap_kwargs is None:
        wrap_kwargs = {}
    if wrap is None:
        wrap = wrapper is not None or keys is not None or len(wrap_kwargs) > 0
    if not isinstance(objs[0], (pd.Series, pd.DataFrame)):
        if isinstance(wrap, str) or wrap:
            new_objs = []
            for i, obj in enumerate(objs):
                _wrap_kwargs = resolve_dict(wrap_kwargs, i)
                if wrapper is not None:
                    new_objs.append(wrapper.wrap(obj, **_wrap_kwargs))
                else:
                    if not isinstance(wrap, str):
                        if isinstance(obj, np.ndarray):
                            ndim = obj.ndim
                        else:
                            ndim = np.asarray(obj).ndim
                        if ndim == 1:
                            wrap = "series"
                        else:
                            wrap = "frame"
                    if isinstance(wrap, str):
                        if wrap.lower() in ("sr", "series"):
                            new_objs.append(pd.Series(obj, **_wrap_kwargs))
                        elif wrap.lower() in ("df", "frame", "dataframe"):
                            new_objs.append(pd.DataFrame(obj, **_wrap_kwargs))
                        else:
                            raise ValueError(f"Invalid wrapping option '{wrap}'")
            objs = new_objs
        else:
            return column_stack(objs)
    return pd.concat(objs, axis=1, keys=keys, **kwargs)


def mixed_merge(
    *objs,
    func_names: tp.Optional[tp.Tuple[str, ...]] = None,
    wrap: tp.Union[None, str, bool] = None,
    wrapper: tp.Optional[ArrayWrapper] = None,
    wrap_kwargs: tp.KwargsLikeSequence = None,
    keys: tp.Optional[tp.Index] = None,
    **kwargs,
) -> tp.MaybeTuple[tp.AnyArray]:
    """Merge objects of mixed types."""
    if len(objs) == 1:
        objs = objs[0]
    objs = list(objs)
    if func_names is None:
        raise ValueError("Merging function names are required")
    if not isinstance(objs[0], tuple):
        raise ValueError("Mixed merging must be applied on tuples")

    outputs = []
    for i, obj_kind in enumerate(zip(*objs)):
        outputs.append(resolve_merge_func(func_names[i])(
            obj_kind,
            keys=keys,
            wrap=wrap,
            wrapper=wrapper,
            wrap_kwargs=wrap_kwargs,
            **kwargs,
        ))
    return tuple(outputs)


def resolve_merge_func(func_name: tp.MaybeTuple[str]) -> tp.Callable:
    """Resolve merging function based on name."""
    if isinstance(func_name, tuple):
        return partial(mixed_merge, func_names=func_name)
    if func_name.lower() == "concat":
        return concat_merge
    if func_name.lower() == "row_stack":
        return row_stack_merge
    if func_name.lower() == "column_stack":
        return column_stack_merge
    raise ValueError(f"Invalid merging function name '{func_name}'")
