# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Decorators for splitting."""

import inspect
from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, merge_dicts
from vectorbtpro.utils.parsing import (
    annotate_args,
    flatten_ann_args,
    unflatten_ann_args,
    ann_args_to_args,
    match_ann_arg,
)
from vectorbtpro.generic.splitting.base import Splitter, Takeable


def split(
    *args,
    splitter: tp.Optional[Splitter] = None,
    splitter_cls: tp.Type[Splitter] = Splitter,
    method: tp.Optional[str] = None,
    method_args: tp.ArgsLike = None,
    method_kwargs: tp.KwargsLike = None,
    index: tp.Optional[tp.IndexLike] = None,
    index_from: tp.Optional[tp.AnnArgQuery] = None,
    takeable_args: tp.Optional[tp.MaybeIterable[tp.AnnArgQuery]] = None,
    template_context: tp.KwargsLike = None,
    **apply_kwargs,
) -> tp.Callable:
    """Decorator that splits the inputs of a function.

    Does the following:

    1. Resolves the splitter of the type `vectorbtpro.generic.splitting.base.Splitter`
    either by using an already provided splitter instance in `splitter`, or by running a splitter method
    (`method`) while passing `index`, `*method_args`, and `**method_kwargs`. Index is getting
    resolved either using an already provided `index`, by parsing the argument under a name/position
    provided in `index_from`, or by parsing the first argument from `takeable_args` (in this order).
    2. Wraps arguments in `takeable_args` with `vectorbtpro.generic.splitting.base.Takeable`
    3. Runs `vectorbtpro.generic.splitting.base.Splitter.apply` with arguments passed
    to the function as `args` and `kwargs`, but also `**apply_kwargs`

    Usage:
        * Split a Series and return its sum:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import numpy as np
        >>> import pandas as pd

        >>> @vbt.split(
        ...     method="from_n_rolling",
        ...     method_kwargs=dict(n=2),
        ...     takeable_args=["sr"]
        ... )
        ... def f(sr):
        ...     return sr.sum()

        >>> index = pd.date_range("2020-01-01", "2020-01-06")
        >>> sr = pd.Series(np.arange(len(index)), index=index)
        >>> f(sr)
        split  set
        0      set_0     3
        1      set_0    12
        dtype: int64
        ```

        * Perform a split manually:

        ```pycon
        >>> @vbt.split(
        ...     method="from_n_rolling",
        ...     method_kwargs=dict(n=2),
        ...     takeable_args=["index"]
        ... )
        ... def f(index, sr):
        ...     return sr[index].sum()

        >>> f(index, sr)
        split  set
        0      set_0     3
        1      set_0    12
        dtype: int64
        ```

        * Construct splitter and mark arguments as "takeable" manually:

        ```pycon
        >>> splitter = vbt.Splitter.from_n_rolling(index, n=2)
        >>> @vbt.split(splitter=splitter)
        ... def f(sr):
        ...     return sr.sum()

        >>> f(vbt.Takeable(sr))
        split  set
        0      set_0     3
        1      set_0    12
        dtype: int64
        ```

        * Split multiple timeframes using a custom index:

        ```pycon
        >>> @vbt.split(
        ...     method="from_n_rolling",
        ...     method_kwargs=dict(n=2),
        ...     index=index,
        ...     takeable_args=["h12_sr", "d2_sr"]
        ... )
        ... def f(h12_sr, d2_sr):
        ...     return h12_sr.sum() + d2_sr.sum()

        >>> h12_index = pd.date_range("2020-01-01", "2020-01-06", freq="12H")
        >>> d2_index = pd.date_range("2020-01-01", "2020-01-06", freq="2D")
        >>> h12_sr = pd.Series(np.arange(len(h12_index)), index=h12_index)
        >>> d2_sr = pd.Series(np.arange(len(d2_index)), index=d2_index)
        >>> f(h12_sr, d2_sr)
        split  set
        0      set_0    15
        1      set_0    42
        dtype: int64
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            splitter = kwargs.pop("_splitter", wrapper.options["splitter"])
            splitter_cls = kwargs.pop("_splitter_cls", wrapper.options["splitter_cls"])
            method = kwargs.pop("_method", wrapper.options["method"])
            method_args = kwargs.pop("_method_args", wrapper.options["method_args"])
            if method_args is None:
                method_args = ()
            method_kwargs = merge_dicts(wrapper.options["method_kwargs"], kwargs.pop("_method_kwargs", {}))
            index = kwargs.pop("_index", wrapper.options["index"])
            index_from = kwargs.pop("_index_from", wrapper.options["index_from"])
            takeable_args = kwargs.pop("_takeable_args", wrapper.options["takeable_args"])
            if takeable_args is None:
                takeable_args = set()
            elif checks.is_iterable(takeable_args) and not isinstance(takeable_args, str):
                takeable_args = set(takeable_args)
            else:
                takeable_args = {takeable_args}
            template_context = merge_dicts(wrapper.options["template_context"], kwargs.pop("_template_context", {}))
            apply_kwargs = merge_dicts(wrapper.options["apply_kwargs"], kwargs.pop("_apply_kwargs", {}))

            ann_args = annotate_args(func, args, kwargs)
            if splitter is None:
                if method is not None:
                    if index is None and index_from is not None:
                        index = splitter_cls.get_obj_index(match_ann_arg(ann_args, index_from))
                    if index is None and len(takeable_args) > 0:
                        index = splitter_cls.get_obj_index(match_ann_arg(ann_args, list(takeable_args)[0]))
                    if index is None:
                        raise ValueError("Must provide splitter, index, index_from, or takeable_args")
                    if isinstance(method, str):
                        method = getattr(splitter_cls, method)
                    splitter = method(
                        index,
                        *method_args,
                        template_context=template_context,
                        **method_kwargs,
                    )
                else:
                    raise ValueError("Must provide splitter or method")
            if len(takeable_args) > 0:
                flat_ann_args = flatten_ann_args(ann_args)
                for takeable_arg in takeable_args:
                    arg_name = match_ann_arg(ann_args, takeable_arg, return_name=True)
                    if not isinstance(flat_ann_args[arg_name]["value"], Takeable):
                        flat_ann_args[arg_name]["value"] = Takeable(flat_ann_args[arg_name]["value"])
                new_ann_args = unflatten_ann_args(flat_ann_args)
                args, kwargs = ann_args_to_args(new_ann_args)
            return splitter.apply(
                func,
                *args,
                **kwargs,
                **apply_kwargs,
            )

        wrapper.options = Config(
            dict(
                splitter=splitter,
                splitter_cls=splitter_cls,
                method=method,
                method_args=method_args,
                method_kwargs=method_kwargs,
                index=index,
                index_from=index_from,
                takeable_args=takeable_args,
                template_context=template_context,
                apply_kwargs=apply_kwargs,
            ),
            frozen_keys_=True,
            as_attrs_=True,
        )
        signature = inspect.signature(wrapper)
        lists_var_kwargs = False
        for k, v in signature.parameters.items():
            if v.kind == v.VAR_KEYWORD:
                lists_var_kwargs = True
                break
        if not lists_var_kwargs:
            var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
            wrapper.__signature__ = signature.replace(parameters=new_parameters)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
