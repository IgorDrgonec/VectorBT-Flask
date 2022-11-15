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
from vectorbtpro.utils.params import parameterized
from vectorbtpro.utils.template import Rep, deep_substitute
from vectorbtpro.generic.splitting.base import Splitter, Takeable


def split(
    *args,
    splitter: tp.Union[None, str, Splitter, tp.Callable] = None,
    splitter_cls: tp.Type[Splitter] = Splitter,
    splitter_kwargs: tp.KwargsLike = None,
    index: tp.Optional[tp.IndexLike] = None,
    index_from: tp.Optional[tp.AnnArgQuery] = None,
    takeable_args: tp.Optional[tp.MaybeIterable[tp.AnnArgQuery]] = None,
    template_context: tp.KwargsLike = None,
    **apply_kwargs,
) -> tp.Callable:
    """Decorator that splits the inputs of a function.

    Does the following:

    1. Resolves the splitter of the type `vectorbtpro.generic.splitting.base.Splitter` using
    the argument `splitter`. It can be either an already provided splitter instance, the name of
    splitter class method, or an arbitrary callable. If any of the latter, it will pass `index` and
    `**splitter_kwargs`. Index is getting resolved either using an already provided `index`,
    by parsing the argument under a name/position provided in `index_from`, or by parsing the
    first argument from `takeable_args` (in this order).
    2. Wraps arguments in `takeable_args` with `vectorbtpro.generic.splitting.base.Takeable`
    3. Runs `vectorbtpro.generic.splitting.base.Splitter.apply` with arguments passed
    to the function as `args` and `kwargs`, but also `**apply_kwargs` (the ones passed to
    the decorator)

    Usage:
        * Split a Series and return its sum:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import numpy as np
        >>> import pandas as pd

        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
        ...     takeable_args=["sr"]
        ... )
        ... def f(sr):
        ...     return sr.sum()

        >>> index = pd.date_range("2020-01-01", "2020-01-06")
        >>> sr = pd.Series(np.arange(len(index)), index=index)
        >>> f(sr)
        split
        0     3
        1    12
        dtype: int64
        ```

        * Perform a split manually:

        ```pycon
        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
        ...     takeable_args=["index"]
        ... )
        ... def f(index, sr):
        ...     return sr[index].sum()

        >>> f(index, sr)
        split
        0     3
        1    12
        dtype: int64
        ```

        * Construct splitter and mark arguments as "takeable" manually:

        ```pycon
        >>> splitter = vbt.Splitter.from_n_rolling(index, n=2)
        >>> @vbt.split(splitter=splitter)
        ... def f(sr):
        ...     return sr.sum()

        >>> f(vbt.Takeable(sr))
        split
        0     3
        1    12
        dtype: int64
        ```

        * Split multiple timeframes using a custom index:

        ```pycon
        >>> @vbt.split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=2),
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
        split
        0    15
        1    42
        dtype: int64
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            splitter = kwargs.pop("_splitter", wrapper.options["splitter"])
            if splitter is None:
                raise ValueError("Must provide splitter")
            splitter_cls = kwargs.pop("_splitter_cls", wrapper.options["splitter_cls"])
            splitter_kwargs = merge_dicts(wrapper.options["splitter_kwargs"], kwargs.pop("_splitter_kwargs", {}))
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
            if not isinstance(splitter, splitter_cls):
                if index is None and index_from is not None:
                    index = splitter_cls.get_obj_index(match_ann_arg(ann_args, index_from))
                if index is None and len(takeable_args) > 0:
                    index = splitter_cls.get_obj_index(match_ann_arg(ann_args, list(takeable_args)[0]))
                if index is None:
                    raise ValueError("Must provide splitter, index, index_from, or takeable_args")
                if isinstance(splitter, str):
                    splitter = getattr(splitter_cls, splitter)
                splitter = splitter(index, template_context=template_context, **splitter_kwargs)
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

        wrapper.is_split = True
        wrapper.options = Config(
            dict(
                splitter=splitter,
                splitter_cls=splitter_cls,
                splitter_kwargs=splitter_kwargs,
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


def cv_split(
    *args,
    parameterized_kwargs: tp.KwargsLike = None,
    selection: tp.Union[None, tp.MaybeIterable[tp.Hashable], tp.Callable] = None,
    selection_args: tp.ArgsLike = None,
    selection_kwargs: tp.KwargsLike = None,
    prepend_grid_result: bool = True,
    return_grid: tp.Union[bool, str] = False,
    template_context: tp.KwargsLike = None,
    **split_kwargs,
) -> tp.Callable:
    """Decorator that combines `split` and `vectorbtpro.utils.params.parameterized` for cross-validation.

    Creates a new apply function that is going to be decorated with `split` and thus applied
    at each single range using `vectorbtpro.generic.splitting.base.Splitter.apply`. Inside
    this apply function, there is a test whether the current range belongs to the first (training) set.
    If yes, parameterizes the underlying function and runs it on the entire grid of parameters.
    The returned results are then stored in a global list. These results are then read by the other
    (testing) sets in the same split, and become the first argument in `selection_args` such that
    if `selection` is a function, it can evaluate the results and return the best parameter combination.
    This parameter combination is then executed by each set (including training).

    Keyword arguments `parameterized_kwargs` will be passed to `vectorbtpro.utils.params.parameterized`
    and will have their templates substituted with a context that will also include the split-related context
    (including `split_idx`, `set_idx`, etc., see `vectorbtpro.generic.splitting.base.Splitter.apply`).

    If `return_grid` is True or 'first', returns both the grid and the selection. If `return_grid`
    is 'all', executes the grid on each set and returns along with the selection.
    Otherwise, returns only the selection.

    Usage:
        * Permutate a series and pick the first value. Make the seed parameterizable.
        Cross-validate based on the highest picked value:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import numpy as np
        >>> import pandas as pd

        >>> @vbt.cv_split(
        ...     splitter="from_n_rolling",
        ...     splitter_kwargs=dict(n=3, split=0.5),
        ...     takeable_args=["sr"],
        ...     parameterized_kwargs=dict(merge_func="concat"),
        ...     selection=lambda x: [np.argmax(x)],
        ...     merge_func="concat",
        ... )
        ... def f(sr, seed):
        ...     np.random.seed(seed)
        ...     return np.random.permutation(sr)[0]

        >>> index = pd.date_range("2020-01-01", "2020-02-01")
        >>> np.random.seed(0)
        >>> sr = pd.Series(np.random.permutation(np.arange(len(index))), index=index)
        >>> f(sr, vbt.Param([41, 42, 43]))
        split  set    seed
        0      set_0  41      22
               set_1  41      28
        1      set_0  43       8
               set_1  43      31
        2      set_0  43      19
               set_1  43       0
        dtype: int64
        ```

        * Extend the example above to also return the grid results of each set:

        ```pycon
        >>> f(sr, vbt.Param([41, 42, 43]), _return_grid="all")
        (split  set    seed
         0      set_0  41      22
                       42      22
                       43       2
                set_1  41      28
                       42      28
                       43      20
         1      set_0  41       5
                       42       5
                       43       8
                set_1  41      23
                       42      23
                       43      31
         2      set_0  41      18
                       42      18
                       43      19
                set_1  41      27
                       42      27
                       43       0
         dtype: int64,
         split  set    seed
         0      set_0  41      22
                set_1  41      28
         1      set_0  43       8
                set_1  43      31
         2      set_0  43      19
                set_1  43       0
         dtype: int64)
        ```
    """
    def decorator(func: tp.Callable) -> tp.Callable:
        if getattr(func, "is_split", False) or getattr(func, "is_parameterized", False):
            raise ValueError("Function is already decorated with split or parameterized")

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            parameterized_kwargs = merge_dicts(
                wrapper.options["parameterized_kwargs"], kwargs.pop("_parameterized_kwargs", {})
            )
            selection = kwargs.pop("_selection", wrapper.options["selection"])
            if selection is None:
                raise ValueError("Must provide selection")
            selection_args = kwargs.pop("_selection_args", wrapper.options["selection_args"])
            if selection_args is None:
                selection_args = ()
            selection_kwargs = merge_dicts(wrapper.options["selection_kwargs"], kwargs.pop("_selection_kwargs", {}))
            prepend_grid_result = kwargs.pop("_prepend_grid_result", wrapper.options["prepend_grid_result"])
            return_grid = kwargs.pop("_return_grid", wrapper.options["return_grid"])
            if isinstance(return_grid, bool):
                if return_grid:
                    return_grid = "first"
                else:
                    return_grid = None
            template_context = merge_dicts(wrapper.options["template_context"], kwargs.pop("_template_context", {}))
            split_kwargs = merge_dicts(wrapper.options["split_kwargs"], kwargs.pop("_split_kwargs", {}))

            grid_results = []

            @wraps(func)
            def apply_wrapper(*_args, __template_context=None, **_kwargs):
                __template_context = dict(__template_context)
                __template_context["grid_results"] = grid_results
                _parameterized_kwargs = deep_substitute(
                    parameterized_kwargs, __template_context, sub_id="parameterized_kwargs"
                )
                parameterized_func = parameterized(func, template_context=__template_context, **_parameterized_kwargs)
                if __template_context["set_idx"] == 0:
                    grid_result = parameterized_func(*_args, **_kwargs)
                    grid_results.append(grid_result)
                if prepend_grid_result:
                    _selection_args = (grid_results[-1], *selection_args)
                else:
                    _selection_args = selection_args
                result = parameterized_func(
                    *_args,
                    _selection=selection,
                    _selection_args=_selection_args,
                    _selection_kwargs=selection_kwargs,
                    _template_context=dict(grid_result=grid_results[-1]),
                    **_kwargs,
                )
                if return_grid is not None:
                    if return_grid.lower() == "first":
                        return grid_results[-1], result
                    if return_grid.lower() == "all":
                        grid_result = parameterized_func(
                            *_args,
                            _template_context=dict(grid_result=grid_results[-1]),
                            **_kwargs,
                        )
                        return grid_result, result
                    else:
                        raise ValueError(f"Invalid option return_grid='{return_grid}'")
                return result

            signature = inspect.signature(apply_wrapper)
            lists_var_kwargs = False
            for k, v in signature.parameters.items():
                if v.kind == v.VAR_KEYWORD:
                    lists_var_kwargs = True
                    break
            if not lists_var_kwargs:
                var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
                new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
                apply_wrapper.__signature__ = signature.replace(parameters=new_parameters)
            split_func = split(apply_wrapper, template_context=template_context, **split_kwargs)
            return split_func(*args, __template_context=Rep("context", sub_id="apply_kwargs"), **kwargs)

        wrapper.is_parameterized = True
        wrapper.is_split = True
        wrapper.options = Config(
            dict(
                parameterized_kwargs=parameterized_kwargs,
                selection=selection,
                selection_args=selection_args,
                selection_kwargs=selection_kwargs,
                prepend_grid_result=prepend_grid_result,
                return_grid=return_grid,
                template_context=template_context,
                split_kwargs=split_kwargs,
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
