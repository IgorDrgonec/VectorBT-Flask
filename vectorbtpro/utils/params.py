# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for working with parameters."""

import attr
import itertools
import inspect
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.config import Config, merge_dicts
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.template import deep_substitute
from vectorbtpro.utils.parsing import annotate_args, ann_args_to_args


def to_typed_list(lst: list) -> List:
    """Cast Python list to typed list.

    Direct construction is flawed in Numba 0.52.0.
    See https://github.com/numba/numba/issues/6651"""
    nb_lst = List()
    for elem in lst:
        nb_lst.append(elem)
    return nb_lst


def flatten_param_tuples(param_tuples: tp.Sequence) -> tp.List[tp.List]:
    """Flattens a nested list of iterables using unzipping."""
    param_list = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = list(unzipped)
        if isinstance(unzipped[0], tuple):
            param_list.extend(flatten_param_tuples(unzipped))
        else:
            param_list.append(unzipped)
    return param_list


def generate_param_combs(op_tree: tp.Tuple, depth: int = 0) -> tp.List[tp.List]:
    """Generate arbitrary parameter combinations from the operation tree `op_tree`.

    `op_tree` is a tuple with nested instructions to generate parameters.
    The first element of the tuple must be a callable that takes remaining elements as arguments.
    If one of the elements is a tuple itself and its first argument is a callable, it will be
    unfolded in the same way as above.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> from itertools import combinations, product
        >>> from vectorbtpro.utils.params import generate_param_combs

        >>> generate_param_combs((product, (combinations, [0, 1, 2, 3], 2), [4, 5]))
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
         [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
         [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]]

        >>> generate_param_combs((product, (zip, [0, 1, 2, 3], [4, 5, 6, 7]), [8, 9]))
        [[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 7, 7], [8, 9, 8, 9, 8, 9, 8, 9]]
        ```
    """
    checks.assert_instance_of(op_tree, tuple)
    checks.assert_instance_of(op_tree[0], Callable)
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple) and isinstance(elem[0], Callable):
            new_op_tree += (generate_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    out = list(new_op_tree[0](*new_op_tree[1:]))
    if depth == 0:
        # do something
        return flatten_param_tuples(out)
    return out


def broadcast_params(param_list: tp.Sequence[tp.Params], to_n: tp.Optional[int] = None) -> tp.List[tp.List]:
    """Broadcast parameters in `param_list`."""
    if to_n is None:
        to_n = max(list(map(len, param_list)))
    new_param_list = []
    for i in range(len(param_list)):
        params = param_list[i]
        if len(params) in [1, to_n]:
            if len(params) < to_n:
                new_param_list.append([p for _ in range(to_n) for p in params])
            else:
                new_param_list.append(list(params))
        else:
            raise ValueError(f"Parameters at index {i} have length {len(params)} that cannot be broadcast to {to_n}")
    return new_param_list


def create_param_product(param_list: tp.Sequence[tp.Params]) -> tp.List[tp.List]:
    """Make Cartesian product out of all params in `param_list`."""
    return list(map(list, zip(*itertools.product(*param_list))))


def params_to_list(params: tp.Params, is_tuple: bool, is_array_like: bool) -> list:
    """Cast parameters to a list."""
    check_against = [list, List]
    if not is_tuple:
        check_against.append(tuple)
    if not is_array_like:
        check_against.append(np.ndarray)
    if isinstance(params, tuple(check_against)):
        new_params = list(params)
    else:
        new_params = [params]
    return new_params


@attr.s(frozen=True)
class Param:
    """Class that represents a parameter."""

    value: tp.Union[tp.Param, tp.Dict[tp.Hashable, tp.Param], tp.Sequence[tp.Param]] = attr.ib()
    """One or more parameter values."""

    is_tuple: bool = attr.ib(default=False)
    """Whether `Param.value` is a tuple.
    
    If so, providing a tuple will be considered as a single value."""

    is_array_like: bool = attr.ib(default=False)
    """Whether `Param.value` is array-like.
    
    If so, providing a NumPy array will be considered as a single value."""

    level: tp.Optional[int] = attr.ib(default=None)
    """Level of the product the parameter takes part in.

    Parameters in the same product are stacked together, not combined, 
    and appear in the index hierarchy next to each other.

    Product index can be used to order index levels: the higher the level, 
    the lower the index level. Index levels with the same level appear in the same 
    order as they are passed to the processor."""

    keys: tp.Optional[tp.IndexLike] = attr.ib(default=None)
    """Keys acting as an index level.

    If None, converts `Param.value` to an index using 
    `vectorbtpro.base.indexes.index_from_values`."""


def combine_params(
    param_dct: tp.Dict[tp.Hashable, Param],
    random_subset: tp.Optional[int] = None,
    seed: tp.Optional[int] = None,
    stack_kwargs: tp.KwargsLike = None,
) -> tp.Tuple[dict, pd.Index]:
    """Combine a dictionary with parameters of the type `Param`.

    Returns a dictionary with combined parameters and an index."""
    from vectorbtpro.base import indexes

    if stack_kwargs is None:
        stack_kwargs = {}

    # Build a product
    param_index = None
    level_values = defaultdict(OrderedDict)
    product_indexes = OrderedDict()
    level_seen = False
    curr_idx = 0
    max_idx = 0
    for k, p in param_dct.items():
        if p.level is None:
            if level_seen:
                raise ValueError("Please provide level for all product parameters")
            level = curr_idx
        else:
            if curr_idx > 0 and not level_seen:
                raise ValueError("Please provide level for all product parameters")
            level_seen = True
            level = p.level
        if level > max_idx:
            max_idx = level

        value = p.value
        keys = p.keys
        if isinstance(value, dict):
            if keys is None:
                keys = list(value.keys())
            value = list(value.values())
        elif isinstance(value, pd.Index):
            if keys is None:
                keys = value
        values = params_to_list(value, is_tuple=p.is_tuple, is_array_like=p.is_array_like)
        level_values[level][k] = values
        if keys is None:
            keys = indexes.index_from_values(values, name=k)
        else:
            if not isinstance(keys, pd.Index):
                keys = pd.Index(keys, name=k)
            elif keys.name is None:
                keys = keys.rename(k)
        product_indexes[k] = keys
        curr_idx += 1

    # Build an operation tree and parameter index
    op_tree_operands = []
    param_keys = []
    for level in range(max_idx + 1):
        if level not in level_values:
            raise ValueError("Group index must come in a strict order starting with 0 and without gaps")
        for k in level_values[level].keys():
            param_keys.append(k)

        # Broadcast parameter arrays
        param_lists = tuple(level_values[level].values())
        if len(param_lists) > 1:
            op_tree_operands.append((zip, *broadcast_params(param_lists)))
        else:
            op_tree_operands.append(param_lists[0])

        # Stack or combine parameter indexes together
        levels = []
        for k in level_values[level].keys():
            levels.append(product_indexes[k])
        if len(levels) > 1:
            _param_index = indexes.stack_indexes(levels, **stack_kwargs)
        else:
            _param_index = levels[0]
        if param_index is None:
            param_index = _param_index
        else:
            param_index = indexes.combine_indexes([param_index, _param_index], **stack_kwargs)

    # Generate parameter combinations using the operation tree
    if len(op_tree_operands) > 1:
        param_product = dict(zip(param_keys, generate_param_combs((itertools.product, *op_tree_operands))))
    elif isinstance(op_tree_operands[0], tuple):
        param_product = dict(zip(param_keys, generate_param_combs(op_tree_operands[0])))
    else:
        param_product = dict(zip(param_keys, op_tree_operands))
    n_params = len(param_product[param_keys[0]])

    # Select a random subset
    if random_subset is not None:
        if seed is not None:
            set_seed(seed)
        random_indices = np.sort(np.random.permutation(np.arange(n_params))[:random_subset])
        param_product = {k: [v[i] for i in range(n_params)] for k, v in param_product.items()}
        if param_index is not None:
            param_index = param_index[random_indices]
    return param_product, param_index


def row_stack_merge_func(results: tp.List[tp.AnyArray], param_index: tp.Index) -> tp.MaybeTuple[tp.SeriesFrame]:
    """Merge multiple Pandas objects along rows."""
    if isinstance(results[0], (tuple, list, List)):
        if len(results[0]) == 1:
            return row_stack_merge_func(list(map(lambda x: x[0], results)), param_index),
        return tuple(map(lambda x: row_stack_merge_func(x, param_index), zip(*results)))
    return pd.concat(results, axis=0, keys=param_index)


def column_stack_merge_func(results: tp.List[tp.AnyArray], param_index: tp.Index) -> tp.MaybeTuple[tp.Frame]:
    """Merge multiple Pandas or `vectorbtpro.base.wrapping.Wrapping` objects along columns."""
    from vectorbtpro.base.wrapping import Wrapping

    if isinstance(results[0], (tuple, list, List)):
        if len(results[0]) == 1:
            return column_stack_merge_func(list(map(lambda x: x[0], results)), param_index),
        return tuple(map(lambda x: column_stack_merge_func(x, param_index), zip(*results)))
    if isinstance(results[0], Wrapping):
        return type(results[0]).column_stack(results, wrapper_kwargs=dict(keys=param_index))
    return pd.concat(results, axis=1, keys=param_index)


def parameterized(
    *args,
    skip_single_param: tp.Optional[bool] = None,
    template_context: tp.Optional[tp.Mapping] = None,
    random_subset: tp.Optional[int] = None,
    stack_kwargs: tp.KwargsLike = None,
    merge_func: tp.Union[None, str, tp.Callable] = None,
    merge_kwargs: tp.KwargsLike = None,
    **execute_kwargs,
) -> tp.Callable:
    """Decorator that parameterizes the function. Engine-agnostic.
    Returns a new function with the same signature as the passed one.

    Does the following:

    1. Searches for arguments wrapped with the class `Param`.
    2. Uses `combine_params` to build parameter combinations.
    3. Generates and resolves parameter configs by combining combinations from the step above over
    `param_configs` that is optionally passed by the user.
    4. Extracts arguments and keyword arguments from each parameter config.
    5. Substitutes any templates
    6. Passes each set of the function and its arguments to `vectorbtpro.utils.execution.execute` for execution.
    7. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`.

    Argument `param_configs` will be added as an extra argument to the function's signature.
    It accepts either a list of dictionaries with arguments named by their names in the signature,
    or a dictionary of dictionaries, where keys are config names. If a list is passed, each dictionary
    can also contain the key `_name` to give the config a name. Variable arguments can be passed
    either in the rolled (`args=(...), kwargs={...}`) or unrolled (`arg_0=..., arg_1=..., some_kwarg=...`) format.

    Any template in both `execute_kwargs` and `merge_kwargs` will be substituted. You can use
    the keys `param_configs`, `param_index`, all keys in `template_context`, and all arguments as found
    in the signature of the function.

    If `skip_single_param` is True, won't use the execution engine, but will execute and
    return the result right away.

    Argument `merge_func` also accepts one of the following strings:

    * 'concat' or 'row_stack': use `row_stack_merge_func`
    * 'column_stack': use `column_stack_merge_func`

    When defining a custom merging function, make sure to use `param_index` (via templates) to build the final
    index/column hierarchy.

    Keyword arguments `**execute_kwargs` are passed directly to `vectorbtpro.utils.execution.execute`.

    Usage:
        * No parameters, no parameter configs:

        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd

        >>> @vbt.parameterized(merge_func="column_stack")
        ... def my_ma(sr_or_df, window, wtype="simple", minp=0, adjust=False):
        ...     return sr_or_df.vbt.ma(window, wtype=wtype, minp=minp, adjust=adjust)

        >>> sr = pd.Series([1, 2, 3, 4, 3, 2, 1])
        >>> my_ma(sr, 3)
        0    1.000000
        1    1.500000
        2    2.000000
        3    3.000000
        4    3.333333
        5    3.000000
        6    2.000000
        dtype: float64
        ```

        * One parameter, no parameter configs:

        ```pycon
        >>> my_ma(sr, vbt.Param([3, 4, 5]))
        window         3    4    5
        0       1.000000  1.0  1.0
        1       1.500000  1.5  1.5
        2       2.000000  2.0  2.0
        3       3.000000  2.5  2.5
        4       3.333333  3.0  2.6
        5       3.000000  3.0  2.8
        6       2.000000  2.5  2.6
        ```

        * Product of two parameters, no parameter configs:

        ```pycon
        >>> my_ma(
        ...     sr,
        ...     vbt.Param([3, 4, 5]),
        ...     wtype=vbt.Param(["simple", "exp"])
        ... )
        window         3                4                5
        wtype     simple       exp simple       exp simple       exp
        0       1.000000  1.000000    1.0  1.000000    1.0  1.000000
        1       1.500000  1.500000    1.5  1.400000    1.5  1.333333
        2       2.000000  2.250000    2.0  2.040000    2.0  1.888889
        3       3.000000  3.125000    2.5  2.824000    2.5  2.592593
        4       3.333333  3.062500    3.0  2.894400    2.6  2.728395
        5       3.000000  2.531250    3.0  2.536640    2.8  2.485597
        6       2.000000  1.765625    2.5  1.921984    2.6  1.990398
        ```

        * No parameters, one partial parameter config:

        ```pycon
        >>> my_ma(sr, param_configs=[dict(window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        * No parameters, one full parameter config:

        ```pycon
        >>> my_ma(param_configs=[dict(sr_or_df=sr, window=3)])
        param_config         0
        0             1.000000
        1             1.500000
        2             2.000000
        3             3.000000
        4             3.333333
        5             3.000000
        6             2.000000
        ```

        * No parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, window=2),
        ...     dict(sr_or_df=sr - 1, window=3)
        ... ], minp=None)
        param_config    0         1
        0             NaN       NaN
        1             2.5       NaN
        2             3.5  1.000000
        3             4.5  2.000000
        4             4.5  2.333333
        5             3.5  2.000000
        6             2.5  1.000000
        ```

        * Multiple parameters, multiple parameter configs:

        ```pycon
        >>> my_ma(param_configs=[
        ...     dict(sr_or_df=sr + 1, minp=0),
        ...     dict(sr_or_df=sr - 1, minp=None)
        ... ], window=vbt.Param([2, 3]))
        window          2              3
        param_config    0    1         0         1
        0             2.0  NaN  2.000000       NaN
        1             2.5  0.5  2.500000       NaN
        2             3.5  1.5  3.000000  1.000000
        3             4.5  2.5  4.000000  2.000000
        4             4.5  2.5  4.333333  2.333333
        5             3.5  1.5  4.000000  2.000000
        6             2.5  0.5  3.000000  1.000000
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            skip_single_param = kwargs.pop("_skip_single_param", wrapper.options["skip_single_param"])
            if skip_single_param is None:
                skip_single_param = params_cfg["skip_single_param"]
            template_context = merge_dicts(wrapper.options["template_context"], kwargs.pop("_template_context", {}))
            random_subset = kwargs.pop("_random_subset", wrapper.options["random_subset"])
            stack_kwargs = merge_dicts(wrapper.options["stack_kwargs"], kwargs.pop("_stack_kwargs", {}))
            merge_func = kwargs.pop("_merge_func", wrapper.options["merge_func"])
            merge_kwargs = merge_dicts(wrapper.options["merge_kwargs"], kwargs.pop("_merge_kwargs", {}))
            execute_kwargs = merge_dicts(wrapper.options["execute_kwargs"], kwargs.pop("_execute_kwargs", {}))
            param_configs = kwargs.pop("param_configs", None)
            if param_configs is None:
                param_configs = []

            # Annotate arguments
            ann_args = annotate_args(func, args, kwargs, allow_partial=True)
            var_args_name = None
            var_kwargs_name = None
            for k, v in ann_args.items():
                if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                    var_args_name = k
                if v["kind"] == inspect.Parameter.VAR_KEYWORD:
                    var_kwargs_name = k

            # Unroll parameter configs
            pc_names = []
            pc_names_none = True
            n_param_configs = 0
            if isinstance(param_configs, dict):
                new_param_configs = []
                for k, v in param_configs.items():
                    v = dict(v)
                    v["_name"] = k
                    new_param_configs.append(v)
                param_configs = new_param_configs
            else:
                param_configs = list(param_configs)
            for i, param_config in enumerate(param_configs):
                param_config = dict(param_config)
                if var_args_name is not None and var_args_name in param_config:
                    for k, arg in enumerate(param_config.pop(var_args_name)):
                        param_config[f"arg_{k}"] = arg
                if var_kwargs_name is not None and var_kwargs_name in param_config:
                    for k, v in param_config.pop(var_kwargs_name).items():
                        param_config[k] = v
                if "_name" in param_config and param_config["_name"] is not None:
                    pc_names.append(param_config.pop("_name"))
                    pc_names_none = False
                else:
                    pc_names.append(n_param_configs)
                param_configs[i] = param_config
                n_param_configs += 1

            # Combine parameters
            paramable_kwargs = {}
            for k, v in ann_args.items():
                if "value" in v:
                    if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                        for i, arg in enumerate(v["value"]):
                            paramable_kwargs[f"arg_{i}"] = arg
                    elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                        for k2, v2 in v["value"].items():
                            paramable_kwargs[k2] = v2
                    else:
                        paramable_kwargs[k] = v["value"]
            param_dct = {}
            for k, v in paramable_kwargs.items():
                if isinstance(v, Param):
                    param_dct[k] = v
            param_columns = None
            if len(param_dct) > 0:
                param_product, param_columns = combine_params(
                    param_dct,
                    random_subset=random_subset,
                    stack_kwargs=stack_kwargs,
                )
                if len(param_configs) == 0:
                    param_configs = []
                    for i in range(len(param_columns)):
                        param_config = dict()
                        for k, v in param_product.items():
                            param_config[k] = v[i]
                        param_configs.append(param_config)
                else:
                    new_param_configs = []
                    for i in range(len(param_columns)):
                        for param_config in param_configs:
                            new_param_config = dict()
                            for k, v in param_config.items():
                                if k in param_product:
                                    raise ValueError(f"Parameter '{k}' is re-defined in a parameter config")
                                new_param_config[k] = v
                            for k, v in param_product.items():
                                new_param_config[k] = v[i]
                            new_param_configs.append(new_param_config)
                    param_configs = new_param_configs

            # Build param index
            n_config_params = len(pc_names)
            if param_columns is not None:
                if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                    param_index = param_columns
                else:
                    from vectorbtpro.base.indexes import combine_indexes

                    param_index = combine_indexes((
                        param_columns,
                        pd.Index(pc_names, name="param_config"),
                    ), **stack_kwargs)
            else:
                if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                    param_index = pd.Index([0], name="param_config")
                else:
                    param_index = pd.Index(pc_names, name="param_config")

            # Create parameter config from arguments if empty
            if len(param_configs) == 0:
                single_param = True
                param_configs.append(dict())
            else:
                single_param = False

            # Roll parameter configs
            new_param_configs = []
            for param_config in param_configs:
                new_param_config = merge_dicts(paramable_kwargs, param_config)
                if var_args_name is not None:
                    _args = ()
                    while True:
                        if f"arg_{len(_args)}" in new_param_config:
                            _args += (new_param_config.pop(f"arg_{len(_args)}"),)
                        else:
                            break
                    new_param_config[var_args_name] = _args
                if var_kwargs_name is not None:
                    new_param_config[var_kwargs_name] = {}
                    for k in list(new_param_config.keys()):
                        if k not in ann_args:
                            new_param_config[var_kwargs_name][k] = new_param_config.pop(k)
                new_param_configs.append(new_param_config)
            param_configs = new_param_configs
            template_context["param_configs"] = param_configs
            template_context["param_index"] = param_index

            # Prepare function and arguments

            def _prepare_args():
                for p, param_config in enumerate(param_configs):
                    _template_context = dict(template_context)
                    _template_context["param_idx"] = p
                    _ann_args = dict()
                    for k, v in ann_args.items():
                        v = dict(v)
                        v["value"] = param_config[k]
                        _ann_args[k] = v
                    _args, _kwargs = ann_args_to_args(_ann_args)
                    _args = deep_substitute(_args, _template_context, sub_id="args")
                    _kwargs = deep_substitute(_kwargs, _template_context, sub_id="kwargs")
                    yield func, _args, _kwargs

            funcs_args = _prepare_args()
            if skip_single_param and single_param:
                funcs_args = list(funcs_args)
                return funcs_args[0][0](*funcs_args[0][1], **funcs_args[0][2])

            # Execute function on each parameter combination
            execute_kwargs = deep_substitute(execute_kwargs, template_context, sub_id="execute_kwargs")
            results = execute(funcs_args, n_calls=len(param_configs), **execute_kwargs)

            # Merge the results
            if merge_func is not None:
                template_context["funcs_args"] = funcs_args
                if isinstance(merge_func, str):
                    if merge_func in ("concat", "row_stack"):
                        merge_func = row_stack_merge_func
                        merge_kwargs = dict(param_index=param_index)
                    elif merge_func == "column_stack":
                        merge_func = column_stack_merge_func
                        merge_kwargs = dict(param_index=param_index)
                    else:
                        raise ValueError(f"Merge function '{merge_func}' not supported")
                merge_kwargs = deep_substitute(merge_kwargs, template_context, sub_id="merge_kwargs")
                return merge_func(results, **merge_kwargs)
            return results

        wrapper.options = Config(
            dict(
                skip_single_param=skip_single_param,
                template_context=template_context,
                random_subset=random_subset,
                stack_kwargs=stack_kwargs,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                execute_kwargs=execute_kwargs,
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
