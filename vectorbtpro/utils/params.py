# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with parameters."""

import math
import inspect
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps

import numpy as np
import pandas as pd
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import define, fld, MISSING, AttrsMixin
from vectorbtpro.utils.config import Config, Configured, merge_dicts
from vectorbtpro.utils.execution import execute
from vectorbtpro.utils.template import CustomTemplate, substitute_templates
from vectorbtpro.utils.parsing import annotate_args, flatten_ann_args, unflatten_ann_args, ann_args_to_args
from vectorbtpro.utils.selection import PosSel, LabelSel, _NoResult, NoResultsException
from vectorbtpro.utils.annotations import Annotatable, has_annotatables
from vectorbtpro.utils.merging import MergeFunc, parse_merge_func
from vectorbtpro.utils.search import find_in_obj, replace_in_obj

__all__ = [
    "generate_param_combs",
    "pick_from_param_grid",
    "Param",
    "combine_params",
    "Parameterizer",
    "parameterized",
]


def to_typed_list(lst: list) -> List:
    """Cast Python list to typed list.

    Direct construction is flawed in Numba 0.52.0.
    See https://github.com/numba/numba/issues/6651"""
    nb_lst = List()
    for elem in lst:
        nb_lst.append(elem)
    return nb_lst


def flatten_param_tuples(param_tuples: tp.Sequence) -> tp.Params:
    """Flattens a nested list of iterables using unzipping."""
    params = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = list(unzipped)
        if isinstance(unzipped[0], tuple):
            params.extend(flatten_param_tuples(unzipped))
        else:
            params.append(unzipped)
    return params


def generate_param_combs(op_tree: tp.Tuple, depth: int = 0) -> tp.Params:
    """Generate arbitrary parameter combinations from the operation tree `op_tree`.

    `op_tree` is a tuple with nested instructions to generate parameters.
    The first element of the tuple must be either the name of a callale from `itertools` or the
    callable itself that takes remaining elements as arguments. If one of the elements is a tuple
    itself and its first argument is a callable, it will be unfolded in the same way as above.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.generate_param_combs(("product", ("combinations", [0, 1, 2, 3], 2), [4, 5]))
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
         [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
         [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]]

        >>> vbt.generate_param_combs(("product", (zip, [0, 1, 2, 3], [4, 5, 6, 7]), [8, 9]))
        [[0, 0, 1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6, 7, 7], [8, 9, 8, 9, 8, 9, 8, 9]]
        ```
    """
    checks.assert_instance_of(op_tree, tuple)
    checks.assert_instance_of(op_tree[0], (Callable, str))
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple) and isinstance(elem[0], (Callable, str)):
            new_op_tree += (generate_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    if isinstance(new_op_tree[0], Callable):
        out = list(new_op_tree[0](*new_op_tree[1:]))
    else:
        import itertools

        out = list(getattr(itertools, new_op_tree[0])(*new_op_tree[1:]))
    if depth == 0:
        return flatten_param_tuples(out)
    return out


def broadcast_params(params_or_dict: tp.ParamsOrDict, to_n: tp.Optional[int] = None) -> tp.ParamsOrDict:
    """Broadcast parameters in `params`."""
    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    if to_n is None:
        to_n = max(list(map(len, params)))
    new_params = []
    for i in range(len(params)):
        param_values = params[i]
        if len(param_values) in [1, to_n]:
            if len(param_values) < to_n:
                new_params.append([p for _ in range(to_n) for p in param_values])
            else:
                new_params.append(list(param_values))
        else:
            raise ValueError(
                f"Parameters at index {i} have length {len(param_values)} that cannot be broadcast to {to_n}"
            )
    if isinstance(params_or_dict, dict):
        return dict(zip(params_or_dict.keys(), new_params))
    return new_params


def create_param_product(params_or_dict: tp.ParamsOrDict) -> tp.ParamsOrDict:
    """Make Cartesian product out of all params in `params`."""
    import itertools

    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    new_params = list(map(list, zip(*itertools.product(*params))))
    if isinstance(params_or_dict, dict):
        return dict(zip(params_or_dict.keys(), new_params))
    return new_params


def is_single_param_value(
    param_values: tp.MaybeParamValues,
    is_tuple: bool = False,
    is_array_like: bool = False,
) -> bool:
    """Check whether `param_values` is a single value."""
    check_against = [list, List]
    if not is_tuple:
        check_against.append(tuple)
    if not is_array_like:
        check_against.append(np.ndarray)
    if isinstance(param_values, tuple(check_against)):
        return False
    if isinstance(param_values, range):
        return False
    return True


def params_to_list(
    param_values: tp.MaybeParamValues,
    is_tuple: bool = False,
    is_array_like: bool = False,
) -> list:
    """Cast parameters to a list."""
    if is_single_param_value(param_values, is_tuple, is_array_like):
        return [param_values]
    return list(param_values)


def get_param_grid_len(params_or_dict: tp.ParamsOrDict) -> int:
    """Get the number of parameter combinations in a parameter grid."""
    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    return math.prod(list(map(len, params)))


def pick_from_param_grid(
    params_or_dict: tp.ParamsOrDict,
    i: tp.Union[None, int, tp.Array1d] = None,
) -> tp.Union[tp.ParamCombOrDict, tp.List[tp.Array1d]]:
    """Pick one or more parameter combinations from a parameter grid."""
    if isinstance(params_or_dict, dict):
        params = list(params_or_dict.values())
    else:
        params = params_or_dict
    grid_len = get_param_grid_len(params)
    if i is None:
        i = np.random.randint(grid_len, dtype=np.int64)
    param_comb = []
    for param_values in params:
        index = i * len(param_values) // grid_len
        block_len = grid_len // len(param_values)
        i = i - index * block_len
        grid_len = block_len
        param_comb.append(param_values[index])
    if isinstance(params_or_dict, dict):
        return dict(zip(params_or_dict.keys(), param_comb))
    return param_comb


ParamT = tp.TypeVar("ParamT", bound="Param")


@define
class Param(Annotatable, AttrsMixin):
    """Class that represents a parameter."""

    value: tp.Union[tp.MaybeParamValues, tp.Dict[tp.Hashable, tp.ParamValue]] = fld(default=MISSING)
    """One or more parameter values."""

    is_tuple: bool = fld(default=False)
    """Whether `Param.value` is a tuple.
    
    If so, providing a tuple will be considered as a single value."""

    is_array_like: bool = fld(default=False)
    """Whether `Param.value` is array-like.
    
    If so, providing a NumPy array will be considered as a single value."""

    map_template: tp.Optional[CustomTemplate] = fld(default=None)
    """Template to map `Param.value` before building parameter combinations."""

    random_subset: tp.Union[None, int, float] = fld(default=None)
    """Random subset of values to select."""

    level: tp.Optional[int] = fld(default=None)
    """Level of the product the parameter takes part in.

    Parameters with the same level are stacked together, while parameters with different levels
    are combined as usual.
    
    Parameters are processed based on their level: a lower-level parameter is processed before 
    (and thus displayed above) a higher-level parameter. If two parameters share the same level, 
    they are processed in the order they were passed to the function.
    
    Levels must come in a strict order starting with 0 and without gaps. If any of the parameters
    have a level specified, all parameters must specify their level."""

    condition: tp.Optional[str] = fld(default=None)
    """Keep a parameter combination only if the condition is met.
    
    Condition should be an expression where `x` denotes this parameter and any other variable
    denotes the name of other parameter(s)."""

    context: tp.KwargsLike = fld(default=None)
    """Context used in evaluation of `Param.condition` and `Param.map_template`."""

    keys: tp.Optional[tp.IndexLike] = fld(default=None)
    """Keys acting as an index level.

    If None, converts `Param.value` to an index using 
    `vectorbtpro.base.indexes.index_from_values`."""

    hide: bool = fld(default=False)
    """Whether to hide the parameter from the parameter index."""

    name: tp.Optional[tp.Hashable] = fld(default=None)
    """Name of the parameter.
    
    If None, defaults to the name of the index in `Param.keys`, or to the key in 
    `param_dct` passed to `combine_params`."""

    mono_reduce: bool = fld(default=False)
    """Whether to reduce a mono-chunk of the same values into one value."""

    mono_merge_func: tp.MergeFuncLike = fld(default=None)
    """Merge function to apply when building a mono-chunk.
    
    Resolved using `vectorbtpro.base.merging.resolve_merge_func`."""

    mono_merge_kwargs: tp.KwargsLike = fld(default=None)
    """Keyword arguments passed to `Param.mono_merge_func`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = fld(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """Return whether the evaluation id of the instance meets the global evaluation id."""
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True

    def check_value(self) -> None:
        """Check whether value is missing."""
        if self.value is MISSING:
            raise ValueError("Parameter value is missing")

    def map_value(self: ParamT, func: tp.Callable) -> ParamT:
        """Execute a function on each value in `Param.value` and create a new `Param` instance."""
        self.check_value()
        attr_dct = self.asdict()
        if isinstance(attr_dct["value"], dict):
            attr_dct["value"] = {k: v for k, v in attr_dct["value"].items()}
        elif isinstance(attr_dct["value"], pd.Index):
            attr_dct["value"] = pd.Index(map(func, attr_dct["value"]))
        elif isinstance(attr_dct["value"], pd.Series):
            attr_dct["value"] = pd.Series(map(func, attr_dct["value"].values), index=attr_dct["value"].index)
        else:
            attr_dct["value"] = list(map(func, attr_dct["value"]))
        return type(self)(**attr_dct)


class Paramable:
    """Class representing an object that can be returned as a parameter."""

    def as_param(self) -> Param:
        """Return this instance as a parameter."""
        raise NotImplementedError


def combine_params(
    param_dct: tp.Dict[tp.Hashable, Param],
    build_grid: tp.Optional[bool] = None,
    grid_indices: tp.Union[None, slice, tp.Sequence[int]] = None,
    random_subset: tp.Union[None, int, float] = None,
    random_replace: bool = False,
    random_sort: bool = True,
    max_guesses: tp.Union[None, int, float] = None,
    max_misses: tp.Union[None, int, float] = None,
    seed: tp.Optional[int] = None,
    index_stack_kwargs: tp.KwargsLike = None,
    name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
    build_index: bool = True,
    raise_empty_error: bool = False,
) -> tp.Union[dict, tp.Tuple[dict, pd.Index]]:
    """Combine a dictionary with parameters of the type `Param`.

    Returns a dictionary with combined parameters and an index if `build_index` is True.

    If `build_grid` is True, first builds the entire grid and then filters parameter
    combinations by conditions and selects random combinations. If `build_grid` is False,
    doesn't build the entire grid, but selects and combines combinations on the fly.
    Materializing the grid is recommended only when the number of combinations is relatively low
    (less than one million) and parameters have conditions.
    
    Argument `grid_indices` can be a slice (for example, `slice(None, None, 2)` for `::2`) or an array
    with indices that map to the length of the grid. It can be used to skip a some combinations
    before a random subset is drawn.

    Argument `random_subset` can be an integer (number of combinations) or a float relative to the
    length of the grid. If parameters have conditions, `random_subset` is drawn from the subset of
    combinations whose conditions have been met, not the other way around. If `random_replace` is True,
    draws random combinations with replacement (that is, duplicate combinations will likely occur).
    If `random_replace` is False (default), each drawn combination will be unique. If `random_sort`
    is True (default), positions of combinations will be sorted. Otherwise, they will remain in their
    randomly-selected positions.

    Arguments `max_guesses` and `max_misses` are effective only when the grid is not buildd
    and parameters have conditions. They mean the maximum number of guesses and misses respectively
    when doing the search for a valid combination in a while-loop. Once any of these two numbers
    is reached, the search will stop. They are useful for limiting the number of guesses; without
    them, the search may continue forever.

    If a name of any parameter is a tuple, can convert this tuple into a string by setting
    `name_tuple_to_str` either to True or providing a callable that does this.

    Keyword arguments `index_stack_kwargs` are passed to `vectorbtpro.base.indexes.stack_indexes`."""
    from vectorbtpro.base import indexes

    if index_stack_kwargs is None:
        index_stack_kwargs = {}
    rng = np.random.default_rng(seed=seed)

    level_map = OrderedDict()
    product_indexes = OrderedDict()
    level_seen = False
    curr_idx = 0
    max_idx = 0
    conditions = {}
    contexts = {}
    names = {}
    for k, p in param_dct.items():
        if isinstance(p, Paramable):
            p = p.to_param()
        if not isinstance(p, Param):
            p = Param(p)
        if isinstance(p.value, Paramable):
            p2 = p.value.to_param()
            p = p.merge_over(p2, value=p2.value)
        if p.condition is not None:
            conditions[k] = p.condition
            if p.context is not None:
                contexts[k] = p.context
            else:
                contexts[k] = {}
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

        keys_name = None
        p_name = p.name
        sr_name = None
        index_name = None

        if not p.hide:
            keys = p.keys
            if keys is not None:
                if not isinstance(keys, pd.Index):
                    keys = pd.Index(keys)
                keys_name = keys.name
        else:
            keys = None

        p.check_value()
        value = p.value
        if isinstance(value, dict):
            if not p.hide and keys is None:
                keys = pd.Index(value.keys())
            value = list(value.values())
        elif isinstance(value, pd.Index):
            if not p.hide and keys is None:
                keys = value
            index_name = value.name
            value = value.tolist()
        elif isinstance(value, pd.Series):
            if not checks.is_default_index(value.index):
                if not p.hide and keys is None:
                    keys = value.index
                index_name = value.index.name
            sr_name = value.name
            value = value.values.tolist()
        values = params_to_list(value, is_tuple=p.is_tuple, is_array_like=p.is_array_like)

        if not p.hide:
            if keys_name is None:
                if p_name is not None:
                    keys_name = p_name
                elif sr_name is not None:
                    keys_name = sr_name
                elif index_name is not None:
                    keys_name = index_name
                else:
                    keys_name = k
            if keys is None:
                keys = indexes.index_from_values(values, name=keys_name)
            else:
                keys = keys.rename(keys_name)

        if p.random_subset is not None:
            if checks.is_float(p.random_subset):
                _random_subset = int(p.random_subset * len(values))
            else:
                _random_subset = p.random_subset
            random_indices = np.sort(rng.permutation(np.arange(len(values)))[:_random_subset])
        else:
            random_indices = None
        if random_indices is not None:
            values = [values[i] for i in random_indices]
            if keys is not None:
                keys = keys[random_indices]

        if p.map_template is not None:
            param_context = merge_dicts(
                dict(
                    param=p,
                    values=values,
                    keys=keys,
                    random_indices=random_indices,
                ),
                p.context,
            )
            values = p.map_template.substitute(param_context, eval_id="map_template")

        if level not in level_map:
            level_map[level] = OrderedDict()
        level_map[level][k] = values
        product_indexes[k] = keys
        names[k] = keys_name
        curr_idx += 1

    level_params = []
    param_keys = []
    level_indexes = []
    shown_levels = []
    hidden_levels = []
    n_combs = None
    for level in range(max_idx + 1):
        if level not in level_map:
            raise ValueError("Levels must come in a strict order starting with 0 and without gaps")
        for k in level_map[level].keys():
            param_keys.append(k)

        params = tuple(level_map[level].values())
        if len(params) > 1:
            params = broadcast_params(params)
        level_params.append(params)
        if n_combs is None:
            n_combs = len(params[0])
        else:
            n_combs *= len(params[0])

        if build_index:
            levels = []
            for k in level_map[level].keys():
                if product_indexes[k] is not None:
                    levels.append(product_indexes[k])
            if len(levels) > 1:
                _param_index = indexes.stack_indexes(levels, **index_stack_kwargs)
                shown_levels.append(level)
            elif len(levels) == 1:
                _param_index = levels[0]
                shown_levels.append(level)
            else:
                _param_index = range(len(params[0]))
                hidden_levels.append(level)
            level_indexes.append(_param_index)

    if len(conditions) > 0:
        condition_funcs = {}
        for k, expr in conditions.items():
            arg_names = {"x"} | set(names.keys()) | set(names.values()) | set(contexts[k].keys())
            condition_funcs[k] = eval(f"lambda {', '.join(arg_names)}: {expr}")
    else:
        condition_funcs = None

    if grid_indices is not None:
        if isinstance(grid_indices, slice):
            slice_start = grid_indices.start
            if slice_start is None:
                slice_start = 0
            slice_stop = grid_indices.stop
            if slice_stop is None:
                slice_stop = n_combs
            grid_indices = np.arange(slice_start, slice_stop, grid_indices.step)
        else:
            grid_indices = np.asarray(grid_indices)

    if build_grid is None:
        if grid_indices is not None:
            build_grid = False
        elif random_subset is None:
            build_grid = True
        else:
            if len(conditions) == 0:
                build_grid = False
            else:
                if checks.is_float(random_subset):
                    build_grid = True
                else:
                    if n_combs >= 1_000_000:
                        build_grid = False
                    else:
                        build_grid = True

    if not build_grid:
        if len(conditions) == 0:
            if grid_indices is not None:
                n_combs = len(grid_indices)
            level_indices = list(map(lambda x: np.arange(len(x[0])), level_params))
            if random_subset is not None:
                if checks.is_float(random_subset):
                    random_subset = int(random_subset * n_combs)
                if grid_indices is not None:
                    random_grid_indices = rng.choice(grid_indices, size=random_subset, replace=random_replace)
                else:
                    random_grid_indices = rng.choice(n_combs, size=random_subset, replace=random_replace)
                if random_sort:
                    random_grid_indices = np.sort(random_grid_indices)
                picked_level_indices = pick_from_param_grid(level_indices, i=random_grid_indices)
            else:
                if grid_indices is not None:
                    picked_grid_indices = grid_indices
                else:
                    picked_grid_indices = np.arange(n_combs)
                picked_level_indices = pick_from_param_grid(level_indices, i=picked_grid_indices)

            params_ready = True
        elif len(conditions) > 0 and grid_indices is None:
            if random_subset is None:
                raise ValueError("Must build the grid for conditions without a random subset")
            if checks.is_float(random_subset):
                raise ValueError("Must build the grid for conditions with a floating random subset")
            if max_guesses is not None and checks.is_float(max_guesses):
                max_guesses = int(max_guesses * random_subset)
            if max_misses is not None and checks.is_float(max_misses):
                max_misses = int(max_misses * random_subset)
            picked_indices_list = []
            picked_indices_set = set()
            visited_indices_set = set()
            n_misses = 0
            n_guesses = 0
            while len(picked_indices_list) < random_subset and len(visited_indices_set) < n_combs:
                n_guesses += 1
                picked_indices = []
                k = 0
                param_values = {}
                for level, params in enumerate(level_params):
                    picked_index = rng.choice(len(params[0]), replace=True)
                    for j in range(len(params)):
                        picked_value = params[j][picked_index]
                        param_values[param_keys[k]] = picked_value
                        param_values[names[param_keys[k]]] = picked_value
                        k += 1
                    picked_indices.append(picked_index)
                visited_indices_set.add(tuple(picked_indices))
                if not random_replace and tuple(picked_indices) in picked_indices_set:
                    continue
                conditions_met = True
                for k, condition_func in condition_funcs.items():
                    param_context = {"x": param_values[k], **param_values, **contexts[k]}
                    if not condition_func(**param_context):
                        conditions_met = False
                        break
                if conditions_met:
                    picked_indices_list.append(tuple(picked_indices))
                    picked_indices_set.add(tuple(picked_indices))
                else:
                    n_misses += 1
                    if max_misses is not None:
                        if n_misses >= max_misses:
                            break
                if max_guesses is not None:
                    if n_guesses >= max_guesses:
                        break
            if random_sort:
                picked_level_indices = list(map(list, zip(*sorted(picked_indices_list))))
            else:
                picked_level_indices = list(map(list, zip(*picked_indices_list)))

            params_ready = True
        else:
            n_combs = len(grid_indices)
            level_indices = list(map(lambda x: np.arange(len(x[0])), level_params))
            picked_level_indices = pick_from_param_grid(level_indices, i=grid_indices)

            params_ready = False

        if len(picked_level_indices) == 0 or len(picked_level_indices[0]) == 0:
            param_product = {k: [] for k in param_keys}
        else:
            param_product = dict()
            k = 0
            for level in range(len(picked_level_indices)):
                for j in range(len(level_params[level])):
                    param_key = param_keys[k]
                    if param_key not in param_product:
                        param_product[param_key] = []
                    param_values = level_params[level][j]
                    picked_param_values = [param_values[i] for i in picked_level_indices[level]]
                    param_product[param_key] = picked_param_values
                    k += 1

        if build_index and len(shown_levels) > 0:
            shown_indexes = []
            for level, index in enumerate(level_indexes):
                if level in shown_levels:
                    if len(picked_level_indices) == 0 or len(picked_level_indices[0]) == 0:
                        shown_indexes.append(index[0:0])
                    else:
                        shown_indexes.append(index[picked_level_indices[level]])
            if len(shown_indexes) > 1:
                param_index = indexes.stack_indexes(shown_indexes, **index_stack_kwargs)
            else:
                param_index = shown_indexes[0]
        else:
            param_index = None
    else:
        op_tree_operands = []
        for params in level_params:
            if len(params) > 1:
                op_tree_operands.append((zip, *broadcast_params(params)))
            else:
                op_tree_operands.append(params[0])
        if len(op_tree_operands) > 1:
            param_product = dict(zip(param_keys, generate_param_combs(("product", *op_tree_operands))))
        elif isinstance(op_tree_operands[0], tuple):
            param_product = dict(zip(param_keys, generate_param_combs(op_tree_operands[0])))
        else:
            param_product = dict(zip(param_keys, op_tree_operands))

        if build_index and len(shown_levels) > 0:
            if len(level_indexes) > 1:
                param_index = indexes.combine_indexes(level_indexes, **index_stack_kwargs)
                if len(hidden_levels) > 0:
                    if len(shown_levels) > 1:
                        param_index = indexes.select_levels(param_index, shown_levels)
                    else:
                        param_index = indexes.select_levels(param_index, shown_levels[0])
            else:
                param_index = level_indexes[0]
        else:
            param_index = None

        if grid_indices is not None:
            n_combs = len(grid_indices)
            param_product = {k: [v[i] for i in grid_indices] for k, v in param_product.items()}
            if build_index and len(shown_levels) > 0:
                param_index = param_index[grid_indices]

        params_ready = False

    if not params_ready:
        if len(conditions) > 0:
            indices = np.arange(n_combs)
            if random_subset is not None and not checks.is_float(random_subset) and not random_replace:
                pre_random_subset = True
            else:
                pre_random_subset = False
            if pre_random_subset:
                indices = rng.permutation(indices)
            keep_indices = []
            any_discarded = False
            for i in indices:
                param_values = {}
                for k in param_product:
                    param_values[k] = param_product[k][i]
                    param_values[names[k]] = param_product[k][i]
                conditions_met = True
                for k, condition_func in condition_funcs.items():
                    param_context = {"x": param_values[k], **param_values, **contexts[k]}
                    if not condition_func(**param_context):
                        conditions_met = False
                        break
                if conditions_met:
                    keep_indices.append(i)
                    if pre_random_subset:
                        if len(keep_indices) == random_subset:
                            break
                else:
                    any_discarded = True
            if any_discarded:
                if len(keep_indices) > 0:
                    if pre_random_subset and random_sort:
                        keep_indices = np.sort(keep_indices)
                    param_product = {k: [v[i] for i in keep_indices] for k, v in param_product.items()}
                    n_combs = len(keep_indices)
                    if build_index and len(shown_levels) > 0:
                        param_index = param_index[keep_indices]
                else:
                    param_product = {k: [] for k, v in param_product.items()}
                    n_combs = 0
                    if build_index and len(shown_levels) > 0:
                        param_index = param_index[0:0]
        else:
            pre_random_subset = False

        if random_subset is not None and not pre_random_subset and n_combs > 0:
            if checks.is_float(random_subset):
                random_subset = int(random_subset * n_combs)
            random_indices = rng.choice(n_combs, size=random_subset, replace=random_replace)
            if random_sort:
                random_indices = np.sort(random_indices)
            param_product = {k: [v[i] for i in random_indices] for k, v in param_product.items()}
            if build_index and len(shown_levels) > 0:
                param_index = param_index[random_indices]

    if build_index and len(shown_levels) > 0:
        if isinstance(name_tuple_to_str, bool):
            if name_tuple_to_str:

                def _name_tuple_to_str(name_tuple):
                    return "_".join(map(lambda x: str(x).strip().lower(), name_tuple))

                name_tuple_to_str = _name_tuple_to_str
            else:
                name_tuple_to_str = None
        if name_tuple_to_str is not None:
            found_tuple = False
            new_names = []
            for name in param_index.names:
                if isinstance(name, tuple):
                    name = name_tuple_to_str(name)
                    found_tuple = True
                new_names.append(name)
            if found_tuple:
                if isinstance(param_index, pd.MultiIndex):
                    param_index.rename(new_names, inplace=True)
                else:
                    param_index.rename(new_names[0], inplace=True)

    if raise_empty_error:
        if len(param_product[list(param_product.keys())[0]]) == 0:
            raise ValueError("Set of parameter combinations is empty")
    if build_index:
        return param_product, param_index
    return param_product


class Parameterizer(Configured):
    """Class responsible for parameterizing and running a function."""

    _settings_path: tp.SettingsPath = "params"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "func",
        "param_search_kwargs",
        "skip_single_comb",
        "template_context",
        "build_grid",
        "grid_indices",
        "random_subset",
        "random_replace",
        "random_sort",
        "max_guesses",
        "max_misses",
        "seed",
        "index_stack_kwargs",
        "name_tuple_to_str",
        "merge_func",
        "merge_kwargs",
        "selection",
        "forward_kwargs_as",
        "mono_min_size",
        "mono_n_chunks",
        "mono_chunk_len",
        "mono_chunk_meta",
        "mono_reduce",
        "mono_merge_func",
        "mono_merge_kwargs",
        "return_meta",
        "return_param_index",
        "execute_kwargs",
    }

    def __init__(
        self,
        func: tp.Callable,
        param_search_kwargs: tp.KwargsLike = None,
        skip_single_comb: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        build_grid: tp.Optional[bool] = None,
        grid_indices: tp.Union[None, slice, tp.Sequence[int]] = None,
        random_subset: tp.Union[None, int, float] = None,
        random_replace: tp.Optional[bool] = None,
        random_sort: tp.Optional[bool] = None,
        max_guesses: tp.Union[None, int, float] = None,
        max_misses: tp.Union[None, int, float] = None,
        seed: tp.Optional[int] = None,
        index_stack_kwargs: tp.KwargsLike = None,
        name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
        merge_func: tp.Optional[tp.MergeFuncLike] = None,
        merge_kwargs: tp.KwargsLike = None,
        selection: tp.Optional[tp.Selection] = None,
        forward_kwargs_as: tp.KwargsLike = None,
        mono_min_size: tp.Optional[int] = None,
        mono_n_chunks: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_len: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
        mono_reduce: tp.Union[bool, tp.Kwargs] = None,
        mono_merge_func: tp.Union[tp.MergeFuncLike, tp.Dict[str, tp.MergeFuncLike]] = None,
        mono_merge_kwargs: tp.KwargsLike = None,
        return_meta: tp.Optional[bool] = None,
        return_param_index: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            func=func,
            param_search_kwargs=param_search_kwargs,
            skip_single_comb=skip_single_comb,
            template_context=template_context,
            build_grid=build_grid,
            grid_indices=grid_indices,
            random_subset=random_subset,
            random_replace=random_replace,
            random_sort=random_sort,
            max_guesses=max_guesses,
            max_misses=max_misses,
            seed=seed,
            index_stack_kwargs=index_stack_kwargs,
            name_tuple_to_str=name_tuple_to_str,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            selection=selection,
            forward_kwargs_as=forward_kwargs_as,
            mono_min_size=mono_min_size,
            mono_n_chunks=mono_n_chunks,
            mono_chunk_len=mono_chunk_len,
            mono_chunk_meta=mono_chunk_meta,
            mono_reduce=mono_reduce,
            mono_merge_func=mono_merge_func,
            mono_merge_kwargs=mono_merge_kwargs,
            return_meta=return_meta,
            return_param_index=return_param_index,
            execute_kwargs=execute_kwargs,
            **kwargs,
        )

        self._func = func
        self._param_search_kwargs = param_search_kwargs
        self._skip_single_comb = skip_single_comb
        self._template_context = template_context
        self._build_grid = build_grid
        self._grid_indices = grid_indices
        self._random_subset = random_subset
        self._random_replace = random_replace
        self._random_sort = random_sort
        self._max_guesses = max_guesses
        self._max_misses = max_misses
        self._seed = seed
        self._index_stack_kwargs = index_stack_kwargs
        self._name_tuple_to_str = name_tuple_to_str
        self._merge_func = merge_func
        self._merge_kwargs = merge_kwargs
        self._selection = selection
        self._forward_kwargs_as = forward_kwargs_as
        self._mono_min_size = mono_min_size
        self._mono_n_chunks = mono_n_chunks
        self._mono_chunk_len = mono_chunk_len
        self._mono_chunk_meta = mono_chunk_meta
        self._mono_reduce = mono_reduce
        self._mono_merge_func = mono_merge_func
        self._mono_merge_kwargs = mono_merge_kwargs
        self._return_meta = return_meta
        self._return_param_index = return_param_index
        self._execute_kwargs = execute_kwargs

    @property
    def func(self) -> tp.Callable:
        """Function."""
        return self._func

    @property
    def param_search_kwargs(self) -> tp.KwargsLike:
        """See `Parameterized.find_params_in_obj`."""
        return self._param_search_kwargs

    @property
    def skip_single_comb(self) -> tp.Optional[bool]:
        """Whether to execute the function directly if there's only one parameter combination."""
        return self._skip_single_comb

    @property
    def template_context(self) -> tp.KwargsLike:
        """Template context.

        Any template in both `execute_kwargs` and `merge_kwargs` will be substituted.
        You can use the keys `param_configs`, `param_index`, all keys in `template_context`,
        and all arguments as found in the signature of the function. To substitute templates
        further down the pipeline, use substitution ids."""
        return self._template_context

    @property
    def build_grid(self) -> tp.Optional[bool]:
        """See `combine_params`."""
        return self._build_grid

    @property
    def grid_indices(self) -> tp.Union[None, slice, tp.Sequence[int]]:
        """See `combine_params`."""
        return self._grid_indices

    @property
    def random_subset(self) -> tp.Union[None, int, float]:
        """See `combine_params`."""
        return self._random_subset

    @property
    def random_replace(self) -> tp.Optional[bool]:
        """See `combine_params`."""
        return self._random_replace

    @property
    def random_sort(self) -> tp.Optional[bool]:
        """See `combine_params`."""
        return self._random_sort

    @property
    def max_guesses(self) -> tp.Union[None, int, float]:
        """See `combine_params`."""
        return self._max_guesses

    @property
    def max_misses(self) -> tp.Union[None, int, float]:
        """See `combine_params`."""
        return self._max_misses

    @property
    def seed(self) -> tp.Optional[int]:
        """See `combine_params`."""
        return self._seed

    @property
    def index_stack_kwargs(self) -> tp.KwargsLike:
        """See `combine_params`."""
        return self._index_stack_kwargs

    @property
    def name_tuple_to_str(self) -> tp.Union[None, bool, tp.Callable]:
        """See `combine_params`."""
        return self._name_tuple_to_str

    @property
    def merge_func(self) -> tp.Optional[tp.MergeFuncLike]:
        """Merging function.

        Resolved using `vectorbtpro.base.merging.resolve_merge_func`."""
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments passed to the merging function.

        When defining a custom merging function, make sure to make use of `param_index`
        (via templates) to build the final index hierarchy."""
        return self._merge_kwargs

    @property
    def selection(self) -> tp.Optional[tp.Selection]:
        """Parameter combination to select.

        The selection can be one or more positions or labels from the parameter index.
        The selection value(s) can be wrapped with `vectorbtpro.utils.selection.PosSel` or
        `vectorbtpro.utils.selection.LabelSel` to instruct vectorbtpro what the value(s) should denote.
        Make sure that it's a sequence (for example, by wrapping it with a list) to attach
        the parameter index to the final result. It can be also `vectorbtpro.utils.selection.NoResult`
        to indicate that there's no result, or a template to yield any of the above."""
        return self._selection

    @property
    def forward_kwargs_as(self) -> tp.KwargsLike:
        """Map to rename keyword arguments.

        Can also pass any variable from the scope of `Parameterized.run`."""
        return self._forward_kwargs_as

    @property
    def mono_min_size(self) -> tp.Optional[int]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`.

        Applied to generate chunk meta."""
        return self._mono_min_size

    @property
    def mono_n_chunks(self) -> tp.Optional[tp.Union[str, int]]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`.

        Applied to generate chunk meta."""
        return self._mono_n_chunks

    @property
    def mono_chunk_len(self) -> tp.Optional[tp.Union[str, int]]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`.

        Applied to generate chunk meta."""
        return self._mono_chunk_len

    @property
    def mono_chunk_meta(self) -> tp.Optional[tp.Iterable[tp.ChunkMeta]]:
        """Chunk meta."""
        return self._mono_chunk_meta

    @property
    def mono_reduce(self) -> tp.Union[bool, tp.Kwargs]:
        """See `Param.mono_reduce`.

        Can be a dictionary with a value per (unpacked) argument name. Otherwise,
        gets applied to each parameter, unless the parameter overrides it."""
        return self._mono_reduce

    @property
    def mono_merge_func(self) -> tp.Union[tp.MergeFuncLike, tp.Dict[str, tp.MergeFuncLike]]:
        """See `Param.mono_merge_func`.

        Can be a dictionary with a value per (unpacked) argument name. Otherwise,
        gets applied to each parameter, unless the parameter overrides it."""
        return self._mono_merge_func

    @property
    def mono_merge_kwargs(self) -> tp.KwargsLike:
        """See `Param.mono_merge_kwargs`.

        Can be a dictionary with a value per (unpacked) argument name. Otherwise,
        gets applied to each parameter, unless the parameter overrides it."""
        return self._mono_merge_kwargs

    @property
    def return_meta(self) -> tp.Optional[bool]:
        """Whether to return all the metadata generated before running the function."""
        return self._return_meta

    @property
    def return_param_index(self) -> tp.Optional[bool]:
        """Whether to return the results along with the parameter index (as a tuple)."""
        return self._return_param_index

    @property
    def execute_kwargs(self) -> tp.KwargsLike:
        """Keyword arguments passed to `vectorbtpro.utils.execution.execute`."""
        return self._execute_kwargs

    @classmethod
    def find_params_in_obj(cls, obj: tp.Any, eval_id: tp.Optional[tp.Hashable] = None, **kwargs) -> dict:
        """Find values wrapped with `Param` in a recursive manner.

        Uses `vectorbtpro.utils.search.find_in_obj`."""
        return find_in_obj(obj, lambda k, v: isinstance(v, Param) and v.meets_eval_id(eval_id), **kwargs)

    @classmethod
    def param_product_to_objs(cls, obj: tp.Any, param_product: dict) -> tp.List[dict]:
        """Resolve parameter product into a list of objects based on the original object.

        Uses `vectorbtpro.utils.search.replace_in_obj`."""
        if len(param_product) == 0:
            return []
        param_product_items = list(param_product.items())
        n_values = len(param_product_items[0][1])
        new_objs = []
        for i in range(n_values):
            param_dct = {k: v[i] for k, v in param_product.items()}
            new_objs.append(replace_in_obj(obj, param_dct))
        return new_objs

    @classmethod
    def parse_and_inject_params(
        cls,
        flat_ann_args: tp.FlatAnnArgs,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.FlatAnnArgs:
        """Parse `Param` instances from function annotations and inject them into flattened annotated arguments."""
        new_flat_ann_args = dict()
        for k, v in flat_ann_args.items():
            new_flat_ann_args[k] = v = dict(v)
            if "annotation" in v:
                if isinstance(v["annotation"], type) and issubclass(v["annotation"], Param):
                    v["annotation"] = v["annotation"]()
                if isinstance(v["annotation"], Param) and v["annotation"].meets_eval_id(eval_id):
                    if "value" in v:
                        if not isinstance(v["value"], Param):
                            v["value"] = v["annotation"].replace(value=v["value"])
                        else:
                            v["value"] = v["value"].merge_over(v["annotation"])
        return new_flat_ann_args

    @classmethod
    def get_var_arg_names(cls, ann_args: tp.AnnArgs) -> tp.Tuple[str, str]:
        """Get the name of any packed variable position arguments and keyword arguments."""
        var_args_name = None
        var_kwargs_name = None
        for k, v in ann_args.items():
            if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                var_args_name = k
            if v["kind"] == inspect.Parameter.VAR_KEYWORD:
                var_kwargs_name = k
        return var_args_name, var_kwargs_name

    @classmethod
    def unroll_param_config(cls, param_config: dict, ann_args: tp.AnnArgs) -> dict:
        """Unroll a parameter config."""
        var_args_name, var_kwargs_name = cls.get_var_arg_names(ann_args)
        new_param_config = dict(param_config)
        if var_args_name is not None and var_args_name in new_param_config:
            for i, arg in enumerate(new_param_config.pop(var_args_name)):
                new_param_config[f"{var_args_name}_{i}"] = arg
        if var_kwargs_name is not None and var_kwargs_name in new_param_config:
            for k, v in new_param_config.pop(var_kwargs_name).items():
                new_param_config[k] = v
        return new_param_config

    @classmethod
    def roll_param_config(cls, param_config: dict, ann_args: tp.AnnArgs) -> dict:
        """Roll a parameter config."""
        var_args_name, var_kwargs_name = cls.get_var_arg_names(ann_args)
        new_param_config = dict(param_config)
        if var_args_name is not None:
            _args = ()
            while True:
                k = f"{var_args_name}_{len(_args)}"
                if k in new_param_config:
                    _args += (new_param_config.pop(k),)
                else:
                    break
            new_param_config[var_args_name] = _args
        if var_kwargs_name is not None:
            new_param_config[var_kwargs_name] = {}
            for k in list(new_param_config.keys()):
                if k not in ann_args:
                    new_param_config[var_kwargs_name][k] = new_param_config.pop(k)
        return new_param_config

    @classmethod
    def select_comb(
        cls,
        param_configs: tp.List[tp.Kwargs],
        param_index: tp.Optional[tp.Index],
        selection: tp.Selection,
        single_comb: bool = False,
        template_context: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.List[tp.Kwargs], tp.Optional[tp.Index], bool]:
        """Select a parameter combination from parameter configs and index."""
        selection = substitute_templates(selection, template_context, eval_id="selection")
        if isinstance(selection, _NoResult):
            raise NoResultsException
        found_param = False
        kind = None
        if isinstance(selection, PosSel):
            selection = selection.value
            kind = "positions"
        elif isinstance(selection, LabelSel):
            selection = selection.value
            kind = "labels"
        if checks.is_hashable(selection):
            if kind == "positions" or (kind is None and checks.is_int(selection)):
                selection = {selection}
                found_param = True
                single_comb = True
            elif kind == "labels" or (kind is None and param_index is not None and selection in param_index):
                selection = {param_index.get_loc(selection)}
                found_param = True
                single_comb = True
        if not found_param:
            if checks.is_iterable(selection):
                new_selection = set()
                for s in selection:
                    if isinstance(s, PosSel):
                        s = s.value
                        kind = "positions"
                    elif isinstance(s, LabelSel):
                        s = s.value
                        kind = "labels"
                    if kind == "positions" or (kind is None and checks.is_int(s)):
                        new_selection.add(s)
                    elif kind == "labels" or (kind is None and param_index is not None and s in param_index):
                        new_selection.add(param_index.get_loc(s))
                    else:
                        raise ValueError(f"Selection {selection} couldn't be matched with parameter index")
                selection = new_selection
                if len(selection) == 0:
                    raise ValueError(f"Selection {selection} is empty")
            else:
                raise ValueError(f"Selection {selection} couldn't be matched with parameter index")
        if param_index is not None:
            param_index = param_index[list(selection)]
        new_param_configs = []
        selection = selection.copy()
        for i, x in enumerate(param_configs):
            if i in selection:
                new_param_configs.append(x)
                selection.remove(i)
                if len(selection) == 0:
                    break
        if len(selection) > 0:
            raise ValueError(f"Selection {selection} couldn't be matched")
        return new_param_configs, param_index, single_comb

    @classmethod
    def yield_funcs_args(
        cls,
        func: tp.Callable,
        ann_args: tp.AnnArgs,
        param_configs: tp.List[tp.Kwargs],
        template_context: tp.KwargsLike = None,
    ) -> tp.FuncsArgs:
        """Yield functions and their arguments for execution."""
        for p, param_config in enumerate(param_configs):
            _template_context = dict(template_context)
            _template_context["config_idx"] = p
            _ann_args = dict()
            for k, v in ann_args.items():
                v = dict(v)
                v["value"] = param_config[k]
                _ann_args[k] = v
            _args, _kwargs = ann_args_to_args(_ann_args)
            _args = substitute_templates(_args, _template_context, eval_id="args")
            _kwargs = substitute_templates(_kwargs, _template_context, eval_id="kwargs")
            yield func, _args, _kwargs

    @classmethod
    def get_mono_chunk_indices(
        cls,
        param_configs: tp.List[tp.Kwargs],
        mono_min_size: tp.Optional[int] = None,
        mono_n_chunks: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_len: tp.Optional[tp.Union[str, int]] = None,
        mono_chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    ) -> tp.List[tp.List[int]]:
        """Get the indices of each mono-chunk."""
        if mono_chunk_meta is None:
            from vectorbtpro.utils.chunking import yield_chunk_meta

            mono_chunk_meta = yield_chunk_meta(
                n_chunks=mono_n_chunks,
                size=len(param_configs),
                min_size=mono_min_size,
                chunk_len=mono_chunk_len,
            )

        last_idx = -1
        indices_sorted = True
        mono_chunk_indices = []
        for _chunk_meta in mono_chunk_meta:
            if _chunk_meta.indices is not None:
                chunk_indices = list(_chunk_meta.indices)
            else:
                if _chunk_meta.start is None or _chunk_meta.end is None:
                    raise ValueError("Each chunk must have a start and an end index")
                chunk_indices = list(range(_chunk_meta.start, _chunk_meta.end))
            if indices_sorted:
                for idx in chunk_indices:
                    if idx != last_idx + 1:
                        indices_sorted = False
                        break
                    last_idx = idx
            mono_chunk_indices.append(chunk_indices)
        return mono_chunk_indices

    def build_mono_chunk_config(
        self,
        chunk_indices: tp.List[int],
        param_configs: tp.List[tp.Kwargs],
        param_config_keys: tp.Set[str],
        ann_args: tp.AnnArgs,
        flat_ann_args: tp.Optional[tp.FlatAnnArgs] = None,
        mono_reduce: tp.Union[bool, tp.Kwargs] = None,
        mono_merge_func: tp.Union[tp.MergeFuncLike, tp.Dict[str, tp.MergeFuncLike]] = None,
        mono_merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        """Build the parameter config for a mono-chunk."""
        if flat_ann_args is None:
            flat_ann_args = flatten_ann_args(ann_args)
        new_param_config = dict()
        all_same = dict()
        for j, i in enumerate(chunk_indices):
            param_config = self.unroll_param_config(param_configs[i], ann_args)
            for k, v in param_config.items():
                if j == 0:
                    new_param_config[k] = [v]
                    all_same[k] = True
                else:
                    if k not in new_param_config:
                        raise ValueError(
                            f"Mono-chunks cannot be built because key '{k}' is not present in all parameter configs"
                        )
                    if v is not new_param_config[k][-1]:
                        all_same[k] = False
                    new_param_config[k].append(v)

        for k, v in new_param_config.items():
            k_reduce = None
            k_merge_func = None
            k_merge_kwargs = None
            if mono_reduce is not None:
                if isinstance(mono_reduce, dict):
                    if k in mono_reduce:
                        k_reduce = mono_reduce[k]
                else:
                    k_reduce = mono_reduce
            if mono_merge_func is not None:
                if isinstance(mono_merge_func, dict):
                    if k in mono_merge_func:
                        k_merge_func = mono_merge_func[k]
                else:
                    k_merge_func = mono_merge_func
            if mono_merge_kwargs is not None:
                if set(mono_merge_kwargs.keys()) <= param_config_keys:
                    if k in mono_merge_kwargs:
                        k_merge_kwargs = mono_merge_kwargs[k]
                else:
                    k_merge_kwargs = mono_merge_kwargs
            if k in flat_ann_args:
                ann_arg = flat_ann_args[k]
                if "value" in ann_arg and isinstance(ann_arg["value"], Param):
                    if k_reduce is None and ann_arg["value"].mono_reduce is not None:
                        k_reduce = ann_arg["value"].mono_reduce
                    if k_merge_func is None and ann_arg["value"].mono_merge_func is not None:
                        k_merge_func = ann_arg["value"].mono_merge_func
                    if k_merge_kwargs is None and ann_arg["value"].mono_merge_kwargs is not None:
                        k_merge_kwargs = ann_arg["value"].mono_merge_kwargs
            if k_reduce is None:
                k_reduce = all_same[k]
            elif k_reduce and not all_same[k]:
                k_reduce = False
            if k_reduce:
                new_param_config[k] = new_param_config[k][0]
            elif k_merge_func is not None:
                if isinstance(k_merge_func, MergeFunc):
                    k_merge_func = k_merge_func.replace(
                        merge_kwargs=k_merge_kwargs,
                        context=template_context,
                        sub_id_prefix="mono_",
                    )
                else:
                    k_merge_func = MergeFunc(
                        k_merge_func,
                        merge_kwargs=k_merge_kwargs,
                        context=template_context,
                        sub_id_prefix="mono_",
                    )
                new_param_config[k] = k_merge_func(v)

        return self.roll_param_config(new_param_config, ann_args)

    def run(
        self,
        *args,
        param_configs: tp.Optional[tp.MaybeSequence[tp.Kwargs]] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Any:
        """Parameterize arguments and run the function.

        Does the following:

        1. Searches for values wrapped with the class `Param` in any nested dicts and tuples
        using `Parameterizer.find_params_in_obj`
        2. Uses `combine_params` to build parameter combinations
        3. Maps parameter combinations to configs using `Parameterizer.param_product_to_objs`
        4. Generates and resolves parameter configs by combining combinations from the step above with
        `param_configs` that is optionally passed by the user. User-defined `param_configs` have more priority.
        5. If `selection` is not None, substitutes it as a template, translates it into indices that
        can be mapped to `param_index`, and selects them from all the objects generated above.
        6. Builds mono-chunks if `mono_n_chunks`, `mono_chunk_len`, or `mono_chunk_meta` is not None
        7. Extracts arguments and keyword arguments from each parameter config and substitutes any templates (lazily)
        8. Passes each set of the function and its arguments to `vectorbtpro.utils.execution.execute` for execution
        9. Optionally, post-processes and merges the results by passing them and `**merge_kwargs` to `merge_func`

        Argument `param_configs` accepts either a list of dictionaries with arguments named by their names
        in the signature, or a dictionary of dictionaries, where keys are config names. If a list is passed,
        each dictionary can also contain the key `_name` to give the config a name. Variable arguments
        can be passed either in the rolled (`args=(...), kwargs={...}`) or unrolled
        (`args_0=..., args_1=..., some_kwarg=...`) format.

        !!! important
            Defining a parameter and listing the same argument in `param_configs` will prioritize
            the config over the parameter, even though the parameter will still be visible in the final columns.
            There are no checks implemented to raise an error when this happens!

        If mono-chunking is enabled, parameter configs will be distributed over chunks. Any argument that
        is wrapped with `Param` or appears in `mono_merge_func` will be aggregated into a list. It will
        be merged using either `Param.mono_merge_func` or `mono_merge_func` in the same way as `merge_func`.
        If an argument satisfies neither of the above requirements and all its values within a chunk are same,
        this value will be passed as a scalar. Arguments `mono_merge_func` and `mono_merge_kwargs`
        must be dictionaries where keys are argument names in the flattened signature and values are
        functions and keyword arguments respectively.

        If `vectorbtpro.utils.selection.NoResult` is returned, will skip the current iteration and
        remove it from the final index.

        For defaults, see `vectorbtpro._settings.params`.
        """
        param_search_kwargs = self.resolve_setting(self.param_search_kwargs, "param_search_kwargs", merge=True)
        skip_single_comb = self.resolve_setting(self.skip_single_comb, "skip_single_comb")
        template_context = self.resolve_setting(self.template_context, "template_context", merge=True)
        build_grid = self.resolve_setting(self.build_grid, "build_grid")
        grid_indices = self.resolve_setting(self.grid_indices, "grid_indices")
        random_subset = self.resolve_setting(self.random_subset, "random_subset")
        random_replace = self.resolve_setting(self.random_replace, "random_replace")
        random_sort = self.resolve_setting(self.random_sort, "random_sort")
        max_guesses = self.resolve_setting(self.max_guesses, "max_guesses")
        max_misses = self.resolve_setting(self.max_misses, "max_misses")
        seed = self.resolve_setting(self.seed, "seed")
        index_stack_kwargs = self.resolve_setting(self.index_stack_kwargs, "index_stack_kwargs", merge=True)
        name_tuple_to_str = self.resolve_setting(self.name_tuple_to_str, "name_tuple_to_str")
        merge_func = self.resolve_setting(self.merge_func, "merge_func")
        merge_kwargs = self.resolve_setting(self.merge_kwargs, "merge_kwargs", merge=True)
        selection = self.resolve_setting(self.selection, "selection")
        forward_kwargs_as = self.resolve_setting(self.forward_kwargs_as, "forward_kwargs_as", merge=True)
        mono_min_size = self.resolve_setting(self.mono_min_size, "mono_min_size")
        mono_n_chunks = self.resolve_setting(self.mono_n_chunks, "mono_n_chunks")
        mono_chunk_len = self.resolve_setting(self.mono_chunk_len, "mono_chunk_len")
        mono_chunk_meta = self.resolve_setting(self.mono_chunk_meta, "mono_chunk_meta")
        mono_reduce = self.resolve_setting(self.mono_reduce, "mono_reduce")
        mono_merge_func = self.resolve_setting(self.mono_merge_func, "mono_merge_func")
        mono_merge_kwargs = self.resolve_setting(self.mono_merge_kwargs, "mono_merge_kwargs", merge=True)
        return_meta = self.resolve_setting(self.return_meta, "return_meta")
        return_param_index = self.resolve_setting(self.return_param_index, "return_param_index")
        execute_kwargs = self.resolve_setting(self.execute_kwargs, "execute_kwargs", merge=True)

        if param_configs is None:
            param_configs = []

        parsed_merge_func = parse_merge_func(self.func)
        if parsed_merge_func is not None:
            if merge_func is not None:
                raise ValueError(
                    f"Two conflicting merge functions: {parsed_merge_func} (annotations) and {merge_func} (merge_func)"
                )
            merge_func = parsed_merge_func

        if len(forward_kwargs_as) > 0:
            new_kwargs = dict()
            for k, v in kwargs.items():
                if k in forward_kwargs_as:
                    new_kwargs[forward_kwargs_as.pop(k)] = v
                else:
                    new_kwargs[k] = v
            kwargs = new_kwargs
        if len(forward_kwargs_as) > 0:
            for k, v in forward_kwargs_as.items():
                kwargs[v] = locals()[k]

        template_context["ann_args"] = annotate_args(
            self.func,
            args,
            kwargs,
            allow_partial=True,
            attach_annotations=True,
        )
        template_context["flat_ann_args"] = flatten_ann_args(template_context["ann_args"])
        if has_annotatables(self.func):
            template_context["flat_ann_args"] = self.parse_and_inject_params(
                template_context["flat_ann_args"],
                eval_id=eval_id,
            )
            template_context["ann_args"] = unflatten_ann_args(
                template_context["flat_ann_args"],
                partial_ann_args=template_context["ann_args"],
            )

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
            param_config = self.unroll_param_config(param_config, template_context["ann_args"])
            if "_name" in param_config and param_config["_name"] is not None:
                pc_names.append(param_config.pop("_name"))
                pc_names_none = False
            else:
                pc_names.append(n_param_configs)
            param_configs[i] = param_config
            n_param_configs += 1

        paramable_kwargs = {}
        for k, v in template_context["flat_ann_args"].items():
            if "value" in v:
                paramable_kwargs[k] = v["value"]
        param_dct = self.find_params_in_obj(paramable_kwargs, eval_id=eval_id, **param_search_kwargs)
        param_index = None
        if len(param_dct) > 0:
            param_product, param_index = combine_params(
                param_dct,
                build_grid=build_grid,
                grid_indices=grid_indices,
                random_subset=random_subset,
                random_replace=random_replace,
                random_sort=random_sort,
                max_guesses=max_guesses,
                max_misses=max_misses,
                seed=seed,
                index_stack_kwargs=index_stack_kwargs,
                name_tuple_to_str=name_tuple_to_str,
                raise_empty_error=True,
            )
            product_param_configs = self.param_product_to_objs(paramable_kwargs, param_product)
            if len(param_configs) == 0:
                param_configs = product_param_configs
            else:
                new_param_configs = []
                for i in range(len(product_param_configs)):
                    for param_config in param_configs:
                        new_param_config = merge_dicts(product_param_configs[i], param_config)
                        new_param_configs.append(new_param_config)
                param_configs = new_param_configs

        n_config_params = len(pc_names)
        if param_index is not None:
            if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                new_param_index = param_index
            else:
                from vectorbtpro.base.indexes import combine_indexes

                new_param_index = combine_indexes(
                    (
                        param_index,
                        pd.Index(pc_names, name="param_config"),
                    ),
                    **index_stack_kwargs,
                )
        else:
            if n_config_params == 0 or (n_config_params == 1 and pc_names_none):
                new_param_index = None
            else:
                new_param_index = pd.Index(pc_names, name="param_config")
        template_context["param_index"] = new_param_index

        if len(param_configs) == 0:
            template_context["single_comb"] = True
            param_configs.append(dict())
        else:
            template_context["single_comb"] = False

        param_config_keys = set()
        new_param_configs = []
        for param_config in param_configs:
            new_param_config = merge_dicts(paramable_kwargs, param_config)
            for k in new_param_config:
                param_config_keys.add(k)
            new_param_config = self.roll_param_config(new_param_config, template_context["ann_args"])
            new_param_configs.append(new_param_config)
        template_context["param_configs"] = new_param_configs

        if selection is not None:
            new_param_configs, new_param_index, new_single_comb = self.select_comb(
                template_context["param_configs"],
                template_context["param_index"],
                selection,
                single_comb=template_context["single_comb"],
                template_context=template_context,
            )
            template_context["param_configs"] = new_param_configs
            template_context["param_index"] = new_param_index
            template_context["single_comb"] = new_single_comb

        if skip_single_comb and template_context["single_comb"]:
            funcs_args = list(
                self.yield_funcs_args(
                    self.func,
                    template_context["ann_args"],
                    template_context["param_configs"],
                    template_context=template_context,
                )
            )
            result = funcs_args[0][0](*funcs_args[0][1], **funcs_args[0][2])
            if isinstance(result, _NoResult):
                raise NoResultsException
            return result

        if mono_n_chunks is not None or mono_chunk_len is not None or mono_chunk_meta is not None:
            template_context["mono_chunk_indices"] = self.get_mono_chunk_indices(
                template_context["param_configs"],
                mono_min_size=mono_min_size,
                mono_n_chunks=mono_n_chunks,
                mono_chunk_len=mono_chunk_len,
                mono_chunk_meta=mono_chunk_meta,
            )
            new_param_configs = []
            new_param_index = []
            for chunk_indices in template_context["mono_chunk_indices"]:
                new_param_configs.append(
                    self.build_mono_chunk_config(
                        chunk_indices,
                        template_context["param_configs"],
                        param_config_keys,
                        template_context["ann_args"],
                        flat_ann_args=template_context["flat_ann_args"],
                        mono_reduce=mono_reduce,
                        mono_merge_func=mono_merge_func,
                        mono_merge_kwargs=mono_merge_kwargs,
                        template_context=template_context,
                    )
                )
                if template_context["param_index"] is not None:
                    new_param_index.append(template_context["param_index"][chunk_indices])
            template_context["param_configs"] = new_param_configs
            if template_context["param_index"] is not None:
                template_context["param_index"] = new_param_index
        else:
            template_context["mono_chunk_indices"] = None

        template_context["funcs_args"] = self.yield_funcs_args(
            self.func,
            template_context["ann_args"],
            template_context["param_configs"],
            template_context=template_context,
        )

        if return_meta:
            return dict(
                ann_args=template_context["ann_args"],
                flat_ann_args=template_context["flat_ann_args"],
                single_comb=template_context["single_comb"],
                param_configs=template_context["param_configs"],
                param_index=template_context["param_index"],
                mono_chunk_indices=template_context["mono_chunk_indices"],
                funcs_args=template_context["funcs_args"],
            )

        results = execute(
            template_context["funcs_args"],
            n_calls=len(template_context["param_configs"]),
            **execute_kwargs,
        )

        skip_indices = set()
        for i, result in enumerate(results):
            if isinstance(result, _NoResult):
                skip_indices.add(i)
        if len(skip_indices) > 0:
            new_results = []
            keep_indices = []
            for i, result in enumerate(results):
                if i not in skip_indices:
                    new_results.append(result)
                    keep_indices.append(i)
            results = new_results
            if template_context["param_index"] is not None:
                template_context["param_index"] = template_context["param_index"][keep_indices]
        if len(results) == 0:
            raise NoResultsException

        if merge_func is not None:
            from vectorbtpro.base.merging import is_merge_func_from_config

            if is_merge_func_from_config(merge_func):
                merge_kwargs = merge_dicts(dict(keys=template_context["param_index"]), merge_kwargs)
            if isinstance(merge_func, MergeFunc):
                merge_func = merge_func.replace(merge_kwargs=merge_kwargs, context=template_context)
            else:
                merge_func = MergeFunc(merge_func, merge_kwargs=merge_kwargs, context=template_context)
            if return_param_index:
                return merge_func(results), template_context["param_index"]
            return merge_func(results)
        if return_param_index:
            return results, template_context["param_index"]
        return results


def parameterized(
    *args,
    parameterizer_cls: tp.Optional[tp.Type[Parameterizer]] = None,
    param_search_kwargs: tp.KwargsLike = None,
    skip_single_comb: tp.Optional[bool] = None,
    template_context: tp.KwargsLike = None,
    build_grid: tp.Optional[bool] = None,
    grid_indices: tp.Union[None, slice, tp.Sequence[int]] = None,
    random_subset: tp.Union[None, int, float] = None,
    random_replace: tp.Optional[bool] = None,
    random_sort: tp.Optional[bool] = None,
    max_guesses: tp.Union[None, int, float] = None,
    max_misses: tp.Union[None, int, float] = None,
    seed: tp.Optional[int] = None,
    index_stack_kwargs: tp.KwargsLike = None,
    name_tuple_to_str: tp.Union[None, bool, tp.Callable] = None,
    merge_func: tp.Optional[tp.MergeFuncLike] = None,
    merge_kwargs: tp.KwargsLike = None,
    selection: tp.Optional[tp.Selection] = None,
    forward_kwargs_as: tp.KwargsLike = None,
    mono_min_size: tp.Optional[int] = None,
    mono_n_chunks: tp.Optional[tp.Union[str, int]] = None,
    mono_chunk_len: tp.Optional[tp.Union[str, int]] = None,
    mono_chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    mono_reduce: tp.Union[bool, tp.Kwargs] = None,
    mono_merge_func: tp.Union[tp.MergeFuncLike, tp.Dict[str, tp.MergeFuncLike]] = None,
    mono_merge_kwargs: tp.KwargsLike = None,
    return_meta: tp.Optional[bool] = None,
    return_param_index: tp.Optional[bool] = None,
    execute_kwargs: tp.KwargsLike = None,
    eval_id: tp.Optional[tp.Hashable] = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that parameterizes the inputs of a function using `Parameterizer`.

    Returns a new function with the same signature as the passed one.

    Each option can be modified in the `options` attribute of the wrapper function or
    directly passed as a keyword argument with a leading underscore.

    Keyword arguments `**kwargs` are passed directly to `vectorbtpro.utils.execution.execute`.

    Usage:
        * No parameters, no parameter configs:

        ```pycon
        >>> from vectorbtpro import *

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

        * Using annotations:

        ```pycon
        >>> @vbt.parameterized
        ... def my_ma(
        ...     sr_or_df,
        ...     window: vbt.Param,
        ...     wtype: vbt.Param = "simple",
        ...     minp=0,
        ...     adjust=False
        ... ) -> vbt.MergeFunc("column_stack"):
        ...     return sr_or_df.vbt.ma(window, wtype=wtype, minp=minp, adjust=adjust)

        >>> my_ma(sr, [3, 4, 5], ["simple", "exp"])
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
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        params_cfg = settings["params"]

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            def _resolve_key(key, merge=False):
                if "_" + key in kwargs:
                    if merge:
                        return merge_dicts(wrapper.options[key], kwargs.pop("_" + key))
                    return kwargs.pop("_" + key)
                return wrapper.options.get(key)

            parameterizer_cls = wrapper.options["parameterizer_cls"]
            if parameterizer_cls is None:
                parameterizer_cls = params_cfg["parameterizer_cls"]
            if parameterizer_cls is None:
                parameterizer_cls = Parameterizer
            return parameterizer_cls(
                func,
                param_search_kwargs=_resolve_key("param_search_kwargs", merge=True),
                skip_single_comb=_resolve_key("skip_single_comb"),
                template_context=_resolve_key("template_context", merge=True),
                build_grid=_resolve_key("build_grid"),
                grid_indices=_resolve_key("grid_indices"),
                random_subset=_resolve_key("random_subset"),
                random_replace=_resolve_key("random_replace"),
                random_sort=_resolve_key("random_sort"),
                max_guesses=_resolve_key("max_guesses"),
                max_misses=_resolve_key("max_misses"),
                seed=_resolve_key("seed"),
                index_stack_kwargs=_resolve_key("index_stack_kwargs", merge=True),
                name_tuple_to_str=_resolve_key("name_tuple_to_str"),
                merge_func=_resolve_key("merge_func"),
                merge_kwargs=_resolve_key("merge_kwargs", merge=True),
                selection=_resolve_key("selection"),
                forward_kwargs_as=_resolve_key("forward_kwargs_as", merge=True),
                mono_min_size=_resolve_key("mono_min_size"),
                mono_n_chunks=_resolve_key("mono_n_chunks"),
                mono_chunk_len=_resolve_key("mono_chunk_len"),
                mono_chunk_meta=_resolve_key("mono_chunk_meta"),
                mono_reduce=_resolve_key("mono_reduce"),
                mono_merge_func=_resolve_key("mono_merge_func"),
                mono_merge_kwargs=_resolve_key("mono_merge_kwargs", merge=True),
                return_meta=_resolve_key("return_meta"),
                return_param_index=_resolve_key("return_param_index"),
                execute_kwargs=_resolve_key("execute_kwargs", merge=True),
            ).run(*args, eval_id=eval_id, **kwargs)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_parameterized = True
        wrapper.options = Config(
            dict(
                parameterizer_cls=parameterizer_cls,
                param_search_kwargs=param_search_kwargs,
                skip_single_comb=skip_single_comb,
                template_context=template_context,
                build_grid=build_grid,
                grid_indices=grid_indices,
                random_subset=random_subset,
                random_replace=random_replace,
                random_sort=random_sort,
                max_guesses=max_guesses,
                max_misses=max_misses,
                seed=seed,
                index_stack_kwargs=index_stack_kwargs,
                name_tuple_to_str=name_tuple_to_str,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                selection=selection,
                forward_kwargs_as=forward_kwargs_as,
                mono_min_size=mono_min_size,
                mono_n_chunks=mono_n_chunks,
                mono_chunk_len=mono_chunk_len,
                mono_chunk_meta=mono_chunk_meta,
                mono_reduce=mono_reduce,
                mono_merge_func=mono_merge_func,
                mono_merge_kwargs=mono_merge_kwargs,
                return_meta=return_meta,
                return_param_index=return_param_index,
                execute_kwargs=merge_dicts(kwargs, execute_kwargs),
                eval_id=eval_id,
            ),
            options_=dict(
                frozen_keys=True,
                as_attrs=True,
            ),
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
