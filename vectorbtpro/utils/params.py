# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for working with parameters."""

import attr
import itertools
from collections import defaultdict, OrderedDict
from collections.abc import Callable

import numpy as np
import pandas as pd
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.random_ import set_seed


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

    value: tp.MaybeSequence[tp.Param] = attr.ib()
    """One or more parameter values."""

    product_idx: tp.Optional[int] = attr.ib(default=None)
    """Index of the product the parameter takes part in.

    Parameters in the same product are stacked together, not combined, 
    and appear in the index hierarchy next to each other.

    Product index can be used to order index levels: the higher the product index, 
    the lower the index level. Index levels with the same product index appear in the same 
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
    product_idx_values = defaultdict(OrderedDict)
    product_indexes = OrderedDict()
    product_idx_seen = False
    curr_idx = 0
    max_idx = 0
    for k, p in param_dct.items():
        if p.product_idx is None:
            if product_idx_seen:
                raise ValueError("Please provide product index for all product parameters")
            product_idx = curr_idx
        else:
            if curr_idx > 0 and not product_idx_seen:
                raise ValueError("Please provide product index for all product parameters")
            product_idx_seen = True
            product_idx = p.product_idx
        if product_idx > max_idx:
            max_idx = product_idx

        product_idx_values[product_idx][k] = list(p.value)
        if p.keys is not None:
            if isinstance(p.keys, pd.Index):
                product_indexes[k] = p.keys
            else:
                product_indexes[k] = pd.Index(p.keys, name=k)
        else:
            product_indexes[k] = indexes.index_from_values(p.value, name=k)
        curr_idx += 1

    # Build an operation tree and parameter index
    op_tree_operands = []
    param_keys = []
    for product_idx in range(max_idx + 1):
        if product_idx not in product_idx_values:
            raise ValueError("Group index must come in a strict order starting with 0 and without gaps")
        for k in product_idx_values[product_idx].keys():
            param_keys.append(k)

        # Broadcast parameter arrays
        param_lists = tuple(product_idx_values[product_idx].values())
        if len(param_lists) > 1:
            op_tree_operands.append((zip, *broadcast_params(param_lists)))
        else:
            op_tree_operands.append(param_lists[0])

        # Stack or combine parameter indexes together
        levels = []
        for k in product_idx_values[product_idx].keys():
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
