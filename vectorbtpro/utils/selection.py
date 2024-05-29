# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for selecting."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define, MISSING

__all__ = [
    "PosSel",
    "LabelSel",
    "NoResult",
    "NoResultsException",
]


@define
class PosSel(DefineMixin):
    """Class that represents a selection by position."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(DefineMixin):
    """Class that represents a selection by label."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more labels."""


class _NoResult:
    pass


NoResult = _NoResult()
"""Return this object to ignore an iteration."""


class NoResultsException(Exception):
    """Gets raised when there are no results."""

    pass


def filter_no_results(
    objs: tp.Iterable[tp.Any],
    keys: tp.Optional[tp.Index] = MISSING,
    raise_error: bool = True,
) -> tp.Union[tp.List, tp.Tuple[tp.List, tp.Optional[tp.Index]]]:
    """Filter objects and keys by removing `NoResult` objects."""
    skip_indices = set()
    for i, obj in enumerate(objs):
        if isinstance(obj, _NoResult):
            skip_indices.add(i)
    if len(skip_indices) > 0:
        new_objs = []
        keep_indices = []
        for i, obj in enumerate(objs):
            if i not in skip_indices:
                new_objs.append(obj)
                keep_indices.append(i)
        objs = new_objs
        if keys is not None and keys is not MISSING:
            if isinstance(keys, pd.Index):
                keys = keys[keep_indices]
            else:
                keys = [keys[i] for i in keep_indices]
        if len(objs) == 0 and raise_error:
            raise NoResultsException
    if keys is not MISSING:
        return objs, keys
    return objs
