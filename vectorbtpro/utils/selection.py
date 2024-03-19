# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for selecting."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import define

__all__ = [
    "PosSel",
    "LabelSel",
    "NoResult",
    "NoResultsException",
]


@define
class PosSel(define.mixin):
    """Class that represents a selection by position."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(define.mixin):
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
