# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for selecting."""

import attr

from vectorbtpro import _typing as tp

__all__ = [
    "PosSel",
    "LabelSel",
    "NoResult",
    "NoResultsException",
]


@attr.s(frozen=True)
class PosSel:
    """Class that represents a selection by position."""

    value: tp.MaybeIterable[tp.Hashable] = attr.ib()
    """Selection of one or more positions."""


@attr.s(frozen=True)
class LabelSel:
    """Class that represents a selection by label."""

    value: tp.MaybeIterable[tp.Hashable] = attr.ib()
    """Selection of one or more labels."""


class _NoResult:
    pass


NoResult = _NoResult()
"""Return this object to ignore an iteration."""


class NoResultsException(Exception):
    """Gets raised when there are no results."""

    pass
