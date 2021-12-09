# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with records.

Records are the second form of data representation in vectorbtpro. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory."""

from vectorbtpro.records.base import Records
from vectorbtpro.records.chunking import (
    ColLensSizer,
    ColLensSlicer,
    ColLensMapper,
    ColMapSlicer,
    ColIdxsMapper
)
from vectorbtpro.records.mapped_array import MappedArray

__all__ = [
    'MappedArray',
    'Records',
    'ColLensSizer',
    'ColLensSlicer',
    'ColLensMapper',
    'ColMapSlicer',
    'ColIdxsMapper'
]

__pdoc__ = {k: False for k in __all__}
