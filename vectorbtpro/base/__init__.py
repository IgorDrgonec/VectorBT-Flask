# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules with base classes and utilities for pandas objects, such as broadcasting."""

from vectorbtpro.base.chunking import GroupLensMapper, FlexArraySelector, FlexArraySlicer
from vectorbtpro.base.grouping import Grouper
from vectorbtpro.base.indexing import PandasIndexer
from vectorbtpro.base.reshaping import BCO, Default, Ref, broadcast, broadcast_to
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping

__all__ = [
    'ArrayWrapper',
    'Wrapping',
    'Grouper',
    'GroupLensMapper',
    'FlexArraySelector',
    'FlexArraySlicer',
    'BCO',
    'Default',
    'Ref',
    'broadcast',
    'broadcast_to',
    'PandasIndexer'
]

__pdoc__ = {k: False for k in __all__}
