# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules with base classes and utilities for pandas objects, such as broadcasting."""

from vectorbtpro.base.accessors import BaseAccessor, BaseSRAccessor, BaseDFAccessor
from vectorbtpro.base.chunking import (
    GroupLensSizer,
    GroupLensSlicer,
    GroupLensMapper,
    GroupMapSlicer,
    GroupIdxsMapper,
    FlexArraySelector,
    FlexArraySlicer,
)
from vectorbtpro.base.grouping import *
from vectorbtpro.base.resampling import *
from vectorbtpro.base.indexes import repeat_index, tile_index, stack_indexes, combine_indexes
from vectorbtpro.base.indexing import (
    PandasIndexer,
    flex_select_auto_nb,
    hslice,
    RowIdx,
    ColIdx,
    RowPoints,
    RowRanges,
    ElemIdx,
    index_dict,
    get_index_points,
    get_index_ranges,
)
from vectorbtpro.base.reshaping import (
    to_1d_array,
    to_2d_array,
    to_per_row_array,
    to_per_col_array,
    BCO,
    Default,
    Ref,
    broadcast,
    broadcast_to,
)
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.base.merging import concat_merge, row_stack_merge, column_stack_merge
from vectorbtpro.utils.module_ import create__all__

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
