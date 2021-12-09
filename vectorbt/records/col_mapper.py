# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Class for mapping column arrays."""

from vectorbt import _typing as tp
from vectorbt.base.reshaping import to_1d_array
from vectorbt.base.wrapping import ArrayWrapper, Wrapping
from vectorbt.jit_registry import jit_registry
from vectorbt.records import nb
from vectorbt.utils.decorators import cached_property, cached_method


class ColumnMapper(Wrapping):
    """Used by `vectorbt.records.base.Records` and `vectorbt.records.mapped_array.MappedArray`
    classes to make use of column and group metadata."""

    def __init__(self, wrapper: ArrayWrapper, col_arr: tp.Array1d, **kwargs) -> None:
        Wrapping.__init__(
            self,
            wrapper,
            col_arr=col_arr,
            **kwargs
        )

        self._col_arr = col_arr

    def _col_idxs_meta(self, col_idxs: tp.Array1d, jitted: tp.JittedOption = None) -> tp.Tuple[tp.Array1d, tp.Array1d]:
        """Get metadata of column indices.

        Returns element indices and new column array.
        Automatically decides whether to use column lengths or column map."""
        if self.is_sorted():
            func = jit_registry.resolve_option(nb.col_lens_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_lens, to_1d_array(col_idxs))  # faster
        else:
            func = jit_registry.resolve_option(nb.col_map_select_nb, jitted)
            new_indices, new_col_arr = func(self.col_map, to_1d_array(col_idxs))  # more flexible
        return new_indices, new_col_arr

    @property
    def col_arr(self) -> tp.Array1d:
        """Column array."""
        return self._col_arr

    @cached_method(whitelist=True)
    def get_col_arr(self, group_by: tp.GroupByLike = None) -> tp.Array1d:
        """Get group-aware column array."""
        group_arr = self.wrapper.grouper.get_groups(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        return col_arr

    @cached_property(whitelist=True)
    def col_lens(self) -> tp.ColLens:
        """Column lengths.

        Faster than `ColumnMapper.col_map` but only compatible with sorted columns."""
        func = jit_registry.resolve_option(nb.col_lens_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_lens(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None) -> tp.ColLens:
        """Get group-aware column lengths."""
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_lens
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_registry.resolve_option(nb.col_lens_nb, jitted)
        return func(col_arr, len(columns))

    @cached_property(whitelist=True)
    def col_map(self) -> tp.ColMap:
        """Column map.

        More flexible than `ColumnMapper.col_lens`.
        More suited for mapped arrays."""
        func = jit_registry.resolve_option(nb.col_map_nb, None)
        return func(self.col_arr, len(self.wrapper.columns))

    @cached_method(whitelist=True)
    def get_col_map(self, group_by: tp.GroupByLike = None, jitted: tp.JittedOption = None) -> tp.ColMap:
        """Get group-aware column map."""
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.col_map
        col_arr = self.get_col_arr(group_by=group_by)
        columns = self.wrapper.get_columns(group_by=group_by)
        func = jit_registry.resolve_option(nb.col_map_nb, jitted)
        return func(col_arr, len(columns))

    @cached_method(whitelist=True)
    def is_sorted(self, jitted: tp.JittedOption = None) -> bool:
        """Check whether column array is sorted."""
        func = jit_registry.resolve_option(nb.is_col_sorted_nb, jitted)
        return func(self.col_arr)
