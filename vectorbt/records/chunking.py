# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Extensions to `vectorbt.utils.chunking`."""

import attr
import uuid

import numpy as np

from vectorbt import _typing as tp
from vectorbt.base.chunking import get_group_lens_slice
from vectorbt.utils.chunking import (
    ArgGetter,
    ChunkMeta,
    ArgSizer,
    ChunkSlicer,
    ChunkMapper
)


class ColLensSizer(ArgSizer):
    """Class for getting the size from column lengths.

    Argument can be either a column map tuple or a column lengths array."""

    def get_size(self, ann_args: tp.AnnArgs) -> int:
        arg = self.get_arg(ann_args)
        if isinstance(arg, tuple):
            return len(arg[1])
        return len(arg)


class ColLensSlicer(ChunkSlicer):
    """Class for slicing multiple elements from column lengths based on the chunk range."""

    def take(self, obj: tp.Union[tp.ColLens, tp.ColMap], chunk_meta: ChunkMeta, **kwargs) -> tp.ColMap:
        if isinstance(obj, tuple):
            return obj[1][chunk_meta.start:chunk_meta.end]
        return obj[chunk_meta.start:chunk_meta.end]


@attr.s(frozen=True)
class ColLensMapper(ChunkMapper, ArgGetter):
    """Class for mapping chunk metadata to per-column record lengths.

    Argument can be either a column map tuple or a column lengths array."""

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        col_lens = self.get_arg(ann_args)
        if isinstance(col_lens, tuple):
            col_lens = col_lens[1]
        col_lens_slice = get_group_lens_slice(col_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=col_lens_slice.start,
            end=col_lens_slice.stop,
            indices=None
        )


col_lens_mapper = ColLensMapper(arg_query=r'(col_lens|col_map)')
"""Default instance of `ColLensMapper`."""


class ColMapSlicer(ChunkSlicer):
    """Class for slicing multiple elements from a column map based on the chunk range."""

    def take(self, obj: tp.ColMap, chunk_meta: ChunkMeta, **kwargs) -> tp.ColMap:
        col_idxs, col_lens = obj
        col_lens = col_lens[chunk_meta.start:chunk_meta.end]
        return np.arange(np.sum(col_lens)), col_lens


@attr.s(frozen=True)
class ColIdxsMapper(ChunkMapper, ArgGetter):
    """Class for mapping chunk metadata to per-column record indices.

    Argument must be a column map tuple."""

    def map(self, chunk_meta: ChunkMeta, ann_args: tp.Optional[tp.AnnArgs] = None, **kwargs) -> ChunkMeta:
        col_map = self.get_arg(ann_args)
        col_idxs, col_lens = col_map
        col_lens_slice = get_group_lens_slice(col_lens, chunk_meta)
        return ChunkMeta(
            uuid=str(uuid.uuid4()),
            idx=chunk_meta.idx,
            start=None,
            end=None,
            indices=col_idxs[col_lens_slice]
        )


col_idxs_mapper = ColIdxsMapper(arg_query='col_map')
"""Default instance of `ColIdxsMapper`."""


def fix_field_in_records(record_arrays: tp.List[tp.RecordArray],
                         chunk_meta: tp.Iterable[ChunkMeta],
                         ann_args: tp.Optional[tp.AnnArgs] = None,
                         mapper: tp.Optional[ChunkMapper] = None,
                         field: str = 'col') -> None:
    """Fix a field of the record array in each chunk."""
    for _chunk_meta in chunk_meta:
        if mapper is None:
            record_arrays[_chunk_meta.idx][field] += _chunk_meta.start
        else:
            _chunk_meta_mapped = mapper.map(_chunk_meta, ann_args=ann_args)
            record_arrays[_chunk_meta.idx][field] += _chunk_meta_mapped.start


def merge_records(results: tp.List[tp.RecordArray],
                  chunk_meta: tp.Iterable[ChunkMeta],
                  ann_args: tp.Optional[tp.AnnArgs] = None,
                  mapper: tp.Optional[ChunkMapper] = None) -> tp.RecordArray:
    """Merge chunks of record arrays.

    Mapper is only applied on the column field."""
    if 'col' in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, ann_args=ann_args, mapper=mapper, field='col')
    if 'group' in results[0].dtype.fields:
        fix_field_in_records(results, chunk_meta, field='group')
    return np.concatenate(results)
