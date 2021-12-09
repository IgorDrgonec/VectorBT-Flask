# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules with utilities that are used throughout vectorbtpro."""

from vectorbtpro.utils.caching import Cacheable
from vectorbtpro.utils.chunking import (
    ChunkMeta,
    ArgChunkMeta,
    LenChunkMeta,
    ArgSizer,
    LenSizer,
    ShapeSizer,
    ArraySizer,
    ChunkMapper,
    ChunkSelector,
    ChunkSlicer,
    CountAdapter,
    ShapeSelector,
    ShapeSlicer,
    ArraySelector,
    ArraySlicer,
    SequenceTaker,
    MappingTaker,
    ArgsTaker,
    KwargsTaker,
    chunked
)
from vectorbtpro.utils.config import (
    atomic_dict,
    merge_dicts,
    Config,
    Configured,
    AtomicConfig
)
from vectorbtpro.utils.decorators import (
    cacheable_property,
    cached_property,
    cacheable,
    cached,
    cacheable_method,
    cached_method
)
from vectorbtpro.utils.docs import stringify
from vectorbtpro.utils.execution import SequenceEngine, DaskEngine, RayEngine
from vectorbtpro.utils.image_ import save_animation
from vectorbtpro.utils.jitting import jitted
from vectorbtpro.utils.parsing import Regex
from vectorbtpro.utils.profiling import Timer, MemTracer
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.schedule_ import AsyncJob, AsyncScheduler, CancelledError, ScheduleManager
from vectorbtpro.utils.template import Sub, Rep, RepEval, RepFunc, deep_substitute

__all__ = [
    'atomic_dict',
    'merge_dicts',
    'Config',
    'Configured',
    'AtomicConfig',
    'Sub',
    'Rep',
    'RepEval',
    'RepFunc',
    'deep_substitute',
    'Regex',
    'cacheable_property',
    'cached_property',
    'cacheable',
    'cached',
    'cacheable_method',
    'cached_method',
    'Cacheable',
    'set_seed',
    'save_animation',
    'AsyncJob',
    'AsyncScheduler',
    'CancelledError',
    'ScheduleManager',
    'SequenceEngine',
    'DaskEngine',
    'RayEngine',
    'ChunkMeta',
    'ArgChunkMeta',
    'LenChunkMeta',
    'ArgSizer',
    'LenSizer',
    'ShapeSizer',
    'ArraySizer',
    'ChunkMapper',
    'ChunkSelector',
    'ChunkSlicer',
    'CountAdapter',
    'ShapeSelector',
    'ShapeSlicer',
    'ArraySelector',
    'ArraySlicer',
    'SequenceTaker',
    'MappingTaker',
    'ArgsTaker',
    'KwargsTaker',
    'chunked',
    'Timer',
    'MemTracer',
    'stringify',
    'jitted'
]

__blacklist__ = []

try:
    import plotly
except ImportError:
    __blacklist__.append('figure')
else:
    from vectorbtpro.utils.figure import Figure, FigureWidget, make_figure, make_subplots

    __all__.append('Figure')
    __all__.append('FigureWidget')
    __all__.append('make_figure')
    __all__.append('make_subplots')

__pdoc__ = {k: False for k in __all__}
