# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Modules with utilities that are used throughout vectorbt."""

from vectorbt.utils.caching import Cacheable
from vectorbt.utils.chunking import (
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
from vectorbt.utils.config import (
    atomic_dict,
    merge_dicts,
    Config,
    Configured,
    AtomicConfig
)
from vectorbt.utils.decorators import (
    cacheable_property,
    cached_property,
    cacheable,
    cached,
    cacheable_method,
    cached_method
)
from vectorbt.utils.docs import stringify
from vectorbt.utils.execution import SequenceEngine, DaskEngine, RayEngine
from vectorbt.utils.image_ import save_animation
from vectorbt.utils.jitting import jitted
from vectorbt.utils.parsing import Regex
from vectorbt.utils.profiling import Timer, MemTracer
from vectorbt.utils.random_ import set_seed
from vectorbt.utils.schedule_ import AsyncJob, AsyncScheduler, CancelledError, ScheduleManager
from vectorbt.utils.template import Sub, Rep, RepEval, RepFunc, deep_substitute

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
    from vectorbt.utils.figure import Figure, FigureWidget, make_figure, make_subplots

    __all__.append('Figure')
    __all__.append('FigureWidget')
    __all__.append('make_figure')
    __all__.append('make_subplots')

__pdoc__ = {k: False for k in __all__}
