# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules with utilities that are used throughout vectorbtpro."""

from vectorbtpro.utils.attr_ import deep_getattr
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
    chunked,
)
from vectorbtpro.utils.config import (
    atomic_dict,
    merge_dicts,
    ChildDict,
    Config,
    FrozenConfig,
    ReadonlyConfig,
    HybridConfig,
    Configured,
    AtomicConfig,
)
from vectorbtpro.utils.datetime_ import to_tzaware_datetime
from vectorbtpro.utils.decorators import (
    cacheable_property,
    cached_property,
    cacheable,
    cached,
    cacheable_method,
    cached_method,
)
from vectorbtpro.utils.execution import SequenceEngine, DaskEngine, RayEngine, execute
from vectorbtpro.utils.formatting import prettify, format_func
from vectorbtpro.utils.image_ import save_animation
from vectorbtpro.utils.jitting import jitted
from vectorbtpro.utils.params import generate_param_combs
from vectorbtpro.utils.parsing import Regex
from vectorbtpro.utils.profiling import Timer, MemTracer
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.schedule_ import AsyncJob, AsyncScheduler, CancelledError, ScheduleManager
from vectorbtpro.utils.template import Sub, Rep, RepEval, RepFunc, deep_substitute
from vectorbtpro.utils.module_ import create__all__

__blacklist__ = []

try:
    import plotly
except ImportError:
    __blacklist__.append("figure")
else:
    from vectorbtpro.utils.figure import Figure, FigureWidget, make_figure, make_subplots

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
