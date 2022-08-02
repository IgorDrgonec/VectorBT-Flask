# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""General types used across vectorbtpro."""

from enum import EnumMeta
from datetime import datetime, timedelta, tzinfo, time
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
from mypy_extensions import VarArg
from pandas import Series, DataFrame as Frame, Index
from pandas.tseries.offsets import BaseOffset
from pandas.core.groupby import GroupBy as PandasGroupBy
from pandas.core.resample import Resampler as PandasResampler

try:
    from plotly.graph_objects import Figure, FigureWidget
    from plotly.basedatatypes import BaseFigure, BaseTraceType
except ImportError:
    Figure = Any
    FigureWidget = Any
    BaseFigure = Any
    BaseTraceType = Any

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if TYPE_CHECKING:
    from vectorbtpro.utils.parsing import Regex
    from vectorbtpro.utils.execution import ExecutionEngine
    from vectorbtpro.utils.chunking import Sizer, ChunkTaker, ChunkMeta, ChunkMetaGenerator
    from vectorbtpro.utils.jitting import Jitter
else:
    Regex = "Regex"
    ExecutionEngine = "ExecutionEngine"
    Sizer = "Sizer"
    ChunkTaker = "ChunkTaker"
    ChunkMeta = "ChunkMeta"
    ChunkMetaGenerator = "ChunkMetaGenerator"
    TraceUpdater = "TraceUpdater"
    Jitter = "Jitter"

# Generic types
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Scalars
Scalar = Union[str, float, int, complex, bool, object, np.generic]
Number = Union[int, float, complex, np.number, np.bool_]
Int = Union[int, np.integer]
Float = Union[float, np.floating]
IntFloat = Union[Int, Float]

# Basic sequences
MaybeTuple = Union[T, Tuple[T, ...]]
MaybeList = Union[T, List[T]]
TupleList = Union[List[T], Tuple[T, ...]]
MaybeTupleList = Union[T, List[T], Tuple[T, ...]]
MaybeIterable = Union[T, Iterable[T]]
MaybeSequence = Union[T, Sequence[T]]
MaybeCollection = Union[T, Collection[T]]
MappingSequence = Union[Mapping[Hashable, T], Sequence[T]]
MaybeMappingSequence = Union[T, Mapping[Hashable, T], Sequence[T]]
SetLike = Union[None, Set[T]]


# Arrays
class SupportsArray(Protocol):
    def __array__(self) -> np.ndarray:
        ...


DTypeLike = Any
PandasDTypeLike = Any
TypeLike = MaybeIterable[Union[Type, str, Regex]]
Shape = Tuple[int, ...]
ShapeLike = Union[int, Shape]
Array = np.ndarray  # ready to be used for n-dim data
Array1d = np.ndarray
Array2d = np.ndarray
Array3d = np.ndarray
Record = np.void
RecordArray = np.ndarray
RecordArray2d = np.ndarray
RecArray = np.recarray
MaybeArray = Union[T, Array]
MaybeIndexArray = Union[int, Array1d, slice, Tuple[Array1d, Array1d]]
SeriesFrame = Union[Series, Frame]
MaybeSeries = Union[T, Series]
MaybeSeriesFrame = Union[T, Series, Frame]
PandasArray = Union[Index, Series, Frame]
AnyArray = Union[Array, PandasArray]
AnyArray1d = Union[Array1d, Index, Series]
AnyArray2d = Union[Array2d, Frame]
ArrayLike = Union[Scalar, Sequence[Scalar], Sequence[Sequence[Any]], SupportsArray]
IndexLike = Union[range, Sequence[Scalar], SupportsArray]
FlexArray = Array
MaybeFlexArray = Union[Scalar, FlexArray]

# Labels
Label = Hashable
Labels = Sequence[Label]
Level = Union[str, int]
LevelSequence = Sequence[Level]
MaybeLevelSequence = Union[Level, LevelSequence]

# Datetime
TimedeltaLike = Union[pd.Timedelta, timedelta, np.timedelta64]
FrequencyLike = Union[str, float, TimedeltaLike, BaseOffset]
PandasFrequencyLike = Union[str, TimedeltaLike, BaseOffset]
PandasGroupByLike = Union[PandasGroupBy, PandasResampler, PandasFrequencyLike]
TimezoneLike = Union[None, str, float, timedelta, tzinfo]
DatetimeLike = Union[str, float, pd.Timestamp, np.datetime64, datetime]
TimeLike = Union[str, time]
PandasFrequency = Union[pd.Timedelta, pd.DateOffset]
PandasDatetimeIndex = Union[pd.DatetimeIndex, pd.PeriodIndex]
AnyFrequency = Union[None, int, float, pd.Timedelta, pd.DateOffset]


class SupportsTZInfo(Protocol):
    tzinfo: tzinfo


# Indexing
PandasIndexingFunc = Callable[[SeriesFrame], MaybeSeriesFrame]

# Grouping
GroupByLike = Union[None, bool, MaybeLevelSequence, IndexLike]
GroupIdxs = Array1d
GroupLens = Array1d
GroupMap = Tuple[GroupIdxs, GroupLens]

# Wrapping
NameIndex = Union[None, Any, Index]

# Config
DictLike = Union[None, dict]
DictLikeSequence = MaybeSequence[DictLike]
Args = Tuple[Any, ...]
ArgsLike = Union[None, Args]
Kwargs = Dict[str, Any]
KwargsLike = Union[None, Kwargs]
KwargsLikeSequence = MaybeSequence[KwargsLike]
PathLike = Union[str, Path]
SettingsKeys = ClassVar[Union[None, Hashable, Dict[Hashable, Hashable]]]

# Data
Symbol = Hashable
Symbols = Sequence[Symbol]

# Plotting
TraceName = Union[str, None]
TraceNames = MaybeSequence[TraceName]

# Generic
MapFunc = Callable[[Scalar, VarArg()], Scalar]
MapMetaFunc = Callable[[int, int, Scalar, VarArg()], Scalar]
ApplyFunc = Callable[[Array1d, VarArg()], MaybeArray]
ApplyMetaFunc = Callable[[int, VarArg()], MaybeArray]
ReduceFunc = Callable[[Array1d, VarArg()], Scalar]
ReduceMetaFunc = Callable[[int, VarArg()], Scalar]
ReduceToArrayFunc = Callable[[Array1d, VarArg()], Array1d]
ReduceToArrayMetaFunc = Callable[[int, VarArg()], Array1d]
ReduceGroupedFunc = Callable[[Array2d, VarArg()], Scalar]
ReduceGroupedMetaFunc = Callable[[GroupIdxs, int, VarArg()], Scalar]
ReduceGroupedToArrayFunc = Callable[[Array2d, VarArg()], Array1d]
ReduceGroupedToArrayMetaFunc = Callable[[GroupIdxs, int, VarArg()], Array1d]
RangeReduceMetaFunc = Callable[[int, int, int, VarArg()], Scalar]
GroupByReduceMetaFunc = Callable[[GroupIdxs, int, int, VarArg()], Scalar]
GroupSqueezeMetaFunc = Callable[[int, GroupIdxs, int, VarArg()], Scalar]

# Signals
PlaceFunc = Callable[[NamedTuple, VarArg()], int]
RankFunc = Callable[[NamedTuple, VarArg()], int]

# Records
RecordsMapFunc = Callable[[np.void, VarArg()], Scalar]
RecordsMapMetaFunc = Callable[[int, VarArg()], Scalar]
MappedReduceMetaFunc = Callable[[GroupIdxs, int, VarArg()], Scalar]
MappedReduceToArrayMetaFunc = Callable[[GroupIdxs, int, VarArg()], Array1d]

# Indicators
Param = Any
Params = Sequence[Param]

# Mappings
MappingLike = Union[str, Mapping, NamedTuple, EnumMeta, IndexLike]

# Parsing
AnnArgs = Dict[str, Kwargs]
FlatAnnArgs = List[Kwargs]
AnnArgQuery = Union[int, str, Regex]

# Execution
FuncArgs = Tuple[Callable, Args, Kwargs]
FuncsArgs = Iterable[FuncArgs]
EngineLike = Union[str, type, ExecutionEngine, Callable]

# Chunking
SizeFunc = Callable[[AnnArgs], int]
SizeLike = Union[int, Sizer, SizeFunc]
ChunkMetaFunc = Callable[[AnnArgs], Iterable[ChunkMeta]]
ChunkMetaLike = Union[Iterable[ChunkMeta], ChunkMetaGenerator, ChunkMetaFunc]
TakeSpec = Union[None, ChunkTaker]
ArgTakeSpec = Mapping[AnnArgQuery, TakeSpec]
ArgTakeSpecFunc = Callable[[AnnArgs, ChunkMeta], Tuple[Args, Kwargs]]
ArgTakeSpecLike = Union[Sequence[TakeSpec], ArgTakeSpec, ArgTakeSpecFunc]
MappingTakeSpec = Mapping[Hashable, TakeSpec]
SequenceTakeSpec = Sequence[TakeSpec]
ContainerTakeSpec = Union[MappingTakeSpec, SequenceTakeSpec]
ChunkedOption = Union[None, bool, str, Kwargs]

# JIT
JittedOption = Union[None, bool, str, Kwargs]
JitterLike = Union[str, Jitter, Type[Jitter]]

# Decorators
ClassWrapper = Callable[[Type[T]], Type[T]]
FlexClassWrapper = Union[Callable[[Type[T]], Type[T]], Type[T]]
