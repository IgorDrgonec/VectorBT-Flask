# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Base class for splitting."""

import attr
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.array_ import is_range
from vectorbtpro.utils.config import resolve_dict, merge_dicts, Config, HybridConfig
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.template import CustomTemplate, Rep, RepFunc, deep_substitute
from vectorbtpro.utils.decorators import class_or_instancemethod
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.datetime_ import try_to_datetime_index
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.base.indexing import hslice, PandasIndexer, get_index_ranges
from vectorbtpro.base.indexes import combine_indexes
from vectorbtpro.base.reshaping import to_dict
from vectorbtpro.base.accessors import BaseIDXAccessor
from vectorbtpro.base.resampling import Resampler
from vectorbtpro.base.grouping import Grouper
from vectorbtpro.generic.analyzable import Analyzable

if tp.TYPE_CHECKING:
    from sklearn.model_selection import BaseCrossValidator as BaseCrossValidatorT
else:
    BaseCrossValidatorT = tp.Any


__pdoc__ = {}


SplitterT = tp.TypeVar("SplitterT", bound="Splitter")


@attr.s(frozen=True)
class RelRange:
    """Class that represents a relative range."""

    offset: tp.Union[int, float] = attr.ib(default=0)
    """Offset.
    
    Floating values between 0 and 1 are considered relative.
    
    Can be negative."""

    offset_anchor: str = attr.ib(default="prev_end")
    """Offset anchor.
    
    Supported are
    
    * 'start': Start of the range
    * 'end': End of the range
    * 'prev_start': Start of the previous range
    * 'prev_end': End of the previous range
    """

    offset_space: str = attr.ib(default="free")
    """Offset space.

    Supported are

    * 'all': All space
    * 'free': Remaining space after the offset anchor
    
    Applied only when `RelRange.offset` is a relative number."""

    length: tp.Union[int, float] = attr.ib(default=1.0)
    """Length.
    
    Floating values between 0 and 1 are considered relative.
    
    Can be negative."""

    length_space: str = attr.ib(default="free")
    """Length space.
    
    Supported are
    
    * 'all': All space
    * 'free': Remaining space after the offset
    
    Applied only when `RelRange.length` is a relative number."""

    out_of_bounds: str = attr.ib(default="warn")
    """Check if start and stop are within bounds.
    
    Supported are
    
    * 'ignore': ignore if out-of-bounds
    * 'warn': emit a warning if out-of-bounds
    * 'raise": raise an error if out-of-bounds
    """

    def __attrs_post_init__(self):
        object.__setattr__(self, "offset_anchor", self.offset_anchor.lower())
        if self.offset_anchor not in ("start", "end", "prev_start", "prev_end", "next_start", "next_end"):
            raise ValueError(f"Invalid offset anchor option '{self.offset_anchor}'")
        object.__setattr__(self, "offset_space", self.offset_space.lower())
        if self.offset_space not in ("all", "free"):
            raise ValueError(f"Invalid offset space option '{self.offset_space}'")
        object.__setattr__(self, "length_space", self.length_space.lower())
        if self.length_space not in ("all", "free"):
            raise ValueError(f"Invalid length space option '{self.length_space}'")
        object.__setattr__(self, "out_of_bounds", self.out_of_bounds.lower())
        if self.out_of_bounds not in ("ignore", "warn", "raise"):
            raise ValueError(f"Invalid out-of-bounds option '{self.out_of_bounds}'")

    def to_slice(
        self,
        total_len: int,
        prev_start: int = 0,
        prev_end: int = 0,
    ) -> slice:
        """Convert the relative range into a slice."""
        if self.offset_anchor == "start":
            offset_anchor = 0
        elif self.offset_anchor == "end":
            offset_anchor = total_len
        elif self.offset_anchor == "prev_start":
            offset_anchor = prev_start
        else:
            offset_anchor = prev_end
        if isinstance(self.offset, (float, np.floating)) and 0 <= abs(self.offset) <= 1:
            if self.offset_space == "all":
                offset = int(self.offset * total_len)
            else:
                if self.offset < 0:
                    offset = int((1 + self.offset) * offset_anchor)
                else:
                    offset = offset_anchor + int(self.offset * (total_len - offset_anchor))
        else:
            if isinstance(self.offset, (float, np.floating)) and not self.offset.is_integer():
                raise TypeError("Floating number for offset must be between 0 and 1")
            offset = int(offset_anchor + self.offset)
        if isinstance(self.length, (float, np.floating)) and 0 <= abs(self.length) <= 1:
            if self.length_space == "all":
                length = int(self.length * total_len)
            else:
                if self.length < 0:
                    if offset > prev_end:
                        length = int(self.length * (offset - prev_end))
                    else:
                        length = int(self.length * offset)
                else:
                    length = int(self.length * (total_len - offset))
        else:
            if isinstance(self.length, (float, np.floating)) and not self.length.is_integer():
                raise TypeError("Floating number for length must be between 0 and 1")
            length = int(self.length)
        start = offset
        stop = start + length
        if length < 0:
            start, stop = stop, start
        if start < 0:
            if self.out_of_bounds == "warn":
                warnings.warn(f"Range start ({start}) is out of bounds", stacklevel=2)
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range start ({start}) is out of bounds")
            start = 0
        if stop > total_len:
            if self.out_of_bounds == "warn":
                warnings.warn(f"Range stop ({stop}) is out of bounds", stacklevel=2)
            elif self.out_of_bounds == "raise":
                raise ValueError(f"Range stop ({stop}) is out of bounds")
            stop = total_len
        if stop - start <= 0:
            raise ValueError("Range length is negative or zero")
        return slice(start, stop)


@attr.s
class GapRange:
    """Class that represents a range acting as a gap."""

    range_: tp.RangeLike = attr.ib()
    """Range."""


class Splitter(Analyzable):
    """Base class for splitting."""

    @classmethod
    def from_splits(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        splits: tp.Splits,
        fix_ranges: bool = True,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        check_constant: bool = True,
        template_context: tp.KwargsLike = None,
        split_labels: tp.Union[None, str, tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from an iterable of splits.

        Argument `splits` supports both absolute and relative ranges.
        To transform relative ranges into the absolute format, enable `fix_ranges`.
        Arguments `split_range_kwargs` are then passed to `Splitter.split_range`.

        Labels for splits can be provided via `split_labels`. The argument can also be set to 'bounds'
        to create an index with start and end row. This requires each range to be fixed."""
        index = try_to_datetime_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                splits = np.asarray(splits)
        except Exception as e:
            splits = np.asarray(splits, dtype=object)
        if splits.size == 0:
            raise ValueError("No splits provided")
        if splits.ndim == 0:
            splits = splits[None]
        if splits.ndim == 1:
            ndim = 1
            splits = splits[:, None]
        else:
            ndim = 2
        if fix_ranges:
            new_splits = []
            for split in splits:
                new_split = cls.split_range(
                    slice(None),
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                new_splits.append(new_split)
            splits = new_splits
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    splits = np.asarray(splits)
            except Exception as e:
                splits = np.asarray(splits, dtype=object)
        if split_labels is None:
            split_labels = pd.RangeIndex(stop=splits.shape[0], name="split")
        if isinstance(split_labels, str):
            if split_labels.lower() == "bounds":
                split_bounds = []
                for split in splits:
                    set_bounds = []
                    for range_ in split:
                        range_bounds = cls.get_range_bounds(
                            range_,
                            template_context=template_context,
                            index=index,
                            check_constant=check_constant,
                            **range_bounds_kwargs,
                        )
                        if check_constant and len(set_bounds) > 0:
                            if set_bounds[-1][1] != range_bounds[0]:
                                raise ValueError("Split does not build a constant range")
                        set_bounds.append(range_bounds)
                    split_bounds.append((set_bounds[0][0], set_bounds[-1][1]))
                split_labels = pd.MultiIndex.from_tuples(split_bounds, names=["start_row", "end_row"])
            else:
                raise ValueError(f"Invalid split labels option '{split_labels}'")
        if not isinstance(split_labels, pd.Index):
            split_labels = pd.Index(split_labels, name="split")
        if set_labels is None:
            set_labels = pd.Index(["set_%d" % i for i in range(splits.shape[1])], name="set")
        if not isinstance(set_labels, pd.Index):
            set_labels = pd.Index(set_labels, name="set")
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper = ArrayWrapper(index=split_labels, columns=set_labels, ndim=ndim, **wrapper_kwargs)
        return cls(wrapper, index, splits, **kwargs)

    @classmethod
    def from_split_func(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        split_func: tp.Callable,
        split_args: tp.ArgsLike = None,
        split_kwargs: tp.KwargsLike = None,
        fix_ranges: bool = True,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a custom split function.

        In a while-loop, substitutes templates in `split_args` and `split_kwargs` and passes
        them to `split_func`, which should return either a split (see `new_split` in `Splitter.split_range`,
        also supports a single range if it's not an iterable) or None to abrupt the while-loop.
        If `fix_ranges` is True, the returned split is then converted into a fixed split using
        `Splitter.split_range` and the bounds of its sets are measured using `Splitter.get_range_bounds`.

        Each template substitution has the following information:

        * `split_idx`: Current split index, starting at 0
        * `splits`: Nested list of splits appended up to this point
        * `bounds`: Nested list of bounds appended up to this point
        * Arguments and keyword arguments passed to `Splitter.from_split_func`

        Usage:
            * Rolling window of 30 elements, 20 for train and 10 for test:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> def split_func(splits, bounds, index):
            ...     if len(splits) == 0:
            ...         new_split = (slice(0, 20), slice(20, 30))
            ...     else:
            ...         # Previous split, first set, right bound
            ...         prev_end = bounds[-1][0][1]
            ...         new_split = (
            ...             slice(prev_end, prev_end + 20),
            ...             slice(prev_end + 20, prev_end + 30)
            ...         )
            ...     if new_split[-1].stop > len(index):
            ...         return None
            ...     return new_split

            >>> splitter = vbt.Splitter.from_split_func(
            ...     index,
            ...     split_func,
            ...     split_args=(
            ...         vbt.Rep("splits"),
            ...         vbt.Rep("bounds"),
            ...         vbt.Rep("index"),
            ...     ),
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_split_func.svg)
        """
        index = try_to_datetime_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}
        if split_args is None:
            split_args = ()
        if split_kwargs is None:
            split_kwargs = {}

        splits = []
        bounds = []
        split_idx = 0
        n_sets = None
        while True:
            _template_context = merge_dicts(
                dict(
                    split_idx=split_idx,
                    splits=splits,
                    bounds=bounds,
                    index=index,
                    split_args=split_args,
                    split_kwargs=split_kwargs,
                    split_range_kwargs=split_range_kwargs,
                    range_bounds_kwargs=range_bounds_kwargs,
                    **kwargs,
                ),
                template_context,
            )
            _split_func = deep_substitute(split_func, _template_context, sub_id="split_func")
            _split_args = deep_substitute(split_args, _template_context, sub_id="split_args")
            _split_kwargs = deep_substitute(split_kwargs, _template_context, sub_id="split_kwargs")
            new_split = _split_func(*_split_args, **_split_kwargs)
            if new_split is None:
                break
            if not checks.is_iterable(new_split):
                new_split = (new_split,)
            if fix_ranges or split is not None:
                new_split = cls.split_range(
                    slice(None),
                    new_split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if split is not None:
                if len(new_split) > 1:
                    raise ValueError("Split function must return only one range if split is already provided")
                new_split = cls.split_range(
                    new_split[0],
                    split,
                    template_context=_template_context,
                    index=index,
                    **split_range_kwargs,
                )
            if n_sets is None:
                n_sets = len(new_split)
            elif n_sets != len(new_split):
                raise ValueError("All splits must have the same number of sets")
            splits.append(new_split)
            if fix_ranges:
                split_bounds = tuple(
                    map(
                        lambda x: cls.get_range_bounds(
                            x,
                            template_context=_template_context,
                            index=index,
                            **range_bounds_kwargs,
                        ),
                        new_split,
                    )
                )
                bounds.append(split_bounds)
            split_idx += 1

        return cls.from_splits(
            index,
            splits,
            fix_ranges=fix_ranges,
            split_range_kwargs=split_range_kwargs,
            range_bounds_kwargs=range_bounds_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        length: tp.Union[int, float],
        offset: tp.Union[int, float] = 0,
        offset_anchor: str = "prev_end",
        offset_anchor_set: tp.Optional[int] = 0,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a rolling range of a fixed length.

        Uses `Splitter.from_splits` to prepare the splits array and labels, and to build the instance.

        Args:
            index (index_like): Index.
            length (int or float): Length of the range to roll over the index.

                Provide it as a float between 0 and 1 to make it relative to the length of the index.
            offset (int or float): Offset relative to the offset anchor.

                Provide it as a float between 0 and 1 to make it relative to the length of the range.
            offset_anchor (str): Offset anchor.

                Can be 'prev_start' and 'prev_end' for the left and right bound of the
                previous range set respectively. By default, it's the right bound.
            offset_anchor_set (int): Offset anchor set.

                Selects the set from the previous range to be used as an offset anchor.
                If None, the whole previous split is considered as a single range.
                By default, it's the first set.
            split (any): Ranges to split the range into.

                If None, will produce the entire range as a single range.
                Otherwise, will use `Splitter.split_range` to split the range into multiple ranges.
            split_range_kwargs (dict): Keyword arguments passed to `Splitter.split_range`.
            range_bounds_kwargs (dict): Keyword arguments passed to `Splitter.get_range_bounds`.
            template_context (dict): Mapping used to substitute templates in ranges.
            **kwargs: Keyword arguments passed to the constructor of `Splitter`.

        Usage:
            * Divide a range into a set of non-overlapping ranges:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_rolling(index, 30)
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_rolling_1.svg)

            * Divide a range into ranges, each split into 1/2:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_rolling_2.svg)

            * Make the ranges above non-overlapping by using the right bound of the last
            set as an offset anchor:

            ```pycon
            >>> splitter = vbt.Splitter.from_rolling(
            ...     index,
            ...     60,
            ...     offset_anchor_set=-1,
            ...     split=1/2,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_rolling_3.svg)
        """
        index = try_to_datetime_index(index)
        if isinstance(length, (float, np.floating)) and 0 <= abs(length) <= 1:
            length = int(len(index) * length)
        if isinstance(length, (float, np.floating)) and not length.is_integer():
            raise TypeError("Floating number for length must be between 0 and 1")
        length = int(length)
        if length < 1 or length > len(index):
            raise TypeError(f"Length must be within [{1}, {len(index)}]")
        if isinstance(offset, (float, np.floating)) and 0 <= abs(offset) <= 1:
            offset = int(length * offset)
        if isinstance(offset, (float, np.floating)) and not offset.is_integer():
            raise TypeError("Floating number for offset must be between 0 and 1")
        offset = int(offset)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = slice(0, length)
            else:
                if offset_anchor.lower() == "prev_start":
                    if offset_anchor_set is None:
                        prev_bounds = bounds[-1][0]
                    else:
                        prev_bounds = bounds[-1][offset_anchor_set]
                    start = prev_bounds[0] + offset
                elif offset_anchor.lower() == "prev_end":
                    if offset_anchor_set is None:
                        prev_bounds = bounds[-1][-1]
                    else:
                        prev_bounds = bounds[-1][offset_anchor_set]
                    start = prev_bounds[1] + offset
                else:
                    raise ValueError(f"Invalid offset anchor option '{offset_anchor}'")
                new_split = slice(start, start + length)
                if new_split.start <= bounds[-1][0][0]:
                    raise ValueError("Infinite loop detected. Provide a non-zero offset.")
            if new_split.stop > len(index):
                break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            range_bounds_kwargs=range_bounds_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_rolling(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        length: tp.Union[None, int, float] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of rolling ranges of a fixed length.

        If `length` is None, splits the index evenly into `n` non-overlapping ranges
        using `Splitter.from_rolling`. Otherwise, picks `n` evenly-spaced, potentially overlapping
        ranges of a fixed length. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll 10 ranges with 100 elements, and split it into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_rolling(
            ...     index,
            ...     10,
            ...     length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_n_rolling.svg)
        """
        index = try_to_datetime_index(index)
        if length is not None:
            if isinstance(length, (float, np.floating)) and 0 <= abs(length) <= 1:
                length = int(len(index) * length)
            if isinstance(length, (float, np.floating)) and not length.is_integer():
                raise TypeError("Floating number for length must be between 0 and 1")
            length = int(length)
            if length < 1 or length > len(index):
                raise TypeError(f"Length must be within [{1}, {len(index)}]")
        if split_range_kwargs is None:
            split_range_kwargs = {}

        if length is None:
            return cls.from_rolling(
                index,
                length=len(index) // n,
                offset=0,
                offset_anchor="prev_end",
                offset_anchor_set=None,
                split=split,
                split_range_kwargs=split_range_kwargs,
                template_context=template_context,
                **kwargs,
            )

        start_rows = np.arange(len(index) - length + 1)
        end_rows = np.arange(length, len(index) + 1)
        if n > len(start_rows):
            n = len(start_rows)
        rows = np.round(np.linspace(0, len(start_rows) - 1, n)).astype(int)
        start_rows = start_rows[rows]
        end_rows = end_rows[rows]
        splits = []
        for i in range(len(start_rows)):
            new_split = slice(start_rows[i], end_rows[i])
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        min_length: tp.Union[int, float],
        gap: tp.Union[int, float],
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        range_bounds_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from an expanding range.

        Argument `min_length` is the minimum length of the expanding range. Provide it as
        a float between 0 and 1 to make it relative to the length of the index. Argument `gap` is
        a gap between the right bounds of each two neighboring ranges. It can also be a float
        relative to the index length. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll an expanding range with a length of 10 and a gap of 10, and split it into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_expanding(
            ...     index,
            ...     10,
            ...     10,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_expanding.svg)
        """
        index = try_to_datetime_index(index)
        if isinstance(min_length, (float, np.floating)) and 0 <= abs(min_length) <= 1:
            min_length = int(len(index) * min_length)
        if isinstance(min_length, (float, np.floating)) and not min_length.is_integer():
            raise TypeError("Floating number for minimum length must be between 0 and 1")
        min_length = int(min_length)
        if min_length < 1 or min_length > len(index):
            raise TypeError(f"Minimum length must be within [{1}, {len(index)}]")
        if isinstance(gap, (float, np.floating)) and 0 <= abs(gap) <= 1:
            gap = int(len(index) * gap)
        if isinstance(gap, (float, np.floating)) and not gap.is_integer():
            raise TypeError("Floating number for gap must be between 0 and 1")
        gap = int(gap)
        if gap < 1 or gap > len(index):
            raise TypeError(f"Gap must be within [{1}, {len(index)}]")
        if split_range_kwargs is None:
            split_range_kwargs = {}
        if range_bounds_kwargs is None:
            range_bounds_kwargs = {}

        splits = []
        bounds = []
        while True:
            if len(splits) == 0:
                new_split = slice(0, min_length)
            else:
                prev_bound = bounds[-1][-1][1]
                new_split = slice(0, prev_bound + gap)
            if new_split.stop > len(index):
                break
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
                bounds.append(
                    tuple(
                        map(
                            lambda x: cls.get_range_bounds(
                                x,
                                template_context=template_context,
                                index=index,
                                **range_bounds_kwargs,
                            ),
                            new_split,
                        )
                    )
                )
            else:
                bounds.append(((new_split.start, new_split.stop),))
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            range_bounds_kwargs=range_bounds_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_expanding(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[None, int, float] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of expanding ranges.

        Picks `n` evenly-spaced, expanding ranges. Argument `min_length` defines the minimum
        length for each range. For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Roll 10 expanding ranges with a minimum length of 100, while reserving 50 elements for test:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_expanding(
            ...     index,
            ...     10,
            ...     min_length=100,
            ...     split=-50,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_n_expanding.svg)
        """
        index = try_to_datetime_index(index)
        if min_length is not None:
            if isinstance(min_length, (float, np.floating)) and 0 <= abs(min_length) <= 1:
                min_length = int(len(index) * min_length)
            if isinstance(min_length, (float, np.floating)) and not min_length.is_integer():
                raise TypeError("Floating number for minimum length must be between 0 and 1")
            min_length = int(min_length)
            if min_length < 1 or min_length > len(index):
                raise TypeError(f"Minimum length must be within [{1}, {len(index)}]")
        else:
            min_length = len(index) // n
        if split_range_kwargs is None:
            split_range_kwargs = {}

        start_rows = np.full(len(index) - min_length + 1, 0)
        end_rows = np.arange(min_length, len(index) + 1)
        if n > len(start_rows):
            n = len(start_rows)
        rows = np.round(np.linspace(0, len(start_rows) - 1, n)).astype(int)
        start_rows = start_rows[rows]
        end_rows = end_rows[rows]
        splits = []
        for i in range(len(start_rows)):
            new_split = slice(start_rows[i], end_rows[i])
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_n_random(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        n: int,
        min_length: tp.Union[int, float],
        max_length: tp.Union[None, int, float] = None,
        min_start: tp.Union[None, int, float] = None,
        max_end: tp.Union[None, int, float] = None,
        length_choice_func: tp.Optional[tp.Callable] = None,
        start_choice_func: tp.Optional[tp.Callable] = None,
        length_p_func: tp.Optional[tp.Callable] = None,
        start_p_func: tp.Optional[tp.Callable] = None,
        seed: tp.Optional[int] = None,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a number of random ranges.

        Randomly picks the length of a range between `min_length` and `max_length` (including) using
        `length_choice_func`, which receives an array of possible values and selects one. It defaults to
        `numpy.random.Generator.choice`. Optional function `length_p_func` takes the same as
        `length_choice_func` and must return either None or probabilities.

        Randomly picks the start position of a range starting at `min_start` and ending at `max_end`
        (excluding) minus the chosen length using `start_choice_func`, which receives an array of possible
        values and selects one. It defaults to `numpy.random.Generator.choice`. Optional function
        `start_p_func` takes the same as `start_choice_func` and must return either None or probabilities.

        !!! note
            Each function must take two arguments: the iteration index and the array with possible values.

        For other arguments, see `Splitter.from_rolling`.

        Usage:
            * Generate 20 random ranges with a length from [40, 100], and split each into 3/4:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_n_random(
            ...     index,
            ...     20,
            ...     min_length=40,
            ...     max_length=100,
            ...     split=3/4,
            ...     set_labels=["train", "test"]
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_n_random.svg)
        """
        index = try_to_datetime_index(index)
        if min_start is None:
            min_start = 0
        if min_start is not None:
            if isinstance(min_start, (float, np.floating)) and 0 <= abs(min_start) <= 1:
                min_start = int(len(index) * min_start)
            if isinstance(min_start, (float, np.floating)) and not min_start.is_integer():
                raise TypeError("Floating number for minimum start position must be between 0 and 1")
            min_start = int(min_start)
            if min_start < 0 or min_start > len(index) - 1:
                raise TypeError(f"Minimum start position must be within [{0}, {len(index) - 1}]")
        if max_end is None:
            max_end = len(index)
        if max_end is not None:
            if isinstance(max_end, (float, np.floating)) and 0 <= abs(max_end) <= 1:
                max_end = int(len(index) * max_end)
            if isinstance(max_end, (float, np.floating)) and not max_end.is_integer():
                raise TypeError("Floating number for maximum end position must be between 0 and 1")
            max_end = int(max_end)
            if max_end < 1 or max_end > len(index):
                raise TypeError(f"Maximum end position must be within [{1}, {len(index)}]")
        space_len = max_end - min_start
        if isinstance(min_length, (float, np.floating)) and 0 <= abs(min_length) <= 1:
            min_length = int(space_len * min_length)
        if isinstance(min_length, (float, np.floating)) and not min_length.is_integer():
            raise TypeError("Floating number for minimum length must be between 0 and 1")
        min_length = int(min_length)
        if min_length < 1 or min_length > space_len:
            raise TypeError(f"Minimum length must be within [{1}, {space_len}]")
        if max_length is not None:
            if isinstance(max_length, (float, np.floating)) and 0 <= abs(max_length) <= 1:
                max_length = int(space_len * max_length)
            if isinstance(max_length, (float, np.floating)) and not max_length.is_integer():
                raise TypeError("Floating number for maximum length must be between 0 and 1")
            max_length = int(max_length)
            if max_length < 1 or max_length > space_len:
                raise TypeError(f"Maximum length must be within [{1}, {space_len}]")
        else:
            max_length = min_length
        rng = np.random.default_rng(seed=seed)
        if length_p_func is None:
            length_p_func = lambda i, x: None
        if start_p_func is None:
            start_p_func = lambda i, x: None
        if length_choice_func is None:
            length_choice_func = lambda i, x: rng.choice(x, p=length_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if start_choice_func is None:
            start_choice_func = lambda i, x: rng.choice(x, p=start_p_func(i, x))
        else:
            if seed is not None:
                np.random.seed(seed)
        if split_range_kwargs is None:
            split_range_kwargs = {}

        length_space = np.arange(min_length, max_length + 1)
        index_space = np.arange(len(index))
        splits = []
        for i in range(n):
            length = length_choice_func(i, length_space)
            start = start_choice_func(i, index_space[min_start : max_end - length + 1])
            new_split = slice(start, start + length)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_ranges(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        *args,
        split: tp.Optional[tp.SplitLike] = None,
        split_range_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from ranges.

        Uses `vectorbtpro.base.indexing.get_index_ranges` to generate start and end indices.
        Other keyword arguments will be passed to `Splitter.from_splits`. For details on
        `split` and `split_range_kwargs`, see `Splitter.from_rolling`.

        Usage:
            * Translate each quarter into a range:

            ```pycon
            >>> import vectorbtpro as vbt
            >>> import pandas as pd

            >>> index = pd.date_range("2020", "2021", freq="D")

            >>> splitter = vbt.Splitter.from_ranges(index, every="QS")
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_ranges_1.svg)

            * In addition to the above, reserve the last month for testing purposes:

            ```pycon
            >>> splitter = vbt.Splitter.from_ranges(
            ...     index,
            ...     every="QS",
            ...     split=(1.0, lambda index: index.month == index.month[-1]),
            ...     split_range_kwargs=dict(backwards=True)
            ... )
            >>> splitter.plot()
            ```

            ![](/assets/images/api/from_ranges_2.svg)
        """
        index = try_to_datetime_index(index)
        if split_range_kwargs is None:
            split_range_kwargs = {}
        func_arg_names = get_func_arg_names(get_index_ranges)
        ranges_kwargs = dict()
        for k in list(kwargs.keys()):
            if k in func_arg_names:
                ranges_kwargs[k] = kwargs.pop(k)

        start_idxs, stop_idxs = get_index_ranges(index, *args, skip_minus_one=True, **ranges_kwargs)
        splits = []
        for start, stop in zip(start_idxs, stop_idxs):
            new_split = slice(start, stop)
            if split is not None:
                new_split = cls.split_range(
                    new_split,
                    split,
                    template_context=template_context,
                    index=index,
                    **split_range_kwargs,
                )
            splits.append(new_split)

        return cls.from_splits(
            index,
            splits,
            split_range_kwargs=split_range_kwargs,
            template_context=template_context,
            **kwargs,
        )

    @classmethod
    def from_sklearn(
        cls: tp.Type[SplitterT],
        index: tp.IndexLike,
        splitter: tp.Union[BaseCrossValidatorT],
        groups: tp.Optional[tp.ArrayLike] = None,
        split_labels: tp.Optional[tp.IndexLike] = None,
        set_labels: tp.Optional[tp.IndexLike] = None,
        **kwargs,
    ) -> SplitterT:
        """Create a new `Splitter` instance from a scikit-learn's splitter.

        The splitter must be an instance of `sklearn.model_selection.BaseCrossValidator`.

        Uses `Splitter.from_splits` to prepare the splits array and labels, and to build the instance."""
        from sklearn.model_selection import BaseCrossValidator

        index = try_to_datetime_index(index)
        checks.assert_instance_of(splitter, BaseCrossValidator)
        if set_labels is None:
            set_labels = ["train", "test"]

        indices_generator = splitter.split(np.arange(len(index))[:, None], groups=groups)
        return cls.from_splits(
            index,
            list(indices_generator),
            split_labels=split_labels,
            set_labels=set_labels,
            **kwargs,
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Splitter` after stacking along rows."""
        if "splits" not in kwargs:
            kwargs["splits"] = kwargs["wrapper"].row_stack_arrs(
                *[obj.splits for obj in objs],
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        reindex_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Splitter` after stacking along columns."""
        if "splits" not in kwargs:
            kwargs["splits"] = kwargs["wrapper"].column_stack_arrs(
                *[obj.splits for obj in objs],
                reindex_kwargs=reindex_kwargs,
                group_by=False,
                wrap=False,
            )
        return kwargs

    @classmethod
    def row_stack(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(
                *[obj.wrapper for obj in objs],
                stack_columns=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[SplitterT],
        *objs: tp.MaybeTuple[SplitterT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> SplitterT:
        """Stack multiple `Splitter` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Splitter):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                union_index=False,
                **wrapper_kwargs,
            )

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "index",
        "splits",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        index: tp.Index,
        splits: tp.SplitsArray,
        **kwargs,
    ) -> None:
        if wrapper.grouper.is_grouped():
            raise ValueError("Splitter cannot be grouped")
        index = try_to_datetime_index(index)
        if splits.shape[0] != wrapper.shape_2d[0]:
            raise ValueError("Number of splits must match wrapper index")
        if splits.shape[1] != wrapper.shape_2d[1]:
            raise ValueError("Number of sets must match wrapper columns")

        Analyzable.__init__(
            self,
            wrapper,
            index=index,
            splits=splits,
            **kwargs,
        )

        self._index = index
        self._splits = splits

    def indexing_func_meta(self, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> dict:
        """Perform indexing on `Splitter` and return metadata."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        if wrapper_meta["rows_changed"] or wrapper_meta["columns_changed"]:
            new_splits = ArrayWrapper.select_from_flex_array(
                self.splits_arr,
                row_idxs=wrapper_meta["row_idxs"],
                col_idxs=wrapper_meta["col_idxs"],
                rows_changed=wrapper_meta["rows_changed"],
                columns_changed=wrapper_meta["columns_changed"],
            )
        else:
            new_splits = self.splits_arr
        return dict(
            wrapper_meta=wrapper_meta,
            new_splits=new_splits,
        )

    def indexing_func(self: SplitterT, *args, splitter_meta: tp.DictLike = None, **kwargs) -> SplitterT:
        """Perform indexing on `Splitter`."""
        if splitter_meta is None:
            splitter_meta = self.indexing_func_meta(*args, **kwargs)
        return self.replace(
            wrapper=splitter_meta["wrapper_meta"]["new_wrapper"],
            splits=splitter_meta["new_splits"],
        )

    @property
    def index(self) -> tp.Index:
        """Index."""
        return self._index

    @property
    def splits_arr(self) -> tp.SplitsArray:
        """Two-dimensional, object-dtype DataFrame with splits.

        First axis represents splits. Second axis represents sets. Elements represent ranges.
        Range must be either a slice, a sequence of indices, a mask, or a callable that returns such."""
        return self._splits

    @property
    def splits(self) -> tp.Frame:
        """`Splitter.splits_arr` as a DataFrame."""
        return self.wrapper.wrap(self._splits, group_by=False)

    @property
    def split_labels(self) -> tp.Index:
        """Split labels."""
        return self.wrapper.index

    @property
    def set_labels(self) -> tp.Index:
        """Set labels."""
        return self.wrapper.columns

    @property
    def n_splits(self) -> int:
        """Number of splits."""
        return self.splits_arr.shape[0]

    @property
    def n_sets(self) -> int:
        """Number of sets."""
        return self.splits_arr.shape[1]

    def get_split_grouper(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Get split grouper."""
        if split_group_by is None:
            return None
        if isinstance(split_group_by, Grouper):
            return split_group_by
        return BaseIDXAccessor(self.split_labels).get_grouper(split_group_by, def_lvl_name="split_group")

    def get_set_grouper(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Optional[Grouper]:
        """Get set grouper."""
        if set_group_by is None:
            return None
        if isinstance(set_group_by, Grouper):
            return set_group_by
        return BaseIDXAccessor(self.set_labels).get_grouper(set_group_by, def_lvl_name="set_group")

    def get_n_splits(self, split_group_by: tp.AnyGroupByLike = None) -> int:
        """Get number of splits while considering the grouper."""
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_group_count()
        return self.n_splits

    def get_n_sets(self, set_group_by: tp.AnyGroupByLike = None) -> int:
        """Get number of sets while considering the grouper."""
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_group_count()
        return self.n_sets

    def get_split_labels(self, split_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Get split labels while considering the grouper."""
        if split_group_by is not None:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            return split_group_by.get_index()
        return self.split_labels

    def get_set_labels(self, set_group_by: tp.AnyGroupByLike = None) -> tp.Index:
        """Get set labels while considering the grouper."""
        if set_group_by is not None:
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            return set_group_by.get_index()
        return self.set_labels

    # ############# Ranges ############# #

    def get_range(
        self,
        split: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        set_: tp.Optional[tp.MaybeIterable[tp.Hashable]] = None,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        split_as_indices: bool = False,
        set_as_indices: bool = False,
        **merge_split_kwargs,
    ) -> tp.RangeLike:
        """Get a range.

        Arguments `split` and `set_` can be either integers and labels. Also, multiple
        values are accepted; in such a case, the corresponding ranges are merged.
        If split/set labels are of the integer data type, treats the provided values as labels
        rather than indices, unless `split_as_indices`/`set_as_indices` is enabled.

        If `split_group_by` and/or `set_group_by` are provided, their groupers get
        created using `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper` and
        arguments `split` and `set_` become relative to the groups.

        If `split`/`set_` is not provided, selects and merges all ranges."""
        if split is not None:
            if not checks.is_iterable(split) or isinstance(split, str):
                split = (split,)
            if split_group_by is not None:
                split_group_by = self.get_split_grouper(split_group_by=split_group_by)
                label_mask = None
                int_index = split_group_by.get_index().is_integer()
                groups = split_group_by.get_groups()
                stretched_index = split_group_by.get_stretched_index()
                for s in split:
                    if isinstance(s, (int, np.integer)) and (split_as_indices or not int_index):
                        mask = groups == s
                    else:
                        mask = stretched_index == s
                    if label_mask is None:
                        label_mask = mask
                    else:
                        label_mask |= mask
                split = np.flatnonzero(label_mask)
                split_as_indices = True
        else:
            split = np.arange(self.n_splits)
            split_as_indices = True
        if set_ is not None:
            if not checks.is_iterable(set_) or isinstance(set_, str):
                set_ = (set_,)
            if set_group_by is not None:
                set_group_by = self.get_set_grouper(set_group_by=set_group_by)
                label_mask = None
                int_index = set_group_by.get_index().is_integer()
                groups = set_group_by.get_groups()
                stretched_index = set_group_by.get_stretched_index()
                for s in set_:
                    if isinstance(s, (int, np.integer)) and (set_as_indices or not int_index):
                        mask = groups == s
                    else:
                        mask = stretched_index == s
                    if label_mask is None:
                        label_mask = mask
                    else:
                        label_mask |= mask
                set_ = np.flatnonzero(label_mask)
                set_as_indices = True
        else:
            set_ = np.arange(self.n_sets)
            set_as_indices = True
        ranges = []
        for s1 in split:
            if isinstance(s1, (int, np.integer)) and (split_as_indices or not self.split_labels.is_integer()):
                i = s1
            else:
                i = self.split_labels.get_indexer([s1])[0]
                if i == -1:
                    raise ValueError(f"Split '{s1}' not found")
            for s2 in set_:
                if isinstance(s2, (int, np.integer)) and (set_as_indices or not self.set_labels.is_integer()):
                    j = s2
                else:
                    j = self.set_labels.get_indexer([s2])[0]
                    if j == -1:
                        raise ValueError(f"Set '{s2}' not found")
                ranges.append(self.splits_arr[i, j])
        if len(ranges) == 1:
            return ranges[0]
        return self.merge_split(ranges, **merge_split_kwargs)

    @classmethod
    def is_range_relative(cls, range_: tp.RangeLike) -> bool:
        """Return whether a range is relative."""
        return isinstance(range_, (int, float, np.number, RelRange))

    @class_or_instancemethod
    def to_ready_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        try_to_slice: bool = True,
        allow_relative: bool = False,
        allow_zero_len: bool = False,
        return_meta: bool = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Union[tp.RelRangeLike, tp.ReadyRangeLike, dict]:
        """Convert a range to be used in array indexing.

        Such a range is either a slice or a one-dimensional NumPy array."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        index = try_to_datetime_index(index)

        meta = dict()
        meta["was_gap"] = False
        meta["was_template"] = False
        meta["was_callable"] = False
        meta["was_relative"] = False
        meta["was_hslice"] = False
        meta["was_slice"] = False
        meta["was_neg_slice"] = False
        meta["was_mask"] = False
        meta["was_indices"] = False
        meta["is_constant"] = False
        meta["start"] = None
        meta["stop"] = None
        meta["length"] = None
        if isinstance(range_, GapRange):
            meta["was_gap"] = True
            range_ = range_.range_
        if isinstance(range_, CustomTemplate):
            meta["was_template"] = True
            if template_context is None:
                template_context = {}
            if "index" not in template_context:
                template_context["index"] = index
            range_ = range_.substitute(context=template_context, sub_id="range")
        if callable(range_):
            meta["was_callable"] = True
            range_ = range_(index)
        if cls_or_self.is_range_relative(range_):
            meta["was_relative"] = True
            if allow_relative:
                if return_meta:
                    meta["range_"] = range_
                    return meta
                return range_
            raise TypeError("Relative ranges must be converted to fixed before indexing")
        if isinstance(range_, hslice):
            meta["was_hslice"] = True
            range_ = range_.to_slice()
        if isinstance(range_, slice):
            meta["was_slice"] = True
            meta["is_constant"] = True
            start = range_.start if range_.start is not None else 0
            stop = range_.stop if range_.stop is not None else len(index)
            if range_.step is not None and range_.step > 1:
                raise ValueError("Step must be either None or 1")
            if start < 0:
                if stop > 0:
                    raise ValueError("Slices must be either strictly negative or positive")
                meta["was_neg_slice"] = True
                start = len(index) + range_.start
                stop = len(index) + range_.stop
            range_ = slice(max(start, 0), min(stop, len(index)))
            meta["start"] = start
            meta["stop"] = stop
            meta["length"] = stop - start
            if not allow_zero_len and meta["length"] == 0:
                raise ValueError("Range has zero length")
        else:
            range_ = np.asarray(range_, dtype=np.asarray([range_[0]]).dtype)
            if range_.dtype == np.bool_:
                if len(range_) != len(index):
                    raise ValueError("Mask must have the same length as index")
                meta["was_mask"] = True
                indices = np.flatnonzero(range_)
                if len(indices) == 0:
                    if not allow_zero_len:
                        raise ValueError("Range has zero length")
                    meta["is_constant"] = True
                    meta["start"] = 0
                    meta["stop"] = 0
                    meta["length"] = 0
                else:
                    meta["is_constant"] = is_range(indices)
                    meta["start"] = indices[0]
                    meta["stop"] = indices[-1] + 1
                    meta["length"] = len(indices)
            else:
                meta["was_indices"] = True
                if len(range_) == 0:
                    if not allow_zero_len:
                        raise ValueError("Range has zero length")
                    meta["is_constant"] = True
                    meta["start"] = 0
                    meta["stop"] = 0
                    meta["length"] = 0
                else:
                    range_ = np.sort(range_)
                    meta["is_constant"] = is_range(range_)
                    meta["start"] = range_[0]
                    meta["stop"] = range_[-1] + 1
                    meta["length"] = len(range_)
            if try_to_slice and meta["is_constant"]:
                range_ = slice(meta["start"], meta["stop"])
        if meta["start"] != meta["stop"]:
            if meta["start"] > meta["stop"]:
                raise ValueError(f"Range start ({meta['start']}) is higher than range stop ({meta['stop']})")
            if meta["start"] < 0 or meta["start"] >= len(index):
                raise ValueError(f"Range start ({meta['start']}) is out of bounds")
            if meta["stop"] < 0 or meta["stop"] > len(index):
                raise ValueError(f"Range stop ({meta['stop']}) is out of bounds")
        if return_meta:
            meta["range_"] = range_
            return meta
        return range_

    @class_or_instancemethod
    def split_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        new_split: tp.SplitLike,
        backwards: bool = False,
        try_to_slice: bool = True,
        allow_zero_len: bool = False,
        to_masks: bool = False,
        to_templates: bool = False,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        **range_split_kwargs,
    ) -> tp.FixSplit:
        """Split a fixed range into a split of multiple fixed ranges.

        Range must be either a template, a callable, a tuple (start and stop), a slice, a sequence
        of indices, or a mask. This range will then be re-mapped into the index.

        Each sub-range in `new_split` can be either a fixed or relative range, that is, an instance
        of `RelRange` or a number that will be used as a length to create an `RelRange`
        instance with `**kwargs`. Each sub-range will then be re-mapped into the main range.
        Argument `new_split` can also be provided as an integer or a float indicating the length;
        in such a case the second part (or the first one depending on `backwards`) will stretch.

        New ranges are returned relative to the index and in the same order as passed.

        Enable `to_masks` to convert the new ranges into masks. Enable `to_templates` to wrap
        the resulting ranges with a template of the type `vectorbtpro.utils.template.Rep`.
        Both options can be combined."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        index = try_to_datetime_index(index)

        # Prepare source range
        range_meta = cls_or_self.to_ready_range(
            range_,
            try_to_slice=try_to_slice,
            allow_zero_len=allow_zero_len,
            return_meta=True,
            template_context=template_context,
            index=index,
        )
        range_ = range_meta["range_"]
        range_was_hslice = range_meta["was_hslice"]
        range_was_mask = range_meta["was_mask"]
        range_length = range_meta["length"]
        if isinstance(range_, np.ndarray) and range_.dtype == np.bool_:
            range_ = np.flatnonzero(range_)

        # Prepare target ranges
        if isinstance(new_split, (int, float, np.number)):
            if new_split < 0:
                backwards = not backwards
                new_split = abs(new_split)
            if not backwards:
                new_split = (new_split, 1.0)
            else:
                new_split = (1.0, new_split)
        elif not checks.is_iterable(new_split):
            new_split = (new_split,)

        # Perform split
        new_ranges = []
        if backwards:
            new_split = new_split[::-1]
            prev_start = range_length
            prev_end = range_length
        else:
            prev_start = 0
            prev_end = 0
        for new_range in new_split:
            # Resolve new range
            new_range_meta = cls_or_self.to_ready_range(
                new_range,
                try_to_slice=try_to_slice,
                allow_relative=True,
                allow_zero_len=allow_zero_len,
                return_meta=True,
                template_context=template_context,
                index=index[range_],
            )
            new_range = new_range_meta["range_"]
            new_range_was_gap = new_range_meta["was_gap"]
            if isinstance(new_range, (int, float, np.number)):
                new_range = RelRange(length=new_range, **range_split_kwargs)
            if isinstance(new_range, RelRange):
                new_range = new_range.to_slice(
                    range_length,
                    prev_start=range_length - prev_end if backwards else prev_start,
                    prev_end=range_length - prev_start if backwards else prev_end,
                )
                if backwards:
                    new_range = slice(range_length - new_range.stop, range_length - new_range.start)

            # Update previous bounds
            if isinstance(new_range, slice):
                prev_start = new_range.start
                prev_end = new_range.stop
            else:
                prev_start = new_range_meta["start"]
                prev_end = new_range_meta["stop"]

            # Remap new range to index
            if new_range_was_gap:
                continue
            if not to_masks and isinstance(range_, slice) and isinstance(new_range, slice):
                new_range = slice(
                    range_.start + new_range.start,
                    range_.start + new_range.stop,
                )
                if range_was_hslice:
                    new_range = hslice.from_slice(new_range)
            else:
                if isinstance(range_, slice):
                    new_range = np.arange(range_.start, range_.stop)[new_range]
                else:
                    new_range = range_[new_range]
                if to_masks or range_was_mask:
                    if index is None:
                        raise ValueError("Must provide index")
                    new_range2 = np.full(len(index), False)
                    new_range2[new_range] = True
                    new_range = new_range2
            if to_templates:
                new_range = Rep("range_", context=dict(range_=new_range))
            new_ranges.append(new_range)

        if backwards:
            return tuple(new_ranges)[::-1]
        return tuple(new_ranges)

    @class_or_instancemethod
    def merge_split(
        cls_or_self,
        split: tp.FixSplit,
        to_mask: bool = False,
        to_template: bool = False,
        try_to_slice: bool = True,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.FixRangeLike:
        """Merge a split of multiple fixed ranges into a fixed range.

        Creates one mask and sets True for each range. If all input ranges are masks,
        returns that mask. If all input ranges are slices, returns a slice if possible.
        Otherwise, returns integer indices.

        Enable `to_mask` to convert the new range into a mask. Enable `to_template` to wrap
        the resulting range with a template of the type `vectorbtpro.utils.template.Rep`.
        Both options can be combined."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        index = try_to_datetime_index(index)
        all_hslices = True
        all_masks = True
        new_ranges = []
        if len(split) == 1:
            raise ValueError("Two or more ranges are required to be merged")
        for range_ in split:
            range_meta = cls_or_self.to_ready_range(
                range_,
                try_to_slice=try_to_slice,
                allow_zero_len=True,
                return_meta=True,
                template_context=template_context,
                index=index,
            )
            if not range_meta["was_hslice"]:
                all_hslices = False
            if not range_meta["was_mask"]:
                all_masks = False
            new_ranges.append(range_meta["range_"])
        ranges = new_ranges

        merged_mask = np.full(len(index), False)
        for range_ in ranges:
            merged_mask[range_] = True
        merged_indices = np.flatnonzero(merged_mask)
        if try_to_slice and is_range(merged_indices):
            if merged_indices[0] == 0 and merged_indices[-1] == len(index) - 1:
                merged_slice = slice(None, None, None)
            else:
                merged_slice = slice(merged_indices[0], merged_indices[-1] + 1, None)
            if all_hslices:
                new_range = hslice.from_slice(merged_slice)
            else:
                new_range = merged_slice
        else:
            if to_mask or all_masks:
                new_range = merged_mask
            else:
                new_range = merged_indices
        if to_template:
            new_range = Rep("range_", context=dict(range_=new_range))
        return new_range

    def to_fixed(self: SplitterT, split_range_kwargs: tp.KwargsLike = None, **kwargs) -> SplitterT:
        """Convert relative ranges into fixed ones and return a new `Splitter` instance.

        Keyword arguments `split_range_kwargs` are passed to `Splitter.split_range`."""
        if split_range_kwargs is None:
            split_range_kwargs = {}
        new_splits = []
        for split in self.splits_arr:
            new_splits.append(self.split_range(slice(None), split, **split_range_kwargs))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_splits = np.asarray(new_splits)
        except Exception as e:
            new_splits = np.asarray(new_splits, dtype=object)
        return self.replace(splits=new_splits, **kwargs)

    # ############# Indexing ############# #

    @class_or_instancemethod
    def get_target_index_range(
        cls_or_self,
        range_: tp.FixRangeLike,
        target_index: tp.IndexLike,
        target_freq: tp.Optional[tp.FrequencyLike] = None,
        try_to_slice: bool = True,
        allow_zero_len: bool = False,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.FixRangeLike:
        """Get a range that can be mapped into a target index.

        If `index` and `target_index` are the same, returns the range. Otherwise,
        uses `vectorbtpro.base.resampling.base.Resampler.resample_source_mask` to resample
        the range into the target index. In such a case, `freq` and `target_freq` must be provided."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        if target_index is None:
            raise ValueError("Must provide target index")
        target_index = try_to_datetime_index(target_index)
        if index.equals(target_index):
            return cls_or_self.to_ready_range(
                range_,
                try_to_slice=try_to_slice,
                allow_zero_len=allow_zero_len,
                template_context=template_context,
                index=target_index,
            )

        mask = cls_or_self.get_range_mask(range_, template_context=template_context, index=index)
        resampler = Resampler(
            source_index=index,
            target_index=target_index,
            source_freq=freq,
            target_freq=target_freq,
        )
        target_mask = resampler.resample_source_mask(mask, jitted=jitted, silence_warnings=silence_warnings)
        return cls_or_self.to_ready_range(
            target_mask,
            try_to_slice=try_to_slice,
            allow_zero_len=allow_zero_len,
            template_context=template_context,
            index=target_index,
        )

    @class_or_instancemethod
    def select_range(
        cls_or_self,
        obj: tp.Any,
        range_: tp.FixRangeLike,
        use_obj_index: bool = True,
        obj_index: tp.Optional[tp.IndexLike] = None,
        obj_freq: tp.Optional[tp.FrequencyLike] = None,
        try_to_slice: bool = True,
        allow_zero_len: bool = False,
        template_context: tp.KwargsLike = None,
        jitted: tp.JittedOption = None,
        silence_warnings: tp.Optional[bool] = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Any:
        """Select a range from an array-like object.

        If the object is Pandas-like and `obj_index` is not None, searches for an index in the object.
        Once found, uses `Splitter.get_target_index_range` to get the range that maps to the object index.
        Finally, uses `obj.iloc` to select the range."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        if use_obj_index and (
            isinstance(obj, (pd.Index, pd.Series, pd.DataFrame, PandasIndexer)) or obj_index is not None
        ):
            if obj_index is None:
                if isinstance(obj, pd.Index):
                    obj_index = obj
                elif hasattr(obj, "index"):
                    obj_index = obj.index
                elif hasattr(obj, "wrapper"):
                    obj_index = obj.wrapper.index
                else:
                    raise ValueError("Must provide object index")
            range_ = cls_or_self.get_target_index_range(
                range_,
                target_index=obj_index,
                target_freq=obj_freq,
                try_to_slice=try_to_slice,
                allow_zero_len=allow_zero_len,
                template_context=template_context,
                jitted=jitted,
                silence_warnings=silence_warnings,
                index=index,
                freq=freq,
            )
        else:
            range_ = cls_or_self.to_ready_range(
                range_,
                try_to_slice=try_to_slice,
                allow_zero_len=allow_zero_len,
                template_context=template_context,
                index=index,
            )
        if isinstance(obj, (pd.Series, pd.DataFrame, PandasIndexer)):
            return obj.iloc[range_]
        return obj[range_]

    # ############# Sets ############# #

    def split_set(
        self: SplitterT,
        new_split: tp.SplitLike,
        column: tp.Optional[tp.Hashable] = None,
        new_set_labels: tp.Optional[tp.Sequence[tp.Hashable]] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **split_range_kwargs,
    ) -> SplitterT:
        """Split a set (column) into multiple sets (columns).

        Arguments `new_split` and `**split_range_kwargs` are passed to `Splitter.split_range`.

        Column must be provided if there are two or more sets.

        Use `new_set_labels` to specify the labels of the new sets; it must have the same length
        as there are new ranges in the new split. To provide final labels, define `columns` in
        `wrapper_kwargs`."""
        if self.n_sets == 0:
            raise ValueError("There are no sets to split")
        if self.n_sets > 1:
            if column is None:
                raise ValueError("Must provide column for multiple sets")
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
        else:
            column = 0
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        new_splits = []
        for split in self.splits_arr:
            new_ranges = self.split_range(split[column], new_split, **split_range_kwargs)
            new_splits.append([*split[:column], *new_ranges, *split[column + 1 :]])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_splits = np.asarray(new_splits)
        except Exception as e:
            new_splits = np.asarray(new_splits, dtype=object)
        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            n_new_sets = new_splits.shape[1] - self.n_sets + 1
            if new_set_labels is None:
                old_set_label = self.set_labels[column]
                if isinstance(old_set_label, str) and len(old_set_label.split("+")) == n_new_sets:
                    new_set_labels = old_set_label.split("+")
                else:
                    new_set_labels = [str(old_set_label) + "/" + str(i) for i in range(n_new_sets)]
            if len(new_set_labels) != n_new_sets:
                raise ValueError(f"Argument new_set_labels must have length {n_new_sets}, not {len(new_set_labels)}")
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(column)
            new_columns = new_columns.insert(column, new_set_labels)
            wrapper_kwargs["columns"] = new_columns
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits=new_splits, **init_kwargs)

    def merge_sets(
        self: SplitterT,
        columns: tp.Optional[tp.Iterable[tp.Hashable]] = None,
        new_set_label: tp.Optional[tp.Hashable] = None,
        insert_at_last: bool = False,
        wrapper_kwargs: tp.KwargsLike = None,
        init_kwargs: tp.KwargsLike = None,
        **merge_split_kwargs,
    ) -> SplitterT:
        """Merge multiple sets (columns) into a set (column).

        Arguments `**merge_split_kwargs` are passed to `Splitter.merge_split`.

        If columns are not provided, merges all columns. If provided and `insert_at_last` is True,
        a new column is inserted at the position of the last column.

        Use `new_set_label` to specify the label of the new set. To provide final labels,
        define `columns` in `wrapper_kwargs`."""
        if self.n_sets < 2:
            raise ValueError("There are no sets to merge")
        if columns is None:
            columns = range(len(self.set_labels))
        new_columns = []
        for column in columns:
            if not isinstance(column, int):
                column = self.set_labels.get_indexer([column])[0]
                if column == -1:
                    raise ValueError(f"Column '{column}' not found")
            new_columns.append(column)
        columns = sorted(new_columns)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        new_splits = []
        for split in self.splits_arr:
            split_to_merge = []
            for j, range_ in enumerate(split):
                if j in columns:
                    split_to_merge.append(range_)
            new_range = self.merge_split(split_to_merge, **merge_split_kwargs)
            new_split = []
            for j in range(self.n_sets):
                if j not in columns:
                    new_split.append(split[j])
                else:
                    if insert_at_last:
                        if j == columns[-1]:
                            new_split.append(new_range)
                    else:
                        if j == columns[0]:
                            new_split.append(new_range)
            new_splits.append(new_split)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_splits = np.asarray(new_splits)
        except Exception as e:
            new_splits = np.asarray(new_splits, dtype=object)
        if "columns" not in wrapper_kwargs:
            wrapper_kwargs = dict(wrapper_kwargs)
            if new_set_label is None:
                old_set_labels = self.set_labels[columns]
                can_aggregate = True
                prefix = None
                suffix = None
                for i, old_set_label in enumerate(old_set_labels):
                    if not isinstance(old_set_label, str):
                        can_aggregate = False
                        break
                    _prefix = "/".join(old_set_label.split("/")[:-1])
                    _suffix = old_set_label.split("/")[-1]
                    if not _suffix.isdigit():
                        can_aggregate = False
                        break
                    _suffix = int(_suffix)
                    if prefix is None:
                        prefix = _prefix
                        suffix = _suffix
                        continue
                    if suffix != 0:
                        can_aggregate = False
                        break
                    if not _prefix == prefix or _suffix != i:
                        can_aggregate = False
                        break
                if can_aggregate and prefix + "/%d" % len(old_set_labels) not in self.set_labels:
                    new_set_label = prefix
                else:
                    new_set_label = "+".join(map(str, old_set_labels))
            new_columns = self.set_labels.copy()
            new_columns = new_columns.delete(columns)
            if insert_at_last:
                new_columns = new_columns.insert(columns[-1] - len(columns) + 1, new_set_label)
            else:
                new_columns = new_columns.insert(columns[0], new_set_label)
            wrapper_kwargs["columns"] = new_columns
        if "ndim" not in wrapper_kwargs:
            if len(wrapper_kwargs["columns"]) == 1:
                wrapper_kwargs["ndim"] = 1
        new_wrapper = self.wrapper.replace(**wrapper_kwargs)
        return self.replace(wrapper=new_wrapper, splits=new_splits, **init_kwargs)

    # ############# Bounds ############# #

    @class_or_instancemethod
    def get_range_bounds(
        cls_or_self,
        range_: tp.FixRangeLike,
        map_to_index: bool = False,
        check_constant: bool = True,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.Tuple[int, int]:
        """Get the left and right bound of a range.

        !!! note
            Ranges are assumed to go strictly from left to right."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        range_meta = cls_or_self.to_ready_range(
            range_,
            return_meta=True,
            template_context=template_context,
            index=index,
        )
        start = range_meta["start"]
        stop = range_meta["stop"]
        if check_constant and not range_meta["is_constant"]:
            raise ValueError("Range is not constant")
        if map_to_index:
            if stop == len(index):
                freq = BaseIDXAccessor(index, freq=freq).any_freq
                if freq is None:
                    raise ValueError("Must provide freq")
                return index[start], index[stop - 1] + freq
            return index[start], index[stop]
        return start, stop

    def get_bounds_arr(
        self,
        map_to_index: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **range_bounds_kwargs,
    ) -> tp.BoundsArray:
        """Three-dimensional integer array with bounds.

        First axis represents splits. Second axis represents sets. Third axis represents bounds.

        Each range is getting selected using `Splitter.get_range` and then measured using
        `Splitter.get_range_bounds`. Keyword arguments `**kwargs` are passed to
        `Splitter.get_range_bounds`."""
        if map_to_index:
            dtype = self.index.dtype
        else:
            dtype = np.int_
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        n_splits = self.get_n_splits(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        n_sets = self.get_n_sets(set_group_by=set_group_by)
        bounds = np.empty((n_splits, n_sets, 2), dtype=dtype)

        for i in range(n_splits):
            for j in range(n_sets):
                range_ = self.get_range(
                    split=i,
                    set_=j,
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    split_as_indices=True,
                    set_as_indices=True,
                    template_context=template_context,
                )
                bounds[i, j, :] = self.get_range_bounds(
                    range_,
                    map_to_index=map_to_index,
                    template_context=template_context,
                    **range_bounds_kwargs,
                )
        return bounds

    @property
    def bounds_arr(self) -> tp.BoundsArray:
        """`GenericAccessor.get_bounds_arr` with default arguments."""
        return self.get_bounds_arr()

    def get_bounds(
        self,
        map_to_index: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Frame:
        """Boolean DataFrame where index are bounds and columns are splits stacked together.

        Keyword arguments `**kwargs` are passed to `Splitter.get_bounds_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        bounds_arr = self.get_bounds_arr(
            map_to_index=map_to_index,
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        )
        out = np.moveaxis(bounds_arr, -1, 0).reshape((2, -1))
        new_index = pd.Index(["start", "end"], name="bound")
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_columns = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=new_index, columns=new_columns)

    @property
    def bounds(self) -> tp.Frame:
        """`GenericAccessor.get_bounds` with default arguments."""
        return self.get_bounds()

    @property
    def index_bounds(self) -> tp.Frame:
        """`GenericAccessor.get_bounds` with `map_to_index=True`."""
        return self.get_bounds(map_to_index=True)

    def get_duration(self, **kwargs) -> tp.Series:
        """Get duration."""
        bounds = self.get_bounds(**kwargs)
        return (bounds.iloc[1] - bounds.iloc[0]).rename("duration")

    @property
    def duration(self) -> tp.Series:
        """`GenericAccessor.get_duration` with default arguments."""
        return self.get_duration()

    @property
    def index_duration(self) -> tp.Series:
        """`GenericAccessor.get_duration` with `map_to_index=True`."""
        return self.get_duration(map_to_index=True)

    # ############# Masks ############# #

    @class_or_instancemethod
    def get_range_mask(
        cls_or_self,
        range_: tp.FixRangeLike,
        template_context: tp.KwargsLike = None,
        index: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Array1d:
        """Get the mask of a range."""
        if index is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide index")
            index = cls_or_self.index
        range_ = cls_or_self.to_ready_range(
            range_,
            allow_zero_len=True,
            template_context=template_context,
            index=index,
        )
        if isinstance(range_, np.ndarray) and range_.dtype == np.bool_:
            return range_
        mask = np.full(len(index), False)
        mask[range_] = True
        return mask

    def get_iter_split_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Array2d, None, None]:
        """Generator of two-dimensional boolean arrays, one per split.

        First axis represents sets. Second axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_range_mask`."""
        if split_group_by is None and set_group_by is None and self.splits_arr.dtype == np.bool_:
            for i in range(self.n_splits):
                yield self.splits_arr[i]
        else:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            n_splits = self.get_n_splits(split_group_by=split_group_by)
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            n_sets = self.get_n_sets(set_group_by=set_group_by)
            for i in range(n_splits):
                out = np.full((n_sets, len(self.index)), False)
                for j in range(n_sets):
                    range_ = self.get_range(
                        split=i,
                        set_=j,
                        split_group_by=split_group_by,
                        set_group_by=set_group_by,
                        split_as_indices=True,
                        set_as_indices=True,
                        template_context=template_context,
                    )
                    out[j, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
                yield out

    @property
    def iter_split_mask_arrs(self) -> tp.Generator[tp.Array2d, None, None]:
        """`GenericAccessor.get_iter_split_mask_arrs` with default arguments."""
        return self.get_iter_split_mask_arrs()

    def get_iter_set_mask_arrs(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Array2d, None, None]:
        """Generator of two-dimensional boolean arrays, one per set.

        First axis represents splits. Second axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_range_mask`."""
        if split_group_by is None and set_group_by is None and self.splits_arr.dtype == np.bool_:
            for j in range(self.n_sets):
                yield self.splits_arr[:, j]
        else:
            split_group_by = self.get_split_grouper(split_group_by=split_group_by)
            n_splits = self.get_n_splits(split_group_by=split_group_by)
            set_group_by = self.get_set_grouper(set_group_by=set_group_by)
            n_sets = self.get_n_sets(set_group_by=set_group_by)
            for j in range(n_sets):
                out = np.full((n_splits, len(self.index)), False)
                for i in range(n_splits):
                    range_ = self.get_range(
                        split=i,
                        set_=j,
                        split_group_by=split_group_by,
                        set_group_by=set_group_by,
                        split_as_indices=True,
                        set_as_indices=True,
                        template_context=template_context,
                    )
                    out[i, :] = self.get_range_mask(range_, template_context=template_context, **kwargs)
                yield out

    @property
    def iter_set_mask_arrs(self) -> tp.Generator[tp.Array2d, None, None]:
        """`GenericAccessor.get_iter_set_mask_arrs` with default arguments."""
        return self.get_iter_set_mask_arrs()

    def get_iter_split_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Frame, None, None]:
        """Generator of boolean DataFrames, one per split.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_split_mask_arrs`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        for mask in self.get_iter_split_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=set_labels)

    @property
    def iter_split_masks(self) -> tp.Generator[tp.Frame, None, None]:
        """`GenericAccessor.get_iter_split_masks` with default arguments."""
        return self.get_iter_split_masks()

    def get_iter_set_masks(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Generator[tp.Frame, None, None]:
        """Generator of boolean DataFrames, one per set.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_set_mask_arrs`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        for mask in self.get_iter_set_mask_arrs(
            split_group_by=split_group_by,
            set_group_by=set_group_by,
            **kwargs,
        ):
            yield pd.DataFrame(np.moveaxis(mask, -1, 0), index=self.index, columns=split_labels)

    @property
    def iter_set_masks(self) -> tp.Generator[tp.Frame, None, None]:
        """`GenericAccessor.get_iter_set_masks` with default arguments."""
        return self.get_iter_set_masks()

    def get_mask_arr(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SplitsMask:
        """Three-dimensional boolean array with splits.

        First axis represents splits. Second axis represents sets. Third axis represents index.

        Keyword arguments `**kwargs` are passed to `Splitter.get_iter_split_mask_arrs`."""
        if split_group_by is None and set_group_by is None and self.splits_arr.dtype == np.bool_:
            return self.splits_arr
        return np.array(
            list(
                self.get_iter_split_mask_arrs(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    template_context=template_context,
                    **kwargs,
                )
            )
        )

    @property
    def mask_arr(self) -> tp.SplitsMask:
        """`GenericAccessor.get_mask_arr` with default arguments."""
        return self.get_mask_arr()

    def get_mask(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Frame:
        """Boolean DataFrame where index is `Splitter.index` and columns are splits stacked together.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`.

        !!! warning
            Boolean arrays for a big number of splits may take a considerable amount of memory."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask_arr = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        out = np.moveaxis(mask_arr, -1, 0).reshape((len(self.index), -1))
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        new_columns = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.DataFrame(out, index=self.index, columns=new_columns)

    @property
    def mask(self) -> tp.Frame:
        """`GenericAccessor.get_mask` with default arguments."""
        return self.get_mask()

    def get_split_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Series:
        """Get the coverage of each split mask.

        If `overlapping` is True, returns the number of overlapping True values between sets in each split.
        If `normalize` is True, returns the number of True values in each split relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each split relative to the total number of True values across all splits.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        mask = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                coverage = (mask.sum(axis=1) > 1).sum(axis=1) / mask.any(axis=1).sum(axis=1)
            else:
                coverage = (mask.sum(axis=1) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask.any(axis=1).sum(axis=1) / mask.any(axis=(0, 1)).sum()
                else:
                    coverage = mask.any(axis=1).mean(axis=1)
            else:
                coverage = mask.any(axis=1).sum(axis=1)
        return pd.Series(coverage, index=split_labels, name="split_coverage")

    @property
    def split_coverage(self) -> tp.Series:
        """`GenericAccessor.get_split_coverage` with default arguments."""
        return self.get_split_coverage()

    def get_set_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> tp.Series:
        """Get the coverage of each set mask.

        If `overlapping` is True, returns the number of overlapping True values between splits in each set.
        If `normalize` is True, returns the number of True values in each set relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each set relative to the total number of True values across all sets.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                coverage = (mask.sum(axis=0) > 1).sum(axis=1) / mask.any(axis=0).sum(axis=1)
            else:
                coverage = (mask.sum(axis=0) > 1).sum(axis=1)
        else:
            if normalize:
                if relative:
                    coverage = mask.any(axis=0).sum(axis=1) / mask.any(axis=(0, 1)).sum()
                else:
                    coverage = mask.any(axis=0).mean(axis=1)
            else:
                coverage = mask.any(axis=0).sum(axis=1)
        return pd.Series(coverage, index=set_labels, name="set_coverage")

    @property
    def set_coverage(self) -> tp.Series:
        """`GenericAccessor.get_set_coverage` with default arguments."""
        return self.get_set_coverage()

    def get_range_coverage(
        self,
        normalize: bool = True,
        relative: bool = False,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        index_combine_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Series:
        """Get the coverage of each range mask.

        If `normalize` is True, returns the number of True values in each range relative to the
        length of the index. If `normalize` and `relative` are True, returns the number of True values
        in each range relative to the total number of True values in its split.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        split_group_by = self.get_split_grouper(split_group_by=split_group_by)
        split_labels = self.get_split_labels(split_group_by=split_group_by)
        set_group_by = self.get_set_grouper(set_group_by=set_group_by)
        set_labels = self.get_set_labels(set_group_by=set_group_by)
        mask = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if normalize:
            if relative:
                coverage = (mask.sum(axis=2) / mask.any(axis=1).sum(axis=1)[:, None]).flatten()
            else:
                coverage = (mask.sum(axis=2) / mask.shape[2]).flatten()
        else:
            coverage = mask.sum(axis=2).flatten()
        if index_combine_kwargs is None:
            index_combine_kwargs = {}
        index = combine_indexes((split_labels, set_labels), **index_combine_kwargs)
        return pd.Series(coverage, index=index, name="range_coverage")

    @property
    def range_coverage(self) -> tp.Series:
        """`GenericAccessor.get_range_coverage` with default arguments."""
        return self.get_range_coverage()

    def get_coverage(
        self,
        overlapping: bool = False,
        normalize: bool = True,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        **kwargs,
    ) -> float:
        """Get the coverage of the entire mask.

        If `overlapping` is True, returns the number of overlapping True values.
        If `normalize` is True, returns the number of True values relative to the length of the index.
        If `overlapping` and `normalize` are True, returns the number of overlapping True values relative
        to the total number of True values.

        Keyword arguments `**kwargs` are passed to `Splitter.get_mask_arr`."""
        mask = self.get_mask_arr(split_group_by=split_group_by, set_group_by=set_group_by, **kwargs)
        if overlapping:
            if normalize:
                return (mask.sum(axis=(0, 1)) > 1).sum() / mask.any(axis=(0, 1)).sum()
            return (mask.sum(axis=(0, 1)) > 1).sum()
        if normalize:
            return mask.any(axis=(0, 1)).mean()
        return mask.any(axis=(0, 1)).sum()

    @property
    def coverage(self) -> float:
        """`GenericAccessor.get_coverage` with default arguments."""
        return self.get_coverage()

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Splitter.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.splitter`."""
        from vectorbtpro._settings import settings

        splitter_stats_cfg = settings["splitter"]["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), splitter_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Index Start",
                calc_func=lambda self: self.index[0],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            end=dict(
                title="Index End",
                calc_func=lambda self: self.index[-1],
                agg_func=None,
                tags=["splitter", "index"],
            ),
            period=dict(
                title="Index Length",
                calc_func=lambda self: len(self.index),
                agg_func=None,
                tags=["splitter", "index"],
            ),
            split_count=dict(
                title="Splits",
                calc_func="n_splits",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            set_count=dict(
                title="Sets",
                calc_func="n_sets",
                agg_func=None,
                tags=["splitter", "splits"],
            ),
            coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                calc_func="coverage",
                overlapping=False,
                post_calc_func=lambda self, out, settings: out * 100 if settings["normalize"] else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_coverage=dict(
                title=RepFunc(lambda normalize: "Coverage [%]" if normalize else "Coverage"),
                check_has_ndim_2=True,
                calc_func="set_coverage",
                overlapping=False,
                relative=False,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_mean_rel_coverage=dict(
                title="Mean Rel Coverage [%]",
                check_has_ndim_2=True,
                check_normalize=True,
                calc_func="range_coverage",
                relative=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out.groupby(self.get_set_labels(
                        settings.get("set_group_by", None)
                    ).names).mean()[self.get_set_labels(
                        settings.get("set_group_by", None)
                    )] * 100,
                    orient="index_series",
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            overlap_coverage=dict(
                title=RepFunc(lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"),
                calc_func="coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: out * 100 if settings["normalize"] else out,
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
            set_overlap_coverage=dict(
                title=RepFunc(lambda normalize: "Overlap Coverage [%]" if normalize else "Overlap Coverage"),
                check_has_ndim_2=True,
                calc_func="set_coverage",
                overlapping=True,
                post_calc_func=lambda self, out, settings: to_dict(
                    out * 100 if settings["normalize"] else out, orient="index_series"
                ),
                agg_func=None,
                tags=["splitter", "splits", "coverage"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot splits as rows and sets as colors.

        Args:
            split_group_by (any): Split groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (any): Set groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (dict): Keyword arguments passed to `Splitter.get_iter_set_masks`.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap`.

                Can be a sequence, one per set.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd
        >>> from sklearn.model_selection import TimeSeriesSplit

        >>> index = pd.date_range("2020", "2021", freq="D")
        >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
        >>> splitter.plot()
        ```

        ![](/assets/images/api/Splitter.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure
        import plotly.express as px

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if self.n_splits > 0 and self.n_sets > 0:
            if mask_kwargs is None:
                mask_kwargs = {}
            set_masks = list(
                self.get_iter_set_masks(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    **mask_kwargs,
                )
            )
            if fig.layout.colorway is not None:
                colorway = fig.layout.colorway
            else:
                colorway = fig.layout.template.layout.colorway
            if len(set_masks) > len(colorway):
                colorway = px.colors.qualitative.Alphabet
            for i, mask in enumerate(set_masks):
                df = mask.vbt.wrapper.fill()
                df[mask] = i
                color = adjust_opacity(colorway[i % len(colorway)], 0.8)
                trace_name = str(df.columns[i])
                _trace_kwargs = merge_dicts(
                    dict(
                        showscale=False,
                        showlegend=True,
                        name=trace_name,
                        colorscale=[color, color],
                        hovertemplate="%{x}<br>Split: %{y}<br>Set: " + trace_name,
                    ),
                    resolve_dict(trace_kwargs, i=i),
                )
                fig = df.vbt.ts_heatmap(
                    trace_kwargs=_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    is_y_category=True,
                    fig=fig,
                )
        return fig

    def plot_coverage(
        self,
        split_group_by: tp.AnyGroupByLike = None,
        set_group_by: tp.AnyGroupByLike = None,
        mask_kwargs: tp.KwargsLike = None,
        trace_kwargs: tp.KwargsLikeSequence = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot index as rows and sets as lines.

        Args:
            split_group_by (any): Split groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            set_group_by (any): Set groups. See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            mask_kwargs (dict): Keyword arguments passed to `Splitter.get_iter_set_masks`.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.

                Can be a sequence, one per set.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
        ```pycon
        >>> import vectorbtpro as vbt
        >>> import pandas as pd
        >>> from sklearn.model_selection import TimeSeriesSplit

        >>> index = pd.date_range("2020", "2021", freq="D")
        >>> splitter = vbt.Splitter.from_sklearn(index, TimeSeriesSplit())
        >>> splitter.plot_coverage()
        ```

        ![](/assets/images/api/Splitter_coverage.svg)
        """
        from vectorbtpro.utils.opt_packages import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure
        import plotly.express as px

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if self.n_splits > 0 and self.n_sets > 0:
            if mask_kwargs is None:
                mask_kwargs = {}
            set_masks = list(
                self.get_iter_set_masks(
                    split_group_by=split_group_by,
                    set_group_by=set_group_by,
                    **mask_kwargs,
                )
            )
            if fig.layout.colorway is not None:
                colorway = fig.layout.colorway
            else:
                colorway = fig.layout.template.layout.colorway
            if len(set_masks) > len(colorway):
                colorway = px.colors.qualitative.Alphabet
            for i, mask in enumerate(set_masks):
                _trace_kwargs = merge_dicts(
                    dict(
                        name=str(mask.columns[i]),
                        line=dict(color=colorway[i % len(colorway)], shape="hv"),
                    ),
                    resolve_dict(trace_kwargs, i=i),
                )
                fig = mask.sum(axis=1).vbt.lineplot(
                    trace_kwargs=_trace_kwargs,
                    add_trace_kwargs=add_trace_kwargs,
                    fig=fig,
                )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Splitter.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.splitter`."""
        from vectorbtpro._settings import settings

        splitter_plots_cfg = settings["splitter"]["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), splitter_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="Splits",
                yaxis_kwargs=dict(title="Split"),
                plot_func="plot",
                tags="splitter",
            ),
            plot_coverage=dict(
                title="Coverage",
                yaxis_kwargs=dict(title="Count"),
                trace_kwargs=dict(showlegend=False),
                plot_func="plot_coverage",
                tags="splitter",
            ),
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


Splitter.override_metrics_doc(__pdoc__)
Splitter.override_subplots_doc(__pdoc__)
