# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Base class for working with data sources.

Class `Data` allows storing, downloading, updating, and managing data. It stores data
as a dictionary of Series/DataFrames keyed by symbol, and makes sure that
all pandas objects have the same index and columns by aligning them.
"""

import warnings
from pathlib import Path
import traceback
import inspect
import string

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_any_array, to_pd_array, to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.utils import checks
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.datetime_ import is_tz_aware, to_timezone, try_to_datetime_index
from vectorbtpro.utils.parsing import get_func_arg_names, extend_args
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.template import RepEval

__pdoc__ = {}


class symbol_dict(dict):
    """Dict that contains symbols as keys."""

    pass


DataT = tp.TypeVar("DataT", bound="Data")


class MetaColumns(type):
    """Meta class that exposes a read-only class property `MetaColumns.column_config`."""

    @property
    def column_config(cls) -> Config:
        """Column config."""
        return cls._column_config


class DataWithColumns(metaclass=MetaColumns):
    """Class exposes a read-only class property `DataWithColumns.field_config`."""

    @property
    def column_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${column_config}
        ```
        """
        return self._column_config


class MetaData(type(Analyzable), type(DataWithColumns)):
    pass


class Data(Analyzable, DataWithColumns, metaclass=MetaData):
    """Class that downloads, updates, and manages data coming from a data source."""

    _setting_keys: tp.SettingsKeys = dict(base="data")

    _writeable_attrs: tp.ClassVar[tp.Optional[tp.Set[str]]] = {"_column_config"}

    _column_config: tp.ClassVar[Config] = HybridConfig()

    @property
    def column_config(self) -> Config:
        """Column config of `${cls_name}`.

        ```python
        ${column_config}
        ```

        Returns `${cls_name}._column_config`, which gets (hybrid-) copied upon creation of each instance.
        Thus, changing this config won't affect the class.

        To change fields, you can either change the config in-place, override this property,
        or overwrite the instance variable `${cls_name}._column_config`.
        """
        return self._column_config

    def use_column_config_of(self, cls: tp.Type[DataT]) -> None:
        """Copy column config from another `Data` class."""
        self._column_config = cls.column_config.copy()

    @classmethod
    def row_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along rows.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to stack the wrappers.

        All objects to be merged must have the same symbols. Metadata such as the last index,
        fetching and returned keyword arguments, are taken from the last object."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        symbols = set()
        for obj in objs:
            symbols = symbols.union(set(obj.data.keys()))
        for obj in objs:
            if len(symbols.difference(set(obj.data.keys()))) > 0:
                raise ValueError("Objects to be merged must have the same symbols")
        if "data" not in kwargs:
            new_data = symbol_dict()
            for s in symbols:
                new_data[s] = kwargs["wrapper"].row_stack_and_wrap(*[obj.data[s] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        if "fetch_kwargs" not in kwargs:
            kwargs["fetch_kwargs"] = objs[-1].fetch_kwargs
        if "returned_kwargs" not in kwargs:
            kwargs["returned_kwargs"] = objs[-1].returned_kwargs
        if "last_index" not in kwargs:
            kwargs["last_index"] = objs[-1].last_index

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    @classmethod
    def column_stack(
        cls: tp.Type[DataT],
        *objs: tp.MaybeTuple[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along columns.

        Uses `vectorbtpro.base.wrapping.ArrayWrapper.column_stack` to stack the wrappers.

        All objects to be merged must have the same symbols and index. Metadata such as the last index,
        fetching and returned keyword arguments, must be the same in all objects."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Records")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                union_index=False,
                **wrapper_kwargs,
            )

        symbols = set()
        for obj in objs:
            symbols = symbols.union(set(obj.data.keys()))
        for obj in objs:
            if len(symbols.difference(set(obj.data.keys()))) > 0:
                raise ValueError("Objects to be merged must have the same symbols")
        if "data" not in kwargs:
            new_data = symbol_dict()
            for s in symbols:
                new_data[s] = kwargs["wrapper"].column_stack_and_wrap(*[obj.data[s] for obj in objs], group_by=False)
            kwargs["data"] = new_data

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        return cls(**kwargs)

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Analyzable._expected_keys or set()) | {
        "data",
        "single_symbol",
        "fetch_kwargs",
        "returned_kwargs",
        "last_index",
        "tz_localize",
        "tz_convert",
        "missing_index",
        "missing_columns",
    }

    def __init__(
        self,
        wrapper: ArrayWrapper,
        data: symbol_dict,
        single_symbol: bool,
        fetch_kwargs: symbol_dict,
        returned_kwargs: symbol_dict,
        last_index: symbol_dict,
        tz_localize: tp.Optional[tp.TimezoneLike],
        tz_convert: tp.Optional[tp.TimezoneLike],
        missing_index: str,
        missing_columns: str,
        **kwargs,
    ) -> None:

        checks.assert_instance_of(data, dict)
        checks.assert_instance_of(fetch_kwargs, dict)
        checks.assert_instance_of(returned_kwargs, dict)
        checks.assert_instance_of(last_index, dict)
        for symbol, obj in data.items():
            checks.assert_meta_equal(obj, data[list(data.keys())[0]])

        Analyzable.__init__(
            self,
            wrapper,
            data=data,
            single_symbol=single_symbol,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

        self._data = symbol_dict(data)
        self._single_symbol = single_symbol
        self._fetch_kwargs = symbol_dict(fetch_kwargs)
        self._returned_kwargs = symbol_dict(returned_kwargs)
        self._last_index = symbol_dict(last_index)
        self._tz_localize = tz_localize
        self._tz_convert = tz_convert
        self._missing_index = missing_index
        self._missing_columns = missing_columns

        # Copy writeable attrs
        self._column_config = type(self)._column_config.copy()

    def indexing_func(self: DataT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> DataT:
        """Perform indexing on `Data`."""
        new_wrapper = pd_indexing_func(self.wrapper)
        new_data = {s: pd_indexing_func(obj) for s, obj in self.data.items()}
        new_last_index = dict()
        for s, obj in self.data.items():
            if s in self.last_index:
                new_last_index[s] = min([self.last_index[s], new_wrapper.index[-1]])
        return self.replace(wrapper=new_wrapper, data=new_data, last_index=new_last_index)

    @property
    def data(self) -> symbol_dict:
        """Data dictionary keyed by symbol of type `symbol_dict`."""
        return self._data

    @property
    def single_symbol(self) -> bool:
        """Whether there is only one symbol in `Data.data`."""
        return self._single_symbol

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        """List of symbols."""
        return list(self.data.keys())

    @property
    def fetch_kwargs(self) -> symbol_dict:
        """Keyword arguments of type `symbol_dict` initially passed to `Data.fetch_symbol`."""
        return self._fetch_kwargs

    @property
    def returned_kwargs(self) -> symbol_dict:
        """Keyword arguments of type `symbol_dict` returned by `Data.fetch_symbol`."""
        return self._returned_kwargs

    @property
    def last_index(self) -> symbol_dict:
        """Last fetched index per symbol of type `symbol_dict`."""
        return self._last_index

    @property
    def tz_localize(self) -> tp.Optional[tp.TimezoneLike]:
        """`tz_localize` initially passed to `Data.fetch_symbol`."""
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Optional[tp.TimezoneLike]:
        """`tz_convert` initially passed to `Data.fetch_symbol`."""
        return self._tz_convert

    @property
    def missing_index(self) -> str:
        """`missing_index` initially passed to `Data.fetch_symbol`."""
        return self._missing_index

    @property
    def missing_columns(self) -> str:
        """`missing_columns` initially passed to `Data.fetch_symbol`."""
        return self._missing_columns

    # ############# Pre- and post-processing ############# #

    @classmethod
    def prepare_tzaware_index(
        cls,
        obj: tp.SeriesFrame,
        tz_localize: tp.Optional[tp.TimezoneLike] = None,
        tz_convert: tp.Optional[tp.TimezoneLike] = None,
    ) -> tp.SeriesFrame:
        """Prepare a timezone-aware index of a pandas object.

        If the index is tz-naive, convert to a timezone using `tz_localize`.
        Convert the index from one timezone to another using `tz_convert`.
        See `vectorbtpro.utils.datetime_.to_timezone`.

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if tz_localize is None:
            tz_localize = data_cfg["tz_localize"]
        if tz_convert is None:
            tz_convert = data_cfg["tz_convert"]

        if isinstance(obj.index, pd.DatetimeIndex):
            if tz_localize is not None:
                if not is_tz_aware(obj.index):
                    obj = obj.tz_localize(to_timezone(tz_localize))
            if tz_convert is not None:
                obj = obj.tz_convert(to_timezone(tz_convert))
        obj.index = try_to_datetime_index(obj.index)
        return obj

    @classmethod
    def align_index(
        cls,
        data: symbol_dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> symbol_dict:
        """Align data to have the same index.

        The argument `missing` accepts the following values:

        * 'nan': set missing data points to NaN
        * 'drop': remove missing data points
        * 'raise': raise an error

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if missing is None:
            missing = data_cfg["missing_index"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        index = None
        for symbol, obj in data.items():
            obj_index = obj.index.sort_values()
            if index is None:
                index = obj_index
            else:
                if not index.equals(obj_index):
                    if missing == "nan":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching index. Setting missing data points to NaN.",
                                stacklevel=2,
                            )
                        index = index.union(obj_index)
                    elif missing == "drop":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching index. Dropping missing data points.",
                                stacklevel=2,
                            )
                        index = index.intersection(obj_index)
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching index")
                    else:
                        raise ValueError(f"Invalid option missing='{missing}'")

        # reindex
        new_data = symbol_dict({symbol: obj.reindex(index=index) for symbol, obj in data.items()})
        return new_data

    @classmethod
    def align_columns(
        cls,
        data: symbol_dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> symbol_dict:
        """Align data to have the same columns.

        See `Data.align_index` for `missing`."""
        if len(data) == 1:
            return data

        data_cfg = cls.get_settings(key_id="base")

        if missing is None:
            missing = data_cfg["missing_columns"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        columns = None
        multiple_columns = False
        name_is_none = False
        for symbol, obj in data.items():
            if checks.is_series(obj):
                if obj.name is None:
                    name_is_none = True
                obj = obj.to_frame()
            else:
                multiple_columns = True
            obj_columns = obj.columns
            if columns is None:
                columns = obj_columns
            else:
                if not columns.equals(obj_columns):
                    if missing == "nan":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching columns. Setting missing data points to NaN.",
                                stacklevel=2,
                            )
                        columns = columns.union(obj_columns)
                    elif missing == "drop":
                        if not silence_warnings:
                            warnings.warn(
                                "Symbols have mismatching columns. Dropping missing data points.",
                                stacklevel=2,
                            )
                        columns = columns.intersection(obj_columns)
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching columns")
                    else:
                        raise ValueError(f"Invalid option missing='{missing}'")

        # reindex
        new_data = symbol_dict()
        for symbol, obj in data.items():
            if checks.is_series(obj):
                obj = obj.to_frame(name=obj.name)
            obj = obj.reindex(columns=columns)
            if not multiple_columns:
                obj = obj[columns[0]]
                if name_is_none:
                    obj = obj.rename(None)
            new_data[symbol] = obj
        return new_data

    @staticmethod
    def select_symbol_kwargs(symbol: tp.Symbol, kwargs: tp.DictLike) -> dict:
        """Select keyword arguments belonging to `symbol`."""
        if kwargs is None:
            return {}
        if isinstance(kwargs, symbol_dict):
            if symbol not in kwargs:
                return {}
            kwargs = kwargs[symbol]
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, symbol_dict):
                if symbol in v:
                    _kwargs[k] = v[symbol]
            else:
                _kwargs[k] = v
        return _kwargs

    @classmethod
    def from_data(
        cls: tp.Type[DataT],
        data: symbol_dict,
        single_symbol: bool = False,
        tz_localize: tp.Optional[tp.TimezoneLike] = None,
        tz_convert: tp.Optional[tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        fetch_kwargs: tp.Optional[symbol_dict] = None,
        returned_kwargs: tp.Optional[symbol_dict] = None,
        last_index: tp.Optional[symbol_dict] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance from data.

        Args:
            data (dict): Dictionary of array-like objects keyed by symbol.
            single_symbol (bool): Whether there is only one symbol in `data`.
            tz_localize (timezone_like): See `Data.prepare_tzaware_index`.
            tz_convert (timezone_like): See `Data.prepare_tzaware_index`.
            missing_index (str): See `Data.align_index`.
            missing_columns (str): See `Data.align_columns`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
            fetch_kwargs (symbol_dict): Keyword arguments initially passed to `Data.fetch_symbol`.
            returned_kwargs (symbol_dict): Keyword arguments returned by `Data.fetch_symbol`.
            last_index (symbol_dict): Last fetched index per symbol.
            silence_warnings (bool): Whether to silence all warnings.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbtpro._settings.data`."""
        data_cfg = cls.get_settings(key_id="base")

        if tz_localize is None:
            tz_localize = data_cfg["tz_localize"]
        if tz_convert is None:
            tz_convert = data_cfg["tz_convert"]
        if missing_index is None:
            missing_index = data_cfg["missing_index"]
        if missing_columns is None:
            missing_columns = data_cfg["missing_columns"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if fetch_kwargs is None:
            fetch_kwargs = symbol_dict()
        if returned_kwargs is None:
            returned_kwargs = symbol_dict()
        if last_index is None:
            last_index = symbol_dict()

        data = symbol_dict(data)
        for symbol, obj in data.items():
            obj = to_pd_array(obj)
            obj = obj[~obj.index.duplicated(keep="last")]
            obj = cls.prepare_tzaware_index(obj, tz_localize=tz_localize, tz_convert=tz_convert)
            data[symbol] = obj
            if symbol not in last_index:
                last_index[symbol] = obj.index[-1]

        data = cls.align_index(data, missing=missing_index, silence_warnings=silence_warnings)
        data = cls.align_columns(data, missing=missing_columns, silence_warnings=silence_warnings)

        for symbol, obj in data.items():
            if isinstance(obj.index, pd.DatetimeIndex):
                obj.index.freq = obj.index.inferred_freq

        symbols = list(data.keys())
        wrapper = ArrayWrapper.from_obj(data[symbols[0]], **wrapper_kwargs)
        return cls(
            wrapper,
            data,
            single_symbol,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

    # ############# Fetching ############# #

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, tp.Tuple[tp.SeriesFrame, tp.Kwargs]]:
        """Fetch a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    @classmethod
    def fetch(
        cls: tp.Type[DataT],
        symbols: tp.Union[tp.Symbol, tp.Symbols] = None,
        *,
        tz_localize: tp.Optional[tp.TimezoneLike] = None,
        tz_convert: tp.Optional[tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Fetch data using `Data.fetch_symbol` and pass to `Data.from_data`.

        Args:
            symbols (hashable or sequence of hashable): One or multiple symbols.

                !!! note
                    Tuple is considered as a single symbol (tuple is a hashable).
            tz_localize (any): See `Data.from_data`.
            tz_convert (any): See `Data.from_data`.
            missing_index (str): See `Data.from_data`.
            missing_columns (str): See `Data.from_data`.
            wrapper_kwargs (dict): See `Data.from_data`.
            show_progress (bool): Whether to show the progress bar.
                Defaults to True if the global flag for data is True and there is more than one symbol.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            skip_on_error (bool): Whether to skip the symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.fetch_symbol` if in the signature.
            **kwargs: Passed to `Data.fetch_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.

        For defaults, see `vectorbtpro._settings.data`.
        """
        data_cfg = cls.get_settings(key_id="base")

        if checks.is_hashable(symbols):
            single_symbol = True
            symbols = [symbols]
        elif not checks.is_sequence(symbols):
            raise TypeError("Symbols must be either a hashable or a sequence of hashable")
        else:
            single_symbol = False
        show_symbol_progress = show_progress
        if show_symbol_progress is None:
            show_symbol_progress = data_cfg["show_progress"]
        if show_progress is None:
            show_progress = data_cfg["show_progress"] and not single_symbol
        pbar_kwargs = merge_dicts(data_cfg["pbar_kwargs"], pbar_kwargs)
        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        data = symbol_dict()
        fetch_kwargs = symbol_dict()
        returned_kwargs = symbol_dict()
        with get_pbar(total=len(symbols), show_progress=show_progress, **pbar_kwargs) as pbar:
            for symbol in symbols:
                if symbol is not None:
                    pbar.set_description(str(symbol))

                _kwargs = cls.select_symbol_kwargs(symbol, kwargs)
                func_arg_names = get_func_arg_names(cls.fetch_symbol)
                if "show_progress" in func_arg_names:
                    _kwargs["show_progress"] = show_symbol_progress
                if "pbar_kwargs" in func_arg_names:
                    _kwargs["pbar_kwargs"] = pbar_kwargs
                if "silence_warnings" in func_arg_names:
                    _kwargs["silence_warnings"] = silence_warnings

                try:
                    out = cls.fetch_symbol(symbol, **_kwargs)
                except Exception as e:
                    if not skip_on_error:
                        raise e
                    if not silence_warnings:
                        warnings.warn(traceback.format_exc(), stacklevel=2)
                        warnings.warn(
                            f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                            stacklevel=2,
                        )
                else:
                    if out is None:
                        if not silence_warnings:
                            warnings.warn(
                                f"Symbol '{str(symbol)}' returned None. Skipping.",
                                stacklevel=2,
                            )
                    else:
                        if isinstance(out, tuple):
                            _data = out[0]
                            _returned_kwargs = out[1]
                        else:
                            _data = out
                            _returned_kwargs = {}
                        _data = to_any_array(_data)
                        if _data.size == 0:
                            if not silence_warnings:
                                warnings.warn(
                                    f"Symbol '{str(symbol)}' returned an empty array. Skipping.",
                                    stacklevel=2,
                                )
                        else:
                            data[symbol] = _data
                            returned_kwargs[symbol] = _returned_kwargs
                            fetch_kwargs[symbol] = _kwargs

                pbar.update(1)
        if len(data) == 0:
            raise ValueError("No symbols could be fetched")

        # Create new instance from data
        return cls.from_data(
            data,
            single_symbol=single_symbol,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            wrapper_kwargs=wrapper_kwargs,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            silence_warnings=silence_warnings,
        )

    # ############# Updating ############# #

    def update_symbol(
        self,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.Union[tp.SeriesFrame, tp.Tuple[tp.SeriesFrame, tp.Kwargs]]:
        """Update a symbol.

        Can also return a dictionary that will be accessible in `Data.returned_kwargs`.

        This is an abstract method - override it to define custom logic."""
        raise NotImplementedError

    def update(
        self: DataT,
        *,
        concat: bool = True,
        show_progress: bool = False,
        pbar_kwargs: tp.KwargsLike = None,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Fetch additional data using `Data.update_symbol` and append it to the existing data.

        Args:
            concat (bool): Whether to concatenate existing and updated/new data.
            show_progress (bool): Whether to show the progress bar.

                Will also forward this argument to `Data.update_symbol` if accepted by `Data.fetch_symbol`.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.

                Will also forward this argument to `Data.update_symbol` if accepted by `Data.fetch_symbol`.
            skip_on_error (bool): Whether to skip the symbol when an exception is raised.
            silence_warnings (bool): Whether to silence all warnings.

                Will also forward this argument to `Data.update_symbol` if accepted by `Data.fetch_symbol`.
            **kwargs: Passed to `Data.update_symbol`.

                If two symbols require different keyword arguments, pass `symbol_dict` for each argument.

        !!! note
            Returns a new `Data` instance instead of changing the data in place."""
        data_cfg = self.get_settings(key_id="base")

        pbar_kwargs = merge_dicts(data_cfg["pbar_kwargs"], pbar_kwargs)
        if skip_on_error is None:
            skip_on_error = data_cfg["skip_on_error"]
        if silence_warnings is None:
            silence_warnings = data_cfg["silence_warnings"]

        new_data = symbol_dict()
        new_last_index = symbol_dict()
        new_returned_kwargs = symbol_dict()
        with get_pbar(total=len(self.data), show_progress=show_progress, **pbar_kwargs) as pbar:
            for symbol, obj in self.data.items():
                if symbol is not None:
                    pbar.set_description(str(symbol))

                _kwargs = self.select_symbol_kwargs(symbol, kwargs)
                func_arg_names = get_func_arg_names(self.fetch_symbol)
                if "show_progress" in func_arg_names:
                    _kwargs["show_progress"] = show_progress
                if "pbar_kwargs" in func_arg_names:
                    _kwargs["pbar_kwargs"] = pbar_kwargs
                if "silence_warnings" in func_arg_names:
                    _kwargs["silence_warnings"] = silence_warnings

                skip_symbol = False
                try:
                    out = self.update_symbol(symbol, **_kwargs)
                except Exception as e:
                    if not skip_on_error:
                        raise e
                    if not silence_warnings:
                        warnings.warn(traceback.format_exc(), stacklevel=2)
                        warnings.warn(
                            f"Symbol '{str(symbol)}' raised an exception. Skipping.",
                            stacklevel=2,
                        )
                    skip_symbol = True
                else:
                    if out is None:
                        if not silence_warnings:
                            warnings.warn(
                                f"Symbol '{str(symbol)}' returned None. Skipping.",
                                stacklevel=2,
                            )
                        skip_symbol = True
                    else:
                        if isinstance(out, tuple):
                            new_obj = out[0]
                            new_returned_kwargs[symbol] = out[1]
                        else:
                            new_obj = out
                            new_returned_kwargs[symbol] = {}
                        new_obj = to_any_array(new_obj)
                        if new_obj.size == 0:
                            if not silence_warnings:
                                warnings.warn(
                                    f"Symbol '{str(symbol)}' returned an empty array. Skipping.",
                                    stacklevel=2,
                                )
                            skip_symbol = True
                        else:
                            if not checks.is_pandas(new_obj):
                                new_obj = to_pd_array(new_obj)
                                new_obj.index = pd.RangeIndex(
                                    start=obj.index[-1],
                                    stop=obj.index[-1] + new_obj.shape[0],
                                    step=1,
                                )
                            new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
                            new_obj = self.prepare_tzaware_index(
                                new_obj,
                                tz_localize=self.tz_localize,
                                tz_convert=self.tz_convert,
                            )
                            new_data[symbol] = new_obj
                            if len(new_obj.index) > 0:
                                new_last_index[symbol] = new_obj.index[-1]
                            else:
                                new_last_index[symbol] = self.last_index[symbol]

                if skip_symbol:
                    new_data[symbol] = obj.iloc[0:0]
                    new_last_index[symbol] = self.last_index[symbol]
                pbar.update(1)

        # Get the last index in the old data from where the new data should begin
        from_index = None
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                index = new_obj.index[0]
            else:
                continue
            if from_index is None or index < from_index:
                from_index = index
        if from_index is None:
            if not silence_warnings:
                warnings.warn(
                    f"None of the symbols have been updated",
                    stacklevel=2,
                )
            return self.copy()

        # Concatenate the updated old data and the new data
        for symbol, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                to_index = new_obj.index[0]
            else:
                to_index = None
            obj = self.data[symbol]
            if checks.is_frame(obj) and checks.is_frame(new_obj):
                shared_columns = obj.columns.intersection(new_obj.columns)
                obj = obj[shared_columns]
                new_obj = new_obj[shared_columns]
            elif checks.is_frame(new_obj):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            elif checks.is_frame(obj):
                if new_obj.name is not None:
                    obj = obj[new_obj.name]
                else:
                    obj = obj[0]
            obj = obj.loc[from_index:to_index]
            new_obj = pd.concat((obj, new_obj), axis=0)
            new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
            new_data[symbol] = new_obj

        # Align the index and columns in the new data
        new_data = self.align_index(new_data, missing=self.missing_index, silence_warnings=silence_warnings)
        new_data = self.align_columns(new_data, missing=self.missing_columns, silence_warnings=silence_warnings)

        # Align the columns and data type in the old and new data
        for symbol, new_obj in new_data.items():
            obj = self.data[symbol]
            if checks.is_frame(obj) and checks.is_frame(new_obj):
                new_obj = new_obj[obj.columns]
            elif checks.is_frame(new_obj):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            if checks.is_frame(obj):
                new_obj = new_obj.astype(obj.dtypes)
            else:
                new_obj = new_obj.astype(obj.dtype)
            new_data[symbol] = new_obj

        if not concat:
            # Do not concatenate with the old data
            for symbol, new_obj in new_data.items():
                if isinstance(new_obj.index, pd.DatetimeIndex):
                    new_obj.index.freq = new_obj.index.inferred_freq
            new_index = new_data[self.symbols[0]].index
            return self.replace(
                wrapper=self.wrapper.replace(index=new_index),
                data=new_data,
                returned_kwargs=new_returned_kwargs,
                last_index=new_last_index,
            )

        # Append the new data to the old data
        for symbol, new_obj in new_data.items():
            obj = self.data[symbol]
            obj = obj.loc[:from_index]
            if obj.index[-1] == from_index:
                obj = obj.iloc[:-1]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if isinstance(new_obj.index, pd.DatetimeIndex):
                new_obj.index.freq = new_obj.index.inferred_freq
            new_data[symbol] = new_obj

        new_index = new_data[self.symbols[0]].index

        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),
            data=new_data,
            returned_kwargs=new_returned_kwargs,
            last_index=new_last_index,
        )

    # ############# Getting ############# #

    def get_symbol_wrapper(self, level_name: str = "symbol", **kwargs) -> ArrayWrapper:
        """Get wrapper where columns are symbols."""
        return self.wrapper.replace(
            columns=pd.Index(self.symbols, name=level_name),
            ndim=1 if self.single_symbol else 2,
            grouper=None,
            **kwargs,
        )

    @property
    def symbol_wrapper(self):
        """`Data.get_symbol_wrapper` with default arguments."""
        return self.get_symbol_wrapper()

    def concat(self, symbols: tp.Optional[tp.Symbols] = None, level_name: str = "symbol") -> dict:
        """Return a dict of Series/DataFrames with symbols as columns, keyed by column name."""
        if symbols is None:
            symbols = self.symbols

        new_data = {}
        first_data = self.data[symbols[0]]
        if self.single_symbol:
            if checks.is_series(first_data):
                new_data[first_data.name] = first_data.rename(symbols[0])
            else:
                for c in first_data.columns:
                    new_data[c] = first_data[c].rename(symbols[0])
        else:
            if checks.is_series(first_data):
                columns = pd.Index([first_data.name])
            else:
                columns = first_data.columns
            for c in columns:
                col_data = []
                for s in symbols:
                    if checks.is_series(self.data[s]):
                        col_data.append(self.data[s].rename(None))
                    else:
                        col_data.append(self.data[s][c].rename(None))
                new_data[c] = pd.concat(col_data, keys=pd.Index(symbols, name=level_name), axis=1)

        return new_data

    def get(
        self,
        columns: tp.Optional[tp.Union[tp.Label, tp.Labels]] = None,
        symbols: tp.Union[None, tp.Symbol, tp.Symbols] = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Get one or more columns of one or more symbols of data.

        If one symbol, returns data for that symbol. If multiple symbols, performs concatenation
        first and returns a DataFrame if one column and a tuple of DataFrames if a list of columns passed."""
        if symbols is None:
            single_symbol = self.single_symbol
            symbols = self.symbols
        else:
            if isinstance(symbols, list):
                single_symbol = False
            else:
                single_symbol = True
                symbols = [symbols]

        if single_symbol:
            if columns is None:
                return self.data[symbols[0]]
            return self.data[symbols[0]][columns]

        concat_data = self.concat(symbols=symbols, **kwargs)
        if len(concat_data) == 1:
            return tuple(concat_data.values())[0]
        if columns is not None:
            if isinstance(columns, list):
                return tuple([concat_data[c] for c in columns])
            return concat_data[columns]
        return tuple(concat_data.values())

    def get_column(self, col_name: str) -> tp.Optional[tp.SeriesFrame]:
        """Get column."""
        if col_name in self.wrapper.columns:
            return self.get(columns=col_name)
        col_name = col_name.lower().strip().replace(" ", "_")
        if hasattr(self.wrapper.columns, "str"):
            column_names = self.wrapper.columns.str.lower().str.strip().str.replace(" ", "_").tolist()
            if col_name in column_names:
                col_index = column_names.index(col_name)
                if col_index != -1:
                    return self.get(columns=self.wrapper.columns[col_index])
        return None

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open (magnet)."""
        return self.get_column("Open")

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High (magnet)."""
        return self.get_column("High")

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low (magnet)."""
        return self.get_column("Low")

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close (magnet)."""
        return self.get_column("Close")

    @property
    def volume(self) -> tp.Optional[tp.SeriesFrame]:
        """Volume (magnet)."""
        return self.get_column("Volume")

    @property
    def trade_count(self) -> tp.Optional[tp.SeriesFrame]:
        """Trade count (magnet)."""
        return self.get_column("Trade count")

    @property
    def vwap(self) -> tp.Optional[tp.SeriesFrame]:
        """VWAP (magnet)."""
        return self.get_column("VWAP")

    @property
    def hlc3(self) -> tp.Optional[tp.SeriesFrame]:
        """HLC/3 (magnet)."""
        return (self.high + self.low + self.close) / 3

    @property
    def ohlc4(self) -> tp.Optional[tp.SeriesFrame]:
        """OHLC/4 (magnet)."""
        return (self.open + self.high + self.low + self.close) / 4

    # ############# Selecting ############# #

    def select(self: DataT, symbols: tp.Union[tp.Symbol, tp.Symbols], **kwargs) -> DataT:
        """Create a new `Data` instance with one or more symbols from this instance."""
        if isinstance(symbols, list):
            single_symbol = False
        else:
            single_symbol = True
            symbols = [symbols]
        return self.replace(
            data=symbol_dict({k: v for k, v in self.data.items() if k in symbols}),
            single_symbol=single_symbol,
            fetch_kwargs=symbol_dict({k: v for k, v in self.fetch_kwargs.items() if k in symbols}),
            returned_kwargs=symbol_dict(
                {k: v for k, v in self.returned_kwargs.items() if k in symbols},
            ),
            last_index=symbol_dict({k: v for k, v in self.last_index.items() if k in symbols}),
            **kwargs,
        )

    # ############# Renaming ############# #

    def rename(self: DataT, rename: tp.Dict[tp.Hashable, tp.Hashable]) -> DataT:
        """Rename symbols using `rename` dict that maps old symbols and new symbols."""
        return self.replace(
            data={rename.get(k, k): v for k, v in self.data.items()},
            fetch_kwargs={rename.get(k, k): v for k, v in self.fetch_kwargs.items()},
            returned_kwargs={rename.get(k, k): v for k, v in self.returned_kwargs.items()},
            last_index={rename.get(k, k): v for k, v in self.last_index.items()},
        )

    # ############# Merging ############# #

    @classmethod
    def merge(
        cls: tp.Type[DataT],
        *datas: DataT,
        rename: tp.Optional[tp.Dict[tp.Hashable, tp.Hashable]] = None,
        **kwargs,
    ) -> DataT:
        """Merge multiple `Data` instances."""
        if len(datas) == 1:
            datas = datas[0]
        datas = list(datas)

        data = symbol_dict()
        fetch_kwargs = symbol_dict()
        returned_kwargs = symbol_dict()
        last_index = symbol_dict()
        for instance in datas:
            for s in instance.symbols:
                if s in data:
                    raise ValueError(f"Found a duplicate symbol '{s}'")
                if rename is None:
                    new_s = s
                else:
                    new_s = rename[s]
                data[new_s] = instance.data[s]
                if s in instance.fetch_kwargs:
                    fetch_kwargs[new_s] = instance.fetch_kwargs[s]
                if s in instance.returned_kwargs:
                    returned_kwargs[new_s] = instance.returned_kwargs[s]
                if s in instance.last_index:
                    last_index[new_s] = instance.last_index[s]

        return cls.from_data(
            data=data,
            single_symbol=False,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            **kwargs,
        )

    # ############# Saving ############# #

    def to_csv(
        self,
        dir_path: tp.Union[tp.PathLike, symbol_dict] = ".",
        ext: tp.Union[str, symbol_dict] = "csv",
        path_or_buf: tp.Optional[tp.Union[str, symbol_dict]] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, symbol_dict] = None,
        **kwargs,
    ) -> None:
        """Save data into CSV file(s).

        Any argument can be provided per symbol using `symbol_dict`."""
        for k, v in self.data.items():
            if path_or_buf is None:
                if isinstance(dir_path, symbol_dict):
                    _dir_path = dir_path[k]
                else:
                    _dir_path = dir_path
                _dir_path = Path(_dir_path)
                if isinstance(ext, symbol_dict):
                    _ext = ext[k]
                else:
                    _ext = ext
                _path_or_buf = str(Path(_dir_path) / f"{k}.{_ext}")
            elif isinstance(path_or_buf, symbol_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf

            _kwargs = self.select_symbol_kwargs(k, kwargs)
            sep = _kwargs.pop("sep", None)
            if isinstance(_path_or_buf, (str, Path)):
                _path_or_buf = Path(_path_or_buf)
                if isinstance(mkdir_kwargs, symbol_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_symbol_kwargs(k, mkdir_kwargs)
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                if _path_or_buf.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path_or_buf.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
            if sep is None:
                sep = ","
            v.to_csv(path_or_buf=_path_or_buf, sep=sep, **_kwargs)

    def to_hdf(
        self,
        file_path: tp.Union[tp.PathLike, symbol_dict] = ".",
        key: tp.Optional[tp.Union[str, symbol_dict]] = None,
        path_or_buf: tp.Optional[tp.Union[str, symbol_dict]] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, symbol_dict] = None,
        **kwargs,
    ) -> None:
        """Save data into an HDF file.

        Any argument can be provided per symbol using `symbol_dict`.

        If `file_path` exists, and it's a directory, will create inside it a file named
        after this class. This won't work with directories that do not exist, otherwise
        they could be confused with file names."""
        for k, v in self.data.items():
            if path_or_buf is None:
                if isinstance(file_path, symbol_dict):
                    _file_path = file_path[k]
                else:
                    _file_path = file_path
                _file_path = Path(_file_path)
                if _file_path.exists() and _file_path.is_dir():
                    _file_path /= type(self).__name__ + ".h5"
                _dir_path = _file_path.parent
                if isinstance(mkdir_kwargs, symbol_dict):
                    _mkdir_kwargs = mkdir_kwargs[k]
                else:
                    _mkdir_kwargs = self.select_symbol_kwargs(k, mkdir_kwargs)
                check_mkdir(_dir_path, **_mkdir_kwargs)
                _path_or_buf = str(_file_path)
            elif isinstance(path_or_buf, symbol_dict):
                _path_or_buf = path_or_buf[k]
            else:
                _path_or_buf = path_or_buf
            if key is None:
                _key = str(k)
            elif isinstance(key, symbol_dict):
                _key = key[k]
            else:
                _key = key
            _kwargs = self.select_symbol_kwargs(k, kwargs)
            v.to_hdf(path_or_buf=_path_or_buf, key=_key, **_kwargs)

    # ############# Transforming ############# #

    def transform(self: DataT, transform_func: tp.Callable, *args, **kwargs) -> DataT:
        """Transform data.

        Concatenates all the data into a single DataFrame and calls `transform_func` on it.
        Then, splits the data by symbol and builds a new `Data` instance."""
        concat_data = pd.concat(self.data.values(), axis=1, keys=pd.Index(self.symbols, name="symbol"))
        new_concat_data = transform_func(concat_data, *args, **kwargs)
        new_wrapper = None
        new_data = symbol_dict()
        for k in self.symbols:
            if isinstance(new_concat_data.columns, pd.MultiIndex):
                new_v = new_concat_data.xs(k, axis=1, level="symbol")
            else:
                if len(self.wrapper.columns) == 1 and self.wrapper.columns[0] != 0:
                    new_v = new_concat_data[k].rename(self.wrapper.columns[0])
                else:
                    new_v = new_concat_data[k].rename(None)
            _new_wrapper = ArrayWrapper.from_obj(new_v)
            if new_wrapper is None:
                new_wrapper = _new_wrapper
            else:
                if not checks.is_index_equal(new_wrapper.columns, _new_wrapper.columns):
                    raise ValueError("Transformed symbols must have the same columns")
            new_data[k] = new_v
        return self.replace(
            wrapper=new_wrapper,
            data=new_data,
        )

    def resample(self: DataT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> DataT:
        """Perform resampling on `Data` based on `Data.column_config`.

        Columns "open", "high", "low", "close", "volume", "trade count", and "vwap" (case-insensitive)
        are recognized and resampled automatically.

        Looks for `resample_func` of each column in `Data.column_config`. The function must
        accept the `Data` instance, object, and resampler."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        new_data = symbol_dict()
        for k, v in self.data.items():
            if checks.is_series(v):
                columns = [v.name]
            else:
                columns = v.columns
            new_v = []
            for c in columns:
                if checks.is_series(v):
                    obj = v
                else:
                    obj = v[c]
                resample_func = self.column_config.get(c, {}).get("resample_func", None)
                if resample_func is not None:
                    new_v.append(resample_func(self, obj, wrapper_meta["resampler"]))
                else:
                    if isinstance(c, str) and c.lower() == "open":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.first_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "high":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.max_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "low":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.min_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "close":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.last_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "volume":
                        new_v.append(obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.sum_reduce_nb))
                    elif isinstance(c, str) and c.lower() == "trade count":
                        new_v.append(
                            obj.vbt.resample_apply(
                                wrapper_meta["resampler"],
                                generic_nb.sum_reduce_nb,
                                wrap_kwargs=dict(dtype=int),
                            )
                        )
                    elif isinstance(c, str) and c.lower() == "vwap":
                        volume_obj = None
                        for c2 in columns:
                            if isinstance(c2, str) and c2.lower() == "volume":
                                volume_obj = v[c2]
                        if volume_obj is None:
                            raise ValueError("Volume is required to resample VWAP")
                        new_v.append(
                            pd.DataFrame.vbt.resample_apply(
                                wrapper_meta["resampler"],
                                generic_nb.wmean_range_reduce_meta_nb,
                                to_2d_array(obj),
                                to_2d_array(volume_obj),
                                wrapper=self.wrapper[c],
                            )
                        )
                    else:
                        raise ValueError(f"Cannot resample column '{c}'. Specify resample_func in column_config.")
            if checks.is_series(v):
                new_v = new_v[0]
            else:
                new_v = pd.concat(new_v, axis=1)
            new_data[k] = new_v
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    # ############# Running ############# #

    def run(
        self,
        func: tp.Union[str, tp.Callable],
        *args,
        on_indices: tp.Union[None, slice, tp.Sequence] = None,
        on_dates: tp.Union[None, slice, tp.Sequence] = None,
        on_symbols: tp.Union[None, tp.Symbol, tp.Symbols] = None,
        pass_as_first: bool = False,
        rename_args: tp.DictLike = None,
        **kwargs,
    ) -> tp.Any:
        """Run a function on data.

        Looks into the signature of the function and searches for arguments with the name `data` or
        those found among magnet columns.

        `func` can be one of the following:

        * "talib_{name}": Name of a TA-Lib indicator
        * "pandas_ta_{name}": Name of a Pandas TA indicator
        * "ta_{name}": Name of a TA indicator
        * "wqa101_{number}": Number of a WQA indicator
        * "{name}": Name of a custom indicator, otherwise of the above if found
        * "from_{mode}": Name of the simulation mode in `vectorbtpro.portfolio.base.Portfolio`
        * Indicator: Any indicator class built with the indicator factory
        * Callable: Function to run

        For example, the argument `open` will be substituted by `Data.open`.

        Use `on_indices` (using `iloc`), `on_dates` (using `loc`), and `on_symbols` (using `Data.select`)
        to filter data. For example, to filter a date range, pass `slice(start_date, end_date)`.

        Use `rename_args` to rename arguments. For example, in `vectorbtpro.portfolio.base.Portfolio`,
        data can be passed instead of `close`."""
        from vectorbtpro.indicators.factory import IndicatorBase, IndicatorFactory
        from vectorbtpro.indicators import custom
        from vectorbtpro.portfolio.base import Portfolio
        from vectorbtpro.utils.opt_packages import check_installed

        _self = self
        if on_indices is not None:
            _self = _self.iloc[on_indices]
        if on_dates is not None:
            _self = _self.loc[on_dates]
        if on_symbols is not None:
            _self = _self.select(on_symbols)

        if pass_as_first:
            return func(_self, *args, **kwargs)

        if isinstance(func, str):
            func = func.lower().strip()
            if func.startswith("from_") and getattr(Portfolio, func):
                func = getattr(Portfolio, func)
                return func(_self, *args, **kwargs)
            if hasattr(custom, func.upper()):
                func = getattr(custom, func.upper())
            elif func.startswith("talib_"):
                func = IndicatorFactory.from_talib(func.replace("talib_", ""))
            elif func.startswith("pandas_ta_"):
                func = IndicatorFactory.from_pandas_ta(func.replace("pandas_ta_", ""))
            elif func.startswith("ta_"):
                func = IndicatorFactory.from_ta(func.replace("ta_", ""))
            elif func.startswith("wqa101_"):
                func = IndicatorFactory.from_wqa101(int(func.replace("wqa101_", "")))
            elif check_installed("talib") and func.upper() in IndicatorFactory.get_talib_indicators():
                func = IndicatorFactory.from_talib(func)
            elif check_installed("pandas_ta") and func.upper() in IndicatorFactory.get_pandas_ta_indicators():
                func = IndicatorFactory.from_pandas_ta(func)
            elif check_installed("ta") and func.upper() in IndicatorFactory.get_ta_indicators():
                func = IndicatorFactory.from_ta(func)
            else:
                raise ValueError(f"Could not find indicator with name '{func}'")
        if isinstance(func, type) and issubclass(func, IndicatorBase):
            func = func.run

        with_kwargs = dict()
        for arg_name in get_func_arg_names(func):
            real_arg_name = arg_name
            if rename_args is not None:
                if arg_name in rename_args:
                    arg_name = rename_args[arg_name]
            if real_arg_name not in kwargs and arg_name == "data":
                with_kwargs[real_arg_name] = _self
            elif real_arg_name not in kwargs and arg_name == "open":
                with_kwargs[real_arg_name] = _self.open
            elif real_arg_name not in kwargs and arg_name == "high":
                with_kwargs[real_arg_name] = _self.high
            elif real_arg_name not in kwargs and arg_name == "low":
                with_kwargs[real_arg_name] = _self.low
            elif real_arg_name not in kwargs and arg_name == "close":
                with_kwargs[real_arg_name] = _self.close
            elif real_arg_name not in kwargs and arg_name == "volume":
                with_kwargs[real_arg_name] = _self.volume
            elif real_arg_name not in kwargs and arg_name == "trade_count":
                with_kwargs[real_arg_name] = _self.trade_count
            elif real_arg_name not in kwargs and arg_name == "vwap":
                with_kwargs[real_arg_name] = _self.vwap
            elif real_arg_name not in kwargs and arg_name == "hlc3":
                with_kwargs[real_arg_name] = _self.hlc3
            elif real_arg_name not in kwargs and arg_name == "ohlc4":
                with_kwargs[real_arg_name] = _self.ohlc4
        new_args, new_kwargs = extend_args(func, args, kwargs, **with_kwargs)
        return func(*new_args, **new_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.stats`.

        Merges `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `stats` from `vectorbtpro._settings.data`."""
        data_stats_cfg = self.get_settings(key_id="base")["stats"]

        return merge_dicts(Analyzable.stats_defaults.__get__(self), data_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start=dict(
                title="Start",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end=dict(
                title="End",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            period=dict(
                title="Period",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_symbols=dict(
                title="Total Symbols",
                calc_func=lambda self: len(self.symbols),
                agg_func=None,
                tags="data",
            ),
            last_index=dict(
                title="Last Index",
                calc_func="last_index",
                agg_func=None,
                tags="data",
            ),
            null_counts=dict(
                title="Null Counts",
                calc_func=lambda self, group_by: {
                    symbol: obj.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by)
                    for symbol, obj in self.data.items()
                },
                agg_func=lambda x: x.sum(),
                tags="data",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        column_names: tp.KwargsLike = None,
        plot_volume: tp.Optional[bool] = None,
        base: tp.Optional[float] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:  # pragma: no cover
        """Plot either one column of multiple symbols, or OHLC(V) of one symbol.

        Args:
            column (str): Name of the column to plot.
            symbol (str): Symbol to plot.
            column_names (sequence of str): Dictionary mapping the column names to OHLCV.

                Applied only if OHLC(V) is plotted.
            plot_volume (bool): Whether to plot volume beneath.

                Applied only if OHLC(V) is plotted.
            base (float): Rebase all series of a column to a given intial base.

                !!! note
                    The column must contain prices.

                Applied only if lines are plotted.
            kwargs (dict): Keyword arguments passed to `vectorbtpro.generic.accessors.GenericAccessor.plot`
                for lines and to `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot` for OHLC(V).

        Usage:
            * Plot the lines of one column across all symbols:

            ```pycon
            >>> import vectorbtpro as vbt

            >>> start = '2021-01-01 UTC'  # crypto is in UTC
            >>> end = '2021-06-01 UTC'
            >>> data = vbt.YFData.fetch(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
            ```

            [=100% "100%"]{: .candystripe}

            ```pycon
            >>> data.plot(column='Close', base=1)
            ```

            * Plot OHLC(V) of one symbol (only if data contains the respective columns):

            ![](/assets/images/data_plot.svg)

            ```pycon
            >>> data.plot(symbol='BTC-USD')
            ```

            ![](/assets/images/data_plot_ohlcv.svg)
        """
        if column is None:
            first_data = self.data[self.symbols[0]]
            if checks.is_frame(first_data):
                ohlc = first_data.vbt.ohlcv(column_names=column_names).ohlc
                if ohlc is not None and len(ohlc.columns) == 4:
                    if symbol is None:
                        if self.single_symbol or len(self.symbols) == 1:
                            symbol = self.symbols[0]
                        else:
                            raise ValueError("Only one symbol is allowed. Use indexing or symbol argument.")
                    return (
                        self.get(
                            symbols=symbol,
                        )
                        .vbt.ohlcv(
                            column_names=column_names,
                        )
                        .plot(
                            plot_volume=plot_volume,
                            **kwargs,
                        )
                    )
        self_col = self.select_col(column=column, group_by=False)
        if symbol is not None:
            self_col = self_col.select(symbol)
        data = self_col.get()
        if base is not None:
            data = data.vbt.rebase(base)
        return data.vbt.lineplot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `Data.plots`.

        Merges `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults` and
        `plots` from `vectorbtpro._settings.data`."""
        data_plots_cfg = self.get_settings(key_id="base")["plots"]

        return merge_dicts(Analyzable.plots_defaults.__get__(self), data_plots_cfg)

    _subplots: tp.ClassVar[Config] = Config(
        dict(
            plot=RepEval(
                """
                if symbols is None:
                    symbols = self.symbols
                if not isinstance(symbols, list):
                    symbols = [symbols]
                [
                    dict(
                        check_is_not_grouped=True,
                        plot_func="plot",
                        plot_volume=False,
                        symbol=s,
                        title=s,
                        pass_add_trace_kwargs=True,
                        xaxis_kwargs=dict(rangeslider_visible=False, showgrid=True),
                        yaxis_kwargs=dict(showgrid=True),
                        tags="data",
                    )
                    for s in symbols
                ]""",
                context=dict(symbols=None),
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots

    # ############# Docs ############# #

    @classmethod
    def build_column_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build column config documentation."""
        if source_cls is None:
            source_cls = Data
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "column_config").__doc__)).substitute(
            {"column_config": cls.column_config.prettify(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_column_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Call this method on each subclass that overrides `Data.column_config`."""
        __pdoc__[cls.__name__ + ".column_config"] = cls.build_column_config_doc(source_cls=source_cls)


Data.override_column_config_doc(__pdoc__)
Data.override_metrics_doc(__pdoc__)
Data.override_subplots_doc(__pdoc__)
